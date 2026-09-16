"""Independent-field ESS decoder with learned queries over recurrent latents.

Each ESS field is decoded in an independent sequence.  The decoder therefore
never sees gold values from the other fields.  Three learned queries attend to
the complete recurrent latent trajectory and produce one soft-prefix token per
field for a frozen Qwen2.5-0.5B decoder with trainable LoRA adapters.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer


FIELD_NAMES: Tuple[str, ...] = (
    "referent",
    "evidence",
    "reason",
)


class IndependentFieldQueryDecoder(nn.Module):
    def __init__(
        self,
        decoder_model: str,
        latent_dim: int,
        latent_steps: int = 4,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.0,
        attention_heads: int = 8,
    ):
        super().__init__()
        self.decoder_model = decoder_model
        self.latent_steps = int(latent_steps)
        self.tokenizer = AutoTokenizer.from_pretrained(
            decoder_model, trust_remote_code=True, use_fast=False
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.lm = AutoModelForCausalLM.from_pretrained(
            decoder_model, torch_dtype=torch.bfloat16, trust_remote_code=True
        )
        hidden_size = int(self.lm.config.hidden_size)
        self.memory_norm = nn.LayerNorm(int(latent_dim))
        self.memory_proj = nn.Linear(int(latent_dim), hidden_size, bias=False)
        self.field_queries = nn.Parameter(torch.empty(len(FIELD_NAMES), hidden_size))
        self.query_attention = nn.MultiheadAttention(
            hidden_size, int(attention_heads), dropout=0.0, batch_first=True
        )
        self.query_norm = nn.LayerNorm(hidden_size)
        # A functional probe maps decoder-input-space field queries into the
        # frozen decoder's semantic hidden space.  We deliberately do not
        # align raw recurrent z with text hidden states: those occupy different
        # functional positions.  This probe is discarded at inference.
        # The recurrent latent memory must be the primary information path.
        # Field queries only disambiguate which ESS field is being decoded.
        self.query_bias_scale = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
        nn.init.normal_(self.memory_proj.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.field_queries, mean=0.0, std=0.02)
        self.lm = get_peft_model(
            self.lm,
            LoraConfig(
                r=int(lora_r),
                lora_alpha=int(lora_alpha),
                target_modules=[
                    "q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj",
                ],
                lora_dropout=float(lora_dropout),
                bias="none",
                task_type="CAUSAL_LM",
            ),
        )

    def read_fields(self, latents: torch.Tensor):
        """Return [B, 4, decoder_hidden] field-conditioned soft prefixes."""
        memory_dtype = self.memory_norm.weight.dtype
        memory = self.memory_proj(self.memory_norm(latents.to(memory_dtype)))
        memory = memory.to(dtype=self.field_queries.dtype)
        batch = int(memory.shape[0])
        queries = self.field_queries.unsqueeze(0).expand(batch, -1, -1)
        attended, attention = self.query_attention(
            queries, memory, memory, need_weights=True, average_attn_weights=False
        )
        scale = self.query_bias_scale.to(attended.device, attended.dtype)
        prefixes = self.query_norm(attended + scale * queries)
        return prefixes, attention

    def forward(
        self,
        latents: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        content_mask: Optional[torch.Tensor] = None,
        stop_mask: Optional[torch.Tensor] = None,
        field_weights: Optional[torch.Tensor] = None,
        stop_weight: float = 0.2,
        semantic_weight: float = 0.0,
        field_prefixes: Optional[torch.Tensor] = None,
        query_attention: Optional[torch.Tensor] = None,
    ):
        """Decode three independent field sequences.

        input_ids has shape [B, F, T], where F == 4.  Only target value tokens
        are selected by content_mask; prompt and field-name tokens are never
        supervised.  stop_mask applies a small explicit EOS/closing-tag loss.
        """
        batch, field_count, seq_len = input_ids.shape
        if field_count != len(FIELD_NAMES):
            raise ValueError(f"expected {len(FIELD_NAMES)} fields, got {field_count}")
        if field_prefixes is None:
            prefixes, computed_attention = self.read_fields(latents)
            if query_attention is None:
                query_attention = computed_attention
        else:
            prefixes = field_prefixes
            if query_attention is None:
                raise ValueError(
                    'query_attention is required when field_prefixes are supplied'
                )
        flat_ids = input_ids.reshape(batch * field_count, seq_len)
        flat_attention = attention_mask.reshape(batch * field_count, seq_len)
        base = self.lm.get_base_model()
        token_embeds = base.get_input_embeddings()(flat_ids)
        prefix = prefixes.reshape(batch * field_count, 1, -1).to(token_embeds.dtype)
        inputs_embeds = torch.cat([prefix, token_embeds], dim=1)
        full_attention = torch.cat(
            [torch.ones((batch * field_count, 1), dtype=flat_attention.dtype,
                        device=flat_attention.device), flat_attention], dim=1
        )
        outputs = self.lm(
            inputs_embeds=inputs_embeds,
            attention_mask=full_attention,
            use_cache=False,
            return_dict=True,
        )
        result: Dict[str, torch.Tensor] = {
            "logits": outputs.logits,
            "query_attention": query_attention,
            "query_prefixes": prefixes,
        }
        if labels is None:
            return result

        flat_labels = labels.reshape(batch * field_count, seq_len)
        full_labels = torch.full(
            (batch * field_count, seq_len + 1), -100,
            dtype=torch.long, device=flat_labels.device
        )
        full_labels[:, 1:] = flat_labels
        shift_logits = outputs.logits[:, :-1].float()
        shift_labels = full_labels[:, 1:]
        token_loss = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.shape[-1]),
            shift_labels.reshape(-1), ignore_index=-100, reduction="none"
        ).view_as(shift_labels)

        flat_content = content_mask.reshape(batch * field_count, seq_len).float()
        flat_stop = stop_mask.reshape(batch * field_count, seq_len).float()
        # shift_logits position 0 is the learned field prefix and predicts
        # labels position 0, so token_loss is already aligned with the
        # unshifted per-field masks.
        valid = shift_labels.ne(-100).float()
        # Out-of-place: in-place *= on the caller's mask tensors bumps the
        # autograd version counter of the shared base tensor, so a second
        # boundary in the same row (or any second backward on the same batch)
        # fails with "modified by an inplace operation".
        full_content = flat_content * valid
        full_stop = flat_stop * valid
        per_field_content = (
            (token_loss * full_content).sum(dim=1)
            / full_content.sum(dim=1).clamp_min(1.0)
        ).view(batch, field_count)
        per_field_stop = (
            (token_loss * full_stop).sum(dim=1)
            / full_stop.sum(dim=1).clamp_min(1.0)
        ).view(batch, field_count)
        if field_weights is None:
            field_weights = torch.ones(field_count, device=token_loss.device)
        field_weights = field_weights.to(token_loss.device, torch.float32)
        content_loss = (
            per_field_content * field_weights.unsqueeze(0)
        ).sum() / (field_weights.sum() * batch).clamp_min(1.0)
        stop_loss = per_field_stop.mean()

        if semantic_weight != 0:
            raise ValueError('This run explicitly disables all hidden/semantic alignment')
        result.update(loss=content_loss + float(stop_weight) * stop_loss,
                      content_loss=content_loss, stop_loss=stop_loss,
                      field_losses=per_field_content.mean(dim=0))
        return result
