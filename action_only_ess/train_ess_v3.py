#!/usr/bin/env python3
"""Trajectory-level recurrent-latent ESS training for Qwen2.5-VL agents.

One dataset item is one complete parent trajectory.  Every executable boundary
is replayed in chronological order.  The K recurrent states created at an
earlier boundary remain numerically present in the KV cache seen by later
boundaries, while the cache is detached at each boundary to bound memory.
Within a boundary, the K-step recurrent chain and its action/ESS losses retain
full BPTT.  Thus this is trajectory-context TBPTT, not boundary-flat training
and not full-trajectory BPTT.
"""

import argparse
import json
import math
import os
import random
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset, Sampler
from torch.utils.checkpoint import checkpoint
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLProcessor,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
from transformers.cache_utils import DynamicCache

from mcp_latent_distill.ar_latent import (
    build_multimodal_embeddings,
    find_token_span,
    qwen_position_ids,
)
from mcp_latent_distill.qwenvl_utils_dynamic import encode_full
from ess_decoder import FIELD_NAMES, IndependentFieldQueryDecoder


def find_last_exact_token_span(sequence, target, min_start=0):
    """Find the final exact target occurrence without relying on vendor API.

    Repeated crop/zoom calls can be valid after a new tool_result/image.  The
    supervised boundary is therefore the last exact occurrence, never the
    first matching action preserved in visible history.
    """
    sequence = list(sequence)
    target = list(target)
    if not target:
        raise ValueError("empty target token sequence")
    lower = max(0, int(min_start))
    for start in range(len(sequence) - len(target), lower - 1, -1):
        if sequence[start:start + len(target)] == target:
            return start, start + len(target)
    raise ValueError(
        f"target token span not found after prompt: target_len={len(target)}, "
        f"sequence_len={len(sequence)}, min_start={lower}"
    )


def find_next_exact_token_span(sequence, target, min_start=0):
    """Find the next chronological target occurrence in a full trajectory."""
    sequence = list(sequence)
    target = list(target)
    if not target:
        raise ValueError("empty target token sequence")
    lower = max(0, int(min_start))
    for start in range(lower, len(sequence) - len(target) + 1):
        if sequence[start:start + len(target)] == target:
            return start, start + len(target)
    raise ValueError(
        f"target token span not found after cursor: target_len={len(target)}, "
        f"sequence_len={len(sequence)}, min_start={lower}"
    )


def clean_question(question):
    return re.sub(r'### User Image Path:\*\* "[^"]*"\n?', '', question or '').strip()


def read_offsets(path):
    offsets = []
    with open(path, "rb") as handle:
        while True:
            offset = handle.tell()
            line = handle.readline()
            if not line:
                break
            if line.strip():
                offsets.append(offset)
    return offsets


FIELD_INSTRUCTIONS = {
    "referent": "Identify the specific target referent encoded in the latent state.",
    "evidence": "State the visual or computational evidence relevant to the next decision.",
    "reason": "Explain why the next tool action or answer is appropriate.",
}


def field_prompt_text(name):
    return (
        "<|im_start|>system\n"
        "Read only the recurrent latent state. Return the requested executable "
        "semantic field and no other field.\n"
        "<|im_end|>\n<|im_start|>assistant\n"
        f"<field>{name}</field><value>"
    )


def encode_independent_fields(tokenizer, ess, max_length=96):
    """Encode four separate prompts; only field values and stop tokens are labels."""
    encoded = []
    for name in FIELD_NAMES:
        value = str((ess or {}).get(name, "")).strip()
        if not value:
            raise ValueError(f"empty ESS field: {name}")
        prompt = field_prompt_text(name)
        tail = "</value><|im_end|>"
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        value_ids = tokenizer.encode(value, add_special_tokens=False)
        tail_ids = tokenizer.encode(tail, add_special_tokens=False)
        room = int(max_length) - len(prompt_ids) - len(tail_ids)
        if room <= 0:
            raise ValueError("ESS field max length too small")
        if len(value_ids) > room:
            raise ValueError(f'ESS field exceeds audited token budget: {name} {len(value_ids)} > {room}')
        ids = prompt_ids + value_ids + tail_ids
        labels = [-100] * len(prompt_ids) + value_ids + tail_ids
        content = [0.0] * len(prompt_ids) + [1.0] * len(value_ids) + [0.0] * len(tail_ids)
        stop = [0.0] * (len(prompt_ids) + len(value_ids)) + [1.0] * len(tail_ids)
        encoded.append((ids, labels, content, stop))
    # A trajectory stacks labels from several boundaries.  Use the audited
    # fixed width so fields from different boundaries can be stacked without
    # a second, error-prone padding pass in the collator.
    width = int(max_length)
    pad_id = int(tokenizer.pad_token_id)
    def pad(values, fill):
        return values + [fill] * (width - len(values))
    return {
        "ess_input_ids": torch.tensor([pad(x[0], pad_id) for x in encoded], dtype=torch.long),
        "ess_attention_mask": torch.tensor([pad([1] * len(x[0]), 0) for x in encoded], dtype=torch.long),
        "ess_labels": torch.tensor([pad(x[1], -100) for x in encoded], dtype=torch.long),
        "ess_content_mask": torch.tensor([pad(x[2], 0.0) for x in encoded], dtype=torch.float32),
        "ess_stop_mask": torch.tensor([pad(x[3], 0.0) for x in encoded], dtype=torch.float32),
    }


class BoundaryDataset(Dataset):
    def __init__(
        self,
        path,
        processor,
        ess_tokenizer,
        system_prompt,
        max_length,
        max_answer_tokens,
        ess_max_length,
        stages,
        image_root="",
    ):
        self.path = str(path)
        self.offsets = read_offsets(self.path)
        self.processor = processor
        self.ess_tokenizer = ess_tokenizer
        self.system_prompt = system_prompt
        self.max_length = int(max_length)
        self.max_answer_tokens = int(max_answer_tokens)
        self.ess_max_length = int(ess_max_length)
        self.stages = int(stages)
        self.image_root = str(image_root or "")
        self._handle = None

    def _resolve_images(self, paths, boundary_id):
        if not paths:
            raise RuntimeError(f'Non-run_code boundary has no images: {boundary_id}')
        resolved = []
        legacy_prefix = "/nvme1/lsl/mcp_v4_images/"
        cluster_prefix = "/public/f_data/lsl/images/"
        for raw_path in paths:
            path = str(raw_path)
            if self.image_root and path.startswith(legacy_prefix):
                path = os.path.join(self.image_root, path[len(legacy_prefix):])
            elif self.image_root and path.startswith(cluster_prefix):
                path = os.path.join(self.image_root, path[len(cluster_prefix):])
            if not os.path.isfile(path):
                raise FileNotFoundError(
                    f"missing image for {boundary_id}: {raw_path} -> {path}"
                )
            resolved.append(path)
        return resolved

    def __len__(self):
        return len(self.offsets)

    def _row(self, index):
        if self._handle is None:
            self._handle = open(self.path, "r", encoding="utf-8")
        self._handle.seek(self.offsets[index])
        return json.loads(self._handle.readline())

    def __getitem__(self, index):
        row = self._row(index)
        response = row["history_prefix"] + row["bridge_text"] + row["target_text"]
        resolved_images = self._resolve_images(row["images"], row["boundary_id"])
        encoded = encode_full(
            self.processor,
            clean_question(row["question"]),
            resolved_images,
            response,
            system=self.system_prompt,
            build_labels=False,
        )
        target_ids = self.processor.tokenizer.encode(row["target_text"], add_special_tokens=False)
        if row.get("kind") == "answer" and len(target_ids) > self.max_answer_tokens:
            raise RuntimeError(
                f"unfiltered overlength answer {row['boundary_id']}: "
                f"{len(target_ids)} > {self.max_answer_tokens} tokens"
            )
        target_start, target_end = find_last_exact_token_span(
            encoded.input_ids.tolist(), target_ids,
            min_start=int(encoded.prompt_len),
        )
        if 'audited_target_end' in row and target_end != row['audited_target_end']:
            raise RuntimeError(f'Processor/image-token mismatch with preflight: {row["boundary_id"]} {target_end} != {row["audited_target_end"]}')
        bridge_ids = self.processor.tokenizer.encode(row["bridge_text"], add_special_tokens=False)
        prefix_end = target_start - len(bridge_ids)
        if prefix_end < int(encoded.prompt_len):
            raise RuntimeError(f"bad prefix/bridge alignment for {row['boundary_id']}")
        if target_end > self.max_length:
            raise RuntimeError(f'Unaudited overlength boundary: {row["boundary_id"]} {target_end}')
        if not isinstance(row.get("ess"),dict):
            raise RuntimeError(f"missing ESS for {row['boundary_id']}")
        ess_tensors=encode_independent_fields(
            self.ess_tokenizer,row["ess"] if row.get('ess_active',True) else {k:'not supervised' for k in FIELD_NAMES},self.ess_max_length)
        result = {
            "input_ids": encoded.input_ids,
            "attention_mask": encoded.attention_mask,
            "pixel_values": encoded.pixel_values,
            "image_grid_thw": encoded.image_grid_thw,
            "prefix_end": torch.tensor(prefix_end, dtype=torch.long),
            "target_start": torch.tensor(target_start, dtype=torch.long),
            "target_end": torch.tensor(target_end, dtype=torch.long),
            "boundary_id": row["boundary_id"],
            "kind": row["kind"],
            "boundary_type": row["boundary_type"],
            "target_tool": row["target_tool"] or "none",
            # Durable provenance for high-gradient diagnostics.  These values
            # never enter the model forward path.
            "question_text": row["question"],
            # Diagnostics must copy the exact files used by the model, not
            # portable paths from another server's manifest.
            "image_paths": resolved_images,
            "target_text": row["target_text"],
            "ess_gold_json": json.dumps(row["ess"], ensure_ascii=False),
        }
        result.update(ess_tensors)
        result['ess_active'] = torch.tensor(float(row.get('ess_active',True)))
        return result


class BatchOneCollator:
    def __call__(self, features):
        if len(features) != 1:
            raise ValueError("CoLT boundary training requires microbatch size 1")
        item = features[0]
        batch = {}
        # Qwen2.5-VL represents pixel_values as [all_patches, patch_dim] and
        # image_grid_thw as [num_images, 3]; unlike text tensors, neither has
        # an outer batch dimension in the model API.
        no_batch_dim = {"pixel_values", "image_grid_thw"}
        for key, value in item.items():
            if torch.is_tensor(value) and key not in no_batch_dim:
                batch[key] = value.unsqueeze(0)
            else:
                batch[key] = value
        return batch


def split_vision_inputs(pixel_values, image_grid_thw):
    """Treat an empty image batch as a text-only forward pass.

    Image-less run_code trajectories are the only samples allowed to carry
    no images.  Qwen represents that case as zero-length tensors, which the
    visual encoder rejects, so normalize the pair to None for the embedding
    build and the RoPE index.
    """
    if pixel_values is not None and int(pixel_values.numel()) == 0:
        return None, None
    if image_grid_thw is not None and int(image_grid_thw.numel()) == 0:
        return pixel_values, None
    return pixel_values, image_grid_thw

def locate_action_span(tokenizer, sequence, target_ids, min_start):
    """Locate an action span, tolerating one byte-pair merge on its left edge.

    Returns ``(start, end, split)``.  ``split`` is ``None`` when the target
    already occupies whole tokens; otherwise it is ``(index, left, right)`` and
    names a token that must be divided so the target starts on a boundary.
    """
    target_ids = [int(value) for value in target_ids]
    length = len(target_ids)
    if length == 0:
        raise ValueError("empty target token sequence")
    for start in range(int(min_start), len(sequence) - length + 1):
        if sequence[start:start + length] == target_ids:
            return start, start + length, None
    if length > 1:
        head_ids = target_ids[1:]
        head_text = tokenizer.decode([target_ids[0]], skip_special_tokens=False)
        for start in range(max(1, int(min_start)), len(sequence) - length + 2):
            if sequence[start:start + length - 1] != head_ids:
                continue
            previous_text = tokenizer.decode(
                [sequence[start - 1]], skip_special_tokens=False
            )
            if not previous_text.endswith(head_text):
                continue
            left_text = previous_text[:len(previous_text) - len(head_text)]
            if not left_text:
                continue
            return start, start + length, (start - 1, left_text, head_text)
    raise ValueError(
        "target token span not found after cursor: "
        "target_len=%d, sequence_len=%d, min_start=%d"
        % (length, len(sequence), int(min_start))
    )


def _split_token_text(tokenizer, token_id, left_text, right_text):
    """Return the token ids that replace one merged token's text."""
    combined = left_text + right_text
    original = tokenizer.decode([int(token_id)], skip_special_tokens=False)
    if original != combined:
        raise ValueError(
            "token split does not reproduce the merged token: %r != %r"
            % (original, combined)
        )
    left_ids = tokenizer.encode(left_text, add_special_tokens=False)
    right_ids = tokenizer.encode(right_text, add_special_tokens=False)
    if not left_ids or not right_ids:
        raise ValueError("empty token split for %r" % combined)
    rebuilt = tokenizer.decode(left_ids + right_ids, skip_special_tokens=False)
    if rebuilt != combined:
        raise ValueError(
            "token split is not text preserving: %r != %r" % (rebuilt, combined)
        )
    return left_ids, right_ids


def apply_token_splits(tokenizer, sequence, mask, splits):
    """Rewrite merged tokens as the two tokens autoregressive decoding emits."""
    tokens = []
    masks = []
    for index, token_id in enumerate(sequence):
        if index in splits:
            left_text, right_text = splits[index]
            left_ids, right_ids = _split_token_text(
                tokenizer, token_id, left_text, right_text
            )
            replacement = left_ids + right_ids
            if len(replacement) < 2:
                raise ValueError(
                    "token split produced %d tokens" % len(replacement)
                )
            tokens.extend(replacement)
            masks.extend([int(mask[index])] * len(replacement))
        else:
            tokens.append(int(token_id))
            masks.append(int(mask[index]))
    return tokens, masks


class TrajectoryDataset(Dataset):
    """Group decision blocks by parent and encode each full trajectory once.

    One block may contain several chronologically consecutive tool calls after
    a single reasoning unit.  Such a block receives one recurrent latent
    rollout and one ESS target, while every active action span in the block is
    supervised.
    """

    def __init__(
        self,
        path,
        processor,
        ess_tokenizer,
        system_prompt,
        max_length,
        max_answer_tokens,
        ess_max_length,
        stages,
        max_images=4,
        image_root="",
        audit_path="",
        length_bucket_width=256,
    ):
        self.path = str(path)
        self.processor = processor
        self.ess_tokenizer = ess_tokenizer
        self.system_prompt = system_prompt
        self.max_length = int(max_length)
        self.max_answer_tokens = int(max_answer_tokens)
        self.ess_max_length = int(ess_max_length)
        self.stages = int(stages)
        self.max_images = int(max_images)
        self.length_bucket_width = max(1, int(length_bucket_width))
        self.image_root = str(image_root or "")
        self._handle = None

        grouped = defaultdict(list)
        # A cheap, deterministic proxy for the eventual multimodal sequence
        # length.  Exact tokenization happens lazily in __getitem__, but this
        # proxy is enough to keep very short and very long trajectories out of
        # the same synchronized step.
        length_estimates = defaultdict(int)
        with open(self.path, "rb") as handle:
            while True:
                offset = handle.tell()
                line = handle.readline()
                if not line:
                    break
                if not line.strip():
                    continue
                row = json.loads(line)
                parent_id = str(row["parent_id"])
                grouped[parent_id].append(
                    (int(row["segment_index"]), int(offset),
                     len(row.get("images") or []), str(row["boundary_id"]))
                )
                audited_end = row.get("audited_target_end")
                if audited_end is not None:
                    # The merged data preflight records the exact multimodal
                    # target end after Qwen image-token expansion.  Use it for
                    # bucketing so synchronized ranks have similar real cost.
                    estimate = int(audited_end) + (self.stages + 1) * 1
                else:
                    text_chars = sum(
                        len(str(row.get(name) or ""))
                        for name in ("question", "history_prefix", "bridge_text", "block_text")
                    )
                    # Fallback only for legacy rows without preflight metadata.
                    estimate = int(text_chars / 3.5) + 900 * len(row.get("images") or [])
                length_estimates[parent_id] = max(length_estimates[parent_id], estimate)

        self.groups = []
        quarantine = []
        for parent_id, entries in grouped.items():
            entries.sort(key=lambda item: item[0])
            if len({item[0] for item in entries}) != len(entries):
                raise RuntimeError(f"duplicate segment index for {parent_id}")
            max_seen_images = max(item[2] for item in entries)
            if max_seen_images > self.max_images:
                quarantine.append({
                    "parent_id": parent_id,
                    "reason": "images_over_limit",
                    "images": max_seen_images,
                    "limit": self.max_images,
                })
                continue
            self.groups.append((parent_id, tuple(item[1] for item in entries)))
        self.groups.sort(key=lambda item: item[0])
        self._length_estimates = {
            parent_id: int(length_estimates[parent_id])
            for parent_id, _ in self.groups
        }
        self.quarantine = quarantine
        if audit_path:
            payload = {
                "schema": "trajectory_context_tbptt_v1",
                "source": self.path,
                "parents_seen": len(grouped),
                "parents_kept_before_token_audit": len(self.groups),
                "parents_quarantined": len(quarantine),
                "quarantine_by_reason": dict(Counter(
                    item["reason"] for item in quarantine
                )),
                "max_images": self.max_images,
                "max_total_tokens_including_latent_queries": self.max_length,
                "image_mapping": {
                    "/nvme1/lsl/mcp_v4_images/": self.image_root,
                    "/public/f_data/lsl/images/": self.image_root,
                },
            }
            Path(audit_path).parent.mkdir(parents=True, exist_ok=True)
            Path(audit_path).write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    def _row_at(self, offset):
        if self._handle is None:
            self._handle = open(self.path, "r", encoding="utf-8")
        self._handle.seek(int(offset))
        return json.loads(self._handle.readline())

    def _resolve_images(self, paths, trajectory_id, allow_image_less_run_code=False):
        if not paths:
            if allow_image_less_run_code:
                return []
            raise RuntimeError(f"Non-run_code trajectory has no images: {trajectory_id}")
        resolved = []
        legacy_prefix = "/nvme1/lsl/mcp_v4_images/"
        cluster_prefix = "/public/f_data/lsl/images/"
        for raw_path in paths:
            path = str(raw_path)
            if self.image_root and path.startswith(legacy_prefix):
                path = os.path.join(self.image_root, path[len(legacy_prefix):])
            elif self.image_root and path.startswith(cluster_prefix):
                path = os.path.join(self.image_root, path[len(cluster_prefix):])
            if not os.path.isfile(path):
                raise FileNotFoundError(
                    f"missing image for {trajectory_id}: {raw_path} -> {path}"
                )
            resolved.append(path)
        if len(resolved) > self.max_images:
            raise RuntimeError(
                f"trajectory image count exceeded audited limit: "
                f"{trajectory_id} {len(resolved)} > {self.max_images}"
            )
        return resolved

    def __len__(self):
        return len(self.groups)

    def boundary_count(self, index):
        return len(self.groups[int(index)][1])

    def length_bucket(self, index, width=None):
        """Physical-length bucket used to balance synchronized distributed steps."""
        parent_id = self.groups[int(index)][0]
        estimate = max(1, int(self._length_estimates.get(parent_id, 1)))
        bucket_width = self.length_bucket_width if width is None else max(1, int(width))
        return estimate // bucket_width

    def _align_action_spans(self, rows, input_ids, attention_mask, prompt_len):
        """Put every action span and latent insertion point on token boundaries.

        Qwen byte-pair merges glue the tag that closes the previous turn to the
        tag that opens the next action, so ``</tool_result></think>`` and
        ``</think><answer>`` are each encoded with one shared token.  A shared
        token has no interior boundary, which would place both the supervised
        span and the latent insertion point inside a single token.  Generation
        cannot produce that merge because the leading character was already
        emitted, so the faithful reconstruction splits the merged token at
        exactly that character and leaves the rest of the sequence untouched.
        """
        tokenizer = self.processor.tokenizer
        sequence = [int(value) for value in input_ids]
        mask = [int(value) for value in attention_mask]
        for _ in range(8):
            splits = {}
            layout = []
            cursor = int(prompt_len)
            for row in rows:
                target_texts = list(row.get("target_texts") or [])
                active_flags = [
                    bool(value) for value in row.get("action_active") or []
                ]
                if not target_texts or len(target_texts) != len(active_flags):
                    raise RuntimeError(
                        f"bad block target metadata: {row['boundary_id']}"
                    )
                starts = []
                ends = []
                local_cursor = cursor
                for target_text in target_texts:
                    target_ids = tokenizer.encode(
                        target_text, add_special_tokens=False
                    )
                    if (
                        row.get("kind") == "answer"
                        and len(target_ids) > self.max_answer_tokens
                    ):
                        raise RuntimeError(
                            f"unfiltered overlength answer {row['boundary_id']}: "
                            f"{len(target_ids)} > {self.max_answer_tokens} tokens"
                        )
                    start, end, split = locate_action_span(
                        tokenizer, sequence, target_ids, local_cursor
                    )
                    if split is not None:
                        splits[split[0]] = (split[1], split[2])
                    starts.append(start)
                    ends.append(end)
                    local_cursor = end
                bridge_text = str(row.get("bridge_text") or "")
                bridge_ids = tokenizer.encode(bridge_text, add_special_tokens=False)
                prefix_end = starts[0] - len(bridge_ids)
                if bridge_ids and prefix_end >= 0:
                    window = tokenizer.decode(
                        sequence[prefix_end:starts[0]], skip_special_tokens=False
                    )
                    if window != bridge_text:
                        merged_text = tokenizer.decode(
                            [sequence[prefix_end]], skip_special_tokens=False
                        )
                        head_text = tokenizer.decode(
                            [bridge_ids[0]], skip_special_tokens=False
                        )
                        if (
                            merged_text.endswith(head_text)
                            and len(merged_text) > len(head_text)
                        ):
                            splits[prefix_end] = (
                                merged_text[:len(merged_text) - len(head_text)],
                                head_text,
                            )
                layout.append((starts, ends, prefix_end))
                cursor = ends[-1]
            if not splits:
                for row, (starts, ends, prefix_end) in zip(rows, layout):
                    bridge_text = str(row.get("bridge_text") or "")
                    window = tokenizer.decode(
                        sequence[prefix_end:starts[0]], skip_special_tokens=False
                    )
                    if window != bridge_text:
                        raise RuntimeError(
                            f"latent insertion point is not a token boundary: "
                            f"{row['boundary_id']} window={window!r} "
                            f"bridge={bridge_text!r}"
                        )
                    for target_text, start, end in zip(
                        row["target_texts"], starts, ends
                    ):
                        span_text = tokenizer.decode(
                            sequence[start:end], skip_special_tokens=False
                        )
                        if span_text != target_text:
                            raise RuntimeError(
                                f"action span mismatch after realignment: "
                                f"{row['boundary_id']} span={span_text!r} "
                                f"target={target_text!r}"
                            )
                return sequence, mask, layout
            sequence, mask = apply_token_splits(tokenizer, sequence, mask, splits)
        raise RuntimeError(
            "could not realign action spans onto token boundaries: %s"
            % rows[0]["boundary_id"]
        )

    def __getitem__(self, index):
        trajectory_id, offsets = self.groups[index]
        rows = [self._row_at(offset) for offset in offsets]
        rows.sort(key=lambda row: int(row["segment_index"]))
        if any(str(row["parent_id"]) != trajectory_id for row in rows):
            raise RuntimeError(f"parent grouping corruption: {trajectory_id}")
        if any(row["question"] != rows[0]["question"] for row in rows):
            raise RuntimeError(f"question changed inside trajectory: {trajectory_id}")

        # Each later flattened boundary was built from the same response with
        # earlier explicit reasoning removed.  Its history must therefore
        # extend the earlier boundary's history/action prefix monotonically.
        for left, right in zip(rows[:-1], rows[1:]):
            completed_left = (
                left["history_prefix"] + left["bridge_text"] + left["block_text"]
            )
            if not right["history_prefix"].startswith(completed_left):
                raise RuntimeError(
                    f"non-monotonic boundary history: {left['boundary_id']} -> "
                    f"{right['boundary_id']}"
                )

        last = rows[-1]
        response = last["history_prefix"] + last["bridge_text"] + last["block_text"]
        image_less_flags = [bool(row.get("image_less_run_code")) for row in rows]
        if len(set(image_less_flags)) != 1:
            raise RuntimeError(f"image-less exception changed inside trajectory: {trajectory_id}")
        if image_less_flags[0]:
            tools = [
                tool
                for row in rows
                for tool in str(row.get("target_tool") or "").split("+")
                if tool and tool != "none"
            ]
            if not tools or any(tool != "run_code" for tool in tools):
                raise RuntimeError(f"invalid image-less non-run_code trajectory: {trajectory_id}")
        resolved_images = self._resolve_images(
            last["images"], trajectory_id, image_less_flags[0]
        )
        encoded = encode_full(
            self.processor,
            clean_question(last["question"]),
            resolved_images,
            response,
            system=self.system_prompt,
            build_labels=False,
        )
        token_sequence, attention_sequence, layout = self._align_action_spans(
            rows,
            encoded.input_ids.tolist(),
            encoded.attention_mask.tolist(),
            int(encoded.prompt_len),
        )
        cursor = int(encoded.prompt_len)
        prefix_ends = []
        target_span_starts = []
        target_span_ends = []
        action_active_flags = []
        block_ends = []
        ess_encoded = []
        for row, (starts, ends, prefix_end) in zip(rows, layout):
            if prefix_end < cursor:
                raise RuntimeError(f"bad trajectory bridge alignment: {row['boundary_id']}")
            prefix_ends.append(prefix_end)
            target_span_starts.append(starts)
            target_span_ends.append(ends)
            action_active_flags.append(
                [bool(value) for value in row.get("action_active") or []]
            )
            block_ends.append(ends[-1])
            cursor = ends[-1]
            label = row["ess"] if row.get("ess_active", True) else {
                name: "not supervised" for name in FIELD_NAMES
            }
            ess_encoded.append(encode_independent_fields(
                self.ess_tokenizer, label, self.ess_max_length
            ))

        physical_tokens = int(block_ends[-1]) + (self.stages + 1) * len(rows)
        if physical_tokens > self.max_length:
            raise RuntimeError(
                f"trajectory exceeds max tokens after latent insertion: "
                f"{trajectory_id} {physical_tokens} > {self.max_length}"
            )
        if int(block_ends[-1]) > len(token_sequence):
            raise RuntimeError(f"target beyond encoded sequence: {trajectory_id}")

        result = {
            "input_ids": torch.tensor(token_sequence, dtype=torch.long),
            "attention_mask": torch.tensor(attention_sequence, dtype=torch.long),
            "pixel_values": encoded.pixel_values,
            "image_grid_thw": encoded.image_grid_thw,
            "prefix_ends": torch.tensor(prefix_ends, dtype=torch.long),
            "target_span_starts": target_span_starts,
            "target_span_ends": target_span_ends,
            "action_active_flags": action_active_flags,
            "block_ends": torch.tensor(block_ends, dtype=torch.long),
            "ess_active": torch.tensor(
                [float(row.get("ess_active", True)) for row in rows],
                dtype=torch.float32,
            ),
            "ess_input_ids": torch.stack([item["ess_input_ids"] for item in ess_encoded]),
            "ess_attention_mask": torch.stack([item["ess_attention_mask"] for item in ess_encoded]),
            "ess_labels": torch.stack([item["ess_labels"] for item in ess_encoded]),
            "ess_content_mask": torch.stack([item["ess_content_mask"] for item in ess_encoded]),
            "ess_stop_mask": torch.stack([item["ess_stop_mask"] for item in ess_encoded]),
            "trajectory_id": trajectory_id,
            "boundary_ids": [row["boundary_id"] for row in rows],
            "kinds": [row["kind"] for row in rows],
            "boundary_types": [row["boundary_type"] for row in rows],
            "target_tools": [row["target_tool"] or "none" for row in rows],
            "question_text": rows[0]["question"],
            "image_paths": resolved_images,
            "target_texts": [row["target_texts"] for row in rows],
            "ess_gold_jsons": [json.dumps(row["ess"], ensure_ascii=False) for row in rows],
            "raw_token_count": torch.tensor(int(block_ends[-1]), dtype=torch.long),
            "physical_token_count": torch.tensor(physical_tokens, dtype=torch.long),
        }
        return result


class TrajectoryCollator:
    def __call__(self, features):
        if len(features) != 1:
            raise ValueError("trajectory-context TBPTT requires microbatch size 1")
        item = features[0]
        batch = {}
        no_batch_dim = {"pixel_values", "image_grid_thw"}
        for key, value in item.items():
            if torch.is_tensor(value) and key not in no_batch_dim:
                batch[key] = value.unsqueeze(0)
            else:
                batch[key] = value
        return batch


class HomogeneousBoundaryDistributedSampler(Sampler):
    """Give every rank the same boundary count at each distributed step.

    Boundary-streaming performs one backward call per boundary.  All ranks must
    therefore execute the same number of backward collectives in a step.  This
    sampler shuffles within boundary-count buckets, pads only the final global
    group in each bucket, and then assigns one member of every homogeneous
    world-size group to each rank.
    """

    def __init__(self, dataset, num_replicas, rank, seed=42):
        self.dataset = dataset
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.seed = int(seed)
        self.epoch = 0
        buckets = defaultdict(list)
        for index in range(len(dataset)):
            # Equal boundary counts are necessary for identical collective
            # counts; similar length buckets avoid long-rank stragglers.
            key = (
                int(dataset.boundary_count(index)),
                int(dataset.length_bucket(index)) if hasattr(dataset, "length_bucket") else 0,
            )
            buckets[key].append(index)
        self.bucket_sizes = {key: len(value) for key, value in buckets.items()}
        self._buckets = dict(buckets)
        # Accelerate.prepare() shards the DataLoader after this sampler is
        # constructed. Therefore this sampler must expose the global padded
        # length and global item order; a rank-local sampler would be split
        # again and reduce one epoch to roughly 1/world_size of the data.
        self.num_samples = sum(
            math.ceil(len(values) / self.num_replicas) * self.num_replicas
            for values in self._buckets.values()
        )

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        global_groups = []
        for bucket_key, original in sorted(self._buckets.items()):
            values = list(original)
            rng.shuffle(values)
            needed = (-len(values)) % self.num_replicas
            if needed:
                # Repeat cyclically when a bucket has fewer than one full
                # world-size group.  `values[:needed]` is insufficient for
                # tiny buckets (e.g. 1 item and 8 ranks), producing short
                # groups and rank-dependent IndexError in Accelerate.
                repeats = (needed + len(values) - 1) // len(values)
                values.extend((values * repeats)[:needed])
            global_groups.extend(
                (bucket_key, values[start:start + self.num_replicas])
                for start in range(0, len(values), self.num_replicas)
            )
        rng.shuffle(global_groups)
        # Consecutive world-size groups contain one member from every
        # boundary/length bucket. Return the global sequence and let
        # Accelerate's prepared DataLoader perform the only rank sharding.
        ordered = []
        for _, members in global_groups:
            ordered.extend(members)
        return iter(ordered)


class CoLTModules(nn.Module):
    def __init__(self, main_hidden, ess_decoder, stages):
        super().__init__()
        self.stages = int(stages)
        self.transition = nn.Sequential(
            nn.Linear(main_hidden, main_hidden // 2),
            nn.GELU(),
            nn.Linear(main_hidden // 2, main_hidden),
            nn.LayerNorm(main_hidden),
        )
        self.alpha = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
        self.ess_decoder = ess_decoder


class HighGradDumpCallback(TrainerCallback):
    """Persist the eight candidate rows behind a high global-gradient step.

    With DDP/ZeRO the optimizer gradient is aggregated across ranks, so a
    single offending row cannot be identified from the global norm alone.
    Every rank therefore writes its one-row microbatch into a rank-specific
    directory.  This preserves the exact candidate set without changing the
    training computation.
    """

    def __init__(self, trainer, output_dir, threshold=20.0, max_events=32):
        self.trainer = trainer
        self.output_dir = Path(output_dir)
        self.threshold = float(threshold)
        self.max_events = int(max_events)
        self.saved_steps = set()

    @staticmethod
    def _as_float(value):
        if value is None:
            return None
        if torch.is_tensor(value):
            return float(value.detach().float().cpu())
        return float(value)

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        engine = getattr(self.trainer, "deepspeed", None)
        grad_norm = None
        if engine is not None and hasattr(engine, "get_global_grad_norm"):
            try:
                grad_norm = self._as_float(engine.get_global_grad_norm())
            except Exception:
                grad_norm = None
        if grad_norm is None:
            squares = []
            for parameter in self.trainer.model.parameters():
                if parameter.grad is not None:
                    squares.append(parameter.grad.detach().float().square().sum())
            if squares:
                grad_norm = float(torch.stack(squares).sum().sqrt().cpu())
        step = int(state.global_step) + 1
        if (grad_norm is None or grad_norm <= self.threshold or
                step in self.saved_steps or len(self.saved_steps) >= self.max_events):
            return control
        snapshot = getattr(self.trainer, "last_case_snapshot", None)
        if not snapshot:
            return control
        self.saved_steps.add(step)
        rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
        case_dir = self.output_dir / f"step_{step:06d}" / f"rank_{rank:02d}"
        image_dir = case_dir / "images"
        image_dir.mkdir(parents=True, exist_ok=True)
        copied = []
        for index, image_path in enumerate(snapshot.get("image_paths", [])):
            source = Path(image_path)
            destination = image_dir / f"image_{index:02d}{source.suffix or '.jpg'}"
            shutil.copy2(source, destination)
            copied.append(str(destination))
        payload = dict(snapshot)
        payload.update({
            "optimizer_step": step,
            "rank": rank,
            "global_grad_norm_preclip": grad_norm,
            "copied_images": copied,
        })
        (case_dir / "metadata.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        ess = payload.get("ess_gold", {})
        report = [
            f"# High-gradient candidate: step {step}, rank {rank}",
            "",
            f"- Global grad norm (pre-clip): `{grad_norm:.6f}`",
            f"- Boundary: `{payload.get('boundary_id', '')}`",
            f"- Kind/tool: `{payload.get('kind', '')}` / `{payload.get('target_tool', '')}`",
            f"- Action CE: `{payload.get('action_ce', '')}`",
            f"- ESS content NLL: `{payload.get('ess_content_nll', '')}`",
            f"- Adjacent cosine: `{payload.get('adjacent_cos', '')}`",
            "", "## Question", "", payload.get("question", ""),
            "", "## Gold action", "", "```text", payload.get("target_text", ""), "```",
            "", "## Gold ESS", "",
        ]
        for key in FIELD_NAMES:
            report.extend([f"### {key}", "", str(ess.get(key, "")), ""])
        for path in copied:
            report.extend(["## Image", "", f"![image]({path})", ""])
        # Snapshot fields can be structured (for example a multi-action
        # block's target_text).  Convert every item before joining so the
        # high-gradient diagnostic can never abort the training step.
        (case_dir / "report.md").write_text(
            "\n".join(str(item) for item in report), encoding="utf-8"
        )
        return control


def freeze_vision(model):
    for name, parameter in model.named_parameters():
        if "visual" in name or "vision" in name:
            parameter.requires_grad = False


def chunked_token_ce(hidden, labels, lm_head, chunk_size=32):
    active = labels.ne(-100)
    if not bool(active.any()):
        return hidden.sum() * 0.0
    hidden = hidden[active]
    labels = labels[active]
    total = hidden.new_zeros((), dtype=torch.float32)
    for left in range(0, labels.numel(), chunk_size):
        right = min(left + chunk_size, labels.numel())
        def fn(h, y):
            return F.cross_entropy(lm_head(h).float(), y, reduction="sum")
        total = total + checkpoint(fn, hidden[left:right], labels[left:right], use_reentrant=False)
    return total / labels.numel()


def detach_kv_cache(cache):
    """Clone a cache interface with every K/V tensor stop-gradient."""
    if cache is None:
        return None
    if hasattr(cache, "to_legacy_cache"):
        legacy = cache.to_legacy_cache()
        detached = tuple(
            tuple(tensor.detach() if torch.is_tensor(tensor) else tensor
                  for tensor in layer)
            for layer in legacy
        )
        if hasattr(DynamicCache, "from_legacy_cache"):
            return DynamicCache.from_legacy_cache(detached)
        return detached
    if isinstance(cache, (tuple, list)):
        return type(cache)(
            type(layer)(
                tensor.detach() if torch.is_tensor(tensor) else tensor
                for tensor in layer
            )
            for layer in cache
        )
    raise TypeError(f"unsupported KV cache type: {type(cache)!r}")


class CoLTTrainer(Trainer):
    def __init__(
        self,
        *args,
        backward_decoder,
        action_weight=1.0,
        forward_weight=0.2,
        backward_weight=0.2,
        prediction_weight=0.2,
        main_lr=1e-5,
        aux_lr=1e-5,
        ess_lr=5e-5,
        ess_weight=0.02,
        semantic_weight=0.2,
        ess_stop_weight=0.2,
        ess_warmup_steps=100,
        chunk_size=32,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.backward_decoder = backward_decoder
        self.action_weight = float(action_weight)
        self.forward_weight = float(forward_weight)
        self.backward_weight = float(backward_weight)
        self.prediction_weight = float(prediction_weight)
        self.main_lr = float(main_lr)
        self.aux_lr = float(aux_lr)
        self.ess_lr = float(ess_lr)
        self.ess_weight = float(ess_weight)
        self.semantic_weight = float(semantic_weight)
        self.ess_stop_weight = float(ess_stop_weight)
        self.ess_warmup_steps = int(ess_warmup_steps)
        self.chunk_size = int(chunk_size)
        self.metric_sums = {
            "train": defaultdict(float),
            "eval": defaultdict(float),
        }
        self.metric_counts = {
            "train": defaultdict(int),
            "eval": defaultdict(int),
        }

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer
        ess, aux, main = [], [], []
        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad:
                continue
            if "colt_modules.ess_decoder" in name:
                ess.append(parameter)
            elif "colt_modules" in name:
                aux.append(parameter)
            else:
                main.append(parameter)
        self.optimizer = torch.optim.AdamW(
            [
                {"params": main, "lr": self.main_lr},
                {"params": aux, "lr": self.aux_lr},
                {"params": ess, "lr": self.ess_lr},
            ],
            betas=(0.9, 0.95),
            weight_decay=self.args.weight_decay,
        )
        return self.optimizer

    @staticmethod
    def _bare(model):
        return model.module if hasattr(model, "module") else model


    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        causal=self._bare(model); backbone=causal.model.model; lm_head=causal.get_output_embeddings()
        modules=causal.colt_modules; mode="train" if model.training else "eval"
        input_ids=inputs["input_ids"]; attention_mask=inputs["attention_mask"]
        pixel_values,image_grid_thw=split_vision_inputs(inputs.get("pixel_values"),inputs.get("image_grid_thw"))
        prefix_end=int(inputs["prefix_end"].item()); target_start=int(inputs["target_start"].item()); target_end=int(inputs["target_end"].item())
        base_embeddings=build_multimodal_embeddings(model,input_ids,pixel_values,image_grid_thw).detach()
        positions=qwen_position_ids(model,input_ids,image_grid_thw,attention_mask)
        # Boundary-flat contract: this prefix contains real prior tool history,
        # but no earlier-boundary latent rollout is replayed or differentiated.
        with torch.no_grad():
            out=backbone(input_ids=input_ids[:,:prefix_end],inputs_embeds=base_embeddings[:,:prefix_end],
                         attention_mask=attention_mask[:,:prefix_end],position_ids=positions[...,:prefix_end],
                         use_cache=True,return_dict=True)
            cache=out.past_key_values; latent=out.last_hidden_state[:,-1]
        del out
        latents=[]; cached_len=prefix_end; last_position=positions[...,prefix_end-1:prefix_end]
        for stage in range(modules.stages):
            out=backbone(inputs_embeds=latent.unsqueeze(1),
                         attention_mask=torch.ones((1,cached_len+1),dtype=torch.long,device=input_ids.device),
                         position_ids=last_position+(stage+1),past_key_values=cache,
                         cache_position=torch.tensor([cached_len],device=input_ids.device),use_cache=True,return_dict=True)
            cache=out.past_key_values; hidden=out.last_hidden_state[:,-1]
            latent=hidden+modules.alpha.to(hidden.device,hidden.dtype)*modules.transition(hidden)
            latents.append(latent); cached_len+=1; del out
        suffix_ids=input_ids[:,prefix_end:target_end]; offset=target_start-prefix_end
        suffix_embeds=base_embeddings[:,prefix_end:target_end]
        action_inputs=torch.cat([latent.unsqueeze(1),suffix_embeds[:,:-1]],dim=1)
        suffix_pos=last_position+torch.arange(modules.stages+2,modules.stages+2+suffix_embeds.shape[1]-1,
                                              device=input_ids.device).view(1,1,-1)
        suffix_pos=suffix_pos.expand(last_position.shape[0],-1,-1)
        action_pos=torch.cat([last_position+(modules.stages+1),suffix_pos],dim=-1)
        action_out=backbone(inputs_embeds=action_inputs,
            attention_mask=torch.ones((1,cached_len+action_inputs.shape[1]),dtype=torch.long,device=input_ids.device),
            position_ids=action_pos,past_key_values=cache,
            cache_position=torch.arange(cached_len,cached_len+action_inputs.shape[1],device=input_ids.device),
            use_cache=False,return_dict=True)
        labels=suffix_ids.clone(); labels[:,:offset]=-100
        action_loss=chunked_token_ce(action_out.last_hidden_state,labels,lm_head,self.chunk_size)
        memory=torch.stack(latents,dim=1)
        if mode=='eval' and bool(inputs['ess_active'].item()) and hasattr(self,'eval_decode_rows'):
            if len(self.eval_decode_rows)<20:
                self.eval_decode_rows.append(dict(boundary_id=inputs['boundary_id'],latents=memory.detach().cpu(),gold=json.loads(inputs['ess_gold_json']),image_paths=inputs['image_paths']))
        ess_result=modules.ess_decoder(
            memory,
            inputs["ess_input_ids"],
            inputs["ess_attention_mask"],
            labels=inputs["ess_labels"],
            content_mask=inputs["ess_content_mask"],
            stop_mask=inputs["ess_stop_mask"],
            field_weights=torch.tensor([1.5,1.5,1.0],device=memory.device),
            stop_weight=self.ess_stop_weight,
            # The global warmup/weight is applied below so this remains a raw
            # interpretable loss rather than being hidden inside ESS NLL.
            semantic_weight=0.0,
        )
        ess_active=bool(inputs['ess_active'].item())
        ess_loss=ess_result["loss"] * inputs['ess_active'].reshape(())
        if self.semantic_weight != 0:
            raise ValueError('Alignment is disabled in this experiment')
        warm=max(1,int(getattr(self,"ess_warmup_steps",100)))
        ess_scale=self.ess_weight*min(1.0,float(self.state.global_step+1)/warm)
        total=self.action_weight*action_loss+ess_scale*ess_loss
        if not bool(torch.isfinite(total)):
            raise FloatingPointError(f'Nonfinite loss: {inputs["boundary_id"]}')
        adjacent=torch.stack([F.cosine_similarity(a.float(),b.float(),dim=-1).mean() for a,b in zip(latents[:-1],latents[1:])]).mean()
        metrics={"action_ce":action_loss,"ess_nll":ess_loss,
                 "ess_content_nll":ess_result["content_loss"],"ess_stop_nll":ess_result["stop_loss"],
                 "ess_effective_weight":torch.tensor(ess_scale,device=action_loss.device),
                 "adjacent_cos":adjacent,"alpha":modules.alpha,"latent_rms":memory.float().square().mean().sqrt()}
        kind=inputs.get("kind","unknown"); kind=kind[0] if isinstance(kind,(list,tuple)) else kind
        metrics[f"action_ce_{kind}"]=action_loss
        for name,val in zip(FIELD_NAMES,ess_result["field_losses"]): metrics[f"ess_{name}"]=val
        attn=ess_result["query_attention"].detach().float().mean(dim=(0,1))
        for field_index,name in enumerate(FIELD_NAMES):
            for latent_index in range(attn.shape[-1]):
                metrics[f"query_attn_{name}_z{latent_index+1}"]=attn[field_index,latent_index]
        if mode == "train":
            def scalar_text(key, default=""):
                value = inputs.get(key, default)
                if isinstance(value, (list, tuple)) and len(value) == 1:
                    value = value[0]
                return str(value)
            try:
                ess_gold = json.loads(scalar_text("ess_gold_json", "{}"))
            except json.JSONDecodeError:
                ess_gold = {}
            image_paths = inputs.get("image_paths", [])
            if isinstance(image_paths, str):
                image_paths = [image_paths]
            self.last_case_snapshot = {
                "boundary_id": scalar_text("boundary_id"),
                "kind": scalar_text("kind"),
                "boundary_type": scalar_text("boundary_type"),
                "target_tool": scalar_text("target_tool"),
                "question": scalar_text("question_text"),
                "image_paths": [str(path) for path in image_paths],
                "target_text": scalar_text("target_text"),
                "ess_gold": ess_gold,
                "total_loss": float(total.detach().float()),
                "action_ce": float(action_loss.detach().float()),
                "ess_content_nll": float(ess_result["content_loss"].detach().float()),
                "ess_stop_nll": float(ess_result["stop_loss"].detach().float()),
                "ess_field_nll": {
                    name: float(value.detach().float())
                    for name, value in zip(FIELD_NAMES, ess_result["field_losses"])
                },
                "adjacent_cos": float(adjacent.detach().float()),
                "latent_rms": float(memory.detach().float().square().mean().sqrt()),
                "alpha": float(modules.alpha.detach().float()),
            }
        if not ess_active:
            metrics={k:v for k,v in metrics.items() if not k.startswith(('ess_','query_attn_'))}
        for k,v in metrics.items(): self.metric_sums[mode][k]+=float(v.detach().float()); self.metric_counts[mode][k]+=1
        return (total,action_out) if return_outputs else total

    def log(self, logs, start_time=None):
        metric_mode = "eval" if any(key.startswith("eval_") for key in logs) else "train"
        if metric_mode=='train':
            engine=getattr(self,'deepspeed',None)
            if engine is not None and hasattr(engine,'get_global_grad_norm'):
                value=engine.get_global_grad_norm()
                if value is not None:logs['grad_norm']=float(value)
        if self.metric_sums[metric_mode]:
            prefix = "eval/colt" if metric_mode == "eval" else "colt"
            payload={'sums':dict(self.metric_sums[metric_mode]),'counts':dict(self.metric_counts[metric_mode])}
            gathered=[payload]
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                gathered=[None]*torch.distributed.get_world_size()
                torch.distributed.all_gather_object(gathered,payload)
            totals=defaultdict(float); counts=defaultdict(int)
            for p in gathered:
                for key,val in p['sums'].items():totals[key]+=val
                for key,val in p['counts'].items():counts[key]+=val
            for key,total in totals.items():
                logs[f"{prefix}/{key}"]=total/counts[key]
            self.metric_sums[metric_mode].clear()
            self.metric_counts[metric_mode].clear()
        return super().log(logs, start_time=start_time)

    def prediction_step(
        self,
        model,
        inputs,
        prediction_loss_only,
        ignore_keys=None,
    ):
        """Evaluate the custom recurrent objective instead of model(**inputs).

        The boundary batch intentionally has no ordinary ``labels`` field, so
        HuggingFace's default prediction path would bypass ``compute_loss`` and
        send metadata such as ``prefix_end`` into Qwen's forward method.
        """
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad(), self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)
        return loss.detach().mean(), None, None

    def evaluate(self,*args,**kwargs):
        self.eval_decode_rows=[]
        metrics=super().evaluate(*args,**kwargs)
        rows=self.eval_decode_rows;self.eval_decode_rows=[]
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            parts=[None]*torch.distributed.get_world_size()
            torch.distributed.all_gather_object(parts,rows)
            rows=[r for part in parts for r in part]
        if self.is_world_process_zero():
            out=Path(self.args.output_dir)/'heldout_free'
            out.mkdir(exist_ok=True)
            report={'step':self.state.global_step,'decoder_input':'latent + fixed field-name prompt only; NO image/question/gold content','cases':[], 'decode_cases': int(getattr(self.args, 'eval_decode_cases', 20)), 'decode_max_new_tokens': int(getattr(self.args, 'eval_decode_max_new_tokens', 256))}
            try:
                dec=self._bare(self.model).colt_modules.ess_decoder
                selected={r['boundary_id']:r for r in rows}
                selected=[selected[k] for k in sorted(selected)[:max(0, int(getattr(self.args, 'eval_decode_cases', 20)))]]
                with torch.inference_mode():
                    for r in selected:
                        z=r['latents'].to(device=dec.field_queries.device,dtype=dec.field_queries.dtype)
                        prefixes,_=dec.read_fields(z)
                        result={k:v for k,v in r.items() if k!='latents'}
                        result['decoded_ess']={};result['raw']={}
                        for i,k in enumerate(FIELD_NAMES):
                            ids=torch.tensor([dec.tokenizer.encode(field_prompt_text(k),add_special_tokens=False)],device=z.device)
                            embeds=dec.lm.get_input_embeddings()(ids)
                            inp=torch.cat([prefixes[:,i:i+1].to(embeds.dtype),embeds],dim=1)
                            tokens=dec.lm.generate(inputs_embeds=inp,attention_mask=torch.ones(inp.shape[:2],dtype=torch.long,device=z.device),max_new_tokens=int(getattr(self.args, 'eval_decode_max_new_tokens', 256)),do_sample=False,use_cache=True,pad_token_id=dec.tokenizer.pad_token_id,eos_token_id=list({dec.tokenizer.eos_token_id,dec.tokenizer.convert_tokens_to_ids('<|im_end|>')}))
                            raw=dec.tokenizer.decode(tokens[0],skip_special_tokens=False)
                            result['raw'][k]=raw
                            result['decoded_ess'][k]=raw.split('</value>')[0].split('<|im_end|>')[0].strip()
                        report['cases'].append(result)
                report['success']=True
            except Exception as error:
                report.update(success=False,error=repr(error))
            (out/f'checkpoint-{self.state.global_step}.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
        if torch.distributed.is_available() and torch.distributed.is_initialized():torch.distributed.barrier()
        return metrics

    def save_model(self, output_dir=None, _internal_call=False):
        super().save_model(output_dir, _internal_call=_internal_call)
        if not self.is_world_process_zero():
            return
        output_dir = output_dir or self.args.output_dir
        causal = self._bare(self.model)
        modules = causal.colt_modules
        torch.save(
            {
                "transition": modules.transition.state_dict(),
                "alpha": modules.alpha.detach().cpu(),
                "stages": modules.stages,
                "protocol":"native_residual_boundary_ess_v2_stable",
                "prefix_history":"visible_history_only_no_prior_boundary_latent_replay",
            },
            os.path.join(output_dir, "colt_modules.pt"),
        )
        # The 0.5B decoder backbone is frozen and referenced by path.  Save
        # only its trainable LoRA adapters plus latent projection; otherwise
        # every 250-step checkpoint redundantly stores ~1.3 GB.
        compact_ess={k:v.detach().cpu() for k,v in modules.ess_decoder.state_dict().items()
                     if k.startswith("memory_proj.") or k.startswith("memory_norm.")
                     or k.startswith("query_attention.") or k.startswith("query_norm.")
                     or k.startswith("semantic_probe.")
                     or k=="field_queries" or k=="query_bias_scale" or "lora_" in k}
        torch.save(compact_ess,os.path.join(output_dir,"ar_ess_decoder.pt"))
        for name in ['run_manifest.json','data_audit.json']:
            src=Path(self.args.output_dir).parents[1]/'config'/name
            if src.exists():shutil.copy2(src,Path(output_dir)/name)

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        super()._load_from_checkpoint(resume_from_checkpoint, model=model)
        causal=self._bare(model if model is not None else self.model)
        p=Path(resume_from_checkpoint)
        state=torch.load(p/'colt_modules.pt',map_location='cpu',weights_only=False)
        causal.colt_modules.transition.load_state_dict(state['transition'],strict=True)
        causal.colt_modules.alpha.data.copy_(state['alpha'])
        sd=torch.load(p/'ar_ess_decoder.pt',map_location='cpu',weights_only=False)
        result=causal.colt_modules.ess_decoder.load_state_dict(sd,strict=False)
        trainable={k for k,v in causal.colt_modules.ess_decoder.named_parameters() if v.requires_grad}
        if result.unexpected_keys or trainable.intersection(result.missing_keys):
            raise RuntimeError(f'Incomplete ESS decoder checkpoint: {result}')


class TrajectoryCoLTTrainer(CoLTTrainer):
    """Chronological boundary replay with detached cross-boundary KV state."""

    def _get_train_sampler(self, dataset=None):
        dataset = dataset if dataset is not None else self.train_dataset
        if dataset is None or not hasattr(dataset, "__len__"):
            return None
        # Trainer may construct the dataloader before Accelerate initializes
        # the process group.  torchrun has already populated WORLD_SIZE/RANK,
        # so prefer those values; otherwise the sampler is built as a
        # single-replica sampler and later receives rank>0 from Accelerate,
        # causing members[self.rank] IndexError.
        env_world = os.environ.get("WORLD_SIZE")
        env_rank = os.environ.get("RANK")
        if env_world is not None and env_rank is not None:
            replicas = int(env_world)
            rank = int(env_rank)
        elif torch.distributed.is_available() and torch.distributed.is_initialized():
            replicas = torch.distributed.get_world_size()
            rank = torch.distributed.get_rank()
        else:
            replicas = 1
            rank = 0
        return HomogeneousBoundaryDistributedSampler(
            dataset,
            num_replicas=replicas,
            rank=rank,
            seed=int(self.args.seed),
        )

    def training_step(self, model, inputs, num_items_in_batch=None):
        """Stream backward once per boundary instead of retaining a full trajectory graph.

        Cross-boundary KV tensors are detached by the trajectory protocol, so the
        boundary graphs are intentionally independent.  Backpropagating each
        weighted boundary loss immediately is mathematically equivalent to one
        backward over their weighted sum, while allowing the just-finished graph
        (including ESS decoder logits) to be released before the next boundary.
        """
        model.train()
        if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
            self.optimizer.train()
        inputs = self._prepare_inputs(inputs)
        self._stream_boundary_backward = True
        try:
            with self.compute_loss_context_manager():
                loss = self.compute_loss(
                    model, inputs, num_items_in_batch=num_items_in_batch
                )
        finally:
            self._stream_boundary_backward = False
        del inputs
        return loss.detach() / max(1, int(self.args.gradient_accumulation_steps))

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        causal = self._bare(model)
        backbone = causal.model.model
        lm_head = causal.get_output_embeddings()
        modules = causal.colt_modules
        mode = "train" if model.training else "eval"

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        prefix_ends = inputs["prefix_ends"][0]
        target_span_starts = inputs["target_span_starts"]
        target_span_ends = inputs["target_span_ends"]
        action_active_flags = inputs["action_active_flags"]
        block_ends = inputs["block_ends"][0]
        ess_active = inputs["ess_active"][0]
        boundary_count = int(prefix_ends.numel())
        if not boundary_count:
            raise RuntimeError("empty trajectory")
        stream_backward = bool(
            model.training and getattr(self, "_stream_boundary_backward", False)
        )
        total_action_tokens = sum(
            end - start
            for starts, ends, active in zip(
                target_span_starts, target_span_ends, action_active_flags
            )
            for start, end, enabled in zip(starts, ends, active)
            if enabled
        )
        if total_action_tokens <= 0:
            raise RuntimeError(f"trajectory has no active action tokens: {inputs['trajectory_id']}")
        active_ess_boundaries = int(ess_active.long().sum().item())
        warm = max(1, int(self.ess_warmup_steps))
        ess_scale = self.ess_weight * min(
            1.0, float(self.state.global_step + 1) / warm
        )

        pixel_values, image_grid_thw = split_vision_inputs(
            inputs.get("pixel_values"), inputs.get("image_grid_thw")
        )
        base_embeddings = build_multimodal_embeddings(
            model, input_ids, pixel_values, image_grid_thw
        ).detach()
        positions = qwen_position_ids(
            model, input_ids, image_grid_thw, attention_mask
        )

        cache = None
        cursor = 0
        inserted = 0
        last_state = None
        action_weighted = base_embeddings.sum() * 0.0
        action_tokens = 0
        ess_losses = []
        adjacent_values = []
        latent_rms_values = []
        query_attention_values = []
        field_loss_values = []
        stop_loss_values = []
        content_loss_values = []
        per_kind = defaultdict(list)
        per_tool_action = defaultdict(list)
        per_tool_field = defaultdict(lambda: defaultdict(list))
        snapshots = []
        last_output = None

        for boundary_index in range(boundary_count):
            prefix_end = int(prefix_ends[boundary_index].item())
            span_starts = [int(value) for value in target_span_starts[boundary_index]]
            span_ends = [int(value) for value in target_span_ends[boundary_index]]
            span_active = [bool(value) for value in action_active_flags[boundary_index]]
            target_start = span_starts[0]
            target_end = int(block_ends[boundary_index].item())
            if not (cursor <= prefix_end <= target_start < target_end):
                raise RuntimeError(
                    f"bad boundary order at {boundary_index}: cursor={cursor} "
                    f"prefix={prefix_end} target=[{target_start},{target_end})"
                )

            if cursor < prefix_end:
                interval = base_embeddings[:, cursor:prefix_end]
                interval_positions = positions[..., cursor:prefix_end] + inserted
                with torch.no_grad():
                    prefix_outputs = backbone(
                        inputs_embeds=interval,
                        attention_mask=torch.ones(
                            (1, prefix_end + inserted),
                            dtype=attention_mask.dtype, device=input_ids.device,
                        ),
                        position_ids=interval_positions,
                        past_key_values=cache,
                        cache_position=(torch.arange(
                            cursor + inserted, prefix_end + inserted,
                            device=input_ids.device, dtype=torch.long,
                        ) if cache is not None else None),
                        use_cache=True,
                        output_hidden_states=False,
                        return_dict=True,
                    )
                cache = detach_kv_cache(prefix_outputs.past_key_values)
                last_state = prefix_outputs.last_hidden_state[:, -1].detach()
                cursor = prefix_end
                del prefix_outputs
            if last_state is None:
                raise RuntimeError("trajectory prefix produced no seed state")

            prefix_last_position = positions[..., prefix_end - 1:prefix_end] + inserted
            local_cache = cache
            current = last_state
            latents = []
            cached_len = prefix_end + inserted
            boundary_adjacent = []
            for stage in range(modules.stages):
                latent_outputs = backbone(
                    inputs_embeds=current.unsqueeze(1),
                    attention_mask=torch.ones(
                        (1, cached_len + 1),
                        dtype=attention_mask.dtype, device=input_ids.device,
                    ),
                    position_ids=prefix_last_position + stage + 1,
                    past_key_values=local_cache,
                    cache_position=torch.tensor(
                        [cached_len], device=input_ids.device, dtype=torch.long
                    ),
                    use_cache=True,
                    output_hidden_states=False,
                    return_dict=True,
                )
                local_cache = latent_outputs.past_key_values
                hidden = latent_outputs.last_hidden_state[:, -1]
                current = hidden + modules.alpha.to(
                    hidden.device, hidden.dtype
                ) * modules.transition(hidden)
                if latents:
                    value = F.cosine_similarity(
                        latents[-1].float(), current.float(), dim=-1
                    ).mean().detach()
                    adjacent_values.append(value)
                    boundary_adjacent.append(value)
                latents.append(current)
                cached_len += 1
                del latent_outputs

            memory = torch.stack(latents, dim=1)
            latent_rms_values.append(memory.float().square().mean().sqrt().detach())
            suffix_ids = input_ids[:, prefix_end:target_end]
            suffix_embeddings = base_embeddings[:, prefix_end:target_end]
            target_offset = target_start - prefix_end
            inserted_after = inserted + modules.stages + 1
            final_latent_position = prefix_last_position + modules.stages + 1
            action_inputs = torch.cat(
                [current.unsqueeze(1), suffix_embeddings[:, :-1]], dim=1
            )
            action_positions = torch.cat([
                final_latent_position,
                positions[..., prefix_end:target_end - 1] + inserted_after,
            ], dim=-1)
            action_outputs = backbone(
                inputs_embeds=action_inputs,
                attention_mask=torch.ones(
                    (1, cached_len + action_inputs.shape[1]),
                    dtype=attention_mask.dtype, device=input_ids.device,
                ),
                position_ids=action_positions,
                past_key_values=local_cache,
                cache_position=torch.arange(
                    cached_len, cached_len + action_inputs.shape[1],
                    device=input_ids.device, dtype=torch.long,
                ),
                use_cache=False,
                output_hidden_states=False,
                return_dict=True,
            )
            labels = torch.full_like(suffix_ids, -100)
            token_count = 0
            for start, end, enabled in zip(span_starts, span_ends, span_active):
                if not enabled:
                    continue
                labels[:, start - prefix_end:end - prefix_end] = suffix_ids[
                    :, start - prefix_end:end - prefix_end
                ]
                token_count += end - start
            if token_count:
                action_loss = chunked_token_ce(
                    action_outputs.last_hidden_state, labels, lm_head, self.chunk_size
                )
            else:
                action_loss = action_outputs.last_hidden_state.sum() * 0.0
            if stream_backward:
                action_weighted = action_weighted + action_loss.detach() * token_count
            else:
                action_weighted = action_weighted + action_loss * token_count
            action_tokens += token_count
            kind = str(inputs["kinds"][boundary_index])
            tool_label = str(inputs["target_tools"][boundary_index] or "answer")
            if tool_label.strip().lower() in {"", "none", "null"}:
                tool_label = "answer"
            # Keep a multi-call block as one explicit combination bucket.  A
            # single ESS supervises the whole combination, so attributing it
            # independently to each tool would double-count the loss.
            tool_label = tool_label.replace("/", "_").replace(" ", "_")
            if token_count:
                per_kind[kind].append(action_loss.detach())
                per_tool_action[tool_label].append(action_loss.detach())
            if not stream_backward:
                last_output = action_outputs

            # Always execute the ESS decoder on every rank and every boundary.
            # Some boundary types intentionally have ess_active=False.  Skipping
            # the decoder on those ranks changes the ZeRO/DDP autograd-hook
            # topology and can deadlock the collective.  Multiplication by the
            # scalar active flag below keeps its gradient exactly zero while
            # preserving an identical distributed graph.
            ess_result = modules.ess_decoder(
                memory,
                inputs["ess_input_ids"][:, boundary_index],
                inputs["ess_attention_mask"][:, boundary_index],
                labels=inputs["ess_labels"][:, boundary_index],
                content_mask=inputs["ess_content_mask"][:, boundary_index],
                stop_mask=inputs["ess_stop_mask"][:, boundary_index],
                field_weights=torch.tensor([1.5, 1.5, 1.0], device=memory.device),
                stop_weight=self.ess_stop_weight,
                semantic_weight=0.0,
            )
            boundary_ess_active = ess_active[boundary_index].reshape(()).to(
                ess_result["loss"].dtype
            )
            if bool(ess_active[boundary_index].item()):
                ess_losses.append(
                    ess_result["loss"].detach() if stream_backward else ess_result["loss"]
                )
                content_loss_values.append(ess_result["content_loss"].detach())
                stop_loss_values.append(ess_result["stop_loss"].detach())
                field_loss_values.append(ess_result["field_losses"].detach())
                for field_index, field_name in enumerate(FIELD_NAMES):
                    per_tool_field[tool_label][field_name].append(
                        ess_result["field_losses"][field_index].detach()
                    )
                query_attention_values.append(
                    ess_result["query_attention"].detach().float().mean(dim=(0, 1))
                )
                if mode == "eval" and hasattr(self, "eval_decode_rows"):
                    if len(self.eval_decode_rows) < 20:
                        self.eval_decode_rows.append({
                            "boundary_id": inputs["boundary_ids"][boundary_index],
                            "latents": memory.detach().cpu(),
                            "gold": json.loads(inputs["ess_gold_jsons"][boundary_index]),
                            "image_paths": inputs["image_paths"],
                        })

            snapshots.append({
                "boundary_index": boundary_index,
                "boundary_id": inputs["boundary_ids"][boundary_index],
                "kind": kind,
                "boundary_type": inputs["boundary_types"][boundary_index],
                "target_tool": inputs["target_tools"][boundary_index],
                "target_text": inputs["target_texts"][boundary_index],
                "target_spans": list(zip(span_starts, span_ends, span_active)),
                "ess_gold": json.loads(inputs["ess_gold_jsons"][boundary_index]),
                "action_ce": float(action_loss.detach().float()),
                "adjacent_cos": float(torch.stack(boundary_adjacent).mean())
                    if boundary_adjacent else float("nan"),
            })

            if stream_backward:
                boundary_loss = (
                    self.action_weight * action_loss * token_count
                    / max(1, total_action_tokens)
                )
                boundary_loss = boundary_loss + (
                    ess_scale * ess_result["loss"] * boundary_ess_active
                    / max(1, active_ess_boundaries)
                )
                if not bool(torch.isfinite(boundary_loss)):
                    raise FloatingPointError(
                        f"nonfinite boundary loss: {inputs['trajectory_id']} "
                        f"boundary={boundary_index}"
                    )
                scaled_boundary_loss = boundary_loss / max(
                    1, int(self.args.gradient_accumulation_steps)
                )
                backward_kwargs = (
                    {"scale_wrt_gas": False} if self.is_deepspeed_enabled else {}
                )
                self.accelerator.backward(scaled_boundary_loss, **backward_kwargs)

            if boundary_index + 1 < boundary_count:
                next_prefix = int(prefix_ends[boundary_index + 1].item())
                if next_prefix <= prefix_end:
                    raise RuntimeError("next boundary did not advance")
                advance_inputs = torch.cat([
                    current.detach().unsqueeze(1),
                    base_embeddings[:, prefix_end:next_prefix],
                ], dim=1)
                advance_positions = torch.cat([
                    final_latent_position,
                    positions[..., prefix_end:next_prefix] + inserted_after,
                ], dim=-1)
                with torch.no_grad():
                    advance_outputs = backbone(
                        inputs_embeds=advance_inputs,
                        attention_mask=torch.ones(
                            (1, cached_len + advance_inputs.shape[1]),
                            dtype=attention_mask.dtype, device=input_ids.device,
                        ),
                        position_ids=advance_positions,
                        past_key_values=detach_kv_cache(local_cache),
                        cache_position=torch.arange(
                            cached_len, cached_len + advance_inputs.shape[1],
                            device=input_ids.device, dtype=torch.long,
                        ),
                        use_cache=True,
                        output_hidden_states=False,
                        return_dict=True,
                    )
                cache = detach_kv_cache(advance_outputs.past_key_values)
                last_state = advance_outputs.last_hidden_state[:, -1].detach()
                cursor = next_prefix
                inserted = inserted_after
                del advance_outputs

            if stream_backward:
                # Do not keep references to a completed boundary graph.  The
                # detached cache above is the only state carried forward.
                del action_outputs, action_loss, action_inputs, memory, latents
                del ess_result
                del current, local_cache

        action_total = action_weighted / max(1, action_tokens)
        ess_total = torch.stack(ess_losses).mean() if ess_losses else action_total * 0.0
        total = self.action_weight * action_total + ess_scale * ess_total
        if not bool(torch.isfinite(total)):
            raise FloatingPointError(
                f"nonfinite trajectory loss: {inputs['trajectory_id']}"
            )

        metrics = {
            "action_ce": action_total.detach(),
            "ess_nll": ess_total.detach(),
            "ess_effective_weight": torch.tensor(ess_scale, device=total.device),
            "adjacent_cos": torch.stack(adjacent_values).mean()
                if adjacent_values else total.detach() * 0.0,
            "alpha": modules.alpha.detach(),
            "latent_rms": torch.stack(latent_rms_values).mean(),
            "boundaries_per_trajectory": torch.tensor(float(boundary_count), device=total.device),
            "raw_tokens": inputs["raw_token_count"].float().mean(),
            "physical_tokens": inputs["physical_token_count"].float().mean(),
        }
        for kind, values in per_kind.items():
            metrics[f"action_ce_{kind}"] = torch.stack(values).mean()
        for tool_label, values in per_tool_action.items():
            metrics[f"action/{tool_label}/ce"] = torch.stack(values).mean()
        if content_loss_values:
            field_matrix = torch.stack(field_loss_values)
            for field_index, name in enumerate(FIELD_NAMES):
                metrics[f"ess/{name}_nll"] = field_matrix[:, field_index].mean()
            for tool_label, fields in per_tool_field.items():
                for field_name, values in fields.items():
                    metrics[f"ess_by_tool/{field_name}/{tool_label}"] = torch.stack(values).mean()
            attention = torch.stack(query_attention_values).mean(dim=0)
            for field_index, name in enumerate(FIELD_NAMES):
                for latent_index in range(attention.shape[-1]):
                    metrics[f"query_attn_{name}_z{latent_index + 1}"] = attention[field_index, latent_index]
        for key, value in metrics.items():
            self.metric_sums[mode][key] += float(value.detach().float())
            self.metric_counts[mode][key] += 1

        if mode == "train":
            worst = max(snapshots, key=lambda item: item["action_ce"])
            self.last_case_snapshot = {
                **worst,
                "trajectory_id": inputs["trajectory_id"],
                "question": inputs["question_text"],
                "image_paths": [str(path) for path in inputs["image_paths"]],
                "total_loss": float(total.detach().float()),
                "ess_content_nll": float(torch.stack(content_loss_values).mean())
                    if content_loss_values else 0.0,
                "ess_stop_nll": float(torch.stack(stop_loss_values).mean())
                    if stop_loss_values else 0.0,
                "latent_rms": float(torch.stack(latent_rms_values).mean()),
                "alpha": float(modules.alpha.detach().float()),
                "boundary_count": boundary_count,
                "raw_token_count": int(inputs["raw_token_count"].item()),
                "physical_token_count": int(inputs["physical_token_count"].item()),
            }
        return (total, last_output) if return_outputs else total

    def save_model(self, output_dir=None, _internal_call=False):
        super().save_model(output_dir, _internal_call=_internal_call)
        if not self.is_world_process_zero():
            return
        output_dir = output_dir or self.args.output_dir
        modules = self._bare(self.model).colt_modules
        torch.save({
            "transition": modules.transition.state_dict(),
            "alpha": modules.alpha.detach().cpu(),
            "stages": modules.stages,
            "protocol": "trajectory_context_boundary_tbptt_v1",
            "cross_boundary_gradient": "detached",
            "cross_boundary_state": "prior latent KV retained",
            "within_boundary_gradient": "full recurrent BPTT",
            "cache": "incremental native KV",
        }, os.path.join(output_dir, "colt_modules.pt"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--decoder-model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--eval-data", default="")
    ap.add_argument("--image-root", default="")
    ap.add_argument("--system-prompt", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--stages", type=int, default=4)
    ap.add_argument("--max-length", type=int, default=8192)
    ap.add_argument("--max-images", type=int, default=4)
    ap.add_argument("--max-answer-tokens", type=int, default=2048)
    ap.add_argument("--ess-max-length", type=int, default=256)
    ap.add_argument("--max-pixels", type=int, default=3211264)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--num-train-epochs", type=float, default=1.0)
    ap.add_argument("--gradient-accumulation-steps", type=int, default=8)
    ap.add_argument("--learning-rate", type=float, default=1e-5)
    ap.add_argument("--aux-learning-rate", type=float, default=1e-5)
    ap.add_argument("--ess-learning-rate", type=float, default=5e-5)
    ap.add_argument("--ess-weight", type=float, default=0.02)
    ap.add_argument("--semantic-weight", type=float, default=0.0)
    ap.add_argument("--ess-stop-weight", type=float, default=0.2)
    ap.add_argument("--ess-warmup-steps", type=int, default=100)
    ap.add_argument("--action-weight", type=float, default=1.0)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--save-steps", type=int, default=250)
    ap.add_argument("--eval-steps", type=int, default=250)
    ap.add_argument("--eval-decode-cases", type=int, default=20,
                    help="Number of held-out boundaries to free-decode after eval; 0 disables free decoding.")
    ap.add_argument("--eval-decode-max-new-tokens", type=int, default=256)
    ap.add_argument("--logging-steps", type=int, default=1)
    ap.add_argument("--dataloader-workers", type=int, default=2)
    ap.add_argument("--lr-scheduler-type", default="constant_with_warmup")
    ap.add_argument("--high-grad-threshold", type=float, default=20.0)
    ap.add_argument("--high-grad-max-events", type=int, default=32)
    ap.add_argument("--high-grad-dir", default="")
    ap.add_argument("--deepspeed", default="")
    ap.add_argument("--resume-from-checkpoint", default="")
    ap.add_argument("--data-audit-out", default="")
    ap.add_argument("--length-bucket-width", type=int, default=256,
                    help="Physical-token width for synchronized length buckets.")
    args = ap.parse_args()
    torch.manual_seed(20260831)
    assert args.semantic_weight == 0

    with open(args.system_prompt, encoding="utf-8") as handle:
        system_prompt = handle.read().strip()
    processor = Qwen2_5_VLProcessor.from_pretrained(
        args.model,
        min_pixels=0,
        max_pixels=args.max_pixels,
        trust_remote_code=True,
    )
    decoder_tokenizer = AutoTokenizer.from_pretrained(args.decoder_model, use_fast=False)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
    )
    freeze_vision(model)
    model = get_peft_model(
        model,
        LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    freeze_vision(model)
    ess_decoder = IndependentFieldQueryDecoder(args.decoder_model,int(model.config.hidden_size),
                                                latent_steps=args.stages,lora_r=16,lora_alpha=32)
    model.colt_modules = CoLTModules(
        int(model.config.hidden_size), ess_decoder, args.stages
    ).to(dtype=torch.bfloat16)
    model.colt_modules.alpha.data=model.colt_modules.alpha.data.float()
    model.colt_modules.ess_decoder.query_bias_scale.data=model.colt_modules.ess_decoder.query_bias_scale.data.float()
    model.print_trainable_parameters()

    audit_out = args.data_audit_out or os.path.join(
        args.output_dir, "trajectory_data_audit.json"
    )
    dataset = TrajectoryDataset(
        args.data,
        processor,
        decoder_tokenizer,
        system_prompt,
        args.max_length,
        args.max_answer_tokens,
        args.ess_max_length,
        args.stages,
        args.max_images,
        args.image_root,
        audit_out,
        args.length_bucket_width,
    )
    eval_dataset = None
    if args.eval_data:
        eval_dataset = TrajectoryDataset(
            args.eval_data,
            processor,
            decoder_tokenizer,
            system_prompt,
            args.max_length,
            args.max_answer_tokens,
            args.ess_max_length,
            args.stages,
            args.max_images,
            args.image_root,
            os.path.join(args.output_dir, "trajectory_eval_data_audit.json"),
            args.length_bucket_width,
        )
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_strategy="steps",
        eval_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=args.eval_steps if eval_dataset is not None else None,
        save_total_limit=4,
        bf16=True,
        max_grad_norm=1.0,
        warmup_ratio=0.02,
        lr_scheduler_type=args.lr_scheduler_type,
        weight_decay=1e-6,
        remove_unused_columns=False,
        report_to=["tensorboard"],
        deepspeed=args.deepspeed or None,
        dataloader_num_workers=args.dataloader_workers,
        dataloader_prefetch_factor=2 if args.dataloader_workers else None,
        ddp_find_unused_parameters=False,
        prediction_loss_only=True,
    )
    trainer = TrajectoryCoLTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        data_collator=TrajectoryCollator(),
        backward_decoder=None,
        action_weight=args.action_weight,
        main_lr=args.learning_rate,
        aux_lr=args.aux_learning_rate,
        ess_lr=args.ess_learning_rate,
        ess_weight=args.ess_weight,
        semantic_weight=args.semantic_weight,
        ess_stop_weight=args.ess_stop_weight,
        ess_warmup_steps=args.ess_warmup_steps,
    )
    high_grad_dir = args.high_grad_dir or os.path.join(args.output_dir, "high_grad_cases")
    trainer.add_callback(HighGradDumpCallback(
        trainer,
        high_grad_dir,
        threshold=args.high_grad_threshold,
        max_events=args.high_grad_max_events,
    ))
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint or None)
    trainer.save_model(os.path.join(args.output_dir, "final"))


if __name__ == "__main__":
    main()
