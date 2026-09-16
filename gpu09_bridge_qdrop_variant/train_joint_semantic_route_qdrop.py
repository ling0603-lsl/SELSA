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
import contextlib
import hashlib
import json
import math
import os
import random
import re
import shutil
from collections import Counter, OrderedDict, defaultdict
from pathlib import Path

import torch
if "LOCAL_RANK" in os.environ:
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
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
    "disambiguation": "State the visual detail that distinguishes the target from alternatives.",
    "grounding": "Describe the target's spatial grounding in natural language.",
    "need": "State what visual evidence the next action must obtain.",
}

BRIDGE_TARGET_SCHEMA = "ess_field_decision_hidden_v1"


def normalize_bridge_text(value):
    return " ".join(str(value or "").strip().split())


def bridge_target_key(field, text):
    payload = BRIDGE_TARGET_SCHEMA + "\0" + str(field) + "\0" + normalize_bridge_text(text)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

def prepare_ess_target(row):
    """Return the frozen ESS target used by both decoding and bridge alignment."""
    if not bool(row.get("ess_active", True)):
        return ({name: "" for name in FIELD_NAMES}, set(), False)
    target = row.get("ess")
    if not isinstance(target, dict):
        raise RuntimeError(f"missing ESS for {row.get('boundary_id', '<unknown>')}")
    return ({name: str(target.get(name, "")).strip() for name in FIELD_NAMES},
            set(FIELD_NAMES), True)


def boundary_flag(row, name, default=True):
    value = row.get(name, default)
    if isinstance(value, (list, tuple)):
        value = value[0] if value else default
    return bool(value)


def field_prompt_text(name):
    return (
        "<|im_start|>system\n"
        "Read only the recurrent latent state. Return the requested executable "
        "semantic field and no other field.\n"
        "<|im_end|>\n<|im_start|>assistant\n"
        f"<field>{name}</field><value>"
    )


def encode_independent_fields(tokenizer, ess, max_length=96, active_fields=None):
    """Encode independent fields with an explicit per-field supervision mask."""
    active_fields = set(FIELD_NAMES) if active_fields is None else set(active_fields)
    encoded = []
    for name in FIELD_NAMES:
        value = str((ess or {}).get(name, "")).strip()
        if name in active_fields and not value:
            raise ValueError(f"empty ESS field: {name}")
        prompt = field_prompt_text(name)
        tail = "</value><|im_end|>"
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        value_ids = tokenizer.encode(value, add_special_tokens=False) if name in active_fields else []
        tail_ids = tokenizer.encode(tail, add_special_tokens=False)
        room = int(max_length) - len(prompt_ids) - len(tail_ids)
        if room <= 0:
            raise ValueError("ESS field max length too small")
        if len(value_ids) > room:
            raise ValueError(f'ESS field exceeds audited token budget: {name} {len(value_ids)} > {room}')
        ids = prompt_ids + value_ids + tail_ids
        labels = ([-100] * len(prompt_ids) + value_ids + tail_ids
                  if name in active_fields else [-100] * (len(prompt_ids) + len(tail_ids)))
        content = ([0.0] * len(prompt_ids) + [1.0] * len(value_ids) + [0.0] * len(tail_ids)
                   if name in active_fields else [0.0] * (len(prompt_ids) + len(tail_ids)))
        stop = ([0.0] * (len(prompt_ids) + len(value_ids)) + [1.0] * len(tail_ids)
                if name in active_fields else [0.0] * (len(prompt_ids) + len(tail_ids)))
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


def split_vision_inputs(pixel_values, image_grid_thw):
    """Normalize the audited image-less run_code exception to text-only input."""
    if pixel_values is not None and int(pixel_values.numel()) == 0:
        return None, None
    if image_grid_thw is not None and int(image_grid_thw.numel()) == 0:
        return pixel_values, None
    return pixel_values, image_grid_thw


def locate_action_span(tokenizer, sequence, target_ids, min_start):
    """Locate an action span, tolerating one BPE merge on its left edge."""
    target_ids = [int(value) for value in target_ids]
    length = len(target_ids)
    if length == 0:
        raise ValueError("empty target token sequence")
    for start in range(int(min_start), len(sequence) - length + 1):
        if sequence[start:start + length] == target_ids:
            return start, start + length, None
    if length > 1:
        tail_ids = target_ids[1:]
        head_text = tokenizer.decode([target_ids[0]], skip_special_tokens=False)
        for start in range(max(1, int(min_start)), len(sequence) - length + 2):
            if sequence[start:start + length - 1] != tail_ids:
                continue
            previous_text = tokenizer.decode(
                [sequence[start - 1]], skip_special_tokens=False
            )
            if not previous_text.endswith(head_text):
                continue
            left_text = previous_text[:len(previous_text) - len(head_text)]
            if left_text:
                return start, start + length, (start - 1, left_text, head_text)
    raise ValueError(
        f"target token span not found after cursor: target_len={length}, "
        f"sequence_len={len(sequence)}, min_start={int(min_start)}"
    )


def _split_token_text(tokenizer, token_id, left_text, right_text):
    combined = left_text + right_text
    original = tokenizer.decode([int(token_id)], skip_special_tokens=False)
    if original != combined:
        raise ValueError(f"token split does not reproduce text: {original!r} != {combined!r}")
    left_ids = tokenizer.encode(left_text, add_special_tokens=False)
    right_ids = tokenizer.encode(right_text, add_special_tokens=False)
    if not left_ids or not right_ids:
        raise ValueError(f"empty token split for {combined!r}")
    if tokenizer.decode(left_ids + right_ids, skip_special_tokens=False) != combined:
        raise ValueError(f"token split is not text preserving: {combined!r}")
    return left_ids, right_ids


def apply_token_splits(tokenizer, sequence, mask, splits):
    tokens, masks = [], []
    for index, token_id in enumerate(sequence):
        if index in splits:
            left_ids, right_ids = _split_token_text(
                tokenizer, token_id, *splits[index]
            )
            replacement = left_ids + right_ids
            if len(replacement) < 2:
                raise ValueError("token split produced fewer than two tokens")
            tokens.extend(replacement)
            masks.extend([int(mask[index])] * len(replacement))
        else:
            tokens.append(int(token_id))
            masks.append(int(mask[index]))
    return tokens, masks


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
        ess_target, active_fields, ess_is_active = prepare_ess_target(row)
        ess_tensors=encode_independent_fields(
            self.ess_tokenizer, ess_target, self.ess_max_length, active_fields)
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
            "ess_gold_json": json.dumps(ess_target, ensure_ascii=False),
        }
        result.update(ess_tensors)
        result['ess_active'] = torch.tensor(float(ess_is_active))
        result['action_active'] = torch.tensor(
            float(boundary_flag(row, 'action_active'))
        )
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


class TrajectoryDataset(Dataset):
    """Group boundary rows by parent and encode each full trajectory once."""

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
        include_bridge_align=False,
        include_qdrop=False,
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
        self.include_bridge_align = bool(include_bridge_align)
        self.include_qdrop = bool(include_qdrop)
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
                    estimate = int(audited_end)
                else:
                    target_chars = sum(
                        len(str(value or ""))
                        for value in (row.get("target_texts") or [])
                    )
                    text_chars = sum(
                        len(str(row.get(name) or ""))
                        for name in ("question", "history_prefix", "bridge_text", "block_text")
                    ) + target_chars
                    # Fallback only for legacy rows without preflight metadata.
                    estimate = int(text_chars / 3.5) + 900 * len(row.get("images") or [])
                length_estimates[parent_id] = max(length_estimates[parent_id], estimate)

        self.groups = []
        quarantine = []
        inserted_tokens_per_boundary = self.stages + len(FIELD_NAMES) + 1
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
            estimated_physical_tokens = (
                int(length_estimates[parent_id])
                + inserted_tokens_per_boundary * len(entries)
            )
            if estimated_physical_tokens > self.max_length:
                quarantine.append({
                    "parent_id": parent_id,
                    "reason": "tokens_over_limit",
                    "physical_tokens": estimated_physical_tokens,
                    "limit": self.max_length,
                })
                continue
            self.groups.append((parent_id, tuple(item[1] for item in entries)))
        self.groups.sort(key=lambda item: item[0])
        self._length_estimates = {
            parent_id: int(length_estimates[parent_id])
            + inserted_tokens_per_boundary * len(offsets)
            for parent_id, offsets in self.groups
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
        """Align every supervised target and latent insertion point to tokens."""
        tokenizer = self.processor.tokenizer
        sequence = [int(value) for value in input_ids]
        mask = [int(value) for value in attention_mask]
        for _ in range(8):
            splits = {}
            layout = []
            cursor = int(prompt_len)
            for row in rows:
                target_texts = list(row.get("target_texts") or [])
                if len(target_texts) != 1:
                    raise RuntimeError(
                        f"Bridge v1 requires one target per boundary: {row['boundary_id']}"
                    )
                target_ids = tokenizer.encode(
                    target_texts[0], add_special_tokens=False
                )
                if row.get("kind") == "answer" and len(target_ids) > self.max_answer_tokens:
                    raise RuntimeError(
                        f"unfiltered overlength answer {row['boundary_id']}: "
                        f"{len(target_ids)} > {self.max_answer_tokens} tokens"
                    )
                start, end, split = locate_action_span(
                    tokenizer, sequence, target_ids, cursor
                )
                if split is not None:
                    splits[split[0]] = (split[1], split[2])
                bridge_text = str(row.get("bridge_text") or "")
                bridge_ids = tokenizer.encode(bridge_text, add_special_tokens=False)
                prefix_end = start - len(bridge_ids)
                if bridge_ids and prefix_end >= 0:
                    window = tokenizer.decode(
                        sequence[prefix_end:start], skip_special_tokens=False
                    )
                    if window != bridge_text:
                        merged_text = tokenizer.decode(
                            [sequence[prefix_end]], skip_special_tokens=False
                        )
                        head_text = tokenizer.decode(
                            [bridge_ids[0]], skip_special_tokens=False
                        )
                        if merged_text.endswith(head_text) and len(merged_text) > len(head_text):
                            splits[prefix_end] = (
                                merged_text[:len(merged_text) - len(head_text)],
                                head_text,
                            )
                layout.append((start, end, prefix_end))
                cursor = end
            if not splits:
                for row, (start, end, prefix_end) in zip(rows, layout):
                    bridge_text = str(row.get("bridge_text") or "")
                    window = tokenizer.decode(
                        sequence[prefix_end:start], skip_special_tokens=False
                    )
                    if window != bridge_text:
                        raise RuntimeError(
                            f"latent insertion point is not a token boundary: "
                            f"{row['boundary_id']} window={window!r} bridge={bridge_text!r}"
                        )
                    span_text = tokenizer.decode(
                        sequence[start:end], skip_special_tokens=False
                    )
                    if span_text != row["target_texts"][0]:
                        raise RuntimeError(
                            f"action span mismatch after realignment: "
                            f"{row['boundary_id']} span={span_text!r} "
                            f"target={row['target_texts'][0]!r}"
                        )
                return sequence, mask, layout
            sequence, mask = apply_token_splits(
                tokenizer, sequence, mask, splits
            )
        raise RuntimeError(
            f"could not realign action spans: {rows[0]['boundary_id']}"
        )

    def __getitem__(self, index):
        trajectory_id, offsets = self.groups[index]
        rows = [self._row_at(offset) for offset in offsets]
        rows.sort(key=lambda row: int(row["segment_index"]))
        for row in rows:
            if "target_text" not in row:
                target_texts = row.get("target_texts")
                if not isinstance(target_texts, list) or len(target_texts) != 1:
                    raise RuntimeError(
                        f"Bridge trainer requires one target action per boundary: {row.get('boundary_id')}"
                )
                row["target_text"] = str(target_texts[0])
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
            raise RuntimeError(
                f"image-less exception changed inside trajectory: {trajectory_id}"
            )
        if image_less_flags[0]:
            tools = [
                tool
                for row in rows
                for tool in str(row.get("target_tool") or "").split("+")
                if tool and tool != "none"
            ]
            if not tools or any(tool != "run_code" for tool in tools):
                raise RuntimeError(
                    f"invalid image-less non-run_code trajectory: {trajectory_id}"
                )
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
        target_starts = []
        target_ends = []
        ess_encoded = []
        ess_active_flags = []
        action_active_flags = []
        ess_targets = []
        bridge_align_rows = [] if self.include_bridge_align else None
        for row, (target_start, target_end, prefix_end) in zip(rows, layout):
            target_ids = self.processor.tokenizer.encode(
                row["target_text"], add_special_tokens=False
            )
            if row.get("kind") == "answer" and len(target_ids) > self.max_answer_tokens:
                raise RuntimeError(
                    f"unfiltered overlength answer {row['boundary_id']}: "
                    f"{len(target_ids)} > {self.max_answer_tokens} tokens"
                )
            if prefix_end < cursor:
                raise RuntimeError(f"bad trajectory bridge alignment: {row['boundary_id']}")
            prefix_ends.append(prefix_end)
            target_starts.append(target_start)
            target_ends.append(target_end)
            cursor = target_end
            label, active_fields, ess_is_active = prepare_ess_target(row)
            ess_encoded.append(encode_independent_fields(
                self.ess_tokenizer, label, self.ess_max_length, active_fields
            ))
            ess_active_flags.append(float(ess_is_active))
            action_active_flags.append(
                float(boundary_flag(row, 'action_active'))
            )
            ess_targets.append(label)
            if self.include_bridge_align:
                field_id_lists = []
                field_key_list = []
                for align_name in FIELD_NAMES:
                    align_text = normalize_bridge_text(label.get(align_name, ""))
                    if not align_text:
                        field_id_lists.append([])
                        field_key_list.append("")
                        continue
                    align_ids = self.processor.tokenizer.encode(
                        align_text, add_special_tokens=False
                    )
                    if len(align_ids) > self.ess_max_length:
                        raise RuntimeError(
                            f"bridge align field exceeds budget: "
                            f"{align_name} {len(align_ids)} > {self.ess_max_length}"
                        )
                    field_id_lists.append(align_ids)
                    field_key_list.append(bridge_target_key(align_name, align_text))
                bridge_align_rows.append((field_id_lists, field_key_list))

        qdrop_encoded = None
        qdrop_sequence = qdrop_attention_sequence = None
        qdrop_prefix_ends = qdrop_target_starts = qdrop_target_ends = None
        if self.include_qdrop:
            # Q-Drop uses the identical image/history/action trajectory but
            # masks the natural-language question.  It must use the same BPE
            # boundary repair as the normal stream: bridge text can merge with
            # the first target token, so exact subsequence search is invalid.
            clean_q = clean_question(last["question"])
            q_img_count = clean_q.count("<image>")
            qdrop_question = (
                ("<image>\n" * q_img_count)
                + "[Task question masked. Execute the next action from visual evidence and latent state.]"
            )
            qdrop_encoded = encode_full(
                self.processor, qdrop_question, resolved_images, response,
                system=self.system_prompt, build_labels=False,
            )
            qdrop_sequence, qdrop_attention_sequence, qdrop_layout = (
                self._align_action_spans(
                    rows,
                    qdrop_encoded.input_ids.tolist(),
                    qdrop_encoded.attention_mask.tolist(),
                    int(qdrop_encoded.prompt_len),
                )
            )
            qdrop_target_starts = [item[0] for item in qdrop_layout]
            qdrop_target_ends = [item[1] for item in qdrop_layout]
            qdrop_prefix_ends = [item[2] for item in qdrop_layout]

        inserted_tokens_per_boundary = self.stages + len(FIELD_NAMES) + 1
        physical_tokens = (
            int(target_ends[-1]) + inserted_tokens_per_boundary * len(rows)
        )
        if physical_tokens > self.max_length:
            raise RuntimeError(
                f"trajectory exceeds max tokens after latent insertion: "
                f"{trajectory_id} {physical_tokens} > {self.max_length}"
            )
        if int(target_ends[-1]) > len(token_sequence):
            raise RuntimeError(f"target beyond encoded sequence: {trajectory_id}")
        align_ids_tensor = None
        align_mask_tensor = None
        align_key_bytes_tensor = None
        if self.include_bridge_align:
            align_width = int(self.ess_max_length)
            pad_id = int(self.processor.tokenizer.pad_token_id)
            if pad_id is None or pad_id < 0:
                pad_id = int(self.processor.tokenizer.eos_token_id)
            align_ids_tensor = torch.full(
                (len(rows), len(FIELD_NAMES), align_width),
                pad_id, dtype=torch.long,
            )
            align_mask_tensor = torch.zeros(
                (len(rows), len(FIELD_NAMES), align_width),
                dtype=torch.float32,
            )
            align_key_bytes_tensor = torch.zeros(
                (len(rows), len(FIELD_NAMES), 32), dtype=torch.uint8,
            )
            for row_index, (field_id_lists, field_key_list) in enumerate(bridge_align_rows):
                for field_index, ids in enumerate(field_id_lists):
                    if not ids:
                        continue
                    align_ids_tensor[row_index, field_index, :len(ids)] = (
                        torch.tensor(ids, dtype=torch.long)
                    )
                    align_mask_tensor[row_index, field_index, :len(ids)] = 1.0
                    key = field_key_list[field_index]
                    if key:
                        align_key_bytes_tensor[row_index, field_index] = torch.tensor(
                            list(bytes.fromhex(key)), dtype=torch.uint8,
                        )

        result = {
            "input_ids": torch.tensor(token_sequence, dtype=torch.long),
            "attention_mask": torch.tensor(attention_sequence, dtype=torch.long),
            "pixel_values": encoded.pixel_values,
            "image_grid_thw": encoded.image_grid_thw,
            "prefix_ends": torch.tensor(prefix_ends, dtype=torch.long),
            "target_starts": torch.tensor(target_starts, dtype=torch.long),
            "target_ends": torch.tensor(target_ends, dtype=torch.long),
            "ess_active": torch.tensor(ess_active_flags, dtype=torch.float32),
            "action_active": torch.tensor(
                action_active_flags, dtype=torch.float32
            ),
            "ess_input_ids": torch.stack([item["ess_input_ids"] for item in ess_encoded]),
            "ess_attention_mask": torch.stack([item["ess_attention_mask"] for item in ess_encoded]),
            "ess_labels": torch.stack([item["ess_labels"] for item in ess_encoded]),
            "ess_content_mask": torch.stack([item["ess_content_mask"] for item in ess_encoded]),
            "ess_stop_mask": torch.stack([item["ess_stop_mask"] for item in ess_encoded]),
            # Stop target is a boundary decision, not the ESS </value> stop
            # token.  Answer boundaries terminate the tool trajectory.
            "stop_targets": torch.tensor([
                1.0 if str(row.get("kind")) == "answer" or
                str(row.get("boundary_type")) in {"post_tool_answer", "direct_answer", "pre_answer"}
                else 0.0 for row in rows
            ], dtype=torch.float32),
            "trajectory_id": trajectory_id,
            "boundary_ids": [row["boundary_id"] for row in rows],
            "kinds": [row["kind"] for row in rows],
            "boundary_types": [row["boundary_type"] for row in rows],
            "target_tools": [row["target_tool"] or "none" for row in rows],
            "question_text": rows[0]["question"],
            "image_paths": resolved_images,
            "target_texts": [row["target_text"] for row in rows],
            "ess_gold_jsons": [json.dumps(label, ensure_ascii=False) for label in ess_targets],
            "raw_token_count": torch.tensor(int(target_ends[-1]), dtype=torch.long),
            "physical_token_count": torch.tensor(physical_tokens, dtype=torch.long),
        }
        if self.include_qdrop:
            result.update({
                "qdrop_input_ids": torch.tensor(qdrop_sequence, dtype=torch.long),
                "qdrop_attention_mask": torch.tensor(
                    qdrop_attention_sequence, dtype=torch.long
                ),
                "qdrop_prompt_len": torch.tensor(
                    int(qdrop_encoded.prompt_len), dtype=torch.long
                ),
                "qdrop_prefix_ends": torch.tensor(
                    qdrop_prefix_ends, dtype=torch.long
                ),
                "qdrop_target_starts": torch.tensor(
                    qdrop_target_starts, dtype=torch.long
                ),
                "qdrop_target_ends": torch.tensor(
                    qdrop_target_ends, dtype=torch.long
                ),
            })
        if self.include_bridge_align:
            result["bridge_align_ids"] = align_ids_tensor
            result["bridge_align_mask"] = align_mask_tensor
            result["bridge_align_key_bytes"] = align_key_bytes_tensor
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


class SemanticActionBridge(nn.Module):
    """Map the ESS decoder's actual field states into main-backbone soft tokens."""

    def __init__(self, decoder_hidden, main_hidden, bottleneck=512, gate_init=0.2):
        super().__init__()
        self.input_norm = nn.LayerNorm(int(decoder_hidden))
        self.down = nn.Linear(int(decoder_hidden), int(bottleneck), bias=False)
        self.up = nn.Linear(int(bottleneck), int(main_hidden), bias=False)
        self.output_norm = nn.LayerNorm(int(main_hidden))
        self.field_type = nn.Parameter(torch.empty(len(FIELD_NAMES), int(main_hidden)))
        gate_init = min(max(float(gate_init), 1e-4), 1.0 - 1e-4)
        self.gate_logit = nn.Parameter(
            torch.tensor(math.log(gate_init / (1.0 - gate_init)), dtype=torch.float32)
        )
        nn.init.normal_(self.down.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.up.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.field_type, mean=0.0, std=0.02)

    def forward(self, field_states, latent_reference):
        dtype = self.down.weight.dtype
        value = self.up(F.gelu(self.down(self.input_norm(field_states.to(dtype)))))
        value = self.output_norm(value + self.field_type.unsqueeze(0).to(value.dtype))
        latent_rms = latent_reference.detach().float().square().mean().sqrt().clamp_min(1e-6)
        gate = torch.sigmoid(self.gate_logit).to(value.device, value.dtype)
        return value * latent_rms.to(value.device, value.dtype) * gate


class CoLTModules(nn.Module):
    def __init__(self, main_hidden, ess_decoder, stages, bridge_bottleneck=512, bridge_gate_init=0.2):
        super().__init__()
        self.stages = int(stages)
        self.transition = nn.Sequential(
            nn.Linear(main_hidden, main_hidden // 2),
            nn.GELU(),
            nn.Linear(main_hidden // 2, main_hidden),
            nn.LayerNorm(main_hidden),
        )
        self.alpha = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
        # Boundary-level controller: 1 means the next action should terminate
        # the tool trajectory and enter an answer, 0 means continue with a
        # tool call.  This is distinct from ESS's </value> stop token loss.
        self.stop_head = nn.Linear(main_hidden, 1)
        self.ess_decoder = ess_decoder
        self.semantic_action_bridge = SemanticActionBridge(
            int(self.ess_decoder.lm.config.hidden_size), main_hidden,
            bottleneck=bridge_bottleneck, gate_init=bridge_gate_init,
        )


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
        (case_dir / "report.md").write_text("\n".join(report), encoding="utf-8")
        return control


def freeze_vision(model):
    for name, parameter in model.named_parameters():
        if "visual" in name or "vision" in name:
            parameter.requires_grad = False


class ChunkedBridgeTargetStore:
    """Lazy reader for the immutable rank/chunk hidden-state cache."""

    def __init__(self, root, max_cached_chunks=8):
        self.root = Path(root)
        self.manifest_dir = self.root / "manifest"
        self.cache_dir = self.root / "cache"
        report_path = self.manifest_dir / "manifest_report.json"
        if not report_path.is_file():
            raise RuntimeError(f"missing bridge manifest report: {report_path}")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        self.schema = str(report.get("schema", ""))
        if self.schema != BRIDGE_TARGET_SCHEMA:
            raise RuntimeError(
                f"bridge target schema mismatch: {self.schema!r} != {BRIDGE_TARGET_SCHEMA!r}"
            )
        self.world_size = int(report.get("world_size", 0))
        self.expected = int(report.get("unique_total", 0))
        if self.world_size <= 0 or self.expected <= 0:
            raise RuntimeError("bridge target manifest is empty")
        self.index = {}
        for rank in range(self.world_size):
            path = self.manifest_dir / f"manifest_rank{rank}.jsonl"
            if not path.is_file():
                raise RuntimeError(f"missing bridge manifest shard: {path}")
            with path.open(encoding="utf-8") as handle:
                for row_index, line in enumerate(handle):
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    key = str(row.get("key", ""))
                    field = str(row.get("field", ""))
                    if len(key) != 64 or field not in FIELD_NAMES:
                        raise RuntimeError(f"invalid bridge manifest row in {path}")
                    if key in self.index:
                        raise RuntimeError(f"duplicate bridge target key: {key}")
                    self.index[key] = (rank, row_index // 8192, row_index % 8192)
        if len(self.index) != self.expected:
            raise RuntimeError(
                f"bridge target coverage mismatch: indexed={len(self.index)} expected={self.expected}"
            )
        self.max_cached_chunks = max(1, int(max_cached_chunks))
        self._chunks = OrderedDict()

    def _load_chunk(self, rank, chunk_id):
        cache_key = (int(rank), int(chunk_id))
        payload = self._chunks.get(cache_key)
        if payload is not None:
            self._chunks.move_to_end(cache_key)
            return payload
        path = self.cache_dir / f"rank{rank}_chunk{chunk_id:05d}.pt"
        if not path.is_file():
            raise RuntimeError(f"missing bridge target chunk: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("schema") != BRIDGE_TARGET_SCHEMA:
            raise RuntimeError(f"bridge target chunk schema mismatch: {path}")
        hidden = payload.get("hidden")
        keys = payload.get("keys")
        if not torch.is_tensor(hidden) or hidden.ndim != 2 or hidden.shape[1] != 3584:
            raise RuntimeError(f"invalid hidden shape in bridge target chunk: {path}")
        if not isinstance(keys, list) or len(keys) != hidden.shape[0]:
            raise RuntimeError(f"invalid key count in bridge target chunk: {path}")
        self._chunks[cache_key] = payload
        self._chunks.move_to_end(cache_key)
        while len(self._chunks) > self.max_cached_chunks:
            self._chunks.popitem(last=False)
        return payload

    def get(self, key, device):
        location = self.index.get(str(key))
        if location is None:
            return None
        rank, chunk_id, row_index = location
        payload = self._load_chunk(rank, chunk_id)
        if payload["keys"][row_index] != str(key):
            raise RuntimeError("bridge target manifest/chunk key mismatch")
        return payload["hidden"][row_index].float().to(device=device)


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
        bridge_align_weight=0.0,
        bridge_specific_align_weight=0.0,
        ess_stop_weight=0.2,
        qdrop_action_weight=0.1,
        stop_weight=0.1,
        stop_pos_weight=1.0,
        ess_warmup_steps=100,
        bridge_lr=1e-4,
        chunk_size=32,
        bridge_align_targets="",
        bridge_target_centroids="",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.bridge_align_targets = str(bridge_align_targets).strip()
        self._offline_align_targets = False
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
        self.bridge_align_weight = float(bridge_align_weight)
        self.bridge_specific_align_weight = float(bridge_specific_align_weight)
        self.ess_stop_weight = float(ess_stop_weight)
        self.qdrop_action_weight = float(qdrop_action_weight)
        self.stop_weight = float(stop_weight)
        self.stop_pos_weight = float(stop_pos_weight)
        self.ess_warmup_steps = int(ess_warmup_steps)
        self.bridge_lr = float(bridge_lr)
        self.chunk_size = int(chunk_size)
        self._bridge_teacher_cache = OrderedDict()
        self._bridge_teacher_cache_max = 1024
        self._bridge_target_store = None
        self._bridge_target_centroids = None
        if self.bridge_specific_align_weight > 0:
            centroid_path = Path(str(bridge_target_centroids).strip())
            if not centroid_path.is_file():
                raise RuntimeError(
                    f"bridge_specific_align_weight={self.bridge_specific_align_weight} > 0 "
                    f"but centroid file is missing: {centroid_path}"
                )
            centroid_payload = torch.load(
                centroid_path, map_location="cpu", weights_only=False
            )
            if centroid_payload.get("schema") != "bridge_target_centroids_v1":
                raise RuntimeError("bridge target centroid schema mismatch")
            if tuple(centroid_payload.get("fields", ())) != tuple(FIELD_NAMES):
                raise RuntimeError("bridge target centroid field order mismatch")
            centroids = centroid_payload.get("centroids")
            if not torch.is_tensor(centroids) or tuple(centroids.shape) != (
                len(FIELD_NAMES), 3584
            ):
                raise RuntimeError("invalid bridge target centroid tensor")
            self._bridge_target_centroids = centroids.float().contiguous()
            print(
                f"[BRIDGE_SPECIFIC] Loaded field centroids from {centroid_path}",
                flush=True,
            )
        if self.bridge_align_weight > 0 or self.bridge_specific_align_weight > 0:
            if not self.bridge_align_targets:
                raise RuntimeError(
                    f"[FATAL GATE 2] bridge alignment is active "
                    f"(raw={self.bridge_align_weight}, "
                    f"specific={self.bridge_specific_align_weight}) but "
                    "--bridge-align-targets was not provided! "
                    "Silent fallback to 7B runtime teacher forward is strictly forbidden."
                )
            if not os.path.exists(self.bridge_align_targets):
                raise RuntimeError(
                    f"[FATAL GATE 2] bridge_align_targets file does not exist: {self.bridge_align_targets}. "
                    "Cannot start training without precomputed alignment targets."
                )
            print(f"[BRIDGE_ALIGN] Loading offline targets from {self.bridge_align_targets}...", flush=True)
            if Path(self.bridge_align_targets).is_dir():
                self._bridge_target_store = ChunkedBridgeTargetStore(self.bridge_align_targets)
                loaded_count = len(self._bridge_target_store.index)
            else:
                loaded = torch.load(self.bridge_align_targets, map_location="cpu", weights_only=False)
                if len(loaded) == 0:
                    raise RuntimeError(
                        f"[FATAL GATE 2] bridge_align_targets file {self.bridge_align_targets} is empty! "
                        "Precomputed targets must contain non-zero items."
                    )
                self._bridge_teacher_cache.update(loaded)
                loaded_count = len(loaded)
            self._offline_align_targets = True
            print(f"[BRIDGE_ALIGN] Loaded {loaded_count} offline targets successfully.", flush=True)
        elif self.bridge_align_targets and os.path.exists(self.bridge_align_targets):
            print(f"[BRIDGE_ALIGN] Loading offline targets from {self.bridge_align_targets}...", flush=True)
            if Path(self.bridge_align_targets).is_dir():
                self._bridge_target_store = ChunkedBridgeTargetStore(self.bridge_align_targets)
            else:
                loaded = torch.load(self.bridge_align_targets, map_location="cpu", weights_only=False)
                self._bridge_teacher_cache.update(loaded)
            self._offline_align_targets = True
            loaded_count = len(self._bridge_target_store.index) if self._bridge_target_store else len(self._bridge_teacher_cache)
            print(f"[BRIDGE_ALIGN] Loaded {loaded_count} offline targets successfully.", flush=True)
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
        bridge, ess, aux, main = [], [], [], []
        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad:
                continue
            if "colt_modules.semantic_action_bridge" in name:
                bridge.append(parameter)
            elif "colt_modules.ess_decoder" in name:
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
                {"params": bridge, "lr": self.bridge_lr},
            ],
            betas=(0.9, 0.95),
            weight_decay=self.args.weight_decay,
        )
        return self.optimizer

    @staticmethod
    def _bare(model):
        return model.module if hasattr(model, "module") else model


    def _bridge_align_loss(
        self, model, semantic_tokens, align_ids, align_mask, content_mask,
        align_key_bytes=None,
    ):
        """Align bridge tokens with frozen Stage1 Decision-probe targets.

        Raw cosine preserves the common action-recognizable target space.  The
        specific term projects both normalized vectors onto the tangent space
        orthogonal to the corresponding field centroid, so a constant
        field-prototype prediction cannot minimize the sample-specific loss.
        """
        causal = self._bare(model)
        device = semantic_tokens.device
        ids = align_ids.to(device=device)
        mask = align_mask.to(device=device)
        content = content_mask.to(device=device)
        valid_text = mask.sum(dim=-1) > 0
        valid_sup = content.sum(dim=-1) > 0
        valid = valid_text & valid_sup
        if not bool(valid.any().item()):
            return None, None, None, None
        key_bytes = None
        if align_key_bytes is not None:
            key_bytes = align_key_bytes.detach().to(device="cpu", dtype=torch.uint8)
            if key_bytes.ndim != 2 or key_bytes.shape[0] != ids.shape[0] or key_bytes.shape[1] != 32:
                raise RuntimeError(
                    f"invalid bridge target key tensor shape: {tuple(key_bytes.shape)}"
                )
        rows = []
        for field_index in range(int(ids.shape[0])):
            if not bool(valid[field_index].item()):
                continue
            length = int(mask[field_index].sum().item())
            if key_bytes is not None:
                key = bytes(key_bytes[field_index].tolist()).hex()
            else:
                key = tuple(int(x) for x in ids[field_index, :length].tolist())
            rows.append((field_index, key))
        targets = [None] * len(rows)
        pending = []
        for pos, (_, key) in enumerate(rows):
            cached = self._bridge_teacher_cache.get(key)
            if cached is None and self._bridge_target_store is not None:
                cached = self._bridge_target_store.get(key, device=device)
            if cached is not None:
                targets[pos] = cached.to(device=device, dtype=torch.float32)
                if not self._offline_align_targets:
                    self._bridge_teacher_cache.move_to_end(key)
            else:
                pending.append(pos)
        if pending and not self._offline_align_targets:
            _mem_before = torch.cuda.memory_allocated() if torch.cuda.is_available() else None
            _peak_before = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None
            computed = self._teacher_hidden_targets(
                causal, [rows[pos][1] for pos in pending]
            )
            if _mem_before is not None:
                self._diag_teacher_delta = getattr(self, "_diag_teacher_delta", 0.0) + (
                    torch.cuda.memory_allocated() - _mem_before
                ) / (1024.0 ** 3)
                self._diag_teacher_peak_delta = getattr(self, "_diag_teacher_peak_delta", 0.0) + (
                    torch.cuda.max_memory_allocated() - _peak_before
                ) / (1024.0 ** 3)
            for pos, vector in zip(pending, computed):
                vector = vector.detach().float()
                key = rows[pos][1]
                self._bridge_teacher_cache[key] = vector.cpu()
                self._bridge_teacher_cache.move_to_end(key)
                while len(self._bridge_teacher_cache) > self._bridge_teacher_cache_max:
                    self._bridge_teacher_cache.popitem(last=False)
                targets[pos] = vector.to(device=device, dtype=torch.float32)
        elif pending and self._offline_align_targets:
            self._bridge_align_misses = getattr(self, "_bridge_align_misses", 0) + len(pending)
            miss_keys = [rows[pos][1] for pos in pending]
            curr_step = getattr(self.state, "global_step", 0)
            print(
                f"[BRIDGE_ALIGN_CRITICAL] MISS DETECTED at step {curr_step}: "
                f"{len(pending)} keys not found in offline targets! Sample miss key: {miss_keys[0][:8]}...",
                flush=True,
            )
            raise RuntimeError(
                f"[FATAL GATE 1] Runtime alignment target cache miss at step {curr_step}: "
                f"{len(pending)} keys missing from offline targets! "
                "Silent degradation is forbidden."
            )
        per_field = torch.full(
            (int(ids.shape[0]),), float("nan"), device=device, dtype=torch.float32
        )
        per_field_specific = torch.full_like(per_field, float("nan"))
        centroids = None
        if self._bridge_target_centroids is not None:
            centroids = self._bridge_target_centroids.to(device=device)
        for (field_index, _), target in zip(rows, targets):
            if target is None:
                continue
            sem = semantic_tokens[0, field_index].float()
            cos = F.cosine_similarity(
                sem.unsqueeze(0), target.unsqueeze(0), dim=-1
            ).squeeze(0)
            per_field[field_index] = cos
            if centroids is not None:
                axis = F.normalize(
                    centroids[field_index].unsqueeze(0), dim=-1
                ).squeeze(0)
                sem_unit = F.normalize(sem.unsqueeze(0), dim=-1).squeeze(0)
                target_unit = F.normalize(
                    target.unsqueeze(0), dim=-1
                ).squeeze(0)
                sem_specific = sem_unit - (sem_unit * axis).sum() * axis
                target_specific = (
                    target_unit - (target_unit * axis).sum() * axis
                )
                specific_cos = F.cosine_similarity(
                    sem_specific.unsqueeze(0),
                    target_specific.unsqueeze(0),
                    dim=-1,
                    eps=1e-6,
                ).squeeze(0)
                per_field_specific[field_index] = specific_cos
        if not bool(torch.isfinite(per_field).any().item()):
            return None, None, None, None
        loss = torch.nanmean(1.0 - per_field)
        specific_loss = None
        if bool(torch.isfinite(per_field_specific).any().item()):
            specific_loss = torch.nanmean(1.0 - per_field_specific)
        return (
            loss,
            per_field.detach(),
            specific_loss,
            per_field_specific.detach(),
        )

    @staticmethod
    def _teacher_hidden_targets(causal, seqs):
        """Final-layer hidden at the last valid token for each bare text row."""
        if not seqs:
            return []
        device = causal.get_input_embeddings().weight.device
        width = max(len(seq) for seq in seqs)
        input_ids = torch.zeros((len(seqs), width), dtype=torch.long, device=device)
        attention_mask = torch.zeros_like(input_ids)
        for row_index, seq in enumerate(seqs):
            offset = width - len(seq)
            input_ids[row_index, offset:] = torch.tensor(
                seq, dtype=torch.long, device=device
            )
            attention_mask[row_index, offset:] = 1
        position_ids = attention_mask.cumsum(dim=-1) - 1
        with torch.inference_mode():
            if hasattr(causal, "disable_adapter"):
                manager = causal.disable_adapter()
            else:
                manager = contextlib.nullcontext()
            with manager:
                embeds = causal.get_input_embeddings()(input_ids)
                outputs = causal.model.model(
                    input_ids=input_ids,
                    inputs_embeds=embeds,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    use_cache=False,
                    return_dict=True,
                )
            hidden = outputs.last_hidden_state.float()
        # Left padding ensures the final valid token of every row is at index -1
        picked = hidden[:, -1]
        return [picked[i].detach() for i in range(picked.shape[0])]



    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        causal=self._bare(model); backbone=causal.model.model; lm_head=causal.get_output_embeddings()
        modules=causal.colt_modules; mode="train" if model.training else "eval"
        input_ids=inputs["input_ids"]; attention_mask=inputs["attention_mask"]
        pixel_values=inputs.get("pixel_values"); image_grid_thw=inputs.get("image_grid_thw")
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
        suffix_ids=input_ids[:,prefix_end:target_end]
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
        # ``suffix_ids`` begins at the action boundary.  For answer rows it
        # contains the control bridge ``</think>\n`` before ``<answer>``.
        # Supervising the whole suffix is essential: masking the bridge taught
        # <answer> only after teacher-forcing the bridge, so free rollout never
        # learned to close <think> on its own.
        labels=suffix_ids.clone()
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
        if getattr(self, "ess_warmup_steps", 0) <= 0:
            ess_scale = self.ess_weight
        else:
            warm = max(1, int(getattr(self, "ess_warmup_steps", 100)))
            ess_scale = self.ess_weight * min(1.0, float(self.state.global_step + 1) / warm)
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
                        active_fields = set(r.get('active_fields', FIELD_NAMES))
                        for i,k in enumerate(FIELD_NAMES):
                            if k not in active_fields:
                                result['decoded_ess'][k] = ''
                                result['raw'][k] = ''
                                continue
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
                "stop_head": modules.stop_head.state_dict(),
                "semantic_action_bridge": modules.semantic_action_bridge.state_dict(),
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
        if 'stop_head' in state:
            causal.colt_modules.stop_head.load_state_dict(state['stop_head'], strict=True)
        if 'semantic_action_bridge' in state:
            causal.colt_modules.semantic_action_bridge.load_state_dict(
                state['semantic_action_bridge'], strict=True
            )
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
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
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
        target_starts = inputs["target_starts"][0]
        target_ends = inputs["target_ends"][0]
        ess_active = inputs["ess_active"][0]
        action_active = inputs["action_active"][0].bool()
        boundary_count = int(prefix_ends.numel())
        if not boundary_count:
            raise RuntimeError("empty trajectory")
        stream_backward = bool(
            model.training and getattr(self, "_stream_boundary_backward", False)
        )
        # Include each boundary bridge in the action objective.  In particular,
        # answer boundaries must learn ``latent -> </think> -> <answer>`` rather
        # than receiving ``</think>`` as an unsupervised teacher-forced prefix.
        boundary_token_counts = target_ends - prefix_ends
        total_action_tokens = int(
            boundary_token_counts[action_active].sum().item()
        )
        active_action_boundaries = int(action_active.long().sum().item())
        active_ess_boundaries = int(ess_active.long().sum().item())
        if getattr(self, "ess_warmup_steps", 0) <= 0:
            ess_scale = self.ess_weight
            align_scale = self.bridge_align_weight
            specific_align_scale = self.bridge_specific_align_weight
        else:
            warm = max(1, int(self.ess_warmup_steps))
            ess_scale = self.ess_weight * min(
                1.0, float(self.state.global_step + 1) / warm
            )
            align_scale = self.bridge_align_weight * min(
                1.0, float(self.state.global_step + 1) / warm
            )
            specific_align_scale = self.bridge_specific_align_weight * min(
                1.0, float(self.state.global_step + 1) / warm
            )

        self._diag_teacher_delta = 0.0
        self._diag_teacher_peak_delta = 0.0
        mem_start_gb = None
        mem_after_vision_gb = None
        mem_after_qd_vision_gb = None
        if torch.cuda.is_available():
            mem_start_gb = torch.cuda.memory_allocated() / (1024.0 ** 3)
        if mode == "train" and torch.cuda.is_available():
            if getattr(self, "_mb_seen_step", None) != self.state.global_step:
                self._mb_seen_step = self.state.global_step
                self._mb_index = 0
            else:
                self._mb_index = getattr(self, "_mb_index", 0) + 1
            _img = inputs.get("image_grid_thw")
            _img_n = int(_img.shape[0]) if _img is not None else 0
            print(
                "[MEMDIAG] gs=%d mb=%d traj=%s raw=%.1f phys=%.1f img=%d mem_start=%.3fGB reserved=%.3fGB"
                % (
                    int(self.state.global_step),
                    int(getattr(self, "_mb_index", 0)),
                    str(inputs.get("trajectory_id", "?")),
                    float(inputs["raw_token_count"].float().mean().item()),
                    float(inputs["physical_token_count"].float().mean().item()),
                    _img_n,
                    float(mem_start_gb),
                    float(torch.cuda.memory_reserved() / (1024.0 ** 3)),
                ),
                flush=True,
            )
        pixel_values, image_grid_thw = split_vision_inputs(
            inputs.get("pixel_values"), inputs.get("image_grid_thw")
        )
        with torch.no_grad():
            base_embeddings = build_multimodal_embeddings(
                model, input_ids, pixel_values, image_grid_thw
            ).detach()
        if mem_start_gb is not None:
            mem_after_vision_gb = torch.cuda.memory_allocated() / (1024.0 ** 3)
        if mode == "train" and torch.cuda.is_available():
            print(
                "[MEMDIAG2] base_vision_done gs=%d mb=%d traj=%s alloc=%.3fGB peak=%.3fGB"
                % (
                    int(self.state.global_step),
                    int(getattr(self, "_mb_index", 0)),
                    str(inputs.get("trajectory_id", "?")),
                    float(mem_after_vision_gb),
                    float(torch.cuda.max_memory_allocated() / (1024.0 ** 3)),
                ),
                flush=True,
            )
        positions = qwen_position_ids(
            model, input_ids, image_grid_thw, attention_mask
        )

        # Parallel Q-Drop stream.  It reuses the normal branch's predicted
        # latents but rebuilds the text/image KV context with the question
        # masked.  Prior-boundary KV is detached exactly like the normal
        # trajectory stream; gradients remain live within the current
        # boundary through the injected latents.
        qdrop_enabled = self.qdrop_action_weight > 0.0
        qd_input_ids = inputs.get("qdrop_input_ids") if qdrop_enabled else None
        qd_attention_mask = inputs.get("qdrop_attention_mask") if qdrop_enabled else None
        qd_prefix_ends = inputs.get("qdrop_prefix_ends") if qdrop_enabled else None
        qd_target_starts = inputs.get("qdrop_target_starts") if qdrop_enabled else None
        qd_target_ends = inputs.get("qdrop_target_ends") if qdrop_enabled else None
        if qdrop_enabled:
            with torch.no_grad():
                qd_base_embeddings = build_multimodal_embeddings(
                    model, qd_input_ids, pixel_values, image_grid_thw
                ).detach()
            qd_positions = qwen_position_ids(
                model, qd_input_ids, image_grid_thw, qd_attention_mask
            )
            if mode == "train" and torch.cuda.is_available():
                print(
                    "[MEMDIAG2] qd_vision_done gs=%d mb=%d traj=%s alloc=%.3fGB peak=%.3fGB"
                    % (
                        int(self.state.global_step),
                        int(getattr(self, "_mb_index", 0)),
                        str(inputs.get("trajectory_id", "?")),
                        float(torch.cuda.memory_allocated() / (1024.0 ** 3)),
                        float(torch.cuda.max_memory_allocated() / (1024.0 ** 3)),
                    ),
                    flush=True,
                )
            qd_cache = None
            # Start from 0 so system prompt and original image visual tokens enter qd_cache
            qd_cursor = 0
            qd_inserted = 0
        else:
            qd_base_embeddings = qd_positions = qd_cache = None
            qd_cursor = qd_inserted = 0
        if mem_start_gb is not None:
            mem_after_qd_vision_gb = torch.cuda.memory_allocated() / (1024.0 ** 3)

        cache = None
        cursor = 0
        inserted = 0
        last_state = None
        action_weighted = base_embeddings.sum() * 0.0
        action_tokens = 0
        qdrop_weighted = base_embeddings.sum() * 0.0
        stop_weighted = base_embeddings.sum() * 0.0
        ess_losses = []
        adjacent_values = []
        latent_rms_values = []
        semantic_rms_values = []
        qdrop_semantic_grad_values = []
        query_attention_values = []
        field_loss_values = []
        stop_loss_values = []
        content_loss_values = []
        per_kind = defaultdict(list)
        # Compatibility buckets: keep the same action/ESS tags as the standalone
        # ESS and NoESS runs so TensorBoard can overlay all three experiments.
        per_tool_action = defaultdict(list)
        per_tool_field = defaultdict(lambda: defaultdict(list))
        snapshots = []
        answer_ready_nll_values = []
        tool_ess_nll_values = []
        branch_nll_values = []
        qdrop_branch_nll_values = []
        branch_correct_values = []
        qdrop_branch_correct_values = []
        field_loss_by_group = {"tool": [], "answer": []}
        content_loss_by_group = {"tool": [], "answer": []}
        stop_loss_by_group = {"tool": [], "answer": []}
        last_output = None
        stop_loss_values_all = []
        qdrop_loss_values = []
        qdrop_per_tool = defaultdict(list)
        bridge_align_values = []
        bridge_specific_align_values = []
        bridge_align_by_group = {"tool": [], "answer": []}
        bridge_specific_align_by_group = {"tool": [], "answer": []}

        for boundary_index in range(boundary_count):
            prefix_end = int(prefix_ends[boundary_index].item())
            target_start = int(target_starts[boundary_index].item())
            target_end = int(target_ends[boundary_index].item())
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
            # A single shared semantic state is consumed by both the ESS
            # decoder and the main action model.  This prevents the two heads
            # from learning unrelated projections of the recurrent memory.
            field_prefixes, field_query_attention = modules.ess_decoder.read_fields(memory)
            semantic_tokens = modules.semantic_action_bridge(field_prefixes, memory)
            semantic_token_count = int(semantic_tokens.shape[1])
            semantic_rms_values.append(
                semantic_tokens.detach().float().square().mean().sqrt()
            )
            align_loss = None
            align_field_cos = None
            specific_align_loss = None
            specific_align_field_cos = None
            if (
                (self.bridge_align_weight > 0 or self.bridge_specific_align_weight > 0)
                and "bridge_align_ids" in inputs
                and "bridge_align_mask" in inputs
            ):
                (
                    align_loss,
                    align_field_cos,
                    specific_align_loss,
                    specific_align_field_cos,
                ) = self._bridge_align_loss(
                    causal,
                    semantic_tokens,
                    inputs["bridge_align_ids"][0, boundary_index],
                    inputs["bridge_align_mask"][0, boundary_index],
                    inputs["ess_content_mask"][0, boundary_index],
                    inputs.get("bridge_align_key_bytes")[0, boundary_index]
                    if inputs.get("bridge_align_key_bytes") is not None else None,
                )
            suffix_ids = input_ids[:, prefix_end:target_end]
            suffix_embeddings = base_embeddings[:, prefix_end:target_end]
            inserted_after = inserted + modules.stages + semantic_token_count + 1
            final_latent_position = prefix_last_position + modules.stages
            # Run Q-Drop before the normal action/ESS forwards.  Q-Drop uses
            # detached latent leaves, is backpropagated without retaining its
            # graph, and passes dL/dz back through a later gradient surrogate.
            # This prevents the two long action graphs (and the ESS decoder
            # graph) from coexisting at the peak-memory point.
            qdrop_loss = current.sum() * 0.0
            qdrop_backed = False
            qdrop_latent_grads = None
            qdrop_semantic_grad = None
            qd_local_cache = None
            qd_cached_len = 0
            qd_final_position = None
            if qdrop_enabled:
                qd_prefix_end = int(qd_prefix_ends[0, boundary_index].item()) if qd_prefix_ends.ndim > 1 else int(qd_prefix_ends[boundary_index].item())
                qd_target_start = int(qd_target_starts[0, boundary_index].item()) if qd_target_starts.ndim > 1 else int(qd_target_starts[boundary_index].item())
                qd_target_end = int(qd_target_ends[0, boundary_index].item()) if qd_target_ends.ndim > 1 else int(qd_target_ends[boundary_index].item())
                if qd_cursor < qd_prefix_end:
                    qd_interval = qd_base_embeddings[:, qd_cursor:qd_prefix_end]
                    qd_interval_pos = qd_positions[..., qd_cursor:qd_prefix_end] + qd_inserted
                    with torch.no_grad():
                        qd_prefix_out = backbone(
                            inputs_embeds=qd_interval,
                            attention_mask=torch.ones(
                                (1, qd_prefix_end + qd_inserted),
                                dtype=attention_mask.dtype, device=input_ids.device,
                            ),
                            position_ids=qd_interval_pos,
                            past_key_values=qd_cache,
                            cache_position=(torch.arange(
                                qd_cursor + qd_inserted,
                                qd_prefix_end + qd_inserted,
                                device=input_ids.device, dtype=torch.long,
                            ) if qd_cache is not None else None),
                            use_cache=True, output_hidden_states=False,
                            return_dict=True,
                        )
                    qd_cache = detach_kv_cache(qd_prefix_out.past_key_values)
                    qd_cursor = qd_prefix_end
                    del qd_prefix_out
                qd_local_cache = qd_cache
                qd_cached_len = qd_prefix_end + qd_inserted
                qd_prefix_last_position = qd_positions[..., qd_prefix_end - 1:qd_prefix_end] + qd_inserted
                qd_final_position = qd_prefix_last_position
                # Same topology as the normal branch; only the Question is
                # masked:  qd_prefix | seed, z1, z2, z3 (KV) | z4 | b1, b2, b3 | action
                qd_semantic_leaf = semantic_tokens.detach().requires_grad_(True)
                qd_z4_embed = latents[-1].detach()
                qd_latent_leaf = qd_z4_embed.detach().requires_grad_(True)
                qd_suffix_ids = qd_input_ids[:, qd_prefix_end:qd_target_end]
                qd_suffix_embeddings = qd_base_embeddings[:, qd_prefix_end:qd_target_end]
                # Q-Drop KV cache: Populated with z1..z3 (detached, under torch.no_grad)
                # NEVER pass normal-branch last_state (which contains unmasked Question!).
                with torch.no_grad():
                    for _qd_stage, _qd_row in enumerate(
                        [z.detach() for z in latents[:-1]]
                    ):
                        _qd_row_out = backbone(
                            inputs_embeds=_qd_row.unsqueeze(1),
                            attention_mask=torch.ones(
                                (1, qd_cached_len + 1), dtype=attention_mask.dtype,
                                device=input_ids.device,
                            ),
                            position_ids=qd_prefix_last_position + _qd_stage + 1,
                            past_key_values=qd_local_cache,
                            cache_position=torch.tensor(
                                [qd_cached_len], device=input_ids.device, dtype=torch.long
                            ),
                            use_cache=True, output_hidden_states=False,
                            return_dict=True,
                        )
                        qd_local_cache = detach_kv_cache(_qd_row_out.past_key_values)
                        qd_cached_len += 1
                        qd_final_position = qd_prefix_last_position + _qd_stage + 1
                        del _qd_row_out
                qd_action_inputs = torch.cat(
                    [
                        qd_latent_leaf.unsqueeze(1),
                        qd_semantic_leaf,
                        qd_suffix_embeddings[:, :-1],
                    ],
                    dim=1,
                )
                qd_final_action_position = qd_final_position + 1
                qd_semantic_positions = qd_final_position + torch.arange(
                    2, semantic_token_count + 2,
                    device=input_ids.device, dtype=qd_final_position.dtype,
                ).view(1, -1)
                qd_action_positions = torch.cat([
                    qd_final_action_position,
                    qd_semantic_positions,
                    qd_positions[..., qd_prefix_end:qd_target_end - 1]
                    + qd_inserted + modules.stages + semantic_token_count,
                ], dim=-1)
                qd_action_mask = torch.ones(
                    (1, qd_cached_len + qd_action_inputs.shape[1]),
                    dtype=attention_mask.dtype, device=input_ids.device,
                )
                qd_action_cache_positions = torch.arange(
                    qd_cached_len, qd_cached_len + qd_action_inputs.shape[1],
                    device=input_ids.device, dtype=torch.long,
                )
                qd_action_hidden = backbone(
                    inputs_embeds=qd_action_inputs,
                    attention_mask=qd_action_mask,
                    position_ids=qd_action_positions,
                    past_key_values=qd_local_cache,
                    cache_position=qd_action_cache_positions,
                    use_cache=False, output_hidden_states=False,
                    return_dict=True,
                ).last_hidden_state
                # Labels: the z4 row and the first two bridge rows are masked;
                # the terminal bridge row (index semantic_token_count) predicts
                # suffix_ids[:, 0], so the bridge is a real bottleneck.
                qd_suffix_labels = (
                    qd_suffix_ids
                    if bool(action_active[boundary_index].item())
                    else torch.full_like(qd_suffix_ids, -100)
                )
                qd_labels = torch.cat([
                    torch.full(
                        (qd_suffix_ids.shape[0], semantic_token_count), -100,
                        dtype=qd_suffix_ids.dtype, device=qd_suffix_ids.device,
                    ),
                    qd_suffix_labels,
                ], dim=1)
                if bool(action_active[boundary_index].item()):
                    with torch.no_grad():
                        # The terminal bridge token directly predicts the control token (<tool_call> or </think>)
                        qd_branch_target = qd_suffix_ids[:, 0].long()
                        qd_branch_logits = lm_head(
                            qd_action_hidden[
                                :, semantic_token_count:semantic_token_count + 1
                            ].detach()
                        ).float().squeeze(1)
                        qd_branch_nll = F.cross_entropy(
                            qd_branch_logits, qd_branch_target
                        )
                        qdrop_branch_nll_values.append(qd_branch_nll.detach())
                        qdrop_branch_correct_values.append(
                            (qd_branch_logits.argmax(dim=-1) == qd_branch_target)
                            .float().mean().detach()
                        )
                qdrop_loss = chunked_token_ce(
                    qd_action_hidden, qd_labels, lm_head,
                    self.chunk_size,
                )
                if bool(action_active[boundary_index].item()) and qd_token_count > 0:
                    qdrop_loss_values.append(qdrop_loss.detach())
                    _qd_tool = str(inputs["target_tools"][boundary_index] or "answer")
                    if _qd_tool.strip().lower() in {"", "none", "null"}:
                        _qd_tool = "answer"
                    qdrop_per_tool[
                        _qd_tool.replace("/", "_").replace(" ", "_")
                    ].append(qdrop_loss.detach())
                qd_boundary_active = bool(action_active[boundary_index].item())
                qd_token_count = (
                    int((qd_target_end - qd_prefix_end))
                    if qd_boundary_active else 0
                )
                if stream_backward:
                    qdrop_scale = (
                        self.qdrop_action_weight * qd_token_count
                        / (
                            max(1, int(self.args.gradient_accumulation_steps))
                            * max(1, total_action_tokens)
                        )
                    )
                    qdrop_backward_kwargs = (
                        {"scale_wrt_gas": False}
                        if self.is_deepspeed_enabled else {}
                    )
                    self.accelerator.backward(
                        qdrop_scale * qdrop_loss,
                        **qdrop_backward_kwargs,
                    )
                    if qd_latent_leaf.grad is None:
                        raise RuntimeError(
                            "latent action seed is disconnected from Q-Drop action loss"
                        )
                    # z1..z3 receive their gradient through the recurrent chain
                    # that produced z4, so only z4 needs a surrogate here.
                    qdrop_latent_grads = [
                        torch.zeros_like(z).detach() for z in latents
                    ]
                    qdrop_latent_grads[-1] = (
                        qd_latent_leaf.grad.detach().clone().to(latents[-1].dtype)
                    )
                    if qd_semantic_leaf.grad is None:
                        raise RuntimeError(
                            "semantic action bridge is disconnected from Q-Drop action loss"
                        )
                    qdrop_semantic_grad = qd_semantic_leaf.grad.detach().clone()
                    qdrop_semantic_grad_values.append(
                        qdrop_semantic_grad.float().square().mean().sqrt()
                    )
                    qdrop_backed = True
                    qdrop_loss = qdrop_loss.detach()
                    qd_local_cache = detach_kv_cache(qd_local_cache)
                    del qd_action_hidden, qd_action_inputs
                    del qd_action_positions, qd_action_mask
                    del qd_action_cache_positions, qd_labels
                    del qd_suffix_ids, qd_suffix_embeddings
                    # qd_branch_* are diagnostic-only and are set inside an
                    # action_active guard, so only drop the references here.
                    qd_branch_logits = qd_branch_nll = qd_branch_target = None
                    del qd_semantic_leaf
                    del qd_latent_leaf
            # Unified topology (normal branch == Q-Drop branch == inference):
            #   ... seed, z1, z2, z3 | z4 | bridge1, bridge2, bridge3 | action
            # ``final_latent_position`` is z3 (the last latent written inside
            # the recurrent loop), so z4 takes the next slot and the three
            # bridge tokens follow it.  The terminal bridge token is the token
            # that predicts suffix_ids[:, 0], which makes the bridge a
            # mandatory bottleneck between the latents and the action policy.
            # Row budget per boundary is unchanged (4 loop rows + z4 + 3).
            final_action_position = final_latent_position + 1
            semantic_positions = final_latent_position + torch.arange(
                2, semantic_token_count + 2,
                device=input_ids.device, dtype=final_latent_position.dtype,
            ).view(1, -1)
            action_inputs = torch.cat(
                [current.unsqueeze(1), semantic_tokens, suffix_embeddings[:, :-1]],
                dim=1,
            )
            action_positions = torch.cat([
                final_action_position,
                semantic_positions,
                positions[..., prefix_end:target_end - 1] + inserted_after,
            ], dim=-1)
            if self.action_weight > 0.0:
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
                action_suffix_labels = (
                    suffix_ids
                    if bool(action_active[boundary_index].item())
                    else torch.full_like(suffix_ids, -100)
                )
                labels = torch.cat([
                    torch.full(
                        (suffix_ids.shape[0], semantic_token_count), -100,
                        dtype=suffix_ids.dtype, device=suffix_ids.device,
                    ),
                    action_suffix_labels,
                ], dim=1)
                # This is also diagnostic-only; keep it outside the action graph.
                if bool(action_active[boundary_index].item()):
                    with torch.no_grad():
                        branch_target = suffix_ids[:, 0].long()
                        branch_logits = lm_head(
                            action_outputs.last_hidden_state[
                                :, semantic_token_count:semantic_token_count + 1
                            ].detach()
                        ).float().squeeze(1)
                        branch_nll = F.cross_entropy(branch_logits, branch_target)
                        branch_nll_values.append(branch_nll.detach())
                        branch_correct_values.append(
                            (branch_logits.argmax(dim=-1) == branch_target)
                            .float().mean().detach()
                        )
                action_loss = chunked_token_ce(
                    action_outputs.last_hidden_state, labels, lm_head, self.chunk_size
                )
            else:
                action_outputs = None
                action_loss = current.sum() * 0.0
            # Q-Drop is evaluated before this normal action forward.  Keep the
            # old inline block disabled so it cannot create a second Q-Drop
            # graph that overlaps the normal branch.
            qdrop_loss = qdrop_loss if qdrop_enabled else action_loss * 0.0
            if False and qdrop_enabled:
                qd_prefix_end = int(qd_prefix_ends[0, boundary_index].item()) if qd_prefix_ends.ndim > 1 else int(qd_prefix_ends[boundary_index].item())
                qd_target_start = int(qd_target_starts[0, boundary_index].item()) if qd_target_starts.ndim > 1 else int(qd_target_starts[boundary_index].item())
                qd_target_end = int(qd_target_ends[0, boundary_index].item()) if qd_target_ends.ndim > 1 else int(qd_target_ends[boundary_index].item())
                if qd_cursor < qd_prefix_end:
                    qd_interval = qd_base_embeddings[:, qd_cursor:qd_prefix_end]
                    qd_interval_pos = qd_positions[..., qd_cursor:qd_prefix_end] + qd_inserted
                    with torch.no_grad():
                        qd_prefix_out = backbone(
                            inputs_embeds=qd_interval,
                            attention_mask=torch.ones(
                                (1, qd_prefix_end + qd_inserted),
                                dtype=attention_mask.dtype, device=input_ids.device,
                            ),
                            position_ids=qd_interval_pos,
                            past_key_values=qd_cache,
                            cache_position=(torch.arange(
                                qd_cursor + qd_inserted,
                                qd_prefix_end + qd_inserted,
                                device=input_ids.device, dtype=torch.long,
                            ) if qd_cache is not None else None),
                            use_cache=True, output_hidden_states=False,
                            return_dict=True,
                        )
                    qd_cache = detach_kv_cache(qd_prefix_out.past_key_values)
                    qd_cursor = qd_prefix_end
                    del qd_prefix_out
                qd_local_cache = qd_cache
                qd_cached_len = qd_prefix_end + qd_inserted
                qd_prefix_last_position = qd_positions[..., qd_prefix_end - 1:qd_prefix_end] + qd_inserted
                for stage, predicted_latent in enumerate(latents):
                    qd_latent_out = backbone(
                        inputs_embeds=predicted_latent.unsqueeze(1),
                        attention_mask=torch.ones(
                            (1, qd_cached_len + 1), dtype=attention_mask.dtype,
                            device=input_ids.device,
                        ),
                        position_ids=qd_prefix_last_position + stage + 1,
                        past_key_values=qd_local_cache,
                        cache_position=torch.tensor(
                            [qd_cached_len], device=input_ids.device, dtype=torch.long
                        ),
                        use_cache=True, output_hidden_states=False,
                        return_dict=True,
                    )
                    qd_local_cache = qd_latent_out.past_key_values
                    qd_cached_len += 1
                    qd_final_position = qd_prefix_last_position + stage + 1
                    del qd_latent_out
                qd_suffix_ids = qd_input_ids[:, qd_prefix_end:qd_target_end]
                qd_suffix_embeddings = qd_base_embeddings[:, qd_prefix_end:qd_target_end]
                qd_action_inputs = torch.cat(
                    [current.unsqueeze(1), qd_suffix_embeddings[:, :-1]], dim=1
                )
                qd_action_positions = torch.cat([
                    qd_final_position,
                    qd_positions[..., qd_prefix_end:qd_target_end - 1]
                    + qd_inserted + modules.stages + 1,
                ], dim=-1)
                qd_action_mask = torch.ones(
                    (1, qd_cached_len + qd_action_inputs.shape[1]),
                    dtype=attention_mask.dtype, device=input_ids.device,
                )
                qd_action_cache_positions = torch.arange(
                    qd_cached_len,
                    qd_cached_len + qd_action_inputs.shape[1],
                    device=input_ids.device, dtype=torch.long,
                )
                # Keep this auxiliary pass as a normal forward.  Wrapping the
                # Qwen-VL action pass in torch.utils.checkpoint is not safe with
                # DynamicCache/FlashAttention: recomputation can return a
                # different tensor metadata (sequence/cache bookkeeping), which
                # raises CheckpointError before backward.  The existing chunked
                # CE and boundary-wise backward release logits promptly; the
                # correctness-first path is preferable to an invalid graph.
                qd_action_hidden = backbone(
                    inputs_embeds=qd_action_inputs,
                    attention_mask=qd_action_mask,
                    position_ids=qd_action_positions,
                    past_key_values=qd_local_cache,
                    cache_position=qd_action_cache_positions,
                    use_cache=False, output_hidden_states=False,
                    return_dict=True,
                ).last_hidden_state
                qd_labels = qd_suffix_ids.clone()
                qdrop_loss = chunked_token_ce(
                    qd_action_hidden, qd_labels, lm_head,
                    self.chunk_size,
                )
                qdrop_loss_values.append(qdrop_loss.detach())

            stop_target = inputs["stop_targets"][0, boundary_index].to(current.dtype)
            # Retain this legacy head only as a read-only diagnostic.  It is
            # no longer part of the optimization objective; branch choice is
            # supervised by the normal autoregressive action CE above.
            with torch.no_grad():
                stop_logit = modules.stop_head(current).reshape(())
            stop_pos_weight = torch.tensor(
                self.stop_pos_weight, device=stop_logit.device, dtype=stop_logit.dtype
            )
            stop_loss = F.binary_cross_entropy_with_logits(
                stop_logit, stop_target, pos_weight=stop_pos_weight
            )
            stop_loss_values_all.append(stop_loss.detach())
            stop_probability = torch.sigmoid(stop_logit).detach()
            stop_correct = ((stop_probability >= 0.5) == (stop_target >= 0.5)).float()
            if not stream_backward:
                qdrop_weighted = qdrop_weighted + qdrop_loss
                stop_weighted = stop_weighted + stop_loss
            boundary_action_active = bool(action_active[boundary_index].item())
            token_count = (target_end - prefix_end) if boundary_action_active else 0
            if stream_backward:
                action_weighted = action_weighted + action_loss.detach() * token_count
            else:
                action_weighted = action_weighted + action_loss * token_count
            action_tokens += token_count
            kind = str(inputs["kinds"][boundary_index])
            tool_label = str(inputs["target_tools"][boundary_index] or kind).strip() or kind
            if kind == "answer" or tool_label.lower() in {"none", "null"}:
                tool_label = "answer"
            if boundary_action_active:
                per_kind[kind].append(action_loss.detach())
                per_tool_action[tool_label].append(action_loss.detach())
            if not stream_backward:
                last_output = action_outputs

            if (self.ess_weight > 0.0 or self.ess_stop_weight > 0.0):
                ess_result = modules.ess_decoder(
                    memory,
                    inputs["ess_input_ids"][:, boundary_index],
                    inputs["ess_attention_mask"][:, boundary_index],
                    labels=inputs["ess_labels"][:, boundary_index],
                    content_mask=inputs["ess_content_mask"][:, boundary_index],
                    stop_mask=inputs["ess_stop_mask"][:, boundary_index],
                    field_weights=(
                        torch.tensor([1.5, 1.5, 1.0], device=memory.device)
                        * (inputs["ess_content_mask"][:, boundary_index]
                           .sum(dim=-1).gt(0).any(dim=0).to(memory.dtype))
                    ),
                    stop_weight=self.ess_stop_weight,
                    semantic_weight=0.0,
                    field_prefixes=field_prefixes,
                    query_attention=field_query_attention,
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
                    is_answer_boundary = kind == "answer" or str(
                        inputs["boundary_types"][boundary_index]
                    ) in {"post_tool_answer", "direct_answer", "pre_answer"}
                    if is_answer_boundary:
                        answer_ready_nll_values.append(ess_result["loss"].detach())
                        ess_group = "answer"
                    else:
                        tool_ess_nll_values.append(ess_result["loss"].detach())
                        ess_group = "tool"
                    field_loss_by_group[ess_group].append(ess_result["field_losses"].detach())
                    for field_index, field_name in enumerate(FIELD_NAMES):
                        per_tool_field[tool_label][field_name].append(
                            ess_result["field_losses"][field_index].detach()
                        )
                    content_loss_by_group[ess_group].append(ess_result["content_loss"].detach())
                    stop_loss_by_group[ess_group].append(ess_result["stop_loss"].detach())
                    query_attention_values.append(
                        ess_result["query_attention"].detach().float().mean(dim=(0, 1))
                    )
            else:
                ess_result = None
                boundary_ess_active = torch.zeros((), device=memory.device, dtype=memory.dtype)
                if mode == "eval" and hasattr(self, "eval_decode_rows"):
                    if len(self.eval_decode_rows) < 20:
                        self.eval_decode_rows.append({
                            "boundary_id": inputs["boundary_ids"][boundary_index],
                            "latents": memory.detach().cpu(),
                            "gold": json.loads(inputs["ess_gold_jsons"][boundary_index]),
                            "active_fields": [
                                name for field_index, name in enumerate(FIELD_NAMES)
                                if bool((inputs["ess_content_mask"][:, boundary_index, field_index].sum() > 0).item())
                            ],
                            "image_paths": inputs["image_paths"],
                        })

            if align_loss is not None:
                bridge_align_values.append(align_loss.detach())
                align_is_answer = kind == "answer" or str(
                    inputs["boundary_types"][boundary_index]
                ) in {"post_tool_answer", "direct_answer", "pre_answer"}
                align_group = "answer" if align_is_answer else "tool"
                bridge_align_by_group[align_group].append(
                    align_field_cos.detach()
                )
                if specific_align_loss is not None:
                    bridge_specific_align_values.append(
                        specific_align_loss.detach()
                    )
                    bridge_specific_align_by_group[align_group].append(
                        specific_align_field_cos.detach()
                    )
            snapshots.append({
                "boundary_index": boundary_index,
                "boundary_id": inputs["boundary_ids"][boundary_index],
                "kind": kind,
                "boundary_type": inputs["boundary_types"][boundary_index],
                "target_tool": inputs["target_tools"][boundary_index],
                "action_active": boundary_action_active,
                "target_text": inputs["target_texts"][boundary_index],
                "ess_gold": json.loads(inputs["ess_gold_jsons"][boundary_index]),
                "action_ce": float(action_loss.detach().float()),
                "qdrop_action_ce": float(qdrop_loss.detach().float()),
                "stop_target": float(stop_target.detach().float()),
                "stop_probability": float(stop_probability.float()),
                "stop_correct": float(stop_correct.float()),
                "adjacent_cos": float(torch.stack(boundary_adjacent).mean())
                    if boundary_adjacent else float("nan"),
            })

            if stream_backward:
                boundary_loss = current.sum() * 0.0
                if self.action_weight > 0.0:
                    boundary_loss = boundary_loss + (
                        self.action_weight * action_loss * token_count
                        / max(1, total_action_tokens)
                    )
                if self.ess_weight > 0.0 and ess_result is not None:
                    boundary_loss = boundary_loss + (
                        ess_scale * ess_result["loss"] * boundary_ess_active
                        / max(1, active_ess_boundaries)
                    )
                if align_loss is not None:
                    boundary_loss = boundary_loss + (
                        align_scale * align_loss / max(1, active_ess_boundaries)
                    )
                if specific_align_loss is not None:
                    boundary_loss = boundary_loss + (
                        specific_align_scale * specific_align_loss
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
                if qdrop_latent_grads is not None:
                    # The stored gradients already include lambda_q / GAS
                    # from the Q-Drop backward above.  Add a zero-valued
                    # surrogate directly to the scaled loss, so its value and
                    # logging stay unchanged while dL/dz is exactly restored.
                    qdrop_latent_surrogate = sum(
                        (latent * grad.to(latent.dtype)).sum()
                        for latent, grad in zip(latents, qdrop_latent_grads)
                    )
                    scaled_boundary_loss = scaled_boundary_loss + (
                        qdrop_latent_surrogate - qdrop_latent_surrogate.detach()
                    )
                if qdrop_semantic_grad is not None:
                    qdrop_semantic_surrogate = (
                        semantic_tokens * qdrop_semantic_grad.to(semantic_tokens.dtype)
                    ).sum()
                    scaled_boundary_loss = scaled_boundary_loss + (
                        qdrop_semantic_surrogate - qdrop_semantic_surrogate.detach()
                    )
                backward_kwargs = (
                    {"scale_wrt_gas": False} if self.is_deepspeed_enabled else {}
                )
                self.accelerator.backward(scaled_boundary_loss, **backward_kwargs)
                del qdrop_latent_grads

            if boundary_index + 1 < boundary_count:
                next_prefix = int(prefix_ends[boundary_index + 1].item())
                if next_prefix <= prefix_end:
                    raise RuntimeError("next boundary did not advance")
                advance_inputs = torch.cat([
                    current.detach().unsqueeze(1),
                    semantic_tokens.detach(),
                    base_embeddings[:, prefix_end:next_prefix],
                ], dim=1)
                advance_positions = torch.cat([
                    final_action_position,
                    semantic_positions,
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

                if qdrop_enabled:
                    next_qd_prefix = int(
                        qd_prefix_ends[0, boundary_index + 1].item()
                        if qd_prefix_ends.ndim > 1
                        else qd_prefix_ends[boundary_index + 1].item()
                    )
                    qd_inserted_after = (
                        qd_inserted + modules.stages + semantic_token_count + 1
                    )
                    qd_advance_inputs = torch.cat([
                        qd_z4_embed.unsqueeze(1),
                        semantic_tokens.detach(),
                        qd_base_embeddings[:, qd_prefix_end:next_qd_prefix],
                    ], dim=1)
                    qd_advance_final_action_position = qd_final_position + 1
                    qd_advance_semantic_positions = qd_final_position + torch.arange(
                        2, semantic_token_count + 2,
                        device=input_ids.device, dtype=qd_final_position.dtype,
                    ).view(1, -1)
                    qd_advance_positions = torch.cat([
                        qd_advance_final_action_position,
                        qd_advance_semantic_positions,
                        qd_positions[..., qd_prefix_end:next_qd_prefix]
                        + qd_inserted_after,
                    ], dim=-1)
                    with torch.no_grad():
                        qd_advance_outputs = backbone(
                            inputs_embeds=qd_advance_inputs,
                            attention_mask=torch.ones(
                                (1, qd_cached_len + qd_advance_inputs.shape[1]),
                                dtype=attention_mask.dtype, device=input_ids.device,
                            ),
                            position_ids=qd_advance_positions,
                            past_key_values=detach_kv_cache(qd_local_cache),
                            cache_position=torch.arange(
                                qd_cached_len,
                                qd_cached_len + qd_advance_inputs.shape[1],
                                device=input_ids.device, dtype=torch.long,
                            ),
                            use_cache=True, output_hidden_states=False,
                            return_dict=True,
                        )
                    qd_cache = detach_kv_cache(qd_advance_outputs.past_key_values)
                    qd_cursor = next_qd_prefix
                    qd_inserted = qd_inserted_after
                    del qd_advance_outputs

            if stream_backward:
                # Do not keep references to a completed boundary graph.  The
                # detached cache above is the only state carried forward.
                del action_outputs, action_loss, action_inputs, memory, latents
                del ess_result
                del current, local_cache, field_prefixes, field_query_attention
                del semantic_tokens
                if qdrop_enabled and not qdrop_backed:
                    del qd_action_hidden, qd_action_inputs

        action_total = action_weighted / max(1, action_tokens)
        ess_total = torch.stack(ess_losses).mean() if ess_losses else action_total * 0.0
        if stream_backward:
            qdrop_total = (
                torch.stack(qdrop_loss_values).mean()
                if qdrop_loss_values else action_total * 0.0
            )
            stop_total = (
                torch.stack(stop_loss_values_all).mean()
                if stop_loss_values_all else action_total * 0.0
            )
        else:
            qdrop_total = qdrop_weighted / max(1, active_action_boundaries)
            stop_total = stop_weighted / max(1, boundary_count)
        total = (
            self.action_weight * action_total
            + ess_scale * ess_total
            + self.qdrop_action_weight * qdrop_total
        )
        if bridge_align_values:
            total = total + align_scale * torch.stack(
                bridge_align_values
            ).mean()
        if bridge_specific_align_values:
            total = total + specific_align_scale * torch.stack(
                bridge_specific_align_values
            ).mean()
        if not bool(torch.isfinite(total)):
            raise FloatingPointError(
                f"nonfinite trajectory loss: {inputs['trajectory_id']}"
            )

        metrics = {
            "action_ce": action_total.detach(),
            "qdrop_action_ce": qdrop_total.detach(),
            "stop_loss": stop_total.detach(),
            "stop_accuracy": torch.tensor(
                sum(item["stop_correct"] for item in snapshots) / max(1, len(snapshots)),
                device=total.device,
            ),
            "stop_probability": torch.tensor(
                sum(item["stop_probability"] for item in snapshots) / max(1, len(snapshots)),
                device=total.device,
            ),
            "ess_nll": ess_total.detach(),
            "ess_effective_weight": torch.tensor(ess_scale, device=total.device),
            "adjacent_cos": torch.stack(adjacent_values).mean()
                if adjacent_values else total.detach() * 0.0,
            "alpha": modules.alpha.detach(),
            "latent_rms": torch.stack(latent_rms_values).mean(),
            "semantic_token_rms": torch.stack(semantic_rms_values).mean(),
            "semantic_gate": torch.sigmoid(
                modules.semantic_action_bridge.gate_logit.detach().float()
            ),
            "boundaries_per_trajectory": torch.tensor(float(boundary_count), device=total.device),
            "active_action_boundaries": torch.tensor(
                float(active_action_boundaries), device=total.device
            ),
            "inactive_action_boundaries": torch.tensor(
                float(boundary_count - active_action_boundaries), device=total.device
            ),
            "active_action_tokens": torch.tensor(
                float(action_tokens), device=total.device
            ),
            "raw_tokens": inputs["raw_token_count"].float().mean(),
            "physical_tokens": inputs["physical_token_count"].float().mean(),
        }
        if bridge_align_values:
            metrics["bridge_align_loss"] = torch.stack(
                bridge_align_values
            ).mean()
            metrics["bridge_align_effective_weight"] = torch.tensor(
                align_scale, device=total.device
            )
        if bridge_specific_align_values:
            metrics["bridge_specific_align_loss"] = torch.stack(
                bridge_specific_align_values
            ).mean()
            metrics["bridge_specific_align_effective_weight"] = torch.tensor(
                specific_align_scale, device=total.device
            )
        if getattr(self, "_offline_align_targets", False):
            metrics["bridge_align_misses"] = torch.tensor(
                float(getattr(self, "_bridge_align_misses", 0)),
                device=total.device,
            )
        all_align_cos = []
        for align_group, values in bridge_align_by_group.items():
            if values:
                align_group_matrix = torch.stack(values)
                all_align_cos.append(align_group_matrix)
                for field_index, name in enumerate(FIELD_NAMES):
                    column = align_group_matrix[:, field_index]
                    if bool(torch.isfinite(column).any().item()):
                        metrics[f"bridge_align_cos_{align_group}_{name}"] = (
                            torch.nanmean(column)
                        )
        if all_align_cos:
            metrics["bridge_align_cos"] = torch.nanmean(torch.cat(all_align_cos))
        all_specific_align_cos = []
        for align_group, values in bridge_specific_align_by_group.items():
            if values:
                align_group_matrix = torch.stack(values)
                all_specific_align_cos.append(align_group_matrix)
                for field_index, name in enumerate(FIELD_NAMES):
                    column = align_group_matrix[:, field_index]
                    if bool(torch.isfinite(column).any().item()):
                        metrics[
                            f"bridge_specific_align_cos_{align_group}_{name}"
                        ] = torch.nanmean(column)
        if all_specific_align_cos:
            metrics["bridge_specific_align_cos"] = torch.nanmean(
                torch.cat(all_specific_align_cos)
            )
        if qdrop_semantic_grad_values:
            metrics["qdrop_semantic_grad_rms"] = torch.stack(
                qdrop_semantic_grad_values
            ).mean()
        if answer_ready_nll_values:
            metrics["ess_answer_ready_nll"] = torch.stack(answer_ready_nll_values).mean()
            metrics["ess_answer_nll"] = metrics["ess_answer_ready_nll"]
        if tool_ess_nll_values:
            metrics["ess_tool_nll"] = torch.stack(tool_ess_nll_values).mean()
        for group, values in field_loss_by_group.items():
            if values:
                group_matrix = torch.stack(values)
                for field_index, name in enumerate(FIELD_NAMES):
                    metrics[f"ess_{group}_{name}"] = group_matrix[:, field_index].mean()
            if content_loss_by_group[group]:
                metrics[f"ess_{group}_content_nll"] = torch.stack(
                    content_loss_by_group[group]
                ).mean()
                metrics[f"ess_{group}_stop_nll"] = torch.stack(
                    stop_loss_by_group[group]
                ).mean()
        if branch_nll_values:
            metrics["branch_marker_nll"] = torch.stack(branch_nll_values).mean()
            metrics["branch_marker_accuracy"] = torch.stack(branch_correct_values).mean()
        if qdrop_branch_nll_values:
            metrics["qdrop_branch_marker_nll"] = torch.stack(qdrop_branch_nll_values).mean()
            metrics["qdrop_branch_marker_accuracy"] = torch.stack(qdrop_branch_correct_values).mean()
        for kind, values in per_kind.items():
            metrics[f"action_ce_{kind}"] = torch.stack(values).mean()
        # Legacy-compatible per-tool action tags.
        for tool_label, values in per_tool_action.items():
            metrics[f"action/{tool_label}/ce"] = torch.stack(values).mean()
        for tool_label, values in qdrop_per_tool.items():
            metrics[f"qdrop_action/{tool_label}/ce"] = torch.stack(values).mean()
        # Legacy-compatible all-active ESS field tags and per-tool field tags.
        if field_loss_values:
            all_fields = torch.stack(field_loss_values)
            for field_index, field_name in enumerate(FIELD_NAMES):
                metrics[f"ess/{field_name}_nll"] = all_fields[:, field_index].mean()
            for tool_label, fields in per_tool_field.items():
                for field_name, values in fields.items():
                    metrics[f"ess_by_tool/{field_name}/{tool_label}"] = torch.stack(values).mean()
        if content_loss_values:
            # Do not emit mixed tool/answer aggregates.  Answer boundaries
            # intentionally mask referent/grounding, so mixing them with tool
            # boundaries makes the resulting curves depend on batch makeup.
            # The un-mixed tool/answer metrics above are the comparison-safe
            # TensorBoard signals; these lists remain available for internal
            # loss bookkeeping and high-gradient snapshots.
            attention = torch.stack(query_attention_values).mean(dim=0)
            for field_index, name in enumerate(FIELD_NAMES):
                for latent_index in range(attention.shape[-1]):
                    metrics[f"query_attn_{name}_z{latent_index + 1}"] = attention[field_index, latent_index]
        if mode == "train" and torch.cuda.is_available() and mem_start_gb is not None:
            metrics["mem_start_gb"] = torch.tensor(mem_start_gb, device=total.device)
            metrics["mem_after_vision_gb"] = torch.tensor(mem_after_vision_gb, device=total.device)
            metrics["mem_after_qd_vision_gb"] = torch.tensor(mem_after_qd_vision_gb, device=total.device)
            metrics["mem_end_gb"] = torch.tensor(
                torch.cuda.memory_allocated() / (1024.0 ** 3), device=total.device
            )
            metrics["mem_peak_gb"] = torch.tensor(
                torch.cuda.max_memory_allocated() / (1024.0 ** 3), device=total.device
            )
            metrics["teacher_delta_gb"] = torch.tensor(
                self._diag_teacher_delta, device=total.device
            )
            metrics["teacher_peak_delta_gb"] = torch.tensor(
                self._diag_teacher_peak_delta, device=total.device
            )
            if inputs.get("image_grid_thw") is not None:
                metrics["image_count"] = torch.tensor(
                    float(inputs["image_grid_thw"].shape[0]), device=total.device
                )
            else:
                metrics["image_count"] = torch.tensor(0.0, device=total.device)
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
                "qdrop_action_ce": float(qdrop_total.detach().float()),
                "stop_loss": float(stop_total.detach().float()),
                "stop_accuracy": float(metrics["stop_accuracy"].detach().float()),
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
            "stop_head": modules.stop_head.state_dict(),
            "stages": modules.stages,
            "protocol": "trajectory_context_boundary_tbptt_v1",
            "cross_boundary_gradient": "detached",
            "cross_boundary_state": "prior latent KV retained",
            "within_boundary_gradient": "full recurrent BPTT",
            "cache": "incremental native KV",
            "answer_bridge_supervision": "full_suffix_including_</think>",
            "branch_marker_definition": "first_generated_control_token_after_zK",
            "qdrop_action_weight": self.qdrop_action_weight,
            "stop_weight": self.stop_weight,
            "semantic_action_bridge": modules.semantic_action_bridge.state_dict(),
            "protocol": "trajectory_joint_ess_semantic_action_routing_v1",
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
    ap.add_argument("--bridge-learning-rate", type=float, default=1e-4)
    ap.add_argument("--bridge-bottleneck", type=int, default=512)
    ap.add_argument("--bridge-gate-init", type=float, default=0.2)
    ap.add_argument(
        "--bridge-align-weight", type=float, default=0.0,
        help="Raw cosine weight aligning semantic bridge tokens with Decision-probe targets.",
    )
    ap.add_argument(
        "--bridge-specific-align-weight", type=float, default=0.0,
        help="Cosine weight on the field-centroid-orthogonal target component.",
    )
    ap.add_argument(
        "--bridge-align-targets", default="",
        help="Precomputed bridge hidden cache directory or legacy .pt file",
    )
    ap.add_argument(
        "--bridge-target-centroids", default="",
        help="Frozen per-field centroid file for sample-specific alignment.",
    )
    ap.add_argument("--ess-weight", type=float, default=0.02)
    ap.add_argument("--semantic-weight", type=float, default=0.0)
    ap.add_argument("--ess-stop-weight", type=float, default=0.2)
    ap.add_argument("--qdrop-action-weight", type=float, default=0.1,
                    help="Weight of action CE under masked-question replay.")
    ap.add_argument("--stop-weight", type=float, default=0.1,
                    help="Weight of boundary continue/stop BCE loss.")
    ap.add_argument("--stop-pos-weight", type=float, default=1.0,
                    help="Positive-class weight for answer/stop boundaries.")
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
    ap.add_argument("--init-from-checkpoint", default="",
                    help="Load model/latent/ESS weights only, but start a new optimizer and step counter.")
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
        int(model.config.hidden_size), ess_decoder, args.stages,
        bridge_bottleneck=args.bridge_bottleneck,
        bridge_gate_init=args.bridge_gate_init,
    ).to(dtype=torch.bfloat16)
    model.colt_modules.alpha.data=model.colt_modules.alpha.data.float()
    model.colt_modules.ess_decoder.query_bias_scale.data=model.colt_modules.ess_decoder.query_bias_scale.data.float()
    model.colt_modules.semantic_action_bridge.gate_logit.data = (
        model.colt_modules.semantic_action_bridge.gate_logit.data.float()
    )
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
        include_bridge_align=(
            args.bridge_align_weight > 0
            or args.bridge_specific_align_weight > 0
        ),
        include_qdrop=args.qdrop_action_weight > 0,
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
            include_bridge_align=(
                args.bridge_align_weight > 0
                or args.bridge_specific_align_weight > 0
            ),
            include_qdrop=args.qdrop_action_weight > 0,
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
        bridge_align_weight=args.bridge_align_weight,
        bridge_specific_align_weight=args.bridge_specific_align_weight,
        ess_stop_weight=args.ess_stop_weight,
        qdrop_action_weight=args.qdrop_action_weight,
        stop_weight=args.stop_weight,
        stop_pos_weight=args.stop_pos_weight,
        ess_warmup_steps=args.ess_warmup_steps,
        bridge_lr=args.bridge_learning_rate,
        bridge_align_targets=args.bridge_align_targets,
        bridge_target_centroids=args.bridge_target_centroids,
    )
    if args.init_from_checkpoint:
        trainer._load_from_checkpoint(args.init_from_checkpoint, model=model)
    # A checkpoint restores the saved gate logit; re-apply the CLI gate init
    # so this run starts from the requested semantic scale.
    if args.init_from_checkpoint or abs(float(args.bridge_gate_init) - 0.2) > 1e-6:
        gate_value = min(max(float(args.bridge_gate_init), 1e-4), 1.0 - 1e-4)
        model.colt_modules.semantic_action_bridge.gate_logit.data = torch.tensor(
            math.log(gate_value / (1.0 - gate_value)),
            dtype=torch.float32,
            device=model.colt_modules.semantic_action_bridge.gate_logit.device,
       )
    for parameter in model.colt_modules.stop_head.parameters():
       parameter.requires_grad_(False)
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
