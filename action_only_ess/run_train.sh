#!/usr/bin/env bash
set -euo pipefail
ROOT=/public/f_data/lsl/ess_v3_nc_v1
PY=/public/f_data/conda_envs/lsl/bin/python
TR=/public/f_data/conda_envs/lsl/bin/torchrun
GUARD=/public/f_data/lsl/data_governance/training_data_guard.py
VENDOR=/public/f_data/lsl/ess_sft_v1/vendor/mcp_latent_distill
MODEL=/public/home/ljt/lsl/models/qwen2_5_vl_7b_teacher_student
DECODER=/public/home/ljt/hf_models/Qwen2.5-0.5B
PROMPT=/public/f_data/lsl/ess_sft_v2/mcp_swift_prompt_short.txt
DEEPSPEED=/public/f_data/lsl/ess_sft_v2/zero2_bf16.json
DATA=${DATA:-$ROOT/data/train_stop.jsonl}
EVAL=${EVAL:-$ROOT/data/eval_stop.jsonl}
RUN_NAME=${RUN_NAME:-main_action_ess}
OUT=${OUT:-$ROOT/checkpoints/$RUN_NAME}
LOG=${LOG:-$ROOT/logs/$RUN_NAME.log}
ESS_WEIGHT=${ESS_WEIGHT:-0.3}
MAX_STEPS=${MAX_STEPS:--1}
GAS=${GAS:-8}
SAVE_STEPS=${SAVE_STEPS:-250}
EVAL_STEPS=${EVAL_STEPS:-250}
EVAL_DECODE_CASES=${EVAL_DECODE_CASES:-20}
export PYTHONPATH=$ROOT/code:$VENDOR
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy || true
mkdir -p "$OUT" "$(dirname "$LOG")"
(cd / && sha256sum -c "$ROOT/reports/train_eval_stop_sha256.txt")
"$PY" "$GUARD" --input "$DATA"
"$PY" "$GUARD" --input "$EVAL"
exec "$TR" --standalone --nproc_per_node=8 "$ROOT/code/train_ess_v3.py" \
  --model "$MODEL" --decoder-model "$DECODER" --system-prompt "$PROMPT" \
  --image-root /public/f_data/lsl/images --data "$DATA" --eval-data "$EVAL" --output-dir "$OUT" \
  --stages 4 --max-images 4 --max-length 6000 --max-answer-tokens 2048 --ess-max-length 384 \
  --max-pixels 3211264 --gradient-accumulation-steps "$GAS" \
  --learning-rate 1e-5 --aux-learning-rate 1e-5 --ess-learning-rate 5e-5 \
  --action-weight 1 --ess-weight "$ESS_WEIGHT" --semantic-weight 0 \
  --ess-stop-weight 0.1 --ess-warmup-steps 100 --num-train-epochs 1 --max-steps "$MAX_STEPS" \
  --save-steps "$SAVE_STEPS" --eval-steps "$EVAL_STEPS" --logging-steps 1 \
  --eval-decode-cases "$EVAL_DECODE_CASES" --eval-decode-max-new-tokens 256 \
  --length-bucket-width 256 --dataloader-workers 2 --lr-scheduler-type cosine \
  --high-grad-threshold 20 --high-grad-max-events 32 \
  --data-audit-out "$OUT/trajectory_data_audit.json" --deepspeed "$DEEPSPEED" \
  2>&1 | tee "$LOG"
