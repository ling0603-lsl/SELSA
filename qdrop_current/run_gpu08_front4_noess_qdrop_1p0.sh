#!/usr/bin/env bash
# Action-only (NoESS) + QDrop(1.0), 4 GPUs (physical 0-3) on gpu08.
# Starts from the previously trained action-only checkpoint
#   ess_v3_nc_v1/checkpoints/ac_noess_stop_v1/checkpoint-2500
# and adapts it with the Question-masked action branch at weight 1.0.
set -euo pipefail
ROOT=/public/f_data/lsl/ess_v3_qdrop_v1
SRC=/public/f_data/lsl/ess_v3_nc_v1
PY=/public/f_data/conda_envs/lsl/bin/python
TR=/public/f_data/conda_envs/lsl/bin/torchrun
GUARD=/public/f_data/lsl/data_governance/training_data_guard.py
VENDOR=/public/f_data/lsl/ess_sft_v1/vendor/mcp_latent_distill
MODEL=/public/home/ljt/lsl/models/qwen2_5_vl_7b_teacher_student
DECODER=/public/home/ljt/hf_models/Qwen2.5-0.5B
PROMPT=/public/f_data/lsl/ess_sft_v2/mcp_swift_prompt_short.txt
DEEPSPEED=/public/f_data/lsl/ess_sft_v2/zero2_bf16.json
DATA=$SRC/data/train_stop.jsonl
EVAL=$SRC/data/eval_stop.jsonl
INIT=$SRC/checkpoints/ac_noess_stop_v1/checkpoint-2500
OUT=$ROOT/checkpoints/qdrop_noess_1p0_v1
LOG=$ROOT/logs/train_noess_1p0_4gpu.log

test -s "$INIT/adapter_model.safetensors"
test -s "$INIT/colt_modules.pt"
test -s "$INIT/ar_ess_decoder.pt"
test -s "$DATA"
test -s "$EVAL"
mkdir -p "$OUT" "$ROOT/logs"
(cd / && sha256sum -c "$SRC/reports/train_eval_stop_sha256.txt")
"$PY" "$GUARD" --input "$DATA"
"$PY" "$GUARD" --input "$EVAL"

export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONPATH=$ROOT/code:$VENDOR
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy || true

echo "[launch] action-only + QDrop(1.0) 4gpu on gpu08 gpus 0-3 init=$INIT" >>"$LOG"
exec "$TR" --standalone --nproc_per_node=4 "$ROOT/code/train_ess_v3_qdrop.py" \
  --model "$MODEL" --decoder-model "$DECODER" --system-prompt "$PROMPT" \
  --image-root /public/f_data/lsl/images --data "$DATA" --eval-data "$EVAL" \
  --output-dir "$OUT" --init-checkpoint "$INIT" \
  --stages 4 --max-images 4 --max-length 6000 --max-answer-tokens 2048 \
  --ess-max-length 384 --max-pixels 3211264 --gradient-accumulation-steps 8 \
  --learning-rate 1e-5 --aux-learning-rate 1e-5 --ess-learning-rate 5e-5 \
  --action-weight 0 --ess-weight 0 --qdrop-action-weight 1.0 \
  --semantic-weight 0 --ess-stop-weight 0.1 --ess-warmup-steps 1 \
  --num-train-epochs 1 --max-steps -1 --save-steps 250 --eval-steps 250 \
  --logging-steps 1 --eval-decode-cases 0 \
  --length-bucket-width 256 --dataloader-workers 2 --lr-scheduler-type cosine \
  --high-grad-threshold 20 --high-grad-max-events 32 \
  --data-audit-out "$OUT/trajectory_data_audit.json" --deepspeed "$DEEPSPEED" \
  >>"$LOG" 2>&1