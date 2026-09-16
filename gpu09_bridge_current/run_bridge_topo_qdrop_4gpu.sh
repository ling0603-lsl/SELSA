#!/usr/bin/env bash
# Bridge (topology-unified) + Action QDrop, 4 GPUs (physical 4-7) on gpu08.
# Uses train_joint_semantic_route_qdrop.py = topology-unified trainer with the
# Q-Drop backward made rank-symmetric (unconditional) to avoid the NCCL hang.
set -euo pipefail
ROOT=/public/f_data/lsl/ess_joint_semantic_route_v2
PY=/public/f_data/conda_envs/lsl/bin/python
TR=/public/f_data/conda_envs/lsl/bin/torchrun
GUARD=/public/f_data/lsl/data_governance/training_data_guard.py
TRAIN=/public/f_data/lsl/ess_v3_nc_v1/data/train_stop.jsonl
EVAL=/public/f_data/lsl/ess_v3_nc_v1/data/eval_stop.jsonl
OUT=${OUTPUT_DIR:-$ROOT/checkpoints/bridge_topo_qdrop_v2}
LOG=${LOG_FILE:-$ROOT/logs/bridge_topo_qdrop_v2.log}
MAX_STEPS=${MAX_STEPS:--1}
TARGETS=/public/f_data/lsl/ess_v3_nc_v1/bridge_hidden_v1
CENTROIDS=$TARGETS/field_centroids.pt
PROMPT=/public/f_data/lsl/ess_sft_v2/mcp_swift_prompt_short.txt
DEEPSPEED=/public/f_data/lsl/ess_sft_v2/zero2_bf16.json
MODEL=/public/home/ljt/lsl/models/qwen2_5_vl_7b_teacher_student
DECODER=/public/home/ljt/hf_models/Qwen2.5-0.5B
TRAINER=$ROOT/code/train_joint_semantic_route_qdrop.py
mkdir -p "$OUT" "$(dirname "$LOG")"
test -s "$CENTROIDS"
test -s "$TRAINER"
export CUDA_VISIBLE_DEVICES=4,5,6,7
export PYTHONPATH="$ROOT/code:/public/f_data/lsl/ess_sft_v1/vendor/mcp_latent_distill:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy || true
(cd / && sha256sum -c /public/f_data/lsl/ess_v3_nc_v1/reports/train_eval_stop_sha256.txt)
"$PY" "$GUARD" --input "$TRAIN"
"$PY" "$GUARD" --input "$EVAL"

echo "[launch] bridge + QDrop (0.2) 4gpu gpus 4-7 trainer=$TRAINER out=$OUT" >>"$LOG"
exec "$TR" --standalone --nproc_per_node=4 "$TRAINER" \
  --model "$MODEL" --decoder-model "$DECODER" \
  --system-prompt "$PROMPT" --image-root /public/f_data/lsl/images \
  --data "$TRAIN" --eval-data "$EVAL" --output-dir "$OUT" \
  --stages 4 --max-images 4 --max-length 6500 --max-answer-tokens 2048 \
  --ess-max-length 384 --max-pixels 3211264 \
  --gradient-accumulation-steps 8 \
  --learning-rate 1e-5 --aux-learning-rate 1e-5 --ess-learning-rate 5e-5 \
  --bridge-learning-rate 2e-5 --bridge-bottleneck 512 --bridge-gate-init 0.5 \
  --bridge-align-weight 0.02 --bridge-specific-align-weight 0.1 \
  --bridge-align-targets "$TARGETS" --bridge-target-centroids "$CENTROIDS" \
  --action-weight 1.0 --qdrop-action-weight 0.2 \
  --ess-weight 0.3 --ess-stop-weight 0.1 --stop-weight 0.0 \
  --ess-warmup-steps 100 --num-train-epochs 1 --max-steps "$MAX_STEPS" \
  --save-steps 250 --eval-steps 250 --eval-decode-cases 20 \
  --eval-decode-max-new-tokens 256 --logging-steps 1 \
  --length-bucket-width 256 --dataloader-workers 2 \
  --lr-scheduler-type cosine --high-grad-threshold 20 --high-grad-max-events 32 \
  --data-audit-out "$OUT/trajectory_data_audit.json" \
  --deepspeed "$DEEPSPEED" >>"$LOG" 2>&1