# SELSA training-code audit bundle (2026-09-16)

This directory contains read-only copies of the training implementations requested for an independent code audit. No remote files or running jobs were changed.

## Versions

- `gpu09_bridge_current/`: the currently running gpu09 Bridge trainer and its launch/support files. The active process uses `train_joint_semantic_route.py` with `--action-weight 1.0`, `--ess-weight 0.3`, `--qdrop-action-weight 0.0`.
- `action_only_ess/`: the common `train_ess_v3.py` trainer used for the historical Action-only and Action+ESS comparison. The two variants differ by command-line weights: Action-only uses `ess_weight=0`; ESS uses a positive ESS weight. The corresponding checkpoints are `ac_noess_stop_v1` and `ac_ess_nc_stop_v1`.
- `qdrop_current/`: current standalone Action-only + Q-Drop implementation and launchers. The pure run uses `action_weight=0`, `ess_weight=0`, `qdrop_action_weight=1.0`.
- `gpu09_bridge_qdrop_variant/`: the Bridge trainer's Q-Drop variant, kept separately because it is not the currently running gpu09 command.

## Provenance

All files were fetched from the shared `/public/f_data/lsl/` tree through the ljt login host. Exact source paths, local paths, byte sizes and SHA256 hashes are recorded in `provenance.tsv`.

The current gpu09 process was inspected before copying; its command points to `ess_joint_semantic_route_v2/code/train_joint_semantic_route.py` and output `ess_joint_semantic_route_v2/checkpoints/bridge_topo_v1`.
