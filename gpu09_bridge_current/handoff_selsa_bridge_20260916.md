# SELSA / Bridge 实验接手文档（2026-09-16）

这份文档给下一位 AI 使用。目标是让接手者不依赖聊天历史，也不接触任何密码或私钥，就能安全地登录、监控当前实验，并理解为什么这样训练。

## 0. 先读这一节：当前状态

- 当前正在运行的是修复 action_active 门控后的 Bridge 版本，从 step 0 重新开始。
- 运行主机：ljt 集群的 gpu09，8 张 A100。
- 输出：/public/f_data/lsl/ess_joint_semantic_route_v2/checkpoints/bridge_specific_g05_v1_gated
- 日志：/public/f_data/lsl/ess_joint_semantic_route_v2/logs/bridge_specific_g05_v1_gated.log
- 启动脚本：/public/f_data/lsl/ess_joint_semantic_route_v2/code/run_bridge_specific_g05_v1.sh
- 最近一次核对时进程仍在运行，约在 global step 45，GPU 显存约 34--35 GB/卡；没有发现 OOM 或 NaN。
- 修复前的 bridge_specific_g05_v1 不要作为最终结果：它把纠错首轮的 inactive action 错误地算进了 action loss，只能用于审计对比。

## 1. 服务器与免密登录

### 服务器角色

| 主机 | 作用 | 重要说明 |
|---|---|---|
| `<amax-relay-host>` | 中转/开发/部分评测与数据整理 | 不是 ljt 的 GPU 主机；不要把 ljt 训练启动在这里。 |
| `<ljt-login-host>` | 集群登录、共享 NFS、TensorBoard 服务 | 登录节点没有 CUDA 训练环境。 |
| gpu08 / gpu09 | ljt 集群的实际 GPU 主机 | 所有训练、vLLM、GPU 评测必须在这里启动。 |

/public/f_data/lsl 是 ljt 登录节点、gpu08、gpu09 之间可见的共享 NFS。amax 的 /nvme/lsl、/ssd/lsl 与 ljt 的路径不是同一块本地盘；迁移后必须重新核对路径和图片映射。

### 登录安全约定

本机应使用已经配置好的 SSH key/agent，目标是命令在无密码提示下成功。文档不保存、不打印任何密码、私钥或 askpass 文件内容。

先做无副作用测试：

~~~bash
ssh -o BatchMode=yes -o ConnectTimeout=10 ljt@<ljt-login-host> 'hostname; whoami'
ssh -o BatchMode=yes -o ConnectTimeout=10 ljt@<ljt-login-host> 'ssh -o BatchMode=yes -o ConnectTimeout=10 gpu09 "hostname; whoami"'
ssh -o BatchMode=yes -o ConnectTimeout=10 <amax-user>@<amax-relay-host> 'hostname; whoami'
~~~

如果第一条失败，不要尝试猜密码，也不要读取 credential/askpass 文件；请让管理员把接手者自己的公钥加入对应账号，或启用 SSH agent。成功标准是 BatchMode=yes 下直接退出 0。

从 Windows PowerShell 进行复杂远程操作时，把脚本写成本地文件后再上传/执行，避免多层引号。不要用 PowerShell 重定向传输二进制；目录传输使用远端 tar 后再复制。

## 2. SELSA 的研究问题

SELSA = Semantic and Executable Latent States Agents。

我们要回答两个问题：

1. 四个紧凑的 recurrent latent 是否保存了下一步工具执行所需的、样本特异的语义？
2. action policy 是否真正读取这些语义，而不是只依赖 image/question 的直达路径？

历史实验的关键现象：

- latent 可以通过 CoLT/ESS reader 解码出目标实体、属性和方位等细粒度信息；
- cross-sample latent swap 对 action 的影响曾经很小，说明表示端和执行端之间存在 semantic--execution gap；
- Q-Drop（部分或全部去掉 Question）能显著增强 latent 的因果作用，但必须和正常 action 分支正确分开统计；
- 因此当前 Bridge 不是为了让 latent“看起来更像 hidden”，而是把 ESS 字段语义变成 action backbone 能消费的语义 token，并用因果干预验证行为变化。

论文叙事应区分：

~~~text
teacher trajectory
        ↓
4 recurrent latent states z1..z4
        ↓ ESS readers：referent / grounding(or evidence) / reason
        ↓ semantic-action bridge
action autoregressive decoder（tool_call 或 </think><answer>）
~~~

ESS 解码文本只用于监督/诊断，不反馈给 action decoder。action 没有独立的离散 action head，而是模型自回归生成文本 token。

## 3. 当前架构与梯度路径

### 3.1 Latent 生成

- 每个可执行 reasoning boundary 产生 K=4 个连续 latent。
- boundary 内使用 Full BPTT；不同 boundary 之间保留 trajectory context，但训练时按边界切断梯度，避免整条长轨迹反传导致 OOM。
- latent 通过增量 KV cache 参与后续状态；历史 latent 的 KV 可在同一 trajectory 中继续保留。
- backbone 使用 FlashAttention/KV cache 的增量路径；action forward 按训练协议使用完整语义上下文。

### 3.2 ESS reader

当前 Bridge 代码的 FIELD_NAMES 是：

~~~text
referent, evidence, reason
~~~

字段独立解码：每个字段有独立 learned query，query 读取完整的 z1..z4 轨迹，送入冻结的 Qwen2.5-0.5B 语言 decoder（带 LoRA 的版本取决于实验脚本）。当前这条 Bridge 训练同时优化 ESS reader 的相关可训练参数；接手时以运行脚本和 checkpoint manifest 为准，不要把旧版字段名（grounding/disambiguation/need）混写进当前结果。

### 3.3 Semantic-action bridge

- 从字段 query/ESS 表征生成 3 个语义 token，并注入 action 路径。
- 使用 LayerNorm、降维/非线性/升维和输出归一化，减少表示尺度漂移。
- bridge 输出按 detached latent RMS 做幅度归一，并乘以 sigmoid gate；本次 gate 初始化为 0.5，保持 gate 的 FP32 精度。
- bridge_align_loss：bridge token 与冻结 teacher Decision-probe hidden 的原始方向对齐。
- bridge_specific_align_loss：先减去字段 centroid，再对齐 sample-specific residual，避免只学公共字段原型。
- 当前权重：raw align=0.02，specific align=0.1；这两个是辅助项，主行为仍由 action CE 和 ESS loss 决定。

梯度关系（本次默认 Q-Drop 关闭）：

~~~text
action CE ----------------→ backbone LoRA / transition / alpha / bridge
ESS field CE --------------→ ESS reader，并通过 recurrent graph 回到 latent 生成参数
raw/specific bridge align → bridge，并通过 bridge 输入路径回到 ESS/latent（按实现中的 detach 点）
~~~

不要把 stop_head 当作真正的 action 分支；当前 stop head 只作诊断，stop-weight=0。模型通过普通 action CE 学习 </think>、<answer> 等协议。

## 4. 数据契约与门禁（训练前必须执行）

### 4.1 数据

- 训练：/public/f_data/lsl/ess_v3_nc_v1/data/train_stop.jsonl
- 评估：/public/f_data/lsl/ess_v3_nc_v1/data/eval_stop.jsonl
- 系统 prompt：/public/f_data/lsl/ess_sft_v2/mcp_swift_prompt_short.txt
- 图片根目录：/public/f_data/lsl/images
- teacher Decision-probe targets：/public/f_data/lsl/ess_v3_nc_v1/bridge_hidden_v1
- field centroids：/public/f_data/lsl/ess_v3_nc_v1/bridge_hidden_v1/field_centroids.pt

当前 train_stop.jsonl 约 679k 行，其中约 76,116 个 boundary 是 action_active=False 且 ess_active=False 的纠错首轮错误动作。它们可以出现在上下文中，但不能产生 action/ESS label 或统计分母。

### 4.2 必须的 blacklist

- ljt：/public/f_data/lsl/data_governance/training_exclusions.json
- amax：/nvme/lsl/data_governance/training_exclusions.json

过滤的是 parent trajectory，不只是某个错误 span。所有训练、ESS/cache 复用、评测样本抽取都必须先过同一黑名单门禁。

### 4.3 图片与协议不变量

- 非 run_code 样本必须有真实图片；空图、缺图、黑图/placeholder 都是硬错误，不能静默 fallback。
- 只有明确标记为 run_code 的样本才允许 image-less。
- portable image path 需要在运行时映射到服务器本地根目录，并把映射写入 run/checkpoint manifest。
- raw portable manifest 保持只读，不直接改写。

门禁示例（使用实际文件前先确认脚本路径）：

~~~bash
python /public/f_data/lsl/data_governance/training_data_guard.py \
  --input /public/f_data/lsl/ess_v3_nc_v1/data/train_stop.jsonl
python /public/f_data/lsl/data_governance/training_data_guard.py \
  --input /public/f_data/lsl/ess_v3_nc_v1/data/eval_stop.jsonl
~~~

如果门禁非 0，停止启动，不要用旧 cache 或旧 filtered data 绕过。

## 5. 本次修复了什么

旧 Bridge trainer 没有读取 action_active，导致 inactive correction-first spans 被无条件纳入：

1. batch 中加入 action_active；
2. inactive 边界的普通 action label 和 Q-Drop label 全部设为 -100；
3. inactive action 的 token count、per-tool/per-kind action CE 不计入；
4. action token denominator 只使用 active token；
5. ESS 原有的 ess_active 门控继续保留；
6. 日志新增 active_action_boundaries、inactive_action_boundaries、active_action_tokens。

修复后的本地源：E:/CodexWork/train_joint_semantic_route_current.py

修复后的 ljt 源：/public/f_data/lsl/ess_joint_semantic_route_v2/code/train_joint_semantic_route.py

两边 SHA256（已核对）为：

~~~text
be03767e1957b55d81e4525fe85fc414eba62b2404671270dd8587cbe8894bbe
~~~

远端 py_compile 已通过；2-step smoke 已通过，峰值约 23.3 GB/卡，无 OOM。

## 6. 当前运行配置

入口：/public/f_data/lsl/ess_joint_semantic_route_v2/code/run_bridge_specific_g05_v1.sh

关键配置如下，完整命令以该脚本和日志首段为准：

~~~text
model                 /public/home/ljt/lsl/models/qwen2_5_vl_7b_teacher_student
decoder-model         /public/home/ljt/hf_models/Qwen2.5-0.5B
stages                4
max-images            4
max-length            6500
max-answer-tokens     2048
ess-max-length        384
max-pixels            3211264
gradient accumulation 8
main lr               1e-5
aux lr                1e-5
ess lr                5e-5
bridge lr             2e-5
bridge bottleneck     512
bridge gate init      0.5
raw align weight      0.02
specific align weight 0.1
action weight          1.0
qdrop action weight    0.0
ess weight             0.3
ess stop weight        0.1
stop weight            0.0
warmup steps           100
epochs                1
save/eval every        250 steps
eval decode cases      20
eval decode max        256 tokens
data loader workers    2
deepspeed              zero2_bf16.json
~~~

当前 run 从 step 0 开始，不要从被污染的旧 Bridge checkpoint resume。

## 7. 安全启动与监控

### 7.1 GPU 进程

训练必须在 GPU 主机上：

~~~bash
ssh -o BatchMode=yes ljt@<ljt-login-host> 'ssh -o BatchMode=yes gpu09 "hostname; nvidia-smi -L"'
~~~

不要在 ljt 登录节点直接运行 torchrun/vLLM。安全的 detached 模式是在 gpu09 上执行 setsid nohup，并把标准输入输出全部断开；启动后用独立命令核对进程和日志，不依赖 launcher 的退出码。

### 7.2 进度查看

~~~bash
ssh -o BatchMode=yes ljt@<ljt-login-host> 'ssh -o BatchMode=yes gpu09 "pgrep -af bridge_specific_g05_v1_gated | head -4; tail -n 20 /public/f_data/lsl/ess_joint_semantic_route_v2/logs/bridge_specific_g05_v1_gated.log"'
~~~

GPU 查询：

~~~bash
ssh -o BatchMode=yes ljt@<ljt-login-host> 'ssh -o BatchMode=yes gpu09 "nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader"'
~~~

只看短尾日志和明确路径，不要对 /public 做无约束递归 find。

### 7.3 TensorBoard

已有共享服务在 ljt 登录节点的 127.0.0.1:16030，包含旧的 bridge_g05_specific、ess、noess 标签。不要杀掉这个服务。

本地端口转发示例：

~~~bash
ssh -N -L 6030:127.0.0.1:16030 ljt@<ljt-login-host>
~~~

新 gated run 尚未自动加入旧 TensorBoard。要比较时，建议在登录节点另起一个服务或显式增加一个 logdir entry，单独使用新本地端口，例如把：

~~~text
bridge_gated:/public/f_data/lsl/ess_joint_semantic_route_v2/checkpoints/bridge_specific_g05_v1_gated/runs
~~~

加入 logdir spec，然后转发到一个空闲端口。不要把 contaminated bridge_specific_g05_v1 误命名为最终 gated run。

## 8. 健康指标与解释

每 250 步 checkpoint/eval 要使用同一套口径比较：

- action_ce：只在 action_active=True token 上统计；分开看 crop、zoom、rotate、enhance、run_code、answer。
- ess：按 referent、evidence、reason 分开看，另看非 answer 与 answer-ready 边界。
- bridge_align_loss 与 bridge_specific_align_loss：分别看 raw 和 sample-specific residual。
- bridge gate、semantic token RMS、latent RMS、adjacent_cos(z_i,z_{i+1})。
- active_action_boundaries、inactive_action_boundaries、active_action_tokens：确认分母没有回到旧口径。
- 梯度：总 grad、transition/alpha、bridge、ESS reader 分开；关注高 grad 峰是否伴随工具 CE 上升。

健康的训练不要求 adjacent cosine 越高越好；需要同时满足：工具 CE 不持续恶化、answer CE 稳定、bridge/ESS loss 有下降趋势、gate/RMS 不爆炸、无 NaN/OOM。若 cosine 下降同时 tool CE 上升，应判为递推不稳定而不是“成功分工”。

## 9. 既有版本与比较纪律

- ess_joint_semantic_route_v1/checkpoints/bridge_v1_gpu09_final：早期 raw-only bridge，gate 较小；并且同样受旧 action_active bug 影响，不能和本次 gated run 直接做最终公平比较。
- ess_joint_semantic_route_v2/checkpoints/bridge_specific_g05_v1：本次修复前的 raw+specific 版本，污染 run，仅作问题定位。
- ess_joint_semantic_route_v2/checkpoints/bridge_specific_g05_v1_gated：本次从 step 0 的修复版，当前主结果。
- .../bridge_specific_g05_v1_gated_smoke：2-step smoke，不能作性能结论。

比较必须统一：checkpoint step、eval manifest、图片映射、active-only 分母、解码协议和 agent runner。不同协议（replay/rebuild/stateful incremental）不得混作模型差异。

## 10. 推理/评测注意事项

- agent runner 应优先使用 stateful incremental KV 版本：eval_stateful_agent_vstar20_v2.py。它在每个 boundary 重新计算真实 MRoPE 位置，但只增量写入新加入的 tool_result/图像区间，并保留历史 latent KV。
- 不要把 force_answer 路径当作自然推理；它可能跳过 latent marker/rollout。若评测停止能力，要明确记录自然输出还是强制输出。
- 工具结果必须真实执行并保存；裁剪框参考当前输入图，而不是默认原图。黑图、缺图、错误坐标应直接标记失败并进入黑名单，不可用 placeholder 代替。
- 评测中若解析不到 </think>、<answer> 或工具闭合标记，保存 raw generation 并标记协议失败，不要静默补答案后计分。

## 11. 接手后的建议顺序

1. 先确认当前 gated 训练进程、GPU 利用率、日志心跳；不动其他用户任务。
2. 到 step 250 时确认 checkpoint 文件完整，读取 active/inactive 统计和 TensorBoard。
3. 用固定 held-out 小集做一次自然 agent 推理，检查 action/ESS/工具结果链路，再跑 VStar 或 HRBench 的正式评测。
4. 在同一 checkpoint 上做 paired intervention：matched latent、cross-sample latent、zero/noise latent；只比较行为变化，不把 ESS reader NLL 当作 action 因果性的替代指标。
5. 只有修复版完成并通过数据/图片/协议门禁后，才与 ESS、NoESS、旧 Bridge 做最终表格。
6. 若要 resume，必须使用该 gated run 自己生成的完整 checkpoint，并先检查 trainer_state.json、optimizer、scheduler、rng state 和 bridge 权重是否都存在；不要 resume 旧污染 run。

## 12. 常见错误清单

- 不要把中转机和 ljt 登录节点当成同一台服务器。
- 在 ljt 登录节点启动 GPU 任务，导致没有 CUDA/libcuda。
- 用密码或 askpass 文件写进脚本/文档；正确做法是本地 SSH key/agent + BatchMode=yes。
- 用 first occurrence 找重复 tool span；重复调用必须优先匹配最后一次真实边界。
- 把 inactive correction-first action 当作监督；它必须保留为上下文但 label=-100。
- 把旧 full-cache、旧 Decision probe cache、不同坐标系的图像路径混合使用。
- 用 rm -rf、git reset --hard 或删除别人的 tmux/进程来“清理”。任何删除前先做精确盘点并保留可恢复副本。

这份文档本身不包含任何服务器密码、私钥或 token。接手者只需拥有经过管理员授权的 SSH key/agent，并遵守上述门禁即可继续工作。
