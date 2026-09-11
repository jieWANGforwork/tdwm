# TDWM

TDWM 基于固定版本的 `stable-worldmodel[all]==0.1.1` 开展 world model
基线复现和后续方法研究。锁定的实验协议、数据来源与评测参数见
[`configs/README.md`](configs/README.md)。

当前正式运行的方法是 **Aligned E2E MC-GT-LeWM**：不加载 LeWM checkpoint 或 latent
cache，从 Cube 原始图像和随机初始化开始联合训练 LeWM 与 GoalTailValue。LeWM 原始
prediction MSE 和 SIGReg 使用 128 条独立的 4-frame clip；另一个 16 条 long-clip 数据流
在同一次 optimizer update 中监督 predicted terminal history 上的有限未来 MC tail。EMA
world model 提供稳定 target representation，value 架构严格保证
`V(h, z_current) = 0`。锁定训练协议见
[`configs/experiment/aligned_e2e_mc_gt_lewm_cube_train.yaml`](configs/experiment/aligned_e2e_mc_gt_lewm_cube_train.yaml)。

最新 paired ablation 固定 training seed 3072，并在六组 matched planning selections、
共 300 个 O50 episodes 上得到：Original LeWM world-only `51.0%`、Aligned world-only
`56.0%`、Aligned world + anchored tail `55.67%`。当前观察到的增益主要来自 Aligned
training 后的 world model，而不是 inference-time tail scoring；这些 planning-selection
seeds 不是独立 training seeds，不能据此声称方法显著优于 baseline。完整逐 episode
归档、Holm-corrected paired statistics 和 provenance 见
[`reports/aligned_acd_cube_o50_seed3072_planning_seeds42_47.md`](reports/aligned_acd_cube_o50_seed3072_planning_seeds42_47.md)。

第一版 **E2E Joint TD-GT-LeWM** 已完成单 seed 训练与 CEM 评测，但其相关窗口伪 batch、
缺失零边界和 `o25` 评测没有严格实现预期 formulation。负结果与适用结论记录在
[`reports/e2e_joint_td_gt_lewm_cube_seed3072.md`](reports/e2e_joint_td_gt_lewm_cube_seed3072.md)。

旧的 **Joint TD-GT-LeWM** checkpoint 初始化实验只属于 frozen-representation dynamics
fine-tuning 诊断，不是端到端正式方法，也不能作为正式训练结果。

LS-LeWM 保留为后续 policy-conditioned successor 方向；其设计见
[`docs/ls_lewm_method.md`](docs/ls_lewm_method.md)，当前阶段不训练该扩展。

新的 **RF-Successor-LeWM** 已作为独立方法实现。它不复用 LS-LeWM 的
goal-conditioned policy 或 TD tail：LeWM 对同一候选 action prefix 预测
`z_1,...,z_K`，causal successor head 预测 `S_1,...,S_K`，EMA future latent 提供
直接 MC target，并用 successor increment 与同 horizon latent feature 的递推关系
连接两种监督。训练与评测入口分别是
[`scripts/train_rf_successor_lewm.py`](scripts/train_rf_successor_lewm.py) 和
[`scripts/evaluate_rf_successor_lewm.py`](scripts/evaluate_rf_successor_lewm.py)。当前状态是
实现和测试完成、正式 checkpoint 与受控多 seed 结果尚未产生，因此不作性能声明。
完整定义见 [`docs/rf_successor_lewm_method.md`](docs/rf_successor_lewm_method.md)。

## 趋动云上的 LeWM Cube 快速训练

Cube 的训练样本是随机 sequence clip。HDF5 在 `/gemini/data-*` 远程挂载上的随机
读取会让 GPU 长时间等待 I/O；把约 74 GB 的文件复制到 `code` 或 `/dev/shm` 也会在
每次新任务中重复付出传输成本，而且 `/dev/shm` 还可能触发容器内存上限。快速训练
使用 `stable-worldmodel==0.1.1` 的公开 Lance 读写接口：先一次性转换并发布数据集，
以后所有训练任务直接读取只读挂载，不再做启动时缓存。

### 前提

- 先按 [`configs/README.md`](configs/README.md) 生成并校验
  `cube_single_expert_chunk1.h5`。
- 使用 `scripts/convert_cube_lance.py` 转成
  `cube_single_expert_jpeg100.lance`。入口只调用 `swm.data.convert(...)`，固定 JPEG
  质量 100，并生成相邻的 `cube_single_expert_jpeg100.lance.manifest.json`。
- 把 Lance 目录和 manifest 一起发布为趋动云数据集，再挂载到离线训练任务。
- 使用离线训练而不是开发环境进行正式训练，并挂载代码、Cube 数据集和结果集。
- 所有 checkpoint、日志和结果写入 `$GEMINI_DATA_OUT`。不要写回只读的数据集
  挂载，也不要把唯一结果留在容器临时目录。

### 一次性转换

在可写环境中执行。脚本会先核对源 HDF5 的大小和 SHA-256，再转换全部 episode；
转换后会核对 episode 边界、`action` 的精确值、`observation` 的确定性 float32
转换，并抽样记录数值与 JPEG 像素误差。已有输出不会被覆盖，失败或未生成 manifest
的 Lance 目录不能用于训练。

```bash
python scripts/convert_cube_lance.py \
  /path/to/cube_single_expert_chunk1.h5 \
  /path/to/cube_single_expert_jpeg100.lance
```

### 单任务启动命令

下面的命令直接读取数据集挂载。冒烟、checkpoint 恢复检查和三个正式 seed 使用
同一个 Lance 版本；任务重新调度时不需要再次转换或复制，`--resume auto` 从持久
结果目录恢复。

```bash
set -euo pipefail

: "${GEMINI_CODE:?GEMINI_CODE is required}"
: "${GEMINI_DATA_IN1:?GEMINI_DATA_IN1 is required}"
: "${GEMINI_DATA_OUT:?GEMINI_DATA_OUT is required}"
test -d "$GEMINI_CODE/tdwm"
test -d "$GEMINI_DATA_IN1/cube_single_expert_jpeg100.lance"
test -r "$GEMINI_DATA_IN1/cube_single_expert_jpeg100.lance.manifest.json"
test -d "$GEMINI_DATA_OUT" && test -w "$GEMINI_DATA_OUT"

repo="$GEMINI_CODE/tdwm"
export TDWM_CUBE_DATASET="$GEMINI_DATA_IN1/cube_single_expert_jpeg100.lance"
export TDWM_RUN_ROOT="$GEMINI_DATA_OUT/tdwm"
export STABLEWM_HOME="$GEMINI_DATA_OUT/stable_worldmodel"
run_dir="$TDWM_RUN_ROOT/lewm_cube_training"
mkdir -p "$run_dir/logs" "$STABLEWM_HOME"

cd "$repo"
python -c "import stable_worldmodel as swm; print(swm.__file__)"

python scripts/train.py --smoke --resume never --seed 3072 \
  --config configs/experiment/lewm_cube_train.yaml \
  --dataset "$TDWM_CUBE_DATASET" --output-dir "$run_dir" \
  2>&1 | tee "$run_dir/logs/smoke_seed_3072.log"

python scripts/train.py --smoke --resume required --seed 3072 \
  --config configs/experiment/lewm_cube_train.yaml \
  --dataset "$TDWM_CUBE_DATASET" --output-dir "$run_dir" \
  2>&1 | tee -a "$run_dir/logs/smoke_seed_3072.log"

for seed in 0 42 3072; do
  python scripts/train.py --resume auto --seed "$seed" \
    --config configs/experiment/lewm_cube_train.yaml \
    --dataset "$TDWM_CUBE_DATASET" --output-dir "$run_dir" \
    2>&1 | tee "$run_dir/logs/seed_${seed}.log"
done
```

### 为什么采用这个布局

- Lance 按训练访问模式读取图像，避免巨型 HDF5 文件在远程挂载上的随机 chunk
  解压和请求延迟；PushT 基线也采用同一公开数据后端。
- JPEG 质量 100 保持分辨率不变，但仍不是逐像素无损。训练 manifest 会保存转换
  协议和抽样误差；所有 LeWM 重训 seed 必须使用同一个 Lance 数据集版本。
- 需要逐像素严格复现时仍使用锁定的 HDF5。JPEG-100 Lance 训练结果必须明确标记为
  快速数据变体，不能和原始像素结果混写。
- 数据不进入镜像或 RAM。平台数据集负责长期保存，`$GEMINI_DATA_OUT` 负责持久化
  checkpoint 和结果。

### 本工程的数据路径优化

训练入口默认启用两个只在 TDWM 内部实现、可以独立关闭的优化，不修改已安装的
`stable-worldmodel`：

- `stride_aware_lance` 只向 Lance 请求 LeWM 实际消费的 4 个观测帧；跨度内的 20 个
  action 仍完整读取并 reshape，因此样本、loss 和随机 clip 顺序不变。
- `device_image_preprocessing` 让 loader 保留 `uint8` 图像，Lightning 把 batch 移到
  GPU 后再执行同一套缩放、ImageNet normalization 和 resize。Cube 原始图像已经是
  224x224，因此主机内存、worker IPC 和 PCIe 上的像素 payload 从 float32 降为 uint8，
  即原来的四分之一。

云端正式训练前，用相同 seed 和 100 step 分别测优化路径与原始路径；两个 run 使用
独立目录，训练 manifest 会记录实际生效的开关和 loader 参数：

```bash
python scripts/train.py --seed 3072 --resume never --max-steps 100 \
  --skip-validation --run-label optimized \
  --dataset "$TDWM_CUBE_DATASET" --output-dir "$run_dir"

python scripts/train.py --seed 3072 --resume never --max-steps 100 \
  --skip-validation --run-label upstream-loader \
  --no-stride-aware-lance --no-device-image-preprocessing \
  --dataset "$TDWM_CUBE_DATASET" --output-dir "$run_dir"
```

## PushT 离线 baseline 入口

PR #1 的 PushT 训练适配已并入现有 Cube 研究入口，但使用独立的
`src/tdwm/training/lewm_pusht.py` 模块，避免覆盖当前的 Cube LeWM 实现。LeWM、PLDM、
DINO-WM、GCBC、GCIVL 和 GCIQL 均读取 `configs/envs/pusht.yaml` 及对应方法配置；
TD-MPC2 仍保持 protocol-gated，不纳入这个 offline、reward-free 比较。

在已解压的 PushT HDF5 数据集上运行单个方法：

```bash
python scripts/train.py \
  --env pusht --method lewm --seed 3072 \
  --dataset /path/to/pusht_expert_train.h5 \
  --run-root /path/to/persistent/tdwm-runs
```

评测使用相同的 dataset-backed `world.evaluate(...)` 协议：

```bash
python scripts/evaluate.py \
  --env pusht --method lewm \
  --checkpoint lewm_pusht_seed3072_FINGERPRINT/weights_epoch_10.pt \
  --dataset /path/to/pusht_expert_train.h5 \
  --run-root /path/to/persistent/tdwm-runs
```

`scripts/launch_pusht_parallel.py` 可在 GPU 节点上按空闲显存排队运行其余五个
baseline。当前 PushT 适配只记录 seed `3072` 的可复现实验入口，不据此作多 seed
性能结论。

## 可复现 GPU 容器

根目录 `Dockerfile` 只接受平台提供的 CUDA base image；PyTorch、torchvision 和 CUDA
不会被 TDWM 选择或升级。构建前会记录并在构建后校验加速器版本，同时安装精确的
`stable-worldmodel[all]==0.1.1`：

```bash
docker build --build-arg BASE_IMAGE=your-existing-gpu-image:tag -t tdwm:runtime .
```

数据集、缓存和 checkpoint 应通过外部只读/持久化挂载提供，不进入镜像。

EffAction / EffActionPlan 的方法、实现状态与训练评测入口见
[方法说明](docs/eff_action.md)。配置中的 provisional 项尚待确认，不能视为正式实验结果。
各次运行保存配置、checkpoint 和逐 episode 结果，以便核查并汇入结果文档。
