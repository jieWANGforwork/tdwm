# EffAction 与 EffActionPlan 方法和训练测试说明

EffAction 用动作条件下的累计特征网络 G 与路径代价网络 V 评价候选动作，再用 CEM 搜索动作。EffActionPlan 使用同一套 G/V 评价，通过学习到的 P 多次修正候选动作。两者共享 G/V 的定义，差别在动作如何产生。

本文对应《Results TD》中“方案二 基于 G 和 V 的动作规划”。数学模块、训练运行器、两个 CLI 与三个 YAML 已存在并通过 CPU smoke。2026-09-12 用户已裁定此前全部待定项，三个配置均为 `protocol_status: user_locked`，取值见下表的“已锁定”。**本文仍不是实验完成报告：尚未执行任何 GPU 训练或真实 Cube 评测，文中没有任何成功率或性能结论。**

## 当前状态与待确认事项

| 项目 | 当前依据或提议 | 状态 |
|---|---|---|
| 方法名 | EffAction；EffActionPlan | 已确定 |
| E 与 e | 沿用冻结的 LeWM 状态编码器及动作编码器 | 已确定 |
| G/V target | 共用分支 b；直接监督与 TD 并存；无折扣 | 已确定 |
| V 的梯度 | 训练 V 时不更新 G；训练 P 时保留输入动作梯度 | 已确定 |
| P | 六项输入、残差动作修正、真实动作重建后再加效率训练 | 已确定 |
| episode 留出 | 训练 G/V/P 使用 0–7999；评测 pair 已改为与 C–G3 完全相同的固定清单（`historical_cg3`，seed 42、50 对，三档 offset 各一份 sha256 锁定） | 已对齐：2026-09-12 用户决定两方法必须做同一批题 |
| 动作跨度 | 一个 25 维动作块，即 5 primitive steps，执行后重规划 | **已锁定**（首版不实现 25 步整计划） |
| cross-episode goals | 首版只使用同轨迹 future goals | **已锁定**（不加跨轨迹目标） |
| 效率分支阈值 | G/V 采样 η≥0.3；Planner 采样 η≥0.8 | **已锁定**，依据见下方只读校准 |
| V 架构 | 两层 256 隐藏单元，非负输出 softplus | **已锁定** |
| P 架构与迭代 | 两层 512 隐藏单元；K=8 | **已锁定** |
| stage 2 系数 | λ_traj=1，λ_eff=0.1；ε=1e-6 | **已锁定** |
| 训练预算 | G/V 阶段 127,960 updates；P stage 1=6,000，stage 2=6,000；seed=3072 | **已锁定** |
| 初始化与动作范围 | normalized zero 初始化；normalized bounds=[−3.5,3.5] | **已锁定**，依据见下方只读审计 |
| GPU 启动 | 当前容器为无卡模式，需切回 GPU 实例后启动 | **待用户操作**；另一方法当前无进程在跑 |

### 2026-09-12 只读校准（未训练）

对冻结缓存中 episodes 0–7999 的全部动作块（1,608,000 行）与 4096 个采样片段做了只读统计，结论直接写入配置：

| 量 | 实测 | 采用 |
|---|---|---|
| 有限动作 \|a\| 全局最大 | 3.4928（p99=2.548，p99.9=3.418） | 标量 bound 3.5，截断 0% |
| 逐坐标 max | [3.4928, 2.5479, 1.5590, 2.5469, 3.1814] 在 5 个 primitive step 上重复 | 统一标量 3.5（接口只支持标量） |
| bound=1.6 截断率 | 坐标 8.77%，动作块 49.24% | 弃用；它使 P 阶段一重建目标不可达 |
| η≥0.9（G/V 采样）直接分支比例 | 3.9% | 弃用 |
| η≥0.3（G/V 采样）直接分支比例 | 17.7% | **采用** |
| η≥0.9（Planner 采样）直接分支比例 | 15.6% | 弃用 |
| η≥0.8（Planner 采样）直接分支比例 | 22.9% | **采用** |
| 非有限动作行 | 40,000 / 1,608,000（2.5%），全部来自 episode 尾部 padding | 采样器校验并排除，不得进入 loss 或评测 |

两个采样器的 goal offset 范围不同（1–40 与 1–10 个块），长程片段的 latent 几何效率天然更低，因此阈值分别校准，不共用同一个数值。

旧 C–G3 的 10 epochs×12,796 updates=127,960 updates 是历史预算。新的 episode sampler 改变了样本总体，因此只写“10 epochs”不能证明等预算。新配置必须显式记录 optimizer updates、batch、采样分布与各阶段边界。

## 状态 动作与时间单位

设冻结状态编码器为 E，状态表示为 z_t=E(o_t)∈R^192；冻结动作编码器为 e，目标为 z_g=z_T，任务向量沿用项目归一化：

\[
m=\sqrt{192}\,\frac{z_g}{\|z_g\|_2}.
\]

这里归一化的是目标状态本身，不是 z_g−z_t，也不掺入旧 C–G3 的随机 Gaussian task。目标范数退化的样本不能直接进入该归一化。

当前冻结 e 的输入是一个 **25 维动作块**：连续 5 个 primitive steps，每步 5 个动作维度；这 5 个动作可以不同。e 将其编码为 192 维。本文 t、t+1 在该接口下相差一个动作块，即原始数据相隔 5 步。样本、target、动作梯度和 P 输出必须使用同一单位。

| 量 | 当前 EffAction 数学 API | RP1 论文中的 Cube refiner |
|---|---|---|
| 一次网络更新的动作对象 | 一个 25 维动作块 | 5 个动作块组成的 125 维完整计划 |
| 覆盖 primitive steps | 5 | 25 |
| 评价 | V(G(z_t,e(a),m),m) | V(H_F(A,z_t),z_g)，评价世界模型 rollout 的终点 |
| P 输入 | z_t、a_ref、a、z_g、J、∇_a J；单块时共 460 维 | A、value、action gradient；共 251 维 |
| P 输出 | 当前为 25 维增量 | 125 维计划增量 |

因此，**25 维动作不等于 25 步动作计划**。把 P 输出维度改成 125，也不会自动使冻结 e 或 G/V 支持整条计划。若用户选择一次计划 25 primitive steps，必须先定义整条计划怎样调用 e/G/V、目标如何监督、是否需要 F 生成中间状态，以及所有计划块如何参与评分，再完成对应实现。

同样，目标 offset O25/O100、计划长度、执行后重规划间隔、episode 总预算是四个独立量。不得因为目标距起点 100 步就认为一次网络要生成 100 步。

## G 和 V 的定义

真实同轨迹片段为 τ_(t:T)=(z_t,a_t,z_(t+1),…,z_T)，本次目标位置 T 不要求是整个 episode 的结尾。定义：

\[
\ell_k=\|z_{k+1}-z_k\|_2,\qquad
D_t=\|z_T-z_t\|_2,\qquad
W_t=\sum_{k=t}^{T-1}\ell_k,\qquad
\eta_t=\frac{D_t}{W_t+\epsilon}.
\]

在当前单块接口下，W 累加的是相邻五步 latent 节点之间的变化；它没有累加未编码的 primitive 中间帧。η 是 latent 几何路径效率，不是物理能量、动作幅度或环境步数奖励。

G 预测执行当前动作后，沿数据后续动作到达本次目标的累计状态特征；V 由累计表示和目标预测路径代价：

\[
\Psi_t=G_\theta(z_t,e(a_t),m),\qquad
v_t=V_\psi(\Psi_t,m)\ge0.
\]

G 的输出为 192 维；V 的输入只有 Ψ 与 m，若拼接则为 384 维。V 不额外绕过 G 读取 z_t。累计特征是表示，不能保证唯一决定路径代价；必须用验证结果检查动作区分能力。

### 共用直接或 TD 分支

b_t∈{0,1} 由真实完整片段决定。高效率片段进入直接分支，其余有效片段进入 TD；若设置额外长度上限，也必须进入同一套显式采样配置。**G 与 V 使用同一个 b_t、同一个目标 T。**

η 阈值已锁定：G/V 采样 η≥0.3，Planner 采样 η≥0.8，依据见前文只读校准。不能用候选动作的预测效率选真实监督分支，也不能把 RP1 的 n-step 窗口判定当成这里的 η 判定。

G 的 target 为：

\[
y^G_{t,\mathrm{direct}}=\sum_{k=t}^{T-1}z_{k+1},
\]
\[
y^G_{t,\mathrm{TD}}=z_{t+1}+(1-d_{t+1})\bar\Psi_{t+1},\qquad
\bar\Psi_{t+1}=\bar G(z_{t+1},e(a^{\mathcal D}_{t+1}),m).
\]

V 的 target 为：

\[
y^V_{t,\mathrm{direct}}=D_t,
\]
\[
y^V_{t,\mathrm{TD}}=\ell_t+(1-d_{t+1})\bar V(\bar\Psi_{t+1},m).
\]

d_(t+1)=1 表示下一节点已到本次采样目标；此时剩余量为零，不读取目标之后的数据动作。普通窗口结束不是目标到达；真实终止但目标未达，也不能被标为成功到达。当前 finite-path sampler 只构造已记录的有效同轨迹连接片段。

G/V 都**没有 γ**。不把旧 C–G3 的 γ=0.95、RP1 的 γ=0.98 或折扣后的累计特征带入本方法。G 的直接 target 是未来特征的**和**，不是平均值。

V 的直接标签是 D_t，不是 W_t。它只在高效率片段上把净变化作为路径代价近似；TD 再传播到其他片段。因此 V 应称为路径代价估计，不能称为每条数据轨迹累计变化的精确值，也不能把所有候选动作都训练成同一个端点距离。

### 损失与梯度归属

训练 V 时使用 stopped G 特征：

\[
\hat v_t=V_\psi(\operatorname{sg}[\Psi_t],m).
\]
\[
L_G=\mathbb E_{\mathrm{valid}}\left[
 b_t\|\Psi_t-\operatorname{sg}[y^G_{t,\mathrm{direct}}]\|_2^2+
 (1-b_t)\|\Psi_t-\operatorname{sg}[y^G_{t,\mathrm{TD}}]\|_2^2
\right],
\]
\[
L_V=\mathbb E_{\mathrm{valid}}\left[
 b_t(\hat v_t-\operatorname{sg}[y^V_{t,\mathrm{direct}}])^2+
 (1-b_t)(\hat v_t-\operatorname{sg}[y^V_{t,\mathrm{TD}}])^2
\right].
\]

当前 API 对 G 的向量坐标求平方和，再对全部有效样本取平均；V 为 scalar MSE。direct/TD 日志分别记录对总体均值的贡献，两者相加等于总损失，不能把两个条件分支均值直接相加后仍称相同目标。

L_G 只更新在线 G，L_V 只更新在线 V。目标网络独立且停止梯度，按

\[
\bar\theta\leftarrow(1-\alpha)\bar\theta+\alpha\theta
\]

更新，V 同理。已有 C–G3 的 EMA decay=.995 对应在线注入 rate=.005；配置必须明确使用哪一种记法。底座 E/e 保持冻结。

## EffAction 怎样训练和测试

EffAction 先训练上述 G/V，评测时固定网络参数，以当前观测和目标定义候选动作成本：

\[
J(a;z_t,z_g)=
-\frac{\|z_g-z_t\|_2}{V_\psi(G_\theta(z_t,e(a),m),m)+\epsilon}.
\]

CEM 在每一轮对**每个候选动作**计算 J，选择最低成本的 elites，并更新分布。固定非退化起终点时，降低 J 与降低正路径代价 V 的排序方向一致；正式配置仍要固定实际使用的评分表达式与数值保护。

当前单块数学接口不需要 F rollout，直接由 G/V 给出对动作及数据 continuation 的长期评价。它不是 RP1 的 terminal-state critic。CEM 搜索预算可参照项目现有的 300 candidates、30 iterations、30 elites、初始 variance=1；搜索范围、动作跨度与实际执行协议仍须由最终配置固定。

测试流程是：编码真实观测与目标 → 初始化候选分布 → CEM 反复评分并更新 → 选择最终动作对象 → 按所选执行协议送入环境 → 在规定时点读取新真实观测。内部候选搜索不调用真实环境取得反馈。

若最终采用单块模式，执行的对象只有 5 primitive steps。若要求执行完整 25 步计划，不能只让第一个块影响 G/V 成本而把另外四块直接执行；完整计划版本需先确定方法定义。

## EffActionPlan 怎样生成动作

在一轮真实环境决策内，z_t、z_g、m 固定，候选动作按 r=0,…,K−1 改善：

\[
a^{(0)}=\Pi_{\mathcal A}(a_{\mathrm{ref}}),\qquad
J^{(r)}=J(a^{(r)};z_t,z_g),\qquad
g^{(r)}=\nabla_{a^{(r)}}J^{(r)},
\]
\[
\Delta a^{(r)}=
P_\omega\left(z_t,a_{\mathrm{ref}},a^{(r)},z_g,
\operatorname{sg}[J^{(r)}],\operatorname{sg}[g^{(r)}]\right),
\]
\[
a^{(r+1)}=\Pi_{\mathcal A}\left(a^{(r)}+\Delta a^{(r)}\right).
\]

同一个 P 在全部 K 轮共享参数。投影作用于更新后的动作，不是只限制增量。每轮更新后重新计算 J 与动作梯度。最终执行 a^(K)，内部修正不推进环境时间。

单块时 P 的输入维度为 192+25+25+192+1+25=460，输出 25。a_ref 是不依赖正确标签的初始化；真实数据动作只用于监督，不能放入 a_ref 或 P 的其他输入泄漏答案。零初始化与 normalized [−3.5,3.5] 范围已锁定。normalized zero 通常对应数据动作均值，不等于物理零动作；测 no-op 基线时不能混用二者。对 a_phys=μ+σ·a_norm，物理合法区间对应逐维 normalized bounds=(a_phys_bounds−μ)/σ；逐坐标实测 max 为 [3.4928, 2.5479, 1.5590, 2.5469, 3.1814] 重复五次，统一标量 3.5 保证零截断，但对部分坐标偏松，正式报告须记录实际触边比例。

### 第一阶段 动作重建

展开 K 轮，以同一起点和采样目标对应的数据动作 a_t^D 监督最终输出：

\[
L_{\mathrm{traj}}=
\mathbb E_{\mathrm{valid}}\|a^{(K)}-\operatorname{sg}[a_t^{\mathcal D}]\|_2^2.
\]

第一阶段仅优化 L_traj。当前 API 按动作坐标求平方和，再按有效样本平均；不会把每轮中间动作都当成额外监督项。标签形状必须与最终输出完全一致。

### 第二阶段 重建与预计效率

从第一阶段的 P 权重继续训练：

\[
L_{\mathrm{eff}}=\mathbb E_{\mathrm{valid}}[J(a^{(K)};z_t,z_g)],\qquad
L_P=\lambda_{\mathrm{traj}}L_{\mathrm{traj}}+
\lambda_{\mathrm{eff}}L_{\mathrm{eff}}.
\]

λ_traj=1、λ_eff=.1、ε=1e-6 已锁定。loss 的坐标 reduction 与系数共同决定梯度尺度；不能在改成逐坐标均值后还声称维持同样权重。

两个阶段都只更新 P，E/e/G/V 参数固定。但梯度必须沿

\[
a^{(K)}\longrightarrow e\longrightarrow G\longrightarrow V\longrightarrow J
\]

回到 P。冻结参数不等于把整个评价器放在 no_grad 内。送入 P 的 J/g 是停止梯度的反馈副本；最终 J 的计算图与 action residual chain 必须保留，不能每轮把当前动作都 detach 后只训练最后一轮。

推理时不更新 P 参数，仍需计算输入动作梯度 g。当前 `iterate_eff_action_plan(track_grad=False)` 会在内部保留计算 g 所需的导数，并返回停止梯度的结果；它不是完全不求任何梯度的前向策略。

EffActionPlan 不执行 CEM，也不以 CEM expert 为监督。它没有方案一的状态中点生成或 L_dyn，未加入 RP1 的平均中间 value 正则、critic co-training 或 Dyna。需要这些扩展时应作为另外的方法定义。

## 数据采样 验证和正式评测

### episode 留出与当前采样实现

按 RP1 已披露的数据划分，学习 G/V/P 使用 episodes [0,8000)，validation/tuning 与最终评测任务从 [8000,10000) 抽取，并使用相互独立、预先保存的 pair draws。该留出指新学习模块的数据使用，不证明历史预训练 LeWM 或其固定归一化统计也未接触这些 episode；必须记录底座原有训练来源。

**2026-09-12 更新**：正式评测不再从 [8000,10000) 自抽，改用 `historical_cg3` 复现 C–G3 的固定 pair 清单，使两个方法做同一批题。理由与代价见「2026-09-12 评测 pair 对齐决定」一节。训练数据的 0–7999 划分不变。

当前 `EffActionSamplingConfig` 接受明确的 episode 范围、goal chunk 上下限、五步网格 phase、η 阈值、ε、近零距离阈值和可选 direct_max_chunks。当前本地约定为：

1. 在配置的 goal chunk offsets 中均匀抽 offset；
2. 在该 offset 的全部有效起点行中均匀抽 anchor；
3. goal=anchor+5×offset，且与 anchor 属于同一 episode；
4. 验证整段 latent 与动作有限，计算真实 η、future-feature sum 和共享分支 b；
5. 到目标后的 next action 不读取，不把缺失动作填成可用于 bootstrap 的标签。

它不是先均匀抽 episode；对不同长度 episode，两种规则会产生不同分布。网格 phase、offset 范围和有效性筛选都需要进入 manifest。当前代码仅支持显式排除退化的零净位移片段；具体 tolerance 尚须随配置固定。

**这是按 RP1 公布约束写出的本地采样规则，不是官方代码逐样本复现。** 跨 episode 目标当前会被拒绝；用户尚未决定是否增加这一分支。跨轨迹 pair 没有已知真实连接路径，不能直接计算 η、future-feature sum 或 Planner 数据动作标签，不能仅添加 0.3 概率便视为定义完整。

### 训练和验证应记录什么

各阶段记录总 updates、有效样本数、η 分布、直接分支比例、goal offsets、被过滤样本数及原因。G/V 分别记录 direct/TD loss，检查目标尺度与非有限值；V 还应检查其对不同候选动作的区分。P 记录重建误差、最终与初始 J、动作修正量、投影触边比例，以及 stage 1/2 checkpoint。

验证 pair 清单应固定并保存。调参、选 checkpoint 和正式报告使用不同 draws；不能根据最终测试成功率挑选 η、K、loss weights、action bounds 或 checkpoint，再报告同一测试集作为独立结果。RP1 报道的调参 seeds 50/51 与报告 seeds 42/43/44 是其协议事实，本任务最终 seeds 尚需与待确认实验配置一致。

### 正式控制评测

每次正式运行必须明确并保存：

| 字段 | 必须说明的含义 |
|---|---|
| selected pairs | episode id、anchor raw step、goal raw step、pair 文件 hash |
| goal offset | O25/O50/O100 等原始数据步数 |
| action object | 一块 25D，或经确认的整计划对象 |
| planning horizon | 候选计划含多少动作块 |
| execution interval | 多少 primitive steps 后读取新真实观测并重规划 |
| episode budget | 最多执行多少 primitive steps；既有 Cube O25/O50/O100 为 50/100/200 |
| solver budget | EffAction 的 CEM 候选/轮数；EffActionPlan 的 K 与模型调用次数 |
| normalization | 动作均值/标准差、数值 bounds、到物理动作域的变换 |
| success | swm/OGBCube-v0 的 cube center 4 cm 准则、terminate_at_goal 设置 |
| provenance | 数据、冻结底座、G/V/P checkpoint hashes、完整配置、代码版本、依赖、seed、renderer/GPU |

单块与整计划执行的反馈频率不同，应分别命名，不能把 H1/RH1 与 H5/RH5 的分数当成同协议结果。相同计算预算的对照要报告实际网络/rollout 调用量和墙钟时间，不能把 K=8 自动等同 CEM 的 8 个候选。

### 与历史 C–G3 结果的关系

历史 C–G3 使用过 90/10 sequence-clip split，部分 baseline 的评测 pair 从全部 10,000 episodes 抽取；这些都与本任务的 RP1 episode 留出不同。旧表可以作为历史背景，但不能直接跨 pair 集合相减来计算提升。

### 2026-09-12 评测 pair 对齐决定

**问题。** EffAction 原使用 `rp1_heldout`，50 对全部来自 episode 8000–9999；C–G3 使用固定清单，三档 offset 各一份 `selection_sha256`（O25 `56546fe8…`、O50 `e46ea81c…`、O100 `8a87815e…`）。实测 C–G3 O50 seed 42 的 50 对 episode 范围为 638–9755，其中只有 10/50 落在 8000–9999，约 40/50 来自训练区 0–7999。两个方法此前做的是几乎不相交的两批题，成功率不能直接相减。

**决定（用户 2026-09-12）。** EffAction 与 EffActionPlan 的 `selection.protocol` 由 `rp1_heldout` 改为 `historical_cg3`，与 C–G3 使用完全相同的 pair 清单。

**落地。**
- 两份评测配置改为 `protocol: historical_cg3`，`protocol_status` 仍为 `user_locked`。
- `historical_cg3` 在实现中锁死 50 对 / seed 42，因此 pair 列表固定；原本的 seed 42/43/44 改为只重置规划器 RNG（评测 CLI 新增 `--planning-seed`），pair 集合不变。
- 复现性已离线验证：以 10000×201 的 episode 长度生成，O25 / O50 / O100 三档的 pair 集合 sha256 与三个锁定值逐一相符。运行时仍保留 `select_eff_action_episodes` 的精确 hash 校验，不符即抛 `RuntimeError`。

**代价与必须随结果一起报告的 caveat。**
1. 这批清单从全部 10,000 episodes 抽取，约 40/50 的题来自 G/V/P 的训练区 0–7999，因此**绝对成功率偏乐观**，应视为训练区成绩，不可当作泛化成绩引用。
2. 只有「同一批 pair 上 EffAction 与 C–G3 的差值」是受控比较；跨 pair 集合的相减仍然无效。
3. 冻结底座 F（LeWM）自身是否见过这些 episode 需单独记录，本对齐只保证两个方法题目相同。

**搜索深度差异保持不变。** C–G3 `horizon: 5`：一次决策搜索 5 个动作块（125 维），用 F 做 25 步 rollout 评分；EffAction `horizon: 1`：只搜索 1 个块（25 维），由 G/V 直接评分。两者 `receding_horizon` 均为 1，重规划节奏一致，都是每执行 5 个环境步重规划一次，此处不存在 5 倍的重规划频率差。`warm_start` 对 EffAction 无效（`horizon == receding_horizon`，没有剩余 plan 尾巴），对 C–G3 有效；这是搜索深度差异的必然结果，不是可单独修复的配置错误。计算量差异由 EfficiencyMetric 记录。

历史 O50 的部分配置每个 5-step block 重规划，而后续 full-plan 协议执行完整 25 步后才反馈。旧结果也包含 OSMesa 与 EGL，且少量归档缺少 renderer 字段；跨后端差异不能直接归因于方法。正式比较应固定 renderer、设备及执行规则，保留 per-episode 结果；必要时以同运行环境下的新 baseline 作为配对参照。

RP1 Cube 的 hard 指标是 `max(0, (s−f)/(100−f)×100)`，s 为原始 success 百分比、f 为同协议 measured no-op floor。它不是在 no-op 失败子集上的条件成功率。若本任务报告该量，必须在本任务自己的 pair/执行协议上测 f，不直接搬用论文的 floor。

## RP1 哪些可核实 哪些仍未公开

[RP1 论文](https://arxiv.org/html/2608.18669v1)可核实：B.1 的 critic future-goal offsets 在可用 horizon 上平衡；C.1 的 8000/2000 episode 留出；Table 6 actor 区的 replay probability=.5（Cube）、max-delta=10 chunks、cross-episode=.3；C.1 又明确 co-trained critic 使用 cross-episode=.3。

论文没有公开 balanced sampler 的 bins/精确概率代码，也没有解释 actor replay=.5 的回放内容和过程。offline critic 初始化没有单独给 cross-episode 的数值；评测使用同轨迹 h-step goals，不能把训练 cross-episode 比例搬入测试任务。

本次此前已核查论文、[Pantheon 官方发布](https://pantheon.inc/research/introducing-rp1)和公开 artifact 入口，未找到可核验的官方实现。因此本文只使用“参照 RP1 已披露划分与采样约束”，不使用“完整/精确复现 RP1”的结论。C–G3 的 `goal_probability=.5` 另指真实目标与随机方向任务混合，不能当作 RP1 replay 或 same/cross 概率。

## 当前代码接口与运行入口

以下文件已经存在；源码路径均相对于本仓库根目录。实现存在不代表 GPU 实验已经执行。

| 文件 | API 或职责 |
|---|---|
| `src/tdwm/methods/eff_action.py` | `EffActionSuccessor`、`EffActionValue`、`build_eff_action_loss`、`eff_action_cost`、`ema_update_eff_action`、`encode_eff_action_blocks` |
| `src/tdwm/methods/eff_action_plan.py` | `EffActionPlanner`、`iterate_eff_action_plan`、`build_eff_action_plan_loss`、`project_eff_action` |
| `src/tdwm/training/eff_action_data.py` | `EffActionSamplingConfig`、`EffActionEpisodeData`、`EFF_ACTION_LOSS_BATCH_KEYS` |
| `src/tdwm/training/eff_action_runtime.py` | `EffActionTrainer`；G/V、P stage 1/2 更新、固定验证、优化器与随机状态恢复 |
| `src/tdwm/training/eff_action.py` | `train_eff_action`、`load_eff_action_protocol`、`load_eff_action_store`、`load_frozen_eff_action_backbone`、`load_eff_action_normalization`、`load_eff_action_deployment`；来源绑定、运行归档与恢复 |
| `src/tdwm/adapters/eff_action.py` | `EffActionCostModel`、`EffActionPlanSolver`、`make_eff_action_policy`；接入 SWM CEM 与 learned updater |
| `src/tdwm/evaluation/eff_action.py` | `select_eff_action_episodes`、`evaluate_eff_action`；pair、数据来源和控制评测结果记录 |
| `scripts/train_eff_action.py` | 三阶段训练 CLI，可在 G/V 或 P 的阶段边界停止 |
| `scripts/evaluate_eff_action.py` | EffAction / EffActionPlan 评测 CLI |
| `configs/experiment/eff_action_cube_train.yaml` | G/V 与 P 两阶段的共同训练配置，`protocol_status: user_locked` |
| `configs/experiment/eff_action_cube_eval.yaml` | EffAction CEM 评测配置，`protocol_status: user_locked` |
| `configs/experiment/eff_action_plan_cube_eval.yaml` | EffActionPlan 评测配置，`protocol_status: user_locked` |

架构宽度、非负输出方式、ε、K、bounds 等由配置显式提供。当前接口只支持冻结 LeWM 的单个 25D block，不能通过改一个维度参数就启用未经定义的完整计划。三份配置已按用户裁定锁定；即便如此，正式实验仍要求真实 GPU、足量磁盘和正式训练产生的 checkpoint，不能仅因配置已锁定就声称结果。

主实现任务已综合运行并通过 86 项方法、数据、训练、评测与 runner CPU 检查，Ruff 检查通过。本地真实 Cube 环境因缺少 MuJoCo 未完成验证；本次服务器会话中 OSMesa GL 不可用，也没有可用 GPU。真实环境评测与 GPU 训练尚未执行，本文没有已验证的成功率或性能提升。

### 环境与输入文件

运行前应正确安装**本仓库**，或在本仓库根目录使用 `PYTHONPATH=src` 隔离其他任务的 editable 安装；以下命令采用后者。Python 环境还须具备项目规定的依赖，真实 Cube 评测另需可工作的 MuJoCo 与指定 renderer。`/path/to/...` 是待替换的外部 artifact 路径，以下是使用说明，并非已执行记录。

`--pretrained` 必须指向 SWM 公开 export 的 `.../checkpoints/<name>/weights.pt` 布局，并保留 loader 所需的导出目录结构；不能用随意改名的单个 `.pt` 文件替代。冻结底座、latent store、数据来源和归一化文件均按锁定 SHA256 核对。

训练与评测的 `--action-normalization` 应指定生成缓存时的原始完整 `column_normalization.json`。训练会原样复制到 `run_dir/column_normalization.json`，评测可使用这份副本。另写出的 `action_normalization.json` 只含 action 统计子集，不能冒充完整 hash 文件传回 CLI，也不能在留出数据上重新拟合后替代原文件。

冻结 latent 缓存由固定底座以 **bfloat16 提取、float32 存储**；文件存储类型不代表提取精度。当前训练 draft 使用 bfloat16 混合精度，正式评测接口要求 **fp32 推理**。这三个环节应分别记录，不能把 float32 缓存描述为 fp32 提取结果。

### 训练与恢复

当前 draft 可显式执行 CPU smoke。该模式缩短阶段预算与 batch，只检查运行链路，不产生正式实验 checkpoint：

```bash
PYTHONPATH=src python scripts/train_eff_action.py \
  --config configs/experiment/eff_action_cube_train.yaml \
  --latent-store /path/to/frozen_latent_store \
  --pretrained /path/to/export/checkpoints/EXPORT_NAME/weights.pt \
  --action-normalization /path/to/original/column_normalization.json \
  --output-dir /path/to/new_training_smoke \
  --device cpu --smoke --stop-after stage2
```

将示例中的 `EXPORT_NAME` 替换为真实导出名称后再执行。`--stop-after` 支持 `gv`、`stage1`、`stage2`：EffAction 只需完成 `gv`；EffActionPlan 默认依次完成三个阶段，stage 2 延续 stage 1 的 P。正式训练使用 `protocol_status: user_locked` 的确认配置，移除 `--smoke`，并在可用的 GPU 实例上选择 `--device cuda`。

恢复时保留相同配置、输入文件、运行模式与输出目录，在相同命令中加 `--resume /path/to/run_dir/latest.pt`。smoke 的恢复仍需 `--smoke`。运行器核对配置与来源，恢复阶段计数、优化器和随机状态；恢复日志保留备份并退回 checkpoint 对应边界。`--stop-after` 不能早于恢复 checkpoint 所在阶段。正常阶段完成文件为 `gv_complete.pt`、`stage1_complete.pt`、`stage2_complete.pt`。

### 控制评测

在满足真实环境依赖后，可用下列 EffAction smoke 检查评测链路；这不是已完成的环境测试：

```bash
PYTHONPATH=src python scripts/evaluate_eff_action.py \
  --config configs/experiment/eff_action_cube_eval.yaml \
  --checkpoint /path/to/training_run/gv_complete.pt \
  --pretrained /path/to/export/checkpoints/EXPORT_NAME/weights.pt \
  --dataset /path/to/cube_dataset.lance \
  --action-normalization /path/to/training_run/column_normalization.json \
  --output-dir /path/to/new_evaluation_smoke \
  --device cpu --goal-offset 25 --seed 42 --smoke
```

EffActionPlan 改用 `configs/experiment/eff_action_plan_cube_eval.yaml`，并提供含已训练 P 的 checkpoint；完整两阶段方法使用 `stage2_complete.pt`。stage 1 checkpoint 的结果只能对应动作重建阶段，须保留其 objective 标签，不能当作已完成 stage 2 的结果。

`--goal-offset` 支持 `25`、`50`、`100`，对应原始数据步数；非 smoke 模式的总执行预算随此参数设为 offset 的两倍。`--seed` 支持 `42`、`43`、`44`，同时设置 pair draw 与 planner seed。`--allow-intermediate-checkpoint` 允许显式评测未完成当前阶段预算的已训练 checkpoint，不绕过方法、来源和正式模式校验。`--smoke` 将任务缩为一个 pair、五个执行步，并缩小 EffAction 的 CEM 搜索预算，其输出不能作为正式对照。

正式评测必须使用 `protocol_status: user_locked` 的确认配置、移除 `--smoke`，并提供由正式训练生成的 checkpoint；smoke checkpoint 会被拒绝。CLI 还核对 ε、EffActionPlan 的 K/初始化/bounds、完整归一化文件与底座来源。评测写出 pair 清单、协议 manifest 和逐 episode 结果；方法不能仅凭 checkpoint 文件名判断，成功率也只能来自实际完成的环境运行。

## 实现核查中不能混淆的事项

- G/V 同一个 b，G direct 用特征和、V direct 用净变化；不混入方案一的 V 标签、效率 exp-weight，或旧 C–G3 的辅助 loss。
- 不引入 γ；不把 sample window 结束当作到达目标；不读取目标之后的动作补 bootstrap。
- V 训练输入 detach，不让 L_V 更新 G；P 训练保留 e/G/V 的输入梯度，不能用被冻结的参数为由截断动作导数。
- P 六项输入完整，a_ref 不泄漏数据标签；最终 loss 作用于 K 轮之后，action residual chain 连续。
- V 非负、ε 显式；近零位移、无效路径、浮点异常与动作触边必须可观测，不能吞掉后继续汇总成功率。
- action width、horizon、feedback interval、goal offset 分别断言；只执行被已定义目标完整评价过的动作对象。
- CEM 的 G/V cost 必须在每轮候选筛选前生效，不能先按旧 cost 搜完，再给选中结果追加一个数值当作新方法。
- 不混用旧/新 pair、renderer 或 loss 尺度；所有结果链接回实际配置与 checkpoint。已锁定的配置不能转换成“已完成训练”或“优于 baseline”的陈述。

正式训练应在可用的 GPU 实例与足量磁盘上开始，运行接口与测试协议已在此前核对完成。之后由真实日志、checkpoint、逐 episode 输出与配对比较填入结果；本文不预先给出实验结论。
