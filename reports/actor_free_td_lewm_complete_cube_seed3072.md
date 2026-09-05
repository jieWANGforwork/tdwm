# Results TD — 全部 Actor-Free TD-LeWM 实验总账（Cube seed 3072）

本报告保留 **477 个已核验正式 O50 基础单元**及其原分析；唯一主矩阵另预留 V1-C2/C3 的 8 个严格 endpoint 单元。基础方法固定 E10，C2 固定最终 E10，C3 固定最终 E12。每格均为同一组 50 个 start-goal pair；训练 seed=3072，planning seed=42。模型均不训练 Actor。

## 一句话结论

- **按原先固定的主评分列 F+G，描述性领先配置为 V1-G3: 27/50 (54%)。**
- **所有固定 E10 单格的最高结果为 V1-C + F + first-Q: 28/50 (56%)。**
- **按四版本、24 个训练配置的固定 E10 均值，描述性领先测试评分为 F + first-Q（44.8%）。** 单 seed 下不把它表述为统计稳健最优。
- **若把五种评分等权平均，描述性领先训练配置为 V1-F, V1-G3（并列 48.8%）。**
- **按六个训练方法 × 五种评分的版本均值，V1 action encoder 最高（47.3%）。**
- V1→V2 联合微调后，F-only 均值由 46.0% 变为 26.0%（-20.0 pp），F+G 由 47.7% 变为 27.3%（-20.3 pp）；这首先提示 world-model/control representation 变化，而不只是 G 的读出形式。

## 完整全账伴随文件

| 文件 | 保留内容 | 路径 | SHA-256 |
| --- | ---: | ---: | ---: |
| CSV scalar ledger | 477 个 O50 单元 | `reports/artifacts/actor_free_td_lewm_complete_cube_seed3072/all_o50_results.csv` | `0e5b541bdb11cf6d647fc1e679499a02c3aa430d64e37c8819d02c44e1dcb900` |
| JSON reconciliation ledger | 477 格 × 50 outcomes = 23,850 | `reports/artifacts/actor_free_td_lewm_complete_cube_seed3072/reconciliation_ledger.json` | `2f1985d5703e795f48bd8b850d470f76e35c998447112c86065e8bdcbbe1372b` |

原 24×5 分析固定使用 E10；全部 477 格和 23,850 个逐-pair 布尔结果仍由上述伴随文件完整保留。C2/C3 的 8 格由独立严格 endpoint ledger 接入，不改写原账。

## 结果覆盖与版本定义

| 版本/家族 | 方法数 | Checkpoint | 评分覆盖 | O50 格数 |
| --- | ---: | ---: | ---: | ---: |
| Legacy | 7 | E10 | F / G(C) / combined | 21 |
| V0 raw action | 6 | E10 | all five scores | 30 |
| V1 action encoder | 6 | E10 | all five scores | 30 |
| V2 joint fine-tune | 6 | E3-E10 | 3 original + E10 first/Mean | 156 |
| V2-EMA-SG | 6 | E3-E10 | all five scores | 240 |
| V1-C2/C3 endpoint extension | 2 | C2 E10 / C3 E12 | First-Q2 + State-V integrated into seven-column matrix | 8 |

## 方法、网络和训练 loss

旧结构消融比较 Successor/critic head 与 LeWM predictor 的连接方式：Serial Decoupled、Serial Coupled、Hybrid、Parallel Real、Goal Hybrid、Imaginary Hybrid、Direct Goal Critic Hybrid。其总目标均为 `L_LeWM + α_u L_TD`，区别在 real/predicted 支路、是否让 TD 梯度进入 LeWM、是否使用 goal projection/imaginary bootstrap/direct scalar critic。DOCX 继续保留前一版的详细结构、loss、训练曲线与 V0/V1 逐方法说明。

C–G3 家族共享同一个 TD-JEPA predictor `G`。V0 输入归一化 raw action；V1 改用冻结的 LeWM Action Encoder；V2 联合微调 LeWM/Action Encoder/G；V2-EMA-SG 进一步用 EMA world model、EMA action encoder 与 EMA G 构造完全 stop-gradient target。

基础 target 与总目标：

$$Y_t=\operatorname{sg}[\bar z_{t+1}+\gamma(1-d_t)G_{\bar\phi}(\bar z_{t+1},\bar e_{t+1},m_t)],\quad \gamma=0.95,$$

$$L_{total}=L_{pred}+0.09L_{SIGReg}+\rho(u)(L_{method}^{real}+L_{method}^{pred}).$$

其中分支 `b∈{real,pred}`；逐样本基础 TD 残差为 `l_i^b=||G_φ(s_i^b,e_i,m_i)-Y_i||²`，`m_i` 是 goal/random task vector，`ρ(u)` 是 TD warm-up 权重。C 额外训练 goal scalar residual；D/F/G1/G2/G3 只用 stop-gradient 的优势信号重加权 `l_i^b`。

| 方法 | 特殊训练信号 | 支路 loss | 作用 |
| --- | ---: | ---: | ---: |
| C | q_i^b = G_phi(s_i^b,e_i,m_i)^T m_i; q_i^Y = Y_i^T m_i | L_C^b = mean(l_i^b) + lambda_C mean_goal[(q_i^b-q_i^Y)^2], lambda_C=1 | Only C adds a trainable scalar projection residual on goal-derived tasks. |
| D | A_i = sg(Y_i^T m_i) | L_D^b = mean_i[w_i(A) l_i^b] | Detached target goal value reweights TD; tau=0.5. |
| F | A_i = sg[Y_i^T m_i - mean_j(Y_i^T m_j)] | L_F^b = mean_i[w_i(A) l_i^b] | Matched future/task is contrasted with all goal tasks in the batch. |
| G1 | A_i = sg[q_i - sum_k softmax(-d_ik/tau_n) q_ik] | L_G1^b = mean_i[w_i(A) l_i^b] | K=8 other-episode latent-neighbour actions; candidates have no TD targets. |
| G2 | A_i = sg[q_i5 - (1/5) sum_{j=1}^5 q_ij] | L_G2^b = mean_i[w_i(A) l_i^b] | Five zero-suffix action prefixes; full-minus-prefix-mean signal. |
| G3 | A_i = sg[(1/4) sum_{j=1}^4(q_i,j+1-q_ij)] | L_G3^b = mean_i[w_i(A) l_i^b] | Five prefixes; mean adjacent marginal score gain. |
| C2 (V1 only) | Frozen-F terminal goal-cost ranking over 16 candidate action sequences | L_C2=L_C+CE(p_F,p_Q); p_F=softmax(-z_cand(J_F)), p_Q=softmax(z_cand(Q_G(z0,A1,g))) | Initialize every parameter from V1-C E10, freeze LeWM/Action Encoder, and fine-tune only G so First-Q follows the planner ranking |
| C3 (V1 only) | Same-episode temporal distance in primitive-step units with an EMA State-V bootstrap | L_C3=E[omega_tau(r)Huber_1(r)], r=V_psi(z,g)-sg(y), tau=0.03; y=delta inside n_eff, otherwise c_gamma(n_eff)+gamma^n_eff V_bar(z_succ,g) | Freeze the complete V1-C parent, including both G copies; train only a nonnegative MRN State-V critic (gamma=0.98, n<=50 primitives) |

## 七种测试方法怎么测

统一约定：`z0` 是当前真实图像经部署 encoder 得到的 latent，`z_g` 是 goal 图像的 latent，`z_k^F` 是 LeWM rollout 的 imagined latent。V1/V2/V2-EMA 先用共享 Action Encoder 得到 `e_k=E_A(A_k)`；V0 直接把归一化 25D action block 输入 G。`Q_G(z,A,g)=G(z,e,w(g))^T w(g)`。CEM 始终最小化 cost。

| 统一评测字段 | 固定设置 |
| --- | ---: |
| Environment | swm/OGBCube-v0 |
| Formal pairs | The same 50 same-episode start-goal pairs; goal offset 50 |
| CEM | 300 candidates, 30 iterations, 30 elites, planning seed 42, warm start |
| Execution | Minimize candidate cost, execute only the first action block A1, then observe and replan |
| Episode success | Object-to-goal distance <= 0.04 m within 100 environment steps |
| Checkpoint | Base rows use fixed E10; C2 uses final E10; C3 uses final E12. No score-specific retraining |

七个评分列的实际计算：

| 评分列 | F/G 的实际路径 | CEM 最小化的 cost | goal/Q 使用位置 |
| --- | ---: | ---: | ---: |
| F-only | F rolls A1...A5 from real z0 and produces imagined z1^F...z5^F; G is not called. | J_F = ||z5^F - z_g||_2^2 | Uses terminal goal distance at z5; no Q and no gamma. |
| G-only | H=1. G scores the real z0 and first candidate action A1; F is not rolled out. | J_G = -Q_G(z0,A1,g) | No explicit goal distance and no gamma; minimizing -Q maximizes Q. |
| F+G tail | F rolls only A1...A4 to z4^F; G evaluates the fifth transition from z4^F with A5. | J_tail = ||z4^F - z_g||_2^2 - gamma^4 Q_G(z4^F,A5,g) | Uses z4 goal distance and the deepest imagined-state Q; gamma=0.95. |
| F + first-Q | F completes the five-step rollout; G is read only once at the real z0 with A1. | J_first = ||z5^F - z_g||_2^2 - 0.25 Q_G(z0,A1,g) | Uses terminal goal distance; the Q term is not multiplied by gamma^4. |
| Mean-Q rollout | F generates predecessors z0,z1^F,...,z4^F; G scores each aligned pair (z{k-1}^F,Ak). | J_mean = -(1/5) sum[k=1..5] Q_G(z{k-1}^F,Ak,g) | No terminal goal distance; z5 is not read by G and gamma is unused. |
| First-Q2 | F completes the five-step rollout; G is read once at real z0 with A1. Each candidate set normalizes F-cost and first-Q separately. | J_first2 = zscore_candidates(J_F) - 0.25 zscore_candidates(Q_G(z0,A1,g)) | No gamma. Population z-score statistics are recomputed inside each CEM candidate set; they never persist across iterations or episodes. |
| State-V terminal | Frozen F completes all five blocks to z5^F; only the EMA State-V critic reads (z5^F,z_g). G is not called. | J_V = V_bar(z5^F,z_g) | No terminal latent L2, G term, actor or gamma factor is added at inference; CEM minimizes predicted temporal cost-to-go. |

V2-EMA 的 EMA world model、EMA Action Encoder 和 EMA G 只构造训练 target；正式 CEM 测试仍部署 online F、online Action Encoder 和 online G。Legacy 7 方法的旧 `G/C-only` 会先由 F 构造 H-1 tail context，不等同于 C-G3 主矩阵里严格 H=1 的 `G-only`。

## Legacy 7 方法：完整 21 格

| 方法 | F-only | G/C-only | Combined |
| --- | ---: | ---: | ---: |
| Serial Decoupled | 20/50 (40%) | 15/50 (30%) | 23/50 (46%) |
| Serial Coupled | 26/50 (52%) | 14/50 (28%) | 20/50 (40%) |
| Hybrid | 25/50 (50%) | 13/50 (26%) | 27/50 (54%) |
| Parallel Real | 17/50 (34%) | 13/50 (26%) | 22/50 (44%) |
| Goal Hybrid | 25/50 (50%) | 19/50 (38%) | 19/50 (38%) |
| Imaginary Hybrid | 24/50 (48%) | 13/50 (26%) | 23/50 (46%) |
| Direct Goal Critic Hybrid | 22/50 (44%) | 16/50 (32%) | 19/50 (38%) |

## 26 个方法 × 7 种评分的唯一主结果矩阵

横向读每一行，可以同时看到训练 loss，并比较同一个训练方法已有的评分；纵向读每一列时，以版本为边界比较该版本内所有可用方法。Markdown 中 **粗体**是行最佳，`◆` 是同版本列最佳；并列全部标记。缺失格显示 `—`，不参加任何最大值；DOCX 使用黄底表示行最佳、蓝底表示同版本列最佳、青色底表示两者同时成立。

Loss 列采用紧凑记号：`l_i` 是逐样本 successor TD 残差，`qY=Y^T m`；D–G3 的 `w_i(·)` 是由括号内 stop-gradient 信号形成的归一化样本权重。V0/V1 只有 real 分支；V2/V2-EMA 的总目标为 `L_pred+0.09L_SIGReg+ρ(L_method^real+L_method^pred)`。精确信号、goal 子集和权重定义见前面的“方法、网络和训练 loss”表。

| 版本 | 训练方法 | 训练 loss | F-only | G-only | F+G tail | First-Q | Mean-Q | First-Q2 | State-V |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| V0 | C | L_C=mean(l)+mean_goal(q-qY)^2 | ◆ 23/50 (46%) | 18/50 (36%) | 24/50 (48%) | ◆ **26/50 (52%)** | 22/50 (44%) | — | — |
| V0 | D | L_D=mean_i w_i[sg(qY)]l_i | ◆ 23/50 (46%) | ◆ 20/50 (40%) | 21/50 (42%) | 24/50 (48%) | ◆ **26/50 (52%)** | — | — |
| V0 | F | L_F=mean_i w_i(A_goal)l_i | ◆ 23/50 (46%) | 19/50 (38%) | 20/50 (40%) | **24/50 (48%)** | **24/50 (48%)** | — | — |
| V0 | G1 | L_G1=mean_i w_i(A_neighbor)l_i | ◆ 23/50 (46%) | 16/50 (32%) | ◆ **25/50 (50%)** | 20/50 (40%) | 20/50 (40%) | — | — |
| V0 | G2 | L_G2=mean_i w_i(A_prefix-mean)l_i | ◆ 23/50 (46%) | 16/50 (32%) | ◆ **25/50 (50%)** | 23/50 (46%) | 23/50 (46%) | — | — |
| V0 | G3 | L_G3=mean_i w_i(A_prefix-gain)l_i | ◆ 23/50 (46%) | 18/50 (36%) | 23/50 (46%) | **24/50 (48%)** | **24/50 (48%)** | — | — |
| V1 | C | L_C=mean(l)+mean_goal(q-qY)^2 | ◆ 23/50 (46%) | 18/50 (36%) | 22/50 (44%) | ◆ **28/50 (56%)** | 21/50 (42%) | ◆ 26/50 (52%) | — |
| V1 | C2 | L_C2=L_C+CE(p_F,p_Qfirst) | ◆ 23/50 (46%) | 18/50 (36%) | 23/50 (46%) | **26/50 (52%)** | 22/50 (44%) | ◆ **26/50 (52%)** | — |
| V1 | C3 | L_C3=mean_i omega_tau(r_i)Huber_1(r_i) | — | — | — | — | — | — | ◆ **26/50 (52%)** |
| V1 | D | L_D=mean_i w_i[sg(qY)]l_i | ◆ 23/50 (46%) | 22/50 (44%) | 21/50 (42%) | 25/50 (50%) | **26/50 (52%)** | — | — |
| V1 | F | L_F=mean_i w_i(A_goal)l_i | ◆ 23/50 (46%) | ◆ 23/50 (46%) | 24/50 (48%) | **26/50 (52%)** | **26/50 (52%)** | — | — |
| V1 | G1 | L_G1=mean_i w_i(A_neighbor)l_i | ◆ 23/50 (46%) | 21/50 (42%) | 24/50 (48%) | **26/50 (52%)** | 25/50 (50%) | — | — |
| V1 | G2 | L_G2=mean_i w_i(A_prefix-mean)l_i | ◆ 23/50 (46%) | 21/50 (42%) | **25/50 (50%)** | **25/50 (50%)** | 24/50 (48%) | — | — |
| V1 | G3 | L_G3=mean_i w_i(A_prefix-gain)l_i | ◆ 23/50 (46%) | 19/50 (38%) | ◆ **27/50 (54%)** | 26/50 (52%) | ◆ **27/50 (54%)** | — | — |
| V2 | C | L_C=mean(l)+mean_goal(q-qY)^2 | 13/50 (26%) | ◆ **20/50 (40%)** | ◆ 16/50 (32%) | 19/50 (38%) | 10/50 (20%) | — | — |
| V2 | D | L_D=mean_i w_i[sg(qY)]l_i | ◆ 16/50 (32%) | 18/50 (36%) | ◆ 16/50 (32%) | ◆ 21/50 (42%) | ◆ **24/50 (48%)** | — | — |
| V2 | F | L_F=mean_i w_i(A_goal)l_i | 15/50 (30%) | 19/50 (38%) | 13/50 (26%) | ◆ 21/50 (42%) | **22/50 (44%)** | — | — |
| V2 | G1 | L_G1=mean_i w_i(A_neighbor)l_i | 12/50 (24%) | 17/50 (34%) | 13/50 (26%) | **19/50 (38%)** | 17/50 (34%) | — | — |
| V2 | G2 | L_G2=mean_i w_i(A_prefix-mean)l_i | 12/50 (24%) | 18/50 (36%) | 13/50 (26%) | **20/50 (40%)** | 13/50 (26%) | — | — |
| V2 | G3 | L_G3=mean_i w_i(A_prefix-gain)l_i | 10/50 (20%) | 14/50 (28%) | 11/50 (22%) | **19/50 (38%)** | 17/50 (34%) | — | — |
| V2-EMA | C | L_C=mean(l)+mean_goal(q-qY)^2 | 12/50 (24%) | 18/50 (36%) | 12/50 (24%) | **20/50 (40%)** | 14/50 (28%) | — | — |
| V2-EMA | D | L_D=mean_i w_i[sg(qY)]l_i | ◆ 15/50 (30%) | 19/50 (38%) | 16/50 (32%) | ◆ **22/50 (44%)** | 19/50 (38%) | — | — |
| V2-EMA | F | L_F=mean_i w_i(A_goal)l_i | ◆ 15/50 (30%) | 19/50 (38%) | ◆ 17/50 (34%) | ◆ 22/50 (44%) | ◆ **23/50 (46%)** | — | — |
| V2-EMA | G1 | L_G1=mean_i w_i(A_neighbor)l_i | ◆ 15/50 (30%) | 15/50 (30%) | 11/50 (22%) | 19/50 (38%) | **21/50 (42%)** | — | — |
| V2-EMA | G2 | L_G2=mean_i w_i(A_prefix-mean)l_i | 12/50 (24%) | ◆ **20/50 (40%)** | 16/50 (32%) | 18/50 (36%) | 17/50 (34%) | — | — |
| V2-EMA | G3 | L_G3=mean_i w_i(A_prefix-gain)l_i | 12/50 (24%) | 17/50 (34%) | 13/50 (26%) | **21/50 (42%)** | 17/50 (34%) | — | — |

### 每个版本内部的逐列赢家

| 版本 | F-only | G-only | F+G tail | First-Q | Mean-Q | First-Q2 | State-V |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| V0 | C/D/F/G1/G2/G3 23/50 | D 20/50 | G1/G2 25/50 | C 26/50 | D 26/50 | — | — |
| V1 | C/C2/D/F/G1/G2/G3 23/50 | F 23/50 | G3 27/50 | C 28/50 | G3 27/50 | C/C2 26/50 | C3 26/50 |
| V2 | D 16/50 | C 20/50 | C/D 16/50 | D/F 21/50 | D 24/50 | — | — |
| V2-EMA | D/F/G1 15/50 | G2 20/50 | F 17/50 | D/F 22/50 | F 23/50 | — | — |

**跨四版本的全局逐列赢家（只用于补充分析，不对应 DOCX 蓝框）：** F-only = 12 tied: V0-C, V0-D, V0-F, …: 23/50 (46%); G-only = V1-F: 23/50 (46%); F+G tail = V1-G3: 27/50 (54%); F + first-Q = V1-C: 28/50 (56%); Mean-Q rollout = V1-G3: 27/50 (54%)。

### V1-C2/C3 endpoint 证据

| 共享评分 | V1-C parent | V1-C2 | C2-parent |
| --- | ---: | ---: | ---: |
| F-only | 23/50 (46%) | 23/50 (46%) | +0/50 (+0 pp) |
| G-only | 18/50 (36%) | 18/50 (36%) | +0/50 (+0 pp) |
| F+G tail | 22/50 (44%) | 23/50 (46%) | +1/50 (+2 pp) |
| F + first-Q | 28/50 (56%) | 26/50 (52%) | -2/50 (-4 pp) |
| Mean-Q rollout | 21/50 (42%) | 22/50 (44%) | +1/50 (+2 pp) |
| First-Q2 | 26/50 (52%) | 26/50 (52%) | +0/50 (+0 pp) |

C2 在 6 个可比评分上为 2 升 / 3 平 / 1 降，平均变化 +0.0 pp。C3 的独立 State-V endpoint 为 26/50 (52%)。这些结论只在严格 8-cell ledger 到齐后由实际 outcome 生成。

C3 的同一组 50 个 pair 早期诊断为 E3 28/50 (56%)，最终 endpoint 为 E12 26/50 (52%)。配对列联为：两者均成功 24、仅 E3 成功 4、仅 E12 成功 2、两者均失败 20；exact McNemar 双侧 p=0.6875。E3 只作为诊断，不进入 8-cell 主表或 endpoint 计数。

#### Endpoint 专项解释与下一步

- **C2 的结果是混合的，而不是稳定的 ranking 增益。** 六种共享评分合计 2 升 / 3 平 / 1 降，平均变化 +0.0 pp；其中 First-Q 为 -4 pp，First-Q2 为 +0 pp。即便个别格上升，单 seed 下也没有跨读出一致、可称为稳健的排序改善。
- **C3 的最终 State-V 是 26/50。** 相对同一个 V1-C parent 的 F-only 为 +6 pp，相对 V1-C 的 First-Q 为 -4 pp；它说明独立时间价值读出有信号，但尚未稳定超过 parent 的最佳 first-action readout。
- E3 与 E12 的同-pair exact McNemar p=0.6875，没有达到常用 0.05 阈值。因此不能在看过正式 O50 后把 E3 当成新的正式 endpoint；下一轮应在独立 dev pairs 上选择 epoch，或事先登记 early-stop 规则。
- C3 训练主要读取真实 encoder latent，推理却在 `F^5` imagined terminal latent 上读 State-V。下一轮应把 stop-gradient 的 F-imagined states 按受控比例混入 State-V 训练，直接缩小这一 terminal-state OOD 间隙。
- 以上比较均为一个 training seed 和同一组 50 pair 的描述性消融，不构成多 seed 总体最优或因果证明。

## First-Q 权重扫描

本轮没有重新训练模型，只在固定 checkpoint 上改变推理评分的权重 `alpha`。C3 两行都使用同一个 V1-C3 E12 checkpoint，训练 loss 仍为 `L_C3`；原 First-Q 使用同一个 V1-C E10 checkpoint，训练 loss 仍为 `L_C`。三种 CEM cost 分别为：

$$J_{C3\text{-}raw}=\bar V_\psi(F^5(z_0,A_{1:5}),z_g)-\alpha Q_G(z_0,A_1,g),$$

$$J_{C3\text{-}z}=Z_{cand}(\bar V_\psi(F^5(z_0,A_{1:5}),z_g))-\alpha Z_{cand}(Q_G(z_0,A_1,g)),$$

$$J_{V1C}=\lVert F^5(z_0,A_{1:5})-z_g\rVert_2^2-\alpha Q_G(z_0,A_1,g),$$

其中 `Q_G(z0,A1,g)=G_phi(z0,E_A(A1),w(g))^T w(g)`，使用 retained online G、真实当前 latent `z0` 和第一个候选 action block；F 仍完整 rollout 五步。`Z_cand` 只在当前 CEM candidate set 内计算 population z-score。`alpha=0` 不重复运行：C3 等价于既有 State-V 结果，V1-C 等价于既有 F-only 结果。

下表把粗扫和针对 C3 Z-score 的边界细扫放在同一张表中。粗体是每行最佳；`◆` 是每个 alpha 列中已有方法的最佳；并列全部保留。DOCX 中黄底表示行最佳、蓝底表示列最佳、青色表示两者同时成立。

| 固定 checkpoint 与评分 | 训练 loss | alpha=0 | 0.025 | 0.05 | 0.075 | 0.1 | 0.15 | 0.2 | 0.25 | 0.5 | 1 | 2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| V1-C3 E12 · Raw State-V + First-Q | `L_C3` | ◆ **26/50 (52%)** | — | — | — | **26/50 (52%)** | — | — | 22/50 (44%) | 21/50 (42%) | 21/50 (42%) | 22/50 (44%) |
| V1-C3 E12 · Z-score State-V + First-Q2 | `L_C3` | ◆ 26/50 (52%) | ◆ 25/50 (50%) | ◆ 28/50 (56%) | ◆ 27/50 (54%) | ◆ **31/50 (62%)** | ◆ 23/50 (46%) | ◆ 28/50 (56%) | 26/50 (52%) | 24/50 (48%) | 24/50 (48%) | ◆ 25/50 (50%) |
| V1-C E10 · Original First-Q | `L_C` | 23/50 (46%) | — | — | — | 24/50 (48%) | — | — | ◆ **28/50 (56%)** | ◆ 27/50 (54%) | ◆ **28/50 (56%)** | ◆ 25/50 (50%) |

### 配对结果与结论

| 对比 | 成功数变化 | 仅新方法成功 | 仅参照成功 | Exact McNemar p |
| --- | ---: | ---: | ---: | ---: |
| C3 Z-score alpha=0.1 vs C3 State-V alpha=0 | 26 -> 31 (+10 pp) | 6 | 1 | 0.125 |
| C3 Z-score alpha=0.1 vs V1-C First-Q alpha=0.25 | 28 -> 31 (+6 pp) | 4 | 1 | 0.375 |
| C3 Z-score alpha=0.1 vs Legacy Hybrid F+G | 27 -> 31 (+8 pp) | 8 | 4 | 0.3877 |

- **本轮最高观察值是 C3 Z-score alpha=0.1 的 31/50 (62%)。** 它比 C3 State-V 增加 5 个成功 pair，也是现有固定表与本轮扫描中观察到的最高单格，但配对检验没有达到常用 0.05 阈值。
- **有效因素是尺度校准，不是简单叠加 Q。** C3 Raw 在 alpha=0.1 只与 State-V 持平，alpha>=0.25 时降到 42%-44%；这说明原始 State-V 与 Q 的数值尺度不匹配。候选内 z-score 后，小权重 Q 才能提供互补的第一步可达性信号。
- **alpha=0.1 是尖锐的局部峰值，而非平滑平台。** 相邻 alpha=0.075、0.15 分别只有 54%、46%，alpha=0.05 与 0.2 都为 56%。CEM 的精英排序会因小幅权重变化而离散改变，因此不能把 0.1 当作已证明的稳健常数。
- **原 V1-C First-Q 对权重相对宽容。** alpha=0.25 和 1 都为 28/50 (56%)，alpha=0.5 为 27/50；但它没有达到 C3 Z-score alpha=0.1 的观察值。
- **alpha 是在同一组 O50 上挑出的，62% 属于探索性调参结果。** 它不能无偏替代预先固定的正式 endpoint。下一步应在独立 dev pairs 上选 alpha，再在未见过的 test pairs 和多个 planning seeds 上确认；若保留 C3，还应改善 State-V 与 First-Q 的校准以降低这种尖峰敏感性。
- 历史 C3 Z-score alpha=0.25 为 27/50，本轮复跑为 26/50；两次有 3 个 pair 的结果翻转，净差 1 个成功。后续结论不应过度解释 1-2 个 episode 的差异，并应固定确定性设置或报告 planning 重复试验。

## 训练 / validation loss 证据

训练总 loss 含不同辅助项，绝对数值不能直接给 C–G3 排名，只用于判断各自是否收敛。Legacy 与 V1 曲线保留在历史来源文档；V0、V2、V2-EMA 的逐 epoch 数值和全部 E3–E10 O50 轨迹继续保留在总账 artifacts 中，但不再塞进主结果表。

V1-C2/C3 的下表和图只读取 endpoint archive 中 hash-bound 的 `training/v1_c2/metrics.csv` 与 `training/v1_c3/metrics.csv`；每个必需指标每个 epoch 必须恰有一个有限 aggregate，否则报告构建失败。

| 方法 | 指标 | 首个 epoch | 最终 epoch | 变化 |
| --- | ---: | ---: | ---: | ---: |
| V1-C2 | Train total loss | 26033.77 | 25074.55 | -3.7% |
| V1-C2 | Validation base TD loss | 965.30 | 812.76 | -15.8% |
| V1-C2 | First-Q ranking CE | 3.1343 | 3.1317 | -0.1% |
| V1-C2 | First-Q top-1 agreement | 9.09 | 9.22 | +0.13 pp; random 6.25% |
| V1-C3 | Train TD loss | 0.3932 | 0.3459 | -12.0% |
| V1-C3 | Validation TD loss | 0.3203 | 0.2940 | -8.2% |
| V1-C3 | Validation MC MAE | 12.750 | 12.029 | -5.7% |
| V1-C3 | Validation TD residual MAE | 9.935 | 9.233 | -7.1% |
| V1-C3 | Validation Spearman | 0.6420 | 0.6455 | +0.0035 |

![V1-C2 and V1-C3 training and validation diagnostics](reports/artifacts/actor_free_td_lewm_complete_cube_seed3072/figures/v1_c2_c3_training_validation_diagnostics.png)

C2 的 validation base TD loss 从 965.30 降至 812.76，但 First-Q ranking CE 仅从 3.1343 到 3.1317，top-1 agreement 从 9.09% 到 9.22%。E10 的 ranking CE 数值只相当于 train total loss 的 0.0125%；这不能单独证明梯度不足，但与其几乎不动和 endpoint 无净增益一致。下一版应先做 loss 标准化，再使用自适应权重或梯度平衡，而不是继续固定权重 1。
C3 的 validation TD loss 从 0.3203 降至 0.2940，MC MAE 从 12.750 到 12.029，TD residual MAE 从 9.935 到 9.233，Spearman 从 0.6420 到 0.6455。优化指标在缓慢改善，但正式控制 success 并未随训练后段单调提高，说明 value calibration 与 planner utility 仍需分开验证。

证据指纹：C2 `reports/artifacts/actor_free_td_lewm_v1_c2_c3_cube_seed3072/training/v1_c2/metrics.csv` (`aec425467a6aea9acd73c9a6de52ca262e05c0f442a450b968114bb5a60653a9`)；C3 `reports/artifacts/actor_free_td_lewm_v1_c2_c3_cube_seed3072/training/v1_c3/metrics.csv` (`28ead3a8ed78ecbf4e65f7193257a25c001399eb42337acd57a290ae4d21af55`)。

## 最佳训练方法与最佳测试评分

### 固定 E10 四版本均值

| 训练版本 | F-only | G-only | F+G | First-Q | Mean-Q | 五评分均值 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| V0 raw action | 46.0% | 35.7% | 46.0% | 47.0% | 46.3% | 44.2% |
| V1 action encoder | 46.0% | 41.3% | 47.7% | 52.0% | 49.7% | 47.3% |
| V2 joint fine-tune | 26.0% | 35.3% | 27.3% | 39.7% | 34.3% | 32.5% |
| V2-EMA-SG | 27.0% | 36.0% | 28.3% | 40.7% | 37.0% | 33.8% |

### 五种评分的四版本汇总

| 评分方式 | 覆盖版本 | 固定格数 | 四版本均值 | 最高固定单格 |
| --- | ---: | ---: | ---: | ---: |
| F-only | V0/V1/V2/V2-EMA | 24 | 36.2% | 12 tied: V0-C, V0-D, V0-F, …: 23/50 (46%) |
| G-only | V0/V1/V2/V2-EMA | 24 | 37.1% | V1-F: 23/50 (46%) |
| F+G tail | V0/V1/V2/V2-EMA | 24 | 37.3% | V1-G3: 27/50 (54%) |
| F + first-Q | V0/V1/V2/V2-EMA | 24 | 44.8% | V1-C: 28/50 (56%) |
| Mean-Q rollout | V0/V1/V2/V2-EMA | 24 | 41.8% | V1-G3: 27/50 (54%) |

### 1. 哪个训练方法最好

不存在脱离测试评分定义的唯一训练赢家。按原研究固定的 F+G 主列，领先配置为 **V1-G3: 27/50 (54%)**；若把五种评分等权平均，则 **V1-F, V1-G3 并列领先（48.8%）**；若寻找最高单格，则为 **V1-C + F + first-Q: 28/50 (56%)**。在 V2-EMA E10 内，五评分均值最高的训练变体为 **F（38.4%）**。这些都是描述性单 seed 结果。
从版本整体看，V1 action encoder 的六方法 × 五评分均值最高（47.3%）。

### 2. 哪个测试方法最好

V2-EMA E10 六个训练方法的均值为：F-only 27.0%、G-only 36.0%、F+G tail 28.3%、F + first-Q 40.7%、Mean-Q rollout 37.0%。跨 V0/V1/V2/V2-EMA 的固定 E10，**F + first-Q** 的 24 配置均值最高（44.8%），因此它是当前描述性默认主测试方式。Mean-Q 的最高固定配置为 V1-G3: 27/50 (54%)。

### 3. 原因分析

- V0→V1 后 G-only 均值从 35.7% 到 41.3%，First-Q 从 47.0% 到 52.0%；这与共享 Action Encoder 改善 G 读出一致。
- V1→V2 后 F-only 与 F+G 同时大幅下降，与 TD 梯度进入 online LeWM/Action Encoder 后产生 latent/control representation drift 的假设一致；单 seed 不能证明因果，问题也不能只归因于 critic。
- 当前均值领先读出是 F + first-Q；不同读出暴露于真实状态、imagined states 与 G 尺度的程度不同，OOD/rollout 误差仍是需要用独立 dev pairs 验证的解释。
- Mean-Q 在 24 个固定训练配置上的均值为 41.8%，最高格为 V1-G3: 27/50 (54%)；它是否应进入主评测由这一完整覆盖结果决定，不再按旧的 V2-only 结论处理。

## 负面结果与下一轮目标

| 发现 | 证据 | 含义/下一步 |
| --- | ---: | ---: |
| V2 world model 退化 | V1→V2 F-only 46.0%→26.0% | 先恢复 F，再谈 G 增益 |
| EMA 未根治 | V2→EMA 五评分变化：F-only +1.0 pp，G-only +0.7 pp，F+G tail +1.0 pp，F + first-Q +1.0 pp，Mean-Q rollout +2.7 pp | 冻结或低 LR 微调 encoder/world |
| tail 效果异质 | F+G 对 F-only：15 升 / 3 平 / 6 降；均值 +1.1 pp | 不能默认 tail 必然增益；以 F + first-Q 为候选并校准 G |
| Mean-Q 完整覆盖 | 24 配置均值 41.8%；V1-G3: 27/50 (54%) | 按完整四版本结果决定主评测或消融地位 |
| checkpoint 选择偏差 | 同一 O50 上看 E3–E10 再取最大 | dev pairs 选 epoch/alpha，正式 O50 只跑锁定配置 |
| 单 seed | 全部训练 seed=3072 | 至少 3 个训练 seeds，保存逐-pair outcome |

下一轮优先目标：

1. 把联合模型固定 E10 的 F-only 六法均值从 27.0% 恢复到至少 V1 的 46.0%。
2. 预注册 `V1-C + F + first-Q` 与 `V1-G3 + F+G tail` 作为主基线；正式 O50 不再事后选 epoch。
3. 若继续 joint training，先冻结 encoder/world 或给 TD 极低学习率，再分阶段解冻；增加对 V1 latent/prediction 的 anchor，并限制 TD 梯度进入 F。
4. 在独立 dev pair set 上选择 α、epoch 与 Q 校准；对 CEM 候选/imagined-state 分布加入 conservative/calibration 训练，抑制 OOD 高估。
5. 至少 3 个、最好 5 个 training seeds；用 paired bootstrap/McNemar 分析固定配置。

## 审计边界

- 477/477 格共享 episode-selection 文件 SHA-256 `e46ea81cce2e6a9a5df05ba04893b4181cbd8979340111a012c30f1efa2d7ee7` 与 action normalization SHA-256 `57f4d3c252e1805f4af1f614d20d1d1a064fa0d1d463ed5eb8ecf9dfc2b1a723`。
- fixed 新评分 launcher 另有 valid-row-ranks SHA-256 `88c204770f33c0b0220057d45b187766e3cfc54912e3f5ca49f2aa93d16437e9`；它是规范化索引哈希，不是 episode-selection 文件哈希，二者不能混写。
- 每格成功数都由 50 个布尔 outcome 重算；CSV `reports/artifacts/actor_free_td_lewm_complete_cube_seed3072/all_o50_results.csv`（SHA-256 `0e5b541bdb11cf6d647fc1e679499a02c3aa430d64e37c8819d02c44e1dcb900`）与 JSON `reports/artifacts/actor_free_td_lewm_complete_cube_seed3072/reconciliation_ledger.json`（SHA-256 `2f1985d5703e795f48bd8b850d470f76e35c998447112c86065e8bdcbbe1372b`）共同保留 477 格 / 23,850 个 outcomes。
- EMA E3 的 G1/F+G 与 G2/F-only 使用隔离 retry attempt_02；原失败调度证据保留，不把失败单元伪装成原调度成功。
- 原固定 E10 Mean-Q 覆盖 V0/V1/V2/V2-EMA × C/D/F/G1/G2/G3，共 24 格且无缺格；新增 First-Q2/State-V 不适用处用中性 `—`，不参与赢家计算。
- 主结果表只展示 E10；E3–E10 全轨迹仍保存在 `all_o50_results.csv`，没有因版式精简而删除。
- 只有一个 training seed；所有跨版本结果都是描述性结构消融，不声称多 seed 总体最优或统计显著。
