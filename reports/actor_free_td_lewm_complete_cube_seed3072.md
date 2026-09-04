# Results TD — 全部 Actor-Free TD-LeWM 实验总账（Cube seed 3072）

本报告基于 **477 个已核验正式 O50 单元**，但主结果只展示每个训练配置的最终 E10 checkpoint，避免把逐 epoch 诊断结果与正式横向比较混在一起。每格均为同一组 50 个 start-goal pair；训练 seed=3072，planning seed=42。模型均不训练 Actor。

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

主表只使用固定 E10；全部 477 格和 23,850 个逐-pair 布尔结果仍由上述伴随文件完整保留。

## 结果覆盖与版本定义

| 版本/家族 | 方法数 | Checkpoint | 评分覆盖 | O50 格数 |
| --- | ---: | ---: | ---: | ---: |
| Legacy | 7 | E10 | F / G(C) / combined | 21 |
| V0 raw action | 6 | E10 | all five scores | 30 |
| V1 action encoder | 6 | E10 | all five scores | 30 |
| V2 joint fine-tune | 6 | E3-E10 | 3 original + E10 first/Mean | 156 |
| V2-EMA-SG | 6 | E3-E10 | all five scores | 240 |

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

## 五种测试方法怎么测

统一约定：`z0` 是当前真实图像经部署 encoder 得到的 latent，`z_g` 是 goal 图像的 latent，`z_k^F` 是 LeWM rollout 的 imagined latent。V1/V2/V2-EMA 先用共享 Action Encoder 得到 `e_k=E_A(A_k)`；V0 直接把归一化 25D action block 输入 G。`Q_G(z,A,g)=G(z,e,w(g))^T w(g)`。CEM 始终最小化 cost。

| 统一评测字段 | 固定设置 |
| --- | ---: |
| Environment | swm/OGBCube-v0 |
| Formal pairs | The same 50 same-episode start-goal pairs; goal offset 50 |
| CEM | 300 candidates, 30 iterations, 30 elites, planning seed 42, warm start |
| Execution | Minimize candidate cost, execute only the first action block A1, then observe and replan |
| Episode success | Object-to-goal distance <= 0.04 m within 100 environment steps |
| Checkpoint | All five columns in one row use the same fixed epoch-10 checkpoint; no score-specific retraining |

五个评分列的实际计算：

| 评分列 | F/G 的实际路径 | CEM 最小化的 cost | goal/Q 使用位置 |
| --- | ---: | ---: | ---: |
| F-only | F rolls A1...A5 from real z0 and produces imagined z1^F...z5^F; G is not called. | J_F = ||z5^F - z_g||_2^2 | Uses terminal goal distance at z5; no Q and no gamma. |
| G-only | H=1. G scores the real z0 and first candidate action A1; F is not rolled out. | J_G = -Q_G(z0,A1,g) | No explicit goal distance and no gamma; minimizing -Q maximizes Q. |
| F+G tail | F rolls only A1...A4 to z4^F; G evaluates the fifth transition from z4^F with A5. | J_tail = ||z4^F - z_g||_2^2 - gamma^4 Q_G(z4^F,A5,g) | Uses z4 goal distance and the deepest imagined-state Q; gamma=0.95. |
| F + first-Q | F completes the five-step rollout; G is read only once at the real z0 with A1. | J_first = ||z5^F - z_g||_2^2 - 0.25 Q_G(z0,A1,g) | Uses terminal goal distance; the Q term is not multiplied by gamma^4. |
| Mean-Q rollout | F generates predecessors z0,z1^F,...,z4^F; G scores each aligned pair (z{k-1}^F,Ak). | J_mean = -(1/5) sum[k=1..5] Q_G(z{k-1}^F,Ak,g) | No terminal goal distance; z5 is not read by G and gamma is unused. |

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

## C–G3 固定 E10 主结果矩阵

横向读每一行，可以比较同一个训练方法最适合哪一种评分；纵向读每一列时，以版本为边界，只比较该版本的 C/D/F/G1/G2/G3。Markdown 中 **粗体**是行最佳，`◆` 是同版本列最佳；并列全部标记。DOCX 使用黄底表示行最佳、蓝底表示同版本列最佳、青色底表示两者同时成立。

| 版本 | 训练方法 | F-only | G-only | F+G tail | First-Q | Mean-Q |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| V0 | C | ◆ 23/50 (46%) | 18/50 (36%) | 24/50 (48%) | ◆ **26/50 (52%)** | 22/50 (44%) |
| V0 | D | ◆ 23/50 (46%) | ◆ 20/50 (40%) | 21/50 (42%) | 24/50 (48%) | ◆ **26/50 (52%)** |
| V0 | F | ◆ 23/50 (46%) | 19/50 (38%) | 20/50 (40%) | **24/50 (48%)** | **24/50 (48%)** |
| V0 | G1 | ◆ 23/50 (46%) | 16/50 (32%) | ◆ **25/50 (50%)** | 20/50 (40%) | 20/50 (40%) |
| V0 | G2 | ◆ 23/50 (46%) | 16/50 (32%) | ◆ **25/50 (50%)** | 23/50 (46%) | 23/50 (46%) |
| V0 | G3 | ◆ 23/50 (46%) | 18/50 (36%) | 23/50 (46%) | **24/50 (48%)** | **24/50 (48%)** |
| V1 | C | ◆ 23/50 (46%) | 18/50 (36%) | 22/50 (44%) | ◆ **28/50 (56%)** | 21/50 (42%) |
| V1 | D | ◆ 23/50 (46%) | 22/50 (44%) | 21/50 (42%) | 25/50 (50%) | **26/50 (52%)** |
| V1 | F | ◆ 23/50 (46%) | ◆ 23/50 (46%) | 24/50 (48%) | **26/50 (52%)** | **26/50 (52%)** |
| V1 | G1 | ◆ 23/50 (46%) | 21/50 (42%) | 24/50 (48%) | **26/50 (52%)** | 25/50 (50%) |
| V1 | G2 | ◆ 23/50 (46%) | 21/50 (42%) | **25/50 (50%)** | **25/50 (50%)** | 24/50 (48%) |
| V1 | G3 | ◆ 23/50 (46%) | 19/50 (38%) | ◆ **27/50 (54%)** | 26/50 (52%) | ◆ **27/50 (54%)** |
| V2 | C | 13/50 (26%) | ◆ **20/50 (40%)** | ◆ 16/50 (32%) | 19/50 (38%) | 10/50 (20%) |
| V2 | D | ◆ 16/50 (32%) | 18/50 (36%) | ◆ 16/50 (32%) | ◆ 21/50 (42%) | ◆ **24/50 (48%)** |
| V2 | F | 15/50 (30%) | 19/50 (38%) | 13/50 (26%) | ◆ 21/50 (42%) | **22/50 (44%)** |
| V2 | G1 | 12/50 (24%) | 17/50 (34%) | 13/50 (26%) | **19/50 (38%)** | 17/50 (34%) |
| V2 | G2 | 12/50 (24%) | 18/50 (36%) | 13/50 (26%) | **20/50 (40%)** | 13/50 (26%) |
| V2 | G3 | 10/50 (20%) | 14/50 (28%) | 11/50 (22%) | **19/50 (38%)** | 17/50 (34%) |
| V2-EMA | C | 12/50 (24%) | 18/50 (36%) | 12/50 (24%) | **20/50 (40%)** | 14/50 (28%) |
| V2-EMA | D | ◆ 15/50 (30%) | 19/50 (38%) | 16/50 (32%) | ◆ **22/50 (44%)** | 19/50 (38%) |
| V2-EMA | F | ◆ 15/50 (30%) | 19/50 (38%) | ◆ 17/50 (34%) | ◆ 22/50 (44%) | ◆ **23/50 (46%)** |
| V2-EMA | G1 | ◆ 15/50 (30%) | 15/50 (30%) | 11/50 (22%) | 19/50 (38%) | **21/50 (42%)** |
| V2-EMA | G2 | 12/50 (24%) | ◆ **20/50 (40%)** | 16/50 (32%) | 18/50 (36%) | 17/50 (34%) |
| V2-EMA | G3 | 12/50 (24%) | 17/50 (34%) | 13/50 (26%) | **21/50 (42%)** | 17/50 (34%) |

### 每个版本内部的逐列赢家

| 版本 | F-only | G-only | F+G tail | First-Q | Mean-Q |
| --- | ---: | ---: | ---: | ---: | ---: |
| V0 | C/D/F/G1/G2/G3 23/50 | D 20/50 | G1/G2 25/50 | C 26/50 | D 26/50 |
| V1 | C/D/F/G1/G2/G3 23/50 | F 23/50 | G3 27/50 | C 28/50 | G3 27/50 |
| V2 | D 16/50 | C 20/50 | C/D 16/50 | D/F 21/50 | D 24/50 |
| V2-EMA | D/F/G1 15/50 | G2 20/50 | F 17/50 | D/F 22/50 | F 23/50 |

**跨四版本的全局逐列赢家（只用于补充分析，不对应 DOCX 蓝框）：** F-only = 12 tied: V0-C, V0-D, V0-F, …: 23/50 (46%); G-only = V1-F: 23/50 (46%); F+G tail = V1-G3: 27/50 (54%); F + first-Q = V1-C: 28/50 (56%); Mean-Q rollout = V1-G3: 27/50 (54%)。

## 训练 / validation loss 证据

训练总 loss 含不同辅助项，绝对数值不能直接给 C–G3 排名，只用于判断各自是否收敛。Legacy 与 V1 曲线保留在历史来源文档；V0、V2、V2-EMA 的逐 epoch 数值和全部 E3–E10 O50 轨迹继续保留在总账 artifacts 中，但不再塞进主结果表。

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
- 固定 E10 Mean-Q 覆盖 V0/V1/V2/V2-EMA × C/D/F/G1/G2/G3，共 24 格；主结果表不允许缺格或使用占位符。
- 主结果表只展示 E10；E3–E10 全轨迹仍保存在 `all_o50_results.csv`，没有因版式精简而删除。
- 只有一个 training seed；所有跨版本结果都是描述性结构消融，不声称多 seed 总体最优或统计显著。
