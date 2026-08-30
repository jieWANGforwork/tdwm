# Actor-Free TD-LeWM C–G3：raw-action V0 与 action-encoder V1

这份报告只比较后来单独实现的 `C/D/F/G1/G2/G3` controlled study。它不把更早的
7-method Actor-Free TD-LeWM 结构消融改名为 V0；旧 7-method 结果仍单独保留在
`actor_free_td_lewm_cube_seed3072.md`。

## 唯一有意改变的核心变量

- **V0**：归一化后的原始 25D action block 直接输入 TD-JEPA predictor：
  `G(z_t, a_t, w)`。
- **V1**：先用预训练 LeWM 自己的共享冻结 Action Encoder 得到
  `stopgrad(E_A(a_t)) ∈ R^192`，再输入 predictor：
  `G(z_t, stopgrad(E_A(a_t)), w)`。
- 当前 action 和数据集真实 `next_action` 都经过同一个 frozen/eval/no-grad
  `world_model.action_encoder`；EMA target 不拥有第二套 action encoder。
- 两版都没有 Actor、policy 或 reward model；C/D/F/G1/G2/G3 的训练目标、EMA/TD
  target、任务采样和 F-only/G-only/F+G 规划语义相同。
- 这不是严格等参数量消融：V0 predictor 为 336,320 参数，V1 为 379,072 参数，因为
  state-action 分支的 action 输入从 25D 变为 192D。

## 正式 O50

两版都使用 training seed 3072、planning seed 42、同一组 50 个 start–goal pair、
goal offset 50，以及 F-only/G-only/F+G 的固定 horizon 5/1/5。

| 方法 | V0 F-only | V0 G-only | V0 F+G | V1 F-only | V1 G-only | V1 F+G | V1−V0 F+G |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| C | 23/50 (46%) | 18/50 (36%) | 24/50 (48%) | 23/50 (46%) | 18/50 (36%) | 22/50 (44%) | −2/50 (−4 pp) |
| D | 23/50 (46%) | 20/50 (40%) | 21/50 (42%) | 23/50 (46%) | 22/50 (44%) | 21/50 (42%) | 0/50 |
| F | 23/50 (46%) | 19/50 (38%) | 20/50 (40%) | 23/50 (46%) | 23/50 (46%) | 24/50 (48%) | +4/50 (+8 pp) |
| G1 | 23/50 (46%) | 16/50 (32%) | 25/50 (50%) | 23/50 (46%) | 21/50 (42%) | 24/50 (48%) | −1/50 (−2 pp) |
| G2 | 23/50 (46%) | 16/50 (32%) | 25/50 (50%) | 23/50 (46%) | 21/50 (42%) | 25/50 (50%) | 0/50 |
| G3 | 23/50 (46%) | 18/50 (36%) | 23/50 (46%) | 23/50 (46%) | 19/50 (38%) | 27/50 (54%) | +4/50 (+8 pp) |

## 排名与结论边界

- V0 的 F+G 首位是 G1/G2，并列 25/50（50%）。
- V1 的 F+G 首位是 G3，27/50（54%）。
- V1 相对 V0 的 F+G：F、G3 各提高 8 pp；D、G2 不变；C 降低 4 pp；G1 降低
  2 pp。
- 这些数值来自一个 training seed 和一个 planning selection，只支持当前固定选择上的
  结构消融；不能声称 action encoder 在多随机种子上总体提高性能。

V0 的 18 个 raw `results.json` 路径与 SHA-256 记录在
`artifacts/actor_free_td_lewm_v0_cube_seed3072/formal_o50_summary.json`；V1 的完整
50-pair outcome、训练曲线、checkpoint/protocol 和验收证据记录在
`artifacts/actor_free_td_lewm_v1_cube_seed3072/`。
