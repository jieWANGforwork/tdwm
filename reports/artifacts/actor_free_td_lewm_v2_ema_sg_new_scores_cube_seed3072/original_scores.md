# Results TD — Actor-Free TD-LeWM V2-EMA-SG Cube O50

本实验唯一新增的局部预测变量是：LeWM MSE 使用 stop-gradient EMA next latent 作为 target；online history、action encoder 和 prediction 保持可训练。

本报告只在 6 个训练全部通过验收、18 个正式 O50 单元全部存在且协议一致后生成。
排名预先固定使用 **F+G**，不是在三种评分模式中事后挑最好结果。

| Rank | Method | F-only | G-only | F+G | Δ F+G − F-only |
| ---: | --- | ---: | ---: | ---: | ---: |
| 5 | V2-EMA-SG-C Coupled Hybrid Goal-Projected TD | 24% (12/50) | 36% (18/50) | 24% (12/50) | +0 pp |
| 2 | V2-EMA-SG-D Coupled Hybrid Goal-Value Weighted TD | 30% (15/50) | 38% (19/50) | 32% (16/50) | +2 pp |
| 1 | V2-EMA-SG-F Coupled Hybrid Same-Future Advantage | 30% (15/50) | 38% (19/50) | 34% (17/50) | +4 pp |
| 6 | V2-EMA-SG-G1 Coupled Hybrid Neighbor Action Advantage | 30% (15/50) | 30% (15/50) | 22% (11/50) | -8 pp |
| 2 | V2-EMA-SG-G2 Coupled Hybrid Prefix-Mean Advantage | 24% (12/50) | 40% (20/50) | 32% (16/50) | +8 pp |
| 4 | V2-EMA-SG-G3 Coupled Hybrid Prefix-Marginal Advantage | 24% (12/50) | 34% (17/50) | 26% (13/50) | +2 pp |

## 方法与推理协议

六种方法都联合微调 online LeWM 与一个 TD-JEPA predictor；25 维原始动作只经过 `world_model.action_encoder` 这一份共享编码器得到 192 维动作表示。没有 Actor，也没有 reward loss。

| Method | Training loss | Special mechanism | Inference |
| --- | --- | --- | --- |
| V2-EMA-SG-C Coupled Hybrid Goal-Projected TD | Coupled real-state and predicted-state feature TD plus goal-projected TD | Goal-derived tasks project both the detached target and online prediction | F-only H=5; G-only H=1; F+G H=5 |
| V2-EMA-SG-D Coupled Hybrid Goal-Value Weighted TD | Coupled real-state and predicted-state feature TD with detached goal-value weights | Goal-subset softmax weights are normalized to mean one | F-only H=5; G-only H=1; F+G H=5 |
| V2-EMA-SG-F Coupled Hybrid Same-Future Advantage | Coupled real-state and predicted-state feature TD with same-future/different-goal weights | The matching goal is contrasted with all goal-derived tasks in the batch | F-only H=5; G-only H=1; F+G H=5 |
| V2-EMA-SG-G1 Coupled Hybrid Neighbor Action Advantage | Coupled real-state and predicted-state feature TD with neighbor-action advantage weights | Other-episode frozen-latent KNN actions are comparison-only candidates | F-only H=5; G-only H=1; F+G H=5 |
| V2-EMA-SG-G2 Coupled Hybrid Prefix-Mean Advantage | Coupled real-state and predicted-state feature TD with prefix-mean advantage weights | The full action score is contrasted with zero-suffix action prefixes | F-only H=5; G-only H=1; F+G H=5 |
| V2-EMA-SG-G3 Coupled Hybrid Prefix-Marginal Advantage | Coupled real-state and predicted-state feature TD with prefix-marginal advantage weights | Mean adjacent prefix-score gains provide detached weights | F-only H=5; G-only H=1; F+G H=5 |

## 审计边界

- 18 个单元共享 selection SHA-256：`e46ea81cce2e6a9a5df05ba04893b4181cbd8979340111a012c30f1efa2d7ee7`。
- CEM 固定为 300 candidates、30 iterations、30 elites，episode budget 100；F+G 只在最后一个 action block 使用 G tail。
- 完整 world-model config canonical SHA-256：`0887bdf5711c4535ce77a2e3c6c7cbdf9f49f78d7ad7b1e6a6b40d708d56e099`。
- 每个方法三种评分模式绑定同一 epoch-10 checkpoint；每格成功率由 50 个布尔 outcome 重算。
- 协议写明 0.04 m，但当前 evaluator 调用的 `stable-worldmodel==0.1.1` `World` 公共构造器没有显式 threshold 参数；归档锁版本并披露此限制，没有声称运行时注入了该数值。
- 只有一个 training seed 和一组 planning selection；这是结构消融，不是多随机种子总体结论。
