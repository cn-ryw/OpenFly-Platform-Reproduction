# AirSim 30 条 Smoke Eval 代表案例

用途：给导师 review 时快速展示本机 4bit AirSim 单场景评估的成功与失败模式。

- Run dir: `demo_results/airsim_smoke/20260629_005458`
- Gallery: `demo_results/review_assets/airsim_env16_30_cases/case_gallery.png`
- 说明：这是单场景 smoke eval，不等同于完整 OpenFly benchmark。

## 案例表

| case | sample | class | steps | GT len | NE | min NE | SR | OSR | SPL | actions | 解释 |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| success_best | 0 | success | 12 | 16 | 7.436 | 7.436 | 1 | 1 | 1.0461 | `[9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 0]` | Successful stop inside the 20m success radius. |
| success_with_turns | 6 | success | 11 | 9 | 19.726 | 6.159 | 1 | 1 | 0.7246 | `[9, 9, 9, 9, 9, 9, 9, 1, 1, 2, 0]` | Successful sample that used more than only forward/stop. |
| near_miss_overshoot | 18 | near_miss_osr | 20 | 5 | 166.716 | 7.436 | 0 | 1 | 0.0000 | `[9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9]...` | Entered the success radius but ended far away. |
| immediate_stop_fail | 17 | fail | 1 | 50 | 164.224 | 164.224 | 0 | 0 | 0.0000 | `[0]` | Predicted stop at step 0 while still far from target. |
| max_steps_forward_bias | 4 | max_steps_fail | 20 | 17 | 83.588 | 46.569 | 0 | 0 | 0.0000 | `[9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9]...` | Hit max steps with a strong action-9 forward bias. |
| mixed_turn_near_miss | 23 | near_miss_osr | 20 | 13 | 41.173 | 15.059 | 0 | 1 | 0.0000 | `[9, 9, 9, 9, 9, 9, 9, 9, 3, 9, 9, 3]...` | Failure case with several turn/vertical actions. |

## 初步结论

- 成功样例证明 4bit 模型、AirSim 图像获取、动作执行、NE/SR/OSR/SPL 记录链路是可运行的。
- near-miss/overshoot 样例说明模型有时进入成功半径，但 stop 时机不稳定。
- immediate stop 样例说明部分场景会过早输出 stop，需要检查 prompt、历史帧和 action decoding 是否与官方流程完全一致。
- max-steps forward-bias 样例说明 action 9 占比偏高，后续要和官方 `train/eval.py` 的 action mapping 与 frame history 构造对齐。
