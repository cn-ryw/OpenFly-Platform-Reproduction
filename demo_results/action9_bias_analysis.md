# Action 9 偏多分析

## 现象

在 `env_airsim_16` 30 条 smoke eval 中，模型预测动作分布为：

| action | count | percent |
|---:|---:|---:|
| 0 | 18 | 5.06% |
| 1 | 15 | 4.21% |
| 2 | 1 | 0.28% |
| 3 | 13 | 3.65% |
| 9 | 309 | 86.80% |

其中 action 9 是 9m forward。当前模型明显偏向长距离直飞。

## 官方数据先验

这并非完全异常。论文中明确提到，flight trajectory 中 Forward action 通常占较大比例，Turn Left/Right 或 Ascend/Descend 多出现在关键 landmark 附近。

本地统计也支持这一点：

| 数据 | action 9 占比 | 说明 |
|---|---:|---|
| `configs/eval_test.json` 全部 240 条 | 50.66% | 包含多个 eval 环境 |
| `env_airsim_16` 30 条，包含 -1/-2 | 44.20% | 原始动作列表中含 up/down 的负编号 |
| `env_airsim_16` 30 条，仅非负动作 | 65.72% | 与模型可执行模板更接近 |
| 本地预测 | 86.80% | 明显高于数据先验 |

结论：action 9 多是合理先验，但本地预测的偏置过强。

## 可能原因

1. **数据分布本身偏向 forward**
   - OpenFly 轨迹多为长距离地标导航。
   - 论文和 annotation 都显示 forward/long-forward 是主动作。

2. **20 step smoke eval 对 stop/turn 的容错低**
   - 官方 `train/eval.py` 设置 `MAX_STEP=100`。
   - 本地 smoke eval 用 `max_steps=20`，长轨迹样例更容易在尚未完成复杂转向前被截断。

3. **没有完整复刻论文 keyframe selection**
   - 论文的关键是通过动作变化和 landmark grounding 选择关键帧。
   - 本地 smoke eval 当前对齐官方 eval 的最近三帧输入，但没有显式实现论文中基于 bounding box / landmark 的 keyframe 选择。
   - 如果模型没有稳定看到“该转向/停止”的关键地标，输出 forward 的概率会偏高。

4. **prompt 形式可能影响 action decoder**
   - 官方 eval 直接传 `gpt_instruction`。
   - 训练 collator 构造过 `What action should the robot take to {instruction.lower()}?`。
   - 两种形式都在仓库中出现，后续应做 prompt wrapping ablation。

5. **4bit 量化可能放大近邻动作的决策偏置**
   - 本地为适配 8GB 显存使用 NF4 4bit。
   - action 8/9、turn/stop 的 token logits 如果接近，量化可能让 greedy decoding 更稳定地落到某个高频 token。
   - 当前没有 BF16 对照，不能定量归因，只能列为风险。

6. **round + exact template match 会把非模板输出变成 stop**
   - 官方和本地都是 raw action -> round -> exact template；不匹配默认 0。
   - 当前 action 9 多不是这个 fallback 造成的，因为 raw action 本身非常接近 `[0, 9, 0, 0, 0, 0, 0, 0]`。
   - fallback 更可能解释部分 early stop，而不是 forward 偏置。

## 与失败案例的关系

代表案例见：

- `demo_results/review_assets/airsim_env16_30_cases/README.md`
- `demo_results/review_assets/airsim_env16_30_cases/case_gallery.png`

主要失败模式：

- **near-miss overshoot**：曾进入 20m 半径，但继续 action 9，最后远离目标。
- **max-steps forward bias**：长时间输出 action 9，未能及时 turn/stop。
- **immediate stop fail**：少数样例 step 0 输出 stop，可能与图像/指令或模板 fallback 相关。
- **mixed turn near-miss**：有转向动作，但仍未在目标附近 stop。

## 可执行验证计划

1. 用 `--max-steps 100` 在 `env_airsim_16` 复跑 30 条，观察 action 9 比例、SR、OSR 是否变化。
2. 对同一 5 条样本做 prompt wrapping ablation。
3. 对同一 5 条样本做 history frame ablation。
4. 记录 raw action 与 nearest template 的 margin，判断 action 9 是否是高置信输出。
5. 如果资源允许，做 8bit 或 BF16/CPU-offload 小样本对照，判断量化影响。

## 汇报用一句话

当前 action 9 偏多不是单纯 bug：OpenFly 数据本身具有 forward-heavy 先验，但本地 4bit smoke eval 的 action 9 占比明显高于官方轨迹分布，说明还需要进一步对齐 max steps、prompt、历史关键帧和量化设置。
