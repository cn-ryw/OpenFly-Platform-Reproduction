# 无人机大模型 Topic Review 材料索引

生成时间：2026-06-29

## 一句话状态

OpenFly 复现材料已经比较完整，AutoFly 已完成论文文本抽取并补充阅读笔记。当前目录用于给导师汇报和答辩，不用于替代 `CLAUDE.md`。

## 推荐阅读顺序

1. `presentation_outline.md`
   - 先看整体汇报结构和时间分配。
2. `openfly_reading_notes.md`
   - 了解 OpenFly 论文贡献，以及我们本地复现做到哪里。
3. `autofly_reading_notes.md`
   - 了解 AutoFly 的问题设定、方法和可批判点。
4. `openfly_vs_autofly_comparison.md`
   - 用于回答“这两篇论文有什么关系和差异”。
5. `prior_work_integration.md`
   - 用于回答“OpenFly/AutoFly 和之前无人机项目有什么联系，以及未来如何继续做”。
6. `defense_qna.md`
   - 用于准备导师提问。

## 源材料

| 材料 | 路径 | 用途 |
|---|---|---|
| OpenFly 论文抽取文本 | `../paper/openfly_paper.txt` | OpenFly 阅读笔记依据 |
| AutoFly 论文抽取文本 | `autofly_paper.txt` | AutoFly 阅读笔记依据 |
| OpenFly 阶段性复现报告 | `../advisor_review_report.md` | 汇报中复现实验部分的主依据 |
| 复现审计包 | `../reproduction_audit.md`, `../manifest.json` | 环境、模型、hash、命令证据 |
| AirSim smoke eval 分析 | `../airsim_smoke/20260629_005458/analysis.md` | 30 条闭环评估结果 |
| 官方 eval 对齐审计 | `../official_eval_alignment_audit.md` | 动作映射、prompt、历史帧逻辑对齐 |
| action 9 偏多分析 | `../action9_bias_analysis.md` | 失败模式解释 |
| 案例图页 | `../review_assets/airsim_env16_30_cases/case_gallery.png` | 展示成功/失败案例 |
| 既有项目关联分析 | `prior_work_integration.md` | 连接 Fast-Drone-250、VINS-Fusion、FAST-LIO2、EGO-Planner、Diff-Planner 与无人机大模型 |

## 为什么不写入 CLAUDE.md

`CLAUDE.md` 是给代码代理或开发工具使用的项目操作说明，不适合作为导师阅读材料。论文综述、复现结论、答辩准备应集中在 `demo_results/topic_review/` 和 `demo_results/advisor_review_report.md` 这类面向人的报告中。

## 当前汇报口径

可以说：

- 已经完成 OpenFly-Agent 在本机 RTX 4060 Laptop 8GB 上的 4bit inference 复现。
- 已经完成 prompt/frame sensitivity 轻量实验。
- 已经跑通 AirSim `env_airsim_16` 单场景 30 条 smoke eval。
- 已经对齐官方 eval 的主要动作映射、三帧输入、`vlnv1` 反归一化和 NE/SR/OSR/SPL 指标。

不能说：

- 已经完整复现 OpenFly 官方 benchmark。
- 当前单场景 4bit 结果可以直接和论文 test-seen/test-unseen 做严格横向比较。
- AutoFly 已经完成工程复现。AutoFly 目前只完成论文阅读和分析。
