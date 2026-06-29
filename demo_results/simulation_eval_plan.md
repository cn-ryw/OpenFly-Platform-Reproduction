# OpenFly 仿真评估复现路线

## 当前结论

当前机器已经完成 OpenFly-Agent 4bit 本地推理复现，并已跑通 `env_airsim_16` 的 30 条 AirSim 4bit smoke eval。当前还没有完成完整 OpenFly benchmark 仿真评估复现；完整指标仍需扩展到更多 AirSim 场景，并在条件允许时与官方 `train/eval.py` 流程对齐。

## 本地状态

- OS/GPU: Ubuntu 22.04, RTX 4060 Laptop 8GB。
- 已有模型推理环境：`openfly` conda 环境，4bit inference 可运行。
- 已有 annotation：`openfly_ann/Annotation/seen.json`、`unseen.json`。
- 已有 demo video 和自有三帧。
- `configs/eval_test.json` 包含 8 个环境，每个 30 条样例：`env_airsim_16/18/23/26/sh/gz`、`env_ue_smallcity`、`env_ue_bigcity`。
- 当前 `envs/airsim/env_airsim_16/LinuxNoEditor/start.sh` 已存在，单场景资产来自本地下载的 `/home/ruan/Downloads/env_airsim_16.zip`。
- 当前 `envs/ue` 只有占位文件，没有 UE 场景包。
- 当前 Python 环境已安装最小 AirSim 客户端依赖：`airsim==1.8.1`、`msgpack-rpc-python==0.4.1`；`unrealcv` 未安装。
- `msgpack-rpc-python` 安装了 `tornado==4.5.3`，`pip check` 当前无 broken requirements；若后续使用 Jupyter/Web 工具，需要留意该旧版本依赖。
- `~/Documents/AirSim/settings.json` 已备份原始最小配置，并替换为 OpenFly 项目配置，使 `front_custom` 相机和 `drone_1` vehicle 生效。
- 当前 3DGS bridge 代码中存在硬编码外部数据路径，不适合作为第一阶段复现目标。
- GTAV 评估需要 Windows、GTA V 和 DeepGTAV，不适合当前 Ubuntu 本机优先推进。

## 场景资产检查

- HuggingFace metadata 显示 `airsim/env_airsim_16.zip` 大小约 `1.91 GB`，同场景 PCD `pcd_map/env_airsim_16.pcd` 约 `107 MB`。
- 本机当前空间足够进行单场景下载和解压，但不适合一次性下载完整 OpenFly 数据资产。
- 直接下载 `IPEC-COMMUNITY/OpenFly_DataGen/airsim/env_airsim_16.zip` 时返回 gated repo `401 Unauthorized`。
- 已按 Hugging Face agents-cli 路线安装 standalone `hf` CLI：`/home/ruan/.local/bin/hf`，版本 `1.21.0`。
- 已安装 HF CLI agent skill：`/home/ruan/.agents/skills/hf-cli`。
- 当前机器 `hf auth whoami` 显示未登录，且 standalone CLI 环境信息显示 `Has saved token ?: False`。
- 环境变量 `HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN` 未设置。
- 已生成单场景下载脚本 `download_openfly_airsim_scene.py`，它只下载 `airsim/<env>.zip`，检查 zip 路径安全性，解压到 `envs/airsim/`，并记录 manifest 到 `demo_results/downloads/<env>/`。
- 已使用本地 zip 完成解压，manifest: `demo_results/downloads/env_airsim_16/20260629_002430.json`。

## 推荐路线：AirSim First

第一阶段优先使用 AirSim 做 smoke evaluation，原因：

- `train/eval.py` 已直接支持 AirSimBridge。
- README 的评估表中 AirSim 场景占比高。
- AirSim 对 Ubuntu 本机更现实；不需要 Windows/GTA。
- 相比 UE，AirSim 场景启动路径更直接：`envs/airsim/<env_name>/LinuxNoEditor/start.sh`。

第一阶段已选择 `env_airsim_16`：

- 已从本地 `/home/ruan/Downloads/env_airsim_16.zip` 解压单个场景包到 `envs/airsim/env_airsim_16/`。
- 已从 `configs/eval_test.json` 中筛出 `env_airsim_16` 样例，并完成该场景 30 条 smoke eval。
- 使用 `eval_airsim_4bit_smoke.py` 作为轻量评估脚本：4bit 加载模型，默认 `unnorm_key="vlnv1"` 贴近官方 `train/eval.py`，避免 README 默认 BF16 + flash-attn 导致 8GB 显存风险。
- 记录每条样例的 action 序列、最终距离、SR/OSR/SPL/NE，以及完整日志。

当前 dry-run 已通过样例筛选和依赖检查：

```bash
python -u eval_airsim_4bit_smoke.py --dry-run --env-name env_airsim_16 --limit 3 --max-steps 20
```

dry-run 输出显示：

- 可选中 `env_airsim_16` 前 3 条官方 eval 样例。
- `airsim` 和 `msgpackrpc` 可 import。
- GPU 可见：RTX 4060 Laptop, Driver 580.159.03, compute capability 8.9。
- `envs/airsim/env_airsim_16/LinuxNoEditor/start.sh` 已存在。

## AirSim Smoke Eval 结果

已完成 30 条 `env_airsim_16` smoke eval：

```bash
python -u eval_airsim_4bit_smoke.py --env-name env_airsim_16 --limit 30 --max-steps 20 --save-frames --prepare-settings
```

结果路径：

- `demo_results/airsim_smoke/20260629_005458/results.json`
- `demo_results/airsim_smoke/20260629_005458/summary.md`
- `demo_results/airsim_smoke/20260629_005458/analysis.md`
- `demo_results/airsim_smoke/20260629_005458/analysis.json`
- `demo_results/airsim_smoke/20260629_005458/run.log`
- `demo_results/airsim_smoke/20260629_005458/contact_sheet.png`
- `demo_results/airsim_smoke/20260629_005458/sample_*/step_*.png`
- Review cases: `demo_results/review_assets/airsim_env16_30_cases/README.md`
- Review gallery: `demo_results/review_assets/airsim_env16_30_cases/case_gallery.png`
- Official eval alignment audit: `demo_results/official_eval_alignment_audit.md`
- Action 9 bias analysis: `demo_results/action9_bias_analysis.md`
- Advisor review report: `demo_results/advisor_review_report.md`

关键结果：

- Samples: `30`
- Mean NE: `76.457`
- Mean SR: `0.1667`
- Mean OSR: `0.5667`
- Mean SPL: `0.1346`
- peak PyTorch allocated GPU memory: `4526.9 MB`
- Action histogram: `{0: 18, 1: 15, 2: 1, 3: 13, 9: 309}`
- Sample classes: `{'fail': 10, 'max_steps_fail': 3, 'near_miss_osr': 12, 'success': 5}`
- Stop predicted count: `18`
- Max-step count: `13`
- Image error count: `0`

说明：

- 首次实际运行失败在 `simGetImages(front_custom)`，原因是 `~/Documents/AirSim/settings.json` 没有 OpenFly 的 `front_custom` 相机配置。
- 修复方式是使用 `--prepare-settings` 备份原配置并复制 `envs/airsim/AirSim/settings.json`。
- `airsim_stdout.log` 中的 `error code: 143` 是脚本结束时主动终止 AirSim 场景进程，不是仿真失败。
- 30 条 smoke eval 说明链路可稳定运行，且没有 OOM、Traceback 或 AirSim 图像错误。
- 指标低于完整论文表格不是意外：当前是 4bit、本机轻量脚本、单场景 smoke eval，不是官方完整 BF16/多场景 benchmark。
- 主要失败模式包括：过近点后没有及时 stop 导致 overshoot、step 0/早期 stop、一直 action 9 前进直到 max steps，以及短轨迹样例上对 stop 时机不稳。
- 当前动作分布明显偏向 action 9，需要后续和官方动作模板、`unnorm_key`、prompt 格式、历史帧构造方式进一步对齐。

历史 3 条试跑结果保留在：

- `demo_results/airsim_smoke/20260629_004035/results.json`
- `demo_results/airsim_smoke/20260629_004035/summary.md`
- `demo_results/airsim_smoke/20260629_004035/analysis.md`

## 后续路线

- UE：等 AirSim 单场景误差分析完成后再考虑。需要下载 UE 场景、UnrealCV、端口配置和更高渲染资源。
- 3DGS：需要先修复 `GSBridge` 中硬编码数据路径，并准备 SIBR viewer 和 3DGS 数据。
- GTAV：放到最后，除非有 Windows/GTA V/DeepGTAV 环境。

## 风险与限制

- 当前可用空间需要谨慎；已下载单个 AirSim 场景，不下载完整 OpenFly 数据集。
- 官方 `train/eval.py` 默认 BF16/flash-attn 不是当前 8GB 复现策略，需要改成 4bit eval。
- 30 条单场景 smoke eval 已能汇报 `NE / SR / OSR / SPL`，但不能代表完整 OpenFly benchmark。

## 建议下一步

1. 检查官方 `train/eval.py` 中 AirSim action 映射、prompt 构造和图像历史帧逻辑，找出轻量脚本与官方流程的差异。
2. 基于 review cases 写一页导师汇报草稿：成功链路、失败模式、限制、下一阶段计划。
3. 后续仍建议在 HuggingFace 账号侧申请/确认 `IPEC-COMMUNITY/OpenFly_DataGen` 访问权限，方便获取其他场景资产。
4. 如磁盘和时间允许，下载第二个小型 AirSim 场景，做跨场景 smoke eval；否则先深挖 `env_airsim_16` 失败模式。
