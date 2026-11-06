# 运行脚本与 Slurm 整理

## 日常入口

在服务器项目根目录运行。默认只打印计划，不创建目录、不启动模型、不调用 sbatch。

```bash
python scripts/jobs/submit.py list
python scripts/jobs/submit.py eval sufficient --model qwen25
python scripts/jobs/submit.py eval merged --model internvl --manifest /path/to/manifest.jsonl --results-dir results/runs/new_run
```

确认参数后在同一命令末尾加 `--submit`。没有 Slurm 的计算节点可显式使用 `--run-local`；它直接执行单个作业，不申请调度资源。本次整理没有提交任何作业。

```bash
python scripts/jobs/submit.py eval sufficient --model qwen25 --submit
python scripts/jobs/submit.py job build.nextgqa.measure --array 0-7%4 --env NUM_SHARDS=8 --submit
python scripts/jobs/submit.py job release.full_small_array --array 0-7%2 --submit -- /absolute/release minicpm
python scripts/jobs/submit.py job release.merge_full --dependency afterok:12345 --submit -- /absolute/release minicpm
```

`--` 后是专用脚本的位置参数；入口选项均放在 `--` 前。资源覆盖例如 `--resource time=02:00:00`、`--resource partition=debug`，换队列时也须自行匹配 `qos`。参数或资源值含空格时加引号。普通 eval 预设不能使用数组，它们不按任务编号切分输入；数组评测应使用 release 作业。

## 文件职责

| 位置 | 用途 |
|---|---|
| `scripts/jobs/submit.py` | 预览、提交、本地执行；创建日志父目录；记录 Slurm 返回值和提交回执 |
| `scripts/configs/cluster.json` | HKUST 根目录、Python 环境、单卡/双卡资源默认值 |
| `scripts/configs/models.json` | 模型目录、服务名与资源配置关联 |
| `scripts/configs/vllm_models.sh` | 各模型服务参数，明确保留视频/图片输入和处理器差异 |
| `scripts/configs/evaluations.json` | 23 个模型×实验预设，包含原有 tag、输出目录、Partial alpha/样本数和续跑种子来源 |
| `scripts/configs/jobs.json` | 53 个专用 Slurm 作业的索引与资源默认值 |
| `scripts/jobs/lib/` | sufficient/insufficient/partial/merged 共用运行器与服务生命周期 |
| `scripts/jobs/build/nextgqa/` | 抽样测量、阈值校准、证据定位、分类、freeze 构建和旧 Qwen3 no-leak 评测 |
| `scripts/jobs/build/etbench/` | Conflict 构建、action 恢复、scene 核验与定稿 |
| `scripts/jobs/analysis/` | QA 审计、视觉复核、干扰项筛选和 benchmark 定稿 |
| `scripts/jobs/experiments/` | 消融、控制实验、Partial seeded controls、风险阶梯及专项指标 |
| `scripts/eval/open_v2/` | 保留冻结发布包所依赖的稳定入口；也在统一作业索引中 |

已有的依赖链提交脚本（submit_grounding_ensemble.sh 等）保留，内部引用已改为新路径。模型服务、下载、探测工具保留独立功能。仅靠名字旧不能判断一个消融或审计实验无用。

## 参数和行为

- `sufficient` 支持 classify，以及 `--env EVAL_TASK=qa`；其余通用条件为 classify。
- `insufficient` 保留 frozen 视频、样本完整性/去重/32 帧检查。
- `partial-n500`、`partial-a010`、`partial-selected4` 保留原样本集、alpha、预期数量；a010 保留从旧 n500 结果初始化的逻辑。
- `merged` 保留 manifest 前后校验；四个模型可选，默认输出到 `results/runs/`。比较历史结果仍须使用对应冻结版本。
- `conflict` 保留独立时间证据采样和专用评测实现，合并的仅是三个模型提交外壳。
- `sufficient-slow` 保留 Qwen 处理器消融。
- LLaVA 的 sufficient/insufficient/partial 使用 video；merged 使用 frames。Qwen sufficient 显式 use_fast，其他条件保留原来未显式指定的设置。没有为了统一形式而改变这些实验参数。
- `--manifest` 对 merged 指 JSONL，对 sufficient/insufficient/partial 指 filtered JSON，对 conflict 指 QA JSONL；它不负责生成或改变输入。
- `--results-dir`、`--model-tag`、`--model-path` 可覆盖通用预设。已有预设保留原输出路径以支持续跑；新实验应指定新目录/tag，避免混合条件。
- `--env KEY=VALUE` 优先于预设；预设优先于已识别的运行环境变量。凭证仍从当前环境或已有密钥文件读取，不写入配置文件或命令行。
- 通用 eval 支持 `--root` / `VID_EVI_ROOT` 和 `--python-env` / `VID_EVI_ENV`。专用历史/冻结脚本仍包含 HKUST 路径，不声称可任意迁移主机，入口会拒绝对它们使用这两个覆盖参数。
- 日志位于 `logs/jobs/<preset>/`，专用作业保留原日志路径；Slurm 回执位于 `reports/job_submissions/`，含任务 ID、命令、脚本哈希。提交被中断或返回值异常时先查回执/队列，不能盲目重复提交。

## Partial 当前状态

预生成 Partial 视频此前已被清空。通用 Partial 预设的运行计划可查看，但真正提交/本地运行前会核对候选数、alpha 数和每个视频是否存在，缺失即拒绝提交。此轮没有实现按需逐帧加噪；保留 seeded control 实验也不等于已经替换主评测输入。

## 删除与迁移

本轮将 19 个重复提交文件替换为配置、迁移 37 个有独立用途的 Slurm 脚本，另删除 5 个过时入口。`job_migration.json` 列出旧路径、替代位置/预设以及原因。

真正弃用的 5 个入口：

| 旧文件 | 原因与保留能力 |
|---|---|
| `scripts/analysis/resubmit_conflict_after_llava.sh` | 写死取消四个历史 Slurm ID；保留通用依赖链和重提能力 |
| `scripts/eval/launch_gemini25pro_v2_retry49_hkstu.sh` | 针对日期固定的 49 条补跑；保留 API evaluator |
| `scripts/eval/gemini25pro_main_table_add479.sbatch` | 针对日期固定的 479 条补跑；保留 API evaluator |
| `scripts/eval/gemini25pro_main_table_replacement.sbatch` | 同一补跑的 replacement 入口；保留 API evaluator |
| `results/build/etbench/conflict/queue_action_content_recovery_smoke.sh` | 轮询 GPU 显存并自行抢占的临时队列；保留 recovery smoke 和 Slurm recovery 作业 |

上述 5 个文件在当前脚本/管线文档中没有调用者，当前用户进程中没有匹配入口；静态检查不能排除外部手工调用。Git 前一提交保留原文件；服务器部署前另存哈希清单及恢复压缩包于 `reports/job_cleanup_20261106/`。

四个 `scripts/eval/run_*_model.sh` 旧通用入口仍可调用，转发到共同实现。19 个已替换的单模型 sbatch 不保留空壳，改用预设。历史 `results/`、`results/archive/codex_runs/` 冻结包不迁移、不改写。新建 merged 冻结包会一并复制共享运行器及模型配置，可离开当前源码目录独立运行。

## 验证边界

验证覆盖所有 shell/Slurm 语法、配置生成、资源覆盖、数组/依赖参数、删除路径引用、Partial 提交拦截、模拟 sbatch 回执、模型输入差异以及 Linux HTTP 模拟服务的启动与进程组清理。没有调用真实模型、收费 API 或 Slurm 调度器；当前 hkstu 节点没有 sbatch/squeue。

验证结果：本地 155 项通过、3 项 Linux 生命周期测试留到服务器执行；服务器完整测试 158 项通过、无跳过。94 个 shell/Slurm 文件通过 bash -n。
