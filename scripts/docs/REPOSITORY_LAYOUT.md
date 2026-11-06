# 仓库目录整理 — 2026-11-07

用户最终要求：可以整理结果目录及路径，但不得删除模型返回的详细 JSON/JSONL。所有历史结果 JSON/JSONL、JSON 压缩文件均纳入迁移前后 SHA-256 核验；本次不重算、不覆盖模型输出。

## 新结构

```text
source_datasets/
models/
  .runtime/                 # 原专用模型运行环境
pipelines/
  nextgqa/
  etbench/
  benchmark/
  media/
scripts/
  eval/                     # 保留已有评测命令入口
  analysis/
  src/vid_evi_qa/            # 评测、适配器、提示词、指标实现
  jobs/                     # 统一入口及 Slurm
  configs/
  tests/
  docs/
results/
  ...                       # 已有实验目录，原名称与内容保留
  build/nextgqa/            # 旧 nextgqa_pipeline 的数据产物
  build/etbench/            # 旧 etbench_pipeline 的数据产物
  reports/                  # 原 reports
  logs/                     # 原 logs
  analysis/nextgqa_leakage_analysis/
  archive/codex_runs/
  archive/scratch/          # 原 tmp，含实际实验结果，未当垃圾整目录删除
```

## 路径对照

| 旧位置 | 新位置 |
|---|---|
| `configs/`、`jobs/`、`src/`、`tests/`、`docs/` | 同名目录置于 `scripts/` 下 |
| `nextgqa_pipeline/` 的数据文件 | `results/build/nextgqa/` |
| `etbench_pipeline/` 的数据文件 | `results/build/etbench/` |
| 上述两个旧目录的运行脚本与代码文档 | `pipelines/nextgqa/`、`pipelines/etbench/`；36 个旧转发 Python 文件移除，使用已有正式实现 |
| `nextgqa_pipeline/insufficient/tests/` | `scripts/tests/nextgqa/` |
| `reports/` | `results/reports/` |
| `logs/` | `results/logs/` |
| `codex_runs/` | `results/archive/codex_runs/` |
| `tmp/` | `results/archive/scratch/` |
| `nextgqa_leakage_analysis/` 核验材料 | `results/analysis/nextgqa_leakage_analysis/`；audit_insufficient.py 改放 scripts/analysis/audit_nextgqa_insufficient.py |
| `.runtime/` | `models/.runtime/`，原位置保留隐藏链接，避免损坏环境内硬编码路径 |
| `MODEL_RESULTS.md` | `results/MODEL_RESULTS.md`，内容不变 |
| `PIPELINE_RESEARCH_HANDOFF.md` | `scripts/docs/PIPELINE_RESEARCH_HANDOFF.md`，内容不变 |
| 清理/迁移清单及 SCRIPT_AUDIT.md | `scripts/docs/maintenance/` |
| `unset_proxy.txt` | `scripts/runtime/unset_proxy.txt` |

当前维护的代码、导入路径、默认数据/结果路径与 Slurm 调用路径已同步更新。旧数据文件内部的历史路径字符串保持原样，避免改写实验记录；历史冻结代码也不批量重写。重跑归档实验时应显式选择新位置输入、核对原协议与输出目录。

新运行的默认结果、日志、媒体构建报告与编译缓存均归入 results；用户显式传入的自定义路径仍由用户决定。

## 保留与删除边界

保留所有模型 JSON/JSONL 详细输出、数据标注、权重、模型环境、人工复核材料和冻结实验记录。没有因为 smoke、retry、tmp 或日期较早就删除一个实验。尤其 tmp/v6tmp2 内确实有实验清单、冻结哈希和核验输出，因此完整归档。

移除的是冗余转发代码、空 runtime_overlays 目录和根目录可重新生成的 TorchInductor 缓存。迁移前源代码和待清理内容均保存到项目同级隐藏备份目录 `.Vid_Evi_QA_cleanup_20261107/`，不在清洁根目录里再次堆备份。

该备份目录还保存 `before_json.jsonl`、迁移映射、事务日志和迁移后核验结果。检查 JSON 的数量、大小和 SHA-256，不仅检查文件名是否存在。已有 results 下的实验目录没有重新命名，新增分类主要接收原来散落于根目录的内容。

## 验证范围

CPU 单元测试、旧评测入口/冻结包导出、模型服务模拟、所有 shell/Slurm 语法和默认输出路径检查。没有启动真实模型、调用收费 API 或提交 Slurm 作业。本次结果整理没有重新评测或重新计分。

## Git 留档范围

本分支保存代码、配置和必要说明。上述目录映射描述服务器布局；数据、模型权重、结果、研究交接材料及服务器历史 README 不上传公开仓库。
