# Vid_Evi_QA

项目按原始数据、模型、构建代码、运行代码、结果分开存放。

| 目录 | 内容 |
|---|---|
| `source_datasets/` | 原始数据及现有视频输入；本轮未清理其内容 |
| `models/` | 模型权重和 `.runtime/` 专用运行环境 |
| `pipelines/` | 数据构建、筛选、证据核验和 benchmark 组装代码 |
| `scripts/` | 评测入口、实现库、Slurm、配置、测试、文档 |
| `results/` | 详细模型返回、统计结果、构建产物、报告、日志及历史实验 |

`.git/`、`.gitignore` 是版本管理文件；隐藏的 `.runtime` 链接指向 `models/.runtime`，保留已有环境的绝对路径兼容性。

## 常用命令

在项目根目录运行：

```bash
python scripts/jobs/submit.py list
python scripts/jobs/submit.py eval sufficient --model qwen25
python scripts/jobs/submit.py eval merged --model internvl --manifest results/reports/final_benchmark_aligned/final/all_condition_records.jsonl
python -m pytest -q scripts/tests -p no:cacheprovider
```

作业入口默认只预览；加 `--submit` 才调用 Slurm，加 `--run-local` 才直接运行。当前 Partial 视频缺失的限制仍然存在，按需图片加噪尚未实现。

- [目录整理与旧路径对照](scripts/docs/REPOSITORY_LAYOUT.md)
- [运行与 Slurm](scripts/docs/JOB_LAYOUT.md)
- [数据构建](pipelines/README.md)
- [评测实现](scripts/src/vid_evi_qa/README.md)

模型返回的详细 JSON/JSONL 保留原文件内容，不以汇总表替代。历史冻结包保留原始内容；其中记录的旧绝对路径应结合迁移对照理解，不能假定归档脚本直接执行即可复现实验。

此 Git 分支仅保存代码、配置及必要说明；原始数据、模型权重、运行环境和实验结果保留在服务器。模型下载与启动辅助脚本仍纳入版本管理。
