# 数据构建管线

这里是数据构建代码的维护位置。数据文件和生成产物仍使用原来的目录；迁移没有改变筛选规则、阈值、标注、视频处理参数或输出格式。

| 目录 | 保留功能 |
|---|---|
| nextgqa/filter_download_check/ | 原始标注筛选与 schema 检查 |
| nextgqa/insufficient/ | 证据定位、集成判定、泄漏检测、校准、无泄漏评测清单、冻结视频和质量报告 |
| etbench/conflict/ | 视觉对比任务、候选配对、scene/action 构建、核验、恢复、合并与最终数据生成 |
| benchmark/ | language-prior 筛选、Conflict 干扰项构建与核验、人工修订、对齐和发布 |
| media/ | 原有冻结视频与 Partial 视频生成实现 |

从项目根目录使用原环境，例如：

```bash
python pipelines/nextgqa/filter_download_check/filter_nextgqa.py --help
python pipelines/etbench/conflict/build_visual_contrast_jobs.py --help
python pipelines/media/build_partial.py --help
```

实际运行前检查输入路径。apply、finalize、publish 和视频生成命令会写入产物，不是只读检查。

旧 nextgqa_pipeline/、etbench_pipeline/、scripts/analysis/ 和 scripts/vqa/ 路径保留 Python 兼容入口。已有 Slurm 脚本继续调用旧路径；维护实现时编辑这里的文件。数据、报告和 Slurm 提交脚本本轮不搬迁。

Partial 预生成视频已清理。生成代码保留，但固定 seed 的运行时逐帧加噪尚未实现，旧评测不能直接读取已清空的视频目录。

完整映射见根目录 script_migration.json 和 docs/SCRIPT_LAYOUT.md。
