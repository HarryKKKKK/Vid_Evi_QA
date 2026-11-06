# 评测实现

| 目录 | 内容 |
|---|---|
| evaluation/ | 主评测、Conflict 评测、GPT API 执行、结果验证与合并、schema 和 smoke gate |
| adapters/ | OpenAI-compatible、GLM、Molmo、VideoLLaMA3、OOM 实验和 prompt-equivalence 适配，以及现有模型注册表 |
| prompts/ | 现有 v2.2 prompt、schema、指纹和响应检查；协议未修改 |
| metrics/ | answerable、unanswerable、Partial、sufficient/insufficient 对比及覆盖检查 |
| media/ | 视频抽帧检查；主评测内已有的媒体处理函数暂未拆分 |
| common/ | 导出独立冻结代码所需的 copy_script 与 resolve_script |

从项目根目录使用原环境：

```bash
python scripts/src/vid_evi_qa/evaluation/evaluate.py --help
python scripts/src/vid_evi_qa/metrics/partial_metrics.py --help
PYTHONPATH=src python -m vid_evi_qa.evaluation.evaluate --help
```

这是保留完整源码目录的重构，不是独立 wheel 发布。默认路径、模型专用参数、prompt、指标、重试和断点续跑规则保持原样。部分适配器导入时会加载模型，不应批量 import 所有模块做健康检查。

旧 Python 路径转发到唯一维护的实现。为保留现有导入、测试 monkeypatch 和冻结副本行为，兼容入口在原模块命名空间执行实现；实现使用旧逻辑路径计算原有数据默认位置。冻结副本使用自身位置，不依赖本目录。

冻结实验包时使用 copy_script，不能直接复制旧兼容入口。已调整 manage_v2_release.py、open_v2/prepare.py 与 prepare_frontier.py；历史冻结目录未改动。
