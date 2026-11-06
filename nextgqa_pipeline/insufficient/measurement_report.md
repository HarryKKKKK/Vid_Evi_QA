# NExT-GQA 候选窗口测量报告

> 注意：窗口阳性表示视觉内容提高了正确答案分数，不等于该窗口包含问题语义对齐的充分 evidence。

## 1. 数据完整性

| 指标 | 数值 |
|---|---:|
| 唯一 item | 5537 |
| 物理 JSONL 记录 | 5537 |
| 重试/重复记录 | 0 |
| status=discarded_invalid_gold | 1 |
| status=needs_review | 1 |
| status=ok | 5535 |
| 视频时长中位数 | 34.0 s |
| 视频时长 P95 / 最大值 | 90.0 / 180.0 s |

## 2. 构造模型表现

模型：`qwen3vl_construct`

| 指标 | 数值 |
|---|---:|
| full accuracy | 83.96% |
| full blind accuracy | 54.89% |
| gold-only accuracy | 82.31% |
| gold-only blind accuracy | 54.60% |
| model-item eligible | 3780 / 5535 (68.29%) |
| single_interval eligible | 3326 / 4949 (67.21%) |
| multi_interval eligible | 454 / 587 (77.34%) |

## 3. 多尺度正对照与候选率

| 尺度 | 正对照窗口 / item | 阈值 | 正对照召回 | seed候选 / 总窗口 | 候选率 | 候选预测gold率 |
|---|---:|---:|---:|---:|---:|---:|
| w4_s2 | 945 / 913 | 0.399 | 95.03% | 36124 / 72891 | 49.56% | 96.55% |
| w8_s4 | 2573 / 2203 | 0.511 | 95.03% | 18007 / 35534 | 50.68% | 98.24% |
| w16_s8 | 3733 / 3004 | 0.635 | 95.02% | 8593 / 16888 | 50.88% | 99.07% |

## 4. 旧遮挡规则的候选覆盖预测

| 指标 | 数值 |
|---|---:|
| eligible item | 3780 |
| 至少一个候选的 item | 3422 (90.53%) |
| uncertainty前候选窗口 | 65586 |
| 最小阳性叶窗口 | 37871 |
| 官方 mask 覆盖率中位数 | 24.21% |
| 新增覆盖率中位数 | 52.43% |
| 预计最终覆盖率中位数 | 90.00% |
| 超过最大覆盖率的 item | 2941 (77.80%) |
| 覆盖率至少80%的 item | 2374 (62.80%) |

## 5. 归一化稳定性

| eligible full gain 小于 | item数 |
|---|---:|
| 0.001 | 3 |
| 0.01 | 5 |
| 0.1 | 13 |
| 0.5 | 51 |
| 1.0 | 102 |
| 2.0 | 248 |

## 6. 级联单调性诊断

| 子尺度 → 父尺度 | 子阳性、父阴性 | 父阳性、全部子阴性 |
|---|---:|---:|
| qwen3vl_construct/w4_s2->w8_s4 | 11.01% | 0.78% |
| qwen3vl_construct/w8_s4->w16_s8 | 12.91% | 1.29% |

## 7. 结论

- 全量抽帧、matched blind、A–E logprob 和多尺度扫描已工程跑通。
- 当前分数适合作为高召回候选生成器，不适合作为最终 evidence 判定器。
- 大约一半 seed 窗口会超过阈值，旧规则会造成严重覆盖膨胀。
- 下一阶段必须使用问题语义 verifier 区分 `sufficient_evidence` 与 `answer_correlated_only`。
- 在 semantic verifier 完成前，不应运行旧 `finalize`。

## 8. 异常记录

- `3842638015_q4`：discarded_invalid_gold；answer text does not map uniquely to one A-E option
- `3167311763_q0`：needs_review；ValueError: missing A-E logprobs for ['E']; observed=[{'token': 'C', 'logprob': -1.0728830375228426e-06}, {'token': 'The', 'logprob': -13.750000953674316}, {'token': 'D', 'logprob': -17.375001907348633}, {'token': 'Based', 'logprob': -18.000001907348633}, {'token': 'B', 'logprob': -18.375001907348633}, {'token': 'A', 'logprob': -18.500001907348633}, {'token': 'There', 'logprob': -19.250001907348633}, {'token': 'Since', 'logprob': -19.375001907348633}, {'token': 'This', 'logprob': -20.625001907348633}, {'token': 'In', 'logprob': -20.750001907348633}, {'token': 'We', 'logprob': -21.500001907348633}, {'token': 'Although', 'logprob': -21.875001907348633}, {'token': 'F', 'logprob': -22.000001907348633}, {'token': 'From', 'logprob': -22.125001907348633}, {'token': 'To', 'logprob': -22.250001907348633}, {'token': 'After', 'logprob': -22.500001907348633}, {'token': '_C', 'logprob': -22.625001907348633}, {'token': 'When', 'logprob': -22.875001907348633}, {'token': 'Given', 'logprob': -22.875001907348633}, {'token': 'I', 'logprob': -23.125001907348633}]
