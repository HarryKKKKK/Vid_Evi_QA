# Vid_Evi_QA — CG-Bench 证据依赖性评测 Pipeline


## 1. 这是做什么的

围绕 CG-Bench 数据集，构造三种"证据条件"下的视频变体，测试 VLM（当前用 Qwen3-VL）
在证据充分/不足/部分缺失时的问答（QA）与可答性判断（classify: ANSWERABLE /
PARTIALLY_ANSWERABLE / UNANSWERABLE）表现：

| 条件 | 含义 | 视频处理方式 |
|---|---|---|
| C1 sufficient | 证据完整 | 原始视频，不做任何处理 |
| C3 insufficient | 证据完全缺失 | 证据区间被冻结成静止帧（或早期版本：涂黑），其余帧/时长不变 |
| C2 partial | 证据部分缺失（dose-response） | 证据区间按 alpha∈[0,1] 做噪声/模糊渐变退化 |
| Hallucination | CG-Bench 自带的幻觉测试子集 | 原始视频，不做处理 |

## 2. 环境

| 用途 | 路径 | 依据 |
|---|---|---|
| 数据构造用 conda 环境（ffmpeg，CPU） | `/aifs4su/hansirui_2nd/harry/envs` | `build_partial_cgbench.sh` / `build_freeze_cgbench.sh` / `build_insufficient_cgbench.sh` 均 `conda activate .../envs` |
| vLLM 推理用 conda 环境（GPU） | `/aifs4su/hansirui_2nd/harry/envs_qwen3vl` | `scripts/eval.sh` `conda activate .../envs_qwen3vl` |
| 代理设置 | `unset_proxy.txt` | 集群内网跑 vLLM/下载数据前 `source` 一下，清掉 http(s)_proxy 等，sbatch 脚本里也各自 `unset` 了一遍 |

两个环境分工明确：`envs` 只用于 ffmpeg 视频构造的 CPU sbatch 作业；`envs_qwen3vl`
只用于起 vLLM server 做推理，两者都在用，没有废弃的。

## 3. 目录结构与数据流

```
dataset/
├── cgbench_pipeline/          # 数据构造：原始数据筛选 + C2/C3 视频生成脚本
│   ├── filter_download_check/ # 阶段0：从 HF 下载/解压/筛选原始 CG-Bench
│   ├── cgbench_filtered.json  # 筛选后的 anchor 元数据（阶段0 的产物，阶段1-3 的输入）
│   ├── build_insufficient.py  # C3 旧版本：涂黑证据区间（见下方"两套 C3 脚本"）
│   ├── build_freeze.py        # C3 当前版本：冻结证据区间为静止帧（evaluate.py 实际用这个）
│   ├── build_partial.py       # C2：证据区间按 alpha 做噪声/模糊 dose-response 退化
│   ├── build_insufficient_cgbench.sh / build_freeze_cgbench.sh / build_partial_cgbench.sh
│   │                           # 对应的 SLURM array 提交脚本
│   └── __pycache__/
├── source_datasets/cg_bench/
│   ├── CG-Bench/               # HF 原始克隆（cgbench.json 等）
│   ├── videos/                 # C1，原始长视频 {video_id}.mp4
│   ├── freeze_videos/          # C3，冻结版 {video_id}_q{qid}_freeze.mp4
│   ├── insufficient_videos/    # C3 旧版（涂黑），build_insufficient.py 的输出目录，当前未在 evaluate.py 中使用
│   └── partial_videos/         # C2，{video_id}_q{qid}_c2_{noise|blur}_a{NNN}.mp4
├── scripts/
│   ├── evaluate.py             # 主评测脚本，起 OpenAI 兼容client 打 vLLM server
│   ├── eval.sh                 # sbatch：起 vLLM server + 跑 evaluate.py
│   ├── compare_suff_insuff.py  # 生成 C2 候选集 suff_correct_insuff_wrong.json
│   ├── answerable_metrics.py / unanswerable_metrics.py / partial_metrics.py
│   │                           # 结果分析脚本
│   └── tmp/                    # 临时/调试脚本（见下）
└── cgbench_result/             # 所有评测结果 + 中间产物 json/jsonl
```

## 4. 数据构造 Pipeline（按执行顺序）

### 阶段 0：原始数据筛选（`cgbench_pipeline/filter_download_check/`）

- `unzip_cgbench_videos.py` — 解压 HF 下载的视频压缩包
- `cgbench_check_exist.py` — 核对视频文件是否齐全
- `cleanup_cgbench_videos.py` — 清理多余/损坏文件
- `filter_cgbench.py` — 从 `source_datasets/cg_bench/CG-Bench/cgbench.json` 读原始数据，
  按 `sub_category` 是否含 "hallucination" 关键词拆出 Hallucination 子集，解析
  `evidence_intervals`，写出 `cgbench_pipeline/cgbench_filtered.json` +
  `cgbench_filter_summary.txt`

`cgbench_filtered.json` 当前共 **1596** 条，其中 `evidence_condition == "sufficient"`
**1152** 条，`== "Hallucination"` **444** 条

### 阶段 1：C3 insufficient 视频

- `build_freeze.py`
  - 把每个 question 的 evidence_intervals 替换成"证据开始前一帧"（或
    `--freeze-source first` 时用区间第一帧）的静止画面，用 ffmpeg overlay
    叠加实现，音频区间内默认静音（`--keep-audio` 可保留）
  - 输出：`source_datasets/cg_bench/freeze_videos/{video_id}_q{qid}_freeze.mp4`
  - 提交：`NUM_SHARDS=8 sbatch --array=0-7%8 cgbench_pipeline/build_freeze_cgbench.sh`
  - `DEFAULT_JSON` 已改回跑全量 `cgbench_filtered.json`（1596 条）。此前临时指向
    `cgbench_filtered.broken_subset.json`（针对一批损坏视频的补跑子集），代码里
    保留了一行注释掉的引用，日常使用不需要它。

### 阶段 2：候选集筛选（`scripts/compare_suff_insuff.py`）

对比 `sufficient.classify.json` 与 `insufficient.classify.json` 里每题模型是否
答对，筛出"sufficient 答对 且 insufficient 答错"的 (video_id, qid) 对，写入
`cgbench_result/suff_correct_insuff_wrong.json`（当前 **245** 条，与筛选逻辑一致：
这批题目是模型证据依赖性最明确的，适合拿来做 C2 dose-response）。

### 阶段 3：C2 partial 视频（`cgbench_pipeline/build_partial.py`）

- 输入：阶段2 产出的 245 条候选 × `cgbench_filtered.json` 里 join 出完整元数据
- 对每题 × 每个 alpha 值（默认 `0.0,0.05,0.1,0.15,0.2,0.3,0.5`）生成一条视频：
  证据区间内按 `alpha` 与原始帧线性混合噪声/模糊，`alpha=1` 等价 C1，`alpha=0`
  等价 C3（区间完全不可辨认，用的是 blend 而非旧版加性 noise，保证与画面明暗
  无关地做到彻底不可辨认）
- 输出命名：`{video_id}_q{qid}_c2_{noise|blur}_a{NNN}.mp4`
  （`NNN = round(alpha*100):03d`，如 alpha=0.05 → `a005`）
- 提交：`NUM_SHARDS=8 sbatch --array=0-7%8 cgbench_pipeline/build_partial_cgbench.sh`
 

三套构造脚本共用同一套并行/分片/断点续跑约定：`--jobs`（进程内并发 ffmpeg 数）×
`--threads`（每个 ffmpeg 的 libx264 线程数）应 ≤ `--cpus-per-task`；
`--num-shards`/`--shard`（或 `SLURM_ARRAY_TASK_ID`）做 round-robin 分片；
已存在的输出文件默认跳过，除非传 `--overwrite`。

## 5. Evaluation（`scripts/evaluate.py` + `scripts/eval.sh`）

- `evaluate.py` 通过 OpenAI 兼容 client 打本地 vLLM server（`VLLM_BASE_URL`，
  默认 `http://localhost:8000/v1`），对每条样本抽帧（默认 32 帧，`--sampling
  evidence` 模式下会按 `--evidence-fraction` 把更多帧集中采在证据区间内）
  → 拼 prompt → 让模型输出 JSON（`--task qa` 只要求选项；`--task classify`
  要求先判断 ANSWERABLE/UNANSWERABLE 再选）。
- `--mode {sufficient|hallucination|insufficient|partial|all}` 控制跑哪个证据
  条件；`partial` 会把候选集按 `--alphas` 展开成多个任务，请求量成倍增长，
  测试时建议先加 `--limit`。
- 断点续跑：先读已有的 `.jsonl`，按 key 跳过已完成条目
  （sufficient/hallucination/insufficient 用 `(video_id, qid)`，partial 用
  `(video_id, qid, alpha)`，因为同一题多个 alpha 不能被同一个 key 合并）。
  每条结果实时 append 到 `.jsonl`，跑完再落一份有序的 `.json`；跳过/报错的
  条目单独存 `.skipped.json(l)`。
- `eval.sh`：起 2 卡 tensor-parallel 的 vLLM server（`models/Qwen3-VL`，即
  **Qwen3-VL-32B**，`--max-model-len 32768`），等 server ready 后跑
  `evaluate.py`。**当前脚本里实际执行的命令**是 NExT-GQA sufficient/qa 测试跑
  （`--filtered-json nextgqa_pipeline/nextgqa_filtered.json --video-dir
  source_datasets/next_gqa/videos --results-dir nextgqa_result`），不再是早期
  CG-Bench 的 `--mode partial --task classify` 那一行——`evaluate.py` 通过
  `--filtered-json`/`--video-dir`/`--results-dir`/`--video-id-mapping` 四个
  CLI 参数支持切换数据集（不传时默认值就是 CG-Bench 的路径，`eval.sh` 不改
  这几个参数就还是跑 CG-Bench）；脚本里 `LIMIT_ARGS` 变量留了一个空位，方便
  跑全量前先设成 `"--limit 5"` 做小范围验证。
- `evaluate.py` 的视频路径解析按每条样本的 `source_dataset` 字段区分：
  `"cgbench"` 走原来的扁平 `{video_id}.mp4`，`"nextgqa"` 走
  `--video-id-mapping`（默认指向 `map_vid_vidorID.json`）解析出的
  `{folder}/{vidorID}.mp4` 嵌套路径。

结果落盘规则：`{results-dir}/{sufficient|hallucination|insufficient|partial}.
{qa|classify}.json`（同名 `.jsonl` 是增量版本，`.skipped.json(l)` 是跳过/报错记录）；
`--results-dir` 默认是 `cgbench_result/`，NExT-GQA 跑的是 `nextgqa_result/`，两边
不会互相覆盖。

> **NOTE（数据质量注意点）**：`--sampling uniform`（当前 `eval.sh` 实际用的
> 默认值）是在整段视频时长上均匀取 `--num-frames` 个时间戳，**不保证**每个
> 时间戳都落在该题的 `evidence_intervals` 内。如果某题的证据区间很短、或者
> 正好落在两个均匀采样点之间，模型实际拿到的帧里可能一帧证据画面都没有——
> 这对 C1/C3 的答题准确率统计是噪声，对 C2 dose-response 更是问题：如果被
> 采样到的帧根本不在做了噪声/模糊处理的区间内，这条 alpha 曲线上的这个点就
> 完全测不出退化效果。用 `scripts/uniform_coverage_check.py`（见第 6、7 节）
> 实测：128 帧 uniform 采样下，`cgbench_filtered.json` 里 **1596 条里有 315
> 条（19.74%）完全踩不中 evidence 区间**——分析 uniform-sampling 结果时应该
> 用 `cgbench_result/uniform_coverage_report.txt` 里列出的 (video_id, qid)
> 把这些题剔除掉，否则统计出来的 success rate / dose-response 曲线会被这批
> "模型其实根本没看到证据"的题目稀释。（另一种规避方式是改用
> `--sampling evidence`，它通过 `--evidence-fraction` 强制保证证据区间内一定
> 有采样点，从设计上不会出现这个问题，但目前实际跑的 `eval.sh` 用的是
> `uniform`。）

## 6. 结果分析脚本（`scripts/`）

- `compare_suff_insuff.py` — 见上方阶段2
- `answerable_metrics.py` — 按 `evidence_condition` 分组统计 success
  rate / evidence hit rate / unanswerable rate 等（吃 sufficient/insufficient
  的 classify 结果）
- `unanswerable_metrics.py` — 专门统计 insufficient 条件下模型的拒答率
  （该拒答却误答的比例、误答里蒙对/蒙错的比例）
- `partial_metrics.py` — 按 alpha 分组统计 C2 dose-response 曲线（success
  rate / evidence hit rate 随 alpha 的变化），支持 CSV 导出
- `uniform_coverage_check.py` — 检测 uniform sampling 是否踩中每题的
  `evidence_intervals`（见第 5 节 NOTE）。直接用 `cgbench_filtered.json`
  里自带的 `video_duration` 字段按 `evaluate.py::plan_timestamps()` 的
  uniform 公式重算时间戳，不需要真实视频文件，本地就能跑：
  `python scripts/uniform_coverage_check.py --num-frames 128`。
  踩不中 evidence 区间的 (video_id, qid) 会写到
  `cgbench_result/uniform_coverage_report.txt`（tab 分隔，一行一条）。

## 7. 当前实际存在的结果文件（`cgbench_result/`）

| 文件 | 说明 |
|---|---|
| `sufficient.qa.json(l)` | C1 QA 任务结果 |
| `sufficient.classify.json(l)` | C1 分类任务结果 |
| `insufficient.classify.json(l)` | C3（freeze）分类任务结果 |
| `suff_correct_insuff_wrong.json` | 245 条 C2 候选集 |
| `partial.classify.json(l)` / `partial.classify_0.json` | C2 分类任务结果（`_0` 大概率是某次未完整跑完/被覆盖前的快照，建议跑分析前确认用哪份） |
| `uniform_coverage_report.txt` | 128 帧 uniform 采样踩不中 evidence 区间的 (video_id, qid) 列表，共 315 条（占 1596 条的 19.74%）。由 `scripts/uniform_coverage_check.py` 生成，见第 5 节 NOTE 和第 6 节。旧的 `uniform_coverage_report.json` 及其生成脚本已不在仓库里，这份 `.txt` 是重新生成的替代版本。 |

## 8. NExT-GQA Data Preparation（阶段0：数据筛选 + 原始视频准备）

与 `cgbench_pipeline/` 平行的第二个数据源，目前只实现了阶段0（sufficient anchor
筛选 + 视频准备），C2/C3/freeze/inference/evaluation 等后续阶段尚未开始。完整说明、
字段含义、筛选规则、每个脚本的输入输出和运行顺序见 **[nextgqa_pipeline/README.md](nextgqa_pipeline/README.md)**。

```
nextgqa_pipeline/
└── filter_download_check/
    ├── inspect_nextgqa.py              # 只读检查 annotation schema
    ├── filter_nextgqa.py               # 筛选 sufficient anchors，输出 schema 对齐 cgbench_filtered.json
    ├── build_nextgqa_video_manifest.py # 按 video_id 去重生成视频 manifest
    ├── download_nextgqa_videos.py      # 从已有视频目录整理 / 按自备 URL 列表下载
    └── check_nextgqa_videos.py         # 检查本地视频是否存在、是否损坏
```

要点：

- Annotation 已在仓库中：`source_datasets/next_gqa/NExT-GQA/datasets/nextgqa/`
  （`test.csv` + `gsub_test.json` + `map_vid_vidorID.json`）；原始视频唯一的官方来源
  是一个 Google Drive 文件，`download_nextgqa_videos.py --fetch-official-archive`
  可以用 `gdown` 包自动下载+解压（尽力而为，非稳定 API，失败时需要手动下载后用
  `--source-video-dir` 整理），准备好后放到 `source_datasets/next_gqa/videos/`。
- 输出 `nextgqa_pipeline/nextgqa_filtered.json` 的字段集合与
  `cgbench_pipeline/cgbench_filtered.json` 严格一致（脚本内建 schema 校验），
  `question_type` 保留 NExT-GQA 原始值 `TN/TC/TP/CW/CH`，本阶段所有样本
  `evidence_condition` 固定为 `"sufficient"`。
- 视频存在性检查（`check_nextgqa_videos.py`）与 annotation 筛选结果彼此独立：
  视频缺失不会从 `nextgqa_filtered.json` 里删除样本。
