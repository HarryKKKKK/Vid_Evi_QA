# NExT-GQA Data Preparation

本目录实现 NExT-GQA（test split）**sufficient anchor** 数据的筛选与原始视频准备，
风格和产物 schema 都以 `cgbench_pipeline/`（见仓库根 [README.md](../README.md)）为模板。

本阶段只做筛选 + 视频准备，不做 C2 partial / C3 insufficient / freeze video /
evidence removal / 模型 inference / answerability evaluation / grounding
metrics / calibration / confidence / partial metrics —— 这些留待后续阶段。

## 1. 目录结构

```
nextgqa_pipeline/
├── README.md                          # 本文件
├── filter_download_check/
│   ├── inspect_nextgqa.py             # 阶段0a：只读检查 annotation schema
│   ├── filter_nextgqa.py              # 阶段0b：筛选 sufficient anchors
│   ├── build_nextgqa_video_manifest.py# 阶段0c：生成去重后的视频 manifest
│   ├── download_nextgqa_videos.py     # 阶段0d：准备本地视频（copy/symlink/下载）
│   └── check_nextgqa_videos.py        # 阶段0e：核对本地视频是否存在/完整
├── nextgqa_filtered.json              # (运行后生成) 筛选结果，schema 对齐 cgbench_filtered.json
├── nextgqa_filter_summary.json/.txt   # (运行后生成) 筛选漏斗统计
├── nextgqa_rejected.json              # (运行后生成) 被拒绝样本及原因
├── nextgqa_video_manifest.json/.txt   # (运行后生成) 去重视频清单
├── nextgqa_download_report.json       # (运行后生成) 视频准备结果报告
└── nextgqa_video_check.json/.txt      # (运行后生成) 本地视频存在性/完整性检查结果
```

以上标注"运行后生成"的文件目前均不存在——本阶段只交付代码，没有执行任何脚本，
详见文末"尚未执行的操作"。

## 2. Annotation 与视频的实际存放位置

仓库里已经存在官方 NExT-GQA 仓库的克隆（`source_datasets/next_gqa/NExT-GQA/`），
其中的 annotation 文件已经就位，**不需要**另外准备：

```
source_datasets/next_gqa/NExT-GQA/datasets/nextgqa/
├── test.csv               # QA：video_id,frame_count,width,height,question,answer,qid,type,a0..a4
├── gsub_test.json          # 时间片段 grounding：{video_id: {duration, location: {qid: [[s,e],...]}, fps}}
├── map_vid_vidorID.json    # video_id -> "{folder}/{vidorID}"（VidOR 相对路径，不含扩展名）
├── frame2time_test.json    # 本阶段未使用
└── README.md               # 官方字段说明
```

**注意**：任务描述里建议的 `source_datasets/next_gqa/annotations/` 路径在当前仓库中
并不存在；所有脚本的 `--annotation-dir` 默认值改为指向上面这个真实存在的路径。
如果你想使用建议的目录名，可以自己建一个 symlink/copy，再用 `--annotation-dir` 覆盖。

视频原始文件本仓库中**不存在**，需要你按第 7 节（视频准备）手动准备到：

```
source_datasets/next_gqa/videos/          # 目标目录（脚本会自动创建）
└── {folder}/{vidorID}.mp4                # 与 map_vid_vidorID.json 给出的相对路径一致
```

## 3. 运行顺序总览

```
inspect_nextgqa.py  →  filter_nextgqa.py  →  build_nextgqa_video_manifest.py
      →  download_nextgqa_videos.py (或手动准备)  →  check_nextgqa_videos.py
```

以下每一节给出对应脚本的输入、输出、字段行为和手动运行命令。**这些命令均未被
agent 执行过**，需要你自己在终端里跑。

## 4. 阶段 0a：`inspect_nextgqa.py`（只读检查）

输入：`test.csv` + `gsub_test.json` + `map_vid_vidorID.json`。
不写任何输出文件，只在 stdout 打印：找到的文件、CSV 列名和样例、原始 `type`
分布、qid/video_id 的类型和样例、answer/options 结构、grounding JSON 结构、
一个多 interval 样例、mapping JSON 样例、以及 CSV↔grounding↔mapping 的 join
覆盖率统计（missing grounding / missing mapping / answer 无法唯一匹配 / 重复行数）。

```bash
python nextgqa_pipeline/filter_download_check/inspect_nextgqa.py \
  --annotation-dir source_datasets/next_gqa/NExT-GQA/datasets/nextgqa
```

## 5. 阶段 0b：`filter_nextgqa.py`（筛选）

```bash
python nextgqa_pipeline/filter_download_check/filter_nextgqa.py \
  --annotation-dir source_datasets/next_gqa/NExT-GQA/datasets/nextgqa \
  --cgbench-template cgbench_pipeline/cgbench_filtered.json \
  --output nextgqa_pipeline/nextgqa_filtered.json \
  --summary-json nextgqa_pipeline/nextgqa_filter_summary.json \
  --summary-txt nextgqa_pipeline/nextgqa_filter_summary.txt \
  --rejected-json nextgqa_pipeline/nextgqa_rejected.json
```

### 5.1 输出 schema（与 `cgbench_pipeline/cgbench_filtered.json` 完全一致的 key 集合）

```json
{
  "video_id": "2574374895",
  "qid": "8",
  "source_dataset": "nextgqa",
  "question": "what did the baby do after throwing the green cup away while on the floor near the end",
  "choices": ["clap proudly", "the lady sitting down", "lay on floor", "just picked it up", "crawl"],
  "answer": "C",
  "evidence_intervals": [{"start": 1.2, "end": 5.8, "description": ""}, {"start": 12.1, "end": 17.1, "description": ""}],
  "evidence_count": 2,
  "video_duration": 34.0,
  "question_type": "TN",
  "source_task": "TN",
  "evidence_condition": "sufficient",
  "expected_behavior": "answer"
}
```

`filter_nextgqa.py` 在写出前会用 `--cgbench-template` 指向的文件做**严格 key 集合比对**
（`validate_schema()`），多一个字段或少一个字段都会直接抛错，不会静默写出不一致的
JSON。上面这个例子里的数值是示意，不是本次运行的真实结果。

字段来源说明：

| 字段 | 来源 |
|---|---|
| `video_id` | `test.csv` 原样列（清理空白后的字符串） |
| `qid` | `test.csv` 原样列，`clean_text()` 清理后**保存为字符串**（不再转成 `int`）——只要 `(video_id, qid)` 能唯一标识样本即可，见 5.2 |
| `question` / `choices`(=a0..a4) | `test.csv`，`strip()` + 合并连续空白 |
| `answer` | 见 5.5，**不是**筛选条件，只是格式转换 |
| `evidence_intervals` / `evidence_count` | `gsub_test.json[video_id]["location"][qid]`，见 5.4 |
| `video_duration` | `gsub_test.json[video_id]["duration"]` |
| `question_type` | `test.csv` 的原始 `type` 列（任意值，不限于 `TN/TC/TP/CW/CH`），**不做**人类可读转换，也**不作为筛选条件**，见 5.3 |
| `source_task` | NExT-GQA 没有比 `type` 更细的分类体系（不像 CG-Bench 有 `sub_category`），因此这里直接复用 `question_type` 的值 |
| `evidence_condition` | 本阶段固定写死 `"sufficient"` |
| `expected_behavior` | 固定写 `"answer"`（对齐 CG-Bench sufficient 样本的约定） |

### 5.2 qid：保存为清理后的字符串，不再要求可转 int

`qid` 不再用 `int()` 解析、也不会因为不是纯数字而被拒绝（旧版本的
`invalid_qid` rejection reason 已删除）。判重和 join 全部基于
`(video_id, qid)` 这对经过 `clean_text()` 清理的字符串。这与
`cgbench_filtered.json` 里 `qid` 是无引号整数（如 `"qid": 12`）不同——这是
为了适配 NExT-GQA `qid` 语义（仅需要能唯一标识问题）而做的、`validate_schema()`
不检查的字段**值类型**差异（只检查 key 集合，不检查每个字段的 value 类型）。

### 5.3 question type：不筛选，原样保留

`question_type` 字段值就是 `test.csv` 里的原始 `type` 列。**不再**限制在
`TN/TC/TP/CW/CH` 范围内（旧版本的 `ALLOWED_TYPES` 白名单和
`type_not_allowed` rejection reason 已删除）——`test.csv` 里出现的任何
`type` 值都会被保留并原样写入 `question_type`，不会被转换成
`Temporal`/`Causal` 这类可读标签，也不会新增 `question_group` 字段。
`type` 只用于 `nextgqa_filter_summary.txt` 里的统计展示，不影响是否保留样本。

### 5.4 evidence intervals 处理逻辑（`normalize_intervals()`）

1. start/end 转 `float`
2. `end < start` 时交换
3. clip 到 `[0, video_duration]`
4. 丢弃裁剪后长度 `<= 0` 的 interval
5. 按 `start` 排序
6. 合并重叠/相接的 interval（`next.start <= prev.end`），中间有正数 gap 的不合并
7. 输出格式 `{"start": float, "end": float, "description": ""}`，与 CG-Bench 一致

多个独立 evidence interval（如 `[10,15]` 和 `[16,20]`）会保留为两条，不会被压缩成
`[10,20]`；但 `[10,15]` 和 `[14,20]` 首尾重叠/相接会合并成 `[10,20]`。
标准化后**只要至少剩一个**有效 interval 即可（旧版本要求至少 2 个的
`MIN_EVIDENCE_INTERVALS` / `insufficient_evidence_intervals` 已删除），相邻
interval 之间也**不再**要求最小间隔（旧版本的 `MIN_GAP_SECONDS` /
`gap_too_small` 已删除）。标准化后一个有效 interval 都不剩时，reject reason
为 `no_valid_evidence_intervals`。

### 5.5 answer：格式转换，不是筛选条件

`test.csv` 的 `answer` 列是文本。转换逻辑（`resolve_answer()`）：先用大小写/
空白不敏感的方式与 `a0..a4` 做唯一匹配，匹配成功则转成对应字母
（`a0→A ... a4→E`）；**匹配不到或匹配到多个都不会拒绝该样本**（旧版本的
`answer_not_unique_match` rejection reason 已删除）——转换失败时直接把清理后的
原始 answer 文本写入 `answer` 字段（不猜测字母），并把这条样本记录进
`nextgqa_filter_summary.json` 的 `answer_conversion_issues` 列表（附
`raw_answer` 和 `options`），方便你人工核对数据格式问题，而不是被脚本静默吞掉。

### 5.6 筛选规则（漏斗顺序，见 `run_filter()`）

```
raw_csv_rows
  → after_duplicate_removal        （去掉重复的 (video_id, qid) 行）
  → after_question_valid           （question 非空）
  → after_options_valid            （a0..a4 都非空）
  → after_grounding_matched        （video_id+qid 在 gsub_test.json 里找到 location）
  → after_mapping_matched          （video_id 在 map_vid_vidorID.json 里找到）
  → after_duration_valid           （video_duration > 0）
  → after_valid_interval_filter    （标准化后至少剩 1 个 interval）
  → after_interval_duration_filter （每个 interval 时长 3.0~30.0s，含边界）
  → after_coverage_filter          （union evidence 时长 / video_duration <= 0.60，含边界）
  → final_retained
```

每一行样本的 rejection reason 是**第一个未通过的检查**（互斥、不重复计数），写入
`nextgqa_rejected.json` 和 `nextgqa_filter_summary.json` 的 `rejection_reason_counts`：

```
duplicate_csv_row
empty_question
incomplete_options
missing_grounding_annotation
missing_video_mapping
invalid_video_duration
no_valid_evidence_intervals
interval_duration_out_of_range
coverage_ratio_too_high
```

`question type`（5.3）和 `answer` 匹配（5.5）都**不在**这个列表里——它们不再是
筛选条件。

> **关于历史文档里出现过的 161 条 / Causal 118 / Temporal 43 这类目标值**：
> 那是基于旧版本（含 type 白名单 + answer 唯一匹配 + 至少 2 个 interval + gap
> 约束）筛选规则的预期校验目标，**不适用于本次更新后的筛选规则**——本版本删除
> 了 type/qid-int/answer-match/最少 interval 数/gap 五个筛选条件，实际
> retained 数量会明显更多，且不再需要向那个旧目标值对齐。实际跑出来的数字
> 请以 `nextgqa_filter_summary.txt` 为准。

## 6. 阶段 0c：`build_nextgqa_video_manifest.py`

```bash
python nextgqa_pipeline/filter_download_check/build_nextgqa_video_manifest.py \
  --filtered-json nextgqa_pipeline/nextgqa_filtered.json \
  --annotation-dir source_datasets/next_gqa/NExT-GQA/datasets/nextgqa \
  --output-json nextgqa_pipeline/nextgqa_video_manifest.json \
  --output-txt nextgqa_pipeline/nextgqa_video_manifest.txt
```

按 `video_id` 去重（同一个视频可能对应多个 qid），每条 manifest entry：

```json
{
  "video_id": "2574374895",
  "mapped_video_id": "0004/2574374895",
  "expected_relative_path": "0004/2574374895.mp4",
  "required_by_qids": ["3", "8"],
  "status": "unknown"
}
```

`expected_relative_path` 直接来自 `map_vid_vidorID.json`，**不是**下载 URL——NExT-GQA
官方没有发布逐视频的下载链接（见第 7 节），这个字段只描述"文件应该放在哪个相对路径"。

## 7. 视频准备：`download_nextgqa_videos.py`

### 7.1 官方视频来源只有一个 Google Drive 文件

`source_datasets/next_gqa/NExT-GQA/README.md`（官方仓库自带）里 "Preparation" 一节
只给出了一个 **Google Drive 文件链接**（"raw videos"，
`https://drive.google.com/file/d/1jTcRCrVHS66ckOUfWRb-rXdzJ52XAWQH/view`），NExT-GQA
没有发布逐视频的 URL 列表。脚本**不**内置任何自己编造的下载 URL，**不**做非官方爬虫
或从 YouTube/第三方站点下载——脚本里唯一出现的 URL/文件 ID，就是这一个官方仓库自己
公开链接出来的 Google Drive 文件。

脚本支持三种路径，按优先级：

* **方式 A（推荐，见 7.2）**：`--fetch-official-archive`，用 `gdown` 包自动下载上面
  这个官方 Drive 文件、解压，然后走正常的 copy/symlink/hardlink 流程整理出 filtered
  子集。需要 `pip install gdown`，需要能连到 Google Drive。
* **方式 B（见 7.3）**：`--source-video-dir`，你已经手动下载好一份完整/部分视频目录
  （不管是自己在浏览器里下载解压的，还是已有的 VidOR 数据集拷贝），脚本帮你整理出
  filtered 需要的子集。
* **方式 C（见 7.4）**：`--url-manifest`，你自己准备了一份合法获取的 URL 列表（比如
  你自己转存后的直链），脚本按这份列表做断点续传下载。

三种都不提供时，脚本只打印说明，不做任何事（见 7.5）。

> **`--fetch-official-archive` 的稳定性说明**：Google Drive 对大文件下载有"无法扫描
> 病毒，是否继续"的确认页机制（`gdown` 负责处理这个确认 token），以及未公开的下载
> 配额限制。这条路径是"尽力而为"的便利选项，**不是**稳定保证的 API——如果 Google
> 改了确认流程，或者这个文件的下载配额被用完/权限被官方改掉，`gdown.download()`
> 就会失败（脚本会给出清晰的报错和后续建议），这时候需要退回方式 B：你自己在能打开
> 浏览器的机器上手动下载，传到服务器，再用 `--source-video-dir` 指过去。

### 7.2 方式 A：`--fetch-official-archive`（用 gdown 自动下载官方 Drive 文件）

前提：服务器上要能 `pip install gdown`，并且网络能连 Google Drive。

```bash
pip install gdown
```

先 dry-run（**不会**真的下载，只打印会做什么）：

```bash
python nextgqa_pipeline/filter_download_check/download_nextgqa_videos.py \
  --manifest nextgqa_pipeline/nextgqa_video_manifest.json \
  --video-dir source_datasets/next_gqa/videos \
  --fetch-official-archive \
  --dry-run
```

确认无误后正式跑（会真正下载一个较大的官方压缩包到
`source_datasets/next_gqa/raw_download/`，解压到
`source_datasets/next_gqa/raw_extracted/`，然后从解压结果里 copy/symlink 出
filtered 需要的子集到 `--video-dir`）：

```bash
python nextgqa_pipeline/filter_download_check/download_nextgqa_videos.py \
  --manifest nextgqa_pipeline/nextgqa_video_manifest.json \
  --video-dir source_datasets/next_gqa/videos \
  --fetch-official-archive \
  --copy-mode symlink \
  --workers 8
```

可选参数：
- `--gdrive-file-id`：覆盖默认的官方文件 ID（正常不需要改，除非官方链接换了）
- `--archive-cache-dir` / `--archive-extract-dir`：改下载/解压的缓存位置
- `--overwrite`：即使 `raw_extracted/` 里已经有内容，也重新解压一次

注意事项：
- 这一步会下载**一个完整的官方压缩包**（不是按需只下 filtered 需要的那几个视频），
  下载和解压都可能占用较大的磁盘空间和时间；
- 解压格式脚本用 `shutil.unpack_archive` 自动识别（支持 zip/tar/tar.gz 等常见格式）；
  如果格式无法识别，脚本会报清晰的错误，提示你手动解压后改用 `--source-video-dir`；
- 如果 `--dry-run` 和 `--fetch-official-archive` 一起用，脚本只打印意图、不下载、
  不解压、也不会进入后面 copy/symlink 那一步的 dry-run 展示（因为还没有源文件可供
  展示会 copy 到哪）。

### 7.3 方式 B：从已有完整视频目录整理 filtered 子集

如果你已经有一份 NExT-GQA/VidOR 原始视频（无论是通过 Google Drive 手动下载解压，
还是已有的 VidOR 数据集拷贝）：

```bash
# 先 dry-run 看看会做什么，不会实际写文件
python nextgqa_pipeline/filter_download_check/download_nextgqa_videos.py \
  --manifest nextgqa_pipeline/nextgqa_video_manifest.json \
  --video-dir source_datasets/next_gqa/videos \
  --source-video-dir /path/to/existing/nextgqa_or_vidor_videos \
  --copy-mode symlink \
  --dry-run
```

确认无误后去掉 `--dry-run` 实际执行（`--copy-mode` 可选 `copy`/`symlink`/`hardlink`）：

```bash
python nextgqa_pipeline/filter_download_check/download_nextgqa_videos.py \
  --manifest nextgqa_pipeline/nextgqa_video_manifest.json \
  --video-dir source_datasets/next_gqa/videos \
  --source-video-dir /path/to/existing/nextgqa_or_vidor_videos \
  --copy-mode symlink \
  --workers 8
```

### 7.4 方式 C：按你自己提供的 URL 列表下载（支持断点续传）

```bash
python nextgqa_pipeline/filter_download_check/download_nextgqa_videos.py \
  --manifest nextgqa_pipeline/nextgqa_video_manifest.json \
  --video-dir source_datasets/next_gqa/videos \
  --url-manifest /path/to/your_video_urls.json \
  --workers 4 --retries 3 --timeout 60 \
  --resume
```

`your_video_urls.json` 格式：`{"video_id": "https://...", ...}`（key 也可以是
`mapped_video_id`）。下载用 `.part` 临时文件 + 完成后原子 rename；已存在的正常文件
默认跳过（`--overwrite` 强制重下）；失败的条目记录在
`nextgqa_pipeline/nextgqa_download_report.json` 里，可重跑同一条命令（默认
`--resume`）继续断点续传，不会重复下载已完成的视频。

### 7.5 什么都不提供时

```bash
python nextgqa_pipeline/filter_download_check/download_nextgqa_videos.py \
  --manifest nextgqa_pipeline/nextgqa_video_manifest.json \
  --video-dir source_datasets/next_gqa/videos
```

脚本会打印提示，说明需要三选一：`--fetch-official-archive`（方式 A）、
`--source-video-dir`（方式 B）、或 `--url-manifest`（方式 C）。不会下载任何东西。

## 8. 阶段 0e：`check_nextgqa_videos.py`（本地视频检查）

```bash
python nextgqa_pipeline/filter_download_check/check_nextgqa_videos.py \
  --filtered-json nextgqa_pipeline/nextgqa_filtered.json \
  --annotation-dir source_datasets/next_gqa/NExT-GQA/datasets/nextgqa \
  --video-dir source_datasets/next_gqa/videos \
  --output-json nextgqa_pipeline/nextgqa_video_check.json \
  --output-txt nextgqa_pipeline/nextgqa_video_check.txt
```

加 `--ffprobe` 会额外用 `ffprobe` 检查每个"找到"的视频是否可解码、时长是否与
`gsub_test.json` 里的 `duration` 相符（容差 `--duration-tolerance`，默认 2 秒）：

```bash
python nextgqa_pipeline/filter_download_check/check_nextgqa_videos.py \
  --filtered-json nextgqa_pipeline/nextgqa_filtered.json \
  --annotation-dir source_datasets/next_gqa/NExT-GQA/datasets/nextgqa \
  --video-dir source_datasets/next_gqa/videos \
  --ffprobe \
  --output-json nextgqa_pipeline/nextgqa_video_check.json \
  --output-txt nextgqa_pipeline/nextgqa_video_check.txt
```

路径匹配顺序：① `map_vid_vidorID.json` 给出的精确相对路径 → ② 视频目录下任意
位置的完整文件名匹配 → ③ 忽略扩展名的 stem 精确匹配 → ④ stem 模糊（子串）匹配。
每一层如果匹配到 2 个及以上候选文件，直接标记 `ambiguous`，不会继续往下一层
"猜"。状态包括 `found` / `missing` / `ambiguous` / `corrupt` / `duration_mismatch`。

**视频缺失不会导致 `nextgqa_filtered.json` 里的样本被删除**——两份 JSON 语义分开：
`nextgqa_filtered.json` 是"哪些样本满足 annotation 筛选规则"，
`nextgqa_video_check.json` 是"这些样本需要的视频本地是否可用"。即使视频缺失，
样本的 `question_type` 和 `evidence_condition` 也保持不变。

## 9. 测试与语法检查

```bash
pytest -q tests/test_nextgqa_data_pipeline.py
```

```bash
python -m py_compile \
  nextgqa_pipeline/filter_download_check/inspect_nextgqa.py \
  nextgqa_pipeline/filter_download_check/filter_nextgqa.py \
  nextgqa_pipeline/filter_download_check/build_nextgqa_video_manifest.py \
  nextgqa_pipeline/filter_download_check/download_nextgqa_videos.py \
  nextgqa_pipeline/filter_download_check/check_nextgqa_videos.py
```

以上两条命令**均未被 agent 执行过**，需要你自己手动运行确认。
