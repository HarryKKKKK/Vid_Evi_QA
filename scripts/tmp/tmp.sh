#!/bin/bash
#SBATCH --job-name=insuff_bench
#SBATCH -p debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/bench_%j.out
#SBATCH --error=logs/bench_%j.err

set -uo pipefail
mkdir -p logs

unset http_proxy; unset https_proxy

source /home/hansirui_2nd/anaconda3/etc/profile.d/conda.sh
conda activate /aifs4su/hansirui_2nd/harry/envs
cd /aifs4su/hansirui_2nd/harry/Vid_Evi_QA

SCRIPT=scripts/vqa/build_insufficient.py
JSON=cgbench_pipeline/cgbench_filtered.json
VIDEO_DIR=source_datasets/cg_bench/videos
SAMPLE=10
hr(){ echo "----------------------------------------------------------------"; }

echo "================================================================"
echo " INSUFFICIENT-VIDEO :: BENCHMARK & DIAGNOSE"
echo " host=$(hostname)  date=$(date)  job=${SLURM_JOB_ID:-NA}"
echo " cpus=${SLURM_CPUS_PER_TASK:-?}"
echo "================================================================"

# ---- 1. 视频时长分布（抽样 60 个）-----------------------------
hr; echo "[1] Video duration distribution (sample 60)"
TMPD=$(mktemp -d)
for f in $(ls "${VIDEO_DIR}" | head -60); do
    d=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "${VIDEO_DIR}/${f}" 2>/dev/null)
    [ -n "${d}" ] && echo "${d}" >> "${TMPD}/durs.txt"
done
if [ -s "${TMPD}/durs.txt" ]; then
    sort -n "${TMPD}/durs.txt" -o "${TMPD}/durs.txt"
    N=$(wc -l < "${TMPD}/durs.txt")
    awk -v n="${N}" '
        {a[NR]=$1; sum+=$1}
        END{
            printf "  count=%d  min=%.1fs  max=%.1fs  mean=%.1fs\n", n, a[1], a[n], sum/n
            printf "  median=%.1fs  p90=%.1fs  p99=%.1fs\n", a[int(n*0.5)+0], a[int(n*0.9)+0], a[int(n*0.99)+0]
        }' "${TMPD}/durs.txt"
    echo "  longest 5 (s):"; tail -5 "${TMPD}/durs.txt" | sed 's/^/    /'
else
    echo "  [WARN] could not read any durations"
fi

# ---- 2. 单条全链路计时（jobs=1，分阶段）-----------------------
hr; echo "[2] Single-clip stage timing (jobs=1, first ${SAMPLE} entries)"
echo "  Each line: probe+extract+encode is one ffmpeg full re-encode."
/usr/bin/time -v python "${SCRIPT}" --json "${JSON}" --video-dir "${VIDEO_DIR}" \
    --encoder libx264 --x264-preset veryfast --x264-threads 16 \
    --all --limit "${SAMPLE}" --jobs 1 --overwrite \
    --report "${TMPD}/r_j1.jsonl" 2>"${TMPD}/time_j1.txt"
echo "  --- per-clip status (jobs=1) ---"
grep -o '"status": "[^"]*"' "${TMPD}/r_j1.jsonl" 2>/dev/null | sort | uniq -c | sed 's/^/    /'
echo "  --- wall/cpu (jobs=1) ---"
grep -E "Elapsed \(wall|Percent of CPU|Maximum resident" "${TMPD}/time_j1.txt" | sed 's/^/    /'

# ---- 3. 真实吞吐（jobs=5，正式配置）---------------------------
hr; echo "[3] Throughput at production config (jobs=5, ${SAMPLE} entries)"
T0=$(date +%s)
python "${SCRIPT}" --json "${JSON}" --video-dir "${VIDEO_DIR}" \
    --encoder libx264 --x264-preset veryfast --x264-threads 3 \
    --all --limit "${SAMPLE}" --jobs 5 --overwrite \
    --report "${TMPD}/r_j5.jsonl" 2>&1 | tail -3 | sed 's/^/    /'
T1=$(date +%s); DT=$((T1-T0))
PER=$(awk "BEGIN{printf \"%.2f\", ${DT}/${SAMPLE}}")
echo "  jobs=5: ${DT}s for ${SAMPLE} clips => ${PER}s/clip effective"

# ---- 4. 失败样本的真实 stderr（若有）-------------------------
hr; echo "[4] First ffmpeg_error stderr (if any)"
ERRLINE=$(grep -m1 ffmpeg_error "${TMPD}"/r_*.jsonl 2>/dev/null)
if [ -n "${ERRLINE}" ]; then
    echo "${ERRLINE}" | python -c "import sys,json; d=json.loads(sys.stdin.readline()); print('  vid=',d.get('video_id'),'qid=',d.get('qid')); print(d.get('stderr_tail',''))" | sed 's/^/    /'
else
    echo "  none (all sample clips encoded OK)"
fi

# ---- 5. evidence 占比（决定 concat 是否值得）-----------------
hr; echo "[5] Evidence coverage vs full duration (sample 200 entries)"
python - "$JSON" "$VIDEO_DIR" << 'PYEOF' 2>/dev/null | sed 's/^/  /'
import sys, json, os, subprocess
json_path, vdir = sys.argv[1], sys.argv[2]
data = json.load(open(json_path))
# build stem index
idx = {}
for n in os.listdir(vdir):
    p = os.path.join(vdir, n)
    if os.path.isfile(p):
        idx.setdefault(os.path.splitext(n)[0], p)
def dur(p):
    try:
        return float(subprocess.run(["ffprobe","-v","error","-show_entries",
            "format=duration","-of","csv=p=0",p],capture_output=True,text=True).stdout.strip())
    except: return None
ratios=[]; n_iv=[]
for e in data[:200]:
    p = idx.get(e["video_id"])
    if not p: continue
    D = dur(p)
    if not D: continue
    ivs = e.get("evidence_intervals",[])
    cov = sum(max(0.0,float(iv["end"])-float(iv["start"])) for iv in ivs)
    ratios.append(min(1.0,cov/D)); n_iv.append(len(ivs))
if ratios:
    ratios.sort()
    m=len(ratios)
    print(f"sampled={m}")
    print(f"evidence_coverage: mean={sum(ratios)/m:.3f} median={ratios[m//2]:.3f} "
          f"min={ratios[0]:.3f} max={ratios[-1]:.3f}")
    print(f"intervals/entry: mean={sum(n_iv)/len(n_iv):.2f} max={max(n_iv)}")
    print(">>> If coverage is small (e.g. <0.2), concat (copy non-evidence, "
          "encode only evidence) gives a large speedup.")
else:
    print("no data")
PYEOF

# ---- 6. 已生成进度 --------------------------------------------
hr; echo "[6] Current output progress"
OUT=source_datasets/cg_bench/insufficient_videos
echo "  existing outputs: $(ls "${OUT}" 2>/dev/null | wc -l) / 1596"

rm -rf "${TMPD}"
hr; echo "BENCH DONE."
