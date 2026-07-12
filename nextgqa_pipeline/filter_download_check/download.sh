#!/bin/bash
#SBATCH --job-name=nextgqa_download
#SBATCH -p debug
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=10:00:00
#SBATCH --output=logs/nextgqa_download_%j.out
#SBATCH --error=logs/nextgqa_download_%j.err

set -euo pipefail

# --------------------------------------------------------------------------
# NExT-GQA official archive download + extraction + filtered-video symlinks
#
# This is a single archive download. Do NOT submit it as a Slurm array.
#
# Submit:
#   mkdir -p logs
#   sbatch nextgqa_pipeline/filter_download_check/download.sh
#
# Monitor:
#   squeue -u "$USER"
#   tail -f logs/nextgqa_download_<JOBID>.out
# --------------------------------------------------------------------------

# Clear proxy variables that could interfere with Google Drive/gdown.
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
unset all_proxy ALL_PROXY

source /home/hansirui_2nd/anaconda3/etc/profile.d/conda.sh
conda activate /aifs4su/hansirui_2nd/harry/envs

REPO_ROOT="/aifs4su/hansirui_2nd/harry/Vid_Evi_QA"
cd "${REPO_ROOT}"

# NOTE: these must match (or intentionally override) the defaults baked into
# download_nextgqa_videos.py / build_nextgqa_video_manifest.py. They are
# passed to the python command below explicitly so this script's mkdir/
# summary output can never silently drift from what python actually used.
MANIFEST="nextgqa_pipeline/nextgqa_video_manifest.json"
VIDEO_DIR="source_datasets/next_gqa/videos"
ARCHIVE_CACHE_DIR="source_datasets/next_gqa/raw_download"
ARCHIVE_EXTRACT_DIR="source_datasets/next_gqa/raw_extracted"
REPORT_JSON="nextgqa_pipeline/nextgqa_download_report.json"

if [[ ! -f "${MANIFEST}" ]]; then
    echo "ERROR: Manifest does not exist: ${MANIFEST}" >&2
    echo "Run build_nextgqa_video_manifest.py first." >&2
    exit 1
fi

# Confirm that gdown is installed in the active environment.
python - <<'PY'
try:
    import gdown
except ImportError as exc:
    raise SystemExit(
        "ERROR: gdown is not installed in the active environment.\n"
        "Install it with: python -m pip install -U gdown"
    ) from exc

print(f"gdown version: {getattr(gdown, '__version__', 'unknown')}")
PY

mkdir -p \
    "${VIDEO_DIR}" \
    "${ARCHIVE_CACHE_DIR}" \
    "${ARCHIVE_EXTRACT_DIR}" \
    "$(dirname "${REPORT_JSON}")"

python nextgqa_pipeline/filter_download_check/download_nextgqa_videos.py \
  --manifest "${MANIFEST}" \
  --video-dir "${VIDEO_DIR}" \
  --archive-cache-dir "${ARCHIVE_CACHE_DIR}" \
  --archive-extract-dir "${ARCHIVE_EXTRACT_DIR}" \
  --report-json "${REPORT_JSON}" \
  --fetch-official-archive \
  --copy-mode symlink \
  --workers 8

echo "============================================================"
echo "Download/preparation command completed"
echo "End time: $(date)"
echo "Report:   ${REPORT_JSON}"
echo "============================================================"

# Count generated video symlinks/files.
echo "Video-directory summary:"
echo "  Symlinks: $(find "${VIDEO_DIR}" -type l 2>/dev/null | wc -l)"
echo "  Files:    $(find "${VIDEO_DIR}" -type f 2>/dev/null | wc -l)"
echo "  Broken symlinks: $(find "${VIDEO_DIR}" -xtype l 2>/dev/null | wc -l)"

BROKEN_LINKS=$(find "${VIDEO_DIR}" -xtype l 2>/dev/null | wc -l)
if [[ "${BROKEN_LINKS}" -gt 0 ]]; then
    echo "WARNING: ${BROKEN_LINKS} broken symlink(s) were found." >&2
    echo "Do not delete or move ${ARCHIVE_EXTRACT_DIR} when using symlink mode." >&2
fi
