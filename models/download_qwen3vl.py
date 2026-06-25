from huggingface_hub import snapshot_download
import os

model_id = "Qwen/Qwen3-VL-32B-Instruct"
local_dir = "/aifs4su/hansirui_2nd/harry/Vid_Evi_QA/models/Qwen3-VL"

os.makedirs(local_dir, exist_ok=True)

print(f"Starting download: {model_id}")
print(f"Target path: {local_dir}")

snapshot_download(
    repo_id=model_id,
    local_dir=local_dir,
    local_dir_use_symlinks=False,
)

print("Download complete.")
