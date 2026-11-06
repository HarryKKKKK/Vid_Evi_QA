"""Download selected pinned frontier checkpoints after an explicit disk check."""
import argparse
from pathlib import Path
import shutil

from huggingface_hub import snapshot_download

from frontier_models import MODELS


ROOT = Path("/aifs4su/hansirui_2nd/harry/Vid_Evi_QA")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", choices=MODELS, required=True)
    args = parser.parse_args()
    requested = sum(MODELS[key]["storage_gib"] for key in args.models)
    free = shutil.disk_usage(ROOT).free / 1024**3
    reserve = 1024
    if free < requested + reserve:
        raise SystemExit(
            f"Need {requested:.1f} GiB plus {reserve} GiB reserve; only {free:.1f} GiB free"
        )
    for key in args.models:
        spec = MODELS[key]
        destination = ROOT / "models" / spec["name"]
        print(f"DOWNLOADING {key} {spec['revision']} -> {destination}", flush=True)
        snapshot_download(
            spec["repo"], revision=spec["revision"], local_dir=destination, max_workers=4
        )
        indexes = list(destination.glob("*.safetensors.index.json"))
        if not indexes and not (destination / "model.safetensors").is_file():
            raise RuntimeError(f"No model weights found in {destination}")
        print(f"READY {key}", flush=True)


if __name__ == "__main__":
    main()
