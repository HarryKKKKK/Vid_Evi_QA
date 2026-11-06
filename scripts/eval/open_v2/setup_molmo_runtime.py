"""Isolate Transformers 4.57.1; reuse the existing torch stack read-only."""
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path('/aifs4su/hansirui_2nd/harry/Vid_Evi_QA')
QENV = Path('/aifs4su/hansirui_2nd/harry/envs_qwen3vl')
RUNTIME = ROOT/'.runtime/molmo2_transformers4571'


def main():
    if Path(sys.prefix) != QENV:
        raise SystemExit('Run setup with envs_qwen3vl/bin/python')
    subprocess.run([sys.executable, '-m', 'venv', str(RUNTIME)], check=True)
    python = RUNTIME/'bin/python'
    site = RUNTIME/'lib/python3.12/site-packages'
    # No package in QENV is edited. Native runtime packages precede this path.
    (site/'existing_torch_stack.pth').write_text(str(QENV/'lib/python3.12/site-packages')+'\n')
    subprocess.run([str(python), '-m', 'pip', 'install', '--no-deps',
                    '--ignore-installed', 'transformers==4.57.1', 'accelerate==1.10.1'], check=True)
    versions = subprocess.check_output([str(python), '-c',
        'import torch,transformers,torchvision,einops,accelerate;'
        'import json;print(json.dumps({"torch":torch.__version__,"transformers":transformers.__version__,"transformers_path":transformers.__file__}))'], text=True)
    (RUNTIME/'deployment.json').write_text(json.dumps({'versions':json.loads(versions.strip()),
        'readonly_dependency_site':str(QENV/'lib/python3.12/site-packages')},indent=2)+'\n')
    print(versions)


if __name__ == '__main__':
    main()
