"""Reviewed, project-scoped retirement plan; never deletes shared environments."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

ROOT = Path('/aifs4su/hansirui_2nd/harry/Vid_Evi_QA')
REPORT = ROOT/'results/reports/model_retirement_20260903'
JOBS = {'376718', '376723', '376725', '376727', '376736'}

def targets():
    found = []
    for base, dirs, files in os.walk(ROOT, followlinks=False):
        parent = Path(base)
        dirs[:] = [d for d in dirs if d not in {'.git', 'node_modules', '.venv'}]
        for name in list(dirs):
            p = parent/name
            if 'nvila' in name.lower():
                found.append(p); dirs.remove(name)
            elif parent == ROOT/'models' and name not in {'model_download','model_running'}:
                dirs.remove(name)
        for name in files:
            p = parent/name
            special = parent == ROOT/'results/logs' and any(name in {f'open_v2_smoke_{j}.out', f'open_v2_smoke_{j}.err'} for j in JOBS)
            if 'nvila' in name.lower() or special:
                found.append(p)
    return sorted(found)

def validate(p):
    if p.is_symlink() or not p.resolve().is_relative_to(ROOT.resolve()) or p.resolve() == ROOT.resolve():
        raise RuntimeError(f'Unsafe target {p}')
    if 'nvila' not in p.name.lower() and not (p.parent == ROOT/'results/logs' and any(p.name in {f'open_v2_smoke_{j}.out', f'open_v2_smoke_{j}.err'} for j in JOBS)):
        raise RuntimeError(f'Not an exact retired-model target {p}')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['plan','apply'])
    args = parser.parse_args()
    REPORT.mkdir(exist_ok=True)
    plan = REPORT/'deletion_plan.json'
    if args.action == 'plan':
        rows = []
        for p in targets():
            validate(p)
            size = sum(q.stat().st_size for q in p.rglob('*') if q.is_file() and not q.is_symlink()) if p.is_dir() else p.stat().st_size
            rows.append({'path':str(p), 'directory':p.is_dir(), 'bytes':size})
        plan.write_text(json.dumps(rows,indent=2)+'\n')
        print(json.dumps({'targets':len(rows),'bytes':sum(x['bytes'] for x in rows),
                          'directories':[x for x in rows if x['directory']], 'plan':str(plan)},indent=2))
        return
    rows = json.loads(plan.read_text())
    if (REPORT/'deleted.json').exists():
        raise SystemExit('Already applied; inspect receipt, do not repeat')
    for row in rows:
        validate(Path(row['path']))
    deleted = []
    for row in rows:
        p = Path(row['path'])
        if p.is_dir(): shutil.rmtree(p)
        elif p.is_file(): p.unlink()
        else: raise RuntimeError(f'Target changed or missing: {p}')
        deleted.append(row)
        (REPORT/'deleted.json').write_text(json.dumps(deleted,indent=2)+'\n')
    for p in (ROOT/'results').glob('open_v2*deployment*'):
        (p/'RETIRED.txt').write_text('Historical deployment retired 2026-09-03: model-specific artifacts removed at user request. Do not resubmit. Other model results retained.\n')
    print(json.dumps({'deleted':len(deleted),'bytes':sum(x['bytes'] for x in deleted),'remaining_targets':[str(p) for p in targets()]},indent=2))

if __name__ == '__main__':
    main()
