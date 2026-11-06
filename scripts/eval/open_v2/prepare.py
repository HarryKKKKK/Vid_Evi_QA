"""Freeze deployment adapters and either a smoke sample or full shards."""
import hashlib
import argparse
import json
from pathlib import Path
import shutil
import sys
from models import MODELS

ROOT = Path('/aifs4su/hansirui_2nd/harry/Vid_Evi_QA')
BASE = ROOT/'results/final_filtered_hierarchical_v2/release_20260902_v21'
SOURCE = ROOT/'scripts/eval/open_v2'


def select_records(rows, profile):
    if profile == 'full':
        return list(rows)
    if profile == 'smoke12':
        selected = []
        for condition in ['sufficient', 'insufficient', 'partially_sufficient']:
            selected.extend([x for x in rows if x['evidence_condition'] == condition][:3])
        conflict = [x for x in rows if x['evidence_condition'] == 'conflicting']
        selected.append(next(x for x in conflict if x.get('conflict_type') == 'scene_context'))
        selected.append(next(x for x in conflict if x.get('conflict_type') == 'action_content'))
        selected.append(next(x for x in conflict if x.get('conflict_type') not in ('scene_context', 'action_content')))
        return selected
    # A disjoint follow-up sample: skip records used by smoke12 and cover four
    # conflict families with two records each. This tests format stability, not
    # a statistically powered accuracy estimate.
    selected = []
    for condition in ['sufficient', 'insufficient', 'partially_sufficient']:
        selected.extend([x for x in rows if x['evidence_condition'] == condition][3:7])
    conflict = [x for x in rows if x['evidence_condition'] == 'conflicting']
    groups = [
        [x for x in conflict if x.get('conflict_type') == 'scene_context'][1:3],
        [x for x in conflict if x.get('conflict_type') == 'action_content'][1:3],
        [x for x in conflict if x.get('conflict_type') == 'temporal_localization'][1:3],
        [x for x in conflict if x.get('conflict_type') not in
         ('scene_context', 'action_content', 'temporal_localization')][:2],
    ]
    if any(len(group) != 2 for group in groups):
        raise RuntimeError('The format20 profile cannot cover every conflict family twice')
    for group in groups:
        selected.extend(group)
    identities = {json.dumps(x, sort_keys=True, ensure_ascii=False) for x in selected}
    if len(selected) != 20 or len(identities) != 20:
        raise RuntimeError('format20 must contain 20 unique records')
    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('destination', type=Path)
    parser.add_argument('--base-release', type=Path, default=BASE)
    parser.add_argument('--models', nargs='+', choices=list(MODELS), default=['minicpm'])
    parser.add_argument('--evaluator-source', type=Path,
                        help='Explicitly freeze an updated evaluator; existing releases remain untouched')
    parser.add_argument('--prompt-source', type=Path,
                        help='Explicitly freeze an updated prompt module; existing releases remain untouched')
    parser.add_argument('--sample-profile', choices=['smoke12', 'format20', 'full'], default='smoke12')
    parser.add_argument('--num-shards', type=int, default=0,
                        help='Write deterministic round-robin manifests for array evaluation')
    args = parser.parse_args()
    base = args.base_release.resolve()
    dest = args.destination.resolve()
    if dest.exists():
        raise SystemExit('Destination already exists; do not overwrite a submitted deployment')
    rows = [json.loads(x) for x in (base/'all_condition_records.jsonl').read_text().splitlines() if x.strip()]
    selected = select_records(rows, args.sample_profile)
    dest.mkdir(parents=True)
    shutil.copytree(base/'code', dest/'eval', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    if args.evaluator_source:
        shutil.copy2(args.evaluator_source, dest/'eval/evaluate.py')
    if args.prompt_source:
        shutil.copy2(args.prompt_source, dest/'eval/prompt_v2.py')
    shutil.copytree(SOURCE, dest/'adapters', ignore=shutil.ignore_patterns('__pycache__'))
    (dest/'smoke.jsonl').write_text(''.join(json.dumps(x, ensure_ascii=False)+'\n' for x in selected))
    if args.num_shards:
        if args.sample_profile != 'full' or args.num_shards < 1:
            raise ValueError('--num-shards is supported only for the full profile')
        (dest/'shards').mkdir()
        for index in range(args.num_shards):
            shard = selected[index::args.num_shards]
            (dest/'shards'/f'{index:03d}.jsonl').write_text(
                ''.join(json.dumps(x, ensure_ascii=False)+'\n' for x in shard))
    info = {'source_release': str(base), 'records': len(selected),
            'sample_profile': args.sample_profile, 'num_shards': args.num_shards,
            'models': {}}
    for key in args.models:
        spec = MODELS[key]
        name = spec['name']
        model = ROOT/'models'/name
        if not list(model.glob('*.safetensors.index.json')) and not (model/'model.safetensors').is_file():
            raise RuntimeError(f'No weight index or single checkpoint in {model}')
        for index in model.rglob('*.index.json'):
            for shard in set(json.loads(index.read_text()).get('weight_map', {}).values()):
                if not (index.parent/shard).is_file():
                    raise RuntimeError(f'Missing weight shard {index.parent/shard}')
        metadata = model/'.cache/huggingface/download/README.md.metadata'
        info['models'][name] = {'path': str(model), 'revision': spec.get('revision') or (metadata.read_text().splitlines()[0] if metadata.exists() else None),
                               'config_sha256': hashlib.sha256((model/'config.json').read_bytes()).hexdigest()}
    info['sha256'] = {str(p.relative_to(dest)):hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in dest.rglob('*') if p.is_file()}
    (dest/'deployment.json').write_text(json.dumps(info, indent=2)+'\n')
    print(json.dumps(info, indent=2))


if __name__ == '__main__':
    main()
