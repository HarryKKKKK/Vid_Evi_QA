"""Merge independently resumable full-evaluation shards in manifest order."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

parent = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(parent if (parent/'prompt_v2.py').is_file() else parent/'eval'))
from prompt_v2 import PROMPT_SHA256, PROMPT_VERSION, record_identity, response_errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('release', type=Path)
    parser.add_argument('kind')
    args = parser.parse_args()
    release = args.release.resolve()
    manifest = [json.loads(x) for x in (release/'smoke.jsonl').read_text().splitlines() if x.strip()]
    wanted = [record_identity(x) for x in manifest]
    if not wanted or any(x is None for x in wanted) or len(set(wanted)) != len(wanted):
        raise SystemExit('Full manifest has missing or duplicate identities')
    found = {}
    duplicates = []
    shard_files = sorted((release/'results'/args.kind).glob('shard_*/merged.classify.json'))
    for path in shard_files:
        for row in json.loads(path.read_text(encoding='utf-8')):
            identity = record_identity(row)
            if identity in found:
                duplicates.append(identity)
            else:
                found[identity] = row
    ordered = [found[x] for x in wanted if x in found]
    outdir = release/'results'/args.kind
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir/'merged.classify.json').write_text(
        json.dumps(ordered, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    (outdir/'merged.classify.jsonl').write_text(
        ''.join(json.dumps(x, ensure_ascii=False)+'\n' for x in ordered), encoding='utf-8')
    invalid = [record_identity(x) for x in ordered if x.get('prompt_version') != PROMPT_VERSION
               or x.get('prompt_sha256') != PROMPT_SHA256 or response_errors(x.get('model_parsed'))]
    missing = [x for x in wanted if x not in found]
    obsolete = sorted(str(x) for x in set(found)-set(wanted))
    report = {'status':'complete' if not missing and not duplicates and not obsolete else 'incomplete',
              'kind':args.kind, 'expected_records':len(wanted), 'result_records':len(ordered),
              'shard_files':len(shard_files), 'missing_ids':missing, 'duplicate_ids':duplicates,
              'obsolete_ids':obsolete, 'invalid_records':len(invalid), 'invalid_ids':invalid,
              'prompt_version':PROMPT_VERSION, 'prompt_sha256':PROMPT_SHA256,
              'conditions':dict(Counter(x.get('evidence_condition') for x in ordered))}
    (outdir/'merged.classify.validation.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report['status'] != 'complete':
        raise SystemExit('One or more full-evaluation shards are incomplete')


if __name__ == '__main__':
    main()
