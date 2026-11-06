"""Remove only reviewed retired-model rows from the historical mixed audit."""
import hashlib
import json
from pathlib import Path
import runpy
from collections import defaultdict

ROOT = Path('/aifs4su/hansirui_2nd/harry/Vid_Evi_QA')

def main():
    claims = ROOT/'results/build/nextgqa/insufficient/multimodel_correct_answer_claims.initial.jsonl'
    reportfile = ROOT/'results/build/nextgqa/insufficient/multimodel_correct_answer_report.initial.json'
    for p in [claims, reportfile]:
        assert not p.is_symlink() and p.resolve().is_relative_to(ROOT.resolve()), p
    original = claims.read_bytes()
    report_bytes = reportfile.read_bytes()
    lines = original.splitlines(keepends=True)
    retained = []; removed = 0
    for line in lines:
        row = json.loads(line)
        if row.get('model_name') == 'NVILA-8B-Video':
            assert row['model_tag'] == 'nvila8b_video_partial_no_evidence_leak'
            removed += 1
        else:
            retained.append(line)
    assert len(lines) == 12514 and removed == 3332, (len(lines), removed)
    rows = [json.loads(line) for line in retained]
    summarize = runpy.run_path(str(ROOT/'scripts/analysis/export_insufficient_correct_claims.py'))['summarize']
    report = json.loads(report_bytes)
    grouped = defaultdict(list)
    for row in rows: grouped[row['model_name']].append(row)
    report['overall'] = summarize(rows)
    report['by_model'] = {name:summarize(items) for name,items in sorted(grouped.items())}
    report['result_files'] = [row for row in report['result_files'] if row['model'] != 'NVILA-8B-Video']
    assert claims.read_bytes() == original and reportfile.read_bytes() == report_bytes, 'Concurrent change'
    temporary = claims.with_suffix('.retirement.tmp')
    temporary.write_bytes(b''.join(retained)); temporary.replace(claims)
    temporary = reportfile.with_suffix('.retirement.tmp')
    temporary.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');temporary.replace(reportfile)
    assert claims.read_bytes().splitlines(keepends=True) == retained
    receipt = {'removed_model_rows':removed, 'retained_rows':len(rows), 'other_rows_byte_identical':True,
               'before_sha256':hashlib.sha256(original).hexdigest(),
               'after_sha256':hashlib.sha256(claims.read_bytes()).hexdigest(), 'overall':report['overall']}
    (ROOT/'results/reports/model_retirement_20260903/mixed_claims_cleanup.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))

if __name__ == '__main__':
    main()
