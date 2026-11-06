"""CPU-only regression tests for job planning, migration and shell lifecycle."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('job_submit', ROOT / 'jobs/submit.py')
submit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(submit)
BASH = shutil.which('bash') or (r'C:\Program Files\Git\bin\bash.exe' if os.name == 'nt' else None)


def arguments(**kw):
    values = dict(action='eval', name='sufficient', model='qwen25', root=None, python_env=None,
                  env=[], resource=[], array=None, dependency=None, model_path=None,
                  manifest=None, results_dir=None, model_tag=None, script_args=[])
    values.update(kw)
    return SimpleNamespace(**values)


@pytest.mark.parametrize('key', sorted(submit.read_config('evaluations.json')))
def test_every_preset_generates_explicit_model_command(key):
    preset = submit.read_config('evaluations.json')[key]
    p = submit.plan(arguments(name=preset['experiment'], model=preset['model']))
    assert p['name'] == key
    assert p['command'][-2] == preset['model_tag']
    assert p['command'][-1].endswith(preset['results_dir'])
    assert p['resources']['gres'] == ('gpu:2' if preset['model'] in ('qwen25', 'qwen3') else 'gpu:1')
    assert 'set -euo pipefail' in p['script']
    assert not p['root'].startswith('C:')  # POSIX target paths stay POSIX on Windows.


def test_dry_run_creates_nothing_and_quotes_shell_characters(tmp_path):
    target = tmp_path / 'not created'
    p = submit.plan(arguments(root=str(target), results_dir='results/a;echo BAD', env=['EVAL_WORKERS=3']))
    assert p['environment']['EVAL_WORKERS'] == '3'
    assert not target.exists()
    assert "'" in p['script'] and p['command'][-1].endswith('a;echo BAD')


def test_resource_override_and_dependency_are_single_arguments():
    p = submit.plan(arguments(resource=['time=02:00:00', 'partition=debug'], dependency='afterok:123'))
    assert '--time=02:00:00' in p['sbatch']
    assert '--dependency=afterok:123' in p['sbatch']
    with pytest.raises(ValueError):
        submit.plan(arguments(resource=['wrap=echo unsafe']))


def test_partial_blocks_before_submission(tmp_path, monkeypatch):
    candidate = tmp_path / 'candidate.json'
    candidate.write_text(json.dumps([{'video_id': 'a', 'qid': 'b'}]))
    env = [f'PARTIAL_CANDIDATES={candidate}', 'PARTIAL_ALPHAS=0.1',
           'PARTIAL_EXPECTED_ANCHORS=1', 'PARTIAL_EXPECTED_TASKS=1']
    p = submit.plan(arguments(name='partial-n500', root=str(tmp_path), env=env))
    monkeypatch.setattr(submit.subprocess, 'run', lambda *a, **k: pytest.fail('must not invoke sbatch'))
    with pytest.raises(ValueError, match='Missing 1 pre-generated Partial'):
        submit.execute(p)
    assert not (tmp_path / 'reports/job_submissions').exists()


def test_submission_creates_logs_first_and_records_job_id(tmp_path, monkeypatch):
    p = submit.plan(arguments(root=str(tmp_path)))
    monkeypatch.setattr(submit.shutil, 'which', lambda name: '/fake/sbatch')
    def fake_run(cmd, **kw):
        assert Path(p['resources']['output']).parent.is_dir()
        assert kw['input'] == p['script']
        assert kw['env']['VID_EVI_ROOT'] == str(tmp_path)
        return SimpleNamespace(returncode=0, stdout='12345;cluster\n', stderr='')
    monkeypatch.setattr(submit.subprocess, 'run', fake_run)
    assert submit.execute(p) == 0
    receipt = json.loads(next((tmp_path/'reports/job_submissions').glob('*.json')).read_text())
    assert receipt['job_id'] == '12345' and receipt['status'] == 'submitted'


def test_ambiguous_submission_is_not_reported_success(tmp_path, monkeypatch):
    p = submit.plan(arguments(root=str(tmp_path)))
    monkeypatch.setattr(submit.shutil, 'which', lambda name: '/fake/sbatch')
    monkeypatch.setattr(submit.subprocess, 'run', lambda *a, **k: SimpleNamespace(returncode=0, stdout='unexpected', stderr=''))
    assert submit.execute(p) == 1
    receipt = json.loads(next((tmp_path/'reports/job_submissions').glob('*.json')).read_text())
    assert receipt['status'] == 'inspect-before-retry'


def test_specialized_array_requires_indices_and_arguments():
    with pytest.raises(ValueError, match='needs 2 positional'):
        submit.plan(arguments(action='job', name='release.full_small_array'))
    p = submit.plan(arguments(action='job', name='release.full_small_array', script_args=['/release', 'minicpm']))
    assert any('Array index required' in w for w in p['warnings'])
    assert p['sbatch'][-2:] == ['/release', 'minicpm']


def test_migration_targets_and_registered_scripts_exist():
    records = json.loads((ROOT/'job_migration.json').read_text())
    for row in records:
        assert not (ROOT/row['old']).exists(), row
        if row['action'] == 'moved':
            assert (ROOT/row['new']).is_file()
        if row['action'] == 'consolidated':
            assert row['preset'] in submit.read_config('evaluations.json')
    for job in submit.read_config('jobs.json').values():
        assert (ROOT/job['script']).is_file()


@pytest.mark.parametrize('mode,model,media,tp,extra', [
    ('sufficient','qwen25','frames','2','use_fast'),
    ('insufficient','qwen25','frames','2','max_pixels'),
    ('partial','qwen25','frames','2','max_pixels'),
    ('merged','qwen3','frames','2','max_pixels'),
    ('sufficient','llava','video','1','num_frames'),
    ('insufficient','llava','video','1','num_frames'),
    ('partial','llava','video','1','num_frames'),
    ('merged','llava','frames','1',''),
    ('merged','internvl','frames','1','max_dynamic_patch'),
])
def test_preserved_model_condition_settings(mode,model,media,tp,extra):
    command = 'MODE=$1; MODEL_KIND=$2; QWEN_USE_FAST=true; source configs/vllm_models.sh; printf "%s\\n" "$MEDIA_MODE" "$TP" "${EXTRA_SERVER_ARGS[*]}"'
    result = subprocess.run([BASH,'-c',command,'test',mode,model],cwd=ROOT,capture_output=True,text=True)
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert lines[:2] == [media,tp]
    assert extra in result.stdout
    if model == 'qwen25' and mode != 'sufficient':
        assert 'use_fast' not in result.stdout


@pytest.mark.skipif(os.name != 'posix', reason='Linux process-group lifecycle; exercised on HKUST CPU')
@pytest.mark.parametrize('detached,fail_eval', [(False,False),(True,False),(False,True)])
def test_service_cleanup_and_detached_bundle(tmp_path,detached,fail_eval):
    # A tiny local HTTP stub replaces vLLM. No CUDA/model imports or API calls.
    runtime = tmp_path/'runtime/bin'; runtime.mkdir(parents=True)
    (runtime/'python').symlink_to(sys.executable)
    fake = runtime/'vllm'
    fake.write_text('#!'+sys.executable+'\n'+'''import os,sys
from pathlib import Path
from http.server import BaseHTTPRequestHandler,HTTPServer
Path(os.environ['VID_EVI_ROOT'],'server.pid').write_text(str(os.getpid()))
class H(BaseHTTPRequestHandler):
 def do_GET(self):
  self.send_response(200); self.end_headers(); self.wfile.write(b'{"data":[{"id":"test-model"}]}')
 def log_message(self,*args): pass
HTTPServer(('127.0.0.1',int(sys.argv[sys.argv.index('--port')+1])),H).serve_forever()
'''); fake.chmod(0o755)
    code=tmp_path/'scripts/eval'; code.mkdir(parents=True)
    evaluator = 'raise SystemExit(7)\n' if fail_eval else '''import json,sys
from pathlib import Path
out=Path(sys.argv[sys.argv.index('--results-dir')+1]); out.mkdir(exist_ok=True)
(out/'sufficient.classify.json').write_text(json.dumps([{'sampled_seconds':list(range(32))}]))
'''
    (code/'evaluate.py').write_text(evaluator)
    (code/'update_model_results.py').write_text('')
    manifest=tmp_path/'manifest.json'; manifest.write_text('[{"evidence_condition":"sufficient"}]')
    model=tmp_path/'model'; model.mkdir()
    (tmp_path/'configs').mkdir()
    shutil.copy2(ROOT/'configs/vllm_models.sh', tmp_path/'configs/vllm_models.sh')
    runner=ROOT/'scripts/eval/run_sufficient_model.sh'
    if detached:
        for name in ('run_eval.sh','vllm_models.sh','vllm_service.sh'):
            source = ROOT/'configs'/name if name == 'vllm_models.sh' else ROOT/'jobs/lib'/name
            shutil.copy2(source,code/name)
        shutil.copy2(runner,code/runner.name); runner=code/runner.name
    import socket
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0)); port=sock.getsockname()[1]
    env=dict(os.environ,VID_EVI_ROOT=str(tmp_path),VID_EVI_ENV=str(runtime.parent),
             FILTERED_JSON=str(manifest),PARTIAL_FILTERED_JSON=str(tmp_path/"must-not-be-used.json"),
             EVAL_CODE_DIR=str(code),VLLM_PORT=str(port))
    result=subprocess.run(['bash',str(runner),'qwen25',str(model),'test-model','test',str(tmp_path/'out')],env=env,capture_output=True,text=True,timeout=30)
    assert result.returncode == (7 if fail_eval else 0), result.stderr
    pid=int((tmp_path/'server.pid').read_text())
    with pytest.raises(ProcessLookupError): os.kill(pid,0)


def test_eval_arrays_cannot_launch_duplicate_writers():
    with pytest.raises(ValueError, match="do not shard"):
        submit.plan(arguments(array="0-3"))
