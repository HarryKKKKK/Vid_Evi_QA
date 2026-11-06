"""CPU-only checks for moved entry points, path stability and detached releases."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/src'))
from vid_evi_qa.common.script_export import copy_script, resolve_script


def load_file(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('old,new', [
    ('pipelines/nextgqa/filter_download_check/filter_nextgqa.py',
     'pipelines/nextgqa/filter_download_check/filter_nextgqa.py'),
    ('pipelines/etbench/conflict/build_visual_contrast_jobs.py',
     'pipelines/etbench/conflict/build_visual_contrast_jobs.py'),
    ('scripts/vqa/build_partial.py', 'pipelines/media/build_partial.py'),
    ('scripts/eval/evaluate.py', 'scripts/src/vid_evi_qa/evaluation/evaluate.py'),
    ('scripts/eval/partial_metrics.py', 'scripts/src/vid_evi_qa/metrics/partial_metrics.py'),
])
def test_old_and_new_cli_help_match(old, new):
    outputs = []
    for path in (old, new):
        result = subprocess.run([sys.executable, str(ROOT / path), '--help'],
                                cwd=ROOT, capture_output=True, text=True, timeout=30)
        assert result.returncode == 0, result.stderr
        outputs.append(result.stdout)
    if old != new:
        assert outputs[0] == outputs[1]
    else:
        assert 'usage:' in outputs[0].lower()


def test_default_artifact_paths_are_separate_from_code():
    jobs = load_file(ROOT / 'pipelines/etbench/conflict/build_visual_contrast_jobs.py', 'layout_jobs')
    pairs = load_file(ROOT / 'pipelines/etbench/conflict/build_pair_judging_jobs.py', 'layout_pairs')
    assert jobs.ROOT == ROOT
    assert jobs.DEFAULT_OUTPUT == ROOT / 'results/build/etbench/conflict/visual_contrast_jobs.jsonl'
    assert pairs.DEFAULT_OUTPUT == ROOT / 'results/build/etbench/conflict/pair_judging_jobs.jsonl'
    filtering = load_file(ROOT / 'pipelines/nextgqa/filter_download_check/filter_nextgqa.py', 'layout_filter')
    assert filtering.REPO_ROOT == ROOT


def test_legacy_import_globals_stay_patchable(monkeypatch):
    module = load_file(ROOT / 'scripts/eval/evaluate.py', 'layout_legacy_evaluator')
    # Import-star or runpy wrappers break patch.object used by existing callers.
    assert module.run_evaluation.__globals__ is module.__dict__
    sentinel = object()
    monkeypatch.setattr(module, 'process_entry', sentinel)
    assert module.run_evaluation.__globals__['process_entry'] is sentinel


def test_detached_frozen_evaluator_uses_its_own_prompt(tmp_path):
    release = tmp_path / 'release'
    release.mkdir()
    for filename in ('evaluate.py', 'prompt_v2.py', 'check_v2_run.py', 'script_export.py'):
        copy_script(ROOT / 'scripts/eval' / filename, release / filename)
        assert not any(line.startswith('# vid-evi-compat-target:') for line in
                       (release / filename).read_text(encoding='utf-8').splitlines())
    prompt = release / 'prompt_v2.py'
    with prompt.open('a', encoding='utf-8') as stream:
        stream.write('\nPROMPT_VERSION = "detached-test-version"\n')
    env = dict(os.environ)
    env.pop('PYTHONPATH', None)
    script = ('import sys; sys.path.insert(0, sys.argv[1]); '
              'import evaluate; assert evaluate.PROMPT_VERSION == "detached-test-version"; '
              'import check_v2_run; assert check_v2_run.PROMPT_VERSION == "detached-test-version"')
    result = subprocess.run([sys.executable, '-I', '-c', script, str(release)],
                            cwd=tmp_path, env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr


def test_adapter_copytree_contains_real_standalone_sources(tmp_path):
    target = tmp_path / 'adapters'
    shutil.copytree(ROOT / 'scripts/eval/open_v2', target, copy_function=copy_script,
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    text = (target / 'evaluate_open.py').read_text(encoding='utf-8')
    assert 'def add_position_aware_image_ids' in text
    assert '# vid-evi-compat-target:' not in text[:512]
    # Preparation scripts must retain their freezer dependency in a detached copy.
    helper = load_file(target / 'script_export.py', 'detached_script_export')
    assert helper.resolve_script(target / 'evaluate_open.py') == target / 'evaluate_open.py'


def test_orphaned_compatibility_file_cannot_silently_be_frozen(tmp_path):
    wrapper = tmp_path / 'entry.py'
    wrapper.write_text('# vid-evi-compat-target: scripts/src/nonexistent/module.py\n', encoding='utf-8')
    with pytest.raises(FileNotFoundError):
        copy_script(wrapper, tmp_path / 'frozen.py')


def test_all_migration_entries_resolve_to_real_sources():
    records = json.loads((ROOT / 'scripts/docs/maintenance/script_migration.json').read_text(encoding='utf-8'))
    for record in records:
        assert resolve_script(ROOT / record['old']) == ROOT / record['new']


def test_package_import_uses_canonical_source():
    script = ('import sys; sys.path.insert(0, sys.argv[1]); '
              'from vid_evi_qa.prompts import prompt_v2; '
              'assert prompt_v2.PROMPT_VERSION == "v2.2"; '
              'assert "scripts/src/vid_evi_qa/prompts/" in prompt_v2.__file__.replace(chr(92), "/")')
    result = subprocess.run([sys.executable, '-I', '-c', script, str(ROOT / 'scripts/src')],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
