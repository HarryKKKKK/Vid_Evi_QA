"""Slurm-only GPU startup and v2 integration smoke; never launches full eval."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
import urllib.request
from models import MODELS

ROOT = Path('/aifs4su/hansirui_2nd/harry/Vid_Evi_QA')
QENV = Path('/aifs4su/hansirui_2nd/harry/envs_qwen3vl')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('release', type=Path)
    parser.add_argument('kind', choices=list(MODELS))
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise SystemExit('Run under a GPU Slurm allocation, not directly on the login node')
    release = args.release.resolve()
    manifest = Path(os.environ.get('V2_MANIFEST', release/'smoke.jsonl')).resolve()
    if not manifest.is_relative_to(release) or not manifest.is_file():
        raise SystemExit('Evaluation manifest must be a frozen file inside the release')
    metadata = json.loads((release/'deployment.json').read_text())
    for name, sha in metadata['sha256'].items():
        if hashlib.sha256((release/name).read_bytes()).hexdigest() != sha:
            raise SystemExit(f'Frozen deployment changed: {name}')
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0)); port = sock.getsockname()[1]
    env = dict(os.environ)
    for name in ['http_proxy','https_proxy','HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','all_proxy']:
        env.pop(name, None)
    env.update(NO_PROXY='localhost,127.0.0.1', no_proxy='localhost,127.0.0.1', PYTHONNOUSERSITE='1',
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', CUDA_HOME='/usr/local/cuda')
    spec = MODELS[args.kind]
    env['V2_MAX_TOKENS'] = str(spec.get('max_output_tokens', 8192))
    model = spec['name']; served = args.kind
    if model not in metadata['models']:
        raise SystemExit('Model was not included in this frozen deployment')
    env.update(VLLM_ATTENTION_BACKEND='TORCH_SDPA', VLLM_MM_ATTENTION_BACKEND='TORCH_SDPA',
               VLLM_ALLREDUCE_USE_SYMM_MEM='0', PATH=str(QENV/'bin')+':'+env['PATH'])
    url = f'http://127.0.0.1:{port}/v1'
    media_mode = spec.get('media_mode', 'frames')
    mm_limits = '{"image":0,"video":1}' if media_mode == 'video' else '{"image":32,"video":0}'
    command = [str(QENV/'bin/vllm'), 'serve', str(ROOT/'models'/model), '--served-model-name', served,
               '--trust-remote-code', '--tensor-parallel-size', '1', '--max-model-len', str(spec['length']),
               '--limit-mm-per-prompt', mm_limits, '--mm-processor-cache-gb','0',
               '--mm-processor-kwargs',json.dumps(spec['processor']), '--no-enable-prefix-caching',
               '--gpu-memory-utilization','0.85','--enforce-eager','--host','127.0.0.1','--port',str(port)]
    if media_mode == 'video':
        command.extend(['--media-io-kwargs', '{"video":{"num_frames":32}}'])
    native = spec.get('backend') == 'transformers'
    native_json = native and os.environ.get('MOLMO_JSON_CONSTRAINT', '0') == '1'
    glm_official = args.kind == 'glm41v_9b' and os.environ.get('GLM_OFFICIAL_SAMPLING', '0') == '1'
    native_python = ROOT/'.runtime/molmo2_transformers4571/bin/python'
    if native:
        command = [str(native_python), '-u', str(release/'adapters/evaluate_molmo.py')]
    configured_resultdir = os.environ.get('V2_RESULT_DIR')
    resultdir = (Path(configured_resultdir).resolve() if configured_resultdir else
                 release/'results'/f'{args.kind}_{os.environ["SLURM_JOB_ID"]}')
    if not resultdir.is_relative_to((release/'results').resolve()):
        raise SystemExit('Result directory must stay inside the frozen release results directory')
    resultdir.mkdir(parents=True, exist_ok=True)
    status = {'status':'starting', 'model':model, 'job_id':os.environ['SLURM_JOB_ID'],
              'command':command, 'url':url, 'manifest':str(manifest),
              'sampling':'32 timestamp-labeled frames, width560, uniform',
              'media_mode':media_mode,
              'max_output_tokens':spec.get('max_output_tokens', 8192),
              'json_constraint':True, 'logprobs_available':True,
              'position_aware_image_cache_ids':args.kind == 'minicpm',
              'note':'The frozen v2.2 JSON schema is enforced during decoding and checked again after generation.'}
    if native:
        status.update(backend='transformers', json_constraint=False, logprobs_available=False,
            url=None, dtype='bfloat16', attention='sdpa',
            note='Engineering smoke only: native generation has no JSON grammar or token logprobs. Prompt and sampled frames unchanged; not a protocol-equivalent formal comparison.')
        if native_json:
            status.update(json_constraint=True, grammar_backend='xgrammar_json_schema',
                note=f'Native frozen-schema diagnostic, {spec.get("max_output_tokens", 8192)} tokens, unchanged frames. No output repair; token logprobs unavailable.')
    elif args.kind == 'glm41v_9b':
        status.update(json_constraint=False, evaluation_scope='engineering_smoke',
            diagnostic_budgets=[8192],
            note='Free-generation diagnostic with an 8192-token ceiling. Preserve raw reasoning; validate final answer only. No output repair.')
        if glm_official:
            status.update(diagnostic_budgets=[8192], temperature=0.8, top_p=0.6, top_k=2, seed=0,
                note='Official generation-config sampling diagnostic with 8192 tokens. Unchanged prompt/frames, no JSON grammar, no answer repair. Engineering-only.')
    statusfile = resultdir/'deployment_status.json'
    statusfile.write_text(json.dumps(status,indent=2)+'\n')
    process = None
    try:
        probe = subprocess.check_output([str(QENV/'bin/python'), '-c',
            'import torch,json,os; print(json.dumps({"cuda_visible_devices":os.environ.get("CUDA_VISIBLE_DEVICES"),'
            '"devices":[{"index":i,"name":torch.cuda.get_device_name(i),"free_bytes":torch.cuda.mem_get_info(i)[0],'
            '"total_bytes":torch.cuda.mem_get_info(i)[1]} for i in range(torch.cuda.device_count())]}))'],env=env,text=True)
        status['gpu_allocation'] = json.loads(probe.strip().splitlines()[-1])
        devices = status['gpu_allocation']['devices']
        if not devices or devices[0]['free_bytes'] < 32*1024**3:
            raise RuntimeError('Allocated GPU has less than 32 GiB free before loading; possible out-of-Slurm occupancy. No other process was terminated.')
        if native:
            env.update(MOLMO_MODEL_PATH=str(ROOT/'models'/model), VLLM_MODEL=served,
                       VLLM_BASE_URL=url, HF_MODULES_CACHE=str(resultdir/'hf_modules'))
            status['status']='testing';statusfile.write_text(json.dumps(status,indent=2)+'\n')
            with (resultdir/'native.log').open('w') as log:
                subprocess.run(command + [
                    '--manifest-jsonl',str(manifest),'--task','classify',
                    '--video-dir',str(ROOT/'source_datasets/next_gqa/videos'),
                    '--results-dir',str(resultdir),'--model-tag',os.environ.get('V2_MODEL_TAG', args.kind+'_native_engineering_v2_2'),
                    '--num-frames','32','--frame-width','560','--sampling','uniform',
                    '--media-mode','frames','--workers','1'],cwd=ROOT,env=env,
                    stdout=log,stderr=subprocess.STDOUT,check=True)
            validation = subprocess.run([str(QENV/'bin/python'),str(release/'eval/check_v2_run.py'),
                str(manifest),'--result',str(resultdir/'merged.classify.json')],cwd=ROOT,env=env)
            report = json.loads((resultdir/'merged.classify.validation.json').read_text())
            status['validation'] = report
            if report['missing_ids'] or report['obsolete_ids'] or report['result_records'] != report['expected_records']:
                raise RuntimeError('Native inference incomplete; inspect native.log and skipped records')
            status['status']='passed' if validation.returncode == 0 else 'ready_with_schema_warnings'
            return
        with (resultdir/'server.log').open('w') as log:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            for _ in range(240):
                if process.poll() is not None:
                    raise RuntimeError(f'Server exited {process.returncode}; inspect server.log')
                try:
                    with urllib.request.urlopen(url+'/models',timeout=3) as response:
                        if served in [x['id'] for x in json.load(response)['data']]: break
                except Exception:
                    time.sleep(5)
            else: raise RuntimeError('Server startup timed out')
            status['status']='testing';statusfile.write_text(json.dumps(status,indent=2)+'\n')
            env.update(VLLM_BASE_URL=url,VLLM_MODEL=served,
                       V2_POSITION_IMAGE_IDS='1' if args.kind == 'minicpm' else '0',
                       V2_REQUEST_EXTRA='{"chat_template_kwargs":{"enable_thinking":false}}' if args.kind == 'minicpm' else '{}')
            if args.kind == 'glm41v_9b':
                status['diagnostics'] = {}
                for budget in [8192]:
                    output = resultdir/(f'official_{budget}' if glm_official else f'free_{budget}')
                    output.mkdir(exist_ok=True)
                    env.update(GLM_MAX_TOKENS=str(budget), GLM_JSON_CONSTRAINT='0')
                    with (output/'evaluation.log').open('w') as eval_log:
                        subprocess.run([str(QENV/'bin/python'),'-u',str(release/'adapters/evaluate_glm.py'),
                            '--manifest-jsonl',str(manifest),'--task','classify',
                            '--video-dir',str(ROOT/'source_datasets/next_gqa/videos'),
                            '--results-dir',str(output),'--model-tag',f'glm41v_free_{budget}_engineering_v2',
                            '--num-frames','32','--frame-width','560','--sampling','uniform',
                            '--media-mode','frames','--workers','1'],cwd=ROOT,env=env,
                            stdout=eval_log,stderr=subprocess.STDOUT,check=True)
                    validation = subprocess.run([str(QENV/'bin/python'),str(release/'eval/check_v2_run.py'),
                        str(manifest),'--result',str(output/'merged.classify.json')],cwd=ROOT,env=env)
                    report = json.loads((output/'merged.classify.validation.json').read_text())
                    status['diagnostics'][str(budget)] = report
                    statusfile.write_text(json.dumps(status,indent=2)+'\n')
                    if report['missing_ids'] or report['obsolete_ids'] or report['result_records'] != report['expected_records']:
                        raise RuntimeError(f'GLM {budget}-token inference incomplete; inspect evaluation.log')
                status['validation'] = report
                status['status']='passed' if validation.returncode == 0 else 'ready_with_schema_warnings'
                return
            subprocess.run([str(QENV/'bin/python'),'-u',str(release/'adapters/evaluate_open.py'),
                            '--manifest-jsonl',str(manifest),'--task','classify',
                            '--video-dir',str(ROOT/'source_datasets/next_gqa/videos'),
                            '--results-dir',str(resultdir),'--model-tag',os.environ.get('V2_MODEL_TAG', args.kind+'_deployment_v2_2'),
                            '--num-frames','32','--frame-width','560','--sampling','uniform','--media-mode',media_mode,
                            '--workers',os.environ.get('V2_WORKERS','1')],cwd=ROOT,env=env,check=True)
            validation = subprocess.run([str(QENV/'bin/python'),str(release/'eval/check_v2_run.py'),str(manifest),
                            '--result',str(resultdir/'merged.classify.json')],cwd=ROOT,env=env)
            report = json.loads((resultdir/'merged.classify.validation.json').read_text())
            status['validation'] = report
            if report['missing_ids'] or report['obsolete_ids'] or report['result_records'] != report['expected_records']:
                raise RuntimeError('Inference/transport incomplete; inspect skipped results')
            # Invalid model-generated JSON is a benchmark outcome, not proof that
            # the model cannot be deployed. Never alter the prompt until it passes.
            status['status']='passed' if validation.returncode == 0 else 'ready_with_schema_warnings'
    except Exception as exc:
        status.update(status='failed',error=str(exc))
        raise
    finally:
        statusfile.write_text(json.dumps(status,indent=2)+'\n')
        if process is not None:
            try: os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError: pass
            try: process.wait(timeout=25)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGKILL); process.wait()
        print(json.dumps(status,indent=2),flush=True)


if __name__ == '__main__':
    main()
