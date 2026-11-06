"""CPU-only checks of deployment adapters; no model/GPU dependency."""
import ast
import copy
import hashlib
from pathlib import Path
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import base64
import io

ROOT = Path(__file__).resolve().parents[1]/'scripts/eval/open_v2'


class DeploymentTests(unittest.TestCase):
    def test_repeated_frames_get_distinct_cache_ids_without_changing_pixels(self):
        tree = ast.parse((ROOT/'evaluate_open.py').read_text(encoding='utf-8'))
        function = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == 'add_position_aware_image_ids')
        namespace = {'hashlib':hashlib}
        exec(compile(ast.Module(body=[function],type_ignores=[]),'cache_ids','exec'),namespace)
        data = [{'role':'user','content':[{'type':'text','text':'time'},
                {'type':'image_url','image_url':{'url':'identical_pixels'}},
                {'type':'image_url','image_url':{'url':'identical_pixels'}}]}]
        original=copy.deepcopy(data)
        namespace['add_position_aware_image_ids'](data)
        self.assertNotEqual(data[0]['content'][1]['uuid'],data[0]['content'][2]['uuid'])
        other=copy.deepcopy(original);namespace['add_position_aware_image_ids'](other)
        self.assertEqual(data,other)
        for item in data[0]['content']:item.pop('uuid',None)
        self.assertEqual(data,original)

    def test_model_registry_only_contains_enabled_models(self):
        import runpy
        models = runpy.run_path(str(ROOT/'models.py'))['MODELS']
        self.assertEqual(set(models), {'minicpm', 'qwen3_8b', 'qwen25_7b', 'internvl_4b', 'molmo2_o_7b', 'glm41v_9b',
                                      'llava_ov_7b', 'llava_next_video_7b'})
        for key, spec in models.items():
            if key != 'minicpm':
                self.assertEqual(len(spec['revision']), 40)
                self.assertIn('repo', spec)

    def test_adapters_parse_and_do_not_modify_approved_prompt(self):
        for source in ROOT.glob('*.py'):
            ast.parse(source.read_text(encoding='utf-8'))
        text = (ROOT/'evaluate_open.py').read_text(encoding='utf-8')
        self.assertNotIn('build_instruction =', text)
        self.assertNotIn('build_prompt =', text)
        self.assertIn('extra_body',text)
        text = (ROOT/'run_smoke.py').read_text(encoding='utf-8')
        self.assertIn('SLURM_JOB_ID',text)
        self.assertIn("'--num-frames','32'", text)
        self.assertIn("'--workers','1'", text)
        self.assertIn("'127.0.0.1'", text)
        self.assertIn("'{\"image\":0,\"video\":1}'", text)
        self.assertIn("'{\"video\":{\"num_frames\":32}}'", text)
        self.assertIn("'--media-mode',media_mode", text)

    def test_schema_preflight_is_cpu_only_and_compiles_frozen_schema(self):
        text = (ROOT/'preflight_schema.py').read_text(encoding='utf-8')
        self.assertIn('compile_json_schema', text)
        self.assertIn('RESPONSE_JSON_SCHEMA', text)
        self.assertNotIn('.cuda(', text)
        self.assertNotIn('AutoModel', text)

    def test_native_adapter_preserves_frame_order_and_prompt(self):
        tree = ast.parse((ROOT/'evaluate_molmo.py').read_text(encoding='utf-8'))
        function = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == 'make_messages')
        namespace = {'base64':base64, 'io':io}
        exec(compile(ast.Module(body=[function],type_ignores=[]),'native_messages','exec'),namespace)
        fake_image = SimpleNamespace(open=lambda stream: SimpleNamespace(convert=lambda mode:stream.getvalue()))
        frames = [base64.b64encode(x).decode() for x in [b'frame1',b'frame2']]
        with patch.dict('sys.modules', {'PIL':SimpleNamespace(Image=fake_image)}):
            messages=namespace['make_messages'](frames,[2.5,8.0],'unchanged v2 prompt')
            with self.assertRaises(ValueError):
                namespace['make_messages'](frames,[2.5],'prompt')
        self.assertEqual(messages,[{'role':'user','content':[
            {'type':'text','text':'Frame at 2.5s:'},{'type':'image','image':b'frame1'},
            {'type':'text','text':'Frame at 8.0s:'},{'type':'image','image':b'frame2'},
            {'type':'text','text':'unchanged v2 prompt'}]}])

    def test_native_generation_limit_and_protocol_differences_are_explicit(self):
        text=(ROOT/'evaluate_molmo.py').read_text(encoding='utf-8')
        self.assertIn("MAX_OUTPUT_TOKENS = int(os.environ.get('V2_MAX_TOKENS', '8192'))", text)
        self.assertIn('max_new_tokens=MAX_OUTPUT_TOKENS, do_sample=False',text)
        self.assertIn("response_format='json_schema' if constrained else 'unconstrained_text'",text)
        self.assertIn('compile_json_schema',text)
        self.assertIn("evaluation_scope='engineering_smoke'",text)
        self.assertIn("'available':False",text)
        self.assertNotIn('build_instruction =',text)
        self.assertIn('evaluate.main(backend_preflight=native_preflight)',text)

    def test_glm_extracts_only_final_not_reasoning_json(self):
        tree=ast.parse((ROOT/'evaluate_glm.py').read_text(encoding='utf-8'))
        function=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='final_answer_text')
        namespace={'json':__import__('json')}
        exec(compile(ast.Module(body=[function],type_ignores=[]),'glm_final','exec'),namespace)
        extract=namespace['final_answer_text']
        answer='{"answer":"R"}'
        self.assertEqual(extract('<think>{"answer":"A"}</think><answer>'+answer+'</answer>'),answer)
        self.assertEqual(extract('<think>'+answer),'')
        # vLLM can omit GLM's </answer> EOS token from returned content.
        self.assertEqual(extract('<think>x</think><answer>'+answer),answer)
        self.assertEqual(extract('<think>x</think>'+answer),answer)
        self.assertEqual(extract(answer),answer)
        self.assertEqual(extract('<think><answer>analysis</answer>'),'')

    def test_glm_preserves_raw_and_does_not_invent_fields(self):
        text=(ROOT/'evaluate_glm.py').read_text(encoding='utf-8')
        self.assertIn("model_response=capture['raw']",text)
        self.assertIn("final_answer_text=capture['final']",text)
        self.assertIn("kwargs.pop('response_format', None)",text)
        self.assertNotIn('build_instruction =',text)

    def test_format20_profile_is_unique_and_balanced(self):
        tree=ast.parse((ROOT/'prepare.py').read_text(encoding='utf-8'))
        function=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='select_records')
        namespace={'json':__import__('json')}
        exec(compile(ast.Module(body=[function],type_ignores=[]),'select_records','exec'),namespace)
        rows=[]
        for condition in ['sufficient','insufficient','partially_sufficient']:
            rows += [{'record_id':f'{condition}-{i}','evidence_condition':condition} for i in range(8)]
        for kind in ['scene_context','action_content','temporal_localization','spatial_relation']:
            rows += [{'record_id':f'{kind}-{i}','evidence_condition':'conflicting','conflict_type':kind} for i in range(4)]
        selected=namespace['select_records'](rows,'format20')
        self.assertEqual(len(selected),20)
        self.assertEqual(len({x['record_id'] for x in selected}),20)
        counts={}
        for x in selected:counts[x['evidence_condition']]=counts.get(x['evidence_condition'],0)+1
        self.assertEqual(counts,{'sufficient':4,'insufficient':4,'partially_sufficient':4,'conflicting':8})

    def test_prepare_can_freeze_prompt_and_evaluator_together(self):
        text = (ROOT/'prepare.py').read_text(encoding='utf-8')
        self.assertIn("'--prompt-source'", text)
        self.assertIn("dest/'eval/prompt_v2.py'", text)
        self.assertIn("dest/'eval/evaluate.py'", text)
        self.assertIn("'__pycache__', '*.pyc'", text)

    def test_full_array_is_sharded_resumable_and_not_debug(self):
        prepare = (ROOT/'prepare.py').read_text(encoding='utf-8')
        runner = (ROOT/'full_small_array.sbatch').read_text(encoding='utf-8')
        merger = (ROOT/'merge_full.py').read_text(encoding='utf-8')
        submit = (ROOT/'submit_full.py').read_text(encoding='utf-8')
        self.assertIn("profile == 'full'", prepare)
        self.assertIn("selected[index::args.num_shards]", prepare)
        self.assertIn('#SBATCH --partition=llm', runner)
        self.assertNotIn('#SBATCH --partition=debug', runner)
        self.assertIn('V2_RESULT_DIR', runner)
        self.assertIn('shard_*/merged.classify.json', merger)
        self.assertIn("'submission.lock'", submit)
        self.assertIn("'--array=0-", submit)


if __name__ == '__main__':
    unittest.main()
