# Open vision model integration tests on hkstu

All enabled models use the frozen v2.2 prompt, 32 timestamp-labeled images,
width 560, uniform sampling, and temperature zero. Baseline output limit is 768
tokens; GLM repair diagnostics explicitly compare 768 and 8192 tokens.
The 12-record smoke sample covers sufficient, insufficient, partial, scene
conflict, action conflict, and another conflict type. It is not an accuracy study.

## Enabled checkpoints

- MiniCPM-V-4_5 (existing).
- Qwen/Qwen3-VL-8B-Instruct.
- Qwen/Qwen2.5-VL-7B-Instruct.
- OpenGVLab/InternVL3_5-4B.
- zai-org/GLM-4.1V-9B-Thinking (existing vLLM backend).
- allenai/Molmo2-O-7B (native Transformers engineering smoke).

The three additional checkpoints are revision-pinned in models.py and downloaded
with download_models.py --models KEY [KEY ...]. The vLLM models reuse envs_qwen3vl without changing installed
packages. These add model-size comparisons, not three new model families.

prepare.py DESTINATION --models qwen3_8b qwen25_7b internvl_4b freezes code,
the approved manifest sample, and model metadata. Existing releases are not
overwritten. Submit DESTINATION/adapters/smoke.sbatch DESTINATION MODEL_KEY.
Each test requests one debug GPU for at most one hour; run sequentially to avoid
competing with full evaluations. The server binds only to localhost and exits
when the job ends. GPU preflight rejects less than 32 GiB free without killing
unrelated processes.

MiniCPM uses position-aware UUIDs for repeated images, disabled prefix caching,
and enable_thinking=false. Other models use their configured image processor
settings, recorded in the server command. No image pixels or prompt text are
rewritten by the adapter.

Read both deployment_status.json and merged.classify.validation.json. `passed`
means complete inference and schema-valid outputs, not correct answers.
`ready_with_schema_warnings` means complete inference but invalid model outputs.
`failed` means incomplete inference or startup/environment failure. Missing
fields, probabilities, or timestamps must not be fabricated. No full evaluations
are automatically submitted by these scripts.

The retired model was removed at the user's request on 2026-09-03. Earlier smoke
releases are historical only; their model-specific artifacts may have been
removed. The four separately frozen full-evaluation releases are not modified.

Molmo2 uses `.runtime/molmo2_transformers4571`, with its own Transformers 4.57.1
and the existing QENV torch/dependency stack imported read-only through a .pth.
It uses BF16, SDPA and the official processor, with unchanged sampled images and
prompt text. Native decoding is unconstrained by default; the repair diagnostic
sets `MOLMO_JSON_CONSTRAINT=1` to compile the frozen v2.2 JSON Schema with XGrammar.
This constrains structure, not factual correctness. Native token logprobs are not collected.
Effective grammar settings are recorded in every result and deployment status.
These are engineering-only results. No invalid output is repaired or fabricated.

GLM's first smoke (376794) kept the JSON-object constraint and 768-token budget:
12 responses, 12 schema-invalid. The repair runner uses paired unconstrained
768/8192-token tests with identical inputs. `evaluate_glm.py` retains the full raw
response in `model_response`, validates only the final answer after reasoning
delimiters, and records `final_answer_text`, `finish_reason`, usage, and policy.
Unfinished thinking is not treated as an answer, even if it contains quoted JSON.
This does not establish a fair thinking-model protocol; full reasoning evaluation
and any different budget require a separate, explicitly recorded decision.
`GLM_OFFICIAL_SAMPLING=1` instead runs an 8192-token diagnostic using the pinned
checkpoint's temperature 0.8, top_p 0.6, top_k 2, with seed zero. It is explicitly
recorded as an official-generation-config sampling test, not a greedy baseline.

Molmo2's first smoke (376795) loaded its weights but the shared evaluator wrongly
probed an HTTP server. The evaluator now accepts an explicit `backend_preflight`
callback. Native inference verifies its actual CUDA model/processor; default
vLLM users retain the HTTP probe. Use `prepare.py --evaluator-source PATH` to
freeze the updated evaluator into a **new** release; never patch submitted code.

Meta Perception-LM-8B is not enabled: the server token lacks gated-repository
access. The user must obtain access and accept the FAIR noncommercial license;
deployment scripts must not accept it or submit personal information for them.
