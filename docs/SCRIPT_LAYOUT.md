# 数据构建与评测实现迁移 — 2026-11-06

后续运行脚本与 Slurm 整理已完成，见 [作业说明](JOB_LAYOUT.md)。下文描述此前的数据/评测实现迁移范围。

本轮迁移 79 个实现：57 个数据管线模块与 22 个评测相关模块。
旧路径保留兼容入口；不调整 Slurm 资源、模型服务、数据、结果或历史冻结包。
另新增独立源码导出工具，并让三个现有冻结包生成器复制实际实现。

## 使用与维护
- 数据构建入口及功能见 pipelines/README.md。
- 评测模块及调用方式见 src/vid_evi_qa/README.md。
- 修改实现时编辑下表右侧路径，不要修改旧路径的兼容入口。
- 包形式运行需设置 PYTHONPATH=src，数据管线模块可从项目根目录直接执行。
- 当前仍保留原服务器路径和原数据产物位置。本轮不改变运行参数或实验定义。
- 旧的扁平导入经兼容入口解析，保留历史冻结包的独立导入语义。后续可以逐步规范内部导入。
- media 本轮只归位现有构建/检查模块；不声称已拆分评测内所有媒体函数或实现新 Partial 加噪。

## 冻结包
旧入口带 vid-evi-compat-target 标记。copy_script 在复制前解析到实际实现。
已调整三个生成器，包含其自身所需的 script_export.py。
若单独把兼容入口复制出项目而未带实现，导出会明确报错，避免生成不可复现的包。
既有 results/、codex_runs/ 中的代码快照没有更新。

## 迁移提交 99b6e76 时的验证与既有问题
- 迁移后去除路径引导代码并还原路径表达式，79 个实现的 Python AST 与迁移前完全一致。
- Windows Python 3.14 与服务器原有 Python 3.12 环境均验证为 56 passed / 4 个同样的既有失败。服务器测试依赖只放在临时目录，未修改训练环境。
- 新增 12 项迁移回归测试：新旧 CLI、默认产物位置、可 monkeypatch 的旧模块、脱离源码树的冻结包、自带 prompt 和适配器导出等。
- 原测试在迁移前为 44 passed / 4 failed，迁移后相同；没有改写测试期望来掩盖协议差异。
- 四项既有失败：test_no_pipe_and_all_three_examples_are_valid_json、test_overlap_conflict_and_empty_insufficient、test_real_resume_flow_and_active_jsonl_cleanup、test_multiple_correct_options_in_one_frame_trigger_rejection。旧用例与当前 v2.2 的版本/响应 schema/prompt 不符，需另行按协议确认修订。
- tests/test_nextgqa_data_pipeline.py 在迁移前即依赖缺失的 build_nextgqa_video_manifest.py，无法收集。本轮保留并记录，未补造实现或静默跳过。
- 未运行 GPU 推理、API 评测或真实数据重建；CPU 回归不代表模型评测通过。

## 后续问题修复
用户随后授权修复既有不合理实现。旧测试已按当前 v2.2 协议和采样规则更新；废弃 manifest 工具的两条测试依赖已按 Git 历史确认移除。
评测响应校验、Conflict 计分和合并完成状态也已修复。最新完整测试不再排除数据管线文件。细节见 [校验修复记录](VALIDATION_FIXES.md)。

## 精确迁移表
| 旧入口 | 维护实现 |
|---|---|
| etbench_pipeline/conflict/build_pair_judging_jobs.py | pipelines/etbench/conflict/build_pair_judging_jobs.py |
| etbench_pipeline/conflict/build_visual_contrast_jobs.py | pipelines/etbench/conflict/build_visual_contrast_jobs.py |
| etbench_pipeline/conflict/conflict_v2_core.py | pipelines/etbench/conflict/conflict_v2_core.py |
| etbench_pipeline/conflict/evaluate_conflict_v2.py | src/vid_evi_qa/evaluation/evaluate_conflict_v2.py |
| etbench_pipeline/conflict/finalize_scene_atomic_verified.py | pipelines/etbench/conflict/finalize_scene_atomic_verified.py |
| etbench_pipeline/conflict/invalidate_scene_results.py | pipelines/etbench/conflict/invalidate_scene_results.py |
| etbench_pipeline/conflict/merge_action_content_recovery_shards.py | pipelines/etbench/conflict/merge_action_content_recovery_shards.py |
| etbench_pipeline/conflict/merge_conflict_v2_shards.py | pipelines/etbench/conflict/merge_conflict_v2_shards.py |
| etbench_pipeline/conflict/recover_action_content_core.py | pipelines/etbench/conflict/recover_action_content_core.py |
| etbench_pipeline/conflict/render_pilot_contact_sheets.py | pipelines/etbench/conflict/render_pilot_contact_sheets.py |
| etbench_pipeline/conflict/run_action_content_recovery.py | pipelines/etbench/conflict/run_action_content_recovery.py |
| etbench_pipeline/conflict/run_conflict_pilot_vlm.py | pipelines/etbench/conflict/run_conflict_pilot_vlm.py |
| etbench_pipeline/conflict/run_conflict_pipeline_v2.py | pipelines/etbench/conflict/run_conflict_pipeline_v2.py |
| etbench_pipeline/conflict/scene_atomic_rebuild.py | pipelines/etbench/conflict/scene_atomic_rebuild.py |
| etbench_pipeline/conflict/validate_conflict_qa.py | pipelines/etbench/conflict/validate_conflict_qa.py |
| etbench_pipeline/conflict/verify_scene_atomic_candidates.py | pipelines/etbench/conflict/verify_scene_atomic_candidates.py |
| nextgqa_pipeline/filter_download_check/filter_nextgqa.py | pipelines/nextgqa/filter_download_check/filter_nextgqa.py |
| nextgqa_pipeline/insufficient/analyze_missed_correct_answer_leaks.py | pipelines/nextgqa/insufficient/analyze_missed_correct_answer_leaks.py |
| nextgqa_pipeline/insufficient/build_no_leak_eval_json.py | pipelines/nextgqa/insufficient/build_no_leak_eval_json.py |
| nextgqa_pipeline/insufficient/build_no_leak_freeze.py | pipelines/nextgqa/insufficient/build_no_leak_freeze.py |
| nextgqa_pipeline/insufficient/build_pre_window_gate_json.py | pipelines/nextgqa/insufficient/build_pre_window_gate_json.py |
| nextgqa_pipeline/insufficient/calibrate.py | pipelines/nextgqa/insufficient/calibrate.py |
| nextgqa_pipeline/insufficient/classify_groundings.py | pipelines/nextgqa/insufficient/classify_groundings.py |
| nextgqa_pipeline/insufficient/common.py | pipelines/nextgqa/insufficient/common.py |
| nextgqa_pipeline/insufficient/dense_ground_ineligible.py | pipelines/nextgqa/insufficient/dense_ground_ineligible.py |
| nextgqa_pipeline/insufficient/export_groundings.py | pipelines/nextgqa/insufficient/export_groundings.py |
| nextgqa_pipeline/insufficient/finalize.py | pipelines/nextgqa/insufficient/finalize.py |
| nextgqa_pipeline/insufficient/ground_correct_answer_evidence.py | pipelines/nextgqa/insufficient/ground_correct_answer_evidence.py |
| nextgqa_pipeline/insufficient/ground_windows.py | pipelines/nextgqa/insufficient/ground_windows.py |
| nextgqa_pipeline/insufficient/measure.py | pipelines/nextgqa/insufficient/measure.py |
| nextgqa_pipeline/insufficient/report_measurements.py | pipelines/nextgqa/insufficient/report_measurements.py |
| nextgqa_pipeline/insufficient/report_no_leak_eval.py | pipelines/nextgqa/insufficient/report_no_leak_eval.py |
| nextgqa_pipeline/insufficient/retry_unparseable_eval.py | pipelines/nextgqa/insufficient/retry_unparseable_eval.py |
| nextgqa_pipeline/insufficient/smoke_test_logprobs.py | pipelines/nextgqa/insufficient/smoke_test_logprobs.py |
| nextgqa_pipeline/insufficient/summarize_correct_answer_evidence.py | pipelines/nextgqa/insufficient/summarize_correct_answer_evidence.py |
| nextgqa_pipeline/insufficient/summarize_dense_ineligible.py | pipelines/nextgqa/insufficient/summarize_dense_ineligible.py |
| scripts/analysis/apply_language_prior_filter.py | pipelines/benchmark/apply_language_prior_filter.py |
| scripts/analysis/apply_personal_scene_repairs.py | pipelines/benchmark/apply_personal_scene_repairs.py |
| scripts/analysis/apply_targeted_conflict_repairs.py | pipelines/benchmark/apply_targeted_conflict_repairs.py |
| scripts/analysis/build_conflict_gpt_candidates.py | pipelines/benchmark/build_conflict_gpt_candidates.py |
| scripts/analysis/final_filter_conflict_gpt.py | pipelines/benchmark/final_filter_conflict_gpt.py |
| scripts/analysis/finalize_aligned_benchmark.py | pipelines/benchmark/finalize_aligned_benchmark.py |
| scripts/analysis/generate_conflict_distractors_gpt.py | pipelines/benchmark/generate_conflict_distractors_gpt.py |
| scripts/analysis/merge_language_prior_runs.py | pipelines/benchmark/merge_language_prior_runs.py |
| scripts/analysis/package_scene_multiplicity_revision.py | pipelines/benchmark/package_scene_multiplicity_revision.py |
| scripts/analysis/prepare_conflict_distractor_retry.py | pipelines/benchmark/prepare_conflict_distractor_retry.py |
| scripts/analysis/prepare_conflict_visual_review.py | pipelines/benchmark/prepare_conflict_visual_review.py |
| scripts/analysis/publish_scene_text_split360.py | pipelines/benchmark/publish_scene_text_split360.py |
| scripts/analysis/review_conflict_distractor_slots_gpt.py | pipelines/benchmark/review_conflict_distractor_slots_gpt.py |
| scripts/analysis/screen_language_priors.py | pipelines/benchmark/screen_language_priors.py |
| scripts/analysis/screen_language_priors_offline.py | pipelines/benchmark/screen_language_priors_offline.py |
| scripts/analysis/targeted_conflict_visual_review.py | pipelines/benchmark/targeted_conflict_visual_review.py |
| scripts/analysis/verify_conflict_distractors_vlm.py | pipelines/benchmark/verify_conflict_distractors_vlm.py |
| scripts/analysis/verify_personal_scene_release.py | pipelines/benchmark/verify_personal_scene_release.py |
| scripts/analysis/verify_scene_multiplicity_release.py | pipelines/benchmark/verify_scene_multiplicity_release.py |
| scripts/analysis/verify_scene_text_split360.py | pipelines/benchmark/verify_scene_text_split360.py |
| scripts/eval/answerable_metrics.py | src/vid_evi_qa/metrics/answerable_metrics.py |
| scripts/eval/check_v2_run.py | src/vid_evi_qa/evaluation/check_v2_run.py |
| scripts/eval/check_video_frame_extraction.py | src/vid_evi_qa/media/check_video_frame_extraction.py |
| scripts/eval/compare_suff_insuff.py | src/vid_evi_qa/metrics/compare_suff_insuff.py |
| scripts/eval/evaluate.py | src/vid_evi_qa/evaluation/evaluate.py |
| scripts/eval/open_v2/check_smoke_gate.py | src/vid_evi_qa/evaluation/check_smoke_gate.py |
| scripts/eval/open_v2/evaluate_glm.py | src/vid_evi_qa/adapters/evaluate_glm.py |
| scripts/eval/open_v2/evaluate_molmo.py | src/vid_evi_qa/adapters/evaluate_molmo.py |
| scripts/eval/open_v2/evaluate_open.py | src/vid_evi_qa/adapters/evaluate_open.py |
| scripts/eval/open_v2/evaluate_prompt_equivalence.py | src/vid_evi_qa/adapters/evaluate_prompt_equivalence.py |
| scripts/eval/open_v2/evaluate_videollama3.py | src/vid_evi_qa/adapters/evaluate_videollama3.py |
| scripts/eval/open_v2/evaluate_videollama3_oomfix.py | src/vid_evi_qa/adapters/evaluate_videollama3_oomfix.py |
| scripts/eval/open_v2/frontier_models.py | src/vid_evi_qa/adapters/frontier_models.py |
| scripts/eval/open_v2/merge_full.py | src/vid_evi_qa/evaluation/merge_full.py |
| scripts/eval/open_v2/models.py | src/vid_evi_qa/adapters/models.py |
| scripts/eval/open_v2/preflight_schema.py | src/vid_evi_qa/evaluation/preflight_schema.py |
| scripts/eval/partial_metrics.py | src/vid_evi_qa/metrics/partial_metrics.py |
| scripts/eval/prompt_v2.py | src/vid_evi_qa/prompts/prompt_v2.py |
| scripts/eval/run_gpt_v2_api.py | src/vid_evi_qa/evaluation/run_gpt_v2_api.py |
| scripts/eval/unanswerable_metrics.py | src/vid_evi_qa/metrics/unanswerable_metrics.py |
| scripts/eval/uniform_coverage_check.py | src/vid_evi_qa/metrics/uniform_coverage_check.py |
| scripts/vqa/build_freeze.py | pipelines/media/build_freeze.py |
| scripts/vqa/build_partial.py | pipelines/media/build_partial.py |
