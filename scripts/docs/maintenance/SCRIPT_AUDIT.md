# Script cleanup audit — 2026-11-06

This is the first-pass historical audit. The later job cleanup consolidates the runner families below; see [job cleanup](scripts/docs/JOB_LAYOUT.md) and job_migration.json for current replacements.

## Scope and evidence
Conservative first-pass cleanup only. No directory restructuring, evaluator/prompt/metric changes, model runs, or job submissions.
The live server lost only the ten files below; data, weights, results, frozen experiment releases, and remaining code were not modified.
A private server backup with SHA-256 manifest is stored at:
reports/script_cleanup_20261106/removed_scripts.tar.gz

The parent commit preserves every removed file (the three credential-containing download helpers are sanitized in all published revisions).
No byte-identical duplicate scripts were found in the selected script tree.

Candidate names were searched across the current scripts, dataset pipeline code, model helpers, tests, codex_runs and results code/Markdown.
No callers were found for the removed entry points. build_insufficient.py is mentioned only in a build_partial.py docstring and the old root README as the legacy blackout implementation.
Static search cannot establish that a script was never invoked externally or manually. The recovery snapshot is retained for that reason.

## Removed files
| Path | Evidence / reason |
|---|---|
| scripts/tmp/test.sh | Unfinished MiniCPM draft: unconditional exit 1 and unresolved TODO parameters; later eval_mini.sh already completed this draft. |
| scripts/eval/eval_inter.sh | Unreferenced early InternVL sufficient launcher; retained full_sufficient_internvl35.sbatch and run_sufficient_model.sh provide the maintained entry. Historical parameters differ; not claimed equivalent. |
| scripts/eval/eval_mini.sh | Unreferenced early launcher hardwired to missing source_datasets/next_gqa/freeze_videos; current MiniCPM adapter retained in scripts/eval/open_v2. |
| scripts/eval/eval_qwen3.sh | Unreferenced early launcher hardwired to missing source_datasets/next_gqa/freeze_videos; no-leak evaluation and merged runners retained. |
| scripts/eval/check_video.sh | Unreferenced old wrapper hardwired to missing freeze_videos; reusable check_video_frame_extraction.py retained. |
| scripts/eval/launch_internvl3_5_8b_tmux.sh | Unreferenced wrapper submits scripts/eval/serve_internvl3_5_8b.sbatch, which no longer exists. |
| scripts/vqa/build_insufficient.py | Legacy CG-Bench blackout builder explicitly superseded by freeze in historical README; CG-Bench inputs absent. Only remaining code-tree reference is a docstring comparison in build_partial.py, not an import/call. |
| scripts/eval/run_sufficient_model.sh.pre_qa_task_20260824 | Unreferenced timestamped backup; live run_sufficient_model.sh retained. |
| scripts/analysis/qwen25_conservatism_experiment.py.pre_diagnostic_20260824 | Unreferenced timestamped backup; live qwen25_conservatism_experiment.py retained. |
| scripts/analysis/partial_seeded_control_experiment.py.before_repr_ess_20260916 | Unreferenced timestamped backup; live partial_seeded_control_experiment.py retained. |

## Similar scripts retained for later consolidation
| Family | Finding | Next step |
|---|---|---|
| run_sufficient_model.sh / run_insufficient_model.sh / run_partial_model.sh / run_merged_model.sh | Repeated service startup, readiness checks, evaluation and validation. Sufficient/insufficient normalized shell line similarity is about 86%; conditions and checks differ. | Extract shared lifecycle helpers after comparing condition-specific behavior. |
| full_* / partial_n500_* / partial_a010_full_* / partial_selected4_full_* | Repeated model/condition launchers. | Move parameters into model and experiment configuration; keep stable entry points during migration. |
| model_download/download_internvl3_5.sh / download_mini.sh | About 93% normalized shell line similarity, but different model targets. | One downloader with explicit model configuration; retain now. |
| model_running/probe_model.sh / prob_mini.sh | About 90% normalized shell line similarity; differing model settings matter. | Parameterize probe settings; retain probe_inter.py shared helper. |
| scripts/eval/open_v2/evaluate_*.py | Different model adapters and historical repair behavior, not proven redundant. | Keep adapters distinct until input/output contracts are compared. |
| scripts/analysis/* and scripts/eval/* metrics/summaries | Related tasks do not guarantee identical metric definitions. | Document metric contracts before deduplication. |
| results/ and results/archive/codex_runs/ code copies | Frozen research provenance. | Keep in place; intentionally excluded from this scripts-only branch. |

Other retained older files include serve_llama4_scout.sh, standalone model service/probe scripts, and generic metrics tools.
A missing optional model or an old filename alone is insufficient deletion evidence.

## Partial status
Pre-generated Partial videos were removed on 2026-11-06. The seeded on-demand frame-noise replacement is not implemented.
Keep build_partial.py, build_partial_nextgqa.sh, run_partial_model.sh and condition launchers: they preserve generation logic and experiment configuration.
They are not represented as runnable against the now-empty Partial video directory.

## Upload-only credential correction
The three model download .sh files previously contained literal HF tokens.
Both published commits replace those literals with a required HF_TOKEN environment variable. Live server copies are unchanged.
Git credentials, original Git history, datasets, weights, logs, outputs and research handoff documents are not published.

## Validation
- All 158 retained Python files compile syntactically without importing or running them.
- All 115 retained shell/Slurm files pass bash -n (LF-normalized input).
- Secret-pattern and literal credential-assignment scans performed before committing/pushing.
- Ten remote deletions were hash-checked before unlink and verified absent afterwards.
- No GPU/API evaluation was run. Syntax checks do not establish runtime correctness.
- Remaining server paths and external data dependencies are documented in README.md.
