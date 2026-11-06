# Vid_Evi_QA script archive

Scripts-only snapshot from the HKUST project, captured on 2026-11-06 (Asia/Shanghai).
This independent branch intentionally has no shared history with the data-heavy working repository.

## Scope
- scripts/, nextgqa_pipeline/, etbench_pipeline/: Python, shell, Slurm scripts and component READMEs.
- models/model_download/, models/model_running/: helper scripts only, no weights.
- tests/ and unset_proxy.txt.
- Excludes datasets, results, logs, environments, research handoff documents and frozen experiment copies inside results/ and codex_runs/.
- Existing relative layout and behavior are preserved; this is not yet a portable or installable package. Server paths and external data dependencies remain.

## History and credentials
The first commit preserves the pre-cleanup script snapshot; the next records conservative cleanup.
Three model download wrappers are sanitized in BOTH commits: supply HF_TOKEN through the environment.
No credentials are included. These three upload-copy substitutions are not applied to the live server scripts.
Use the parent commit of the cleanup commit to recover removed scripts.

## Current limitation
Pre-generated Partial videos were deleted at the user's request on 2026-11-06.
Fixed-seed on-demand frame noise is not implemented yet. Old Partial runners/builders are kept for reference and regeneration, not declared runnable.
See SCRIPT_AUDIT.md in the cleanup revision for deletion evidence and consolidation candidates.
