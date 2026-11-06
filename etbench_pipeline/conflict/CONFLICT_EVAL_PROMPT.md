# Conflict multi-answer evaluation contract

User-confirmed 2026-09-02: at least two distinct correct options trigger UNANSWERABLE / R / CONFLICTING_EVIDENCE. The evidence may occur at different times or simultaneously in one frame; factual contradiction is not required. Each option must independently satisfy the question. Two evidence groups may share timestamps.

Prompt version: hierarchical_classify_multi_answer_v2. Previous-version caches are not reused. Temporal / spatial / mixed / unclassified labels are analysis metadata only; they are not supplied to the model. No new evaluation is launched by the manual data publisher.

The output retains decision probabilities, conditional option probabilities, evidence groups, and resolvability. Option-set diagnostics are separate from the required rejection decision. Both TAL and repaired TEM records may be evaluated; the existing null-official-action guard remains enabled.
