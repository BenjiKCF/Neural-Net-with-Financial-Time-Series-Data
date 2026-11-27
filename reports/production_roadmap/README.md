# Roadmap implementation evidence

**27 November 2026 · version 2.3.0.** The implementation passes 563 local tests,
Ruff, dependency compatibility checks and an isolated wheel installation.
[acceptance.json](acceptance.json) binds the tested source, commands and evidence.

| Evidence | Scope and outcome |
| --- | --- |
| [Rolling screen](rolling_screening/README.md) | 40/40 registered trials and 160 native scenario replays; 1,724 archived files and all 1,556 receipt-listed artifacts verified. Both validation-selected candidates have negative held-out rank IC. |
| [Tree/neural blend](../../docs/rolling-blend.md) | Two folds, eight forecast panels and 32 native scenario replays; weights chosen using validation only, with both constituents and fixed 50/50 controls. |
| [Advanced models](advanced_screening/results.json) | Single/joint 1/5/20-session forecasts, quantiles/calibration and frozen/incremental/retrained comparisons; small screen with mixed results. |
| [DoubleAdapt-style comparator](../../docs/doubleadapt.md) | Mature-label, prediction-before-update experiment; negative result retained with all matched controls. |
| [Precision profile](../../docs/precision-profile.md) | Full 300-stock CPU FP32/bfloat16 comparison; FP32 retained after bfloat16 fails the rank gate. |
| [Operations](operations_acceptance/summary.json) | Three actual exported models × 300 stocks × 11 checks, including recovery, idempotency, version pinning and rollback. |
| [Lifecycle assessment](lifecycle_transformer/assessment.json) | Archived terminated-series exposure reproduced; unresolved account-level quantities remain unavailable rather than being silently liquidated. |
| [Paper session](paper_session_demo/completion.json) | Synthetic durable replay, cash reconciliation and duplicate-run recovery. |
| [Bootstrap audit](../../docs/rolling-analysis-audit.md) | Original frozen analysis retained; noninformative full-period blocks corrected in a separate source-bound report. |

These are implementation tests and explicitly bounded historical experiments.
They do not constitute prospective observations, a deployment decision, licensed
market-data approval or actual broker execution. The original 12-setup campaign
and native gallery remain unchanged. No model was promoted and no live order was
submitted.

The compact [frozen runtime](rolling_runtime_source.tar.gz) and its
[source receipt](runtime_source.json) preserve the original training code.
The rolling evidence archive includes all numerical bundles, fixtures,
predictions, scenario ledgers and attempts. Its README explains verification
and reproduction at fresh paths. Earlier scientific runner attempts are also
retained with their recorded diagnostics and corrections.
