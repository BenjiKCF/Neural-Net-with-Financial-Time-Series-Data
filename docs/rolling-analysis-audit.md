# Auditing a completed rolling summary

A circular bootstrap block must be shorter than its paired evaluation period.
When the block length equals the number of dates, every resample contains every
date exactly once. Rotating their order leaves the forecast errors, mean
correlations and portfolio return statistics unchanged. The resulting
zero-width interval cannot measure sampling uncertainty.

Both paired interval functions reject `block_length >= date_count`. Campaign
summaries retain the registered block length and emit `insufficient_dates` with
an explanation and no interval. Forecast date counts include dates with mature
labels; missing labels do not create usable bootstrap dates. Shorter accepted
blocks do not by themselves establish that the evaluation period is long enough
for reliable statistical inference.

Every interval reports `effective_blocks = date_count / block_length`. When
fewer than two full block lengths fit in the evaluation period, the report
explicitly qualifies the interval as having very limited resampling information.
For example, a 20-session block on a 21-session native portfolio period gives
only 1.05 block lengths. Its narrow interval cannot be treated as strong
evidence even though it avoids the exact full-period degeneracy.

The initial frozen runtime used by the November 2026 exploratory campaign
allowed equality. Its 20-session forecast panels therefore cannot support the
registered 20-session bootstrap intervals. The deterministic reproduction is
retained in
[`reproduction.json`](../reports/production_roadmap/bootstrap_audit/reproduction.json).
This reporting correction does not change the registered experiment, model
selection, forecasts, portfolios or point estimates.

After the campaign completes, regenerate a derived summary in a separate
directory using the corrected checkout:

```bash
PYTHONPATH=src python - <<'PY'
from financial_ts.rolling import summarize_rolling_campaign

summarize_rolling_campaign(
    "runs/roadmap-screening-20261127",
    summary_output_dir="reports/production_roadmap/audited_rolling",
)
PY
```

The summary function verifies the archived trial manifests before reading their
data. The separate output contains `comparison.json`, `comparison.csv`,
`trial_history.json`, regenerated seed ensembles, and `analysis_receipt.json`.
The analysis receipt binds the original comparison checksum, registered source
identity, current analysis source identity, the analysis module checksum, and
the corrected bootstrap rule. Original campaign summaries and trial artifacts
remain unchanged. Resuming model fitting still requires the original registered
source identity; summary analysis is a separate derived operation.
