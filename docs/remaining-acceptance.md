# Remaining acceptance work

This continuation starts from merged PR #24, revision
`23b6023f8cd5a07bf79ddff52d8f01180e63a9c9`. It implements evidence collection
and verification for the four remaining requirements. Completion is measured
from retained observations and artifacts, separately from software delivery.

## Current work

| Requirement | Delivered in this continuation | Acceptance boundary |
| --- | --- | --- |
| Larger five-seed campaign | Frozen runtime, resumable isolated workers, an exact-grid auditor and an 80-shard GitHub Actions workflow for the unchanged 560-trial recipe. The local campaign has started. | All 560 trials and 2,800 portfolio scenarios must complete and pass artifact verification. Execution completion alone does not demonstrate an economic advantage. |
| Genuinely uninspected confirmation | Immutable real-clock registration, paired forecast capture, first-vintage mature labels and dependence-aware evaluation. A 60-session confirmation template is available. | The longer confirmation is not registered or passed. The separate 20-session operations pilot is explicitly ineligible for candidate confirmation. |
| Account-level corporate-action settlement | Evidence-bound raw-share account reconstruction, cash/tax/receivable reconciliation, independent closing NAV and late-allocation recovery. The archived evidence inventory was executed and verified. | The historical hypothetical account still requires complete raw-fill/action reconstruction and a separate qualified replay. The original 6.489845% terminated-series exposure remains unresolved. |
| Prospective shadow-operation acceptance | A real future pilot for 30 November–25 December 2026, frozen Transformer/ridge bundles, missing-session accounting and strict admission of on-time forecasts. | Forecast delivery, deployment fault rehearsals and prospective execution reconciliation are separate requirements. No future observation or production acceptance is fabricated. |

See the [five-seed campaign](five-seed-campaign.md),
[account settlement](account-settlement.md), and
[prospective acceptance](prospective-acceptance.md) guides for commands,
receipt schemas and limits.

## One evidence report

```bash
fts roadmap-acceptance \
  --campaign runs/five-seed-20261127 \
  --account-settlement reports/production_roadmap/account_settlement \
  --prospective runs/prospective-dec2026-pilot-19h \
  --output runs/remaining-acceptance.json
```

The command rechecks trial artifacts, recomputes supplied-account settlement
and verifies prospective custody before deriving the report. It exits with code
1 while any requirement remains open. An absent or altered evidence store is
reported as missing or invalid; an old summary cannot substitute for its inputs.

The account report distinguishes successful checks of a supplied account window
from a repaired archived campaign. The prospective report distinguishes actual
forecast delivery from broader operation and execution acceptance. Neither a
synthetic fixture nor the short operations pilot can close model-quality gates.

`production_ready` and `orders_submitted` remain false. The report does not
install a service, submit orders, assign a production alias or record a deployment
approval.

## Preserve chronology

The original 12-setup CSI300 benchmark and 40-trial rolling screen remain intact.
The larger campaign uses a new protocol and frozen source tree. Its historical
dates remain development evidence even if results improve.

The prospective pilot uses a fixed Transformer challenger and ridge reference
for operational diversity. The pair was selected before the future observation
window, without declaring a new test-score winner. A short pilot can expose
arrival, inference and recovery problems while the larger research campaign runs.

The first pilot registration used an 18:00 Hong Kong cutoff. The retained source
metadata showed that the 26 November release appeared at 18:08:41 and its
manifest at 18:09:25. The original registration was preserved, and the cutoff
was superseded to 19:00 with a 19:30 publication deadline before any observations.
Model identities, sessions and acceptance thresholds were unchanged. See the
[source timing receipt](../reports/remaining_acceptance/prospective_pilot/source_timing.json)
and [supersession receipt](../reports/remaining_acceptance/prospective_pilot/supersession_receipt.json).

The 20 forecast sessions end on 25 December; the final two-session label matures
on 29 December. The sourced 2026 exchange calendar supports that window. The
confirmation template does not invent 2027 sessions or relabel inspected history
as a fresh holdout.

## Remaining inputs and elapsed time

Account data and elapsed future sessions cannot be created by passing software
tests. The archived Qlib account is a simulation, so a real brokerage statement
for that hypothetical account does not exist. Its reconstruction needs the raw
execution chain, complete sourced actions and declared counterfactual allocation
assumptions or conservative bounds. Real account statements belong to the separate
actual-account reconciliation workflow.

The pilot currently has no observations. Its registration is a committed protocol,
not evidence that a data feed or scheduled service is already running. Actual
arrivals must bind admitted snapshots and be captured before the cutoff. Late,
missing, stale or revised records retain explicit failures rather than a backdated
success.
