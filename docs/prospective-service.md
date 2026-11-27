# Scheduled prospective research controller

The December operations pilot now has a clock-bound arrival and publication
controller. Its original protocol remains
`e0130e4dbcb1e6a8240e1ce91e428623775c052e95f39c57b5cf83a5ccaf1edb`:
20 forecasts from **30 November to 25 December 2026**, a **19:00 HKT** cutoff,
**19:30** publication deadline, and final two-session labels eligible on
**29 December**. The Transformer/ridge pair, calendar and thresholds are unchanged.

A research automation was enabled on 27 November for weekdays at **18:15 and
19:15 HKT**, through 29 December. Its first scheduled invocation is 30 November
at 18:15. The first invocation collects and audits the actual dated source; the
second publishes inside the registered window and captures due mature labels.
The recorded schedule does not certify a completed run or a delivery SLA.
The [schedule receipt](../reports/remaining_acceptance/continuation/schedule.json)
separates enabled scheduling from observed sessions. The pilot still has **0/20**
observations before its first session.

This is research shadow execution. The controller neither grants production
approval nor submits orders. A separate hourly completion watch follows the
560-trial historical campaign; it does not count historical results as future
observations.

## Run a single actual-time tick

Use the registered interpreter and the reviewed checkout. Do not upgrade that
interpreter in place: the original protocol pins the environment as well as its
inference/evaluation source. Installing the current package into an arbitrary
environment is not sufficient to resume this existing protocol.

```bash
PYTHONPATH=src /path/to/registered/python scripts/run_prospective_service.py \
  --store runs/prospective-live/store \
  --feed-root runs/prospective-arrivals
```

The command uses the actual host clock and has no clock override. Before the
cutoff it invokes the [arrival collector](prospective-arrival-feed.md). During
the publication window it fully verifies the admitted provider and calls the
unchanged frozen model pair. Missing source data cannot be acquired late and
backdated. Publication failures enter the registered incident budget; source
polling before the cutoff remains separately recorded acquisition evidence.
Outside the registered calendar it does not invent a forecast, but still checks
retained label custody and can finish an interrupted label capture.

For a source not yet published at the first check, the scheduled task preserves
the failed attempt and retries before 18:50, leaving time for verification.
The real final arrival and publication times determine acceptance. Neither the
schedule nor an eventual successful download can repair a missed session.

Every tick retains its actual start/finish, source-code hash, phase and result
under `store/service_ticks`. The original prospective evaluator determines the
forecast and candidate-quality gates. Service logs cannot substitute for its
observation receipts.

## First-vintage labels

The controller chooses the earliest retained admitted snapshot whose actual
observation is after the registered label maturity. It reads the two observed
close endpoints and applies the existing `close[t+2]/close[t+1]-1` target.
Every predicted instrument remains in the label file; unavailable or invalid
outcomes remain null. It does not substitute a later revision to improve results.

`store/label_sources/SESSION/source.json` retains the endpoint dates, observed
close values, arrival identity, full provider fingerprint and exact label-file
hash. The frozen label receipt binds this source file by SHA256. Subsequent ticks
check the source file, label bytes and referenced arrival metadata. Missing or
modified source records fail. New label derivation always verifies the chosen
provider bytes before and after reading them.

Older provider materializations may be offloaded after a verified durable
checkpoint. Metadata-only receipt checking explicitly reports that the full
provider was not re-read. It is not interchangeable with full provider
verification before inference, new labels or the next vintage comparison.

## Restore without losing prior observations

For the original **zero-observation** registration only:

```bash
python -I scripts/reassemble_rolling_evidence.py \
  --manifest reports/remaining_acceptance/prospective_pilot/archive_parts.json
python -I scripts/restore_prospective_store.py \
  --manifest reports/remaining_acceptance/prospective_pilot/store_archive_manifest.json \
  --destination runs/prospective-live
```

The restore verifies all 67 retained files before extraction, rejects unsafe
members and conflicting destinations, and preserves the required empty evidence
directories. It does not reconstruct the interpreter or waive its identity check.
The actual 27 November restoration and pre-window idle tick are retained in the
[continuation evidence](../reports/remaining_acceptance/continuation).

After collection begins, recover the **latest verified checkpoint**, including
all observations, incidents, labels, and source receipts. Never replace an
observed study with this initial empty archive. See
[checkpoint packaging and restoration](prospective-checkpoints.md).

## Bounded local storage and durable custody

The reference compressed source is 573,153,280 bytes and its materialized provider
is about 849 MiB. Keeping every daily full snapshot in the current small workspace
would exhaust it. A verified transfer of the exact reference archive in two
bounded parts established the save path; the
[receipt](../reports/remaining_acceptance/continuation/archive_preflight.json)
is historical storage evidence, not a new arrival.

The scheduled task packages each admitted source archive and full state/metadata
checkpoint, saves every part and manifest durably, and records each successful
save identity privately. Only after complete successful saving may older local
materializations and duplicate compressed caches be evicted. The latest two
admitted providers remain available for the next snapshot comparison. All
receipts, metadata, forecasts, labels and failure evidence remain retained.
An old provider needed for full verification must be restored from its exact
hash-verified parts. A failed save prevents eviction. The scripts themselves
do not delete historical evidence or infer a successful remote save.

## Acceptance that remains open

The pilot must actually deliver all 20 paired forecasts within their windows and
retain complete mature labels. It is still an `operations_pilot`, so it cannot
establish candidate confirmation even with favorable metrics. Unexplained source
revisions can quarantine a real arriving snapshot. Account settlement, execution
reconciliation, deployment fault rehearsals and longer confirmation retain their
separate gates. No prospective economic result is claimed before observation.
