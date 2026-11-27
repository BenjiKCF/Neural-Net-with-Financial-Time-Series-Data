# Real source arrivals for the prospective pilot

The arrival collector fills the gap between a registered prospective protocol
and an actual daily source observation. It acquires the exact dated chenditc
release, verifies its published hashes and calendar, retains an immutable local
snapshot and records the time at which that snapshot became usable on this host.
It does not fit models, capture forecasts or submit orders.

The command requires the expected exchange session and its frozen cutoff:

```bash
PYTHONPATH=src /path/to/registered/python scripts/collect_prospective_arrival.py collect \
  --feed-root runs/prospective-arrivals \
  --session 2026-11-30 \
  --decision-cutoff 2026-11-30T19:00:00+08:00
```

The dates above are command examples, not observed arrivals. The first collection
must run on that actual Shanghai/Hong Kong trading date and complete before the
cutoff. A historical source publication time cannot backdate this host's
observation. The collector has no command-line clock override.

## Retained chronology and identity

The arrival stores five separate times: source availability, release discovery,
completed acquisition, completed verification and durable receipt publication.
Source availability is the latest timestamp among the release publication and
both assets' creation/update records. The observed time is the real local clock
after provider fingerprinting and audit, rather than the older timestamp in a
download cache. All source bytes and the source identity are checked again as
part of the acquisition, so an asset replacement between discovery and download
cannot quietly change the admitted input.

The collector reuses a verified compressed asset cache when available. It always
creates and verifies a new immutable provider snapshot for a new arrival. A
repeated successful session verifies and returns its original receipt and time,
including when the retry is after cutoff. A changed cutoff cannot replace it.

The JSON result includes `provider_dir` and `arrival_receipt`. The latter is
`FEED/sessions/YYYY-MM-DD/arrival.json`, suitable for the existing frozen runner:

```bash
PYTHONPATH=FROZEN_RUNTIME/src /path/to/registered/python \
  scripts/run_prospective_acceptance.py --store REGISTERED_STORE run-day \
  --provider ACTUAL_RESULT_PROVIDER_DIR \
  --arrival-receipt ACTUAL_RESULT_ARRIVAL_RECEIPT
```

Inference still has to run during its separately registered 19:00–19:30 HKT
publication window. Collecting the source before cutoff does not establish that
the forecast was delivered on time. The arrival collector keeps
`prospective_forecasts_captured` and `orders_submitted` false.

## Snapshot admission

The immutable snapshot retains the exact source release response, a separate
`provider_metadata.json` copy of downloader provenance, the full-provider
fingerprint and audit. It initially retains all provider bytes as well. The audit
checks every daily binary's offset, extent and values, the observed calendar,
universe membership, close coverage and required OHLCV/adjustment fields.

`collector_source.json` retains the exact acquisition/audit Python source text,
SHA256 values and runtime environment. Its `collector_id` is bound into both
acquisition and arrival receipts. The collector verifies that this identity did
not change during acquisition. Failed acquisition attempts retain the source
identity too. These additions do not change the pilot's frozen inference or
evaluation implementation.

After the first admission, each snapshot is compared with the preceding admitted
vintage. New sessions may extend a series. Historical values, membership,
calendar entries, rows and files cannot change silently. Unexplained price or
adjustment-factor changes quarantine the snapshot. This deliberately strict
collector offers no global revision exemption. A separately sourced and reviewed
corporate-action explanation requires its own admission workflow; it is not
inferred from a changed factor or a favorable model result.

The first snapshot explicitly records **no prior-vintage comparison**. Its
admission checks source custody and current data structure, and cannot establish
point-in-time accuracy of every historical value. Revision summaries retain all
changed-cell counts and up to 100 examples; the complete before/after provider
vintages remain available for inspection.

```bash
PYTHONPATH=src /path/to/registered/python scripts/collect_prospective_arrival.py verify \
  --receipt runs/prospective-arrivals/sessions/2026-11-30/arrival.json
```

Verification rechecks the receipt, source/audit files, metadata and full provider
fingerprint. It also checks the recorded clock ordering and completed durable
publication. An interrupted prepared receipt cannot be accepted as published.

### Verification after archival

The Python API distinguishes two scopes explicitly:

| Function | Checked evidence | Returned scope |
| --- | --- | --- |
| `verify_arrival(path)` | Receipt, retained source/audit/provenance, collector identity, chronology, current provider metadata and complete materialized provider fingerprint | `provider_bytes_verified: true` |
| `verify_arrival_metadata(path)` | Receipt and exact retained metadata inventory, source/audit/provenance consistency, collector identity and chronology; does not hash market data | `provider_bytes_verified: false` |

The metadata check remains usable after an independently verified durable
archive offload removes an older materialized provider. It cannot establish
that provider bytes or an external archive are currently available, and cannot
authorize inference, computation of a new label, or a new historical-vintage
comparison. Those operations require the strict full verification. A changed
or missing retained metadata file still fails even if market data was offloaded.

An archival policy must retain the original compressed source archive under its
bound size/SHA256 and independently verify durable storage before removing
local materializations. Keep the current and preceding providers available for
the next full historical comparison. Preserve the immutable receipt and metadata
files locally, including the original provider provenance; retain restoration
and archival receipts separately from the hash-bound snapshot inventory. The
collector and these verification APIs perform no offload or deletion themselves.

## Failure and storage behavior

Each actual attempt writes its request, observed release metadata when available,
space check and outcome under `FEED/attempts`. Failed or late provider bytes are
retained there; no successful arrival receipt is issued. Late final writes are
moved to the attempt evidence. Missing releases, stale target dates, revised
history, clock violations and insufficient space remain explicit failures.

Before acquiring data, the collector requires at least the larger of 3 GiB or
four compressed archive sizes free. This protects acquisition and retained
failure evidence; it is not a month-long storage-capacity guarantee. There is no
automatic retention deletion or installed scheduler in this command. A durable
deployment must provision sufficient retained storage or introduce an explicitly
audited archival policy before available space is exhausted.

The evidence assumes a trusted local clock and filesystem. It is not an external
timestamp attestation. Failed pre-release polls do not turn into successful
arrivals; the prospective session is assessed from the actual admitted receipt
and actual forecast publication.

## Focused verification

```bash
PYTHONPATH=src /path/to/registered/python -m pytest -q tests/test_arrival_feed.py
```

The tests acquire a hash-verified local source archive, then exercise immutable
retries, future/late source timestamps, download and receipt deadline overruns,
asset replacement, metadata/provider/audit tampering, low storage, historical
price and factor revisions, membership changes and missing required fields.
