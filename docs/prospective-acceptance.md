# Prospective evaluation and shadow acceptance

**Continuation, 27 November:** the unchanged registered pilot now has a
[scheduled research controller](prospective-service.md), a
[real source-arrival collector](prospective-arrival-feed.md), and portable
checkpoint custody. The original registration receipts below are preserved
as history. Scheduling is enabled for the future window; observed forecasts
remain 0/20 until actual sessions are captured.

The prospective runner freezes a future observation window, candidate and
reference numerical bundles, source hashes, dependency versions, daily cutoffs,
label maturity endpoints, service objectives and statistical thresholds before
any forecast in that window can be admitted. Its first capture of each paired
forecast uses the host's actual wall clock. A historical replay receipt with a
future-looking label cannot become prospective evidence.

This supports two separately named uses. `confirmation` assesses a candidate
against its frozen control after the entire registered window and mature labels
are available. `operations_pilot` measures a shorter forecast-delivery exercise;
its candidate-quality status is always `research_only`, even if all observed
metrics look favorable. Neither mode grants production authorization.

## Current recipes and actual limits

The [December pilot recipe](../configs/production/prospective_dec2026_pilot.json)
predeclares 20 forecast sessions, **30 November through 25 December 2026**, with
the final two-session labels eligible no earlier than **29 December at 16:30
Shanghai/Hong Kong time**. The temporal Transformer challenger and ridge control
come from the retained fold-001 seed-42 screen, chosen as fixed software
comparators. They are not newly selected winners. Their inspected historical
outcomes remain exploratory and their data/model rights remain unreviewed.

The pilot's session list uses weekday sessions and the published
[SSE 2026 schedule](https://www.sse.com.cn/disclosure/announcement/general/c/c_20251222_10802507.shtml)
and [SZSE 2026 schedule](https://investor.szse.cn/disclosure/notice/general/t20251222_618087.html),
which declare no holiday closures in that interval. It includes two trailing
sessions for the label endpoint and does not invent 2027 holiday dates.

The [60-session confirmation template](../configs/production/prospective_confirmation.template.json)
is intentionally unregistered. Supply the actual future exchange calendar,
frozen model/control pair, operator and revision before registration; placeholders
are rejected. Sixty sessions and the template's thresholds are a proposed
protocol, not a universal adequacy claim. A new candidate or changed threshold
requires a new prospective registration before its own first decision.

The retained historical provider ends before these future dates. Registration
alone does not start collection, configure an upstream arrival feed, install a
service, or create future observations. Current status must come from the actual
store's status command, never from the existence of a recipe or passing unit test.

## Retained registration

The December operations pilot was actually registered on **27 November 2026 at
17:52:30 Hong Kong time**, before its first forecast session. Its immutable
protocol ID is `e0130e4dbcb1e6a8240e1ce91e428623775c052e95f39c57b5cf83a5ccaf1edb`.
The [registration receipt](../reports/remaining_acceptance/prospective_pilot/registration_receipt.json)
records both bundle identities and the local store. The
[preregistered protocol](../reports/remaining_acceptance/prospective_pilot/protocol.json)
retains the actual timestamp, calendar, thresholds and source/runtime identities.
The [initial status](../reports/remaining_acceptance/prospective_pilot/status.json)
is **pending, 0 of 20 sessions captured**, with candidate quality explicitly
`research_only`. These are registration records; no prospective result is
claimed. No arrival feed or service is installed by registration.

### Pre-observation schedule correction

The original 18:00-cutoff pilot was retained and superseded before any forecast
or label observation. The pinned source release was published at **18:08:41
HKT**, and its manifest upload completed at **18:09:25 HKT**, making that cutoff
incompatible with the observed source schedule. The active pilot therefore uses
**19:00 HKT** with the same 30-minute publication budget (**19:30 deadline**).
Models, session dates, label endpoints and acceptance thresholds are unchanged.

The [source timing receipt](../reports/remaining_acceptance/prospective_pilot/source_timing.json)
records the [release](https://github.com/chenditc/investment_data/releases/tag/2026-11-26)
metadata and SHA256 of the retained provider metadata. The
[supersession receipt](../reports/remaining_acceptance/prospective_pilot/supersession_receipt.json)
links both protocol IDs and hashes the original registration files under
[superseded_18h](../reports/remaining_acceptance/prospective_pilot/superseded_18h/protocol.json).
The old store remains at `runs/prospective-dec2026-pilot-18h-superseded`; the active
store is `runs/prospective-dec2026-pilot-19h`. This observed publication time does
not guarantee future delivery: late arrivals must still fail the frozen window.

The 19:00 recipe was subsequently re-registered on the final reviewed source,
also before any observations. The earlier 19:00 registration is retained at
`runs/prospective-dec2026-pilot-19h-pre-review-superseded` with its original
[receipt and protocol](../reports/remaining_acceptance/prospective_pilot/superseded_19h_pre_review/protocol.json).
This source-only supersession adds interrupted-attempt evidence checks and
post-publication deadline rejection. The active 19:00/19:30 schedule, model pair,
observation dates, labels and thresholds did not change.

## Register once before the first decision

```bash
python scripts/run_prospective_acceptance.py --store runs/prospective-dec2026-pilot-19h register \
  --spec configs/production/prospective_dec2026_pilot.json \
  --candidate-bundle PATH_TO_FROZEN_TRANSFORMER_BUNDLE \
  --reference-bundle PATH_TO_FROZEN_RIDGE_BUNDLE
```

The store copies and verifies both numerical bundles, records their training and
selection lineage, and freezes the exact inference/evaluation source files and
runtime versions. Model fitting must predate the actual registration time.
Known equivalent spellings of the existing two-session Qlib return target are
accepted; other targets require a separately implemented evaluator. Repeating
an identical registration returns the original identity. It cannot reset the
clock or replace a candidate, threshold, calendar or completed observation.

The source and dependency checks intentionally reject a changed implementation
when resuming a registered experiment. Preserve the registered checkout and
runtime for its lifetime. Documentation or unrelated CLI changes do not change
the pinned inference source identity; changing inference or evaluation logic
requires a new study or an explicitly separate audit of the original evidence.

## Restore the retained store and frozen source

The [archive parts](../reports/remaining_acceptance/prospective_pilot/archive_parts.json)
retain the registered store, both numerical bundles and its frozen Python source.
The [file manifest](../reports/remaining_acceptance/prospective_pilot/store_archive_manifest.json)
binds all **67 files** and the original protocol ID. The
[restore receipt](../reports/remaining_acceptance/prospective_pilot/store_restore_receipt.json)
records a verified reconstruction with **zero forecast observations**.

From the repository root, reassemble the archive. This standard-library command
checks each part's size/hash, checks the combined size/hash, and refuses to
overwrite a different existing archive:

```bash
python -I scripts/reassemble_rolling_evidence.py \
  --manifest reports/remaining_acceptance/prospective_pilot/archive_parts.json
```

Then verify the complete member inventory and hashes before extracting into a
**new, empty directory**. This example requires Python 3.12 for its extraction
filter and also verifies the extracted bytes:

```bash
python -I - <<'PY'
import hashlib, json, tarfile
from pathlib import Path, PurePosixPath

base = Path("reports/remaining_acceptance/prospective_pilot")
manifest = json.loads((base / "store_archive_manifest.json").read_text())
archive = base / manifest["archive"]
raw = archive.read_bytes()
assert len(raw) == manifest["bytes"]
assert hashlib.sha256(raw).hexdigest() == manifest["sha256"]
destination = Path("runs/prospective-restored")
with tarfile.open(archive, "r:xz") as stream:
    members = stream.getmembers()
    for member in members:
        path = PurePosixPath(member.name)
        assert not path.is_absolute() and ".." not in path.parts
        assert member.isfile() or member.isdir()  # Reject links and special files.
    files = [member for member in members if member.isfile()]
    assert len(files) == len(manifest["files"])
    assert {member.name for member in files} == set(manifest["files"])
    for member in files:
        data = stream.extractfile(member).read()
        expected = manifest["files"][member.name]
        assert len(data) == expected["bytes"]
        assert hashlib.sha256(data).hexdigest() == expected["sha256"]
    destination.mkdir(parents=True, exist_ok=False)
    stream.extractall(destination, filter="data")
for name, expected in manifest["files"].items():
    path = destination / name
    assert path.stat().st_size == expected["bytes"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected["sha256"]
protocol = json.loads((destination / "store/protocol.json").read_text())
assert protocol["protocol_id"] == manifest["protocol_id"]
print("Verified", len(manifest["files"]), "files; protocol", protocol["protocol_id"])
PY
```

The restored store is `runs/prospective-restored/store`; frozen source is
`runs/prospective-restored/runtime/src`. Use the **registered interpreter and
environment**, with that frozen source first on `PYTHONPATH`:

```bash
PYTHONPATH=runs/prospective-restored/runtime/src /path/to/registered/python \
  scripts/run_prospective_acceptance.py --store runs/prospective-restored/store status
```

Inspect `store/protocol.json` → `environment` for the exact Python, platform and
package/distribution versions. The archive preserves source and evidence; it
does not install or recreate that environment. Portable custody therefore does
**not** waive the environment pin. Resume on a matching runtime, or reconstruct
it faithfully; a different runtime requires a new registration before its own
first observation. Do not edit the protocol or version metadata to bypass the
check. Restoring these zero-observation records also does not install an arrival
feed or scheduler, reopen an expired window, or create prospective results.

## Run each real arriving session

The upstream arrival receipt must contain:

```json
{
  "expected_session": "2026-11-30",
  "decision_cutoff": "2026-11-30T19:00:00+08:00",
  "data_available_at": "2026-11-30T18:10:00+08:00",
  "observed_at": "2026-11-30T18:15:00+08:00",
  "provider_fingerprint": {"sha256": "ACTUAL_ADMITTED_PROVIDER_SHA256"},
  "snapshot_audit_status": "admitted",
  "source": "INSPECTABLE_REAL_ARRIVAL_AND_SNAPSHOT_AUDIT_REFERENCE"
}
```

This is a schema example, not an actual arrival receipt. Fill the fingerprint
with the complete actual provider identity used by the existing adapter and
retain the upstream audit. Source publication and collector observation must
precede the preregistered cutoff. The existing snapshot auditor is the source of
admission; a historical timestamp must never be substituted for an observed feed
arrival. An explicit publication/arrival policy cannot reconstruct missing
historical publication vintages.

```bash
python scripts/run_prospective_acceptance.py --store runs/prospective-dec2026-pilot-19h run-day \
  --provider PATH_TO_ADMITTED_CURRENT_PROVIDER --arrival-receipt arrival.json
```

The runner loads the frozen pair, builds causal feature-only packets, verifies
the admitted provider identity and predicts both models on the same eligible
panel. Both artifacts must be captured between the cutoff and the registered
30-minute deadline. No labels are read by this command and no orders are sent.
An operator may instead capture already generated version-2 daily artifacts:

```bash
python scripts/run_prospective_acceptance.py --store runs/prospective-dec2026-pilot-19h capture \
  --candidate-artifact candidate/2026-11-30 --reference-artifact reference/2026-11-30 \
  --arrival-receipt arrival.json
```

The same real-time deadline applies. Identical retries verify and reuse the
original receipt, without adding another day or replacing its timestamp.
Changed model identities, panels, snapshots or forecasts for an existing day
fail. Invalid or late captures record separate incident receipts. Missing days
are counted against the entire expected session list, not silently dropped.

## Admit mature labels and assess

A label file has exactly `datetime,instrument,label,available_at` columns and
covers every instrument in that session's frozen predictions. Use a null label
for an unavailable outcome; deleting the instrument is rejected. The registered
calendar, two-session endpoint and market close provide a lower bound on
availability regardless of the caller's timestamp. The actual availability
must also be no later than the current clock. The first sourced label vintage
is frozen; changed revisions require separate follow-up research.

```bash
python scripts/run_prospective_acceptance.py --store runs/prospective-dec2026-pilot-19h labels \
  --session 2026-11-30 --csv matured-labels.csv --source INSPECTABLE_LABEL_SOURCE
python scripts/run_prospective_acceptance.py --store runs/prospective-dec2026-pilot-19h status \
  --output runs/prospective-dec2026-pilot-19h-status.json
```

The report separates forecast-delivery acceptance from candidate quality:

| Result | Meaning |
| --- | --- |
| `operational_acceptance: pending` | The registered observation window has not ended. |
| `operational_acceptance: failed` | Closed window missed a session, service objective or incident budget. |
| `operational_acceptance: passed` | The frozen forecast-delivery objectives passed; deployment fault rehearsals and account execution remain separate. |
| `candidate_quality: pending` | The full confirmation window or required mature labels are unavailable. |
| `candidate_quality: failed` | Coverage, absolute Rank IC or paired dependence-aware thresholds failed. |
| `candidate_quality: research_only` | Operations pilot; it is ineligible to establish candidate confirmation. |
| `candidate_quality: passed` | A separately registered confirmation met its declared forecast-quality tests and is ready for review. |

Matched candidate/reference forecast panels retain full daily cross sections.
Paired moving date-block intervals use every predeclared block length. Each
length must have enough effective blocks and its Rank IC gain lower 95% bound
must exceed the frozen threshold. Null labels and undefined correlations reduce
coverage or usable dates. Early monitoring does not end the fixed window or
silently change the selected model.

A successful forecast study still reports `production_ready: false`. It does
not establish realized net profitability, settle corporate actions, exercise a
broker, validate model/data usage rights, test all deployment failure modes, or
record a deployment decision. The roadmap retains those separate gates.

## Evidence custody and tests

A post-publication clock check rejects a slow final rename or receipt write that
crosses the deadline. Such artifacts retain a late status in
`rejected_observations`, with a separate incident, and cannot count as captured
sessions. A prepared directory without a committed receipt fails verification.

Each session is an atomically published directory with both verified forecast
artifacts, its original arrival receipt, capture time and SHA256 inventory.
Label vintages bind the corresponding observation receipt. Protocol and stored
artifacts are checked on each status read. Unexpected sessions, changed files,
symlinks, stale sources and impossible timestamp order fail closed.

These protections assume a trusted host clock, source collector and filesystem.
They cannot prove a statement supplied by a dishonest source or prevent an
administrator from replacing an entire local history and its hashes. Preserve
registration and receipt hashes outside the mutable working directory when an
independent timestamp or audit authority is required.

```bash
python -m pytest -q tests/test_prospective.py
```

Tests use an explicitly replaced private clock and synthetic models/data. They
cover retrospective registration, early/late capture, capture crossing its
deadline, conflicting duplicates, source/panel mismatches, tampering, premature
labels, unavailable outcomes, frozen vintages and pilot/confirmation separation.
Their success is software evidence, not a real prospective observation.
