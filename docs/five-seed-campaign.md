# Full five-seed campaign

The full recipe is **16 chronological folds × 7 candidates × 5 seeds = 560
trials**, with a maximum of 12 neural epochs and the registered early-stopping
rule. Each trial retains five cost/capital scenarios, giving **2,800 portfolio
replays**. This is historical research; completing it does not establish fresh
confirmation, valid account settlement, or permission to trade.

The [original recipe](../configs/rolling_full.json) remains unchanged. Its
canonical specification SHA256 is
`83b04ee3ee08b3d6e7180a48c529d95c7de4b550c044b8e0982a2b7af9081aa8`.
The candidates are zero return, ridge, LightGBM, temporal Transformer,
daily Transformer, MASTER-style and QuantFormer-MoE. Seeds are 7, 42, 2026,
2027 and 2028. The selection rule consumes all five seeds for every candidate
within a fold. Failed or incomplete trials prevent that fold's final selection.

## Register and run

Use the existing installation and pinned 26 November 2026 provider:

```bash
python scripts/run_full_campaign.py register \
  --provider .data/chenditc/2026-11-26 \
  --config configs/rolling_full.json \
  --output runs/five-seed

python scripts/run_full_campaign.py run \
  --output runs/five-seed --workers 2
```

Registration copies the exact Python package into a separate immutable runtime,
records every source-file hash and dependency version, and registers the
scientific protocol from that copy. Worker subprocesses import only the copied
package. Subsequent development-tree edits cannot silently change a running
campaign. The frozen source is checked before launching work, and every trial
rechecks provider bytes, data metadata, package versions and source identity.
The data provider stays in its original location and is not copied or modified.

Each trial has its own process, log, exclusive lock and numbered attempt.
Process isolation releases Qlib/PyTorch memory between trials. Two workers are
the conservative starting point on the observed CPU-only host with roughly
10 GB of memory; increasing concurrency needs measured memory headroom.
The complete campaign is substantially larger than the earlier two-epoch,
120-session screen. A short invocation cannot close its completion gate.

The host must keep the controller alive, for example through a persistent job
runner or service. The command does not install a service. The supplied GitHub
workflow provides a separate durable execution option and must not be launched
as a duplicate of an already running campaign without an explicit handoff.

## Inspect, resume and audit

```bash
# Fast progress: receipt counts, with no claim that all artifact bytes were checked.
python scripts/run_full_campaign.py status --output runs/five-seed

# Acceptance check: every retained receipt and completed artifact is verified.
python scripts/run_full_campaign.py status \
  --output runs/five-seed --verify-artifacts

# Continue pending work; verified completed trials are reused.
python scripts/run_full_campaign.py run --output runs/five-seed --workers 2

# Finish active trials and pause the remaining queue.
python scripts/run_full_campaign.py stop --output runs/five-seed

# Explicitly resume a stopped queue.
python scripts/run_full_campaign.py run --output runs/five-seed --workers 2 --clear-stop

# Retry failed/interrupted trials as NEW numbered attempts, preserving old evidence.
python scripts/run_full_campaign.py run \
  --output runs/five-seed --workers 2 --retry-failed

# One audited summary writer after all workers finish.
python scripts/run_full_campaign.py summarize --output runs/five-seed
```

An invocation can use `--max-trials N` to bound a single host session. This
does not remove trials from the protocol and cannot make an incomplete campaign
pass. Deterministic `--shard-index I --shard-count N` partitions are disjoint and
cover the declared grid exactly. On a shared campaign directory the supervisor
lock admits one controller; it uses worker processes for parallelism. Isolated
remote shards must restore the same registration artifact before executing.

The `status` output distinguishes receipt counts from
`verified_completed_trials`. `full_campaign_execution_gate` passes only when
all 560 trials and their scenario artifacts are complete and independently
checked against the exact unchanged full recipe. A synthetic fixture, a smaller
fold budget, a missing seed, a changed recipe, or a counter-only report cannot
pass. `promotion_ready` remains false; a separate quality and operational review
must consume the evidence.

Interrupted workers retain their last receipt. A process that exits between
creating an attempt directory and writing its first receipt is recovered under
the exclusive trial lock: the missing-evidence attempt is recorded as failed,
without inventing a start time or completed work. A retry gets a new directory.

## Retained launch

On 27 November 2026 at 17:39:37 Hong Kong time, the unchanged recipe was
registered locally with protocol
`e8f74600571cfc491ac897541d7338d9abb3ddc9e6a1dcc219a60277be4b2660`.
Two CPU workers were started against the already pinned provider. This is a
launch receipt, not completion evidence. The current status report records
actual finished, failed, running and pending trials.

That initial runtime predates the graceful-stop interface. New registrations
record execution API version 2 and support `stop`; the command rejects the older
runtime rather than falsely claiming that it will stop. Cancellation of an old
runtime leaves active attempts explicitly incomplete and requires a fresh
numbered retry if they are resumed.

## Durable GitHub execution

The [full-campaign workflow](../.github/workflows/full-five-seed.yml) can be
dispatched manually or started by pushing a dedicated `codex/run-five-seed-*`
branch with `[run-five-seed]` in the push's head commit message. A normal fix
commit on that branch does not launch a duplicate campaign. It registers once,
restores that immutable artifact in 80 disjoint
seven-trial jobs, and runs at most eight jobs concurrently. All jobs use the
pinned release and exact installed dependency versions. The original full
recipe, source identities, five seeds and all cost scenarios are unchanged.

A cache miss can produce new retrieval timestamps and API download counters
without changing market data. The registration retains the exact original
provider metadata bytes. Each worker verifies the downloaded provider fingerprint
and semantic source/archive identity, preserves its newly observed retrieval
metadata in a separate receipt, then restores the registered manifest bytes.
A changed source, archive, release, adjustment, universe or provider byte fails;
new retrieval provenance is not relabeled as the original capture time.

Worker artifacts retain forecasts, fitted parameters, numerical bundles,
scenario ledgers, logs and failed attempts for 30 days. Rerunning failed jobs
first restores earlier artifacts for the same shard, verifies their protocol
identity, and refuses any conflicting overwrite. Completed attempts are reused;
failed/interrupted attempts get a new number. A dead worker's running receipt is
preserved byte for byte before its interruption is recorded. The merger permits
only that hash-bound, independently checked receipt transition; unrelated
conflicts still fail. The registration's artifact name
is stable across workflow attempts so a failed-job-only rerun can find it.

The final job merges all shards, verifies every registered trial and scenario,
and fails while any required evidence is absent. Only after that audit does a
single process calculate the registered paired comparisons. Neither a successful
worker exit nor a partial subset becomes an accepted full campaign. Individual
jobs have a six-hour limit; timeout evidence remains incomplete and can be
resumed from retained completed trials. A worker job fails if its assigned shard
has failed or incomplete trials, so GitHub's failed-job rerun includes that
shard rather than merely repeating the final audit. Download the final artifacts before
their retention period expires.

The complete baseline screen and its original numerical evidence are preserved.
Full-budget results must be published under their own campaign identity. Even
favorable retrospective results cannot rename an already inspected interval as
an uninspected holdout.

### Recover the 27 November workflow startup failure

The first remote run,
[45978766798](https://github.com/BenjiKCF/Neural-Net-with-Financial-Time-Series-Data/actions/runs/45978766798),
registered successfully but its workers failed before training. The cache had
already restored the pinned provider; the next step tried to download into that
nonempty directory, which the downloader correctly rejects. The failure is in
the execution wrapper and provides no completed model trials. Those workflow
logs and the original registration remain evidence of the attempt.

The wrapper now distinguishes an absent provider from a populated cache. It
downloads only into an empty destination and leaves cached metadata and market
bytes intact. A populated cache with missing, foreign or synthetic metadata
fails for inspection. Both cache hits and fresh downloads still pass the
complete registered fingerprint and semantic metadata checks before a worker
can run; cache availability alone is not acceptance.

A failed-job rerun of the old revision would execute the same broken workflow.
To continue its original experiment with the repaired wrapper, dispatch the new
workflow with `resume_run_id: "45978766798"`, or push the repair on the existing
dedicated branch with this exact marker in the head commit message:

```text
[resume-five-seed:45978766798]
```

Recovery is refused while the source run is active. Once it is terminal, the new
workflow downloads the source run's original registration artifact, preserves
its protocol, frozen Python source and dependencies, and restores any retained
shard attempts before executing pending trials. It does not call registration
again. Source-run jobs, failed steps, artifact identity and the hashes of the
unchanged registration files are recorded under `full-five-seed-recovery/` in
the new registration and audit artifacts. A later failed-job retry first reuses
the new run's retained registration and attempts, while preserving the original
source reference. Do not use `[run-five-seed]` to repair an existing campaign;
that marker is for a separately registered new campaign.

This handoff changes the workflow revision, not the scientific recipe or frozen
training implementation. The historical local launch and its 15-trial snapshot
remain a separate campaign; their counts are never added to the remote run.
