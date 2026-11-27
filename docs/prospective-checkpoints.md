# Prospective source and state checkpoints

The checkpoint command retains the exact compressed data vintage separately
from the prospective study's smaller state. It prepares verifiable files for
durable storage. It does not upload them, delete a provider or download cache,
change an arrival timestamp, or declare an experiment accepted.

## Package one admitted session

Use the registered interpreter and acquire the checkpoint between service
ticks. The command takes the same service, collection, observation and label
locks; it fails immediately while another operation holds them.

```bash
PYTHONPATH=src /path/to/registered/python scripts/package_prospective_checkpoint.py package \
  --store PATH_TO_REGISTERED_STORE \
  --feed PATH_TO_ARRIVAL_FEED \
  --session YYYY-MM-DD \
  --runtime-root PATH_TO_FROZEN_RUNTIME_SRC \
  --output NEW_CHECKPOINT_DIRECTORY
```

The session's original archive must be present at
`FEED/downloads/YYYY-MM-DD/qlib_bin.tar.gz`. Its size and SHA256 must agree with
the retained GitHub release, source manifest and admitted provider provenance.
The complete archive is validated before it is split. The command does not
redownload a later revision or substitute a provider with the same date.

Each checkpoint contains:

| File | Contents |
| --- | --- |
| `qlib_bin.tar.gz.part-0000`, subsequent numbered parts | Exact original archive bytes, in order. Each part is at most **280 MiB**. |
| `state.tar.gz` | All registered study files, including both numerical bundles, protocol, observations, labels, source evidence, incidents and service ticks. It also includes feed sessions, snapshot metadata and successful/failed attempt metadata. |
| `manifest.json` | Per-part and combined hashes, every state file's size/hash, protocol and arrival identities, original root paths, and exact retained source identities. |

Materialized `provider` directories and compressed download caches are excluded
from the state archive. The selected compressed source exists only as numbered
parts in the checkpoint. Metadata from rejected attempts is retained; their
large provider directories are excluded. Process lock files are excluded.

`--runtime-root` is optional. When supplied, its registered inference source
files must match every hash in the prospective protocol; all files under that
runtime source root are included. This preserves source, not the interpreter
or installed dependency environment. Without it, the manifest explicitly says
that registered source files were not included in this checkpoint.

The checkpoint records its actual creation time. Original source publication,
collector observation, admission, prediction and label times remain in their
unchanged files. The original receipt bytes and identifiers are preserved.

The command requires a new output directory and checks that source membership
and file contents did not change during packaging. A failed or interrupted
attempt is left for inspection; it is not presented as a completed checkpoint.

## Verify before and after durable saving

```bash
PYTHONPATH=src /path/to/registered/python scripts/package_prospective_checkpoint.py verify \
  --checkpoint CHECKPOINT_DIRECTORY
```

Verification reads every part, verifies the combined original archive hash,
checks the state archive, streams and hashes every state member, and rechecks
the protocol, arrival and source identities. Missing, reordered, changed,
unregistered or unsafe members are rejected.

The result is a local custody check. It has `durable_save_status: not_recorded`
when first packaged. An external storage workflow must save **all parts, the
state archive and the manifest**, retain the successful save receipts and
verify the saved identities. A generated filename, attempted upload or local
checksum is not evidence that durable saving succeeded.

This script has no eviction command. Any later storage-management workflow
must require explicit successful durable-save evidence before removing an
older materialized provider or compressed cache, keep the current and previous
snapshots available for revision comparison, and preserve all arrival metadata.

## Restore an offloaded provider

The following command streams the selected source parts directly into a new
staging directory. It does not recreate a second compressed download cache.

```bash
PYTHONPATH=src /path/to/registered/python scripts/package_prospective_checkpoint.py restore-provider \
  --checkpoint CHECKPOINT_DIRECTORY \
  --feed EXISTING_RETAINED_FEED
```

The existing feed must still contain the original session receipt and its
snapshot metadata. They must agree with the selected checkpoint. Restoration
uses the retained release, source manifest and exact `provider_metadata.json`;
it never runs the downloader or invents a new retrieval time.

Every source archive member is checked for safe type and path, duplicates are
rejected, and the complete stream digest is verified again. The reconstructed
calendar and universe coverage must match the original provenance. The full
provider fingerprint must match the admitted arrival before the staging
directory is moved to its original relative location:

```text
FEED/snapshots/ORIGINAL_DATE_AND_ID/provider
```

Finally, `verify_arrival` verifies both the retained metadata and all consumed
provider bytes. A valid existing provider is reused. A changed existing provider
is rejected and is never overwritten. Failed staging is retained for inspection.
The restoration result contains a new actual restoration time; the original
arrival and acquisition files remain byte-for-byte unchanged.

## Restore portable study state for inspection

```bash
PYTHONPATH=src /path/to/registered/python scripts/package_prospective_checkpoint.py restore-state \
  --checkpoint CHECKPOINT_DIRECTORY \
  --destination NEW_REVIEW_ROOT
```

This creates `NEW_REVIEW_ROOT/store`, `NEW_REVIEW_ROOT/feed`, and, when supplied,
`NEW_REVIEW_ROOT/runtime`. Every state member is verified before the recovered
bundle and arrival metadata checks run. Required empty observation, label and
incident directories are recreated for the frozen evaluator without adding
any observation records. The destination must not already exist.

Portable restoration does not rewrite absolute paths inside historical source
references. Its result therefore has `operational_resume_ready: false`. Inspect
the recorded `original_roots` and restore the original operational paths and
matching registered runtime before resuming an existing study. A different
location or environment must not silently change the frozen protocol. Full
provider restoration remains a separate explicit operation.

## Focused checks

```bash
PYTHONPATH=src /path/to/registered/python -m pytest -q \
  tests/test_prospective_checkpoint.py
```

The fixtures use a miniature valid chenditc archive. Tests cover bounded parts,
complete numerical-bundle/state retention, exact original-provider restoration,
portable state restoration, unchanged arrival timestamps/provenance, archive
corruption, wrong sources, changed runtimes, lock contention, symlinks, existing
provider preservation and excluded rejected-provider bytes. No new external
dataset is downloaded by the tests.
