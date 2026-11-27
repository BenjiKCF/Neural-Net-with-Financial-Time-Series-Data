# Portable rolling campaign evidence

This directory contains a complete, independently hash-verified evidence archive
for protocol `52489beb3bc0a813660da1d939962bbd2bc39aee8120150ff4dcfdaf973e87df`. The original campaign files and
all attempt receipts were preserved byte-for-byte. Their absolute source and
output paths are historical provenance and have deliberately not been rewritten.

## Inspect and audit

The original archive is distributed as four ordered byte parts to fit the
transfer size limit. `archive_parts.json` records each part's byte count and
SHA256, plus the original archive identity. Reassembly preserves the original
compressed bytes exactly; it does not recompress or change the evidence.

Run from the repository root. The reassembler uses only Python's standard
library, verifies every part and the combined archive, and atomically creates
the sibling `rolling_evidence.tar.xz`. Repeating it verifies and reuses an
identical archive; it never overwrites a different existing file.

```bash
python scripts/reassemble_rolling_evidence.py \
  --manifest reports/production_roadmap/rolling_screening/archive_parts.json
python scripts/export_rolling_evidence.py \
  --audit reports/production_roadmap/rolling_screening/archive_manifest.json
```

For a copy in another directory, keep the parts and both manifests together and
pass that directory's manifest paths. An already reassembled archive can be
audited directly from an installed checkout:

```bash
python scripts/export_rolling_evidence.py --audit PATH/TO/archive_manifest.json
```

The audit streams every archived file, rejects duplicate or unsafe paths and
non-regular entries, checks the complete inventory, verifies the original
protocol/trial identities, follows every attempt chain, and validates every
receipt-listed artifact hash. It does not load native model pickles or extract
files. Both completed and failed historical attempts remain visible.

After a successful audit, extract to a fresh inspection directory if needed:

```bash
mkdir -p runs/rolling-evidence-inspection
tar -xJf PATH/TO/rolling_evidence.tar.xz -C runs/rolling-evidence-inspection
```

The extracted tree is under `campaign/`. Numerical inference bundles use the
verified JSON/NPZ loader; native `params.pkl` files are retained solely as opaque
historical checkpoint evidence. Do not load a pickle from an untrusted source.

## Repeat the experiment at fresh paths

Do not edit the retained `protocol.json` to make its original paths fit a new
machine. Use its registered specification to create a new campaign identity:

```python
import json
from pathlib import Path
from financial_ts.rolling import prepare_rolling_campaign, run_rolling_campaign

old = json.loads(Path("PATH/TO/protocol.json").read_text())
new = prepare_rolling_campaign(
    provider=".data/chenditc/2026-11-26",
    output="runs/rolling-reproduction-new",
    spec=old["specification"],
)
result = run_rolling_campaign(new)
```

Restore the pinned provider release and confirm its recorded fingerprint. Use
matching source/package versions when reproducing numerical results. A corrected
analysis implementation or a different numerical environment needs its own
identity and an explicit comparison with the retained original/follow-up
receipts. Never treat inspected historical folds as a new confirmation holdout.

`comparison.json`, `comparison.csv`, `aggregation_summary.json`, and individual
receipt copies are readable without opening the archive. All other retained
campaign evidence, including analysis corrections or frozen-original reports,
is included inside the archive and listed in `archive_manifest.json`.

If an audited follow-up was supplied, its derived files are attached inside
`campaign/audited_summary/`. Readable `comparison.json` and `comparison.csv`
then show that corrected analysis; `original_comparison.json` and
`original_comparison.csv` preserve the registered implementation's reports.
`analysis_receipt.json` binds both source-code identities and the untouched
original comparison hash. No experimental predictions or trial receipts are
replaced by the follow-up analysis.

## Storage and exclusions

Compression is deterministic tar metadata plus XZ/LZMA2 preset 6 with a
64 MiB dictionary. Artifacts with the same
filename are adjacent in the stream so repeated large fixture arrays can be
reused by the compressor. Every inference bundle, prediction file, scenario
ledger and attempt is retained.

Generic handler caches, lock files, Python bytecode caches and unused MLflow
internals are excluded. Exact files explicitly listed in an attempt receipt
are always retained, even inside an excluded directory; the manifest lists
these exceptions. No original campaign evidence was deleted.
