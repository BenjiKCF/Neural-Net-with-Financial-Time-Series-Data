# Account-level corporate-action settlement

The account settlement workflow compares a reconstructed **raw-share account**
with an independent closing statement. It checks actual share allocations,
cash credits, withholding tax, outstanding receivables and closing NAV. It does
not round a hypothetical registrar allocation or balance a ledger by copying
the closing statement into the reconstructed account.

The retained historical strategy has adjusted Qlib holdings, rather than an
actual broker account. Converting those adjusted quantities with a later factor
does not establish originally executed shares or registrar allocations. The
existing 6.489845% terminated-holding exposure therefore remains qualified.
Resolving a supplied account window also does not, by itself, repair the archived
strategy return. A full corrected strategy replay needs separate, complete
execution and corporate-action history.

There are two distinct evidence paths:

1. **Historical hypothetical strategy:** reconstruct the executed raw-share
   ledger, sourced action history and successor tradability, then run a new
   action-aware strategy replay. Because that simulated account never received
   registrar allocations, ambiguous fractional allocations need explicitly
   registered counterfactual assumptions or conservative outcome bounds. A
   nonexistent broker statement cannot resolve that ambiguity. The original
   campaign and its qualification must remain preserved.
2. **Actual account or future paper account with observed records:** use the
   workflow below to reconcile externally supplied opening/closing statements,
   real account allocations and cash credits. It establishes only that supplied
   account window. A synthetic paper account can exercise the software but does
   not become actual broker evidence.

## Run the evidence inventory

```bash
PYTHONPATH=src python scripts/run_account_settlement.py \
  --spec configs/account_settlement_archived.json \
  --output runs/account-settlement

PYTHONPATH=src python scripts/run_account_settlement.py \
  --verify --output runs/account-settlement
```

The provided registration deliberately has `account_export: null`. It binds
the existing lifecycle assessment and event catalog by SHA256, then produces a
**blocked** report identifying unsupplied account records, independent closing
statement and event coverage, with separate archived-simulation replay blockers.
It is not a source of fabricated
account allocations. Each affected historical event retains its original
quantity discrepancy and missing-evidence reasons.

The public APIs are `run_account_settlement(spec_path, output)` and
`verify_account_settlement(output)` in
[`settlement.py`](../src/financial_ts/settlement.py). Verification checks retained
source bytes and independently reruns reconciliation; it does not trust a
standalone JSON field saying `passed`.

## Evidence registration

A version-1 specification contains exactly these fields:

| Field | Meaning |
| --- | --- |
| `schema_version` | `1` |
| `campaign_id` | New accounting campaign identity |
| `account_id`, `currency` | Exact account and currency shared by all account records |
| `asof` | Timezone-aware audit cutoff; evidence available later is rejected |
| `account_export` | `{path, sha256}` for a normalized account export, or `null` when unavailable |
| `source_artifacts` | Original source documents, each with `source_id`, `kind`, `path`, `sha256` |
| `archived_assessment` | Optional `{path, sha256}` for historical context |
| `event_catalog` | Optional `{path, sha256}` for sourced event identities |

Paths are relative to the specification. Source kinds are `broker`, `custodian`,
`registrar`, `issuer`, `market_data`, and `calendar`. Every record references one
of the verified source IDs. Original bytes are copied into the evidence package
and hashed before execution. Account documents can contain private information;
the workflow writes locally and does not upload them.

Hashes establish integrity. They do not authenticate a broker, independently
certify a manual extraction, or establish original historical publication times.
The normalized export and its completeness attestation must be checked against
the original account/provider records. A reviewer or operator identity in
`attested_by` records that external attestation; the software does not invent it.

## Normalized account export

An export declares `schema_version: 1`, `quantity_units: raw_shares`, exact
`account_id`/`currency`, and `evidence_kind: account_records` or
`synthetic_fixture`. Synthetic fixtures can prove arithmetic, but always leave
the account-evidence gate blocked.

Every statement, operation, mark and calendar record has `record_id`,
`source_id`, and a timezone-aware `available_at`. Account statements and
operations additionally repeat `account_id` and `currency` so a document from
another account cannot silently enter the replay.

| Record | Required contents |
| --- | --- |
| `opening` | `asof`, `cash`, integer raw-share `positions`, and event-keyed `receivables` |
| `closing` | Independently observed `asof`, `cash`, `positions`, `receivables`, and `nav` |
| `coverage` | Exact `from`/`through`, `complete: true`, ordered `operation_ids`, all `event_ids`, and `attested_by` |
| `operations` | Chronological list of the operation types below |
| `marks` | Instrument-keyed records containing `price`, `observed_at`, HTTPS `source`, and `price_convention: raw` |
| `calendar` | Ordered exchange `sessions` covering the closing cutoff |

Nonzero opening receivables require matching `opening.receivable_terms` entries
containing sourced `record_id`, `source_id`, `account_id`, `currency`,
`available_at`, `effective_at` and `payment_at`; otherwise the gate remains
blocked. Each opening obligation must already exist at the opening cutoff.
Decimal strings are supported for money. Share quantities must be whole raw
shares. Cash, receivables and NAV reconcile within **one cent**; share quantities
and instrument identities must match exactly.

Each operation supplies `kind` and `effective_at`:

| Kind | Account information used |
| --- | --- |
| `fill` | `instrument`, `side`, executed `quantity`, raw `price`, `fees`, optional `tax` |
| `action` | Sourced `CorporateAction` object under `event`, plus actual `entitled_quantity` |
| `allocation` | `event_id`, frozen `entitled_quantity`, observed `allocated_quantity`, exact `successor`; merger allocations also need sourced `tradable_at` and `tradability_source_id` |
| `cash_credit` | `event_id`, `gross_amount`, `tax_withheld`, `net_amount` |
| `cash_transfer` | Signed external deposit/withdrawal `amount`, retained separately from trading income |

The action's `cash_per_share` is the **net** account entitlement. Gross payment
minus observed withholding must equal the actual net credit and reconcile with
the entitlement. Taxes are not inferred. A payment due date creates an
obligation; cash is credited only when an actual `cash_credit` is supplied.
Overdue unobserved cash remains a blocker. Partial cash payments are deliberately
rejected by this schema; a split-payment schedule needs a separately versioned
record format rather than an assumed residual credit.

An issuer's conversion ratio does not supply an actual account allocation.
Separate registrar/account evidence is required for share conversion, including
integral theoretical entitlements. The old entitlement stays frozen until that
record is applied. Floor/ceiling bounds validate an observed allocation; they
are never a rule for choosing one. A late allocation receipt preserves the
original event, payment and tradability dates. Preexisting successor shares are
retained and the new allocation is added to them.

Unknown terminal proceeds, missing allocations, unknown successor tradability,
stale/missing marks, mismatched ending balances or incomplete coverage block
acceptance. The implementation rejects duplicate records, unsupported units,
overdrawn cash, sales exceeding raw holdings and fills that spend unresolved
shares. No live order is submitted.

## Interpretation of a passed report

`status: passed` means that the supplied, externally attested account window
reconciles. The receipt always records `scope: supplied_account_window`,
`archived_backtest_repaired: false`, and `production_promoted: false`.
Point-in-time data quality, the strategy's economic value, a corrected full
portfolio replay and prospective operations remain their own gates.

The output retains `protocol.json`, `assessment.json`, `completion.json` and
all referenced input bytes. Identical calls reuse the verified receipt; changed
inputs, implementation or tampering require a fresh output path. Later verifier
versions should explicitly migrate the protocol rather than rewriting an old
accounting receipt.

## Focused verification

```bash
PYTHONPATH=src python -m pytest -q tests/test_settlement.py tests/test_lifecycle.py
```

The tests independently calculate raw shares, cash, withholding and NAV; cover
late/duplicate allocation receipts, future and wrong-account evidence, adjusted
units, missing credits, independent closing mismatches and evidence tampering.
