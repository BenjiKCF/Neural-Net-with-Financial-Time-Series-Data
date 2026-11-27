# Data availability, lifecycle accounting and paper execution

The roadmap adds explicit input contracts and an independent raw-share ledger.
This guide describes their interfaces, reproducible checks and the retained
[archived-holdings assessment](../reports/production_roadmap/lifecycle_transformer/assessment.json).
That assessment identifies sourced successors and unresolved raw-share/account
allocations. It does not establish repaired investment returns. The
[existing CSI300 accounting qualification](chenditc-benchmark.md#terminated-series-and-portfolio-valuation)
continues to apply.

## Availability and immutable refreshes

[`data_contracts.py`](../src/financial_ts/data_contracts.py) separates an
observation's market date from the time at which it could be used.
`FeatureAvailabilityContract` declares feature order, exchange timezone,
calendar identity, adjustment convention, revision policy, freshness and
missingness limits, and historical eligibility requirements.

`validate_feature_frame` accepts a numeric DataFrame with a unique
`datetime,instrument` index. Availability comes from an exactly aligned Series
of timezone-aware timestamps or a declared conservative release lag. Every row
must be available by the decision cutoff; the latest row for each expected
instrument must satisfy the session-based freshness limit. Feature reordering,
duplicate keys, unknown sessions, infinite values, excessive missingness and
unexpected membership reject the batch.

The observed calendar must extend through the decision cutoff's local date.
For a nontrading cutoff, include the following session so the runner can
establish which preceding session was current. A calendar truncated at the
last data row cannot prove freshness and is rejected. Extending the calendar
does not make future feature rows admissible; observations still stop at the
decision cutoff.

```python
from financial_ts.data_contracts import FeatureAvailabilityContract, validate_feature_frame

contract = FeatureAvailabilityContract(
    feature_order=("close", "volume"),
    availability_mode="explicit",
)
receipt = validate_feature_frame(
    features,
    contract,
    cutoff=decision_cutoff,
    observed_calendar=calendar,
    expected_instruments=historically_eligible,
    available_at=source_publication_times,
)
```

The application supplies the variables above from its admitted source and
decision schedule. The [daily template](../configs/data_contract_daily.json)
instead assumes availability at 08:00 on the next observed session and requires
raw OHLCV inputs. It is an explicit conservative assumption, not a certification
that a retrospective adjusted provider contains original publication times.
An adjusted input needs a separately appropriate contract. Learned processors
must still fit only within the allowed training period.

`audit_snapshot` compares a proposed snapshot with its predecessor. Identities
include the release, archive SHA256 and provider SHA256. It reports schema,
coverage, missingness, instrument changes, removed history and revised cells,
including adjustment factors. Unexplained historical changes quarantine the
refresh. A supported explanation identifies one date/instrument/column, its
source URL and reason. It does not excuse unrelated revisions.

```bash
fts snapshot-audit --current incoming.csv --identity incoming-identity.json \
  --previous previous.csv --previous-identity previous-identity.json \
  --output runs/snapshot-audits
```

The CSV keys are `datetime,instrument` followed by numeric feature columns.
`save_snapshot_audit` writes a content-addressed receipt under `admitted` or
`quarantined`. Retain the immutable old snapshot. Admission records these checks;
publication-time accuracy and permitted source use need their own evidence.
See [the source contract](../src/financial_ts/data_contracts.py) and
[refresh tests](../tests/test_data_contracts.py) for exact fields and failures.

## Corporate actions in raw share units

[`EventLedger`](../src/financial_ts/lifecycle.py) tracks cash, positions,
receivables, restrictions, unresolved events and chronological receipts. It
supports splits, dividends, mergers, ticker changes, suspensions, resumptions
and delistings with known or unavailable proceeds. A dividend creates an
entitlement at its declared effective time and enters cash at its payment time.
The caller supplies the actual entitlement and net amount; tax is not inferred.

Events carry a stable ID, source URL, known time and effective time. The ledger
rejects events unavailable at the requested cutoff. Retroactive corrections
after intervening ledger activity require a replay from history. Identical
events/fills are idempotent, while changed contents under a reused ID fail.
`reconcile` independently recomputes cash and quantities from event deltas.

Prices and corporate-action quantities must use consistent units. For a Qlib
adjusted position, `qlib_to_raw_position` uses the contemporaneous factor:

```text
implied raw quantity = adjusted quantity × source factor
raw quote = adjusted quote ÷ source factor
```

The conversion preserves that position's quoted value. It does not establish
originally executed shares, all intervening entitlements or final registrar
allocations. A later adjustment factor cannot substitute for the factor at the
position date. `PriceMark` rejects an adjusted price convention.

The [event catalog](../configs/lifecycle_events.json) links issuer disclosures
and records the qualification of each fixture. Announcement dates use a
conservative known-time policy where exact publication times were unavailable.
The catalog is a set of sourced fixtures, not a complete historical action feed.

Fractional merger entitlements remain unresolved until actual allocation
evidence is supplied through `resolve_allocation`. Successor shares also need
known tradability. Unknown-value delistings retain the old position and its
unresolved status. `mark` returns `nav: null` when a required raw mark or
settlement is unavailable. An indicative stale marking is labeled separately;
it is not a usable NAV or a fabricated last-price liquidation.

Valuation freshness requires a session calendar covering the cutoff date in
`Asia/Shanghai`. For a weekend or holiday cutoff, include the following session
so the calendar establishes the intervening nontrading dates. A truncated
calendar is rejected before it can make an old mark appear current.

## Assess an existing run without overwriting it

```bash
fts lifecycle-assess --run-dir runs/my-csi300/transformer \
  --provider .data/cn-20261126 --events configs/lifecycle_events.json \
  --output runs/my-lifecycle-assessment
```

Use a completed run and the provider that generated it. The assessment checks
provider identity, hashes source positions and events, and evaluates pre-event
holdings with contemporaneous factors. It writes to a separate directory.
Review unresolved allocations, raw-unit discrepancies and successor trading
dates before attempting a complete action-aware strategy replay. Merely finding
a successor instrument does not repair historical portfolio returns.

### Sourced terminated-series fixtures

The [event catalog](../configs/lifecycle_events.json) records these issuer
notices. Announcement dates were verified; `known_at` is conservatively the start
of the following local calendar day, because exact publication times were not
established.

| Former instrument | Successor | Final shares per old share | Termination date | Primary issuer disclosure |
| --- | --- | ---: | --- | --- |
| SH600837, Haitong Securities | SH601211 | 0.62 | 2025-03-04 | [Haitong 2025-026](https://static.cninfo.com.cn/finalpage/2025-03-01/1222675645.PDF) |
| SH601989, China Shipbuilding Industry | SH600150 | **0.1339** | 2025-09-05 | [Issuer 2025-062](https://static.cninfo.com.cn/finalpage/2025-08-30/1224625027.PDF) |
| SH601059, Cinda Securities | SH601995 | 0.5210 | 2026-10-19 | [Cinda 2026-060, issuer disclosure in China Securities Journal](https://epaper.cs.com.cn/zgzqb/html/2026-10/16/nw.D110000zgzqb_20261016_5-B017.htm) |

The final SH601989 ratio supersedes the earlier proposed 0.1335. Each notice
describes whole-share allocations across holders based on fractional remainders,
with a lottery for tied remainders. An isolated account therefore cannot simply
round a fractional entitlement. Such an event remains unresolved and the old
position remains frozen until `resolve_allocation` receives actual allocation
evidence. Unknown successor trading dates also remain unresolved; a sourced
resumption event can clear that restriction. Restrictions are conservative at
the instrument level, including where an account already held the successor.

### Retained assessment findings

`assess_archived_lifecycles` checks the supplied provider fingerprint against
the archived run, records the source-position and event-catalog hashes, and
evaluates the pre-event holdings with contemporaneous source factors. It writes
to a new directory and refuses to modify the archived run or provider.

The retained [assessment](../reports/production_roadmap/lifecycle_transformer/assessment.json)
reproduces **6.489845%** of the archived Transformer's final simulated NAV
in the three terminated holdings. SH600837 implies 196,600 raw quote units and
121,892 successor shares. The other two implied quantities are not integers;
the report retains those discrepancies. It flags account allocation, successor
tradability and the unexecuted complete action-aware strategy replay. No new
return figure is presented as a repaired backtest.

## Dated paper execution

[`PaperExecutor`](../src/financial_ts/execution.py) requires explicit
per-instrument `TradingRule` intervals with sources and known times. It rejects
missing or overlapping applicable rules. Board rules are not inferred from
ticker strings. Rule inputs specify order size/step, tick size, daily limits,
odd-lot liquidation and settlement behavior. The caller supplies the date's
raw reference price, including ex-action adjustments where relevant.

An order must exist before its execution bar. Suspensions, limit locks,
unsettled shares and insufficient cash produce inspectable rejections.
Participation limits accumulate across all orders for the same instrument and
session. Partial fills respect lots and minimum fees. The explicit impact
assumption uses a square-root participation curve; side-aware slippage and
adverse tick rounding then determine the simulated fill price. This daily-bar
model does not measure queue priority or establish market capacity.

The [scenario template](../configs/execution_scenarios.json) separates fee,
slippage and account-size assumptions. Its numbers are registered research
settings, not a verified brokerage tariff or historical rule database. Supply
the appropriate sourced rule/fee inputs for each new use.

## Durable replay and recovery

```bash
fts paper-replay --spec configs/paper_session_demo.json --output runs/paper-demo
fts paper-replay --spec configs/paper_session_demo.json --output runs/paper-demo
```

The [synthetic specification](../configs/paper_session_demo.json) declares an
initial raw-share account, observed calendar, dated rules, execution scenario
and chronological operations. Operations cover orders, corporate actions,
cash settlement and valuation. A session ends with an explicit valuation.

The runner records its protocol, source and runtime before executing operations.
A lock serializes duplicate runners, outputs are replaced atomically, and
`completion.json` is written last with hashes for the ledger, operations and
valuations. Identical reuse verifies the completed artifacts. Conflicting
specifications, changed runtime/source identity and altered files fail.
Interrupted sessions can be replayed from the registered initial account;
failed attempts retain their own receipts. No live order is submitted.

The retained [synthetic completion receipt](../reports/production_roadmap/paper_session_demo/completion.json)
records one partial and one full fill, with independently reconciled final
cash/NAV of CNY 9,975.105. Identical replay reused the completed receipt. These
are synthetic execution checks, not live fills or a deployment decision, and
remain tied to their original source/runtime identity.

## Verification and remaining acceptance

```bash
python -m pytest -q tests/test_data_contracts.py tests/test_lifecycle.py tests/test_execution.py
```

The tests cover hand-calculated events, missing values, fractional allocations,
stale/future marks, historical revisions, participation, settlement, cash and
duplicate recovery. Full historical account reconstruction still requires
complete action coverage, actual allocations, qualified prices, source arrival
records and a separately audited replay. See the
[implementation boundaries](production-implementation.md#remaining-acceptance-gates).
