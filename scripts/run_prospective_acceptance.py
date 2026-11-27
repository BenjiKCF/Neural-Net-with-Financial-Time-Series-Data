"""Preregister, observe and assess real-clock prospective shadow forecasts."""

import argparse
import json
from pathlib import Path

from financial_ts.bundles import atomic_json
from financial_ts.prospective import (
    evaluate_prospective,
    record_mature_labels,
    record_observation,
    register_prospective,
    run_prospective_day,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, required=True)
    sub = parser.add_subparsers(dest="command", required=True)
    register = sub.add_parser("register")
    register.add_argument("--spec", type=Path, required=True)
    register.add_argument("--candidate-bundle", type=Path, required=True)
    register.add_argument("--reference-bundle", type=Path, required=True)
    observe = sub.add_parser("capture")
    observe.add_argument("--candidate-artifact", type=Path, required=True)
    observe.add_argument("--reference-artifact", type=Path, required=True)
    observe.add_argument("--arrival-receipt", type=Path, required=True)
    run = sub.add_parser("run-day")
    run.add_argument("--provider", type=Path, required=True)
    run.add_argument("--arrival-receipt", type=Path, required=True)
    labels = sub.add_parser("labels")
    labels.add_argument("--session", required=True)
    labels.add_argument("--csv", type=Path, required=True)
    labels.add_argument("--source", required=True)
    status = sub.add_parser("status")
    status.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.command == "register":
        result = register_prospective(
            args.store, json.loads(args.spec.read_text()),
            candidate_bundle=args.candidate_bundle, reference_bundle=args.reference_bundle,
        )
        result = {key: result[key] for key in ("protocol_id", "registered_at", "production_ready", "reused")}
    elif args.command == "capture":
        result = record_observation(
            args.store, candidate_artifact=args.candidate_artifact,
            reference_artifact=args.reference_artifact, arrival_receipt=args.arrival_receipt,
        )
    elif args.command == "run-day":
        result = run_prospective_day(
            args.store, provider=args.provider, arrival_receipt=args.arrival_receipt,
        )
    elif args.command == "labels":
        result = record_mature_labels(
            args.store, session=args.session, labels=args.csv, source=args.source,
        )
    else:
        result = evaluate_prospective(args.store)
        if args.output is not None:
            atomic_json(args.output, result)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
