#!/usr/bin/env python3
"""Reparse existing successful solver logs and rebuild CSVs without rerunning GPUs.

Run only after the controller has exited, so its in-memory ledger cannot
overwrite the repaired state. A nonzero solver exit remains a failed run.
"""
import argparse
from pathlib import Path
from run import aggregate, parse_metrics, read_json, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="outputs/<run-id> directory")
    args = parser.parse_args()
    for state_path in sorted(args.output.glob("*/run.json")):
        state = read_json(state_path)
        if any(r["status"] == "running" for r in state["runs"]):
            raise RuntimeError(f"Controller may still be running: {state_path}")
        for r in state["runs"]:
            if r.get("exit_code") != 0 or r["status"] not in {"invalid", "completed"}:
                continue
            config = next(c for c in state["configurations"] if c["id"] == r["config_id"])
            unit = args.output / r["log"]
            unit = unit.parent
            try:
                metrics = parse_metrics(config, unit)
                write_json(unit / "metrics.json", metrics)
                r.update({k: v for k, v in metrics.items() if k != "native_metrics"})
                r.update(status="completed", error="")
                write_json(unit / "run.json", r)
            except (ValueError, KeyError, OSError) as exc:
                r.update(status="invalid", error=str(exc))
        state["status"] = "completed" if all(r["status"] == "completed" for r in state["runs"]) else "failed"
        write_json(state_path, state)
        aggregate(state["configurations"], state["runs"], state_path.parent)
        print(state_path.parent / "results.csv")


if __name__ == "__main__":
    main()
