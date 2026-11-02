#!/usr/bin/env python3
"""Compare completed new table results with the separate historical references.

Min/max bands are engineering screens, not statistical confidence intervals.
Incomplete means are not marked as passing. Timing is not compared.
"""
import argparse
import csv
from decimal import Decimal
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[2]


def read(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True, help="outputs/<run-id> directory")
    a = p.parse_args()
    comparisons = []
    for results in sorted(a.output.glob("*/results.csv")):
        table = results.parent.name
        reference = ROOT / "reproduce/reference" / table
        if not (reference / "bands.csv").exists():
            continue
        summary = read(results)
        native = read(results.parent / "per_seed.csv")
        historical = read(reference / "per_seed.csv")
        bands = read(reference / "bands.csv")
        for r in summary:
            config, setting = r["config_id"], r["setting"]
            selected = [s for s in native if s["config_id"] == config and s["setting"] == setting and s["status"] == "completed"]
            seeds = {s["seed"] for s in selected}
            for metric in ("cost", "gap_pct"):
                match = next((b for b in bands if b["config_id"] == config and b["setting"] == setting and b["metric"] == metric), None)
                if not match:
                    continue
                refs = [s for s in historical if s["config_id"] == config and s["setting"] == setting and s[metric]]
                precision = max(float(Decimal(10) ** Decimal(s[metric]).as_tuple().exponent) / 2 for s in refs)
                same = [float(s[metric]) for s in refs if s["seed"] in seeds]
                value = float(r[metric + "_mean"]) if r.get(metric + "_mean") else None
                low, high = float(match["min"]), float(match["max"])
                status = ("pending" if r["status"] != "completed" else "missing_metric" if value is None else
                          "within_reference_band" if low - precision <= value <= high + precision else "outside_reference_band")
                comparisons.append(dict(table=table, config_id=config, setting=setting, metric=metric,
                                        new_mean=value if value is not None else "", historical_same_seed_mean=statistics.mean(same) if same else "",
                                        historical_five_seed_mean=match["mean"], historical_min=low, historical_max=high,
                                        rounding_allowance=precision, n_completed=r["n_completed"], n_expected=r["n_expected"], status=status))
        for config in sorted({r["config_id"] for r in summary}):
            info = next(r for r in summary if r["config_id"] == config)
            metric = "gap_pct" if info["task"] == "tsp" else "cost"
            band = next((b for b in bands if b["config_id"] == config and b["setting"] == "paired_delta" and b["metric"] == metric), None)
            if not band:
                continue
            component = [c for c in comparisons if c["table"] == table and c["config_id"] == config and c["metric"] == metric]
            allowance = sum(c["rounding_allowance"] for c in component)
            value = float(info["delta_mean"]) if info["delta_mean"] else None
            complete = all(r["status"] == "completed" for r in summary if r["config_id"] == config)
            low, high = float(band["min"]), float(band["max"])
            selected = set(info["paired_seeds"].split())
            by_seed = {}
            for r in historical:
                if r["config_id"] == config and r["seed"] in selected and r[metric]:
                    by_seed.setdefault(r["seed"], {})[r["setting"]] = float(r[metric])
            same = [(r["baseline"] - r["pisg"]) if metric == "gap_pct" else (r["pisg"] - r["baseline"])
                    for r in by_seed.values() if set(r) == {"baseline", "pisg"}]
            status = "pending" if not complete else "missing_metric" if value is None else "within_reference_band" if low - allowance <= value <= high + allowance else "outside_reference_band"
            comparisons.append(dict(table=table, config_id=config, setting="paired_delta", metric=metric,
                                    new_mean=value if value is not None else "", historical_same_seed_mean=statistics.mean(same) if same else "",
                                    historical_five_seed_mean=band["mean"], historical_min=low, historical_max=high,
                                    rounding_allowance=allowance, n_completed=info["n_pairs"], n_expected=info["n_expected"], status=status))
    if not comparisons:
        raise ValueError("No comparable table CSVs found")
    target = a.output / "comparison.csv"
    with target.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(comparisons[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(comparisons)
    print(target)


if __name__ == "__main__":
    main()
