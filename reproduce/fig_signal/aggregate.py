#!/usr/bin/env python3
"""Selected Figure 3 statistics from native FastT2T saved probability tensors.

Rank and conflict computations preserve the original MIT-licensed FastT2T
analysis. Only the two panels shown in the paper are emitted.
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path
import statistics
import numpy as np
import torch

REFERENCE = Path(__file__).resolve().parents[1] / "reference/fig_signal"


def read(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def write(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def from_captures(out):
    tsp_rows, mis_instances = [], defaultdict(list)
    for seed in (41, 42, 43, 44, 45):
        ranks = defaultdict(list)
        for task in ("tsp", "mis"):
            directory = out / "captures" / f"{task}_s{seed}" / "intermediates"
            files = sorted(directory.glob("intermediates_rank*_batch*.pt"))
            if not files:
                files = sorted(directory.glob("intermediates_batch*.pt"))
            if not files:
                raise ValueError(f"Missing capture: {directory}")
            for path in files:
                data = torch.load(path, map_location="cpu", weights_only=False)
                n = data["metadata"]["num_nodes"]
                if task == "mis":
                    src, dst = data["edge_index"].numpy()
                    valid = (src < n) & (dst < n) & (src != dst)
                    edges = np.unique(np.stack([np.minimum(src[valid], dst[valid]),
                                                np.maximum(src[valid], dst[valid])], axis=1), axis=0)
                for item in data["steps"]:
                    if item.get("x0_normal") is None or item.get("x0_perturbed") is None:
                        continue
                    step = int(item["step"])
                    if task == "tsp":
                        normal = item["x0_normal"][..., 1].squeeze(0)[:n, :n].clone()
                        delta = (item["x0_normal"][..., 1] - item["x0_perturbed"][..., 1]).squeeze(0)[:n, :n].numpy()
                        normal.fill_diagonal_(float("-inf"))
                        order = torch.argsort(normal, dim=1, descending=True).numpy()
                        for city in range(n):
                            for rank in range(n - 1):
                                label = str(rank) if rank < 3 else "rest"
                                ranks[(step, label)].append(float(delta[city, order[city, rank]]))
                    else:
                        normal = item["x0_normal"][..., 1].float().reshape(-1)[:n].numpy()
                        perturbed = item["x0_perturbed"][..., 1].float().reshape(-1)[:n].numpy()
                        delta = normal - perturbed
                        src, dst = edges[:, 0], edges[:, 1]
                        conflict = (normal[src] > 0.3) & (normal[dst] > 0.3)
                        gap = np.abs(delta[src] - delta[dst])
                        # Original CMA averages both groups on instances having conflict edges.
                        if conflict.any():
                            mis_instances[(seed, step)].append((float(gap[conflict].mean()), float(gap[~conflict].mean())))
        for (step, rank), values in sorted(ranks.items()):
            tsp_rows.append(dict(seed=seed, step=step, rank=rank, mean_delta=round(float(np.mean(values)), 6)))
    mis_seed, mis_all = [], []
    for (seed, step), values in sorted(mis_instances.items()):
        a = np.array(values)
        mis_seed.append(dict(seed=seed, step=step, threshold=0.3, n_instances_with_conflict=len(values),
                             delta_gap_conflict_mean=a[:, 0].mean(), delta_gap_nonconflict_mean=a[:, 1].mean()))
    for step in range(5):
        values = [v for (seed, s), vs in mis_instances.items() if s == step for v in vs]
        a = np.array(values)
        mis_all.append(dict(step=step, threshold=0.3, n_instances_with_conflict=len(values),
                            delta_gap_conflict_mean=a[:, 0].mean(), delta_gap_conflict_std=a[:, 0].std(),
                            delta_gap_nonconflict_mean=a[:, 1].mean(),
                            delta_gap_ratio_conflict_vs_nonconflict=a[:, 0].mean() / a[:, 1].mean()))
    return tsp_rows, mis_seed, mis_all


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--from-reference", action="store_true")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    source = "historical_reference" if args.from_reference else "new_run"
    if args.from_reference:
        tsp, mis_seed, mis = [read(REFERENCE / name) for name in
                              ("tsp_per_seed.csv", "mis_per_seed.csv", "mis_aggregate.csv")]
    else:
        tsp, mis_seed, mis = from_captures(args.output)
    rows, per_seed = [], []
    fields = dict(panel="", task="", step="", rank="", group="", threshold="", mean="", std="",
                  n_seeds=5, n_instances="", ratio="", signal_space="probability", source=source, status="completed")
    for step in (0, 3):
        for rank in ("0", "1", "2", "rest"):
            chosen = [r for r in tsp if int(r["step"]) == step and str(r["rank"]) == rank]
            assert sorted(int(r["seed"]) for r in chosen) == [41, 42, 43, 44, 45]
            values = [float(r["mean_delta"]) for r in chosen]
            rows.append(dict(fields, panel="a", task="tsp100", step=step, rank=rank,
                             mean=round(statistics.mean(values), 6), std=round(float(np.std(values)), 6)))
            for r in chosen:
                per_seed.append(dict(fields, panel="a", task="tsp100", step=step, rank=rank,
                                     seed=r["seed"], mean=r["mean_delta"], n_seeds=1))
    for r in mis:
        step = int(r["step"])
        if step not in (0, 4) or float(r["threshold"]) != 0.3:
            continue
        for group in ("conflict", "nonconflict"):
            rows.append(dict(fields, panel="b", task="mis_er", step=step, group=group, threshold=0.3,
                             mean=r[f"delta_gap_{group}_mean"], std=r.get(f"delta_gap_{group}_std", ""),
                             n_instances=r["n_instances_with_conflict"], ratio=r["delta_gap_ratio_conflict_vs_nonconflict"]))
            for s in mis_seed:
                if int(s["step"]) == step and float(s["threshold"]) == 0.3:
                    per_seed.append(dict(fields, panel="b", task="mis_er", step=step, group=group, threshold=0.3,
                                         seed=s["seed"], mean=s[f"delta_gap_{group}_mean"], n_seeds=1,
                                         n_instances=s["n_instances_with_conflict"]))
    assert len(rows) == 12 and len(per_seed) == 60
    assert all(np.isfinite(float(r["mean"])) for r in rows)
    write(args.output / "results.csv", rows)
    write(args.output / "per_seed.csv", per_seed)
    print(args.output / "results.csv")


if __name__ == "__main__":
    main()
