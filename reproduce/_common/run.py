#!/usr/bin/env python3
"""Run the selected paper configurations in their native solver environments."""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
TABLES = ("main_tsp", "main_mis", "fig_signal", "gensco_tsp", "gensco_mis", "deployment")


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def write_csv(path, rows, fields):
    with Path(path).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def command_for(config, setting, seed, out, local):
    solver = config["solver"]
    python = local.get("python", {}).get(solver)
    prefix = [python] if python else ["conda", "run", "--no-capture-output", "-n", solver, "python"]
    source = Path(local.get("source", {}).get(solver, ROOT / "solvers" / solver))
    if solver == "gensco" and solver not in local.get("source", {}):
        source /= "upstream"
    assets = Path(local.get("resources", {}).get(solver, ROOT / "resources" / solver)).resolve()
    tokens = config["args"] + config.get(setting + "_args", [])
    replacements = {"{assets}": str(assets), "{seed}": str(seed),
                    "{out}": str(out), "{root}": str(ROOT)}
    args = []
    for token in tokens:
        for old, new in replacements.items():
            token = str(token).replace(old, new)
        args.append(token)
    if solver == "gensco" and config["task"] == "mis":
        cmd = prefix + [str(ROOT / "reproduce" / "_common" / "gensco_mis.py")] + args
    else:
        entry = {"difusco": ["difusco/train.py"], "t2t": ["diffusion/train.py"],
                 "fastt2t": ["train.py"], "strudico": ["train.py"],
                 "gensco": ["-m", "decoding.tsp"]}[solver]
        cmd = prefix + entry + args
    return cmd, source


def parse_metrics(config, out):
    metrics = {}
    path = out / "metrics.raw.json"
    if path.exists():
        raw = read_json(path)
        metrics = raw.get("metrics", {})
        if "test" in raw:
            metrics = raw["test"][0]
    text = (out / "run.log").read_text(errors="replace")
    text = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", text).replace("│", " ")
    for key, value in re.findall(r"(?:^|\n)\s*(test/[\w_]+)\s+([-+\d.eE]+)", text):
        metrics[key] = float(value)
    if config["solver"] == "gensco":
        if config["task"] == "tsp":
            costs = re.findall(r"^mean cost:\s*([-+\d.eE]+)", text, re.M)
            gaps = re.findall(r"^Gap:\s*([-+\d.eE]+)\s*%", text, re.M)
            if len(costs) != 1 or len(gaps) != 1:
                raise ValueError("Expected one GenSCO cost and gap")
            result = {"cost": float(costs[0]), "gap_pct": float(gaps[0])}
        else:
            sizes = re.findall(r"^Avg Size:\s*([-+\d.eE]+)", text, re.M)
            if len(sizes) != 1:
                raise ValueError("Expected one GenSCO Avg Size")
            result = {"cost": float(sizes[0]), "gap_pct": None}
    else:
        final = config.get("tg", 0) > 0
        cost_keys = (["test/final_solved_cost", "test/guided_solved_cost"] if final else []) + ["test/solved_cost"]
        gap_keys = (["test/final_gap", "test/guided_gap"] if final else []) + ["test/gap"]
        result = {"cost": next((metrics[k] for k in cost_keys if k in metrics), None),
                  "gap_pct": next((metrics[k] for k in gap_keys if k in metrics), None)}
        if config["solver"] == "difusco" and result["gap_pct"] is None:
            # DIFUSCO reports mean solved/ground-truth costs, not a gap metric.
            # The paper's extraction uses the ratio of these means.
            gt = metrics.get("test/gt_cost")
            if gt and result["cost"] is not None:
                direction = 1 if config["task"] == "tsp" else -1
                result["gap_pct"] = direction * 100 * (result["cost"] / gt - 1)
        if config["task"] == "mis" and config["solver"] == "fastt2t" and result["gap_pct"] is not None:
            result["gap_pct"] *= -1
        result["initial_cost"] = metrics.get("test/solved_cost")
        result["initial_gap_pct"] = metrics.get("test/gap")
        if config["task"] == "mis" and config["solver"] == "fastt2t" and result["initial_gap_pct"] is not None:
            result["initial_gap_pct"] *= -1
        if config["solver"] == "difusco":
            result["initial_gap_pct"] = result["gap_pct"]
    if result["cost"] is None or not math.isfinite(float(result["cost"])):
        raise ValueError("Missing or non-finite cost")
    if not (config["solver"] == "gensco" and config["task"] == "mis") and result["gap_pct"] is None:
        raise ValueError("Missing gap")
    for value in result.values():
        if value is not None and not math.isfinite(float(value)):
            raise ValueError("Non-finite metric")
    result["native_metrics"] = metrics
    return result


def aggregate(configs, ledger, out):
    rows = []
    summary = []
    for config in configs:
        records = [r for r in ledger if r["config_id"] == config["id"]]
        for record in records:
            rows.append({**config, **record, "task_size": config["size"]})
        good = [r for r in records if r["status"] == "completed"]
        pairs = {}
        for r in good:
            pairs.setdefault(r["seed"], {})[r["setting"]] = r
        pairs = {s: p for s, p in pairs.items() if set(p) == {"baseline", "pisg"}}
        target = "gap_pct" if config["task"] == "tsp" else "cost"
        deltas = [(p["baseline"][target] - p["pisg"][target]) if target == "gap_pct"
                  else (p["pisg"][target] - p["baseline"][target]) for p in pairs.values()]
        for setting in ("baseline", "pisg"):
            selected = [r for r in good if r["setting"] == setting]
            n_expected = sum(r["setting"] == setting for r in records)
            row = {"config_id": config["id"], "solver": config["solver"], "task": config["task"],
                   "task_size": config["size"], "ts": config.get("ts", ""), "tg": config.get("tg", ""),
                   "cycles": config.get("cycles", ""), "two_opt": config.get("two_opt", 0),
                   "setting": setting, "seeds": " ".join(str(r["seed"]) for r in selected),
                   "n_completed": len(selected), "n_expected": n_expected,
                   "status": "completed" if len(selected) == n_expected else "incomplete",
                   "paired_seeds": " ".join(map(str, sorted(pairs))), "n_pairs": len(pairs),
                   "delta_mean": statistics.mean(deltas) if deltas else "",
                   "wins": sum(d > 0 for d in deltas) if deltas else ""}
            for key in ("cost", "gap_pct", "initial_cost", "initial_gap_pct"):
                values = [r[key] for r in selected if r.get(key) is not None]
                row[key + "_mean"] = statistics.mean(values) if values else ""
                row[key + "_std"] = statistics.stdev(values) if len(values) > 1 else ""
            summary.append(row)
    write_csv(out / "per_seed.csv", rows, ["config_id", "solver", "task", "task_size", "ts", "tg", "cycles",
              "two_opt", "setting", "seed", "status", "cost", "gap_pct", "initial_cost", "initial_gap_pct",
              "attempt", "log", "error"])
    fields = ["config_id", "solver", "task", "task_size", "ts", "tg", "cycles", "two_opt", "setting",
              "seeds", "n_completed", "n_expected", "status", "cost_mean", "cost_std", "gap_pct_mean",
              "gap_pct_std", "initial_cost_mean", "initial_gap_pct_mean", "paired_seeds", "n_pairs", "delta_mean", "wins"]
    write_csv(out / "results.csv", summary, fields)


def run_table(table, args, local, base_out):
    out = base_out / table
    out.mkdir(parents=True, exist_ok=True)
    if table == "fig_signal":
        if args.mode == "sample":
            print("fig_signal: no GPU sample is scheduled", flush=True)
            return True
        cmd = [sys.executable, str(ROOT / "reproduce" / table / "run.py"),
               "--output", str(out), "--settings", str(args.settings)]
        if args.dry_run:
            cmd.append("--dry-run")
        return subprocess.run(cmd).returncode == 0
    definition = read_json(ROOT / "reproduce" / table / "configs.json")
    configs = [c for c in definition["configs"] if args.mode == "full" or c.get("sample")]
    if args.only:
        configs = [c for c in configs if c["id"] in args.only]
        if not configs:
            raise ValueError("No selected configurations in " + table)
    seeds = args.seeds or definition[args.mode + "_seeds"]
    digest = hashlib.sha256(json.dumps([configs, seeds], sort_keys=True).encode()).hexdigest()
    state_path = out / "run.json"
    previous = read_json(state_path) if state_path.exists() else None
    if previous and previous["protocol_sha256"] != digest:
        raise ValueError("Output already belongs to a different protocol; use a new run ID")
    ledger = previous["runs"] if previous else [
        {"config_id": c["id"], "setting": setting, "seed": seed, "status": "pending", "attempt": 0}
        for c in configs for seed in seeds for setting in ("baseline", "pisg")]
    state = {"table": table, "mode": args.mode, "protocol_sha256": digest, "configurations": configs,
             "seeds": seeds, "status": "running", "runs": ledger}
    write_json(state_path, state)
    for record in ledger:
        if record["status"] == "completed":
            continue
        if record["status"] in {"failed", "invalid"} and not args.retry_failed:
            continue
        if record["attempt"] >= 2:
            continue
        config = next(c for c in configs if c["id"] == record["config_id"])
        attempt = record["attempt"] + 1
        unit = out / "runs" / f'{config["id"]}_{record["setting"]}_s{record["seed"]}_a{attempt}'
        unit.mkdir(parents=True, exist_ok=True)
        cmd, cwd = command_for(config, record["setting"], record["seed"], unit, local)
        write_json(unit / "command.json", {"argv": cmd, "cwd": str(cwd), "config": config,
                   "setting": record["setting"], "seed": record["seed"], "environment":
                   {k: os.environ[k] for k in ("CUDA_VISIBLE_DEVICES", "XLA_PYTHON_CLIENT_PREALLOCATE") if k in os.environ}})
        (unit / "command.txt").write_text(shlex.join(cmd) + "\n")
        if args.dry_run:
            record["status"] = "planned"
            print(shlex.join(cmd), flush=True)
            continue
        record.update(status="running", attempt=attempt, log=str((unit / "run.log").relative_to(base_out)))
        write_json(state_path, state)
        print(f'{table}: {config["id"]} {record["setting"]} seed={record["seed"]} attempt={attempt}', flush=True)
        env = dict(os.environ, WANDB_MODE="disabled", PYTHONUNBUFFERED="1")
        env["PYTHONPATH"] = str(cwd) + os.pathsep + env.get("PYTHONPATH", "")
        env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
        try:
            with (unit / "run.log").open("w") as log:
                p = subprocess.run(cmd, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT)
            record["exit_code"] = p.returncode
            if p.returncode:
                record.update(status="failed", error=f"solver exit code {p.returncode}")
            else:
                result = parse_metrics(config, unit)
                write_json(unit / "metrics.json", result)
                record.update({k: v for k, v in result.items() if k != "native_metrics"})
                record.update(status="completed", error="")
        except (OSError, ValueError, KeyError) as exc:
            record.update(status="invalid", error=str(exc))
        except KeyboardInterrupt:
            record.update(status="interrupted", error="Interrupted")
            state["status"] = "interrupted"
            write_json(state_path, state)
            aggregate(configs, ledger, out)
            raise
        write_json(unit / "run.json", record)
        write_json(state_path, state)
        aggregate(configs, ledger, out)
        print(f'  {record["status"]}: cost={record.get("cost", "")} gap={record.get("gap_pct", "")}', flush=True)
    aggregate(configs, ledger, out)
    done = all(r["status"] == "completed" for r in ledger)
    state["status"] = "planned" if args.dry_run else "completed" if done else "failed"
    write_json(state_path, state)
    print(out / "results.csv", flush=True)
    return done or args.dry_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", choices=TABLES + ("all",), default="all")
    parser.add_argument("--mode", choices=("sample", "full"), default="sample")
    parser.add_argument("--run-id", default=dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    parser.add_argument("--settings", type=Path, default=ROOT / "reproduce" / "local.json")
    parser.add_argument("--only", nargs="+")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--retry-failed", action="store_true", help="One retry; prior attempt logs are preserved")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.run_id):
        parser.error("run-id must be a single directory name")
    local = read_json(args.settings) if args.settings.exists() else {}
    tables = TABLES if args.table == "all" else (args.table,)
    out = ROOT / "outputs" / args.run_id
    out.mkdir(parents=True, exist_ok=True)
    results = [run_table(table, args, local, out) for table in tables]
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
