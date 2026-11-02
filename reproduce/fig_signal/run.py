#!/usr/bin/env python3
"""Capture Figure 3 intermediates, or aggregate explicitly labelled historical data."""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_common"))
from run import ROOT, command_for, read_json, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--settings", type=Path, default=ROOT / "reproduce/local.json")
    p.add_argument("--from-reference", action="store_true", help="CSV check only; does not launch GPU runs")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    local = read_json(args.settings) if args.settings.exists() else {}
    python = local.get("python", {}).get("fastt2t")
    prefix = [python] if python else ["conda", "run", "--no-capture-output", "-n", "fastt2t", "python"]
    if not args.from_reference:
        import os
        import shlex
        import copy
        for task, table, config_id in [("tsp", "main_tsp", "fastt2t_tsp100_ts5_tg0"),
                                       ("mis", "main_mis", "fastt2t_miser_ts5_tg0")]:
            definition = read_json(ROOT / "reproduce" / table / "configs.json")
            config = copy.deepcopy(next(c for c in definition["configs"] if c["id"] == config_id))
            for seed in (41, 42, 43, 44, 45):
                unit = out / "captures" / f"{task}_s{seed}"
                marker = unit / "completed.json"
                if marker.exists():
                    continue
                if unit.exists() and not args.dry_run:
                    raise RuntimeError(f"Unfinished capture at {unit}; preserve it and use a new output directory")
                current = copy.deepcopy(config)
                current["args"] += ["--save_intermediates", "--intermediate_dir", "{out}/intermediates"]
                cmd, cwd = command_for(current, "pisg", seed, unit, local)
                command_dir = out / "plans" / f"{task}_s{seed}" if args.dry_run else unit
                command_dir.mkdir(parents=True, exist_ok=args.dry_run)
                write_json(command_dir / "command.json", {"argv": cmd, "cwd": str(cwd), "config": current, "seed": seed})
                (command_dir / "command.txt").write_text(shlex.join(cmd) + "\n")
                if args.dry_run:
                    print(shlex.join(cmd), flush=True)
                    continue
                env = dict(os.environ, WANDB_MODE="disabled", PYTHONUNBUFFERED="1")
                env["PYTHONPATH"] = str(cwd) + os.pathsep + env.get("PYTHONPATH", "")
                print(f"fig_signal: capturing {task} seed={seed}", flush=True)
                with (unit / "run.log").open("w") as log:
                    result = subprocess.run(cmd, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT)
                if result.returncode:
                    return result.returncode
                if not list((unit / "intermediates").glob("intermediates*.pt")):
                    raise RuntimeError(f"No captured intermediates at {unit}")
                write_json(marker, {"status": "completed", "seed": seed})
    if args.dry_run:
        return 0
    cmd = prefix + [str(Path(__file__).with_name("aggregate.py")), "--output", str(out)]
    if args.from_reference:
        cmd.append("--from-reference")
    return subprocess.run(cmd).returncode


if __name__ == "__main__":
    raise SystemExit(main())
