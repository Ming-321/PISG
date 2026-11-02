#!/usr/bin/env python3
"""Check one prepared resource snapshot against the recorded relative-path hashes."""
import argparse
import csv
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--solver", choices=("difusco", "t2t", "fastt2t", "strudico", "gensco"), required=True)
    p.add_argument("--root", type=Path)
    a = p.parse_args()
    base = a.root or ROOT / "resources" / a.solver
    with (ROOT / "docs/resources.csv").open() as f:
        rows = [r for r in csv.DictReader(f) if r["solver"] == a.solver]
    failures = 0
    for row in rows:
        path = base / row["path"]
        if not path.is_file():
            print(f"missing: {row['path']}")
            failures += 1
            continue
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
                h.update(chunk)
        if path.stat().st_size != int(row["size_bytes"]) or h.hexdigest() != row["sha256"]:
            print(f"mismatch: {row['path']}")
            failures += 1
    expected_graphs = {row["path"] for row in rows if row["path"].endswith("gpickle")}
    for directory in sorted({str(Path(path).parent) for path in expected_graphs}):
        actual_graphs = {str(path.relative_to(base)) for path in (base / directory).glob("*gpickle")}
        for extra in sorted(actual_graphs - expected_graphs):
            print(f"extra evaluation graph: {extra}")
            failures += 1
    print(f"{a.solver}: checked {len(rows)} files, {failures} missing or mismatched")
    return bool(failures)


if __name__ == "__main__":
    raise SystemExit(main())
