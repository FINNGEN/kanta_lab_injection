#!/usr/bin/env python3
"""
release_check.py — check the unharmonized suggestions' release files before they are published.

Reads groups.tsv, candidates.tsv and plot_data.json.gz from a release directory (default: the
repo's data/unharmonized/) and lists everything that breaks the rules in
scripts/common/disclosure.py (see suggest.py's privacy notes):

  * group people (N_SUBJECTS) under the minimum, a concept's harmonized rows / people published
    under the minimum, rows with a range (N_RANGED) published at all;
  * counts written into the reason texts ("… rows");
  * row-level detail in the plot data (curves, histograms), unknown keys, percentiles of groups too
    small to show a distribution, range pairs / history paths with fewer rows than the minimum.

Only the published files are read, so it also runs in CI; the people counts it checks are the ones
suggest.py wrote from the person keys.

  python3 scripts/unharmonized/release_check.py [DIR] [--min-target-n 100]   # exit 1 on problems
"""

import argparse
import gzip
import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import disclosure as dc   # noqa: E402

GROUP_KEYS = {"group", "ranges", "candidates", "concepts", "context", "picked", "reason"}
CAND_KEYS = {"unit", "cf", "omop", "target_unit", "same", "plot"}
CONCEPT_KEYS = {"target_unit", "name", "n_target", "clusters"}
CLUSTER_KEYS = {"LOWER", "UPPER", "SHARE"}
ROW_COUNT = re.compile(r"\d[\d,]*\s+rows?\b")


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def check_groups(path, problems, min_target_n):
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    if "N_RANGED" in df.columns:
        problems.append(f"{path.name}: N_RANGED must not be released")
    if "N_SUBJECTS" not in df.columns:
        problems.append(f"{path.name}: no N_SUBJECTS column (people are not counted)")
        return
    for i, r in df.iterrows():
        where = f"{path.name} row {i + 2} ({r.get('TEST_NAME')} · {r.get('SOURCE_UNIT')})"
        dc.check_count(r["N_SUBJECTS"], f"{where} N_SUBJECTS", problems)
        if r["N_SUBJECTS"] in ("", "NA", dc.BELOW):
            problems.append(f"{where}: a released group must show its people")
        n_t, p_t = _num(r.get("N_TARGET")), r.get("N_TARGET_SUBJECTS", "")
        if r.get("N_TARGET", "") != "" and (n_t is None or n_t < min_target_n or p_t == ""):
            problems.append(f"{where}: N_TARGET {r.get('N_TARGET')} published (minimum {min_target_n} rows, with people)")
        dc.check_count(p_t, f"{where} N_TARGET_SUBJECTS", problems)
        for col in ("REASON", "NEEDS"):
            if ROW_COUNT.search(str(r.get(col, ""))):
                problems.append(f"{where}: a row count in {col}: {ROW_COUNT.search(str(r[col])).group(0)!r}")


def check_plot_data(path, problems, min_target_n):
    with gzip.open(path, "rb") as fh:
        data = json.loads(fh.read())
    dc.check_tree(data, path.name, problems)
    for gid, g in data.items():
        where = f"{path.name}[{gid}]"
        for k in set(g) - GROUP_KEYS:
            problems.append(f"{where}.{k}: unexpected key")
        if g.get("group") is not None:
            dc.check_side(g["group"], f"{where}.group", problems)
        for cid, c in (g.get("candidates") or {}).items():
            for k in set(c) - CAND_KEYS:
                problems.append(f"{where}.candidates[{cid}].{k}: unexpected key")
            if c.get("plot") is not None:
                dc.check_comparison(c["plot"], f"{where}.candidates[{cid}].plot", problems)
        for o, con in (g.get("concepts") or {}).items():
            for k in set(con) - CONCEPT_KEYS:
                problems.append(f"{where}.concepts[{o}].{k}: unexpected key")
            n = con.get("n_target")
            if n not in (None, dc.BELOW) and (_num(n) is None or _num(n) < min_target_n):
                problems.append(f"{where}.concepts[{o}].n_target {n} (minimum {min_target_n})")
            for c in con.get("clusters") or []:
                if set(c) - CLUSTER_KEYS:
                    problems.append(f"{where}.concepts[{o}]: cluster keys {sorted(set(c) - CLUSTER_KEYS)}")
        for p in g.get("ranges") or []:
            if len(p) != 3 or _num(p[2]) is None or _num(p[2]) < dc.MIN_SUBJECTS:
                problems.append(f"{where}.ranges: {p}")
        for h in (g.get("context") or {}).get("history") or []:
            if _num(h[-1]) is None or _num(h[-1]) < dc.MIN_SUBJECTS:
                problems.append(f"{where}.context.history: {h}")
        if ROW_COUNT.search(str(g.get("reason", ""))):
            problems.append(f"{where}.reason: a row count")


def check_release(rel_dir, min_target_n=100):
    rel_dir = Path(rel_dir)
    problems = []
    for name in ("groups.tsv", "decidable.tsv", "not_decidable.tsv"):
        if (rel_dir / name).exists():
            check_groups(rel_dir / name, problems, min_target_n)
    path = rel_dir / "plot_data.json.gz"
    if path.exists():
        check_plot_data(path, problems, min_target_n)
    else:
        problems.append(f"{path}: missing")
    return problems


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("dir", nargs="?", type=Path,
                    default=Path(__file__).resolve().parents[2] / "data" / "unharmonized")
    ap.add_argument("--min-target-n", type=int, default=100)
    args = ap.parse_args()
    problems = check_release(args.dir, args.min_target_n)
    for p in problems[:200]:
        print(p)
    if len(problems) > 200:
        print(f"... and {len(problems) - 200} more")
    print(f"{args.dir}: {len(problems)} problem(s)")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
