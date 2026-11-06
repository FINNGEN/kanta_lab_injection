#!/usr/bin/env python3
"""
release_check.py — check the primary injection's release files before they are published.

Reads injection_results.tsv and plot_data.json.gz from a release directory (default: the
repo's data/) and lists everything that breaks the rules in scripts/common/disclosure.py:

  * people counts under the minimum (they must be BELOW_THRESHOLD instead), rows shown for a
    group whose people are hidden;
  * deciles / percentiles of groups too small to show a distribution;
  * any row-level detail in the plot data (ECDF or KDE curves, histograms, fitted samples),
    or keys this checker does not know (a new field has to be added here on purpose).

Only the published files are read, so it also runs in CI. The people counts it checks are the
ones written by test_name_injection.py, which counts people from the person keys.

  python3 scripts/primary/release_check.py [DATA_DIR]     # exit 1 when anything is found
"""

import gzip
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import disclosure as dc   # noqa: E402

BIMODAL_KEYS = {"status", "separator", "bc", "dip_p", "overlap_pct", "winner", "linear", "log", "candidate"}
FIT_KEYS = {"lognormal", "bic", "sep_native", "components"}


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def check_results(path, problems):
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    for i, r in df.iterrows():
        where = f"{path.name} row {i + 2} ({r.get('TEST_NAME')}, {r.get('SUB_DIST')}, {r.get('UNIT')})"
        for n_col, p_col, dec_col in (("N_CANDIDATE", "N_CANDIDATE_SUBJECTS", "CAND_DECILES"),
                                      ("N_TARGET", "N_TARGET_SUBJECTS", "TARG_DECILES")):
            if p_col not in df.columns:
                problems.append(f"{path.name}: no {p_col} column (people are not counted)")
                return
            people, rows = r[p_col], r[n_col]
            dc.check_count(people, f"{where} {p_col}", problems)
            if (people == dc.BELOW) != (rows == dc.BELOW):
                problems.append(f"{where}: {n_col} and {p_col} must be hidden together")
            if r.get(dec_col, "NA") not in ("", "NA"):
                p, n = _num(people), _num(rows)
                if p is None or n is None or p < dc.MIN_DIST_SUBJECTS or n < dc.MIN_DIST_RECORDS:
                    problems.append(f"{where}: {dec_col} of {people} people / {rows} rows")


def check_entry(e, where, problems):
    """One comparison (an unambiguous test, or one engine section of an ambiguous test)."""
    dc.check_comparison(e, where, problems)


def check_plot_data(path, problems):
    with gzip.open(path, "rb") as fh:
        data = json.loads(fh.read())
    dc.check_tree(data, path.name, problems)
    for name, entry in data.items():
        where = f"{path.name}[{name}]"
        if "engine" in entry or "bimodal" in entry:                       # ambiguous test
            for k in set(entry) - {"engine", "bimodal"}:
                problems.append(f"{where}.{k}: unexpected key")
            for key, e in (entry.get("engine") or {}).items():
                check_entry(e, f"{where}.engine[{key}]", problems)
            bim = entry.get("bimodal")
            if bim:
                for k in set(bim) - BIMODAL_KEYS:
                    problems.append(f"{where}.bimodal.{k}: unexpected key")
                if "candidate" in bim:
                    dc.check_side(bim["candidate"], f"{where}.bimodal.candidate", problems)
                for space in ("linear", "log"):
                    fit = bim.get(space)
                    if fit:
                        for k in set(fit) - FIT_KEYS:
                            problems.append(f"{where}.bimodal.{space}.{k}: unexpected key")
                        for c in fit.get("components", []):
                            if set(c) - {"mean", "sigma", "weight"}:
                                problems.append(f"{where}.bimodal.{space}: unexpected component keys")
        else:                                                              # unambiguous test
            check_entry(entry, where, problems)


def check_release(data_dir):
    """All problems found in the release files of data_dir (empty list: fine to publish)."""
    data_dir = Path(data_dir)
    problems = []
    for fname, fn in (("injection_results.tsv", check_results), ("plot_data.json.gz", check_plot_data)):
        path = data_dir / fname
        if path.exists():
            fn(path, problems)
        else:
            problems.append(f"{path}: missing")
    return problems


def main():
    data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[2] / "data"
    problems = check_release(data_dir)
    for p in problems[:200]:
        print(p)
    if len(problems) > 200:
        print(f"... and {len(problems) - 200} more")
    print(f"{data_dir}: {len(problems)} problem(s)")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
