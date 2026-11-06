#!/usr/bin/env python3
"""
ref_primary_injection.py  —  version 1: reference ranges are the decision maker.

Same tests and categories as scripts/primary/test_name_injection.py (unit-less rows of each
TEST_NAME; unambiguous / ambiguous by top-unit prevalence), but the unit is decided from
the reference ranges instead of the value distributions:

1. Per test, learn each candidate unit's range profile (median lower / median upper over
   the unit's own rows; units = top 3 with > 1 %, as the value engine) and group units with
   the same profile into scale groups (scripts_ref/common/ref_ranges.py).
2. Label every unit-less row that has a range with the scale group its range matches, and
   confirm it with the row's value.
3. Decide the test (scripts_ref/common/ref_decide.py): SINGLE (one group), MIXED (several:
   one unit per group plus a value cutoff learned from the labelled rows), or, when the
   ranges cannot decide (NO_EVIDENCE, REVIEW), fall back to the current value engine.

Outputs (in --out-dir)
----------------------
  ref_unit_profiles.tsv     per test x unit: median lower / upper limit (+ p05/p95, rows,
                            scale group) — the per-unit reference a row's own range can be
                            compared with downstream
  ref_range_rules.tsv       per test x distinct candidate range (lower, upper): rows, the unit
                            it points to, value-check counts (groups of >= --min-rule-rows)
  ref_primary_results.tsv   per test x SUB_DIST, in the injection_results.tsv shape
                            (SUB_DIST, CUTOFF, UNIT, OUTCOME) plus DECIDED_BY (REF / VALUE)
                            and the REF_* evidence columns
  ref_primary_summary.md    decisions by category
  plots/ref_<test>.png      with --plots: values by range label + range clusters

Usage
-----
  python3 scripts_ref/primary/ref_primary_injection.py --out-dir OUT [--plots]            # latest run in /mnt/disks/data/kanta/engine
  python3 scripts_ref/primary/ref_primary_injection.py --input ENGINE.parquet --test b-hkr
  (synthetic data: add --backend local)
"""

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "common"))

import numpy as np    # noqa: E402
import pandas as pd   # noqa: E402

import ref_data as rd      # noqa: E402
import ref_decide as rdec  # noqa: E402
import ref_ranges as rr    # noqa: E402

RESULT_COLS = [
    "TEST_NAME", "TYPE", "DECIDED_BY", "REF_DECISION", "SUB_DIST", "CUTOFF", "UNIT", "OUTCOME",
    "SCALE_GROUP_UNITS", "GROUP_SHARE", "CUTOFF_ACC", "CUTOFF_BAL_ACC", "CUTOFF_RELIABLE",
    "N_CANDIDATE", "N_RANGED", "REF_COVERAGE", "N_DECIDED", "REF_SHARES", "REF_NO_MATCH",
    "REF_STATUS", "SCALE_GROUPS", "MEDIAN_LOWER", "MEDIAN_UPPER",
    "UNIT_PREVALENCE", "PREVALENCE_DICT", "VALUE_TESTS_PASSED", "NOTES",
]
RULE_COLS = ["TEST_NAME", "LOWER", "UPPER", "N", "N_DECIDED", "UNIT", "UNIT_SHARE", "STATUS",
             "DIST_MEDIAN", "VALUE_MEDIAN", "N_VALUE_OVERRIDE", "N_VALUE_CONFLICT"]


def range_rules(name, cand, res, min_rows=5):
    lo, hi = rr.valid_limits(cand["LOWER"].to_numpy(dtype=float), cand["UPPER"].to_numpy(dtype=float))
    status, group, rep = res["status"], res["group"], res["group_rep"]
    ranged = np.isfinite(lo) | np.isfinite(hi)
    if not ranged.any():
        return pd.DataFrame(columns=RULE_COLS)
    dec = np.isin(status, rr.DECIDED) & (group >= 0)
    df = pd.DataFrame({
        "LOWER": lo[ranged], "UPPER": hi[ranged], "STATUS": status[ranged],
        "UNIT": np.where(dec, np.array([rep.get(g, "") for g in group], dtype=object), "")[ranged],
        "DEC": dec[ranged], "DIST": res["dist"][ranged], "VALUE": cand["VALUE"].to_numpy()[ranged],
    })
    rows = []
    for (l, u), g in df.groupby(["LOWER", "UPPER"], dropna=False):
        if len(g) < min_rows:
            continue
        d = g[g["DEC"]]
        top = d["UNIT"].value_counts()
        rows.append(dict(
            TEST_NAME=name, LOWER=l, UPPER=u, N=len(g), N_DECIDED=len(d),
            UNIT=top.index[0] if len(top) else "",
            UNIT_SHARE=top.iloc[0] / len(d) if len(top) else np.nan,
            STATUS=g["STATUS"].value_counts().index[0],
            DIST_MEDIAN=float(np.nanmedian(g["DIST"])) if np.isfinite(g["DIST"]).any() else np.nan,
            VALUE_MEDIAN=float(np.median(g["VALUE"])),
            N_VALUE_OVERRIDE=int((g["STATUS"] == "VALUE_OVERRIDE").sum()),
            N_VALUE_CONFLICT=int((g["STATUS"] == "VALUE_CONFLICT").sum()),
        ))
    return pd.DataFrame(rows, columns=RULE_COLS).sort_values("N", ascending=False)


def run_primary(backend, plot_name, details, args, keep_rows=False, log_path=None, plots_dir=None):
    """
    Returns dict: results, profiles, rules (DataFrames) and, with keep_rows, rows:
    name -> DataFrame of the candidate rows with PRED_UNIT / PRED_SOURCE / STATUS.
    """
    prev_dict = dict(zip(details["TEST_NAME"], details["PREVALENCE_DICT"]))
    tests = plot_name[plot_name["CATEGORY"].isin(["UNAMBIGUOUS", "AMBIGUOUS"])]
    results, profiles, rules, rows_out, fallback = [], [], [], {}, []
    state = {}

    for i, (name, cat) in enumerate(zip(tests["TEST_NAME"], tests["CATEGORY"]), 1):
        typ   = cat.lower()
        prev  = prev_dict.get(name, "NA")
        units = rd.units_to_consider(prev)
        cand, res = rd.analyse(backend, name, units, args)
        dec = rdec.ref_decision(cand["VALUE"].to_numpy(), res, args)
        info = rdec.ref_info(dec, res)
        print(f"[{i:>4}/{len(tests)}] {name:<40} {typ:<11} {dec['decision']:<11} "
              f"N={dec['n']:,} ranged={info['REF_COVERAGE']:.0%} decided={dec['n_decided']:,} "
              f"{info['REF_SHARES']}", flush=True)

        p = res["profiles"].copy()
        p.insert(0, "TYPE", typ)
        p.insert(0, "TEST_NAME", name)
        profiles.append(p)
        rules.append(range_rules(name, cand, res, args.min_rule_rows))
        prof = res["profiles"].set_index("UNIT")
        prev_of = dict(units)

        if dec["decision"] in ("SINGLE", "MIXED"):
            for _, r in dec["rows"].iterrows():
                results.append(dict(
                    TEST_NAME=name, TYPE=typ, DECIDED_BY="REF", SUB_DIST=r["SUB_DIST"],
                    CUTOFF=r["CUTOFF"], UNIT=r["UNIT"], OUTCOME="PASS",
                    SCALE_GROUP_UNITS=r["SCALE_GROUP_UNITS"], GROUP_SHARE=r["GROUP_SHARE"],
                    CUTOFF_ACC=r["CUTOFF_ACC"], CUTOFF_BAL_ACC=r["CUTOFF_BAL_ACC"],
                    N_CANDIDATE=dec["n"], MEDIAN_LOWER=prof.at[r["UNIT"], "MEDIAN_LOWER"],
                    MEDIAN_UPPER=prof.at[r["UNIT"], "MEDIAN_UPPER"],
                    UNIT_PREVALENCE=prev_of.get(r["UNIT"], np.nan), PREVALENCE_DICT=prev,
                    NOTES=("" if dec["reliable"] or dec["decision"] == "SINGLE"
                           else "cutoff_unreliable: only the per-row range rules are trustworthy"),
                    **info))
        else:
            fallback.append(name)
        state[name] = (typ, prev, cand, res, dec, info)

        if keep_rows:
            pred, src = rdec.predict_rows(cand["VALUE"].to_numpy(), res, dec)
            r = cand.copy()
            r["STATUS"] = res["status"]
            r["PRED_UNIT"] = pred if pred is not None else ""
            r["PRED_SOURCE"] = src if src is not None else "value"
            rows_out[name] = r
        if plots_dir is not None and (typ == "ambiguous" or dec["decision"] == "MIXED"):
            import ref_plots
            ref_plots.plot_test(name, cand, res, dec, str(Path(plots_dir) / f"ref_{rd.tag_of(name)}.png"))

    # fallback: the value engine on the tests the ranges could not decide
    if fallback:
        print(f"\nValue-engine fallback for {len(fallback)} test(s): "
              f"{', '.join(fallback[:8])}{' ...' if len(fallback) > 8 else ''}", flush=True)
        base = rd.run_value_engine(backend, plot_name[plot_name["TEST_NAME"].isin(fallback)],
                                   details, args.dump_dir, args, log_path=log_path)
        for name in fallback:
            typ, prev, cand, res, dec, info = state[name]
            b = base[base["TEST_NAME"] == name]
            for _, r in b.iterrows():
                results.append(dict(
                    TEST_NAME=name, TYPE=typ, DECIDED_BY="VALUE", SUB_DIST=r["SUB_DIST"],
                    CUTOFF=r["CUTOFF"], UNIT=r["UNIT"], OUTCOME=r["OUTCOME"],
                    N_CANDIDATE=dec["n"], UNIT_PREVALENCE=r.get("UNIT_PREVALENCE", np.nan),
                    PREVALENCE_DICT=prev, VALUE_TESTS_PASSED=r.get("TESTS_PASSED", ""),
                    NOTES=f"ranges: {dec['decision']}", **info))
            if not len(b):
                results.append(dict(TEST_NAME=name, TYPE=typ, DECIDED_BY="VALUE", SUB_DIST="all",
                                    CUTOFF=np.inf, UNIT="", OUTCOME="SKIP", N_CANDIDATE=dec["n"],
                                    PREVALENCE_DICT=prev, NOTES=f"ranges: {dec['decision']}; value engine skipped",
                                    **info))
            if keep_rows:
                rows_out[name]["PRED_UNIT"] = rd.value_rule_units(cand["VALUE"].to_numpy(), b)
                rows_out[name]["PRED_SOURCE"] = "value"

    out = dict(
        results=pd.DataFrame(results, columns=RESULT_COLS),
        profiles=pd.concat(profiles, ignore_index=True) if profiles else pd.DataFrame(),
        rules=pd.concat([r for r in rules if len(r)], ignore_index=True) if any(len(r) for r in rules)
        else pd.DataFrame(columns=RULE_COLS),
    )
    if keep_rows:
        out["rows"] = rows_out
    return out


def summary_md(results, plot_name):
    count = dict(zip(plot_name["TEST_NAME"], plot_name["COUNT"]))
    t = results.drop_duplicates("TEST_NAME")[["TEST_NAME", "TYPE", "REF_DECISION", "DECIDED_BY"]].copy()
    t["MEAS"] = t["TEST_NAME"].map(count).fillna(0).astype(int)
    passed = set(results[results["OUTCOME"] == "PASS"]["TEST_NAME"])
    t["ASSIGNED"] = t["TEST_NAME"].isin(passed)
    lines = ["# Range-primary injection summary", "",
             "| TYPE | REF_DECISION | DECIDED_BY | tests | unit-less rows | tests assigned |",
             "| --- | --- | --- | ---: | ---: | ---: |"]
    for (typ, d, by), g in t.groupby(["TYPE", "REF_DECISION", "DECIDED_BY"]):
        lines.append(f"| {typ} | {d} | {by} | {len(g):,} | {g['MEAS'].sum():,} | {g['ASSIGNED'].sum():,} |")
    lines += ["", "REF_DECISION: SINGLE / MIXED decided by the ranges; NO_EVIDENCE / REVIEW fall back to "
              "the value engine (DECIDED_BY = VALUE)."]
    return "\n".join(lines) + "\n"


def build_parser():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    rd.add_common_args(p)
    p.add_argument("--min-rule-rows", type=int, default=5,
                   help="ref_range_rules.tsv: candidate ranges with fewer rows are left out")
    return p


def main():
    args = build_parser().parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    backend = rd.make_backend(args)
    plot_name, details = rd.load_plot_name(backend, args, out_dir)
    plots_dir = out_dir / "plots" if args.plots else None
    res = run_primary(backend, plot_name, details, args,
                      log_path=None if args.test else str(out_dir / "value_engine_fallback.log"),
                      plots_dir=plots_dir)

    if args.test:
        with pd.option_context("display.max_columns", None, "display.width", 200):
            print("\nProfiles:\n", res["profiles"].drop(columns=["TEST_NAME"]).to_string(index=False))
            print("\nResult:\n", res["results"].T.to_string(header=False))
        return
    res["profiles"].to_csv(out_dir / "ref_unit_profiles.tsv", sep="\t", index=False, float_format="%.5g", na_rep="NA")
    res["rules"].to_csv(out_dir / "ref_range_rules.tsv", sep="\t", index=False, float_format="%.5g", na_rep="NA")
    res["results"].to_csv(out_dir / "ref_primary_results.tsv", sep="\t", index=False, float_format="%.5g", na_rep="NA")
    md = summary_md(res["results"], plot_name)
    (out_dir / "ref_primary_summary.md").write_text(md)
    print("\n" + md)
    print(f"Wrote ref_unit_profiles.tsv, ref_range_rules.tsv, ref_primary_results.tsv, "
          f"ref_primary_summary.md → {out_dir}/")


if __name__ == "__main__":
    main()
