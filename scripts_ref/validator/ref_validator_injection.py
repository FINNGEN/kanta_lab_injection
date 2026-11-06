#!/usr/bin/env python3
"""
ref_validator_injection.py  —  version 2: the value engine decides, reference ranges
validate and correct.

1. Run the current value engine (scripts/primary/test_name_injection.py: run_unambiguous +
   run_ambiguous, imported unchanged) -> the injection_results.tsv rows.
2. Label the unit-less rows by their reference ranges (scripts_ref/common/ref_ranges.py),
   the same way as version 1.
3. Validate each value-engine row on the range-labelled rows of its sub-distribution:
     REF_MATCH       share whose range points to the row's unit (same scale group)
     REF_VALIDATION  PASS (>= --pass-share), CONFLICT (another group has >= 50 %), WEAK,
                     NO_EVIDENCE (< --min-ranged decided rows), NO_PROFILE (the unit has no
                     usable range profile, so ranges cannot judge it)
4. Correct the test when the ranges clearly disagree. Both rules are scored on the
   range-labelled rows: the value engine's (BASE_AGREEMENT) and the range decision's in
   cutoff format (REF_AGREEMENT). The range rule replaces the value rule when it agrees
   better by more than --improve (a REF_AGREEMENT below --min-cut-acc is still noted as
   cutoff_unreliable: better than the value rule, but the cutoff format loses rows that
   only the per-row ranges of version 1 get right). CORRECTION says what changed:
     unit    same split, other unit(s)          split   value engine did not split
     cutoff  same units, other cutoff           merge   value engine split, ranges say one unit
     rescue  value engine had no PASS           none    kept
   Nothing is corrected when one of the value engine's units has no range profile.

The corrected table keeps the injection_results.tsv format, so it is usable downstream as is.

Outputs (in --out-dir)
----------------------
  value_injection_results.tsv   the value engine's table, as written today
  ref_validated_results.tsv     the same rows + REF_* validation columns + the test's CORRECTION
  ref_corrected_results.tsv     cutoff-format table after the corrections (SOURCE VALUE / REF)
  ref_unit_profiles.tsv         per test x unit: median lower / upper limit
  ref_validator_summary.md      validation and corrections by category
  plots/ref_<test>.png          with --plots: values by range label, both cutoffs

Usage
-----
  python3 scripts_ref/validator/ref_validator_injection.py --out-dir OUT [--plots]          # latest run in /mnt/disks/data/kanta/engine
  python3 scripts_ref/validator/ref_validator_injection.py --input ENGINE.parquet --test u-ph
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
import test_name_injection as tni  # noqa: E402  (on sys.path via ref_data)

VALID_COLS = ["REF_N", "REF_MATCH", "REF_TOP_UNIT", "REF_TOP_SHARE", "REF_VALIDATION"]
TEST_COLS  = ["REF_DECISION", "N_RANGED", "REF_COVERAGE", "N_DECIDED", "REF_SHARES", "REF_NO_MATCH",
              "REF_STATUS", "SCALE_GROUPS", "BASE_AGREEMENT", "REF_AGREEMENT", "CORRECTION"]
CORR_COLS  = ["TEST_NAME", "TYPE", "SUB_DIST", "CUTOFF", "UNIT", "OUTCOME", "SOURCE", "CORRECTION",
              "REF_VALIDATION", "REF_MATCH", "BASE_AGREEMENT", "REF_AGREEMENT", "REF_DECISION",
              "REF_SHARES", "REF_COVERAGE", "UNIT_PREVALENCE", "PREVALENCE_DICT", "NOTES"]


def _sub_masks(values, rows):
    """Masks of the values each cutoff-format row covers (same rule as rd.value_rule_units)."""
    r = rows.copy()
    r["_c"] = pd.to_numeric(r["CUTOFF"], errors="coerce").fillna(np.inf)
    r = r.sort_values("_c", kind="stable")
    lower, out = -np.inf, []
    for idx, row in r.iterrows():
        out.append((idx, (values > lower) & (values <= row["_c"])))
        lower = row["_c"]
    return out


def validate_rows(values, res, base_rows, args):
    """REF_* validation columns for each value-engine row of one test (index -> dict)."""
    status, group, rep, ug = res["status"], res["group"], res["group_rep"], res["unit_group"]
    decided = np.isin(status, rr.DECIDED) & (group >= 0)
    out = {}
    for idx, m in _sub_masks(values, base_rows):
        unit = base_rows.at[idx, "UNIT"]
        d = decided & m
        n = int(d.sum())
        shares = {}
        if n:
            cnt = np.bincount(group[d], minlength=len(rep))
            shares = {g: cnt[g] / n for g in np.argsort(-cnt) if cnt[g] > 0}
        top_g = next(iter(shares), None)
        g_unit = ug.get(unit)
        match = shares.get(g_unit, 0.0) if n else np.nan
        if g_unit is None:
            v = "NO_PROFILE"
        elif n < args.min_ranged:
            v = "NO_EVIDENCE"
        elif match >= args.pass_share:
            v = "PASS"
        elif top_g is not None and top_g != g_unit and shares[top_g] >= 0.5:
            v = "CONFLICT"
        else:
            v = "WEAK"
        out[idx] = dict(REF_N=n, REF_MATCH=match,
                        REF_TOP_UNIT=rep[top_g] if top_g is not None else "",
                        REF_TOP_SHARE=shares.get(top_g, np.nan) if top_g is not None else np.nan,
                        REF_VALIDATION=v)
    return out


def correction_kind(base_rows, dec):
    b = base_rows[base_rows["OUTCOME"] == "PASS"]
    if not len(b):
        return "rescue"
    b_split, r_split = len(base_rows) > 1, dec["decision"] == "MIXED"
    if not b_split and r_split:
        return "split"
    if b_split and not r_split:
        return "merge"
    if b_split and r_split:
        same_units = list(base_rows.sort_values("CUTOFF")["UNIT"]) == list(dec["units"])
        return "cutoff" if same_units else "unit"
    return "unit"


def run_validator(backend, plot_name, details, args, keep_rows=False, log_path=None, plots_dir=None):
    prev_dict = dict(zip(details["TEST_NAME"], details["PREVALENCE_DICT"]))
    tests = plot_name[plot_name["CATEGORY"].isin(["UNAMBIGUOUS", "AMBIGUOUS"])]
    print(f"Value engine on {len(tests)} test(s) ...", flush=True)
    base = rd.run_value_engine(backend, tests, details, args.dump_dir, args, log_path=log_path)

    validated, corrected, profiles, rows_out = [], [], [], {}
    for i, (name, cat) in enumerate(zip(tests["TEST_NAME"], tests["CATEGORY"]), 1):
        typ  = cat.lower()
        prev = prev_dict.get(name, "NA")
        b = base[base["TEST_NAME"] == name].copy()
        units = rd.units_to_consider(prev)
        prev_all = dict(tni.parse_top_units(prev, n=10))
        for u in b["UNIT"].dropna():
            if u and u not in dict(units):
                units.append((u, prev_all.get(u, 0.0)))
        cand, res = rd.analyse(backend, name, units, args)
        values = cand["VALUE"].to_numpy(dtype=float)
        dec  = rdec.ref_decision(values, res, args)
        info = rdec.ref_info(dec, res)

        p = res["profiles"].copy()
        p.insert(0, "TYPE", typ)
        p.insert(0, "TEST_NAME", name)
        profiles.append(p)

        if not len(b):
            b = pd.DataFrame([dict(TEST_NAME=name, TYPE=typ, SUB_DIST="all", CUTOFF=np.inf,
                                   UNIT="", OUTCOME="SKIP", PREVALENCE_DICT=prev)])
        vrows = validate_rows(values, res, b, args)
        base_pred = rd.value_rule_units(values, b)
        base_agree = rdec.range_agreement(base_pred, res)

        ref_agree, corr = np.nan, "none"
        if dec["decision"] in ("SINGLE", "MIXED"):
            ref_pred = rr.apply_cutoffs(values, [t for t, _, _ in dec["cutoffs"]], dec["units"])
            ref_agree = rdec.range_agreement(ref_pred, res)
            no_profile = any(u and u not in res["unit_group"]
                             for u in b.loc[b["OUTCOME"] == "PASS", "UNIT"])
            better = (np.nan_to_num(ref_agree) - np.nan_to_num(base_agree)) > args.improve
            if not no_profile and better:
                corr = correction_kind(b, dec)

        tinfo = dict(info, BASE_AGREEMENT=base_agree, REF_AGREEMENT=ref_agree, CORRECTION=corr)
        for idx, r in b.iterrows():
            validated.append({**r.to_dict(), "TYPE": typ, **vrows[idx], **tinfo})

        common = dict(TEST_NAME=name, TYPE=typ, CORRECTION=corr, BASE_AGREEMENT=base_agree,
                      REF_AGREEMENT=ref_agree, REF_DECISION=dec["decision"],
                      REF_SHARES=info["REF_SHARES"], REF_COVERAGE=info["REF_COVERAGE"],
                      PREVALENCE_DICT=prev)
        if corr == "none":
            for idx, r in b.iterrows():
                corrected.append(dict(common, SUB_DIST=r["SUB_DIST"], CUTOFF=r["CUTOFF"], UNIT=r["UNIT"],
                                      OUTCOME=r["OUTCOME"], SOURCE="VALUE",
                                      REF_VALIDATION=vrows[idx]["REF_VALIDATION"],
                                      REF_MATCH=vrows[idx]["REF_MATCH"],
                                      UNIT_PREVALENCE=r.get("UNIT_PREVALENCE", np.nan), NOTES=""))
            final_pred = base_pred
        else:
            for _, r in dec["rows"].iterrows():
                corrected.append(dict(common, SUB_DIST=r["SUB_DIST"], CUTOFF=r["CUTOFF"], UNIT=r["UNIT"],
                                      OUTCOME="PASS", SOURCE="REF", REF_VALIDATION="",
                                      REF_MATCH=np.nan, UNIT_PREVALENCE=dict(units).get(r["UNIT"], np.nan),
                                      NOTES=("cutoff_unreliable; " if ref_agree < args.min_cut_acc else "")
                                      + "value engine: " + "; ".join(
                                          f"{s}:{u}:{o}" for s, u, o in zip(b["SUB_DIST"], b["UNIT"], b["OUTCOME"]))))
            final_pred = rr.apply_cutoffs(values, [t for t, _, _ in dec["cutoffs"]], dec["units"])

        vals = ",".join(v["REF_VALIDATION"] for v in vrows.values())
        print(f"[{i:>4}/{len(tests)}] {name:<40} {typ:<11} value={'|'.join(b['UNIT'].astype(str))}"
              f"({'|'.join(b['OUTCOME'])})  ranges={dec['decision']} {info['REF_SHARES']}  "
              f"valid={vals}  agree base/ref={_pct(base_agree)}/{_pct(ref_agree)}  → {corr}", flush=True)

        if keep_rows:
            r = cand.copy()
            r["STATUS"] = res["status"]
            r["BASE_UNIT"] = base_pred
            r["PRED_UNIT"] = final_pred
            rows_out[name] = r
        if plots_dir is not None and (typ == "ambiguous" or dec["decision"] == "MIXED" or corr != "none"):
            import ref_plots
            low = b[b["SUB_DIST"] == "low"]
            vc = float(low["CUTOFF"].iloc[0]) if len(low) else None
            ref_plots.plot_test(name, cand, res, dec, str(Path(plots_dir) / f"ref_{rd.tag_of(name)}.png"),
                                value_cutoff=vc, title_extra=f"   correction: {corr}")

    out = dict(
        base=base,
        validated=pd.DataFrame(validated),
        corrected=pd.DataFrame(corrected, columns=CORR_COLS),
        profiles=pd.concat(profiles, ignore_index=True) if profiles else pd.DataFrame(),
    )
    if len(out["validated"]):
        front = [c for c in tni._UNIFIED_COLS if c in out["validated"].columns]
        out["validated"] = out["validated"][front + VALID_COLS + TEST_COLS]
    if keep_rows:
        out["rows"] = rows_out
    return out


def _pct(x):
    return "NA" if not np.isfinite(x) else f"{x:.0%}"


def summary_md(v, c, plot_name):
    count = dict(zip(plot_name["TEST_NAME"], plot_name["COUNT"]))
    lines = ["# Range validator summary", "", "## Validation of the value engine's rows", "",
             "| TYPE | OUTCOME | REF_VALIDATION | rows | tests | unit-less rows |",
             "| --- | --- | --- | ---: | ---: | ---: |"]
    for (typ, o, val), g in v.groupby(["TYPE", "OUTCOME", "REF_VALIDATION"]):
        lines.append(f"| {typ} | {o} | {val} | {len(g):,} | {g['TEST_NAME'].nunique():,} | "
                     f"{sum(count.get(n, 0) for n in g['TEST_NAME'].unique()):,} |")
    t = c.drop_duplicates("TEST_NAME")
    lines += ["", "## Corrections", "", "| TYPE | CORRECTION | tests | unit-less rows |", "| --- | --- | ---: | ---: |"]
    for (typ, corr), g in t.groupby(["TYPE", "CORRECTION"]):
        lines.append(f"| {typ} | {corr} | {len(g):,} | {sum(count.get(n, 0) for n in g['TEST_NAME']):,} |")
    return "\n".join(lines) + "\n"


def build_parser():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    rd.add_common_args(p)
    p.add_argument("--pass-share", type=float, default=0.8,
                   help="REF_VALIDATION PASS: share of range-labelled rows pointing to the row's unit")
    p.add_argument("--improve", type=float, default=0.02,
                   help="Correct only when the range rule agrees with the range labels by more than this")
    return p


def main():
    args = build_parser().parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    backend = rd.make_backend(args)
    plot_name, details = rd.load_plot_name(backend, args, out_dir)
    plots_dir = out_dir / "plots" if args.plots else None
    res = run_validator(backend, plot_name, details, args,
                        log_path=None if args.test else str(out_dir / "value_engine.log"),
                        plots_dir=plots_dir)
    if args.test:
        with pd.option_context("display.max_columns", None, "display.width", 200):
            print("\nProfiles:\n", res["profiles"].drop(columns=["TEST_NAME"]).to_string(index=False))
            print("\nValidated:\n", res["validated"].T.to_string(header=False))
            print("\nCorrected:\n", res["corrected"].T.to_string(header=False))
        return
    fmt = dict(sep="\t", index=False, float_format="%.5g", na_rep="NA")
    res["base"].to_csv(out_dir / "value_injection_results.tsv", **fmt)
    res["validated"].to_csv(out_dir / "ref_validated_results.tsv", **fmt)
    res["corrected"].to_csv(out_dir / "ref_corrected_results.tsv", **fmt)
    res["profiles"].to_csv(out_dir / "ref_unit_profiles.tsv", **fmt)
    md = summary_md(res["validated"], res["corrected"], plot_name)
    (out_dir / "ref_validator_summary.md").write_text(md)
    print("\n" + md)
    print(f"Wrote value_injection_results.tsv, ref_validated_results.tsv, ref_corrected_results.tsv, "
          f"ref_unit_profiles.tsv, ref_validator_summary.md → {out_dir}/")


if __name__ == "__main__":
    main()
