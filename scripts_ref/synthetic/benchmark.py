#!/usr/bin/env python3
"""
benchmark.py

Scores the injection methods on a synthetic parquet (make_synthetic.py) against the truth:
for every unit-less row, is the unit it ends up with on the right scale?

Methods
-------
  VALUE       the current value engine (scripts/primary/test_name_injection.py, unchanged): its
              injection_results rows applied by value (SUB_DIST / CUTOFF / UNIT)
  REF_ROWS    version 1 (scripts_ref/primary): each row with a range gets the unit its range
              points to; the others get the range-learned cutoff rule (or the value engine
              when the ranges cannot decide)
  REF_CUTOFF  version 1 in the injection_results format only: the range decision's
              SUB_DIST / CUTOFF / UNIT applied by value to every row (usable downstream today)
  VALIDATOR   version 2 (scripts_ref/validator): the value engine, corrected where the
              ranges disagree; cutoff format

A row is CORRECT when its unit has the scale of its true unit (same factor: e.g. any of
form / 1 / ph for pH), WRONG when another scale, UNASSIGNED when no unit is injected.

Writes to --out: benchmark_rows.tsv (test x method x row group), benchmark.md,
benchmark_accuracy.png, and with --sweep sweep.tsv (REF_ROWS over margin x combine).

Usage
-----
  python3 scripts_ref/synthetic/make_synthetic.py --out /mnt/disks/data/claude/kanta_ref/synth
  python3 scripts_ref/synthetic/benchmark.py --synth /mnt/disks/data/claude/kanta_ref/synth \\
          --out /mnt/disks/data/claude/kanta_ref/bench [--plots] [--sweep]
"""

import argparse
import contextlib
import io
import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for sub in ("common", "primary", "validator"):
    sys.path.insert(0, str(_ROOT / sub))

import numpy as np    # noqa: E402
import pandas as pd   # noqa: E402

import ref_data as rd                     # noqa: E402
import ref_primary_injection as prim      # noqa: E402
import ref_validator_injection as valid   # noqa: E402

METHODS = ["VALUE", "REF_ROWS", "REF_CUTOFF", "VALIDATOR"]
COLORS  = {"VALUE": "#2a78d6", "REF_ROWS": "#eb6834", "REF_CUTOFF": "#1baf7a", "VALIDATOR": "#eda100"}


def score(pred, true_factor, factors):
    pf = np.array([factors.get(u, np.nan) if u else np.nan for u in pred], dtype=float)
    assigned = np.array([bool(u) for u in pred])
    correct = assigned & np.isclose(pf, true_factor)
    return int(correct.sum()), int((assigned & ~correct).sum()), int((~assigned).sum())


def make_args(parquet, dump_dir, out_dir, extra=()):
    p = argparse.ArgumentParser()
    rd.add_common_args(p)
    p.add_argument("--min-rule-rows", type=int, default=5)
    p.add_argument("--pass-share", type=float, default=0.8)
    p.add_argument("--improve", type=float, default=0.02)
    return p.parse_args(["--input", str(parquet), "--backend", "local", "--dump-dir", dump_dir, "--out-dir", out_dir,
                         "--min-count", "1000", *extra])


def describe(rows):
    """Short text of a test's cutoff-format rows: unit (or low<cut<high)."""
    if rows is None or not len(rows):
        return "-"
    r = rows.copy()
    r["_c"] = pd.to_numeric(r["CUTOFF"], errors="coerce").fillna(np.inf)
    r = r.sort_values("_c")
    parts = []
    for _, x in r.iterrows():
        u = x["UNIT"] if x["OUTCOME"] == "PASS" else f"({x['OUTCOME']})"
        parts.append(u if not np.isfinite(x["_c"]) else f"{u} ≤{x['_c']:.3g}")
    return " | ".join(parts)


def run(args, truth, scen, out, plots=False, quiet=True):
    backend = rd.make_backend(args)
    plot_name, details = rd.load_plot_name(backend, args, out)
    tests = plot_name[plot_name["CATEGORY"].isin(["UNAMBIGUOUS", "AMBIGUOUS"])]
    log = str(out / "value_engine.log")
    sink = io.StringIO()
    ctx = contextlib.redirect_stdout(sink) if quiet else contextlib.nullcontext()
    with ctx:
        base = rd.run_value_engine(backend, tests, details, args.dump_dir, args, log_path=log)
        pr = prim.run_primary(backend, plot_name, details, args, keep_rows=True, log_path=log,
                              plots_dir=(out / "plots_primary") if plots else None)
        vr = valid.run_validator(backend, plot_name, details, args, keep_rows=True, log_path=log,
                                 plots_dir=(out / "plots_validator") if plots else None)
    (out / "run_progress.log").write_text(sink.getvalue())

    recs, desc = [], []
    for name, cat in zip(tests["TEST_NAME"], tests["CATEGORY"]):
        cand = backend.candidate_rows(name)
        v = cand["VALUE"].to_numpy(dtype=float)
        tf = cand["TRUE_FACTOR"].to_numpy(dtype=float)
        ranged = pr["rows"][name]["STATUS"].to_numpy() != "NO_RANGE"
        b_rows = base[base["TEST_NAME"] == name]
        p_rows = pr["results"][pr["results"]["TEST_NAME"] == name]
        c_rows = vr["corrected"][vr["corrected"]["TEST_NAME"] == name]
        preds = {
            "VALUE":      rd.value_rule_units(v, b_rows),
            "REF_ROWS":   pr["rows"][name]["PRED_UNIT"].to_numpy(dtype=object),
            "REF_CUTOFF": rd.value_rule_units(v, p_rows),
            "VALIDATOR":  vr["rows"][name]["PRED_UNIT"].to_numpy(dtype=object),
        }
        for m, pred in preds.items():
            for grp, mask in (("all", np.ones(len(v), bool)), ("ranged", ranged), ("unranged", ~ranged)):
                if not mask.any():
                    continue
                c, w, u = score(pred[mask], tf[mask], truth[name])
                recs.append(dict(TEST_NAME=name, CATEGORY=cat.lower(), METHOD=m, ROWS=grp,
                                 N=int(mask.sum()), CORRECT=c, WRONG=w, UNASSIGNED=u,
                                 PCT_CORRECT=100 * c / mask.sum()))
        desc.append(dict(
            TEST_NAME=name, VALUE=describe(b_rows),
            REF=f"{p_rows['REF_DECISION'].iloc[0]}/{p_rows['DECIDED_BY'].iloc[0]}: {describe(p_rows)}"
            if len(p_rows) else "-",
            VALIDATOR=f"{c_rows['CORRECTION'].iloc[0]}: {describe(c_rows)}" if len(c_rows) else "-",
            REF_SHARES=p_rows["REF_SHARES"].iloc[0] if len(p_rows) else "",
            COVERAGE=p_rows["REF_COVERAGE"].iloc[0] if len(p_rows) else np.nan,
        ))
    return pd.DataFrame(recs), pd.DataFrame(desc), pr, vr, base


def write_md(rec, desc, scen, out, args):
    a = rec[rec["ROWS"] == "all"]
    piv = a.pivot_table(index="TEST_NAME", columns="METHOD", values="PCT_CORRECT")[METHODS]
    order = scen["TEST_NAME"].tolist()
    piv = piv.reindex([t for t in order if t in piv.index])
    note = dict(zip(scen["TEST_NAME"], scen["NOTE"]))
    lines = ["# Range-based injection: synthetic benchmark", "",
             f"Parameters: margin {args.margin}, combine {args.combine}, far {args.far}, "
             f"min-ranged {args.min_ranged}, mix-share {args.mix_share}, min-cut-acc {args.min_cut_acc}.", "",
             "## % of unit-less rows that get a unit of the right scale", "",
             "| test | " + " | ".join(METHODS) + " | scenario |",
             "| --- | " + " | ".join("---:" for _ in METHODS) + " | --- |"]
    for t, r in piv.iterrows():
        best = r.max()
        cells = [(f"**{x:.1f}**" if x >= best - 0.05 else f"{x:.1f}") for x in r]
        lines.append(f"| {t} | " + " | ".join(cells) + f" | {note.get(t, '')} |")
    tot = a.groupby("METHOD")[["CORRECT", "N"]].sum()
    lines.append("| **all rows** | " + " | ".join(
        f"**{100 * tot.at[m, 'CORRECT'] / tot.at[m, 'N']:.1f}**" for m in METHODS) + " | |")
    amb = a[a["CATEGORY"] == "ambiguous"].groupby("METHOD")[["CORRECT", "N"]].sum()
    lines.append("| **ambiguous tests** | " + " | ".join(
        f"**{100 * amb.at[m, 'CORRECT'] / amb.at[m, 'N']:.1f}**" for m in METHODS) + " | |")

    lines += ["", "## Wrong vs unassigned (all rows)", "",
              "| test | " + " | ".join(f"{m} wrong / unassigned" for m in METHODS) + " |",
              "| --- | " + " | ".join("---:" for _ in METHODS) + " |"]
    for t in piv.index:
        cells = []
        for m in METHODS:
            r = a[(a["TEST_NAME"] == t) & (a["METHOD"] == m)].iloc[0]
            cells.append(f"{100 * r['WRONG'] / r['N']:.1f} / {100 * r['UNASSIGNED'] / r['N']:.1f}")
        lines.append(f"| {t} | " + " | ".join(cells) + " |")

    lines += ["", "## REF_ROWS: rows with a range vs without", "",
              "| test | coverage | % correct, ranged rows | % correct, unranged rows |", "| --- | ---: | ---: | ---: |"]
    rr_ = rec[rec["METHOD"] == "REF_ROWS"]
    for t in piv.index:
        g = rr_[rr_["TEST_NAME"] == t].set_index("ROWS")
        cov = g.at["ranged", "N"] / g.at["all", "N"] if "ranged" in g.index else 0
        rc = f"{g.at['ranged', 'PCT_CORRECT']:.1f}" if "ranged" in g.index else "-"
        uc = f"{g.at['unranged', 'PCT_CORRECT']:.1f}" if "unranged" in g.index else "-"
        lines.append(f"| {t} | {100 * cov:.0f}% | {rc} | {uc} |")

    lines += ["", "## What each method decided", "",
              "| test | ranges say | VALUE | REF (decision/decided by: rows) | VALIDATOR (correction: rows) |",
              "| --- | --- | --- | --- | --- |"]
    d = desc.set_index("TEST_NAME")
    for t in piv.index:
        r = d.loc[t]
        lines.append(f"| {t} | {r['REF_SHARES']} | {r['VALUE']} | {r['REF']} | {r['VALIDATOR']} |")
    (out / "benchmark.md").write_text("\n".join(lines) + "\n")
    return piv


def plot(piv, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tests = list(piv.index)[::-1]
    y = np.arange(len(tests))
    h = 0.2
    fig, ax = plt.subplots(figsize=(9, 0.55 * len(tests) + 1.5), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    for i, m in enumerate(METHODS):
        ax.barh(y + (1.5 - i) * h, piv.loc[tests, m], height=h - 0.03, color=COLORS[m], label=m)
    ax.set_yticks(y)
    ax.set_yticklabels(tests, fontsize=8, color="#0b0b0b")
    ax.set_xlim(0, 100)
    ax.set_xlabel("% of unit-less rows with a unit of the right scale", fontsize=9, color="#52514e")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="x", color="#e6e5e0", lw=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(colors="#52514e", labelsize=8)
    ax.legend(fontsize=8, frameon=False, ncol=4, loc="lower center", bbox_to_anchor=(0.4, 1.0))
    fig.tight_layout()
    fig.savefig(out / "benchmark_accuracy.png", dpi=120, facecolor="#fcfcfb")
    plt.close(fig)


def sweep(parquet, dump, out, truth):
    rows = []
    for combine in ("max", "mean"):
        for margin in (1.2, 1.5, 2.0, 3.0):
            args = make_args(parquet, dump, str(out), ["--margin", str(margin), "--combine", combine])
            backend = rd.make_backend(args)
            plot_name, details = rd.load_plot_name(backend, args, out)
            with contextlib.redirect_stdout(io.StringIO()):
                pr = prim.run_primary(backend, plot_name, details, args, keep_rows=True,
                                      log_path=str(out / "value_engine.log"))
            c = n = ca = na = 0
            for name, r in pr["rows"].items():
                cc, _, _ = score(r["PRED_UNIT"].to_numpy(dtype=object), r["TRUE_FACTOR"].to_numpy(dtype=float),
                                 truth[name])
                c += cc
                n += len(r)
                if plot_name.set_index("TEST_NAME").at[name, "CATEGORY"] == "AMBIGUOUS":
                    ca += cc
                    na += len(r)
            rows.append(dict(COMBINE=combine, MARGIN=margin, PCT_CORRECT=100 * c / n,
                             PCT_CORRECT_AMBIGUOUS=100 * ca / na))
            print(f"  sweep combine={combine} margin={margin}: {100 * c / n:.2f}% (ambiguous {100 * ca / na:.2f}%)")
    df = pd.DataFrame(rows)
    df.to_csv(out / "sweep.tsv", sep="\t", index=False, float_format="%.3f")
    return df


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--synth", required=True, help="Folder written by make_synthetic.py")
    p.add_argument("--out", required=True, help="Output folder (scratch)")
    p.add_argument("--plots", action="store_true", help="Per-test diagnostic PNGs of both versions")
    p.add_argument("--sweep", action="store_true", help="Also sweep --margin x --combine for REF_ROWS")
    p.add_argument("--verbose", action="store_true", help="Show the progress lines instead of logging them")
    a = p.parse_args()
    synth, out = Path(a.synth), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    parquet = str(synth / "synthetic_inject.parquet")
    truth = json.loads((synth / "synthetic_truth.json").read_text())
    scen = pd.read_csv(synth / "synthetic_scenarios.tsv", sep="\t")
    dump = str(out / f"dump_{time.strftime('%Y%m%d_%H%M%S')}")   # fresh cache per run
    args = make_args(parquet, dump, str(out))

    t0 = time.time()
    rec, desc, pr, vr, base = run(args, truth, scen, out, plots=a.plots, quiet=not a.verbose)
    rec.to_csv(out / "benchmark_rows.tsv", sep="\t", index=False, float_format="%.3f")
    pr["results"].to_csv(out / "ref_primary_results.tsv", sep="\t", index=False, float_format="%.5g", na_rep="NA")
    pr["profiles"].to_csv(out / "ref_unit_profiles.tsv", sep="\t", index=False, float_format="%.5g", na_rep="NA")
    vr["corrected"].to_csv(out / "ref_corrected_results.tsv", sep="\t", index=False, float_format="%.5g", na_rep="NA")
    vr["validated"].to_csv(out / "ref_validated_results.tsv", sep="\t", index=False, float_format="%.5g", na_rep="NA")
    base.to_csv(out / "value_injection_results.tsv", sep="\t", index=False, float_format="%.5g", na_rep="NA")
    piv = write_md(rec, desc, scen, out, args)
    plot(piv, out)
    print((out / "benchmark.md").read_text())
    print(f"({time.time() - t0:.0f}s)  → {out}/")
    if a.sweep:
        sweep(parquet, dump, out, truth)


if __name__ == "__main__":
    main()
