#!/usr/bin/env python3
"""
range_type_check.py — does the reference range say *who* was tested rather than *which unit*?

The validator split p-tt-inr into inr (range ~0.9-1.2) and form (range ~2-3). 2-3 is the
warfarin target range, so the hypothesis is: some labs print the therapeutic target as the
"reference range", and the range type tracks the patient group (anticoagulated or not),
not the unit. This script checks it on the rows that HAVE a unit (pre-injection unit) and
on the raw source unit before cleanup, plus the unit-less rows for comparison.

Each row is classified by its reference range:
  NORMAL       upper <= --normal-max          (e.g. 0.9-1.2)
  THERAPEUTIC  lower >= --ther-min            (e.g. 2-3, 2-3.5)
  OTHER        has a range, neither of the above
  NONE         no lower and no upper limit

If the hypothesis holds:
  * both pre-injection units (inr, form, 1, ...) contain NORMAL and THERAPEUTIC rows;
  * within every unit, THERAPEUTIC rows have clearly higher values than NORMAL rows;
  * the range type follows the lab (SERVICE_PROVIDER), not the unit text.

Usage (data VM; default input = latest run in /mnt/disks/data/kanta/engine):
  python3 scripts_ref/range_type_check.py --out-dir OUT
  python3 scripts_ref/range_type_check.py --out-dir OUT --test p-tt-inr --test p-inr
Writes OUT/range_type_check.md (shareable), OUT/range_type_<test>.png and the TSVs behind them.
"""

import argparse
import subprocess
import sys
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "common"))
from ref_data import ENGINE_DIR, latest_engine_output   # noqa: E402

COL_TEST  = "TEST_NAME_ABBREVIATION"
COL_UNIT  = "cleaned-pre-inj::MEASUREMENT_UNIT"
COL_CVAL  = "MEASUREMENT_VALUE"           # value of unit-less rows (candidates)
COL_TVAL  = "source::MEASUREMENT_VALUE"   # value of rows with a unit (targets)
# INR tests (OMOP 3032080 in LABfi.tsv) that are in the injection run
DEF_TESTS = ["p-inr", "p-tt-inr", "p-tromboplastiiniaika,inr-tulostus", "p-inr-vt",
             "p-tromboplastiiniaika,inr-tulos", "p-inr-pika",
             "tromboplastiiniaika,inr-tulostus,plasmasta", "b-inrpika"]
CLASSES   = ["NORMAL", "THERAPEUTIC", "OTHER", "NONE"]
COLORS    = {"NORMAL": "#2a78d6", "THERAPEUTIC": "#eb6834", "OTHER": "#1baf7a", "NONE": "#b5b4ae"}


def clickhouse(query, **params):
    cmd = ["clickhouse", "-q", query] + [f"--param_{k}={v}" for k, v in params.items()]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.exit(f"ClickHouse query failed:\n{proc.stderr}")
    return proc.stdout


def columns(parquet):
    out = clickhouse(f"DESCRIBE TABLE file('{parquet}') FORMAT TSV")
    return [line.split("\t")[0] for line in out.splitlines() if line]


def pick(cols, wanted, label):
    if wanted in cols:
        return wanted
    print(f"  (no column {wanted!r}: {label} skipped)")
    return None


def query(parquet, test, src_unit, lab, lower, upper, step):
    num = lambda c: f"toFloat64OrNull(toString(`{c}`))"
    src = f"toString(`{src_unit}`)" if src_unit else "'-'"
    lb  = f"toString(`{lab}`)" if lab else "'-'"
    out = clickhouse(f"""
        SELECT pre_unit, src_unit, lab, lower, upper, round(value / {step}) * {step} AS v, count() AS n
        FROM (
            SELECT `{COL_UNIT}` AS pre_unit, {src} AS src_unit, {lb} AS lab,
                   {num(lower)} AS lower, {num(upper)} AS upper,
                   if(`{COL_UNIT}` = 'NA', {num(COL_CVAL)}, {num(COL_TVAL)}) AS value
            FROM file('{parquet}')
            WHERE {COL_TEST} = {{name:String}}
        )
        WHERE value IS NOT NULL
        GROUP BY pre_unit, src_unit, lab, lower, upper, v
        FORMAT TSV
    """, name=test)
    df = pd.read_csv(StringIO(out), sep="\t", header=None, keep_default_na=False,
                     names=["PRE_UNIT", "SRC_UNIT", "LAB", "LOWER", "UPPER", "VALUE", "N"],
                     dtype={"PRE_UNIT": str, "SRC_UNIT": str, "LAB": str})
    for c in ("LOWER", "UPPER", "VALUE"):
        df[c] = pd.to_numeric(df[c].replace("\\N", np.nan), errors="coerce")
    df["PRE_UNIT"] = df["PRE_UNIT"].replace("NA", "(none)")
    df["SRC_UNIT"] = df["SRC_UNIT"].replace({"NA": "(none)", "": "(empty)", "\\N": "(null)"})
    return df


def classify(df, normal_max, ther_min):
    lo, up = df["LOWER"], df["UPPER"]
    cls = np.select([lo.isna() & up.isna(), up <= normal_max, lo >= ther_min],
                    ["NONE", "NORMAL", "THERAPEUTIC"], default="OTHER")
    return pd.Series(cls, index=df.index)


def wq(values, weights, qs):
    o = np.argsort(values)
    v, w = np.asarray(values)[o], np.asarray(weights, dtype=float)[o]
    c = np.cumsum(w) / w.sum()
    return [float(v[min(np.searchsorted(c, q), len(v) - 1)]) for q in qs]


def summary(df, keys, thr):
    rows = []
    for k, g in df.groupby(keys, dropna=False):
        k = k if isinstance(k, tuple) else (k,)
        n = int(g["N"].sum())
        p10, p50, p90 = wq(g["VALUE"], g["N"], [0.1, 0.5, 0.9])
        rows.append((*k, n, p10, p50, p90, float(g.loc[g["VALUE"] >= thr, "N"].sum() / n)))
    out = pd.DataFrame(rows, columns=[*keys, "N", "P10", "MEDIAN", "P90", f"SHARE_GE_{thr:g}"])
    out["SHARE_OF_KEY"] = out["N"] / out.groupby(keys[0])["N"].transform("sum")
    return out.sort_values([keys[0], "N"], ascending=[True, False])


def top_ranges(df, n_top):
    r = df[df["CLASS"] != "NONE"].groupby(["PRE_UNIT", "LOWER", "UPPER", "CLASS"], dropna=False)["N"].sum()
    r = r.reset_index().sort_values(["PRE_UNIT", "N"], ascending=[True, False])
    return r.groupby("PRE_UNIT").head(n_top)


def lab_table(df, min_n):
    t = df.pivot_table(index="LAB", columns="CLASS", values="N", aggfunc="sum", fill_value=0)
    t = t.reindex(columns=CLASSES, fill_value=0)
    t["N"] = t.sum(axis=1)
    units = df.groupby(["LAB", "PRE_UNIT"])["N"].sum().reset_index().sort_values("N", ascending=False)
    t["PRE_UNITS"] = units.groupby("LAB").apply(
        lambda g: ", ".join(f"{u}:{100 * n / g['N'].sum():.0f}%" for u, n in zip(g["PRE_UNIT"], g["N"])),
        include_groups=False)
    t = t[t["N"] >= min_n].sort_values("N", ascending=False)
    for c in CLASSES:
        t[c] = (t[c] / t["N"]).round(3)
    # a lab is "mixed" when it prints both normal and therapeutic ranges
    t["MIXED_TYPES"] = (t["NORMAL"] >= 0.05) & (t["THERAPEUTIC"] >= 0.05)
    return t.reset_index()


def plot(df, test, out_path, thr, step):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    units = df.groupby("PRE_UNIT")["N"].sum().sort_values(ascending=False).index[:4]
    fig, axes = plt.subplots(1, len(units), figsize=(4.2 * len(units), 3.4), squeeze=False)
    hi = wq(df["VALUE"], df["N"], [0.99])[0]
    lo = max(0.0, float(df["VALUE"].min()))
    bins = np.arange(lo - step / 2, hi + step, step) if (hi - lo) / step <= 200 else np.linspace(lo, hi, 100)
    for ax, u in zip(axes[0], units):
        g = df[df["PRE_UNIT"] == u]
        for c in CLASSES:
            s = g[g["CLASS"] == c]
            if s["N"].sum() == 0:
                continue
            h, _ = np.histogram(s["VALUE"].clip(upper=hi), bins=bins, weights=s["N"])
            ax.step(bins[:-1], h / g["N"].sum(), where="post", color=COLORS[c], lw=2,
                    label=f"{c} ({100 * s['N'].sum() / g['N'].sum():.0f}%)")
        ax.axvline(thr, color="#52514e", lw=1, ls=":")
        ax.set_title(f"unit: {u}  (N={int(g['N'].sum()):,})", fontsize=10, color="#0b0b0b")
        ax.set_xlabel("value", color="#52514e")
        ax.grid(color="#e6e5e0", lw=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.legend(fontsize=8, frameon=False)
    axes[0][0].set_ylabel("share of the unit's rows", color="#52514e")
    fig.suptitle(f"{test}: value by reference-range type, per pre-injection unit", fontsize=11)
    fig.patch.set_facecolor("#fcfcfb")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, facecolor=fig.get_facecolor())
    plt.close(fig)


def md_table(df):
    """Markdown table without the optional `tabulate` dependency."""
    fmt = lambda x: f"{x:.3g}" if isinstance(x, (float, np.floating)) else str(x)
    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(fmt(x) for x in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def md(df, name):
    return f"\n**{name}**\n\n" + md_table(df) + "\n"


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--input", type=Path, default=latest_engine_output(),
                   help=f"Engine output parquet (default: the most recent run in {ENGINE_DIR}, now %(default)s)")
    p.add_argument("--out-dir", type=Path, default=Path("range_type_check"))
    p.add_argument("--test", action="append", help=f"TEST_NAME (repeatable; default {', '.join(DEF_TESTS)})")
    p.add_argument("--source-unit-col", default="source::MEASUREMENT_UNIT", help="Raw unit before cleanup")
    p.add_argument("--lab-col", default="SERVICE_PROVIDER", help="Lab column ('' to skip)")
    p.add_argument("--lower-col", default="REFERENCE_RANGE_LOWER_VALUE")
    p.add_argument("--upper-col", default="REFERENCE_RANGE_UPPER_VALUE")
    p.add_argument("--normal-max", type=float, default=1.6, help="NORMAL: upper limit <= this")
    p.add_argument("--ther-min", type=float, default=1.5, help="THERAPEUTIC: lower limit >= this")
    p.add_argument("--value-threshold", type=float, default=1.5, help="Report the share of values >= this")
    p.add_argument("--value-step", type=float, default=0.1, help="Values are binned to this step in the query")
    p.add_argument("--min-lab-rows", type=int, default=200)
    p.add_argument("--top-ranges", type=int, default=8)
    args = p.parse_args()
    if args.input is None:
        sys.exit(f"No engine output found in {ENGINE_DIR}: pass --input")
    tests = args.test or DEF_TESTS
    args.out_dir.mkdir(parents=True, exist_ok=True)

    cols = columns(args.input)
    print(f"Input: {args.input}")
    print("Unit / range / lab columns in the parquet:")
    for c in cols:
        if any(k in c.upper() for k in ("UNIT", "REFERENCE", "PROVIDER", "LAB")):
            print(f"  {c}")
    src = pick(cols, args.source_unit_col, "raw source unit")
    lab = pick(cols, args.lab_col, "lab breakdown") if args.lab_col else None
    thr = args.value_threshold

    report = [f"# Reference-range type vs unit\n",
              f"Input `{args.input}`. NORMAL = upper ≤ {args.normal_max:g}, THERAPEUTIC = lower ≥ "
              f"{args.ther_min:g}. Values of rows with a unit are `{COL_TVAL}`, of unit-less rows "
              f"`{COL_CVAL}`; binned to {args.value_step:g}. `SHARE_GE_{thr:g}` = share of values ≥ {thr:g}.\n",
              "Hypothesis: the range type tracks the patient group (THERAPEUTIC rows have high values "
              "in every unit) and the lab, not the unit.\n"]
    for t in tests:
        print(f"\n== {t}")
        df = query(args.input, t, src, lab, args.lower_col, args.upper_col, args.value_step)
        if df.empty:
            print("  no rows")
            report.append(f"\n## {t}\n\nNo rows.\n")
            continue
        df["CLASS"] = classify(df, args.normal_max, args.ther_min)
        tag = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in t)

        by_unit = summary(df, ["PRE_UNIT", "CLASS"], thr)
        by_src  = summary(df, ["SRC_UNIT", "PRE_UNIT", "CLASS"], thr) if src else None
        ranges  = top_ranges(df, args.top_ranges)
        labs    = lab_table(df, args.min_lab_rows) if lab else None

        by_unit.to_csv(args.out_dir / f"range_type_{tag}_by_unit.tsv", sep="\t", index=False)
        ranges.to_csv(args.out_dir / f"range_type_{tag}_top_ranges.tsv", sep="\t", index=False)
        if by_src is not None:
            by_src.to_csv(args.out_dir / f"range_type_{tag}_by_source_unit.tsv", sep="\t", index=False)
        if labs is not None:
            labs.to_csv(args.out_dir / f"range_type_{tag}_by_lab.tsv", sep="\t", index=False)
        plot(df, t, args.out_dir / f"range_type_{tag}.png", thr, args.value_step)

        print(by_unit.to_string(index=False, float_format=lambda x: f"{x:.3g}"))
        report.append(f"\n## {t}\n\n![{t}](range_type_{tag}.png)\n")
        report.append(md(by_unit, "Pre-injection unit × range type"))
        report.append(md(ranges, f"Most common ranges per pre-injection unit (top {args.top_ranges})"))
        if by_src is not None:
            report.append(md(by_src, "Raw source unit × pre-injection unit × range type"))
        if labs is not None:
            n_mixed = int(labs["MIXED_TYPES"].sum())
            report.append(f"\n**Labs (≥ {args.min_lab_rows} rows): range-type shares**  "
                          f"{n_mixed} of {len(labs)} labs print both NORMAL and THERAPEUTIC ranges (≥5% each).\n\n"
                          + md_table(labs) + "\n")

    out = args.out_dir / "range_type_check.md"
    out.write_text("".join(report))
    print(f"\nReport: {out}")


if __name__ == "__main__":
    main()
