#!/usr/bin/env python3
"""
make_synthetic.py

Synthetic INJECT-like parquet for testing the range-based injection (one TEST_NAME per
scenario). Same columns and string conventions as the engine output the injection
scripts read ('NA' = missing), plus the truth for scoring:

  TEST_NAME_ABBREVIATION, MEASUREMENT_VALUE, source::MEASUREMENT_VALUE,
  cleaned-pre-inj::MEASUREMENT_UNIT ('NA' on the unit-less candidate rows),
  harmonization_omop::OMOP_ID, REFERENCE_RANGE_LOWER_VALUE, REFERENCE_RANGE_UPPER_VALUE,
  REFERENCE_RANGE_SOURCE, TRUE_UNIT, TRUE_FACTOR (truth; no method reads them)

Each scenario has a base analyte distribution and base reference ranges; unit u reports
value x factor_u and ranges x factor_u. Labs vary a range by a few % (3 variants) and
round to 3 significant digits. Target rows (with a unit) carry a range with --target-cov
probability, 2 % of them the range of another unit (lab mislabels). Candidate rows (no
unit) are a mix of the units, with their own coverage and range-noise rate.

Writes <out>/synthetic_inject.parquet, <out>/synthetic_truth.json (test -> unit -> factor)
and <out>/synthetic_scenarios.tsv.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

# name: dict(dist, ranges (base scale: (lower|None, upper|None, weight)), units: [(unit, factor,
#            n_target)], cand: (n, {unit: share}), cand_cov, cand_noise, note)
SCENARIOS = {
    "syn-clean-1000x": dict(
        dist=("lognormal", np.log10(5), 0.12), ranges=[(3.5, 6.5, 1)],
        units=[("mmol/l", 1, 6000), ("umol/l", 1000, 4000)],
        cand=(4000, {"mmol/l": .5, "umol/l": .5}), cand_cov=.8, cand_noise=0,
        note="two units 1000x apart, 50/50: value split should work too"),
    "syn-close-2.5x": dict(
        dist=("lognormal", np.log10(90), 0.25), ranges=[(70, 110, 1)],
        units=[("mg/dl", 1, 6000), ("mmol/l", 0.4, 4000)],
        cand=(5000, {"mg/dl": .6, "mmol/l": .4}), cand_cov=.8, cand_noise=0,
        note="units 2.5x apart, wide values: the value distributions overlap"),
    "syn-minority-10x": dict(
        dist=("lognormal", np.log10(140), 0.07), ranges=[(117, 155, .5), (134, 167, .5)],
        units=[("g/l", 1, 8000), ("g/dl", 0.1, 1500)],
        cand=(5000, {"g/l": .93, "g/dl": .07}), cand_cov=.8, cand_noise=0,
        note="Hb-like, sex-specific ranges, 7 % of the unit-less rows in the minority unit"),
    "syn-heavy-tail-10x": dict(
        dist=("lognormal", np.log10(4), 0.6), ranges=[(None, 10, 1)],
        units=[("mg/l", 1, 6000), ("mg/dl", 0.1, 3000)],
        cand=(4000, {"mg/l": .5, "mg/dl": .5}), cand_cov=.8, cand_noise=0,
        note="CRP-like, very wide values, one-sided ranges (upper only)"),
    "syn-same-scale": dict(
        dist=("normal_step", 6.0, 0.7, 0.5, 4.5, 9.0), ranges=[(5, 8, 1)],
        units=[("form", 1, 9500), ("1", 1, 350), ("ph", 1, 150)],
        cand=(6000, {"form": 1.0}), cand_cov=.8, cand_noise=0,
        note="pH-like: three labels of the same scale; any of them is right"),
    "syn-pct-ratio-100x": dict(
        dist=("lognormal", np.log10(41), 0.05), ranges=[(35, 46, 1)],
        units=[("%", 1, 7000), ("ratio", 0.01, 3000)],
        cand=(4000, {"%": .8, "ratio": .2}), cand_cov=.8, cand_noise=0,
        note="hematocrit-like % vs ratio, 80/20"),
    "syn-low-coverage-2.5x": dict(
        dist=("lognormal", np.log10(90), 0.25), ranges=[(70, 110, 1)],
        units=[("mg/dl", 1, 6000), ("mmol/l", 0.4, 4000)],
        cand=(5000, {"mg/dl": .6, "mmol/l": .4}), cand_cov=.1, cand_noise=0,
        note="as syn-close-2.5x, only 10 % of the unit-less rows have a range"),
    "syn-range-noise-5x": dict(
        dist=("lognormal", np.log10(20), 0.15), ranges=[(12, 30, 1)],
        units=[("umol/l", 1, 6000), ("mg/l", 0.2, 4000)],
        cand=(4000, {"umol/l": .5, "mg/l": .5}), cand_cov=.8, cand_noise=.1,
        note="10 % of the unit-less rows carry the other unit's range; the value check must catch it"),
    "syn-unambiguous": dict(
        dist=("lognormal", np.log10(80), 0.10), ranges=[(60, 105, 1)],
        units=[("umol/l", 1, 10000), ("mmol/l", 0.001, 40)],
        cand=(3000, {"umol/l": 1.0}), cand_cov=.8, cand_noise=0,
        note="unambiguous test (99.6 % one unit)"),
    "syn-wrong-dominant-2x": dict(
        dist=("lognormal", np.log10(50), 0.25), ranges=[(35, 70, 1)],
        units=[("u/l", 1, 8000), ("iu/dl", 2.0, 1400)],
        cand=(3000, {"iu/dl": 1.0}), cand_cov=.8, cand_noise=0,
        note="the unit-less rows are all in the minority unit, 2x from the dominant one"),
    "syn-three-units": dict(
        dist=("lognormal", np.log10(3), 0.15), ranges=[(2, 4.5, 1)],
        units=[("mmol/l", 1, 5000), ("mg/dl", 10, 2000), ("umol/l", 1000, 3000)],
        cand=(5000, {"mmol/l": .5, "mg/dl": .2, "umol/l": .3}), cand_cov=.8, cand_noise=0,
        note="three units (1x, 10x, 1000x) mixed in the unit-less rows"),
    "syn-no-cand-ranges": dict(
        dist=("lognormal", np.log10(5), 0.12), ranges=[(3.5, 6.5, 1)],
        units=[("mmol/l", 1, 6000), ("umol/l", 1000, 4000)],
        cand=(3000, {"mmol/l": .5, "umol/l": .5}), cand_cov=0, cand_noise=0,
        note="no ranges on the unit-less rows: the ranges must fall back to the value engine"),
}


def _base_values(dist, n, rng):
    if dist[0] == "lognormal":
        return 10 ** rng.normal(dist[1], dist[2], n)
    _, mean, sd, step, lo, hi = dist
    return np.clip(np.round(rng.normal(mean, sd, n) / step) * step, lo, hi)


def _sig(x, digits=3):
    x = np.asarray(x, dtype=float)
    out = np.full(len(x), "NA", dtype=object)
    ok = np.isfinite(x)
    for i in np.flatnonzero(ok):
        v = x[i]
        out[i] = "0" if v == 0 else np.format_float_positional(
            float(f"{v:.{digits}g}"), trim="-")
    return out


def _ranges_for(rows_units, units, ranges, rng, lab_var=0.03):
    """Pick a (lower, upper) per row for its unit: a base range x factor x lab variant."""
    factor = {u: f for u, f, _ in units}
    w = np.array([r[2] for r in ranges], dtype=float)
    pick = rng.choice(len(ranges), size=len(rows_units), p=w / w.sum())
    var  = rng.choice([1 - lab_var, 1.0, 1 + lab_var], size=len(rows_units))
    f    = np.array([factor[u] for u in rows_units], dtype=float)
    lo   = np.array([np.nan if ranges[k][0] is None else ranges[k][0] for k in pick]) * f * var
    hi   = np.array([np.nan if ranges[k][1] is None else ranges[k][1] for k in pick]) * f * var
    return lo, hi


def _swap_unit(rows_units, units, frac, rng):
    """For a fraction of rows, the unit whose range they get is another one (range noise)."""
    names = [u for u, _, _ in units]
    out = np.array(rows_units, dtype=object)
    if frac <= 0 or len(names) < 2:
        return out
    m = rng.random(len(out)) < frac
    for i in np.flatnonzero(m):
        out[i] = rng.choice([u for u in names if u != out[i]])
    return out


def build(seed=1, target_cov=0.8, target_noise=0.02):
    rng = np.random.default_rng(seed)
    frames, truth, desc = [], {}, []
    for k, (name, sc) in enumerate(SCENARIOS.items()):
        units = sc["units"]
        factor = {u: f for u, f, _ in units}
        truth[name] = factor
        omop = str(3000000 + k)
        # target rows
        t_units = np.concatenate([[u] * n for u, _, n in units])
        t_base  = _base_values(sc["dist"], len(t_units), rng)
        t_val   = t_base * np.array([factor[u] for u in t_units])
        r_units = _swap_unit(t_units, units, target_noise, rng)
        lo, hi  = _ranges_for(r_units, units, sc["ranges"], rng)
        has     = rng.random(len(t_units)) < target_cov
        lo[~has] = np.nan
        hi[~has] = np.nan
        frames.append(_frame(name, omop, t_units, t_val, lo, hi, t_units, factor))
        # candidate rows
        n_c, mix = sc["cand"]
        cu = rng.choice(list(mix), size=n_c, p=np.array(list(mix.values())) / sum(mix.values()))
        c_base = _base_values(sc["dist"], n_c, rng)
        c_val  = c_base * np.array([factor[u] for u in cu])
        r_units = _swap_unit(cu, units, sc["cand_noise"], rng)
        lo, hi  = _ranges_for(r_units, units, sc["ranges"], rng)
        has     = rng.random(n_c) < sc["cand_cov"]
        lo[~has] = np.nan
        hi[~has] = np.nan
        frames.append(_frame(name, omop, np.array(["NA"] * n_c, dtype=object), c_val, lo, hi, cu, factor))
        desc.append(dict(TEST_NAME=name, UNITS="|".join(f"{u}x{f:g}:{n}" for u, f, n in units),
                         CANDIDATES=n_c, CAND_MIX="|".join(f"{u}:{s:g}" for u, s in mix.items()),
                         CAND_RANGE_COV=sc["cand_cov"], CAND_RANGE_NOISE=sc["cand_noise"], NOTE=sc["note"]))
    df = pd.concat(frames, ignore_index=True)
    return df.sample(frac=1, random_state=seed).reset_index(drop=True), truth, pd.DataFrame(desc)


def _frame(name, omop, units, values, lo, hi, true_units, factor):
    v = _sig(values)
    return pd.DataFrame({
        "TEST_NAME_ABBREVIATION": name,
        "MEASUREMENT_VALUE": v,
        "source::MEASUREMENT_VALUE": v,
        "cleaned-pre-inj::MEASUREMENT_UNIT": units,
        "harmonization_omop::OMOP_ID": np.where(units == "NA", "NA", omop),
        "REFERENCE_RANGE_LOWER_VALUE": _sig(lo),
        "REFERENCE_RANGE_UPPER_VALUE": _sig(hi),
        "REFERENCE_RANGE_SOURCE": np.where(np.isfinite(lo) | np.isfinite(hi), "text", "NA"),
        "TRUE_UNIT": true_units,
        "TRUE_FACTOR": np.array([factor[u] for u in true_units], dtype=float),
    })


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--out", required=True, help="Output folder (scratch, e.g. /mnt/disks/data/claude/...)")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--target-cov", type=float, default=0.8, help="Share of target rows with a range")
    p.add_argument("--target-noise", type=float, default=0.02,
                   help="Share of target rows carrying another unit's range")
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df, truth, desc = build(args.seed, args.target_cov, args.target_noise)
    df.to_parquet(out / "synthetic_inject.parquet", index=False)
    (out / "synthetic_truth.json").write_text(json.dumps(truth, indent=1))
    desc.to_csv(out / "synthetic_scenarios.tsv", sep="\t", index=False)
    print(f"Wrote {len(df):,} rows, {df['TEST_NAME_ABBREVIATION'].nunique()} tests → {out}/synthetic_inject.parquet")


if __name__ == "__main__":
    main()
