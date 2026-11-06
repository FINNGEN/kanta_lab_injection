#!/usr/bin/env python3
"""
make_synthetic.py — a small engine-like parquet for testing suggest.py and the portal locally.

Uses real test names, units and OMOP concepts from reference/LABfi.tsv, so the Usagi and
conversion-table lookups behave as on real data, but every value, range, lab and date is
invented. One scenario per case the suggestions have to handle:

  u-ph · 1                 sent without a unit (or "__________"), injected as 1 by the primary
                           injection; (u-ph, 1) is not in Usagi → 1 → ph (unit history on the page)
  b-hkr · osuus            pair missing, concept from the test's other units → osuus → ratio (high)
  p-ferrit · (no unit)     mapped, unit missing → ug/l (high)
  p-krea · uml/l           pair waiting for Usagi review, typo unit → umol/l (high)
  du-alb · mg              pair waiting for review, few ranges → mg/24h (medium, values only)
  p-inr-hy · (no unit)     anticoagulated patients: values differ, ranges (2-3) match → inr (medium)
  du-na · mmol             current rule says mmol/l, evidence says mmol/24h (different from rule)
  fp-pth · pg/ml           test maps to two concepts (ng/l, pmol/l) → ng/l
  s-d-25 · (no unit)       values look like nmol/l, ranges are on the umol/l scale (conflict)
  u-alb · (no unit)        half the labs report mg/l, half g/l (needs a split)
  b-hb-a1c · (no unit)     values and ranges in mmol/mol, only % converts (no unit fits)
  b-hkr · fl               unchecked pair; the values are red-cell volumes, not hematocrit (no unit fits)
  -l-ind · (no unit)       concept with too few harmonized rows
  syn-no-usagi · (no unit) test not in Usagi
  l-monosyytti · (no unit) test only unchecked in Usagi, no concept
  rakkoaika · h            unchecked pair proposing a concept that has no harmonized rows
  b-t-cd4 · e6/l           the unit is right (e6/l, × 0.001 into e9/l), but the pair is unchecked in Usagi
                           without a concept → approve the Usagi row (no unit fix)

Every row has a FINNGENID (people repeat: about 3–4 rows each), and a few groups carry cases the
minimum-N rules must catch (scripts/common/disclosure.py):

  syn-few-people · mg      1500 failed rows from 3 people: left out before anything else
  u-ph · 1                 the rows spelled "__________" come from 2 people: that unit-history path
                           (and as many smaller ones as needed) is not shown
  p-ferrit · (no unit)     40 rows with the range 7–77, all from 2 people: that range pair is not shown
  du-alb · mg              its few rows with a range come from 3 people: the coverage and the range
                           shares are not published

Writes OUT/synthetic_engine.parquet and OUT/omop_injection.tsv (a stand-in for the current
rules, so "same as / different from the current rule" can be tested).

  python3 scripts/unharmonized/make_synthetic.py --out /mnt/disks/data/claude/unharm_synth
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

LABS = [f"SYN_LAB_{i:02d}" for i in range(1, 9)]
YEARS = list(range(2014, 2025))

# concept -> harmonized ("target") populations: (test, unit, cf to target unit, target unit,
#            log10 median in target unit, log10 sd, n rows, range in target unit (lo, hi) or None, coverage)
TARGETS = {
    "3009542":  [("b-hkr", "ratio", 1, "ratio", np.log10(0.41), 0.04, 9000, (0.35, 0.46), 0.9),
                 ("b-hkr", "%", 0.01, "ratio", np.log10(0.41), 0.04, 3000, (0.35, 0.46), 0.9)],
    "3001122":  [("p-ferrit", "ug/l", 1, "ug/l", np.log10(85), 0.38, 6000, (15, 300), 0.8)],
    "3020564":  [("p-krea", "umol/l", 1, "umol/l", np.log10(80), 0.12, 8000, (50, 110), 0.85)],
    "3027035":  [("du-alb", "mg/24h", 1, "mg/24h", np.log10(12), 0.5, 3000, (None, 30), 0.7)],
    "3032080":  [("p-tt-inr", "inr", 1, "inr", np.log10(1.05), 0.05, 7000, (0.9, 1.2), 0.85),
                 ("p-inr", "inr", 1, "inr", np.log10(2.4), 0.1, 3000, (2, 3), 0.85)],
    "3002079":  [("du-na", "mmol/24h", 1, "mmol/24h", np.log10(140), 0.2, 3000, (40, 220), 0.8)],
    "3003181":  [("du-na", "mmol/l", 1, "mmol/l", np.log10(70), 0.3, 1500, (20, 150), 0.6)],
    "3000067":  [("fp-pth", "ng/l", 1, "ng/l", np.log10(50), 0.22, 4000, (15, 65), 0.8)],
    "3010566":  [("fp-pth", "pmol/l", 1, "pmol/l", np.log10(5.3), 0.22, 2000, (1.6, 6.9), 0.8)],
    "40771025": [("s-d-25", "nmol/l", 1, "nmol/l", np.log10(70), 0.2, 6000, (50, 125), 0.7)],
    "3012516":  [("u-alb", "mg/l", 1, "mg/l", np.log10(8), 0.45, 5000, (None, 20), 0.8)],
    "3004410":  [("b-hb-a1c", "%", 1, "%", np.log10(5.7), 0.05, 6000, (4, 6), 0.85)],
    "3051611":  [("-l-ind", "index", 1, "index", np.log10(1.0), 0.2, 40, (None, 1.2), 0.5)],
    "3015736":  [("u-ph", "ph", 1, "ph", np.log10(6.0), 0.05, 4000, (5, 8), 0.7)],
    "3028167":  [("b-t-cd4", "e9/l", 1, "e9/l", np.log10(0.8), 0.25, 4000, (0.4, 1.6), 0.7)],
}

# failed groups: (test, unit, omop_raw, n, components, coverage, source-unit spellings)
#   component = (share, log10 median of the *reported* value, log10 sd, reported range (lo, hi) or None)
GROUPS = [
    ("u-ph", "1", "NA", 1500, [(1, np.log10(6.1), 0.05, (5, 8))], 0.6, ["NA", "__________", "1"]),
    ("b-hkr", "osuus", "NA", 2400, [(1, np.log10(0.40), 0.04, (0.35, 0.46))], 0.9, ["osuus", "Osuus"]),
    ("p-ferrit", "NA", "3001122", 1800, [(1, np.log10(95), 0.38, (15, 300))], 0.7, ["NA"]),
    ("p-krea", "uml/l", "0", 700, [(1, np.log10(82), 0.12, (50, 110))], 0.8, ["uml/l", "UML/L"]),
    ("du-alb", "mg", "0", 650, [(1, np.log10(13), 0.5, (None, 30))], 0.03, ["mg"]),
    ("p-inr-hy", "NA", "3032080", 1500, [(1, np.log10(2.5), 0.09, (2, 3))], 0.6, ["NA"]),
    ("du-na", "mmol", "0", 900, [(1, np.log10(150), 0.2, (40, 220))], 0.6, ["mmol", "MMOL"]),
    ("fp-pth", "pg/ml", "NA", 800, [(1, np.log10(52), 0.22, (15, 65))], 0.75, ["pg/ml", "pg/mL"]),
    ("s-d-25", "NA", "40771025", 1200, [(1, np.log10(68), 0.2, (0.05, 0.125))], 0.5, ["NA"]),
    ("u-alb", "NA", "3012516", 2000, [(0.55, np.log10(8), 0.45, (None, 20)),
                                      (0.45, np.log10(0.008), 0.45, (None, 0.02))], 0.8, ["NA"]),
    ("b-hb-a1c", "NA", "3004410", 1300, [(1, np.log10(38), 0.08, (20, 42))], 0.8, ["NA"]),
    ("-l-ind", "NA", "3051611", 300, [(1, np.log10(1.1), 0.2, (None, 1.2))], 0.5, ["NA"]),
    ("syn-no-usagi", "NA", "NA", 400, [(1, np.log10(3), 0.3, (1, 5))], 0.4, ["NA"]),
    ("b-hkr", "fl", "0", 900, [(1, np.log10(90), 0.04, (82, 98))], 0.8, ["fl"]),
    ("l-monosyytti", "NA", "0", 600, [(1, np.log10(7), 0.2, (2, 12))], 0.6, ["NA"]),
    ("rakkoaika", "h", "0", 500, [(1, np.log10(3), 0.2, None)], 0.0, ["h"]),
    ("syn-few-people", "mg", "NA", 1500, [(1, np.log10(3), 0.3, (1, 5))], 0.4, ["mg"]),
    ("b-t-cd4", "e6/l", "0", 900, [(1, np.log10(800), 0.25, (400, 1600))], 0.7, ["e6/l"]),
]

RULES = [("b-hkr", "osuus", "ratio"), ("du-na", "mmol", "mmol/l")]


def sig3(x):
    x = np.asarray(x, dtype=float)
    out = x.copy()
    ok = np.isfinite(x) & (x > 0)
    p = 10.0 ** (2 - np.floor(np.log10(x[ok])))
    out[ok] = np.round(x[ok] * p) / p
    return out


def lab_ranges(rng, lo, hi, n, coverage, labs):
    """Reported limits per row: each lab's range = the base range × a small lab factor."""
    jit = {lab: f for lab, f in zip(LABS, [1, 1, 0.97, 1.03, 1, 0.95, 1.05, 1])}
    f = np.array([jit[l] for l in labs])
    has = rng.random(n) < coverage
    lower = np.where(has & (lo is not None), sig3((lo or np.nan) * f), np.nan)
    upper = np.where(has & (hi is not None), sig3((hi or np.nan) * f), np.nan)
    return lower, upper


_NEXT_PERSON = [0]


def people(rng, n, per_person):
    """A FINNGENID per row: n rows from about n / per_person new people."""
    k = max(1, int(round(n / per_person)))
    ids = np.array([f"FG{_NEXT_PERSON[0] + i:08d}" for i in range(k)])
    _NEXT_PERSON[0] += k
    return rng.choice(ids, n)


def dates(rng, n):
    y = rng.choice(YEARS, n)
    m = rng.integers(1, 13, n)
    d = rng.integers(1, 28, n)
    return [f"{a}-{b:02d}-{c:02d}T09:00:00" for a, b, c in zip(y, m, d)]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    frames = []

    for omop, pops in TARGETS.items():
        for test, unit, cf, tunit, mu, sd, n, rng_t, cov in pops:
            vt = 10 ** (mu + rng.normal(0, sd, n))            # target-unit values
            reported = sig3(vt / cf)
            labs = rng.choice(LABS[:5], n)
            lo, hi = rng_t if rng_t else (None, None)
            lower, upper = lab_ranges(rng, None if lo is None else lo / cf, None if hi is None else hi / cf, n, cov, labs)
            frames.append(pd.DataFrame({
                "TEST_NAME_ABBREVIATION": test, "MEASUREMENT_UNIT": unit, "source::MEASUREMENT_UNIT": unit,
                "cleaned-pre-inj::MEASUREMENT_UNIT": unit, "cleaned-post-inj::MEASUREMENT_UNIT": unit,
                "MEASUREMENT_VALUE": [f"{v:g}" for v in reported],
                "harmonization_omop::OMOP_ID": omop,
                "harmonization_omop::MEASUREMENT_VALUE": [f"{v * cf:g}" for v in reported],
                "harmonization_omop::MEASUREMENT_UNIT": tunit,
                "harmonization_omop::CONVERSION_FACTOR": f"{cf:g}",
                "REFERENCE_RANGE_LOWER_VALUE": lower, "REFERENCE_RANGE_UPPER_VALUE": upper,
                "SERVICE_PROVIDER": labs, "APPROX_EVENT_DATETIME": dates(rng, n),
                "FINNGENID": people(rng, n, 4),
            }))

    for test, unit, omop_raw, n, comps, cov, spell in GROUPS:
        shares = np.array([c[0] for c in comps])
        which = rng.choice(len(comps), n, p=shares / shares.sum())
        labs_pool = [LABS[5:8], LABS[2:5]] if len(comps) > 1 else [LABS[3:8]]
        vals = np.empty(n)
        lower = np.full(n, np.nan)
        upper = np.full(n, np.nan)
        labs = np.empty(n, dtype=object)
        for k, (_, mu, sd, rg) in enumerate(comps):
            m = which == k
            vals[m] = sig3(10 ** (mu + rng.normal(0, sd, m.sum())))
            labs[m] = rng.choice(labs_pool[k % len(labs_pool)], m.sum())
            if rg:
                lower[m], upper[m] = lab_ranges(rng, rg[0], rg[1], m.sum(), cov, labs[m])
        src = rng.choice(spell, n, p=None if len(spell) == 1 else [0.9] + [0.1 / (len(spell) - 1)] * (len(spell) - 1))
        frames.append(pd.DataFrame({
            "TEST_NAME_ABBREVIATION": test, "MEASUREMENT_UNIT": unit,
            "source::MEASUREMENT_UNIT": src,
            "cleaned-pre-inj::MEASUREMENT_UNIT": ["NA" if x in ("NA", "__________", "-") else x.lower() for x in src],
            "cleaned-post-inj::MEASUREMENT_UNIT": unit,
            "MEASUREMENT_VALUE": [f"{v:g}" for v in vals],
            "harmonization_omop::OMOP_ID": omop_raw, "harmonization_omop::MEASUREMENT_VALUE": "NA",
            "harmonization_omop::MEASUREMENT_UNIT": "NA", "harmonization_omop::CONVERSION_FACTOR": "NA",
            "REFERENCE_RANGE_LOWER_VALUE": lower, "REFERENCE_RANGE_UPPER_VALUE": upper,
            "SERVICE_PROVIDER": labs, "APPROX_EVENT_DATETIME": dates(rng, n),
            "FINNGENID": people(rng, n, 1500 if test == "syn-few-people" else 3),
        }))

    df = pd.concat(frames, ignore_index=True)
    # cases for the minimum-N rules (see the docstring)
    failed = df["harmonization_omop::MEASUREMENT_VALUE"] == "NA"
    m = failed & (df["TEST_NAME_ABBREVIATION"] == "u-ph") & (df["source::MEASUREMENT_UNIT"] == "__________")
    df.loc[m, "FINNGENID"] = rng.choice(["FG90000001", "FG90000002"], m.sum())
    m = np.flatnonzero(failed & (df["TEST_NAME_ABBREVIATION"] == "p-ferrit"))[:40]
    df.loc[m, ["REFERENCE_RANGE_LOWER_VALUE", "REFERENCE_RANGE_UPPER_VALUE"]] = [7.0, 77.0]
    df.loc[m, "FINNGENID"] = rng.choice(["FG90000003", "FG90000004"], len(m))
    m = (failed & (df["TEST_NAME_ABBREVIATION"] == "du-alb")
         & (df["REFERENCE_RANGE_LOWER_VALUE"].notna() | df["REFERENCE_RANGE_UPPER_VALUE"].notna()))
    df.loc[m, "FINNGENID"] = rng.choice(["FG90000005", "FG90000006", "FG90000007"], m.sum())
    df = df.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    args.out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out / "synthetic_engine.parquet", index=False)
    pd.DataFrame([{"TEST_NAME_ABBREVIATION": t, "source_unit_clean": u, "source_unit_clean_fix": f,
                   "notes": "synthetic", "INJECTION_NOTES": "VALID"} for t, u, f in RULES]
                 ).to_csv(args.out / "omop_injection.tsv", sep="\t", index=False)
    print(f"Wrote {args.out}/synthetic_engine.parquet ({len(df):,} rows) and omop_injection.tsv")


if __name__ == "__main__":
    main()
