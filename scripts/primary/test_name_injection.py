#!/usr/bin/env python3
"""
test_name_injection.py

Identifies lab measurements that have a numeric value but no unit, characterises
the unit distribution of matching records that do have a unit, and optionally
runs the injection engine to validate each candidate TEST_NAME.

Usage
-----
  python3 scripts/primary/test_name_injection.py [PARQUET] [options]

PARQUET defaults to the most recent engine run in /mnt/disks/data/kanta/engine
(<prefix>_<YYYY_MM_DD>/<prefix>_<YYYY_MM_DD>.parquet, latest date in the folder name).
"""

import argparse
import gzip
import json
import os
import re
import subprocess
import sys
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd

_SCRIPTS = Path(__file__).resolve().parents[1]          # scripts/
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from common import disclosure, injection_engine, split_eval   # noqa: E402
from common.clickhouse import ENGINE_DIR, clickhouse, latest_engine_output   # noqa: E402  (tni.clickhouse is used by other scripts)
from plot_exploration import make_scatter_plot         # noqa: E402
from release_check import check_release                # noqa: E402


# ---------------------------------------------------------------------------
# ClickHouse
# ---------------------------------------------------------------------------

# clickhouse() lives in common/clickhouse.py (imported above).


# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------

_COUNTS_MIN = 50  # fixed baseline for the cached query; --min-count filters post-load


def query_counts(parquet, out="test_name_counts.tsv"):
    if Path(out).exists():
        df = pd.read_csv(out, sep="\t", na_values=["\\N"])
        if "OMOP_CONCEPT_ID" in df.columns:
            df["OMOP_CONCEPT_ID"] = pd.to_numeric(df["OMOP_CONCEPT_ID"], errors="coerce").astype("Int64")
            print(f"{out} already exists, skipping.")
            return df
        print(f"{out} exists but missing OMOP_CONCEPT_ID — re-querying.")

    result = clickhouse(f"""
        SELECT TEST_NAME, count() AS COUNT, any(OMOP_CONCEPT_ID) AS OMOP_CONCEPT_ID
        FROM (
            SELECT
                TEST_NAME_ABBREVIATION                     AS TEST_NAME,
                MEASUREMENT_VALUE                          AS MEASUREMENT_VALUE,
                `cleaned-pre-inj::MEASUREMENT_UNIT`         AS MEASUREMENT_UNIT_PRE_INJ,
                `harmonization_omop::OMOP_ID`                AS OMOP_CONCEPT_ID
            FROM file('{parquet}')
        )
        WHERE MEASUREMENT_VALUE != 'NA'
          AND MEASUREMENT_UNIT_PRE_INJ = 'NA'
          AND TEST_NAME != 'NA'
        GROUP BY TEST_NAME
        HAVING COUNT > {_COUNTS_MIN}
        ORDER BY COUNT DESC
        FORMAT TSVWithNames
    """)
    Path(out).write_text(result)
    df = pd.read_csv(out, sep="\t", na_values=["\\N"])
    df["OMOP_CONCEPT_ID"] = pd.to_numeric(df["OMOP_CONCEPT_ID"], errors="coerce").astype("Int64")
    print(f"Wrote {out}  ({len(df)} rows)")
    return df


def query_details(parquet, counts_file="test_name_counts.tsv", out="test_name_details.tsv"):
    if Path(out).exists():
        print(f"{out} already exists, skipping.")
        return pd.read_csv(out, sep="\t")

    counts_abs = str(Path(counts_file).resolve())
    result = clickhouse(f"""
        WITH
        renamed AS (
            SELECT
                TEST_NAME_ABBREVIATION              AS TEST_NAME,
                `source::MEASUREMENT_VALUE`          AS MEASUREMENT_VALUE_SOURCE,
                `cleaned-pre-inj::MEASUREMENT_UNIT`  AS MEASUREMENT_UNIT_PRE_INJ
            FROM file('{parquet}')
            WHERE TEST_NAME_ABBREVIATION != 'NA'
        ),
        global_names AS (
            SELECT DISTINCT TEST_NAME
            FROM file('{counts_abs}', TSVWithNames)
        ),
        top3_units AS (
            SELECT
                TEST_NAME,
                argMax(MEASUREMENT_UNIT_PRE_INJ, unit_cnt) AS UNIT,
                concat('{{', arrayStringConcat(groupArray(unit_json), ','), '}}') AS PREVALENCE_DICT
            FROM (
                SELECT TEST_NAME, MEASUREMENT_UNIT_PRE_INJ, unit_cnt, unit_json
                FROM (
                    SELECT
                        TEST_NAME, MEASUREMENT_UNIT_PRE_INJ, unit_cnt,
                        ROW_NUMBER() OVER (PARTITION BY TEST_NAME ORDER BY unit_cnt DESC) AS rn,
                        concat(
                            MEASUREMENT_UNIT_PRE_INJ, ':',
                            toString(round(100.0 * unit_cnt / SUM(unit_cnt) OVER (PARTITION BY TEST_NAME), 2))
                        ) AS unit_json
                    FROM (
                        SELECT TEST_NAME, MEASUREMENT_UNIT_PRE_INJ, count() AS unit_cnt
                        FROM renamed
                        WHERE MEASUREMENT_VALUE_SOURCE != 'NA'
                          AND MEASUREMENT_UNIT_PRE_INJ != 'NA'
                        GROUP BY TEST_NAME, MEASUREMENT_UNIT_PRE_INJ
                    ) AS sub
                ) AS ranked
                WHERE rn <= 3
            ) AS top3
            GROUP BY TEST_NAME
        ),
        total_counts AS (
            SELECT TEST_NAME, count() AS COUNT
            FROM renamed
            WHERE MEASUREMENT_VALUE_SOURCE != 'NA'
              AND MEASUREMENT_UNIT_PRE_INJ != 'NA'
            GROUP BY TEST_NAME
        )
        SELECT
            t.TEST_NAME       AS TEST_NAME,
            t.COUNT           AS COUNT,
            u.UNIT            AS UNIT,
            u.PREVALENCE_DICT AS PREVALENCE_DICT
        FROM total_counts t
        LEFT JOIN top3_units u USING (TEST_NAME)
        INNER JOIN global_names g USING (TEST_NAME)
        ORDER BY t.COUNT DESC
        FORMAT TSVWithNames
    """)
    Path(out).write_text(result)
    df = pd.read_csv(out, sep="\t")
    print(f"Wrote {out}  ({len(df)} rows)")
    return df


def query_summaries(parquet, out="test_name_unit_summaries.tsv"):
    """
    Rows and people per TEST_NAME x pre-injection unit ('NA' = no unit), counted once up front
    so that every published group can be checked against the minimum number of people.

    The value column follows the rest of the pipeline: MEASUREMENT_VALUE for the no-unit rows
    (the candidates, as in query_counts), source::MEASUREMENT_VALUE for rows with a unit (the
    targets and the unit prevalence, as in query_details / _query_test_values).

      N_RECORDS, N_SUBJECTS          rows with a value (the rows behind COUNT and PREVALENCE_DICT)
      N_RECORDS_NUM, N_SUBJECTS_NUM  rows whose value is a number (the rows behind N_CANDIDATE /
                                     N_TARGET and the distributions)

    Row-level counts, small ones included: a cache for the pipeline, never released.
    """
    if Path(out).exists():
        print(f"{out} already exists, skipping.")
        return _read_summaries(out)
    result = clickhouse(f"""
        SELECT TEST_NAME, UNIT,
               count()                         AS N_RECORDS,
               uniqExact(FINNGENID)            AS N_SUBJECTS,
               countIf(IS_NUM)                 AS N_RECORDS_NUM,
               uniqExactIf(FINNGENID, IS_NUM)  AS N_SUBJECTS_NUM
        FROM (
            SELECT
                TEST_NAME_ABBREVIATION                        AS TEST_NAME,
                `cleaned-pre-inj::MEASUREMENT_UNIT`           AS UNIT,
                CASE WHEN `cleaned-pre-inj::MEASUREMENT_UNIT` = 'NA'
                     THEN MEASUREMENT_VALUE
                     ELSE `source::MEASUREMENT_VALUE` END     AS VALUE,
                FINNGENID                                     AS FINNGENID,
                toFloat64OrNull(VALUE) IS NOT NULL            AS IS_NUM
            FROM file('{parquet}')
            WHERE TEST_NAME_ABBREVIATION != 'NA'
        )
        WHERE VALUE != 'NA'
        GROUP BY TEST_NAME, UNIT
        ORDER BY TEST_NAME, UNIT
        FORMAT TSVWithNames
    """)
    Path(out).write_text(result)
    df = _read_summaries(out)
    print(f"Wrote {out}  ({len(df)} rows)")
    return df


def _read_summaries(path):
    # keep_default_na=False: the unit 'NA' (no unit) is a key, not a missing value
    return pd.read_csv(path, sep="\t", keep_default_na=False, na_values=["\\N"],
                       dtype={"TEST_NAME": str, "UNIT": str})


def published_units(summaries, min_subjects):
    """
    TEST_NAME -> [(unit, share %), ...] (largest first) for publication: only units with at
    least min_subjects people, and the shares are taken over those units alone, so the rows of
    the rarer units cannot be worked back from the shares and the published counts.
    """
    u = summaries[(summaries["UNIT"] != "NA") & (summaries["N_SUBJECTS"] >= min_subjects)]
    out = {}
    for name, g in u.groupby("TEST_NAME", sort=False):
        g = g.sort_values("N_RECORDS", ascending=False)
        tot = g["N_RECORDS"].sum()
        out[name] = [(unit, 100.0 * n / tot) for unit, n in zip(g["UNIT"], g["N_RECORDS"])]
    return out


def format_units(units, n=3):
    """[(unit, share), ...] -> '{unit:share,...}' (top n), the PREVALENCE_DICT format."""
    if not units:
        return "NA"
    return "{" + ",".join(f"{u}:{round(p, 2):g}" for u, p in units[:n]) + "}"


# ---------------------------------------------------------------------------
# Plot table
# ---------------------------------------------------------------------------


def add_people(plot_name, summaries, min_subjects):
    """
    People behind each TEST_NAME's groups, from the summaries:

      N_SUBJECTS       people with no-unit rows (the candidates; rows: COUNT)
      TARGET_SUBJECTS  people with rows in the dominant unit (TOP_UNIT)

    TEST_NAMEs whose no-unit rows come from fewer than min_subjects people are dropped here:
    they are not analysed, plotted or counted anywhere.
    """
    s = summaries.set_index(["TEST_NAME", "UNIT"])
    df = plot_name.copy()
    keys_c = list(zip(df["TEST_NAME"], ["NA"] * len(df)))
    keys_t = list(zip(df["TEST_NAME"], df["TOP_UNIT"].fillna("NA")))
    df["N_SUBJECTS"] = s["N_SUBJECTS"].reindex(keys_c).fillna(0).astype(int).values
    df["TARGET_SUBJECTS"] = [int(s["N_SUBJECTS"].get(k, 0)) if k[1] != "NA" else 0 for k in keys_t]

    n_rec = s["N_RECORDS"].reindex(keys_c).values
    off = df["COUNT"].values != n_rec
    if off.any():
        print(f"WARNING: {int(off.sum())} TEST_NAME(s) count differently in the summaries and "
              f"test_name_counts.tsv (stale cache?): {df.loc[off, 'TEST_NAME'].tolist()[:5]}")

    small = df["N_SUBJECTS"] < min_subjects
    if small.any():
        print(f"Dropped {int(small.sum())} TEST_NAME(s) whose no-unit rows come from fewer than "
              f"{min_subjects} people")
    return df[~small].reset_index(drop=True)


def build_plot_table(counts, details):
    def parse_top_prevalence(d):
        if pd.isna(d):
            return None
        pcts = re.findall(r':(\d+\.?\d*)[,}]', str(d))
        return max(float(p) for p in pcts) if pcts else None

    details = details.copy()
    details["top_prevalence"] = details["PREVALENCE_DICT"].apply(parse_top_prevalence)

    plot_name = counts.merge(
        details[["TEST_NAME", "top_prevalence", "COUNT", "UNIT"]].rename(
            columns={"COUNT": "N_WITH_UNIT", "UNIT": "TOP_UNIT"}),
        on="TEST_NAME", how="left",
    )
    plot_name["top_prevalence"] = plot_name["top_prevalence"].fillna(0)
    plot_name["N_WITH_UNIT"]    = plot_name["N_WITH_UNIT"].fillna(0).astype(int)
    return plot_name


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

_SUMMARY_THRESHOLDS = [95, 98, 99, 100]


def _cat_at(df, t, min_target_n, min_subjects=0):
    """NO_DATA also when the dominant unit's rows come from fewer than min_subjects people
    (TARGET_SUBJECTS, when known): such a reference is neither tested nor published."""
    few_people = (df["TARGET_SUBJECTS"] < min_subjects) if "TARGET_SUBJECTS" in df else False
    return np.select(
        [
            (df["N_WITH_UNIT"] < min_target_n) | few_people,
            df["top_prevalence"] >= t,
            df["top_prevalence"] > 0,
        ],
        ["NO_DATA", "UNAMBIGUOUS", "AMBIGUOUS"],
        default="NO_DATA",
    )


def _add_category(plot_name, threshold, min_target_n, min_subjects=0):
    """Add CATEGORY (operational) and CATEGORY_{t} (exploratory) columns."""
    df = plot_name.copy()
    df["CATEGORY"] = _cat_at(df, threshold, min_target_n, min_subjects)
    for t in _SUMMARY_THRESHOLDS:
        df[f"CATEGORY_{t}"] = _cat_at(df, t, min_target_n, min_subjects)
    return df


def print_summary(plot_name, threshold):
    total   = len(plot_name)
    total_n = int(plot_name["COUNT"].sum())

    tw = 13  # test_names cell width
    mw = 18  # measurements cell width

    cats = ["UNAMBIGUOUS", "AMBIGUOUS", "NO_DATA"]

    # Header
    print()
    h1 = f"{'':>12}"
    h2 = f"{'Threshold':>10}  "
    sep = f"{'':>12}"
    for cat in cats:
        h1  += f"  {cat:^{tw+mw+2}}"
        h2  += f"  {'TEST_NAMES':>{tw}}  {'MEASUREMENTS':>{mw}}"
        sep += f"  {'-'*tw}  {'-'*mw}"
    print(h1)
    print(h2)
    print(sep)

    for t in _SUMMARY_THRESHOLDS:
        marker = " *" if t == threshold else "  "
        row = f"{t:>9}%{marker}"
        for cat in cats:
            sub = plot_name[plot_name[f"CATEGORY_{t}"] == cat]
            n   = len(sub)
            m   = int(sub["COUNT"].sum())
            row += f"  {f'{n:,} ({100*n/total:.1f}%)':>{tw}}  {f'{m:,} ({100*m/total_n:.1f}%)':>{mw}}"
        print(row)

    print(sep)
    print(f"{'TOTAL':>12}  {total:>{tw},}{'':>{mw+4}}"
          f"  {'':>{tw}}  {total_n:>{mw},}")
    print(f"\n  * = active threshold ({threshold}%)\n")


def print_assignment_summary(udf, adf, plot_name, out_doc=None, out_tsv=None):
    """Fraction of TEST_NAMEs/measurements successfully assigned a unit (UNAMBIGUOUS + AMBIGUOUS only)."""
    count_map  = plot_name.set_index("TEST_NAME")["COUNT"].to_dict()
    omop_names = set(plot_name[plot_name["OMOP_CONCEPT_ID"].notna()]["TEST_NAME"])

    def _safe(df):
        return df if df is not None and len(df) else None

    udf_ = _safe(udf)
    adf_ = _safe(adf)

    assigned = {
        "UNAMBIGUOUS": set(udf_[udf_["OUTCOME"] == "PASS"]["TEST_NAME"]) if udf_ is not None else set(),
        "AMBIGUOUS":   set(adf_[adf_["OUTCOME"] == "PASS"]["TEST_NAME"]) if adf_ is not None else set(),
    }
    processed = {
        "UNAMBIGUOUS": set(udf_["TEST_NAME"]) if udf_ is not None else set(),
        "AMBIGUOUS":   set(adf_["TEST_NAME"]) if adf_ is not None else set(),
    }

    def _meas(names):
        return sum(count_map.get(n, 0) for n in names)

    def _build_row(name_filter=None):
        cells    = []
        all_asgn = set()
        all_proc = set()
        for cat in ["UNAMBIGUOUS", "AMBIGUOUS"]:
            asgn = assigned[cat]  if name_filter is None else assigned[cat]  & name_filter
            proc = processed[cat] if name_filter is None else processed[cat] & name_filter
            all_asgn |= asgn
            all_proc |= proc
            cells.append((f"{len(asgn):,}/{len(proc):,}", f"{_meas(asgn):,}/{_meas(proc):,}"))
        cells.append((f"{len(all_asgn):,}/{len(all_proc):,}", f"{_meas(all_asgn):,}/{_meas(all_proc):,}"))
        return cells

    rows = [
        ("All",         _build_row()),
        ("OMOP-mapped", _build_row(omop_names)),
    ]

    nw  = max(len(ns) for _, cells in rows for ns, _  in cells)
    mw  = max(len(ms) for _, cells in rows for _,  ms in cells)
    lw  = 12
    cw  = nw + 2 + mw
    cols = ["UNAMBIGUOUS", "AMBIGUOUS", "TOTAL"]

    text_lines = [
        "\nAssignment summary (successfully assigned)\n",
        f"{'':>{lw}}" + "".join(f"  {c:^{cw}}" for c in cols),
        f"{'':>{lw}}" + "".join(f"  {'names':>{nw}}  {'meas':>{mw}}" for _ in cols),
    ]
    div = f"{'':>{lw}}" + "".join(f"  {'-'*nw}  {'-'*mw}" for _ in cols)
    text_lines.append(div)
    for label, cells in rows:
        text_lines.append(f"{label:<{lw}}" + "".join(f"  {ns:>{nw}}  {ms:>{mw}}" for ns, ms in cells))
    text_lines.append(div)

    for line in text_lines:
        print(line)

    if out_doc:
        Path(out_doc).write_text("```\n" + "\n".join(text_lines) + "\n```\n")
        print(f"Wrote {out_doc}")

    if out_tsv:
        tsv_rows = []
        for label, cells in rows:
            row = {"subset": label}
            for col, (ns, ms) in zip(cols, cells):
                row[f"{col}_names"] = ns
                row[f"{col}_meas"]  = ms
            tsv_rows.append(row)
        pd.DataFrame(tsv_rows).to_csv(out_tsv, sep="\t", index=False)
        print(f"Wrote {out_tsv}")


def dump_summary_md(plot_name, threshold, min_count, out="summary_table.md", out_tsv=None):
    total   = len(plot_name)
    total_n = int(plot_name["COUNT"].sum())
    cats    = ["UNAMBIGUOUS", "AMBIGUOUS", "NO_DATA"]

    lines = []
    lines.append(f"*Active threshold: {threshold}% — min count: {min_count:,} — "
                 f"{total:,} TEST_NAMEs, {total_n:,} measurements*\n")

    header = "| Threshold | " + " | ".join(
        f"{c} test names | {c} measurements" for c in cats
    ) + " |"
    sep = "| --- | " + " | ".join(["---: | ---:" for _ in cats]) + " |"
    lines.append(header)
    lines.append(sep)

    tsv_rows = []
    for t in _SUMMARY_THRESHOLDS:
        marker = " \\*" if t == threshold else ""
        row = f"| {t}%{marker} |"
        tsv_row = {"threshold": f"{t}%"}
        for cat in cats:
            sub = plot_name[plot_name[f"CATEGORY_{t}"] == cat]
            n   = len(sub)
            m   = int(sub["COUNT"].sum())
            row += f" {n:,} ({100*n/total:.1f}%) | {m:,} ({100*m/total_n:.1f}%) |"
            tsv_row[f"{cat}_names"] = n
            tsv_row[f"{cat}_meas"]  = m
        lines.append(row)
        tsv_rows.append(tsv_row)

    lines.append(f"| **TOTAL** | **{total:,}** | | " + "| |" * (len(cats) - 1))

    Path(out).write_text("\n".join(lines) + "\n")
    print(f"Wrote {out}")

    if out_tsv:
        pd.DataFrame(tsv_rows).to_csv(out_tsv, sep="\t", index=False)
        print(f"Wrote {out_tsv}")


# ---------------------------------------------------------------------------
# Injection engine
# ---------------------------------------------------------------------------

def _query_test_values(parquet, name, mode, unit=None):
    """
    Numeric values of one TEST_NAME's candidate (no-unit) or target (`unit`) rows, sorted, with
    a person key per value (cityHash64 of FINNGENID: people can be counted in any subset, e.g.
    a bimodal split's low side, without the IDs). Returns (values, persons).
    """
    if mode == "candidate":
        q = f"""
            SELECT MEASUREMENT_VALUE AS value, cityHash64(FINNGENID) AS person
            FROM file('{parquet}')
            WHERE TEST_NAME_ABBREVIATION = {{name:String}}
              AND `cleaned-pre-inj::MEASUREMENT_UNIT` = 'NA'
              AND MEASUREMENT_VALUE != 'NA'
            FORMAT TSV
        """
        out = clickhouse(q, name=name)
    else:
        q = f"""
            SELECT `source::MEASUREMENT_VALUE` AS value, cityHash64(FINNGENID) AS person
            FROM file('{parquet}')
            WHERE TEST_NAME_ABBREVIATION = {{name:String}}
              AND `cleaned-pre-inj::MEASUREMENT_UNIT` = {{unit:String}}
              AND `source::MEASUREMENT_VALUE` != 'NA'
            FORMAT TSV
        """
        out = clickhouse(q, name=name, unit=unit)
    vals, persons = [], []
    for line in out.strip().split("\n") if out.strip() else []:
        value, _, person = line.partition("\t")
        try:
            v = float(value.strip())
        except ValueError:
            continue
        vals.append(v)
        persons.append(int(person))
    vals = np.array(vals, dtype=float)
    persons = np.array(persons, dtype=np.uint64)
    order = np.argsort(vals, kind="stable")
    return vals[order], persons[order]


def _load_values(parquet, dump_dir, tag, name, mode, unit=None):
    """(values, persons) of a candidate / target, cached in dump_dir as {cand,targ}_{tag}.npz
    (the earlier value-only .npy caches have no person keys and are not used)."""
    path = os.path.join(dump_dir, f"{'cand' if mode == 'candidate' else 'targ'}_{tag}.npz")
    if os.path.exists(path):
        with np.load(path) as z:
            return z["v"], z["p"], True
    v, p = _query_test_values(parquet, name, mode, unit=unit)
    np.savez(path, v=v, p=p)
    return v, p, False


def _sample_deciles(df):
    """Return one row per COUNT decile (by rank), up to 10 rows."""
    df = df.reset_index(drop=True)
    n  = len(df)
    indices = sorted({int(round((i / 9) * (n - 1))) for i in range(10)})
    return df.iloc[indices]


_PCTS = [10, 20, 30, 40, 50, 60, 70, 80, 90]


def _fmt_deciles(arr):
    vals = np.percentile(arr, _PCTS)
    return "[" + ",".join(f"{v:.4g}" for v in vals) + "]"


# ---------------------------------------------------------------------------
# Per-test result cache
# ---------------------------------------------------------------------------

# Bump whenever the engine, the release rules (common/disclosure.py) or the result columns change:
# cached results computed by older code are then recomputed.
CACHE_VERSION = 1


def _jsonable(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serializable: {type(o)}")


class ResultCache:
    """
    One file per test in <dump_dir>/results/<pass>_<tag>.json: the test's result rows and its
    plot-data entry, with the key they were computed with. The key holds CACHE_VERSION, the
    release thresholds, the arguments and summary decisions that shape the test's result
    (`params`), and size + mtime of its value arrays (.npz). A cached result is reused only when
    its key equals the current one; force (--force) and plots (--plots: the figures are drawn
    while computing) always recompute. Files are written only for tests that ran to the end.
    Row-level-derived and private, like the .npz arrays: never released.
    """

    def __init__(self, dump_dir, pass_name, force=False):
        self.dir = Path(dump_dir) / "results"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.pass_name, self.force = pass_name, force
        self.pending = None
        self.hits = self.misses = 0

    def key(self, files, params):
        k = {"version": CACHE_VERSION, "pass": self.pass_name,
             "release": {"min_dist_subjects": disclosure.MIN_DIST_SUBJECTS,
                         "min_dist_records": disclosure.MIN_DIST_RECORDS, "pcts": list(disclosure.PCTS)},
             "params": params,
             "inputs": {Path(f).name: ([Path(f).stat().st_size, Path(f).stat().st_mtime]
                                       if Path(f).exists() else None) for f in files}}
        return json.loads(json.dumps(k, default=_jsonable))     # as it reads back from the file

    def get(self, tag, files, params):
        """(rows, plot_entry) when a valid cached result exists, else None."""
        path = self.dir / f"{self.pass_name}_{tag}.json"
        if self.force or not path.exists():
            self.misses += 1
            return None
        data = json.loads(path.read_text())
        if data.get("key") != self.key(files, params):
            self.misses += 1
            return None
        self.hits += 1
        return data["rows"], data.get("plot_data")

    def begin(self, tag, name, files, params, start):
        """A test is about to be computed; its result is saved by the next finish()."""
        self.pending = (tag, name, list(files), params, start)

    def finish(self, rows, plot_data):
        """Save the test begun last (it ran to the end: the loop moved on)."""
        if self.pending is None:
            return
        tag, name, files, params, start = self.pending
        self.pending = None
        payload = {"key": self.key(files, params), "rows": rows[start:], "plot_data": plot_data.get(name)}
        path = self.dir / f"{self.pass_name}_{tag}.json"
        tmp = path.with_suffix(".json.part")
        tmp.write_text(json.dumps(payload, default=_jsonable))
        os.replace(tmp, path)


def _run_engine(c_vals, c_persons, t_vals, t_persons, name, dump_dir, plots_dir=None,
                prevalence=None, tag_override=None, min_subjects=disclosure.MIN_SUBJECTS, plots=True):
    """Run KS / T / MAD pipeline, save the (private) diagnostic plot, return
    (updates, ks, t, mad, plot_data) with plot_data in its publishable form."""
    ks_step  = injection_engine._ks(c_vals, t_vals, ks_threshold=0.3, sig_level=0.05)
    t_step   = injection_engine._t(c_vals, t_vals, sig_level=0.05)
    mad_step = injection_engine._mad(c_vals, t_vals, n_mad=3.0)

    ks_mlogp = -np.log10(np.clip(ks_step.details["pval"], 1e-300, 1.0))
    t_mlogp  = -np.log10(np.clip(t_step.details["pval"],  1e-300, 1.0))

    if ks_step.passed:
        decided_by, outcome = "KS", "PASS"
    elif t_step.passed:
        decided_by, outcome = "T", "PASS"
    else:
        decided_by = "MAD"
        outcome    = "PASS" if mad_step.passed else "FAIL"

    result = injection_engine.PipelineResult(outcome, decided_by,
                                             [ks_step, t_step, mad_step])
    if plots:                              # private diagnostic PNG (raw values), never released
        injection_engine.plot_result(c_vals, t_vals, result, name, plots_dir or dump_dir,
                                     prevalence=prevalence, tag_override=tag_override)
    plot_data = injection_engine.compute_release_plot_data(c_vals, c_persons, t_vals, t_persons, result,
                                                           min_subjects=min_subjects)
    shown_c = disclosure.shows_distribution(plot_data["candidate"])
    shown_t = disclosure.shows_distribution(plot_data["target"])

    passed_tests = ",".join(n for n, s in [("KS", ks_step), ("T", t_step), ("MAD", mad_step)] if s.passed)
    updates = dict(
        KS_STAT=ks_step.details["stat"],  KS_MLOGP=ks_mlogp,
        KS_PASS="PASS" if ks_step.passed else "FAIL",
        T_STAT=t_step.details["stat"],    T_MLOGP=t_mlogp,
        T_PASS="PASS" if t_step.passed else "FAIL",
        MAD_DIST=mad_step.details["distance"],
        MAD_THRESHOLD=mad_step.details["threshold"],
        MAD_PASS="PASS" if mad_step.passed else "FAIL",
        OUTCOME=outcome,
        NOTES=passed_tests,
        # deciles are percentiles too: only for groups large enough to show a distribution
        CAND_DECILES=_fmt_deciles(c_vals) if shown_c else "NA",
        TARG_DECILES=_fmt_deciles(t_vals) if shown_t else "NA",
    )
    return updates, ks_step, t_step, mad_step, plot_data


def run_unambiguous(parquet, plot_name, details, dump_dir, plots_dir=None, test_mode=False,
                    out="unambiguous_results.tsv", min_subjects=disclosure.MIN_SUBJECTS, plots=True,
                    force=False):
    plot_data = {}
    os.makedirs(dump_dir, exist_ok=True)

    unambig = plot_name[plot_name["CATEGORY"] == "UNAMBIGUOUS"].sort_values("COUNT", ascending=False)
    if test_mode:
        unambig = _sample_deciles(unambig)
    test_names = unambig["TEST_NAME"].tolist()
    print(f"Unambiguous TEST_NAMEs: {len(test_names)}")

    dom_unit  = dict(zip(details["TEST_NAME"], details["UNIT"]))
    prev_dict = dict(zip(details["TEST_NAME"], details["PREVALENCE_DICT"]))

    rows = []
    n_pass = n_fail = n_skip = 0
    cache = ResultCache(dump_dir, "unambiguous", force=force or plots)

    for i, name in enumerate(test_names, 1):
        cache.finish(rows, plot_data)
        print(f"  [{i:>4}/{len(test_names)}] {name:<40}", end="  ", flush=True)

        tag  = name.replace("/", "_").replace(" ", "_")
        unit = dom_unit.get(name, "NA")
        prev = prev_dict.get(name, "NA")

        files  = [os.path.join(dump_dir, f"cand_{tag}.npz"), os.path.join(dump_dir, f"targ_{tag}.npz")]
        params = {"unit": unit, "prevalence": prev, "min_subjects": min_subjects}
        hit = cache.get(tag, files, params)
        if hit:
            cached_rows, entry = hit
            rows.extend(cached_rows)
            if entry is not None:
                plot_data[name] = entry
            outcome = cached_rows[0]["OUTCOME"] if cached_rows else "SKIP"
            n_pass += outcome == "PASS"; n_fail += outcome == "FAIL"; n_skip += outcome not in ("PASS", "FAIL")
            print(f"result=cache  {outcome}  [running: {n_pass}P/{n_fail}F/{n_skip}S]")
            continue
        cache.begin(tag, name, files, params, len(rows))

        c_vals, c_pers, cached = _load_values(parquet, dump_dir, tag, name, "candidate")
        print(f"cand={'cache' if cached else 'query'}", end="  ", flush=True)
        t_vals, t_pers, cached = _load_values(parquet, dump_dir, tag, name, "target", unit=unit)
        print(f"targ={'cache' if cached else 'query'}({unit})", end="  ", flush=True)
        c_people, t_people = disclosure.n_people(c_pers), disclosure.n_people(t_pers)

        print(f"N={len(c_vals):,}/{len(t_vals):,}  people={c_people:,}/{t_people:,}", end="  ", flush=True)

        row = dict(TEST_NAME=name, UNIT=unit, PREVALENCE_DICT=prev,
                   N_CANDIDATE=len(c_vals), N_TARGET=len(t_vals),
                   N_CANDIDATE_SUBJECTS=c_people, N_TARGET_SUBJECTS=t_people,
                   KS_STAT=np.nan, KS_MLOGP=np.nan, KS_PASS="NA",
                   T_STAT=np.nan,  T_MLOGP=np.nan,  T_PASS="NA",
                   MAD_DIST=np.nan, MAD_THRESHOLD=np.nan, MAD_PASS="NA",
                   OUTCOME="SKIP", NOTES="")

        # Both sides must come from enough people to be compared (and published) at all
        if min(c_people, t_people) < min_subjects:
            print(f"too few people (< {min_subjects})", end="  ", flush=True)
        elif len(c_vals) >= 2 and len(t_vals) >= 2:
            print("engine...", end="  ", flush=True)
            updates, ks, t, mad, pd_ = _run_engine(c_vals, c_pers, t_vals, t_pers, name, dump_dir,
                                                    plots_dir=plots_dir, prevalence=prev,
                                                    min_subjects=min_subjects, plots=plots)
            row.update(updates)
            plot_data[name] = pd_
            print(
                f"KS={'P' if ks.passed else 'F'}"
                f"(stat={ks.details['stat']:.3g},mlogp={updates['KS_MLOGP']:.1f},{ks.details['method']})"
                f"  T={'P' if t.passed else 'F'}(mlogp={updates['T_MLOGP']:.1f})"
                f"  MAD={'P' if mad.passed else 'F'}"
                f"(dist={mad.details['distance']:.3f},thr={mad.details['threshold']:.3f})",
                end="  ", flush=True,
            )

        if row["OUTCOME"] == "PASS":   n_pass += 1
        elif row["OUTCOME"] == "FAIL": n_fail += 1
        else:                          n_skip += 1

        rows.append(row)
        print(f"{row['OUTCOME']}  ({row['NOTES']})  "
              f"[running: {n_pass}P/{n_fail}F/{n_skip}S]  {prev}")
    cache.finish(rows, plot_data)
    print(f"Result cache (unambiguous): {cache.hits} reused, {cache.misses} computed")

    cols = ["TEST_NAME", "UNIT", "PREVALENCE_DICT", "N_CANDIDATE", "N_TARGET",
            "N_CANDIDATE_SUBJECTS", "N_TARGET_SUBJECTS",
            "CAND_DECILES", "TARG_DECILES",
            "KS_STAT", "KS_MLOGP", "KS_PASS",
            "T_STAT",  "T_MLOGP",  "T_PASS",
            "MAD_DIST", "MAD_THRESHOLD", "MAD_PASS",
            "OUTCOME", "NOTES"]
    results = pd.DataFrame(rows, columns=cols).rename(columns={"NOTES": "TESTS_PASSED"})
    for col in ("TEST_NAME", "UNIT", "OUTCOME"):
        results[col] = results[col].str.replace(" ", "_", regex=False)
    if out is not None:
        results.to_csv(out, sep="\t", index=False, float_format="%.4g", na_rep="NA")
        print(f"\nWrote {out}  ({len(results)} rows)")
    return results, plot_data


# ---------------------------------------------------------------------------
# Ambiguous injection
# ---------------------------------------------------------------------------

def parse_top_units(prevalence_dict, n=3):
    """Parse '{unit1:pct1,unit2:pct2,...}' → list of (unit, pct) sorted by pct desc."""
    if pd.isna(prevalence_dict) or str(prevalence_dict) in ("", "NA", "nan"):
        return []
    matches = re.findall(r'([^:{},\s][^:{},]*):(\d+\.?\d*)', str(prevalence_dict))
    pairs = [(u.strip(), float(p)) for u, p in matches if u.strip()]
    pairs.sort(key=lambda x: x[1], reverse=True)
    return pairs[:n]


def _unit_tag(unit):
    return unit.replace("/", "_").replace(" ", "_").replace("%", "pct")


def run_ambiguous(parquet, plot_name, details, dump_dir, plots_dir=None,
                  dip_threshold=0.05, min_target_n=30, test_mode=False,
                  split_threshold=0.15, out="ambiguous_results.tsv",
                  min_subjects=disclosure.MIN_SUBJECTS, plots=True, force=False):
    os.makedirs(dump_dir, exist_ok=True)
    plot_data = {}

    ambig = plot_name[plot_name["CATEGORY"] == "AMBIGUOUS"].sort_values("COUNT", ascending=False)
    if test_mode:
        ambig = ambig.head(10)
    n_names = len(ambig)
    print(f"\nAmbiguous TEST_NAMEs: {n_names}", flush=True)

    prev_dict = dict(zip(details["TEST_NAME"], details["PREVALENCE_DICT"]))

    rows = []

    cache = ResultCache(dump_dir, "ambiguous", force=force or plots)

    for i, name in enumerate(ambig["TEST_NAME"].tolist(), 1):
        cache.finish(rows, plot_data)
        tag = name.replace("/", "_").replace(" ", "_")

        print(f"\n[{i:>4}/{n_names}] {name}", flush=True)

        units_k = [(u, p) for u, p in parse_top_units(prev_dict.get(name, ""), n=3) if p > 1.0]
        files   = ([os.path.join(dump_dir, f"cand_{tag}.npz")]
                   + [os.path.join(dump_dir, f"targ_{tag}_{_unit_tag(u)}.npz") for u, _ in units_k])
        params  = {"units": units_k, "min_subjects": min_subjects, "min_target_n": min_target_n,
                   "dip_threshold": dip_threshold, "split_threshold": split_threshold}
        hit = cache.get(tag, files, params)
        if hit:
            cached_rows, entry = hit
            rows.extend(cached_rows)
            if entry is not None:
                plot_data[name] = entry
            print(f"  result=cache  ({len(cached_rows)} row(s))", flush=True)
            continue
        cache.begin(tag, name, files, params, len(rows))

        # Candidate values (reuse cache from unambiguous run if present)
        c_vals, c_pers, cached = _load_values(parquet, dump_dir, tag, name, "candidate")
        print(f"  cand={'cache' if cached else 'query'}  N={len(c_vals):,}  "
              f"people={disclosure.n_people(c_pers):,}", flush=True)

        if len(c_vals) < 2 or disclosure.n_people(c_pers) < min_subjects:
            print("  SKIP: too few candidate values or people", flush=True)
            continue

        # Filter to units with prevalence > 1% (top 3)
        top_units = [(u, p) for u, p in parse_top_units(prev_dict.get(name, ""), n=3) if p > 1.0]
        if not top_units:
            print("  SKIP: no units with prevalence > 1%", flush=True)
            continue

        # Load & cache target arrays once (min_target_n rows and min_subjects people)
        unit_data = {}   # unit -> t_vals
        unit_pers = {}   # unit -> person keys
        for unit, upct in top_units:
            t_vals_u, t_pers_u, _ = _load_values(parquet, dump_dir, f"{tag}_{_unit_tag(unit)}",
                                                 name, "target", unit=unit)
            if len(t_vals_u) < min_target_n:
                print(f"  SKIP unit {unit}({upct:.1f}%): N_TARGET={len(t_vals_u)}<{min_target_n}", flush=True)
            elif disclosure.n_people(t_pers_u) < min_subjects:
                print(f"  SKIP unit {unit}({upct:.1f}%): fewer than {min_subjects} people", flush=True)
            else:
                unit_data[unit] = t_vals_u
                unit_pers[unit] = t_pers_u

        if not unit_data:
            print("  SKIP: no units with sufficient reference data", flush=True)
            continue

        # Pre-check: run engine on full candidate vs each qualifying unit.
        # If any unit already passes, record those results and skip bimodality.
        print(f"  pre-check (full candidate)  units={list(unit_data)}", flush=True)
        precheck = []   # (unit, upct, t_vals, updates, ks_step, t_step, mad_step)
        for unit, upct in [(u, p) for u, p in top_units if u in unit_data]:
            t_vals_u = unit_data[unit]
            upd, ks, t, mad, pd_ = _run_engine(
                c_vals, c_pers, t_vals_u, unit_pers[unit],
                f"{name} [all] [{unit}]", dump_dir, plots_dir=plots_dir,
                prevalence=f"{upct:.1f}%",
                tag_override=f"{tag}_all_{_unit_tag(unit)}", min_subjects=min_subjects, plots=plots,
            )
            upd.update(N_CANDIDATE_SUBJECTS=disclosure.n_people(c_pers),
                       N_TARGET_SUBJECTS=disclosure.n_people(unit_pers[unit]))
            plot_data.setdefault(name, {}).setdefault("engine", {})[f"all_{unit}"] = pd_
            precheck.append((unit, upct, t_vals_u, upd, ks, t, mad))
            print(f"    [all][{unit}({upct:.1f}%)]"
                  f"  N={len(c_vals):,}/{len(t_vals_u):,}"
                  f"  KS={'P' if ks.passed else 'F'}(stat={ks.details['stat']:.3g})"
                  f"  T={'P' if t.passed else 'F'}"
                  f"  MAD={'P' if mad.passed else 'F'}"
                  f"  → {upd['OUTCOME']}", flush=True)

        any_precheck_pass = any(upd["OUTCOME"] == "PASS" for _, _, _, upd, *_ in precheck)

        # Bimodal check always runs — needed for split score even when pre-check passed
        bim = injection_engine.bimodal_check(c_vals, dip_threshold=dip_threshold)
        if plots:
            injection_engine.plot_bimodal_check(bim, name, plots_dir or dump_dir)
        plot_data.setdefault(name, {})["bimodal"] = injection_engine.compute_bimodal_release(
            bim, c_vals, c_pers, min_subjects=min_subjects)
        # the fitted components (and so the separator / overlap) are only published when each
        # component has enough people; a split additionally needs both sides to (checked below)
        fit_shown = (plot_data[name]["bimodal"] or {}).get("separator") is not None
        print(f"  bimodal={bim.status}  sep={bim.separator:.4g}"
              f"  BC={bim.bc:.3f}  dip_p={bim.dip_p:.3g}", flush=True)

        # Evaluate whether splitting improves the fit
        prefer_split = False
        si    = {}
        sep   = bim.separator
        c_low  = c_vals[c_vals <= sep] if not np.isnan(sep) else np.array([])
        c_high = c_vals[c_vals >  sep] if not np.isnan(sep) else np.array([])
        if len(c_low) >= 2 and len(c_high) >= 2:
            si = injection_engine.split_improvement(c_vals, c_low, c_high, unit_data)
            print(f"  split_improvement={si['improvement']:+.1%}"
                  f"  same_unit={si['same_best_unit']}"
                  f"  (global_KS={si['global_score']:.4f}"
                  f"  split_KS={si['split_score']:.4f})", flush=True)
            if si["improvement"] > split_threshold and not si["same_best_unit"]:
                prefer_split = True
                print(f"  → SPLIT preferred (threshold={split_threshold:.0%})", flush=True)

        # Decision-tree figure (private; its unit rankings are only used for the figure)
        if plots:
            unit_data_fig = {unit: (unit_data[unit], upct)
                             for unit, upct in top_units if unit in unit_data}
            global_ranks  = split_eval.rank_units(c_vals, unit_data_fig)
            low_ranks     = split_eval.rank_units(c_low,  unit_data_fig) if len(c_low)  >= 2 else []
            high_ranks    = split_eval.rank_units(c_high, unit_data_fig) if len(c_high) >= 2 else []
            split_eval.make_figure(
                name=name, c_vals=c_vals, unit_data=unit_data_fig,
                global_ranks=global_ranks, low_ranks=low_ranks, high_ranks=high_ranks,
                c_low=c_low, c_high=c_high, sep=sep, bim=bim,
                g_score=global_ranks[0][2] if global_ranks else np.nan,
                s_score=si.get("split_score", np.nan),
                improvement=si.get("improvement", 0.0),
                out_path=os.path.join(plots_dir or dump_dir, f"split_eval_{tag}.png"),
            )

        if any_precheck_pass and not prefer_split:
            print(f"  pre-check passed → global result kept", flush=True)
            for unit, upct, t_vals_u, upd, ks, t, mad in precheck:
                rows.append(dict(
                    TEST_NAME=name,
                    BIMODAL_STATUS="skipped",
                    BIMODAL_SEP=np.nan, BIMODAL_BC=np.nan, BIMODAL_DIP_P=np.nan, BIMODAL_OVERLAP=np.nan,
                    SCORE_GLOBAL=si.get("global_score", np.nan),
                    SCORE_SPLIT=si.get("split_score", np.nan),
                    SCORE_IMPROVEMENT=si.get("improvement", np.nan),
                    SUB_DIST="all",
                    UNIT=unit, UNIT_PREVALENCE=upct,
                    PREVALENCE_DICT=prev_dict.get(name, "NA"),
                    N_CANDIDATE=len(c_vals), N_TARGET=len(t_vals_u),
                    **upd,
                ))
            continue

        # Decide whether to split: bimodal detected, or split preferred by score
        # bimodal/bimodal_cautious takes priority over split_by_score when dip confirms bimodality
        do_split      = prefer_split or bim.status in ("bimodal", "bimodal_cautious")
        bimodal_label = (bim.status if bim.status in ("bimodal", "bimodal_cautious")
                         else ("split_by_score" if prefer_split else bim.status))

        # A split needs both sides from at least min_subjects people: a "mode" of a handful of
        # people is not one, and its rows could be worked out from the total minus the other side.
        if do_split and np.isnan(bim.separator):
            do_split = False                      # nothing to split at
        elif do_split:
            low_people  = disclosure.n_people(c_pers[c_vals <= bim.separator])
            high_people = disclosure.n_people(c_pers[c_vals > bim.separator])
            if min(low_people, high_people) < min_subjects:
                print(f"  split dropped: a side has fewer than {min_subjects} people", flush=True)
                do_split, bimodal_label = False, "split_too_small"

        if not do_split:
            # Unimodal, pre-check failed, split not preferred (or too small) — reuse pre-check results
            for unit, upct, t_vals_u, upd, ks, t, mad in precheck:
                rows.append(dict(
                    TEST_NAME=name,
                    BIMODAL_STATUS=bimodal_label,
                    BIMODAL_SEP=bim.separator if fit_shown else np.nan, BIMODAL_BC=bim.bc, BIMODAL_DIP_P=bim.dip_p,
                    BIMODAL_OVERLAP=bim.overlap_pct if fit_shown else np.nan,
                    SCORE_GLOBAL=si.get("global_score", np.nan),
                    SCORE_SPLIT=si.get("split_score", np.nan),
                    SCORE_IMPROVEMENT=si.get("improvement", np.nan),
                    SUB_DIST="all",
                    UNIT=unit, UNIT_PREVALENCE=upct,
                    PREVALENCE_DICT=prev_dict.get(name, "NA"),
                    N_CANDIDATE=len(c_vals), N_TARGET=len(t_vals_u),
                    **upd,
                ))
        else:
            sep  = bim.separator
            is_low = c_vals <= sep
            # both sides have at least min_subjects people (checked above), so >= 2 values each
            sub_dists = [("low", c_vals[is_low], c_pers[is_low]),
                         ("high", c_vals[~is_low], c_pers[~is_low])]

            print(f"  sub_dists={[s for s, *_ in sub_dists]}", flush=True)
            for sub_name, c_sub, c_sub_pers in sub_dists:
                for unit, upct in [(u, p) for u, p in top_units if u in unit_data]:
                    t_vals_u = unit_data[unit]
                    upd, ks, t, mad, pd_ = _run_engine(
                        c_sub, c_sub_pers, t_vals_u, unit_pers[unit],
                        f"{name} [{sub_name}] [{unit}]", dump_dir, plots_dir=plots_dir,
                        prevalence=f"{upct:.1f}%",
                        tag_override=f"{tag}_{sub_name}_{_unit_tag(unit)}", min_subjects=min_subjects,
                        plots=plots,
                    )
                    upd.update(N_CANDIDATE_SUBJECTS=disclosure.n_people(c_sub_pers),
                               N_TARGET_SUBJECTS=disclosure.n_people(unit_pers[unit]))
                    plot_data.setdefault(name, {}).setdefault("engine", {})[f"{sub_name}_{unit}"] = pd_
                    rows.append(dict(
                        TEST_NAME=name,
                        BIMODAL_STATUS=bimodal_label,
                        BIMODAL_SEP=bim.separator, BIMODAL_BC=bim.bc, BIMODAL_DIP_P=bim.dip_p, BIMODAL_OVERLAP=bim.overlap_pct,
                        SCORE_GLOBAL=si.get("global_score", np.nan),
                        SCORE_SPLIT=si.get("split_score", np.nan),
                        SCORE_IMPROVEMENT=si.get("improvement", np.nan),
                        SUB_DIST=sub_name,
                        UNIT=unit, UNIT_PREVALENCE=upct,
                        PREVALENCE_DICT=prev_dict.get(name, "NA"),
                        N_CANDIDATE=len(c_sub), N_TARGET=len(t_vals_u),
                        **upd,
                    ))
                    print(f"    [{sub_name}][{unit}({upct:.1f}%)]"
                          f"  N={len(c_sub):,}/{len(t_vals_u):,}"
                          f"  KS={'P' if ks.passed else 'F'}(stat={ks.details['stat']:.3g})"
                          f"  T={'P' if t.passed else 'F'}"
                          f"  MAD={'P' if mad.passed else 'F'}"
                          f"  → {upd['OUTCOME']}", flush=True)

    cache.finish(rows, plot_data)
    print(f"Result cache (ambiguous): {cache.hits} reused, {cache.misses} computed")
    if not rows:
        print("No ambiguous results to write.", flush=True)
        return pd.DataFrame(), {}

    results = pd.DataFrame(rows)

    # Best unit per (TEST_NAME, SUB_DIST):
    # Rank by deciding test quality (KS > T > MAD > FAIL > SKIP),
    # then by UNIT_PREVALENCE (prefer the dominant unit when several pass the same test:
    # a small target's KS stat is noisy), then by KS stat (lower = better fit).
    def _pass_rank(row):
        if row["KS_PASS"] == "PASS": return 0
        if row["T_PASS"]  == "PASS": return 1
        if row["OUTCOME"] == "PASS": return 2   # MAD decided
        if row["OUTCOME"] == "FAIL": return 3
        return 4

    results["_rank"] = results.apply(_pass_rank, axis=1)
    best = (results
            .sort_values(["_rank", "UNIT_PREVALENCE", "KS_STAT"],
                         ascending=[True, False, True])
            .groupby(["TEST_NAME", "SUB_DIST"])["UNIT"]
            .first()
            .reset_index())
    results = results.merge(best.rename(columns={"UNIT": "_best_unit"}),
                            on=["TEST_NAME", "SUB_DIST"], how="left")
    results = results[results["UNIT"] == results["_best_unit"]].drop(columns=["_rank", "_best_unit"])

    col_order = [
        "TEST_NAME", "BIMODAL_STATUS", "BIMODAL_SEP", "BIMODAL_BC", "BIMODAL_DIP_P", "BIMODAL_OVERLAP",
        "SCORE_GLOBAL", "SCORE_SPLIT", "SCORE_IMPROVEMENT",
        "SUB_DIST",
        "UNIT", "UNIT_PREVALENCE", "PREVALENCE_DICT",
        "N_CANDIDATE", "N_TARGET", "N_CANDIDATE_SUBJECTS", "N_TARGET_SUBJECTS",
        "CAND_DECILES", "TARG_DECILES",
        "KS_STAT", "KS_MLOGP", "KS_PASS",
        "T_STAT",  "T_MLOGP",  "T_PASS",
        "MAD_DIST", "MAD_THRESHOLD", "MAD_PASS",
        "OUTCOME", "TESTS_PASSED",
    ]
    results = results.rename(columns={"NOTES": "TESTS_PASSED"})[col_order]
    if out is not None:
        results.to_csv(out, sep="\t", index=False, float_format="%.4g", na_rep="NA")

    _print_ambig_summary(results, out)
    return results, plot_data


def _print_ambig_summary(results, out=None):
    n_tot  = results["TEST_NAME"].nunique()
    tested = results[results["OUTCOME"] != "SKIP"]

    best_rows = tested.copy()

    n_pass = (best_rows["OUTCOME"] == "PASS").sum()
    n_fail = (best_rows["OUTCOME"] == "FAIL").sum()
    n_skip = n_tot - best_rows["TEST_NAME"].nunique()

    print(f"\n{'=' * 72}")
    print("AMBIGUOUS INJECTION SUMMARY")
    print(f"{'=' * 72}")
    print(f"  TEST_NAMEs: {n_tot}  "
          f"PASS={n_pass}  FAIL={n_fail}  SKIP={n_skip}")
    print()

    w = 40
    print(f"  {'TEST_NAME':<{w}}  {'BIMODAL':<18}  {'SUB':<5}  {'UNIT':<14}  OUTCOME")
    print(f"  {'-'*w}  {'-'*18}  {'-'*5}  {'-'*14}  -------")
    for _, r in best_rows.iterrows():
        print(f"  {r['TEST_NAME']:<{w}}  {r['BIMODAL_STATUS']:<18}  {r['SUB_DIST']:<5}"
              f"  {r['UNIT']:<14}  {r['OUTCOME']}")

    if out is not None:
        print(f"\nWrote {out}  ({len(results)} rows, {n_tot} TEST_NAMEs)")


# ---------------------------------------------------------------------------
# Unified output
# ---------------------------------------------------------------------------

_UNIFIED_COLS = [
    "TEST_NAME",
    "TYPE",
    "SUB_DIST",
    "CUTOFF",
    "UNIT",
    "OUTCOME",
    "NOTES",
    "TESTS_PASSED",
    "UNIT_PREVALENCE", "PREVALENCE_DICT",
    "BIMODAL_STATUS", "BIMODAL_SEP", "BIMODAL_BC", "BIMODAL_DIP_P", "BIMODAL_OVERLAP",
    "SCORE_GLOBAL", "SCORE_SPLIT", "SCORE_IMPROVEMENT",
    "N_CANDIDATE", "N_TARGET", "N_CANDIDATE_SUBJECTS", "N_TARGET_SUBJECTS",
    "CAND_DECILES", "TARG_DECILES",
    "KS_STAT", "KS_MLOGP", "KS_PASS",
    "T_STAT",  "T_MLOGP",  "T_PASS",
    "MAD_DIST", "MAD_THRESHOLD", "MAD_PASS",
]


def _make_notes(row):
    """Generate a human-readable NOTES entry describing how the split decision was made."""
    if str(row.get("TYPE", "")) == "unambiguous":
        return ""
    status   = str(row.get("BIMODAL_STATUS", ""))
    impr     = row.get("SCORE_IMPROVEMENT")
    sep      = row.get("BIMODAL_SEP")
    dip      = row.get("BIMODAL_DIP_P")
    ovl      = row.get("BIMODAL_OVERLAP")
    sep_str  = f"{sep:.4g}"   if pd.notna(sep)  else "?"
    impr_str = f"{impr:+.1%}" if pd.notna(impr) else "?"

    if status == "split_by_score":
        core = f"split_by_score ({impr_str}) at {sep_str}"
    elif status == "bimodal":
        core = f"split_by_bimodal at {sep_str}"
    elif status == "bimodal_cautious":
        core = f"split_by_bimodal_cautious at {sep_str}"
    elif status == "skipped":
        return "NO_SPLIT"
    else:
        return "NO_SPLIT"

    metrics = []
    if pd.notna(dip):
        metrics.append(f"DIP:{dip:.3g}")
    if pd.notna(ovl):
        metrics.append(f"BL:{ovl:.1f}%")
    return core + (" | " + ",".join(metrics) if metrics else "")


def _publishable(combined, units_pub, min_subjects):
    """
    The released injection_results.tsv describes groups of rows, so it follows the same rules
    as the plot data:
      PREVALENCE_DICT / UNIT_PREVALENCE  shares among the units used by >= min_subjects people
                                         (published_units), not among all rows with a unit
      N_CANDIDATE / N_TARGET (+ _SUBJECTS) rows and people, BELOW under min_subjects people
    """
    df = combined.copy()
    df["PREVALENCE_DICT"] = [format_units(units_pub.get(n)) for n in df["TEST_NAME"]]

    def _share(name, unit):
        for u, p in units_pub.get(name, []):
            if unit in (u, u.replace(" ", "_")):
                return round(p, 2)
        return np.nan
    df["UNIT_PREVALENCE"] = [_share(n, u) for n, u in zip(df["TEST_NAME"], df["UNIT"])]

    for n_col, p_col in (("N_CANDIDATE", "N_CANDIDATE_SUBJECTS"), ("N_TARGET", "N_TARGET_SUBJECTS")):
        people = pd.to_numeric(df[p_col], errors="coerce").fillna(0)
        df[n_col] = [disclosure.count_or_below(n, p, min_subjects) if pd.notna(n) else np.nan
                     for n, p in zip(df[n_col], people)]
        df[p_col] = [disclosure.count_or_below(p, p, min_subjects) for p in people]
    return df


def _write_unified(udf, adf, out="injection_results.tsv", units_pub=None,
                   min_subjects=disclosure.MIN_SUBJECTS):
    """injection_results.tsv (released): both passes, in the publishable form (_publishable)
    when units_pub is given."""
    parts = []

    if udf is not None and len(udf):
        u = udf.copy()
        u["TYPE"]           = "unambiguous"
        u["SUB_DIST"]       = "all"
        # Derive UNIT_PREVALENCE from the dominant unit in PREVALENCE_DICT
        def _top_pct(row):
            pairs = parse_top_units(row["PREVALENCE_DICT"], n=10)
            for unit, pct in pairs:
                if unit == row["UNIT"]:
                    return pct
            return np.nan
        u["UNIT_PREVALENCE"] = u.apply(_top_pct, axis=1)
        parts.append(u)

    if adf is not None and len(adf):
        a = adf.copy()
        a["TYPE"] = "ambiguous"
        parts.append(a)

    if not parts:
        return None

    combined = pd.concat(parts, ignore_index=True)
    if "BIMODAL_SEP" not in combined.columns:
        combined["BIMODAL_SEP"] = np.nan
    combined["CUTOFF"] = np.where(
        combined["SUB_DIST"] == "low",
        combined["BIMODAL_SEP"],
        np.inf,
    )
    combined["NOTES"] = combined.apply(_make_notes, axis=1)
    combined = combined.reindex(columns=_UNIFIED_COLS)
    if units_pub is not None:
        combined = _publishable(combined, units_pub, min_subjects)
    if out is not None:
        combined.to_csv(out, sep="\t", index=False, float_format="%.4g", na_rep="NA")
        print(f"\nWrote {out}  ({len(combined)} rows)")
    return combined


def _print_result_rows(df):
    """Print injection_results rows vertically, one column per line."""
    if df is None or df.empty:
        return
    skip = {"PREVALENCE_DICT", "CAND_DECILES", "TARG_DECILES"}
    for i, (_, row) in enumerate(df.iterrows()):
        print(f"\n{'─'*60}")
        print(f"  injection_results row {i+1}/{len(df)}")
        print(f"{'─'*60}")
        for col in df.columns:
            if col in skip:
                continue
            val = row[col]
            if isinstance(val, float):
                s = f"{val:.4g}" if not pd.isna(val) else "NA"
            else:
                s = str(val) if not pd.isna(val) else "NA"
            print(f"  {col:<22} {s}")


# ---------------------------------------------------------------------------
# Coverage check
# ---------------------------------------------------------------------------

def _check_coverage(plot_name, udf, adf):
    expected = set(plot_name[plot_name["CATEGORY"].isin(["UNAMBIGUOUS", "AMBIGUOUS"])]["TEST_NAME"])

    covered = set()
    for label, df in [("unambiguous", udf), ("ambiguous", adf)]:
        if df is not None and len(df):
            names = set(df["TEST_NAME"])
            overlap = covered & names
            if overlap:
                print(f"  COVERAGE ERROR: {len(overlap)} TEST_NAME(s) appear in multiple files: "
                      f"{sorted(overlap)[:5]}{'...' if len(overlap) > 5 else ''}")
            covered |= names

    missing  = expected - covered
    extra    = covered - expected

    print(f"\nCoverage check: {len(expected)} expected  "
          f"{len(covered)} covered  "
          f"{len(missing)} missing  "
          f"{len(extra)} extra")

    if missing:
        print(f"  MISSING: {sorted(missing)[:10]}{'...' if len(missing) > 10 else ''}")
    if extra:
        print(f"  EXTRA:   {sorted(extra)[:10]}{'...' if len(extra) > 10 else ''}")
    if not missing and not extra:
        print("  OK: all TEST_NAMEs accounted for across 3 output files")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(
        description="Explore unit injection targets in Kanta lab data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("parquet", nargs="?", default=None,
                   help=f"Engine output parquet (default: the most recent run in {ENGINE_DIR})")
    p.add_argument("--min-count", type=int, default=1000, metavar="INT",
                   help="Minimum no-unit records per TEST_NAME to include")
    p.add_argument("--prevalence-threshold", type=float, default=98, metavar="FLOAT",
                   help="Min top-unit prevalence (%%) for a TEST_NAME to be unambiguous")
    p.add_argument("--dump-dir", default="/mnt/disks/data/kanta/inject/tmp/", metavar="PATH",
                   help="Private cache: per-test value arrays with person keys (*.npz), per-test "
                        "results (results/*.json) and, with --plots, the diagnostic PNGs")
    p.add_argument("--inject", action="store_true",
                   help="Run the injection engine on all tests (unambiguous + ambiguous)")
    p.add_argument("--dip-threshold", type=float, default=0.05, metavar="FLOAT",
                   help="Hartigan dip test p-value threshold for bimodality detection")
    p.add_argument("--split-threshold", type=float, default=0.15, metavar="FLOAT",
                   help="Minimum relative KS improvement to prefer a split over the global fit")
    p.add_argument("--min-target-n", type=int, default=30, metavar="INT",
                   help="Minimum reference records a unit must have to be tested (ambiguous pass)")
    p.add_argument("--force", action="store_true",
                   help="Recompute every test's result instead of reusing <dump-dir>/results/ "
                        "(results are reused only when inputs, arguments and code version match)")
    p.add_argument("--plots", action="store_true",
                   help="Also draw the private diagnostic PNGs into --dump-dir (per comparison, "
                        "bimodal and split figures; drawn from raw values, never released). Off by "
                        "default: they are slow and the website does not use them")
    p.add_argument("--min-subjects", type=int, default=disclosure.MIN_SUBJECTS, metavar="INT",
                   help="Minimum people behind every group that is analysed or published: a "
                        "TEST_NAME's no-unit rows, a reference unit, a split side, a listed unit")
    p.add_argument("--test", nargs="?", const=True, default=None, metavar="TEST_NAME",
                   help="Test mode. Bare --test: small sample (one per COUNT decile for "
                        "unambiguous, top 10 for ambiguous), writes to _test-suffixed files. "
                        "--test NAME: run only that TEST_NAME, print to screen, no file output.")
    p.add_argument("--out-dir", default=".", metavar="PATH",
                   help="Output directory for injection results/ (and test/results/ in --test mode). "
                        "Exploration outputs (counts, summary, scatter) go to --dump-dir.")
    p.add_argument("--release", action="store_true",
                   help="After a full --inject run, copy the release files (injection_results.tsv, "
                        "assignment_summary.md, summary_table.md, plot_data.json.gz, "
                        "test_names_exploration_scatter.png) into the repo's data/ "
                        "(not --out-dir, regardless of cwd), after release_check.py finds "
                        "nothing to fix (minimum-N rules). Local copy only — does not commit or push.")
    return p


_RELEASE_FILES = [
    "injection_results.tsv",
    "assignment_summary.md",
    "summary_table.md",
    "plot_data.json.gz",
    "test_names_exploration_scatter.png",
]


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.parquet is None:
        latest = latest_engine_output()
        if latest is None:
            parser.error(f"no engine output found in {ENGINE_DIR}: pass the parquet")
        args.parquet = str(latest)
    print(f"Input: {args.parquet}")
    if args.release and (not args.inject or args.test):
        parser.error("--release requires --inject without --test")

    out_dir  = Path(args.out_dir)
    # test (sample mode) → test/; named test (screen only) and normal → results/
    work_dir = out_dir / ("test" if args.test is True else "results")
    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    # Cached: only written on first run or when missing
    counts  = query_counts(args.parquet, out=str(out_dir / "test_name_counts.tsv"))
    effective_min = 0 if isinstance(args.test, str) else args.min_count
    counts  = counts[counts["COUNT"] > effective_min].reset_index(drop=True)
    details = query_details(args.parquet,
                            counts_file=str(out_dir / "test_name_counts.tsv"),
                            out=str(out_dir / "test_name_details.tsv"))
    # Rows and people per TEST_NAME x unit: every published group is checked against these
    summaries = query_summaries(args.parquet, out=str(out_dir / "test_name_unit_summaries.tsv"))
    units_pub = published_units(summaries, args.min_subjects)

    # Recomputed every run → work_dir
    plot_name = build_plot_table(counts, details)
    plot_name = add_people(plot_name, summaries, args.min_subjects)
    plot_name = _add_category(plot_name, args.prevalence_threshold, args.min_target_n, args.min_subjects)
    plot_name.to_csv(work_dir / "plot_name_level.tsv", sep="\t", index=False)
    print(f"Built plot_name_level.tsv  ({len(plot_name)} rows, "
          f"min_count={args.min_count}, threshold={args.prevalence_threshold}%)")

    make_scatter_plot(plot_name, output=str(work_dir / "test_names_exploration_scatter.png"))
    print_summary(plot_name, args.prevalence_threshold)
    dump_summary_md(plot_name, args.prevalence_threshold, args.min_count,
                    out=str(work_dir / "summary_table.md"),
                    out_tsv=str(work_dir / "summary_table.tsv"))

    if args.inject:
        if isinstance(args.test, str):
            plot_name = plot_name[plot_name["TEST_NAME"] == args.test].reset_index(drop=True)
            if plot_name.empty:
                print(f"'{args.test}' not found in any category — check name spelling")
                return

        def _out(name):
            return None if isinstance(args.test, str) else str(work_dir / name)

        plots_dir = None
        sample_mode = args.test is True

        udf, udf_plots = run_unambiguous(args.parquet, plot_name, details, args.dump_dir,
                                         plots_dir=plots_dir,
                                         test_mode=sample_mode,
                                         out=_out("unambiguous_results.tsv"),
                                         min_subjects=args.min_subjects, plots=args.plots,
                                         force=args.force)
        adf, adf_plots = run_ambiguous(args.parquet, plot_name, details, args.dump_dir,
                                       plots_dir=plots_dir,
                                       dip_threshold=args.dip_threshold,
                                       min_target_n=args.min_target_n, test_mode=sample_mode,
                                       split_threshold=args.split_threshold,
                                       out=_out("ambiguous_results.tsv"),
                                       min_subjects=args.min_subjects, plots=args.plots,
                                       force=args.force)
        combined = _write_unified(udf, adf, out=_out("injection_results.tsv"),
                                  units_pub=units_pub, min_subjects=args.min_subjects)
        if isinstance(args.test, str):
            _print_result_rows(combined)

        all_plot_data = {**udf_plots, **adf_plots}
        plot_data_out = _out("plot_data.json.gz")
        if plot_data_out is not None:
            def _round(obj, sig=5):
                if isinstance(obj, float):
                    return float(f"{obj:.{sig}g}")
                if isinstance(obj, dict):
                    return {k: _round(v, sig) for k, v in obj.items()}
                if isinstance(obj, list):
                    return [_round(v, sig) for v in obj]
                if isinstance(obj, (np.bool_,)):   return bool(obj)
                if isinstance(obj, np.integer):    return int(obj)
                if isinstance(obj, np.floating):   return _round(float(obj), sig)
                return obj
            payload = json.dumps(_round(all_plot_data)).encode("utf-8")
            with gzip.open(plot_data_out, "wb") as fh:
                fh.write(payload)
            print(f"Wrote {plot_data_out}  ({len(all_plot_data)} TEST_NAMEs, "
                  f"{len(payload)/1024:.0f} KB → {Path(plot_data_out).stat().st_size/1024:.0f} KB gz)")

        if not args.test:
            _check_coverage(plot_name, udf, adf)

        print_assignment_summary(udf, adf, plot_name,
                                 out_doc=None if args.test else str(work_dir / "assignment_summary.md"),
                                 out_tsv=None if args.test else str(work_dir / "assignment_summary.tsv"))

        # Copy injection_results.tsv to OUT/ for quick access after a full run
        if not args.test:
            import shutil
            shutil.copy2(work_dir / "injection_results.tsv", out_dir / "injection_results.tsv")
            print(f"Copied injection_results.tsv → {out_dir}/")

            problems = check_release(work_dir)
            for msg in problems[:20]:
                print(f"  RELEASE CHECK: {msg}")
            print(f"Release check: {len(problems)} problem(s) in {work_dir}")

            if args.release:
                if problems:
                    raise RuntimeError("release check failed: nothing copied to data/")
                repo_dir = Path(__file__).resolve().parents[2]
                release_dir = repo_dir / "data"
                release_dir.mkdir(parents=True, exist_ok=True)
                for fname in _RELEASE_FILES:
                    shutil.copy2(work_dir / fname, release_dir / fname)
                print(f"Released {len(_RELEASE_FILES)} files → {release_dir}/")

                site_dir = repo_dir / "html"
                proc = subprocess.run(
                    [sys.executable, str(repo_dir / "scripts" / "site" / "build.py"),
                     "--data-dir", str(release_dir), "--out-dir", str(site_dir)],
                    capture_output=True, text=True,
                )
                if proc.returncode != 0:
                    print(proc.stdout)
                    print(proc.stderr, file=sys.stderr)
                    raise RuntimeError("scripts/site/build.py failed")
                print(proc.stdout)
                print(f"Built local site preview → {site_dir}/index.html")


if __name__ == "__main__":
    main()
