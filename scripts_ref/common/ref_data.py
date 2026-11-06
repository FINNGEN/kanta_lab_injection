"""
ref_data.py

Data access and shared plumbing for the range-based injection scripts.

Backends
--------
  clickhouse  (default) the same access as scripts/primary/test_name_injection.py: `clickhouse -q`
              on the INJECT parquet, with per-test results cached in --dump-dir. Use it on
              the data VM.
  local       pandas/pyarrow reading the whole parquet into memory. Only for small
              (synthetic) parquets: it lets everything, including the value engine of
              scripts/primary/test_name_injection.py, run without ClickHouse.

Columns read (engine names)
---------------------------
  TEST_NAME_ABBREVIATION, MEASUREMENT_VALUE (candidate value), `source::MEASUREMENT_VALUE`
  (target value), `cleaned-pre-inj::MEASUREMENT_UNIT` ('NA' = no unit),
  `harmonization_omop::OMOP_ID`, and the reference limits REFERENCE_RANGE_LOWER_VALUE /
  REFERENCE_RANGE_UPPER_VALUE (--lower-col / --upper-col). Either limit may be missing.

The value engine (scripts/primary/test_name_injection.py) is imported, not copied, so the baseline
used for comparison is exactly the current one.
"""

import contextlib
import os
import sys
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd

_HERE    = Path(__file__).resolve().parent
_SCRIPTS = _HERE.parent.parent / "scripts"
for _p in (_SCRIPTS, _SCRIPTS / "primary"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import test_name_injection as tni                                  # noqa: E402  (value engine, categories, writers)
from common.clickhouse import ENGINE_DIR, latest_engine_output     # noqa: E402,F401
import ref_ranges as rr             # noqa: E402

COL_TEST   = "TEST_NAME_ABBREVIATION"
COL_CVAL   = "MEASUREMENT_VALUE"
COL_TVAL   = "source::MEASUREMENT_VALUE"
COL_UNIT   = "cleaned-pre-inj::MEASUREMENT_UNIT"
COL_OMOP   = "harmonization_omop::OMOP_ID"
DEF_LOWER  = "REFERENCE_RANGE_LOWER_VALUE"
DEF_UPPER  = "REFERENCE_RANGE_UPPER_VALUE"
TRUTH_COLS = ("TRUE_UNIT", "TRUE_FACTOR")   # synthetic data only; ignored by every method

_ORIG_QUERY_TEST_VALUES = tni._query_test_values


def tag_of(name):
    return name.replace("/", "_").replace(" ", "_")


def _to_float(s):
    return pd.to_numeric(pd.Series(s, dtype="object"), errors="coerce").to_numpy(dtype=float)


# ---------------------------------------------------------------------------
# ClickHouse backend
# ---------------------------------------------------------------------------

class ClickHouseBackend:
    name = "clickhouse"

    def __init__(self, parquet, dump_dir, lower_col=DEF_LOWER, upper_col=DEF_UPPER):
        self.parquet   = parquet
        self.dump_dir  = Path(dump_dir)
        self.lower_col = lower_col
        self.upper_col = upper_col
        self.dump_dir.mkdir(parents=True, exist_ok=True)

    def _lim(self, col):
        return f"toFloat64OrNull(toString(`{col}`))"

    # categories: exactly the cached queries of test_name_injection.py
    def counts_details(self, out_dir):
        counts_file = str(Path(out_dir) / "test_name_counts.tsv")
        counts  = tni.query_counts(self.parquet, out=counts_file)
        details = tni.query_details(self.parquet, counts_file=counts_file,
                                    out=str(Path(out_dir) / "test_name_details.tsv"))
        return counts, details

    def test_values(self, name, mode, unit=None):
        return _ORIG_QUERY_TEST_VALUES(self.parquet, name, mode, unit)[0]   # values (without person keys)

    def candidate_rows(self, name):
        """Unit-less rows of the test with a numeric value: VALUE, LOWER, UPPER."""
        path = self.dump_dir / f"refcand_{tag_of(name)}.npy"
        if path.exists():
            a = np.load(path)
            return pd.DataFrame({"VALUE": a[:, 0], "LOWER": a[:, 1], "UPPER": a[:, 2]})
        out = tni.clickhouse(f"""
            SELECT MEASUREMENT_VALUE, {self._lim(self.lower_col)}, {self._lim(self.upper_col)}
            FROM file('{self.parquet}')
            WHERE {COL_TEST} = {{name:String}}
              AND `{COL_UNIT}` = 'NA'
              AND MEASUREMENT_VALUE != 'NA'
            FORMAT TSV
        """, name=name)
        df = _read_tsv(out, ["VALUE", "LOWER", "UPPER"])
        df = df[np.isfinite(df["VALUE"])].sort_values("VALUE", kind="stable").reset_index(drop=True)
        np.save(path, df[["VALUE", "LOWER", "UPPER"]].to_numpy(dtype=float))
        return df

    def target_pairs(self, name):
        """Target rows (with a unit and a value) aggregated by unit x lower x upper."""
        path = self.dump_dir / f"refpairs_{tag_of(name)}.tsv"
        if path.exists():
            return pd.read_csv(path, sep="\t", keep_default_na=False,
                               na_values={"LOWER": ["", "NA", "nan"], "UPPER": ["", "NA", "nan"]})
        out = tni.clickhouse(f"""
            SELECT unit, lower, upper, count() AS n
            FROM (
                SELECT `{COL_UNIT}` AS unit,
                       {self._lim(self.lower_col)} AS lower,
                       {self._lim(self.upper_col)} AS upper
                FROM file('{self.parquet}')
                WHERE {COL_TEST} = {{name:String}}
                  AND `{COL_UNIT}` != 'NA'
                  AND `{COL_TVAL}` != 'NA'
            )
            GROUP BY unit, lower, upper
            FORMAT TSV
        """, name=name)
        df = _read_tsv(out, ["UNIT", "LOWER", "UPPER", "N"], str_cols=("UNIT",))
        df["N"] = df["N"].astype(int)
        df.to_csv(path, sep="\t", index=False, na_rep="NA")
        return df

    def target_value_stats(self, name):
        """Per unit: N, and quartiles of log10 of the positive target values."""
        path = self.dump_dir / f"refvstats_{tag_of(name)}.tsv"
        if path.exists():
            return pd.read_csv(path, sep="\t", keep_default_na=False,
                               na_values={c: ["", "NA", "nan"] for c in ("Q25", "Q50", "Q75")})
        out = tni.clickhouse(f"""
            SELECT unit, count() AS n, countIf(v > 0) AS n_pos,
                   quantilesExactIf(0.25, 0.5, 0.75)(log10(v), v > 0) AS q
            FROM (
                SELECT `{COL_UNIT}` AS unit, toFloat64OrNull(`{COL_TVAL}`) AS v
                FROM file('{self.parquet}')
                WHERE {COL_TEST} = {{name:String}}
                  AND `{COL_UNIT}` != 'NA'
                  AND `{COL_TVAL}` != 'NA'
            )
            WHERE v IS NOT NULL
            GROUP BY unit
            FORMAT TSV
        """, name=name)
        rows = []
        for line in out.strip().split("\n"):
            if not line.strip():
                continue
            unit, n, n_pos, q = line.split("\t")
            qs = [float(x) for x in q.strip("[]").split(",") if x] if q.strip("[]") else []
            qs = (qs + [np.nan] * 3)[:3]
            rows.append(dict(UNIT=unit, N=int(n), N_POS=int(n_pos), Q25=qs[0], Q50=qs[1], Q75=qs[2]))
        df = pd.DataFrame(rows, columns=["UNIT", "N", "N_POS", "Q25", "Q50", "Q75"])
        df.to_csv(path, sep="\t", index=False, na_rep="NA")
        return df


def _read_tsv(out, names, str_cols=()):
    if not out.strip():
        return pd.DataFrame({c: pd.Series(dtype="object" if c in str_cols else float) for c in names})
    df = pd.read_csv(StringIO(out), sep="\t", header=None, names=names, dtype=str,
                     keep_default_na=False, quoting=3)
    for c in names:
        if c not in str_cols:
            df[c] = pd.to_numeric(df[c].replace({"\\N": None, "NA": None, "": None}), errors="coerce")
    return df


# ---------------------------------------------------------------------------
# Local (in-memory) backend, for synthetic parquets
# ---------------------------------------------------------------------------

class LocalBackend:
    name = "local"

    def __init__(self, parquet, dump_dir=None, lower_col=DEF_LOWER, upper_col=DEF_UPPER):
        import pyarrow.parquet as pq
        self.parquet = parquet
        have = set(pq.read_schema(parquet).names)
        need = [COL_TEST, COL_CVAL, COL_TVAL, COL_UNIT, COL_OMOP, lower_col, upper_col]
        missing = [c for c in need if c not in have]
        if missing:
            raise SystemExit(f"{parquet}: missing columns {missing}")
        cols = need + [c for c in TRUTH_COLS if c in have]
        df = pd.read_parquet(parquet, columns=cols)
        self.truth_cols = [c for c in TRUTH_COLS if c in have]
        self.df = pd.DataFrame({
            "TEST":  df[COL_TEST].astype(str).to_numpy(),
            "UNIT":  df[COL_UNIT].astype(str).to_numpy(),
            "CRAW":  df[COL_CVAL].astype(str).to_numpy(),
            "TRAW":  df[COL_TVAL].astype(str).to_numpy(),
            "OMOP":  df[COL_OMOP].astype(str).to_numpy(),
            "CVAL":  _to_float(df[COL_CVAL]),
            "TVAL":  _to_float(df[COL_TVAL]),
            "LOWER": _to_float(df[lower_col]),
            "UPPER": _to_float(df[upper_col]),
        })
        for c in self.truth_cols:
            self.df[c] = df[c].to_numpy()

    def _cand(self, name):
        d = self.df
        return d[(d["TEST"] == name) & (d["UNIT"] == "NA") & (d["CRAW"] != "NA")]

    def _targ(self, name):
        d = self.df
        return d[(d["TEST"] == name) & (d["UNIT"] != "NA") & (d["TRAW"] != "NA")]

    def counts_details(self, out_dir=None):
        d = self.df
        c = d[(d["CRAW"] != "NA") & (d["UNIT"] == "NA") & (d["TEST"] != "NA")]
        counts = (c.groupby("TEST").agg(COUNT=("CRAW", "size"), OMOP_CONCEPT_ID=("OMOP", "first"))
                   .reset_index().rename(columns={"TEST": "TEST_NAME"}))
        counts = counts[counts["COUNT"] > tni._COUNTS_MIN].sort_values("COUNT", ascending=False)
        counts["OMOP_CONCEPT_ID"] = pd.to_numeric(counts["OMOP_CONCEPT_ID"], errors="coerce").astype("Int64")

        t = d[(d["TRAW"] != "NA") & (d["UNIT"] != "NA") & (d["TEST"] != "NA")]
        rows = []
        for name, g in t.groupby("TEST"):
            if name not in set(counts["TEST_NAME"]):
                continue
            vc = g["UNIT"].value_counts()
            tot = vc.sum()
            top = vc.head(3)
            pd_ = "{" + ",".join(f"{u}:{_round2(100.0 * n / tot)}" for u, n in top.items()) + "}"
            rows.append(dict(TEST_NAME=name, COUNT=int(tot), UNIT=top.index[0], PREVALENCE_DICT=pd_))
        details = pd.DataFrame(rows, columns=["TEST_NAME", "COUNT", "UNIT", "PREVALENCE_DICT"])
        return counts.reset_index(drop=True), details.sort_values("COUNT", ascending=False).reset_index(drop=True)

    def test_values(self, name, mode, unit=None):
        if mode == "candidate":
            v = self._cand(name)["CVAL"].to_numpy()
        else:
            t = self._targ(name)
            v = t[t["UNIT"] == unit]["TVAL"].to_numpy()
        v = v[np.isfinite(v)]
        return np.sort(v.astype(float))

    def candidate_rows(self, name):
        c = self._cand(name)
        c = c[np.isfinite(c["CVAL"])]
        out = pd.DataFrame({"VALUE": c["CVAL"].to_numpy(), "LOWER": c["LOWER"].to_numpy(),
                            "UPPER": c["UPPER"].to_numpy()})
        for col in self.truth_cols:
            out[col] = c[col].to_numpy()
        return out.sort_values("VALUE", kind="stable").reset_index(drop=True)

    def target_pairs(self, name):
        t = self._targ(name)
        g = (t.groupby(["UNIT", "LOWER", "UPPER"], dropna=False).size()
              .reset_index(name="N"))
        return g

    def target_value_stats(self, name):
        t = self._targ(name)
        rows = []
        for unit, g in t.groupby("UNIT"):
            v = g["TVAL"].to_numpy()
            v = v[np.isfinite(v)]
            pos = v[v > 0]
            q = np.percentile(np.log10(pos), [25, 50, 75]) if len(pos) else [np.nan] * 3
            rows.append(dict(UNIT=unit, N=len(v), N_POS=len(pos), Q25=q[0], Q50=q[1], Q75=q[2]))
        return pd.DataFrame(rows, columns=["UNIT", "N", "N_POS", "Q25", "Q50", "Q75"])


def _round2(x):
    r = round(x, 2)
    return str(int(r)) if r == int(r) else f"{r:g}"


def make_backend(args):
    if args.parquet is None:
        sys.exit(f"No engine output found in {ENGINE_DIR}: pass --input")
    if args.backend == "local":
        return LocalBackend(str(args.parquet), args.dump_dir, args.lower_col, args.upper_col)
    return ClickHouseBackend(str(args.parquet), args.dump_dir, args.lower_col, args.upper_col)


# ---------------------------------------------------------------------------
# Shared plumbing
# ---------------------------------------------------------------------------

def add_common_args(p):
    p.add_argument("--input", dest="parquet", type=Path, default=latest_engine_output(),
                   help="Engine output parquet (needs the reference limit columns) "
                        f"(default: the most recent run in {ENGINE_DIR}, now %(default)s)")
    p.add_argument("--backend", choices=["clickhouse", "local"], default="clickhouse",
                   help="clickhouse: as test_name_injection.py (data VM); local: in-memory, small/synthetic parquets only")
    p.add_argument("--out-dir", default=".", metavar="PATH", help="Output directory")
    p.add_argument("--dump-dir", default="/mnt/disks/data/kanta/inject/tmp/", metavar="PATH",
                   help="Cache directory (shared with test_name_injection.py: same cand_/targ_ arrays; "
                        "the range caches are refcand_/refpairs_/refvstats_)")
    p.add_argument("--test", default=None, metavar="TEST_NAME",
                   help="Run only this TEST_NAME and print the result (no files)")
    p.add_argument("--lower-col", default=DEF_LOWER, help="Lower reference limit column")
    p.add_argument("--upper-col", default=DEF_UPPER, help="Upper reference limit column")
    g = p.add_argument_group("categories (as test_name_injection.py)")
    g.add_argument("--min-count", type=int, default=1000, help="Minimum unit-less rows per TEST_NAME")
    g.add_argument("--prevalence-threshold", type=float, default=98,
                   help="Min top-unit prevalence (%%) for a TEST_NAME to be unambiguous")
    g.add_argument("--min-target-n", type=int, default=30, help="Minimum target rows (categories, value engine)")
    g.add_argument("--dip-threshold", type=float, default=0.05, help="Value engine: dip test p threshold")
    g.add_argument("--split-threshold", type=float, default=0.15, help="Value engine: split improvement threshold")
    r = p.add_argument_group("reference ranges")
    r.add_argument("--margin", type=float, default=1.5,
                   help="A row matches a unit when its limits are within this factor of the unit's median limits")
    r.add_argument("--combine", choices=["max", "mean"], default="max",
                   help="Combine the lower and upper distances by max (strict) or mean")
    r.add_argument("--far", type=float, default=3.0,
                   help="Value check: robust SDs (log10) beyond which a value contradicts its range's unit")
    r.add_argument("--min-profile-rows", type=int, default=20, help="Target rows with a range a unit profile needs")
    r.add_argument("--mixed-ratio", type=float, default=10.0,
                   help="A profile whose limit p95/p05 reaches this is mixed-scale (not used)")
    r.add_argument("--min-ranged", type=int, default=30,
                   help="Decided (range-labelled) candidate rows needed for the ranges to decide/validate a test")
    r.add_argument("--mix-share", type=float, default=0.05,
                   help="Share of decided rows a scale group needs to count as present")
    r.add_argument("--min-group-rows", type=int, default=20, help="Decided rows a scale group needs to count as present")
    r.add_argument("--min-cut-acc", type=float, default=0.9,
                   help="Supervised value cutoff counts as reliable from this accuracy on the range-labelled rows")
    p.add_argument("--plots", action="store_true", help="Per ambiguous test: values by range label + range clusters (PNG)")
    return p


def load_plot_name(backend, args, out_dir):
    """counts, details -> plot_name with CATEGORY, exactly as test_name_injection.main()."""
    counts, details = backend.counts_details(out_dir)
    effective_min = 0 if args.test else args.min_count
    counts = counts[counts["COUNT"] > effective_min].reset_index(drop=True)
    plot_name = tni.build_plot_table(counts, details)
    plot_name = tni._add_category(plot_name, args.prevalence_threshold, args.min_target_n)
    if args.test:
        plot_name = plot_name[plot_name["TEST_NAME"] == args.test].reset_index(drop=True)
        if plot_name.empty:
            raise SystemExit(f"'{args.test}' not found — check the name")
    return plot_name, details


def units_to_consider(prevalence_dict):
    """The units the value engine tests for an ambiguous test: top 3 with > 1 %."""
    return [(u, p) for u, p in tni.parse_top_units(prevalence_dict, n=3) if p > 1.0]


def vstats_dict(stats_df):
    out = {}
    for _, r in stats_df.iterrows():
        if np.isfinite(r["Q50"]):
            out[r["UNIT"]] = (float(r["Q50"]), max(float((r["Q75"] - r["Q25"]) / 1.349), rr._RSD_FLOOR))
    return out


def analyse(backend, name, units, args):
    """Load one test's data and run the range analysis. Returns (cand, result dict)."""
    cand  = backend.candidate_rows(name)
    pairs = backend.target_pairs(name)
    vst   = backend.target_value_stats(name)
    n_t   = dict(zip(vst["UNIT"], vst["N"]))
    res = rr.analyse_test(cand, pairs, vstats_dict(vst), units, n_target=n_t,
                          margin=args.margin, combine=args.combine, far=args.far,
                          min_profile_rows=args.min_profile_rows, mixed_ratio=args.mixed_ratio)
    res["margin"] = args.margin
    return cand, res


@contextlib.contextmanager
def _quiet(log_path):
    if log_path is None:
        yield
        return
    with open(log_path, "a") as fh, contextlib.redirect_stdout(fh):
        yield


def run_value_engine(backend, plot_name, details, dump_dir, args, log_path=None):
    """
    The current value engine (test_name_injection.run_unambiguous + run_ambiguous) on the
    given tests. Returns the unified table (injection_results.tsv format) or an empty frame.
    With the local backend the value queries are redirected to it.
    """
    if backend.name == "local":
        # the value engine wants a person key per value; the local backend has none, so every row
        # counts as its own person (experiments only: nothing from here is released)
        def _values(parquet, name, mode, unit=None):
            v = backend.test_values(name, mode, unit)
            return v, np.arange(len(v), dtype=np.uint64)
        tni._query_test_values = _values
    else:
        tni._query_test_values = _ORIG_QUERY_TEST_VALUES
    os.makedirs(dump_dir, exist_ok=True)
    with _quiet(log_path):
        udf, _ = tni.run_unambiguous(backend.parquet, plot_name, details, dump_dir, out=None)
        adf, _ = tni.run_ambiguous(backend.parquet, plot_name, details, dump_dir,
                                   dip_threshold=args.dip_threshold, min_target_n=args.min_target_n,
                                   split_threshold=args.split_threshold, out=None)
        combined = tni._write_unified(udf, adf, out=None)
    return combined if combined is not None else pd.DataFrame(columns=tni._UNIFIED_COLS)


def value_rule_units(values, rows):
    """
    Apply cutoff-format rows (SUB_DIST / CUTOFF / UNIT / OUTCOME of one test) to values.
    Rows are ordered by CUTOFF; a value goes to the first row whose CUTOFF >= value.
    Rows that are not PASS leave their values unassigned (''). Returns an object array.
    """
    v = np.asarray(values, dtype=float)
    out = np.full(len(v), "", dtype=object)
    if rows is None or len(rows) == 0:
        return out
    r = rows.copy()
    r["_c"] = pd.to_numeric(r["CUTOFF"], errors="coerce").fillna(np.inf)
    r = r.sort_values("_c", kind="stable")
    lower = -np.inf
    for _, row in r.iterrows():
        m = (v > lower) & (v <= row["_c"])
        if row["OUTCOME"] == "PASS":
            out[m] = row["UNIT"]
        lower = row["_c"]
    return out
