#!/usr/bin/env python3
"""
suggest.py — automatic fix suggestions for lab rows that failed unit harmonization.

A group is TEST_NAME_ABBREVIATION × MEASUREMENT_UNIT over the rows that have a numeric
value but no harmonized value (`harmonization_omop::MEASUREMENT_VALUE` == 'NA'). For every
group the script works out which OMOP concept the rows belong to, which units could
convert them into the concept's unit, and then weighs two independent kinds of evidence:

  values   the engine's KS / T / MAD tests: the group's values against the concept's
           harmonized values expressed in the candidate unit (harmonized ÷ factor);
  ranges   the group's reference limits × factor against the concept's typical ranges
           (learned from the rows already harmonized, each converted by its own factor).

The result is a suggestion for omop_injection.tsv (source unit → fixed unit), a case
that needs review, or the reason why nothing can be suggested. The output is a proposal
for review; the injection table itself is maintained in kanta_lab_preprocessing.

Concept (CONCEPT_CASE)
  Mapped in Usagi, conversion failed     the row already has an OMOP concept
  Pair waiting for Usagi review          (test, unit) is in Usagi but not approved
  Concept taken from the test's other units   the pair is missing; the test's approved
                                         units agree on one concept
  Test maps to several concepts          the test's approved units point to 2+ concepts;
                                         every concept's candidates are tried
  No concept in Usagi                    nothing to convert to

Unit (UNIT_CASE)
  Unit missing / Unit is a synonym or incomplete / Unit does not match the values /
  Conversion missing from the table / Only a formula conversion exists /
  Too few harmonized rows to compare / Unit not convertible

Result (RESULT)
  Suggest fix                            CONFIDENCE High: values (KS or T) and ranges agree;
                                         Medium: one kind of evidence only, or values pass
                                         on the loose MAD test alone
  Review: values and ranges disagree
  Review: needs a split                  the ranges point to two units; omop_injection.tsv
                                         holds one fix per (test, unit)
  No unit fits                           none of the concept's units (the conversion-table units into
                                         its unit) passes; the reason says how far off the ranges are
  Not decidable: test not in Usagi / test only unchecked in Usagi (no approved unit, so no
                 concept; the reason names a candidate concept when an unchecked row has one) /
                 too few harmonized rows / no conversion into the
                                         concept's unit — nothing to compare against; these
                                         form the second list (DECIDABLE = no)

Only rows with a numeric MEASUREMENT_VALUE are counted; rows without a value never enter.
Each group's list is decided first, from the cached data alone (no concept, too few harmonized
rows, no conversion → not decidable). Then each list has its own minimum size: --min-count (1000)
failed rows for decidable groups, --min-count-undecidable (1000) for not-decidable ones. 1000 is the
primary injection's threshold: smaller groups are left to it.

Privacy (FinnGen minimum N, scripts/common/disclosure.py): everything written for the portal is
aggregated, and every published group, count or share comes from at least --min-cell (5)
PEOPLE, counted from a person key (cityHash64 of FINNGENID) kept with every cached row:
  * groups whose failed rows come from fewer people are not analysed; a concept is compared
    against only with --min-target-n rows from --min-cell people;
  * value distributions (the group's, and both sides of each candidate comparison) are published
    as percentiles P5..P95 only, under disclosure.side_summary()'s rules (60 people, 100 rows,
    every 5% stretch of rows from 5 people): no curves, histograms, minimum or maximum;
  * reference-range pairs, typical-range clusters (lower bound: their largest pair) and
    unit-history paths need min_cell people each, and the paths left out together as well;
  * shares of rows (range coverage, range match) are published only when the part and the rest
    each come from min_cell people; the reasons then say "most" / "some" instead of a number.
Reference texts are never read, only the extracted lower / upper limits. release_check.py checks
the release files; --release copies nothing when it finds a problem.

Speed: the engine parquet is scanned once for every row that failed harmonization, and three
more times for the harmonized side of the concepts involved; both are cached under --dump-dir
(one folder per input file) and everything after that runs locally. The cache is rebuilt when
the input file changes, or with --force.

Usage (data VM; default input = latest run in /mnt/disks/data/kanta/engine)
  python3 scripts/unharmonized/suggest.py --out-dir OUT                # groups with ≥ --min-count rows
  python3 scripts/unharmonized/suggest.py --out-dir OUT --release      # also copy to data/unharmonized/
  python3 scripts/unharmonized/suggest.py --input X.parquet --test b-hkr --out-dir OUT
"""

import argparse
import gzip
import hashlib
import json
import re
import shutil
import sys
import time
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd

_SCRIPTS = Path(__file__).resolve().parents[1]
for _p in (_SCRIPTS, _SCRIPTS / "unharmonized"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from common import disclosure as dc                                        # noqa: E402
from common import injection_engine as ie                                   # noqa: E402
from common import ranges as rr                                            # noqa: E402
from common import reference_data as rd                                    # noqa: E402
from common.clickhouse import ENGINE_DIR, clickhouse, latest_engine_output  # noqa: E402
from release_check import check_release                                    # noqa: E402

REPO = Path(__file__).resolve().parents[2]
RELEASE_DIR = REPO / "data" / "unharmonized"
DEFAULT_INJECTION_TABLE = REPO.parent / "kanta_lab_preprocessing" / "src" / "kanta" / "engine" / "data" / "omop_injection.tsv"

DUMP_FORMAT = 3      # bump when the dump's columns change: old caches are rebuilt (3: PERSON)
FIGURE_FORMAT = 2    # bump when the issue figures change (2: percentiles only): figures of an older
                     # format, in --out-dir/figures or the release, are redrawn and replaced
CONCEPT_FORMAT = 2   # bump when the per-concept cache changes (2: people per unit / range pair,
                     # a person key per sampled value): the concept cache is rebuilt

COL = dict(
    test="TEST_NAME_ABBREVIATION", unit="MEASUREMENT_UNIT", src_unit="source::MEASUREMENT_UNIT",
    pre_unit="cleaned-pre-inj::MEASUREMENT_UNIT", post_unit="cleaned-post-inj::MEASUREMENT_UNIT",
    value="MEASUREMENT_VALUE", h_value="harmonization_omop::MEASUREMENT_VALUE",
    h_unit="harmonization_omop::MEASUREMENT_UNIT", h_omop="harmonization_omop::OMOP_ID",
    h_cf="harmonization_omop::CONVERSION_FACTOR",
    lower="REFERENCE_RANGE_LOWER_VALUE", upper="REFERENCE_RANGE_UPPER_VALUE",
    lab="SERVICE_PROVIDER", date="APPROX_EVENT_DATETIME", person="FINNGENID",
)

# person key: people can be counted in any subset of rows without keeping the IDs
PERSON = f"cityHash64(toString(`{COL['person']}`))"

CONCEPT_CASES = {
    "mapped":    "Mapped in Usagi, conversion failed",
    "unchecked": "Pair waiting for Usagi review",
    "inferred":  "Concept taken from the test's other units",
    "several":   "Test maps to several concepts",
    "noconcept": "No concept in Usagi",
}
UNIT_CASES = {
    "missing":     "Unit missing",
    "synonym":     "Unit is a synonym or incomplete",
    "wrong":       "Unit does not match the values",
    "noconv":      "Conversion missing from the table",
    "nomatch":     "Values match none of the concept's units",
    "formula":     "Only a formula conversion exists",
    "lowtarget":   "Too few harmonized rows to compare",
    "unconverted": "Unit not convertible",
    "notassessed": "Not assessed (no concept)",
}
RESULTS = {
    "suggest":      "Suggest fix",
    "conflict":     "Review: values and ranges disagree",
    "split":        "Review: needs a split",
    "nofit":        "No unit fits",
    "nd_missing":   "Not decidable: test not in Usagi",
    "nd_unchecked": "Not decidable: test only unchecked in Usagi",
    "nd_lowtarget": "Not decidable: too few harmonized rows",
    "nd_noconv":    "Not decidable: no conversion into the concept's unit",
}
NOT_DECIDABLE = ("nd_missing", "nd_unchecked", "nd_lowtarget", "nd_noconv")

GROUP_COLS = ["GROUP_ID", "TEST_NAME", "SOURCE_UNIT", "N_ROWS", "N_SUBJECTS", "CONCEPT_CASE", "USAGI_PAIR_STATUS",
              "OMOP_ID", "OMOP_NAME", "OMOP_QUANTITY", "TARGET_UNIT", "N_TARGET", "N_TARGET_SUBJECTS", "UNIT_CASE",
              "RESULT", "DECIDABLE", "CONFIDENCE", "FIX_UNIT", "CONVERSION_FACTOR", "VALUES", "VALUES_DECIDED_BY",
              "KS_STAT", "RANGE_COVERAGE", "N_RANGED", "RANGE_MATCH", "NEEDS", "CURRENT_RULE",
              "VS_CURRENT_RULE", "REASON", "SUGGESTION_ID", "ISSUE_TITLE"]
CAND_COLS = ["GROUP_ID", "CAND_ID", "OMOP_ID", "TARGET_UNIT", "UNIT", "SAME_SCALE_UNITS", "CONVERSION_FACTOR",
             "KS_STAT", "KS_PASS", "T_PASS", "MAD_DIST", "MAD_THRESHOLD", "MAD_PASS", "VALUES",
             "VALUES_DECIDED_BY", "RANGE_MATCH", "RANGE_ASSIGNED", "PICKED"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def slug(s):
    return re.sub(r"[^a-zA-Z0-9_-]", "_", str(s))


def unit_label(u):
    return "(no unit)" if u in ("NA", "", None) else u


def num(col):
    return f"toFloat64OrNull(toString(`{col}`))"


def fmt_cf(cf):
    return f"{cf:.6g}"


def _read(out, names):
    if not out.strip():
        return pd.DataFrame(columns=names)
    return pd.read_csv(StringIO(out), sep="\t", header=None, names=names, keep_default_na=False,
                       na_values=["\\N", "nan", "NaN", "inf"], dtype=str)


def _f(df, cols):
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


# ---------------------------------------------------------------------------
# Progress output
# ---------------------------------------------------------------------------

_TTY = sys.stdout.isatty()


def _fmt_t(sec):
    sec = int(sec)
    return f"{sec // 3600}h{sec % 3600 // 60:02d}m" if sec >= 3600 else (f"{sec // 60}m{sec % 60:02d}s" if sec >= 60 else f"{sec}s")


class ticking:
    """Elapsed-time counter for a step whose progress cannot be measured (a ClickHouse scan)."""

    def __init__(self, label):
        self.label = label

    def __enter__(self):
        import threading
        self.t0 = time.time()
        self.stop = threading.Event()
        print(f"{self.label} ...", end="" if _TTY else "\n", flush=True)
        if _TTY:
            def run():
                while not self.stop.wait(1):
                    print(f"\r{self.label} ... {_fmt_t(time.time() - self.t0)}", end="", flush=True)
            self.th = threading.Thread(target=run, daemon=True)
            self.th.start()
        return self

    def __exit__(self, *exc):
        self.stop.set()
        if _TTY:
            self.th.join()
        print(f"\r{self.label} ... done in {_fmt_t(time.time() - self.t0)}", flush=True)
        return False


class GroupStatus:
    """What the group being evaluated is doing: one line that overwrites itself on a terminal (the
    step and the time spent on the group, ticking every second), replaced by the group's result
    line when it is done. Without a terminal (a log file) only the steps of large groups are
    printed, as plain lines."""

    def __init__(self, prefix, n_rows, big=100_000):
        import threading
        self.prefix, self.t0, self.width, self.step = prefix, time.time(), 0, "starting"
        self.plain = not _TTY and n_rows >= big
        self.lock, self.stop = threading.Lock(), threading.Event()
        if _TTY:
            def tick():
                while not self.stop.wait(1):
                    self._draw()
            threading.Thread(target=tick, daemon=True).start()

    def _draw(self):
        with self.lock:
            if self.stop.is_set():
                return
            line = f"{self.prefix}  … {self.step} ({_fmt_t(time.time() - self.t0)})"
            print("\r" + line.ljust(self.width), end="", flush=True)
            self.width = max(self.width, len(line))

    def __call__(self, step):
        self.step = step
        if _TTY:
            self._draw()
        elif self.plain:
            print(f"{self.prefix}  … {step} ({_fmt_t(time.time() - self.t0)})", flush=True)

    def clear(self):
        with self.lock:
            self.stop.set()
            if _TTY and self.width:
                print("\r" + " " * self.width + "\r", end="", flush=True)


# ---------------------------------------------------------------------------
# Data access: a few full scans of the engine parquet, everything else from a local cache
# ---------------------------------------------------------------------------

def sql_str(v):
    return "'" + str(v).replace("\\", "\\\\").replace("'", "\\'") + "'"


class Data:
    """
    Cache folder per input file (<dump-dir>/<input file stem>/):

      failed.parquet        every row with a numeric value and no harmonized value: test, unit,
                            raw OMOP id, raw unit spelling, value, limits, lab, year (1 full scan)
      concepts/<id>.units.tsv / .ranges.tsv / .values.npy
                            per concept: harmonized rows per unit, distinct (lower, upper, factor)
                            pairs, and a value sample in the main unit. Fetched for all concepts
                            not cached yet in one batch (3 full scans), so a new --min-count or new
                            thresholds only scan for concepts never seen before.
      manifest.json         the input's path, size and modification time

    The cache is rebuilt when the input file changes (another name gives another folder; the
    same name with a different size or time is detected through the manifest) or with --force.
    """

    def __init__(self, parquet, cache_dir, args):
        self.p = str(Path(parquet).resolve())
        self.cache = Path(cache_dir)
        self.args = args
        st = Path(parquet).stat()
        manifest = {"input": self.p, "size": st.st_size, "mtime": int(st.st_mtime), "dump_format": DUMP_FORMAT,
                    "concept_format": CONCEPT_FORMAT}
        mpath = self.cache / "manifest.json"
        old = json.loads(mpath.read_text()) if mpath.exists() else None
        same_input = old is not None and all(old.get(k) == manifest[k] for k in ("input", "size", "mtime"))
        if self.cache.exists() and (args.force or (old is not None and not same_input)):
            print(f"Rebuilding the cache {self.cache} ({'--force' if args.force else 'input file changed'})")
            shutil.rmtree(self.cache)
        elif same_input and old.get("dump_format") != DUMP_FORMAT:
            # only the dump of failed rows changed shape; the per-concept cache is still valid
            print("The dump of failed rows has a new format: re-dumping it (concept cache kept)")
            (self.cache / "failed.parquet").unlink(missing_ok=True)
        if same_input and old.get("concept_format") != CONCEPT_FORMAT and (self.cache / "concepts").exists():
            print("The concept cache has a new format: rebuilding it (dump of failed rows kept)")
            shutil.rmtree(self.cache / "concepts")
        (self.cache / "concepts").mkdir(parents=True, exist_ok=True)
        mpath.write_text(json.dumps(manifest))

        cols = [l.split("\t")[0] for l in clickhouse(f"DESCRIBE TABLE file('{self.p}') FORMAT TSV").splitlines() if l]
        self.has = set(cols)
        missing = [COL[k] for k in ("test", "unit", "value", "h_value", "h_unit", "h_omop", "person")
                   if COL[k] not in self.has]
        if missing:
            sys.exit(f"Input lacks required columns: {missing}")
        for k in ("src_unit", "pre_unit", "post_unit", "h_cf", "lower", "upper", "lab", "date"):
            if COL[k] not in self.has:
                print(f"  (no column {COL[k]!r}: {k} left out)")
        self.failed = self._load_failed()
        self._idx = self.failed.groupby(["TEST", "UNIT"], observed=True, sort=False).indices
        self._rng = np.random.default_rng(0)

    # -- helpers --
    def _col(self, key, expr=None, default="NULL"):
        return (expr or f"`{COL[key]}`") if COL[key] in self.has else default

    def _outfile(self, select, path):
        tmp = path.with_name(path.name + ".tmp")
        tmp.unlink(missing_ok=True)
        clickhouse(f"{select} INTO OUTFILE '{tmp}' FORMAT Parquet")
        tmp.replace(path)

    # -- failed rows --
    def _load_failed(self):
        path = self.cache / "failed.parquet"
        if not path.exists():
            year = self._col("date", f"toString(toYear(parseDateTimeBestEffortOrNull(toString(`{COL['date']}`))))", "'-'")
            with ticking("Dumping the rows that failed harmonization (full scan 1)"):
                self._outfile(f"""
                SELECT toString(`{COL['test']}`) AS TEST, toString(`{COL['unit']}`) AS UNIT,
                       toString(`{COL['h_omop']}`) AS OMOP_RAW,
                       {self._col('src_unit', f"toString(`{COL['src_unit']}`)", "'-'")} AS SRC_UNIT,
                       {self._col('pre_unit', f"toString(`{COL['pre_unit']}`)", "'-'")} AS PRE_UNIT,
                       {self._col('post_unit', f"toString(`{COL['post_unit']}`)", "'-'")} AS POST_UNIT,
                       {num(COL['value'])} AS VALUE,
                       {self._col('lower', num(COL['lower']))} AS LOWER,
                       {self._col('upper', num(COL['upper']))} AS UPPER,
                       {self._col('lab', f"toString(`{COL['lab']}`)", "'-'")} AS LAB,
                       ifNull({year}, 'NA') AS YEAR,
                       {PERSON} AS PERSON
                FROM file('{self.p}')
                WHERE VALUE IS NOT NULL AND `{COL['h_value']}` = 'NA'""", path)
        import pyarrow.parquet as pq
        df = pq.read_table(path).to_pandas(strings_to_categorical=True)
        for c in ("TEST", "UNIT", "OMOP_RAW", "SRC_UNIT", "PRE_UNIT", "POST_UNIT", "LAB", "YEAR"):
            df[c] = df[c].cat.add_categories([x for x in ("NA",) if x not in df[c].cat.categories]).fillna("NA")
        print(f"Rows that failed harmonization: {len(df):,}")
        return df

    def groups(self):
        g = self.failed.groupby(["TEST", "UNIT", "OMOP_RAW"], observed=True).size().reset_index(name="N")
        for c in ("TEST", "UNIT", "OMOP_RAW"):
            g[c] = g[c].astype(str)
        g["UNIT"] = g["UNIT"].replace("", "NA")
        return g

    def group_people(self):
        """(test, unit) -> distinct people among the group's failed rows."""
        p = self.failed.groupby(["TEST", "UNIT"], observed=True)["PERSON"].nunique()
        return {(str(t), "NA" if str(u) == "" else str(u)): int(n) for (t, u), n in p.items()}

    def _rows(self, test, unit):
        idx = self._idx.get((test, unit))
        if idx is None and unit == "NA":
            idx = self._idx.get((test, ""))
        return self.failed.iloc[idx] if idx is not None else self.failed.iloc[[]]

    def group_rows(self, test, unit):
        """VALUE, LOWER, UPPER (reported scale) and PERSON of one group's rows (sampled)."""
        df = self._rows(test, unit)
        if len(df) > self.args.max_group_rows:
            df = df.sample(self.args.max_group_rows, random_state=0)
        out = df[["VALUE", "LOWER", "UPPER"]].reset_index(drop=True).astype(float)
        out["PERSON"] = df["PERSON"].to_numpy(dtype=np.uint64)
        return out[np.isfinite(out["VALUE"])].reset_index(drop=True)

    def group_context(self, test, unit):
        """The unit path of every row of the group (sent → after cleaning → after primary
        injection), with its person key."""
        df = self._rows(test, unit)[["SRC_UNIT", "PRE_UNIT", "POST_UNIT", "PERSON"]]
        return df.astype({"SRC_UNIT": str, "PRE_UNIT": str, "POST_UNIT": str}).reset_index(drop=True)

    # -- harmonized side, per concept --
    def _cpath(self, omop, kind):
        return self.cache / "concepts" / f"{slug(omop)}.{kind}"

    def prefetch_counts(self, concepts):
        """Harmonized rows per unit for every concept not counted yet (one scan for all of them)."""
        todo = sorted({str(o) for o in concepts if re.fullmatch(r"\d+", str(o))
                       and not self._cpath(o, "units.tsv").exists()})
        if not todo:
            return
        ids = ",".join(sql_str(o) for o in todo)
        with ticking(f"Counting harmonized rows of {len(todo)} concept(s) (full scan)"):
            units = _read(clickhouse(f"""
            SELECT toString(`{COL['h_omop']}`), toString(`{COL['h_unit']}`), count(), uniqExact({PERSON})
            FROM file('{self.p}') WHERE toString(`{COL['h_omop']}`) IN ({ids}) AND {num(COL['h_value'])} IS NOT NULL
            GROUP BY 1, 2 FORMAT TSV"""), ["OMOP", "UNIT", "N", "P"])
        units["N"] = units["N"].astype(int)
        units["P"] = units["P"].astype(int)
        for o in todo:
            units[units["OMOP"] == o][["UNIT", "N", "P"]].sort_values("N", ascending=False) \
                .to_csv(self._cpath(o, "units.tsv"), sep="\t", index=False)

    def target_count(self, omop):
        """(main unit, harmonized rows in it) from the counts cache; (None, 0) if none."""
        path = self._cpath(omop, "units.tsv")
        if not path.exists():
            self.prefetch_counts([omop])
        if not path.exists():
            return None, 0
        units = pd.read_csv(path, sep="\t", keep_default_na=False, dtype={"UNIT": str})
        return (units.iloc[0]["UNIT"], int(units.iloc[0]["N"])) if len(units) else (None, 0)

    def target_people(self, omop):
        """People behind the concept's harmonized rows in its main unit (0 if none)."""
        self.target_count(omop)
        path = self._cpath(omop, "units.tsv")
        if not path.exists():
            return 0
        units = pd.read_csv(path, sep="\t", keep_default_na=False, dtype={"UNIT": str})
        return int(units.iloc[0]["P"]) if len(units) else 0

    def target_usable(self, omop):
        """Enough harmonized rows (--min-target-n) and people (--min-cell) to compare against."""
        tu, n = self.target_count(omop)
        return tu is not None and n >= self.args.min_target_n and self.target_people(omop) >= self.args.min_cell

    def prefetch(self, concepts):
        """Ranges and a value sample for every concept not cached yet (two scans for all of them)."""
        self.prefetch_counts(concepts)
        todo = sorted({str(o) for o in concepts if re.fullmatch(r"\d+", str(o))
                       and not self._cpath(o, "values.npz").exists()})
        if not todo:
            return
        ids = ",".join(sql_str(o) for o in todo)
        hv = num(COL["h_value"])
        if {COL["lower"], COL["upper"], COL["h_cf"]} <= self.has:
            with ticking(f"Reference ranges of {len(todo)} concept(s) (full scan)"):
                ranges = _f(_read(clickhouse(f"""
                SELECT toString(`{COL['h_omop']}`) AS o, {num(COL['lower'])} AS lo, {num(COL['upper'])} AS hi,
                       {num(COL['h_cf'])} AS cf, count(), uniqExact({PERSON})
                FROM file('{self.p}')
                WHERE o IN ({ids}) AND {hv} IS NOT NULL AND cf IS NOT NULL AND (lo IS NOT NULL OR hi IS NOT NULL)
                GROUP BY o, lo, hi, cf FORMAT TSV"""), ["OMOP", "LO", "HI", "CF", "N", "P"]),
                    ["LO", "HI", "CF", "N", "P"])
        else:
            ranges = pd.DataFrame(columns=["OMOP", "LO", "HI", "CF", "N", "P"])
        main = [(o, self.target_count(o)[0]) for o in todo]
        main = [(o, u) for o, u in main if u is not None]
        vals = pd.DataFrame(columns=["OMOP", "V", "P"])
        if main:
            pairs = ",".join(f"({sql_str(o)}, {sql_str(u)})" for o, u in main)
            vpath = self.cache / "concepts" / "_values_batch.parquet"
            with ticking(f"Value samples of {len(main)} concept(s) (full scan)"):
                self._outfile(f"""
                SELECT o AS OMOP, t.1 AS V, t.2 AS P FROM (
                    SELECT toString(`{COL['h_omop']}`) AS o,
                           groupArraySample({self.args.max_target_rows}, 1)(({hv}, {PERSON})) AS s
                    FROM file('{self.p}')
                    WHERE (toString(`{COL['h_omop']}`), toString(`{COL['h_unit']}`)) IN ({pairs})
                      AND {hv} IS NOT NULL
                    GROUP BY o) ARRAY JOIN s AS t""", vpath)
            vals = pd.read_parquet(vpath)
            vpath.unlink()
            vals["OMOP"] = vals["OMOP"].astype(str)
        for o in todo:
            ranges[ranges["OMOP"] == o][["LO", "HI", "CF", "N", "P"]].to_csv(self._cpath(o, "ranges.tsv"), sep="\t", index=False)
            sub = vals[vals["OMOP"] == o]
            np.savez(self._cpath(o, "values.npz"), v=sub["V"].to_numpy(dtype=float), p=sub["P"].to_numpy(dtype=np.uint64))

    def target(self, omop):
        """Main unit, harmonized values in it (sampled) with their person keys, and the count of
        rows. Values are only fetched when the concept has enough rows and people to compare against."""
        tu, n = self.target_count(omop)
        if not self.target_usable(omop):
            return tu, np.array([]), np.array([], dtype=np.uint64), n
        self.prefetch([omop])
        with np.load(self._cpath(omop, "values.npz")) as z:
            return tu, z["v"], z["p"], n

    def target_ranges(self, omop):
        """Distinct (lower, upper) pairs of the concept's harmonized rows, converted to its unit,
        with their rows (N) and people (P)."""
        path = self._cpath(omop, "ranges.tsv")
        if not path.exists():
            return pd.DataFrame(columns=["LOWER", "UPPER", "N", "P"])
        df = pd.read_csv(path, sep="\t")
        return pd.DataFrame({"LOWER": df["LO"] * df["CF"], "UPPER": df["HI"] * df["CF"], "N": df["N"], "P": df["P"]})


# ---------------------------------------------------------------------------
# Reference tables
# ---------------------------------------------------------------------------

class Usagi:
    def __init__(self):
        m = rd.get_usagi_mapping().copy()
        m.columns = ["TEST", "UNIT", "OMOP", "QUANTITY", "STATUS"]
        self.m = m
        self.pair = {(t, u): (s, o) for t, u, s, o in zip(m["TEST"], m["UNIT"], m["STATUS"], m["OMOP"])}
        app = m[(m["STATUS"] == "APPROVED") & ~m["OMOP"].isin(["0", "NA"])]
        self.approved = app.groupby("TEST")["OMOP"].agg(lambda s: sorted(set(s))).to_dict()
        self.approved_pairs = set(zip(app["TEST"], app["UNIT"]))
        self.concept_units = app.groupby("OMOP")["UNIT"].agg(lambda s: sorted(set(s) - {"NA", ""})).to_dict()
        self.tests = set(m["TEST"])
        unc = m[(m["STATUS"] != "APPROVED") & ~m["OMOP"].isin(["0", "NA"])]
        self.unchecked_concepts = unc.groupby("TEST")["OMOP"].agg(lambda s: sorted(set(s))).to_dict()
        q = m[~m["QUANTITY"].isin(["NA", ""]) & ~m["OMOP"].isin(["0", "NA"])]
        self.quantity = q.groupby("OMOP")["QUANTITY"].agg(lambda s: s.value_counts().index[0]).to_dict()
        try:
            names = pd.read_csv(rd.USAGI_MAPPING_FILE, sep="\t", dtype=str, keep_default_na=False)
            self.name = dict(zip(names["harmonization_omop::OMOP_ID"], names["harmonization_omop::OMOP_NAME"]))
        except Exception:
            self.name = {}
        conv = rd.get_unit_conversion().copy()
        conv.columns = ["QUANTITY", "SRC", "TO", "CF", "ONLY"]
        self.conv = conv.fillna("")

    def candidates(self, omop, target_unit, test):
        """
        Every unit of the concept's quantity that converts into target_unit, directly or through
        other units (vol% → % → ratio), with its overall factor (value in target = value × cf).
        The target unit itself is always a candidate (× 1). Units with the same factor are one
        candidate, named after the unit Usagi approves for this test, else the target unit, else
        the first. Formula conversions are listed apart (not tested).
        """
        q = self.quantity.get(omop)
        c = self.conv[(self.conv["QUANTITY"] == q)
                      & ((self.conv["ONLY"] == "") | (self.conv["ONLY"] == str(omop)))]
        edges, formula = {}, set()
        for src, to, cf in zip(c["SRC"], c["TO"], c["CF"]):
            if src in ("", "NA") or to in ("", "NA"):
                continue
            try:
                f = float(cf)
            except ValueError:
                formula.add(src)
                continue
            if f > 0:
                edges.setdefault(to, []).append((src, f))
        factor = {target_unit: 1.0}          # unit -> factor into target_unit
        queue = [target_unit]
        while queue:                          # walk conversions backwards from the target
            to = queue.pop(0)
            for src, f in edges.get(to, []):
                if src not in factor:
                    factor[src] = f * factor[to]
                    queue.append(src)
        numeric = {}
        for u, f in factor.items():
            numeric.setdefault(round(np.log10(f), 6), (f, []))[1].append(u)
        cands = []
        for f, units in sorted(numeric.values(), key=lambda x: x[0]):
            rep = next((u for u in units if (test, u) in self.approved_pairs), None) \
                or (target_unit if target_unit in units else sorted(units)[0])
            cands.append(dict(unit=rep, cf=float(f"{f:.10g}"), same=sorted(u for u in units if u != rep)))
        return cands, sorted(formula - set(factor)), q


def load_injection_table(path):
    if not path or not Path(path).exists():
        return {}
    t = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    t["source_unit_clean"] = t["source_unit_clean"].replace("", "NA")
    t["source_unit_clean_fix"] = t["source_unit_clean_fix"].replace("", "NA")
    return {(a, u): f for a, u, f in zip(t["TEST_NAME_ABBREVIATION"], t["source_unit_clean"], t["source_unit_clean_fix"])}


# ---------------------------------------------------------------------------
# Evidence
# ---------------------------------------------------------------------------

def value_tests(c_vals, t_vals):
    ks = ie._ks(c_vals, t_vals, ks_threshold=0.3, sig_level=0.05)
    tt = ie._t(c_vals, t_vals, sig_level=0.05)
    md = ie._mad(c_vals, t_vals, n_mad=3.0)
    if ks.passed:
        by, outcome = "KS", "PASS"
    elif tt.passed:
        by, outcome = "T", "PASS"
    else:
        by, outcome = "MAD", ("PASS" if md.passed else "FAIL")
    res = ie.PipelineResult(outcome, by, [ks, tt, md])
    rank = {"KS": 0, "T": 1, "MAD": 2}[by] if outcome == "PASS" else 3
    return res, dict(KS_STAT=float(ks.details["stat"]), KS_PASS=ks.passed, T_PASS=tt.passed,
                     MAD_DIST=float(md.details["distance"]), MAD_THRESHOLD=float(md.details["threshold"]),
                     MAD_PASS=md.passed, VALUES=outcome, VALUES_DECIDED_BY=by if outcome == "PASS" else "",
                     _rank=rank)


def with_counts(side, n_rows, n_people, min_cell):
    """A side_summary() computed on a sample, with the group's full rows and people instead of
    the sample's (hidden together under min_cell people)."""
    if side is None:
        return None
    out = dict(side)
    out["n_records"] = dc.count_or_below(n_rows, n_people, min_cell)
    out["n_subjects"] = dc.count_or_below(n_people, n_people, min_cell)
    if out["n_subjects"] == dc.BELOW:
        out.pop("mean", None)
        out.pop("pct", None)
    return out


def share_ok(part, base, persons, min_cell):
    """May the share part / base be published? The base, the part and the rest (base without
    the part) each come from at least min_cell people; an empty part or rest is fine. Otherwise
    the share, with the group's rows, would describe a handful of people."""
    def people(m):
        return dc.n_people(persons[m])
    if people(base) < min_cell:
        return False
    return all(not m.any() or people(m) >= min_cell for m in (part & base, base & ~part))


def share_txt(x, ok):
    """A share for the reason texts: '85%' when it may be published, else only its drift."""
    if ok and np.isfinite(x):
        return f"{100 * x:.0f}%"
    return "most" if np.isfinite(x) and x >= 0.5 else "some"


def _round(o, nd=5):
    if isinstance(o, float):
        return float(f"{o:.{nd}g}") if np.isfinite(o) else None
    if isinstance(o, dict):
        return {k: _round(v, nd) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_round(v, nd) for v in o]
    if isinstance(o, (np.floating,)):
        return _round(float(o), nd)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def range_pairs(df, min_cell):
    """The group's (lower, upper) reference ranges with their rows, each from at least min_cell people."""
    r = df[np.isfinite(df["LOWER"]) | np.isfinite(df["UPPER"])]
    g = (r.groupby([r["LOWER"].round(6).fillna(-1), r["UPPER"].round(6).fillna(-1)])["PERSON"]
          .agg(N="size", P="nunique").reset_index())
    g = g[g["P"] >= min_cell]
    return [[None if a < 0 else a, None if b < 0 else b, int(n)] for a, b, n, _ in g.itertuples(index=False)]


# ---------------------------------------------------------------------------
# One group
# ---------------------------------------------------------------------------

def resolve_concepts(test, unit, omop_raw, usagi):
    status, pair_omop = usagi.pair.get((test, unit), ("MISSING", "NA"))
    if omop_raw not in ("0", "NA", ""):
        return "mapped", [omop_raw], status
    if status == "UNCHECKED" and pair_omop not in ("0", "NA", ""):
        return "unchecked", [pair_omop], status
    approved = usagi.approved.get(test, [])
    if status == "UNCHECKED" and len(approved) == 1:
        return "unchecked", approved, status
    if len(approved) == 1:
        return "inferred", approved, status
    if len(approved) > 1:
        return "several", approved, status
    return "noconcept", [], status


def pre_classify(test, concepts, concept_case, usagi, data, args):
    """None when the group can be decided by the evidence, else its not-decidable result key."""
    if concept_case == "noconcept":
        return "nd_missing" if test not in usagi.tests else "nd_unchecked"
    usable = []
    for o in concepts:
        if data.target_usable(o):
            usable.append((o, data.target_count(o)[0]))
    if not usable:
        return "nd_lowtarget"
    if not any(usagi.candidates(o, tu, test)[0] for o, tu in usable):
        return "nd_noconv"
    return None


def evaluate_group(g, data, usagi, rules, args, cache_targets, status=lambda step: None):
    test, unit, n_rows, omop_raw = g["TEST"], g["UNIT"], int(g["N"]), g["OMOP_RAW"]
    n_people, mc = int(g["P"]), args.min_cell
    # file-safe id; the hash keeps tests apart that differ only in punctuation (". " vs ",")
    gid = f"{slug(test)}__{slug(unit)}-{hashlib.sha1(f'{test}|{unit}'.encode()).hexdigest()[:6]}"
    concept_case, concepts, pair_status = resolve_concepts(test, unit, omop_raw, usagi)
    status("loading the group's rows")
    rows = data.group_rows(test, unit)
    vals = rows["VALUE"].to_numpy(dtype=float)
    gpers = rows["PERSON"].to_numpy(dtype=np.uint64)
    lo_r, hi_r = rows["LOWER"].to_numpy(dtype=float), rows["UPPER"].to_numpy(dtype=float)
    ranged = rr.has_range(lo_r, hi_r)
    coverage = float(ranged.mean()) if len(rows) else 0.0
    n_ranged = int(round(coverage * n_rows))
    evidence = n_ranged >= args.min_ranged and coverage >= args.min_coverage
    # the coverage is published only when the rows with and without a range each come from enough people
    cov_ok = share_ok(ranged, np.ones(len(rows), dtype=bool), gpers, mc)

    info = dict(GROUP_ID=gid, TEST_NAME=test, SOURCE_UNIT=unit_label(unit), N_ROWS=n_rows, N_SUBJECTS=n_people,
                CONCEPT_CASE=CONCEPT_CASES[concept_case], USAGI_PAIR_STATUS=pair_status,
                RANGE_COVERAGE=round(coverage, 4) if cov_ok else "", N_RANGED=n_ranged,
                CURRENT_RULE=(f"{unit_label(unit)} → {unit_label(rules[(test, unit)])}"
                              if (test, unit) in rules else ""),
                ISSUE_TITLE=f"[unharm] {test} · {unit_label(unit)}")
    # the group's own values: percentiles only (as for any published distribution), full counts
    status("summarising the group's values")
    plot = {"group": with_counts(dc.side_summary(vals, gpers, min_subjects=mc), n_rows, n_people, mc),
            "ranges": range_pairs(rows, mc), "candidates": {}, "concepts": {}}

    cands, formula_units, lowtarget = [], [], []
    light = g.get("LIST") == "not decidable"      # no values or ranges needed, only counts
    for omop in concepts:
        status(f"concept {omop}: harmonized values and typical ranges")
        if omop not in cache_targets or (cache_targets[omop][6] and not light):
            no_clusters = pd.DataFrame(columns=rr.CLUSTER_COLS + ["P_MIN"])
            if light:
                tu, tn = data.target_count(omop)
                tv, tp, clusters = np.array([]), np.array([], dtype=np.uint64), no_clusters
            else:
                tu, tv, tp, tn = data.target(omop)
                tr = data.target_ranges(omop)
                clusters = rr.range_clusters(tr["LOWER"], tr["UPPER"], tr["N"], margin=args.margin,
                                             people=tr["P"]) if len(tr) else no_clusters
            cache_targets[omop] = (tu, tv, tp, tn, data.target_people(omop), clusters, light)
        tu, tv, tp, tn, tpeople, clusters, _ = cache_targets[omop]
        usable = tu is not None and tn >= args.min_target_n and tpeople >= mc
        plot["concepts"][omop] = {"target_unit": tu, "name": usagi.name.get(omop, ""),
                                  # the count is not released below --min-target-n rows or --min-cell
                                  # people, only that it is below
                                  "n_target": tn if usable else dc.BELOW,
                                  # typical ranges: each from at least min_cell people (lower bound: the
                                  # largest of its pairs); only limits and share are published
                                  "clusters": [{"LOWER": c["LOWER"], "UPPER": c["UPPER"], "SHARE": c["SHARE"]}
                                               for c in clusters.to_dict(orient="records")
                                               if c.get("P_MIN", 0) >= mc]}
        if not usable:
            lowtarget.append(omop)
            continue
        cs, fs, quantity = usagi.candidates(omop, tu, test)
        formula_units += fs
        for c in cs:
            c.update(omop=omop, target_unit=tu, quantity=quantity, t_vals=tv, t_pers=tp, t_n=tn,
                     t_people=tpeople, clusters=clusters)
            cands.append(c)

    # --- evidence per candidate ---
    cand_rows = []
    for k, c in enumerate(cands, 1):
        status(f"candidate {k}/{len(cands)}: {c['unit']} × {fmt_cf(c['cf'])} (value tests, ranges)")
        cid = f"{c['omop']}|{c['unit']}|{fmt_cf(c['cf'])}"
        t_in_cand = c["t_vals"] / c["cf"]
        if len(vals) >= 2 and len(t_in_cand) >= 2:
            res, v = value_tests(vals, t_in_cand)
            # publishable: percentiles and test results only (no curves through the raw values)
            pdata = ie.compute_release_plot_data(vals, gpers, t_in_cand, c["t_pers"], res, min_subjects=mc)
            pdata["candidate"] = with_counts(pdata["candidate"], n_rows, n_people, mc)
            pdata["target"] = with_counts(pdata["target"], c["t_n"], c["t_people"], mc)
        else:
            v = dict(KS_STAT=np.nan, KS_PASS=False, T_PASS=False, MAD_DIST=np.nan, MAD_THRESHOLD=np.nan,
                     MAD_PASS=False, VALUES="FAIL", VALUES_DECIDED_BY="", _rank=3)
            pdata = None
        match = rr.match_clusters(lo_r * c["cf"], hi_r * c["cf"], c["clusters"], args.margin) & ranged \
            if len(c["clusters"]) else np.zeros(len(rows), dtype=bool)
        c.update(cid=cid, v=v, match=match,
                 range_match=float(match.sum() / ranged.sum()) if ranged.any() else np.nan,
                 rm_ok=bool(ranged.any()) and share_ok(match, ranged, gpers, mc))
        plot["candidates"][cid] = {"unit": c["unit"], "cf": c["cf"], "omop": c["omop"],
                                   "target_unit": c["target_unit"], "same": c["same"], "plot": pdata}
    # rows that match exactly one candidate
    if cands and ranged.any():
        M = np.vstack([c["match"] for c in cands])
        uniq = M.sum(axis=0) == 1
        for i, c in enumerate(cands):
            c["assigned"] = float((M[i] & uniq).sum() / ranged.sum())
            c["as_ok"] = share_ok(M[i] & uniq, ranged, gpers, mc)
    for c in cands:
        c.setdefault("assigned", np.nan)
        c.setdefault("as_ok", False)

    # --- decision ---
    out = dict(info)
    pick, result, conf, reason = None, "nofit", "", ""
    passing = sorted([c for c in cands if c["v"]["VALUES"] == "PASS"],
                     key=lambda c: (c["v"]["_rank"], c["v"]["KS_STAT"]))
    best_v = passing[0] if passing else None
    by_range = sorted(cands, key=lambda c: -(c["range_match"] if np.isfinite(c["range_match"]) else -1))
    best_r = by_range[0] if (evidence and by_range and by_range[0]["range_match"] >= args.range_pass) else None
    two = [c for c in cands if np.isfinite(c["assigned"]) and c["assigned"] >= args.split_share]
    rm = lambda c: share_txt(c["range_match"], c["rm_ok"])     # a candidate's range match, publishable

    if concept_case == "noconcept":
        if test not in usagi.tests:
            result = "nd_missing"
            reason = (f"{test} is not in Usagi at all, so there is no concept to convert to. "
                      f"It needs to be added and mapped first.")
        else:
            result = "nd_unchecked"
            unc = usagi.unchecked_concepts.get(test, [])
            n_rows_usagi = int((usagi.m["TEST"] == test).sum())
            reason = (f"{test} has {n_rows_usagi} Usagi entr{'ies' if n_rows_usagi != 1 else 'y'}, all UNCHECKED"
                      + (f"; they propose concept {', '.join(unc)}, so approving the right row may be enough."
                         if unc else ", none with a concept yet. It needs a mapping first."))
    elif not cands and lowtarget:
        result = "nd_lowtarget"
        reason = "; ".join(
            f"Concept {o}{' (' + usagi.name[o] + ')' if usagi.name.get(o) else ''} has "
            f"too few harmonized rows to compare against" for o in lowtarget) + \
            f" (at least {args.min_target_n} harmonized rows from several people are needed)."
    elif not cands:
        result = "nd_noconv"
        parts = []
        for o in concepts:
            tu, q = cache_targets[o][0], usagi.quantity.get(o)
            if not q:
                parts.append(f"Concept {o} has no quantity in Usagi, so no conversion applies")
            else:
                parts.append(f"The conversion table has no numeric conversion into {tu} for {q}")
        reason = "; ".join(parts) + (f" (only formula conversions from {', '.join(sorted(set(formula_units)))})"
                                     if formula_units else "") + "."
    elif evidence and len(two) >= 2 and not best_r:
        result = "split"
        reason = ("The ranges point to two units: " + " and ".join(
            f"{c['unit']} ({share_txt(c['assigned'], c['as_ok'])} of the rows with a range)" for c in two[:2])
            + ". omop_injection.tsv holds one fix per test and unit, so this needs a split by lab or period.")
    elif best_v and best_r and best_v is best_r and best_v["v"]["_rank"] < 2:
        result, conf, pick = "suggest", "High", best_v
        reason = (f"The values match the harmonized {best_v['target_unit']} values after × {fmt_cf(best_v['cf'])} "
                  f"(decided by {best_v['v']['VALUES_DECIDED_BY']}), and {rm(best_r)} of the rows "
                  f"with a range sit on the concept's typical range.")
    elif best_v and best_r and best_v is best_r:
        result, conf, pick = "suggest", "Medium", best_r
        reason = (f"{rm(best_r).capitalize()} of the rows with a range sit on the {best_r['unit']} range. "
                  f"The values pass only the loose MAD test (KS {best_v['v']['KS_STAT']:.2f}), so the patients "
                  f"differ from the harmonized rows rather than the unit.")
    elif best_v and best_r:
        result = "conflict"
        reason = (f"The values fit {best_v['unit']} best (decided by {best_v['v']['VALUES_DECIDED_BY']}), but "
                  f"{rm(best_r)} of the rows with a range point to {best_r['unit']}.")
    elif best_v and not evidence:
        result, conf, pick = "suggest", "Medium", best_v
        reason = (f"The values match after × {fmt_cf(best_v['cf'])} (decided by {best_v['v']['VALUES_DECIDED_BY']}). "
                  f"Too few rows have a range to confirm it.")
    elif best_r and not best_v:
        result, conf, pick = "suggest", "Medium", best_r
        reason = (f"No value test passes, but {rm(best_r)} of the rows with a range sit on the "
                  f"{best_r['unit']} range: the patients differ from the harmonized rows, not the unit.")
    elif best_v and evidence:
        result = "conflict"
        reason = (f"The values fit {best_v['unit']}, but only {rm(by_range[0])} of the rows with a "
                  f"range match any candidate unit.")
    else:
        result = "nofit"
        accepted = ", ".join(f"{c['unit']} (× {fmt_cf(c['cf'])})" for c in cands)
        bv = min(cands, key=lambda c: c["v"]["KS_STAT"] if np.isfinite(c["v"]["KS_STAT"]) else 9)
        txt = (f"None of the concept's units fits ({accepted} into {bv['target_unit']}). "
               f"Values: closest is {bv['unit']} (KS {bv['v']['KS_STAT']:.2f}, MAD distance "
               f"{bv['v']['MAD_DIST']:.3g} vs {bv['v']['MAD_THRESHOLD']:.3g}). ")
        if not evidence:
            txt += "Ranges: too few rows have one."
        else:
            br = by_range[0]
            txt += (f"Ranges: {share_txt(coverage, cov_ok)} of the rows have one; the best match is {br['unit']} "
                    f"with {rm(br) if np.isfinite(br['range_match']) else '0%'}.")
            est = implied_factor(lo_r, hi_r, ranged, cands, args)
            if est:
                txt += (f" Most of them sit about × {fmt_cf(est['cf'])} away from the {est['target_unit']} typical "
                        f"range, which is not the factor of any of these units: the rows may hold another "
                        f"measurement, or a unit outside this concept.")
        tested = {u for c in cands for u in [c["unit"]] + c["same"]}
        no_conv = sorted({u for o in concepts for u in usagi.concept_units.get(o, [])} - tested - set(formula_units))
        if no_conv:
            txt += (f" Usagi accepts {', '.join(no_conv)} for this concept, but the conversion table has no numeric "
                    f"conversion from {'it' if len(no_conv) == 1 else 'them'} into {bv['target_unit']}.")
        if formula_units:
            txt += f" Formula conversions are not tested ({', '.join(sorted(set(formula_units)))})."
        reason = txt

    # --- unit case ---
    if concept_case == "noconcept":
        ucase = "missing" if unit == "NA" else "notassessed"
    elif lowtarget and not cands:
        ucase = "lowtarget"
    elif not cands and formula_units:
        ucase = "formula"
    elif not cands:
        ucase = "noconv"
    elif result == "nofit":
        ucase = "nomatch"
    elif unit == "NA":
        ucase = "missing"
    elif pick is not None:
        ucase = "synonym" if abs(pick["cf"] - 1) < 1e-9 else "wrong"
    else:
        ucase = "unconverted"

    # --- needs ---
    needs = []
    if result == "nd_missing":
        needs.append(f"Add {test} to Usagi and map it")
    elif result == "nd_unchecked":
        unc = usagi.unchecked_concepts.get(test, [])
        needs.append(f"Usagi review of {test}" + (f" (proposed concept {', '.join(unc)})" if unc else ""))
    elif result == "nd_lowtarget":
        needs.append("Harmonized rows for concept " + ", ".join(lowtarget)
                     + " (or a check that this is the right concept)")
    elif result == "nd_noconv":
        needs.append("A numeric conversion into " + ", ".join(
            f"{cache_targets[o][0]} ({usagi.quantity.get(o) or 'no quantity'})" for o in concepts if o in cache_targets))
    if pick is not None and (test, pick["unit"]) not in usagi.approved_pairs:
        needs.append(f"Usagi row ({test}, {pick['unit']}) → {pick['omop']}")

    for c in cands:
        cand_rows.append(dict(GROUP_ID=gid, CAND_ID=c["cid"], OMOP_ID=c["omop"], TARGET_UNIT=c["target_unit"],
                              UNIT=c["unit"], SAME_SCALE_UNITS="|".join(c["same"]), CONVERSION_FACTOR=c["cf"],
                              **{k: v for k, v in c["v"].items() if not k.startswith("_")},
                              RANGE_MATCH=c["range_match"] if c["rm_ok"] else "",
                              RANGE_ASSIGNED=c["assigned"] if c["as_ok"] else "",
                              PICKED=c is pick))

    ref = pick or best_v or best_r
    out.update(
        OMOP_ID=(pick or ref or {}).get("omop", concepts[0] if len(concepts) == 1 else "|".join(concepts)),
        UNIT_CASE=UNIT_CASES[ucase], RESULT=RESULTS[result], CONFIDENCE=conf,
        DECIDABLE="no" if result in NOT_DECIDABLE else "yes",
        FIX_UNIT=pick["unit"] if pick else "",
        NEEDS="; ".join(needs),
    )
    if pick is not None:
        out["CONVERSION_FACTOR"] = pick["cf"]
        out["TARGET_UNIT"] = pick["target_unit"]
    if ref is not None:
        out.setdefault("TARGET_UNIT", ref["target_unit"])
        out.update(VALUES=ref["v"]["VALUES"], VALUES_DECIDED_BY=ref["v"]["VALUES_DECIDED_BY"],
                   KS_STAT=ref["v"]["KS_STAT"], RANGE_MATCH=ref["range_match"] if ref["rm_ok"] else "")
    om = out["OMOP_ID"]
    out["OMOP_NAME"] = usagi.name.get(om, "") if "|" not in str(om) else ""
    out["OMOP_QUANTITY"] = usagi.quantity.get(om, "") if "|" not in str(om) else ""
    if om in cache_targets:
        out["N_TARGET"] = cache_targets[om][3]
        out["N_TARGET_SUBJECTS"] = cache_targets[om][4]
        out.setdefault("TARGET_UNIT", cache_targets[om][0])
    if (test, unit) in rules and pick is not None:
        out["VS_CURRENT_RULE"] = "Same" if rules[(test, unit)] == pick["unit"] else "Different"
    elif pick is not None:
        out["VS_CURRENT_RULE"] = "New"
    else:
        out["VS_CURRENT_RULE"] = "Rule exists" if (test, unit) in rules else ""
    out["REASON"] = reason
    key = "|".join(str(out.get(k, "")) for k in ("TEST_NAME", "SOURCE_UNIT", "RESULT", "FIX_UNIT",
                                                 "CONVERSION_FACTOR", "OMOP_ID"))
    out["SUGGESTION_ID"] = hashlib.sha1(key.encode()).hexdigest()[:10]

    status("unit history")
    plot["context"] = context_tables(data.group_context(test, unit), mc)
    plot["picked"] = pick["cid"] if pick else (ref["cid"] if ref else None)
    plot["reason"] = reason
    return out, cand_rows, plot


def implied_factor(lo_r, hi_r, ranged, cands, args):
    """When no candidate fits: does the group agree on one factor to the main typical range?"""
    clusters = next((c["clusters"] for c in cands if len(c["clusters"])), None)
    if clusters is None or not len(clusters):
        return None
    main = clusters.iloc[0]
    f = np.full(len(hi_r), np.nan)
    ok = ranged & np.isfinite(hi_r) & (hi_r > 0)
    if np.isfinite(main["UPPER"]):
        f[ok] = main["UPPER"] / hi_r[ok]
    ok2 = ranged & ~np.isfinite(f) & np.isfinite(lo_r) & (lo_r > 0)
    if np.isfinite(main["LOWER"]):
        f[ok2] = main["LOWER"] / lo_r[ok2]
    f = f[np.isfinite(f) & (f > 0)]
    if len(f) < args.min_ranged:
        return None
    med = float(np.median(f))
    share = float((np.abs(np.log10(f / med)) <= np.log10(args.margin)).mean())
    if share < args.range_pass:
        return None
    tu = next(c["target_unit"] for c in cands if len(c["clusters"]))
    return dict(cf=float(f"{med:.3g}"), share=share, target_unit=tu)


def unit_step(sent, pre, post):
    """What happened to the unit between the lab and the primary injection."""
    none = ("NA", "", "-", "(no unit)")
    if post == "-" or pre == "-":
        return ""
    steps = []
    if sent != pre:
        steps.append("cleaned" if pre not in none else "cleaned to no unit")
    if pre in none and post not in none:
        steps.append("injected")
    elif pre != post:
        steps.append("changed at injection")
    return ", ".join(steps) or "unchanged"


def context_tables(ctx, min_cell):
    """Unit history of the group's rows: one row per path sent → cleaned → after primary
    injection, with its rows. Only paths from at least min_cell people are shown. The paths left
    out must together come from at least min_cell people too (or be none): otherwise the smallest
    shown paths are left out as well, so the group's total minus the shown rows never describes
    a handful of people."""
    lab = lambda u: "(no unit)" if u in ("NA", "") else u
    keys = ["SRC_UNIT", "PRE_UNIT", "POST_UNIT"]
    g = ctx.groupby(keys)["PERSON"].agg(N="size", P="nunique").sort_values("N", ascending=False)
    shown = list(g[g["P"] >= min_cell].index)

    def left_out_people():
        out = ~ctx.set_index(keys).index.isin(shown)
        return dc.n_people(ctx["PERSON"].to_numpy()[out])
    while shown and 0 < left_out_people() < min_cell:
        shown.pop()                        # the smallest shown path joins the left-out ones
    return {"history": [[lab(a), lab(b), lab(c), unit_step(a, b, c), int(g.loc[(a, b, c), "N"])]
                        for a, b, c in shown]}


def publish_counts(gdf, min_target_n, min_cell):
    """Group-level counts for the released tables: the concept's harmonized rows and people are
    blank under min_target_n rows or min_cell people (too few to compare against), rows with a
    range (N_RANGED) are not released, and the coverage is rounded to whole percent (it is
    already blank when the rows with or without a range come from too few people)."""
    out = gdf.copy()
    n_t = pd.to_numeric(out["N_TARGET"], errors="coerce")
    p_t = pd.to_numeric(out["N_TARGET_SUBJECTS"], errors="coerce")
    ok = (n_t >= min_target_n) & (p_t >= min_cell)
    out["N_TARGET"] = pd.Series([int(n) if k else "" for n, k in zip(n_t, ok)], index=out.index, dtype=object)
    out["N_TARGET_SUBJECTS"] = pd.Series([int(p) if k else "" for p, k in zip(p_t, ok)], index=out.index, dtype=object)
    out["RANGE_COVERAGE"] = pd.to_numeric(out["RANGE_COVERAGE"], errors="coerce").round(2)
    return out.drop(columns=["N_RANGED"])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def summary_md(gdf, args, parquet):
    dec = gdf[gdf["DECIDABLE"] == "yes"]
    lines = ["# Unharmonized suggestions", "",
             f"Input `{parquet}`. {len(gdf):,} groups (test × unit, rows with a numeric value), "
             f"{int(gdf['N_ROWS'].sum()):,} rows: {len(dec):,} decidable with ≥ {args.min_count:,} failed rows "
             f"({int(dec['N_ROWS'].sum()):,} rows), {len(gdf) - len(dec):,} not decidable with "
             f"≥ {args.min_count_undecidable:,}.", "",
             "| Result | Confidence | Groups | Rows |", "| --- | --- | ---: | ---: |"]
    s = gdf.groupby(["RESULT", "CONFIDENCE"], dropna=False).agg(G=("GROUP_ID", "size"), N=("N_ROWS", "sum")).reset_index()
    order = {v: i for i, v in enumerate(RESULTS.values())}
    s = s.sort_values(["RESULT", "CONFIDENCE"], key=lambda x: x.map(order) if x.name == "RESULT" else x)
    for r in s.itertuples(index=False):
        lines.append(f"| {r.RESULT} | {r.CONFIDENCE or ''} | {r.G:,} | {int(r.N):,} |")
    lines += ["", "| Concept | Groups | Rows |", "| --- | ---: | ---: |"]
    for k, sub in gdf.groupby("CONCEPT_CASE"):
        lines.append(f"| {k} | {len(sub):,} | {int(sub['N_ROWS'].sum()):,} |")
    return "\n".join(lines) + "\n"


def build_parser():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1],
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--input", type=Path, default=latest_engine_output(),
                   help=f"Engine output parquet (default: the most recent run in {ENGINE_DIR})")
    p.add_argument("--out-dir", type=Path, default=Path("unharmonized_suggestions"))
    p.add_argument("--dump-dir", type=Path, default=Path("/mnt/disks/data/kanta/inject/tmp/unharm"),
                   help="Cache: one subfolder per input file, rebuilt when that file changes")
    p.add_argument("--force", action="store_true", help="Rebuild this input's cache from scratch")
    p.add_argument("--test", action="append", help="Only these TEST_NAMEs (repeatable)")
    p.add_argument("--min-count", type=int, default=1000, help="Minimum failed rows per decidable group")
    p.add_argument("--min-count-undecidable", type=int, default=1000,
                   help="Minimum failed rows per not-decidable group")
    p.add_argument("--min-target-n", type=int, default=100, help="Minimum harmonized rows per concept")
    p.add_argument("--max-target-rows", type=int, default=200_000, help="Harmonized values sampled per concept")
    p.add_argument("--max-group-rows", type=int, default=200_000, help="Rows sampled per group")
    p.add_argument("--margin", type=float, default=1.5, help="Range match tolerance (factor)")
    p.add_argument("--min-ranged", type=int, default=30, help="Minimum rows with a range for range evidence")
    p.add_argument("--min-coverage", type=float, default=0.05, help="Minimum share of rows with a range")
    p.add_argument("--range-pass", type=float, default=0.8, help="Share of ranged rows that must match")
    p.add_argument("--split-share", type=float, default=0.2, help="Share for a second unit to call a split")
    p.add_argument("--min-cell", type=int, default=dc.MIN_SUBJECTS,
                   help="Fewest people behind any published group, count or share")
    p.add_argument("--injection-table", type=Path, default=DEFAULT_INJECTION_TABLE,
                   help="Current omop_injection.tsv, to compare against (skipped if missing)")
    p.add_argument("--release", action="store_true",
                   help=f"Copy the outputs to {RELEASE_DIR}, after release_check.py finds nothing to fix")
    rd.add_branch_argument(p)
    return p


def main():
    args = build_parser().parse_args()
    if args.input is None:
        sys.exit(f"No engine output found in {ENGINE_DIR}: pass --input")
    rd.set_harmonization_branch(args.harmonization_branch)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    data = Data(args.input, args.dump_dir / slug(Path(args.input).stem), args)
    usagi = Usagi()
    rules = load_injection_table(args.injection_table)
    print(f"Input: {args.input}\nCurrent injection rules: {len(rules)}"
          + ("" if rules else f" (no table at {args.injection_table})"))

    raw = data.groups()
    agg = (raw.sort_values("N", ascending=False)
              .groupby(["TEST", "UNIT"], as_index=False)
              .agg(N=("N", "sum"), OMOP_RAW=("OMOP_RAW", "first")))
    no_name = agg["TEST"].isin(["NA", ""])
    if no_name.any():
        print(f"Left out: {int(agg.loc[no_name, 'N'].sum()):,} failed rows without a test name "
              f"(TEST_NAME_ABBREVIATION = NA)")
    agg = agg[~no_name & (agg["N"] >= min(args.min_count, args.min_count_undecidable))].reset_index(drop=True)
    # people behind each group: a group from fewer than --min-cell people is not analysed or published
    people = data.group_people()
    agg["P"] = [people.get((t, u), 0) for t, u in zip(agg["TEST"], agg["UNIT"])]
    few = agg["P"] < args.min_cell
    if few.any():
        print(f"Left out: {int(few.sum())} group(s) whose failed rows come from fewer than {args.min_cell} people")
    agg = agg[~few].reset_index(drop=True)
    if args.test:
        agg = agg[agg["TEST"].isin(args.test)].reset_index(drop=True)

    # Which list does each group go to? Decided before any evaluation: no concept, too few
    # harmonized rows, or no conversion into the concept's unit → not decidable. Only needs
    # the per-concept row counts (one scan for all concepts, cached).
    res = [resolve_concepts(t, u, o, usagi) for t, u, o in zip(agg["TEST"], agg["UNIT"], agg["OMOP_RAW"])]
    data.prefetch_counts({o for _, cs, _ in res for o in cs})
    pre = [pre_classify(t, cs, case, usagi, data, args) for t, (case, cs, _) in zip(agg["TEST"], res)]
    agg["CONCEPT_CASE"] = [CONCEPT_CASES[c] for c, _, _ in res]
    agg["OMOP_IDS"] = ["|".join(cs) for _, cs, _ in res]
    agg["LIST"] = ["decidable" if p is None else "not decidable" for p in pre]
    agg["PRE_RESULT"] = ["" if p is None else RESULTS[p] for p in pre]
    # each list has its own minimum group size
    cut = np.where(agg["LIST"] == "decidable", args.min_count, args.min_count_undecidable)
    agg = agg[agg["N"] >= cut].reset_index(drop=True)

    selected = [agg[agg["LIST"] == lst].sort_values("N", ascending=False) for lst in ("decidable", "not decidable")]
    print(f"{len(agg):,} groups ({int(agg['N'].sum()):,} rows): {len(selected[0]):,} decidable "
          f"(≥ {args.min_count:,} failed rows), {len(selected[1]):,} not decidable (≥ {args.min_count_undecidable:,})")

    # harmonized side only for the decidable groups
    data.prefetch({o for o in "|".join(selected[0]["OMOP_IDS"]).split("|") if o and data.target_usable(o)})

    groups, cands, plots, cache_targets = [], [], {}, {}
    for lst, sel in zip(("Decidable", "Not decidable"), selected):
        print(f"\n== {lst}: {sel['TEST'].nunique()} test(s), {len(sel)} group(s), {int(sel['N'].sum()):,} rows ==")
        t_list = time.time()
        for i, (_, g) in enumerate(sel.iterrows(), 1):
            st = GroupStatus(f"[{i:>4}/{len(sel)}] {g['TEST']:<28} {unit_label(g['UNIT']):<12} {int(g['N']):>9,}",
                             int(g["N"]))
            out, crow, plot = evaluate_group(g, data, usagi, rules, args, cache_targets, status=st)
            st.clear()
            groups.append(out)
            cands += crow
            plots[out["GROUP_ID"]] = plot
            el = time.time() - t_list
            print(f"[{i:>4}/{len(sel)}] {g['TEST']:<28} {unit_label(g['UNIT']):<12} {int(g['N']):>9,}  "
                  f"{out['RESULT']:<36} {out['CONFIDENCE']:<6} {out['FIX_UNIT']:<10} "
                  f"({_fmt_t(el)}{', ~' + _fmt_t(el / i * (len(sel) - i)) + ' left' if i < len(sel) else ''})",
                  flush=True)

    nd_cols = ["TEST_NAME", "SOURCE_UNIT", "N_ROWS", "N_SUBJECTS", "RESULT", "REASON", "NEEDS", "CONCEPT_CASE",
               "USAGI_PAIR_STATUS", "OMOP_ID", "OMOP_NAME", "TARGET_UNIT", "N_TARGET", "N_TARGET_SUBJECTS",
               "RANGE_COVERAGE", "ISSUE_TITLE"]
    gdf = publish_counts(pd.DataFrame(groups).reindex(columns=GROUP_COLS), args.min_target_n, args.min_cell)
    dec_df = gdf[gdf["DECIDABLE"] == "yes"]
    nd_df = gdf[gdf["DECIDABLE"] == "no"].reindex(columns=nd_cols)
    cdf = pd.DataFrame(cands).reindex(columns=CAND_COLS)
    prop = gdf[gdf["RESULT"] == RESULTS["suggest"]]
    proposed = pd.DataFrame({
        "TEST_NAME_ABBREVIATION": prop["TEST_NAME"],
        "source_unit_clean": prop["SOURCE_UNIT"].replace("(no unit)", ""),
        "source_unit_clean_fix": prop["FIX_UNIT"],
        "notes": prop["REASON"],
        "INJECTION_NOTES": "SUGGESTED (" + prop["CONFIDENCE"] + ")",
        "CONFIDENCE": prop["CONFIDENCE"], "VS_CURRENT_RULE": prop["VS_CURRENT_RULE"],
        "NEEDS": prop["NEEDS"], "SUGGESTION_ID": prop["SUGGESTION_ID"],
    })

    files = {
        "groups.tsv": lambda p: gdf.to_csv(p, sep="\t", index=False),
        "decidable.tsv": lambda p: dec_df.to_csv(p, sep="\t", index=False),
        "not_decidable.tsv": lambda p: nd_df.to_csv(p, sep="\t", index=False),
        "candidates.tsv": lambda p: cdf.to_csv(p, sep="\t", index=False),
        "proposed_omop_injection.tsv": lambda p: proposed.to_csv(p, sep="\t", index=False),
        "summary.md": lambda p: Path(p).write_text(summary_md(gdf, args, args.input)),
        "plot_data.json.gz": lambda p: Path(p).write_bytes(gzip.compress(
            json.dumps(_round(plots), allow_nan=False).encode())),
    }
    for name, write in files.items():
        write(args.out_dir / name)
    print(f"\nWrote {', '.join(files)} → {args.out_dir}/")

    # one figure per suggestion id, drawn only once: an issue keeps the figure it was opened with
    fig_dir, rel_fig = args.out_dir / "figures", RELEASE_DIR / "figures"
    fig_dir.mkdir(exist_ok=True)
    # a folder's figures count only when drawn by the current figure code (format.txt)
    current_format = lambda d: (d / "format.txt").exists() and (d / "format.txt").read_text().strip() == str(FIGURE_FORMAT)
    out_ok, rel_ok = current_format(fig_dir), current_format(rel_fig)
    # decidable groups only: a not-decidable issue is about what is missing, said in its text
    with_fig = [r for r in groups if r["DECIDABLE"] == "yes"]
    todo = [r for r in with_fig if not (out_ok and (fig_dir / f"{r['SUGGESTION_ID']}.png").exists())
            and not (rel_ok and (rel_fig / f"{r['SUGGESTION_ID']}.png").exists())]
    if todo:
        from figures import draw
        t0 = time.time()
        for k, r in enumerate(todo, 1):
            draw(r, plots[r["GROUP_ID"]], fig_dir / f"{r['SUGGESTION_ID']}.png")
            if _TTY:
                print(f"\rFigures for new suggestions: {k:,}/{len(todo):,} · {_fmt_t(time.time() - t0)}",
                      end="\n" if k == len(todo) else "", flush=True)
            elif k % 100 == 0 or k == len(todo):
                print(f"Figures for new suggestions: {k:,}/{len(todo):,} · {_fmt_t(time.time() - t0)}", flush=True)
    (fig_dir / "format.txt").write_text(f"{FIGURE_FORMAT}\n")
    print(f"Figures (decidable groups): {len(todo):,} new, {len(with_fig) - len(todo):,} already drawn")
    print(summary_md(gdf, args, args.input))
    problems = check_release(args.out_dir, args.min_target_n)
    for msg in problems[:20]:
        print(f"  RELEASE CHECK: {msg}")
    print(f"Release check: {len(problems)} problem(s) in {args.out_dir}")
    if args.release:
        if problems:
            sys.exit("release check failed: nothing copied to data/unharmonized/")
        RELEASE_DIR.mkdir(parents=True, exist_ok=True)
        for name in files:
            shutil.copy2(args.out_dir / name, RELEASE_DIR / name)
        rel_fig.mkdir(exist_ok=True)
        current = [fig_dir / f"{r['SUGGESTION_ID']}.png" for r in with_fig]   # this run's decidable groups only
        # added, never replaced or removed, unless the release holds figures of an older format
        new_figs = [f for f in current if f.exists() and (not rel_ok or not (rel_fig / f.name).exists())]
        for f in new_figs:
            shutil.copy2(f, rel_fig / f.name)
        (rel_fig / "format.txt").write_text(f"{FIGURE_FORMAT}\n")
        print(f"Released → {RELEASE_DIR}/ ({len(new_figs):,} new figure(s) in figures/)")


if __name__ == "__main__":
    main()
