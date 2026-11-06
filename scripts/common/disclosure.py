"""
disclosure.py — what may be published about groups of lab rows (FinnGen minimum-N rules).

Every published number or picture describes a group of rows, and FinnGen's rule is about the
people behind them: each group has to come from at least MIN_SUBJECTS people. Rows are not
people (one person can have hundreds of INR rows), so groups are counted by person throughout.

Two levels:
  * a count, a share or a single statistic (mean, median, test result) of a group needs
    at least MIN_SUBJECTS (5) people;
  * the shape of a distribution is only published as percentiles PCTS (P5 .. P95), and only
    for groups of at least MIN_DIST_SUBJECTS (60) people and MIN_DIST_RECORDS (100) rows.
    60 is the decile rule of kanta_lab_preprocessing's summaries (people // 10 > 5); 100 rows
    put P5 and P95 beyond the 5 most extreme rows. No minimum or maximum, no histogram and
    no curve drawn through the raw values is ever published.

Below a threshold the count is replaced by BELOW (never by a "fewer than 5" text).
"""

import numpy as np

MIN_SUBJECTS = 5
MIN_DIST_SUBJECTS = 60
MIN_DIST_RECORDS = 100
PCTS = tuple(range(5, 100, 5))          # P5, P10, ..., P95
BELOW = "BELOW_THRESHOLD"

# Keys that hold row-level detail (raw curves, histograms, fitted samples). None of them may
# appear in published plot data; check_tree() looks for them.
FORBIDDEN_KEYS = {
    "ecdf", "kde_linear", "kde_log", "c_x", "c_y", "t_x", "t_y", "ks_marker",
    "hist_counts", "hist_edges", "counts", "edges", "x_fit", "x_orig", "values", "curves",
}


def sig(x, digits=4):
    """x rounded to `digits` significant digits (None for missing / non-finite)."""
    if x is None:
        return None
    x = float(x)
    if not np.isfinite(x):
        return None
    return float(f"{x:.{digits}g}")


def n_people(persons):
    """Number of distinct people in an array of person keys."""
    return int(len(np.unique(np.asarray(persons))))


def count_or_below(n, people, min_subjects=MIN_SUBJECTS):
    """A count for publication: n when the group has at least min_subjects people, else BELOW."""
    return int(n) if people >= min_subjects else BELOW


def bands_have_people(values, persons, min_subjects=MIN_SUBJECTS, pcts=PCTS):
    """
    True when every stretch of rows between two published percentiles, and each tail beyond
    the first and last one, holds rows of at least min_subjects different people.

    Percentiles are positions among rows, and rows are not people: one person with hundreds of
    rows can fill the top 20% on their own, and P80..P95 would then describe that person. Read
    as an equal-mass histogram (each stretch is a bar of 5% of the rows), this is FinnGen's
    histogram rule: every bar from at least min_subjects people. Stretches are cut by rank, so
    tied values do not leave empty stretches.
    """
    order = np.argsort(np.asarray(values, dtype=float), kind="stable")
    p = np.asarray(persons)[order]
    n = len(p)
    cuts = [0] + [int(np.floor(q / 100 * n)) for q in pcts] + [n]
    return all(len(np.unique(p[a:b])) >= min_subjects for a, b in zip(cuts[:-1], cuts[1:]))


def side_summary(values, persons, min_subjects=MIN_SUBJECTS,
                 min_dist_subjects=MIN_DIST_SUBJECTS, min_dist_records=MIN_DIST_RECORDS):
    """
    What may be published about one group of values (one side of a comparison).

      n_records, n_subjects   rows and people, or BELOW under min_subjects people
      mean                    from min_subjects people
      pct                     P5..P95 (PCTS), from min_dist_subjects people and min_dist_records
                              rows, and only when every 5% stretch of rows (and each tail) comes
                              from min_subjects people (bands_have_people)
    """
    values = np.asarray(values, dtype=float)
    persons = np.asarray(persons)
    people = n_people(persons)
    out = {"n_records": count_or_below(len(values), people, min_subjects),
           "n_subjects": count_or_below(people, people, min_subjects)}
    ok = np.isfinite(values)
    values, persons = values[ok], persons[ok]
    if people >= min_subjects and len(values):
        out["mean"] = sig(np.mean(values))
    if (people >= min_dist_subjects and len(values) >= min_dist_records
            and bands_have_people(values, persons, min_subjects)):
        out["pct"] = [sig(v) for v in np.percentile(values, PCTS)]
    return out


def shows_distribution(side):
    return bool(side) and "pct" in side


# ---------------------------------------------------------------------------
# Checks on what is about to be published
# ---------------------------------------------------------------------------

def check_count(n, where, problems, min_subjects=MIN_SUBJECTS):
    """A published people count must be BELOW or >= min_subjects."""
    if n is None or n == BELOW or n in ("", "NA"):
        return
    try:
        v = float(n)
    except (TypeError, ValueError):
        problems.append(f"{where}: count {n!r} is neither a number nor {BELOW}")
        return
    if np.isfinite(v) and v < min_subjects:
        problems.append(f"{where}: {v:g} people (< {min_subjects})")


def check_side(side, where, problems, min_subjects=MIN_SUBJECTS,
               min_dist_subjects=MIN_DIST_SUBJECTS, min_dist_records=MIN_DIST_RECORDS):
    """A published side_summary(): counts above the minimum, percentiles only for large groups."""
    if not isinstance(side, dict):
        problems.append(f"{where}: not a side summary")
        return
    people, rows = side.get("n_subjects"), side.get("n_records")
    check_count(people, f"{where}.n_subjects", problems, min_subjects)
    if (people == BELOW) != (rows == BELOW):
        problems.append(f"{where}: rows and people must be hidden together")
    if people == BELOW and ("mean" in side or "pct" in side):
        problems.append(f"{where}: statistics of a group under {min_subjects} people")
    if "pct" in side:
        pct = side["pct"]
        if people == BELOW or people < min_dist_subjects or rows < min_dist_records:
            problems.append(f"{where}: percentiles of {people} people / {rows} rows "
                            f"(need {min_dist_subjects} people and {min_dist_records} rows)")
        if not isinstance(pct, list) or len(pct) != len(PCTS):
            problems.append(f"{where}: expected {len(PCTS)} percentiles")
    extra = set(side) - {"n_records", "n_subjects", "mean", "pct"}
    if extra:
        problems.append(f"{where}: unexpected keys {sorted(extra)}")


def check_tree(obj, where, problems):
    """No forbidden (row-level) key anywhere in a published JSON tree."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in FORBIDDEN_KEYS:
                problems.append(f"{where}.{k}: row-level detail must not be published")
            check_tree(v, f"{where}.{k}", problems)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            check_tree(v, f"{where}[{i}]", problems)


# One published comparison (candidate / target): what compute_release_plot_data() writes
COMPARISON_KEYS = {"outcome", "decided_by", "candidate", "target", "tests"}
TEST_KEYS = {"ks": {"stat", "mlogp", "passed"}, "t": {"stat", "mlogp", "passed"},
             "mad": {"c_median", "t_median", "MAD", "n_mad", "threshold", "distance", "passed"}}


def check_comparison(e, where, problems):
    """A published comparison: known keys only, both sides as side_summary() allows."""
    if not isinstance(e, dict):
        problems.append(f"{where}: not a comparison")
        return
    for k in set(e) - COMPARISON_KEYS:
        problems.append(f"{where}.{k}: unexpected key")
    for side in ("candidate", "target"):
        if side in e:
            check_side(e[side], f"{where}.{side}", problems)
    for name, t in (e.get("tests") or {}).items():
        extra = set(t) - TEST_KEYS.get(name, set())
        if extra:
            problems.append(f"{where}.tests.{name}: unexpected keys {sorted(extra)}")
