"""
ranges.py

Reference-range logic shared by the range-based experiments (scripts_ref/) and the
unharmonized suggestions (scripts/unharmonized/suggest.py). Pure numpy/pandas, no I/O.

Idea (plans/references.md): a row's reference range is on the scale of the value the
lab reported, whatever unit it declared (or failed to declare). Each unit of a test
has its ranges in one narrow cluster, so the range tells the unit of a unit-less row
even where the value distributions of two units overlap.

Steps
-----
1. learn_profiles      per unit of the test: median lower / median upper limit over the
                       unit's target rows that carry a range (weighted, 5th/95th kept).
2. scale_groups        units whose profiles match each other (e.g. pH in `form`, `1`,
                       `ph`) are one scale group: ranges cannot tell them apart, and any
                       of them is a valid answer. The group is represented by its most
                       prevalent unit.
3. assign_rows         per candidate row with a range: distance to each unit profile
                       = |log10(row limit / profile limit)|, combined over the limits both
                       sides have (max by default, or mean). A row matches a group when
                       its distance is within log10(margin). Lower or upper alone is
                       fine: only the limits present on both sides are compared.
4. value_check         the row's own value confirms the range: robust z of log10(value)
                       under the assigned group's target values. When the value is
                       implausible there (> far) and plausible under exactly one other
                       group, the value wins (VALUE_OVERRIDE, e.g. a 41 % hematocrit with
                       a 0.35-0.46 range); when it is plausible under several, the row
                       is left undecided (VALUE_CONFLICT).
5. present_groups /    group shares over the decided rows; groups with enough rows are
   best_cutoff         "present". Two or more present -> a value cutoff between
                       consecutive groups, learned from the range-labelled rows
                       (supervised), with its accuracy on those rows.

Row statuses
------------
  MATCH            exactly one scale group within the margin
  VALUE_OVERRIDE   range pointed elsewhere, the value decided (see 4)
  VALUE_CONFLICT   range and value disagree, undecided
  AMBIGUOUS        several scale groups within the margin
  NO_MATCH         compared with at least one profile, none within the margin
  NO_PROFILE       has a limit, but no usable profile shares a limit with it
  NO_RANGE         no usable limit (both missing or <= 0)
"""

import numpy as np
import pandas as pd

DECIDED = ("MATCH", "VALUE_OVERRIDE")
STATUSES = ("MATCH", "VALUE_OVERRIDE", "VALUE_CONFLICT", "AMBIGUOUS",
            "NO_MATCH", "NO_PROFILE", "NO_RANGE")
_RSD_FLOOR = 0.01   # log10 robust SD floor (discrete values such as pH can give MAD = 0)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def wquantile(x, w, q):
    """Weighted quantile (lower step): first sorted value whose cumulative weight >= q."""
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    m = np.isfinite(x) & np.isfinite(w) & (w > 0)
    if not m.any():
        return np.nan
    o  = np.argsort(x[m], kind="stable")
    xs = x[m][o]
    cw = np.cumsum(w[m][o])
    i  = int(np.searchsorted(cw, q * cw[-1], side="left"))
    return float(xs[min(i, len(xs) - 1)])


def valid_limits(lower, upper):
    """Usable limits: finite and > 0. A pair with lower >= upper is dropped (both)."""
    lo = np.asarray(lower, dtype=float).copy()
    hi = np.asarray(upper, dtype=float).copy()
    lo[~(np.isfinite(lo) & (lo > 0))] = np.nan
    hi[~(np.isfinite(hi) & (hi > 0))] = np.nan
    bad = np.isfinite(lo) & np.isfinite(hi) & (lo >= hi)
    lo[bad] = np.nan
    hi[bad] = np.nan
    return lo, hi


def has_range(lower, upper):
    lo, hi = valid_limits(lower, upper)
    return np.isfinite(lo) | np.isfinite(hi)


def _limit_distance(x, p):
    """|log10(x / p)| where both usable, else NaN. x array, p scalar."""
    if not (np.isfinite(p) and p > 0):
        return np.full(len(x), np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.abs(np.log10(x / p))


def range_distance(lower, upper, p_lower, p_upper, combine="max"):
    """Distance (log10 factor) of each row's range to one profile; NaN if no shared limit."""
    lo, hi = valid_limits(lower, upper)
    d_l = _limit_distance(lo, p_lower)
    d_u = _limit_distance(hi, p_upper)
    if combine == "max":
        return np.fmax(d_l, d_u)
    if combine == "mean":
        both = np.vstack([d_l, d_u])
        n    = np.isfinite(both).sum(axis=0)
        s    = np.nansum(both, axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(n > 0, s / np.maximum(n, 1), np.nan)
    raise ValueError(f"combine must be 'max' or 'mean', not {combine!r}")


def fmt_shares(d):
    """{unit: share} -> '{u1:97.2,u2:2.8}' (percent, the PREVALENCE_DICT style)."""
    return "{" + ",".join(f"{u}:{100 * s:.1f}" for u, s in d.items()) + "}"


# ---------------------------------------------------------------------------
# 1. Unit profiles
# ---------------------------------------------------------------------------

PROFILE_COLS = ["UNIT", "UNIT_PREVALENCE", "N_TARGET", "N_RANGED", "N_LOWER", "N_UPPER",
                "MEDIAN_LOWER", "MEDIAN_UPPER",
                "LOWER_P05", "LOWER_P95", "UPPER_P05", "UPPER_P95",
                "MIXED_SCALE", "USE", "NOTE"]


def learn_profiles(pairs, units, min_rows=20, mixed_ratio=10.0, n_target=None):
    """
    pairs : DataFrame UNIT, LOWER, UPPER, N  (target rows aggregated by range; any NaN)
    units : list of (unit, prevalence %) to profile, in prevalence order
    Returns one row per unit (PROFILE_COLS). USE = enough ranged rows and not mixed-scale.
    """
    n_target = n_target or {}
    rows = []
    for unit, prev in units:
        sub = pairs[pairs["UNIT"] == unit]
        lo, hi = valid_limits(sub["LOWER"].to_numpy(), sub["UPPER"].to_numpy())
        n  = sub["N"].to_numpy(dtype=float)
        ml, mh = np.isfinite(lo), np.isfinite(hi)
        n_ranged = int(n[ml | mh].sum())
        r = dict(UNIT=unit, UNIT_PREVALENCE=prev, N_TARGET=n_target.get(unit, np.nan),
                 N_RANGED=n_ranged, N_LOWER=int(n[ml].sum()), N_UPPER=int(n[mh].sum()))
        for tag, x, m in (("LOWER", lo, ml), ("UPPER", hi, mh)):
            r[f"MEDIAN_{tag}"] = wquantile(x[m], n[m], 0.50) if m.any() else np.nan
            r[f"{tag}_P05"]    = wquantile(x[m], n[m], 0.05) if m.any() else np.nan
            r[f"{tag}_P95"]    = wquantile(x[m], n[m], 0.95) if m.any() else np.nan
        mixed = any(
            np.isfinite(r[f"{t}_P05"]) and r[f"{t}_P95"] / r[f"{t}_P05"] >= mixed_ratio
            for t in ("LOWER", "UPPER")
        )
        notes = []
        if n_ranged < min_rows:
            notes.append(f"few_ranged<{min_rows}")
        if mixed:
            notes.append(f"mixed_scale(p95/p05>={mixed_ratio:g})")
        if not (np.isfinite(r["MEDIAN_LOWER"]) or np.isfinite(r["MEDIAN_UPPER"])):
            notes.append("no_limits")
        r["MIXED_SCALE"] = bool(mixed)
        r["USE"]  = not notes
        r["NOTE"] = ",".join(notes)
        rows.append(r)
    return pd.DataFrame(rows, columns=PROFILE_COLS)


# ---------------------------------------------------------------------------
# 2. Scale groups
# ---------------------------------------------------------------------------

def scale_groups(profiles, margin=1.5, combine="max"):
    """
    Group usable units whose profiles are within the margin of each other (union-find).
    Returns (unit -> group id, group id -> representative unit = most prevalent).
    Unusable units get no group.
    """
    use = profiles[profiles["USE"]].sort_values("UNIT_PREVALENCE", ascending=False)
    units = use["UNIT"].tolist()
    lo = use["MEDIAN_LOWER"].to_numpy(dtype=float)
    hi = use["MEDIAN_UPPER"].to_numpy(dtype=float)
    parent = list(range(len(units)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    tol = np.log10(margin)
    for i in range(len(units)):
        d = range_distance(lo, hi, lo[i], hi[i], combine)
        for j in range(i + 1, len(units)):
            if np.isfinite(d[j]) and d[j] <= tol:
                parent[find(j)] = find(i)

    roots, unit_group, group_rep = {}, {}, {}
    for i, u in enumerate(units):          # prevalence order -> first unit of a group is its rep
        r = find(i)
        if r not in roots:
            roots[r] = len(roots)
            group_rep[roots[r]] = u
        unit_group[u] = roots[r]
    return unit_group, group_rep


# ---------------------------------------------------------------------------
# 3. Row assignment
# ---------------------------------------------------------------------------

def assign_rows(lower, upper, profiles, unit_group, margin=1.5, combine="max"):
    """
    Per row: status, group (-1 = none), distance to the nearest group, distance to the
    second nearest group. Groups are compared by their closest member unit.
    """
    n = len(lower)
    status = np.full(n, "NO_RANGE", dtype=object)
    group  = np.full(n, -1, dtype=int)
    dist   = np.full(n, np.nan)
    dist2  = np.full(n, np.nan)

    ranged = has_range(lower, upper)
    status[ranged] = "NO_PROFILE"
    if not unit_group or not ranged.any():
        return status, group, dist, dist2

    n_groups = max(unit_group.values()) + 1
    D = np.full((n, n_groups), np.nan)
    prof = profiles.set_index("UNIT")
    for u, g in unit_group.items():
        d = range_distance(lower, upper, prof.at[u, "MEDIAN_LOWER"], prof.at[u, "MEDIAN_UPPER"], combine)
        D[:, g] = np.fmin(D[:, g], d)

    compared = np.isfinite(D).any(axis=1) & ranged
    tol      = np.log10(margin)
    within   = np.isfinite(D) & (D <= tol)
    n_within = within.sum(axis=1)

    Dsort = np.sort(np.where(np.isfinite(D), D, np.inf), axis=1)
    dist[compared]  = Dsort[compared, 0]
    if n_groups > 1:
        d2 = Dsort[:, 1]
        dist2[compared] = np.where(np.isfinite(d2[compared]), d2[compared], np.nan)

    status[compared & (n_within == 0)] = "NO_MATCH"
    status[compared & (n_within > 1)]  = "AMBIGUOUS"
    one = compared & (n_within == 1)
    status[one] = "MATCH"
    group[one]  = np.argmax(within[one], axis=1)
    return status, group, dist, dist2


# ---------------------------------------------------------------------------
# 4. Value confirmation
# ---------------------------------------------------------------------------

def value_stats_from_values(values):
    """(median, robust SD) of log10 of the positive values; NaN when there are none."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v) & (v > 0)]
    if len(v) < 2:
        return np.nan, np.nan
    lv = np.log10(v)
    q25, q50, q75 = np.percentile(lv, [25, 50, 75])
    return float(q50), max(float((q75 - q25) / 1.349), _RSD_FLOOR)


def value_check(values, status, group, unit_group, vstats, far=3.0):
    """
    vstats: unit -> (median log10, robust SD log10) of the unit's target values.
    Rows with status MATCH and a positive value are checked; returns new (status, group).
    A value implausible under every group keeps the range decision (it is an outlier
    whatever its unit).
    """
    status = status.copy()
    group  = group.copy()
    if not unit_group:
        return status, group
    n_groups = max(unit_group.values()) + 1
    v  = np.asarray(values, dtype=float)
    ok = (status == "MATCH") & np.isfinite(v) & (v > 0)
    if not ok.any():
        return status, group

    lv = np.log10(v[ok])
    Z  = np.full((ok.sum(), n_groups), np.inf)
    for u, g in unit_group.items():
        med, rsd = vstats.get(u, (np.nan, np.nan))
        if np.isfinite(med) and np.isfinite(rsd):
            Z[:, g] = np.fmin(Z[:, g], np.abs(lv - med) / rsd)

    g_now     = group[ok]
    z_now     = Z[np.arange(len(g_now)), g_now]
    plausible = Z <= far
    bad       = z_now > far
    n_alt     = plausible.sum(axis=1) - plausible[np.arange(len(g_now)), g_now]

    idx = np.flatnonzero(ok)
    over = bad & (n_alt == 1)
    conf = bad & (n_alt > 1)
    if over.any():
        alt = plausible[over].copy()
        alt[np.arange(over.sum()), g_now[over]] = False
        group[idx[over]]  = np.argmax(alt, axis=1)
        status[idx[over]] = "VALUE_OVERRIDE"
    if conf.any():
        group[idx[conf]]  = -1
        status[idx[conf]] = "VALUE_CONFLICT"
    return status, group


# ---------------------------------------------------------------------------
# 5. Group shares, present groups, supervised cutoff
# ---------------------------------------------------------------------------

def group_shares(status, group, group_rep):
    """Share of the decided rows per group (rep unit -> share), largest first, and N decided."""
    dec = np.isin(status, DECIDED) & (group >= 0)
    n_dec = int(dec.sum())
    if n_dec == 0:
        return {}, 0
    counts = np.bincount(group[dec], minlength=len(group_rep))
    order  = np.argsort(-counts, kind="stable")
    return {group_rep[g]: counts[g] / n_dec for g in order if counts[g] > 0}, n_dec


def present_groups(status, group, values, group_rep, mix_share=0.05, min_group_rows=20):
    """
    Groups with >= mix_share of the decided rows and >= min_group_rows rows, ordered by the
    median value of their decided rows (low -> high). Returns list of group ids.
    """
    dec = np.isin(status, DECIDED) & (group >= 0)
    n_dec = dec.sum()
    out = []
    for g in range(len(group_rep)):
        m = dec & (group == g)
        k = int(m.sum())
        if k >= min_group_rows and n_dec and k / n_dec >= mix_share:
            out.append((float(np.median(values[m])), g))
    return [g for _, g in sorted(out)]


def best_cutoff(v_low, v_high):
    """
    Value cutoff t separating two labelled samples (rule: v <= t -> low group), chosen to
    maximise accuracy over both. Returns (t, accuracy, balanced accuracy); t is NaN when the
    best rule puts everything on one side.
    """
    v_low  = np.asarray(v_low, dtype=float)
    v_high = np.asarray(v_high, dtype=float)
    v_low, v_high = v_low[np.isfinite(v_low)], v_high[np.isfinite(v_high)]
    na, nb = len(v_low), len(v_high)
    if na == 0 or nb == 0:
        return np.nan, np.nan, np.nan
    v = np.concatenate([v_low, v_high])
    y = np.concatenate([np.zeros(na, int), np.ones(nb, int)])
    o = np.argsort(v, kind="stable")
    v, y = v[o], y[o]
    n = len(v)
    a_prefix = np.concatenate([[0], np.cumsum(y == 0)])   # low-group rows among the first k
    b_prefix = np.concatenate([[0], np.cumsum(y == 1)])
    k = np.arange(n + 1)
    correct = a_prefix + (nb - b_prefix)
    # a cut is only possible between distinct values
    valid = np.ones(n + 1, bool)
    valid[1:n] = v[1:] != v[:-1]
    correct = np.where(valid, correct, -1)
    kb = int(np.argmax(correct))
    acc = correct[kb] / n
    if kb == 0 or kb == n:
        return np.nan, float(acc), 0.5
    lo_v, hi_v = v[kb - 1], v[kb]
    t = float(np.sqrt(lo_v * hi_v)) if lo_v > 0 and hi_v > 0 else float((lo_v + hi_v) / 2)
    bal = 0.5 * (a_prefix[kb] / na + (nb - b_prefix[kb]) / nb)
    return t, float(acc), float(bal)


def cutoffs_for_groups(values, status, group, ordered_groups):
    """Supervised cutoffs between consecutive ordered groups: list of (t, acc, bal_acc)."""
    dec = np.isin(status, DECIDED)
    out = []
    for g_lo, g_hi in zip(ordered_groups[:-1], ordered_groups[1:]):
        out.append(best_cutoff(values[dec & (group == g_lo)], values[dec & (group == g_hi)]))
    return out


def apply_cutoffs(values, cutoffs, units):
    """Value rule: v <= c1 -> units[0], c1 < v <= c2 -> units[1], ... ; NaN cutoff = no split."""
    v = np.asarray(values, dtype=float)
    out = np.full(len(v), units[-1], dtype=object)
    for i in range(len(cutoffs) - 1, -1, -1):
        c = cutoffs[i]
        if np.isfinite(c):
            out[v <= c] = units[i]
    return out


# ---------------------------------------------------------------------------
# One call for a test
# ---------------------------------------------------------------------------

def analyse_test(cand, pairs, vstats, units, n_target=None, margin=1.5, combine="max",
                 far=3.0, min_profile_rows=20, mixed_ratio=10.0):
    """
    cand   : DataFrame VALUE, LOWER, UPPER (candidate = unit-less rows)
    pairs  : DataFrame UNIT, LOWER, UPPER, N (target rows aggregated by range)
    vstats : unit -> (median log10 value, robust SD) of the unit's target values
    units  : [(unit, prevalence %)] to consider
    Returns dict: profiles, unit_group, group_rep, status, group, dist, dist2
    """
    profiles = learn_profiles(pairs, units, min_rows=min_profile_rows,
                              mixed_ratio=mixed_ratio, n_target=n_target)
    unit_group, group_rep = scale_groups(profiles, margin=margin, combine=combine)
    profiles["SCALE_GROUP"] = profiles["UNIT"].map(
        lambda u: group_rep[unit_group[u]] if u in unit_group else "")
    lo = cand["LOWER"].to_numpy(dtype=float)
    hi = cand["UPPER"].to_numpy(dtype=float)
    status, group, dist, dist2 = assign_rows(lo, hi, profiles, unit_group, margin, combine)
    status, group = value_check(cand["VALUE"].to_numpy(dtype=float), status, group,
                                unit_group, vstats, far=far)
    return dict(profiles=profiles, unit_group=unit_group, group_rep=group_rep,
                status=status, group=group, dist=dist, dist2=dist2)


# ---------------------------------------------------------------------------
# 6. Range clusters (unharmonized suggestions)
# ---------------------------------------------------------------------------

CLUSTER_COLS = ["LOWER", "UPPER", "N", "SHARE"]


def range_clusters(lower, upper, weight, margin=1.5, min_share=0.05, max_clusters=3, people=None):
    """
    Typical ranges of one concept, from distinct (lower, upper) pairs on one scale
    (already converted to the concept's unit) and their row counts.

    A concept can have several typical ranges that are not units: INR has its normal
    range (0.9-1.2) and the warfarin target (2-3). So instead of one median profile the
    pairs are grouped greedily: most common pair first, every pair within `margin` of a
    cluster's seed joins it, otherwise it seeds a new cluster. Each cluster's limits are
    the weighted medians of its members. Clusters under `min_share` of the ranged rows
    are dropped; at most `max_clusters` are kept, largest first.

    With `people` (distinct people per pair, aligned with lower/upper/weight) each cluster also
    gets P_MIN: the most people behind any one of its pairs, a lower bound of its people (the
    pairs' people may overlap, so they cannot be added up).
    """
    lo, hi = valid_limits(lower, upper)
    w = np.asarray(weight, dtype=float)
    keep = (np.isfinite(lo) | np.isfinite(hi)) & (w > 0)
    lo, hi, w = lo[keep], hi[keep], w[keep]
    ppl = np.asarray(people, dtype=float)[keep] if people is not None else None
    cols = CLUSTER_COLS + (["P_MIN"] if people is not None else [])
    if not len(w):
        return pd.DataFrame(columns=cols)
    tol = np.log10(margin)
    order = np.argsort(-w, kind="stable")
    seeds, label = [], np.full(len(w), -1)
    for i in order:
        for k, (sl, sh) in enumerate(seeds):
            d = range_distance(lo[i:i + 1], hi[i:i + 1], sl, sh)[0]
            if np.isfinite(d) and d <= tol:
                label[i] = k
                break
        else:
            seeds.append((lo[i], hi[i]))
            label[i] = len(seeds) - 1
    total = w.sum()
    rows = []
    for k in range(len(seeds)):
        m = label == k
        rows.append(dict(LOWER=wquantile(lo[m], w[m], 0.5), UPPER=wquantile(hi[m], w[m], 0.5),
                         N=int(w[m].sum()), SHARE=float(w[m].sum() / total),
                         **({"P_MIN": int(ppl[m].max())} if ppl is not None else {})))
    out = pd.DataFrame(rows, columns=cols).sort_values("N", ascending=False)
    out = out[out["SHARE"] >= min_share].head(max_clusters).reset_index(drop=True)
    return out


def match_clusters(lower, upper, clusters, margin=1.5):
    """Per row: True when its range (same scale as the clusters) is within `margin` of any cluster."""
    lo = np.asarray(lower, dtype=float)
    out = np.zeros(len(lo), dtype=bool)
    tol = np.log10(margin)
    for _, c in clusters.iterrows():
        d = range_distance(lower, upper, c["LOWER"], c["UPPER"])
        out |= np.isfinite(d) & (d <= tol)
    return out
