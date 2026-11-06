"""
ref_decide.py

Test-level decision from the range-labelled candidate rows (shared by both versions).

  NO_EVIDENCE  fewer than --min-ranged decided rows: the ranges cannot decide
  REVIEW       more than half of the compared rows match no unit profile: the unit-less
               rows' ranges look like none of the test's units
  SINGLE       one scale group present -> one unit for the whole test
  MIXED        two or more groups present -> one unit per group, with supervised value
               cutoffs between consecutive groups (low -> high)

Cutoff-format rows (TEST_NAME, SUB_DIST, CUTOFF, UNIT) use the injection_results.tsv
convention: a value goes to the first row (by CUTOFF) whose CUTOFF >= value; the last row
has CUTOFF inf. For two groups SUB_DIST is low/high, exactly as the value engine writes it.
"""

import numpy as np
import pandas as pd

import ref_ranges as rr

COMPARED = ("MATCH", "VALUE_OVERRIDE", "VALUE_CONFLICT", "AMBIGUOUS", "NO_MATCH")


def sub_names(k):
    if k == 1:
        return ["all"]
    if k == 2:
        return ["low", "high"]
    if k == 3:
        return ["low", "mid", "high"]
    return [f"sub{i + 1}" for i in range(k)]


def ref_decision(values, res, args):
    values = np.asarray(values, dtype=float)
    status, group, rep = res["status"], res["group"], res["group_rep"]
    n = len(values)
    n_ranged   = int((status != "NO_RANGE").sum())
    n_compared = int(np.isin(status, COMPARED).sum())
    n_no_match = int((status == "NO_MATCH").sum())
    shares, n_dec = rr.group_shares(status, group, rep)
    present = rr.present_groups(status, group, values, rep,
                                mix_share=args.mix_share, min_group_rows=args.min_group_rows)
    no_match_share = n_no_match / n_compared if n_compared else np.nan

    if n_dec < args.min_ranged:
        decision = "NO_EVIDENCE"
    elif no_match_share > 0.5:
        decision = "REVIEW"
    elif len(present) == 0:
        decision = "NO_EVIDENCE"
    elif len(present) == 1:
        decision = "SINGLE"
    else:
        decision = "MIXED"

    units, cutoffs = [], []
    if decision in ("SINGLE", "MIXED"):
        units   = [rep[g] for g in present]
        cutoffs = rr.cutoffs_for_groups(values, status, group, present)
    reliable = all(np.isfinite(t) and acc >= args.min_cut_acc for t, acc, _ in cutoffs)

    counts = {s: int((status == s).sum()) for s in rr.STATUSES if (status == s).any()}
    out = dict(
        decision=decision, n=n, n_ranged=n_ranged, n_compared=n_compared, n_decided=n_dec,
        shares=shares, present=present, units=units, cutoffs=cutoffs, reliable=reliable,
        no_match_share=no_match_share, status_counts=counts,
    )
    out["rows"] = cutoff_rows(out, res)
    return out


def cutoff_rows(dec, res):
    """Cutoff-format rows for SINGLE / MIXED (empty otherwise)."""
    if dec["decision"] not in ("SINGLE", "MIXED"):
        return pd.DataFrame(columns=["SUB_DIST", "CUTOFF", "UNIT", "GROUP_SHARE",
                                     "CUTOFF_ACC", "CUTOFF_BAL_ACC"])
    rep, unit_group = res["group_rep"], res["unit_group"]
    k = len(dec["units"])
    names = sub_names(k)
    rows = []
    for i, (g, unit) in enumerate(zip(dec["present"], dec["units"])):
        t, acc, bal = dec["cutoffs"][i] if i < k - 1 else (np.inf, np.nan, np.nan)
        members = [u for u, gg in unit_group.items() if gg == g]
        rows.append(dict(SUB_DIST=names[i], CUTOFF=t, UNIT=unit,
                         SCALE_GROUP_UNITS="|".join(members),
                         GROUP_SHARE=dec["shares"].get(rep[g], np.nan),
                         CUTOFF_ACC=acc, CUTOFF_BAL_ACC=bal))
    return pd.DataFrame(rows)


def predict_rows(values, res, dec):
    """
    Row-level prediction of the range-primary version:
      decided rows       -> their scale group's unit (source 'range')
      NO_MATCH rows      -> unassigned (their range fits none of the units)
      other rows         -> the cutoff rule (source 'cutoff', or 'single' for SINGLE)
    For NO_EVIDENCE / REVIEW returns None: the caller falls back to the value engine.
    """
    if dec["decision"] not in ("SINGLE", "MIXED"):
        return None, None
    values = np.asarray(values, dtype=float)
    status, group, rep = res["status"], res["group"], res["group_rep"]
    pred   = np.full(len(values), "", dtype=object)
    source = np.full(len(values), "none", dtype=object)
    decided = np.isin(status, rr.DECIDED) & (group >= 0)
    pred[decided]   = [rep[g] for g in group[decided]]
    source[decided] = "range"
    rest = ~decided & (status != "NO_MATCH")
    if dec["decision"] == "SINGLE":
        pred[rest]   = dec["units"][0]
        source[rest] = "single"
    else:
        cut = [t for t, _, _ in dec["cutoffs"]]
        pred[rest]   = rr.apply_cutoffs(values[rest], cut, dec["units"])
        source[rest] = "cutoff"
    return pred, source


def range_agreement(pred_units, res, dec_mask=None):
    """
    Share of the decided rows whose predicted unit is in the scale group their range
    points to. Units without a profile never agree. pred_units: object array ('' = none).
    """
    status, group, unit_group = res["status"], res["group"], res["unit_group"]
    dec = np.isin(status, rr.DECIDED) & (group >= 0)
    if dec_mask is not None:
        dec &= dec_mask
    if not dec.any():
        return np.nan
    pg = np.array([unit_group.get(u, -2) for u in pred_units[dec]])
    return float((pg == group[dec]).mean())


def ref_info(dec, res):
    """Test-level REF_* columns shared by every output row of a test."""
    return dict(
        REF_DECISION=dec["decision"],
        N_RANGED=dec["n_ranged"],
        REF_COVERAGE=dec["n_ranged"] / dec["n"] if dec["n"] else np.nan,
        N_DECIDED=dec["n_decided"],
        REF_SHARES=rr.fmt_shares(dec["shares"]),
        REF_NO_MATCH=dec["no_match_share"],
        CUTOFF_RELIABLE=dec["reliable"] if dec["decision"] == "MIXED" else np.nan,
        REF_STATUS="{" + ",".join(f"{k}:{v}" for k, v in dec["status_counts"].items()) + "}",
        SCALE_GROUPS="{" + ",".join(
            f"{res['group_rep'][g]}:" + "|".join(u for u, gg in res["unit_group"].items() if gg == g)
            for g in sorted(res["group_rep"])) + "}",
    )
