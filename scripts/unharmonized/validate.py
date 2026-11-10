"""
validate.py — check the secondary-injection rules (omop_injection.tsv) against the data.

In the engine, a rule (test, unit) → fixed unit rewrites MEASUREMENT_UNIT right after the snapshot
cleaned-post-inj::MEASUREMENT_UNIT, and nothing changes the unit afterwards (the value is never
touched). So the rows a rule changed are exactly the rows with cleaned-post-inj::MEASUREMENT_UNIT !=
MEASUREMENT_UNIT (suggest.py dumps them to ruled.parquet), and their unit before the rule is
cleaned-post-inj::MEASUREMENT_UNIT.

Every rule applied in the run is checked with suggest.py's own decision engine, as if the rule did
not exist: the rows are evaluated like a failed group with the unit they had before the rule, the
concept is resolved from (test, that unit) and Usagi alone (not from the concept the rows got after
the rule), and the harmonized rows compared against never include rows a rule changed. The rule is

  Confirmed        the engine suggests the rule's unit, or a unit with the same conversion (same scale;
                   a factor, or a linear formula with a shift such as HbA1c % → mmol/mol);
                   CONFIDENCE is the engine's
  Disagrees        the engine suggests a unit with another conversion, or says the unit before the rule
                   was already right
  No effect        none of the rule's rows harmonize after the fix: the (test, unit) it produces is
                   missing or unchecked in Usagi, or has no conversion. Takes precedence over the
                   verdicts above (the scale may be right, but the rule changes nothing); the reason
                   gives the Usagi status, and the engine's unit when that one is approved in Usagi
  Not confirmed    the evidence picks no unit: values and ranges disagree, a split, or no unit fits
  Cannot check     not decidable (no concept, too few harmonized rows, no conversion into the
                   concept's unit), or the rule's rows come from fewer than --min-cell people
                   (their counts are then BELOW_THRESHOLD)
  Not applied      the rule is in the table, but no row was changed by it in this run (no counts)

RULE_STATUS says whether the rule that was applied is still in the current table. CONCEPT_MATCH
compares the engine's concept with the concept the rule's rows harmonized to. SHARE_HARMONIZED is
the share of the rule's rows that harmonized after the fix (published under the same rule as every
share: the part and the rest each from --min-cell people).

Outputs (written by suggest.py): validated.tsv, the candidates in candidates.tsv and the plot data in
plot_data.json.gz under the group ids "val__…", and one figure per VALIDATION_ID.
"""

import hashlib

import numpy as np
import pandas as pd

VERDICTS = ["Confirmed", "Disagrees", "No effect", "Not confirmed", "Cannot check", "Not applied"]

VAL_COLS = ["GROUP_ID", "TEST_NAME", "SOURCE_UNIT", "RULE_FIX", "RULE_STATUS", "N_ROWS", "N_SUBJECTS",
            "SHARE_HARMONIZED", "VALIDATION", "CONFIDENCE", "ENGINE_RESULT", "ENGINE_FIX", "ENGINE_CONVERSION",
            "RULE_CONVERSION", "CONCEPT_CASE", "OMOP_ID", "OMOP_NAME", "CONCEPT_MATCH", "TARGET_UNIT", "N_TARGET",
            "N_TARGET_SUBJECTS", "VALUES", "VALUES_DECIDED_BY", "KS_STAT", "RANGE_COVERAGE", "RANGE_MATCH",
            "REASON", "VALIDATION_ID", "ISSUE_TITLE"]


def _units(c):
    return [c["UNIT"]] + [u for u in str(c.get("SAME_SCALE_UNITS") or "").split("|") if u]


def _conv(sg, c):
    """A candidate row's conversion into the target unit, as text ('× 100', '× 10.93 − 23.5 into mmol/mol')."""
    shift = c.get("CONVERSION_SHIFT") or 0.0
    return f"{sg.conv_txt(float(c['CONVERSION_FACTOR']), float(shift))} into {c['TARGET_UNIT']}"


def _verdict(sg, out, crow, post, fix):
    """(verdict, engine fix, engine conversion, rule conversion, text) from the engine's result on the rows."""
    res = out["RESULT"]
    picked = next((c for c in crow if c["PICKED"]), None)
    if res in [sg.RESULTS[k] for k in sg.NOT_DECIDABLE]:
        why = res.split(": ", 1)[-1]          # "Not decidable: too few harmonized rows" → the why
        return "Cannot check", "", "", "", f"The engine cannot decide these rows: {why}."
    if res in (sg.RESULTS["suggest"], sg.RESULTS["usagi"]) and picked is not None:
        engine_fix = post if res == sg.RESULTS["usagi"] else picked["UNIT"]
        # the rule's unit among the candidates, preferably of the picked concept
        rule_c = (next((c for c in crow if fix in _units(c) and c["OMOP_ID"] == picked["OMOP_ID"]), None)
                  or next((c for c in crow if fix in _units(c)), None))
        rule_cf = _conv(sg, rule_c) if rule_c else ""
        into = _conv(sg, picked)
        if rule_c is not None and rule_c["CAND_ID"] == picked["CAND_ID"]:
            same = "" if sg.unit_label(engine_fix) == sg.unit_label(fix) else \
                f" ({sg.unit_label(engine_fix)} and {sg.unit_label(fix)} are the same scale)"
            if res == sg.RESULTS["usagi"]:
                txt = (f"Not knowing the rule, the engine finds the unit {sg.unit_label(post)} already right ({into}); "
                       f"the rule's {sg.unit_label(fix)} has the same factor.")
            else:
                txt = f"Not knowing the rule, the engine suggests {sg.unit_label(engine_fix)} ({into}), as the rule{same}."
            return "Confirmed", engine_fix, into, rule_cf, txt
        rule_txt = rule_cf if rule_c is not None else "not a unit that converts into the concept's unit"
        what = ("finds the unit before the rule already right" if res == sg.RESULTS["usagi"]
                else f"suggests {sg.unit_label(engine_fix)}")
        return ("Disagrees", engine_fix, into, rule_cf,
                f"Not knowing the rule, the engine {what} ({into}), but the rule sets {sg.unit_label(fix)} ({rule_txt}).")
    return "Not confirmed", "", "", "", f"The engine picks no unit for these rows ({res[0].lower()}{res[1:]})."


def _rule_status(rules, test, post, fix):
    now = rules.get((test, post))
    if now is None:
        return "Not in the table any more"
    if now != fix:
        return f"Changed in the table (now → {now if now != 'NA' else '(no unit)'})"
    return "In the table"


def run(sg, data, usagi, rules, args, cache_targets):
    """Validate every rule applied in the run. Returns (validated rows, candidate rows, plots,
    figure rows) for suggest.py to write."""
    mc = args.min_cell
    groups = data.ruled_groups()
    if args.test:
        groups = groups[groups["TEST"].isin(args.test)]
    groups = groups.sort_values("N", ascending=False).reset_index(drop=True)
    seen = set(zip(groups["TEST"], groups["POST_UNIT"]))
    print(f"\n== Rule validation: {len(groups)} applied rule(s), {int(groups['N'].sum()):,} rows "
          f"({len(rules)} rule(s) in the table) ==")

    # concepts resolved blind (from the unit before the rule), harmonized side fetched in one batch
    big = groups[groups["P"] >= mc]
    res = [sg.resolve_concepts(t, p, "NA", usagi) for t, p in zip(big["TEST"], big["POST_UNIT"])]
    data.prefetch_counts({o for _, cs, _ in res for o in cs})
    data.prefetch({o for _, cs, _ in res for o in cs if data.target_usable(o)})

    rows_out, cand_rows, plots, fig_rows = [], [], {}, []
    for i, g in enumerate(groups.itertuples(index=False), 1):
        test, post, fix, n, p = g.TEST, g.POST_UNIT, g.UNIT, int(g.N), int(g.P)
        key = hashlib.sha1(f"{test}|{post}|{fix}".encode()).hexdigest()[:6]
        gid = f"val__{sg.slug(test)}__{sg.slug(post)}__{sg.slug(fix)}-{key}"
        base = dict(GROUP_ID=gid, TEST_NAME=test, SOURCE_UNIT=sg.unit_label(post), RULE_FIX=sg.unit_label(fix),
                    RULE_STATUS=_rule_status(rules, test, post, fix),
                    ISSUE_TITLE=f"[unharm-check] {test} · {sg.unit_label(post)} → {sg.unit_label(fix)}")
        label = f"[{i:>4}/{len(groups)}] {test:<28} {sg.unit_label(post):<10} → {sg.unit_label(fix):<10} {n:>9,}"
        if p < mc:
            base.update(N_ROWS=sg.dc.BELOW, N_SUBJECTS=sg.dc.BELOW, VALIDATION="Cannot check",
                        REASON="The rule's rows come from too few people to be checked or described.")
            rows_out.append(base)
            print(f"{label}  Cannot check (too few people)", flush=True)
            continue

        allrows = data.ruled_all(test, post, fix)
        persons = allrows["PERSON"].to_numpy(dtype=np.uint64)
        ok = allrows["H_OK"].to_numpy(dtype=float) > 0
        share_h = round(float(ok.mean()), 4) if sg.share_ok(ok, np.ones(len(ok), dtype=bool), persons, mc) else ""
        # the rule's rows that still fail are also in the failed group (test, fix): when that group's
        # rows from elsewhere (its total minus this rule's) come from a handful of people, nothing may
        # give the rule's failing rows away: no share, and for "No effect" (all fail) no row count
        fr = data._rows(test, fix)
        rest = fr.loc[fr["POST_UNIT"].astype(str) != post, "PERSON"].to_numpy()
        rest_small = 0 < sg.dc.n_people(rest) < mc
        if rest_small:
            share_h = ""
        h_omop = allrows.loc[ok, "H_OMOP"].astype(str)
        h_omop = h_omop[~h_omop.isin(["NA", "0", ""])]
        rows_concept = h_omop.value_counts().index[0] if len(h_omop) else ""

        st = sg.GroupStatus(label, n)
        out, crow, plot = sg.evaluate_group(
            {"TEST": test, "UNIT": post, "N": n, "OMOP_RAW": "NA", "P": p, "LIST": "decidable"},
            data, usagi, rules, args, cache_targets, status=st,
            rows=data.ruled_rows(test, post, fix), ctx=data.ruled_context(test, post, fix))
        st.clear()
        verdict, e_fix, e_cf, r_cf, txt = _verdict(sg, out, crow, post, fix)
        if not ok.any():
            # the rule fires but changes nothing: none of its rows harmonize with the unit it produces
            verdict = "No effect"
            status, pair_omop = usagi.pair.get((test, fix), ("MISSING", "NA"))
            pair = f"({test}, {sg.unit_label(fix)})"
            why = {"MISSING": f"{pair} is not in Usagi",
                   "UNCHECKED": f"{pair} is unchecked in Usagi"}.get(
                status, f"{pair} is {status.lower()} in Usagi (→ {pair_omop}), but its rows still do not convert")
            hint = ""
            if e_fix and sg.unit_label(e_fix) != sg.unit_label(fix) and (test, e_fix) in usagi.approved_pairs:
                hint = (f" ({test}, {sg.unit_label(e_fix)}) is approved in Usagi: a rule "
                        f"{sg.unit_label(post)} → {sg.unit_label(e_fix)} would let them harmonize.")
            txt = f"None of the rule's rows harmonize after the fix: {why}.{hint} {txt}"
        omop = str(out.get("OMOP_ID", ""))
        match = ("" if not rows_concept else "Same" if rows_concept == omop
                 else f"Different (the rule's rows harmonized to {rows_concept})")
        if match.startswith("Different"):
            txt += f" The rule's rows harmonized to concept {rows_concept}, the engine's concept is {omop or 'none'}."
        base.update(N_ROWS="" if (rest_small and verdict == "No effect") else n, N_SUBJECTS=p,
                    SHARE_HARMONIZED=share_h, VALIDATION=verdict,
                    CONFIDENCE=out.get("CONFIDENCE", "") if verdict in ("Confirmed", "Disagrees") else "",
                    ENGINE_RESULT=out["RESULT"], ENGINE_FIX=sg.unit_label(e_fix) if e_fix else "",
                    ENGINE_CONVERSION=e_cf, RULE_CONVERSION=r_cf, CONCEPT_MATCH=match,
                    REASON=f"{txt} Engine: {out['REASON']}",
                    **{k: out.get(k, "") for k in ("CONCEPT_CASE", "OMOP_ID", "OMOP_NAME", "TARGET_UNIT", "N_TARGET",
                                                    "N_TARGET_SUBJECTS", "VALUES", "VALUES_DECIDED_BY", "KS_STAT",
                                                    "RANGE_COVERAGE", "RANGE_MATCH")})
        vkey = "|".join(str(x) for x in (test, post, fix, verdict, e_fix, e_cf, omop))
        base["VALIDATION_ID"] = hashlib.sha1(vkey.encode()).hexdigest()[:10]
        rows_out.append(base)
        for c in crow:
            c["GROUP_ID"] = gid
        cand_rows += crow
        plot["reason"] = base["REASON"]
        plots[gid] = plot
        # the figure: the engine's evidence, titled with the rule and the verdict
        fig_rows.append(dict(out, GROUP_ID=gid, SOURCE_UNIT=sg.unit_label(post), SUGGESTION_ID=base["VALIDATION_ID"],
                             RESULT=f"rule → {sg.unit_label(fix)}: {verdict}", ID_LABEL="validation"))
        print(f"{label}  {verdict:<14} {base['CONFIDENCE']:<6} engine: {base['ENGINE_FIX'] or out['RESULT']}", flush=True)

    # rules in the table that changed no row in this run
    for (test, post), fix in sorted(rules.items()):
        if (test, post) in seen or (args.test and test not in args.test):
            continue
        rows_out.append(dict(
            GROUP_ID=f"val__{sg.slug(test)}__{sg.slug(post)}__{sg.slug(fix)}-"
                     f"{hashlib.sha1(f'{test}|{post}|{fix}'.encode()).hexdigest()[:6]}",
            TEST_NAME=test, SOURCE_UNIT=sg.unit_label(post), RULE_FIX=sg.unit_label(fix), RULE_STATUS="In the table",
            VALIDATION="Not applied",
            REASON="The rule is in the table, but no row with a value was changed by it in this run "
                   "(no such rows, or the run is older than the rule).",
            ISSUE_TITLE=f"[unharm-check] {test} · {sg.unit_label(post)} → {sg.unit_label(fix)}"))

    vdf = pd.DataFrame(rows_out).reindex(columns=VAL_COLS)
    vdf = sg.publish_counts(vdf, args.min_target_n, mc)
    order = {v: k for k, v in enumerate(VERDICTS)}
    vdf = vdf.sort_values("VALIDATION", key=lambda s: s.map(order), kind="stable").reset_index(drop=True)
    return vdf, cand_rows, plots, fig_rows


def summary_lines(vdf):
    """Markdown lines for summary.md: rules per verdict."""
    if vdf is None or not len(vdf):
        return []
    lines = ["", "## Secondary-injection rules", "", "| Validation | Rules |", "| --- | ---: |"]
    counts = vdf["VALIDATION"].value_counts()
    lines += [f"| {v} | {int(counts[v])} |" for v in VERDICTS if v in counts]
    return lines
