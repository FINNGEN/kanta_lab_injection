"""
unharmonized.py

Portal pages for the unharmonized suggestions (scripts/unharmonized/suggest.py output):

  unharmonized/index.html            overview of every group, filterable
  unharmonized/groups/<group>.html   one page per group: the suggestion, the evidence per
                                     candidate unit, the values and reference-range figures,
                                     the raw unit spellings
  unharmonized/assets/portal.js      shared chart code
  unharmonized/proposed_omop_injection.tsv   the accepted-format proposal, for download

Review happens in GitHub issues in FINNGEN/kanta_lab_harmonisation_public: each group page has a
button that opens a pre-filled issue titled "[unharm] <test> · <unit>". The issue status shown on the pages is read from
GitHub when the site is built (see build.py), and re-checked live on each group page.

Only pandas + markdown are needed (the GitHub Pages build installs nothing else).
"""

import gzip
import html
import json
import shutil
import urllib.parse
from pathlib import Path

import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))     # scripts/
from common import disclosure   # noqa: E402

CDN = """
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css">
  <link rel="stylesheet" href="https://cdn.datatables.net/1.13.8/css/dataTables.bootstrap5.min.css">"""

STYLE = """
  <style>
    body   { padding: 2rem; font-size: 0.9rem; }
    nav a  { margin-right: 1.5rem; font-weight: 500; }
    .stat-label { color: #6c757d; font-size: 0.75rem; text-transform: uppercase; letter-spacing: .03em; }
    .stat-value { font-size: 1rem; font-weight: 600; font-variant-numeric: tabular-nums; }
    .section-box { border: 1px solid #dee2e6; border-radius: 6px; padding: 1rem; margin-bottom: 1.5rem; }
    .chart     { width: 100%; height: 360px; }
    .chart-sm  { width: 100%; height: 340px; }
    .res-suggest  { background: #198754; }
    .res-usagi    { background: #0d6efd; }
    .res-review   { background: #c27c0e; }
    .res-none     { background: #6c757d; }
    .conf-High    { background: #0f5132; }
    .conf-Medium  { background: #997404; }
    .case-chip { display: inline-block; border: 1px solid #ced4da; border-radius: 4px; padding: .15rem .5rem;
                 margin: 0 .4rem .4rem 0; background: #f8f9fa; }
    .case-chip b { font-weight: 600; color: #495057; margin-right: .3rem; }
    .fix-line { font-size: 1.15rem; font-weight: 600; }
    table.cands td, table.cands th { font-variant-numeric: tabular-nums; white-space: nowrap; }
    table.cands tr.sel { outline: 2px solid #0d6efd; outline-offset: -2px; }
    table.cands tr { cursor: pointer; }
    #groups td, #groups th { white-space: nowrap; }
    #groups td.wrap { white-space: normal; min-width: 180px; }
    thead tr.search-row input { width: 100%; font-size: .78rem; padding: 2px 4px; border: 1px solid #ced4da;
                                border-radius: 3px; box-sizing: border-box; }
    .issue-open   { background: #198754; }
    .issue-closed { background: #6f42c1; }
    .issue-none   { background: #adb5bd; }
    .summary-tables table { border-collapse: collapse; font-size: .85rem; margin: 0 2rem 1rem 0; display: inline-table; }
    .summary-tables th, .summary-tables td { border: 1px solid #dee2e6; padding: .25rem .6rem; }
    .summary-tables th { background: #f8f9fa; }
    .mock { background: #fff3cd; border: 1px solid #ffe69c; border-radius: 4px; padding: .4rem .7rem; margin-bottom: 1rem; }
  </style>"""

USAGI_RESULT = "Unit is right: approve the Usagi row"

RESULT_CLASS = {
    "Suggest fix": "res-suggest",
    "Unit is right: approve the Usagi row": "res-usagi",
    "Review: values and ranges disagree": "res-review",
    "Review: needs a split": "res-review",
    "No unit fits": "res-none",
    "Not decidable: test not in Usagi": "res-none",
    "Not decidable: test only unchecked in Usagi": "res-none",
    "Not decidable: too few harmonized rows": "res-none",
    "Not decidable: no conversion into the concept's unit": "res-none",
}


def _nav(prefix, current):
    links = [("Primary injection", f"{prefix}../index.html"), ("Unharmonized", f"{prefix}index.html"),
             ("Documentation", f"{prefix}../doc.html")]
    return "<nav class=\"mb-3\">" + "".join(
        f'<a href="{h}">{"<b>" + t + "</b>" if t == current else t}</a>' for t, h in links) + "</nav><hr>"


def _json(o):
    return json.dumps(o, allow_nan=False, default=str).replace("</", "<\\/")


def _clean(v):
    if v is None:
        return ""
    if isinstance(v, float) and v != v:
        return ""
    return v


def _rng(lo, hi):
    f = lambda v: f"{float(f'{v:.6g}'):g}"
    if lo is None and hi is None:
        return "—"
    return f"≤ {f(hi)}" if lo is None else (f"≥ {f(lo)}" if hi is None else f"{f(lo)}–{f(hi)}")


def ranges_md(row, plot, max_rows=5):
    """Markdown table of the concept's typical ranges (in the suggested or closest unit) and the
    group's most common ranges, for the issue body."""
    ok = lambda v: v is None or v > 0
    pairs = sorted([p for p in plot.get("ranges", []) if ok(p[0]) and ok(p[1]) and (p[0] is not None or p[1] is not None)
                    and (p[0] is None or p[1] is None or p[0] < p[1])], key=lambda p: -p[2])
    cand = plot.get("candidates", {}).get(plot.get("picked") or "")
    typical = []
    if cand:
        for c in plot.get("concepts", {}).get(cand["omop"], {}).get("clusters", []):
            lo = c["LOWER"] / cand["cf"] if c.get("LOWER") is not None else None
            hi = c["UPPER"] / cand["cf"] if c.get("UPPER") is not None else None
            typical.append((lo, hi, c.get("SHARE")))
    if not pairs and not typical:
        return []
    unit = cand["unit"] if cand else ""
    out = ["", f"**Reference ranges.** Unharmonized rows: the {row['TEST_NAME']} · {row['SOURCE_UNIT']} rows that fail "
               f"harmonization, ranges as the lab reported them."
           + (f" Harmonized: the typical range(s) of the rows already harmonized for this concept, converted to {unit}."
              " Matching ranges mean the rows are in that unit." if typical else ""),
           "", "| Rows | Reference range | Count |", "|---|---|---|"]
    for lo, hi, share in typical:
        out.append(f"| harmonized (typical, in {unit}) | {_rng(lo, hi)} | "
                   f"{'' if share is None else f'{100 * share:.0f}% of harmonized ranges'} |")
    for lo, hi, n in pairs[:max_rows]:
        out.append(f"| unharmonized rows | {_rng(lo, hi)} | {n:,} rows |")
    if len(pairs) > max_rows:
        out.append(f"| unharmonized rows | {len(pairs) - max_rows} other range(s) | {sum(p[2] for p in pairs[max_rows:]):,} rows |")
    return out


def _count(v):
    """A count from the tables with thousands separators; '—' when empty (not released)."""
    v = _clean(v)
    return "—" if v == "" else f"{int(float(v)):,}"


def issue_body(row, page_url, fig_url, crow, history, plot=None):
    """Self-contained review record: it stays readable after the group leaves the data."""
    fix = (f"{row['SOURCE_UNIT']} → {row['FIX_UNIT']} (× {row['CONVERSION_FACTOR']:g} into {row['TARGET_UNIT']})"
           if row["RESULT"] == "Suggest fix" else
           f"none: approve the Usagi row ({row['TEST_NAME']}, {row['SOURCE_UNIT']}) → {_clean(row['OMOP_ID'])} "
           f"(× {row['CONVERSION_FACTOR']:g} into {row['TARGET_UNIT']}, already in the conversion table)"
           if row["RESULT"] == USAGI_RESULT else "—")
    lines = [
        f"**Group page:** {page_url}",
        "",
        *([f"![evidence]({fig_url})", ""] if fig_url else []),
        f"- **Test · unit:** {row['TEST_NAME']} · {row['SOURCE_UNIT']} ({int(row['N_ROWS']):,} rows)",
        f"- **Result:** {row['RESULT']}" + (f" ({row['CONFIDENCE']} confidence)" if _clean(row["CONFIDENCE"]) else ""),
        f"- **Suggested fix:** {fix}",
        f"- **Concept:** {_clean(row['OMOP_ID'])} {_clean(row['OMOP_NAME'])} — {row['CONCEPT_CASE']}",
        f"- **Unit problem:** {row['UNIT_CASE']}",
        (f"- **Ranges:** {100 * float(row['RANGE_COVERAGE']):.0f}% of rows have one"
         if _clean(row["RANGE_COVERAGE"]) != "" else "- **Ranges:** —")
        + (f"; {100 * float(row['RANGE_MATCH']):.0f}% of those match the "
           f"{'suggested unit' if row['RESULT'] == 'Suggest fix' else 'unit'}"
           if row["RESULT"] in ("Suggest fix", USAGI_RESULT) and _clean(row.get("RANGE_MATCH")) != "" else ""),
        f"- **Needs:** {_clean(row['NEEDS']) or '—'}",
        f"- **Current rule:** {_clean(row['CURRENT_RULE']) or '—'} ({_clean(row['VS_CURRENT_RULE']) or '—'})",
        "",
        f"> {row['REASON']}",
    ]
    lines += ranges_md(row, plot or {})
    if crow:
        lines += ["", "| Candidate unit | Factor | KS | Values | Range match |", "|---|---|---|---|---|"]
        for c in list(crow.values())[:6]:
            ks = f"{float(c['KS_STAT']):.2f}" if c.get("KS_STAT") != "" else "—"
            rm = f"{100 * float(c['RANGE_MATCH']):.0f}%" if c.get("RANGE_MATCH") != "" else "—"
            v = c.get("VALUES") or "—"
            v += f" ({c['VALUES_DECIDED_BY']})" if c.get("VALUES_DECIDED_BY") else ""
            lines.append(f"| {c['UNIT']}{' ✔' if c.get('PICKED') else ''} | × {float(c['CONVERSION_FACTOR']):g} "
                         f"| {ks} | {v} | {rm} |")
    if history:
        lines += ["", "| Sent | Cleaned | After injection | Rows |", "|---|---|---|---|"]
        lines += [f"| {h[0]} | {h[1]} | {h[2]} | {h[4]:,} |" for h in history[:4]]
    lines += ["", f"Suggestion id: `{row['SUGGESTION_ID']}`"]
    return "\n".join(lines)


def issue_url(repo, row, page_url, fig_url, crow, history, plot=None):
    body = issue_body(row, page_url, fig_url, crow, history, plot)
    q = urllib.parse.urlencode({"title": row["ISSUE_TITLE"], "body": body})
    return f"https://github.com/{repo}/issues/new?{q}"


def issue_note(issue, sid):
    """What an existing issue means for a group that is still in the data."""
    if not issue:
        return ""
    if issue.get("sid") and issue["sid"] != sid:
        return "suggestion changed since the issue"
    if issue["state"] == "closed":
        return "issue closed, but the rows still fail"
    return ""


# ---------------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------------

_INDEX = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Unharmonized suggestions</title>__CDN____STYLE__
</head>
<body>
  __NAV__
  <h2>Unharmonized lab data: suggested fixes</h2>
  __MOCK__
  <p class="text-muted" style="max-width:62rem">Rows that have a numeric value but whose unit could not be harmonized,
  grouped by test and unit (rows without a value are not counted). For each group the concept, the candidate units and
  two independent kinds of evidence (the values against the harmonized values, the reference ranges against the
  concept's typical ranges) give a suggested fix for <code>omop_injection.tsv</code>, a case to review, or the reason
  there is none. Suggestions are proposals: open an issue on the group page to review one.
  <a href="proposed_omop_injection.tsv">Download the proposed rules (TSV)</a>.</p>

  <ul class="nav nav-tabs mb-3" role="tablist">
    <li class="nav-item"><button class="nav-link active" data-list="dec" type="button">Decidable
      <span class="badge bg-secondary">__N_DEC__</span></button></li>
    <li class="nav-item"><button class="nav-link" data-list="nd" type="button">Not decidable
      <span class="badge bg-secondary">__N_ND__</span></button></li>
    <li class="nav-item"><button class="nav-link" data-list="gone" type="button">Resolved
      <span class="badge bg-secondary">__N_GONE__</span></button></li>
  </ul>

  <section id="list-dec">
    <p class="text-muted mb-2" style="max-width:62rem">Groups with a concept and at least one unit that converts into
    it, so the evidence can decide: a fix, a case to review, or no unit fits.</p>
    <div class="summary-tables">__SUMMARY_DEC__</div>
    __TABLE_DEC__
  </section>
  <section id="list-nd" hidden>
    <p class="text-muted mb-2" style="max-width:62rem">Groups the evidence cannot decide, because there is nothing to
    compare against: no concept in Usagi, too few harmonized rows for the concept, or no conversion into its unit.
    The reason says what is missing; these still get a review issue.
    <a href="not_decidable.tsv">Download this list (TSV)</a>.</p>
    <div class="summary-tables">__SUMMARY_ND__</div>
    __TABLE_ND__
  </section>
  <section id="list-gone" hidden>
    <p class="text-muted mb-2" style="max-width:62rem">Review issues whose test and unit no longer fail harmonization
    in this data, e.g. because the fix is now in the engine. The issue keeps the evidence it was opened with.</p>
    __TABLE_GONE__
  </section>
  <p class="mt-2"><small class="text-muted">Issue status as of __ISSUES_AT__. Per-column filters take regular
  expressions (e.g. <code>^Suggest</code>).</small></p>

  <script src="https://code.jquery.com/jquery-3.7.0.min.js"></script>
  <script src="https://cdn.datatables.net/1.13.8/js/jquery.dataTables.min.js"></script>
  <script src="https://cdn.datatables.net/1.13.8/js/dataTables.bootstrap5.min.js"></script>
  <script>
  const LISTS = __LISTS__;
  const RESULT_CLASS = __RESULT_CLASS__;
  function esc(s) { return String(s == null ? '' : s).replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c])); }
  const render = {
    TEST_NAME: (d, t, r) => t === 'display' ? '<a href="groups/' + r.GROUP_ID + '.html">' + esc(d) + '</a>' : d,
    RESULT: (d, t, r) => {
      if (t !== 'display') return d;
      const badge = '<span class="badge ' + (RESULT_CLASS[d] || 'res-none') + '">' + esc(d) + (r.RESULT_LINK ? ' ↗' : '') + '</span>';
      // a clear-cut suggestion is clickable: it opens its review issue (or the existing one)
      return r.RESULT_LINK ? '<a href="' + esc(r.RESULT_LINK) + '" target="_blank" rel="noopener" title="' +
                             (r.ISSUE_NEW ? 'Open a review issue for this fix' : 'Go to the review issue') + '">' + badge + '</a>' : badge;
    },
    CONFIDENCE: (d, t) => t === 'display' && d ? '<span class="badge conf-' + d + '">' + esc(d) + '</span>' : d,
    N_ROWS: (d, t) => t === 'display' ? Number(d).toLocaleString('en-US') : d,
    RANGE_COVERAGE: (d, t) => t === 'display' && d !== '' ? Math.round(100 * d) + '%' : d,
    RANGE_MATCH: (d, t) => t === 'display' && d !== '' ? Math.round(100 * d) + '%' : d,
    ISSUE: (d, t, r) => {
      if (t !== 'display') return d;
      if (!r.ISSUE_NUMBER) return '<span class="badge issue-none">none</span>';
      return '<a class="badge ' + (r.ISSUE_STATE === 'open' ? 'issue-open' : 'issue-closed') + '" href="' + r.ISSUE_URL +
             '">#' + r.ISSUE_NUMBER + ' ' + r.ISSUE_STATE + '</a>' +
             (r.ISSUE_NOTE ? ' <span class="badge bg-warning text-dark">' + esc(r.ISSUE_NOTE) + '</span>' : '');
    },
  };
  $(function () {
    Object.keys(LISTS).forEach(key => {
      const L = LISTS[key], id = '#groups-' + key;
      const rowsCol = L.cols.findIndex(c => c[0] === 'N_ROWS');
      const table = $(id).DataTable({
        data: L.rows, pageLength: 50, order: [[rowsCol, 'desc']], orderCellsTop: true, autoWidth: false,
        columns: L.cols.map(c => ({ data: c[0], render: render[c[0]] || null, className: c[2] || '' })),
      });
      $(id + ' thead tr.search-row th').each(function (i) {
        $('input', this).on('keyup change', function () { table.column(i).search(this.value, true, false).draw(); });
      });
    });
    document.querySelectorAll('[data-list]').forEach(btn => btn.addEventListener('click', () => {
      document.querySelectorAll('[data-list]').forEach(b => b.classList.toggle('active', b === btn));
      ['dec', 'nd', 'gone'].forEach(k => { document.getElementById('list-' + k).hidden = k !== btn.dataset.list; });
      $.fn.dataTable.tables({ visible: true, api: true }).columns.adjust();
    }));
  });
  </script>
</body>
</html>
"""

COLS_DEC = [
    ("TEST_NAME", "Test"), ("SOURCE_UNIT", "Unit"), ("RESULT", "Result"), ("N_ROWS", "Rows"),
    ("CONFIDENCE", "Confidence"), ("FIX_UNIT", "Fix"), ("CONVERSION_FACTOR", "Factor"),
    ("REASON", "Reason", "wrap"), ("CONCEPT_CASE", "Concept", "wrap"), ("UNIT_CASE", "Unit problem", "wrap"),
    ("VALUES", "Values"), ("RANGE_COVERAGE", "With range"), ("RANGE_MATCH", "Range match"),
    ("NEEDS", "Needs", "wrap"), ("VS_CURRENT_RULE", "vs current rule"), ("ISSUE", "Issue"),
]
COLS_ND = [
    ("TEST_NAME", "Test"), ("SOURCE_UNIT", "Unit"), ("RESULT", "Result"), ("N_ROWS", "Rows"),
    ("REASON", "Reason", "wrap"), ("NEEDS", "Needs", "wrap"), ("CONCEPT_CASE", "Concept", "wrap"),
    ("RANGE_COVERAGE", "With range"), ("ISSUE", "Issue"),
]


def _gone_html(gone):
    if not gone:
        return "<p class='text-muted'>No review issue belongs to a group that has left the data.</p>"
    rows = "".join(
        f"<tr><td><a class='badge {'issue-closed' if g['state'] == 'closed' else 'issue-open'}' href='{g['url']}'>"
        f"#{g['number']} {g['state']}</a></td><td>{html.escape(g['test'])}</td><td>{html.escape(g['unit'])}</td>"
        f"<td>{html.escape(g['note'])}</td></tr>" for g in gone)
    return ("<table class='table table-sm table-striped' style='max-width:60rem'><thead><tr><th>Issue</th><th>Test</th>"
            f"<th>Unit</th><th>Status</th></tr></thead><tbody>{rows}</tbody></table>")


def _table_html(key, cols):
    filters = "".join(f'<th><input type="text" placeholder="{html.escape(c[1])}"></th>' for c in cols)
    headers = "".join(f"<th>{html.escape(c[1])}</th>" for c in cols)
    return (f'<div style="overflow-x:auto"><table id="groups-{key}" class="table table-sm table-striped '
            f'table-bordered" style="width:100%"><thead><tr class="search-row">{filters}</tr><tr>{headers}</tr>'
            f'</thead><tbody></tbody></table></div>')


def _summary_html(groups, by_concept=True):
    if not len(groups):
        return "<p class='text-muted'>None.</p>"
    order = list(RESULT_CLASS)
    s = groups.assign(CONFIDENCE=groups["CONFIDENCE"].fillna("")).groupby(["RESULT", "CONFIDENCE"]).agg(
        G=("GROUP_ID", "size"), N=("N_ROWS", "sum")).reset_index()
    s["o"] = s["RESULT"].map({r: i for i, r in enumerate(order)})
    s = s.sort_values(["o", "CONFIDENCE"])
    t1 = "<table><tr><th>Result</th><th>Confidence</th><th>Groups</th><th>Rows</th></tr>" + "".join(
        f"<tr><td>{html.escape(r.RESULT)}</td><td>{html.escape(str(_clean(r.CONFIDENCE)))}</td>"
        f"<td style='text-align:right'>{r.G:,}</td><td style='text-align:right'>{int(r.N):,}</td></tr>"
        for r in s.itertuples()) + "</table>"
    if not by_concept:      # not-decidable list: no confidence column
        r2 = groups.groupby("RESULT").agg(G=("GROUP_ID", "size"), N=("N_ROWS", "sum")).reset_index() \
            .sort_values("N", ascending=False)
        return "<table><tr><th>Result</th><th>Groups</th><th>Rows</th></tr>" + "".join(
            f"<tr><td>{html.escape(r.RESULT)}</td><td style='text-align:right'>{r.G:,}</td>"
            f"<td style='text-align:right'>{int(r.N):,}</td></tr>" for r in r2.itertuples()) + "</table>"
    c = groups.groupby("CONCEPT_CASE").agg(G=("GROUP_ID", "size"), N=("N_ROWS", "sum")).reset_index() \
        .sort_values("N", ascending=False)
    t2 = "<table><tr><th>Concept</th><th>Groups</th><th>Rows</th></tr>" + "".join(
        f"<tr><td>{html.escape(r.CONCEPT_CASE)}</td><td style='text-align:right'>{r.G:,}</td>"
        f"<td style='text-align:right'>{int(r.N):,}</td></tr>" for r in c.itertuples()) + "</table>"
    return t1 + t2


# ---------------------------------------------------------------------------
# Group page
# ---------------------------------------------------------------------------

_GROUP = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>__TITLE__</title>__CDN____STYLE__
</head>
<body>
  __NAV__
  __MOCK__
  <div class="d-flex flex-wrap align-items-center gap-2 mb-2">
    <h3 class="mb-0 me-2">__TEST__ <span class="text-muted">·</span> __UNIT__</h3>
    <span class="badge __RESULT_CLASS__ fs-6">__RESULT__</span>
    __CONF_BADGE__
  </div>
  <div class="fix-line mb-3">__FIX_LINE__</div>

  <div class="row g-3 mb-3">
    <div class="col-auto"><div class="stat-label">Rows</div><div class="stat-value">__N_ROWS__</div></div>
    <div class="col-auto"><div class="stat-label">Concept</div><div class="stat-value">__CONCEPT__</div></div>
    <div class="col-auto"><div class="stat-label">Concept unit</div><div class="stat-value">__TARGET_UNIT__</div></div>
    <div class="col-auto"><div class="stat-label">Harmonized rows</div><div class="stat-value">__N_TARGET__</div></div>
    <div class="col-auto"><div class="stat-label">Rows with a range</div><div class="stat-value">__COVERAGE__</div></div>
    <div class="col-auto"><div class="stat-label">Current rule</div><div class="stat-value">__RULE__</div></div>
  </div>

  <div class="mb-2">
    <span class="case-chip"><b>Concept</b>__CONCEPT_CASE__</span>
    <span class="case-chip"><b>Unit problem</b>__UNIT_CASE__</span>
    <span class="case-chip"><b>Usagi pair</b>__PAIR_STATUS__</span>
    __NEEDS_CHIP__
  </div>
  <p style="max-width:60rem">__REASON__</p>

  <div class="section-box d-flex flex-wrap align-items-center gap-3">
    <div><div class="stat-label">Review</div><div id="issue-status">__ISSUE_STATUS__</div></div>
    <a id="issue-btn" class="btn btn-outline-primary btn-sm" href="__ISSUE_URL__" target="_blank" rel="noopener">__ISSUE_BTN__</a>
    <small class="text-muted">Issue title <code>__ISSUE_TITLE__</code>. Status as of __ISSUES_AT__; checked again live
    when the page opens.</small>
  </div>

  <div class="section-box">
    <h5>Candidate units</h5>
    <p class="text-muted mb-2">Every unit that converts into the concept's unit. Units with the same factor are one
    candidate. Click a row to show it in the figures below.</p>
    <div style="overflow-x:auto"><table class="table table-sm cands mb-0" id="cands"></table></div>
  </div>

  <div class="section-box">
    <h5>Values <small class="text-muted" id="values-sub"></small></h5>
    <p class="text-muted mb-2">The values of the unharmonized rows (blue) against the concept's harmonized values converted to the
    selected unit (orange), as percentiles P5 to P95 only: the percentiles themselves, bars between neighbouring
    percentiles (each holds 5% of the rows) with a smooth line through them, and a P5 · P25 · median · P75 · P95 box
    with the mean. Green bands: the concept's typical reference ranges in the selected unit.</p>
    <div class="row g-2" id="values-row">
      <div class="col-lg-4"><div id="chart-ecdf" class="chart"></div></div>
      <div class="col-lg-4"><div id="chart-lin" class="chart"></div></div>
      <div class="col-lg-4"><div id="chart-log" class="chart"></div></div>
    </div>
    <p class="small text-muted mb-0" id="values-note"></p>
  </div>

  <div class="section-box">
    <h5>Reference ranges</h5>
    <p class="text-muted mb-2">Each bar is a reference range, from its lower to its upper limit, on the scale the lab
    reported. Top: the concept's typical ranges, converted to each candidate unit (green: the selected one). Below: the
    most common ranges of the unharmonized rows (this test · unit), with the number of rows. If the blue bars line up with the green one, the
    ranges agree with that unit. A range with only an upper limit starts at the left edge; one with only a lower limit
    runs to the right edge.</p>
    <div id="chart-ranges"></div>
  </div>

  <div class="section-box">
    <h5>Context</h5>
    <div class="row g-4">
      <div class="col-lg-9"><div class="stat-label mb-1">Unit history</div><div id="history"></div></div>
    </div>
    <small class="text-muted">How the unit of these rows changed on its way through the pipeline: as sent by the lab
    (<code>source::MEASUREMENT_UNIT</code>), after cleaning (<code>cleaned-pre-inj</code>), after the primary injection
    (<code>cleaned-post-inj</code>); all of them now have <b>__UNIT__</b>. Paths followed by only a few people are not shown, so the counts can add up to less than the group's rows.</small>
  </div>

  <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
  <script>const PAGE = __PAGE__;</script>
  <script src="../assets/portal.js"></script>
</body>
</html>
"""

PORTAL_JS = r"""
(function () {
  const BLUE = 'steelblue', ORANGE = 'darkorange', GREEN = '#1baf7a', GREY = '#adb5bd';
  const P = PAGE, cands = P.candidates, ids = Object.keys(cands);
  let sel = P.picked && cands[P.picked] ? P.picked : ids[0];

  function fmt(x, d) {
    if (x === null || x === undefined || x === '' || Number.isNaN(x)) return '—';
    return typeof x === 'number' ? Number(x.toPrecision(d || 3)).toString() : String(x);
  }
  function pct(x) { return (x === null || x === undefined || x === '') ? '—' : Math.round(100 * x) + '%'; }
  function esc(s) { return String(s == null ? '' : s).replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c])); }

  // ---------- candidates table ----------
  function drawTable() {
    const t = document.getElementById('cands');
    if (!ids.length) { t.outerHTML = '<p class="text-muted mb-0">No candidate units (see the reason above).</p>'; return; }
    let h = '<thead><tr><th>Unit</th><th>Same scale</th><th>Factor</th><th>Concept</th><th>KS</th><th>T</th>' +
            '<th>MAD</th><th>Values</th><th>Range match</th><th>Only this unit</th><th></th></tr></thead><tbody>';
    ids.forEach(id => {
      const c = cands[id], e = P.cand_rows[id] || {};
      const v = e.VALUES === 'PASS' ? '<span class="text-success fw-semibold">PASS</span> (' + esc(e.VALUES_DECIDED_BY) + ')'
                                    : '<span class="text-danger fw-semibold">FAIL</span>';
      h += '<tr data-id="' + esc(id) + '"' + (id === sel ? ' class="sel"' : '') + '><td><b>' + esc(c.unit) + '</b></td><td>' +
           esc((c.same || []).join(', ')) + '</td><td>× ' + fmt(c.cf, 4) + ' → ' + esc(c.target_unit) + '</td><td>' + esc(c.omop) +
           '</td><td>' + fmt(e.KS_STAT) + (e.KS_PASS ? ' ✓' : '') + '</td><td>' + (e.T_PASS ? '✓' : '✗') + '</td><td>' +
           fmt(e.MAD_DIST) + ' / ' + fmt(e.MAD_THRESHOLD) + (e.MAD_PASS ? ' ✓' : '') + '</td><td>' + v + '</td><td>' +
           pct(e.RANGE_MATCH) + '</td><td>' + pct(e.RANGE_ASSIGNED) + '</td><td>' + (e.PICKED ? '<span class="badge bg-success">suggested</span>' : '') +
           '</td></tr>';
    });
    t.innerHTML = h + '</tbody>';
    t.querySelectorAll('tbody tr').forEach(tr => tr.addEventListener('click', () => { sel = tr.dataset.id; drawAll(); }));
  }

  // concept typical ranges converted into the selected candidate's (reported) unit
  function bandsFor(id) {
    const c = cands[id]; if (!c) return [];
    const con = P.concepts[c.omop] || {};
    return (con.clusters || []).map(k => ({ lo: k.LOWER == null ? null : k.LOWER / c.cf,
                                            hi: k.UPPER == null ? null : k.UPPER / c.cf, share: k.SHARE }));
  }

  // ---------- values ----------
  // Only what the plot data publishes per side (scripts/common/disclosure.py): rows and people,
  // the mean and the percentiles P5..P95, plus the test results. Nothing is drawn from raw values.
  const PCTS = P.pcts || [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95];
  const allPos = arrs => arrs.every(a => a.every(v => v > 0));
  const normPdf = (x, m, s) => Math.exp(-0.5 * Math.pow((x - m) / s, 2)) / (s * Math.sqrt(2 * Math.PI));
  // bars between neighbouring percentiles: each holds 5% of the rows (height = share / width)
  function equalMassBars(p, tf) {
    const xs = p.map(tf), bars = [];
    let start = 0;
    for (let i = 1; i < xs.length; i++) {
      if (xs[i] > xs[start]) {
        bars.push({ x0: xs[start], x1: xs[i], mass: 0.05 * (i - start), h: 0.05 * (i - start) / (xs[i] - xs[start]),
                    label: 'P' + PCTS[start] + '–P' + PCTS[i] });
        start = i;
      }
    }
    return bars;
  }
  // smooth line through the bars: each bar's share spread as a normal curve of half its width
  // (at least 1/25 of the range)
  function smoothBars(bars) {
    const lo = bars[0].x0, hi = bars[bars.length - 1].x1, x = [], y = [];
    for (let i = 0; i <= 200; i++) {
      const v = lo + (hi - lo) * i / 200;
      x.push(v);
      y.push(bars.reduce((s, b) => s + b.mass * normPdf(v, (b.x0 + b.x1) / 2, Math.max((b.x1 - b.x0) / 2, (hi - lo) / 25, 1e-12)), 0));
    }
    return { x, y };
  }
  function sideTxt(name, s) {
    if (!s || s.n_subjects === undefined) return '';
    if (s.n_subjects === 'BELOW_THRESHOLD') return name + ': too few people to describe';
    return name + ': ' + s.n_records.toLocaleString('en-US') + ' rows · ' + s.n_subjects.toLocaleString('en-US') +
           ' people' + (s.pct ? '' : ' (distribution not shown: too few people)');
  }

  function drawValues() {
    const c = cands[sel], pd = c && c.plot;
    const sub = document.getElementById('values-sub'), noteEl = document.getElementById('values-note');
    const sides = pd ? [['unharmonized rows', pd.candidate || {}, BLUE], ['harmonized', pd.target || {}, ORANGE]]
                     : [['unharmonized rows', P.group || {}, BLUE]];
    sub.textContent = pd ? '— candidate ' + c.unit + ' (× ' + fmt(c.cf, 4) + ' → ' + c.target_unit + ')'
                         : '— no candidate to compare';
    noteEl.textContent = sides.map(s => sideTxt(s[0], s[1])).filter(Boolean).join(' · ');
    const shown = sides.filter(s => s[1].pct);
    ['chart-ecdf', 'chart-lin', 'chart-log'].forEach(d => Plotly.purge(d));
    if (!shown.length) return;
    const useLog = allPos(shown.map(s => s[1].pct));
    const tf = useLog ? (v => Math.log10(v)) : (v => v);
    const lay = (xt, yt, extra) => Object.assign({ xaxis: { title: xt }, yaxis: { title: yt },
      margin: { t: 30, b: 50, l: 55, r: 10 }, showlegend: false }, extra || {});
    const note = (txt) => [{ xref: 'paper', yref: 'paper', x: 0.98, y: 0.98, xanchor: 'right', yanchor: 'top', text: txt,
      showarrow: false, align: 'left', bgcolor: 'rgba(255,255,200,0.85)', bordercolor: '#ccc', borderwidth: 1, font: { size: 10 } }];
    const t = (pd && pd.tests) || {};

    // 1. percentiles
    Plotly.newPlot('chart-ecdf', shown.map(s => ({ x: PCTS, y: s[1].pct, mode: 'lines+markers', name: s[0],
        line: { color: s[2] }, marker: { size: 4 }, hovertemplate: 'P%{x}: %{y}<extra>' + s[0] + '</extra>' })),
      lay('percentile', 'value', { yaxis: { title: 'value', type: useLog ? 'log' : 'linear' },
        title: { text: 'Percentiles', font: { size: 12 } }, showlegend: true,
        legend: { x: 0, y: 1, bgcolor: 'rgba(255,255,255,0.7)' },
        annotations: t.ks ? note('KS=' + t.ks.stat.toFixed(3) + '<br>→ ' + (t.ks.passed ? 'PASS' : 'FAIL')) : [] }),
      { responsive: true });

    // 2. distribution: equal-mass bars + smooth line, with the typical ranges of the selected unit
    const dist = [];
    shown.forEach(s => {
      const bars = equalMassBars(s[1].pct, tf);
      if (!bars.length) return;
      dist.push({ type: 'bar', x: bars.map(b => (b.x0 + b.x1) / 2), width: bars.map(b => b.x1 - b.x0),
                  y: bars.map(b => b.h), customdata: bars.map(b => b.label),
                  hovertemplate: '%{customdata}<extra>' + s[0] + '</extra>', marker: { color: s[2], opacity: 0.3 } });
      const sm = smoothBars(bars);
      dist.push({ x: sm.x, y: sm.y, mode: 'lines', line: { color: s[2], width: 2 }, hoverinfo: 'skip' });
    });
    const shapes = [];
    (pd ? bandsFor(sel) : []).forEach(b => {
      const ok = v => v != null && (!useLog || v > 0);
      if (ok(b.lo) && ok(b.hi)) {
        shapes.push({ type: 'rect', xref: 'x', yref: 'paper', x0: tf(b.lo), x1: tf(b.hi), y0: 0, y1: 1,
                      fillcolor: 'rgba(27,175,122,0.13)', line: { width: 0 }, layer: 'below' });
      } else if (ok(b.lo) || ok(b.hi)) {          // one-sided typical range: a line at the limit it has
        const lim = tf(ok(b.lo) ? b.lo : b.hi);
        shapes.push({ type: 'line', xref: 'x', yref: 'paper', x0: lim, x1: lim, y0: 0, y1: 1,
                      line: { color: GREEN, width: 2, dash: 'dash' } });
      }
    });
    Plotly.newPlot('chart-lin', dist, lay(useLog ? 'log₁₀(value)' : 'value', 'density',
      { barmode: 'overlay', bargap: 0, shapes: shapes, title: { text: 'Distribution (P5–P95)', font: { size: 12 } },
        annotations: t.t ? note('t=' + t.t.stat.toFixed(2) + '<br>→ ' + (t.t.passed ? 'PASS' : 'FAIL')) : [] }),
      { responsive: true });

    // 3. summary: P5 · P25 · median · P75 · P95 and the mean, with the MAD band
    const boxes = shown.map(s => {
      const p = s[1].pct;
      return { type: 'box', orientation: 'h', name: s[0], y: [s[0]], lowerfence: [p[0]], q1: [p[4]], median: [p[9]],
               q3: [p[14]], upperfence: [p[18]], mean: s[1].mean != null ? [s[1].mean] : undefined,
               boxmean: s[1].mean != null, boxpoints: false, marker: { color: s[2] }, line: { color: s[2] },
               hoverinfo: 'x' };
    });
    const m = t.mad, bshapes = [];
    if (m && m.t_median != null && m.threshold != null && shown.length === 2) {
      let lo = m.t_median - m.threshold;
      if (useLog && lo <= 0) lo = Math.min(shown[0][1].pct[0], shown[1][1].pct[0]);
      bshapes.push({ type: 'rect', xref: 'x', yref: 'paper', x0: lo, x1: m.t_median + m.threshold, y0: 0, y1: 1,
                     layer: 'below', line: { width: 0 },
                     fillcolor: m.passed ? 'rgba(50,205,50,0.15)' : 'rgba(250,128,114,0.18)' });
    }
    Plotly.newPlot('chart-log', boxes, lay('value', '', { xaxis: { title: 'value', type: useLog ? 'log' : 'linear' },
      yaxis: { automargin: true }, shapes: bshapes, title: { text: 'Summary', font: { size: 12 } },
      annotations: m ? note('MAD dist=' + fmt(m.distance) + '<br>thr=' + fmt(m.threshold) + '<br>→ ' +
                            (m.passed ? 'PASS' : 'FAIL')) : [] }), { responsive: true });
  }

  // ---------- ranges ----------
  const MAX_BARS = 20;
  function drawRanges() {
    const el = document.getElementById('chart-ranges');
    // placeholder ranges (a limit of 0 or below, lower ≥ upper) are ignored, as in the range evidence
    const okLim = v => v == null || v > 0;
    const allPairs = (P.ranges || []).slice().sort((a, b) => b[2] - a[2]);
    const pairs = allPairs.filter(p => okLim(p[0]) && okLim(p[1]) && (p[0] == null || p[1] == null || p[0] < p[1])
                                       && (p[0] != null || p[1] != null));
    const skipped = allPairs.length - pairs.length;
    if (!pairs.length) {
      Plotly.purge(el);
      el.innerHTML = '<p class="text-muted p-3">No reference range is common enough to show.</p>';
      return;
    }
    if (!el.data) el.innerHTML = '';
    const shown = pairs.slice(0, MAX_BARS);
    // scale: the group's limits and the selected unit's typical ranges
    const lims = [];
    shown.forEach(p => { if (p[0] != null) lims.push(p[0]); if (p[1] != null) lims.push(p[1]); });
    bandsFor(sel).forEach(b => { if (b.lo != null) lims.push(b.lo); if (b.hi != null) lims.push(b.hi); });
    // fit the axis to the limits shown: from the smallest to the largest, with a margin; start at 0
    // only when the values are close to 0 anyway (smallest limit below the spread)
    let lo0 = Math.min.apply(null, lims), hi0 = Math.max.apply(null, lims);
    const spread = hi0 - lo0 || Math.abs(hi0) * 0.1 || 1;
    const xmin = lo0 < spread ? 0 : lo0 - 0.15 * spread;
    const xmax = hi0 + 0.15 * spread;

    const rows = [], off = [];        // rows: {label, lo, hi, color, text, hover}
    ids.forEach(id => {
      const c = cands[id], con = P.concepts[c.omop] || {}, isSel = id === sel;
      (con.clusters || []).forEach((k, j) => {
        const lo = k.LOWER == null ? null : k.LOWER / c.cf, hi = k.UPPER == null ? null : k.UPPER / c.cf;
        const name = 'typical · ' + c.unit + (j ? ' (' + (j + 1) + ')' : '');
        const ref = lo != null ? lo : hi;
        const far = (lo != null && lo > xmax) || (hi != null && hi < xmin) || (hi != null && hi < xmax / 1000);
        if (far && !isSel) { off.push(c.unit + ' ' + rangeTxt(lo, hi)); return; }
        rows.push({ label: name, lo: lo, hi: hi, color: isSel ? GREEN : GREY, text: rangeTxt(lo, hi),
                    hover: name + ': ' + rangeTxt(lo, hi) + '<br>' + Math.round(100 * k.SHARE) + '% of harmonized ranges' });
      });
    });
    const nTyp = rows.length;
    shown.forEach(p => {
      rows.push({ label: rangeTxt(p[0], p[1]), lo: p[0], hi: p[1], color: 'rgba(70,130,180,0.75)',
                  text: p[2].toLocaleString('en-US') + ' rows',
                  hover: rangeTxt(p[0], p[1]) + '<br>' + p[2].toLocaleString('en-US') + ' rows' });
    });
    const base = rows.map(r => r.lo == null ? xmin : r.lo);
    const width = rows.map(r => (r.hi == null ? xmax : r.hi) - (r.lo == null ? xmin : r.lo));
    const ylab = rows.map((r, i) => (i < nTyp ? '' : '') + r.label);
    const trace = { type: 'bar', orientation: 'h', base: base, x: width, y: rows.map((r, i) => i),
      marker: { color: rows.map(r => r.color) }, text: rows.map(r => r.text), textposition: 'outside',
      cliponaxis: false, hovertext: rows.map(r => r.hover), hoverinfo: 'text', width: 0.6 };
    const more = pairs.length - shown.length;
    const title = [off.length ? 'off this scale: ' + off.join(', ') : '',
                   more > 0 ? (more + ' less common range' + (more > 1 ? 's' : '') + ' not shown') : '',
                   skipped > 0 ? (skipped + ' placeholder range' + (skipped > 1 ? 's' : '') + ' (0 or lower ≥ upper) left out') : '']
                  .filter(Boolean).join(' · ');
    Plotly.newPlot(el, [trace], {
      height: 90 + 26 * rows.length,
      xaxis: { title: 'value (as reported)', range: [xmin, xmax], zeroline: xmin === 0 },
      yaxis: { tickvals: rows.map((r, i) => i), ticktext: ylab, autorange: 'reversed', automargin: true },
      shapes: nTyp && rows.length > nTyp ? [{ type: 'line', xref: 'paper', x0: 0, x1: 1, y0: nTyp - 0.5, y1: nTyp - 0.5,
                                               line: { color: '#adb5bd', width: 1, dash: 'dot' } }] : [],
      title: title ? { text: title, font: { size: 10, color: '#6c757d' }, x: 0.99, xanchor: 'right' } : undefined,
      margin: { t: title ? 40 : 15, b: 45, l: 10, r: 70 }, showlegend: false, bargap: 0.3 }, { responsive: true });
  }
  function rangeTxt(lo, hi) {
    const f = v => Number(v.toPrecision(6)).toString();    // limits as stored, not rounded
    if (lo == null && hi == null) return '—';
    if (lo == null) return '≤ ' + f(hi);
    if (hi == null) return '≥ ' + f(lo);
    return f(lo) + '–' + f(hi);
  }

  // ---------- context ----------
  function drawContext() {
    const s = P.context || {};
    const el = document.getElementById('history');
    if (!s.history) { el.innerHTML = '<p class="text-muted">Not available (rebuild the suggestions).</p>'; return; }
    let h = '<table class="table table-sm"><thead><tr><th>Sent by the lab</th><th></th><th>After cleaning</th><th></th>' +
            '<th>After primary injection</th><th>What happened</th><th class="text-end">Rows</th></tr></thead><tbody>';
    s.history.forEach(r => {
      h += '<tr><td><code>' + esc(r[0]) + '</code></td><td class="text-muted">→</td><td><code>' + esc(r[1]) +
           '</code></td><td class="text-muted">→</td><td><code>' + esc(r[2]) + '</code></td><td>' + esc(r[3]) +
           '</td><td class="text-end">' + r[4].toLocaleString('en-US') + '</td></tr>';
    });
    el.innerHTML = h + '</tbody></table>';
  }

  // ---------- live issue status ----------
  function liveIssue() {
    if (!P.repo) return;
    const q = 'repo:' + P.repo + ' is:issue in:title "' + P.issue_title + '"';
    fetch('https://api.github.com/search/issues?q=' + encodeURIComponent(q))
      .then(r => r.ok ? r.json() : null)
      .then(j => {
        if (!j || !j.items) return;
        const it = j.items.find(i => i.title === P.issue_title);
        if (!it) return;
        document.getElementById('issue-status').innerHTML = '<a class="badge ' + (it.state === 'open' ? 'issue-open' : 'issue-closed') +
          '" href="' + it.html_url + '">#' + it.number + ' ' + it.state + '</a>';
        const b = document.getElementById('issue-btn');
        b.href = it.html_url; b.textContent = 'Go to issue #' + it.number;
      })
      .catch(() => {});
  }

  function drawAll() { drawTable(); drawValues(); drawRanges(); }
  drawAll(); drawContext(); liveIssue();
})();
"""


def _status_html(issue, note=""):
    if not issue:
        return '<span class="badge issue-none">no issue yet</span>'
    cls = "issue-open" if issue["state"] == "open" else "issue-closed"
    out = f'<a class="badge {cls}" href="{issue["url"]}">#{issue["number"]} {issue["state"]}</a>'
    return out + (f' <span class="badge bg-warning text-dark">{html.escape(note)}</span>' if note else "")


def build(out_dir: Path, data_dir: Path, repo: str, site_url: str, issues: dict, issues_at: str,
          mock_note: str = "", fig_base: str = "") -> bool:
    """Write the portal into out_dir. Returns False (and writes nothing) if there is no data."""
    groups_path, plot_path = data_dir / "groups.tsv", data_dir / "plot_data.json.gz"
    if not groups_path.exists() or not plot_path.exists():
        print(f"Skipping unharmonized portal: no {groups_path} / {plot_path}")
        return False
    groups = pd.read_csv(groups_path, sep="\t", keep_default_na=False, na_values=[""],
                         dtype={"OMOP_ID": str, "SUGGESTION_ID": str, "FIX_UNIT": str, "TARGET_UNIT": str})
    cands = pd.read_csv(data_dir / "candidates.tsv", sep="\t", keep_default_na=False, na_values=[""],
                        dtype={"OMOP_ID": str, "CAND_ID": str}) \
        if (data_dir / "candidates.tsv").exists() else pd.DataFrame(columns=["GROUP_ID", "CAND_ID"])
    with gzip.open(plot_path, "rb") as fh:
        plots = json.loads(fh.read())
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "groups").mkdir(exist_ok=True)
    for stale in (out_dir / "groups").glob("*.html"):   # pages of an earlier build (ids may have changed)
        stale.unlink()
    (out_dir / "assets").mkdir(exist_ok=True)
    (out_dir / "assets" / "portal.js").write_text(PORTAL_JS, encoding="utf-8")
    for f in ("proposed_omop_injection.tsv", "groups.tsv", "decidable.tsv", "not_decidable.tsv"):
        if (data_dir / f).exists():
            shutil.copy2(data_dir / f, out_dir / f)
    mock = f'<div class="mock">{html.escape(mock_note)}</div>' if mock_note else ""
    base = site_url.rstrip("/") + "/unharmonized/groups/"

    rows_out = []
    for _, r in groups.iterrows():
        gid = r["GROUP_ID"]
        issue = issues.get(r["ISSUE_TITLE"])
        note = issue_note(issue, r["SUGGESTION_ID"])
        page_url = base + gid + ".html"
        plot = plots.get(gid, {})
        crow = {c["CAND_ID"]: {k: _clean(v) for k, v in c.items()}
                for c in cands[cands["GROUP_ID"] == gid].to_dict(orient="records")}
        for c in crow.values():
            c["PICKED"] = str(c.get("PICKED")) == "True"
            for k in ("KS_PASS", "T_PASS", "MAD_PASS"):
                c[k] = str(c.get(k)) == "True"
        fig_url = f"{fig_base}{r['SUGGESTION_ID']}.png" if r.get("DECIDABLE", "yes") != "no" else ""
        new_issue = issue_url(repo, r, page_url, fig_url, crow,
                              plot.get("context", {}).get("history", []), plot)
        url = issue["url"] if issue and not note.startswith("suggestion changed") else new_issue
        page = dict(candidates=plot.get("candidates", {}), concepts=plot.get("concepts", {}),
                    ranges=plot.get("ranges", []), group=plot.get("group"), pcts=list(disclosure.PCTS),
                    context=plot.get("context", {}), picked=plot.get("picked"), cand_rows=crow,
                    margin=1.5, repo=repo, issue_title=r["ISSUE_TITLE"])
        fix = "No fix suggested"
        if r["RESULT"] == "Suggest fix":
            fix = (f"Fix: {html.escape(r['SOURCE_UNIT'])} → {html.escape(r['FIX_UNIT'])} "
                   f"<span class='text-muted fw-normal'>(× {r['CONVERSION_FACTOR']:g} into "
                   f"{html.escape(str(r['TARGET_UNIT']))})</span>")
        elif r["RESULT"] == USAGI_RESULT:
            fix = (f"Approve in Usagi: ({html.escape(r['TEST_NAME'])}, {html.escape(r['SOURCE_UNIT'])}) → "
                   f"{html.escape(str(_clean(r['OMOP_ID'])))} <span class='text-muted fw-normal'>(× "
                   f"{r['CONVERSION_FACTOR']:g} into {html.escape(str(r['TARGET_UNIT']))}, already in the "
                   f"conversion table)</span>")
        conf = _clean(r["CONFIDENCE"])
        concept = f"{_clean(r['OMOP_ID'])}" + (f" <span class='fw-normal text-muted'>{html.escape(str(r['OMOP_NAME']))}</span>"
                                               if _clean(r["OMOP_NAME"]) else "")
        needs = _clean(r["NEEDS"])
        page_html = (_GROUP
                     .replace("__CDN__", CDN).replace("__STYLE__", STYLE)
                     .replace("__NAV__", _nav("../", "")).replace("__MOCK__", mock)
                     .replace("__TITLE__", html.escape(f"{r['TEST_NAME']} · {r['SOURCE_UNIT']}"))
                     .replace("__TEST__", html.escape(r["TEST_NAME"])).replace("__UNIT__", html.escape(r["SOURCE_UNIT"]))
                     .replace("__RESULT_CLASS__", RESULT_CLASS.get(r["RESULT"], "res-none"))
                     .replace("__RESULT__", html.escape(r["RESULT"]))
                     .replace("__CONF_BADGE__", f'<span class="badge conf-{conf} fs-6">{conf} confidence</span>' if conf else "")
                     .replace("__FIX_LINE__", fix)
                     .replace("__N_ROWS__", f"{int(r['N_ROWS']):,}")
                     .replace("__CONCEPT__", concept or "—")
                     .replace("__TARGET_UNIT__", html.escape(str(_clean(r["TARGET_UNIT"]) or "—")))
                     .replace("__N_TARGET__", _count(r["N_TARGET"]))
                     .replace("__COVERAGE__", f"{100 * float(r['RANGE_COVERAGE']):.0f}%"
                              if _clean(r["RANGE_COVERAGE"]) != "" else "—")
                     .replace("__RULE__", html.escape(str(_clean(r["CURRENT_RULE"]) or "none"))
                              + (f" <span class='text-muted fw-normal'>({html.escape(str(r['VS_CURRENT_RULE']))})</span>"
                                 if _clean(r["CURRENT_RULE"]) and _clean(r["VS_CURRENT_RULE"]) else ""))
                     .replace("__CONCEPT_CASE__", html.escape(r["CONCEPT_CASE"]))
                     .replace("__UNIT_CASE__", html.escape(r["UNIT_CASE"]))
                     .replace("__PAIR_STATUS__", html.escape(str(r["USAGI_PAIR_STATUS"]).lower()))
                     .replace("__NEEDS_CHIP__", f'<span class="case-chip"><b>Needs</b>{html.escape(needs)}</span>' if needs else "")
                     .replace("__REASON__", html.escape(r["REASON"]))
                     .replace("__ISSUE_STATUS__", _status_html(issue, note))
                     .replace("__ISSUE_URL__", html.escape(url))
                     .replace("__ISSUE_BTN__", "Open a new review issue" if note.startswith("suggestion changed")
                              else (f"Go to issue #{issue['number']}" if issue else "Open review issue"))
                     .replace("__ISSUE_TITLE__", html.escape(r["ISSUE_TITLE"]))
                     .replace("__ISSUES_AT__", html.escape(issues_at))
                     .replace("__MARGIN__", "1.5")
                     .replace("__PAGE__", _json(page)))
        (out_dir / "groups" / f"{gid}.html").write_text(page_html, encoding="utf-8")
        row = {c[0]: _clean(r.get(c[0])) for c in COLS_DEC + COLS_ND if c[0] in groups.columns}
        row["DECIDABLE"] = _clean(r.get("DECIDABLE")) or "yes"
        row.update(GROUP_ID=gid, ISSUE="", ISSUE_NUMBER=issue["number"] if issue else "",
                   ISSUE_STATE=issue["state"] if issue else "", ISSUE_URL=issue["url"] if issue else "",
                   ISSUE_NOTE=note,
                   # a "new issue" link right in the overview, for clear-cut suggestions only
                   ISSUE_NEW=new_issue if r["RESULT"] in ("Suggest fix", USAGI_RESULT)
                   and (not issue or note.startswith("suggestion changed")) else "")
        row["RESULT_LINK"] = ((row["ISSUE_NEW"] or (issue["url"] if issue else ""))
                              if r["RESULT"] in ("Suggest fix", USAGI_RESULT) else "")
        if row.get("CONVERSION_FACTOR") != "":
            row["CONVERSION_FACTOR"] = f"{float(row['CONVERSION_FACTOR']):g}"
        rows_out.append(row)

    current = set(groups["ISSUE_TITLE"])
    gone = []
    for title, it in sorted(issues.items(), key=lambda kv: -kv[1]["number"]):
        if title in current or not title.startswith("[unharm] "):
            continue
        test, _, unit = title[len("[unharm] "):].partition(" · ")
        gone.append({"number": it["number"], "state": it["state"], "url": it["url"], "test": test, "unit": unit,
                     "note": "resolved: no longer failing" if it["state"] == "closed"
                             else "no longer failing, but the issue is still open"})
    dec = [r for r in rows_out if r["DECIDABLE"] != "no"]
    nd = [r for r in rows_out if r["DECIDABLE"] == "no"]
    gdec = groups[groups.get("DECIDABLE", pd.Series("yes", index=groups.index)).fillna("yes") != "no"]
    gnd = groups.drop(gdec.index)
    lists = {"dec": {"rows": dec, "cols": [list(c) for c in COLS_DEC]},
             "nd": {"rows": nd, "cols": [list(c) for c in COLS_ND]}}
    idx = (_INDEX.replace("__CDN__", CDN).replace("__STYLE__", STYLE)
           .replace("__NAV__", _nav("", "Unharmonized")).replace("__MOCK__", mock)
           .replace("__N_DEC__", str(len(dec))).replace("__N_ND__", str(len(nd)))
           .replace("__N_GONE__", str(len(gone))).replace("__TABLE_GONE__", _gone_html(gone))
           .replace("__SUMMARY_DEC__", _summary_html(gdec)).replace("__SUMMARY_ND__", _summary_html(gnd, False))
           .replace("__TABLE_DEC__", _table_html("dec", COLS_DEC)).replace("__TABLE_ND__", _table_html("nd", COLS_ND))
           .replace("__ISSUES_AT__", html.escape(issues_at))
           .replace("__LISTS__", _json(lists))
           .replace("__RESULT_CLASS__", _json(RESULT_CLASS)))
    (out_dir / "index.html").write_text(idx, encoding="utf-8")
    print(f"Wrote {out_dir}/index.html and {len(groups)} group pages")
    return True
