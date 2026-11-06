#!/usr/bin/env python3
"""
primary.py

Pages for the primary (test-name) injection results: index.html, doc.html and tests/*.html.
Normally run through scripts/site/build.py, which also builds the unharmonized portal.

Usage (this section only):
  python3 scripts/site/primary.py [--out-dir PATH]

Reads from {out_dir}/data/ and {out_dir}/plots/, writes index.html and doc.html
to {out_dir}/.  Pass --out-dir to match what you used with test_name_injection.py.
"""

import argparse
import gzip
import os
import re
import json
import shutil
from pathlib import Path

import sys

import markdown
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))     # scripts/
from common import disclosure   # noqa: E402


# ---------------------------------------------------------------------------
# Column label overrides and preferred display order.
# Columns not listed here are appended after these in their original TSV order.
# ---------------------------------------------------------------------------

COL_LABELS = {
    "TEST_NAME":        "Test name",
    "TYPE":             "Type",
    "CUTOFF":           "Cutoff",
    "UNIT":             "Unit",
    "OUTCOME":          "Outcome",
    "NOTES":            "Notes",
    "TESTS_PASSED":     "Tests passed",
    "UNIT_PREVALENCE":  "Unit prev (%)",
    "N_CANDIDATE":      "N cand",
    "N_TARGET":         "N target",
    "N_CANDIDATE_SUBJECTS": "People cand",
    "N_TARGET_SUBJECTS":    "People target",
    "KS_STAT":          "KS stat",
    "KS_MLOGP":         "KS −log10p",
    "KS_PASS":          "KS pass",
    "T_STAT":           "T stat",
    "T_MLOGP":          "T −log10p",
    "T_PASS":           "T pass",
    "MAD_DIST":         "MAD dist",
    "MAD_THRESHOLD":    "MAD thr",
    "MAD_PASS":         "MAD pass",
    "BIMODAL_OVERLAP":  "Overlap (%)",
}

COL_ORDER = list(COL_LABELS.keys())

# Pixel widths for columns that are inherently short
COL_WIDTHS = {
    "TYPE":            80,
    "CUTOFF":          80,
    "OUTCOME":         75,
    "KS_PASS":         70,
    "T_PASS":          70,
    "MAD_PASS":        70,
    "KS_STAT":         75,
    "KS_MLOGP":        85,
    "T_STAT":          75,
    "T_MLOGP":         85,
    "MAD_DIST":        80,
    "MAD_THRESHOLD":   80,
    "UNIT_PREVALENCE": 90,
    "BIMODAL_OVERLAP": 85,
    "TESTS_PASSED":    80,
    "N_CANDIDATE":     85,
    "N_TARGET":        75,
    "N_CANDIDATE_SUBJECTS": 85,
    "N_TARGET_SUBJECTS":    85,
    "SUB_DIST":        70,
}

# Columns to drop entirely from the landing table (reserved for per-test pages)
EXCLUDE_COLS = {
    "SUB_DIST",
    "BIMODAL_STATUS", "BIMODAL_SEP", "BIMODAL_BC", "BIMODAL_DIP_P",
    "SCORE_GLOBAL", "SCORE_SPLIT", "SCORE_IMPROVEMENT",
    "CAND_DECILES", "TARG_DECILES",
    "PREVALENCE_DICT",
}


# ---------------------------------------------------------------------------
# HTML templates  (use __MARKER__ substitution to avoid Python/JS brace clash)
# ---------------------------------------------------------------------------

_INDEX = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Unit injection results</title>
  <link rel="stylesheet"
    href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css">
  <link rel="stylesheet"
    href="https://cdn.datatables.net/1.13.8/css/dataTables.bootstrap5.min.css">
  <style>
    body     { padding: 2rem; font-size: 0.9rem; }
    nav a    { margin-right: 1.5rem; font-weight: 500; }
    .scatter { text-align: center; margin-bottom: 2rem; }
    .scatter img { max-width: 100%; max-height: 480px; }
    .pass { color: #198754; font-weight: 600; }
    .fail { color: #dc3545; font-weight: 600; }
    .skip { color: #6c757d; }
    small.hint { color: #6c757d; font-size: 0.8rem; }
    tfoot input, thead tr.search-row input {
                  width: 100%; font-size: 0.78rem; padding: 2px 4px;
                  border: 1px solid #ced4da; border-radius: 3px; box-sizing: border-box; }
    #results td, #results th { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    #results td.col-wide, #results th.col-wide { white-space: normal; min-width: 160px; }
    .summary-tables { margin-bottom: 2rem; }
    .summary-tables h4 { margin-top: 1.2rem; font-size: 1rem; font-weight: 600; }
    .summary-tables table { border-collapse: collapse; font-size: 0.85rem; margin-bottom: 0.5rem; }
    .summary-tables th, .summary-tables td { border: 1px solid #dee2e6; padding: 0.25rem 0.6rem; }
    .summary-tables th { background: #f8f9fa; }
    .summary-tables pre { background: #f8f9fa; padding: 0.75rem; border-radius: 4px;
                           font-size: 0.78rem; overflow-x: auto; }
  </style>
</head>
<body>
  <nav class="mb-3">
    <a href="index.html"><b>Results</b></a>
    <a href="doc.html">Documentation</a>
    __NAV_UNHARM__
  </nav>
  <hr>
  <h2>Unit injection results</h2>

  __SCATTER_BLOCK__

  __SUMMARY_SECTION__

  <p><small class="hint">Per-column search: text columns support regex (e.g. <code>^PASS$</code>,
  <code>mmol</code>). Numeric columns support comparisons: <code>&lt;3</code>,
  <code>&gt;=20</code>, <code>!=0</code>. The global box top-right searches all columns with regex.</small></p>

  <div style="overflow-x:auto">
  <table id="results" class="table table-sm table-striped table-bordered" style="width:100%">
    <thead>
      <tr class="search-row">__FOOTERS__</tr>
      <tr>__HEADERS__</tr>
    </thead>
    <tbody></tbody>
  </table>
  </div>

  <script src="https://code.jquery.com/jquery-3.7.0.min.js"></script>
  <script src="https://cdn.datatables.net/1.13.8/js/jquery.dataTables.min.js"></script>
  <script src="https://cdn.datatables.net/1.13.8/js/dataTables.bootstrap5.min.js"></script>
  <script>
  const DATA        = __DATA_JSON__;
  const COLUMNS     = __COLUMNS_JSON__;
  const OUTCOME_COL = __OUTCOME_COL_IDX__;
  const NUMERIC_COLS = __NUMERIC_COLS_JSON__;  // set of column indices with numeric data

  // Per-column filter state
  const colFilters = new Array(COLUMNS.length).fill('');

  // Parse a numeric comparison expression like "<3", ">=20", "!=0"
  function parseNumeric(expr) {
    var m = expr.trim().match(/^([<>]=?|!=?)\s*(-?\d+\.?\d*(?:e[+-]?\d+)?)$/i);
    if (!m) return null;
    var op = m[1], val = parseFloat(m[2]);
    return function (x) {
      var n = parseFloat(x);
      if (isNaN(n)) return false;
      if (op === '<')  return n <  val;
      if (op === '>')  return n >  val;
      if (op === '<=') return n <= val;
      if (op === '>=') return n >= val;
      if (op === '=' || op === '==') return n === val;
      if (op === '!=') return n !== val;
      return true;
    };
  }

  // Global custom filter handles both numeric comparisons and regex
  $.fn.dataTable.ext.search.push(function (settings, data) {
    for (var i = 0; i < colFilters.length; i++) {
      var expr = colFilters[i].trim();
      if (!expr) continue;
      var numFn = NUMERIC_COLS[i] ? parseNumeric(expr) : null;
      if (numFn) {
        if (!numFn(data[i])) return false;
      } else {
        try {
          if (!new RegExp(expr, 'i').test(data[i])) return false;
        } catch (e) { /* invalid regex — ignore */ }
      }
    }
    return true;
  });

  $(document).ready(function () {
    var table = $('#results').DataTable({
      data:       DATA,
      columns:    COLUMNS,
      pageLength: 25,
      autoWidth:  false,
      order:      [[OUTCOME_COL, 'asc']],
      search:     { regex: true, smart: false },
      columnDefs: [{
        targets: OUTCOME_COL,
        render: function (data, type) {
          if (type !== 'display') return data;
          var cls = { PASS: 'pass', FAIL: 'fail', SKIP: 'skip' }[data] || '';
          return '<span class="' + cls + '">' + data + '</span>';
        }
      }]
    });

    // Wire per-column search inputs via event delegation (works with any DOM structure)
    $(document).on('input', '#results thead tr.search-row input', function () {
      var idx = $(this).closest('th').index();
      colFilters[idx] = this.value;
      table.draw();
    });
  });
  </script>
</body>
</html>
"""

_DOC = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Documentation — unit injection</title>
  <link rel="stylesheet"
    href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css">
  <style>
    body    { padding: 2rem; max-width: 960px; }
    nav a   { margin-right: 1.5rem; font-weight: 500; }
    img     { max-width: 100%; }
    table   { border-collapse: collapse; width: 100%; margin: 1rem 0; }
    th, td  { border: 1px solid #dee2e6; padding: 0.4rem 0.75rem; vertical-align: top; }
    th      { background: #f8f9fa; font-weight: 600; }
    pre     { background: #f8f9fa; padding: 1rem; border-radius: 4px; overflow-x: auto; }
    code    { background: #f0f0f0; padding: 0.1rem 0.3rem; border-radius: 3px;
              font-size: 0.85em; }
    pre code { background: none; padding: 0; }
  </style>
</head>
<body>
  <nav class="mb-3">
    <a href="index.html">Results</a>
    <a href="doc.html"><b>Documentation</b></a>
    __NAV_UNHARM__
  </nav>
  <hr>
  __README_HTML__
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Charts shared by the test pages. They only use what the plot data publishes per group (see
# scripts/common/disclosure.py): rows and people, the mean, P5..P95 for large enough groups,
# test results and fitted mixture parameters. Nothing is drawn from raw values.
# ---------------------------------------------------------------------------

_CHART_JS = """
  var BLUE = 'steelblue', ORANGE = 'darkorange';
  var PCTS = __PCTS__;

  function allPos(arrs) { return arrs.every(function (a) { return a.every(function (v) { return v > 0; }); }); }

  // Candidate vs target: percentiles, their ratio, and a P5-P25-P50-P75-P95 box with the mean
  function renderCompare(divId, pd) {
    var el = document.getElementById(divId);
    if (!el || !pd) return;
    var c = pd.candidate || {}, t = pd.target || {};
    var sides = [['candidate', c, BLUE], ['target', t, ORANGE]].filter(function (s) { return s[1].pct; });
    if (!sides.length) {
      el.style.height = 'auto';
      el.innerHTML = '<p class="text-muted">Distributions not shown: the groups are too small.</p>';
      return;
    }
    var useLog = allPos(sides.map(function (s) { return s[1].pct; }));
    var traces = [], shapes = [];
    sides.forEach(function (s) {
      traces.push({x: PCTS, y: s[1].pct, mode: 'lines+markers', name: s[0], line: {color: s[2]},
                   marker: {size: 4}, xaxis: 'x', yaxis: 'y',
                   hovertemplate: 'P%{x}: %{y}<extra>' + s[0] + '</extra>'});
    });
    // Distribution: bars between neighbouring published percentiles, each holding 5% of the rows
    // (height = share / width), in log10 space when all values are positive. Built from the
    // percentiles alone, so it shows nothing beyond them.
    var tf = useLog ? function (v) { return Math.log10(v); } : function (v) { return v; };
    sides.forEach(function (s) {
      var bars = equalMassBars(s[1].pct, tf);
      if (!bars.length) return;
      traces.push({type: 'bar', x: bars.map(function (b) { return (b.x0 + b.x1) / 2; }),
                   width: bars.map(function (b) { return b.x1 - b.x0; }), y: bars.map(function (b) { return b.h; }),
                   customdata: bars.map(function (b) { return b.label; }),
                   hovertemplate: '%{customdata}<extra>' + s[0] + '</extra>',
                   marker: {color: s[2], opacity: 0.3, line: {width: 0}},
                   xaxis: 'x2', yaxis: 'y2', showlegend: false});
      var sm = smoothBars(bars);
      traces.push({x: sm.x, y: sm.y, mode: 'lines', line: {color: s[2], width: 2}, hoverinfo: 'skip',
                   xaxis: 'x2', yaxis: 'y2', showlegend: false});
    });
    sides.forEach(function (s) {
      var p = s[1].pct;
      traces.push({type: 'box', orientation: 'h', name: s[0], y: [s[0]],
                   lowerfence: [p[0]], q1: [p[4]], median: [p[9]], q3: [p[14]], upperfence: [p[18]],
                   mean: s[1].mean != null ? [s[1].mean] : undefined, boxmean: s[1].mean != null,
                   boxpoints: false, marker: {color: s[2]}, line: {color: s[2]},
                   xaxis: 'x3', yaxis: 'y3', showlegend: false, hoverinfo: 'x'});
    });
    var m = pd.tests && pd.tests.mad;
    if (m && m.t_median != null && m.threshold != null && c.pct && t.pct) {
      var lo = m.t_median - m.threshold, hi = m.t_median + m.threshold;
      if (useLog && lo <= 0) lo = Math.min(c.pct[0], t.pct[0]);
      shapes.push({type: 'rect', xref: 'x3', yref: 'y3 domain', x0: lo, x1: hi, y0: 0, y1: 1, layer: 'below',
                   line: {width: 0}, fillcolor: m.passed ? 'rgba(50,205,50,0.15)' : 'rgba(250,128,114,0.18)'});
    }
    Plotly.newPlot(divId, traces, {
      grid: {rows: 1, columns: 3, pattern: 'independent'},
      xaxis: {title: 'percentile'}, yaxis: {title: 'value', type: useLog ? 'log' : 'linear'},
      xaxis2: {title: useLog ? 'log₁₀(value)' : 'value'}, yaxis2: {title: 'density'}, barmode: 'overlay', bargap: 0,
      xaxis3: {title: 'value (P5 · P25 · median · P75 · P95)', type: useLog ? 'log' : 'linear'},
      yaxis3: {automargin: true},
      shapes: shapes, legend: {x: 0, y: 1.15, orientation: 'h'}, margin: {t: 40, b: 50, l: 60, r: 10},
      title: {text: 'Percentiles &nbsp;&nbsp;·&nbsp;&nbsp; Distribution (P5–P95, 5% of rows per bar) &nbsp;&nbsp;·&nbsp;&nbsp; Summary', font: {size: 13}},
    }, {responsive: true});
  }

  // Bimodal check: the fitted two-component mixtures, over the candidate's percentiles drawn as
  // equal-mass bars (each bar holds 5% of the rows between two published percentiles)
  function equalMassBars(p, tf) {
    var xs = p.map(tf), bars = [], start = 0;
    for (var i = 1; i < xs.length; i++) {
      if (xs[i] > xs[start]) {
        bars.push({x0: xs[start], x1: xs[i], mass: 0.05 * (i - start),
                   h: 0.05 * (i - start) / (xs[i] - xs[start]), label: 'P' + PCTS[start] + '–P' + PCTS[i]});
        start = i;
      }
    }
    return bars;
  }

  // Smooth line through the bars: each bar's share spread as a normal curve centred on the bar
  // (sd = half its width, at least 1/25 of the range), drawn over the bars' range only. Uses the
  // bars alone: nothing new.
  function smoothBars(bars) {
    var lo = bars[0].x0, hi = bars[bars.length - 1].x1, xs = [], ys = [];
    for (var i = 0; i <= 200; i++) {
      var x = lo + (hi - lo) * i / 200, y = 0;
      bars.forEach(function (b) { y += b.mass * normPdf(x, (b.x0 + b.x1) / 2, Math.max((b.x1 - b.x0) / 2, (hi - lo) / 25, 1e-12)); });
      xs.push(x); ys.push(y);
    }
    return {x: xs, y: ys};
  }

  function normPdf(x, m, s) { return Math.exp(-0.5 * Math.pow((x - m) / s, 2)) / (s * Math.sqrt(2 * Math.PI)); }

  function renderBimodal(divId, bim) {
    if (!bim || !document.getElementById(divId)) return;
    var pct = bim.candidate && bim.candidate.pct, traces = [], shapes = [];
    [['linear', function (v) { return v; }], ['log', function (v) { return Math.log10(v); }]].forEach(function (sp, col) {
      var f = bim[sp[0]], ax = col === 0 ? '' : '2';
      var bars = (pct && (sp[0] === 'linear' || allPos([pct]))) ? equalMassBars(pct, sp[1]) : [];
      if (bars.length) {
        traces.push({type: 'bar', x: bars.map(function (b) { return (b.x0 + b.x1) / 2; }),
                     width: bars.map(function (b) { return b.x1 - b.x0; }), y: bars.map(function (b) { return b.h; }),
                     customdata: bars.map(function (b) { return b.label; }), hovertemplate: '%{customdata}<extra></extra>',
                     marker: {color: 'rgba(70,130,180,0.35)'}, xaxis: 'x' + ax, yaxis: 'y' + ax, showlegend: false});
      }
      if (!f || !f.components) return;          // components not published: too few people
      var lo, hi;
      if (bars.length) {
        var span = bars[bars.length - 1].x1 - bars[0].x0;
        lo = bars[0].x0 - 0.3 * span; hi = bars[bars.length - 1].x1 + 0.3 * span;
      } else {
        lo = Math.min.apply(null, f.components.map(function (k) { return k.mean - 4 * k.sigma; }));
        hi = Math.max.apply(null, f.components.map(function (k) { return k.mean + 4 * k.sigma; }));
      }
      var xs = [];
      for (var i = 0; i <= 200; i++) xs.push(lo + (hi - lo) * i / 200);
      f.components.forEach(function (k, ci) {
        traces.push({x: xs, y: xs.map(function (x) { return k.weight * normPdf(x, k.mean, k.sigma); }), mode: 'lines',
                     line: {color: ci === 0 ? BLUE : ORANGE, width: 2}, xaxis: 'x' + ax, yaxis: 'y' + ax,
                     showlegend: false, hoverinfo: 'skip'});
      });
      if (f.sep_native != null) {
        shapes.push({type: 'line', xref: 'x' + ax, yref: 'paper', x0: f.sep_native, x1: f.sep_native, y0: 0, y1: 1,
                     line: {color: 'red', width: 2, dash: 'dash'}});
      }
    });
    if (!traces.length) {
      var el = document.getElementById(divId);
      el.style.height = 'auto';
      el.innerHTML = '<p class="text-muted mb-0">Fit not shown: one of its components, or the no-unit values, ' +
                     'come from too few people.</p>';
      return;
    }
    var isLog = bim.winner === 'log';
    function badge(xr, yr, name, f, win) {
      return {xref: xr, yref: yr, x: 0.98, y: 0.97, xanchor: 'right', yanchor: 'top', showarrow: false,
              text: name + (win ? ' ✓ winner' : '') + '<br>BIC=' + (f && f.bic != null ? f.bic.toFixed(1) : 'NA'),
              bgcolor: win ? 'rgba(200,255,200,0.85)' : 'rgba(240,240,240,0.8)', bordercolor: '#ccc',
              borderwidth: 1, font: {size: 10}};
    }
    Plotly.newPlot(divId, traces, {
      grid: {rows: 1, columns: 2, pattern: 'independent'},
      xaxis: {title: 'value'}, xaxis2: {title: 'log₁₀(value)'}, yaxis: {title: 'density'}, yaxis2: {title: ''},
      shapes: shapes, bargap: 0, margin: {t: 30, b: 50, l: 55, r: 10},
      annotations: [badge('x domain', 'y domain', 'linear', bim.linear, !isLog),
                    badge('x2 domain', 'y2 domain', 'log', bim.log, isLog)],
    }, {responsive: true});
  }
"""


def _chart_js() -> str:
    return _CHART_JS.replace("__PCTS__", json.dumps(list(disclosure.PCTS)))


def _fmt_count(v) -> str:
    """A published count: thousands separators, or the marker as is."""
    if v is None or (isinstance(v, float) and pd.isna(v)) or v == "":
        return "—"
    try:
        return f"{int(float(v)):,}"
    except (TypeError, ValueError):
        return str(v)


def _side_text(rows, people) -> str:
    """'1,500 rows · 394 people', or the marker when the group is too small to be described."""
    if disclosure.BELOW in (str(rows), str(people)):
        return disclosure.BELOW
    return f"{_fmt_count(rows)} rows · {_fmt_count(people)} people"


def _tests_table(pd_) -> str:
    """KS / t / MAD results of one comparison as a small table."""
    t = (pd_ or {}).get("tests") or {}
    def res(x):
        return ('<span class="pass">PASS</span>' if x.get("passed") else '<span class="fail">FAIL</span>')
    def num(v, f):
        return format(v, f) if isinstance(v, (int, float)) else "NA"
    cells = []
    if "ks" in t:
        cells.append(f"<td>KS</td><td>stat {num(t['ks'].get('stat'), '.3g')}, −log10 p {num(t['ks'].get('mlogp'), '.1f')}</td><td>{res(t['ks'])}</td>")
    if "t" in t:
        cells.append(f"<td>Welch t</td><td>t {num(t['t'].get('stat'), '.3g')}, −log10 p {num(t['t'].get('mlogp'), '.1f')}</td><td>{res(t['t'])}</td>")
    if "mad" in t:
        m = t["mad"]
        cells.append(f"<td>MAD</td><td>|median difference| {num(m.get('distance'), '.3g')} vs "
                     f"{num(m.get('n_mad'), 'g')} × MAD = {num(m.get('threshold'), '.3g')}</td><td>{res(m)}</td>")
    if not cells:
        return ""
    return ('<table class="table table-sm tests mb-0"><tbody>'
            + "".join(f"<tr>{c}</tr>" for c in cells) + "</tbody></table>")


GATE_NOTE = (f"Distributions are shown as percentiles only (P5 to P95), for groups of at least "
             f"{disclosure.MIN_DIST_SUBJECTS} people and {disclosure.MIN_DIST_RECORDS} rows in which "
             f"every 5% of the rows comes from several people. See the documentation.")


_AMBIG_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>__TEST_NAME__</title>
  <link rel="stylesheet"
    href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css">
  <style>
    body   { padding: 2rem; font-size: 0.9rem; }
    nav a  { margin-right: 1.5rem; font-weight: 500; }
    .badge-pass { background: #198754; }
    .badge-fail { background: #dc3545; }
    .badge-skip { background: #6c757d; }
    .stat-label { color: #6c757d; font-size: 0.78rem; text-transform: uppercase; }
    .stat-value { font-size: 1rem; font-weight: 600; }
    .overlap-low  { color: #198754; }
    .overlap-mid  { color: #fd7e14; }
    .overlap-high { color: #dc3545; font-weight: 700; }
    .section-box { border: 1px solid #dee2e6; border-radius: 6px;
                   padding: 1rem; margin-bottom: 1.5rem; }
    .winner-box  { border-color: #198754; }
    .chart       { width: 100%; height: 360px; }
    .chart-bim   { width: 100%; height: 340px; }
    .pass { color: #198754; font-weight: 600; }
    .fail { color: #dc3545; font-weight: 600; }
    table.tests  { width: auto; font-size: 0.85rem; }
  </style>
</head>
<body>
  <nav class="mb-3">
    <a href="../index.html">← Results</a>
    <a href="../doc.html">Documentation</a>
    __NAV_UNHARM__
  </nav>
  <hr>

  <div class="d-flex align-items-center gap-3 mb-3">
    <h3 class="mb-0">__TEST_NAME__</h3>
    <span class="text-muted">ambiguous</span>
  </div>

  <div class="row g-3 mb-3">
    <div class="col-auto"><div class="stat-label">Bimodal status</div>
      <div class="stat-value">__BIMODAL_STATUS__</div></div>
    <div class="col-auto"><div class="stat-label">Split improvement</div>
      <div class="stat-value">__SCORE_IMPROVEMENT__</div></div>
    <div class="col-auto"><div class="stat-label">Distribution overlap</div>
      <div class="stat-value __OVERLAP_CLASS__">__OVERLAP_PCT__</div></div>
  </div>

  <p class="text-muted">__EXPLANATION__</p>

  <!-- Bimodal section -->
  <div class="section-box">
    <h5>Bimodal check
      <small class="text-muted ms-2" style="font-size:0.85rem">
        winner: __BIM_WINNER__ &nbsp;|&nbsp; sep=__BIM_SEP__ &nbsp;|&nbsp;
        BC=__BIM_BC__ &nbsp;|&nbsp; dip_p=__BIM_DIP_P__ &nbsp;|&nbsp;
        overlap=__OVERLAP_PCT__
      </small>
    </h5>
    <div id="chart-bimodal" class="chart-bim"></div>
    <p class="small text-muted mb-0">Curves: the two fitted components. Bars: the no-unit values
    between their percentiles (each bar holds 5% of the rows).</p>
  </div>

  <p class="small text-muted">__GATE_NOTE__</p>

  <!-- Engine sections (one per winning sub-dist) -->
  __ENGINE_SECTIONS__

  <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
  <script>
  const PAGE = __PAGE_DATA_JSON__;
__CHART_JS__
  renderBimodal('chart-bimodal', PAGE.bimodal);
  PAGE.sections.forEach(function (s) { renderCompare('chart-' + s.chart_id, s.plot_data); });
  </script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_scatter(data_dir: Path, out_dir: Path) -> str:
    p = data_dir / "test_names_exploration_scatter.png"
    if p.exists():
        dest = out_dir / "test_names_exploration_scatter.png"
        if not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dest)
        data_dest = out_dir / "data" / "test_names_exploration_scatter.png"
        if not data_dest.exists():
            data_dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, data_dest)
        return "test_names_exploration_scatter.png"
    return ""


def _md_to_html(path: Path) -> str:
    if not path.exists():
        return ""
    return markdown.markdown(path.read_text(), extensions=["tables", "fenced_code"])


def _summary_section(data_dir: Path) -> str:
    parts = []
    for fname, title in (("summary_table.md",    "Category summary"),
                         ("assignment_summary.md", "Assignment summary")):
        html = _md_to_html(data_dir / fname)
        if html:
            parts.append(f'<h4>{title}</h4>{html}')
    if not parts:
        return ""
    return '<div class="summary-tables">' + "".join(parts) + "</div>"


def _scatter_block(rel_path: str) -> str:
    if not rel_path:
        return '<p class="text-muted">Scatter plot not found.</p>'
    return (f'<div class="scatter">'
            f'<img src="{rel_path}" alt="Scatter: TEST_NAMEs by count and prevalence">'
            f'</div>')


# ---------------------------------------------------------------------------
# Page builders
# ---------------------------------------------------------------------------

def build_index(out_dir: Path, data_dir: Path,
                inj_path: Path = None, scatter_override: str = None) -> None:
    inj = inj_path or data_dir / "injection_results.tsv"
    if not inj.exists():
        print(f"Warning: {inj} not found — table will be empty")
        df = pd.DataFrame(columns=COL_ORDER)
    else:
        df = pd.read_csv(inj, sep="\t", na_values=["NA"])

    # COL_ORDER first (if present), then any remaining TSV columns, minus excluded ones
    ordered   = [c for c in COL_ORDER  if c in df.columns]
    remaining = [c for c in df.columns if c not in set(ordered) and c not in EXCLUDE_COLS]
    all_cols  = ordered + remaining
    cols      = [(c, COL_LABELS.get(c, c)) for c in all_cols]

    df_out = df[all_cols].copy().fillna("")
    if "CUTOFF" in df_out.columns:
        df_out["CUTOFF"] = df_out["CUTOFF"].replace(float("inf"), "∞")
    rows   = df_out.to_dict(orient="records")

    # Which column indices hold numeric data
    numeric_set = {i for i, c in enumerate(all_cols)
                   if pd.api.types.is_numeric_dtype(df[c])}

    # Build columns array with widths embedded; mark wide columns for CSS
    wide_cols = {"TEST_NAME", "NOTES", "UNIT"}
    def _col_def(k, l):
        d = {"data": k, "title": l}
        if k in COL_WIDTHS:
            d["width"] = f"{COL_WIDTHS[k]}px"
        if k in wide_cols:
            d["className"] = "col-wide"
        if k == "TEST_NAME":
            d["render"] = "__TEST_NAME_RENDER__"
        return d

    headers           = "".join(f"<th>{label}</th>" for _, label in cols)
    footers           = "".join(f'<th><input type="text" placeholder="{label}"></th>'
                                for _, label in cols)
    data_json         = json.dumps(rows, allow_nan=False)
    columns_json      = json.dumps([_col_def(k, l) for k, l in cols])
    numeric_cols_json = json.dumps({i: True for i in numeric_set})
    outcome_idx       = next((i for i, (k, _) in enumerate(cols) if k == "OUTCOME"), 3)
    scatter_path      = scatter_override if scatter_override is not None else _find_scatter(data_dir, out_dir)

    # TEST_NAME render function: link to per-test page
    test_name_render = ("function(data,type){if(type!=='display')return data;"
                        "var slug=data.replace(/[^a-zA-Z0-9_-]/g,'_');"
                        "return '<a href=\"tests/'+slug+'.html\">'+data+'</a>';}")

    html = (_INDEX
            .replace("__SCATTER_BLOCK__",     _scatter_block(scatter_path))
            .replace("__SUMMARY_SECTION__",   _summary_section(data_dir))
            .replace("__HEADERS__",           headers)
            .replace("__FOOTERS__",           footers)
            .replace("__DATA_JSON__",         data_json)
            .replace("__COLUMNS_JSON__",      columns_json)
            .replace("__NUMERIC_COLS_JSON__",  numeric_cols_json)
            .replace("__OUTCOME_COL_IDX__",   str(outcome_idx))
            .replace('"__TEST_NAME_RENDER__"', test_name_render))

    out = out_dir / "index.html"
    out.write_text(_with_nav(html, ""), encoding="utf-8")
    print(f"Wrote {out}  ({len(rows)} rows)")


# Link to the unharmonized portal in every page's nav; set by build.py when that portal is built.
NAV_UNHARM = False


def _with_nav(html: str, prefix: str) -> str:
    link = f'<a href="{prefix}unharmonized/index.html">Unharmonized</a>' if NAV_UNHARM else ""
    return html.replace("__NAV_UNHARM__", link)


def _slugify(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]", "_", name)


_TEST_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>__TEST_NAME__</title>
  <link rel="stylesheet"
    href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css">
  <style>
    body  { padding: 2rem; font-size: 0.9rem; }
    nav a { margin-right: 1.5rem; font-weight: 500; }
    .badge-pass { background: #198754; }
    .badge-fail { background: #dc3545; }
    .stat-label { color: #6c757d; font-size: 0.78rem; text-transform: uppercase; }
    .stat-value { font-size: 1rem; font-weight: 600; }
    .pass { color: #198754; font-weight: 600; }
    .fail { color: #dc3545; font-weight: 600; }
    table.tests { width: auto; font-size: 0.85rem; }
    #chart      { width: 100%; height: 400px; }
  </style>
</head>
<body>
  <nav class="mb-3">
    <a href="../index.html">← Results</a>
    <a href="../doc.html">Documentation</a>
    __NAV_UNHARM__
  </nav>
  <hr>

  <div class="d-flex align-items-center gap-3 mb-3">
    <h3 class="mb-0">__TEST_NAME__</h3>
    <span class="badge __OUTCOME_CLASS__ fs-6">__OUTCOME__</span>
    <span class="text-muted">__TYPE__</span>
  </div>

  <div class="row g-3 mb-3">
    <div class="col-auto"><div class="stat-label">Unit</div><div class="stat-value">__UNIT__</div></div>
    <div class="col-auto"><div class="stat-label">Unit share</div><div class="stat-value">__UNIT_PREV__</div></div>
    <div class="col-auto"><div class="stat-label">Candidate (no unit)</div><div class="stat-value">__N_CAND__</div></div>
    <div class="col-auto"><div class="stat-label">Target (__UNIT__)</div><div class="stat-value">__N_TARGET__</div></div>
    <div class="col-auto"><div class="stat-label">Decided by</div><div class="stat-value">__DECIDED_BY__</div></div>
  </div>

  <p class="text-muted">__EXPLANATION__</p>

  <div id="chart"></div>
  __TESTS_TABLE__
  <p class="small text-muted mt-2">__GATE_NOTE__</p>

  <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
  <script>
  const PD = __PLOT_DATA_JSON__;
__CHART_JS__
  renderCompare('chart', PD);
  </script>
</body>
</html>
"""


def _explanation(row, pd) -> str:
    outcome    = row.get("OUTCOME", "")
    decided_by = pd.get("decided_by", "")
    unit       = row.get("UNIT", "")
    prev       = row.get("UNIT_PREVALENCE", "")

    if outcome == "PASS":
        by_map = {
            "KS":  f"The KS test confirmed the distributions match (stat={row.get('KS_STAT',''):.3g}, −log10p={row.get('KS_MLOGP',''):.1f}).",
            "T":   f"The KS test was inconclusive but the Welch t-test confirmed similar means (−log10p={row.get('T_MLOGP',''):.1f}).",
            "MAD": f"KS and t-tests were inconclusive, but the MAD test confirmed medians are within the threshold (dist={row.get('MAD_DIST',''):.3g}).",
        }
        desc = by_map.get(decided_by, "")
        share = (f"{prev:.1f}% of the rows with a unit use <b>{unit}</b> (rarely used units left out). "
                 if isinstance(prev, (int, float)) and prev == prev else "")   # prev == prev: not NaN
        return f"{share}{desc} Unit <b>{unit}</b> assigned."
    else:
        return (f"No unit could be confidently assigned. "
                f"The candidate distribution did not match any reference unit "
                f"(KS stat={row.get('KS_STAT',''):.3g}, MAD dist={row.get('MAD_DIST',''):.3g}).")


def _ambig_explanation(rows, bim_data: dict) -> str:
    status = str(rows.iloc[0].get("BIMODAL_STATUS", ""))
    sep    = rows.iloc[0].get("BIMODAL_SEP")
    bc     = rows.iloc[0].get("BIMODAL_BC")
    dip_p  = rows.iloc[0].get("BIMODAL_DIP_P")
    impr   = rows.iloc[0].get("SCORE_IMPROVEMENT")

    # Bimodal sentence
    if status == "skipped":
        bim_sentence = "No split was performed — the pre-check passed globally without needing to split the distribution."
    elif status == "split_by_score":
        impr_str = f"{impr:+.1%}" if pd.notna(impr) else "?"
        sep_str  = f"{sep:.4g}"   if pd.notna(sep)  else "?"
        bim_sentence = (f"The distribution was split by score improvement ({impr_str}) at cutoff {sep_str}, "
                        f"even though the dip test did not flag bimodality.")
    elif status in ("bimodal", "bimodal_cautious"):
        dip_str = f"{dip_p:.3g}" if pd.notna(dip_p) else "?"
        bc_str  = f"{bc:.3f}"   if pd.notna(bc)    else "?"
        sep_str = f"{sep:.4g}"  if pd.notna(sep)   else "?"
        cautious = " (modes overlap)" if status == "bimodal_cautious" else ""
        bim_sentence = (f"Hartigan's dip test detected bimodality{cautious} "
                        f"(dip p={dip_str}, BC={bc_str}). Distribution split at {sep_str}.")
    else:
        bim_sentence = ""

    # Per sub-dist sentences
    sub_sentences = []
    for _, row in rows.iterrows():
        sub     = str(row.get("SUB_DIST", "all"))
        unit    = str(row.get("UNIT", ""))
        outcome = str(row.get("OUTCOME", ""))
        decided = ("KS" if row.get("KS_PASS") == "PASS"
                   else "T" if row.get("T_PASS") == "PASS"
                   else "MAD" if row.get("OUTCOME") not in ("SKIP", "")
                   else "")
        prev    = row.get("UNIT_PREVALENCE")
        prev_str = f"{prev:.1f}%" if pd.notna(prev) else "?"
        if sub == "all":
            label = "The full candidate distribution"
        else:
            label = f"The <b>{sub}</b> sub-distribution"
        if outcome == "PASS":
            sub_sentences.append(f"{label} matched unit <b>{unit}</b> ({prev_str} of the rows with a unit, decided by {decided}).")
        else:
            sub_sentences.append(f"{label} did not match any unit confidently (outcome: {outcome}).")

    return " ".join(filter(None, [bim_sentence] + sub_sentences))


def _print_progress(current: int, total: int, prefix: str = "Building test pages") -> None:
    width = 40
    filled = int(width * current / total) if total else width
    bar = "█" * filled + "░" * (width - filled)
    pct = 100 * current / total if total else 100
    print(f"\r{prefix}  [{bar}] {current}/{total} ({pct:.0f}%)",
          end="\n" if current == total else "", flush=True)


def build_test_pages(out_dir: Path, data_dir: Path) -> None:
    plot_data_path = data_dir / "plot_data.json.gz"
    inj_path       = data_dir / "injection_results.tsv"

    if not plot_data_path.exists():
        print(f"Skipping test pages: {plot_data_path} not found")
        return
    if not inj_path.exists():
        print(f"Skipping test pages: {inj_path} not found")
        return

    with gzip.open(plot_data_path, "rb") as fh:
        plot_data = json.loads(fh.read())
    df        = pd.read_csv(inj_path, sep="\t", na_values=["NA"])

    # Unambiguous tests have one comparison (candidate / target) at the top level
    unamb = df[df["TYPE"] == "unambiguous"].copy()

    tests_dir = out_dir / "tests"
    tests_dir.mkdir(exist_ok=True)

    # Pre-filter so total count is accurate
    unamb_rows   = [(row, plot_data[row["TEST_NAME"]])
                    for _, row in unamb.iterrows()
                    if row["TEST_NAME"] in plot_data and "candidate" in plot_data[row["TEST_NAME"]]]
    ambig_names  = [n for n in df[df["TYPE"] == "ambiguous"]["TEST_NAME"].unique()
                    if n in plot_data and "engine" in plot_data[n]]
    total        = len(unamb_rows) + len(ambig_names)

    written = 0
    for row, pd_ in unamb_rows:
        written += 1
        _print_progress(written, total)
        name = row["TEST_NAME"]
        slug    = _slugify(name)
        outcome = str(row.get("OUTCOME", ""))
        prev    = pd.to_numeric(row.get("UNIT_PREVALENCE"), errors="coerce")
        html = (_TEST_PAGE
                .replace("__TEST_NAME__",     name)
                .replace("__OUTCOME_CLASS__",  "badge-pass" if outcome == "PASS" else "badge-fail")
                .replace("__OUTCOME__",        outcome)
                .replace("__TYPE__",           str(row.get("TYPE", "")))
                .replace("__UNIT__",           str(row.get("UNIT", "")))
                .replace("__UNIT_PREV__",      f"{prev:.1f}%" if pd.notna(prev) else "—")
                .replace("__N_CAND__",         _side_text(row.get("N_CANDIDATE"), row.get("N_CANDIDATE_SUBJECTS")))
                .replace("__N_TARGET__",       _side_text(row.get("N_TARGET"), row.get("N_TARGET_SUBJECTS")))
                .replace("__DECIDED_BY__",     str(pd_.get("decided_by", "")))
                .replace("__EXPLANATION__",    _explanation(row, pd_))
                .replace("__TESTS_TABLE__",    _tests_table(pd_))
                .replace("__GATE_NOTE__",      GATE_NOTE)
                .replace("__CHART_JS__",       _chart_js())
                .replace("__PLOT_DATA_JSON__", json.dumps(pd_)))
        (tests_dir / f"{slug}.html").write_text(_with_nav(html, "../"), encoding="utf-8")

    # Ambiguous pages
    for name in ambig_names:
        written += 1
        _print_progress(written, total)
        pd_   = plot_data[name]
        rows  = df[df["TEST_NAME"] == name]
        bim   = pd_.get("bimodal") or {}
        first = rows.iloc[0]

        # Build one engine section per winning row
        engine_sections_html = ""
        sections_data = []
        for _, row in rows.iterrows():
            sub   = str(row.get("SUB_DIST", "all"))
            unit  = str(row.get("UNIT", ""))
            key   = f"{sub}_{unit}"
            epd   = pd_["engine"].get(key)
            outcome = str(row.get("OUTCOME", ""))
            badge = "badge-pass" if outcome == "PASS" else ("badge-fail" if outcome == "FAIL" else "badge-skip")
            chart_id = _slugify(key)
            label = f"{sub} — {unit}"
            prev  = pd.to_numeric(row.get("UNIT_PREVALENCE"), errors="coerce")
            engine_sections_html += (
                f'<div class="section-box{" winner-box" if outcome=="PASS" else ""}">'
                f'<h5>{label} <span class="badge {badge} ms-2">{outcome}</span>'
                f'<small class="text-muted ms-3" style="font-size:0.8rem">'
                f'candidate {_side_text(row.get("N_CANDIDATE"), row.get("N_CANDIDATE_SUBJECTS"))} &nbsp;·&nbsp; '
                f'target {_side_text(row.get("N_TARGET"), row.get("N_TARGET_SUBJECTS"))} &nbsp;·&nbsp; '
                f'unit share {f"{prev:.1f}%" if pd.notna(prev) else "—"}</small></h5>'
                f'<div id="chart-{chart_id}" class="chart"></div>'
                f'{_tests_table(epd)}'
                f'</div>'
            )
            sections_data.append({"chart_id": chart_id, "plot_data": epd})

        bim_sep = f"{bim.get('separator'):.4g}" if bim.get('separator') is not None else "NA"
        bim_bc  = f"{bim.get('bc'):.3f}"        if bim.get('bc')        is not None else "NA"
        bim_dip = f"{bim.get('dip_p'):.3g}"     if bim.get('dip_p')     is not None else "NA"
        score_impr = first.get("SCORE_IMPROVEMENT")
        score_str  = f"{score_impr:+.1%}" if pd.notna(score_impr) else "NA"
        overlap_val = first.get("BIMODAL_OVERLAP")
        overlap_str = f"{overlap_val:.1f}%" if pd.notna(overlap_val) else "—"
        overlap_cls = ("overlap-low"  if pd.notna(overlap_val) and overlap_val < 5
                  else "overlap-mid"  if pd.notna(overlap_val) and overlap_val < 15
                  else "overlap-high" if pd.notna(overlap_val)
                  else "")

        page_data = {"bimodal": bim, "sections": sections_data}
        html = (_AMBIG_PAGE
                .replace("__TEST_NAME__",        name)
                .replace("__EXPLANATION__",      _ambig_explanation(rows, bim))
                .replace("__BIMODAL_STATUS__",   str(first.get("BIMODAL_STATUS", "NA")))
                .replace("__SCORE_IMPROVEMENT__", score_str)
                .replace("__OVERLAP_CLASS__",    overlap_cls)
                .replace("__OVERLAP_PCT__",      overlap_str)
                .replace("__BIM_WINNER__",       bim.get("winner", "NA"))
                .replace("__BIM_SEP__",          bim_sep)
                .replace("__BIM_BC__",           bim_bc)
                .replace("__BIM_DIP_P__",        bim_dip)
                .replace("__ENGINE_SECTIONS__",  engine_sections_html)
                .replace("__GATE_NOTE__",        GATE_NOTE)
                .replace("__CHART_JS__",         _chart_js())
                .replace("__PAGE_DATA_JSON__",   json.dumps(page_data)))
        slug = _slugify(name)
        (tests_dir / f"{slug}.html").write_text(_with_nav(html, "../"), encoding="utf-8")

    print(f"Wrote {written} test pages → {tests_dir}")


def build_doc(out_dir: Path, readme_path: Path) -> None:
    text = readme_path.read_text()
    readme_html = markdown.markdown(text, extensions=["tables", "fenced_code"])
    html = _DOC.replace("__README_HTML__", readme_html)
    out = out_dir / "doc.html"
    out.write_text(_with_nav(html, ""))
    print(f"Wrote {out}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Build HTML site from injection pipeline outputs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _root = Path(__file__).resolve().parents[2]
    p.add_argument("--data-dir", default=str(_root / "data"), metavar="PATH",
                   help="Directory containing all pipeline outputs (TSVs, PNGs, MDs)")
    p.add_argument("--out-dir", default=None, metavar="PATH",
                   help="Directory where index.html and doc.html are written "
                        "(default: same as --data-dir)")
    p.add_argument("--injection-results", default=None, metavar="PATH",
                   help="Override path for injection_results.tsv")
    p.add_argument("--scatter", default=None, metavar="PATH",
                   help="Scatter plot filename relative to --data-dir (default: auto-detected)")
    p.add_argument("--readme", default=None, metavar="PATH",
                   help="Path to README.md (default: README.md at the repo root)")
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    out_dir  = Path(args.out_dir) if args.out_dir else Path(".")
    readme   = Path(args.readme) if args.readme else _root / "README.md"
    inj      = Path(args.injection_results) if args.injection_results else None
    scatter  = args.scatter

    build_index(out_dir, data_dir, inj_path=inj, scatter_override=scatter)
    build_doc(out_dir, readme)
    build_test_pages(out_dir, data_dir)


if __name__ == "__main__":
    main()
