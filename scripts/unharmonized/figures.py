"""
figures.py — one small PNG per suggestion, embedded in its review issue.

The file is named after the suggestion id (test, unit, result, fix, factor, concept), so it is
drawn once when a suggestion first appears and never overwritten: an issue keeps showing the
evidence it was opened with, whatever later runs find. Two panels:

  left   reference ranges as bars: the concept's typical range(s) in the suggested (or closest)
         unit in green, the group's most common ranges in blue with their row counts
  right  values: the group's log10 percentiles against the harmonized ones in that unit (bars of 5% of
         the rows + a smooth line), with the typical ranges shaded; the group alone when there is no candidate

Aggregates only, from the published plot data: value percentiles (no curves through raw values),
range pairs and typical ranges from enough people (see scripts/common/disclosure.py).
"""

import io
import math

import numpy as np

GREEN, BLUE, ORANGE, GREY, INK, MUTED = "#1baf7a", "#4682b4", "#ff8c00", "#adb5bd", "#212529", "#6c757d"
MAX_BARS = 8


def _nn(v):
    """None for a missing or non-finite limit (plot data in memory has NaN, from JSON None)."""
    return None if v is None or not np.isfinite(v) else float(v)


def _rtxt(lo, hi):
    f = lambda v: f"{float(f'{v:.6g}'):g}"
    if lo is None:
        return f"≤ {f(hi)}"
    if hi is None:
        return f"≥ {f(lo)}"
    return f"{f(lo)}–{f(hi)}"


def _ok(lo, hi):
    good = lambda v: v is None or v > 0
    return good(lo) and good(hi) and (lo is not None or hi is not None) and (lo is None or hi is None or lo < hi)


def _equal_mass_bars(xs, step=0.05):
    """Bars between neighbouring percentiles (5% of the rows each; tied ones merged):
    [(x0, x1, height = share / width)]."""
    bars, start = [], 0
    for i in range(1, len(xs)):
        if xs[i] > xs[start]:
            bars.append((xs[start], xs[i], step * (i - start) / (xs[i] - xs[start])))
            start = i
    return bars


def _smooth(bars):
    """A smooth line through the bars: each bar's share as a normal curve of half its width (at
    least 1/25 of the range)."""
    lo, hi = bars[0][0], bars[-1][1]
    xs = np.linspace(lo, hi, 201)
    ys = np.zeros_like(xs)
    for a, b, h in bars:
        sd = max((b - a) / 2, (hi - lo) / 25, 1e-12)
        ys += h * (b - a) * np.exp(-0.5 * ((xs - (a + b) / 2) / sd) ** 2) / (sd * math.sqrt(2 * math.pi))
    return xs, ys


def draw(row, plot, path):
    """Write the evidence figure for one group (row from groups.tsv, plot from plot_data)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    cid = plot.get("picked")
    cand = plot.get("candidates", {}).get(cid) if cid else None
    clusters = (plot.get("concepts", {}).get(cand["omop"], {}).get("clusters", []) if cand else [])
    bands = [(None if _nn(c.get("LOWER")) is None else c["LOWER"] / cand["cf"],
              None if _nn(c.get("UPPER")) is None else c["UPPER"] / cand["cf"]) for c in clusters] if cand else []

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.4, 3.3), gridspec_kw={"width_ratios": [1.1, 1]})
    fix = (f": {row['SOURCE_UNIT']} → {row['FIX_UNIT']} (× {float(row['CONVERSION_FACTOR']):g})"
           if row["RESULT"] == "Suggest fix" else
           f" ({row['TEST_NAME']}, {row['SOURCE_UNIT']}) → {row['OMOP_ID']}"
           if row["RESULT"] == "Unit is right: approve the Usagi row" else "")
    fig.suptitle(f"{row['TEST_NAME']} · {row['SOURCE_UNIT']} — {row['RESULT']}{fix}", fontsize=10, color=INK,
                 x=0.01, ha="left")

    # ---- ranges ----
    pairs = [(_nn(p[0]), _nn(p[1]), p[2]) for p in plot.get("ranges", [])]
    pairs = sorted([p for p in pairs if _ok(p[0], p[1])], key=lambda p: -p[2])[:MAX_BARS]
    rows = [(f"typical{' (' + str(i + 1) + ')' if i else ''}", lo, hi, GREEN, _rtxt(lo, hi)) for i, (lo, hi) in enumerate(bands)]
    rows += [(_rtxt(lo, hi), lo, hi, BLUE, f"{n:,}") for lo, hi, n in pairs]
    if rows:
        lims = [v for _, lo, hi, _, _ in rows for v in (lo, hi) if v is not None]
        lo0, hi0 = min(lims), max(lims)
        spread = (hi0 - lo0) or abs(hi0) * 0.1 or 1
        xmin = 0 if lo0 < spread else lo0 - 0.15 * spread
        xmax = hi0 + 0.15 * spread
        for i, (label, lo, hi, col, txt) in enumerate(rows):
            a, b = (xmin if lo is None else lo), (xmax if hi is None else hi)
            ax1.barh(i, b - a, left=a, height=0.6, color=col)
            ax1.text(b, i, " " + txt, va="center", fontsize=7, color=MUTED, clip_on=False)
        ax1.set_yticks(range(len(rows)), [r[0] for r in rows], fontsize=7)
        ax1.set_ylim(len(rows) - 0.5, -0.5)
        ax1.set_xlim(xmin, xmax)
        if bands and pairs:
            ax1.axhline(len(bands) - 0.5, color=GREY, lw=0.8, ls=":")
        unit = cand["unit"] if cand else "?"
        ax1.set_title(f"Reference ranges (green: typical, in {unit} · blue: these rows)",
                      fontsize=8, color=MUTED, loc="left")
    else:
        ax1.text(0.5, 0.5, "no reference ranges", ha="center", va="center", color=MUTED, fontsize=8,
                 transform=ax1.transAxes)
        ax1.set_yticks([])
    ax1.tick_params(axis="x", labelsize=7)

    # ---- values: percentiles only (equal-mass bars + smooth line, log10), as on the group page ----
    pd_ = cand.get("plot") if cand else None
    sides = ([("unharmonized rows", (pd_ or {}).get("candidate"), BLUE),
              (f"harmonized (in {cand['unit']})", (pd_ or {}).get("target"), ORANGE)] if pd_
             else [("unharmonized rows", plot.get("group"), BLUE)])
    drawn = []
    for label, side, col in sides:
        p = (side or {}).get("pct")
        if not p or min(p) <= 0:
            continue
        bars = _equal_mass_bars([math.log10(v) for v in p])
        if not bars:
            continue
        ax2.bar([(a + b) / 2 for a, b, _ in bars], [h for _, _, h in bars], width=[b - a for a, b, _ in bars],
                color=col, alpha=0.25, lw=0)
        xs, ys = _smooth(bars)
        ax2.plot(xs, ys, color=col, lw=1.4, label=label)
        drawn += [bars[0][0], bars[-1][1]]
    if drawn:
        for lo, hi in bands:
            a = math.log10(lo) if lo and lo > 0 else min(drawn)
            b = math.log10(hi) if hi and hi > 0 else max(drawn)
            ax2.axvspan(a, b, color=GREEN, alpha=0.12, lw=0)
        ax2.legend(fontsize=7, frameon=False, loc="upper right")
        ax2.set_title(f"Values P5–P95, log10 (as {cand['unit']} × {cand['cf']:g})" if pd_
                      else "Values P5–P95, log10 (no candidate unit)",
                      fontsize=8, color=MUTED, loc="left")
    else:
        ax2.text(0.5, 0.5, "no values to show", ha="center", va="center", color=MUTED, fontsize=8,
                 transform=ax2.transAxes)
    ax2.set_yticks([])
    ax2.tick_params(axis="x", labelsize=7)
    for ax in (ax1, ax2):
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.text(0.99, 0.01, f"suggestion {row['SUGGESTION_ID']}", ha="right", fontsize=6, color=GREY)
    fig.tight_layout(rect=(0, 0.02, 1, 0.93))

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, facecolor="white")
    plt.close(fig)
    buf.seek(0)
    Image.open(buf).convert("RGB").quantize(colors=64).save(path, format="PNG", optimize=True)
