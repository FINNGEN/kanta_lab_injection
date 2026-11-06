"""
ref_plots.py

One diagnostic PNG per test: what the ranges say about the unit-less rows.

  left   histogram of log10(candidate value), stacked by the unit the row's range points to
         (grey: no decision); solid lines = supervised cutoffs from the range labels,
         dashed line = the value engine's cutoff when given
  right  the candidate rows' distinct ranges (log10 lower vs log10 upper, size ~ rows),
         coloured the same way; crosses = each unit profile's median limits, boxes = the
         match margin around them. One-sided ranges are drawn on the axis edge.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import ref_ranges as rr

SERIES  = ["#2a78d6", "#eb6834", "#1baf7a"]   # categorical slots 1-3 (validated all-pairs)
NONE    = "#b5b4ae"
TEXT    = "#0b0b0b"
TEXT2   = "#52514e"
SURFACE = "#fcfcfb"
GRID    = "#e6e5e0"


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=TEXT2, labelsize=8)
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def plot_test(name, cand, res, dec, out_path, value_cutoff=None, title_extra=""):
    values = cand["VALUE"].to_numpy(dtype=float)
    status, group, rep = res["status"], res["group"], res["group_rep"]
    decided = np.isin(status, rr.DECIDED) & (group >= 0)
    groups = sorted(rep)
    colors = {g: SERIES[i % len(SERIES)] for i, g in enumerate(groups)}

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6), facecolor=SURFACE)
    for a in (a1, a2):
        _style(a)

    # left: values by range label
    pos = values > 0
    lv = np.log10(values[pos])
    if len(lv):
        bins = np.linspace(np.percentile(lv, 0.5), np.percentile(lv, 99.5), 60)
        stacks, labels, cols = [], [], []
        for g in groups:
            m = (decided & (group == g))[pos]
            if m.any():
                stacks.append(lv[m]); labels.append(f"{rep[g]} (range)"); cols.append(colors[g])
        m = (~decided)[pos]
        if m.any():
            stacks.append(lv[m]); labels.append("no range decision"); cols.append(NONE)
        a1.hist(stacks, bins=bins, stacked=True, color=cols, label=labels,
                edgecolor=SURFACE, linewidth=0.5)
        for t, acc, _ in dec.get("cutoffs", []):
            if np.isfinite(t) and t > 0:
                a1.axvline(np.log10(t), color=TEXT, lw=1.5)
                a1.text(np.log10(t), a1.get_ylim()[1] * 0.97, f" ranges: {t:.3g} (acc {acc:.0%})",
                        color=TEXT, fontsize=8, va="top")
        if value_cutoff is not None and np.isfinite(value_cutoff) and value_cutoff > 0:
            a1.axvline(np.log10(value_cutoff), color=TEXT2, lw=1.5, ls="--")
            a1.text(np.log10(value_cutoff), a1.get_ylim()[1] * 0.85, f" value engine: {value_cutoff:.3g}",
                    color=TEXT2, fontsize=8, va="top")
        a1.legend(fontsize=8, frameon=False, labelcolor=TEXT)
    a1.set_xlabel("log10(candidate value)", color=TEXT2, fontsize=9)
    a1.set_ylabel("unit-less rows", color=TEXT2, fontsize=9)
    a1.set_title("Candidate values, by the unit their range points to", color=TEXT, fontsize=10, loc="left")

    # right: range clusters
    lo, hi = rr.valid_limits(cand["LOWER"].to_numpy(dtype=float), cand["UPPER"].to_numpy(dtype=float))
    prof = res["profiles"]
    pl = np.log10(prof["MEDIAN_LOWER"].to_numpy(dtype=float))
    ph = np.log10(prof["MEDIAN_UPPER"].to_numpy(dtype=float))
    llo = np.log10(lo)
    lhi = np.log10(hi)
    allx = np.concatenate([llo[np.isfinite(llo)], pl[np.isfinite(pl)]])
    ally = np.concatenate([lhi[np.isfinite(lhi)], ph[np.isfinite(ph)]])
    xmin = (allx.min() - 0.4) if len(allx) else -1
    ymin = (ally.min() - 0.4) if len(ally) else -1
    x = np.where(np.isfinite(llo), llo, xmin + 0.1)     # one-sided upper -> left edge
    y = np.where(np.isfinite(lhi), lhi, ymin + 0.1)     # one-sided lower -> bottom edge
    ranged = np.isfinite(llo) | np.isfinite(lhi)
    lab = np.where(decided, group, -1)
    import pandas as pd
    df = pd.DataFrame({"x": np.round(x[ranged], 4), "y": np.round(y[ranged], 4), "g": lab[ranged]})
    agg = df.groupby(["x", "y", "g"]).size().reset_index(name="n")
    if len(agg):
        smax = agg["n"].max()
        for g in [-1] + groups:
            s = agg[agg["g"] == g]
            if len(s):
                a2.scatter(s["x"], s["y"], s=12 + 300 * np.sqrt(s["n"] / smax),
                           color=colors.get(g, NONE), alpha=0.75, edgecolor=SURFACE, linewidth=1,
                           label=(f"{rep[g]}" if g >= 0 else "no decision"))
    tol = np.log10(res.get("margin", 1.5))
    for i, r in prof.iterrows():
        if not r["USE"]:
            continue
        g = res["unit_group"].get(r["UNIT"])
        c = colors.get(g, NONE)
        px = pl[i] if np.isfinite(pl[i]) else xmin + 0.1
        py = ph[i] if np.isfinite(ph[i]) else ymin + 0.1
        a2.plot(px, py, marker="x", color=TEXT, ms=9, mew=2)
        a2.add_patch(plt.Rectangle((px - tol, py - tol), 2 * tol, 2 * tol, fill=False,
                                   edgecolor=c, lw=1.5, ls="--"))
        a2.annotate(f" {r['UNIT']}", (px, py), color=TEXT, fontsize=8)
    a2.set_xlabel("log10(lower limit)   (left edge: no lower)", color=TEXT2, fontsize=9)
    a2.set_ylabel("log10(upper limit)   (bottom edge: no upper)", color=TEXT2, fontsize=9)
    a2.set_title("Candidate ranges vs unit profiles (x, margin box)", color=TEXT, fontsize=10, loc="left")
    if len(agg):
        a2.legend(fontsize=8, frameon=False, labelcolor=TEXT, markerscale=0.5)

    fig.suptitle(f"{name}   ranges: {dec['decision']}  {rr.fmt_shares(dec['shares'])}  "
                 f"coverage {dec['n_ranged'] / max(dec['n'], 1):.0%}{title_extra}",
                 color=TEXT, fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=110, facecolor=SURFACE)
    plt.close(fig)
    return out_path
