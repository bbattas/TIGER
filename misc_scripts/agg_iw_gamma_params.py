from PIL import Image
import glob
import os
import argparse
import re
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
from scipy.stats import linregress
# Temp for paper1 stuff
import logging
import os
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from itertools import product
from pathlib import Path
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.colors as mcolors


p = argparse.ArgumentParser(
        description="Checking new gbe/kappa/mu for IW and gamma balance.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
p.add_argument("--plot", action="store_true",
                      help="Save heatmaps of variable changes.")
args = p.parse_args()


def poly_g(g):
    g2 = g * g
    a1 = -3.0944
    a2 = -1.8169
    a3 = 10.323
    a4 = -8.1819
    a5 = 2.0033
    poly = (((a1 * g2 + a2) * g2 + a3) * g2 + a4) * g2 + a5
    return poly

def f0(g):
    pol = poly_g(g)
    f = (((((0.0788 * pol - 0.4955) * pol + 1.2244) * pol - 1.5281) * pol + 1.0686) * pol - 0.5563) * pol + 0.2907
    return f

def f0p(pg):
    pol = pg
    f = (((((0.0788 * pol - 0.4955) * pol + 1.2244) * pol - 1.5281) * pol + 1.0686) * pol - 0.5563) * pol + 0.2907
    return f

def gamma_iw(gbe,kappa=2.07337e7,m_base=5.521269e6):
    g = gbe / (np.sqrt(kappa * m_base))
    # bad = (g < G_MIN) | (g > G_MAX)
    pg = poly_g(g)
    gamma = 1 / pg
    f0_iw = f0p(pg)
    iw = (np.sqrt(kappa / m_base)) * (np.sqrt(1 / f0_iw))
    bad = (g < G_MIN) | (g > G_MAX) | (gamma < GAMMA_MIN) | (gamma > GAMMA_MAX)
    return gamma, iw, bad




plt.rcParams["lines.markersize"] = 2


G_MIN = 0.098546
G_MAX = 0.765691
GAMMA_MIN = 0.53
GAMMA_MAX = 40.0

# L0 = 1.15382e-6
# gbe_iso = 4.60748e6
# kappa = 2.07337e7
# m_base = 5.521269e6

def iw_param_space(gbe, kappa, m):
    """
    Evaluate iw and validity over all combinations of gbe, kappa, m,
    and for gbe scaled by [0.3, 1.0].
    Parameters
    ----------
    gbe : array-like
        Base grain boundary energy values
    kappa : array-like
        Kappa values
    m : array-like
        Mobility values
    Returns
    -------
    iw_vals : ndarray
        Shape (ngbe, nkappa, nm, 2), where last dim corresponds to
        gbe scaling factors [0.3, 1.0]
    bad_vals : ndarray
        Same shape as iw_vals, boolean
    iw_max : ndarray
        Shape (ngbe, nkappa, nm), max iw across the two gbe scalings
    iw_min : ndarray
        Shape (ngbe, nkappa, nm), min iw across the two gbe scalings
    any_bad : ndarray
        Shape (ngbe, nkappa, nm), True if either scaled gbe is invalid
    gamma_vals : ndarray
        Shape (ngbe, nkappa, nm, 2), gamma values for each scaling
    gamma_max : ndarray
        Shape (ngbe, nkappa, nm), max gamma across the two gbe scalings
    gamma_min : ndarray
        Shape (ngbe, nkappa, nm), min gamma across the two gbe scalings
    """
    gbe = np.asarray(gbe, dtype=float)
    kappa = np.asarray(kappa, dtype=float)
    m = np.asarray(m, dtype=float)
    # scaling factors for the gbe range to check
    pair = np.array([0.3, 1.0], dtype=float)
    # reshape for broadcasting:
    # gbe   -> (ngbe, 1, 1, 1)
    # kappa -> (1, nkappa, 1, 1)
    # m     -> (1, 1, nm, 1)
    # pair  -> (1, 1, 1, 2)
    gbe4 = gbe[:, None, None, None]
    kappa4 = kappa[None, :, None, None]
    m4 = m[None, None, :, None]
    pair4 = pair[None, None, None, :]
    gbe_eff = gbe4 * pair4
    g = gbe_eff / np.sqrt(kappa4 * m4)
    pg = poly_g(g)
    gamma = 1.0 / pg
    f0_iw = f0p(pg)
    iw_vals = np.sqrt(kappa4 / m4) * np.sqrt(1.0 / f0_iw)
    bad_vals = (
        (g < G_MIN) | (g > G_MAX) |
        (gamma < GAMMA_MIN) | (gamma > GAMMA_MAX)
    )
    # summarize over the two gbe scale endpoints
    iw_max = np.max(iw_vals, axis=-1)
    iw_min = np.min(iw_vals, axis=-1)
    any_bad = np.any(bad_vals, axis=-1)
    gamma_vals = gamma
    gamma_max = np.max(gamma_vals, axis=-1)
    gamma_min = np.min(gamma_vals, axis=-1)
    return iw_vals, bad_vals, iw_max, iw_min, any_bad, gamma_vals, gamma_max, gamma_min







# # DETERMINE NEW L
# gbe_new = 1.388384e+07
# kappa_new = 2.590909e+07
# m_new = 1.282828e+07
# mgb = kappa * L0 / gbe_iso
# newL = mgb * gbe_new / kappa_new
# print(' ')
# print('Single param set:')
# print(f'gbe = {gbe_new}, kappa = {kappa_new}, m = {m_new}')
# print(f'L0 = {newL}')
# print(f'iso IW = {old_iw_min(1,gbe_new,kappa_new,m_new)}')






# Central values
gbe_c   = 1.388384e+07
kappa_c = 2.590909e+07
m_c     = 1.282828e+07
gamma_03, iw_03, bad_03 = gamma_iw(gbe_c*0.3,kappa_c,m_c)
gamma_1, iw_1, bad_1 = gamma_iw(gbe_c,kappa_c,m_c)
print(f'Central/Old values: ')
print(f'IW min: {iw_1:.2f}, IW max: {iw_03:.2f}')
print(f'gamma min: {gamma_03:.2f}, gamma max: {gamma_1:.2f}')

# Build ranges (e.g. ±50% around central, 30 points each)
n = 30
gbe_range   = np.linspace(gbe_c   * 0.5, gbe_c   * 1.5, n)
kappa_range = np.linspace(kappa_c * 0.5, kappa_c * 1.5, n)
m_range     = np.linspace(m_c     * 0.5, m_c     * 1.5, n)

# Find indices closest to central values (for fixing the third axis)
gbe_ci   = np.argmin(np.abs(gbe_range   - gbe_c))
kappa_ci = np.argmin(np.abs(kappa_range - kappa_c))
m_ci     = np.argmin(np.abs(m_range     - m_c))

# Run the full parameter space sweep
iw_vals, bad_vals, iw_max, iw_min, any_bad, gamma_vals, gamma_max, gamma_min = \
    iw_param_space(gbe_range, kappa_range, m_range)

# ── Plot helpers ──────────────────────────────────────────────────────────────

def make_plot(ax, data, mask, xlabel, ylabel, title, cmap="viridis"):
    """Plot a single heatmap, masking invalid cells."""
    masked = np.ma.array(data, mask=mask)
    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad(color="lightgray")
    im = ax.imshow(
        masked,
        origin="lower",
        aspect="auto",
        cmap=cmap_obj,
    )
    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    return im

def axis_ticks(ax, xvals, yvals, nticks=5):
    """Set readable tick labels from the actual parameter arrays."""
    xi = np.linspace(0, len(xvals) - 1, nticks, dtype=int)
    yi = np.linspace(0, len(yvals) - 1, nticks, dtype=int)
    ax.set_xticks(xi)
    ax.set_xticklabels([f"{xvals[i]:.2e}" for i in xi], rotation=30, ha="right", fontsize=7)
    ax.set_yticks(yi)
    ax.set_yticklabels([f"{yvals[i]:.2e}" for i in yi], fontsize=7)

def make_figure(slices, masks, titles, row_labels, col_labels, suptitle):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle(suptitle, fontsize=13, fontweight="bold")
    xvals, yvals = col_labels
    for r in range(2):
        for c in range(2):
            make_plot(axes[r, c], slices[r][c].T, masks[r][c].T,
                      xvals[0], yvals[0], titles[r][c])
            axis_ticks(axes[r, c], xvals[1], yvals[1])
    plt.tight_layout()
    return fig


if args.plot:
    # ── Plot 1: gbe vs kappa  (m fixed at m_c) ───────────────────────────────────
    # iw_min/max shape: (ngbe, nkappa, nm) → slice [:, :, m_ci]
    # gamma_min/max shape same

    fig1 = make_figure(
        slices=[[iw_min[:, :, m_ci],    iw_max[:, :, m_ci]],
                [gamma_min[:, :, m_ci], gamma_max[:, :, m_ci]]],
        masks=[[any_bad[:, :, m_ci]]*2,
            [any_bad[:, :, m_ci]]*2],
        titles=[["iw min", "iw max"], ["gamma min", "gamma max"]],
        row_labels=None,
        col_labels=(("gbe", gbe_range), ("kappa", kappa_range)),
        suptitle=f"gbe vs kappa  |  m = {m_c:.3e} (fixed)",
    )
    fig1.savefig("P01_gbe_vs_kappa.png", dpi=150, bbox_inches="tight")
    plt.close(fig1)

    # ── Plot 2: gbe vs m  (kappa fixed at kappa_c) ───────────────────────────────

    fig2 = make_figure(
        slices=[[iw_min[:, kappa_ci, :],    iw_max[:, kappa_ci, :]],
                [gamma_min[:, kappa_ci, :], gamma_max[:, kappa_ci, :]]],
        masks=[[any_bad[:, kappa_ci, :]]*2,
            [any_bad[:, kappa_ci, :]]*2],
        titles=[["iw min", "iw max"], ["gamma min", "gamma max"]],
        row_labels=None,
        col_labels=(("gbe", gbe_range), ("m", m_range)),
        suptitle=f"gbe vs m  |  kappa = {kappa_c:.3e} (fixed)",
    )
    fig2.savefig("P02_gbe_vs_m.png", dpi=150, bbox_inches="tight")
    plt.close(fig2)


    # ── Plot 3: kappa vs m  (gbe fixed at gbe_c) ─────────────────────────────────

    fig3 = make_figure(
        slices=[[iw_min[gbe_ci, :, :],    iw_max[gbe_ci, :, :]],
                [gamma_min[gbe_ci, :, :], gamma_max[gbe_ci, :, :]]],
        masks=[[any_bad[gbe_ci, :, :]]*2,
            [any_bad[gbe_ci, :, :]]*2],
        titles=[["iw min", "iw max"], ["gamma min", "gamma max"]],
        row_labels=None,
        col_labels=(("kappa", kappa_range), ("m", m_range)),
        suptitle=f"kappa vs m  |  gbe = {gbe_c:.3e} (fixed)",
    )
    fig3.savefig("P03_kappa_vs_m.png", dpi=150, bbox_inches="tight")
    plt.close(fig3)


# ── Selection / scoring ───────────────────────────────────────────────────────

# Targets
IW_TARGET    = 10.0
GAMMA_TARGET = 10.0
IW_MIN_TARGET = 4

# Only consider valid points
valid_mask = ~any_bad  # shape (ngbe, nkappa, nm)

# # Normalised distance from each target (0 = at target, grows beyond)
# iw_score    = iw_max    / IW_TARGET
# gamma_score = gamma_max / GAMMA_TARGET

# # Combined score: geometric mean keeps neither from dominating
# # Lower is better; scores < 1 mean we're under the target
# combined_score = np.sqrt(iw_score * gamma_score)

# How hard to penalise exceeding each target — tune these to shift the balance
IW_WEIGHT    = 1.0
GAMMA_WEIGHT = 1.0

# Excess above target (0 if at or below target, positive if above)
iw_excess    = np.maximum(0, iw_max    - IW_TARGET)    / IW_TARGET
gamma_excess = np.maximum(0, gamma_max - GAMMA_TARGET) / GAMMA_TARGET

# Weighted sum of excesses — lower is better
primary_score = IW_WEIGHT * iw_excess + GAMMA_WEIGHT * gamma_excess

# Tiebreaker: among zero-score points, rank by iw_min largest first
# Subtract so larger iw_min → smaller tiebreaker → ranks higher
iw_min_tiebreak = -iw_min / IW_TARGET

combined_score = primary_score + 1e-6 * iw_min_tiebreak

# Mask out invalid points so they never rank
combined_score_masked = np.where(valid_mask, combined_score, np.inf)

# ── Retrieve top N ────────────────────────────────────────────────────────────

N_TOP = 30
flat_idx = np.argsort(combined_score_masked, axis=None)[:N_TOP]
idx_3d   = np.unravel_index(flat_idx, combined_score_masked.shape)

print(f"{'Rank':>4}  {'gbe':>12}  {'kappa':>12}  {'m':>12}  "
      f"{'iw_min':>10}  {'iw_max':>10}  {'gamma_min':>10}  {'gamma_max':>10}  {'score':>8}")
print("-" * 100)

results = []
for rank, (gi, ki, mi) in enumerate(zip(*idx_3d), start=1):
    row = dict(
        rank      = rank,
        gbe       = gbe_range[gi],
        kappa     = kappa_range[ki],
        m         = m_range[mi],
        iw_min    = iw_min[gi, ki, mi],
        iw_max    = iw_max[gi, ki, mi],
        gamma_min = gamma_min[gi, ki, mi],
        gamma_max = gamma_max[gi, ki, mi],
        score     = combined_score_masked[gi, ki, mi],
    )
    results.append(row)
    print(f"{rank:>4}  {row['gbe']:>12.3e}  {row['kappa']:>12.3e}  {row['m']:>12.3e}  "
          f"{row['iw_min']:>10.4f}  {row['iw_max']:>10.4f}  "
          f"{row['gamma_min']:>10.4f}  {row['gamma_max']:>10.4f}  {row['score']:>8.4f}")




# ── Calculate new L0 ────────────────────────────────────────────────────────────

def old_iw_min(f,gbe,kappa,m):
    gbe2 = f * gbe
    g = gbe2 / (np.sqrt(kappa * m))
    pg = poly_g(g)
    gamma = 1 / pg
    f0_iw = f0p(pg)
    iw = (np.sqrt(kappa / m)) * (np.sqrt(1 / f0_iw))
    bad = (g < G_MIN) | (g > G_MAX) | (gamma < GAMMA_MIN) | (gamma > GAMMA_MAX)
    return iw, bad

# Rank           gbe         kappa             m      iw_min      iw_max   gamma_min   gamma_max     score
# ----------------------------------------------------------------------------------------------------
# 382     1.125e+07     2.100e+07     1.305e+07      2.5791      8.8875      0.5950      9.3897   -0.0000

# OLD VALUES
L0 = 1.15382e-6
gbe_iso = 4.60748e6
kappa = 2.07337e7
m_base = 5.521269e6


gbe_new = 1.125e7
kappa_new = 2.100e7
m_new = 1.305e7

mgb = kappa * L0 / gbe_iso
newL = mgb * gbe_new / kappa_new
print(' ')
print('New param set:')
print(f'gbe = {gbe_new:.3e}, kappa = {kappa_new:.3e}, m = {m_new:.3e}')
print(f'L0 = {newL}')
print(f'iso IW = {old_iw_min(1,gbe_new,kappa_new,m_new)}')
gamma_03, iw_03, bad_03 = gamma_iw(gbe_new*0.3,kappa_new,m_new)
gamma_1, iw_1, bad_1 = gamma_iw(gbe_new,kappa_new,m_new)
print(f'Central/Old values: ')
print(f'IW min: {iw_1:.2f}, IW max: {iw_03:.2f}')
print(f'gamma min: {gamma_03:.2f}, gamma max: {gamma_1:.2f}')
