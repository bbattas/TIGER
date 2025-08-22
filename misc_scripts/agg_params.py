#!/usr/bin/env python3
import argparse
import glob
import json
from pathlib import Path
import sys
from typing import List, Tuple, Dict, Any
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from itertools import product
from tqdm import tqdm


class default_vals:
    L = 1.15382e-6
    gbe = 4.60748e6
    kappa = 2.07337e7
    m = 4.5e6


def base_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Check the parameter space for varying levels of anisotropy."
    )
    p.add_argument("--gbe", type=float, default=default_vals.gbe, help="Iso GB Energy.")
    p.add_argument("--kappa", type=float, default=default_vals.kappa, help="Kappa.")
    p.add_argument("--m","-m", type=float, default=default_vals.m, help="FE constant m.")
    p.add_argument("--multi", action="store_true", help="Parametric analysis of max aniso ceneterd around input values.")
    p.add_argument("--maxm", action="store_true", help="Find the m value(s) for maximum aniso for given kappa and GBE.")
    # Make all options non-required here; we'll enforce presence after merging JSON+CLI.
    # p.add_argument("--input_pattern","-i", help="Substring or glob for the input file.")
    # p.add_argument("--max-x","-x", type=float, help="Maximum allowed x (inclusive).")
    # p.add_argument("--max-y","-y", type=float, help="Maximum allowed y (inclusive).")
    # p.add_argument("--min-x", type=float, default=None, help="Minimum allowed x (inclusive).")
    # p.add_argument("--min-y", type=float, default=None, help="Minimum allowed y (inclusive).")
    # p.add_argument("-o","--output", help="Output filename ('.txt' added automatically if omitted).")
    p.add_argument("--plot", action="store_true", help="Show plots.")
    p.add_argument("--save", metavar="PATH", help="Save plots to file (implies --plot).")
    # p.add_argument("--json","-j", action="store_true",
    #                help=f"Use {JSON_NAME} if present; create it if missing. CLI values override JSON.")
    # p.add_argument("--decimals","-d", type=int, default=None,
    #                help="Number of decimal places to write in the output file (default: 5).")
    return p

# def parser_defaults_dict(p: argparse.ArgumentParser) -> Dict[str, Any]:
#     # Pull argparse defaults into a dict we can merge with JSON/CLI
#     defaults = {}
#     for a in p._actions:
#         if a.dest != "help":
#             defaults[a.dest] = a.default
#     return defaults

# def load_json_config() -> Dict[str, Any] | None:
#     path = Path(JSON_NAME)
#     if not path.exists():
#         return None
#     try:
#         with path.open("r", encoding="utf-8") as f:
#             data = json.load(f)
#         if not isinstance(data, dict):
#             raise ValueError("JSON root must be an object")
#         return data
#     except Exception as e:
#         sys.exit(f"[error] Failed to read {JSON_NAME}: {e}")

# def save_json_config(cfg: Dict[str, Any]) -> None:
#     # Never store the json flag itself
#     cfg = {k: v for k, v in cfg.items() if k != "json"}
#     try:
#         with Path(JSON_NAME).open("w", encoding="utf-8") as f:
#             json.dump(cfg, f, indent=2, sort_keys=True)
#         print(f"[info] Wrote {JSON_NAME}")
#     except Exception as e:
#         sys.exit(f"[error] Failed to write {JSON_NAME}: {e}")

# def merge_config(cli: Dict[str, Any], json_cfg: Dict[str, Any] | None, defaults: Dict[str, Any]) -> Dict[str, Any]:
#     # Start with defaults, then JSON, then CLI (CLI wins on non-None/non-False values)
#     cfg = dict(defaults)
#     if json_cfg:
#         cfg.update({k: v for k, v in json_cfg.items() if k in cfg})
#     # Apply CLI overrides: treat None/False/"" as "not provided" except booleans explicitly True
#     for k, v in cli.items():
#         if k not in cfg:
#             continue
#         if isinstance(v, bool):
#             if v:  # only override if True, to avoid stomping JSON with False-by-default
#                 cfg[k] = v
#         else:
#             if v is not None:
#                 cfg[k] = v
#     return cfg

# def require_fields(cfg: Dict[str, Any], fields: List[str]) -> None:
#     missing = [f for f in fields if cfg.get(f) in (None, "")]
#     if missing:
#         sys.exit(f"[error] Missing required option(s): {', '.join(missing)} "
#                  f"(provide via CLI or {JSON_NAME})")

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
    g2 = g * g
    pol = poly_g(g)
    f = (((((0.0788 * pol - 0.4955) * pol + 1.2244) * pol - 1.5281) * pol + 1.0686) * pol - 0.5563) * pol + 0.2907
    return f

def solve_poly(poly_val):
    a1, a2, a3, a4, a5 = -3.0944, -1.8169, 10.323, -8.1819, 2.0033
    coeffs = [a1, a2, a3, a4, a5 - poly_val]
    roots = np.roots(coeffs)
    # Filter real solutions
    real_roots = [r.real for r in roots if abs(r.imag) < 1e-8]
    return real_roots

G_MIN = 0.098546
G_MAX = 0.765691
GAMMA_MIN = 0.5
GAMMA_MAX = 40.0

def clip_gamma_by_g(gamma, g,
                    gmin=G_MIN, gmax=G_MAX,
                    gamma_min=GAMMA_MIN, gamma_max=GAMMA_MAX):
    """
    Clamp gamma ONLY where g is out of [gmin, gmax].
    Returns: gamma_clipped, mask_out, mask_low, mask_high
    """
    g = np.asarray(g)
    gamma = np.asarray(gamma)
    mask_low  = (g < gmin)
    mask_high = (g > gmax)
    mask_out  = mask_low | mask_high

    gamma_clipped = gamma.copy()
    gamma_clipped[mask_low]  = gamma_min   # <-- set to lower limit
    gamma_clipped[mask_high] = gamma_max   # <-- set to upper limit

    return gamma_clipped, mask_out, mask_low, mask_high

def contiguous_runs(mask):
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0:
        return []
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    # breaks where the index jumps by > 1
    breaks = np.where(np.diff(idx) > 1)[0]
    starts = np.r_[0, breaks + 1]
    ends   = np.r_[breaks, idx.size - 1]
    return [(idx[s], idx[e]) for s, e in zip(starts, ends)]


def calculate_single_aniso(args):
    a_range = np.linspace(0,1,1000)
    anis = np.stack([1 - a_range, 1 + a_range], axis=-1)
    MIN, MAX = 0, 1

    gbe = args.gbe * anis # (n,2)
    g = gbe / (np.sqrt(args.kappa * args.m))
    g2 = g * g
    polyg = poly_g(g)
    gamma = 1 / polyg
    f0_int = f0(g)
    iw = (np.sqrt(args.kappa / args.m)) * (np.sqrt(1 / f0_int))

    # Put it in a DF
    df = pd.DataFrame({
        "a": a_range,
        "gbe_min": gbe[:, MIN], "gbe_max": gbe[:, MAX],
        "g_min": g[:, MIN], "g_max": g[:, MAX],
        "g2_min": g2[:, MIN], "g2_max": g2[:, MAX],
        "polyg_min": polyg[:, MIN], "polyg_max": polyg[:, MAX],
        "gamma_min": gamma[:, MIN], "gamma_max": gamma[:, MAX],
        "iw_min": iw[:, MIN], "iw_max": iw[:, MAX],
    })
    # Valid parameter space
    ok = (
        (df["g_min"] >= G_MIN) & (df["g_max"] <= G_MAX) &
        (df["gamma_min"] > GAMMA_MIN) & (df["gamma_max"] < GAMMA_MAX)
    )
    bad = ~ok
    max_valid_a = 0
    if ok.any():
        max_valid_a = df.loc[ok, "a"].max()
        print(f"Maximum anisotropy with valid g & gamma: {max_valid_a:.4f}")
    else:
        print("No anisotropy values are fully valid within the specified ranges.")

    # Plots
    # My own version simpler
    fig, ax = plt.subplots()
    ax.fill_between(df["a"][ok], df["gamma_min"][ok], df["gamma_max"][ok],
                alpha=0.3, label="Gamma band (in-bounds)")
    ax.fill_between(df["a"][bad], 0.5,40,#df["gamma_min"][bad], df["gamma_max"][bad],
                alpha=0.3,color="red", label="Gamma band (out-bounds)")
    ax.plot(df["a"][ok], 0.5*(df["gamma_min"][ok]+df["gamma_max"][ok]), label="Gamma center")

    ax.axhline(0.5,  ls="--", lw=1, color="k", alpha=0.4)
    ax.axhline(40.,  ls="--", lw=1, color="k", alpha=0.4)

    ax.set_xlabel("Anisotropy Magnitude +/-")
    ax.set_ylabel("Gamma")
    ax.legend()
    plt.tight_layout()
    if args.save is not None:
        plt.savefig(args.save+'_gamma_valid_single')
    if args.plot:
        plt.show()
    plt.close('all')

    # IW my way
    fig, ax = plt.subplots()
    ax.fill_between(df["a"][ok], df["iw_min"][ok], df["iw_max"][ok],
                alpha=0.3, label="IW band (in-bounds)")
    ax.fill_between(df["a"][bad], df["iw_min"][bad], df["iw_max"][bad],
                alpha=0.3,color="red", label="IW band (out-bounds)")
    ax.plot(df["a"], 0.5*(df["iw_min"]+df["iw_max"]), label="IW center")

    ax.set_xlabel("Anisotropy Magnitude +/-")
    ax.set_ylabel("IW")
    ax.legend()
    plt.tight_layout()
    if args.save is not None:
        plt.savefig(args.save+'_IW_valid_single')
    if args.plot:
        plt.show()
    plt.close('all')

    return max_valid_a


def minimal_a_calc(gbe,kappa,m,n=100):
    a_range = np.linspace(0,1,n)
    anis = np.stack([1 - a_range, 1 + a_range], axis=-1)
    MIN, MAX = 0, 1

    gbe = gbe * anis # (n,2)
    g = gbe / (np.sqrt(kappa * m))
    # g2 = g * g
    polyg = poly_g(g)
    gamma = 1 / polyg
    f0_int = f0(g)
    iw = (np.sqrt(kappa / m)) * (np.sqrt(1 / f0_int))

    # Put it in a DF
    df = pd.DataFrame({
        "a": a_range,
        # "gbe_min": gbe[:, MIN], "gbe_max": gbe[:, MAX],
        "g_min": g[:, MIN], "g_max": g[:, MAX],
        # "g2_min": g2[:, MIN], "g2_max": g2[:, MAX],
        # "polyg_min": polyg[:, MIN], "polyg_max": polyg[:, MAX],
        "gamma_min": gamma[:, MIN], "gamma_max": gamma[:, MAX],
        # "iw_min": iw[:, MIN], "iw_max": iw[:, MAX],
    })
    # Valid parameter space
    ok = (
        (df["g_min"] >= G_MIN) & (df["g_max"] <= G_MAX) &
        (df["gamma_min"] > GAMMA_MIN) & (df["gamma_max"] < GAMMA_MAX)
    )
    max_valid_a = 0
    if ok.any():
        max_valid_a = df.loc[ok, "a"].max()

    return max_valid_a

def _valid_at_a(gbe, kappa, m, a):
    """
    Return True/False if BOTH min/max branches at this 'a' are within g and gamma bounds.
    """
    # aniso multipliers for min/max branches
    anis = np.array([1 - a, 1 + a])  # shape (2,)
    gbepm = gbe * anis               # (2,)
    g = gbepm / np.sqrt(kappa * m)   # (2,)

    # g bounds must hold on both branches
    if not (g[0] >= G_MIN and g[1] <= G_MAX):
        return False

    # gamma from poly_g(g) = 1 / P(g)
    polyg = poly_g(g)
    gamma = 1.0 / polyg              # (2,)

    # gamma bounds must hold on both branches
    if not (gamma[0] > GAMMA_MIN and gamma[1] < GAMMA_MAX):
        return False

    return True

def minimal_a_bisect(gbe, kappa, m, tol=1e-4, max_iter=128):
    lo, hi = 0.0, 1.0

    # If even a=0 is invalid, return 0 right away
    if not _valid_at_a(gbe, kappa, m, 0.0):
        return 0.0

    # If a=1 is valid (unlikely), then 1.0 is the answer
    if _valid_at_a(gbe, kappa, m, 1.0):
        return 1.0

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if _valid_at_a(gbe, kappa, m, mid):
            lo = mid  # mid is valid; push upward
        else:
            hi = mid  # mid invalid; move down
        if (hi - lo) <= tol:
            break
    return lo

# def calculate_multi_aniso(args, oom=2):
def calculate_multi_aniso(args, oom=2, per_axis=9, method="bisect", outfile=None):
    """
    Sweep gbe, kappa, m over ±oom orders of magnitude around args.* values.
    per_axis: number of points per parameter (>=2 recommended).
    method: "bisect" (fast) or "grid" (uses minimal_a_calc).
    """
    # Base values
    base_gbe   = float(getattr(args, "gbe",   default_vals.gbe))
    base_kappa = float(getattr(args, "kappa", default_vals.kappa))
    base_m     = float(getattr(args, "m",     default_vals.m))

    factors = np.logspace(-oom, +oom, per_axis)  # multiplicative factors
    total   = len(factors) ** 3   # total combos

    rows = []
    use_bisect = (method.lower() == "bisect")
    for fg, fk, fm in tqdm(product(factors, factors, factors), total=total, desc='Sweeping parameter space'):
        gbe   = base_gbe   * fg
        kappa = base_kappa * fk
        m     = base_m     * fm

        if use_bisect:
            max_a = minimal_a_bisect(gbe, kappa, m)
        else:
            max_a = minimal_a_calc(gbe, kappa, m, n=600)  # smaller n to start

        rows.append({
            "gbe": gbe, "kappa": kappa, "m": m,
            "f_gbe": fg, "f_kappa": fk, "f_m": fm,
            "max_valid_a": max_a,
        })

    df = pd.DataFrame(rows)

    if outfile:
        Path(outfile).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(outfile, index=False)

    return df


# PLOTTING
def _edges_from_centers(v):
    """Log-safe bin edges from positive center points v (ascending)."""
    v = np.asarray(v, float)
    assert np.all(v > 0), "All centers must be positive for log edges."
    logv = np.log(v)
    edges = np.empty(v.size + 1)
    edges[1:-1] = np.exp(0.5 * (logv[1:] + logv[:-1]))
    edges[0]    = np.exp(logv[0] - (logv[1] - logv[0]) / 2)
    edges[-1]   = np.exp(logv[-1] + (logv[-1] - logv[-2]) / 2)
    return edges

def plot_max_a_heatmap_for_m(df_sweep, m_target, ax=None):
    """
    df_sweep: output of calculate_multi_aniso(...), with columns:
              ['gbe','kappa','m','max_valid_a', ...]
    m_target: the m value you'd like to slice at (choose nearest in log space)
    """
    # choose nearest m (log distance)
    m_unique = np.array(sorted(df_sweep["m"].unique()))
    i = np.argmin(np.abs(np.log(m_unique / m_target)))
    m_sel = m_unique[i]

    sub = df_sweep[df_sweep["m"] == m_sel].copy()

    # pivot to 2D grid (kappa x gbe)
    gbe_vals   = np.array(sorted(sub["gbe"].unique()))
    kappa_vals = np.array(sorted(sub["kappa"].unique()))
    P = sub.pivot(index="kappa", columns="gbe", values="max_valid_a")
    P = P.reindex(index=kappa_vals, columns=gbe_vals)  # ensure sorted axes

    Z = P.values  # shape (len(kappa_vals), len(gbe_vals))

    if ax is None:
        fig, ax = plt.subplots()

    # build log-space cell edges for pcolormesh
    xe = _edges_from_centers(gbe_vals)
    ye = _edges_from_centers(kappa_vals)

    # mask NaNs (no valid result for that cell)
    Zm = np.ma.masked_invalid(Z)

    pcm = ax.pcolormesh(xe, ye, Zm, shading="auto")
    cbar = plt.colorbar(pcm, ax=ax)
    cbar.set_label("Max valid anisotropy (a*)")

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("gbe")
    ax.set_ylabel("kappa")
    ax.set_title(f"Max valid a* at m ≈ {m_sel:.3e} (target {m_target:.3e})")

    # optional: mark cells with zero (no valid anisotropy) as red dots
    # r, c = np.where(np.nan_to_num(Z, nan=-1.0) == 0.0)
    # ax.plot(gbe_vals[c], kappa_vals[r], 'o', ms=3, color='red', alpha=0.7)

    return ax



# COMBINED K and M
def _valid_at_a_km(gbe, km, a):
    """
    True if BOTH min/max branches are within g & gamma bounds at this 'a'.
    """
    anis = np.array([1 - a, 1 + a])     # (2,)
    g = (gbe * anis) / np.sqrt(km)      # (2,)

    if not (g[0] >= G_MIN and g[1] <= G_MAX):
        return False

    polyg = poly_g(g)                   # expects vectorized input
    gamma = 1.0 / polyg

    if not (gamma[0] > GAMMA_MIN and gamma[1] < GAMMA_MAX):
        return False

    return True


def minimal_a_bisect_km(gbe, km, tol=1e-4, max_iter=128):
    """
    Fast: find the largest a in [0,1] that remains valid, using bisection.
    """
    lo, hi = 0.0, 1.0
    if not _valid_at_a_km(gbe, km, 0.0):
        return 0.0
    if _valid_at_a_km(gbe, km, 1.0):
        return 1.0

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if _valid_at_a_km(gbe, km, mid):
            lo = mid
        else:
            hi = mid
        if (hi - lo) <= tol:
            break
    return lo

def calculate_multi_km(args, oom=2, per_axis=9, method="bisect", outfile=None):
    """
    Sweep gbe and km over ±oom orders around base values.
    km base = args.kappa * args.m

    Returns a DataFrame with columns:
      ['gbe','km','f_gbe','f_km','max_valid_a']
    """
    base_gbe = float(getattr(args, "gbe",   default_vals.gbe))
    base_k   = float(getattr(args, "kappa", default_vals.kappa))
    base_m   = float(getattr(args, "m",     default_vals.m))
    base_km  = base_k * base_m

    f = np.logspace(-oom, +oom, per_axis)
    total = len(f) ** 2

    rows = []
    use_bisect = (method.lower() == "bisect")

    for fg, fkm in tqdm(product(f, f), total=total, desc="Sweep gbe×km"):
        gbe = base_gbe * fg
        km  = base_km  * fkm

        if use_bisect:
            max_a = minimal_a_bisect_km(gbe, km)
        else:
            raise ValueError("calculate_multi_km: Not setup to hadle the calculation instead of bisect")
            max_a = minimal_a_calc_km(gbe, km, n=600)

        rows.append({
            "gbe": gbe, "km": km,
            "f_gbe": fg, "f_km": fkm,
            "max_valid_a": max_a,
        })

    df = pd.DataFrame(rows)
    if outfile:
        Path(outfile).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(outfile, index=False)
    return df


# def plot_max_a_heatmap_km(df, ax=None):
#     """
#     df: output of calculate_multi_km with columns ['gbe','km','max_valid_a']
#     """
#     gbe_vals = np.array(sorted(df["gbe"].unique()))
#     km_vals  = np.array(sorted(df["km"].unique()))
#     P = df.pivot(index="km", columns="gbe", values="max_valid_a")
#     P = P.reindex(index=km_vals, columns=gbe_vals)

#     Z = P.values
#     xe = _edges_from_centers(gbe_vals)
#     ye = _edges_from_centers(km_vals)

#     if ax is None:
#         fig, ax = plt.subplots()

#     pcm = ax.pcolormesh(xe, ye, np.ma.masked_invalid(Z), shading="auto")
#     cbar = plt.colorbar(pcm, ax=ax)
#     cbar.set_label("Max valid anisotropy (a*)")

#     ax.set_xscale("log"); ax.set_yscale("log")
#     ax.set_xlabel("gbe")
#     ax.set_ylabel("km = kappa·m")
#     ax.set_title("Max valid a* over (gbe, kappa*m)")

#     return ax

def plot_max_a_heatmap_km(df, gbe_base=None, km_base=None, ax=None, show_lines=True, show_box=False):
    """
    df: output of calculate_multi_km with ['gbe','km','max_valid_a']
    gbe_base, km_base: base values to highlight
    show_lines: if True, draw dotted hline/vline at base values
    show_box:  if True, draw a rectangle (±half a grid step) around base point
    """
    gbe_vals = np.array(sorted(df["gbe"].unique()))
    km_vals  = np.array(sorted(df["km"].unique()))
    P = df.pivot(index="km", columns="gbe", values="max_valid_a")
    P = P.reindex(index=km_vals, columns=gbe_vals)

    Z = P.values
    xe = _edges_from_centers(gbe_vals)
    ye = _edges_from_centers(km_vals)

    if ax is None:
        fig, ax = plt.subplots()

    pcm = ax.pcolormesh(xe, ye, np.ma.masked_invalid(Z), shading="auto")
    cbar = plt.colorbar(pcm, ax=ax)
    cbar.set_label("Max valid anisotropy (a*)")

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("gbe")
    ax.set_ylabel("km = kappa·m")
    ax.set_title("Max valid a* over (gbe, km)")

    # --- Highlight base values ---
    if gbe_base and km_base:
        if show_lines:
            ax.axvline(gbe_base, ls="--", lw=1, color="k", alpha=0.7)
            ax.axhline(km_base, ls="--", lw=1, color="k", alpha=0.7)
        if show_box:
            # compute nearest grid spacing in log space
            import matplotlib.patches as patches
            dx = np.min(np.diff(np.log(gbe_vals))) / 2
            dy = np.min(np.diff(np.log(km_vals))) / 2
            rect = patches.Rectangle(
                (gbe_base * np.exp(-dx), km_base * np.exp(-dy)),
                gbe_base * (np.exp(dx) - np.exp(-dx)),
                km_base * (np.exp(dy) - np.exp(-dy)),
                linewidth=1.5, edgecolor="red", facecolor="none", ls="--"
            )
            ax.add_patch(rect)

    return ax

# For calculating the best m value to use
def _max_a_at_m(gbe, kappa, m):
    # Objective for maximization: max valid anisotropy at this m
    return float(minimal_a_bisect(gbe, kappa, m))

def _golden_section_max_logm(gbe, kappa, logm_lo, logm_hi, tol=1e-4, max_iter=64):
    """
    Maximize f(log m) := max_a(m) over logm in [logm_lo, logm_hi] using golden-section.
    Returns (logm_star, f_star).
    """
    phi = (1 + np.sqrt(5)) / 2
    invphi = 1 / phi

    a, b = logm_lo, logm_hi
    c = b - invphi * (b - a)
    d = a + invphi * (b - a)

    fc = _max_a_at_m(gbe, kappa, np.exp(c))
    fd = _max_a_at_m(gbe, kappa, np.exp(d))

    for _ in range(max_iter):
        if (b - a) <= tol:
            break
        if fc < fd:
            a, c, fc = c, d, fd
            d = a + invphi * (b - a)
            fd = _max_a_at_m(gbe, kappa, np.exp(d))
        else:
            b, d, fd = d, c, fc
            c = b - invphi * (b - a)
            fc = _max_a_at_m(gbe, kappa, np.exp(c))

    # pick best endpoint
    if fc > fd:
        return c, fc
    else:
        return d, fd

def find_best_m(gbe, kappa, m_base, coarse_pts=41, coarse_oom=3, refine_tol=1e-4):
    """
    1) Coarse scan m in [m_base/10^coarse_oom, m_base*10^coarse_oom] (logspace)
    2) Bracket the max using the best coarse point and its neighbors
    3) Golden-section refine on log(m)
    Returns (m_star, a_star)
    """
    # 1) coarse scan
    m_lo = m_base / (10 ** coarse_oom)
    m_hi = m_base * (10 ** coarse_oom)
    logm_grid = np.linspace(np.log(m_lo), np.log(m_hi), coarse_pts)
    vals = np.array([_max_a_at_m(gbe, kappa, np.exp(Lm)) for Lm in logm_grid])

    best_idx = int(np.nanargmax(vals))
    a_star0 = float(vals[best_idx])

    # If best is at an endpoint, expand a bit (or just refine over the whole interval)
    i0 = max(0, best_idx - 1)
    i1 = min(coarse_pts - 1, best_idx + 1)
    logm_lo = logm_grid[i0]
    logm_hi = logm_grid[i1]
    if logm_hi - logm_lo < 1e-9:  # degenerate (e.g., flat or endpoint)
        logm_lo, logm_hi = logm_grid[0], logm_grid[-1]

    # 2) refine with golden-section on log m
    logm_star, a_star = _golden_section_max_logm(
        gbe, kappa, logm_lo, logm_hi, tol=refine_tol, max_iter=64
    )
    return float(np.exp(logm_star)), float(a_star)

def plot_max_a_vs_m_around(gbe, kappa, m_star, span_oom=1.0, n=61, ax=None):
    """
    Plot max_valid_a vs m over a logspace window around m_star:
    m in [m_star/10^span_oom, m_star*10^span_oom].
    """
    m_vals = np.logspace(np.log10(m_star) - span_oom,
                         np.log10(m_star) + span_oom, n)
    a_vals = [ _max_a_at_m(gbe, kappa, m) for m in tqdm(m_vals, desc="a*(m) line") ]

    if ax is None:
        fig, ax = plt.subplots()

    ax.plot(m_vals, a_vals, lw=2)
    ax.axvline(m_star, ls="--", lw=1, color="k", alpha=0.7)
    ax.set_xscale("log")
    ax.set_xlabel("m")
    ax.set_ylabel("Max valid anisotropy a*")
    ax.set_title(f"a*(m) around m* ≈ {m_star:.3e}")
    return ax, m_vals, a_vals


def main():
    p = base_parser()
    args = p.parse_args()

    # Valid g(gamma) range:
    # Gamma = 0.52 - 40
    # g(gamma) = 0.098546 - 0.765691
    # f0 is 0.004921 - 0.281913
    # g for gamma 0.5 is about 0.0201 (round it up to be safe)
    if args.multi:
        # df_sweep = calculate_multi_aniso(args, oom=2, per_axis=30, method="bisect")
        # #, outfile="multi_aniso.csv")
        # plot_max_a_heatmap_for_m(df_sweep, m_target=default_vals.m)
        # plt.tight_layout()
        # plt.show()
        df_km = calculate_multi_km(args, oom=1, per_axis=333, method="bisect")
        #, outfile="gbe_km_sweep.csv")
        # plot_max_a_heatmap_km(df_km)
        plot_max_a_heatmap_km(df_km, gbe_base=args.gbe, km_base=args.kappa*args.m,
                           show_lines=True, show_box=True)
        plt.tight_layout()
        # plt.show()
        if args.save is not None:
            plt.savefig(args.save+'_multi_paramSpace')
        if args.plot:
            plt.show()
        plt.close('all')

    # else:
    #     calculate_single_aniso(args)

    if args.maxm:
        # Find the m value that corresponds to the maximum anisotropy for the input kappa and gbe
        # Plot a small range of m around that value to show the anisotropy as a function of m
        gbe = float(args.gbe)
        kappa = float(args.kappa)
        m_base = float(args.m)

        # 1) Find the argmax m*
        m_star, a_star = find_best_m(gbe, kappa, m_base,
                                     coarse_pts=41,  # adjust if you want
                                     coarse_oom=3,   # ±3 orders to search broadly
                                     refine_tol=1e-4)
        print(f"Best m ≈ {m_star:.6e} yields a* ≈ {a_star:.6f}")

        # 2) Plot a*(m) over ±1 OOM around m*
        ax, m_vals, a_vals = plot_max_a_vs_m_around(
            gbe, kappa, m_star, span_oom=1.0, n=301
        )
        plt.tight_layout()
        if args.save is not None:
            plt.savefig(args.save + "_amax_vs_m")
        if args.plot:
            plt.show()
        plt.close('all')








if __name__ == "__main__":
    main()
