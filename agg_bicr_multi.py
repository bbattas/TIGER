#!/usr/bin/env python3
"""
bicrystal_contour.py

Read ExodusII results using ExodusBasics and extract the gr0-gr1 interface
contour (x,y coordinates) at a selected timestep.

Two contour definitions (choose with --method):
  - gr0     : contour of nodal variable gr0 at level 0.99
  - moelans : contour of gr0^2 / (gr0^2 + gr1^2) at level 0.5

Outputs a compact .npz by default:
  xy      : (N,2) stacked vertices across all paths
  starts  : (n_paths,) start indices into xy
  lengths : (n_paths,) number of vertices per path
  plus metadata arrays (step, time, level, method, etc.)

Examples:
  python bicrystal_contour.py -t 5.0 --method gr0
  python bicrystal_contour.py --step 40 --method moelans --largest-only
"""
from __future__ import annotations

from vector.ExodusBasics import ExodusBasics

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt


# ---------------------------
# CLI + logging (vector_inclination-style)
# ---------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Extract bicrystal interface contour coordinates from ExodusII output.")

    # ---- General ----
    gen = p.add_argument_group("General")
    gen.add_argument("-v", "--verbose", action="count", default=0,
                     help="Increase verbosity (-v, -vv, -vvv).")
    gen.add_argument("-s", "--subdirs", action="store_true",
                     help="Search for *.e files one level down (./*/.e). "
                          "If not set, only search current directory.")
    gen.add_argument("--pattern", default="*.e",
                     help="Filename glob pattern to search (default: *.e)")
    gen.add_argument("--outdir", type=Path, default=Path("."),
                     help="Output directory (default: current directory).")

    # ---- Time selection ----
    tim = p.add_argument_group("Timestep selection (choose one)")
    grp = tim.add_mutually_exclusive_group(required=True)
    grp.add_argument("-t", "--time", type=float,
                     help="Target time; chooses timestep with time_whole closest to this value.")
    grp.add_argument("--step", type=int,
                     help="Timestep index (0-based). If provided, no closest-time search is used.")

    # ---- Contour definition ----
    cnt = p.add_argument_group("Contour definition")
    cnt.add_argument("--method", choices=("gr0", "moelans"), required=True,
                     help="Contour definition. 'gr0' uses gr0=0.99. "
                          "'moelans' uses gr0^2/(gr0^2+gr1^2)=0.5")
    cnt.add_argument("--level", type=float, default=None,
                     help="Override contour level (default depends on --method).")
    cnt.add_argument("--gr0-name", default="gr0",
                     help="Nodal variable name for grain 0 (default: gr0).")
    cnt.add_argument("--gr1-name", default="gr1",
                     help="Nodal variable name for grain 1 (default: gr1).")

    # ---- Grid mapping + output ----
    out = p.add_argument_group("Grid & output")
    out.add_argument("--tol", type=float, default=None,
                     help="Tolerance for snapping x/y to a structured grid. "
                          "If not provided, choose a conservative default.")
    out.add_argument("--largest-only", action="store_false",
                     help="Keep only the longest contour path (useful if tiny islands exist), default=true.")
    out.add_argument("--save", choices=("npz", "npy", "csv"), default="npz",
                     help="Output format: npz (default), npy (xy only), csv (x,y with path_id).")
    out.add_argument("--no-plot", action="store_true",
                     help="Do not write a quicklook PNG. (By default, writes '<stem>_contour_<method>.png').")

    return p.parse_args()


def setup_logging(verbosity: int) -> logging.Logger:
    level = logging.WARNING
    if verbosity >= 2:
        level = logging.DEBUG
    elif verbosity >= 1:
        level = logging.INFO
    logging.basicConfig(level=level, format="%(message)s")
    return logging.getLogger("TIGER")


def tf(ti, log, extra=None):
    dt = time.perf_counter() - ti
    if extra is None:
        log.warning(f"Time: {dt:.4f}s")
    else:
        log.warning(f"{extra}Time: {dt:.4f}s")


# ---------------------------
# Helpers (file discovery, namebase, timestep choice)
# ---------------------------

def find_exodus_files(*, subdirs: bool = False, pattern: str = "*.e") -> list[Path]:
    cwd = Path.cwd()
    files = sorted(cwd.glob(f"*/{pattern}" if subdirs else pattern))
    return [p for p in files if p.is_file()]


def exodus_stem(exo_path: Path) -> str:
    name = exo_path.name
    if name.endswith(".e"):
        name = name[:-2]
    if name.endswith("_out"):
        name = name[:-4]
    return name


def closest_index(values: np.ndarray, target: float) -> int:
    values = np.asarray(values, dtype=float)
    return int(np.argmin(np.abs(values - target)))


def select_step(exo: ExodusBasics, *, step: int | None, time_value: float | None, log: logging.Logger) -> int:
    times = exo.time()
    if step is not None:
        if step < 0 or step >= len(times):
            raise ValueError(f"--step out of range. Got {step}, but valid is [0, {len(times)-1}]")
        log.info(f"Frame selected by step: step={step}, time={times[step]}")
        return step

    idx = closest_index(times, float(time_value))
    log.info(f"Frame selected by time: requested={time_value}, chosen step={idx}, time={times[idx]}")
    return idx


# ---------------------------
# Grid mapping + contour extraction
# ---------------------------

def _default_tol(x: np.ndarray, y: np.ndarray) -> float:
    scale = max(float(np.ptp(x)), float(np.ptp(y)), 1.0)
    return scale * 1e-12


def map_to_grid_xy(x: np.ndarray, y: np.ndarray, v: np.ndarray,
                   *, tol: float, fill_value=np.nan, reduce: str | None = None):
    """
    Map scattered node values onto a structured (y,x) grid by snapping x and y with tolerance.

    Returns:
      V2d: (ny, nx)
      x_axis: (nx,) sorted unique x coords
      y_axis: (ny,) sorted unique y coords
    """
    x = np.asarray(x)
    y = np.asarray(y)
    v = np.asarray(v)

    kx = np.rint(x / tol).astype(np.int64)
    ky = np.rint(y / tol).astype(np.int64)

    ux, j = np.unique(kx, return_inverse=True)   # x columns
    uy, i = np.unique(ky, return_inverse=True)   # y rows

    x_axis = ux.astype(np.float64) * tol
    y_axis = uy.astype(np.float64) * tol

    V = np.full((len(y_axis), len(x_axis)), fill_value, dtype=np.float64)

    if reduce is None:
        V[i, j] = v
        return V, x_axis, y_axis

    if reduce == "max":
        V[:, :] = -np.inf
        np.maximum.at(V, (i, j), v)
    elif reduce == "min":
        V[:, :] = np.inf
        np.minimum.at(V, (i, j), v)
    elif reduce == "sum":
        V[:, :] = 0.0
        np.add.at(V, (i, j), v)
    elif reduce == "mean":
        V[:, :] = 0.0
        counts = np.zeros_like(V)
        np.add.at(V, (i, j), v)
        np.add.at(counts, (i, j), 1.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            V = np.where(counts > 0, V / counts, fill_value)
    else:
        raise ValueError("reduce must be None, 'max', 'min', 'sum', or 'mean'")

    return V, x_axis, y_axis


def compute_field(exo: ExodusBasics, step: int, *, method: str, gr0_name: str, gr1_name: str) -> np.ndarray:
    """
    Returns nodal scalar field to contour (shape: num_nodes,)
    """
    g0 = exo.nodal_var_at_step(gr0_name, step)
    if method == "gr0":
        return g0

    g1 = exo.nodal_var_at_step(gr1_name, step)
    denom = g0 * g0 + g1 * g1
    with np.errstate(divide="ignore", invalid="ignore"):
        phi = np.where(denom > 0, (g0 * g0) / denom, 0.0)
    return phi


def extract_contours(x_axis: np.ndarray, y_axis: np.ndarray, Z: np.ndarray, *,
                     level: float, largest_only: bool = False):
    """
    Use matplotlib contouring to extract paths in physical x/y coordinates.

    Returns:
      xy: (N,2) stacked
      starts: (n_paths,)
      lengths: (n_paths,)
    """
    X, Y = np.meshgrid(x_axis, y_axis)  # shapes (ny,nx)
    fig, ax = plt.subplots(figsize=(6, 5))
    cs = ax.contour(X, Y, Z, levels=[level])

    verts_list = []
    # if cs.collections:
    #     for path in cs.collections[0].get_paths():
    #         v = path.vertices
    #         if v is not None and len(v) >= 2:
    #             verts_list.append(v)
    segs = cs.allsegs[0]   # list of (Ni,2) arrays for the first contour level
    verts_list = [s for s in segs if s is not None and len(s) >= 2]

    plt.close(fig)

    if not verts_list:
        xy = np.zeros((0, 2), dtype=float)
        starts = np.zeros((0,), dtype=np.int64)
        lengths = np.zeros((0,), dtype=np.int64)
        return xy, starts, lengths

    if largest_only:
        verts_list = [max(verts_list, key=lambda a: a.shape[0])]

    lengths = np.array([v.shape[0] for v in verts_list], dtype=np.int64)
    starts = np.zeros_like(lengths)
    if len(lengths) > 1:
        starts[1:] = np.cumsum(lengths[:-1])

    xy = np.vstack(verts_list).astype(np.float64)
    return xy, starts, lengths


def quicklook_png(out_png: Path, x_axis, y_axis, Z, xy, starts, lengths, level, title):
    X, Y = np.meshgrid(x_axis, y_axis)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.pcolormesh(X, Y, Z, shading="nearest")
    fig.colorbar(im, ax=ax)
    ax.contour(X, Y, Z, levels=[level], linewidths=2)

    for s, L in zip(starts, lengths):
        seg = xy[s:s+L]
        ax.plot(seg[:, 0], seg[:, 1], linewidth=1.5)

    ax.set_aspect("equal")
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.tight_layout()
    fig.savefig(out_png, dpi=250)
    plt.close(fig)


def level_tag(level: float) -> str:
    s = f"{level:.6g}"
    return s.replace(".", "p").replace("-", "m")


# ---------------------------
# amag calculation
# ---------------------------

def contour_anisotropy_metric(seg_xy: np.ndarray, use_arc_weights: bool = True):
    """
    A = sum |r_i - r_avg| / (r_avg * n_sites)

    If use_arc_weights=True, weights points by local arc-length:
      r_avg = sum(w_i r_i)/sum(w_i)
      A     = sum(w_i |r_i - r_avg|)/(r_avg * sum(w_i))
    """
    seg_xy = np.asarray(seg_xy, dtype=float)
    if seg_xy.ndim != 2 or seg_xy.shape[1] != 2 or seg_xy.shape[0] < 3:
        return np.nan, np.array([np.nan, np.nan]), np.nan

    # If contour is explicitly closed by repeating the first point at the end, drop the duplicate
    scale = max(np.ptp(seg_xy[:, 0]), np.ptp(seg_xy[:, 1]), 1.0)
    eps = 1e-12 * scale
    if np.linalg.norm(seg_xy[0] - seg_xy[-1]) <= eps:
        seg_xy = seg_xy[:-1]
        if seg_xy.shape[0] < 3:
            return np.nan, np.array([np.nan, np.nan]), np.nan

    # "Center of grain": fast estimate
    center = seg_xy.mean(axis=0)

    # Radii
    r = np.linalg.norm(seg_xy - center, axis=1)

    if not use_arc_weights:
        r_avg = r.mean()
        A = np.sum(np.abs(r - r_avg)) / (r_avg * r.size) if r_avg > 0 else np.nan
        return float(A), center, float(r_avg)

    # Closed-curve arc-length weights
    diffs = np.diff(np.vstack([seg_xy, seg_xy[0]]), axis=0)     # N segments, wraps
    seglen = np.linalg.norm(diffs, axis=1)                      # (N,)

    W = seglen.sum()
    if W <= 0:
        return np.nan, center, np.nan

    # Vertex weights: w_i = 0.5*(seglen_{i-1}+seglen_i)
    w = 0.5 * (seglen + np.roll(seglen, 1))                    # (N,)

    r_avg = (w * r).sum() / w.sum()
    A = (w * np.abs(r - r_avg)).sum() / (r_avg * w.sum()) if r_avg > 0 else np.nan
    return float(A), center, float(r_avg)

# ---------------------------
# Main
# ---------------------------

def main():
    ti = time.perf_counter()
    args = parse_args()
    log = setup_logging(args.verbose)
    log.info("Setup:")
    log.info(f"Arguments: {args}")

    exo_files = find_exodus_files(subdirs=args.subdirs, pattern=args.pattern)
    if not exo_files:
        where = "subdirectories" if args.subdirs else "current directory"
        raise SystemExit(f"No Exodus files found in {where} using pattern '{args.pattern}'.")

    args.outdir.mkdir(parents=True, exist_ok=True)

    for cnt, exofile in enumerate(exo_files, start=1):
        til = time.perf_counter()
        stem = exodus_stem(exofile)
        log.warning("\033[1m\033[96m" + f"File {cnt}/{len(exo_files)}: " + "\x1b[0m" + str(stem))

        try:
            with ExodusBasics(str(exofile)) as exo:
                step = select_step(exo, step=args.step, time_value=args.time, log=log)
                tval = float(exo.time()[step])

                if args.level is None:
                    level = 0.99 if args.method == "gr0" else 0.5
                else:
                    level = float(args.level)

                x, y = exo.coords_xy()
                field = compute_field(exo, step, method=args.method,
                                      gr0_name=args.gr0_name, gr1_name=args.gr1_name)

                tol = args.tol if args.tol is not None else _default_tol(x, y)
                Z, x_axis, y_axis = map_to_grid_xy(x, y, field, tol=tol, reduce="mean")

                xy, starts, lengths = extract_contours(
                    x_axis, y_axis, Z, level=level, largest_only=args.largest_only
                )

                # A mag calculation
                # choose main path for the metric
                if lengths.size > 0:
                    k = int(np.argmax(lengths))
                    seg = xy[starts[k]:starts[k] + lengths[k]]
                    A, center, r_avg = contour_anisotropy_metric(seg, use_arc_weights=True)
                else:
                    A, center, r_avg = np.nan, np.array([np.nan, np.nan]), np.nan

                tag = level_tag(level)
                base = f"{stem}_contour_{args.method}_lvl{tag}"
                out_npz = args.outdir / f"{base}.npz"
                out_npy = args.outdir / f"{base}.npy"
                out_csv = args.outdir / f"{base}.csv"
                out_png = args.outdir / f"{base}.png"

                if args.save == "npz":
                    np.savez_compressed(
                        out_npz,
                        xy=xy,
                        starts=starts,
                        lengths=lengths,
                        step=np.array([step], dtype=np.int64),
                        time=np.array([tval], dtype=np.float64),
                        level=np.array([level], dtype=np.float64),
                        method=np.array([args.method]),
                        gr0_name=np.array([args.gr0_name]),
                        gr1_name=np.array([args.gr1_name]),
                        exofile=np.array([str(exofile)]),
                        stem=np.array([stem]),
                        tol=np.array([tol], dtype=np.float64),
                        anisotropy=np.array([A], dtype=np.float64),
                        center=np.array(center, dtype=np.float64),
                        r_avg=np.array([r_avg], dtype=np.float64),
                    )
                    log.warning(f"Wrote {out_npz}")
                elif args.save == "npy":
                    np.save(out_npy, xy)
                    log.warning(f"Wrote {out_npy} (xy only)")
                else:
                    if len(lengths) == 0:
                        arr = np.zeros((0, 3))
                    else:
                        pid = np.empty((xy.shape[0],), dtype=np.int64)
                        for k, (s, L) in enumerate(zip(starts, lengths)):
                            pid[s:s+L] = k
                        arr = np.column_stack([xy[:, 0], xy[:, 1], pid])
                    np.savetxt(out_csv, arr, delimiter=",", header="x,y,path_id", comments="")
                    log.warning(f"Wrote {out_csv}")

                if not args.no_plot:
                    title = f"{stem} | method={args.method} | step={step} | time={tval:.6g} | level={level}"
                    quicklook_png(out_png, x_axis, y_axis, Z, xy, starts, lengths, level, title)
                    log.info(f"Wrote {out_png}")

        except Exception as e:
            log.warning(f"ERROR: {e}")
            sys.exit(2)

        tf(til, log, extra=f"File {cnt} ")

    tf(ti, log, extra="Total ")


if __name__ == "__main__":
    main()
