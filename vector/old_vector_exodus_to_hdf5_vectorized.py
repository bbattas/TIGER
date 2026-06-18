#!/usr/bin/env python3
"""
vector_exodus_to_hdf5.py

Convert phase field grain growth results to curvature measurements using
Lin's VECTOR smoothing algorithms. Outputs raw GB curvature data for
GB velocity calculations at the specified time.

This code assumes uniform mesh elements.

Usage example (standard path):
    python vector_exodus_to_hdf5.py -n 32 -t 120 -l 20 --stream
        --hdf5-frames 50 --hdf5-dt 2.4
        --out my_output -v

Usage example (fast vectorized path):
    python vector_exodus_to_hdf5.py -n 32 -t 120 -l 20 --stream
        --hdf5-frames 50 --hdf5-dt 2.4
        --fast --chunk-size 50000
        --out my_output -v

--fast enables:
    1. Vectorized boundary/junction detection in accumulate_gb_properties
       (replaces Python double-loops with NumPy roll ops + scipy distance
       transform for TJ exclusion).
    2. Vectorized find_window + chunked einsum normal vector calculation
       in PACKAGE_MP_Linear (replaces per-pixel find_window calls with
       batched NumPy advanced indexing).

    Recommended for meshes > 1000x1000 or grain counts > 5000.
    At 2400x2400 with 16900 grains (step=8), reduces wall time from
    hours to minutes.

--chunk-size controls memory per vectorized batch:
    chunk_size=50000, loop_times=20 => ~740 MB peak per chunk (transient).
    Reduce if OOM; increase for speed on memory-rich nodes.
"""

from ExodusBasics import ExodusBasics
import PACKAGE_MP_Linear_Vectorized as smooth
import myInput

import glob
import time
import numpy as np
import argparse
import sys
import logging
import matplotlib.pyplot as plt
import csv
from pathlib import Path
import h5py
from tqdm import tqdm
from scipy.ndimage import distance_transform_edt


# Assumes uniform constant unchanging mesh, dx = dy and treats all
# index values as the coords

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    VALID_CPUS = (1, 2, 4, 8, 16, 32, 64, 128)
    p = argparse.ArgumentParser(
        description=(
            "Convert phase field GG results to curvature measurements, "
            "using Lin's VECTOR smoothing algorithms. Also optionally outputs "
            "raw data for GB velocity calculations at the specified time.  "
            "This code assumes uniform mesh elements."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---- General ----
    log = p.add_argument_group("General")
    log.add_argument(
        "-v", "--verbose", action="count", default=0,
        help="Increase verbosity (-v, -vv, -vvv).",
    )
    log.add_argument(
        "-n", "--cpus", type=int, default=4, choices=VALID_CPUS,
        help=(
            "Number of CPUs for smoothing/curvature calculations. "
            "Allowed: powers of 2 (1, 2, 4, 8, 16, 32, 64, 128...)."
        ),
    )
    log.add_argument(
        "-s", "--subdirs", action="store_true",
        help=(
            "Search for *.e files one level down (./*/.e). "
            "If not set, only search current directory."
        ),
    )

    # ---- Time selection ----
    tim = p.add_argument_group("Time (choose one)")
    grp = tim.add_mutually_exclusive_group(required=True)
    grp.add_argument(
        "-g", "--grains", type=int,
        help=(
            "Target grain count; chooses timestep with grain_tracker "
            "closest to this value."
        ),
    )
    grp.add_argument(
        "-t", "--time", type=float,
        help="Target time; chooses timestep with time_whole closest to this value.",
    )

    # ---- Curvature options ----
    curv = p.add_argument_group("Curvature/Inclination")
    curv.add_argument(
        "--loop-times", "-l", type=int, default=5,
        help="Smoothing window size. Larger = smoother.",
    )
    curv.add_argument(
        "--tj-distance", type=int, default=6, metavar="N",
        help=(
            "Euclidean pixel distance threshold for excluding "
            "TJ-proximal boundary pixels."
        ),
    )
    curv.add_argument(
        "--unsigned", action="store_true", default=False,
        help="If set, use an unsigned abs() curvature.",
    )

    # ---- Fast vectorized path ----
    fast_grp = p.add_argument_group("Fast vectorized path (--fast)")
    fast_grp.add_argument(
        "--fast", action="store_true", default=False,
        help=(
            "Use vectorized boundary detection, GB accumulation, and "
            "smoothing core. Recommended for large meshes (>1000x1000) "
            "or high grain counts (>5000). Uses chunked processing to "
            "control memory usage. Affects both accumulate_gb_properties "
            "(scipy distance transform for TJ exclusion) and "
            "PACKAGE_MP_Linear (batched find_window via NumPy einsum)."
        ),
    )
    fast_grp.add_argument(
        "--chunk-size", type=int, default=50_000, metavar="N",
        dest="chunk_size",
        help=(
            "Number of boundary pixels processed per vectorized chunk "
            "in --fast mode. Reduce if you hit OOM; increase for speed. "
            "Default 50000 => ~740 MB peak per chunk at loop_times=20, "
            "fw_len=43."
        ),
    )

    # ---- Multi-frame HDF5 ----
    mf = p.add_argument_group("Multi-frame HDF5")
    mf.add_argument(
        "--out", "-o", type=str, default=None, metavar="NAME",
        help=(
            "Output name for the HDF5 file (with or without .h5 extension). "
            "If not set, defaults to <stem>_multiframe.h5."
        ),
    )
    mf.add_argument(
        "--stream", action="store_true", default=False,
        help=(
            "Write each HDF5 frame immediately and discard from memory. "
            "Minimizes RAM usage for large frame counts."
        ),
    )
    mf.add_argument(
        "--hdf5-frames", type=int, default=5, metavar="N",
        help="Number of frames to save in the HDF5 file.",
    )
    mf.add_argument(
        "--hdf5-dt", type=float, default=None, metavar="DT",
        help=(
            "Target time spacing between saved frames. "
            "If not set, defaults to 1%% of max simulation time."
        ),
    )

    # ---- Plot options ----
    plot = p.add_argument_group("Plotting")
    plot.add_argument(
        "--debug-plot", "-d", action="store_true",
        help="Save the debugging plots. -vv also enables this.",
    )

    args = p.parse_args()
    args.signed = not args.unsigned
    return args


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def setup_logging(verbosity: int) -> logging.Logger:
    level = logging.WARNING
    if verbosity >= 2:
        level = logging.DEBUG
    elif verbosity >= 1:
        level = logging.INFO
    logging.basicConfig(level=level, format="%(message)s")
    return logging.getLogger("TIGER")


def tf(ti, log, extra=None):
    if extra is not None:
        log.warning(extra + f"Time: {(time.perf_counter() - ti):.4}s")
    else:
        log.warning(f"Time: {(time.perf_counter() - ti):.4}s")


def vtf(ti, log, extra=None):
    if extra is not None:
        log.info(extra + f"Time: {(time.perf_counter() - ti):.4}s")
    else:
        log.info(f"Time: {(time.perf_counter() - ti):.4}s")


def progress(iterable, desc=None, *, verbose=0, **kwargs):
    """Wrap iterable in tqdm only when not in verbose mode."""
    if verbose > 0:
        return iterable
    return tqdm(iterable, desc=desc, **kwargs)


# ---------------------------------------------------------------------------
# File utilities
# ---------------------------------------------------------------------------

def find_exodus_files(*, subdirs: bool = False,
                      pattern: str = "*.e") -> list:
    """
    Find Exodus files in current directory or one-level-down subdirectories.
    Returns sorted list of Paths.
    """
    cwd = Path.cwd()
    if subdirs:
        files = sorted(cwd.glob(f"*/{pattern}"))
    else:
        files = sorted(cwd.glob(pattern))
    files = [p for p in files if p.is_file()]
    return files


def exodus_stem(exo_path: Path) -> str:
    """
    Convert '/a/b/file_out.e' -> 'file'
              './file.e'       -> 'file'
    """
    name = exo_path.name
    if name.endswith(".e"):
        name = name[:-2]
    if name.endswith("_out"):
        name = name[:-4]
    return name


def closest_index(values: np.ndarray, target: float) -> int:
    """Return index of entry closest to target. Ties -> first occurrence."""
    values = np.asarray(values)
    return int(np.argmin(np.abs(values - target)))


# ---------------------------------------------------------------------------
# Timestep selection
# ---------------------------------------------------------------------------

def select_step(exo, *, grains=None, time_value=None,
                log: logging.Logger) -> int:
    """
    Select the single anchor timestep index (0-based).

    Parameters
    ----------
    exo        : open ExodusBasics instance
    grains     : int or None  — target grain count
    time_value : float or None — target time
    log        : Logger

    Returns
    -------
    int : timestep index
    """
    times = exo.time()

    if time_value is not None:
        step = closest_index(times, float(time_value))
        log.info(
            f"Frame selected by time: requested={time_value}, "
            f"chosen step={step}, time={times[step]}"
        )
        return step

    # grains path: require grain_tracker
    glo_names = exo.glo_varnames()
    if "grain_tracker" not in glo_names:
        raise RuntimeError(
            "You requested --grains, but this Exodus file has no global "
            f"variable 'grain_tracker'. Available global vars: {glo_names}"
        )

    gt = exo.glo_var_series("grain_tracker")
    gt_counts = np.rint(gt).astype(np.int64)
    step = closest_index(gt_counts, int(grains))
    log.info(
        f"Frame selected by grains: requested={grains}, "
        f"chosen step={step}, grain_tracker={gt_counts[step]}"
    )
    return step


def select_multi_frame_steps(
    exo,
    target_step: int,
    n_frames: int,
    dt=None,
    mode: str = "center",
    log: logging.Logger = None,
) -> list:
    """
    Select up to n_frames timestep indices anchored to target_step.

    Parameters
    ----------
    exo         : ExodusBasics open instance
    target_step : int
        The step selected by -t / -g. Acts as center or end anchor.
    n_frames    : int
        Desired number of frames (including the target frame).
    dt          : float or None
        Target time spacing between frames.
        If None, defaults to 1% of the target frame's time
        (or t_max if target time is 0).
    mode        : "center" | "end"
        "center" — target_step is the middle frame.
        "end"    — target_step is the last frame.
    log         : logging.Logger or None

    Returns
    -------
    list of (step_index, time_value) tuples, sorted ascending by step index.
    """
    times = np.asarray(exo.time(), dtype=float)
    n_total = len(times)
    t_anchor = times[target_step]

    # Grain tracker (optional)
    glo_names = exo.glo_varnames()
    has_gt = "grain_tracker" in glo_names
    gt = (
        np.rint(exo.glo_var_series("grain_tracker")).astype(np.int64)
        if has_gt else None
    )

    # --- Default dt ---
    if dt is None:
        ref = t_anchor if t_anchor > 0 else times[-1]
        dt = ref * 0.01
        if log:
            log.info(
                f"HDF5 dt defaulted to {dt:.4g} "
                f"(1% of anchor time {ref:.4g})"
            )

    # --- Build ideal target times centered on or ending at anchor ---
    if mode == "end":
        offsets = np.arange(n_frames - 1, -1, -1)
        ideal_times = t_anchor - offsets * dt
    else:  # "center"
        half = (n_frames - 1) / 2.0
        ideal_times = t_anchor + np.arange(n_frames) * dt - half * dt

    # --- Map each ideal time to the closest available step ---
    raw_indices = [closest_index(times, t) for t in ideal_times]

    # --- Safety 1: remove duplicates while preserving order ---
    seen = set()
    unique_indices = []
    for idx in raw_indices:
        if idx not in seen:
            seen.add(idx)
            unique_indices.append(idx)

    # --- Safety 2: if dt collapsed frames, fill gaps ---
    if len(unique_indices) < n_frames:
        i_min = max(0, min(unique_indices))
        i_max = min(n_total - 1, max(unique_indices))
        if (i_max - i_min + 1) < n_frames:
            i_min = max(0, i_max - (n_frames - 1))
            i_max = min(n_total - 1, i_min + (n_frames - 1))
        extra_indices = list(np.unique(
            np.round(np.linspace(i_min, i_max, n_frames)).astype(int)
        ))
        unique_indices = sorted(
            set(unique_indices) | set(extra_indices)
        )[:n_frames]
        if log:
            log.warning(
                f"\033[31mHDF5:\033[0m dt={dt:.4g} too small/large — "
                f"only {len(seen)} unique frames found. "
                f"Filled to {len(unique_indices)} by spreading within "
                f"[{times[i_min]:.4g}, {times[i_max]:.4g}]."
            )

    unique_indices = sorted(unique_indices)
    result = [(int(idx), float(times[idx])) for idx in unique_indices]

    if log:
        log.info(
            f"HDF5 multi-frame: anchor step={target_step} t={t_anchor:.4g}, "
            f"mode='{mode}', dt={dt:.4g}, "
            f"requested={n_frames}, selected={len(result)}"
        )
        for idx, t in result:
            if has_gt:
                log.info(f"  step={idx}, time={t:.6g}, grains={int(gt[idx])}")
            else:
                log.info(f"  step={idx}, time={t:.6g}")

    return result


# ---------------------------------------------------------------------------
# Grid mapping
# ---------------------------------------------------------------------------

def map_to_grid(xc, yc, val, *, tol=1e-6, fill_value=np.nan, reduce=None):
    """
    Map per-element values onto a structured 2D grid using xc, yc element
    locations.

    Parameters
    ----------
    xc, yc     : np.ndarray, shape (n_elem,)
    val        : np.ndarray, shape (n_elem,)
    tol        : float  — quantization tolerance for floating noise
    fill_value : scalar — value for unfilled grid cells
    reduce     : None | 'max' | 'min' | 'sum' — collision handling

    Returns
    -------
    P0       : np.ndarray, shape (ny, nx)
    x_centers: np.ndarray, shape (nx,)
    y_centers: np.ndarray, shape (ny,)
    """
    # Quantize coordinates to stable integer keys
    kx = np.rint(xc / tol).astype(np.int64)
    ky = np.rint(yc / tol).astype(np.int64)

    # Unique axes + inverse indices
    ux, j = np.unique(kx, return_inverse=True)   # columns
    uy, i = np.unique(ky, return_inverse=True)   # rows

    x_centers = ux.astype(np.float64) * tol
    y_centers = uy.astype(np.float64) * tol

    P0 = np.full(
        (len(y_centers), len(x_centers)),
        fill_value,
        dtype=np.float64 if np.isnan(fill_value) else val.dtype,
    )

    if reduce is None:
        P0[i, j] = val
    else:
        if reduce == "max":
            P0[i, j] = -np.inf
            np.maximum.at(P0, (i, j), val)
        elif reduce == "min":
            P0[i, j] = np.inf
            np.minimum.at(P0, (i, j), val)
        elif reduce == "sum":
            P0[i, j] = 0.0
            np.add.at(P0, (i, j), val)
        else:
            raise ValueError("reduce must be None, 'max', 'min', or 'sum'")

    return P0, x_centers, y_centers


# ---------------------------------------------------------------------------
# Smoothing wrappers
# ---------------------------------------------------------------------------

def get_both(P0: np.ndarray, args) -> tuple:
    """
    Standard (scalar) path: compute normal vectors and curvature via
    linear_combined_core. Preserved unchanged from original.

    Parameters
    ----------
    P0   : np.ndarray, shape (nx, ny) — grain ID grid
    args : argparse.Namespace

    Returns
    -------
    C : np.ndarray, shape (2, nx, ny)
    P : np.ndarray, shape (3, nx, ny)
    """
    nx, ny = P0.shape
    ng = int(np.nanmax(P0)) - int(np.nanmin(P0)) + 1
    cores = args.cpus
    loop_times = args.loop_times
    R = np.zeros((nx, ny, 3))

    smooth_class = smooth.linear_class(
        nx, ny, ng, cores, loop_times, P0, R,
        verification_system=False,
        curvature_sign=args.signed,
    )
    smooth_class.linear_main("both")

    C = smooth_class.get_C()
    P = smooth_class.get_P()
    return C, P


def get_both_fast(P0: np.ndarray, args) -> tuple:
    """
    Fast (vectorized) path: compute normal vectors and curvature via
    linear_combined_core_fast with chunked window processing.

    Passes fast=True and chunk_size to linear_main, which:
      - stores chunk_size on the instance (self.fast_chunk_size) before
        spawning worker processes so the picklable dispatcher can read it
      - dispatches linear_combined_core_fast_dispatch to the pool instead
        of linear_combined_core

    Parameters
    ----------
    P0   : np.ndarray, shape (nx, ny) — grain ID grid
    args : argparse.Namespace
           Must have: .cpus, .loop_times, .signed, .chunk_size

    Returns
    -------
    C : np.ndarray, shape (2, nx, ny)
    P : np.ndarray, shape (3, nx, ny)
    """
    nx, ny = P0.shape
    ng = int(np.nanmax(P0)) - int(np.nanmin(P0)) + 1
    cores = args.cpus
    loop_times = args.loop_times
    R = np.zeros((nx, ny, 3))

    smooth_class = smooth.linear_class(
        nx, ny, ng, cores, loop_times, P0, R,
        verification_system=False,
        curvature_sign=args.signed,
    )
    smooth_class.linear_main(
        "both",
        fast=True,
        chunk_size=args.chunk_size,
    )

    C = smooth_class.get_C()
    P = smooth_class.get_P()
    return C, P


def get_normal_vector(P0: np.ndarray, args, log: logging.Logger) -> tuple:
    """
    Calculate normal vectors for grain boundaries in 2D microstructure.
    Uses smooth linear interpolation method.

    Parameters
    ----------
    P0   : np.ndarray, shape (nx, ny)
    args : argparse.Namespace
    log  : Logger

    Returns
    -------
    tuple : (P, sites_together, sites)
    """
    nx = P0.shape[0]
    ny = P0.shape[1]
    ng = int(np.nanmax(P0)) - int(np.nanmin(P0)) + 1
    cores = args.cpus
    loop_times = 5
    R = np.zeros((nx, ny, 2))
    verb = False
    if args.verbose >= 1:
        verb = True

    smooth_class = smooth.linear_class(
        nx, ny, ng, cores, loop_times, P0, R,
        verification_system=verb,
    )
    smooth_class.linear_main("inclination")
    P = smooth_class.get_P()
    sites = smooth_class.get_all_gb_list()
    sites_together = []
    for id in range(len(sites)):
        sites_together += sites[id]
    log.info(f"Total num of GB sites: {len(sites_together)}")

    return P, sites_together, sites


# ---------------------------------------------------------------------------
# GB pair utilities
# ---------------------------------------------------------------------------

def get_pair_id(grain_a: int, grain_b: int) -> tuple:
    """
    Returns a consistent, hashable key for a grain boundary pair.
    Sorting ensures the same pair always maps to the same key regardless
    of which grain is 'central' at a given pixel.

    Parameters
    ----------
    grain_a : int
    grain_b : int

    Returns
    -------
    tuple : (min_id, max_id)
    """
    return (min(grain_a, grain_b), max(grain_a, grain_b))


# ---------------------------------------------------------------------------
# Standard (scalar) GB property accumulation — original path, unchanged
# ---------------------------------------------------------------------------

def get_boundary_pixels(C0: np.ndarray) -> list:
    """
    Identifies all pixels lying on a grain boundary using periodic BCs.
    A pixel is a boundary pixel if any of its 4 cardinal neighbors
    has a different grain ID.

    Parameters
    ----------
    C0 : np.ndarray, shape (nx, ny)

    Returns
    -------
    list of tuple : (i, j) coordinates of boundary pixels
    """
    nx, ny = C0.shape
    boundary_pixels = []

    for i in range(nx):
        for j in range(ny):
            ip = (i + 1) % nx
            im = (i - 1) % nx
            jp = (j + 1) % ny
            jm = (j - 1) % ny
            current = C0[i, j]
            if (C0[ip, j] != current or
                    C0[im, j] != current or
                    C0[i, jp] != current or
                    C0[i, jm] != current):
                boundary_pixels.append((i, j))

    return boundary_pixels


def get_junction_pixels(C0: np.ndarray, boundary_pixels: list) -> dict:
    """
    Identifies junction pixels where 3 or more grains meet, and records
    them under every grain pair key that passes through that junction.

    Parameters
    ----------
    C0              : np.ndarray, shape (nx, ny)
    boundary_pixels : list of tuple

    Returns
    -------
    dict : pair_id -> list of (i, j) junction pixel coordinates
    """
    nx, ny = C0.shape
    junction_dict = {}

    for (i, j) in boundary_pixels:
        ip = (i + 1) % nx
        im = (i - 1) % nx
        jp = (j + 1) % ny
        jm = (j - 1) % ny

        central = int(C0[i, j])
        neighbors = set([
            int(C0[ip, j]),
            int(C0[im, j]),
            int(C0[i, jp]),
            int(C0[i, jm]),
        ])
        neighbors.discard(central)

        if len(neighbors) > 1:
            for neighbor_id in neighbors:
                pair_id = get_pair_id(central, neighbor_id)
                if pair_id in junction_dict:
                    junction_dict[pair_id].append((i, j))
                else:
                    junction_dict[pair_id] = [(i, j)]

    return junction_dict


def accumulate_gb_properties(C: np.ndarray, TJ_distance_max: int = 6,
                              signed: bool = True) -> tuple:
    """
    Standard single-pass accumulation of grain boundary properties.
    Simultaneously identifies boundary pixels, detects junction pixels,
    and accumulates curvature per grain boundary pair.

    Parameters
    ----------
    C               : np.ndarray, shape (2, nx, ny)
                      C[0] is grain ID map, C[1] is curvature map.
    TJ_distance_max : int
    signed          : bool

    Returns
    -------
    tuple : (raw_gb_dict, boundary_pixels, junction_pixels)
        raw_gb_dict : pair_id -> np.array([valid_count, sum_curvature,
                                           raw_area, grain_id1, grain_id2])
        boundary_pixels : np.ndarray, shape (N, 2)
        junction_pixels : np.ndarray, shape (M, 2)
    """
    C0 = C[0]
    C1 = C[1]
    nx, ny = C0.shape

    raw_gb_dict   = {}
    junction_dict = {}
    boundary_pixels = []
    junction_pixels = []

    # ------------------------------------------------------------------
    # PASS 1: Full pixel scan — classify boundary vs interior,
    #         junction vs clean GB, initialize raw_gb_dict entries
    # ------------------------------------------------------------------
    for i in range(nx):
        for j in range(ny):
            ip = (i + 1) % nx
            im = (i - 1) % nx
            jp = (j + 1) % ny
            jm = (j - 1) % ny

            central = int(C0[i, j])
            neighbors = set([
                int(C0[ip, j]),
                int(C0[im, j]),
                int(C0[i, jp]),
                int(C0[i, jm]),
            ])
            neighbors.discard(central)

            if len(neighbors) == 0:
                continue

            boundary_pixels.append((i, j))

            if len(neighbors) > 1:
                junction_pixels.append((i, j))
                for neighbor_id in neighbors:
                    pair_id = get_pair_id(central, neighbor_id)
                    if pair_id not in junction_dict:
                        junction_dict[pair_id] = []
                    junction_dict[pair_id].append((i, j))
                continue

            # Clean GB pixel
            neighbor_id = next(iter(neighbors))
            pair_id = get_pair_id(central, neighbor_id)
            grain_id1, grain_id2 = pair_id

            if pair_id not in raw_gb_dict:
                raw_gb_dict[pair_id] = np.array(
                    [0.0, 0.0, 0.0, float(grain_id1), float(grain_id2)]
                )
            raw_gb_dict[pair_id][2] += 1.0

    # ------------------------------------------------------------------
    # PASS 2: Boundary pixels only — apply TJ filter and accumulate
    # ------------------------------------------------------------------
    for (i, j) in boundary_pixels:
        central = int(C0[i, j])
        ip = (i + 1) % nx
        im = (i - 1) % nx
        jp = (j + 1) % ny
        jm = (j - 1) % ny

        neighbors = set([
            int(C0[ip, j]),
            int(C0[im, j]),
            int(C0[i, jp]),
            int(C0[i, jm]),
        ])
        neighbors.discard(central)

        if len(neighbors) > 1:
            continue

        neighbor_id = next(iter(neighbors))
        pair_id = get_pair_id(central, neighbor_id)
        grain_id1, grain_id2 = pair_id

        # TJ proximity check
        too_close = False
        if pair_id in junction_dict:
            pixel_coord = np.array([i, j])
            for tj_site in junction_dict[pair_id]:
                if np.linalg.norm(pixel_coord - np.array(tj_site)) < TJ_distance_max:
                    too_close = True
                    break
        if too_close:
            continue

        curvature_val = C1[i, j]
        if signed:
            if central != grain_id1:
                curvature_val = -curvature_val
        else:
            curvature_val = abs(curvature_val)

        raw_gb_dict[pair_id][0] += 1.0
        raw_gb_dict[pair_id][1] += curvature_val

    boundary_pixels = np.array(boundary_pixels)
    junction_pixels = np.array(junction_pixels)

    return raw_gb_dict, boundary_pixels, junction_pixels


# ---------------------------------------------------------------------------
# Fast (vectorized) GB property accumulation — new --fast path
# ---------------------------------------------------------------------------

def _detect_boundary_and_junction_vectorized(C0: np.ndarray) -> tuple:
    """
    Vectorized replacement for Pass 1 of accumulate_gb_properties.

    Uses np.roll to compare each pixel to all four cardinal neighbors
    simultaneously across the entire grid, with periodic boundary
    conditions implicit in the roll.

    At 2400x2400: each boolean array is ~5.5 MB. Total overhead ~50 MB.

    Parameters
    ----------
    C0 : np.ndarray, shape (nx, ny) — grain ID map (int)

    Returns
    -------
    is_boundary  : np.ndarray bool, shape (nx, ny)
        True where the pixel has at least one foreign cardinal neighbor.
    is_junction  : np.ndarray bool, shape (nx, ny)
        True where the pixel has 2 or more foreign cardinal neighbors
        (triple/higher junction pixel).
    neighbor_map : np.ndarray int32, shape (nx, ny)
        For clean GB pixels (exactly 1 foreign neighbor), holds the
        foreign grain ID. 0 elsewhere. Priority: right > left > up > down.
    """
    # Four cardinal neighbors with periodic BC via np.roll
    right = np.roll(C0, -1, axis=0)   # i+1
    left  = np.roll(C0,  1, axis=0)   # i-1
    up    = np.roll(C0, -1, axis=1)   # j+1
    down  = np.roll(C0,  1, axis=1)   # j-1

    diff_r = (right != C0)
    diff_l = (left  != C0)
    diff_u = (up    != C0)
    diff_d = (down  != C0)

    is_boundary = diff_r | diff_l | diff_u | diff_d

    # A pixel is a junction if at least two of its foreign neighbors
    # belong to DIFFERENT foreign grains — i.e. at least one pair of
    # foreign neighbors disagrees with each other.
    # Check all 6 pairs among the 4 neighbors, but only where both differ
    # from center.  If any two foreign neighbors also differ from each
    # other, it's a junction.
    is_junction = np.zeros_like(C0, dtype=bool)

    neighbor_arrays  = [(right, diff_r), (left, diff_l),
                        (up,    diff_u), (down, diff_d)]

    for idx_a in range(4):
        for idx_b in range(idx_a + 1, 4):
            n_a, d_a = neighbor_arrays[idx_a]
            n_b, d_b = neighbor_arrays[idx_b]
            # Both neighbors are foreign AND they differ from each other
            is_junction |= (d_a & d_b & (n_a != n_b))

    is_clean    = is_boundary & ~is_junction

    # Build neighbor_map for clean GB pixels only.
    # Priority right > left > up > down (arbitrary but consistent).
    neighbor_map = np.zeros_like(C0, dtype=np.int32)
    neighbor_map[is_clean & diff_d] = down [is_clean & diff_d].astype(np.int32)
    neighbor_map[is_clean & diff_u] = up   [is_clean & diff_u].astype(np.int32)
    neighbor_map[is_clean & diff_l] = left [is_clean & diff_l].astype(np.int32)
    neighbor_map[is_clean & diff_r] = right[is_clean & diff_r].astype(np.int32)

    return is_boundary, is_junction, neighbor_map


def _build_tj_exclusion_mask(is_junction: np.ndarray,
                              TJ_distance_max: int) -> np.ndarray:
    """
    Build a boolean mask marking every pixel within TJ_distance_max of
    any junction pixel.

    Uses scipy.ndimage.distance_transform_edt — O(nx*ny), no Python loop.
    At 2400x2400: distance array is ~46 MB, computed in < 1 second.

    Parameters
    ----------
    is_junction     : np.ndarray bool, shape (nx, ny)
    TJ_distance_max : int

    Returns
    -------
    np.ndarray bool, shape (nx, ny)
        True where the pixel is within TJ_distance_max of any junction.
    """
    # distance_transform_edt measures distance from nearest True pixel
    # in the foreground. We want distance from nearest junction pixel,
    # so foreground = junction (is_junction == True).
    # The function operates on the background (False pixels), so we
    # invert: distance from nearest True = edt(~foreground).
    # Guard: if no junction pixels exist, no pixel is excluded
    if not np.any(is_junction):
        return np.zeros_like(is_junction, dtype=bool)
    dist_from_tj = distance_transform_edt(~is_junction)
    # Subtract small epsilon to handle pixels at exactly the boundary radius
    return dist_from_tj < (float(TJ_distance_max) - 1e-10)


def accumulate_gb_properties_fast(C: np.ndarray,
                                   TJ_distance_max: int = 6,
                                   signed: bool = True) -> tuple:
    """
    Vectorized replacement for accumulate_gb_properties.

    Replaces:
      - Pass 1 double Python loop (nx*ny iterations) with NumPy roll ops.
      - Pass 2 per-pixel TJ proximity inner loop with a scipy distance
        transform over the whole grid.
      - Growing Python lists of tuples with np.argwhere + np.bincount.

    Memory at 2400x2400:
      - Boolean arrays (x6): ~33 MB total
      - neighbor_map (int32): ~22 MB
      - dist_from_tj (float64): ~46 MB
      - Temporary pair encoding arrays: ~proportional to N_boundary

    Parameters
    ----------
    C               : np.ndarray, shape (2, nx, ny)
    TJ_distance_max : int
    signed          : bool

    Returns
    -------
    tuple : (raw_gb_dict, boundary_pixels, junction_pixels)
        Same structure as accumulate_gb_properties — fully drop-in
        compatible with average_gb_properties and compute_gb_curvature.
    """
    C0 = C[0].astype(np.int32)
    C1 = C[1]
    nx, ny = C0.shape

    # ------------------------------------------------------------------
    # Step 1: Vectorized boundary structure detection
    # ------------------------------------------------------------------
    is_boundary, is_junction, neighbor_map = \
        _detect_boundary_and_junction_vectorized(C0)

    is_clean = is_boundary & ~is_junction

    # ------------------------------------------------------------------
    # Step 2: TJ exclusion mask via distance transform
    # ------------------------------------------------------------------
    tj_exclusion_mask = _build_tj_exclusion_mask(is_junction, TJ_distance_max)

    # Pixels that contribute to curvature accumulation:
    # clean GB pixel AND not excluded by TJ proximity
    accumulate_mask = is_clean & ~tj_exclusion_mask

    # ------------------------------------------------------------------
    # Step 3: Extract coordinate arrays (no Python loops)
    # ------------------------------------------------------------------
    boundary_pixels = np.argwhere(is_boundary)    # (N_b, 2)
    junction_pixels = np.argwhere(is_junction)    # (N_j, 2)
    clean_pixels    = np.argwhere(is_clean)        # (N_c, 2)
    accum_pixels    = np.argwhere(accumulate_mask) # (N_a, 2)

    # ------------------------------------------------------------------
    # Step 4: Build raw_area counts per pair via np.bincount
    #
    # Encode each (min_id, max_id) pair as a single int64 for hashing:
    #   key = min_id * LARGE + max_id
    # where LARGE > max possible grain ID.
    # np.unique then gives us one row per unique pair with its pixel
    # count from bincount — no per-pixel Python loop needed.
    # ------------------------------------------------------------------
    raw_gb_dict = {}

    if len(clean_pixels) > 0:
        ci = clean_pixels[:, 0]
        cj = clean_pixels[:, 1]
        central_ids  = C0[ci, cj].astype(np.int64)
        neighbor_ids = neighbor_map[ci, cj].astype(np.int64)
        pair_min = np.minimum(central_ids, neighbor_ids)
        pair_max = np.maximum(central_ids, neighbor_ids)

        unique_pairs, inverse = np.unique(
            np.stack([pair_min, pair_max], axis=1),
            axis=0,
            return_inverse=True,
        )
        raw_area_counts = np.bincount(inverse, minlength=len(unique_pairs))

        for k, (g1, g2) in enumerate(unique_pairs):
            pid = (int(g1), int(g2))
            raw_gb_dict[pid] = np.array([
                0.0,                        # valid_count (filled in step 5)
                0.0,                        # sum_curvature (filled in step 5)
                float(raw_area_counts[k]),  # raw_area (before TJ exclusion)
                float(g1),
                float(g2),
            ])

    # ------------------------------------------------------------------
    # Step 5: Accumulate curvature for TJ-excluded-clean pixels
    # ------------------------------------------------------------------
    if len(accum_pixels) > 0:
        ai = accum_pixels[:, 0]
        aj = accum_pixels[:, 1]
        central_ids  = C0[ai, aj].astype(np.int64)
        neighbor_ids = neighbor_map[ai, aj].astype(np.int64)
        pair_min = np.minimum(central_ids, neighbor_ids)
        pair_max = np.maximum(central_ids, neighbor_ids)

        curv_vals = C1[ai, aj].copy()

        if signed:
            # Flip curvature sign where the central grain is NOT grain_id1
            # (grain_id1 = pair_min by construction)
            flip_mask = central_ids != pair_min
            curv_vals[flip_mask] *= -1.0
        else:
            curv_vals = np.abs(curv_vals)

        unique_pairs_a, inverse_a = np.unique(
            np.stack([pair_min, pair_max], axis=1),
            axis=0,
            return_inverse=True,
        )
        valid_counts  = np.bincount(inverse_a,
                                    minlength=len(unique_pairs_a))
        sum_curvature = np.bincount(inverse_a,
                                    weights=curv_vals,
                                    minlength=len(unique_pairs_a))

        for k, (g1, g2) in enumerate(unique_pairs_a):
            pid = (int(g1), int(g2))
            if pid in raw_gb_dict:
                raw_gb_dict[pid][0] = float(valid_counts[k])
                raw_gb_dict[pid][1] = float(sum_curvature[k])
            # If pid missing from raw_gb_dict something is inconsistent
            # with the clean pixel map — skip safely.

    return raw_gb_dict, boundary_pixels, junction_pixels


# ---------------------------------------------------------------------------
# GB property averaging — shared by both paths
# ---------------------------------------------------------------------------

def average_gb_properties(raw_gb_dict: dict) -> dict:
    """
    Removes GBs with no valid pixels after TJ filtering, then computes
    average curvature for each remaining grain boundary.

    Parameters
    ----------
    raw_gb_dict : dict
        Output of accumulate_gb_properties or accumulate_gb_properties_fast.

    Returns
    -------
    dict
        pair_id -> np.array([avg_curvature, gb_area, grain_id1,
                             grain_id2, raw_gb_area])
        gb_area     : TJ-filtered clean GB pixel count
        raw_gb_area : clean GB pixel count before TJ exclusion
    """
    gb_dict = {}

    for pair_id, data in raw_gb_dict.items():
        valid_count = data[0]
        if valid_count == 0:
            continue
        avg_curvature = data[1] / valid_count
        gb_area       = valid_count
        raw_gb_area   = data[2]
        grain_id1     = data[3]
        grain_id2     = data[4]
        gb_dict[pair_id] = np.array([
            avg_curvature, gb_area, grain_id1, grain_id2, raw_gb_area
        ])

    return gb_dict


# ---------------------------------------------------------------------------
# Top-level curvature orchestrators — standard and fast
# ---------------------------------------------------------------------------

def compute_gb_curvature(C: np.ndarray, TJ_distance_max: int = 6,
                          signed: bool = True) -> tuple:
    """
    Standard path top-level orchestrator for GB curvature calculation.

    Parameters
    ----------
    C               : np.ndarray, shape (2, nx, ny)
    TJ_distance_max : int
    signed          : bool

    Returns
    -------
    tuple : (gb_dict, boundary_pixels, junction_pixels)
        gb_dict : pair_id -> np.array([avg_curvature, gb_area,
                                       grain_id1, grain_id2, raw_gb_area])
    """
    raw_gb_dict, boundary_pixels, junction_pixels = accumulate_gb_properties(
        C, TJ_distance_max=TJ_distance_max, signed=signed
    )
    gb_dict = average_gb_properties(raw_gb_dict)
    return gb_dict, boundary_pixels, junction_pixels


def compute_gb_curvature_fast(C: np.ndarray, TJ_distance_max: int = 6,
                               signed: bool = True) -> tuple:
    """
    Fast path top-level orchestrator for GB curvature calculation.

    Drop-in replacement for compute_gb_curvature using vectorized internals.
    Shares average_gb_properties with the standard path — only the
    accumulation step differs.

    Parameters
    ----------
    C               : np.ndarray, shape (2, nx, ny)
    TJ_distance_max : int
    signed          : bool

    Returns
    -------
    tuple : (gb_dict, boundary_pixels, junction_pixels)
        Identical structure to compute_gb_curvature output.
    """
    raw_gb_dict, boundary_pixels, junction_pixels = \
        accumulate_gb_properties_fast(
            C, TJ_distance_max=TJ_distance_max, signed=signed
        )
    gb_dict = average_gb_properties(raw_gb_dict)
    return gb_dict, boundary_pixels, junction_pixels


# ---------------------------------------------------------------------------
# Frame processing — routes to standard or fast path based on args.fast
# ---------------------------------------------------------------------------

def process_frame(exo, step: int, args,
                  log: logging.Logger) -> tuple:
    """
    Process a single exodus timestep into grids and GB data.

    Routes to the standard or fast path based on args.fast:
      - Standard: get_both + compute_gb_curvature
      - Fast:     get_both_fast + compute_gb_curvature_fast

    Parameters
    ----------
    exo  : open ExodusBasics instance
    step : int — 0-based timestep index
    args : argparse.Namespace
    log  : Logger

    Returns
    -------
    tuple : (step, time_val, P0, C, P, gb_dict,
             boundary_pixels, junction_pixels)
    """
    time_val = float(exo.time()[step])

    log.debug(f"  [process_frame] step={step}: reading mesh data...")
    xc, yc = exo.element_centers_xy(method="mean")
    ug = exo.elem_var_at_step("unique_grains", step=step)
    ug = np.rint(ug).astype(np.int32)

    P0, _, _ = map_to_grid(xc, yc, ug, tol=1e-12, fill_value=np.nan)

    if args.fast:
        log.debug(
            f"  [process_frame] step={step}: P0 built {P0.shape}. "
            f"Starting get_both_fast..."
        )
        C, P = get_both_fast(P0, args=args)

        log.debug(
            f"  [process_frame] step={step}: get_both_fast done. "
            f"Starting compute_gb_curvature_fast..."
        )
        gb_dict, boundary_pixels, junction_pixels = compute_gb_curvature_fast(
            C,
            TJ_distance_max=args.tj_distance,
            signed=args.signed,
        )
    else:
        log.debug(
            f"  [process_frame] step={step}: P0 built {P0.shape}. "
            f"Starting get_both..."
        )
        C, P = get_both(P0, args=args)

        log.debug(
            f"  [process_frame] step={step}: get_both done. "
            f"Starting compute_gb_curvature..."
        )
        gb_dict, boundary_pixels, junction_pixels = compute_gb_curvature(
            C,
            TJ_distance_max=args.tj_distance,
            signed=args.signed,
        )

    log.debug(f"  [process_frame] step={step}: done.")
    return step, time_val, P0, C, P, gb_dict, boundary_pixels, junction_pixels


# ---------------------------------------------------------------------------
# HDF5 I/O
# ---------------------------------------------------------------------------

def save_frame_times_csv(filepath: str, frames_data: list, exo,
                         log: logging.Logger = None) -> None:
    """
    Write a CSV with step, time, and grain count (if available) for each frame.
    filepath should be the .h5 path — _times.csv will be substituted.
    """
    csv_path = filepath.replace(".h5", "_times.csv")

    glo_names = exo.glo_varnames()
    has_gt = "grain_tracker" in glo_names
    gt = (
        np.rint(exo.glo_var_series("grain_tracker")).astype(np.int64)
        if has_gt else None
    )

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        if has_gt:
            writer.writerow(["frame", "step", "time", "grains"])
            for frame_num, (step, time_val, *_) in enumerate(frames_data):
                writer.writerow([frame_num, step, time_val, int(gt[step])])
        else:
            writer.writerow(["frame", "step", "time"])
            for frame_num, (step, time_val, *_) in enumerate(frames_data):
                writer.writerow([frame_num, step, time_val])

    if log:
        log.info(f"Frame times CSV written: {csv_path}")
    else:
        print(f"Frame times CSV written: {csv_path}")


def _initialize_streamed_hdf5(filepath: str, args: argparse.Namespace,
                               log: logging.Logger = None) -> None:
    """
    Create a new HDF5 file with the provenance group and an empty
    frames group. Called once before streaming begins.
    """
    with h5py.File(filepath, "w") as hf:
        prov = hf.create_group("provenance")
        prov.create_dataset("tj_distance", data=int(args.tj_distance))
        prov.create_dataset("loop_times",  data=int(args.loop_times))
        prov.create_dataset("signed",      data=bool(args.signed))
        prov.create_dataset("cpus",        data=int(args.cpus))
        prov.create_dataset("hdf5_frames", data=int(args.hdf5_frames))
        prov.create_dataset(
            "hdf5_dt",
            data=float(args.hdf5_dt) if args.hdf5_dt is not None
            else float("nan"),
        )
        prov.create_dataset("fast_mode",   data=bool(args.fast))
        prov.create_dataset("chunk_size",  data=int(args.chunk_size))
        hf.create_group("frames")

    if log:
        log.info(f"Initialized streamed HDF5 with provenance: {filepath}")


def _stream_frame_to_hdf5(filepath: str, frame_num: int,
                           frame_tuple: tuple,
                           log: logging.Logger = None) -> None:
    """Append a single frame to an existing HDF5 file."""
    step, time_val, P0, C, P, gb_dict, boundary_pixels, junction_pixels = \
        frame_tuple

    with h5py.File(filepath, "a") as hf:
        fg = hf["frames"].create_group(f"frame_{frame_num:04d}")

        fg.create_dataset("step", data=int(step))
        fg.create_dataset("time", data=float(time_val))
        fg.create_dataset("P0",   data=np.rint(P0).astype(np.int32),
                          compression="gzip")
        fg.create_dataset("C",    data=C, compression="gzip")
        fg.create_dataset("P",    data=P, compression="gzip")

        if len(boundary_pixels) > 0:
            fg.create_dataset(
                "boundary_pixels",
                data=np.array(boundary_pixels, dtype=np.int32),
                compression="gzip",
            )
        else:
            fg.create_dataset("boundary_pixels",
                              data=np.empty((0, 2), dtype=np.int32))

        if len(junction_pixels) > 0:
            fg.create_dataset(
                "junction_pixels",
                data=np.array(junction_pixels, dtype=np.int32),
                compression="gzip",
            )
        else:
            fg.create_dataset("junction_pixels",
                              data=np.empty((0, 2), dtype=np.int32))

        gb_grp = fg.create_group("gb_dict")
        if gb_dict:
            gb_grp.create_dataset(
                "pair_ids",
                data=np.array(list(gb_dict.keys()), dtype=np.int32),
                compression="gzip",
            )
            gb_grp.create_dataset(
                "data",
                data=np.array(list(gb_dict.values()), dtype=np.float64),
                compression="gzip",
            )
        else:
            gb_grp.create_dataset("pair_ids",
                                  data=np.empty((0, 2), dtype=np.int32))
            gb_grp.create_dataset("data",
                                  data=np.empty((0, 5), dtype=np.float64))

    if log:
        log.info(
            f"  Streamed frame_{frame_num:04d} (step={step}) to {filepath}."
        )


def save_hdf5_multiframe(filepath: str, frames_data: list,
                          args: argparse.Namespace,
                          log: logging.Logger = None) -> None:
    """
    Save multi-frame grain boundary curvature data to an HDF5 file.

    HDF5 structure
    --------------
    /provenance/
        tj_distance    (scalar int)
        loop_times     (scalar int)
        signed         (scalar bool)
        cpus           (scalar int)
        hdf5_frames    (scalar int)
        hdf5_dt        (scalar float or NaN)
        fast_mode      (scalar bool)
        chunk_size     (scalar int)
    /frames/
        frame_0000/
            step               (scalar int)
            time               (scalar float)
            P0                 (nx x ny int32)
            C                  (2 x nx x ny float64)
            P                  (3 x nx x ny float64)
            boundary_pixels    (N x 2 int32)
            junction_pixels    (M x 2 int32)
            gb_dict/
                pair_ids       (K x 2 int32)
                data           (K x 5 float64)
                               [avg_curv, gb_area, grain_id1,
                                grain_id2, raw_gb_area]

    Parameters
    ----------
    filepath    : str
    frames_data : list of tuples
        Each: (step, time_val, P0, C, P, gb_dict,
               boundary_pixels, junction_pixels)
    args        : argparse.Namespace — for provenance
    log         : Logger or None
    """
    with h5py.File(filepath, "w") as hf:

        # Provenance
        prov = hf.create_group("provenance")
        prov.create_dataset("tj_distance", data=int(args.tj_distance))
        prov.create_dataset("loop_times",  data=int(args.loop_times))
        prov.create_dataset("signed",      data=bool(args.signed))
        prov.create_dataset("cpus",        data=int(args.cpus))
        prov.create_dataset("hdf5_frames", data=int(args.hdf5_frames))
        prov.create_dataset(
            "hdf5_dt",
            data=float(args.hdf5_dt) if args.hdf5_dt is not None
            else float("nan"),
        )
        prov.create_dataset("fast_mode",  data=bool(args.fast))
        prov.create_dataset("chunk_size", data=int(args.chunk_size))

        # Frames
        frames_grp = hf.create_group("frames")

        for frame_num, (step, time_val, P0, C, P, gb_dict,
                        boundary_pixels, junction_pixels) in \
                enumerate(frames_data):
            fg = frames_grp.create_group(f"frame_{frame_num:04d}")

            fg.create_dataset("step", data=int(step))
            fg.create_dataset("time", data=float(time_val))
            fg.create_dataset("P0",   data=np.rint(P0).astype(np.int32),
                              compression="gzip")
            fg.create_dataset("C",    data=C, compression="gzip")
            fg.create_dataset("P",    data=P, compression="gzip")

            if len(boundary_pixels) > 0:
                fg.create_dataset(
                    "boundary_pixels",
                    data=np.array(boundary_pixels, dtype=np.int32),
                    compression="gzip",
                )
            else:
                fg.create_dataset("boundary_pixels",
                                  data=np.empty((0, 2), dtype=np.int32))

            if len(junction_pixels) > 0:
                fg.create_dataset(
                    "junction_pixels",
                    data=np.array(junction_pixels, dtype=np.int32),
                    compression="gzip",
                )
            else:
                fg.create_dataset("junction_pixels",
                                  data=np.empty((0, 2), dtype=np.int32))

            gb_grp = fg.create_group("gb_dict")
            if gb_dict:
                pair_ids = np.array(list(gb_dict.keys()), dtype=np.int32)
                data_arr = np.array(list(gb_dict.values()), dtype=np.float64)
                gb_grp.create_dataset("pair_ids", data=pair_ids,
                                      compression="gzip")
                gb_grp.create_dataset("data",     data=data_arr,
                                      compression="gzip")
            else:
                gb_grp.create_dataset("pair_ids",
                                      data=np.empty((0, 2), dtype=np.int32))
                gb_grp.create_dataset("data",
                                      data=np.empty((0, 5), dtype=np.float64))

    if log:
        log.info(f"Saved {len(frames_data)} frames to {filepath}")
    else:
        print(f"Saved {len(frames_data)} frames to {filepath}")


# ---------------------------------------------------------------------------
# Debug plotting
# ---------------------------------------------------------------------------

def plot_gb_curvature_debug(C: np.ndarray, P: np.ndarray, gb_dict: dict,
                             boundary_pixels: np.ndarray,
                             junction_pixels: np.ndarray,
                             TJ_distance_max: int = 6,
                             stem: str = "debug",
                             figsize: tuple = (18, 12)) -> None:
    """
    Multi-panel diagnostic plot for verifying GB curvature results.

    Panel 1: Raw grain ID map
    Panel 2: Boundary and junction pixel overlay
    Panel 3: Raw curvature field (C[1])
    Panel 4: Per-GB averaged curvature mapped back onto grain structure
    Panel 5: Histogram of averaged GB curvatures
    Panel 6: Inclination quiver plot
    """
    C0 = C[0]
    C1 = C[1]
    nx, ny = C0.shape

    # Build averaged curvature overlay
    curvature_overlay = np.full((nx, ny), np.nan)
    for (i, j) in boundary_pixels:
        central = int(C0[i, j])
        ip, im = (i + 1) % nx, (i - 1) % nx
        jp, jm = (j + 1) % ny, (j - 1) % ny
        neighbors = set([
            int(C0[ip, j]), int(C0[im, j]),
            int(C0[i, jp]), int(C0[i, jm]),
        ])
        neighbors.discard(central)
        if len(neighbors) == 1:
            pair_id = get_pair_id(central, next(iter(neighbors)))
            if pair_id in gb_dict:
                curvature_overlay[i, j] = gb_dict[pair_id][0]

    valid_overlay = curvature_overlay[~np.isnan(curvature_overlay)]
    curv_max = (
        np.nanpercentile(np.abs(valid_overlay), 95)
        if len(valid_overlay) > 0 else 1.0
    )

    avg_curvatures = np.array([v[0] for v in gb_dict.values()])
    gb_areas       = np.array([v[1] for v in gb_dict.values()])

    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.flatten()

    # Panel 1: Grain ID map
    ax = axes[0]
    im0 = ax.imshow(C0, origin="lower", cmap="viridis",
                    interpolation="nearest")
    ax.set_title("Panel 1: Grain ID Map")
    ax.set_xlabel("i")
    ax.set_ylabel("j")
    plt.colorbar(im0, ax=ax, label="Grain ID")

    # Panel 2: Boundary and junction pixel overlay
    ax = axes[1]
    ax.imshow(C0, origin="lower", cmap="gray", alpha=0.3,
              interpolation="nearest")
    if len(boundary_pixels) > 0:
        ax.scatter(boundary_pixels[:, 1], boundary_pixels[:, 0],
                   c="steelblue", s=1, label="Boundary pixels")
    if len(junction_pixels) > 0:
        ax.scatter(junction_pixels[:, 1], junction_pixels[:, 0],
                   c="red", s=4, label="Junction pixels", zorder=3)
        for (ji, jj) in junction_pixels[::max(1, len(junction_pixels) // 20)]:
            circle = plt.Circle(
                (jj, ji), TJ_distance_max,
                color="orange", fill=False, linewidth=0.8, alpha=0.6,
            )
            ax.add_patch(circle)
    ax.set_title(
        "Panel 2: Boundary & Junction Pixels\n(orange = TJ exclusion radius)"
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="upper right", markerscale=4, fontsize=7)

    # Panel 3: Raw curvature field
    ax = axes[2]
    c1_max = np.nanpercentile(np.abs(C1), 95)
    im2 = ax.imshow(C1, origin="lower", cmap="coolwarm",
                    vmin=-c1_max, vmax=c1_max, interpolation="nearest")
    ax.set_title("Panel 3: Raw Curvature Field C[1]")
    ax.set_xlabel("i")
    ax.set_ylabel("j")
    plt.colorbar(im2, ax=ax, label="Curvature")

    # Panel 4: Averaged GB curvature overlay
    ax = axes[3]
    ax.imshow(C0, origin="lower", cmap="gray", alpha=0.3,
              interpolation="nearest")
    im3 = ax.imshow(curvature_overlay, origin="lower", cmap="plasma",
                    vmin=0, vmax=curv_max, interpolation="nearest")
    ax.set_title("Panel 4: Averaged GB Curvature Overlay")
    ax.set_xlabel("i")
    ax.set_ylabel("j")
    plt.colorbar(im3, ax=ax, label="Avg Curvature")

    # Panel 5: Histogram of averaged GB curvatures
    ax = axes[4]
    ax.hist(avg_curvatures, bins=50, color="steelblue",
            edgecolor="white", linewidth=0.4)
    ax.axvline(np.mean(avg_curvatures), color="red", linestyle="--",
               label=f"Mean: {np.mean(avg_curvatures):.4f}")
    ax.axvline(np.median(avg_curvatures), color="orange", linestyle="--",
               label=f"Median: {np.median(avg_curvatures):.4f}")
    ax.set_title("Panel 5: Distribution of Avg GB Curvatures")
    ax.set_xlabel("Avg Curvature")
    ax.set_ylabel("Count")
    ax.legend(fontsize=8)

    # Panel 6: Inclination quiver
    ax = axes[5]
    ax.imshow(C0, origin="lower", cmap="gray", alpha=0.3,
              interpolation="nearest")
    if len(boundary_pixels) > 0:
        xs, ys, us, vs = [], [], [], []
        stride = max(1, len(boundary_pixels) // 2000)
        for (i, j) in boundary_pixels[::stride]:
            dx, dy = myInput.get_grad(P, int(i), int(j))
            mag = np.sqrt(dx ** 2 + dy ** 2)
            if mag > 0:
                xs.append(j)
                ys.append(i)
                us.append(dx)
                vs.append(dy)
        ax.quiver(xs, ys, us, vs, angles="xy", scale_units="xy",
                  scale=0.5, width=0.003, color="cyan", alpha=0.7)
    ax.set_title("Panel 6: GB Inclination Normals")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    plt.suptitle(
        f"GB Curvature Debug — {len(gb_dict)} boundaries detected",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    figname = stem + "_DEBUG_curvature_plot.png"
    fig.savefig(figname, dpi=500, transparent=True)
    plt.close(fig)


def debug_plot(P0, sites, stem):
    dfig = plt.figure(figsize=(10, 4))
    ax0 = dfig.add_subplot(1, 2, 1)
    ax1 = dfig.add_subplot(1, 2, 2)

    im = ax0.imshow(P0, origin="lower", aspect="auto")
    plt.colorbar(im, ax=ax0)
    ax0.set_aspect("equal")

    nx = P0.shape[0]
    ny = P0.shape[1]
    arr = np.array(sites)
    x = arr[:, 0]
    y = arr[:, 1]
    ax1.scatter(x, y, s=5, c="red")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_aspect("equal")
    ax1.set_xlim([0, nx])
    ax1.set_ylim([0, ny])

    dfig.tight_layout()
    figname = stem + "_DEBUG_GB_plot.png"
    dfig.savefig(figname, dpi=500, transparent=True)
    plt.close(dfig)


def debug_quiver(
    P,
    sites,
    x_centers,
    y_centers,
    stem,
    *,
    get_grad=myInput.get_grad,
    normalize=True,
    scale=0.5,
    width=0.003,
    background=None,
    figsize=(7, 6),
    title="Normals from get_grad at boundary sites",
    ax=None,
    show=True,
):
    """
    Plot quiver arrows at given (i,j) sites using dx,dy from get_grad(P,i,j).
    """
    x_centers = np.asarray(x_centers)
    y_centers = np.asarray(y_centers)

    xs, ys, us, vs = [], [], [], []
    for (i, j) in sites:
        dx, dy = get_grad(P, int(i), int(j))
        xs.append(x_centers[j])
        ys.append(y_centers[i])
        us.append(dx)
        vs.append(dy)

    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    us = np.asarray(us, dtype=float)
    vs = np.asarray(vs, dtype=float)

    if normalize:
        mag = np.sqrt(us ** 2 + vs ** 2)
        nz = mag > 0
        us[nz] /= mag[nz]
        vs[nz] /= mag[nz]

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    if background is not None:
        bg = np.asarray(background)
        X, Y = np.meshgrid(x_centers, y_centers)
        ax.pcolormesh(X, Y, bg, shading="nearest")

    ax.set_aspect("equal")
    ax.quiver(xs, ys, us, vs, angles="xy", scale_units="xy",
              scale=scale, width=width)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    plt.tight_layout()
    figname = stem + "_DEBUG_GB_quiver_plot.png"
    fig.savefig(figname, dpi=500, transparent=True)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ti = time.perf_counter()
    args = parse_args()
    log = setup_logging(args.verbose)

    log.info("Setup:")
    log.info(f"Arguments: {args}")

    if args.fast:
        log.warning(
            f"\033[32m--fast mode enabled\033[0m  "
            f"(chunk_size={args.chunk_size}, "
            f"loop_times={args.loop_times})"
        )

    exo_files = find_exodus_files(subdirs=args.subdirs)
    if not exo_files:
        where = "subdirectories" if args.subdirs else "current directory"
        raise SystemExit(f"No .e files found in {where}.")

    log.info(" ")
    log.info("Exodus Files:")
    for ef in exo_files:
        log.info(f"  File: {ef}")
        log.info(f"  Namebase: {exodus_stem(ef)}")
    log.info(" ")

    for cnt, exofile in enumerate(exo_files):
        til = time.perf_counter()
        stem = exodus_stem(exofile)

        if len(exo_files) > 1:
            log.warning(" ")
            log.warning(
                "\033[1m\033[96m" + "File " + str(cnt + 1) + "/" +
                str(len(exo_files)) + ": " + "\x1b[0m" + str(stem)
            )

        try:
            with ExodusBasics(exofile) as exo:

                step = select_step(
                    exo, grains=args.grains, time_value=args.time, log=log
                )

                log.info("Starting multi-frame HDF5 export...")
                tih = time.perf_counter()

                frame_steps = select_multi_frame_steps(
                    exo,
                    target_step=step,
                    n_frames=args.hdf5_frames,
                    dt=args.hdf5_dt,
                    mode="end",
                    log=log,
                )

                # Resolve output path
                if args.out is not None:
                    hdf5_out = Path(args.out)
                    hdf5_out.parent.mkdir(parents=True, exist_ok=True)
                    hdf5_name = (
                        str(hdf5_out)
                        if str(hdf5_out).endswith(".h5")
                        else str(hdf5_out) + ".h5"
                    )
                    hdf5_path = hdf5_name
                else:
                    hdf5_path = stem + "_multiframe.h5"

                frames_data = []
                log.info("Processing HDF5 Frames:")

                if args.stream:
                    frames_data_for_csv = []
                    _initialize_streamed_hdf5(hdf5_path, args, log=log)

                    for frame_num, (s, t) in progress(
                        enumerate(frame_steps),
                        desc="Streaming frames",
                        verbose=args.verbose,
                        total=len(frame_steps),
                    ):
                        log.info(f"  HDF5 frame: step={s}, time={t:.6g}")
                        tif = time.perf_counter()

                        frame_tuple = process_frame(exo, s, args, log)

                        # Keep only (step, time_val) for CSV
                        frames_data_for_csv.append(
                            (frame_tuple[0], frame_tuple[1])
                        )
                        _stream_frame_to_hdf5(
                            hdf5_path, frame_num, frame_tuple, log=log
                        )
                        vtf(tif, log,
                            extra=f"  Frame {frame_num} (step={s}) process: ")

                        # Retain the target frame for debug plotting;
                        # discard all others immediately.
                        if s == step:
                            frames_data = [frame_tuple]
                        # frame_tuple goes out of scope — GC can collect it

                    save_frame_times_csv(
                        hdf5_path, frames_data_for_csv, exo, log=log
                    )

                else:
                    for frame_num, (s, t) in progress(
                        enumerate(frame_steps),
                        desc="Processing all frames",
                        verbose=args.verbose,
                        total=len(frame_steps),
                    ):
                        log.info(f"  HDF5 frame: step={s}, time={t:.6g}")
                        tif = time.perf_counter()
                        frame_tuple = process_frame(exo, s, args, log)
                        vtf(tif, log,
                            extra=f"  Frame {frame_num} (step={s}) process: ")
                        frames_data.append(frame_tuple)

                    tif = time.perf_counter()
                    save_hdf5_multiframe(hdf5_path, frames_data, args, log=log)
                    save_frame_times_csv(
                        hdf5_path, frames_data, exo, log=log
                    )
                    vtf(tif, log,
                        extra=f"  Batch write ({len(frames_data)} frames): ")

                log.info(
                    f"HDF5 written: {hdf5_path} ({len(frame_steps)} frames)"
                )
                vtf(tih, log, "End of HDF5 generation: ")
                log.info(" ")

                # Debug plot on the target frame only
                if args.debug_plot or args.verbose >= 2:
                    tid = time.perf_counter()

                    target_tuple = next(
                        (f for f in frames_data if f[0] == step),
                        frames_data[-1],
                    )
                    _, _, P0_dbg, C_dbg, P_dbg, gb_dict_dbg, \
                        bp_dbg, jp_dbg = target_tuple

                    log.info("--- Debug Frame Shape Info ---")
                    log.info(f"  P0 shape:              {P0_dbg.shape}")
                    log.info(f"  C  shape:              {C_dbg.shape}")
                    log.info(f"  P  shape:              {P_dbg.shape}")
                    log.info(f"  boundary_pixels shape: {bp_dbg.shape}")
                    log.info(f"  junction_pixels shape: {jp_dbg.shape}")
                    log.info(f"  gb_dict entries:       {len(gb_dict_dbg)}")
                    log.info(
                        f"  unique grains in P0:   "
                        f"{len(np.unique(np.rint(P0_dbg).astype(int)))}"
                    )
                    log.info(" ")
                    log.info(
                        "  * nx and ny are mesh coordinate specific here, "
                        "not the VECTOR internal (nx,ny) = P0.shape"
                    )
                    log.info("------------------------------")

                    plot_gb_curvature_debug(
                        C_dbg, P_dbg, gb_dict_dbg, bp_dbg, jp_dbg,
                        stem=stem,
                        TJ_distance_max=args.tj_distance,
                    )
                    log.info(" ")
                    vtf(tid, log, "End of debug plotting: ")
                    log.info(" ")

        except Exception as e:
            log.exception(f"ERROR: {e}")
            sys.exit(2)

        if len(exo_files) > 1:
            tf(til, log, extra=f"File {cnt + 1} ")

    if len(exo_files) > 1:
        log.warning(" ")
    tf(ti, log, extra="Total ")


if __name__ == "__main__":
    main()
