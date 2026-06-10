#!/usr/bin/env python3
"""
mat_to_hdf5_vectorized.py

Convert SPPARKS Monte Carlo Potts model grain growth results (stored as a
MATLAB v7.3 .mat file) to the same HDF5 curvature output as
vector_exodus_to_hdf5_vectorized.py.

The input .mat file must contain a variable 'Grainims_id' with shape
(n_steps, 1, ny, nx), dtype uint16.

Usage example (standard path):
    python mat_to_hdf5_vectorized.py -i SPPARKS2400_Simulation1_Grainims_id.mat
        -g 5000 -l 20 --stream --hdf5-frames 50 --hdf5-dt 2.4 --out my_output -v

Usage example (fast vectorized path):
    python mat_to_hdf5_vectorized.py -i SPPARKS2400_Simulation1_Grainims_id.mat
        -g 5000 -l 20 --stream --hdf5-frames 50 --hdf5-dt 2.4
        --fast --chunk-size 50000 --out my_output -v

The .mat file is read via h5py (MATLAB v7.3 / HDF5 format).
Frames are accessed lazily one at a time to keep peak RAM at ~11 MB per frame
rather than loading all 1601 x 2400 x 2400 x 2 bytes (~18.5 GB) at once.
"""

import PACKAGE_MP_Linear_Vectorized as smooth
import myInput

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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    VALID_CPUS = (1, 2, 4, 8, 16, 32, 64, 128)
    p = argparse.ArgumentParser(
        description=(
            "Convert SPPARKS Monte Carlo Potts model GG results to curvature "
            "measurements, using Lin's VECTOR smoothing algorithms. Outputs the "
            "same HDF5 schema as vector_exodus_to_hdf5_vectorized.py. "
            "This code assumes a uniform structured 2D mesh."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---- General ----
    gen = p.add_argument_group("General")
    gen.add_argument(
        "-v", "--verbose", action="count", default=0,
        help="Increase verbosity (-v, -vv, -vvv).",
    )
    gen.add_argument(
        "-n", "--cpus", type=int, default=4, choices=VALID_CPUS,
        help=(
            "Number of CPUs for smoothing/curvature calculations. "
            "Allowed: powers of 2 (1, 2, 4, 8, 16, 32, 64, 128)."
        ),
    )

    # ---- Input ----
    inp = p.add_argument_group("Input")
    inp.add_argument(
        "-i", "--input", type=str, required=True, metavar="PATH",
        help=(
            "Path to the MATLAB v7.3 .mat file containing 'Grainims_id' "
            "with shape (n_steps, 1, ny, nx), dtype uint16."
        ),
    )
    inp.add_argument(
        "--mat-dt", type=float, default=1.0, metavar="DT",
        help=(
            "Physical time spacing between MAT frames (used to synthesize "
            "the time array). Defaults to 1.0 if unknown."
        ),
    )
    inp.add_argument(
        "--mat-var", type=str, default="Grainims_id", metavar="VAR",
        help="Name of the variable inside the .mat file to load.",
    )

    # ---- Time selection ----
    tim = p.add_argument_group("Time (choose one)")
    grp = tim.add_mutually_exclusive_group(required=True)
    grp.add_argument(
        "-g", "--grains", type=int,
        help=(
            "Target grain count; chooses the frame whose unique grain count "
            "is closest to this value."
        ),
    )
    grp.add_argument(
        "-t", "--time", type=float,
        help="Target time; chooses the frame with synthesized time closest to this value.",
    )

    # ---- Curvature/Inclination ----
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
        help="If set, use unsigned abs() curvature.",
    )

    # ---- Fast vectorized path ----
    fast_grp = p.add_argument_group("Fast vectorized path (--fast)")
    fast_grp.add_argument(
        "--fast", action="store_true", default=False,
        help=(
            "Use vectorized boundary detection, GB accumulation, and "
            "smoothing core. Recommended for large meshes (>1000x1000) "
            "or high grain counts (>5000)."
        ),
    )
    fast_grp.add_argument(
        "--chunk-size", type=int, default=50_000, metavar="N",
        dest="chunk_size",
        help=(
            "Number of boundary pixels processed per vectorized chunk "
            "in --fast mode. Reduce if OOM; increase for speed. "
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
            "If not set, defaults to <mat_stem>_multiframe.h5."
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
            "If not set, defaults to 1%% of the anchor frame's synthesized time."
        ),
    )
    mf.add_argument(
        "--mode", type=str, default="end", choices=["end", "center"],
        help=(
            "Frame selection mode relative to the anchor step. "
            "'end' = anchor is the last frame; 'center' = anchor is the middle frame."
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
    return logging.getLogger("MAT2HDF5")


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
# MAT file I/O layer
# ---------------------------------------------------------------------------

def open_mat_grainims(mat_path: str, var_name: str = "Grainims_id"):
    """
    Open a MATLAB v7.3 .mat file via h5py and return the dataset handle.

    The dataset is NOT loaded into memory here — callers slice it lazily
    one frame at a time via grainims_ds[step, 0, :, :].

    MATLAB v7.3 files store arrays in Fortran (column-major) order under
    h5py, so axes appear transposed relative to MATLAB notation.
    The on-disk shape is (nx, ny, 1, n_steps) in h5py, which we transpose
    to (n_steps, 1, ny, nx) to match the documented (1601, 1, 2400, 2400)
    convention. We handle this transparently in read_mat_frame().

    Parameters
    ----------
    mat_path : str
    var_name : str

    Returns
    -------
    h5py.File handle (must be closed by caller) and the dataset object.
    Call hf.close() when done.
    """
    hf = h5py.File(mat_path, "r")
    if var_name not in hf:
        available = list(hf.keys())
        raise KeyError(
            f"Variable '{var_name}' not found in {mat_path}. "
            f"Available keys: {available}"
        )
    ds = hf[var_name]
    return hf, ds


def get_mat_shape(ds) -> tuple:
    """
    Return (n_steps, ny, nx) from the h5py dataset, accounting for
    MATLAB's Fortran-order transposition.

    MATLAB saves (n_steps, 1, ny, nx) but h5py reads it as (nx, ny, 1, n_steps).
    We detect the layout by inspecting which axis is size-1 (the singleton).
    If no axis is size 1, we assume the last axis is n_steps and the first
    two are spatial.
    """
    shape = ds.shape  # h5py shape — Fortran-transposed from MATLAB
    # MATLAB (1601, 1, 2400, 2400) -> h5py (2400, 2400, 1, 1601)
    # Find the singleton axis
    singleton_axes = [i for i, s in enumerate(shape) if s == 1]
    if len(singleton_axes) == 1:
        # Remove singleton, remaining are (spatial..., n_steps) or (n_steps, spatial...)
        reduced = [s for i, s in enumerate(shape) if i not in singleton_axes]
        # Convention: MATLAB stores timestep as slowest (first) axis,
        # h5py transposes it to fastest (last) axis.
        n_steps = reduced[-1]
        spatial = reduced[:-1]  # [nx, ny] in h5py = [ny, nx] after transpose
        ny, nx = spatial[1], spatial[0]
    else:
        # Fallback: assume shape is (nx, ny, n_steps)
        n_steps = shape[-1]
        ny, nx = shape[1], shape[0]
    return n_steps, ny, nx


def read_mat_frame(ds, step: int) -> np.ndarray:
    """
    Read one frame from the h5py dataset and return it as a 2D int32 array
    of shape (ny, nx), accounting for MATLAB's Fortran-order transposition.

    MATLAB (n_steps, 1, ny, nx) is stored on disk as (nx, ny, 1, n_steps)
    by h5py. We read ds[:, :, 0, step] and transpose to get (ny, nx).

    Parameters
    ----------
    ds   : h5py Dataset
    step : int — 0-based frame index

    Returns
    -------
    np.ndarray, shape (ny, nx), dtype int32
    """
    shape = ds.shape
    singleton_axes = [i for i, s in enumerate(shape) if s == 1]

    if len(singleton_axes) == 1:
        sa = singleton_axes[0]
        # Build a full index tuple; slice everything except singleton and step axes.
        # For shape (nx, ny, 1, n_steps): read [:, :, 0, step] -> (nx, ny), then .T
        idx = []
        step_axis = len(shape) - 1  # last axis in h5py = first (n_steps) in MATLAB
        for ax, s in enumerate(shape):
            if ax == sa:
                idx.append(0)
            elif ax == step_axis:
                idx.append(step)
            else:
                idx.append(slice(None))
        frame = ds[tuple(idx)]          # shape (nx, ny)
        frame = frame.T                 # -> (ny, nx)
    else:
        # Fallback: last axis = steps, first two = spatial
        frame = ds[:, :, step]          # (nx, ny)
        frame = frame.T                 # -> (ny, nx)

    return frame.astype(np.int32)


def synthesize_time(n_steps: int, dt: float) -> np.ndarray:
    """
    Build a synthetic time array equivalent to exo.time().

    Returns np.arange(n_steps) * dt.
    """
    return np.arange(n_steps, dtype=np.float64) * dt


def compute_grain_count_at_step(ds, step: int) -> int:
    """
    Count unique grain IDs at a single frame.
    Reads only that one frame to avoid loading the full dataset.
    """
    frame = read_mat_frame(ds, step)
    return int(np.unique(frame).size)


def compute_grain_counts_for_steps(ds, steps: list) -> dict:
    """
    Compute unique grain counts for the specified step indices only.

    Returns
    -------
    dict : {step_index: grain_count}
    """
    counts = {}
    for s in steps:
        counts[s] = compute_grain_count_at_step(ds, s)
    return counts


# ---------------------------------------------------------------------------
# Step selection (adapted from the Exodus version — operates on plain arrays)
# ---------------------------------------------------------------------------

def closest_index(values: np.ndarray, target: float) -> int:
    """Return index of entry closest to target. Ties -> first occurrence."""
    values = np.asarray(values)
    return int(np.argmin(np.abs(values - target)))


def select_step_mat(ds, times: np.ndarray, *,
                    grains: int = None,
                    time_value: float = None,
                    log: logging.Logger) -> int:
    """
    Select the single anchor timestep index (0-based) from a MAT dataset.

    Mirrors select_step() from the Exodus version, but reads grain counts
    lazily from the h5py dataset rather than from a grain_tracker variable.

    Parameters
    ----------
    ds         : h5py Dataset (Grainims_id)
    times      : np.ndarray — synthesized time array
    grains     : int or None — target grain count
    time_value : float or None — target synthesized time
    log        : Logger

    Returns
    -------
    int : timestep index
    """
    if time_value is not None:
        step = closest_index(times, float(time_value))
        log.info(
            f"Frame selected by time: requested={time_value}, "
            f"chosen step={step}, time={times[step]}"
        )
        return step

    if grains is not None:
        n_steps = len(times)
        log.info(
            f"Scanning all {n_steps} frames for grain counts "
            f"(target={grains}). This may take a moment..."
        )
        all_counts = np.array(
            [compute_grain_count_at_step(ds, s) for s in range(n_steps)],
            dtype=np.int64,
        )
        step = closest_index(all_counts, int(grains))
        log.info(
            f"Frame selected by grains: requested={grains}, "
            f"chosen step={step}, grain_count={all_counts[step]}"
        )
        return step

    raise ValueError("Either grains or time_value must be provided.")


def select_multi_frame_steps_mat(
    ds,
    times: np.ndarray,
    target_step: int,
    n_frames: int,
    dt=None,
    mode: str = "end",
    log: logging.Logger = None,
) -> list:
    """
    Select up to n_frames timestep indices anchored to target_step.

    Mirrors select_multi_frame_steps() exactly, but uses the synthesized
    times array and lazy grain count reads instead of an open ExodusBasics.

    Parameters
    ----------
    ds          : h5py Dataset (for optional grain count annotation in logs)
    times       : np.ndarray — synthesized time array
    target_step : int
    n_frames    : int
    dt          : float or None — target time spacing between frames
    mode        : "end" | "center"
    log         : Logger or None

    Returns
    -------
    list of (step_index, time_value) tuples, sorted ascending.
    """
    times = np.asarray(times, dtype=float)
    n_total = len(times)
    t_anchor = times[target_step]

    # --- Default dt ---
    if dt is None:
        ref = t_anchor if t_anchor > 0 else times[-1]
        dt = ref * 0.01
        if log:
            log.info(
                f"HDF5 dt defaulted to {dt:.4g} "
                f"(1% of anchor time {ref:.4g})"
            )

    # --- Build ideal target times ---
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
            log.info(f"  step={idx}, time={t:.6g}")

    return result


# ---------------------------------------------------------------------------
# Smoothing wrappers (identical to the Exodus version)
# ---------------------------------------------------------------------------

def get_both(P0: np.ndarray, args) -> tuple:
    """
    Standard (scalar) path: compute normal vectors and curvature.
    Identical to the Exodus version — operates purely on the P0 grid.
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
    Fast (vectorized) path: compute normal vectors and curvature.
    Identical to the Exodus version — operates purely on the P0 grid.
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


# ---------------------------------------------------------------------------
# GB pair utilities (identical to the Exodus version)
# ---------------------------------------------------------------------------

def get_pair_id(grain_a: int, grain_b: int) -> tuple:
    return (min(grain_a, grain_b), max(grain_a, grain_b))


# ---------------------------------------------------------------------------
# Standard (scalar) GB property accumulation (identical to the Exodus version)
# ---------------------------------------------------------------------------

def accumulate_gb_properties(C: np.ndarray, TJ_distance_max: int = 6,
                              signed: bool = True) -> tuple:
    C0 = C[0]
    C1 = C[1]
    nx, ny = C0.shape

    raw_gb_dict   = {}
    junction_dict = {}
    boundary_pixels = []
    junction_pixels = []

    # PASS 1
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

            neighbor_id = next(iter(neighbors))
            pair_id = get_pair_id(central, neighbor_id)
            grain_id1, grain_id2 = pair_id

            if pair_id not in raw_gb_dict:
                raw_gb_dict[pair_id] = np.array(
                    [0.0, 0.0, 0.0, float(grain_id1), float(grain_id2)]
                )
            raw_gb_dict[pair_id][2] += 1.0

    # PASS 2
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
# Vectorized GB property accumulation (identical to the Exodus version)
# ---------------------------------------------------------------------------

def _detect_boundary_and_junction_vectorized(C0: np.ndarray) -> tuple:
    right = np.roll(C0, -1, axis=0)
    left  = np.roll(C0,  1, axis=0)
    up    = np.roll(C0, -1, axis=1)
    down  = np.roll(C0,  1, axis=1)

    diff_r = (right != C0)
    diff_l = (left  != C0)
    diff_u = (up    != C0)
    diff_d = (down  != C0)

    is_boundary = diff_r | diff_l | diff_u | diff_d

    is_junction = np.zeros_like(C0, dtype=bool)
    neighbor_arrays = [(right, diff_r), (left, diff_l),
                       (up,    diff_u), (down, diff_d)]

    for idx_a in range(4):
        for idx_b in range(idx_a + 1, 4):
            n_a, d_a = neighbor_arrays[idx_a]
            n_b, d_b = neighbor_arrays[idx_b]
            is_junction |= (d_a & d_b & (n_a != n_b))

    is_clean = is_boundary & ~is_junction

    neighbor_map = np.zeros_like(C0, dtype=np.int32)
    neighbor_map[is_clean & diff_d] = down [is_clean & diff_d].astype(np.int32)
    neighbor_map[is_clean & diff_u] = up   [is_clean & diff_u].astype(np.int32)
    neighbor_map[is_clean & diff_l] = left [is_clean & diff_l].astype(np.int32)
    neighbor_map[is_clean & diff_r] = right[is_clean & diff_r].astype(np.int32)

    return is_boundary, is_junction, neighbor_map


def _build_tj_exclusion_mask(is_junction: np.ndarray,
                              TJ_distance_max: int) -> np.ndarray:
    if not np.any(is_junction):
        return np.zeros_like(is_junction, dtype=bool)
    dist_from_tj = distance_transform_edt(~is_junction)
    return dist_from_tj < (float(TJ_distance_max) - 1e-10)


def accumulate_gb_properties_fast(C: np.ndarray,
                                   TJ_distance_max: int = 6,
                                   signed: bool = True) -> tuple:
    C0 = C[0].astype(np.int32)
    C1 = C[1]
    nx, ny = C0.shape

    is_boundary, is_junction, neighbor_map = \
        _detect_boundary_and_junction_vectorized(C0)

    is_clean = is_boundary & ~is_junction

    tj_exclusion_mask = _build_tj_exclusion_mask(is_junction, TJ_distance_max)

    accumulate_mask = is_clean & ~tj_exclusion_mask

    boundary_pixels = np.argwhere(is_boundary)
    junction_pixels = np.argwhere(is_junction)
    clean_pixels    = np.argwhere(is_clean)
    accum_pixels    = np.argwhere(accumulate_mask)

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
                0.0,
                0.0,
                float(raw_area_counts[k]),
                float(g1),
                float(g2),
            ])

    if len(accum_pixels) > 0:
        ai = accum_pixels[:, 0]
        aj = accum_pixels[:, 1]
        central_ids  = C0[ai, aj].astype(np.int64)
        neighbor_ids = neighbor_map[ai, aj].astype(np.int64)
        pair_min = np.minimum(central_ids, neighbor_ids)
        pair_max = np.maximum(central_ids, neighbor_ids)

        curv_vals = C1[ai, aj].copy()

        if signed:
            flip_mask = central_ids != pair_min
            curv_vals[flip_mask] *= -1.0
        else:
            curv_vals = np.abs(curv_vals)

        unique_pairs_a, inverse_a = np.unique(
            np.stack([pair_min, pair_max], axis=1),
            axis=0,
            return_inverse=True,
        )
        valid_counts  = np.bincount(inverse_a, minlength=len(unique_pairs_a))
        sum_curvature = np.bincount(inverse_a, weights=curv_vals,
                                    minlength=len(unique_pairs_a))

        for k, (g1, g2) in enumerate(unique_pairs_a):
            pid = (int(g1), int(g2))
            if pid in raw_gb_dict:
                raw_gb_dict[pid][0] = float(valid_counts[k])
                raw_gb_dict[pid][1] = float(sum_curvature[k])

    return raw_gb_dict, boundary_pixels, junction_pixels


# ---------------------------------------------------------------------------
# GB property averaging (identical to the Exodus version)
# ---------------------------------------------------------------------------

def average_gb_properties(raw_gb_dict: dict) -> dict:
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
# Top-level curvature orchestrators (identical to the Exodus version)
# ---------------------------------------------------------------------------

def compute_gb_curvature(C: np.ndarray, TJ_distance_max: int = 6,
                         signed: bool = True) -> tuple:
    raw_gb_dict, boundary_pixels, junction_pixels = accumulate_gb_properties(
        C, TJ_distance_max=TJ_distance_max, signed=signed
    )
    gb_dict = average_gb_properties(raw_gb_dict)
    return gb_dict, boundary_pixels, junction_pixels


def compute_gb_curvature_fast(C: np.ndarray, TJ_distance_max: int = 6,
                               signed: bool = True) -> tuple:
    raw_gb_dict, boundary_pixels, junction_pixels = \
        accumulate_gb_properties_fast(
            C, TJ_distance_max=TJ_distance_max, signed=signed
        )
    gb_dict = average_gb_properties(raw_gb_dict)
    return gb_dict, boundary_pixels, junction_pixels


# ---------------------------------------------------------------------------
# Frame processing — MAT version (replaces process_frame)
# ---------------------------------------------------------------------------

def process_frame_mat(ds, step: int, times: np.ndarray, args,
                      log: logging.Logger) -> tuple:
    """
    Process a single MAT timestep into grids and GB data.

    Replaces process_frame() from the Exodus version. The key difference is
    that we read a pre-structured 2D grid directly from the h5py dataset
    rather than reading an unstructured mesh and calling map_to_grid().

    The grain ID grid P0 is read as:
        P0 = Grainims_id[step, 0, :, :].astype(float64)

    Because linear_class infers id_offset from np.nanmin(P0), no manual
    grain ID normalization is needed even if the minimum ID is not 0 or 1.

    Parameters
    ----------
    ds    : h5py Dataset (Grainims_id)
    step  : int — 0-based frame index
    times : np.ndarray — synthesized time array
    args  : argparse.Namespace
    log   : Logger

    Returns
    -------
    tuple : (step, time_val, P0, C, P, gb_dict, boundary_pixels, junction_pixels)
            Identical structure to process_frame() output.
    """
    time_val = float(times[step])

    log.debug(f"  [process_frame_mat] step={step}: reading frame from .mat...")
    frame = read_mat_frame(ds, step)          # shape (ny, nx), int32
    P0 = frame.astype(np.float64)             # linear_class expects float

    log.debug(
        f"  [process_frame_mat] step={step}: P0 shape={P0.shape}, "
        f"unique grains={len(np.unique(frame))}, "
        f"id range=[{frame.min()}, {frame.max()}]"
    )

    if args.fast:
        log.debug(
            f"  [process_frame_mat] step={step}: starting get_both_fast..."
        )
        C, P = get_both_fast(P0, args=args)
        log.debug(
            f"  [process_frame_mat] step={step}: get_both_fast done. "
            f"starting compute_gb_curvature_fast..."
        )
        gb_dict, boundary_pixels, junction_pixels = compute_gb_curvature_fast(
            C,
            TJ_distance_max=args.tj_distance,
            signed=args.signed,
        )
    else:
        log.debug(
            f"  [process_frame_mat] step={step}: starting get_both..."
        )
        C, P = get_both(P0, args=args)
        log.debug(
            f"  [process_frame_mat] step={step}: get_both done. "
            f"starting compute_gb_curvature..."
        )
        gb_dict, boundary_pixels, junction_pixels = compute_gb_curvature(
            C,
            TJ_distance_max=args.tj_distance,
            signed=args.signed,
        )

    log.debug(f"  [process_frame_mat] step={step}: done.")
    return step, time_val, P0, C, P, gb_dict, boundary_pixels, junction_pixels


# ---------------------------------------------------------------------------
# HDF5 I/O (identical to the Exodus version except provenance additions)
# ---------------------------------------------------------------------------

def _initialize_streamed_hdf5(filepath: str, args: argparse.Namespace,
                               mat_path: str,
                               log: logging.Logger = None) -> None:
    """
    Create a new HDF5 file with the provenance group and an empty frames group.
    Adds source_file and mat_dt to provenance compared to the Exodus version.
    """
    with h5py.File(filepath, "w") as hf:
        prov = hf.create_group("provenance")
        prov.create_dataset("tj_distance",  data=int(args.tj_distance))
        prov.create_dataset("loop_times",   data=int(args.loop_times))
        prov.create_dataset("signed",       data=bool(args.signed))
        prov.create_dataset("cpus",         data=int(args.cpus))
        prov.create_dataset("hdf5_frames",  data=int(args.hdf5_frames))
        prov.create_dataset(
            "hdf5_dt",
            data=float(args.hdf5_dt) if args.hdf5_dt is not None else float("nan"),
        )
        prov.create_dataset("fast_mode",    data=bool(args.fast))
        prov.create_dataset("chunk_size",   data=int(args.chunk_size))
        prov.create_dataset("source_file",  data=str(mat_path))
        prov.create_dataset("mat_dt",       data=float(args.mat_dt))
        hf.create_group("frames")

    if log:
        log.info(f"Initialized streamed HDF5 with provenance: {filepath}")


def _stream_frame_to_hdf5(filepath: str, frame_num: int,
                           frame_tuple: tuple,
                           log: logging.Logger = None) -> None:
    """Append a single frame to an existing HDF5 file. Identical to the Exodus version."""
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
                         mat_path: str,
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
        source_file    (string — path to the .mat file)
        mat_dt         (float  — dt used to synthesize time)
    /frames/
        frame_0000/
            step               (scalar int)
            time               (scalar float)
            P0                 (ny x nx int32)
            C                  (2 x ny x nx float64)
            P                  (3 x ny x nx float64)
            boundary_pixels    (N x 2 int32)
            junction_pixels    (M x 2 int32)
            gb_dict/
                pair_ids       (K x 2 int32)
                data           (K x 5 float64)
                               [avg_curv, gb_area, grain_id1,
                                grain_id2, raw_gb_area]
    """
    with h5py.File(filepath, "w") as hf:

        # Provenance
        prov = hf.create_group("provenance")
        prov.create_dataset("tj_distance",  data=int(args.tj_distance))
        prov.create_dataset("loop_times",   data=int(args.loop_times))
        prov.create_dataset("signed",       data=bool(args.signed))
        prov.create_dataset("cpus",         data=int(args.cpus))
        prov.create_dataset("hdf5_frames",  data=int(args.hdf5_frames))
        prov.create_dataset(
            "hdf5_dt",
            data=float(args.hdf5_dt) if args.hdf5_dt is not None
            else float("nan"),
        )
        prov.create_dataset("fast_mode",    data=bool(args.fast))
        prov.create_dataset("chunk_size",   data=int(args.chunk_size))
        prov.create_dataset("source_file",  data=str(mat_path))
        prov.create_dataset("mat_dt",       data=float(args.mat_dt))

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


def save_frame_times_csv_mat(filepath: str, frames_data: list,
                              grain_counts_map: dict,
                              log: logging.Logger = None) -> None:
    """
    Write a CSV with step, time, and grain count for each frame.

    Mirrors save_frame_times_csv() from the Exodus version but reads grain
    counts from a pre-built dict {step_index: grain_count} instead of an
    open ExodusBasics instance. If the dict is empty, the grains column
    is omitted.

    Parameters
    ----------
    filepath          : str — path to the .h5 file (used to derive CSV path)
    frames_data       : list of frame tuples (step, time_val, ...)
    grain_counts_map  : dict {step_index: grain_count}
    log               : Logger or None
    """
    csv_path = filepath.replace(".h5", "_times.csv")
    has_gt = len(grain_counts_map) > 0

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        if has_gt:
            writer.writerow(["frame", "step", "time", "grains"])
            for frame_num, (step, time_val, *_) in enumerate(frames_data):
                grains = grain_counts_map.get(step, "")
                writer.writerow([frame_num, step, time_val, grains])
        else:
            writer.writerow(["frame", "step", "time"])
            for frame_num, (step, time_val, *_) in enumerate(frames_data):
                writer.writerow([frame_num, step, time_val])

    if log:
        log.info(f"Frame times CSV written: {csv_path}")
    else:
        print(f"Frame times CSV written: {csv_path}")


# ---------------------------------------------------------------------------
# Debug plotting (identical to the Exodus version)
# ---------------------------------------------------------------------------

def plot_gb_curvature_debug(C: np.ndarray, P: np.ndarray, gb_dict: dict,
                             boundary_pixels: np.ndarray,
                             junction_pixels: np.ndarray,
                             TJ_distance_max: int = 6,
                             stem: str = "debug",
                             figsize: tuple = (18, 12)) -> None:
    C0 = C[0]
    C1 = C[1]
    nx, ny = C0.shape

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

    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.flatten()

    ax = axes[0]
    im0 = ax.imshow(C0, origin="lower", cmap="viridis",
                    interpolation="nearest")
    ax.set_title("Panel 1: Grain ID Map")
    ax.set_xlabel("i")
    ax.set_ylabel("j")
    plt.colorbar(im0, ax=ax, label="Grain ID")

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

    ax = axes[2]
    c1_max = np.nanpercentile(np.abs(C1), 95)
    im2 = ax.imshow(C1, origin="lower", cmap="coolwarm",
                    vmin=-c1_max, vmax=c1_max, interpolation="nearest")
    ax.set_title("Panel 3: Raw Curvature Field C[1]")
    ax.set_xlabel("i")
    ax.set_ylabel("j")
    plt.colorbar(im2, ax=ax, label="Curvature")

    ax = axes[3]
    ax.imshow(C0, origin="lower", cmap="gray", alpha=0.3,
              interpolation="nearest")
    im3 = ax.imshow(curvature_overlay, origin="lower", cmap="plasma",
                    vmin=0, vmax=curv_max, interpolation="nearest")
    ax.set_title("Panel 4: Averaged GB Curvature Overlay")
    ax.set_xlabel("i")
    ax.set_ylabel("j")
    plt.colorbar(im3, ax=ax, label="Avg Curvature")

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

    mat_path = Path(args.input)
    if not mat_path.is_file():
        raise SystemExit(f"Input file not found: {mat_path}")

    stem = mat_path.stem

    log.warning(f"Input .mat file: {mat_path}")
    log.warning(f"Variable name:   {args.mat_var}")
    log.warning(f"Synthetic dt:    {args.mat_dt}")

    # -----------------------------------------------------------------------
    # Open the .mat file — keep it open for lazy frame access throughout
    # -----------------------------------------------------------------------
    try:
        hf, ds = open_mat_grainims(str(mat_path), var_name=args.mat_var)
    except Exception as e:
        raise SystemExit(f"Failed to open .mat file: {e}")

    try:
        n_steps, ny, nx = get_mat_shape(ds)
        log.warning(
            f"Dataset shape (h5py raw): {ds.shape}  "
            f"-> interpreted as n_steps={n_steps}, ny={ny}, nx={nx}"
        )

        times = synthesize_time(n_steps, args.mat_dt)
        log.info(
            f"Synthesized time array: [{times[0]:.4g} ... {times[-1]:.4g}], "
            f"n_steps={n_steps}, dt={args.mat_dt}"
        )

        # -------------------------------------------------------------------
        # Step selection
        # -------------------------------------------------------------------
        step = select_step_mat(
            ds, times,
            grains=args.grains,
            time_value=args.time,
            log=log,
        )

        # -------------------------------------------------------------------
        # Multi-frame step list
        # -------------------------------------------------------------------
        log.info("Starting multi-frame HDF5 export...")
        tih = time.perf_counter()

        frame_steps = select_multi_frame_steps_mat(
            ds,
            times,
            target_step=step,
            n_frames=args.hdf5_frames,
            dt=args.hdf5_dt,
            mode=args.mode,
            log=log,
        )

        # -------------------------------------------------------------------
        # Resolve output path
        # -------------------------------------------------------------------
        if args.out is not None:
            hdf5_out = Path(args.out)
            hdf5_out.parent.mkdir(parents=True, exist_ok=True)
            hdf5_path = (
                str(hdf5_out)
                if str(hdf5_out).endswith(".h5")
                else str(hdf5_out) + ".h5"
            )
        else:
            hdf5_path = stem + "_multiframe.h5"

        # -------------------------------------------------------------------
        # Process frames and write HDF5
        # -------------------------------------------------------------------
        frames_data = []
        grain_counts_map = {}
        log.info("Processing HDF5 Frames:")

        if args.stream:
            frames_data_for_csv = []
            _initialize_streamed_hdf5(hdf5_path, args, str(mat_path), log=log)

            for frame_num, (s, t) in progress(
                enumerate(frame_steps),
                desc="Streaming frames",
                verbose=args.verbose,
                total=len(frame_steps),
            ):
                log.info(f"  HDF5 frame: step={s}, time={t:.6g}")
                tif = time.perf_counter()

                frame_tuple = process_frame_mat(ds, s, times, args, log)

                # Record grain count for CSV (cheap since frame is already read)
                grain_counts_map[s] = int(
                    np.unique(np.rint(frame_tuple[2]).astype(np.int32)).size
                )

                frames_data_for_csv.append(
                    (frame_tuple[0], frame_tuple[1])
                )
                _stream_frame_to_hdf5(
                    hdf5_path, frame_num, frame_tuple, log=log
                )
                vtf(tif, log,
                    extra=f"  Frame {frame_num} (step={s}) process: ")

            save_frame_times_csv_mat(
                hdf5_path, frames_data_for_csv, grain_counts_map, log=log
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
                frame_tuple = process_frame_mat(ds, s, times, args, log)

                grain_counts_map[s] = int(
                    np.unique(np.rint(frame_tuple[2]).astype(np.int32)).size
                )

                vtf(tif, log,
                    extra=f"  Frame {frame_num} (step={s}) process: ")
                frames_data.append(frame_tuple)

            tif = time.perf_counter()
            save_hdf5_multiframe(
                hdf5_path, frames_data, args, str(mat_path), log=log
            )
            save_frame_times_csv_mat(
                hdf5_path, frames_data, grain_counts_map, log=log
            )
            vtf(tif, log,
                extra=f"  Batch write ({len(frames_data)} frames): ")

        log.info(
            f"HDF5 written: {hdf5_path} ({len(frame_steps)} frames)"
        )
        vtf(tih, log, "End of HDF5 generation: ")
        log.info(" ")

        # -------------------------------------------------------------------
        # Debug plot on the target frame only
        # -------------------------------------------------------------------
        if args.debug_plot or args.verbose >= 2:
            tid = time.perf_counter()

            # In streaming mode frames_data is empty; re-process the target
            if args.stream:
                log.info("Re-processing target frame for debug plot...")
                target_tuple = process_frame_mat(ds, step, times, args, log)
            else:
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
        hf.close()
        sys.exit(2)
    finally:
        hf.close()

    tf(ti, log, extra="Total ")


if __name__ == "__main__":
    main()
