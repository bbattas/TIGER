#!/usr/bin/env python3
"""
vector_exodus_to_hdf5.py

Convert phase field grain growth results to curvature measurements using
Lin's VECTOR smoothing algorithms. Outputs raw GB curvature data for
GB velocity calculations at the specified time.

This code assumes uniform mesh elements.

Usage example (standard path):
    python vector_exodus_to_hdf5.py -n 32 -t 120 -l 20 --stream
        --hdf5-frames 50 --hdf5-t0 0.0
        --out my_output -v

Usage example (fast vectorized path):
    python vector_exodus_to_hdf5.py -n 32 -t 120 -l 20 --stream
        --hdf5-frames 50 --hdf5-t0 0.0
        --fast --chunk-size 50000
        --out my_output -v

--hdf5-t0 sets the start of the multi-frame time range. Frames are
    evenly spaced between t0 and the anchor time (-t / -g). Defaults
    to the first available timestep if omitted.
    The anchor frame is always pinned as the final frame.
    The actual frame count used is embedded in the output filename:
        my_output_50f.h5   (all 50 requested frames obtained)
        my_output_38f.h5   (only 38 unique steps available in range)

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
            "If not set, defaults to <stem>_<N>f_multiframe.h5. "
            "The actual frame count used is always embedded in the filename "
            "(e.g. my_output_50f.h5)."
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
        "--hdf5-frames", type=int, default=100, metavar="N",
        help="Number of frames to save in the HDF5 file.",
    )
    mf.add_argument(                                        # CHANGED from --hdf5-dt
        "--hdf5-t0", type=float, default=None, metavar="T0",
        dest="hdf5_t0",
        help=(
            "Start time for the multi-frame range. Frames are selected by "
            "evenly spacing --hdf5-frames steps between T0 and the anchor "
            "time chosen by -t / -g. "
            "If not set, defaults to the first available timestep (t=0). "
            "The anchor frame is always pinned as the final frame."
        ),
    )

    # ---- Recovery mode ----
    rec = p.add_argument_group(
        "Recovery mode",
        description=(
            "Use when a MOOSE simulation was recovered and produced multiple "
            ".e files that together form one continuous run."
        ),
    )
    rec.add_argument(
        "--recover", action="store_true", default=False,
        help=(
            "Enable recovery mode. Discovers all .e files in cwd and all "
            "subdirectories (or in --recover-dirs if supplied), merges their "
            "timesteps into a single deduplicated index, and treats the whole "
            "set as one simulation."
        ),
    )
    rec.add_argument(
        "--recover-dirs", nargs="+", default=None, metavar="DIR",
        dest="recover_dirs",
        help=(
            "Explicit list of directories to search for .e files in recovery "
            "mode. If omitted, defaults to cwd + all subdirectories recursively. "
            "Use this for sibling-directory layouts."
        ),
    )
    rec.add_argument(
        "--rebuild-index", action="store_true", default=False,
        dest="rebuild_index",
        help=(
            "Force regeneration of recovery_index.csv even if it already "
            "exists in the current directory."
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
    t_start: float | None = None,
    log: logging.Logger = None,
) -> tuple[list, int]:
    """
    Select up to n_frames timestep indices spanning [t_start, t_anchor].

    The anchor frame (selected by -t / -g) is always the final frame in
    the returned list. Frames are chosen by evenly spacing n_frames ideal
    times between t_start and t_anchor via np.linspace, then snapping each
    to the nearest available timestep.

    If fewer than n_frames unique steps are available in that range, a
    warning is emitted and the reduced set is returned as-is — the range
    is never silently expanded.

    Parameters
    ----------
    exo         : ExodusBasics open instance
    target_step : int
        The step selected by -t / -g. Pinned as the last frame.
    n_frames    : int
        Desired number of frames (including the anchor frame).
    t_start     : float or None
        Start of the time range. If None, defaults to times[0].
    log         : logging.Logger or None

    Returns
    -------
    tuple of:
        list of (step_index, time_value) tuples, sorted ascending.
        int : actual number of frames selected (may be < n_frames).
    """
    times    = np.asarray(exo.time(), dtype=float)
    t_anchor = times[target_step]
    anchor_idx = target_step  # already exact — no need to search

    # --- Resolve t_start ---
    if t_start is None:
        t_start = float(times[0])
        if log:
            log.info(
                f"HDF5 t_start defaulted to {t_start:.4g} (first available timestep)"
            )

    if t_start > t_anchor:
        raise ValueError(
            f"--hdf5-t0 ({t_start:.4g}) is later than the anchor time "
            f"({t_anchor:.4g}). t_start must be <= anchor time."
        )

    # --- Grain tracker (optional, for logging only) ---
    glo_names = exo.glo_varnames()
    has_gt    = "grain_tracker" in glo_names
    gt = (
        np.rint(exo.glo_var_series("grain_tracker")).astype(np.int64)
        if has_gt else None
    )

    # --- Build ideal times and snap to nearest available steps ---
    ideal_times = np.linspace(t_start, t_anchor, n_frames)
    raw_indices = [closest_index(times, t) for t in ideal_times]

    # --- Deduplicate while preserving order ---
    seen           = set()
    unique_indices = []
    for idx in raw_indices:
        if idx not in seen:
            seen.add(idx)
            unique_indices.append(idx)

    # --- Anchor pin: ensure target_step is present and is the last frame ---
    if anchor_idx not in seen:
        unique_indices.append(anchor_idx)
        if log:
            log.warning(
                f"\033[31mHDF5:\033[0m anchor step={anchor_idx} "
                f"(t={t_anchor:.4g}) was not in snap result — pinned manually."
            )
    unique_indices = sorted(unique_indices)
    # Guarantee anchor is last (it should be after sort, but be explicit)
    if unique_indices[-1] != anchor_idx:
        unique_indices = [i for i in unique_indices if i != anchor_idx]
        unique_indices.append(anchor_idx)

    actual_n = len(unique_indices)

    # --- Warn if we got fewer frames than requested ---
    if actual_n < n_frames:
        if log:
            log.warning(
                f"\033[31mHDF5:\033[0m Requested {n_frames} frames between "
                f"t={t_start:.4g} and t={t_anchor:.4g}, but only "
                f"{actual_n} unique timesteps are available in that range. "
                f"Proceeding with {actual_n} frames."
            )

    result = [(int(idx), float(times[idx])) for idx in unique_indices]

    if log:
        log.info(
            f"HDF5 multi-frame: anchor step={anchor_idx} t={t_anchor:.4g}, "
            f"t_start={t_start:.4g}, "
            f"requested={n_frames}, selected={actual_n}"
        )
        for idx, t in result:
            if has_gt:
                log.info(f"  step={idx}, time={t:.6g}, grains={int(gt[idx])}")
            else:
                log.info(f"  step={idx}, time={t:.6g}")

    return result, actual_n


# ---------------------------------------------------------------------------
# Recovery mode — file discovery
# ---------------------------------------------------------------------------

def find_recovery_files(args, log: logging.Logger) -> list[Path]:
    """
    Discover all .e files for recovery mode.

    If args.recover_dirs is provided, search each listed directory
    non-recursively. Otherwise, search cwd and every subdirectory
    recursively (Option 1 layout).

    Returns sorted, deduplicated list of absolute Paths. Exits cleanly
    if no files are found.
    """
    if args.recover_dirs:
        found = []
        for d in args.recover_dirs:
            dp = Path(d).resolve()
            if not dp.is_dir():
                log.warning(
                    f"  [recover] --recover-dirs entry is not a directory, "
                    f"skipping: {dp}"
                )
                continue
            found.extend(dp.glob("*.e"))
    else:
        found = list(Path.cwd().rglob("*.e"))

    seen = set()
    unique = []
    for p in found:
        rp = p.resolve()
        if rp.is_file() and rp not in seen:
            seen.add(rp)
            unique.append(rp)
    unique.sort()

    if not unique:
        dirs_desc = (
            ", ".join(str(d) for d in args.recover_dirs)
            if args.recover_dirs
            else "cwd + subdirectories"
        )
        raise SystemExit(
            f"Recovery mode: no .e files found in {dirs_desc}."
        )

    log.info(f"[recover] Found {len(unique)} .e file(s):")
    for p in unique:
        log.info(f"  {p}")
    return unique


# ---------------------------------------------------------------------------
# Recovery mode — index build and I/O
# ---------------------------------------------------------------------------

def build_recovery_index(
    exo_files: list[Path],
    log: logging.Logger,
) -> list[dict]:
    """
    Build a per-timestep index across all recovery .e files.

    Reads time_whole and grain_tracker (if present) from each file cheaply,
    sorts files by first timestep, then merges into a flat deduplicated table.
    Overlapping time ranges are resolved by preferring the later file (the
    canonical post-recovery continuation).

    Returns list of dicts: {time, grain_count, file_path, file_step}
    grain_count is -1 if grain_tracker is not present.
    """
    file_meta = []
    for fp in exo_files:
        try:
            with ExodusBasics(str(fp)) as exo:
                times = np.asarray(exo.time(), dtype=float)
                glo_names = exo.glo_varnames()
                if "grain_tracker" in glo_names:
                    gt = np.rint(
                        exo.glo_var_series("grain_tracker")
                    ).astype(np.int64)
                else:
                    gt = None
        except Exception as e:
            log.warning(
                f"  [recover] Could not read {fp}, skipping. Reason: {e}"
            )
            continue

        if len(times) == 0:
            log.warning(f"  [recover] {fp} has no timesteps, skipping.")
            continue

        file_meta.append({
            "path":        fp,
            "times":       times,
            "grain_tracker": gt,
            "first_time":  float(times[0]),
        })
        log.info(
            f"  [recover] {fp.name}: {len(times)} steps, "
            f"t=[{times[0]:.4g}, {times[-1]:.4g}]"
        )

    if not file_meta:
        raise RuntimeError(
            "Recovery index: no readable .e files with timesteps found."
        )

    file_meta.sort(key=lambda m: m["first_time"])

    rows: list[dict] = []
    for meta in file_meta:
        fp_str   = str(meta["path"])
        times    = meta["times"]
        gt       = meta["grain_tracker"]
        cut_time = meta["first_time"]

        # Drop rows from earlier files that overlap with this file's range
        rows = [r for r in rows if r["time"] < cut_time]

        for step_idx, t in enumerate(times):
            gc = int(gt[step_idx]) if gt is not None else -1
            rows.append({
                "time":        float(t),
                "grain_count": gc,
                "file_path":   fp_str,
                "file_step":   step_idx,
            })

    rows.sort(key=lambda r: r["time"])
    log.warning(
        f"[recover] Index built: {len(rows)} unique timesteps across "
        f"{len(file_meta)} file(s)."
    )
    return rows


def save_recovery_index(
    rows: list[dict],
    index_path: Path,
    log: logging.Logger,
) -> None:
    """Write recovery index to CSV: time, grain_count, file_path, file_step."""
    with open(index_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time", "grain_count", "file_path", "file_step"])
        for r in rows:
            writer.writerow([
                r["time"], r["grain_count"], r["file_path"], r["file_step"],
            ])
    log.warning(
        f"[recover] Recovery index written: {index_path} ({len(rows)} rows)"
    )


def load_recovery_index(index_path: Path, log: logging.Logger) -> list[dict]:
    """Load a previously-saved recovery index CSV."""
    rows = []
    with open(index_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "time":        float(row["time"]),
                "grain_count": int(row["grain_count"]),
                "file_path":   str(row["file_path"]),
                "file_step":   int(row["file_step"]),
            })
    log.warning(
        f"[recover] Loaded recovery index: {index_path} ({len(rows)} rows)"
    )
    return rows


# ---------------------------------------------------------------------------
# Recovery mode — timestep selection from merged index
# ---------------------------------------------------------------------------

def select_step_from_index(
    rows: list[dict],
    *,
    grains: int | None = None,
    time_value: float | None = None,
    log: logging.Logger,
) -> dict:
    """
    Select the single anchor row from the merged recovery index.
    Recovery-mode replacement for select_step().
    Returns the chosen row: {time, grain_count, file_path, file_step}
    """
    if time_value is not None:
        times = np.array([r["time"] for r in rows], dtype=float)
        idx   = closest_index(times, float(time_value))
        row   = rows[idx]
        log.info(
            f"[recover] Anchor by time: requested={time_value}, "
            f"chosen time={row['time']:.6g}, "
            f"file={Path(row['file_path']).name}, "
            f"file_step={row['file_step']}"
        )
        return row

    gc_vals = np.array([r["grain_count"] for r in rows], dtype=np.int64)
    if np.all(gc_vals == -1):
        raise RuntimeError(
            "Recovery mode: --grains requested but no file in the index "
            "contains a 'grain_tracker' global variable."
        )
    idx = closest_index(gc_vals, int(grains))
    row = rows[idx]
    log.info(
        f"[recover] Anchor by grains: requested={grains}, "
        f"chosen grain_count={row['grain_count']}, "
        f"time={row['time']:.6g}, "
        f"file={Path(row['file_path']).name}, "
        f"file_step={row['file_step']}"
    )
    return row


def select_multi_frame_steps_from_index(
    rows: list[dict],
    anchor_row: dict,
    n_frames: int,
    t_start: float | None = None,
    log: logging.Logger = None,
) -> tuple[list[dict], int]:
    """
    Select up to n_frames rows from the merged index spanning [t_start, t_anchor].
    Recovery-mode replacement for select_multi_frame_steps().

    The anchor row is always the final entry in the returned list.
    If fewer than n_frames unique steps are available in the range, a
    warning is emitted and the reduced set is returned — the range is
    never silently expanded.

    Parameters
    ----------
    rows       : list of index row dicts (full merged recovery index)
    anchor_row : dict
        The row selected by -t / -g. Pinned as the last frame.
    n_frames   : int
        Desired number of frames (including the anchor frame).
    t_start    : float or None
        Start of the time range. If None, defaults to times[0].
    log        : logging.Logger or None

    Returns
    -------
    tuple of:
        list of index row dicts, sorted ascending by time.
        int : actual number of frames selected (may be < n_frames).
    """
    times      = np.array([r["time"] for r in rows], dtype=float)
    t_anchor   = anchor_row["time"]
    anchor_idx = closest_index(times, t_anchor)  # now used

    # --- Resolve t_start ---
    if t_start is None:
        t_start = float(times[0])
        if log:
            log.info(
                f"[recover] HDF5 t_start defaulted to {t_start:.4g} "
                f"(first available timestep)"
            )

    if t_start > t_anchor:
        raise ValueError(
            f"[recover] --hdf5-t0 ({t_start:.4g}) is later than the anchor "
            f"time ({t_anchor:.4g}). t_start must be <= anchor time."
        )

    # --- Build ideal times and snap to nearest available steps ---
    ideal_times = np.linspace(t_start, t_anchor, n_frames)
    raw_indices = [closest_index(times, t) for t in ideal_times]

    # --- Deduplicate while preserving order ---
    seen           = set()
    unique_indices = []
    for idx in raw_indices:
        if idx not in seen:
            seen.add(idx)
            unique_indices.append(idx)

    # --- Anchor pin: ensure anchor_idx is present and is the last frame ---
    if anchor_idx not in seen:
        unique_indices.append(anchor_idx)
        if log:
            log.warning(
                f"\033[31m[recover] HDF5:\033[0m anchor idx={anchor_idx} "
                f"(t={t_anchor:.4g}) was not in snap result — pinned manually."
            )
    unique_indices = sorted(unique_indices)
    # Guarantee anchor is last (explicit, not assumed from sort)
    if unique_indices[-1] != anchor_idx:
        unique_indices = [i for i in unique_indices if i != anchor_idx]
        unique_indices.append(anchor_idx)

    actual_n = len(unique_indices)

    # --- Warn if fewer frames than requested ---
    if actual_n < n_frames:
        if log:
            log.warning(
                f"\033[31m[recover] HDF5:\033[0m Requested {n_frames} frames "
                f"between t={t_start:.4g} and t={t_anchor:.4g}, but only "
                f"{actual_n} unique timesteps are available in that range. "
                f"Proceeding with {actual_n} frames."
            )

    result = [rows[i] for i in unique_indices]

    if log:
        log.info(
            f"[recover] Multi-frame: anchor t={t_anchor:.4g}, "
            f"t_start={t_start:.4g}, "
            f"requested={n_frames}, selected={actual_n}"
        )
        for r in result:
            gc_str = (
                f", grains={r['grain_count']}"
                if r["grain_count"] != -1 else ""
            )
            log.info(
                f"  time={r['time']:.6g}{gc_str}, "
                f"file={Path(r['file_path']).name}, "
                f"file_step={r['file_step']}"
            )

    return result, actual_n


def _resolve_hdf5_path(out_arg: str | None, stem: str, actual_n: int,
                        suffix: str = "") -> str:
    """
    Build the output HDF5 filepath, embedding the actual frame count.

    Examples
    --------
    out_arg=None,          stem="myfile",  actual_n=50  -> "myfile_50f_multiframe.h5"
    out_arg=None,          stem="myfile",  actual_n=50,
                           suffix="_recovery"           -> "myfile_50f_recovery_multiframe.h5"
    out_arg="my_output",   actual_n=50                  -> "my_output_50f.h5"
    out_arg="my_output.h5",actual_n=50                  -> "my_output_50f.h5"
    """
    if out_arg is not None:
        base = out_arg
        if base.endswith(".h5"):
            base = base[:-3]          # strip extension
        return f"{base}_{actual_n}f.h5"
    else:
        return f"{stem}_{actual_n}f{suffix}_multiframe.h5"


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

def process_frame(
    exo_or_path,        # open ExodusBasics OR a file path string/Path
    step: int,
    args,
    log: logging.Logger,
) -> tuple:
    """
    Process a single exodus timestep into grids and GB data.

    In standard mode, exo_or_path is an already-open ExodusBasics instance.
    In recovery mode, exo_or_path is a file path (str or Path); this function
    opens its own ExodusBasics context so frames from different files can be
    processed without keeping all files open simultaneously.

    Routes to the standard or fast path based on args.fast.

    Returns
    -------
    tuple : (step, time_val, P0, C, P, gb_dict, boundary_pixels, junction_pixels)
    """
    if isinstance(exo_or_path, (str, Path)):
        # Recovery mode: open the file ourselves
        with ExodusBasics(str(exo_or_path)) as exo:
            return _process_frame_inner(exo, step, args, log)
    else:
        # Standard mode: use the already-open instance
        return _process_frame_inner(exo_or_path, step, args, log)


def _process_frame_inner(exo, step: int, args, log: logging.Logger) -> tuple:
    """Inner frame processing logic shared by both call paths."""
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
            C, TJ_distance_max=args.tj_distance, signed=args.signed,
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
            C, TJ_distance_max=args.tj_distance, signed=args.signed,
        )

    log.debug(f"  [process_frame] step={step}: done.")
    return step, time_val, P0, C, P, gb_dict, boundary_pixels, junction_pixels


# ---------------------------------------------------------------------------
# HDF5 I/O
# ---------------------------------------------------------------------------

def save_frame_times_csv(
    filepath: str,
    frames_data: list,
    exo_or_index_rows,          # open ExodusBasics OR list of index row dicts
    log: logging.Logger = None,
    recovery_mode: bool = False,
) -> None:
    """
    Write a CSV with step, time, grain count, and (in recovery mode)
    source file path for each frame.

    In standard mode, exo_or_index_rows is an open ExodusBasics instance
    and grain counts are read from grain_tracker if present.

    In recovery mode, exo_or_index_rows is the list of index row dicts;
    grain counts are taken directly from the index (already read during
    index build) and a source_file column is added.
    """
    csv_path = filepath.replace(".h5", "_times.csv")

    if recovery_mode:
        # Build a lookup from (file_path, file_step) -> index row
        index_rows = exo_or_index_rows
        row_lookup = {
            (r["file_path"], r["file_step"]): r for r in index_rows
        }
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["frame", "step", "time", "grain_count", "source_file"]
            )
            for frame_num, frame in enumerate(frames_data):
                # frames_data entries are either full tuples or (step, time_val)
                if isinstance(frame, dict):
                    # came directly from index rows
                    step      = frame["file_step"]
                    time_val  = frame["time"]
                    gc        = frame["grain_count"]
                    src       = Path(frame["file_path"]).name
                else:
                    step, time_val = frame[0], frame[1]
                    # Try to look up grain count from index
                    gc  = -1
                    src = ""
                writer.writerow([frame_num, step, time_val, gc, src])
    else:
        exo = exo_or_index_rows
        glo_names = exo.glo_varnames()
        has_gt    = "grain_tracker" in glo_names
        gt        = (
            np.rint(exo.glo_var_series("grain_tracker")).astype(np.int64)
            if has_gt else None
        )
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            if has_gt:
                writer.writerow(["frame", "step", "time", "grains"])
                for frame_num, (step, time_val, *_) in enumerate(frames_data):
                    writer.writerow(
                        [frame_num, step, time_val, int(gt[step])]
                    )
            else:
                writer.writerow(["frame", "step", "time"])
                for frame_num, (step, time_val, *_) in enumerate(frames_data):
                    writer.writerow([frame_num, step, time_val])

    if log:
        log.info(f"Frame times CSV written: {csv_path}")
    else:
        print(f"Frame times CSV written: {csv_path}")


def _initialize_streamed_hdf5(filepath: str, args: argparse.Namespace,
                               actual_n: int,                          # NEW
                               log: logging.Logger = None) -> None:
    """
    Create a new HDF5 file with the provenance group and an empty
    frames group. Called once before streaming begins.
    """
    with h5py.File(filepath, "w") as hf:
        prov = hf.create_group("provenance")
        prov.create_dataset("tj_distance",   data=int(args.tj_distance))
        prov.create_dataset("loop_times",    data=int(args.loop_times))
        prov.create_dataset("signed",        data=bool(args.signed))
        prov.create_dataset("cpus",          data=int(args.cpus))
        prov.create_dataset("hdf5_frames",   data=int(args.hdf5_frames))
        prov.create_dataset("hdf5_t0",
            data=float(args.hdf5_t0) if args.hdf5_t0 is not None else 0.0,
        )
        prov.create_dataset("actual_frames", data=int(actual_n))
        prov.create_dataset("fast_mode",     data=bool(args.fast))
        prov.create_dataset("chunk_size",    data=int(args.chunk_size))
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
                          actual_n: int,
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
        hdf5_t0        (scalar float)
        actual_frames  (scalar int)
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
        prov.create_dataset("hdf5_t0",
            data=float(args.hdf5_t0) if args.hdf5_t0 is not None else 0.0,
        )
        prov.create_dataset("actual_frames", data=int(actual_n))
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
    ti   = time.perf_counter()
    args = parse_args()
    log  = setup_logging(args.verbose)

    log.info("Setup:")
    log.info(f"Arguments: {args}")

    if args.fast:
        log.warning(
            f"\033[32m--fast mode enabled\033[0m  "
            f"(chunk_size={args.chunk_size}, loop_times={args.loop_times})"
        )

    # -----------------------------------------------------------------------
    # RECOVERY MODE BRANCH
    # -----------------------------------------------------------------------
    if args.recover:
        log.warning("\033[1m\033[95m[recover] Recovery mode active.\033[0m")

        index_path = Path.cwd() / "recovery_index.csv"

        # Step 1: Load or build the merged timestep index
        if index_path.exists() and not args.rebuild_index:
            index_rows = load_recovery_index(index_path, log)
        else:
            rec_files  = find_recovery_files(args, log)
            index_rows = build_recovery_index(rec_files, log)
            save_recovery_index(index_rows, index_path, log)

        # Step 2: Select anchor step and frame list from merged index
        anchor_row = select_step_from_index(
            index_rows,
            grains=args.grains,
            time_value=args.time,
            log=log,
        )
        frame_rows, actual_n = select_multi_frame_steps_from_index(
            index_rows,
            anchor_row,
            n_frames=args.hdf5_frames,
            t_start=args.hdf5_t0,
            log=log,
        )

        # Step 3: Resolve output HDF5 path
        hdf5_path = _resolve_hdf5_path(
            args.out,
            stem=exodus_stem(Path(anchor_row["file_path"])),
            actual_n=actual_n,
            suffix="_recovery",
        )
        Path(hdf5_path).parent.mkdir(parents=True, exist_ok=True)
        log.warning(f"[recover] Output HDF5: {hdf5_path}")
        tih = time.perf_counter()

        log.warning(f"[recover] Output HDF5: {hdf5_path}")
        tih = time.perf_counter()

        # Step 4: Process frames — each row carries its own file_path + file_step
        try:
            if args.stream:
                frames_data_for_csv = []
                _initialize_streamed_hdf5(hdf5_path, args, actual_n=actual_n, log=log)
                anchor_frame_tuple = None

                for frame_num, row in progress(
                    enumerate(frame_rows),
                    desc="[recover] Streaming frames",
                    verbose=args.verbose,
                    total=len(frame_rows),
                ):
                    log.info(
                        f"  [recover] HDF5 frame {frame_num}: "
                        f"file={Path(row['file_path']).name}, "
                        f"file_step={row['file_step']}, "
                        f"time={row['time']:.6g}"
                    )
                    tif = time.perf_counter()
                    frame_tuple = process_frame(
                        row["file_path"], row["file_step"], args, log
                    )
                    frames_data_for_csv.append(row)
                    _stream_frame_to_hdf5(
                        hdf5_path, frame_num, frame_tuple, log=log
                    )
                    vtf(tif, log,
                        extra=f"  Frame {frame_num} "
                              f"(file_step={row['file_step']}) process: ")

                    # Retain anchor frame for debug plotting
                    if row["file_path"] == anchor_row["file_path"] and \
                            row["file_step"] == anchor_row["file_step"]:
                        anchor_frame_tuple = frame_tuple

                save_frame_times_csv(
                    hdf5_path, frames_data_for_csv, index_rows,
                    log=log, recovery_mode=True,
                )

            else:
                frames_data     = []
                anchor_frame_tuple = None

                for frame_num, row in progress(
                    enumerate(frame_rows),
                    desc="[recover] Processing frames",
                    verbose=args.verbose,
                    total=len(frame_rows),
                ):
                    log.info(
                        f"  [recover] HDF5 frame {frame_num}: "
                        f"file={Path(row['file_path']).name}, "
                        f"file_step={row['file_step']}, "
                        f"time={row['time']:.6g}"
                    )
                    tif = time.perf_counter()
                    frame_tuple = process_frame(
                        row["file_path"], row["file_step"], args, log
                    )
                    vtf(tif, log,
                        extra=f"  Frame {frame_num} "
                              f"(file_step={row['file_step']}) process: ")
                    frames_data.append(frame_tuple)

                    if row["file_path"] == anchor_row["file_path"] and \
                            row["file_step"] == anchor_row["file_step"]:
                        anchor_frame_tuple = frame_tuple

                tif = time.perf_counter()
                save_hdf5_multiframe(hdf5_path, frames_data, args, actual_n=actual_n, log=log)
                save_frame_times_csv(
                    hdf5_path, frame_rows, index_rows,
                    log=log, recovery_mode=True,
                )
                vtf(tif, log,
                    extra=f"  Batch write ({len(frames_data)} frames): ")

            log.info(
                f"[recover] HDF5 written: {hdf5_path} "
                f"({len(frame_rows)} frames)"
            )
            vtf(tih, log, "[recover] End of HDF5 generation: ")

            # Step 5: Debug plot on anchor frame
            if (args.debug_plot or args.verbose >= 2) \
                    and anchor_frame_tuple is not None:
                tid = time.perf_counter()
                _, _, P0_dbg, C_dbg, P_dbg, gb_dict_dbg, \
                    bp_dbg, jp_dbg = anchor_frame_tuple
                stem = exodus_stem(Path(anchor_row["file_path"]))
                plot_gb_curvature_debug(
                    C_dbg, P_dbg, gb_dict_dbg, bp_dbg, jp_dbg,
                    stem=stem + "_recovery",
                    TJ_distance_max=args.tj_distance,
                )
                vtf(tid, log, "[recover] End of debug plotting: ")

        except Exception as e:
            log.exception(f"[recover] ERROR: {e}")
            sys.exit(2)

        tf(ti, log, extra="[recover] Total ")
        return  # <-- exit main after recovery path

    # -----------------------------------------------------------------------
    # STANDARD MODE BRANCH (unchanged from original)
    # -----------------------------------------------------------------------
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
        til  = time.perf_counter()
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

                frame_steps, actual_n = select_multi_frame_steps(
                    exo,
                    target_step=step,
                    n_frames=args.hdf5_frames,
                    t_start=args.hdf5_t0,
                    log=log,
                )

                # Resolve output path
                hdf5_path = _resolve_hdf5_path(args.out, stem=stem, actual_n=actual_n)
                if args.out is not None:
                    Path(hdf5_path).parent.mkdir(parents=True, exist_ok=True)

                frames_data = []
                log.info("Processing HDF5 Frames:")

                if args.stream:
                    frames_data_for_csv = []
                    _initialize_streamed_hdf5(hdf5_path, args, actual_n=actual_n, log=log)

                    for frame_num, (s, t) in progress(
                        enumerate(frame_steps),
                        desc="Streaming frames",
                        verbose=args.verbose,
                        total=len(frame_steps),
                    ):
                        log.info(f"  HDF5 frame: step={s}, time={t:.6g}")
                        tif = time.perf_counter()

                        frame_tuple = process_frame(exo, s, args, log)
                        frames_data_for_csv.append(
                            (frame_tuple[0], frame_tuple[1])
                        )
                        _stream_frame_to_hdf5(
                            hdf5_path, frame_num, frame_tuple, log=log
                        )
                        vtf(tif, log,
                            extra=f"  Frame {frame_num} (step={s}) process: ")

                        if s == step:
                            frames_data = [frame_tuple]

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
                    save_hdf5_multiframe(hdf5_path, frames_data, args, actual_n=actual_n, log=log)
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
