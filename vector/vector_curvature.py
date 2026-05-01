from ExodusBasics import ExodusBasics
from MultiExodusReaderDerivs import MultiExodusReaderDerivs
import PACKAGE_MP_Linear as smooth
import myInput

import glob
import time
import numpy as np
import argparse
import sys
import logging
import matplotlib.pyplot as plt
import math
from pathlib import Path
import h5py



# Assumes uniform constant unchanging mesh, dx = dy and treats all index values as the coords

def parse_args():
    VALID_CPUS = (1, 2, 4, 8, 16, 32, 64, 128)
    p = argparse.ArgumentParser(
        description="Convert phase field GG results to curvature measurements,"
                    " using Lin's VECTOR smooting algorithms. Also optionally outputs raw data"
                    " for GB velocity calculations at the specified time.  "
                    "This code assumes uniform mesh elements.",
                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # ---- General ----
    log = p.add_argument_group("General")
    log.add_argument("-v", "--verbose", action="count", default=0,
                   help="Increase verbosity (-v, -vv, -vvv).")
    log.add_argument('-n','--cpus',type=int,default=4, choices=VALID_CPUS,
                     help='Number of CPUs for smoothing/curvature calculations.'
                     'Allowed: powers of 2 (1, 2, 4, 8, 16, 32, 64, 128...).')
    log.add_argument("-s", "--subdirs",action="store_true",
            help="Search for *.e files one level down (./*/.e). If not set, only search current directory.")

    # ---- Time selection ----
    tim = p.add_argument_group("Time (choose one)")
    grp = tim.add_mutually_exclusive_group(required=True)
    grp.add_argument("-g", "--grains", type=int,
                     help="Target grain count; chooses timestep with grain_tracker closest to this value.")
    grp.add_argument("-t", "--time", type=float,
                     help="Target time; chooses timestep with time_whole closest to this value.")

    # ---- Curvature options ----
    curv = p.add_argument_group("Curvature")
    curv.add_argument('--loop-times','-l', type=int, default=5,
                    help='Smoothing window size. Larger = smoother.')
    curv.add_argument("--tj-distance", type=int, default=6, metavar="N",
                    help="Euclidean pixel distance threshold for excluding "
                        "TJ-proximal boundary pixels.")
    curv.add_argument("--signed", action="store_true", default=False,
                    help="If set, preserve curvature sign relative to grain "
                        "orientation. Default=False (absolute curvature).")


    # ---- Velocity selection ----
    # ---- Multi-frame HDF5 ----
    mf = p.add_argument_group("Multi-frame HDF5")
    mf.add_argument("--hdf5", action="store_true",
                    help="Enable multi-frame HDF5 output.")
    mf.add_argument("--hdf5-frames", type=int, default=3, metavar="N",
                    help="Number of frames to save in the HDF5 file.")
    mf.add_argument("--hdf5-dt", type=float, default=None, metavar="DT",
                    help="Target time spacing between saved frames. "
                        "If not set, defaults to 1%% of max simulation time.")

    # ---- CSV options ----
    cs = p.add_argument_group("CSV")
    cs.add_argument("--skip-csv", action="store_true",
               help="Skip computing/writing the single frame curvature CSV output.")
    cs.add_argument("--csv-up", action="store_true",
               help="Write csv output up one directory from current (../file.csv).")
    # cs.add_argument("--raw", "-r", action="store_true",
    #             help="Write raw inclination angles (degrees) as a single-column CSV.")

    # ---- Plot options ----
    plot = p.add_argument_group("Plotting")
    # plot.add_argument('--plot','-p',action='store_true',
    #                         help='Save a polar plot of inclination, default=False')
    plot.add_argument('--debug-plot','-d',action='store_true',
                            help='Save the debugging plots. -vv also enables this.')


    return p.parse_args()


def setup_logging(verbosity: int) -> logging.Logger:
    level = logging.WARNING
    if verbosity >= 2:
        level = logging.DEBUG
    elif verbosity >= 1:
        level = logging.INFO
    # log.warning("This prints by default")
    # log.info("This prints with -v")
    # log.debug("This prints with -vv")
    logging.basicConfig(level=level, format="%(message)s")
    return logging.getLogger("TIGER")


def tf(ti,log,extra=None):
    if extra is not None:
        log.warning(extra + f"Time: {(time.perf_counter()-ti):.4}s")
    else:
        log.warning(f"Time: {(time.perf_counter()-ti):.4}s")

def vtf(ti,log,extra=None):
    if extra is not None:
        log.info(extra + f"Time: {(time.perf_counter()-ti):.4}s")
    else:
        log.info(f"Time: {(time.perf_counter()-ti):.4}s")


def find_exodus_files(*, subdirs: bool = False, pattern: str = "*.e") -> list[Path]:
    """
    Find Exodus files in current directory or one-level-down subdirectories.
    Returns sorted list of Paths.
    """
    cwd = Path.cwd()

    if subdirs:
        files = sorted(cwd.glob(f"*/{pattern}"))
    else:
        files = sorted(cwd.glob(pattern))

    # Optional: ignore hidden dirs/files or non-files
    files = [p for p in files if p.is_file()]
    return files


def exodus_stem(exo_path: Path) -> str:
    """
    Convert '/a/b/file_out.e' -> 'file'
            './file.e'       -> 'file'
    """
    name = exo_path.name  # just filename
    if name.endswith(".e"):
        name = name[:-2]   # drop '.e'
    if name.endswith("_out"):
        name = name[:-4]   # drop '_out'
    return name


def closest_index(values: np.ndarray, target: float) -> int:
    """Return index of entry closest to target. Ties -> first occurrence."""
    values = np.asarray(values)
    return int(np.argmin(np.abs(values - target)))


def select_step(exo, *, grains: int | None, time_value: float | None, log: logging.Logger) -> int:
    """
    exo: an open ExodusBasics instance
    Returns: timestep index (0-based)
    """
    times = exo.time()

    if time_value is not None:
        step = closest_index(times, float(time_value))
        log.info(f"Frame selected by time: requested={time_value}, chosen step={step}, time={times[step]}")
        return step

    # grains path: require grain_tracker
    glo_names = exo.glo_varnames()
    if "grain_tracker" not in glo_names:
        raise RuntimeError(
            "You requested --grains, but this Exodus file has no global variable 'grain_tracker'. "
            f"Available global vars: {glo_names}"
        )

    gt = exo.glo_var_series("grain_tracker")
    # Often stored as float; treat as counts for matching
    gt_counts = np.rint(gt).astype(np.int64)

    step = closest_index(gt_counts, int(grains))
    log.info(f"Frame selected by grains: requested={grains}, chosen step={step}, grain_tracker={gt_counts[step]}")
    return step


def select_multi_frame_steps(
    exo,
    target_step: int,
    n_frames: int,
    dt: float | None = None,
    mode: str = "center",   # "center" or "end"
    log: logging.Logger = None,
) -> list[tuple[int, float]]:
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
        If None, defaults to 1% of the target frame's time (or t_max if target time is 0).
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

    # --- Default dt ---
    if dt is None:
        ref = t_anchor if t_anchor > 0 else times[-1]
        dt = ref * 0.01
        if log:
            log.info(f"HDF5 dt defaulted to {dt:.4g} (1% of anchor time {ref:.4g})")

    # --- Build ideal target times centered on or ending at anchor ---
    if mode == "end":
        # frames at: anchor - (n-1)*dt, ..., anchor - dt, anchor
        offsets = np.arange(n_frames - 1, -1, -1)   # [n-1, n-2, ..., 0]
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

    # --- Safety 2: if dt was too large and collapsed frames, fill gaps ---
    # Spread remaining needed frames as evenly as possible within available range
    if len(unique_indices) < n_frames:
        # Determine the available index range to draw from
        i_min = max(0, min(unique_indices))
        i_max = min(n_total - 1, max(unique_indices))

        # Expand range if we still can't fit n_frames
        if (i_max - i_min + 1) < n_frames:
            i_min = max(0, i_max - (n_frames - 1))
            i_max = min(n_total - 1, i_min + (n_frames - 1))

        extra_indices = list(np.unique(
            np.round(np.linspace(i_min, i_max, n_frames)).astype(int)
        ))
        unique_indices = sorted(set(unique_indices) | set(extra_indices))[:n_frames]
        if log:
            log.warning(
                f"\033[31mHDF5:\033[0m dt={dt:.4g} too small/large — only {len(seen)} unique frames found. "
                f"Filled to {len(unique_indices)} by spreading within [{times[i_min]:.4g}, {times[i_max]:.4g}]."
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


def process_frame(exo, step: int, args,
                  log: logging.Logger) -> tuple:
    """
    Process a single exodus timestep into grids and GB data.

    Returns
    -------
    tuple : (step, time_val, P0, C, gb_dict, boundary_pixels, junction_pixels)
    """
    time_val = float(exo.time()[step])
    log.info(f"  Processing step={step}, time={time_val:.6g}")

    xc, yc = exo.element_centers_xy(method="mean")
    ug = exo.elem_var_at_step("unique_grains", step=step)
    ug = np.rint(ug).astype(np.int32)

    P0, _, _ = map_to_grid(xc, yc, ug, tol=1e-12, fill_value=np.nan)
    C = get_curvature(P0, args=args)

    gb_dict, boundary_pixels, junction_pixels = compute_gb_curvature(
        C, TJ_distance_max=args.tj_distance, signed=args.signed)

    return step, time_val, P0, C, gb_dict, boundary_pixels, junction_pixels


def map_to_grid(xc, yc, val, *, tol=1e-6, fill_value=np.nan, reduce=None):
    """
    Map per-element values onto a structured 2D grid using xc,yc element "locations".
    - tol: quantization tolerance for floating noise (in same units as x/y)
    - reduce: None | 'max' | 'min' | 'sum' (handles collisions)
    """
    # Quantize coordinates to stable integer keys (avoids fragile rounding)
    kx = np.rint(xc / tol).astype(np.int64)
    ky = np.rint(yc / tol).astype(np.int64)

    # Unique axes + inverse indices (direct i/j mapping)
    ux, j = np.unique(kx, return_inverse=True)   # columns
    uy, i = np.unique(ky, return_inverse=True)   # rows

    # Convert axis keys back to coordinate values (for labels)
    x_centers = ux.astype(np.float64) * tol
    y_centers = uy.astype(np.float64) * tol

    P0 = np.full((len(y_centers), len(x_centers)), fill_value, dtype=np.float64 if np.isnan(fill_value) else val.dtype)

    if reduce is None:
        # If you *know* it's one element per cell, this is fastest
        P0[i, j] = val
    else:
        # Handle collisions safely
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


def get_curvature(P0, args):
    nx = P0.shape[0]
    ny = P0.shape[1]
    # ng = np.max(P0)
    ng = int(np.nanmax(P0)) - int(np.nanmin(P0)) + 1
    # ng = int(np.nanmax(P0))
    cores = args.cpus
    loop_times = args.loop_times
    R = np.zeros((nx,ny,3)) #2
    verb = False
    if args.verbose >=2:
        verb = True
    smooth_class = smooth.linear_class(nx,ny,ng,cores,loop_times,P0,R,verification_system=verb)
    smooth_class.linear_main("curvature")
    C = smooth_class.get_C()
    return C




# Process to go from C to curvature average per grain boundary

def get_pair_id(grain_a: int, grain_b: int) -> tuple:
    """
    Returns a consistent, hashable key for a grain boundary pair.
    Sorting ensures the same pair always maps to the same key regardless
    of which grain is 'central' at a given pixel.

    Parameters
    ----------
    grain_a : int
        ID of the first grain.
    grain_b : int
        ID of the second grain.

    Returns
    -------
    tuple
        Sorted tuple (min_id, max_id).
    """
    return (min(grain_a, grain_b), max(grain_a, grain_b))


def get_boundary_pixels(C0: np.ndarray) -> list:
    """
    Identifies all pixels lying on a grain boundary using periodic BCs.
    A pixel is a boundary pixel if any of its 4 cardinal neighbors
    has a different grain ID.

    Parameters
    ----------
    C0 : np.ndarray, shape (nx, ny)
        Grain ID map.

    Returns
    -------
    list of tuple
        List of (i, j) coordinates of boundary pixels.
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
    C0 : np.ndarray, shape (nx, ny)
        Grain ID map.
    boundary_pixels : list of tuple
        Output of get_boundary_pixels.

    Returns
    -------
    dict
        pair_id (tuple) -> list of (i, j) junction pixel coordinates
        associated with that grain pair.
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
            int(C0[i, jm])
        ])
        neighbors.discard(central)

        # Junction: more than one foreign neighbor grain
        if len(neighbors) > 1:
            for neighbor_id in neighbors:
                pair_id = get_pair_id(central, neighbor_id)
                if pair_id in junction_dict:
                    junction_dict[pair_id].append((i, j))
                else:
                    junction_dict[pair_id] = [(i, j)]

    return junction_dict


def accumulate_gb_properties(C: np.ndarray, TJ_distance_max: int = 6,
                              signed: bool = False) -> dict:
    """
    Single-pass accumulation of grain boundary properties.
    Simultaneously identifies boundary pixels, detects junction pixels,
    and accumulates curvature per grain boundary pair.

    Parameters
    ----------
    C : np.ndarray, shape (2, nx, ny)
        C[0] is grain ID map, C[1] is curvature map.
    TJ_distance_max : int
        Euclidean distance threshold for excluding TJ-proximal pixels.
    signed : bool
        If True, flips curvature sign based on grain orientation.
        If False (default), uses absolute curvature value.

    Returns
    -------
    dict
        pair_id (tuple) -> np.array([valid_count, sum_curvature, total_area,
                                     grain_id1, grain_id2])
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
                int(C0[i, jm])
            ])
            neighbors.discard(central)

            # Interior pixel — skip entirely
            if len(neighbors) == 0:
                continue

            boundary_pixels.append((i, j))

            # Junction pixel — record under each relevant pair, skip accumulation
            if len(neighbors) > 1:
                junction_pixels.append((i, j))
                for neighbor_id in neighbors:
                    pair_id = get_pair_id(central, neighbor_id)
                    if pair_id not in junction_dict:
                        junction_dict[pair_id] = []
                    junction_dict[pair_id].append((i, j))
                continue

            # Clean GB pixel — initialize entry if needed, increment area
            neighbor_id = next(iter(neighbors))
            pair_id = get_pair_id(central, neighbor_id)
            grain_id1, grain_id2 = pair_id

            if pair_id not in raw_gb_dict:
                # [valid_count, sum_curvature, total_area, grain_id1, grain_id2]
                raw_gb_dict[pair_id] = np.array([0.0, 0.0, 0.0,
                                                  float(grain_id1), float(grain_id2)])
            raw_gb_dict[pair_id][2] += 1.0  # total_area

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
            int(C0[i, jm])
        ])
        neighbors.discard(central)

        # Skip junction pixels
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

        # Curvature sign handling
        curvature_val = C1[i, j]
        if signed:
            if central != grain_id1:
                curvature_val = -curvature_val
        else:
            curvature_val = abs(curvature_val)

        raw_gb_dict[pair_id][0] += 1.0           # valid_count
        raw_gb_dict[pair_id][1] += curvature_val  # sum_curvature

        boundary_pixels = np.array(boundary_pixels)
        junction_pixels = np.array(junction_pixels)

    return raw_gb_dict, boundary_pixels, junction_pixels


def average_gb_properties(raw_gb_dict: dict) -> dict:
    """
    Removes GBs with no valid pixels after TJ filtering, then computes
    average curvature for each remaining grain boundary.

    Parameters
    ----------
    raw_gb_dict : dict
        Output of accumulate_gb_properties.

    Returns
    -------
    dict
        pair_id (tuple) -> np.array([avg_curvature, area, grain_id1, grain_id2])
    """
    gb_dict = {}

    for pair_id, data in raw_gb_dict.items():
        valid_count = data[0]

        if valid_count == 0:
            continue

        avg_curvature = data[1] / valid_count
        area          = data[2]
        grain_id1     = data[3]
        grain_id2     = data[4]

        gb_dict[pair_id] = np.array([avg_curvature, area, grain_id1, grain_id2])

    return gb_dict


def compute_gb_curvature(C: np.ndarray, TJ_distance_max: int = 6,
                         signed: bool = False) -> dict:
    """
    Top-level orchestrator for GB curvature calculation.

    Parameters
    ----------
    C : np.ndarray, shape (2, nx, ny)
        C[0] is grain ID map, C[1] is signed curvature map.
    TJ_distance_max : int
        Euclidean distance threshold for TJ exclusion. Default is 6.
    signed : bool
        If True, preserves curvature sign relative to grain orientation.
        If False (default), uses absolute curvature.

    Returns
    -------
    dict
        pair_id (tuple) -> np.array([avg_curvature, area, grain_id1, grain_id2])
    """
    raw_gb_dict, boundary_pixels, junction_pixels = accumulate_gb_properties(
        C, TJ_distance_max=TJ_distance_max, signed=signed)

    gb_dict = average_gb_properties(raw_gb_dict)

    return gb_dict, boundary_pixels, junction_pixels


def save_gb_dict_to_csv(gb_dict: dict, filepath: str, log: logging.Logger) -> None:
    """
    Saves grain boundary curvature data to a CSV file.

    Parameters
    ----------
    gb_dict : dict
        Output of compute_gb_curvature.
        pair_id (tuple) -> np.array([avg_curvature, area, grain_id1, grain_id2])
    filepath : str
        Full path to the output CSV file.
    """
    import csv

    header = ['grain_id1', 'grain_id2', 'avg_curvature', 'area']

    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for pair_id, data in gb_dict.items():
            row = [
                int(data[2]),  # grain_id1
                int(data[3]),  # grain_id2
                data[0],       # avg_curvature
                int(data[1])   # area
            ]
            writer.writerow(row)

    log.info(f"Saved {len(gb_dict)} grain boundaries to {filepath}")


def load_gb_dict_from_csv(filepath: str, log: logging.Logger) -> dict:
    """
    Loads grain boundary curvature data from a CSV file back into
    the gb_dict format.

    Parameters
    ----------
    filepath : str
        Path to the CSV file saved by save_gb_dict_to_csv.

    Returns
    -------
    dict
        pair_id (tuple) -> np.array([avg_curvature, area, grain_id1, grain_id2])
    """
    import csv

    gb_dict = {}

    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            grain_id1 = int(float(row['grain_id1']))
            grain_id2 = int(float(row['grain_id2']))
            pair_id   = get_pair_id(grain_id1, grain_id2)
            gb_dict[pair_id] = np.array([
                float(row['avg_curvature']),
                float(row['area']),
                float(grain_id1),
                float(grain_id2)
            ])

    log.info(f"Loaded {len(gb_dict)} grain boundaries from {filepath}")
    return gb_dict



def save_hdf5_multiframe(filepath: str, frames_data: list,
                         log: logging.Logger = None) -> None:
    """
    Save multi-frame grain boundary curvature data to an HDF5 file.

    HDF5 structure:
        /frames/
            frame_0000/
                step                 (scalar int)
                time                 (scalar float)
                P0                   (nx x ny float array)
                C                    (2 x nx x ny float array)
                boundary_pixels      (N x 2 int array)
                junction_pixels      (M x 2 int array)
                gb_dict/
                    pair_ids         (K x 2 int array)
                    data             (K x 4 float array)
                                     [avg_curv, area, grain_id1, grain_id2]

    Parameters
    ----------
    filepath : str
        Output .h5 file path.
    frames_data : list of tuples
        Each tuple: (step, time_val, P0, C, gb_dict, boundary_pixels, junction_pixels)
    log : logging.Logger or None
    """
    with h5py.File(filepath, 'w') as hf:
        frames_grp = hf.create_group("frames")

        for frame_num, (step, time_val, P0, C, gb_dict,
                        boundary_pixels, junction_pixels) in enumerate(frames_data):
            fg = frames_grp.create_group(f"frame_{frame_num:04d}")

            fg.create_dataset("step",  data=int(step))
            fg.create_dataset("time",  data=float(time_val))
            fg.create_dataset("P0",    data=np.rint(P0).astype(np.int32), compression="gzip") #data=P0
            fg.create_dataset("C",     data=C, compression="gzip")

            # boundary/junction pixels
            if len(boundary_pixels) > 0:
                fg.create_dataset("boundary_pixels",
                                  data=np.array(boundary_pixels, dtype=np.int32),
                                  compression="gzip")
            else:
                fg.create_dataset("boundary_pixels",
                                  data=np.empty((0, 2), dtype=np.int32))

            if len(junction_pixels) > 0:
                fg.create_dataset("junction_pixels",
                                  data=np.array(junction_pixels, dtype=np.int32),
                                  compression="gzip")
            else:
                fg.create_dataset("junction_pixels",
                                  data=np.empty((0, 2), dtype=np.int32))

            # gb_dict as two parallel arrays
            gb_grp = fg.create_group("gb_dict")
            if gb_dict:
                pair_ids = np.array(list(gb_dict.keys()), dtype=np.int32)
                data_arr = np.array(list(gb_dict.values()), dtype=np.float64)
                gb_grp.create_dataset("pair_ids", data=pair_ids, compression="gzip")
                gb_grp.create_dataset("data",     data=data_arr, compression="gzip")
            else:
                gb_grp.create_dataset("pair_ids", data=np.empty((0, 2), dtype=np.int32))
                gb_grp.create_dataset("data",     data=np.empty((0, 4), dtype=np.float64))

    if log:
        log.info(f"Saved {len(frames_data)} frames to {filepath}")
    else:
        print(f"Saved {len(frames_data)} frames to {filepath}")







def plot_gb_curvature_debug(C: np.ndarray, gb_dict: dict,
                             boundary_pixels: np.ndarray,
                             junction_pixels: np.ndarray,
                             TJ_distance_max: int = 6,
                             stem: str = 'debug',
                             figsize: tuple = (18, 12)) -> None:
    """
    Multi-panel diagnostic plot for verifying GB curvature results.

    Panel 1: Raw grain ID map
    Panel 2: Boundary and junction pixel overlay
    Panel 3: Raw curvature field (C[1])
    Panel 4: Per-GB averaged curvature mapped back onto the grain structure
    Panel 5: Histogram of averaged GB curvatures
    Panel 6: GB area (pixel count) distribution

    Parameters
    ----------
    C : np.ndarray, shape (2, nx, ny)
        C[0] is grain ID map, C[1] is curvature map.
    gb_dict : dict
        Output of compute_gb_curvature.
    TJ_distance_max : int
        Must match the value used in compute_gb_curvature.
    figsize : tuple
        Figure size.
    """
    import matplotlib.pyplot as plt

    C0 = C[0]
    C1 = C[1]
    nx, ny = C0.shape

    # ------------------------------------------------------------------
    # Build averaged curvature overlay image
    # ------------------------------------------------------------------
    curvature_overlay = np.full((nx, ny), np.nan)
    for (i, j) in boundary_pixels:
        central = int(C0[i, j])
        ip, im = (i + 1) % nx, (i - 1) % nx
        jp, jm = (j + 1) % ny, (j - 1) % ny
        neighbors = set([
            int(C0[ip, j]), int(C0[im, j]),
            int(C0[i, jp]), int(C0[i, jm])
        ])
        neighbors.discard(central)
        if len(neighbors) == 1:
            pair_id = get_pair_id(central, next(iter(neighbors)))
            if pair_id in gb_dict:
                curvature_overlay[i, j] = gb_dict[pair_id][0]

    # Guard against all-NaN overlay (e.g. all GBs filtered by TJ proximity)
    valid_overlay = curvature_overlay[~np.isnan(curvature_overlay)]
    curv_max = np.nanpercentile(np.abs(valid_overlay), 95) if len(valid_overlay) > 0 else 1.0

    # ------------------------------------------------------------------
    # Extract GB-level data for histograms
    # ------------------------------------------------------------------
    avg_curvatures = np.array([v[0] for v in gb_dict.values()])
    gb_areas       = np.array([v[1] for v in gb_dict.values()])

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.flatten()

    # --- Panel 1: Grain ID map ---
    ax = axes[0]
    im0 = ax.imshow(C0.T, origin='lower', cmap='viridis', interpolation='nearest')
    ax.set_title('Panel 1: Grain ID Map')
    ax.set_xlabel('i')
    ax.set_ylabel('j')
    plt.colorbar(im0, ax=ax, label='Grain ID')

    # --- Panel 2: Boundary and junction pixel overlay ---
    ax = axes[1]
    ax.imshow(C0.T, origin='lower', cmap='gray', alpha=0.3, interpolation='nearest')
    if len(boundary_pixels) > 0:
        ax.scatter(boundary_pixels[:, 0], boundary_pixels[:, 1],
                   c='steelblue', s=1, label='Boundary pixels')
    if len(junction_pixels) > 0:
        ax.scatter(junction_pixels[:, 0], junction_pixels[:, 1],
                   c='red', s=4, label='Junction pixels', zorder=3)
        for (ji, jj) in junction_pixels[::max(1, len(junction_pixels) // 20)]:
            circle = plt.Circle((ji, jj), TJ_distance_max,
                                 color='orange', fill=False, linewidth=0.8, alpha=0.6)
            ax.add_patch(circle)
    ax.set_title('Panel 2: Boundary & Junction Pixels\n(orange = TJ exclusion radius)')
    ax.set_xlabel('i')
    ax.set_ylabel('j')
    ax.legend(loc='upper right', markerscale=4, fontsize=7)

    # --- Panel 3: Raw curvature field ---
    ax = axes[2]
    c1_max = np.nanpercentile(np.abs(C1), 95)
    im2 = ax.imshow(C1.T, origin='lower', cmap='coolwarm',
                    vmin=-c1_max, vmax=c1_max, interpolation='nearest')
    ax.set_title('Panel 3: Raw Curvature Field C[1]')
    ax.set_xlabel('i')
    ax.set_ylabel('j')
    plt.colorbar(im2, ax=ax, label='Curvature')

    # --- Panel 4: Averaged GB curvature overlay ---
    ax = axes[3]
    ax.imshow(C0.T, origin='lower', cmap='gray', alpha=0.3, interpolation='nearest')
    curv_max = np.nanpercentile(np.abs(curvature_overlay[~np.isnan(curvature_overlay)]), 95)
    im3 = ax.imshow(curvature_overlay.T, origin='lower', cmap='plasma',
                    vmin=0, vmax=curv_max, interpolation='nearest')
    ax.set_title('Panel 4: Averaged GB Curvature Overlay')
    ax.set_xlabel('i')
    ax.set_ylabel('j')
    plt.colorbar(im3, ax=ax, label='Avg Curvature')

    # --- Panel 5: Histogram of averaged GB curvatures ---
    ax = axes[4]
    ax.hist(avg_curvatures, bins=50, color='steelblue', edgecolor='white', linewidth=0.4)
    ax.axvline(np.mean(avg_curvatures), color='red', linestyle='--',
               label=f'Mean: {np.mean(avg_curvatures):.4f}')
    ax.axvline(np.median(avg_curvatures), color='orange', linestyle='--',
               label=f'Median: {np.median(avg_curvatures):.4f}')
    ax.set_title('Panel 5: Distribution of Avg GB Curvatures')
    ax.set_xlabel('Avg Curvature')
    ax.set_ylabel('Count')
    ax.legend(fontsize=8)

    # --- Panel 6: GB area distribution ---
    ax = axes[5]
    ax.hist(gb_areas, bins=50, color='mediumseagreen', edgecolor='white', linewidth=0.4)
    ax.axvline(np.mean(gb_areas), color='red', linestyle='--',
               label=f'Mean: {np.mean(gb_areas):.1f} px')
    ax.set_title('Panel 6: Distribution of GB Areas (pixels)')
    ax.set_xlabel('Area (pixels)')
    ax.set_ylabel('Count')
    ax.legend(fontsize=8)

    plt.suptitle(f'GB Curvature Debug — {len(gb_dict)} boundaries detected',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    figname = stem + '_DEBUG_curvature_plot.png'
    fig.savefig(figname,dpi=500, transparent=True)
    plt.close(fig)





def main():
    ti = time.perf_counter()
    args = parse_args()
    log = setup_logging(args.verbose)
    log.info('Setup:')
    log.info(f'Arguments: {args}')


    # exofile = glob.glob('*.e')[0]
    exo_files = find_exodus_files(subdirs=args.subdirs)
    if not exo_files:
        where = "subdirectories" if args.subdirs else "current directory"
        raise SystemExit(f"No .e files found in {where}.")

    log.info(' ')
    log.info('Exodus Files:')
    for ef in exo_files:
        log.info(f'  File: {ef}')
        log.info(f'  Namebase: {exodus_stem(ef)}')
        log.info(' ')

    for cnt,exofile in enumerate(exo_files):
        til = time.perf_counter()
        stem = exodus_stem(exofile)
        if len(exo_files) > 1:
            log.warning(' ')
        log.warning('\033[1m\033[96m'+'File '+str(cnt+1)+'/'+str(len(exo_files))+': '+'\x1b[0m'+str(stem))

        try:
            with ExodusBasics(exofile) as exo:
                # Pick the right timestep
                step = select_step(exo, grains=args.grains, time_value=args.time, log=log)
                # # Pull a couple reference values:
                # ug_max = exo.elem_var_at_step("unique_grains", step=0).max()
                # t = exo.time()[step]
                # Get data for inclination
                xc, yc = exo.element_centers_xy(method="mean")
                ug = exo.elem_var_at_step("unique_grains", step=step)
                ug = np.rint(ug).astype(np.int32)
                vtf(ti,log,"End of Exodus ripping: ")
                log.info(' ')

                # VECTOR PORTION
                tiv = time.perf_counter()
                # Make Array for Lins approach
                # MIGHT need to adjust or rotate it?
                P0, xcen, ycen = map_to_grid(xc, yc, ug, tol=1e-12, fill_value=np.nan)
                # Curvature
                C = get_curvature(P0, args=args)
                # GB data from the curvature
                gb_dict, boundary_pixels, junction_pixels = compute_gb_curvature(
                    C, TJ_distance_max=args.tj_distance, signed=args.signed)


                # Write histogram inclination to csv
                if not args.skip_csv:
                    prefix = '../' if args.csv_up else ''
                    csv_name = prefix + stem + f'_curvature_step{step}.csv'
                    save_gb_dict_to_csv(gb_dict, csv_name, log=log)


                # Debug Plots
                if args.verbose >= 2 or args.debug_plot:
                    plot_gb_curvature_debug(C, gb_dict, boundary_pixels, junction_pixels,
                            stem=stem, TJ_distance_max=args.tj_distance)

                vtf(tiv,log,"End of VECTOR calculations: ")
                log.info(' ')

                # ---- Multi-frame HDF5 output ----
                if args.hdf5:
                    log.info("Starting multi-frame HDF5 export...")
                    tih = time.perf_counter()
                    frame_steps = select_multi_frame_steps(
                        exo,
                        target_step=step,
                        n_frames=args.hdf5_frames,
                        dt=args.hdf5_dt,
                        mode="end", # or "center", your preference
                        log=log,
                    )
                    frames_data = []
                    for s, t in frame_steps:
                        log.info(f"  HDF5 frame: step={s}, time={t:.6g}")
                        frame_tuple = process_frame(exo, s, args, log)
                        frames_data.append(frame_tuple)

                    hdf5_path = stem + '_multiframe.h5'
                    save_hdf5_multiframe(hdf5_path, frames_data, log=log)
                    log.info(f"HDF5 written: {hdf5_path} ({len(frames_data)} frames)")
                    vtf(tih,log,"End of HDF5 ripping for velocity calculations: ")
                    log.info(' ')


                if len(exo_files) > 1:
                    tf(til,log,extra=f"File {cnt+1} ")




        except Exception as e:
            # argparse-style error output (clean for CLI)
            log.exception(f"ERROR: {e}")#, file=sys.stderr)
            sys.exit(2)

    # TOTAL end time
    if len(exo_files) > 1:
        log.warning(' ')
    tf(ti,log,extra="Total ")

if __name__ == "__main__":
    main()

