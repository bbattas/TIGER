#!/usr/bin/env python3
"""
vector_hdf5_processing.py

Post-processing script for multi-frame HDF5 files produced by
vector_exodus_to_hdf5.py.  Currently implements:
  - GBE calculation at every boundary pixel (four modes)
  - Optional triple-junction exclusion (default: enabled)
"""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass, replace
from pathlib import Path

import h5py
import myInput
import numpy as np
import pandas as pd
import math

# ──────────────────────────────────────────────────────────────────────────────
#  Constants
# ──────────────────────────────────────────────────────────────────────────────

THETA_MAX_DEG = 62.0
THETA_MAX_RAD = np.deg2rad(THETA_MAX_DEG)

GBE_MODES = ("iso","inclination_cos", "inclination", "misorientation", "both")

# ──────────────────────────────────────────────────────────────────────────────
#  CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Post-process multi-frame HDF5 files from vector_exodus_to_hdf5.py. "
            "Computes grain-boundary energy (GBE) at every boundary pixel."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---- I/O ----
    io = p.add_argument_group("I/O")
    io.add_argument(
        "--hdf5", "-i", type=Path, required=True, metavar="FILE",
        help="Input .h5 file produced by vector_exodus_to_hdf5.py.",
    )
    io.add_argument(
        "--parquet", "-p", type=Path, default=None, metavar="FILE",
        help=(
            "Misorientation parquet file. Required when --gbe-mode is "
            "'misorientation' or 'both'."
        ),
    )
    io.add_argument(
        "--output-dir", "-o", type=Path, default=Path("."), metavar="DIR",
        help="Directory for all output files.",
    )
    io.add_argument(
        "--inclination-csv", action="store_true",
        help="Write a CSV of the inclination angle distribution (binned, normalized).",
    )
    io.add_argument(
        "--dpi", type=int, default=300,
        help="DPI for saving the actual plots from --plot",
    )
    io.add_argument(
        "--final", action="store_true",
        help="Write final plotting data to CSV files.",
    )

    # ---- GBE ----
    gbe = p.add_argument_group("GBE")
    gbe.add_argument(
        "--gbe-mode", type=str, default="misorientation", choices=GBE_MODES,
        help="Energy function to use for GBE calculation.",
    )
    gbe.add_argument(
        "--tj-exclude", action=argparse.BooleanOptionalAction, default=True,
        help="Exclude boundary pixels within --tj-distance of a triple junction.",
    )
    gbe.add_argument(
        "--tj-distance", type=int, default=3, metavar="N",
        help=(
            "Euclidean pixel distance threshold for TJ exclusion. "
            "Matches the default used in vector_exodus_to_hdf5.py. Lin used 6"
        ),
    )
    gbe.add_argument(
        "--inclination-anisotropy", "-a", type=float, default=0.05,
        metavar="A",
        help=(
            "Anisotropy amplitude 'a' for the inclination_cos GBE mode: "
            "GBE = 1 + a * cos(2 * theta). Must be in [0, 0.95]."
        ),
    )
    gbe.add_argument(
        "--min-area", type=int, default=20, metavar="N",
        help="Minimum TJ-filtered GB pixel area for keeping an entire ij GB pair. Lin used 100",
    )
    gbe.add_argument(
        "--min-curvature", type=float, default=0.0, metavar="F",
        help="Minimum absolute TJ-filtered average GB curvature for keeping an entire ij GB pair. Lin used 0.0182",
    )
    gbe.add_argument(
        "--antic-confidence", type=float, default=0.99, metavar="F",
        help="Percent confidence for anticurvature calculation. Lin used 0.99",
    )

    # ---- Plotting ----
    tog = p.add_argument_group("Toggles")
    tog.add_argument('--debug-plot','-d',action='store_true',
        help='Save the debugging plots.')
    tog.add_argument('--plot',action='store_true',
        help='Save individual plots for critical values.')
    tog.add_argument(
        "--skip-velocity", action="store_true",
        help="Skip GB velocity calculation.",
    )
    tog.add_argument(
        "--drop-isolated-antic", action="store_true",
        help="Use the sliding window 00100 approach to drop isolated anticurvature events.",
    )


    # ---- Verbosity ----
    p.add_argument(
        "-v", "--verbose", action="count", default=0,
        help="Increase verbosity (-v = INFO, -vv = DEBUG).",
    )

    args = p.parse_args()

    # ---- Cross-argument validation ----
    if args.gbe_mode in ("misorientation", "both") and args.parquet is None:
        p.error(
            f"--gbe-mode '{args.gbe_mode}' requires --parquet to be specified."
        )
    if args.parquet is not None and not args.parquet.exists():
        p.error(f"Parquet file not found: {args.parquet}")
    if not args.hdf5.exists():
        p.error(f"HDF5 file not found: {args.hdf5}")
    if not (0.0 <= args.inclination_anisotropy <= 0.95):
        p.error(
            f"--inclination-anisotropy must be in [0, 0.95], "
            f"got {args.inclination_anisotropy}."
        )

    return args


# ──────────────────────────────────────────────────────────────────────────────
#  Logging
# ──────────────────────────────────────────────────────────────────────────────

def setup_logging(verbosity: int) -> logging.Logger:
    level = logging.WARNING
    if verbosity >= 2:
        level = logging.DEBUG
    elif verbosity >= 1:
        level = logging.INFO
    logging.basicConfig(level=level, format="%(message)s")
    return logging.getLogger("VHP")  # Vector HDF5 Processing


def tf(ti: float, log: logging.Logger, extra: str = "") -> None:
    log.warning(f"{extra}Time: {time.perf_counter() - ti:.4f}s")

def vtf(ti: float, log: logging.Logger, extra: str = "") -> None:
    log.info(f"{extra}Time: {time.perf_counter() - ti:.4f}s")


# ──────────────────────────────────────────────────────────────────────────────
#  Data structures
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class FrameData:
    step:            int
    time:            float
    P0:              np.ndarray   # (nx, ny)         grain ID map
    C:               np.ndarray   # (2, nx, ny)      C[0]=grain ID, C[1]=curvature
    P:               np.ndarray   # (3, nx, ny)      P[0]=grain ID, P[1]=dx, P[2]=dy
    gb_dict:         dict         # pair_id -> np.array([avg_curv, gb_area, gid1, gid2, raw_gb_area])
    boundary_pixels: np.ndarray   # (N, 2)
    junction_pixels: np.ndarray   # (M, 2)


# ──────────────────────────────────────────────────────────────────────────────
#  Layer 1 — I/O
# ──────────────────────────────────────────────────────────────────────────────

def load_hdf5_frames(filepath: Path, log: logging.Logger) -> list[FrameData]:
    """
    Read all frames from an HDF5 file produced by vector_exodus_to_hdf5.py.
    Reconstructs gb_dict from the parallel pair_ids / data arrays stored
    under each frame's gb_dict group.
    """
    frames: list[FrameData] = []

    with h5py.File(filepath, "r") as hf:
        frames_grp = hf["frames"]
        frame_keys = sorted(frames_grp.keys())  # frame_0000, frame_0001, ...
        log.info(f"HDF5 contains {len(frame_keys)} frame(s): {frame_keys}")

        for key in frame_keys:
            fg = frames_grp[key]

            step     = int(fg["step"][()])
            time_val = float(fg["time"][()])
            P0       = fg["P0"][:]
            C        = fg["C"][:]
            P        = fg["P"][:]
            bp       = fg["boundary_pixels"][:]
            jp       = fg["junction_pixels"][:]

            # Reconstruct gb_dict from parallel arrays
            gb_grp   = fg["gb_dict"]
            pair_ids = gb_grp["pair_ids"][:]   # (K, 2)
            data_arr = gb_grp["data"][:]        # (K, 4 legacy) or (K, 5 current)

            # Current schema: [avg_curv, gb_area, gid1, gid2, raw_gb_area].
            # Legacy schema:  [avg_curv, area,    gid1, gid2].  Preserve compatibility
            # by treating legacy area as both gb_area and raw_gb_area.
            if data_arr.size > 0 and data_arr.shape[1] == 4:
                data_arr = np.column_stack([
                    data_arr[:, 0],  # avg_curv
                    data_arr[:, 1],  # gb_area fallback from legacy area
                    data_arr[:, 2],  # gid1
                    data_arr[:, 3],  # gid2
                    data_arr[:, 1],  # raw_gb_area fallback
                ])
            elif data_arr.size == 0:
                data_arr = np.empty((0, 5), dtype=np.float64)
            elif data_arr.shape[1] != 5:
                raise ValueError(
                    f"Unexpected gb_dict/data shape {data_arr.shape}; expected Kx4 or Kx5."
                )

            gb_dict: dict[tuple[int, int], np.ndarray] = {
                (int(pair_ids[k, 0]), int(pair_ids[k, 1])): data_arr[k]
                for k in range(len(pair_ids))
            }

            frames.append(FrameData(
                step=step, time=time_val,
                P0=P0, C=C, P=P,
                gb_dict=gb_dict,
                boundary_pixels=bp,
                junction_pixels=jp,
            ))
            log.info(
                f"  Loaded frame '{key}': step={step}, time={time_val:.6g}, "
                f"gb_pairs={len(gb_dict)}, "
                f"boundary_px={len(bp)}, junction_px={len(jp)}"
            )

    return frames


def get_last_frame(frames: list[FrameData]) -> FrameData:
    """Return the last frame — used for all single-frame analyses."""
    return frames[-1]


def filter_gb_dict(
    gb_dict: dict[tuple[int, int], np.ndarray],
    *,
    min_area: float = 0.0,
    min_curvature: float = 0.0,
    log: logging.Logger | None = None,
    label: str = "",
) -> dict[tuple[int, int], np.ndarray]:
    """Filter entire GB pairs using TJ-filtered GB properties.

    Expected gb_dict[pair_id] schema:
        [avg_curvature, gb_area, grain_id1, grain_id2, raw_gb_area]

    The filters are GB-level filters applied after TJ exclusion:
        - gb_area >= min_area
        - abs(avg_curvature) >= min_curvature
    """
    out: dict[tuple[int, int], np.ndarray] = {}
    n_area = 0
    n_curv = 0

    for pair_id, data in gb_dict.items():
        avg_curv = float(data[0])
        gb_area = float(data[1])

        if gb_area < min_area:
            n_area += 1
            continue
        if abs(avg_curv) < min_curvature:
            n_curv += 1
            continue
        out[pair_id] = data

    if log is not None:
        tag = f" {label}" if label else ""
        log.warning(
            f"GB filter{tag}: {len(out)}/{len(gb_dict)} kept, "
            f"{n_area} removed by gb_area<{min_area}, "
            f"{n_curv} removed by |avg_curvature|<{min_curvature}."
        )
    return out


def filter_frames_gb_dicts(
    frames: list[FrameData],
    *,
    min_area: float,
    min_curvature: float,
    log: logging.Logger,
) -> list[FrameData]:
    """Return FrameData copies whose gb_dict values pass the canonical GB filter."""
    filtered: list[FrameData] = []
    for idx, frame in enumerate(frames):
        filtered.append(replace(
            frame,
            gb_dict=filter_gb_dict(
                frame.gb_dict,
                min_area=min_area,
                min_curvature=min_curvature,
                log=log,
                label=f"frame {idx}",
            ),
        ))
    return filtered


def summarize_velocity_input(flat: dict, log: logging.Logger, label: str = "Velocity plot input") -> None:
    """Print compact diagnostics for the filtered velocity lists feeding the plots."""
    n = len(flat.get("all_curvatures", []))
    if n == 0:
        log.warning(f"{label}: 0 entries.")
        return
    curv = np.asarray(flat["all_curvatures"], dtype=float)
    vel = np.asarray(flat["all_velocities"], dtype=float)
    area = np.asarray(flat["all_areas"], dtype=float)
    log.warning(
        f"{label}: N={n}, "
        f"|kappa| min/median/max={np.min(np.abs(curv)):.5g}/"
        f"{np.median(np.abs(curv)):.5g}/{np.max(np.abs(curv)):.5g}, "
        f"area min/median/max={np.min(area):.1f}/{np.median(area):.1f}/{np.max(area):.1f}, "
        f"velocity min/median/max={np.min(vel):.5g}/{np.median(vel):.5g}/{np.max(vel):.5g}."
    )


def load_misorientation_parquet(
    parquet_path: Path,
    neighbor_pairs: set[tuple[int, int]],
    log: logging.Logger,
) -> dict[tuple[int, int], dict]:
    """
    Load only the rows from the Parquet file that correspond to the
    requested neighbor_pairs, using predicate push-down on column 'i'
    to minimise I/O for large files.

    Returns
    -------
    dict  pair_id -> {"angle_deg", "ax_x", "ax_y", "ax_z"}
    """
    if not neighbor_pairs:
        return {}

    log.info(f"Loading misorientation Parquet (pair-selective): {parquet_path}")

    grain_ids = set()
    for (a, b) in neighbor_pairs:
        grain_ids.add(a)
        grain_ids.add(b)

    filters = [("i", "in", list(grain_ids))]
    df = pd.read_parquet(
        parquet_path,
        columns=["i", "j", "angle_deg", "ax_x", "ax_y", "ax_z"],
        filters=filters,
    )

    result: dict[tuple[int, int], dict] = {}
    for row in df.itertuples(index=False):
        key = (min(int(row.i), int(row.j)), max(int(row.i), int(row.j)))
        if key in neighbor_pairs:
            result[key] = {
                "angle_deg": float(row.angle_deg),
                "ax_x":      float(row.ax_x),
                "ax_y":      float(row.ax_y),
                "ax_z":      float(row.ax_z),
            }

    log.info(
        f"  Parquet selective load: requested={len(neighbor_pairs)}, "
        f"found={len(result)} pairs."
    )
    return result


# ──────────────────────────────────────────────────────────────────────────────
#  Layer 2 — GBE energy functions
# ──────────────────────────────────────────────────────────────────────────────

def compute_fmiso_single(
    angle_deg: float,
    ax_x: float,
    ax_y: float,
    ax_z: float,
) -> float:
    """
    Misorientation-only GBE function f_miso for a single GB pair.

    1. Normalize misorientation axis.
    2. polar   = acos(ax_z)
       azimuth = atan2(ax_y, ax_x), remapped to [0, 2π)
    3. ang_energy = (θ/62°) * (1 - ln(θ/62°)), clamped to [0, 1]
       ax_energy  = |cos(polar)|^0.4 + |cos(azimuth/2)|^0.4, clamped to [0, 1]
    4. f_miso = 0.3 + 0.7 * (ang_energy * ax_energy)

    Returns f_miso in [0.3, ~1.0].
    """
    norm = np.sqrt(ax_x**2 + ax_y**2 + ax_z**2)
    if norm < 1e-12:
        return 0.3
    ax_x /= norm
    ax_y /= norm
    ax_z /= norm

    polar   = np.arccos(np.clip(ax_z, -1.0, 1.0))
    azimuth = np.arctan2(ax_y, ax_x)
    if azimuth < 0.0:
        azimuth += 2.0 * np.pi

    theta_rad = np.deg2rad(angle_deg)
    ratio     = theta_rad / THETA_MAX_RAD
    if ratio <= 0.0:
        ang_energy = 0.0
    elif ratio >= 1.0:
        ang_energy = 1.0
    else:
        ang_energy = ratio * (1.0 - np.log(ratio))
    ang_energy = min(ang_energy, 1.0)

    ax_energy = abs(np.cos(polar))**0.4 + abs(np.cos(azimuth / 2.0))**0.4
    ax_energy = min(ax_energy, 1.0)

    return float(0.3 + 0.7 * (ang_energy * ax_energy))


def compute_finclination_cosine(dx: float, dy: float, a: float) -> float:
    """
    Inclination-only GBE using a two-fold cosine function:
        GBE = 1 + a * cos(2 * theta)
    where theta is the inclination angle computed from the normal vector
    via atan2(-dy, dx) + pi, matching the convention in
    get_normal_vector_slope and compute_inclination_per_pixel.

    Parameters
    ----------
    dx, dy : float  — normal vector components from myInput.get_grad
    a      : float  — anisotropy amplitude in [0, 0.95]

    Returns
    -------
    float  — GBE in [1 - a, 1 + a]
    """
    theta = math.atan2(-dy, dx) + math.pi   # [0, 2π), consistent with inclination convention [1]
    return 1.0 + a * math.cos(2.0 * theta)


def compute_finclination(dx: float, dy: float) -> float:
    """
    Inclination-only GBE using a direct inclination-dependent function.
    Placeholder — replace with your target formulation.
    """
    # TODO: implement direct inclination energy function
    raise NotImplementedError("inclination GBE not yet implemented.")


def compute_fboth(
    dx: float,
    dy: float,
    angle_deg: float,
    ax_x: float,
    ax_y: float,
    ax_z: float,
) -> float:
    """
    Combined inclination + misorientation GBE.
    Placeholder — replace with your target formulation.
    """
    # TODO: implement combined inclination + misorientation energy function
    raise NotImplementedError("'both' GBE not yet implemented.")


# ──────────────────────────────────────────────────────────────────────────────
#  Layer 2 — TJ proximity filter
# ──────────────────────────────────────────────────────────────────────────────

def build_tj_proximity_set(
    junction_pixels: np.ndarray,
    tj_distance: int,
) -> set[tuple[int, int]]:
    """
    Pre-compute the set of all pixel coordinates that lie within
    tj_distance of any triple-junction pixel.  Using a set allows O(1)
    membership tests in the hot loop below.

    This mirrors the per-pair proximity check in accumulate_gb_properties
    in vector_exodus_to_hdf5.py [1], but is applied globally (not per-pair)
    so it only needs to be built once per frame.
    """
    if len(junction_pixels) == 0:
        return set()

    # For each candidate boundary pixel we only need to know whether *any*
    # junction is close, not which one — so a global KD-style approach is
    # fastest.  For typical image sizes the nested loop is fine; swap for
    # scipy.spatial.cKDTree if performance becomes a concern.
    excluded: set[tuple[int, int]] = set()
    for (ji, jj) in junction_pixels:
        excluded.add((int(ji), int(jj)))

    # Expand: mark every pixel within tj_distance of a junction pixel.
    # We use a bounding-box pre-filter to avoid the sqrt for most candidates.
    junctions = np.array(list(excluded), dtype=np.int32)
    r = tj_distance

    for (ji, jj) in junctions:
        for di in range(-r, r + 1):
            for dj in range(-r, r + 1):
                if di * di + dj * dj < r * r:
                    excluded.add((ji + di, jj + dj))

    return excluded


def compute_grain_areas(P0: np.ndarray) -> dict[int, int]:
    """
    Returns a dict mapping grain_id -> pixel count for every grain in P0.

    Parameters
    ----------
    P0 : np.ndarray, shape (nx, ny)
        Grain ID map (will be rounded to nearest int).

    Returns
    -------
    dict  grain_id -> pixel count
    """
    unique, counts = np.unique(
        np.rint(P0).astype(np.int32), return_counts=True
    )
    return {int(gid): int(cnt) for gid, cnt in zip(unique, counts)}


def build_small_grain_set(
    grain_areas: dict[int, int],
    min_area: int,
) -> set[int]:
    """
    Returns the set of grain IDs whose total pixel count is below min_area.

    Parameters
    ----------
    grain_areas : dict  grain_id -> pixel count (from compute_grain_areas)
    min_area    : int   minimum pixel count threshold

    Returns
    -------
    set of grain IDs that fall below the threshold
    """
    return {gid for gid, area in grain_areas.items() if area < min_area}


# ──────────────────────────────────────────────────────────────────────────────
#  Layer 2 — Core GBE per-pixel computation
# ──────────────────────────────────────────────────────────────────────────────

def compute_gbe_per_pixel(
    frame: FrameData,
    mode: str,
    tj_excluded: set[tuple[int, int]],
    valid_gb_dict: dict[tuple[int, int], np.ndarray],
    misorientation_data: dict | None = None,
    inclination_anisotropy: float = 0.05,
    log: logging.Logger | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute GBE at every (non-excluded) boundary pixel in the last frame.

    Parameters
    ----------
    frame               : FrameData  (last frame)
    mode                : one of GBE_MODES
    tj_excluded         : pre-built set from build_tj_proximity_set()
    valid_gb_dict       : already-filtered GB dictionary. A pixel is kept only if
                          its pair_id is present in this dict. This applies the
                          GB-level min_area and min_curvature filters after TJ
                          exclusion.
    misorientation_data : output of load_misorientation_parquet, or None
    inclination_anisotropy : float anisotropy amplitude
    log                 : logger

    Returns
    -------
    pixel_coords : np.ndarray  shape (N, 2)   — (i, j) of each kept pixel
    gbe_values   : np.ndarray  shape (N,)     — GBE value at each pixel
    """
    if log is None:
        log = logging.getLogger("VHP")

    C0 = frame.C[0]          # grain ID map
    C1 = frame.C[1]          # curvature map
    P  = frame.P              # normal-vector field
    nx, ny = C0.shape

    pixel_coords: list[tuple[int, int]] = []
    gbe_values:   list[float]           = []
    n_skipped_tj         = 0
    n_skipped_no_data    = 0
    n_skipped_pair_filter = 0

    for (i, j) in frame.boundary_pixels:

        # ── TJ exclusion ──────────────────────────────────────────────────
        if (int(i), int(j)) in tj_excluded:
            n_skipped_tj += 1
            continue

        # ── Identify the grain pair at this pixel ─────────────────────────
        central = int(C0[i, j])
        ip = (i + 1) % nx;  im = (i - 1) % nx
        jp = (j + 1) % ny;  jm = (j - 1) % ny
        neighbors = {
            int(C0[ip, j]), int(C0[im, j]),
            int(C0[i,  jp]), int(C0[i,  jm]),
        }
        neighbors.discard(central)

        # Skip junction pixels (more than one foreign neighbor)
        if len(neighbors) != 1:
            continue

        neighbor_id = next(iter(neighbors))
        pair_id = (min(central, neighbor_id), max(central, neighbor_id))

        # ── Canonical GB-level filter ─────────────────────────────────────
        if pair_id not in valid_gb_dict:
            n_skipped_pair_filter += 1
            continue

        # ── Compute inclination (dx, dy) via myInput.get_grad ─────────────
        dx, dy = myInput.get_grad(P, int(i), int(j))

        # ── Dispatch to energy function ───────────────────────────────────
        try:
            if mode == "iso":
                gbe = 1.0

            elif mode == "inclination_cos":
                gbe = compute_finclination_cosine(dx, dy, inclination_anisotropy)

            elif mode == "inclination":
                gbe = compute_finclination(dx, dy)

            elif mode == "misorientation":
                if pair_id not in misorientation_data:
                    n_skipped_no_data += 1
                    continue
                m = misorientation_data[pair_id]
                gbe = compute_fmiso_single(
                    m["angle_deg"], m["ax_x"], m["ax_y"], m["ax_z"]
                )

            elif mode == "both":
                if pair_id not in misorientation_data:
                    n_skipped_no_data += 1
                    continue
                m = misorientation_data[pair_id]
                gbe = compute_fboth(
                    dx, dy,
                    m["angle_deg"], m["ax_x"], m["ax_y"], m["ax_z"]
                )

            else:
                raise ValueError(f"Unknown GBE mode: {mode}")

        except NotImplementedError:
            raise

        pixel_coords.append((int(i), int(j)))
        gbe_values.append(gbe)

    log.info(
        f"  GBE computed: {len(gbe_values)} pixels kept, "
        f"{n_skipped_tj} skipped (TJ proximity), "
        f"{n_skipped_pair_filter} skipped (GB-level area/curvature filter), "
        f"{n_skipped_no_data} skipped (no misorientation data)."
    )

    return np.array(pixel_coords, dtype=np.int32), np.array(gbe_values, dtype=np.float64)

# Inclination
def compute_inclination_per_pixel(
    frame: FrameData,
    tj_excluded: set[tuple[int, int]],
    valid_gb_dict: dict[tuple[int, int], np.ndarray],
    log: logging.Logger | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute inclination angle (radians, [0, 2π)) at every non-excluded
    boundary pixel using myInput.get_grad on the P field.

    Follows the same convention as get_normal_vector_slope:
        angle = atan2(-dy, dx) + π

    Parameters
    ----------
    frame         : FrameData (last frame)
    tj_excluded   : pre-built TJ proximity set — shared with GBE calculation
    valid_gb_dict : already-filtered GB dictionary. A pixel is kept only if
                    its pair_id is present in this dict.
    log           : logger

    Returns
    -------
    pixel_coords       : np.ndarray, shape (N, 2)
    inclination_angles : np.ndarray, shape (N,)
        Angle in radians at each kept boundary pixel, in [0, 2π).
    """
    if log is None:
        log = logging.getLogger("VHP")

    P  = frame.P
    C0 = frame.C[0]
    C1 = frame.C[1]
    nx, ny = C0.shape

    angles: list[float]           = []
    coords: list[tuple[int, int]] = []
    n_skipped_tj        = 0
    n_skipped_pair_filter = 0

    for (i, j) in frame.boundary_pixels:

        # ── TJ exclusion ──────────────────────────────────────────────────
        if (int(i), int(j)) in tj_excluded:
            n_skipped_tj += 1
            continue

        # ── Confirm clean GB pixel (not a junction) ───────────────────────
        central  = int(C0[i, j])
        ip = (i + 1) % nx;  im = (i - 1) % nx
        jp = (j + 1) % ny;  jm = (j - 1) % ny
        neighbors = {
            int(C0[ip, j]), int(C0[im, j]),
            int(C0[i, jp]), int(C0[i, jm]),
        }
        neighbors.discard(central)
        if len(neighbors) != 1:
            continue

        neighbor_id = next(iter(neighbors))
        pair_id = (min(central, neighbor_id), max(central, neighbor_id))

        # ── Canonical GB-level filter ─────────────────────────────────────
        if pair_id not in valid_gb_dict:
            n_skipped_pair_filter += 1
            continue

        dx, dy = myInput.get_grad(P, int(i), int(j))
        angle  = math.atan2(-dy, dx) + math.pi   # [0, 2π)
        angles.append(angle)
        coords.append((int(i), int(j)))

    log.info(
        f"  Inclination: {len(angles)} pixels computed, "
        f"{n_skipped_tj} skipped (TJ proximity), "
        f"{n_skipped_pair_filter} skipped (GB-level area/curvature filter)."
    )

    return np.array(coords, dtype=np.int32), np.array(angles, dtype=np.float64)


def compute_inclination_distribution(
    inclination_angles: np.ndarray,
    bin_width_deg: float = 10.01,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Bin inclination angles into a normalized frequency distribution,
    matching the convention in get_normal_vector_slope.

    Parameters
    ----------
    inclination_angles : (N,) array of angles in radians, [0, 2π)
    bin_width_deg      : bin width in degrees (default 10.01 matches reference fn)

    Returns
    -------
    theta_closed : (B+1,) bin centres in radians, loop-closed for polar plot
    r_closed     : (B+1,) normalized frequency, loop-closed
    bin_centers  : (B,)   bin centres in degrees (for CSV output)
    freq         : (B,)   normalized frequency   (for CSV output)
    """
    x_lim    = (0.0, 360.0)
    bin_num  = round((x_lim[1] - x_lim[0]) / bin_width_deg)
    bin_centers_deg = np.linspace(
        x_lim[0] + bin_width_deg / 2,
        x_lim[1] - bin_width_deg / 2,
        bin_num,
    )

    freq = np.zeros(bin_num)
    degrees = np.degrees(inclination_angles)   # [0°, 360°)

    for angle_deg in degrees:
        idx = int((angle_deg - x_lim[0]) / bin_width_deg)
        idx = min(idx, bin_num - 1)            # guard upper edge
        freq[idx] += 1

    freq = freq / np.sum(freq * bin_width_deg)  # normalize to PDF

    # Close the loop for polar plotting (0° connects back to 360°)
    theta        = bin_centers_deg / 180.0 * math.pi
    theta_closed = np.r_[theta, theta[0] + 2 * math.pi]
    r_closed     = np.r_[freq,  freq[0]]

    return theta_closed, r_closed, bin_centers_deg, freq


# ──────────────────────────────────────────────────────────────────────────────
#  Velocity
# ──────────────────────────────────────────────────────────────────────────────

def compute_dV_split(
    P0_current: np.ndarray,
    P0_next: np.ndarray,
    grain_id1: int,
    grain_id2: int,
) -> tuple[int, int, int]:
    """
    Count pixels that swapped grain identity between two frames for a GB pair.
    Mirrors compute_dV_split from the reference code, implemented in pure numpy
    using your P0 grain ID maps [1].

    A pixel "swaps" if it belonged to grain A in the current frame and grain B
    in the next frame — indicating the boundary moved through that pixel.

    positive dV means net label transition 1 -> 2,
    i.e. grain 2 gained area from grain 1.

    Parameters
    ----------
    P0_current : (nx, ny) int array  — grain ID map at time t
    P0_next    : (nx, ny) int array  — grain ID map at time t + dt
    grain_id1  : int
    grain_id2  : int

    Returns
    -------
    dV_1_to_2 : net pixel swap count (grain1->grain2 minus grain2->grain1)
    n_1_to_2  : pixels where grain1 grew into grain2
    n_2_to_1  : pixels where grain2 grew into grain1
    """
    n_1_to_2 = int(np.sum((P0_current == grain_id1) & (P0_next == grain_id2)))
    n_2_to_1 = int(np.sum((P0_current == grain_id2) & (P0_next == grain_id1)))
    dV_1_to_2 = n_1_to_2 - n_2_to_1
    return dV_1_to_2 , n_1_to_2, n_2_to_1


def compute_gb_velocity_one_interval(
    frame_current: FrameData,
    frame_next: FrameData,
    log: logging.Logger | None = None,
) -> dict[tuple[int, int], dict]:
    """
    Compute GB velocity for all qualifying pairs between two consecutive frames.

    Velocity is defined as:
        v = dV / dt / (area / 2)

    Parameters
    ----------
    frame_current : FrameData at time t
    frame_next    : FrameData at time t + dt
    frame_current.gb_dict and frame_next.gb_dict are expected to already be
    filtered with the canonical GB-level filters.

    Returns
    -------
    dict  pair_id -> {
        "velocity":          float,
        "dV":                int,
        "dV_forward":        int,
        "dV_backward":       int,
        "avg_curvature":     float,
        "area":              float,
        "is_anti_curvature": bool,
    }
    """
    if log is None:
        log = logging.getLogger("VHP")

    dt = frame_next.time - frame_current.time
    if dt <= 0.0:
        raise ValueError(
            f"Non-positive time interval between frames: "
            f"t_current={frame_current.time}, t_next={frame_next.time}"
        )

    P0_current = np.rint(frame_current.P0).astype(np.int32)
    P0_next    = np.rint(frame_next.P0).astype(np.int32)

    results: dict[tuple[int, int], dict] = {}
    n_skipped_gone  = 0

    for pair_id, data_current in frame_current.gb_dict.items():
        avg_curvature = float(data_current[0])
        area          = float(data_current[1])
        grain_id1     = int(data_current[2])
        grain_id2     = int(data_current[3])

        if pair_id not in frame_next.gb_dict:
            n_skipped_gone += 1
            continue                          # GB disappears in next frame

        # ── dV calculation ────────────────────────────────────────────────
        dV, dV_forward, dV_backward = compute_dV_split(
            P0_current, P0_next, grain_id1, grain_id2
        )

        # Normalize by time and half-area
        velocity = dV / dt / (area / 2.0)

        results[pair_id] = {
            "velocity":          velocity,
            "dV":                dV,
            "dV_forward":        dV_forward,
            "dV_backward":       dV_backward,
            "avg_curvature":     avg_curvature,
            "area":              area,
            "is_anti_curvature": (avg_curvature * velocity) < 0.0,
        }

    log.info(
        f"  Velocity [{frame_current.step}->{frame_next.step}]: "
        f"{len(results)} pairs computed, "
        f"{n_skipped_gone} skipped (not present after next-frame filters or GB gone)."
    )
    return results


def compute_gb_velocity_averaged(
    frames: list[FrameData],
    log: logging.Logger | None = None,
) -> pd.DataFrame:
    """
    Compute per-GB velocity averaged across all consecutive frame pairs.

    Parameters
    ----------
    frames : all loaded frames (in time order), with gb_dict already filtered
    """
    if log is None:
        log = logging.getLogger("VHP")

    if len(frames) < 2:
        raise ValueError("Velocity calculation requires at least 2 frames.")

    accum: dict[tuple[int, int], list[dict]] = {}

    for i in range(len(frames) - 1):
        interval_results = compute_gb_velocity_one_interval(
            frame_current = frames[i],
            frame_next    = frames[i + 1],
            log           = log,
        )
        for pair_id, res in interval_results.items():
            accum.setdefault(pair_id, []).append(res)

    # Aggregate into one row per pair  (body unchanged from original)
    rows = []
    for pair_id, res_list in accum.items():
        velocities  = [r["velocity"]          for r in res_list]
        anti_curv   = [r["is_anti_curvature"] for r in res_list]
        dV_fwd_tot  = sum(r["dV_forward"]     for r in res_list)
        dV_bwd_tot  = sum(r["dV_backward"]    for r in res_list)
        last = res_list[-1]

        rows.append({
            "pair_id":                 pair_id,
            "grain_id1":               int(frames[0].gb_dict.get(pair_id, [0,0,0,0])[2]),
            "grain_id2":               int(frames[0].gb_dict.get(pair_id, [0,0,0,0])[3]),
            "velocity_mean":           float(np.mean(velocities)),
            "velocity_std":            float(np.std(velocities)),
            "n_intervals":             len(velocities),
            "avg_curvature":           last["avg_curvature"],
            "area":                    last["area"],
            "dV_forward_total":        dV_fwd_tot,
            "dV_backward_total":       dV_bwd_tot,
            "anti_curvature_fraction": float(np.mean(anti_curv)),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        log.warning(
            "\033[31mVelocity:\033[0m Velocity DataFrame is empty — all GB pairs were "
            "filtered out. Check --min-area against your data, and confirm "
            "gb_dict pair_ids match across frames."
        )
        return pd.DataFrame(columns=[
            "pair_id", "grain_id1", "grain_id2",
            "velocity_mean", "velocity_std", "n_intervals",
            "avg_curvature", "area",
            "dV_forward_total", "dV_backward_total",
            "anti_curvature_fraction",
        ])
    log.warning(
        f"Velocity averaged: {len(df)} GB pairs across "
        f"{len(frames) - 1} interval(s)."
    )
    return df



def accumulate_velocity_flat_lists(
    frames: list[FrameData],
    log: logging.Logger | None = None,
) -> dict:
    """
    Accumulate per-interval velocity results into flat lists.

    Parameters
    ----------
    frames : all loaded frames (in time order), with gb_dict already filtered
    """
    if log is None:
        log = logging.getLogger("VHP")

    flat = {
        "all_curvatures":       [],
        "all_velocities":       [],
        "all_dV_forward":       [],
        "all_dV_backward":      [],
        "all_areas":            [],
        "all_pair_ids":         [],
        "all_timestep_indices": [],
        "all_interval_results": [],
    }

    for i in range(len(frames) - 1):
        interval_results = compute_gb_velocity_one_interval(
            frame_current = frames[i],
            frame_next    = frames[i + 1],
            log           = log,
        )
        flat["all_interval_results"].append(interval_results)

        for pair_id, res in interval_results.items():
            flat["all_curvatures"].append(res["avg_curvature"])
            flat["all_velocities"].append(res["velocity"])
            flat["all_dV_forward"].append(res["dV_forward"])
            flat["all_dV_backward"].append(res["dV_backward"])
            flat["all_areas"].append(res["area"])
            flat["all_pair_ids"].append(pair_id)
            flat["all_timestep_indices"].append(i)

        log.info(
            f"  Interval {i}->{i+1}: {len(interval_results)} pairs accumulated. "
            f"Flat list size now: {len(flat['all_curvatures'])}"
        )

    return flat



def apply_curvature_sign_convention(
    curvatures:   list[float],
    velocities:   list[float],
    dV_forward:   list[int],
    dV_backward:  list[int],
) -> tuple[list[float], list[float], list[int], list[int]]:
    """
    Flip signs so all curvatures are positive (|κ|) and the velocity
    sign carries the physical meaning.

    For each entry where curvature < 0:
      - flip sign of both curvature and velocity
      - swap dV_forward and dV_backward so 'forward' always means
        the curvature-driven direction [1]

    This means the scatter x-axis is always |κ| ≥ 0, matching
    the reference figures exactly.
    """
    out_curv = []
    out_vel  = []
    out_fwd  = []
    out_bwd  = []

    for kappa, v, fwd, bwd in zip(curvatures, velocities, dV_forward, dV_backward):
        if kappa < 0.0:
            out_curv.append(-kappa) # flip to positive
            out_vel.append(-v) # flip velocity too
            out_fwd.append(bwd)   # swap: grain2->grain1 becomes "forward"
            out_bwd.append(fwd)
        else:
            out_curv.append(kappa)
            out_vel.append(v)
            out_fwd.append(fwd)
            out_bwd.append(bwd)

    return out_curv, out_vel, out_fwd, out_bwd


def apply_confidence_filter(
    curvatures:  list[float],
    velocities:  list[float],
    dV_forward:  list[int],
    dV_backward: list[int],
    areas:       list[float],
    pair_ids:    list[tuple[int,int]],
    mode:        str,           # "antic" or "normc"
    confidence:  float = 0.99,
    log: logging.Logger | None = None,
) -> dict:
    """
    Filter entries by voxel-level directional confidence [1].

    For anti-curvature entries (mode='antic'):
        keep if  dV_forward / (dV_forward + dV_backward) > confidence
        i.e. >= 99% of moved voxels went AGAINST curvature

    For normal-curvature entries (mode='normc'):
        keep if  dV_forward / (dV_forward + dV_backward) > confidence
        i.e. >= 99% of moved voxels went WITH curvature

    After apply_curvature_sign_convention, dV_forward is always the
    curvature-direction count, so the ratio is the same expression
    for both modes.
    """
    if log is None:
        log = logging.getLogger("VHP")

    kept = {
        "curvatures":  [],
        "velocities":  [],
        "dV_forward":  [],
        "dV_backward": [],
        "areas":       [],
        "pair_ids":    [],
    }
    n_total  = len(curvatures)
    n_kept   = 0
    n_zero   = 0

    for kappa, v, fwd, bwd, area, pid in zip(
        curvatures, velocities, dV_forward, dV_backward, areas, pair_ids
    ):
        total_dV = fwd + bwd
        if total_dV == 0:
            n_zero += 1
            continue
        # Anti-c GBs move against curvature: backward direction dominates
        if mode == "antic":
            ratio = bwd / total_dV
        else:
            ratio = fwd / total_dV
        if ratio > confidence:
            kept["curvatures"].append(kappa)
            kept["velocities"].append(v)
            kept["dV_forward"].append(fwd)
            kept["dV_backward"].append(bwd)
            kept["areas"].append(area)
            kept["pair_ids"].append(pid)
            n_kept += 1

    log.warning(
        f"  Confidence filter ({mode}, >{confidence*100:.0f}%): "
        f"{n_kept}/{n_total} kept, {n_zero} skipped (zero dV)."
    )
    return kept


def compute_velocity_bins(
    curvatures:    list[float],
    velocities:    list[float],
    x_lim:         tuple[float, float] = (0.0, 0.1),
    bin_interval:  float = 0.002,
    min_count:     int   = 10,
) -> dict:
    """
    Bin velocities by curvature and compute mean ± std per bin,
    matching the reference binning loop exactly [1].

    Returns
    -------
    dict with keys:
        bin_centers  : (B,) bin centre coordinates
        bin_means    : (B,) mean velocity per bin (0 if empty)
        bin_stds     : (B,) std  velocity per bin (0 if empty)
        bin_counts   : (B,) number of entries per bin
        valid_mask   : (B,) bool — bins with count > min_count
    """
    bin_number = int((x_lim[1] - x_lim[0]) / bin_interval)
    bin_centers = (
        np.arange(x_lim[0], x_lim[1], bin_interval) + bin_interval / 2.0
    )

    counts  = np.zeros(bin_number, dtype=int)
    sums    = np.zeros(bin_number, dtype=float)
    sq_sums = np.zeros(bin_number, dtype=float)

    for kappa, v in zip(curvatures, velocities):
        if kappa < x_lim[0] or kappa >= x_lim[1]:
            continue
        idx = int((kappa - x_lim[0]) / bin_interval)
        idx = min(idx, bin_number - 1)
        counts[idx]  += 1
        sums[idx]    += v
        sq_sums[idx] += v * v

    means = np.zeros(bin_number)
    stds  = np.zeros(bin_number)
    for i in range(bin_number):
        if counts[i] > 0:
            means[i] = sums[i] / counts[i]
            stds[i]  = np.sqrt(
                max(0.0, sq_sums[i] / counts[i] - means[i] ** 2)
            )

    return {
        "bin_centers": bin_centers,
        "bin_means":   means,
        "bin_stds":    stds,
        "bin_counts":  counts,
        "valid_mask":  counts > min_count,
    }



def find_isolated_anticurvature_events(
    all_interval_results: list[dict],
    log: logging.Logger | None = None,
) -> set[tuple[tuple[int, int], int]]:
    """
    Find isolated "00100" anti-curvature events from Lin's sliding-window filter.

    A (pair_id, timestep_index) is returned if:
      - it IS anti-curvature at center step t
      - it is NOT anti-curvature at t-2, t-1, t+1, or t+2

    In Lin's code, these returned events are treated as transient/noisy and
    removed from anti-curvature tracking.

    Parameters
    ----------
    all_interval_results : list of per-interval result dicts,
                           one per consecutive frame pair, in order.
                           Each dict maps pair_id -> result dict
                           with key "is_anti_curvature".

    Returns
    -------
    rejected_set : set of (pair_id, timestep_index) tuples that trip
               the filter. Empty if fewer than 5 intervals available.
    """
    if log is None:
        log = logging.getLogger("VHP")

    n = len(all_interval_results)

    if n < 5:
        log.warning(
            f"  Sliding window filter: only {n} interval(s) available — "
            f"need ≥5. Filter skipped; returning empty set. "
            f"Add more frames to enable this filter."
        )
        return set()

    # Build per-step sets of anti-curvature pair_ids (mirrors GB_filter_kernel [1])
    antic_sets: list[set[tuple[int,int]]] = []
    for step_idx, interval in enumerate(all_interval_results):
        step_set = {
            pid for pid, res in interval.items() if res["is_anti_curvature"]
        }
        antic_sets.append(step_set)

    rejected_set: set[tuple[tuple[int,int], int]] = set()
    n_total   = 0
    n_removed = 0

    # Slide a 5-step window: centre at t, flanks at t-2, t-1, t+1, t+2
    for t in range(2, n - 2):
        centre   = antic_sets[t]
        flanks   = antic_sets[t-2] | antic_sets[t-1] | antic_sets[t+1] | antic_sets[t+2]

        # "00100": in centre but NOT in any flank step
        filtered = centre - flanks
        n_total  += len(centre)
        n_removed += len(filtered) #len(centre) - len(filtered)

        for pid in filtered:
            rejected_set.add((pid, t))

    log.warning(
        f"  Sliding window filter: {len(rejected_set)} (pair, step) entries rejected "
        f"from {n_total} anti-curvature candidates "
        f"({n_removed} removed as transient)."
    )
    return rejected_set




# ──────────────────────────────────────────────────────────────────────────────
#  Debugging
# ──────────────────────────────────────────────────────────────────────────────

def plot_gbe_debug(
    frame: FrameData,
    pixel_coords: np.ndarray,
    gbe_values: np.ndarray,
    mode: str,
    tj_exclude: bool,
    output_dir: Path,
    stem: str = "gbe_debug",
) -> None:
    """
    Debug heatmap of GBE values overlaid on the grain ID map.

    Boundary pixels not included in the GBE calculation (e.g. TJ-excluded)
    are shown in a neutral colour so the exclusion pattern is visible.

    Parameters
    ----------
    frame        : FrameData (last frame)
    pixel_coords : (N, 2) array of (i, j) coords returned by compute_gbe_per_pixel
    gbe_values   : (N,)   GBE value at each coord
    mode         : GBE mode string — used in title
    tj_exclude   : bool — noted in title
    output_dir   : Path  — where to save the figure
    stem         : str   — filename prefix
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    C0 = frame.C[0]
    nx, ny = C0.shape

    # ── Build a 2D GBE image (NaN everywhere except computed boundary pixels) ──
    gbe_map = np.full((nx, ny), np.nan, dtype=np.float64)
    for (i, j), val in zip(pixel_coords, gbe_values):
        gbe_map[i, j] = val

    fig, ax = plt.subplots(figsize=(8, 7))

    # Grain ID background — faint, for structural context
    ax.imshow(C0, origin="lower", cmap="gray", alpha=0.25, interpolation="nearest")

    # GBE heatmap — only paints pixels with a real value
    im = ax.imshow(
        gbe_map,
        origin="lower",
        cmap="plasma",
        interpolation="nearest",
        vmin=np.nanmin(gbe_values),
        vmax=np.nanmax(gbe_values),
    )

    # Optionally mark junction pixels to show what was excluded
    if tj_exclude and len(frame.junction_pixels) > 0:
        jp = frame.junction_pixels
        ax.scatter(
            jp[:, 1], jp[:, 0],
            c="cyan", s=2, linewidths=0,
            label="Junction pixels (excluded region centres)",
            zorder=3,
        )
        ax.legend(loc="upper right", fontsize=7, markerscale=3)

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("GBE", fontsize=10)

    tj_str = "TJ-excluded" if tj_exclude else "TJ-included"
    ax.set_title(
        f"GBE Debug Heatmap\nmode={mode}  |  {tj_str}  |  "
        f"N={len(gbe_values)} pixels\n"
        f"min={np.nanmin(gbe_values):.4f}  "
        f"mean={np.nanmean(gbe_values):.4f}  "
        f"max={np.nanmax(gbe_values):.4f}",
        fontsize=10,
    )
    ax.set_xlabel("j (x)")
    ax.set_ylabel("i (y)")

    plt.tight_layout()
    outpath = output_dir / f"{stem}_DEBUG_gbe_heatmap.png"
    fig.savefig(outpath, dpi=300, transparent=True)
    plt.close(fig)
    print(f"Debug GBE heatmap saved: {outpath}")



def plot_inclination_polar(
    theta_closed: np.ndarray,
    r_closed: np.ndarray,
    output_dir: Path,
    stem: str,
    tj_exclude: bool,
    n_pixels: int,
    log: logging.Logger,
) -> None:
    """
    Polar plot of the inclination angle distribution.
    Saved as <stem>_DEBUG_inclination_polar.png, consistent with
    the debug plot naming convention used in vector_exodus_to_hdf5.py [1].
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(6, 6))

    ax.plot(theta_closed, r_closed, linewidth=2, color="steelblue", label="Inclination")
    ax.fill(theta_closed, r_closed, alpha=0.15, color="steelblue")

    ax.set_rgrids(np.arange(0, 0.01, 0.004))  # Radial grid lines every 0.004 units
    ax.set_rlabel_position(0.0)  # Position radial labels at 0-degree angle
    ax.set_rlim(0.0, 0.01)       # Radial axis limits for probability density range
    ax.set_yticklabels(['0', '4e-3', '8e-3'],fontsize=8)

    tj_str = "TJ-excluded" if tj_exclude else "TJ-included"
    ax.set_title(
        f"Inclination Distribution\n{tj_str}  |  N={n_pixels} pixels",
        fontsize=10, pad=25,
    )
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8)

    plt.tight_layout()
    outpath = output_dir / f"{stem}_DEBUG_inclination_polar.png"
    fig.savefig(outpath, dpi=300, transparent=True)
    plt.close(fig)
    log.warning(f"Inclination polar plot saved: {outpath}")


def plot_velocity_debug(
    flat_lists:      dict,
    normc_flat:      dict,
    antic_flat:      dict,
    normc_flat_conf: dict,
    antic_flat_conf: dict,
    last_frame:      FrameData,
    frames:          list[FrameData],
    velocity_df:     pd.DataFrame,
    output_dir:      Path,
    stem:            str,
    tj_exclude:      bool,
    log:             logging.Logger,
    curvature_limit: float = 0.0182,
    x_lim:           tuple[float, float] = (0.0, 0.1),
    bin_interval:    float = 0.002,
) -> None:
    """
    Four-panel debug figure matching the reference output style [1]:

    Panel 1 : Velocity heatmap (unchanged from original)
    Panel 2 : Raw scatter — normc (blue) vs antic (orange), signed curvature
    Panel 3 : Density contour + binned mean ± std + linear fits (pre-confidence)
    Panel 4 : Confidence-filtered scatter — normc vs antic (|κ| on x-axis)
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    xmin_curve = curvature_limit * 0.95 if curvature_limit > 0 else -0.005
    C0 = last_frame.C[0]
    nx, ny = C0.shape
    tj_str = "TJ-excluded" if tj_exclude else "TJ-included"

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # ── Panel 1: Velocity heatmap ─────────────────────────────────────────
    ax = axes[0, 0]

    # Build velocity map from averaged df for the heatmap (visual only)
    vel_lookup = {
        row.pair_id: row.velocity_mean
        for row in velocity_df.itertuples(index=False)
    } if velocity_df is not None and not velocity_df.empty else {}

    vel_map = np.full((nx, ny), np.nan, dtype=np.float64)
    for (i, j) in last_frame.boundary_pixels:
        central = int(C0[i, j])
        ip = (i+1)%nx; im = (i-1)%nx
        jp = (j+1)%ny; jm = (j-1)%ny
        neighbors = {
            int(C0[ip,j]), int(C0[im,j]),
            int(C0[i,jp]), int(C0[i,jm]),
        }
        neighbors.discard(central)
        if len(neighbors) != 1:
            continue
        pid = (min(central, next(iter(neighbors))),
               max(central, next(iter(neighbors))))
        if pid in vel_lookup:
            vel_map[i, j] = vel_lookup[pid]

    valid_vals = vel_map[~np.isnan(vel_map)]
    vabs = np.percentile(np.abs(valid_vals), 95) if len(valid_vals) > 0 else 1.0

    ax.imshow(C0, origin="lower", cmap="gray", alpha=0.25, interpolation="nearest")
    im = ax.imshow(vel_map, origin="lower", cmap="RdBu",
                   interpolation="nearest", vmin=-vabs, vmax=vabs)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label(
        "Velocity mean (px/s)", fontsize=9)
    ax.set_title(
        f"Velocity Heatmap\n{tj_str}  |  N={len(velocity_df) if velocity_df is not None else 0} GB pairs",
        fontsize=9)
    ax.set_xlabel("j (x)"); ax.set_ylabel("i (y)")

    # ── Panel 2: Raw scatter — normc vs antic (signed curvature) ─────────
    ax = axes[0, 1]
    ax.axhline(0, color="gray", linewidth=0.8, linestyle=":")
    ax.axvline(0, color="gray", linewidth=0.8, linestyle=":")
    ax.plot([curvature_limit, x_lim[1]], [0, 0], "-", color="grey", linewidth=1.5)

    ax.scatter(
        normc_flat["curvatures"], normc_flat["velocities"],
        s=10, alpha=0.8, color="C0", label=f"Normal-c ({len(normc_flat['curvatures'])})",
    )
    # ax.scatter(
    #     [-v for v in antic_flat["velocities"]],   # anti-c plotted with negative v
    #     antic_flat["velocities"],
    #     s=8, alpha=0.8, color="C1",
    #     label=f"Anti-c ({len(antic_flat['velocities'])})",
    # )
    # Correct: x-axis uses |κ| for both (already flipped by sign convention)
    ax.scatter(
        antic_flat["curvatures"], antic_flat["velocities"],
        s=10, alpha=0.8, color="C1",label=f"Anti-c ({len(antic_flat['velocities'])})"
    )

    ax.set_xlim([xmin_curve, x_lim[1]]) #[curvature_limit, x_lim[1]])
    ax.set_ylim([
        min(flat_lists["all_velocities"]) * 1.1 if flat_lists["all_velocities"] else -1,
        max(flat_lists["all_velocities"]) * 1.1 if flat_lists["all_velocities"] else  1,
    ])
    ax.set_xlabel("|κ| (px⁻¹)", fontsize=11)
    ax.set_ylabel("velocity (px/s)", fontsize=11)
    ax.set_title("Raw Scatter: Normal-c vs Anti-c\n(pre-confidence filter)", fontsize=9)
    ax.legend(fontsize=8)

    # ── Panel 3: Density contour + binned mean ± std ──────────────────────
    ax = axes[1, 0]

    all_c = normc_flat["curvatures"] + antic_flat["curvatures"]
    all_v = normc_flat["velocities"] + antic_flat["velocities"]

    x_bins = np.linspace(x_lim[0], x_lim[1], 40)
    y_abs  = max(abs(v) for v in all_v) * 1.1 if all_v else 1.0
    y_bins = np.linspace(-y_abs, y_abs, 40)

    # if len(all_c) >= 2:
    #     hist, x_edges, y_edges = np.histogram2d(all_c, all_v,
    #                                               bins=[x_bins, y_bins])
    #     x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    #     y_centers = (y_edges[:-1] + y_edges[1:]) / 2
    #     X, Y = np.meshgrid(x_centers, y_centers)
    #     hist.T[hist.T == 0] = 1
    #     ax.contourf(X, Y, np.log10(hist.T), levels=20,
    #                 cmap="coolwarm", alpha=0.9, vmin=0)
    #     ax.contour(X, Y, np.log10(hist.T), levels=20,
    #                cmap="gray", alpha=0.1, vmin=0)
    if len(all_c) >= 2:
        hist, x_edges, y_edges = np.histogram2d(all_c, all_v,
                                                bins=[x_bins, y_bins])
        x_centers = (x_edges[:-1] + x_edges[1:]) / 2
        y_centers = (y_edges[:-1] + y_edges[1:]) / 2
        X, Y = np.meshgrid(x_centers, y_centers)
        hist_log = np.full_like(hist.T, np.nan, dtype=float)
        nz = hist.T > 0
        hist_log[nz] = np.log10(hist.T[nz])
        cf = ax.contourf(X, Y, hist_log, levels=20,
                        cmap="coolwarm", alpha=0.9, vmin=0)
        ax.contour(X, Y, hist_log, levels=20,
                cmap="gray", alpha=0.1, vmin=0)
        cbar = plt.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(r"log$_{10}$(count)", fontsize=9)

    # Binned mean ± std overlay
    bins = compute_velocity_bins(all_c, all_v, x_lim=x_lim,
                                  bin_interval=bin_interval, min_count=10)
    valid = bins["valid_mask"]
    if valid.any():
        ax.errorbar(
            bins["bin_centers"][valid],
            bins["bin_means"][valid],
            yerr=bins["bin_stds"][valid],
            fmt="o", color="k", linewidth=1, capsize=2,
            ecolor="black", markersize=3, label="Bin mean ± std",
        )

        # Linear fit — all valid bins
        x_all = bins["bin_centers"][valid]
        y_all = bins["bin_means"][valid]
        p_all = np.polyfit(x_all, y_all, 1)
        y_pred_all = np.polyval(p_all, x_all)
        ss_res = np.sum((y_all - y_pred_all)**2)
        ss_tot = np.sum((y_all - y_all.mean())**2)
        r2_all = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        ax.plot(x_all, y_pred_all, "-", color="C1", linewidth=2,
                label=rf"All (R²={r2_all:.3f}, M*={p_all[0]:.4f})")
        log.warning(f"Binned fit (all):   slope={p_all[0]:.4f}  R²={r2_all:.4f}")

        # Linear fit — low-κ bins only (κ < 0.03, matching reference [1])
        mask_sub = x_all < 0.03
        if mask_sub.sum() > 1:
            x_sub = x_all[mask_sub]; y_sub = y_all[mask_sub]
            p_sub = np.polyfit(x_sub, y_sub, 1)
            y_pred_sub = np.polyval(p_sub, x_sub)
            ss_res_s = np.sum((y_sub - y_pred_sub)**2)
            ss_tot_s = np.sum((y_sub - y_sub.mean())**2)
            r2_sub = 1.0 - ss_res_s / ss_tot_s if ss_tot_s > 0 else float("nan")
            ax.plot(x_sub, y_pred_sub, "-", color="C2", linewidth=2,
                    label=rf"κ<0.03 (R²={r2_sub:.3f}, M*={p_sub[0]:.4f})")
            log.warning(f"Binned fit (κ<0.03): slope={p_sub[0]:.4f}  R²={r2_sub:.4f}")

    ax.axhline(0, color="gray", linewidth=0.8, linestyle=":")
    ax.plot([curvature_limit, x_lim[1]], [0, 0], "-", color="grey", linewidth=1.5)
    ax.set_xlim([curvature_limit, x_lim[1]]) #curvature_limit
    ax.set_xlabel("|κ| (px⁻¹)", fontsize=11)
    ax.set_ylabel("velocity (px/s)", fontsize=11)
    ax.set_title("Density + Binned Mean ± Std\n(pre-confidence filter)", fontsize=9)
    ax.legend(fontsize=8, loc="lower right")

    # ── Panel 4: Confidence-filtered scatter ─────────────────────────────
    ax = axes[1, 1]
    ax.axhline(0, color="gray", linewidth=0.8, linestyle=":")
    ax.plot([curvature_limit, x_lim[1]], [0, 0], "-", color="grey", linewidth=1.5)

    ax.scatter(
        normc_flat_conf["curvatures"], normc_flat_conf["velocities"],
        s=4, alpha=0.5, color="C0",
        label=f"Normal-c ({len(normc_flat_conf['curvatures'])})",
    )
    ax.scatter(
        antic_flat_conf["curvatures"], antic_flat_conf["velocities"],
        s=8, alpha=0.5, color="C1",
        label=f"Anti-c ({len(antic_flat_conf['curvatures'])})",
    )
    ax.set_xlim([xmin_curve, x_lim[1]]) #curvature_limit
    ax.set_xlabel("|κ| (px⁻¹)", fontsize=11)
    ax.set_ylabel("velocity (px/s)", fontsize=11)
    ax.set_title("Confidence-Filtered Scatter (99%)\nnormc (blue) vs anti-c (orange)", fontsize=9)
    ax.legend(fontsize=8)

    plt.suptitle(
        f"Velocity Analysis — {stem} — {len(frames)-1} interval(s)",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()

    outpath = output_dir / f"{stem}_DEBUG_velocity.png"
    fig.savefig(outpath, dpi=300, transparent=True)
    plt.close(fig)
    log.warning(f"Velocity debug plot saved: {outpath}")


def plot_curvature_debug(
    frame: FrameData,
    tj_excluded: set[tuple[int, int]],
    tj_exclude: bool,
    output_dir: Path,
    stem: str,
    log: logging.Logger,
) -> None:
    """
    Debug heatmap of signed curvature values overlaid on the grain ID map.

    Generates TWO figures side by side for direct comparison:
        Figure A : All GB pixels (TJ-included) — full curvature field
        Figure B : TJ-excluded GB pixels only   — mirrors the filtering
                   applied in the velocity and GBE pipelines [2]

    Each figure has four panels:
        Panel 1 : Signed curvature heatmap (diverging, zero = white)
        Panel 2 : |curvature| heatmap (sequential)
        Panel 3 : Histogram of signed curvature at all plotted GB pixels
        Panel 4 : |curvature| histogram with reference threshold lines

    Parameters
    ----------
    frame       : FrameData  (any frame — typically last_frame)
    tj_excluded : pre-built TJ proximity set from build_tj_proximity_set()
                  Pass the same set used by the velocity/GBE pipeline [2]
                  so the exclusion zones match exactly.
    tj_exclude  : bool — if False, Figure B is skipped and a warning is logged
    output_dir  : Path
    stem        : filename prefix
    log         : logger
    """
    import matplotlib.pyplot as plt

    C0        = frame.C[0]   # grain ID map
    curv_full = frame.C[1]   # signed curvature field (full grid)
    nx, ny    = C0.shape

    # ── Helper: build curvature map for a given exclusion set ─────────────
    def _build_curv_map(exclude_set: set | None) -> tuple[np.ndarray, list[float]]:
        """
        Walk boundary_pixels and paint curvature values into a 2D map.

        Parameters
        ----------
        exclude_set : set of (i,j) to skip, or None to skip nothing
                      (None = TJ-included mode)

        Returns
        -------
        curv_map  : (nx, ny) float array, NaN where no GB pixel
        curv_vals : flat list of curvature values at kept pixels
        """
        curv_map  = np.full((nx, ny), np.nan, dtype=np.float64)
        curv_vals: list[float] = []

        for (i, j) in frame.boundary_pixels:
            # Apply exclusion if requested
            if exclude_set is not None and (int(i), int(j)) in exclude_set:
                continue

            central = int(C0[i, j])
            ip = (i + 1) % nx;  im = (i - 1) % nx
            jp = (j + 1) % ny;  jm = (j - 1) % ny
            neighbors = {
                int(C0[ip, j]), int(C0[im, j]),
                int(C0[i, jp]), int(C0[i, jm]),
            }
            neighbors.discard(central)
            # Include junction pixels in TJ-included mode so the full
            # boundary is visible; they are simply not in exclude_set.
            kappa = float(curv_full[i, j])
            curv_map[i, j] = kappa
            curv_vals.append(kappa)

        return curv_map, curv_vals

    # ── Helper: render a single 4-panel figure ────────────────────────────
    def _render_figure(
        curv_map:   np.ndarray,
        curv_vals:  list[float],
        title_tag:  str,        # e.g. "TJ-included" or "TJ-excluded"
        out_suffix: str,        # appended to stem for filename
    ) -> None:
        if len(curv_vals) == 0:
            log.warning(
                f"plot_curvature_debug ({title_tag}): "
                f"no boundary pixels found — skipping."
            )
            return

        curv_arr = np.array(curv_vals)
        abs_arr  = np.abs(curv_arr)

        # Symmetric color limit — robust to outliers
        vabs = float(np.percentile(abs_arr, 98))

        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        # ── Panel 1: Signed curvature heatmap ────────────────────────────
        ax = axes[0, 0]
        ax.imshow(C0, origin="lower", cmap="gray", alpha=0.25,
                  interpolation="nearest")
        im1 = ax.imshow(
            curv_map, origin="lower", cmap="RdBu",
            interpolation="nearest", vmin=-vabs, vmax=vabs,
        )
        plt.colorbar(im1, ax=ax, fraction=0.046, pad=0.04).set_label(
            "Signed curvature (px⁻¹)", fontsize=9)
        ax.set_title(
            f"Signed Curvature — {title_tag}\n"
            f"N={len(curv_vals)} px  "
            f"min={curv_arr.min():.5f}  max={curv_arr.max():.5f}  "
            f"mean={curv_arr.mean():.5f}",
            fontsize=9,
        )
        ax.set_xlabel("j (x)"); ax.set_ylabel("i (y)")

        # ── Panel 2: |curvature| heatmap ─────────────────────────────────
        ax = axes[0, 1]
        abs_map = np.where(np.isnan(curv_map), np.nan, np.abs(curv_map))
        ax.imshow(C0, origin="lower", cmap="gray", alpha=0.25,
                  interpolation="nearest")
        im2 = ax.imshow(
            abs_map, origin="lower", cmap="plasma",
            interpolation="nearest", vmin=0.0, vmax=vabs,
        )
        plt.colorbar(im2, ax=ax, fraction=0.046, pad=0.04).set_label(
            "|curvature| (px⁻¹)", fontsize=9)
        ax.set_title(
            f"|Curvature| heatmap — {title_tag}\n"
            f"98th pct={vabs:.5f}  median={float(np.median(abs_arr)):.5f}",
            fontsize=9,
        )
        ax.set_xlabel("j (x)"); ax.set_ylabel("i (y)")

        # ── Panel 3: Signed curvature histogram ──────────────────────────
        ax = axes[1, 0]
        ax.hist(curv_arr, bins=120, color="steelblue", alpha=0.8,
                edgecolor="none")
        ax.axvline(0, color="black", linewidth=1.2, linestyle="--",
                   label="κ = 0")
        ax.axvline( 0.0182, color="C1", linewidth=1.5, linestyle=":",
                    label="ref threshold +0.0182 [1]")
        ax.axvline(-0.0182, color="C1", linewidth=1.5, linestyle=":")
        ax.set_xlabel("Signed curvature (px⁻¹)", fontsize=11)
        ax.set_ylabel("Pixel count", fontsize=11)
        ax.set_title(
            f"Signed Curvature Distribution — {title_tag}\n"
            f"(all plotted GB pixels, pre-filter)",
            fontsize=9,
        )
        ax.legend(fontsize=8)

        # ── Panel 4: |curvature| histogram + threshold lines ─────────────
        ax = axes[1, 1]
        ax.hist(abs_arr, bins=100, color="darkorange", alpha=0.8,
                edgecolor="none")

        pct_ref = 100.0 * (abs_arr >= 0.0182).mean()
        pct_0   = 100.0 * (abs_arr >  0.0).mean()

        ax.axvline(
            0.0182, color="C1", linewidth=2.0, linestyle="--",
            label=f"ref min_curvature=0.0182 [1]\n"
                  f"({(abs_arr >= 0.0182).sum()} / {len(abs_arr)} px survive)",
        )
        ax.set_xlabel("|curvature| (px⁻¹)", fontsize=11)
        ax.set_ylabel("Pixel count", fontsize=11)
        ax.set_title(
            f"|Curvature| Distribution — {title_tag}\n"
            f">{0.0182:.4f}: {pct_ref:.1f}% pass  |  >0: {pct_0:.1f}%",
            fontsize=9,
        )
        ax.legend(fontsize=8)

        plt.suptitle(
            f"Curvature Debug ({title_tag}) — {stem}  |  step={frame.step}",
            fontsize=12, fontweight="bold",
        )
        plt.tight_layout()

        outpath = output_dir / f"{stem}_DEBUG_curvature_{out_suffix}.png"
        fig.savefig(outpath, dpi=300, transparent=True)
        plt.close(fig)
        log.warning(f"Curvature debug plot saved: {outpath}")

    # ── Figure A: TJ-included (no exclusion) ─────────────────────────────
    curv_map_all, curv_vals_all = _build_curv_map(exclude_set=None)
    _render_figure(
        curv_map   = curv_map_all,
        curv_vals  = curv_vals_all,
        title_tag  = "TJ-included",
        out_suffix = "tj_included",
    )

    # ── Figure B: TJ-excluded (uses the same set as velocity/GBE) ────────
    if not tj_exclude:
        log.warning(
            "plot_curvature_debug: --tj-exclude is False — "
            "skipping TJ-excluded figure."
        )
        return

    if len(tj_excluded) == 0:
        log.warning(
            "plot_curvature_debug: tj_excluded set is empty — "
            "skipping TJ-excluded figure. "
            "Check that build_tj_proximity_set() was called before this."
        )
        return

    curv_map_excl, curv_vals_excl = _build_curv_map(exclude_set=tj_excluded)
    n_removed = len(curv_vals_all) - len(curv_vals_excl)
    log.warning(
        f"Curvature debug TJ exclusion: "
        f"{n_removed} pixels removed "
        f"({100.0 * n_removed / max(len(curv_vals_all), 1):.1f}% of boundary pixels), "
        f"{len(curv_vals_excl)} remaining."
    )
    _render_figure(
        curv_map   = curv_map_excl,
        curv_vals  = curv_vals_excl,
        title_tag  = "TJ-excluded",
        out_suffix = "tj_excluded",
    )



# ──────────────────────────────────────────────────────────────────────────────
#  Output Writing
# ──────────────────────────────────────────────────────────────────────────────


def save_inclination_csv(
    bin_centers_deg: np.ndarray,
    freq: np.ndarray,
    output_dir: Path,
    stem: str,
    log: logging.Logger,
) -> None:
    """
    Write the inclination distribution to a two-column CSV:
        angle_deg, normalized_frequency
    """
    import csv
    outpath = output_dir / f"{stem}_inclination_distribution.csv"
    with open(outpath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["angle_deg", "normalized_frequency"])
        for angle, nf in zip(bin_centers_deg, freq):
            writer.writerow([f"{angle:.4f}", f"{nf:.8f}"])
    log.warning(f"Inclination CSV saved: {outpath}")


# ──────────────────────────────────────────────────────────────────────────────
#  Final Plots
# ──────────────────────────────────────────────────────────────────────────────

def plot_final_gbe_cdf(
    gbe_values: np.ndarray,
    output_dir: Path,
    stem: str,
    mode: str,
    log: logging.Logger,
) -> None:
    """
    CDF of per-pixel GBE values (TJ-excluded, fully filtered).

    Parameters
    ----------
    gbe_values : (N,) array of GBE values at kept boundary pixels
    output_dir : output directory
    stem       : filename prefix
    mode       : GBE mode string — used in labels
    log        : logger
    """
    import matplotlib.pyplot as plt

    sorted_vals = np.sort(gbe_values)
    cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(sorted_vals, cdf, linewidth=2, color="steelblue")
    ax.set_xlabel("GBE", fontsize=12)
    ax.set_ylabel("Cumulative Fraction", fontsize=12)
    ax.set_title(
        f"GBE CDF — mode={mode}\n"
        f"N={len(gbe_values)} pixels  "
        f"(TJ-excluded)",
        fontsize=11,
    )
    ax.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()

    outpath = output_dir / f"{stem}_final_gbe_cdf.png"
    fig.savefig(outpath, dpi=300, transparent=True)
    plt.close(fig)
    log.warning(f"Final GBE CDF saved: {outpath}")


def plot_final_gbe_heatmap(
    frame: FrameData,
    pixel_coords: np.ndarray,
    gbe_values: np.ndarray,
    tj_excluded: set,
    mode: str,
    output_dir: Path,
    stem: str,
    log: logging.Logger,
) -> None:
    """
    Heatmap of per-pixel GBE with TJ exclusion zone centres overlaid.

    TJ exclusion zone centre pixels (junction_pixels from the frame) are
    plotted as cyan dots so the reader can see which regions were masked.

    Parameters
    ----------
    frame        : FrameData (last frame)
    pixel_coords : (N, 2) kept boundary pixel coordinates
    gbe_values   : (N,)   GBE values at those pixels
    tj_excluded  : pre-built TJ proximity set (used for the mask label)
    mode         : GBE mode string
    output_dir   : output directory
    stem         : filename prefix
    log          : logger
    """
    import matplotlib.pyplot as plt

    C0 = frame.C[0]
    nx, ny = C0.shape

    gbe_map = np.full((nx, ny), np.nan, dtype=np.float64)
    for (i, j), val in zip(pixel_coords, gbe_values):
        gbe_map[i, j] = val

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.imshow(C0, origin="lower", cmap="gray", alpha=0.25,
              interpolation="nearest")
    im = ax.imshow(
        gbe_map,
        origin="lower",
        cmap="plasma",
        interpolation="nearest",
        vmin=float(np.nanmin(gbe_values)),
        vmax=float(np.nanmax(gbe_values)),
    )
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("GBE", fontsize=10)

    if len(frame.junction_pixels) > 0:
        jp = frame.junction_pixels
        ax.scatter(
            jp[:, 1], jp[:, 0],
            c="cyan", s=4, linewidths=0,
            label=f"TJ centres ({len(jp)} px)",
            zorder=3,
        )
        ax.legend(loc="upper right", fontsize=7, markerscale=3)

    ax.set_title(
        f"GBE Heatmap — mode={mode}  |  TJ-excluded\n"
        f"N={len(gbe_values)} pixels  "
        f"min={np.nanmin(gbe_values):.4f}  "
        f"mean={np.nanmean(gbe_values):.4f}  "
        f"max={np.nanmax(gbe_values):.4f}",
        fontsize=10,
    )
    ax.set_xlabel("j (x)")
    ax.set_ylabel("i (y)")
    plt.tight_layout()

    outpath = output_dir / f"{stem}_final_gbe_heatmap.png"
    fig.savefig(outpath, dpi=300, transparent=True)
    plt.close(fig)
    log.warning(f"Final GBE heatmap saved: {outpath}")


def plot_final_inclination_polar(
    theta_closed: np.ndarray,
    r_closed: np.ndarray,
    n_pixels: int,
    output_dir: Path,
    stem: str,
    log: logging.Logger,
) -> None:
    """
    Polar plot of the inclination angle distribution (TJ-excluded).

    Parameters
    ----------
    theta_closed : (B+1,) bin centres in radians, loop-closed
    r_closed     : (B+1,) normalised frequency, loop-closed
    n_pixels     : number of pixels used
    output_dir   : output directory
    stem         : filename prefix
    log          : logger
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(6, 6))
    ax.plot(theta_closed, r_closed, linewidth=2, color="steelblue",
            label="Inclination PDF")
    ax.fill(theta_closed, r_closed, alpha=0.15, color="steelblue")

    ax.set_rgrids(np.arange(0, 0.01, 0.004))
    ax.set_rlabel_position(0.0)
    ax.set_rlim(0.0, 0.01)
    ax.set_yticklabels(["0", "4e-3", "8e-3"], fontsize=8)
    ax.set_title(
        f"Inclination Distribution\nTJ-excluded  |  N={n_pixels} pixels",
        fontsize=10, pad=25,
    )
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8)
    plt.tight_layout()

    outpath = output_dir / f"{stem}_final_inclination_polar.png"
    fig.savefig(outpath, dpi=300, transparent=True)
    plt.close(fig)
    log.warning(f"Final inclination polar plot saved: {outpath}")


def plot_final_gbe_histograms_by_curvature(
    frame: FrameData,
    pixel_coords: np.ndarray,
    gbe_values: np.ndarray,
    output_dir: Path,
    stem: str,
    mode: str,
    log: logging.Logger,
) -> None:
    """
    Two side-by-side histograms of per-pixel GBE split by the sign of the
    local curvature at each boundary pixel.

    Pixels where curvature > 0  → "normal curvature" (blue)
    Pixels where curvature <= 0 → "anti-curvature"   (orange)

    The curvature value used is C[1] from the last frame, evaluated at the
    same kept boundary pixels that were used for GBE.

    Parameters
    ----------
    frame        : FrameData (last frame) — supplies C[1] curvature field
    pixel_coords : (N, 2) kept boundary pixel coordinates
    gbe_values   : (N,)   GBE values at those pixels
    output_dir   : output directory
    stem         : filename prefix
    mode         : GBE mode string — used in title
    log          : logger
    """
    import matplotlib.pyplot as plt

    curv_field = frame.C[1]
    curvatures_at_pixels = np.array(
        [curv_field[i, j] for (i, j) in pixel_coords], dtype=np.float64
    )

    normc_mask = curvatures_at_pixels > 0.0
    antic_mask = ~normc_mask

    gbe_normc = gbe_values[normc_mask]
    gbe_antic = gbe_values[antic_mask]

    all_vals = gbe_values
    bins = np.linspace(float(all_vals.min()), float(all_vals.max()), 50)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)

    for ax, gbe_sub, label, color, n in [
        (axes[0], gbe_normc, "Normal curvature (κ > 0)", "steelblue", normc_mask.sum()),
        (axes[1], gbe_antic, "Anti-curvature (κ ≤ 0)",   "darkorange", antic_mask.sum()),
    ]:
        if len(gbe_sub) > 0:
            ax.hist(gbe_sub, bins=bins, color=color, alpha=0.85, edgecolor="none")
            ax.axvline(float(np.mean(gbe_sub)), color="black", linewidth=1.5,
                       linestyle="--", label=f"mean={np.mean(gbe_sub):.4f}")
            ax.legend(fontsize=9)
        ax.set_xlabel("GBE", fontsize=11)
        ax.set_ylabel("Pixel count", fontsize=11)
        ax.set_title(f"{label}\nN={n} pixels", fontsize=10)

    fig.suptitle(
        f"GBE Distribution by Curvature Sign — mode={mode}  |  TJ-excluded",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()

    outpath = output_dir / f"{stem}_final_gbe_hist_by_curvature.png"
    fig.savefig(outpath, dpi=300, transparent=True)
    plt.close(fig)
    log.warning(f"Final GBE histograms saved: {outpath}")


def plot_final_velocity_scatter(
    normc_flat_conf: dict,
    antic_flat_conf: dict,
    output_dir: Path,
    stem: str,
    curvature_limit: float,
    x_lim: tuple,
    log: logging.Logger,
) -> None:
    """
    Confidence-filtered velocity vs |κ| scatter plot.

    Mirrors debug Panel 4 as a clean, standalone figure.
    Normal-curvature GBs in blue, anti-curvature in orange.

    Parameters
    ----------
    normc_flat_conf : confidence-filtered normal-curvature flat dict
    antic_flat_conf : confidence-filtered anti-curvature flat dict
    output_dir      : output directory
    stem            : filename prefix
    curvature_limit : min_curvature threshold — shown as reference line
    x_lim           : (x_min, x_max) for the curvature axis
    log             : logger
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.axhline(0, color="gray", linewidth=0.8, linestyle=":")
    ax.plot([curvature_limit, x_lim[1]], [0, 0], "-", color="grey",
            linewidth=1.5)

    ax.scatter(
        normc_flat_conf["curvatures"], normc_flat_conf["velocities"],
        s=6, alpha=0.6, color="C0",
        label=f"Normal-c ({len(normc_flat_conf['curvatures'])})",
    )
    ax.scatter(
        antic_flat_conf["curvatures"], antic_flat_conf["velocities"],
        s=10, alpha=0.6, color="C1",
        label=f"Anti-c ({len(antic_flat_conf['curvatures'])})",
    )

    ax.set_xlim([max(0.0, curvature_limit * 0.95), x_lim[1]])
    ax.set_xlabel("|κ| (px⁻¹)", fontsize=12)
    ax.set_ylabel("Velocity (px/s)", fontsize=12)
    ax.set_title(
        "Velocity vs Curvature — Confidence-Filtered (99%)\n"
        "Normal-c (blue) vs Anti-c (orange)  |  TJ-excluded",
        fontsize=11,
    )
    ax.legend(fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.4)
    plt.tight_layout()

    outpath = output_dir / f"{stem}_final_velocity_scatter.png"
    fig.savefig(outpath, dpi=300, transparent=True)
    plt.close(fig)
    log.warning(f"Final velocity scatter saved: {outpath}")


def plot_final_velocity_density_with_fits(
    normc_flat: dict,
    antic_flat: dict,
    output_dir: Path,
    stem: str,
    curvature_limit: float,
    x_lim: tuple,
    bin_interval: float,
    log: logging.Logger,
    filtered: bool,
) -> None:
    """
    Density contour plot of velocity vs |κ| with binned mean ± std and
    linear fits overlaid.

    Mirrors debug Panel 3 as a clean, standalone figure using the
    post-sign-convention, pre-confidence data (all normc + antic combined).

    Parameters
    ----------
    normc_flat      : sign-convention-corrected normal-curvature flat dict
    antic_flat      : sign-convention-corrected anti-curvature flat dict
    output_dir      : output directory
    stem            : filename prefix
    curvature_limit : min_curvature threshold line
    x_lim           : (x_min, x_max) for the curvature axis
    bin_interval    : bin width for compute_velocity_bins
    log             : logger
    """
    import matplotlib.pyplot as plt

    all_c = normc_flat["curvatures"] + antic_flat["curvatures"]
    all_v = normc_flat["velocities"] + antic_flat["velocities"]

    if len(all_c) < 2:
        log.warning("plot_final_velocity_density_with_fits: insufficient data, skipping.")
        return

    fig, ax = plt.subplots(figsize=(8, 6))

    # ── Density contour ──────────────────────────────────────────────────────
    x_bins = np.linspace(x_lim[0], x_lim[1], 40)
    y_abs  = max(abs(v) for v in all_v) * 1.1
    y_bins = np.linspace(-y_abs, y_abs, 40)

    hist, x_edges, y_edges = np.histogram2d(all_c, all_v,
                                             bins=[x_bins, y_bins])
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2
    X, Y = np.meshgrid(x_centers, y_centers)
    hist_log = np.full_like(hist.T, 0, dtype=float) #was np.nan not 1 meant no background
    nz = hist.T > 0
    hist_log[nz] = np.log10(hist.T[nz])
    cf = ax.contourf(X, Y, hist_log, levels=20,
                     cmap="coolwarm", alpha=0.9, vmin=0)
    ax.contour(X, Y, hist_log, levels=20,
               cmap="gray", alpha=0.15, vmin=0)
    cbar = plt.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r"log$_{10}$(count)", fontsize=9)

    # ── Binned mean ± std ────────────────────────────────────────────────────
    bins = compute_velocity_bins(all_c, all_v, x_lim=x_lim,
                                 bin_interval=bin_interval, min_count=10)
    valid = bins["valid_mask"]
    if valid.any():
        ax.errorbar(
            bins["bin_centers"][valid],
            bins["bin_means"][valid],
            yerr=bins["bin_stds"][valid],
            fmt="o", color="k", linewidth=1, capsize=2,
            ecolor="black", markersize=4, label="Bin mean ± std",
        )

        # Linear fit — all valid bins
        x_all = bins["bin_centers"][valid]
        y_all = bins["bin_means"][valid]
        p_all = np.polyfit(x_all, y_all, 1)
        y_pred_all = np.polyval(p_all, x_all)
        ss_res = np.sum((y_all - y_pred_all) ** 2)
        ss_tot = np.sum((y_all - y_all.mean()) ** 2)
        r2_all = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        ax.plot(x_all, y_pred_all, "-", color="C1", linewidth=2,
                label=rf"All bins  R²={r2_all:.3f}  M*={p_all[0]:.4f}")
        log.warning(f"Final density fit (all):    slope={p_all[0]:.4f}  R²={r2_all:.4f}")

        # Linear fit — low-κ bins (κ < 0.03)
        mask_sub = x_all < 0.03
        if mask_sub.sum() > 1:
            x_sub = x_all[mask_sub]
            y_sub = y_all[mask_sub]
            p_sub = np.polyfit(x_sub, y_sub, 1)
            y_pred_sub = np.polyval(p_sub, x_sub)
            ss_res_s = np.sum((y_sub - y_pred_sub) ** 2)
            ss_tot_s = np.sum((y_sub - y_sub.mean()) ** 2)
            r2_sub = 1.0 - ss_res_s / ss_tot_s if ss_tot_s > 0 else float("nan")
            ax.plot(x_sub, y_pred_sub, "-", color="C2", linewidth=2,
                    label=rf"κ<0.03  R²={r2_sub:.3f}  M*={p_sub[0]:.4f}")
            log.warning(f"Final density fit (κ<0.03): slope={p_sub[0]:.4f}  R²={r2_sub:.4f}")

    ax.axhline(0, color="gray", linewidth=0.8, linestyle=":")
    ax.plot([curvature_limit, x_lim[1]], [0, 0], "-", color="grey",
            linewidth=1.5)
    ax.set_xlim([curvature_limit, x_lim[1]])
    ax.set_xlabel("|κ| (px⁻¹)", fontsize=12)
    ax.set_ylabel("Velocity (px/s)", fontsize=12)
    if filtered:
        ax.set_title(
            "Velocity vs Curvature — Density + Binned Fits\n"
            "TJ-excluded  |  post-confidence filter",
            fontsize=11,
        )
    else:
        ax.set_title(
            "Velocity vs Curvature — Density + Binned Fits\n"
            "TJ-excluded  |  pre-confidence filter",
            fontsize=11,
        )
    ax.legend(fontsize=9, loc="lower right")
    plt.tight_layout()
    filname = "" if filtered else "un"
    outpath = output_dir / f"{stem}_final_{filname}filtered_velocity_density.png"
    fig.savefig(outpath, dpi=300, transparent=True)
    plt.close(fig)
    log.warning(f"Final velocity density plot saved: {outpath}")





# ──────────────────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    ti = time.perf_counter()
    args = parse_args()
    log  = setup_logging(args.verbose)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"Arguments: {args}")

    # ── 1. Load HDF5 ──────────────────────────────────────────────────────
    t0 = time.perf_counter()
    frames = load_hdf5_frames(args.hdf5, log)
    tf(t0, log, "HDF5 load: ")

    raw_last_frame = get_last_frame(frames)
    log.warning(
        f"Using raw last frame: step={raw_last_frame.step}, time={raw_last_frame.time:.6g}, "
        f"boundary_pixels={len(raw_last_frame.boundary_pixels)}, "
        f"junction_pixels={len(raw_last_frame.junction_pixels)}, "
        f"gb_pairs={len(raw_last_frame.gb_dict)}"
    )

    # ── 1b. Apply canonical GB-level filters to every frame ────────────────
    # HDF5 gb_dict entries are already TJ-excluded during HDF5 generation.
    # The filters below remove whole ij pairs based on TJ-filtered GB area
    # and TJ-filtered average curvature.
    t0 = time.perf_counter()
    frames_filtered = filter_frames_gb_dicts(
        frames,
        min_area=args.min_area,
        min_curvature=args.min_curvature,
        log=log,
    )
    tf(t0, log, "GB-level filtering: ")
    last_frame = get_last_frame(frames_filtered)
    log.warning(
        f"Using filtered last frame: step={last_frame.step}, time={last_frame.time:.6g}, "
        f"gb_pairs={len(last_frame.gb_dict)} after min_area/min_curvature."
    )

    # ── 2. Load misorientation parquet (if needed) ─────────────────────────
    misorientation_data: dict = {}
    if args.gbe_mode in ("misorientation", "both"):
        neighbor_pairs = set(last_frame.gb_dict.keys())
        t0 = time.perf_counter()
        misorientation_data = load_misorientation_parquet(
            args.parquet, neighbor_pairs, log
        )
        tf(t0, log, "Parquet load: ")

    # ── 3. Build TJ exclusion set ONCE — shared by all per-pixel calculations ──
    tj_excluded: set[tuple[int, int]] = set()
    if args.tj_exclude and len(raw_last_frame.junction_pixels) > 0:
        t0 = time.perf_counter()
        tj_excluded = build_tj_proximity_set(
            raw_last_frame.junction_pixels, args.tj_distance
        )
        tf(t0, log, "TJ exclusion set: ")
        log.warning(
            f"TJ exclusion: {len(tj_excluded)} coords excluded "
            f"(radius={args.tj_distance}px, "
            f"from {len(raw_last_frame.junction_pixels)} junction pixels)"
        )

    log.warning(
        "Filter convention: --min-area is interpreted as TJ-filtered GB pixel area; "
        "--min-curvature is interpreted as abs(TJ-filtered average GB curvature). "
        "For velocity, --tj-distance is governed by the HDF5 gb_dict; regenerate the HDF5 "
        "with a different --tj-distance to change the velocity TJ filtering."
    )

    # ── 4. Compute GBE per pixel ──────────────────────────────────────────
    t0 = time.perf_counter()
    pixel_coords, gbe_values = compute_gbe_per_pixel(
        frame                  = last_frame,
        mode                   = args.gbe_mode,
        tj_excluded            = tj_excluded,
        valid_gb_dict          = last_frame.gb_dict,
        misorientation_data    = misorientation_data,
        inclination_anisotropy = args.inclination_anisotropy,
        log                    = log,
    )
    tf(t0, log, "GBE calculation: ")

    log.info(
        f"GBE result: {len(gbe_values)} pixels, "
        f"min={gbe_values.min():.4f}, max={gbe_values.max():.4f}, "
        f"mean={gbe_values.mean():.4f}"
    )

    if args.debug_plot:
        stem = args.hdf5.stem
        plot_gbe_debug(
            frame       = last_frame,
            pixel_coords= pixel_coords,
            gbe_values  = gbe_values,
            mode        = args.gbe_mode,
            tj_exclude  = args.tj_exclude,
            output_dir  = args.output_dir,
            stem        = stem,
        )
        plot_curvature_debug(
            frame       = last_frame,
            tj_excluded = tj_excluded,
            tj_exclude  = args.tj_exclude,
            output_dir  = args.output_dir,
            stem        = stem,
            log         = log,
        )

    # ── 5. Compute inclination per pixel ──────────────────────────────────
    t0 = time.perf_counter()
    inc_coords, inc_angles = compute_inclination_per_pixel(
        frame         = last_frame,
        tj_excluded   = tj_excluded,
        valid_gb_dict = last_frame.gb_dict,
        log           = log,
    )
    tf(t0, log, "Inclination calculation: ")

    theta_closed, r_closed, bin_centers_deg, freq = compute_inclination_distribution(
        inc_angles
    )

    if args.inclination_csv:
        stem = args.hdf5.stem
        save_inclination_csv(bin_centers_deg, freq, args.output_dir, stem, log)

    if args.debug_plot:
        stem = args.hdf5.stem
        plot_inclination_polar(
            theta_closed = theta_closed,
            r_closed     = r_closed,
            output_dir   = args.output_dir,
            stem         = stem,
            tj_exclude   = args.tj_exclude,
            n_pixels     = len(inc_angles),
            log          = log,
        )


        # ── 6. Compute velocity ───────────────────────────────────────────────
    if not args.skip_velocity:
        t0 = time.perf_counter()

        flat_lists = accumulate_velocity_flat_lists(
            frames       = frames_filtered,
            log          = log,
        )
        vtf(t0, log, "Velocity flat accumulation: ")

        t02 = time.perf_counter()
        velocity_df = compute_gb_velocity_averaged(
            frames       = frames_filtered,
            log          = log,
        )
        vtf(t02, log, "Velocity averaged (secondary): ")
        tf(t0,  log,  "Velocity calculation: ")

        # ── 6.1 Sign convention ─────────────────────────────────────────────────
        (flat_lists["all_curvatures"],
        flat_lists["all_velocities"],
        flat_lists["all_dV_forward"],
        flat_lists["all_dV_backward"]) = apply_curvature_sign_convention(
            flat_lists["all_curvatures"],
            flat_lists["all_velocities"],
            flat_lists["all_dV_forward"],
            flat_lists["all_dV_backward"],
        )
        summarize_velocity_input(flat_lists, log, label="Velocity plot input after GB filters")

        # ── 6.2 Sliding window filter (optional, --drop-isolated-antic) ─────────
        isolated_antic = find_isolated_anticurvature_events(
            flat_lists["all_interval_results"], log=log
        )

        if args.drop_isolated_antic and isolated_antic:
            # Build a set of flat-list indices to drop
            rejected_flat_indices = {
                i for i, (pid, t) in enumerate(
                    zip(flat_lists["all_pair_ids"], flat_lists["all_timestep_indices"])
                )
                if (pid, t) in isolated_antic
            }
            log.warning(
                f"Sliding window: dropping {len(rejected_flat_indices)} isolated "
                f"anti-curvature entries from flat lists."
            )
            keep_indices = [
                i for i in range(len(flat_lists["all_velocities"]))
                if i not in rejected_flat_indices
            ]
            for key in ("all_curvatures", "all_velocities", "all_dV_forward",
                        "all_dV_backward", "all_areas", "all_pair_ids",
                        "all_timestep_indices"):
                flat_lists[key] = [flat_lists[key][i] for i in keep_indices]
        else:
            if args.drop_isolated_antic:
                log.warning("Sliding window filter enabled but returned no rejections.")

        # ── 6.3 Split into normc / antic ────────────────────────────────────────
        normc_indices = [
            i for i, v in enumerate(flat_lists["all_velocities"]) if v >= 0.0
        ]
        antic_indices = [
            i for i, v in enumerate(flat_lists["all_velocities"]) if v < 0.0
        ]

        def _extract(flat, indices):
            return {
                "curvatures": [flat["all_curvatures"][i]  for i in indices],
                "velocities": [flat["all_velocities"][i]  for i in indices],
                "dV_forward": [flat["all_dV_forward"][i]  for i in indices],
                "dV_backward":[flat["all_dV_backward"][i] for i in indices],
                "areas":      [flat["all_areas"][i]        for i in indices],
                "pair_ids":   [flat["all_pair_ids"][i]     for i in indices],
            }

        normc_flat = _extract(flat_lists, normc_indices)
        antic_flat = _extract(flat_lists, antic_indices)

        # ── 6.4 Confidence filtering ────────────────────────────────────────────
        CONFIDENCE = args.antic_confidence

        normc_flat_conf = apply_confidence_filter(
            **{k: normc_flat[k] for k in
            ("curvatures","velocities","dV_forward","dV_backward","areas","pair_ids")},
            mode="normc", confidence=CONFIDENCE, log=log,
        )
        antic_flat_conf = apply_confidence_filter(
            **{k: antic_flat[k] for k in
            ("curvatures","velocities","dV_forward","dV_backward","areas","pair_ids")},
            mode="antic", confidence=CONFIDENCE, log=log,
        )
    else:
        flat_lists  = None
        velocity_df = None



    if args.debug_plot:
        if flat_lists is not None and len(flat_lists["all_velocities"]) > 0:
            stem = args.hdf5.stem
            plot_velocity_debug(
                flat_lists      = flat_lists,
                normc_flat      = normc_flat,
                antic_flat      = antic_flat,
                normc_flat_conf = normc_flat_conf,
                antic_flat_conf = antic_flat_conf,
                last_frame      = last_frame,
                frames          = frames_filtered,
                velocity_df     = velocity_df,
                output_dir      = args.output_dir,
                stem            = stem,
                tj_exclude      = args.tj_exclude,
                curvature_limit = args.min_curvature,
                log             = log,
            )
        else:
            log.warning("Skipping velocity debug plot — no velocity data.")



    # ── 7. Final individual plots (--plot) ───────────────────────────────────
    if args.plot:
        stem = args.hdf5.stem

        # 7.1  GBE CDF
        plot_final_gbe_cdf(
            gbe_values = gbe_values,
            output_dir = args.output_dir,
            stem       = stem,
            mode       = args.gbe_mode,
            log        = log,
        )

        # 7.2  GBE heatmap with TJ exclusion centres
        plot_final_gbe_heatmap(
            frame        = last_frame,
            pixel_coords = pixel_coords,
            gbe_values   = gbe_values,
            tj_excluded  = tj_excluded,
            mode         = args.gbe_mode,
            output_dir   = args.output_dir,
            stem         = stem,
            log          = log,
        )

        # 7.3  Inclination polar plot
        plot_final_inclination_polar(
            theta_closed = theta_closed,
            r_closed     = r_closed,
            n_pixels     = len(inc_angles),
            output_dir   = args.output_dir,
            stem         = stem,
            log          = log,
        )

        # 7.4  GBE histograms split by curvature sign
        plot_final_gbe_histograms_by_curvature(
            frame        = last_frame,
            pixel_coords = pixel_coords,
            gbe_values   = gbe_values,
            output_dir   = args.output_dir,
            stem         = stem,
            mode         = args.gbe_mode,
            log          = log,
        )

        # 7.5 & 7.6  Velocity plots — only if velocity was computed
        if flat_lists is not None and len(flat_lists["all_velocities"]) > 0:

            # 7.5  Confidence-filtered scatter
            plot_final_velocity_scatter(
                normc_flat_conf = normc_flat_conf,
                antic_flat_conf = antic_flat_conf,
                output_dir      = args.output_dir,
                stem            = stem,
                curvature_limit = args.min_curvature,
                x_lim           = (0.0, 0.1),
                log             = log,
            )

            # 7.6  Density + fits (pre-confidence, post sign convention)
            plot_final_velocity_density_with_fits(
                normc_flat      = normc_flat,
                antic_flat      = antic_flat,
                output_dir      = args.output_dir,
                stem            = stem,
                curvature_limit = args.min_curvature,
                x_lim           = (0.0, 0.1),
                bin_interval    = 0.002,
                filtered        = False,
                log             = log,
            )
            # 7.6  Density + fits (POST-confidence, post sign convention)
            plot_final_velocity_density_with_fits(
                normc_flat      = normc_flat_conf,
                antic_flat      = antic_flat_conf,
                output_dir      = args.output_dir,
                stem            = stem,
                curvature_limit = args.min_curvature,
                x_lim           = (0.0, 0.1),
                bin_interval    = 0.002,
                filtered        = True,
                log             = log,
            )
        else:
            log.warning("Skipping final velocity plots — no velocity data available.")

    tf(ti, log, "Total: ")


if __name__ == "__main__":
    main()
