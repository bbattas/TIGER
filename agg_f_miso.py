from __future__ import annotations
from vector.ExodusBasics import ExodusBasics

import time
import numpy as np
import pandas as pd
import argparse
import sys
import logging
import h5py
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
from tqdm import tqdm
from scipy.spatial import cKDTree


# ─────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────

THETA_MAX_DEG = 62.0
THETA_MAX_RAD = np.deg2rad(THETA_MAX_DEG)


# ─────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Grain-boundary f_miso energy analysis from unique_grains Exodus variable. "
            "Requires a CSV of grain-pair quaternion misorientations with columns: "
            "i, j, angle_deg, ax_x, ax_y, ax_z  (misorientation axis in the fundamental "
            "zone of [100]/[110]/[111])."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---- General ----
    gen = p.add_argument_group("General")
    gen.add_argument("-v", "--verbose", action="count", default=0,
                     help="Increase verbosity (-v, -vv, -vvv).")
    gen.add_argument("-s", "--subdirs", action="store_true",
                     help="Search for *.e files one level down.")
    gen.add_argument("--misorientation", "-m", type=str, required=True,
                     help="Partial path / glob pattern to locate the misorientation CSV.")

    # ---- Target frame selection ----
    tim = p.add_argument_group("Target frame selection (choose one)")
    grp = tim.add_mutually_exclusive_group(required=True)
    grp.add_argument("-g", "--grains", type=int,
                     help="Target grain count (grain_tracker global variable).")
    grp.add_argument("-t", "--time", type=float,
                     help="Target simulation time.")
    grp.add_argument("--full", action="store_true",
                     help="Use the final timestep.")

    # ---- Timestepping ----
    out = p.add_argument_group("Timestepping")
    out.add_argument("--frame-mode",
                     choices=("single", "all", "sequence"),
                     default="single",
                     help=(
                         "'single' = target frame only, "
                         "'all' = every frame up to target, "
                         "'sequence' = evenly spaced up to target."
                     ))
    out.add_argument("--nframes", "-f", type=int, default=40,
                     help="Number of frames for --frame-mode sequence.")

    # ---- Neighbor detection ----
    nbr = p.add_argument_group("Neighbor detection")
    nbr.add_argument("--neighbor-method",
                     choices=("kdtree", "connectivity"),
                     default="connectivity",
                     help=(
                         "'connectivity' uses shared element edges (exact, preferred for "
                         "structured meshes). 'kdtree' falls back to spatial proximity."
                     ))
    nbr.add_argument("--neighbor-radius", type=float, default=None,
                     help="Search radius for kdtree method. Defaults to 1.5 * median spacing.")

    # ---- Output ----
    ogrp = p.add_argument_group("Output")
    ogrp.add_argument("--hdf5", action="store_true",
                      help="Write an HDF5 file of f_miso values per frame.")
    ogrp.add_argument("--plot", action="store_true",
                      help="Save overlay plot of grain map + f_miso-colored GB points.")
    ogrp.add_argument("--view", action="store_true",
                      help="Show plots interactively instead of saving.")
    ogrp.add_argument("--dpi", type=int, default=300)
    ogrp.add_argument("--minimal", action="store_true",
                      help="No axes, no colorbar, no title in plots.")
    ogrp.add_argument("--no-axes", action="store_true")
    ogrp.add_argument("--no-colorbar", action="store_true")
    ogrp.add_argument("--no-title", action="store_true")
    ogrp.add_argument("--cmap", type=str, default="plasma",
                      help="Colormap for f_miso overlay.")
    ogrp.add_argument("--gb-lw", type=float, default=0.6,
                      help="Marker size scale for GB points in overlay plot.")

    args = p.parse_args()

    if args.minimal:
        args.no_axes = True
        args.no_colorbar = True
        args.no_title = True

    return args


# ─────────────────────────────────────────────
#  Logging / timing
# ─────────────────────────────────────────────

def setup_logging(verbosity: int) -> logging.Logger:
    level = logging.WARNING
    if verbosity >= 2:
        level = logging.DEBUG
    elif verbosity >= 1:
        level = logging.INFO
    logging.basicConfig(level=level, format="%(message)s")
    return logging.getLogger("FMISO")


def tf(ti, log, extra=None):
    tag = (extra or "") + f"Time: {(time.perf_counter() - ti):.4}s"
    log.warning(tag)


def vtf(ti, log, extra=None):
    tag = (extra or "") + f"Time: {(time.perf_counter() - ti):.4}s"
    log.info(tag)


# ─────────────────────────────────────────────
#  File helpers
# ─────────────────────────────────────────────

def find_exodus_files(*, subdirs: bool = False, pattern: str = "*.e") -> list[Path]:
    cwd = Path.cwd()
    files = sorted(cwd.glob(f"*/{pattern}") if subdirs else cwd.glob(pattern))
    return [p for p in files if p.is_file()]


def exodus_stem(exo_path: Path) -> str:
    name = exo_path.name
    if name.endswith(".e"):
        name = name[:-2]
    if name.endswith("_out"):
        name = name[:-4]
    return name


def closest_index(values: np.ndarray, target: float) -> int:
    return int(np.argmin(np.abs(np.asarray(values) - target)))


# ─────────────────────────────────────────────
#  Step selection
# ─────────────────────────────────────────────

def select_step(exo, *, grains, time_value, full, log) -> int:
    times = exo.time()
    if full:
        step = len(times) - 1
        log.info(f"Final step={step}, time={times[step]}")
        return step
    if time_value is not None:
        step = closest_index(times, float(time_value))
        log.info(f"Time step={step}, time={times[step]}")
        return step
    glo_names = exo.glo_varnames()
    if "grain_tracker" not in glo_names:
        raise RuntimeError(f"No 'grain_tracker' global var. Available: {glo_names}")
    gt = exo.glo_var_series("grain_tracker")
    gt_counts = np.rint(gt).astype(np.int64)
    step = closest_index(gt_counts, int(grains))
    log.info(f"Grain step={step}, grain_tracker={gt_counts[step]}")
    return step


def select_steps_by_time(times, target_step, nframes) -> list[int]:
    times = np.asarray(times)
    if nframes == 1:
        return [target_step]
    target_times = np.linspace(float(times[0]), float(times[target_step]), nframes)
    return [closest_index(times[: target_step + 1], tt) for tt in target_times]


def select_steps(exo, *, grains, time_value, full, frame_mode, nframes, log) -> list[int]:
    times = exo.time()
    target = select_step(exo, grains=grains, time_value=time_value, full=full, log=log)
    if frame_mode == "single":
        steps = [target]
    elif frame_mode == "all":
        steps = list(range(target + 1))
    elif frame_mode == "sequence":
        steps = select_steps_by_time(times, target, nframes)
    else:
        raise ValueError(f"Unknown frame_mode: {frame_mode}")
    log.info(f"frame_mode={frame_mode}, target={target}, steps={steps}")
    return steps


# ─────────────────────────────────────────────
#  Misorientation CSV loader
#  Expected columns: i, j, angle_deg, ax_x, ax_y, ax_z
#  (misorientation axis already in the fundamental zone)
# ─────────────────────────────────────────────

def find_misorientation_csv(partial: str) -> Path:
    exact = Path(partial)
    if exact.is_file():
        return exact
    cwd = Path.cwd()
    pattern = partial if partial.endswith(".csv") else f"{partial}*.csv"
    candidates = sorted(cwd.glob(pattern)) + sorted(cwd.glob(f"*/{pattern}"))
    candidates = [p for p in candidates if p.is_file()]
    if not candidates:
        raise FileNotFoundError(
            f"No misorientation CSV found matching '{pattern}'. "
            f"Searched cwd and one level down."
        )
    return candidates[0]


def load_misorientation(csv_path: Path, log) -> dict[tuple[int, int], dict]:
    """
    Parse CSV with columns: i, j, angle_deg, ax_x, ax_y, ax_z.
    Returns dict keyed by canonical (min(i,j), max(i,j)) ->
        {"angle_deg": float, "ax_x": float, "ax_y": float, "ax_z": float}
    """
    log.info(f"Loading misorientation CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]

    required = {"i", "j", "angle_deg", "ax_x", "ax_y", "ax_z"}
    missing_cols = required - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"Misorientation CSV missing required columns: {missing_cols}. "
            f"Found: {list(df.columns)}"
        )

    result = {}
    for _, row in df.iterrows():
        i_val = int(row["i"])
        j_val = int(row["j"])
        key = (min(i_val, j_val), max(i_val, j_val))
        result[key] = {
            "angle_deg": float(row["angle_deg"]),
            "ax_x":      float(row["ax_x"]),
            "ax_y":      float(row["ax_y"]),
            "ax_z":      float(row["ax_z"]),
        }

    log.info(f"  Loaded {len(result)} i-j misorientation pairs.")
    return result


# ─────────────────────────────────────────────
#  f_miso energy calculation
# ─────────────────────────────────────────────

def compute_fmiso_single(angle_deg: float, ax_x: float, ax_y: float, ax_z: float) -> float:
    """
    Compute the grain-boundary energy function f_miso for a single GB pair.

    Steps:
      1. Normalize the misorientation axis (ax_x, ax_y, ax_z).
      2. polar   = acos(ax_z)
         azimuth = atan2(ax_y, ax_x), remapped to [0, 2π)
      3. ang_energy = (θ/62°) * (1 - ln(θ/62°)),  clamped to [0, 1]
         ax_energy  = |cos(polar)|^0.4 + |cos(azimuth/2)|^0.4, clamped to [0, 1]
      4. f_miso = 0.3 + 0.7 * (ang_energy * ax_energy)

    Returns f_miso in [0.3, ~1.0].
    """
    # --- normalize axis ---
    norm = np.sqrt(ax_x**2 + ax_y**2 + ax_z**2)
    if norm < 1e-12:
        # Degenerate axis (zero rotation): assign minimum energy
        return 0.3
    ax_x /= norm
    ax_y /= norm
    ax_z /= norm

    # --- polar / azimuth ---
    # Clamp ax_z to [-1, 1] for numerical safety before acos
    polar   = np.arccos(np.clip(ax_z, -1.0, 1.0))
    azimuth = np.arctan2(ax_y, ax_x)
    if azimuth < 0.0:
        azimuth += 2.0 * np.pi          # remap (-π, π] → [0, 2π)

    # --- ang_energy ---
    theta_rad = np.deg2rad(angle_deg)
    ratio = theta_rad / THETA_MAX_RAD
    if ratio <= 0.0:
        ang_energy = 0.0
    elif ratio >= 1.0:
        ang_energy = 1.0                # already at max; ln(1) = 0 → value = 1
    else:
        ang_energy = ratio * (1.0 - np.log(ratio))
    ang_energy = min(ang_energy, 1.0)

    # --- ax_energy ---
    ax_energy = abs(np.cos(polar))**0.4 + abs(np.cos(azimuth / 2.0))**0.4
    ax_energy = min(ax_energy, 1.0)

    # --- f_miso ---
    f_miso = 0.3 + 0.7 * (ang_energy * ax_energy)
    return float(f_miso)


def compute_fmiso_for_pairs(
    neighbor_pairs: set[tuple[int, int]],
    misorientation: dict[tuple[int, int], dict],
    log,
) -> tuple[list[tuple[int, int]], np.ndarray, list[tuple[int, int]]]:
    """
    Compute f_miso for every neighbor pair found in the mesh.

    Returns
    -------
    pairs_found : list of (i, j) tuples that had data in the CSV
    fmiso_values : np.ndarray of f_miso values aligned with pairs_found
    missing     : list of (i, j) tuples absent from the CSV
    """
    pairs_found = []
    fmiso_list  = []
    missing     = []

    for pair in sorted(neighbor_pairs):
        data = misorientation.get(pair)
        if data is None:
            missing.append(pair)
            continue
        val = compute_fmiso_single(
            data["angle_deg"], data["ax_x"], data["ax_y"], data["ax_z"]
        )
        pairs_found.append(pair)
        fmiso_list.append(val)

    fmiso_values = np.array(fmiso_list, dtype=np.float64)

    log.info(
        f"  f_miso: computed={len(pairs_found)}, "
        f"missing_from_csv={len(missing)}, "
        f"mean={fmiso_values.mean():.4f} min={fmiso_values.min():.4f} "
        f"max={fmiso_values.max():.4f}"
        if len(fmiso_values) > 0 else "  f_miso: no pairs found."
    )
    if missing:
        log.debug(f"  Missing pairs (first 10): {missing[:10]}")

    return pairs_found, fmiso_values, missing


# ─────────────────────────────────────────────
#  Neighbor identification  (unchanged from original)
# ─────────────────────────────────────────────

def find_neighbors_connectivity(exo, step: int, eb: int = 1,
                                log=None) -> set[tuple[int, int]]:
    grain_ids = exo.elem_var_at_step("unique_grains", step, eb=eb)
    grain_ids = np.rint(grain_ids).astype(np.int64)
    conn = exo.connectivity(which=eb, zero_based=True)
    n_elem = conn.shape[0]

    max_node = int(conn.max()) + 1
    node_to_elems: list[list[int]] = [[] for _ in range(max_node)]
    for elem_idx in range(n_elem):
        for node in conn[elem_idx]:
            node_to_elems[node].append(elem_idx)

    pairs: set[tuple[int, int]] = set()
    for elem_idx in range(n_elem):
        g_i = int(grain_ids[elem_idx])
        for node in conn[elem_idx]:
            for nbr_idx in node_to_elems[node]:
                if nbr_idx == elem_idx:
                    continue
                g_j = int(grain_ids[nbr_idx])
                if g_i != g_j:
                    pairs.add((min(g_i, g_j), max(g_i, g_j)))

    if log:
        log.info(f"  Connectivity neighbor pairs found: {len(pairs)}")
    return pairs


def find_neighbors_kdtree(exo, step: int, eb: int = 1,
                          radius: float | None = None,
                          log=None) -> set[tuple[int, int]]:
    grain_ids = exo.elem_var_at_step("unique_grains", step, eb=eb)
    grain_ids = np.rint(grain_ids).astype(np.int64)
    xc, yc = exo.element_centers_xy(eb=eb, method="bbox", cache=True)
    pts = np.column_stack([xc, yc])

    if radius is None:
        tree_est = cKDTree(pts)
        dists, _ = tree_est.query(pts, k=2)
        median_spacing = float(np.median(dists[:, 1]))
        radius = 1.5 * median_spacing
        if log:
            log.info(f"  Auto kdtree radius = {radius:.4g}")

    tree = cKDTree(pts)
    pairs_idx = tree.query_pairs(radius)

    pairs: set[tuple[int, int]] = set()
    for a, b in pairs_idx:
        g_i, g_j = int(grain_ids[a]), int(grain_ids[b])
        if g_i != g_j:
            pairs.add((min(g_i, g_j), max(g_i, g_j)))

    if log:
        log.info(f"  KDTree neighbor pairs found: {len(pairs)}")
    return pairs


def find_neighbors(exo, step: int, method: str, eb: int = 1,
                   radius: float | None = None,
                   log=None) -> set[tuple[int, int]]:
    if method == "connectivity":
        return find_neighbors_connectivity(exo, step, eb=eb, log=log)
    elif method == "kdtree":
        return find_neighbors_kdtree(exo, step, eb=eb, radius=radius, log=log)
    else:
        raise ValueError(f"Unknown neighbor method: {method}")


# ─────────────────────────────────────────────
#  Structured grid helpers  (unchanged)
# ─────────────────────────────────────────────

def centers_to_edges(vals):
    vals = np.asarray(vals)
    if vals.size == 1:
        d = 0.5
        return np.array([vals[0] - d, vals[0] + d])
    mids = 0.5 * (vals[:-1] + vals[1:])
    first = vals[0] - 0.5 * (vals[1] - vals[0])
    last  = vals[-1] + 0.5 * (vals[-1] - vals[-2])
    return np.concatenate(([first], mids, [last]))


def build_structured_grid(x, y, c):
    xu, yu = np.unique(x), np.unique(y)
    nx, ny = len(xu), len(yu)
    if nx * ny != len(c):
        raise ValueError("Not a full structured grid.")
    ix = np.searchsorted(xu, x)
    iy = np.searchsorted(yu, y)
    C = np.full((ny, nx), np.nan)
    C[iy, ix] = c
    if np.isnan(C).any():
        raise ValueError("Grid has missing cells.")
    return centers_to_edges(xu), centers_to_edges(yu), C


# ─────────────────────────────────────────────
#  GB interface-point geometry  (unchanged)
# ─────────────────────────────────────────────

def boundary_interface_points_with_fmiso(
    pairs: list[tuple[int, int]],
    fmiso_values: np.ndarray,
    grain_ids_all: np.ndarray,
    xc: np.ndarray,
    yc: np.ndarray,
    radius: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    For each grain pair, find element-center points from either grain that lie
    within `radius` of the other grain's elements.

    Returns (x_pts, y_pts, fmiso_pts) where fmiso_pts holds the f_miso value
    replicated for every interface point belonging to that pair.
    """
    pts = np.column_stack([xc, yc])

    xs, ys, fs = [], [], []
    for (g_i, g_j), fval in zip(pairs, fmiso_values):
        mask_i = grain_ids_all == g_i
        mask_j = grain_ids_all == g_j
        pts_i = pts[mask_i]
        pts_j = pts[mask_j]
        if len(pts_i) == 0 or len(pts_j) == 0:
            continue

        tree_j = cKDTree(pts_j)
        dists, _ = tree_j.query(pts_i)
        border_i = pts_i[dists <= radius]

        tree_i = cKDTree(pts_i)
        dists2, _ = tree_i.query(pts_j)
        border_j = pts_j[dists2 <= radius]

        if len(border_i) and len(border_j):
            border = np.vstack([border_i, border_j])
        elif len(border_i):
            border = border_i
        elif len(border_j):
            border = border_j
        else:
            continue

        n_pts = len(border)
        xs.append(border[:, 0])
        ys.append(border[:, 1])
        fs.append(np.full(n_pts, fval))

    if xs:
        return np.concatenate(xs), np.concatenate(ys), np.concatenate(fs)
    return np.array([]), np.array([]), np.array([])


# ─────────────────────────────────────────────
#  Plotting
# ─────────────────────────────────────────────

def plot_fmiso_overlay(
    exo,
    step: int,
    pairs_found: list[tuple[int, int]],
    fmiso_values: np.ndarray,
    stem: str,
    frame_idx: int,
    total_frames: int,
    *,
    eb: int = 1,
    cmap: str = "plasma",
    gb_lw: float = 0.6,
    grain_cmap: str = "tab20",
    dpi: int = 300,
    show_axes: bool = True,
    show_colorbar: bool = True,
    show_title: bool = True,
    open_plot: bool = False,
    log,
):
    """
    Plot unique_grains as background, overlay GB interface points
    colored continuously by f_miso value.
    """
    grain_ids = np.rint(exo.elem_var_at_step("unique_grains", step, eb=eb)).astype(np.int64)
    xc, yc   = exo.element_centers_xy(eb=eb, method="bbox", cache=True)
    t        = float(exo.time()[step])

    # Estimate element spacing for interface radius
    all_pts = np.column_stack([xc, yc])
    tree_est = cKDTree(all_pts)
    dists, _ = tree_est.query(all_pts, k=2)
    elem_spacing   = float(np.median(dists[:, 1]))
    interface_radius = 1.5 * elem_spacing

    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)

    # ── background: unique_grains ───────────────────────────────
    try:
        xedges, yedges, C = build_structured_grid(xc, yc, grain_ids.astype(float))
        ax.pcolormesh(xedges, yedges, C, cmap=grain_cmap, shading="flat", alpha=0.6)
    except ValueError:
        ax.scatter(xc, yc, c=grain_ids, cmap=grain_cmap, s=4, marker="s", alpha=0.6)

    # ── f_miso overlay ──────────────────────────────────────────
    if len(pairs_found) > 0 and len(fmiso_values) > 0:
        bx, by, bf = boundary_interface_points_with_fmiso(
            pairs_found, fmiso_values, grain_ids, xc, yc, interface_radius
        )
        if len(bx):
            norm  = mcolors.Normalize(vmin=0.3, vmax=1.0)
            sc = ax.scatter(
                bx, by,
                c=bf,
                cmap=cmap,
                norm=norm,
                s=gb_lw * 4,
                marker="s",
                linewidths=0,
                zorder=3,
            )
            if show_colorbar:
                cb = fig.colorbar(sc, ax=ax, label="$f_{miso}$", shrink=0.85)
                cb.set_ticks([0.3, 0.5, 0.7, 0.9, 1.0])

    ax.set_xlim(xc.min(), xc.max())
    ax.set_ylim(yc.min(), yc.max())
    ax.set_aspect("equal", adjustable="box")
    ax.margins(0)

    if show_title:
        mean_f = fmiso_values.mean() if len(fmiso_values) > 0 else float("nan")
        ax.set_title(
            f"t = {t:.4g}s  |  GB pairs = {len(pairs_found)}  "
            f"|  mean $f_{{miso}}$ = {mean_f:.3f}"
        )
    if show_axes:
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    else:
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="both", which="both",
                       bottom=False, top=False, left=False, right=False,
                       labelbottom=False, labelleft=False)

    if open_plot:
        plt.show()
    else:
        outdir = Path("pics")
        outdir.mkdir(parents=True, exist_ok=True)
        if total_frames == 1:
            fname = outdir / f"{stem}_fmiso_overlay_step{step}.png"
        else:
            fname = outdir / f"{stem}_fmiso_overlay_{frame_idx:04d}.png"
        fig.savefig(fname, dpi=dpi, bbox_inches="tight")
        log.info(f"  Saved plot: {fname}")
    plt.close(fig)


# ─────────────────────────────────────────────
#  HDF5 output
# ─────────────────────────────────────────────

def write_hdf5_frame(
    h5file: h5py.File,
    frame_idx: int,
    step: int,
    t_val: float,
    n_grains: int,
    pairs_found: list[tuple[int, int]],
    fmiso_values: np.ndarray,
    missing: list[tuple[int, int]],
    misorientation: dict[tuple[int, int], dict],
):
    """
    Write one frame of results into an open HDF5 file.

    Layout
    ------
    /frame_{frame_idx:04d}/
        attrs:  simulation_step, time, n_grains
        pairs            : int64   (N, 2)  — grain-id pairs (i, j)
        fmiso            : float64 (N,)    — f_miso values
        misorientation_deg: float64 (N,)   — misorientation angle in degrees
        polar_rad        : float64 (N,)    — polar axis angle (radians)
        azimuth_rad      : float64 (N,)    — azimuth axis angle (radians)
        missing          : int64   (M, 2)  — pairs absent from CSV
    """
    grp = h5file.create_group(f"frame_{frame_idx:04d}")
    grp.attrs["simulation_step"] = step
    grp.attrs["time"]            = t_val
    grp.attrs["n_grains"]        = n_grains

    if len(pairs_found) > 0:
        pairs_arr = np.array(pairs_found, dtype=np.int64)   # (N, 2)

        # Pull misorientation angle, polar, azimuth for each found pair
        angle_arr   = np.array([misorientation[p]["angle_deg"] for p in pairs_found], dtype=np.float64)
        polar_arr   = np.array([
            np.arccos(np.clip(misorientation[p]["ax_z"] /
                np.maximum(np.sqrt(misorientation[p]["ax_x"]**2 +
                                   misorientation[p]["ax_y"]**2 +
                                   misorientation[p]["ax_z"]**2), 1e-12), -1.0, 1.0))
            for p in pairs_found], dtype=np.float64)
        azimuth_arr = np.array([
            (lambda a: a + 2*np.pi if a < 0 else a)(
                np.arctan2(misorientation[p]["ax_y"], misorientation[p]["ax_x"]))
            for p in pairs_found], dtype=np.float64)
    else:
        pairs_arr   = np.empty((0, 2), dtype=np.int64)
        angle_arr   = np.empty((0,),   dtype=np.float64)
        polar_arr   = np.empty((0,),   dtype=np.float64)
        azimuth_arr = np.empty((0,),   dtype=np.float64)

    grp.create_dataset("pairs",             data=pairs_arr,   compression="gzip")
    grp.create_dataset("fmiso",             data=fmiso_values, compression="gzip")
    grp.create_dataset("misorientation_deg", data=angle_arr,  compression="gzip")
    grp.create_dataset("polar_rad",         data=polar_arr,   compression="gzip")
    grp.create_dataset("azimuth_rad",       data=azimuth_arr, compression="gzip")

    missing_arr = np.array(missing, dtype=np.int64) if len(missing) > 0 else np.empty((0, 2), dtype=np.int64)
    grp.create_dataset("missing", data=missing_arr, compression="gzip")


# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────

def main():
    ti   = time.perf_counter()
    args = parse_args()
    log  = setup_logging(args.verbose)

    log.info(f"Arguments: {args}")

    # ── Load misorientation CSV once (shared across all files) ──
    csv_path       = find_misorientation_csv(args.misorientation)
    misorientation = load_misorientation(csv_path, log)
    log.warning(f"Misorientation CSV: {csv_path}  ({len(misorientation)} pairs)")

    # ── Find Exodus files ────────────────────────────────────────
    exo_files = find_exodus_files(subdirs=args.subdirs)
    if not exo_files:
        where = "subdirectories" if args.subdirs else "current directory"
        raise SystemExit(f"No .e files found in {where}.")

    log.info("Exodus files:")
    for ef in exo_files:
        log.info(f"  {ef}")

    # ── Per-file processing ──────────────────────────────────────
    for cnt, exofile in enumerate(exo_files):
        til  = time.perf_counter()
        stem = exodus_stem(exofile)

        if len(exo_files) > 1:
            log.warning(
                "\033[1m\033[96m" + f"File {cnt+1}/{len(exo_files)}: "
                + "\x1b[0m" + str(stem)
            )

        try:
            with ExodusBasics(exofile) as exo:

                steps = select_steps(
                    exo,
                    grains=args.grains,
                    time_value=args.time,
                    full=args.full,
                    frame_mode=args.frame_mode,
                    nframes=args.nframes,
                    log=log,
                )
                times_arr = exo.time()

                # Open HDF5 file for this Exodus file if requested
                h5_handle = None
                if args.hdf5:
                    h5_path = Path(f"{stem}_fmiso.h5")
                    h5_handle = h5py.File(h5_path, "w")
                    h5_handle.attrs["exodus_file"]       = str(exofile)
                    h5_handle.attrs["misorientation_csv"] = str(csv_path)
                    h5_handle.attrs["theta_max_deg"]      = THETA_MAX_DEG
                    log.warning(f"  HDF5 output: {h5_path}")

                use_tqdm = (args.verbose == 0 and len(steps) > 1)
                frame_iter = tqdm(
                    steps,
                    desc=f"Analyzing {stem}",
                    unit="frames",
                    leave=False,
                ) if use_tqdm else steps

                try:
                    for frame_idx, step in enumerate(frame_iter):
                        t_val = float(times_arr[step])
                        log.info(" ")
                        log.info(f"  Step {step}, t={t_val:.6g}")

                        # ── 1. Neighbor identification ──────────────
                        neighbor_pairs = find_neighbors(
                            exo, step,
                            method=args.neighbor_method,
                            radius=args.neighbor_radius,
                            log=log,
                        )

                        # ── 2. Compute f_miso for each GB pair ───────
                        pairs_found, fmiso_values, missing = compute_fmiso_for_pairs(
                            neighbor_pairs, misorientation, log
                        )

                        log.info(
                            f"  t={t_val:.4g}  |  "
                            f"GB_pairs={len(pairs_found)}  "
                            f"missing={len(missing)}  "
                            f"mean_fmiso={fmiso_values.mean():.4f}"
                            if len(fmiso_values) > 0
                            else f"  t={t_val:.4g}  |  no GB pairs found."
                        )

                        # ── 3. HDF5 output ───────────────────────────
                        if h5_handle is not None:
                            # Get grain_tracker count at this step (same logic as select_step)
                            gt = exo.glo_var_series("grain_tracker")
                            gt_counts = np.rint(gt).astype(np.int64)
                            n_grains = int(gt_counts[step])

                            write_hdf5_frame(
                                h5_handle, frame_idx, step, t_val,
                                n_grains,
                                pairs_found, fmiso_values, missing,
                                misorientation,
                            )

                        # ── 4. Optional overlay plot ──────────────────
                        if args.plot or args.view:
                            plot_fmiso_overlay(
                                exo, step,
                                pairs_found, fmiso_values,
                                stem=stem,
                                frame_idx=frame_idx,
                                total_frames=len(steps),
                                cmap=args.cmap,
                                gb_lw=args.gb_lw,
                                dpi=args.dpi,
                                show_axes=not args.no_axes,
                                show_colorbar=not args.no_colorbar,
                                show_title=not args.no_title,
                                open_plot=args.view,
                                log=log,
                            )

                        vtf(til, log, f"Frame {frame_idx} ")

                finally:
                    if h5_handle is not None:
                        h5_handle.close()
                        log.warning(f"  Closed HDF5: {h5_path}")

        except Exception as e:
            log.error("Failed on %s: %s: %s", exofile, type(e).__name__, e)
            sys.exit(2)

        if len(exo_files) > 1:
            tf(til, log, extra=f"File {cnt+1} ")

    if len(exo_files) > 1:
        log.warning(" ")
    tf(ti, log, extra="Total ")


if __name__ == "__main__":
    main()
