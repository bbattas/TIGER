from __future__ import annotations
from vector.ExodusBasics import ExodusBasics

import time
import numpy as np
import pandas as pd
import argparse
import sys
import logging
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from tqdm import tqdm
from scipy.spatial import cKDTree


# ─────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Grain-boundary angle analysis from unique_grains Exodus variable. " \
        "Needs a csv file of the ij misorientation angles, which can be created from " \
        "the txt file of euler angles using misc_scripts/miso_txt_check.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---- General ----
    gen = p.add_argument_group("General")
    gen.add_argument("-v", "--verbose", action="count", default=0,
                     help="Increase verbosity (-v, -vv, -vvv).")
    gen.add_argument("-s", "--subdirs", action="store_true",
                     help="Search for *.e files one level down.")
    gen.add_argument("--misorientation", "-m", type=str, required=True,
                     help="Partial path / glob pattern to locate the misorientation CSV "
                          "(e.g. 'misorientation' or 'data/misorientation').")

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
                     help="'single' = target frame only, 'all' = every frame up to target, "
                          "'sequence' = evenly spaced up to target.")
    out.add_argument("--nframes", "-f", type=int, default=40,
                     help="Number of frames for --frame-mode sequence.")

    # ---- Physics ----
    phy = p.add_argument_group("Physics")
    phy.add_argument("--threshold", type=float, default=15.0,
                     help="Low-angle GB threshold in degrees. "
                          "angle <= threshold → low-angle; > threshold → high-angle.")

    # ---- Neighbor detection ----
    nbr = p.add_argument_group("Neighbor detection")
    nbr.add_argument("--neighbor-method",
                     choices=("kdtree", "connectivity"),
                     default="connectivity",
                     help="'connectivity' uses shared element edges (exact, preferred for "
                          "structured meshes). 'kdtree' falls back to spatial proximity.")
    nbr.add_argument("--neighbor-radius", type=float, default=None,
                     help="Search radius for kdtree method. Defaults to 1.5 * median "
                          "element spacing.")

    # ---- Output ----
    ogrp = p.add_argument_group("Output")
    ogrp.add_argument("--csv", action="store_true",
                      help="Write a CSV table of time, low_count, high_count per frame.")
    ogrp.add_argument("--plot", action="store_true",
                      help="Save overlay plot of grain map + colored GB lines.")
    ogrp.add_argument("--view", action="store_true",
                      help="Show plots interactively instead of saving.")
    ogrp.add_argument("--dpi", type=int, default=300)
    ogrp.add_argument("--minimal", action="store_true",
                      help="No axes, no colorbar, no title in plots.")
    ogrp.add_argument("--no-axes", action="store_true")
    ogrp.add_argument("--no-colorbar", action="store_true")
    ogrp.add_argument("--no-title", action="store_true")
    ogrp.add_argument("--low-color", type=str, default="blue",
                      help="Color for low-angle GBs in overlay plot.")
    ogrp.add_argument("--high-color", type=str, default="red",
                      help="Color for high-angle GBs in overlay plot.")
    ogrp.add_argument("--gb-lw", type=float, default=0.6,
                      help="Line width for GB lines in overlay plot.")

    args = p.parse_args()

    if args.minimal:
        args.no_axes = True
        args.no_colorbar = True
        args.no_title = True

    return args


# ─────────────────────────────────────────────
#  Logging / timing  (mirrors plotting_general)
# ─────────────────────────────────────────────

def setup_logging(verbosity: int) -> logging.Logger:
    level = logging.WARNING
    if verbosity >= 2:
        level = logging.DEBUG
    elif verbosity >= 1:
        level = logging.INFO
    logging.basicConfig(level=level, format="%(message)s")
    return logging.getLogger("GB_ANGLE")


def tf(ti, log, extra=None):
    tag = (extra or "") + f"Time: {(time.perf_counter() - ti):.4}s"
    log.warning(tag)


def vtf(ti, log, extra=None):
    tag = (extra or "") + f"Time: {(time.perf_counter() - ti):.4}s"
    log.info(tag)


# ─────────────────────────────────────────────
#  File helpers  (mirrors plotting_general)
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
#  Step selection  (mirrors plotting_general)
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
# ─────────────────────────────────────────────

def find_misorientation_csv(partial: str) -> Path:
    """
    Locate a misorientation CSV file by partial name or path.

    If `partial` ends with '.csv', it is used as the glob pattern directly.
    Otherwise, `partial` + '*.csv' is used, allowing partial name matching
    (e.g. 'misorientation' will match 'misorientation_angles.csv').

    Search order:
      1. Exact path (if the string resolves directly to an existing file)
      2. cwd glob with the resolved pattern
      3. One-level-down glob (i.e. */<pattern>)

    Returns the first matching file found, or raises FileNotFoundError
    if no match exists in either location.
    """
    exact = Path(partial)
    if exact.is_file():
        return exact

    cwd = Path.cwd()

    if partial.endswith(".csv"):
        pattern = partial
    else:
        pattern = f"{partial}*.csv"

    candidates = sorted(cwd.glob(pattern)) + sorted(cwd.glob(f"*/{pattern}"))
    candidates = [p for p in candidates if p.is_file()]

    if not candidates:
        raise FileNotFoundError(
            f"No misorientation CSV found matching '{pattern}'. "
            f"Searched cwd and one level down."
        )
    return candidates[0]


def load_misorientation(csv_path: Path, log) -> dict[tuple[int, int], float]:
    """
    Parse CSV with columns i, j, degrees (and optionally radians).
    Returns dict keyed by canonical (min(i,j), max(i,j)) → degrees.

    Supports both the YAML-style format in the example (i: 0 / j: 1 / degrees: ...)
    and a standard comma-separated format.
    """
    log.info(f"Loading misorientation CSV: {csv_path}")
    text = csv_path.read_text()

    # ── detect format ──────────────────────────────────────────────
    # YAML-style: each record is three lines "i: X\nj: Y\ndegrees: Z"
    if "i:" in text and "j:" in text and "degrees:" in text and "," not in text:
        records = []
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        idx = 0
        while idx < len(lines):
            try:
                i_val = int(lines[idx].split(":")[1].strip())
                j_val = int(lines[idx + 1].split(":")[1].strip())
                deg   = float(lines[idx + 2].split(":")[1].strip())
                records.append((i_val, j_val, deg))
                idx += 4  # skip the optional 'radians:' line too
            except (IndexError, ValueError):
                idx += 1
    else:
        # Standard CSV with header row containing 'i', 'j', 'degrees'
        df = pd.read_csv(csv_path)
        df.columns = [c.strip().lower() for c in df.columns]
        records = list(zip(df["i"].astype(int), df["j"].astype(int), df["degrees"].astype(float)))

    result = {}
    for i_val, j_val, deg in records:
        key = (min(i_val, j_val), max(i_val, j_val))
        result[key] = deg

    log.info(f"  Loaded {len(result)} i-j misorientation pairs.")
    return result


# ─────────────────────────────────────────────
#  Neighbor identification
# ─────────────────────────────────────────────

def find_neighbors_connectivity(exo, step: int, eb: int = 1,
                                log=None) -> set[tuple[int, int]]:
    """
    Find all grain-grain neighbor pairs by scanning element connectivity.

    Two elements are neighbors if they share at least one mesh node.
    Their grain IDs are read from unique_grains at `step`.
    Pairs with the same grain ID are skipped.

    Returns set of canonical (min_id, max_id) integer tuples.
    """
    grain_ids = exo.elem_var_at_step("unique_grains", step, eb=eb)
    grain_ids = np.rint(grain_ids).astype(np.int64)

    conn = exo.connectivity(which=eb, zero_based=True)  # (n_elem, n_nodes_per_elem)
    n_elem = conn.shape[0]

    # Build node → list of element indices
    # max node index
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
    """
    Find grain-grain neighbor pairs via spatial proximity of element centers.
    Falls back for unstructured or non-standard meshes.
    """
    grain_ids = exo.elem_var_at_step("unique_grains", step, eb=eb)
    grain_ids = np.rint(grain_ids).astype(np.int64)

    xc, yc = exo.element_centers_xy(eb=eb, method="bbox", cache=True)
    pts = np.column_stack([xc, yc])

    if radius is None:
        # estimate from median nearest-neighbor distance
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
#  GB angle classification
# ─────────────────────────────────────────────

def classify_boundaries(
    neighbor_pairs: set[tuple[int, int]],
    misorientation: dict[tuple[int, int], float],
    threshold: float,
    log,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]], list[tuple[int, int]]]:
    """
    Classify each neighbor pair as low-angle, high-angle, or missing.

    Returns (low_pairs, high_pairs, missing_pairs).
    missing_pairs = neighbors found in the mesh but absent from the CSV.
    """
    low, high, missing = [], [], []
    for pair in sorted(neighbor_pairs):
        angle = misorientation.get(pair)
        if angle is None:
            missing.append(pair)
        elif angle <= threshold:
            low.append(pair)
        else:
            high.append(pair)

    log.info(
        f"  Classification: low={len(low)}, high={len(high)}, "
        f"missing_from_csv={len(missing)}"
    )
    if missing:
        log.debug(f"  Missing pairs (first 10): {missing[:10]}")

    return low, high, missing


# ─────────────────────────────────────────────
#  Structured grid helpers  (from plotting_general)
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
#  GB boundary line computation
# ─────────────────────────────────────────────

def grain_id_to_elem_centers(grain_ids_all, xc, yc) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """
    Pre-build a mapping: grain_id → (xc_array, yc_array) of element centers in that grain.
    """
    unique_ids = np.unique(grain_ids_all)
    result = {}
    for gid in unique_ids:
        mask = grain_ids_all == gid
        result[int(gid)] = (xc[mask], yc[mask])
    return result


def boundary_midpoints(
    pairs: list[tuple[int, int]],
    grain_to_centers: dict[int, tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    """
    For each grain pair, find the midpoint between the two grains' centroid.
    Used as scatter positions to mark a boundary visually.

    Returns (x_pts, y_pts).
    """
    xs, ys = [], []
    for g_i, g_j in pairs:
        if g_i not in grain_to_centers or g_j not in grain_to_centers:
            continue
        xi, yi = grain_to_centers[g_i]
        xj, yj = grain_to_centers[g_j]
        cx_i, cy_i = xi.mean(), yi.mean()
        cx_j, cy_j = xj.mean(), yj.mean()
        xs.append(0.5 * (cx_i + cx_j))
        ys.append(0.5 * (cy_i + cy_j))
    return np.array(xs), np.array(ys)


def boundary_interface_points(
    pairs: list[tuple[int, int]],
    grain_ids_all: np.ndarray,
    xc: np.ndarray,
    yc: np.ndarray,
    radius: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    For each grain pair, find element-center points from either grain that lie
    within `radius` of the other grain's elements.  These are the interface
    element centers — giving a dense, accurate tracing of the boundary.

    Returns (x_pts, y_pts) — may be empty if grains are not in contact.
    """
    # Build spatial index over all element centers
    pts = np.column_stack([xc, yc])
    tree = cKDTree(pts)

    xs, ys = [], []
    for g_i, g_j in pairs:
        mask_i = grain_ids_all == g_i
        mask_j = grain_ids_all == g_j
        pts_i = pts[mask_i]
        pts_j = pts[mask_j]
        if len(pts_i) == 0 or len(pts_j) == 0:
            continue
        # Elements of grain i that are close to grain j
        tree_j = cKDTree(pts_j)
        dists, _ = tree_j.query(pts_i)
        border_i = pts_i[dists <= radius]
        # Elements of grain j that are close to grain i
        tree_i = cKDTree(pts_i)
        dists2, _ = tree_i.query(pts_j)
        border_j = pts_j[dists2 <= radius]
        border = np.vstack([border_i, border_j]) if (len(border_i) and len(border_j)) \
                 else (border_i if len(border_i) else border_j)
        if len(border):
            xs.append(border[:, 0])
            ys.append(border[:, 1])

    if xs:
        return np.concatenate(xs), np.concatenate(ys)
    return np.array([]), np.array([])


# ─────────────────────────────────────────────
#  Plotting
# ─────────────────────────────────────────────

def plot_gb_overlay(
    exo,
    step: int,
    low_pairs: list[tuple[int, int]],
    high_pairs: list[tuple[int, int]],
    stem: str,
    frame_idx: int,
    total_frames: int,
    *,
    eb: int = 1,
    low_color: str = "blue",
    high_color: str = "red",
    gb_lw: float = 0.6,
    cmap: str = "tab20",
    dpi: int = 300,
    show_axes: bool = True,
    show_colorbar: bool = True,
    show_title: bool = True,
    open_plot: bool = False,
    log,
):
    """
    Plot unique_grains as background, overlay GB interface points colored by angle class.
    """
    grain_ids = np.rint(exo.elem_var_at_step("unique_grains", step, eb=eb)).astype(np.int64)
    xc, yc = exo.element_centers_xy(eb=eb, method="bbox", cache=True)
    t = float(exo.time()[step])

    # ── background: unique_grains ───────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)

    try:
        xedges, yedges, C = build_structured_grid(xc, yc, grain_ids.astype(float))
        ax.pcolormesh(xedges, yedges, C, cmap=cmap, shading="flat")
    except ValueError:
        sc = ax.scatter(xc, yc, c=grain_ids, cmap=cmap, s=4, marker="s")
        if show_colorbar:
            fig.colorbar(sc, ax=ax, label="unique_grains")

    # ── GB overlay ──────────────────────────────────────────────────
    # Estimate element spacing for interface radius
    all_pts = np.column_stack([xc, yc])
    tree_est = cKDTree(all_pts)
    dists, _ = tree_est.query(all_pts, k=2)
    elem_spacing = float(np.median(dists[:, 1]))
    interface_radius = 1.5 * elem_spacing

    for pairs, color in [(low_pairs, low_color), (high_pairs, high_color)]:
        if not pairs:
            continue
        bx, by = boundary_interface_points(pairs, grain_ids, xc, yc, interface_radius)
        if len(bx):
            ax.scatter(bx, by, c=color, s=gb_lw * 4, marker="s",
                       linewidths=0, zorder=3)

    # ── legend ──────────────────────────────────────────────────────
    handles = [
        mpatches.Patch(color=low_color,  label=f"Low-angle GB"),
        mpatches.Patch(color=high_color, label=f"High-angle GB"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=7, framealpha=0.7)

    ax.set_xlim(xc.min(), xc.max())
    ax.set_ylim(yc.min(), yc.max())
    ax.set_aspect("equal", adjustable="box")
    ax.margins(0)

    if show_title:
        ax.set_title(f"t = {t:.4g}s  |  low={len(low_pairs)}  high={len(high_pairs)}")
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
            fname = outdir / f"{stem}_gb_overlay_angle_step{step}.png"
        else:
            fname = outdir / f"{stem}_gb_overlay_angle_{frame_idx:04d}.png"
        fig.savefig(fname, dpi=dpi, bbox_inches="tight")
        log.info(f"  Saved plot: {fname}")
        plt.close(fig)


# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────

def main():
    ti = time.perf_counter()
    args = parse_args()
    log  = setup_logging(args.verbose)

    log.info(f"Arguments: {args}")

    # ── Load misorientation CSV once (shared across all files) ──────
    csv_path     = find_misorientation_csv(args.misorientation)
    misorientation = load_misorientation(csv_path, log)
    log.warning(f"Misorientation CSV: {csv_path}  ({len(misorientation)} pairs)")

    # ── Find Exodus files ───────────────────────────────────────────
    exo_files = find_exodus_files(subdirs=args.subdirs)
    if not exo_files:
        where = "subdirectories" if args.subdirs else "current directory"
        raise SystemExit(f"No .e files found in {where}.")

    log.info("Exodus files:")
    for ef in exo_files:
        log.info(f"  {ef}")

    # ── Per-file processing ─────────────────────────────────────────
    for cnt, exofile in enumerate(exo_files):
        til   = time.perf_counter()
        stem  = exodus_stem(exofile)

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

                # Accumulate per-frame results for CSV output
                csv_rows: list[dict] = []

                use_tqdm = (args.verbose == 0 and len(steps) > 1)
                frame_iter = tqdm(
                    steps,
                    desc=f"Analyzing {stem}",
                    unit="frames",
                    leave=False,
                ) if use_tqdm else steps

                for frame_idx, step in enumerate(frame_iter):
                    t_val = float(times_arr[step])
                    log.info(" ")
                    log.info(f"  Step {step}, t={t_val:.6g}")

                    # ── 1. Neighbor identification ──────────────────
                    neighbor_pairs = find_neighbors(
                        exo, step,
                        method=args.neighbor_method,
                        radius=args.neighbor_radius,
                        log=log,
                    )

                    # ── 2. Classify by misorientation angle ─────────
                    low_pairs, high_pairs, missing = classify_boundaries(
                        neighbor_pairs, misorientation, args.threshold, log
                    )

                    log.info(
                        f"  t={t_val:.4g}  |  "
                        f"low={len(low_pairs)}  high={len(high_pairs)}  "
                        f"missing={len(missing)}"
                    )

                    csv_rows.append({
                        "time":     t_val,
                        "step":     step,
                        "low":      len(low_pairs),
                        "high":     len(high_pairs),
                        "missing":  len(missing),
                    })

                    # ── 3. Optional overlay plot ────────────────────
                    if args.plot or args.view:
                        plot_gb_overlay(
                            exo, step, low_pairs, high_pairs,
                            stem=stem,
                            frame_idx=frame_idx,
                            total_frames=len(steps),
                            low_color=args.low_color,
                            high_color=args.high_color,
                            gb_lw=args.gb_lw,
                            dpi=args.dpi,
                            show_axes=not args.no_axes,
                            show_colorbar=not args.no_colorbar,
                            show_title=not args.no_title,
                            open_plot=args.view,
                            log=log,
                        )

                # ── 4. Optional summary CSV ─────────────────────────
                if args.csv and csv_rows:
                    df = pd.DataFrame(csv_rows)[["time", "step", "low", "high", "missing"]]
                    # outdir = Path("data_out")
                    # outdir.mkdir(parents=True, exist_ok=True)
                    # csv_out = outdir / f"{stem}_gb_counts.csv"
                    csv_out = f"{stem}_gb_counts.csv"
                    df.to_csv(csv_out, index=False)
                    log.warning(f"  Saved CSV: {csv_out}")

                vtf(til, log, f"File {cnt+1} analysis ")

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
