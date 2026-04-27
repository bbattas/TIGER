from __future__ import annotations
from vector.ExodusBasics import ExodusBasics

import time
import numpy as np
import argparse
import sys
import logging
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.collections import LineCollection
from pathlib import Path
from tqdm import tqdm



# Assumes uniform constant unchanging mesh, dx = dy and treats all index values as the coords

def parse_args():
    p = argparse.ArgumentParser(
        description="Plot Exodus results for any specified variable/property.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # ---- General ----
    log = p.add_argument_group("General")
    log.add_argument("-v", "--verbose", action="count", default=0,
                     help="Increase verbosity (-v, -vv, -vvv).")
    log.add_argument("-s", "--subdirs", action="store_true",
                     help="Search for *.e files one level down (./*/.e). If not set, only search current directory.")

    # ---- Target frame selection ----
    tim = p.add_argument_group("Target frame selection (choose one)")
    grp = tim.add_mutually_exclusive_group(required=True)
    grp.add_argument(
        "-g", "--grains", type=int,
        help="Target grain count; chooses timestep with grain_tracker closest to this value."
    )
    grp.add_argument(
        "-t", "--time", type=float,
        help="Target time; chooses timestep with time_whole closest to this value."
    )
    grp.add_argument(
        "--full", action="store_true",
        help="Use the final Exodus timestep as the target/end frame."
    )

    # ---- Output frame mode ----
    out = p.add_argument_group("Timestepping")
    out.add_argument(
        "--frame-mode",
        choices=("single", "all", "sequence"),
        default="single",
        help=(
            "How many frames to output relative to the selected target time: "
            "'single' = only the selected time, "
            "'all' = every frame from start to selected time, "
            "'sequence' = evenly spaced frames from start to selected time."
        ),
    )
    out.add_argument(
        "--nframes", "-f",
        type=int,
        default=40,
        help=(
            "Number of evenly spaced frames to output when --frame-mode sequence. "
            "Includes the first frame and the selected target frame."
        ),
    )

    # ---- Plot options ----
    plot = p.add_argument_group("Plotting")
    plot.add_argument("--var", "-p", type=str, required=True,
                      help="What variable to plot.")
    plot.add_argument("--view", action="store_true",
                      help="Do NOT save figure, just view the plot.")
    plot.add_argument("--dpi", type=int, default=300,
                      help="Plotting dpi.")
    plot.add_argument("--minimal", action="store_true",
                      help="Remove titles and axes and colorbar from plot images.")
    plot.add_argument("--no-axes", action="store_true",
                      help="Remove X and Y axes from plot images.")
    plot.add_argument("--no-colorbar", action="store_true",
                      help="Remove colorbar from plot images.")
    plot.add_argument("--no-title", action="store_true",
                      help="Remove title with time value from plot images.")
    plot.add_argument("--gbs", action="store_true",
                      help="Overlay grain boundary edges on elemental feature plots.")
    plot.add_argument("--boundary-color", type=str, default="black",
                      help="Color of grain boundary overlay lines.")
    plot.add_argument("--boundary-lw", type=float, default=0.5,
                      help="Line width of grain boundary overlay.")

    args = p.parse_args()

    if args.minimal:
        args.no_axes = True
        args.no_colorbar = True
        args.no_title = True

    return args


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


def select_step(exo, *, grains: int | None, time_value: float | None, full: bool, log: logging.Logger) -> int:
    """
    exo: an open ExodusBasics instance
    Returns: timestep index (0-based)
    """
    times = exo.time()

    if full:
        step = len(times) - 1
        log.info(f"Frame selected by full run: chosen final step={step}, time={times[step]}")
        return step

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


def unique_preserve_order(values) -> list[int]:
    """Return unique ints in original order."""
    out = []
    seen = set()
    for v in values:
        iv = int(v)
        if iv not in seen:
            out.append(iv)
            seen.add(iv)
    return out


def select_steps_by_time(times: np.ndarray, target_step: int, nframes: int) -> list[int]:
    """
    Build a sequence from the first frame to target_step using evenly spaced
    target TIMES, then map each target time to the nearest actual Exodus step.

    Returns a unique, ordered list of step indices.
    """
    times = np.asarray(times)

    if target_step < 0:
        raise ValueError("target_step must be >= 0")
    if nframes <= 0:
        raise ValueError("--nframes must be >= 1")

    # For a single requested frame, just use the target frame
    if nframes == 1:
        return [target_step]

    t0 = float(times[0])
    t1 = float(times[target_step])

    target_times = np.linspace(t0, t1, nframes)
    steps = [closest_index(times[:target_step + 1], tt) for tt in target_times]

    # Remove duplicates caused by irregular or sparse output times
    # return unique_preserve_order(steps)
    # Output all including duplicates to preserve time in videos
    return steps


def select_steps(
    exo,
    *,
    grains: int | None,
    time_value: float | None,
    full: bool,
    frame_mode: str,
    nframes: int,
    log: logging.Logger,
) -> list[int]:
    """
    Return a list of timestep indices (0-based) to plot.

    frame_mode:
        - 'single'   : [target_step]
        - 'all'      : [0, 1, ..., target_step]
        - 'sequence' : evenly spaced frames from 0 to target_step inclusive
    """
    times = exo.time()
    target_step = select_step(exo, grains=grains, time_value=time_value, full=full, log=log)

    if frame_mode == "single":
        steps = [target_step]

    elif frame_mode == "all":
        steps = list(range(target_step + 1))

    elif frame_mode == "sequence":
        steps = select_steps_by_time(times, target_step, nframes)

    else:
        raise ValueError("frame_mode must be 'single', 'all', or 'sequence'")

    log.info(
        f"Frame mode={frame_mode}, target_step={target_step}, "
        f"target_time={times[target_step]}, selected_steps={steps}, "
        f"selected_times={[float(times[s]) for s in steps]}"
    )
    return steps



def centers_to_edges(vals):
    vals = np.asarray(vals)
    if vals.size == 1:
        d = 0.5
        return np.array([vals[0] - d, vals[0] + d])

    mids = 0.5 * (vals[:-1] + vals[1:])
    first = vals[0] - 0.5 * (vals[1] - vals[0])
    last = vals[-1] + 0.5 * (vals[-1] - vals[-2])
    return np.concatenate(([first], mids, [last]))

def build_structured_grid(x, y, c):
    """
    Try to reshape center-based data onto a structured rectangular grid.
    Returns xedges, yedges, C if successful, otherwise raises ValueError.
    """
    xu = np.unique(x)
    yu = np.unique(y)

    nx = len(xu)
    ny = len(yu)

    if nx * ny != len(c):
        raise ValueError(
            "Data do not form a full structured rectangular grid."
        )

    ix = np.searchsorted(xu, x)
    iy = np.searchsorted(yu, y)

    C = np.full((ny, nx), np.nan)
    C[iy, ix] = c

    if np.isnan(C).any():
        raise ValueError(
            "Structured grid contains missing cells; cannot use pcolormesh cleanly."
        )

    xedges = centers_to_edges(xu)
    yedges = centers_to_edges(yu)
    return xedges, yedges, C


def draw_grain_boundaries(ax, x, y, c, color="black", lw=0.5):
    """
    Overlay grain boundary lines on an elemental plot.
    Uses vectorized numpy diff + LineCollection for performance.
    """
    try:
        xedges, yedges, C = build_structured_grid(x, y, c)
    except ValueError:
        from scipy.interpolate import griddata
        xi = np.linspace(x.min(), x.max(), 500)
        yi = np.linspace(y.min(), y.max(), 500)
        Xi, Yi = np.meshgrid(xi, yi)
        Ci = griddata((x, y), c, (Xi, Yi), method="nearest")
        ax.contour(Xi, Yi, Ci, levels=np.unique(c), colors=color, linewidths=lw)
        return

    # Cast to integer to avoid float comparison artifacts
    C_int = np.rint(C).astype(np.int64)

    # Boolean masks where neighboring cells differ
    h_diff = C_int[:, :-1] != C_int[:, 1:]
    v_diff = C_int[:-1, :] != C_int[1:, :]

    iy_v, ix_v = np.where(h_diff)
    iy_h, ix_h = np.where(v_diff)

    if len(iy_v) == 0 and len(iy_h) == 0:
        return

    # Vertical segments
    vsegs = np.column_stack([
        xedges[ix_v + 1], yedges[iy_v],
        xedges[ix_v + 1], yedges[iy_v + 1],
    ]).reshape(-1, 2, 2)

    # Horizontal segments
    hsegs = np.column_stack([
        xedges[ix_h],     yedges[iy_h + 1],
        xedges[ix_h + 1], yedges[iy_h + 1],
    ]).reshape(-1, 2, 2)

    all_segs = np.vstack([vsegs, hsegs])
    lc = LineCollection(all_segs, colors=color, linewidths=lw, zorder=2)
    ax.add_collection(lc)


def plot_exodus_var(
    exo,
    name: str,
    step: int,
    savename: str | None = None,
    *,
    eb: int = 1,
    method: str = "auto",
    elem_center_method: str = "mean",
    cmap: str = "viridis",
    s: float = 12,
    figsize=(6, 5),
    ax=None,
    show_colorbar: bool = True,
    square_scatter: bool = True,
    vmin=None,
    vmax=None,
    show_axes: bool = True,
    dpi: int = 300,
    show_title: bool = True,
    open_plot: bool = False,
    show_boundaries: bool = False,
    boundary_color: str = "black",
    boundary_lw: float = 0.5,
):
    """
    Plot Exodus variable with multiple plotting styles.

    Parameters
    ----------
    exo : ExodusBasics
        Open ExodusBasics reader
    name : str
        Variable name
    step : int
        Timestep index (0-based)
    eb : int, optional
        Element block for elemental variables
    method : {"auto", "scatter", "tripcolor", "pcolormesh"}
        Plotting method
    elem_center_method : {"min", "mean", "bbox"}
        Representative coordinate choice for elemental variables
    cmap : str
        Matplotlib colormap name
    s : float
        Marker size for scatter
    figsize : tuple
        Figure size if ax is not provided
    ax : matplotlib axis, optional
        Existing axis to plot on
    show_colorbar : bool
        Whether to add a colorbar
    square_scatter : bool
        If True, use square markers for scatter plots of elemental data

    Returns
    -------
    fig, ax, artist
    """
    kind = exo.var_kind(name)
    t = exo.time()[step]
    if name == "unique_grains":
        if vmin is None:
            vmin = 0 #exo.elem_var_at_step("unique_grains", step=0, eb=eb).min()
        if vmax is None:
            vmax = exo.elem_var_at_step("unique_grains", step=0, eb=eb).max()

    x, y, z, c = exo.xyzc_at_step(
        name,
        step,
        eb=eb,
        elem_center_method=elem_center_method,
    )

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    else:
        fig = ax.figure

    if method == "auto":
        if kind == "nodal":
            method = "tripcolor"
        else:
            try:
                xedges, yedges, C = build_structured_grid(x, y, c)
                method = "pcolormesh"
            except ValueError:
                method = "scatter"

    if method == "scatter":
        marker = "s" if (kind == "elemental" and square_scatter) else "o"
        artist = ax.scatter(x, y, c=c, s=s, cmap=cmap, marker=marker, vmin=vmin, vmax=vmax)

    elif method == "tripcolor":
        tri = mtri.Triangulation(x, y)
        shading = "gouraud" if kind == "nodal" else "flat"
        artist = ax.tripcolor(tri, c, cmap=cmap, shading=shading, vmin=vmin, vmax=vmax)

    elif method == "pcolormesh":
        xedges, yedges, C = build_structured_grid(x, y, c)
        artist = ax.pcolormesh(xedges, yedges, C, cmap=cmap, shading="flat", vmin=vmin, vmax=vmax)

    else:
        raise ValueError(
            "method must be 'auto', 'scatter', 'tripcolor', or 'pcolormesh'"
        )

    if show_boundaries:
        draw_grain_boundaries(ax, x, y, c, color=boundary_color, lw=boundary_lw)

    if show_title:
        ax.set_title(f"t = {t:.2f}s")

    if show_axes:
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    else:
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(
            axis="both",
            which="both",
            bottom=False,
            top=False,
            left=False,
            right=False,
            labelbottom=False,
            labelleft=False,
        )

    # Boundary of plot domain to limits
    ax.set_xlim(np.min(x), np.max(x))
    ax.set_ylim(np.min(y), np.max(y))
    ax.margins(0)

    ax.set_aspect("equal", adjustable="box")

    if show_colorbar:
        fig.colorbar(artist, ax=ax, label=name)

    if open_plot:
        plt.show()
    elif savename is not None:
        outdir = Path("pics")
        outdir.mkdir(parents=True, exist_ok=True)
        outfile = outdir / f"{savename}.png"
        fig.savefig(outfile, dpi=dpi, bbox_inches="tight")
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
                steps = select_steps(
                    exo,
                    grains=args.grains,
                    time_value=args.time,
                    full=args.full,
                    frame_mode=args.frame_mode,
                    nframes=args.nframes,
                    log=log,
                )

                times = exo.time()
                minimal_tag = "_minimal" if args.minimal else ""
                use_tqdm = (args.verbose == 0 and len(steps) > 1)
                frame_iter = tqdm(
                    steps,
                    desc=f"Plotting {args.var} in {stem}",
                    unit="frames",
                    leave=False,
                ) if use_tqdm else steps

                for i, step in enumerate(frame_iter):
                    if len(steps) == 1:
                        frame_name = f"{stem}_{args.var}{minimal_tag}_step{step}"
                    else:
                        frame_name = f"{stem}_{args.var}{minimal_tag}_{i:04d}"

                    log.info(
                        f"Plotting frame {i+1}/{len(steps)}: "
                        f"  step={step}, time={times[step]:.6g}, savename={frame_name}"
                    )

                    plot_exodus_var(
                        exo,
                        name=args.var,
                        step=step,
                        savename=frame_name,
                        method="auto",
                        show_axes=not args.no_axes,
                        show_colorbar=not args.no_colorbar,
                        show_title=not args.no_title,
                        open_plot=args.view,
                        show_boundaries=args.gbs,
                        boundary_color=args.boundary_color,
                        boundary_lw=args.boundary_lw
                    )

                vtf(ti, log, "Finished plotting selected frame(s) ")
                log.info(' ')

                if len(exo_files) > 1:
                    tf(til,log,extra=f"File {cnt+1} ")


        except Exception as e:
            # argparse-style error output (clean for CLI)
            # log.warning(f"ERROR: {e}")#, file=sys.stderr)
            log.error("Failed in file %s:  %s: %s",
              exofile, type(e).__name__, e)
            sys.exit(2)

    # TOTAL end time
    if len(exo_files) > 1:
        log.warning(' ')
    tf(ti,log,extra="Total ")

if __name__ == "__main__":
    main()

