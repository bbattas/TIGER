from __future__ import annotations
from vector.ExodusBasics import ExodusBasics

import time
import numpy as np
import argparse
import sys
import logging
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.tri as mtri
import matplotlib.ticker as mticker
from pathlib import Path
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser(
        description="Plot BISON Exodus results for any specified variable/property.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # ---- General ----
    log = p.add_argument_group("General")
    log.add_argument("-v", "--verbose", action="count", default=0,
                     help="Increase verbosity (-v, -vv, -vvv).")
    log.add_argument("-s", "--subdirs", action="store_true",
                     help="Search for *.e files one level down (./*/.e). If not set, only search current directory.")
    log.add_argument("--input", "-i", type=str, default=None, metavar="PATTERN",
                    help="Only process .e files whose name contains this string (e.g. 'job_042').")

    # ---- Target frame selection ----
    tim = p.add_argument_group("Target frame selection (choose one)")
    grp = tim.add_mutually_exclusive_group(required=False)
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
    plot.add_argument("--var", "-p", type=str, default=None,
                      help="What variable to plot. Required unless --debug-blocks is set.")
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
    plot.add_argument("--eb", type=int, default=2,
                      help="Element block index to plot.")
    plot.add_argument("--xscale", type=float, default=10.0,
                      help="Multiply x coordinates by this factor for display only (does not affect data).")
    plot.add_argument("--whitespace", action="store_true",
                      help=("Fix plot limits using timestep 0, with 10% padding on each "
                            "side of the initial x and y ranges."))
    # ---- Debug ----
    dbg = p.add_argument_group("Debug")
    dbg.add_argument("--debug-blocks", action="store_true",
                     help=(
                         "Show bounding boxes for every element block in the Exodus file "
                         "and exit. Useful for identifying the correct --eb for your region of interest. "
                         "--xscale is applied. Does not require --var."
                     ))

    args = p.parse_args()

    # Validate: --var is required unless --debug-blocks is set
    if not args.debug_blocks and args.var is None:
        p.error("--var / -p is required unless --debug-blocks is set.")

    # Validate: a frame selection mode is required unless --debug-blocks is set
    if not args.debug_blocks and not args.full and args.grains is None and args.time is None:
        p.error("One of --grains, --time, or --full is required unless --debug-blocks is set.")

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
    logging.basicConfig(level=level, format="%(message)s")
    return logging.getLogger("BISON")


def tf(ti, log, extra=None):
    if extra is not None:
        log.warning(extra + f"Time: {(time.perf_counter()-ti):.4}s")
    else:
        log.warning(f"Time: {(time.perf_counter()-ti):.4}s")

def vtf(ti, log, extra=None):
    if extra is not None:
        log.info(extra + f"Time: {(time.perf_counter()-ti):.4}s")
    else:
        log.info(f"Time: {(time.perf_counter()-ti):.4}s")


def find_exodus_files(*, subdirs: bool = False, pattern: str = "*.e", filter_str: str | None = None) -> list[Path]:
    """
    Find Exodus files in current directory or one-level-down subdirectories.
    If filter_str is given, only return files whose name contains that string.
    Returns sorted list of Paths.
    """
    cwd = Path.cwd()

    if filter_str:
        glob_pattern = f"*{filter_str}*.e"
    else:
        glob_pattern = pattern

    if subdirs:
        files = sorted(cwd.glob(f"*/{glob_pattern}"))
    else:
        files = sorted(cwd.glob(glob_pattern))

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
    Returns ordered list of step indices (duplicates preserved for video timing).
    """
    times = np.asarray(times)

    if target_step < 0:
        raise ValueError("target_step must be >= 0")
    if nframes <= 0:
        raise ValueError("--nframes must be >= 1")

    if nframes == 1:
        return [target_step]

    t0 = float(times[0])
    t1 = float(times[target_step])

    target_times = np.linspace(t0, t1, nframes)
    steps = [closest_index(times[:target_step + 1], tt) for tt in target_times]
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
    last  = vals[-1] + 0.5 * (vals[-1] - vals[-2])
    return np.concatenate(([first], mids, [last]))


def build_structured_grid(x, y, c):
    """
    Try to reshape center-based data onto a structured rectangular grid.
    Returns xedges, yedges, C if successful, otherwise raises ValueError.
    x here should already be the display-scaled copy.
    """
    xu = np.unique(x)
    yu = np.unique(y)

    nx = len(xu)
    ny = len(yu)

    if nx * ny != len(c):
        raise ValueError("Data do not form a full structured rectangular grid.")

    ix = np.searchsorted(xu, x)
    iy = np.searchsorted(yu, y)

    C = np.full((ny, nx), np.nan)
    C[iy, ix] = c

    if np.isnan(C).any():
        raise ValueError("Structured grid contains missing cells; cannot use pcolormesh cleanly.")

    xedges = centers_to_edges(xu)
    yedges = centers_to_edges(yu)
    return xedges, yedges, C


def plot_block_debug(exo, xscale: float = 1.0):
    """
    Display bounding boxes for every element block in the open Exodus file.
    Each block gets its own colored rectangle + legend entry.
    xscale is applied to x coordinates for display only.
    Calls plt.show() and does NOT save anything.
    """
    connect_names = exo.connect_varnames()
    n_blocks = len(connect_names)

    if n_blocks == 0:
        print("No element blocks (connect*) found in this Exodus file.")
        return

    # Use a qualitative colormap with enough colors
    cmap = plt.get_cmap("tab10") if n_blocks <= 10 else plt.get_cmap("tab20")
    colors = [cmap(i % cmap.N) for i in range(n_blocks)]

    fig, ax = plt.subplots(figsize=(6, 8), constrained_layout=True)

    for i, cname in enumerate(connect_names):
        eb = int(cname.replace("connect", ""))
        try:
            xc, yc = exo.element_centers_xy(eb=eb, method="bbox")
        except Exception as e:
            print(f"  Warning: could not compute centers for {cname}: {e}")
            continue

        xc_plot = xc * xscale  # display-only scaled copy

        xmin, xmax = float(xc_plot.min()), float(xc_plot.max())
        ymin, ymax = float(yc.min()),       float(yc.max())

        width  = xmax - xmin
        height = ymax - ymin

        rect = mpatches.FancyBboxPatch(
            (xmin, ymin), width, height,
            boxstyle="square,pad=0",
            linewidth=2,
            edgecolor=colors[i],
            facecolor=(*colors[i][:3], 0.15),  # semi-transparent fill
            label=f"eb={eb}  x=[{xmin:.4g}, {xmax:.4g}]  y=[{ymin:.4g}, {ymax:.4g}]",
            zorder=2,
        )
        ax.add_patch(rect)

        # Label the block at center of bbox
        cx = 0.5 * (xmin + xmax)
        cy = 0.5 * (ymin + ymax)
        ax.text(cx, cy, f"eb={eb}", ha="center", va="center",
                fontsize=9, color=colors[i], fontweight="bold", zorder=3)

    ax.autoscale_view()
    # ax.set_aspect("auto")   # intentional: xscale may distort aspect
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(f"x  (scaled ×{xscale})" if xscale != 1.0 else "x")
    ax.set_ylabel("y")
    ax.set_title("Element block bounding boxes — debug view")
    fig.legend(loc="outside right center", fontsize=8, framealpha=0.9)
    fig.tight_layout()

    print(f"\nDebug block view: {n_blocks} block(s) found.")
    for cname in connect_names:
        eb = int(cname.replace("connect", ""))
        print(f"  {cname}  (use --eb {eb})")

    plt.show()


def plot_exodus_var(
    exo,
    name: str,
    step: int,
    savename: str | None = None,
    *,
    eb: int = 1,
    xscale: float = 1.0,
    method: str = "auto",
    elem_center_method: str = "mean",
    cmap: str = "viridis",
    s: float = 12,
    figsize=(3,5), #(6, 5),
    ax=None,
    show_colorbar: bool = True,
    square_scatter: bool = True,
    vmin=None,
    vmax=None,
    show_axes: bool = True,
    dpi: int = 300,
    show_title: bool = True,
    open_plot: bool = False,
    plot_limits=None,
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
    savename : str, optional
        Base name for saved file (saved under pics/)
    eb : int
        Element block for elemental variables (default 1)
    xscale : float
        Multiply x coordinates by this factor for display only.
        The underlying data is never modified.
    method : {"auto", "scatter", "tripcolor", "pcolormesh"}
        Plotting method
    elem_center_method : {"min", "mean", "bbox"}
        Representative coordinate choice for elemental variables

    Returns
    -------
    fig, ax, artist
    """
    kind = exo.var_kind(name)
    t = exo.time()[step]

    if name == "unique_grains":
        if vmin is None:
            vmin = 0
        if vmax is None:
            vmax = exo.elem_var_at_step("unique_grains", step=0, eb=eb).max()

    x, y, z, c = exo.xyzc_at_step(
        name,
        step,
        eb=eb,
        elem_center_method=elem_center_method,
    )

    # Display-only scaled x — the raw x array from exo is never mutated
    x_plot = x * xscale

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    else:
        fig = ax.figure

    if method == "auto":
        if kind == "nodal":
            method = "tripcolor"
        else:
            try:
                build_structured_grid(x_plot, y, c)
                method = "pcolormesh"
            except ValueError:
                method = "scatter"

    if method == "scatter":
        marker = "s" if (kind == "elemental" and square_scatter) else "o"
        artist = ax.scatter(x_plot, y, c=c, s=s, cmap=cmap, marker=marker, vmin=vmin, vmax=vmax)

    elif method == "tripcolor":
        tri = mtri.Triangulation(x_plot, y)
        shading = "gouraud" if kind == "nodal" else "flat"
        artist = ax.tripcolor(tri, c, cmap=cmap, shading=shading, vmin=vmin, vmax=vmax)

    elif method == "pcolormesh":
        xedges, yedges, C = build_structured_grid(x_plot, y, c)
        artist = ax.pcolormesh(xedges, yedges, C, cmap=cmap, shading="flat", vmin=vmin, vmax=vmax)

    else:
        raise ValueError("method must be 'auto', 'scatter', 'tripcolor', or 'pcolormesh'")

    if show_title:
        ax.set_title(f"t = {t:.3e}s")

    if show_axes:
        ax.set_xlabel(f"x  (scaled ×{xscale})" if xscale != 1.0 else "x")
        ax.set_ylabel("y")
        if xscale != 1.0:
            ax.xaxis.set_major_formatter(
                mticker.FuncFormatter(lambda val, _: f"{val / xscale:.4g}")
            )
        ax.tick_params(axis="x", labelrotation=45, labelsize=8)
    else:
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(
            axis="both",
            which="both",
            bottom=False, top=False, left=False, right=False,
            labelbottom=False, labelleft=False,
        )

    # ax.set_xlim(np.min(x_plot), np.max(x_plot))
    # ax.set_ylim(np.min(y),      np.max(y))
    if plot_limits is None:
        ax.set_xlim(np.min(x_plot), np.max(x_plot))
        ax.set_ylim(np.min(y),      np.max(y))
    else:
        xmin, xmax, ymin, ymax = plot_limits
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
    ax.margins(0)

    # Use 'auto' aspect since xscale intentionally distorts the domain
    # ax.set_aspect("auto", adjustable="box")
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
    log.info("Setup:")
    log.info(f"Arguments: {args}")

    exo_files = find_exodus_files(subdirs=args.subdirs, filter_str=args.input)
    if not exo_files:
        where = "subdirectories" if args.subdirs else "current directory"
        raise SystemExit(f"No .e files found in {where}.")

    log.info(" ")
    log.info("Exodus Files:")
    for ef in exo_files:
        log.info(f"  File: {ef}")
        log.info(f"  Namebase: {exodus_stem(ef)}")
        log.info(" ")

    # ---- Debug-blocks mode: open first file, show block map, exit ----
    if args.debug_blocks:
        exofile = exo_files[0]
        log.warning(f"Debug-blocks mode: inspecting {exofile}")
        with ExodusBasics(exofile) as exo:
            plot_block_debug(exo, xscale=args.xscale)
        sys.exit(0)

    # ---- Normal plotting loop ----
    for cnt, exofile in enumerate(exo_files):
        til = time.perf_counter()
        stem = exodus_stem(exofile)
        if len(exo_files) > 1:
            log.warning(" ")
        log.warning("\033[1m\033[96m" + "File " + str(cnt+1) + "/" + str(len(exo_files)) + ": " + "\x1b[0m" + str(stem))

        try:
            with ExodusBasics(exofile) as exo:
                nod_vars = exo.nodal_varnames()
                if "disp_x" in nod_vars or "disp_y" in nod_vars:
                    log.warning("Displacement variables detected (disp_x/disp_y): coordinates will be updated per timestep.")
                else:
                    log.warning("No displacement variables found: using static mesh coordinates.")
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

                # Optional fixed view: bounds come from the first Exodus timestep
                # and include 10% padding on each side.
                plot_limits = None
                if args.whitespace:
                    x0, y0, z0, c0 = exo.xyzc_at_step(
                        args.var,
                        step=0,
                        eb=args.eb,
                    )
                    x0_plot = x0 * args.xscale

                    xmin, xmax = float(np.nanmin(x0_plot)), float(np.nanmax(x0_plot))
                    ymin, ymax = float(np.nanmin(y0)),      float(np.nanmax(y0))
                    xpad = 0.10 * (xmax - xmin)
                    ypad = 0.10 * (ymax - ymin)

                    plot_limits = (
                        xmin - xpad, xmax + xpad,
                        ymin - ypad, ymax + ypad,
                    )
                    log.info(
                        "Using fixed timestep-0 plot limits with 10%% padding: "
                        "x=[%.6g, %.6g], y=[%.6g, %.6g]",
                        *plot_limits,
                    )

                use_tqdm = (args.verbose == 0 and len(steps) > 1)
                frame_iter = tqdm(
                    steps,
                    desc=f"Plotting {args.var} in {stem}",
                    unit="frames",
                    leave=False,
                ) if use_tqdm else steps

                # Pre-pass: compute global vmin/vmax across all selected steps
                log.info("Computing global vmin/vmax across all selected steps...")
                vmin_global, vmax_global = np.inf, -np.inf
                for step in steps:
                    x, y, z, c = exo.xyzc_at_step(args.var, step, eb=args.eb)
                    vmin_global = min(vmin_global, float(np.nanmin(c)))
                    vmax_global = max(vmax_global, float(np.nanmax(c)))
                log.info(f"Global range: vmin={vmin_global:.6g}, vmax={vmax_global:.6g}")

                for i, step in enumerate(frame_iter):
                    if len(steps) == 1:
                        frame_name = f"{stem}_{args.var}{minimal_tag}_eb{args.eb}_step{step}"
                    else:
                        frame_name = f"{stem}_{args.var}{minimal_tag}_eb{args.eb}_{i:04d}"

                    log.info(
                        f"Plotting frame {i+1}/{len(steps)}: "
                        f"  step={step}, time={times[step]:.6g}, savename={frame_name}"
                    )

                    plot_exodus_var(
                        exo,
                        name=args.var,
                        step=step,
                        savename=frame_name,
                        eb=args.eb,
                        xscale=args.xscale,
                        vmin=vmin_global,
                        vmax=vmax_global,
                        method="auto",
                        show_axes=not args.no_axes,
                        show_colorbar=not args.no_colorbar,
                        show_title=not args.no_title,
                        open_plot=args.view,
                        dpi=args.dpi,
                        plot_limits=plot_limits,
                    )

                vtf(ti, log, "Finished plotting selected frame(s) ")
                log.info(" ")

                if len(exo_files) > 1:
                    tf(til, log, extra=f"File {cnt+1} ")

        except Exception as e:
            log.error("Failed in file %s:  %s: %s", exofile, type(e).__name__, e)
            sys.exit(2)

    if len(exo_files) > 1:
        log.warning(" ")
    tf(ti, log, extra="Total ")


if __name__ == "__main__":
    main()
