# extract_contours.py
"""
Extract iso-contours of a nodal variable from Exodus files and save to JSON.

Usage:
    python extract_contours.py --var sumetasqu --level 0.75 [options]

See --help for full options.
"""

from __future__ import annotations
from vector.ExodusBasics import ExodusBasics

import time
import json
import numpy as np
import argparse
import sys
import logging
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from pathlib import Path
from tqdm import tqdm

from plotting_general import (
    find_exodus_files,
    exodus_stem,
    setup_logging,
    tf,
    vtf,
)


def parse_args():
    p = argparse.ArgumentParser(
        description="Extract iso-contours from Exodus nodal variables and save to JSON.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---- General (matching plotting_general style) ----
    log = p.add_argument_group("General")
    log.add_argument("-v", "--verbose", action="count", default=0,
                     help="Increase verbosity (-v, -vv, -vvv).")
    log.add_argument("-s", "--subdirs", action="store_true",
                     help="Search for *.e files one level down (./*/*.e).")
    log.add_argument("--input", "-i", type=str, default=None, metavar="PATTERN",
                     help="Only process .e files whose name contains this string.")

    # ---- Contour options ----
    c = p.add_argument_group("Contour")
    c.add_argument("--var", "-p", type=str, default="sumetasqu",
                   help="Nodal variable name to contour.")
    c.add_argument("--level", type=float, default=0.75,
                   help="Iso-value to extract contour at.")

    return p.parse_args()


def extract_contours_at_step(
    x: np.ndarray,
    y: np.ndarray,
    c: np.ndarray,
    level: float,
) -> list[list[list[float]]]:
    """
    Use matplotlib tricontour on an unstructured triangulation to extract
    contour path segments at `level`.

    Returns a list of path segments. Each segment is a list of [x, y] pairs.
    Returns an empty list if no contour exists at this level for this step.
    """
    tri = mtri.Triangulation(x, y)

    # Temporary figure: we only want path geometry, not a rendered plot.
    fig, ax = plt.subplots()
    cs = ax.tricontour(tri, c, levels=[level])
    plt.close(fig)

    segments = []
    # cs.allsegs: list-of-lists, allsegs[i] = all disconnected paths for level i.
    # We requested exactly one level so allsegs[0] has everything.
    for seg in cs.allsegs[0]:
        # seg is ndarray shape (N, 2)
        segments.append(seg.tolist())

    return segments


def save_debug_plot(
    x: np.ndarray,
    y: np.ndarray,
    c: np.ndarray,
    segments: list,
    level: float,
    var: str,
    t: float,
    stem: str,
    outdir: Path,
) -> Path:
    """
    Save a debug plot of the nodal field with extracted contour overlaid.
    Only called once, for the first timestep of the first file.
    Saves to outdir/<stem>_<var>_contour_debug.png
    """
    tri = mtri.Triangulation(x, y)
    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)

    # Background: the nodal field rendered as a smooth tripcolor
    artist = ax.tripcolor(tri, c, cmap="viridis", shading="gouraud")
    fig.colorbar(artist, ax=ax, label=var)

    # Overlay: each extracted contour segment in red
    first = True
    for seg in segments:
        seg_arr = np.array(seg)
        ax.plot(
            seg_arr[:, 0], seg_arr[:, 1],
            color="red", lw=1.5,
            label=f"{var} = {level}" if first else "_nolegend_",
        )
        first = False

    if segments:
        ax.legend(fontsize=8)

    ax.set_title(f"{stem}  |  {var} = {level}  |  t = {t:.4g}  (step 0, debug)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="box")

    outdir.mkdir(parents=True, exist_ok=True)
    outfile = outdir / f"{stem}_{var}_contour_debug.png"
    fig.savefig(outfile, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return outfile


def main():
    ti = time.perf_counter()
    args = parse_args()
    log = setup_logging(args.verbose)

    exo_files = find_exodus_files(subdirs=args.subdirs, filter_str=args.input)
    if not exo_files:
        where = "subdirectories" if args.subdirs else "current directory"
        raise SystemExit(f"No .e files found in {where}.")

    log.info("Exodus Files Found:")
    for ef in exo_files:
        log.info(f"  {ef}")

    outdir = Path("pics")
    debug_saved = False  # debug plot saved once only: first file, first step

    for cnt, exofile in enumerate(exo_files):
        til = time.perf_counter()
        stem = exodus_stem(exofile)
        debug_saved = False  # reset per file: save one debug plot per exodus file

        if len(exo_files) > 1:
            log.warning(
                "\033[1m\033[96m" + f"File {cnt+1}/{len(exo_files)}: " + "\x1b[0m" + str(stem)
            )

        try:
            with ExodusBasics(exofile) as exo:

                # Validate that the requested variable exists before looping
                nodal_names = exo.nodal_varnames()
                if args.var not in nodal_names:
                    raise KeyError(
                        f"'{args.var}' not found in nodal variables. "
                        f"Available: {nodal_names}"
                    )

                times = exo.time()
                n_steps = len(times)

                # Read mesh coordinates once — mesh is static across timesteps [1]
                x, y = exo.coords_xy()

                log.info(f"  Nodes: {len(x)}, Steps: {n_steps}")

                all_steps = []

                use_tqdm = (args.verbose == 0)
                step_iter = tqdm(
                    range(n_steps),
                    desc=f"Extracting contours from {stem}",
                    unit="steps",
                    leave=False,
                ) if use_tqdm else range(n_steps)

                for step in step_iter:
                    t = float(times[step])
                    c = exo.nodal_var_at_step(args.var, step)  # shape (n_nodes,) [1]

                    segments = extract_contours_at_step(x, y, c, args.level)

                    log.info(
                        f"  step={step:>4d}, time={t:.6g}, "
                        f"contour_segments={len(segments)}"
                    )

                    all_steps.append({
                        "step": step,
                        "time": t,
                        # Each entry is one disconnected contour path: [[x0,y0], [x1,y1], ...]
                        # Multiple entries mean multiple disconnected iso-lines at this level.
                        "contours": segments,
                    })

                    # Debug plot: first timestep of the very first file only
                    if not debug_saved:
                        outfile = save_debug_plot(
                            x, y, c, segments,
                            level=args.level,
                            var=args.var,
                            t=t,
                            stem=stem,
                            outdir=outdir,
                        )
                        log.warning(f"Debug plot saved: {outfile}")
                        debug_saved = True

                # Assemble full output structure
                output = {
                    "file": str(exofile),
                    "variable": args.var,
                    "level": args.level,
                    "steps": all_steps,
                }

                json_path = Path(f"{stem}_{args.var}_contours.json")
                with open(json_path, "w") as f:
                    json.dump(output, f, indent=2)

                log.warning(f"Contours saved → {json_path}  ({n_steps} steps)")
                vtf(til, log, f"File {cnt+1} ")

        except Exception as e:
            log.error("Failed in file %s:  %s: %s", exofile, type(e).__name__, e)
            sys.exit(2)

    if len(exo_files) > 1:
        log.warning(" ")
    tf(ti, log, "Total ")


if __name__ == "__main__":
    main()
