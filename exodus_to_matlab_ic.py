#!/usr/bin/env python3
"""
exodus_to_matlab_ic.py

Read gr0-gr5 nodal variables from a MOOSE ExodusII output at a selected
timestep and save them as a .mat file for use as a MATLAB phase-field
initial condition.

Each variable is reshaped onto a 2D grid with axis 0 = x, axis 1 = y,
matching the convention of the MATLAB solver (anisotropic_grgr_incl_in_gamma.m).

Usage:
    python exodus_to_matlab_ic.py                      # closest frame to t=0
    python exodus_to_matlab_ic.py -t 0.5               # closest frame to t=0.5
    python exodus_to_matlab_ic.py -i myjob -t 1.0      # filter by filename
    python exodus_to_matlab_ic.py -s                   # search subdirectories
    python exodus_to_matlab_ic.py -o my_ic.mat         # custom output filename
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import scipy.io
from tqdm import tqdm

from vector.ExodusBasics import ExodusBasics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert MOOSE Exodus gr0-gr5 nodal variables to a MATLAB .mat IC file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # General
    gen = p.add_argument_group("General")
    gen.add_argument("-v", "--verbose", action="count", default=0,
                     help="Increase verbosity (-v, -vv, -vvv).")
    gen.add_argument("-s", "--subdirs", action="store_true",
                     help="Search for *.e files one level down (./*/*.e).")
    gen.add_argument("--input", "-i", type=str, default=None, metavar="PATTERN",
                     help="Only process .e files whose name contains this string.")

    # Frame selection
    tim = p.add_argument_group("Target frame selection")
    tim.add_argument(
        "-t", "--time", type=float, default=None,
        help=(
            "Target exodus time value; picks the closest available frame. "
            "If not specified, defaults to the frame closest to t=0."
        ),
    )

    # Output
    out = p.add_argument_group("Output")
    out.add_argument(
        "-o", "--output", type=str, default=None,
        help=(
            "Output .mat filename. Defaults to '<exodus_stem>_ic_t<time>.mat' "
            "in the current directory."
        ),
    )
    out.add_argument(
        "--no-validate", action="store_true",
        help="Skip the [0, 1] range validation check on order parameter values.",
    )

    return p.parse_args()


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(verbosity: int) -> logging.Logger:
    level = logging.WARNING
    if verbosity >= 2:
        level = logging.DEBUG
    elif verbosity >= 1:
        level = logging.INFO
    logging.basicConfig(level=level, format="%(message)s")
    return logging.getLogger("exo2mat")


# ---------------------------------------------------------------------------
# File discovery  (mirrors plotting_general.py pattern)
# ---------------------------------------------------------------------------

def find_exodus_files(
    *,
    subdirs: bool = False,
    pattern: str = "*.e",
    filter_str: str | None = None,
) -> list[Path]:
    cwd = Path.cwd()
    glob_pattern = f"*{filter_str}*.e" if filter_str else pattern
    if subdirs:
        files = sorted(cwd.glob(f"*/{glob_pattern}"))
    else:
        files = sorted(cwd.glob(glob_pattern))
    return [p for p in files if p.is_file()]


def exodus_stem(exo_path: Path) -> str:
    name = exo_path.name
    if name.endswith(".e"):
        name = name[:-2]
    if name.endswith("_out"):
        name = name[:-4]
    return name


# ---------------------------------------------------------------------------
# Step selection  (mirrors plotting_general.py closest_index pattern)
# ---------------------------------------------------------------------------

def closest_index(values: np.ndarray, target: float) -> int:
    """Return index of entry closest to target. Ties -> first occurrence."""
    values = np.asarray(values)
    return int(np.argmin(np.abs(values - target)))


def select_step(times: np.ndarray, target_time: float, log: logging.Logger) -> int:
    step = closest_index(times, target_time)
    log.info(
        f"Frame selected: requested t={target_time}, "
        f"chosen step={step}, actual time={float(times[step]):.6g}"
    )
    return step


# ---------------------------------------------------------------------------
# Core: Exodus -> structured 2D grids
# ---------------------------------------------------------------------------

GR_NAMES = ["gr0", "gr1", "gr2", "gr3", "gr4", "gr5"]


def validate_varnames(available: list[str], log: logging.Logger) -> None:
    missing = [n for n in GR_NAMES if n not in available]
    if missing:
        raise RuntimeError(
            f"Expected nodal variables {GR_NAMES} but these are missing: {missing}. "
            f"Available nodal vars: {available}"
        )
    log.info(f"Confirmed nodal variables: {GR_NAMES}")


def build_eta_grids(
    exo: ExodusBasics,
    step: int,
    log: logging.Logger,
) -> dict[str, np.ndarray]:
    """
    Read gr0-gr5 at `step` and scatter each flat nodal array onto a
    structured 2D grid with shape (nx, ny) — axis 0 = x, axis 1 = y —
    matching the MATLAB solver index convention.

    Returns dict: {'gr0': array(nx,ny), ..., 'gr5': array(nx,ny)}
    """
    x, y = exo.coords_xy()  # flat arrays, length = num_nodes

    # Build sorted unique coordinate axes
    xu = np.unique(x)
    yu = np.unique(y)
    nx, ny = len(xu), len(yu)

    log.info(f"Grid: nx={nx}, ny={ny}, total nodes={nx*ny} (file has {len(x)})")

    if nx * ny != len(x):
        raise RuntimeError(
            f"Node count mismatch: nx*ny={nx*ny} but Exodus reports {len(x)} nodes. "
            "Mesh may not be a full structured rectangular grid."
        )

    # Map each node to its (ix, iy) grid index
    ix = np.searchsorted(xu, x)  # shape (num_nodes,)
    iy = np.searchsorted(yu, y)  # shape (num_nodes,)

    grids: dict[str, np.ndarray] = {}

    for gr_name in tqdm(GR_NAMES, desc="Reading order parameters", unit="var"):
        vals = exo.nodal_var_at_step(gr_name, step)   # shape (num_nodes,)

        # Fill grid: G[ix, iy] -> axis 0 = x direction, axis 1 = y direction
        # This matches the MATLAB solver which left-multiplies rows for x-derivatives
        G = np.full((nx, ny), np.nan)
        G[ix, iy] = vals

        if np.isnan(G).any():
            n_missing = int(np.isnan(G).sum())
            raise RuntimeError(
                f"{gr_name}: {n_missing} grid cells are unfilled after scatter. "
                "Coordinate uniqueness or node count may be wrong."
            )

        grids[gr_name] = G
        log.info(f"  {gr_name}: shape={G.shape}, min={vals.min():.4f}, max={vals.max():.4f}")

    return grids


def validate_range(grids: dict[str, np.ndarray], log: logging.Logger) -> None:
    """Warn if any order parameter has values outside [0, 1]."""
    for name, G in grids.items():
        vmin, vmax = float(G.min()), float(G.max())
        if vmin < -1e-6 or vmax > 1.0 + 1e-6:
            log.warning(
                f"WARNING: {name} has values outside [0, 1]: "
                f"min={vmin:.6f}, max={vmax:.6f}. "
                "Small violations near grain boundaries are expected; "
                "large violations may indicate a problem."
            )
        else:
            log.info(f"  {name}: range OK [{vmin:.4f}, {vmax:.4f}]")


# ---------------------------------------------------------------------------
# Save to .mat
# ---------------------------------------------------------------------------

def save_mat(
    grids: dict[str, np.ndarray],
    outpath: Path,
    metadata: dict,
    log: logging.Logger,
) -> None:
    """
    Save order parameter grids to a MATLAB v5 .mat file.

    The saved variables are named gr0..gr5 directly so MATLAB can load
    and assign them:
        data = load('ic_from_moose.mat');
        eta{1} = data.gr0;  % etc.

    Metadata (source file, time, step) is saved as a struct field.
    """
    save_dict: dict = {}
    for name, G in grids.items():
        save_dict[name] = G.astype(np.float64)

    # Pack metadata as simple scalars/strings (scipy.io compatible)
    save_dict["source_file"] = str(metadata["source_file"])
    save_dict["exodus_time"] = float(metadata["exodus_time"])
    save_dict["exodus_step"] = int(metadata["exodus_step"])
    save_dict["nx"] = int(metadata["nx"])
    save_dict["ny"] = int(metadata["ny"])

    scipy.io.savemat(str(outpath), save_dict, format="5", do_compression=False)
    log.warning(f"Saved: {outpath}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    log = setup_logging(args.verbose)

    # --- Find Exodus file(s) ---
    exo_files = find_exodus_files(subdirs=args.subdirs, filter_str=args.input)
    if not exo_files:
        where = "subdirectories" if args.subdirs else "current directory"
        raise SystemExit(f"No .e files found in {where}.")

    log.info(f"Found {len(exo_files)} Exodus file(s):")
    for ef in exo_files:
        log.info(f"  {ef}")

    if len(exo_files) > 1:
        log.warning(
            f"Multiple .e files found; processing all {len(exo_files)}. "
            "Use --input to filter to a specific file."
        )

    for exo_path in exo_files:
        stem = exodus_stem(exo_path)
        log.warning(f"\nProcessing: {exo_path.name}")

        try:
            with ExodusBasics(exo_path) as exo:

                # --- Validate variable names ---
                available = exo.nodal_varnames()
                log.info(f"Available nodal variables: {available}")
                validate_varnames(available, log)

                # --- Select timestep ---
                times = exo.time()
                log.info(f"Available timesteps: {len(times)}, times: {times[:]}")

                target_time = args.time if args.time is not None else 0.0
                step = select_step(times, target_time, log)
                actual_time = float(times[step])
                log.warning(
                    f"Selected step={step}, time={actual_time:.6g}"
                    + (" (default: closest to t=0)" if args.time is None else "")
                )

                # --- Build 2D grids ---
                grids = build_eta_grids(exo, step, log)

                # --- Validate value ranges ---
                if not args.no_validate:
                    validate_range(grids, log)

                # --- Determine output path ---
                if args.output:
                    outpath = Path(args.output)
                else:
                    outpath = Path(f"{stem}_ic_t{actual_time:.6g}.mat")

                # --- Save ---
                nx, ny = next(iter(grids.values())).shape
                save_mat(
                    grids,
                    outpath,
                    metadata={
                        "source_file": exo_path.name,
                        "exodus_time": actual_time,
                        "exodus_step": step,
                        "nx": nx,
                        "ny": ny,
                    },
                    log=log,
                )

        except Exception as e:
            log.error("Failed on file %s:  %s: %s", exo_path, type(e).__name__, e)
            sys.exit(2)

    log.warning("Done.")


if __name__ == "__main__":
    main()
