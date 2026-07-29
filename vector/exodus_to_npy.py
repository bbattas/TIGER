#!/usr/bin/env python3
"""
exodus_to_npy.py

Reads one or more ExodusII (.e) files, extracts the `unique_grains` elemental
variable across all timesteps, maps values onto a regular 3D spatial grid,
and saves the result as a .npy file with shape (frames, x, y, z).

Usage:
    # Single file (explicit):
    python exodus_to_npy.py simulation.e

    # All .e files in the current directory:
    python exodus_to_npy.py

    # All .e files one subdirectory level down:
    python exodus_to_npy.py --subdirs

    # Verbose output:
    python exodus_to_npy.py -v
    python exodus_to_npy.py --subdirs -v
"""

import argparse
import logging
import re
import sys
from pathlib import Path

import numpy as np

from ExodusBasics import ExodusBasics

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
VAR_NAME     = "unique_grains"
QUANTIZE_TOL = 1e-6   # snap float coords to avoid degenerate uniqueness
EB           = 1      # default element block

logger = logging.getLogger("exodus_to_npy")


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def setup_logging(verbose: bool) -> None:
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(
        format="%(levelname)s: %(message)s",
        level=level,
    )
    logger.setLevel(level)


# ---------------------------------------------------------------------------
# Output name derivation
# ---------------------------------------------------------------------------
def derive_output_path(input_path: Path) -> Path:
    """
    Strip trailing _out or _exodus from the stem, then append .npy.

    Examples:
        mysim_out.e     -> mysim.npy
        mysim_exodus.e  -> mysim.npy
        mysim.e         -> mysim.npy
    """
    stem = re.sub(r"_(out|exodus)$", "", input_path.stem)
    output = input_path.parent / f"{stem}.npy"
    logger.info("Output path derived: %s -> %s", input_path.name, output)
    return output


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------
def discover_files(args: argparse.Namespace) -> list[Path]:
    """Return a sorted list of .e files to process."""
    if args.subdirs:
        files = sorted(Path(".").glob("*/*.e"))
        logger.info("Subdirectory search found %d .e file(s).", len(files))
        for f in files:
            logger.info("  Found: %s", f)
        if not files:
            logger.warning("No .e files found in any subdirectory.")
        return files

    if args.input:
        p = Path(args.input)
        if not p.is_file():
            logger.warning("Specified input file does not exist: %s", p)
            return []
        if p.suffix != ".e":
            logger.warning("Specified file does not have a .e extension: %s", p)
        logger.info("Single file mode: %s", p)
        return [p]

    # Default: all .e files in CWD
    files = sorted(Path(".").glob("*.e"))
    logger.info("CWD search found %d .e file(s).", len(files))
    for f in files:
        logger.info("  Found: %s", f)
    if not files:
        logger.warning("No .e files found in the current directory.")
    return files


# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------
def build_index_map(
    xc: np.ndarray,
    yc: np.ndarray,
    zc: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int, int]:
    """
    Map flat element coordinates to 3D grid indices (ix, iy, iz).

    Uses np.unique with return_inverse to assign each unique coordinate
    value a consecutive integer index. Works for both 2D meshes (nz==1)
    and full 3D meshes.

    Returns:
        ix, iy, iz : integer index arrays of shape (n_elements,)
        nx, ny, nz : grid dimensions
    """
    _, ix = np.unique(xc, return_inverse=True)
    _, iy = np.unique(yc, return_inverse=True)
    _, iz = np.unique(zc, return_inverse=True)

    nx = int(ix.max()) + 1
    ny = int(iy.max()) + 1
    nz = int(iz.max()) + 1

    expected = nx * ny * nz
    actual   = len(xc)

    if expected != actual:
        logger.warning(
            "Grid size mismatch: nx=%d * ny=%d * nz=%d = %d, "
            "but element count is %d. Mesh may not be a perfect structured grid. "
            "Some grid cells may be empty (zero-filled).",
            nx, ny, nz, expected, actual,
        )
    else:
        logger.info("Grid dimensions: nx=%d, ny=%d, nz=%d", nx, ny, nz)

    return ix, iy, iz, nx, ny, nz


def process_file(path: Path) -> None:
    """Open one Exodus file, extract unique_grains, and save as .npy."""
    logger.info("=" * 60)
    logger.info("Processing: %s", path)

    output_path = derive_output_path(path)

    if output_path.exists():
        logger.warning("Output file already exists and will be overwritten: %s", output_path)

    with ExodusBasics(str(path)) as exo:

        # ---- Validate variable presence --------------------------------
        elem_vars = exo.elem_varnames()
        logger.info("Element variables in file: %s", elem_vars)

        if VAR_NAME not in elem_vars:
            logger.warning(
                "Variable '%s' not found in %s. Available element vars: %s. Skipping.",
                VAR_NAME, path.name, elem_vars,
            )
            return

        # ---- Check for multiple element blocks -------------------------
        block_names = exo.connect_varnames()
        logger.info("Element blocks found: %s", block_names)
        if len(block_names) > 1:
            logger.warning(
                "%s contains %d element blocks. Processing block eb=%d only.",
                path.name, len(block_names), EB,
            )

        # ---- Timesteps -------------------------------------------------
        times     = exo.time()
        n_frames  = len(times)
        logger.info("Timesteps: %d", n_frames)

        if n_frames == 0:
            logger.warning("No timesteps found in %s. Skipping.", path.name)
            return

        # ---- Grid construction -----------------------------------------
        logger.info("Computing element centers (eb=%d, method='mean', quantize_tol=%s)...", EB, QUANTIZE_TOL)
        xc, yc, zc = exo.element_centers_xyz(
            eb=EB,
            method="mean",
            quantize_tol=QUANTIZE_TOL,
        )
        logger.info("Element count in block eb=%d: %d", EB, len(xc))

        ix, iy, iz, nx, ny, nz = build_index_map(xc, yc, zc)

        # Check for duplicate (ix, iy, iz) assignments
        coords = np.stack([ix, iy, iz], axis=1)
        unique_coords, counts = np.unique(coords, axis=0, return_counts=True)
        duplicates = counts[counts > 1]
        if len(duplicates) > 0:
            logger.warning("Found %d grid cells with multiple elements mapped to them!", len(duplicates))

        # ---- Allocate output -------------------------------------------
        out = np.zeros((n_frames, nx, ny, nz), dtype=np.int32)
        logger.info("Allocated output array: shape=%s, dtype=%s", out.shape, out.dtype)

        # ---- Fill frames -----------------------------------------------
        for step in range(n_frames):
            vals = exo.elem_var_at_step(VAR_NAME, step, eb=EB)
            out[step, ix, iy, iz] = vals.astype(np.int32)
            logger.info("  Frame %d / %d done.", step + 1, n_frames)

        # ---- Save ------------------------------------------------------
        np.save(output_path, out)

    logger.warning("Saved: %s  shape=%s", output_path, out.shape)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract 'unique_grains' from ExodusII (.e) file(s) and save "
            "as .npy arrays with shape (frames, x, y, z)."
        )
    )
    parser.add_argument(
        "input",
        nargs="?",
        default=None,
        help="Path to a single .e file. If omitted, all .e files in CWD are used.",
    )
    parser.add_argument(
        "-s", "--subdirs",
        action="store_true",
        help="Search one level of subdirectories for .e files and process each.",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose (INFO-level) logging.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)

    files = discover_files(args)

    if not files:
        logger.warning("No files to process. Exiting.")
        sys.exit(0)

    logger.warning("Processing %d file(s)...", len(files))

    for path in files:
        try:
            process_file(path)
        except Exception as exc:
            logger.warning("Failed to process %s: %s", path, exc, exc_info=args.verbose)

    logger.warning("Done.")


if __name__ == "__main__":
    main()
