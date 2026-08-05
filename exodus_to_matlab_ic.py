#!/usr/bin/env python3
"""
exodus_to_matlab_ic.py

Read grN nodal variables from a MOOSE ExodusII output at a selected
timestep and save them as a .mat file for use as a MATLAB phase-field
initial condition.

Each variable is reshaped onto a 2D grid with axis 0 = x and axis 1 = y,
matching the convention of the MATLAB solver
(anisotropic_grgr_incl_in_gamma.m).

Grid modes
----------
all-nodes
    Preserve every Exodus node. A QUAD4 mesh and corresponding QUAD9 mesh
    will generally produce different numerical grid dimensions.

vertices
    Use only element corner nodes. For matching QUAD4 and QUAD9 meshes this
    produces the same first-order-equivalent grid without interpolation.

Usage examples
--------------
python exodus_to_matlab_ic.py
python exodus_to_matlab_ic.py -t 0.5
python exodus_to_matlab_ic.py -i myjob -t 1.0
python exodus_to_matlab_ic.py -s
python exodus_to_matlab_ic.py -o my_ic.mat
python exodus_to_matlab_ic.py -n 4
python exodus_to_matlab_ic.py --grid-mode vertices -t 0.01 -vv
"""
from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

import numpy as np
import scipy.io
from tqdm import tqdm

try:
    from vector.ExodusBasics import ExodusBasics
except ModuleNotFoundError as exc:
    # Allow this script and ExodusBasics.py to be run side-by-side while still
    # preserving the normal project import when the vector package is present.
    if exc.name != "vector":
        raise
    from ExodusBasics import ExodusBasics


# Connectivity ordering for the supported quadrilateral element families:
# the first four entries are the element corner nodes in Exodus convention.
CORNER_NODES_PER_ELEMENT = {
    "QUAD4": 4,
    "QUAD8": 4,
    "QUAD9": 4,
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Convert MOOSE Exodus grN nodal variables to a MATLAB .mat IC file."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # General
    gen = p.add_argument_group("General")
    gen.add_argument(
        "-v", "--verbose", action="count", default=0,
        help="Increase verbosity (-v, -vv, -vvv).",
    )
    gen.add_argument(
        "-s", "--subdirs", action="store_true",
        help="Search for *.e files one level down (./*/*.e).",
    )
    gen.add_argument(
        "--input", "-i", type=str, default=None, metavar="PATTERN",
        help="Only process .e files whose name contains this string.",
    )
    gen.add_argument(
        "-n", "--num-grains", type=int, default=None, metavar="N",
        help=(
            "Maximum number of gr* order parameters to extract. "
            "Defaults to all grN variables found in the Exodus file."
        ),
    )

    # Frame selection
    tim = p.add_argument_group("Target frame selection")
    tim.add_argument(
        "-t", "--time", type=float, default=None,
        help=(
            "Target Exodus time value; picks the closest available frame. "
            "If not specified, defaults to the frame closest to t=0."
        ),
    )

    # Output grid
    mesh = p.add_argument_group("Output grid")
    mesh.add_argument(
        "--grid-mode",
        choices=("all-nodes", "vertices"),
        default="all-nodes",
        help=(
            "'all-nodes' uses every Exodus node. 'vertices' uses only element "
            "corner nodes, reducing higher-order quadrilateral elements to "
            "their first-order-equivalent nodal grid without interpolation."
        ),
    )

    # Output
    out = p.add_argument_group("Output")
    out.add_argument(
        "-o", "--output", type=str, default=None,
        help=(
            "Output .mat filename. Defaults to "
            "'<exodus_stem>_ic_<N>gr_t<time>.mat' in the current directory."
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
# File discovery
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
# Step selection
# ---------------------------------------------------------------------------

def closest_index(values: np.ndarray, target: float) -> int:
    """Return index of entry closest to target. Ties select the first entry."""
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

def discover_gr_names(available: list[str]) -> list[str]:
    """Return a numerically sorted list of all grN variables in available."""
    return sorted(
        [n for n in available if re.fullmatch(r"gr\d+", n)],
        key=lambda s: int(s[2:]),
    )


def resolve_gr_names(
    available: list[str],
    num_grains: int | None,
    log: logging.Logger,
) -> list[str]:
    """
    Discover all grN variables, apply an optional count cap, and raise only
    when no grN variables are available.
    """
    found = discover_gr_names(available)

    if not found:
        raise RuntimeError(
            f"No grN nodal variables found in file. Available vars: {available}"
        )

    if num_grains is not None and num_grains < len(found):
        log.warning(
            f"WARNING: --num-grains={num_grains} requested but {len(found)} "
            f"gr* variables found; truncating to {found[:num_grains]}."
        )
        found = found[:num_grains]
    elif num_grains is not None and num_grains > len(found):
        log.warning(
            f"WARNING: --num-grains={num_grains} requested but only "
            f"{len(found)} gr* variable(s) found in file: {found}. "
            "Proceeding with available variables only."
        )
    else:
        log.info(f"Auto-detected gr* variables: {found}")

    log.warning(f"Extracting {len(found)} order parameter(s): {found}")
    return found


def canonical_element_type(elem_type: str) -> str:
    """Normalize common separators so, for example, QUAD_9 becomes QUAD9."""
    return re.sub(r"[\s_-]+", "", elem_type.strip().upper())


def select_grid_node_ids(
    exo: ExodusBasics,
    grid_mode: str,
    log: logging.Logger,
    eb: int = 1,
) -> tuple[np.ndarray, dict[str, object]]:
    """
    Select zero-based global node IDs for the requested output-grid mode.

    ``all-nodes`` returns every Exodus node. ``vertices`` collects the unique
    corner-node IDs from the element connectivity. No interpolation is used.

    Returns
    -------
    node_ids, metadata
        ``node_ids`` is a one-dimensional integer array. ``metadata`` contains
        mesh and selection information suitable for verbose logging and MAT
        output metadata.
    """
    all_x, all_y = exo.coords_xy()
    source_num_nodes = len(all_x)
    if len(all_y) != source_num_nodes:
        raise RuntimeError(
            f"Coordinate length mismatch: len(x)={source_num_nodes}, "
            f"len(y)={len(all_y)}."
        )

    conn = np.asarray(exo.connectivity(which=eb, zero_based=True))
    if conn.ndim != 2:
        raise RuntimeError(
            f"Expected connect{eb} to be 2D; found connectivity shape {conn.shape}."
        )

    elem_type_raw = exo.element_type(eb)
    elem_type = canonical_element_type(elem_type_raw)
    num_elements, nodes_per_element = map(int, conn.shape)

    if conn.size:
        conn_min = int(conn.min())
        conn_max = int(conn.max())
        if conn_min < 0 or conn_max >= source_num_nodes:
            raise RuntimeError(
                f"connect{eb} contains node IDs outside the valid zero-based "
                f"range [0, {source_num_nodes - 1}]: min={conn_min}, max={conn_max}."
            )

    corner_count = CORNER_NODES_PER_ELEMENT.get(elem_type)

    if grid_mode == "all-nodes":
        node_ids = np.arange(source_num_nodes, dtype=np.int64)
    elif grid_mode == "vertices":
        if corner_count is None:
            supported = ", ".join(sorted(CORNER_NODES_PER_ELEMENT))
            raise RuntimeError(
                f"--grid-mode vertices does not currently support element type "
                f"{elem_type_raw!r}. Supported types: {supported}."
            )
        if nodes_per_element < corner_count:
            raise RuntimeError(
                f"Element type {elem_type_raw!r} requires {corner_count} corner "
                f"entries, but connect{eb} has only {nodes_per_element} nodes "
                "per element."
            )

        # Exodus QUAD4/8/9 connectivity stores the four corner nodes first.
        node_ids = np.unique(conn[:, :corner_count].reshape(-1)).astype(
            np.int64, copy=False
        )
    else:
        raise ValueError(
            f"Unknown grid_mode {grid_mode!r}; expected 'all-nodes' or 'vertices'."
        )

    source_nx = int(len(np.unique(all_x)))
    source_ny = int(len(np.unique(all_y)))
    output_num_nodes = int(len(node_ids))
    excluded_nodes = int(source_num_nodes - output_num_nodes)

    metadata: dict[str, object] = {
        "grid_mode": grid_mode,
        "element_block": int(eb),
        "element_type": elem_type_raw,
        "element_type_canonical": elem_type,
        "num_elements": num_elements,
        "nodes_per_element": nodes_per_element,
        "corner_nodes_per_element": (
            int(corner_count) if corner_count is not None else -1
        ),
        "source_num_nodes": int(source_num_nodes),
        "output_num_nodes": output_num_nodes,
        "excluded_nodes": excluded_nodes,
        "source_nx": source_nx,
        "source_ny": source_ny,
    }

    log.info(f"Element block: {eb}")
    log.info(f"Element type: {elem_type_raw}")
    log.info(f"Elements: {num_elements}")
    log.info(f"Nodes per element: {nodes_per_element}")
    log.info(f"Source nodes: {source_num_nodes}")
    log.info(f"Source coordinate grid: {source_nx} x {source_ny}")
    log.info(f"Grid mode: {grid_mode}")
    if grid_mode == "vertices":
        log.info(f"Corner nodes per element: {corner_count}")
        log.info(f"Unique selected vertex nodes: {output_num_nodes}")
    else:
        log.info(f"Selected nodes: {output_num_nodes}")
    log.info(f"Excluded nodes: {excluded_nodes}")

    return node_ids, metadata


def build_eta_grids(
    exo: ExodusBasics,
    step: int,
    gr_names: list[str],
    log: logging.Logger,
    *,
    node_ids: np.ndarray | None = None,
    grid_mode: str = "all-nodes",
) -> dict[str, np.ndarray]:
    """
    Read each variable at ``step`` and scatter selected nodal values onto a
    structured 2D grid with shape ``(nx, ny)``.

    Axis 0 is x and axis 1 is y. If ``node_ids`` is omitted, all nodes are
    used, preserving the original behavior.
    """
    all_x, all_y = exo.coords_xy()
    source_num_nodes = len(all_x)

    if node_ids is None:
        node_ids = np.arange(source_num_nodes, dtype=np.int64)
    else:
        node_ids = np.asarray(node_ids, dtype=np.int64)

    if node_ids.ndim != 1:
        raise ValueError(f"node_ids must be one-dimensional; found {node_ids.shape}.")
    if node_ids.size == 0:
        raise RuntimeError("The selected output grid contains no nodes.")
    if int(node_ids.min()) < 0 or int(node_ids.max()) >= source_num_nodes:
        raise IndexError(
            f"Selected node IDs are outside [0, {source_num_nodes - 1}]."
        )
    if len(np.unique(node_ids)) != len(node_ids):
        raise RuntimeError("Selected node IDs contain duplicates.")

    x = np.asarray(all_x)[node_ids]
    y = np.asarray(all_y)[node_ids]

    coordinate_pairs = np.column_stack((x, y))
    unique_pairs = np.unique(coordinate_pairs, axis=0)
    if len(unique_pairs) != len(node_ids):
        raise RuntimeError(
            "Multiple selected nodes occupy the same x-y coordinate. "
            f"Selected nodes={len(node_ids)}, unique coordinates={len(unique_pairs)}."
        )

    xu = np.unique(x)
    yu = np.unique(y)
    nx, ny = len(xu), len(yu)

    log.info(
        f"Output grid: nx={nx}, ny={ny}, total cells={nx * ny}, "
        f"selected nodes={len(node_ids)}"
    )
    if nx == ny:
        log.info("Output grid is square: yes")
    else:
        log.warning(f"WARNING: output grid is not square: nx={nx}, ny={ny}")

    if nx * ny != len(node_ids):
        raise RuntimeError(
            "Selected nodes do not form a complete structured rectangular grid: "
            f"grid_mode={grid_mode}, nx={nx}, ny={ny}, nx*ny={nx * ny}, "
            f"selected_nodes={len(node_ids)}."
        )

    # Map each selected node to its structured-grid index.
    ix = np.searchsorted(xu, x)
    iy = np.searchsorted(yu, y)

    grids: dict[str, np.ndarray] = {}

    for gr_name in tqdm(gr_names, desc="Reading order parameters", unit="var"):
        all_vals = np.asarray(exo.nodal_var_at_step(gr_name, step))
        if len(all_vals) != source_num_nodes:
            raise RuntimeError(
                f"{gr_name}: nodal value count {len(all_vals)} does not match "
                f"coordinate count {source_num_nodes}."
            )
        vals = all_vals[node_ids]

        # Fill grid: G[ix, iy] gives axis 0 = x and axis 1 = y.
        G = np.full((nx, ny), np.nan, dtype=np.float64)
        G[ix, iy] = vals

        if np.isnan(G).any():
            n_missing = int(np.isnan(G).sum())
            raise RuntimeError(
                f"{gr_name}: {n_missing} grid cells are unfilled after scatter. "
                "Coordinate uniqueness or node selection may be wrong."
            )

        grids[gr_name] = G
        log.info(
            f"  {gr_name}: shape={G.shape}, min={float(vals.min()):.6g}, "
            f"max={float(vals.max()):.6g}, mean={float(vals.mean()):.6g}"
        )

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
    metadata: dict[str, object],
    log: logging.Logger,
) -> None:
    """
    Save order-parameter grids and simple mesh/conversion metadata to a MATLAB
    v5 MAT file.
    """
    save_dict: dict[str, object] = {
        name: G.astype(np.float64, copy=False) for name, G in grids.items()
    }

    string_metadata = (
        "source_file",
        "grid_mode",
        "element_type",
        "element_type_canonical",
    )
    integer_metadata = (
        "exodus_step",
        "nx",
        "ny",
        "element_block",
        "num_elements",
        "nodes_per_element",
        "corner_nodes_per_element",
        "source_num_nodes",
        "output_num_nodes",
        "excluded_nodes",
        "source_nx",
        "source_ny",
    )

    for key in string_metadata:
        if key in metadata:
            save_dict[key] = str(metadata[key])
    for key in integer_metadata:
        if key in metadata:
            save_dict[key] = int(metadata[key])

    save_dict["exodus_time"] = float(metadata["exodus_time"])

    outpath.parent.mkdir(parents=True, exist_ok=True)
    scipy.io.savemat(str(outpath), save_dict, format="5", do_compression=False)
    log.warning(f"Saved: {outpath}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    log = setup_logging(args.verbose)

    exo_files = find_exodus_files(subdirs=args.subdirs, filter_str=args.input)
    if not exo_files:
        where = "subdirectories" if args.subdirs else "current directory"
        raise SystemExit(f"No .e files found in {where}.")

    log.info(f"Found {len(exo_files)} Exodus file(s):")
    for ef in exo_files:
        log.info(f"  {ef}")

    if len(exo_files) > 1:
        if args.output:
            raise SystemExit(
                "--output may only be used when exactly one Exodus file is selected. "
                "Use --input to select one file or omit --output."
            )
        log.warning(
            f"Multiple .e files found; processing all {len(exo_files)}. "
            "Use --input to filter to a specific file."
        )

    for exo_path in exo_files:
        stem = exodus_stem(exo_path)
        log.warning(f"\nProcessing: {exo_path.name}")

        try:
            with ExodusBasics(exo_path) as exo:
                available = exo.nodal_varnames()
                log.info(f"Available nodal variables: {available}")
                gr_names = resolve_gr_names(available, args.num_grains, log)

                times = exo.time()
                log.info(f"Available timesteps: {len(times)}, times: {times[:]}")

                target_time = args.time if args.time is not None else 0.0
                step = select_step(times, target_time, log)
                actual_time = float(times[step])
                log.warning(
                    f"Selected step={step}, time={actual_time:.6g}"
                    + (" (default: closest to t=0)" if args.time is None else "")
                )

                node_ids, grid_metadata = select_grid_node_ids(
                    exo,
                    grid_mode=args.grid_mode,
                    log=log,
                    eb=1,
                )

                grids = build_eta_grids(
                    exo,
                    step,
                    gr_names,
                    log,
                    node_ids=node_ids,
                    grid_mode=args.grid_mode,
                )

                if not args.no_validate:
                    validate_range(grids, log)

                if args.output:
                    outpath = Path(args.output)
                else:
                    outpath = Path(
                        f"{stem}_ic_{len(gr_names)}gr_t{actual_time:.6g}.mat"
                    )

                nx, ny = next(iter(grids.values())).shape
                metadata: dict[str, object] = {
                    "source_file": exo_path.name,
                    "exodus_time": actual_time,
                    "exodus_step": step,
                    "nx": nx,
                    "ny": ny,
                    **grid_metadata,
                }
                save_mat(grids, outpath, metadata=metadata, log=log)

        except Exception as exc:
            log.error(
                "Failed on file %s: %s: %s",
                exo_path,
                type(exc).__name__,
                exc,
            )
            sys.exit(2)

    log.warning("Done.")


if __name__ == "__main__":
    main()
