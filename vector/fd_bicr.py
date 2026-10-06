#!/usr/bin/env python3
"""
Measure bicrystal geometry and grain-boundary inclination from ExodusII files.

For every timestep, this script writes:

    <stem>_bicrystal_metrics.csv
        time, area, x_length, y_length, aspect_ratio

    <stem>_inclination.csv
        time, step, angle_deg, count, probability_density

It also writes a polar inclination plot for one selected timestep:

    <stem>_inclination_stepXXXXXX.png

Assumptions
-----------
1. The simulation is two-dimensional.
2. The mesh is static and consists of TRI or QUAD elements.
3. Element centers form a complete rectangular structured grid.
4. ``unique_grains`` contains categorical grain IDs.
5. The center grain has the same grain ID throughout the simulation.
6. PACKAGE_MP_Linear_Vectorized.py and myInput.py are importable.
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import os
import sys
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
import numpy as np
from tqdm import tqdm

from ExodusBasics import ExodusBasics
import PACKAGE_MP_Linear_Vectorized as smooth


LOGGER_NAME = "bicrystal_analysis"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Measure center-grain area, dimensions, aspect ratio, and "
            "grain-boundary inclination distributions from bicrystal "
            "ExodusII simulations."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "files",
        nargs="*",
        type=Path,
        help=(
            "Exodus files to process. If omitted, files are discovered using "
            "--pattern in the current directory or one-level subdirectories."
        ),
    )
    parser.add_argument(
        "-s",
        "--subdirs",
        action="store_true",
        help=(
            "When no files are supplied explicitly, search one directory "
            "level down instead of the current directory."
        ),
    )
    parser.add_argument(
        "--pattern",
        default="*.e",
        help="Glob pattern used for automatic Exodus-file discovery.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Directory in which CSV and PNG files are written.",
    )
    parser.add_argument(
        "--variable",
        default="unique_grains",
        help="Exodus nodal or elemental variable containing grain IDs.",
    )
    parser.add_argument(
        "--element-block",
        type=int,
        default=1,
        help="One-based Exodus element-block number to analyze.",
    )
    parser.add_argument(
        "--grain-id",
        type=int,
        default=None,
        help=(
            "Center-grain ID. By default, it is determined at timestep zero "
            "from the element whose center is nearest the domain center."
        ),
    )
    parser.add_argument(
        "--grid-tol",
        type=float,
        default=1.0e-10,
        help=(
            "Coordinate quantization tolerance used when mapping element "
            "centers to a structured grid."
        ),
    )

    inclination = parser.add_argument_group("inclination analysis")
    inclination.add_argument(
        "--skip-inclination",
        action="store_true",
        help="Skip smoothing, inclination CSV creation, and polar plotting.",
    )
    inclination.add_argument(
        "--bins",
        type=int,
        default=36,
        help="Number of equal-width inclination bins over 0 to 360 degrees.",
    )
    inclination.add_argument(
        "--loop-times",
        type=int,
        default=5,
        help="Smoothing-window parameter passed to linear_class.",
    )
    inclination.add_argument(
        "-n",
        "--cpus",
        type=int,
        default=1,
        help="Number of worker processes used by the smoothing algorithm.",
    )
    inclination.add_argument(
        "--plot-step",
        type=int,
        default=-1,
        help=(
            "Zero-based timestep to use for the polar plot. Negative values "
            "use Python indexing, so -1 selects the final timestep."
        ),
    )
    inclination.add_argument(
        "--plot-dpi",
        type=int,
        default=300,
        help="Resolution of the saved polar plot.",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help=(
            "Enable informational logging. Progress bars are disabled when "
            "verbose logging is enabled."
        ),
    )
    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        help=(
            "Save a six-panel debugging figure showing the initial and final "
            "grain geometry, inclination quivers, area history, and aspect-ratio "
            "history. Debugging requires inclination analysis."
        ),
    )

    geometry = parser.add_argument_group("geometry measurement")

    geometry.add_argument(
        "--geometry-source",
        choices=("unique-grains", "order-parameter"),
        default="unique-grains",
        help=(
            "Field used to calculate center-grain area, x length, y length, "
            "and aspect ratio. Inclination always uses unique_grains."
        ),
    )
    geometry.add_argument(
        "--eta0-variable",
        default="gr0",
        help=(
            "Exodus variable containing the first order parameter when "
            "--geometry-source=order-parameter."
        ),
    )
    geometry.add_argument(
        "--eta1-variable",
        default="gr1",
        help=(
            "Exodus variable containing the second order parameter when "
            "--geometry-source=order-parameter."
        ),
    )
    geometry.add_argument(
        "--contour-level",
        type=float,
        default=0.5,
        help=(
            "Contour level applied to eta0^2/(eta0^2+eta1^2) when using "
            "order-parameter geometry."
        ),
    )
    geometry.add_argument(
        "--order-parameter-epsilon",
        type=float,
        default=1.0e-14,
        help=(
            "Minimum eta0^2+eta1^2 denominator accepted when constructing "
            "the order-parameter ratio."
        ),
    )

    args = parser.parse_args()

    if args.element_block < 1:
        parser.error("--element-block must be at least 1.")
    if args.grid_tol <= 0.0:
        parser.error("--grid-tol must be positive.")
    if args.bins < 1:
        parser.error("--bins must be at least 1.")
    if args.loop_times < 0:
        parser.error("--loop-times cannot be negative.")
    if args.cpus < 1:
        parser.error("--cpus must be at least 1.")
    if args.plot_dpi < 1:
        parser.error("--plot-dpi must be at least 1.")
    if args.debug and args.skip_inclination:
        parser.error("--debug cannot be combined with --skip-inclination.")
    if not 0.0 < args.contour_level < 1.0:
        parser.error("--contour-level must be between 0 and 1.")
    if args.order_parameter_epsilon <= 0.0:
        parser.error("--order-parameter-epsilon must be positive.")

    return args


def setup_logging(verbose: bool) -> logging.Logger:
    """Configure warning and informational logging levels."""

    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(level=level, format="%(message)s")
    return logging.getLogger(LOGGER_NAME)


def find_exodus_files(
    *,
    subdirs: bool = False,
    pattern: str = "*.e",
) -> list[Path]:
    """
    Find Exodus files in the current directory or one level below it.

    Parameters
    ----------
    subdirs
        If true, search ``./*/<pattern>``. Otherwise search ``./<pattern>``.
    pattern
        Glob pattern used to identify Exodus files.

    Returns
    -------
    list[Path]
        Sorted paths referring to regular files.
    """

    cwd = Path.cwd()

    if subdirs:
        files = sorted(cwd.glob(f"*/{pattern}"))
    else:
        files = sorted(cwd.glob(pattern))

    return [path for path in files if path.is_file()]


def exodus_stem(path: Path) -> str:
    """Return a clean filename stem for generated output files."""

    name = path.name

    if name.endswith(".e"):
        name = name[:-2]
    else:
        name = path.stem

    if name.endswith("_out"):
        name = name[:-4]

    return name


def normalize_step(step: int, number_of_steps: int) -> int:
    """Convert a possibly negative timestep index to a valid positive index."""

    normalized = step if step >= 0 else number_of_steps + step

    if not 0 <= normalized < number_of_steps:
        raise IndexError(
            f"Plot step {step} is outside the available range "
            f"0 to {number_of_steps - 1}."
        )

    return normalized


def read_scalar_variable(
    exo: ExodusBasics,
    variable: str,
    step: int,
    element_block: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Read a nodal or elemental scalar variable and its physical coordinates.

    Nodal variables are returned at nodal coordinates. Elemental variables
    are returned at element-center coordinates.

    Returns
    -------
    x, y, values
        One-dimensional arrays with matching lengths.
    """

    kind = exo.var_kind(variable)

    if kind == "nodal":
        x, y = exo.coords_xy_at_step(step)
        values = exo.nodal_var_at_step(variable, step)
    else:
        x, y = exo.element_centers_xy(
            eb=element_block,
            method="mean",
        )
        values = exo.elem_var_at_step(
            variable,
            step=step,
            eb=element_block,
        )

    return (
        np.asarray(x, dtype=float),
        np.asarray(y, dtype=float),
        np.asarray(values, dtype=float),
    )


def scalar_values_to_grid(
    values: np.ndarray,
    row_indices: np.ndarray,
    column_indices: np.ndarray,
    shape: tuple[int, int],
) -> np.ndarray:
    """Map scalar nodal or elemental values onto a structured grid."""

    values = np.asarray(values, dtype=float)

    if values.size != row_indices.size:
        raise ValueError(
            "Scalar field size does not match its structured-grid mapping: "
            f"{values.size} values versus {row_indices.size} positions."
        )

    grid = np.full(shape, np.nan, dtype=float)
    grid[row_indices, column_indices] = values
    return grid


def calculate_order_parameter_ratio(
    eta0: np.ndarray,
    eta1: np.ndarray,
    epsilon: float,
) -> np.ndarray:
    """
    Calculate eta0^2 / (eta0^2 + eta1^2).

    Grid locations with a denominator smaller than ``epsilon`` are assigned
    NaN and excluded from contour interpolation.
    """

    eta0 = np.asarray(eta0, dtype=float)
    eta1 = np.asarray(eta1, dtype=float)

    if eta0.shape != eta1.shape:
        raise ValueError(
            "The two order-parameter grids must have matching shapes; "
            f"found {eta0.shape} and {eta1.shape}."
        )

    eta0_squared = np.square(eta0)
    eta1_squared = np.square(eta1)
    denominator = eta0_squared + eta1_squared

    ratio = np.full(eta0.shape, np.nan, dtype=float)
    valid = (
        np.isfinite(eta0_squared)
        & np.isfinite(eta1_squared)
        & np.isfinite(denominator)
        & (denominator > epsilon)
    )

    ratio[valid] = eta0_squared[valid] / denominator[valid]
    return ratio


def polygon_area(vertices: np.ndarray) -> float:
    """Calculate the unsigned area of a two-dimensional polygon."""

    vertices = np.asarray(vertices, dtype=float)

    if vertices.ndim != 2 or vertices.shape[0] < 3 or vertices.shape[1] != 2:
        return 0.0

    x = vertices[:, 0]
    y = vertices[:, 1]

    return 0.5 * abs(
        float(
            np.dot(x, np.roll(y, -1))
            - np.dot(y, np.roll(x, -1))
        )
    )


def close_contour(
    vertices: np.ndarray,
    tolerance: float,
) -> np.ndarray | None:
    """
    Return a closed contour polygon, or None if the segment is genuinely open.

    Matplotlib normally repeats the first contour point at the end of a closed
    segment, but this function also accepts endpoints separated only by
    numerical roundoff.
    """

    vertices = np.asarray(vertices, dtype=float)

    if vertices.ndim != 2 or vertices.shape[0] < 3 or vertices.shape[1] != 2:
        return None

    finite = np.all(np.isfinite(vertices), axis=1)
    vertices = vertices[finite]

    if vertices.shape[0] < 3:
        return None

    endpoint_distance = float(np.linalg.norm(vertices[-1] - vertices[0]))

    if endpoint_distance <= tolerance:
        vertices = vertices.copy()
        vertices[-1] = vertices[0]
    else:
        return None

    if polygon_area(vertices) <= 0.0:
        return None

    return vertices


def extract_center_contour(
    ratio_grid: np.ndarray,
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    level: float,
) -> np.ndarray | None:
    """
    Extract the closed contour surrounding the center of the domain.

    If numerical noise prevents a contour from being identified as containing
    the exact domain center, the largest closed contour is used as a fallback.

    Returns
    -------
    ndarray or None
        Array with shape ``(N, 2)`` containing physical ``(x, y)`` contour
        coordinates.
    """

    ratio_grid = np.asarray(ratio_grid, dtype=float)
    x_axis = np.asarray(x_axis, dtype=float)
    y_axis = np.asarray(y_axis, dtype=float)

    finite = ratio_grid[np.isfinite(ratio_grid)]

    if finite.size == 0:
        return None

    if level < float(np.min(finite)) or level > float(np.max(finite)):
        return None

    figure, axis = plt.subplots()

    try:
        contour_set = axis.contour(
            x_axis,
            y_axis,
            np.ma.masked_invalid(ratio_grid),
            levels=[level],
        )
        segments = contour_set.allsegs[0]
    finally:
        plt.close(figure)

    if not segments:
        return None

    coordinate_scale = max(
        float(np.ptp(x_axis)),
        float(np.ptp(y_axis)),
        1.0,
    )
    closure_tolerance = 1.0e-8 * coordinate_scale

    closed_segments = []

    for segment in segments:
        polygon = close_contour(segment, closure_tolerance)

        if polygon is not None:
            closed_segments.append(polygon)

    if not closed_segments:
        return None

    center_point = (
        0.5 * (float(np.min(x_axis)) + float(np.max(x_axis))),
        0.5 * (float(np.min(y_axis)) + float(np.max(y_axis))),
    )

    containing_segments = [
        polygon
        for polygon in closed_segments
        if MplPath(polygon).contains_point(
            center_point,
            radius=closure_tolerance,
        )
    ]

    candidates = containing_segments if containing_segments else closed_segments

    # The center grain should be represented by the center-containing curve.
    # Selecting the largest candidate suppresses tiny contours caused by noise.
    return max(candidates, key=polygon_area)


def measure_contour_geometry(
    contour_vertices: np.ndarray | None,
) -> tuple[
    float,
    float,
    float,
    float,
    tuple[float, float, float, float] | None,
]:
    """
    Measure area and bounding dimensions from an interpolated contour.

    Returns
    -------
    area, x_length, y_length, aspect_ratio, bounds
        ``bounds`` is ``(x_min, x_max, y_min, y_max)``.
    """

    if contour_vertices is None or len(contour_vertices) < 3:
        return 0.0, 0.0, 0.0, math.nan, None

    contour_vertices = np.asarray(contour_vertices, dtype=float)

    x = contour_vertices[:, 0]
    y = contour_vertices[:, 1]

    x_min = float(np.min(x))
    x_max = float(np.max(x))
    y_min = float(np.min(y))
    y_max = float(np.max(y))

    area = polygon_area(contour_vertices)
    x_length = x_max - x_min
    y_length = y_max - y_min
    aspect_ratio = x_length / y_length if y_length > 0.0 else math.nan

    bounds = (x_min, x_max, y_min, y_max)

    return area, x_length, y_length, aspect_ratio, bounds


def categorical_ids(values: np.ndarray) -> np.ndarray:
    """
    Convert floating-point categorical Exodus values to integer grain IDs.

    Values are expected to be integer-valued apart from ordinary numerical
    roundoff.
    """

    values = np.asarray(values, dtype=float)

    if not np.all(np.isfinite(values)):
        raise ValueError("The grain-ID variable contains NaN or infinite values.")

    rounded = np.rint(values)

    if not np.allclose(values, rounded, atol=1.0e-5, rtol=0.0):
        maximum_error = float(np.max(np.abs(values - rounded)))
        raise ValueError(
            "The grain-ID variable is not categorical within tolerance; "
            f"maximum distance from an integer is {maximum_error:g}."
        )

    return rounded.astype(np.int64)


def nodal_ids_to_element_ids(
    nodal_values: np.ndarray,
    connectivity: np.ndarray,
) -> np.ndarray:
    """
    Assign each element its majority nodal grain ID.

    A tie is resolved deterministically in favor of the smaller grain ID.
    For a categorical ``unique_grains`` field, ties should occur only on
    elements intersected exactly by the interface.
    """

    nodal_ids = categorical_ids(nodal_values)
    values_by_element = nodal_ids[connectivity]
    unique_ids = np.unique(nodal_ids)

    counts = np.stack(
        [(values_by_element == grain_id).sum(axis=1) for grain_id in unique_ids],
        axis=1,
    )
    return unique_ids[np.argmax(counts, axis=1)]


def read_element_grain_ids(
    exo: ExodusBasics,
    variable: str,
    step: int,
    element_block: int,
    connectivity: np.ndarray,
) -> np.ndarray:
    """
    Read grain IDs at one timestep and return one ID per element.

    Elemental variables are read directly. Nodal variables are converted using
    majority voting over each element's connectivity.
    """

    kind = exo.var_kind(variable)

    if kind == "element":
        values = exo.elem_var_at_step(
            variable,
            step=step,
            eb=element_block,
        )
        return categorical_ids(values)

    nodal_values = exo.nodal_var_at_step(variable, step=step)
    return nodal_ids_to_element_ids(nodal_values, connectivity)


def element_corner_count(element_type: str, nodes_per_element: int) -> int:
    """
    Return the number of corner nodes used for area calculations.

    Exodus higher-order TRI and QUAD elements ordinarily list their corner
    nodes before midside and center nodes.
    """

    element_type = element_type.upper()

    if element_type.startswith("TRI"):
        if nodes_per_element < 3:
            raise ValueError(
                f"{element_type} has fewer than three connectivity entries."
            )
        return 3

    if element_type.startswith(("QUAD", "SHELL")):
        if nodes_per_element < 4:
            raise ValueError(
                f"{element_type} has fewer than four connectivity entries."
            )
        return 4

    raise ValueError(
        f"Unsupported element type {element_type!r}. "
        "This script currently supports 2D TRI and QUAD elements."
    )


def calculate_element_areas(
    x_nodes: np.ndarray,
    y_nodes: np.ndarray,
    connectivity: np.ndarray,
    corner_count: int,
) -> np.ndarray:
    """
    Calculate each element's area using the shoelace formula.

    Parameters
    ----------
    x_nodes, y_nodes
        Nodal physical coordinates.
    connectivity
        Zero-based element-to-node connectivity.
    corner_count
        Number of leading connectivity entries representing corner nodes.
    """

    corners = connectivity[:, :corner_count]
    x = np.asarray(x_nodes, dtype=float)[corners]
    y = np.asarray(y_nodes, dtype=float)[corners]

    area = 0.5 * np.abs(
        np.sum(
            x * np.roll(y, shift=-1, axis=1)
            - y * np.roll(x, shift=-1, axis=1),
            axis=1,
        )
    )

    if np.any(area <= 0.0):
        bad = int(np.count_nonzero(area <= 0.0))
        raise ValueError(f"Found {bad} zero-area or invalid elements.")

    return area


def choose_center_grain(
    element_ids: np.ndarray,
    x_centers: np.ndarray,
    y_centers: np.ndarray,
) -> int:
    """
    Identify the grain occupying the geometric center of the domain.

    The selected element center is the one closest to the midpoint of the
    element-center coordinate bounds.
    """

    domain_x = 0.5 * (float(np.min(x_centers)) + float(np.max(x_centers)))
    domain_y = 0.5 * (float(np.min(y_centers)) + float(np.max(y_centers)))

    distance_squared = (
        (np.asarray(x_centers) - domain_x) ** 2
        + (np.asarray(y_centers) - domain_y) ** 2
    )
    center_element = int(np.argmin(distance_squared))
    return int(element_ids[center_element])


def measure_center_grain(
    element_ids: np.ndarray,
    target_id: int,
    element_areas: np.ndarray,
    x_nodes: np.ndarray,
    y_nodes: np.ndarray,
    connectivity: np.ndarray,
) -> tuple[float, float, float, float]:
    """
    Measure center-grain area, x length, y length, and aspect ratio.

    Lengths are physical-coordinate bounding-box dimensions calculated using
    all nodes belonging to selected center-grain elements. The aspect ratio is
    ``x_length / y_length``.
    """

    selected = np.asarray(element_ids) == target_id

    if not np.any(selected):
        return 0.0, 0.0, 0.0, math.nan

    area = float(np.sum(element_areas[selected]))

    grain_nodes = np.unique(connectivity[selected].ravel())
    selected_x = np.asarray(x_nodes, dtype=float)[grain_nodes]
    selected_y = np.asarray(y_nodes, dtype=float)[grain_nodes]

    x_length = float(np.max(selected_x) - np.min(selected_x))
    y_length = float(np.max(selected_y) - np.min(selected_y))
    aspect_ratio = x_length / y_length if y_length > 0.0 else math.nan

    return area, x_length, y_length, aspect_ratio


def center_grain_bounds(
    element_ids: np.ndarray,
    target_id: int,
    x_nodes: np.ndarray,
    y_nodes: np.ndarray,
    connectivity: np.ndarray,
) -> tuple[float, float, float, float] | None:
    """
    Return the physical bounding box of the selected center grain.

    Returns
    -------
    tuple or None
        ``(x_min, x_max, y_min, y_max)``. Returns ``None`` if the
        center grain is absent.
    """

    selected = np.asarray(element_ids) == target_id

    if not np.any(selected):
        return None

    grain_nodes = np.unique(connectivity[selected].ravel())
    selected_x = np.asarray(x_nodes, dtype=float)[grain_nodes]
    selected_y = np.asarray(y_nodes, dtype=float)[grain_nodes]

    return (
        float(np.min(selected_x)),
        float(np.max(selected_x)),
        float(np.min(selected_y)),
        float(np.max(selected_y)),
    )


def make_structured_grid_mapping(
    x_centers: np.ndarray,
    y_centers: np.ndarray,
    tolerance: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a reusable element-to-structured-grid mapping.

    Returns
    -------
    row_indices, column_indices, x_axis, y_axis
        Arrays mapping each element to ``grid[row, column]`` and the physical
        coordinate axes corresponding to grid columns and rows.
    """

    x_keys = np.rint(np.asarray(x_centers) / tolerance).astype(np.int64)
    y_keys = np.rint(np.asarray(y_centers) / tolerance).astype(np.int64)

    unique_x_keys, column_indices = np.unique(x_keys, return_inverse=True)
    unique_y_keys, row_indices = np.unique(y_keys, return_inverse=True)

    expected_elements = len(unique_x_keys) * len(unique_y_keys)
    actual_elements = len(x_centers)

    if expected_elements != actual_elements:
        raise ValueError(
            "Element centers do not form a complete rectangular grid: "
            f"{len(unique_y_keys)} rows x {len(unique_x_keys)} columns "
            f"requires {expected_elements} elements, but found {actual_elements}."
        )

    linear_indices = row_indices * len(unique_x_keys) + column_indices
    if np.unique(linear_indices).size != actual_elements:
        raise ValueError(
            "Multiple elements map to the same structured-grid position. "
            "Try reducing --grid-tol or verify that the mesh is structured."
        )

    x_axis = unique_x_keys.astype(float) * tolerance
    y_axis = unique_y_keys.astype(float) * tolerance

    return row_indices, column_indices, x_axis, y_axis


def element_ids_to_grid(
    element_ids: np.ndarray,
    row_indices: np.ndarray,
    column_indices: np.ndarray,
    shape: tuple[int, int],
) -> np.ndarray:
    """Map one grain ID per element onto a complete structured 2D grid."""

    grid = np.empty(shape, dtype=np.int64)
    grid[row_indices, column_indices] = element_ids
    return grid


def representative_spacing(axis: np.ndarray, name: str) -> float:
    """Return the representative spacing of a uniformly spaced axis."""

    axis = np.asarray(axis, dtype=float)

    if axis.size < 2:
        raise ValueError(f"The structured grid needs at least two {name} positions.")

    differences = np.diff(axis)

    if np.any(differences <= 0.0):
        raise ValueError(f"The {name} coordinate axis is not strictly increasing.")

    spacing = float(np.median(differences))

    if not np.allclose(differences, spacing, rtol=1.0e-6, atol=1.0e-12):
        raise ValueError(
            f"The smoothing calculation requires uniform {name} spacing."
        )

    return spacing


def center_grain_boundary_sites(
    grain_grid: np.ndarray,
    target_id: int,
) -> np.ndarray:
    """
    Return target-grain cells adjacent to another grain.

    Cardinal neighbors and periodic indexing are used to match the smoothing
    package's boundary convention.
    """

    target = grain_grid == target_id
    boundary = target & (
        (np.roll(grain_grid, 1, axis=0) != grain_grid)
        | (np.roll(grain_grid, -1, axis=0) != grain_grid)
        | (np.roll(grain_grid, 1, axis=1) != grain_grid)
        | (np.roll(grain_grid, -1, axis=1) != grain_grid)
    )

    return np.argwhere(boundary)


def run_smoothing(
    grain_grid: np.ndarray,
    target_id: int,
    cpus: int,
    loop_times: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate smoothed interface normals for one grain structure.

    Returns
    -------
    P
        Smoothing result with shape ``(3, rows, columns)``.
    sites
        ``(row, column)`` indices on the target grain's boundary.
    """

    rows, columns = grain_grid.shape

    if np.count_nonzero(grain_grid == target_id) == 0:
        return np.zeros((3, rows, columns), dtype=float), np.empty(
            (0, 2), dtype=int
        )

    grain_count = int(np.unique(grain_grid).size)
    reference = np.zeros((rows, columns, 2), dtype=float)

    smoother = smooth.linear_class(
        rows,
        columns,
        grain_count,
        cpus,
        loop_times,
        grain_grid,
        reference,
        verification_system=False,
        id_offset=int(np.min(grain_grid)),
    )
    smoother.linear_main("inclination")

    result = smoother.get_P()
    sites = center_grain_boundary_sites(grain_grid, target_id)
    return result, sites

def extract_outward_vectors(
    smoothed_field: np.ndarray,
    sites: np.ndarray,
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    dx: float,
    dy: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract normalized, outward-facing physical normal vectors.

    Parameters
    ----------
    smoothed_field
        Output from the linear smoothing algorithm.
    sites
        Boundary-site indices stored as ``(row, column)``.
    x_axis, y_axis
        Physical coordinates of structured-grid columns and rows.
    dx, dy
        Physical grid spacing.

    Returns
    -------
    tuple[np.ndarray, ...]
        Physical boundary coordinates ``x, y`` and normalized vector
        components ``u, v``.
    """

    if sites.size == 0:
        empty = np.empty(0, dtype=float)
        return empty, empty, empty, empty

    rows = sites[:, 0].astype(int)
    columns = sites[:, 1].astype(int)

    # Convert the smoothing package's array-index components to physical
    # Cartesian components. This follows the convention used by the supplied
    # vector-inclination code.
    vector_x = -smoothed_field[2, rows, columns] / dx
    vector_y = smoothed_field[1, rows, columns] / dy

    magnitudes = np.hypot(vector_x, vector_y)
    valid = (
        np.isfinite(magnitudes)
        & np.isfinite(vector_x)
        & np.isfinite(vector_y)
        & (magnitudes > 0.0)
    )

    if not np.any(valid):
        empty = np.empty(0, dtype=float)
        return empty, empty, empty, empty

    rows = rows[valid]
    columns = columns[valid]
    vector_x = vector_x[valid] / magnitudes[valid]
    vector_y = vector_y[valid] / magnitudes[valid]

    boundary_x = np.asarray(x_axis)[columns]
    boundary_y = np.asarray(y_axis)[rows]

    # Orient each normal away from the approximate grain center. This removes
    # the arbitrary 180-degree sign ambiguity in an interface normal.
    center_x = float(np.mean(boundary_x))
    center_y = float(np.mean(boundary_y))

    radial_x = boundary_x - center_x
    radial_y = boundary_y - center_y

    inward = vector_x * radial_x + vector_y * radial_y < 0.0
    vector_x[inward] *= -1.0
    vector_y[inward] *= -1.0

    return boundary_x, boundary_y, vector_x, vector_y


def extract_outward_angles(
    smoothed_field: np.ndarray,
    sites: np.ndarray,
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    dx: float,
    dy: float,
) -> np.ndarray:
    """Convert smoothed outward normal vectors into angles in degrees."""

    _, _, vector_x, vector_y = extract_outward_vectors(
        smoothed_field,
        sites,
        x_axis,
        y_axis,
        dx,
        dy,
    )

    if vector_x.size == 0:
        return np.empty(0, dtype=float)

    return np.mod(
        np.degrees(np.arctan2(vector_y, vector_x)),
        360.0,
    )


def inclination_histogram(
    angles_deg: np.ndarray,
    bins: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create a 0–360 degree inclination histogram.

    Probability density is reported per degree and therefore integrates to
    one when multiplied by the bin width and summed.
    """

    edges = np.linspace(0.0, 360.0, bins + 1)
    counts, _ = np.histogram(angles_deg, bins=edges)
    centers = 0.5 * (edges[:-1] + edges[1:])
    bin_width = 360.0 / bins

    if counts.sum() > 0:
        density = counts.astype(float) / (counts.sum() * bin_width)
    else:
        density = np.zeros(bins, dtype=float)

    return centers, counts, density


def write_metrics_csv(
    output_path: Path,
    rows: Iterable[tuple[int, float, float, float, float, float]],
    geometry_source: str,
) -> None:
    """Write center-grain geometric measurements."""

    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "step",
                "time",
                "area",
                "x_length",
                "y_length",
                "aspect_ratio_x_over_y",
                "geometry_source",
            ]
        )

        for row in rows:
            writer.writerow([*row, geometry_source])


def write_inclination_csv(
    output_path: Path,
    rows: Iterable[tuple[int, float, float, int, float]],
) -> None:
    """Write long-form inclination distributions for all timesteps."""

    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "step",
                "time",
                "angle_deg",
                "count",
                "probability_density_per_degree",
            ]
        )
        writer.writerows(rows)


def save_polar_plot(
    output_path: Path,
    angle_centers_deg: np.ndarray,
    density: np.ndarray,
    *,
    title: str,
    dpi: int,
) -> None:
    """Save a closed 360-degree polar inclination-distribution plot."""

    theta = np.radians(angle_centers_deg)
    theta_closed = np.append(theta, theta[0] + 2.0 * np.pi)
    density_closed = np.append(density, density[0])

    figure, axis = plt.subplots(
        figsize=(6.0, 6.0),
        subplot_kw={"projection": "polar"},
    )

    axis.plot(theta_closed, density_closed, linewidth=2.0)
    axis.fill(theta_closed, density_closed, alpha=0.15)
    axis.set_theta_zero_location("E")
    axis.set_theta_direction(1)
    axis.set_thetagrids(np.arange(0.0, 360.0, 45.0))
    axis.set_title(title, pad=20.0)
    axis.grid(True, alpha=0.5)

    figure.tight_layout()
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def draw_length_overlay(
    axis: plt.Axes,
    frame: dict,
    target_id: int,
) -> None:
    """
    Draw the selected geometry field with measured dimensions overlaid.

    For unique-grains geometry, the center-grain categorical mask is shown.
    For order-parameter geometry, the continuous ratio field and interpolated
    contour are shown.
    """

    geometry_source = frame["geometry_source"]
    bounds = frame["bounds"]

    if geometry_source == "order-parameter":
        geometry_x_axis = frame["geometry_x_axis"]
        geometry_y_axis = frame["geometry_y_axis"]
        ratio_grid = frame["ratio_grid"]
        contour_vertices = frame["contour_vertices"]
        contour_level = frame["contour_level"]

        image = axis.pcolormesh(
            geometry_x_axis,
            geometry_y_axis,
            ratio_grid,
            shading="nearest",
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
            rasterized=True,
        )

        if contour_vertices is not None:
            axis.plot(
                contour_vertices[:, 0],
                contour_vertices[:, 1],
                color="white",
                linewidth=3.0,
                label=(
                    r"$\eta_0^2/(\eta_0^2+\eta_1^2)"
                    f"={contour_level:g}$"
                ),
                zorder=4,
            )
            axis.plot(
                contour_vertices[:, 0],
                contour_vertices[:, 1],
                color="black",
                linewidth=1.0,
                zorder=5,
            )

        axis.figure.colorbar(
            image,
            ax=axis,
            fraction=0.046,
            pad=0.04,
            label=r"$\eta_0^2/(\eta_0^2+\eta_1^2)$",
        )

        x_limits = (
            float(np.min(geometry_x_axis)),
            float(np.max(geometry_x_axis)),
        )
        y_limits = (
            float(np.min(geometry_y_axis)),
            float(np.max(geometry_y_axis)),
        )
        source_title = "Order-parameter contour"

    else:
        grain_grid = frame["grain_grid"]
        x_axis = frame["x_axis"]
        y_axis = frame["y_axis"]
    axis.set_ylim(*y_limits)


def draw_quiver_overlay(
    axis: plt.Axes,
    frame: dict,
    target_id: int,
    *,
    maximum_arrows: int = 400,
) -> None:
    """Draw normalized inclination vectors over the center-grain map."""

    grain_grid = frame["grain_grid"]
    x_axis = frame["x_axis"]
    y_axis = frame["y_axis"]

    center_mask = (grain_grid == target_id).astype(float)

    axis.pcolormesh(
        x_axis,
        y_axis,
        center_mask,
        shading="nearest",
        cmap="Greys",
        vmin=0.0,
        vmax=1.0,
        alpha=0.65,
    )

    x, y, u, v = extract_outward_vectors(
        frame["smoothed_field"],
        frame["sites"],
        x_axis,
        y_axis,
        frame["dx"],
        frame["dy"],
    )

    if x.size:
        # Keep large interfaces readable while sampling uniformly around the
        # complete list of boundary sites.
        stride = max(1, int(math.ceil(x.size / maximum_arrows)))

        x = x[::stride]
        y = y[::stride]
        u = u[::stride]
        v = v[::stride]

        # Normals are unit vectors, so give each arrow a visible physical
        # length of approximately three grid cells.
        arrow_length = 3.0 * min(frame["dx"], frame["dy"])

        axis.quiver(
            x,
            y,
            arrow_length * u,
            arrow_length * v,
            angles="xy",
            scale_units="xy",
            scale=1.0,
            width=0.005,
            headwidth=4.5,
            headlength=6.0,
            headaxislength=5.0,
            minlength=0.0,
            minshaft=1.0,
            color="tab:red",
            edgecolor="black",
            linewidth=0.25,
            pivot="middle",
            zorder=5,
        )

    axis.set_title(
        f"Inclination normals: step {frame['step']}, "
        f"time={frame['time']:.5g}"
    )
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.set_aspect("equal")
    axis.set_xlim(float(np.min(x_axis)), float(np.max(x_axis)))
    axis.set_ylim(float(np.min(y_axis)), float(np.max(y_axis)))


def save_debugging_plot(
    output_path: Path,
    initial_frame: dict,
    final_frame: dict,
    metrics_rows: list[tuple[int, float, float, float, float, float]],
    target_id: int,
    *,
    dpi: int,
) -> None:
    """
    Save a six-panel geometry and inclination debugging figure.

    Layout
    ------
    Row 1
        Initial dimensions, initial quiver, area versus time.
    Row 2
        Final dimensions, final quiver, aspect ratio versus time.
    """

    metrics = np.asarray(metrics_rows, dtype=float)
    times = metrics[:, 1]
    areas = metrics[:, 2]
    aspect_ratios = metrics[:, 5]

    figure, axes = plt.subplots(
        2,
        3,
        figsize=(18.0, 11.0),
        constrained_layout=True,
    )

    draw_length_overlay(axes[0, 0], initial_frame, target_id)
    draw_quiver_overlay(axes[0, 1], initial_frame, target_id)

    axes[0, 2].plot(
        times,
        areas,
        color="tab:green",
        linewidth=2.0,
        marker="o" if len(times) <= 25 else None,
        markersize=3.0,
    )
    axes[0, 2].scatter(
        [times[0], times[-1]],
        [areas[0], areas[-1]],
        color=["tab:blue", "tab:red"],
        s=55,
        zorder=3,
        label="First and final frames",
    )
    axes[0, 2].set_title("Center-grain area")
    axes[0, 2].set_xlabel("Time")
    axes[0, 2].set_ylabel("Area")
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].legend()

    draw_length_overlay(axes[1, 0], final_frame, target_id)
    draw_quiver_overlay(axes[1, 1], final_frame, target_id)

    finite = np.isfinite(aspect_ratios)
    axes[1, 2].plot(
        times[finite],
        aspect_ratios[finite],
        color="tab:purple",
        linewidth=2.0,
        marker="o" if np.count_nonzero(finite) <= 25 else None,
        markersize=3.0,
    )

    if finite[0]:
        axes[1, 2].scatter(
            times[0],
            aspect_ratios[0],
            color="tab:blue",
            s=55,
            zorder=3,
            label="Initial",
        )

    if finite[-1]:
        axes[1, 2].scatter(
            times[-1],
            aspect_ratios[-1],
            color="tab:red",
            s=55,
            zorder=3,
            label="Final",
        )

    axes[1, 2].axhline(
        1.0,
        color="black",
        linestyle="--",
        linewidth=1.0,
        alpha=0.6,
        label="Circular/isotropic reference",
    )
    axes[1, 2].set_title("Center-grain aspect ratio")
    axes[1, 2].set_xlabel("Time")
    axes[1, 2].set_ylabel("x length / y length")
    axes[1, 2].grid(True, alpha=0.3)
    axes[1, 2].legend()

    figure.suptitle(
        f"Bicrystal analysis checks — center-grain ID {target_id}",
        fontsize=16,
    )
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)

def analyze_file(
    exodus_path: Path,
    args: argparse.Namespace,
    log: logging.Logger,
) -> None:
    """
    Analyze one bicrystal Exodus file and write its output products.

    Geometry measurements can be calculated from either:

    1. ``unique_grains`` element membership, or
    2. the interpolated order-parameter contour

           eta0^2 / (eta0^2 + eta1^2) = contour_level.

    Inclination analysis always uses the smoothed ``unique_grains`` field,
    regardless of the selected geometry source.

    Parameters
    ----------
    exodus_path
        Path to the ExodusII file.
    args
        Parsed command-line arguments.
    log
        Configured logger.
    """

    stem = exodus_stem(exodus_path)
    geometry_suffix = args.geometry_source.replace("-", "_")

    metrics_path = (
        args.output_dir
        / f"{stem}_bicrystal_metrics_{geometry_suffix}.csv"
    )
    inclination_path = args.output_dir / f"{stem}_inclination.csv"

    log.warning(f"Processing {exodus_path}")

    with ExodusBasics(str(exodus_path)) as exo:
        # ------------------------------------------------------------------
        # Read general mesh and time information.
        # ------------------------------------------------------------------
        times = np.asarray(exo.time(), dtype=float)

        if times.size == 0:
            raise ValueError(f"{exodus_path} contains no timesteps.")

        connectivity = np.asarray(
            exo.connectivity(
                which=args.element_block,
                zero_based=True,
            ),
            dtype=np.int64,
        )

        x_nodes, y_nodes = exo.coords_xy()
        x_nodes = np.asarray(x_nodes, dtype=float)
        y_nodes = np.asarray(y_nodes, dtype=float)

        x_centers, y_centers = exo.element_centers_xy(
            eb=args.element_block,
            method="mean",
        )
        x_centers = np.asarray(x_centers, dtype=float)
        y_centers = np.asarray(y_centers, dtype=float)

        element_type = exo.element_type(args.element_block)
        corners = element_corner_count(
            element_type,
            connectivity.shape[1],
        )

        element_areas = calculate_element_areas(
            x_nodes,
            y_nodes,
            connectivity,
            corners,
        )

        # ------------------------------------------------------------------
        # Read initial unique_grains and identify the center-grain ID.
        #
        # This is always required because inclination uses unique_grains even
        # when area and aspect ratio use the order-parameter contour.
        # ------------------------------------------------------------------
        initial_ids = read_element_grain_ids(
            exo,
            args.variable,
            step=0,
            element_block=args.element_block,
            connectivity=connectivity,
        )

        initial_unique_ids = np.unique(initial_ids)

        if initial_unique_ids.size != 2:
            raise ValueError(
                "Expected exactly two grain IDs at timestep zero, but found "
                f"{initial_unique_ids.tolist()}."
            )

        target_id = (
            int(args.grain_id)
            if args.grain_id is not None
            else choose_center_grain(
                initial_ids,
                x_centers,
                y_centers,
            )
        )

        if target_id not in initial_unique_ids:
            raise ValueError(
                f"Requested center-grain ID {target_id} is absent at "
                f"timestep zero. Available IDs are "
                f"{initial_unique_ids.tolist()}."
            )

        log.info(f"Element type: {element_type}")
        log.info(f"unique_grains variable: {args.variable}")
        log.info(f"Variable kind: {exo.var_kind(args.variable)}")
        log.info(f"Center-grain ID: {target_id}")
        log.info(f"Geometry source: {args.geometry_source}")
        log.info(f"Timesteps: {times.size}")

        # ------------------------------------------------------------------
        # Prepare the structured-grid mapping used for unique_grains
        # smoothing and inclination.
        #
        # It is also required by the debugging quiver plots.
        # ------------------------------------------------------------------
        row_indices = None
        column_indices = None
        x_axis = None
        y_axis = None
        dx = None
        dy = None
        grain_grid_shape = None

        if not args.skip_inclination:
            (
                row_indices,
                column_indices,
                x_axis,
                y_axis,
            ) = make_structured_grid_mapping(
                x_centers,
                y_centers,
                args.grid_tol,
            )

            grain_grid_shape = (
                len(y_axis),
                len(x_axis),
            )

            dx = representative_spacing(x_axis, "x")
            dy = representative_spacing(y_axis, "y")

            log.info(
                f"unique_grains structured grid: "
                f"{grain_grid_shape[0]} rows x "
                f"{grain_grid_shape[1]} columns"
            )
            log.info(f"Physical grid spacing: dx={dx:g}, dy={dy:g}")

        # ------------------------------------------------------------------
        # Prepare the optional order-parameter structured-grid mapping.
        #
        # The order parameters may be nodal or elemental. This mapping is
        # separate from the element-centered unique_grains mapping.
        # ------------------------------------------------------------------
        eta_row_indices = None
        eta_column_indices = None
        eta_x_axis = None
        eta_y_axis = None
        eta_shape = None

        initial_eta0_values = None
        initial_eta1_values = None

        if args.geometry_source == "order-parameter":
            (
                eta0_x,
                eta0_y,
                initial_eta0_values,
            ) = read_scalar_variable(
                exo,
                args.eta0_variable,
                step=0,
                element_block=args.element_block,
            )

            (
                eta1_x,
                eta1_y,
                initial_eta1_values,
            ) = read_scalar_variable(
                exo,
                args.eta1_variable,
                step=0,
                element_block=args.element_block,
            )

            if initial_eta0_values.shape != initial_eta1_values.shape:
                raise ValueError(
                    f"{args.eta0_variable!r} and "
                    f"{args.eta1_variable!r} contain different numbers "
                    "of values."
                )

            if eta0_x.shape != eta1_x.shape or eta0_y.shape != eta1_y.shape:
                raise ValueError(
                    f"{args.eta0_variable!r} and "
                    f"{args.eta1_variable!r} are defined on grids with "
                    "different shapes."
                )

            if not (
                np.allclose(eta0_x, eta1_x)
                and np.allclose(eta0_y, eta1_y)
            ):
                raise ValueError(
                    f"{args.eta0_variable!r} and "
                    f"{args.eta1_variable!r} are not defined at the same "
                    "physical coordinates."
                )

            (
                eta_row_indices,
                eta_column_indices,
                eta_x_axis,
                eta_y_axis,
            ) = make_structured_grid_mapping(
                eta0_x,
                eta0_y,
                args.grid_tol,
            )

            eta_shape = (
                len(eta_y_axis),
                len(eta_x_axis),
            )

            log.info(
                "Order-parameter geometry enabled using "
                f"{args.eta0_variable!r} and "
                f"{args.eta1_variable!r}."
            )
            log.info(
                f"Order-parameter grid: "
                f"{eta_shape[0]} rows x {eta_shape[1]} columns"
            )
            log.info(
                "Geometry contour: "
                f"{args.eta0_variable}^2 / "
                f"({args.eta0_variable}^2 + "
                f"{args.eta1_variable}^2) = "
                f"{args.contour_level:g}"
            )

        # ------------------------------------------------------------------
        # Select the timestep used for the standalone polar plot.
        # ------------------------------------------------------------------
        plot_step = None

        if not args.skip_inclination:
            plot_step = normalize_step(
                args.plot_step,
                len(times),
            )

        # ------------------------------------------------------------------
        # Prepare output storage.
        # ------------------------------------------------------------------
        metrics_rows = []
        inclination_rows = []

        plot_angles = None
        plot_density = None

        debug_frames: dict[int, dict] = {}
        debug_steps = {0, len(times) - 1}

        frame_indices = range(len(times))

        if not args.verbose:
            frame_indices = tqdm(
                frame_indices,
                total=len(times),
                desc=stem,
                unit="step",
            )

        # ------------------------------------------------------------------
        # Process every timestep.
        # ------------------------------------------------------------------
        for step in frame_indices:
            # --------------------------------------------------------------
            # Read unique_grains.
            #
            # This is required at every timestep for inclination, even when
            # contour geometry is selected.
            # --------------------------------------------------------------
            element_ids = (
                initial_ids
                if step == 0
                else read_element_grain_ids(
                    exo,
                    args.variable,
                    step=step,
                    element_block=args.element_block,
                    connectivity=connectivity,
                )
            )

            unique_ids = np.unique(element_ids)

            if unique_ids.size > 2:
                raise ValueError(
                    f"Timestep {step} contains more than two grain IDs: "
                    f"{unique_ids.tolist()}."
                )

            # These remain None when unique_grains geometry is selected.
            ratio_grid = None
            contour_vertices = None

            # --------------------------------------------------------------
            # Measure geometry using the selected method.
            # --------------------------------------------------------------
            if args.geometry_source == "order-parameter":
                if step == 0:
                    eta0_values = initial_eta0_values
                    eta1_values = initial_eta1_values
                else:
                    _, _, eta0_values = read_scalar_variable(
                        exo,
                        args.eta0_variable,
                        step=step,
                        element_block=args.element_block,
                    )
                    _, _, eta1_values = read_scalar_variable(
                        exo,
                        args.eta1_variable,
                        step=step,
                        element_block=args.element_block,
                    )

                if eta0_values.shape != initial_eta0_values.shape:
                    raise ValueError(
                        f"{args.eta0_variable!r} changes size at "
                        f"timestep {step}."
                    )

                if eta1_values.shape != initial_eta1_values.shape:
                    raise ValueError(
                        f"{args.eta1_variable!r} changes size at "
                        f"timestep {step}."
                    )

                eta0_grid = scalar_values_to_grid(
                    eta0_values,
                    eta_row_indices,
                    eta_column_indices,
                    eta_shape,
                )

                eta1_grid = scalar_values_to_grid(
                    eta1_values,
                    eta_row_indices,
                    eta_column_indices,
                    eta_shape,
                )

                ratio_grid = calculate_order_parameter_ratio(
                    eta0_grid,
                    eta1_grid,
                    args.order_parameter_epsilon,
                )

                contour_vertices = extract_center_contour(
                    ratio_grid,
                    eta_x_axis,
                    eta_y_axis,
                    args.contour_level,
                )

                (
                    area,
                    x_length,
                    y_length,
                    aspect_ratio,
                    bounds,
                ) = measure_contour_geometry(contour_vertices)

                if contour_vertices is None:
                    log.warning(
                        f"Step {step}: no closed center contour was found "
                        f"at level {args.contour_level:g}. Geometry values "
                        "were written as zero or NaN."
                    )

            else:
                (
                    area,
                    x_length,
                    y_length,
                    aspect_ratio,
                ) = measure_center_grain(
                    element_ids,
                    target_id,
                    element_areas,
                    x_nodes,
                    y_nodes,
                    connectivity,
                )

                bounds = center_grain_bounds(
                    element_ids,
                    target_id,
                    x_nodes,
                    y_nodes,
                    connectivity,
                )

            metrics_rows.append(
                (
                    step,
                    float(times[step]),
                    area,
                    x_length,
                    y_length,
                    aspect_ratio,
                )
            )

            log.info(
                f"Step {step}: time={times[step]:g}, "
                f"geometry={args.geometry_source}, "
                f"area={area:g}, "
                f"x_length={x_length:g}, "
                f"y_length={y_length:g}, "
                f"aspect_ratio={aspect_ratio:g}"
            )

            # Geometry calculations are complete, so inclination may now be
            # skipped without affecting the metrics CSV.
            if args.skip_inclination:
                continue

            # --------------------------------------------------------------
            # Map unique_grains to the structured grid and smooth it.
            # --------------------------------------------------------------
            grain_grid = element_ids_to_grid(
                element_ids,
                row_indices,
                column_indices,
                grain_grid_shape,
            )

            smoothed_field, sites = run_smoothing(
                grain_grid,
                target_id,
                args.cpus,
                args.loop_times,
            )

            # --------------------------------------------------------------
            # Extract and histogram outward interface-normal angles.
            # --------------------------------------------------------------
            angles = extract_outward_angles(
                smoothed_field,
                sites,
                x_axis,
                y_axis,
                dx,
                dy,
            )

            angle_centers, counts, density = inclination_histogram(
                angles,
                args.bins,
            )

            inclination_rows.extend(
                (
                    step,
                    float(times[step]),
                    float(angle),
                    int(count),
                    float(probability),
                )
                for angle, count, probability in zip(
                    angle_centers,
                    counts,
                    density,
                )
            )

            # Retain the selected distribution for the standalone polar plot.
            if step == plot_step:
                plot_angles = angle_centers.copy()
                plot_density = density.copy()

            # --------------------------------------------------------------
            # Retain complete first/final debugging frames.
            #
            # Quiver information always comes from unique_grains smoothing.
            # Geometry overlays come from the selected geometry source.
            # --------------------------------------------------------------
            if args.debug and step in debug_steps:
                debug_frames[step] = {
                    "step": step,
                    "time": float(times[step]),

                    # unique_grains data for the quiver panel
                    "grain_grid": grain_grid.copy(),
                    "smoothed_field": smoothed_field.copy(),
                    "sites": sites.copy(),
                    "x_axis": x_axis.copy(),
                    "y_axis": y_axis.copy(),
                    "dx": float(dx),
                    "dy": float(dy),

                    # Selected geometry measurement
                    "geometry_source": args.geometry_source,
                    "bounds": bounds,

                    # Order-parameter data for contour-based geometry panels
                    "ratio_grid": (
                        None
                        if ratio_grid is None
                        else ratio_grid.copy()
                    ),
                    "contour_vertices": (
                        None
                        if contour_vertices is None
                        else contour_vertices.copy()
                    ),
                    "geometry_x_axis": (
                        None
                        if eta_x_axis is None
                        else eta_x_axis.copy()
                    ),
                    "geometry_y_axis": (
                        None
                        if eta_y_axis is None
                        else eta_y_axis.copy()
                    ),
                    "contour_level": float(args.contour_level),
                }

    # ----------------------------------------------------------------------
    # The Exodus file is now closed. Write geometry measurements.
    # ----------------------------------------------------------------------
    write_metrics_csv(
        metrics_path,
        metrics_rows,
        args.geometry_source,
    )
    log.warning(f"Wrote {metrics_path}")

    # ----------------------------------------------------------------------
    # Write inclination output and the standalone polar plot.
    # ----------------------------------------------------------------------
    if not args.skip_inclination:
        write_inclination_csv(
            inclination_path,
            inclination_rows,
        )
        log.warning(f"Wrote {inclination_path}")

        if plot_angles is None or plot_density is None:
            raise RuntimeError(
                f"No inclination distribution was retained for "
                f"timestep {plot_step}."
            )

        polar_plot_path = (
            args.output_dir
            / f"{stem}_inclination_step{plot_step:06d}.png"
        )

        save_polar_plot(
            polar_plot_path,
            plot_angles,
            plot_density,
            title=(
                f"{stem}\n"
                f"Inclination distribution, step {plot_step}, "
                f"time={times[plot_step]:g}"
            ),
            dpi=args.plot_dpi,
        )
        log.warning(f"Wrote {polar_plot_path}")

    # ----------------------------------------------------------------------
    # Save the combined first/final debugging figure.
    # ----------------------------------------------------------------------
    if args.debug:
        first_step = 0
        final_step = len(times) - 1

        if first_step not in debug_frames:
            raise RuntimeError(
                "The initial debugging frame was not retained."
            )

        if final_step not in debug_frames:
            raise RuntimeError(
                "The final debugging frame was not retained."
            )

        debug_path = (
            args.output_dir
            / f"{stem}_debug_summary_{geometry_suffix}.png"
        )

        save_debugging_plot(
            debug_path,
            debug_frames[first_step],
            debug_frames[final_step],
            metrics_rows,
            target_id,
            dpi=args.plot_dpi,
        )
        log.warning(f"Wrote {debug_path}")


def main() -> int:
    """Run bicrystal analysis for all selected Exodus files."""

    args = parse_args()
    log = setup_logging(args.verbose)

    if args.files:
        exodus_files = sorted(path.resolve() for path in args.files)
        missing = [path for path in exodus_files if not path.is_file()]
        if missing:
            for path in missing:
                log.error(f"File not found: {path}")
            return 2
    else:
        exodus_files = find_exodus_files(
            subdirs=args.subdirs,
            pattern=args.pattern,
        )

    if not exodus_files:
        location = (
            "one-level subdirectories"
            if args.subdirs
            else "the current directory"
        )
        log.error(
            f"No Exodus files matching {args.pattern!r} were found in {location}."
        )
        return 2

    args.output_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Arguments: {args}")
    log.info(f"Found {len(exodus_files)} Exodus file(s).")

    failures = []

    for exodus_file in exodus_files:
        try:
            analyze_file(exodus_file, args, log)
        except Exception as error:
            failures.append(exodus_file)
            if args.verbose:
                log.exception(f"Failed to process {exodus_file}: {error}")
            else:
                log.error(f"Failed to process {exodus_file}: {error}")

    if failures:
        log.error(
            f"{len(failures)} of {len(exodus_files)} file(s) failed."
        )
        return 1

    log.warning(
        f"Completed analysis of {len(exodus_files)} Exodus file(s)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
