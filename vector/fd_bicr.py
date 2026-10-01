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
            ]
        )
        writer.writerows(rows)


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
    """Draw a grain map with measured x and y dimensions overlaid."""

    grain_grid = frame["grain_grid"]
    x_axis = frame["x_axis"]
    y_axis = frame["y_axis"]
    bounds = frame["bounds"]

    center_mask = (grain_grid == target_id).astype(float)

    axis.pcolormesh(
        x_axis,
        y_axis,
        center_mask,
        shading="nearest",
        cmap="Greys",
        vmin=0.0,
        vmax=1.0,
    )

    if bounds is not None:
        x_min, x_max, y_min, y_max = bounds
        x_mid = 0.5 * (x_min + x_max)
        y_mid = 0.5 * (y_min + y_max)

        # Horizontal x-length measurement.
        axis.annotate(
            "",
            xy=(x_max, y_mid),
            xytext=(x_min, y_mid),
            arrowprops={
                "arrowstyle": "<->",
                "color": "tab:red",
                "linewidth": 2.0,
            },
        )
        axis.text(
            x_mid,
            y_mid,
            f"  x = {x_max - x_min:.5g}",
            color="tab:red",
            fontsize=9,
            ha="center",
            va="bottom",
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.75,
            },
        )

        # Vertical y-length measurement.
        axis.annotate(
            "",
            xy=(x_mid, y_max),
            xytext=(x_mid, y_min),
            arrowprops={
                "arrowstyle": "<->",
                "color": "tab:blue",
                "linewidth": 2.0,
            },
        )
        axis.text(
            x_mid,
            y_mid,
            f"  y = {y_max - y_min:.5g}",
            color="tab:blue",
            fontsize=9,
            ha="left",
            va="center",
            rotation=90,
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.75,
            },
        )

        # Show the bounding box itself as an additional consistency check.
        rectangle = plt.Rectangle(
            (x_min, y_min),
            x_max - x_min,
            y_max - y_min,
            fill=False,
            edgecolor="tab:green",
            linestyle="--",
            linewidth=1.2,
            label="Measured bounding box",
        )
        axis.add_patch(rectangle)

    axis.set_title(
        f"Grain dimensions: step {frame['step']}, "
        f"time={frame['time']:.5g}"
    )
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.set_aspect("equal")
    axis.set_xlim(float(np.min(x_axis)), float(np.max(x_axis)))
    axis.set_ylim(float(np.min(y_axis)), float(np.max(y_axis)))


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
    """Analyze one Exodus file and write its measurement products."""

    stem = exodus_stem(exodus_path)
    metrics_path = args.output_dir / f"{stem}_bicrystal_metrics.csv"
    inclination_path = args.output_dir / f"{stem}_inclination.csv"

    log.warning(f"Processing {exodus_path}")

    with ExodusBasics(str(exodus_path)) as exo:
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
                f"Expected two grain IDs at timestep zero, but found "
                f"{initial_unique_ids.tolist()}."
            )

        target_id = (
            int(args.grain_id)
            if args.grain_id is not None
            else choose_center_grain(initial_ids, x_centers, y_centers)
        )

        if target_id not in initial_unique_ids:
            raise ValueError(
                f"Requested center-grain ID {target_id} is absent at timestep "
                f"zero. Available IDs: {initial_unique_ids.tolist()}."
            )

        log.info(f"Element type: {element_type}")
        log.info(f"Variable kind: {exo.var_kind(args.variable)}")
        log.info(f"Center-grain ID: {target_id}")
        log.info(f"Timesteps: {times.size}")

        row_indices = column_indices = None
        x_axis = y_axis = None
        dx = dy = None

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
            dx = representative_spacing(x_axis, "x")
            dy = representative_spacing(y_axis, "y")

            log.info(
                f"Structured grid: {len(y_axis)} rows x {len(x_axis)} columns"
            )
            log.info(f"Physical spacing: dx={dx:g}, dy={dy:g}")

        plot_step = normalize_step(args.plot_step, len(times))
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

        for step in frame_indices:
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

            area, x_length, y_length, aspect_ratio = measure_center_grain(
                element_ids,
                target_id,
                element_areas,
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
                f"Step {step}: time={times[step]:g}, area={area:g}, "
                f"x_length={x_length:g}, y_length={y_length:g}, "
                f"aspect_ratio={aspect_ratio:g}"
            )

            if args.skip_inclination:
                continue

            grain_grid = element_ids_to_grid(
                element_ids,
                row_indices,
                column_indices,
                (len(y_axis), len(x_axis)),
            )

            bounds = center_grain_bounds(
                element_ids,
                target_id,
                x_nodes,
                y_nodes,
                connectivity,
            )

            smoothed_field, sites = run_smoothing(
                grain_grid,
                target_id,
                args.cpus,
                args.loop_times,
            )

            if args.debug and step in debug_steps:
                debug_frames[step] = {
                    "step": step,
                    "time": float(times[step]),
                    "grain_grid": grain_grid.copy(),
                    "smoothed_field": smoothed_field.copy(),
                    "sites": sites.copy(),
                    "bounds": bounds,
                    "x_axis": x_axis.copy(),
                    "y_axis": y_axis.copy(),
                    "dx": float(dx),
                    "dy": float(dy),
                }

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

            if step == plot_step:
                plot_angles = angle_centers.copy()
                plot_density = density.copy()

    write_metrics_csv(metrics_path, metrics_rows)
    log.warning(f"Wrote {metrics_path}")

    if not args.skip_inclination:
        write_inclination_csv(inclination_path, inclination_rows)
        log.warning(f"Wrote {inclination_path}")

        if plot_angles is None or plot_density is None:
            raise RuntimeError(
                f"No inclination distribution was retained for step {plot_step}."
            )

        plot_path = (
            args.output_dir
            / f"{stem}_inclination_step{plot_step:06d}.png"
        )
        save_polar_plot(
            plot_path,
            plot_angles,
            plot_density,
            title=(
                f"{stem}\n"
                f"Inclination distribution, step {plot_step}, "
                f"time={times[plot_step]:g}"
            ),
            dpi=args.plot_dpi,
        )
        log.warning(f"Wrote {plot_path}")

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

            debug_path = args.output_dir / f"{stem}_debug_summary.png"

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
