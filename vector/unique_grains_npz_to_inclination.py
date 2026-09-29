#!/usr/bin/env python3
"""
Read unique-grain IDs from an NPZ produced by identify_unique_grains.py,
reconstruct a structured 2D grain-ID grid, smooth it with VECTOR's
linear smoothing path, and calculate grain-boundary inclination.

Required NPZ arrays:
    unique_grains
    element_center_x
    element_center_y

Example:
    python unique_grains_npz_to_inclination.py \
        01_cosinc_05_1k_unique_grains_step000010.npz \
        --output-dir inclination_output \
        --loop-times 5 \
        --bins 72

Outputs:
    <stem>_grain_grid.npy
    <stem>_inclination_distribution.csv
    <stem>_inclination_polar.png
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

import PACKAGE_MP_Linear_Vectorized as smooth
import myInput


def setup_logger(verbose: bool) -> logging.Logger:
    logger = logging.getLogger("unique_grains_inclination")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)

    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
        logger.addHandler(handler)

    return logger


def coordinate_groups(
    coordinates: np.ndarray,
    tolerance: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert nearly equal floating-point element-center coordinates into
    integer grid indices.

    Returns:
        unique_coordinates : representative coordinate of each row/column
        indices            : integer row/column index per input coordinate
    """
    coordinates = np.asarray(coordinates, dtype=np.float64)

    if coordinates.size == 0:
        raise ValueError("Cannot construct a grid from an empty coordinate array.")

    sorted_indices = np.argsort(coordinates)
    sorted_values = coordinates[sorted_indices]

    if tolerance is None:
        span = max(float(sorted_values[-1] - sorted_values[0]), 1.0)
        tolerance = span * 1.0e-10

    group_for_sorted = np.zeros(sorted_values.size, dtype=np.int64)
    group = 0

    for index in range(1, sorted_values.size):
        if abs(sorted_values[index] - sorted_values[index - 1]) > tolerance:
            group += 1
        group_for_sorted[index] = group

    indices = np.empty_like(group_for_sorted)
    indices[sorted_indices] = group_for_sorted

    unique_coordinates = np.empty(group + 1, dtype=np.float64)
    for group_index in range(group + 1):
        unique_coordinates[group_index] = np.mean(
            coordinates[indices == group_index]
        )

    return unique_coordinates, indices


def reconstruct_grain_grid(
    unique_grains: np.ndarray,
    center_x: np.ndarray,
    center_y: np.ndarray,
    coordinate_tolerance: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Rebuild a structured grain-ID grid from element-centered grain IDs.

    The output convention is:
        grain_grid[iy, ix]

    where iy increases with physical y and ix increases with physical x.

    A value of 0 remains background/unassigned.
    """
    unique_grains = np.asarray(unique_grains, dtype=np.int64)
    center_x = np.asarray(center_x, dtype=np.float64)
    center_y = np.asarray(center_y, dtype=np.float64)

    if not (
        unique_grains.size == center_x.size == center_y.size
    ):
        raise ValueError(
            "unique_grains, element_center_x, and element_center_y "
            "must have the same number of values."
        )

    x_coords, ix = coordinate_groups(center_x, coordinate_tolerance)
    y_coords, iy = coordinate_groups(center_y, coordinate_tolerance)

    grain_grid = np.zeros((y_coords.size, x_coords.size), dtype=np.int64)
    assigned = np.zeros(grain_grid.shape, dtype=bool)

    for grain_id, x_index, y_index in zip(unique_grains, ix, iy):
        if assigned[y_index, x_index]:
            raise ValueError(
                "More than one Exodus element maps to the same reconstructed "
                f"grid position (iy={y_index}, ix={x_index}). This NPZ may "
                "not represent a single uniform structured 2D mesh."
            )

        grain_grid[y_index, x_index] = grain_id
        assigned[y_index, x_index] = True

    if not np.all(assigned):
        missing = int(np.count_nonzero(~assigned))
        raise ValueError(
            f"The reconstructed grid has {missing} empty cells. "
            "This script expects one element at every structured-grid location."
        )

    return grain_grid, x_coords, y_coords


def calculate_inclination(
    grain_grid: np.ndarray,
    loop_times: int,
    cpus: int,
    verbose: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run the same VECTOR linear smoothing inclination path used by the
    existing workflow.

    Returns:
        normal_field : VECTOR normal/inclination field
        sites        : grain-boundary pixel locations
    """
    nx, ny = grain_grid.shape

    grain_count = int(np.max(grain_grid)) + 1
    if grain_count <= 1:
        raise ValueError(
            "The grain grid contains no positive grain IDs to analyze."
        )

    R = np.zeros((nx, ny, 2), dtype=np.float64)

    smoothing = smooth.linear_class(
        nx,
        ny,
        grain_count,
        cpus,
        loop_times,
        grain_grid,
        R,
        verification_system=verbose,
    )

    smoothing.linear_main("inclination")

    normal_field = smoothing.get_P()
    sites_by_grain = smoothing.get_all_gb_list()

    if sites_by_grain:
        sites = np.asarray(
            [site for grain_sites in sites_by_grain for site in grain_sites],
            dtype=np.int64,
        )
    else:
        sites = np.empty((0, 2), dtype=np.int64)

    return normal_field, sites






def write_inclination_csv(
    centers_deg: np.ndarray,
    frequency: np.ndarray,
    counts: np.ndarray,
    output_path: Path,
) -> None:
    with output_path.open("w", encoding="utf-8") as output_file:
        output_file.write("angle_deg,normalized_frequency,pixel_count\n")

        for angle, normalized_frequency, count in zip(
            centers_deg,
            frequency,
            counts,
        ):
            output_file.write(
                f"{angle:.8f},{normalized_frequency:.12g},{int(count)}\n"
            )


def inclination_angles_from_gradient_field(
    vector_field: np.ndarray,
    boundary_sites: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate directed inclination angles in [0, 360) degrees from the
    exact dx/dy vectors used for the debug quiver plot.

    Returns
    -------
    angles_deg : (N,) array
        Directed gradient angles in degrees, in [0, 360).
    sites_used : (N, 2) array
        Boundary sites associated with valid nonzero vectors.
    dx_values, dy_values : (N,) arrays
        Gradient-vector components used to calculate each angle.
    """
    if boundary_sites.size == 0:
        return (
            np.empty(0, dtype=np.float64),
            np.empty((0, 2), dtype=np.int64),
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
        )

    angles = []
    valid_sites = []
    dx_values = []
    dy_values = []

    for i, j in boundary_sites:
        # This intentionally matches the VECTOR debug/quiver path:
        # dx, dy = myInput.get_grad(P, i, j)
        dx, dy = myInput.get_grad(vector_field, int(i), int(j))

        magnitude = np.hypot(dx, dy)

        if magnitude <= 0.0 or not np.isfinite(magnitude):
            continue

        # Keep vector direction: 0 <= angle < 360 degrees.
        angle_deg = np.degrees(np.arctan2(dy, dx)) % 360.0

        angles.append(angle_deg)
        valid_sites.append((i, j))
        dx_values.append(dx)
        dy_values.append(dy)

    return (
        np.asarray(angles, dtype=np.float64),
        np.asarray(valid_sites, dtype=np.int64),
        np.asarray(dx_values, dtype=np.float64),
        np.asarray(dy_values, dtype=np.float64),
    )


def inclination_distribution(
    angles_deg: np.ndarray,
    bins: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate a directed inclination distribution over [0, 360).
    """
    edges = np.linspace(0.0, 360.0, bins + 1)
    counts, _ = np.histogram(angles_deg, bins=edges)

    centers = 0.5 * (edges[:-1] + edges[1:])

    if counts.sum() == 0:
        frequency = np.zeros_like(centers, dtype=np.float64)
    else:
        frequency = counts / counts.sum()

    return centers, frequency, counts


def plot_inclination_polar(
    centers_deg: np.ndarray,
    frequency: np.ndarray,
    output_path: Path,
    title: str,
) -> None:
    """
    Full 360-degree polar inclination plot.
    """
    theta = np.deg2rad(centers_deg)

    # Close the curve for plotting.
    theta_closed = np.append(theta, theta[0])
    frequency_closed = np.append(frequency, frequency[0])

    figure, axis = plt.subplots(
        subplot_kw={"projection": "polar"},
        figsize=(6.5, 6.5),
    )

    axis.plot(
        theta_closed,
        frequency_closed,
        color="steelblue",
        linewidth=2,
        label="Inclination PDF",
    )
    axis.fill(
        theta_closed,
        frequency_closed,
        color="steelblue",
        alpha=0.18,
    )

    # Default matplotlib polar axes span the full 360 degrees.
    axis.set_theta_zero_location("E")
    axis.set_theta_direction(1)
    axis.set_title(title, pad=18)
    axis.legend(loc="upper right", bbox_to_anchor=(1.25, 1.12))

    figure.tight_layout()
    figure.savefig(output_path, dpi=300, transparent=True)
    plt.close(figure)


def plot_inclination_quiver_debug(
    grain_grid: np.ndarray,
    boundary_sites: np.ndarray,
    dx_values: np.ndarray,
    dy_values: np.ndarray,
    x_centers: np.ndarray,
    y_centers: np.ndarray,
    output_path: Path,
    max_arrows: int = 2_000,
    normalize: bool = True,
) -> None:
    """
    Plot the same dx/dy gradient vectors used for inclination angles over
    the reconstructed unique-grain microstructure.

    The boundary-site convention is (i, j):
        i -> row / y index
        j -> column / x index
    """
    figure, axis = plt.subplots(figsize=(9, 8))

    x_mesh, y_mesh = np.meshgrid(x_centers, y_centers)

    image = axis.pcolormesh(
        x_mesh,
        y_mesh,
        grain_grid,
        shading="nearest",
        cmap="nipy_spectral",
    )
    colorbar = figure.colorbar(image, ax=axis)
    colorbar.set_label("Unique grain ID")

    if boundary_sites.size > 0:
        stride = max(1, len(boundary_sites) // max_arrows)

        sites_plot = boundary_sites[::stride]
        dx_plot = dx_values[::stride].astype(np.float64, copy=True)
        dy_plot = dy_values[::stride].astype(np.float64, copy=True)

        if normalize:
            magnitude = np.hypot(dx_plot, dy_plot)
            valid = magnitude > 0.0
            dx_plot[valid] /= magnitude[valid]
            dy_plot[valid] /= magnitude[valid]

        # This mapping matches the existing VECTOR debug plotting:
        # site (i, j) maps to physical coordinate (x_centers[j], y_centers[i]).
        x_plot = x_centers[sites_plot[:, 1]]
        y_plot = y_centers[sites_plot[:, 0]]

        axis.quiver(
            x_plot,
            y_plot,
            dx_plot,
            dy_plot,
            angles="xy",
            scale_units="xy",
            scale=0.5,
            width=0.0025,
            color="cyan",
            alpha=0.80,
            pivot="middle",
        )

    axis.set_aspect("equal")
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.set_title(
        "Inclination-gradient debug plot\n"
        f"{len(boundary_sites)} valid GB sites; arrows subsampled"
    )

    figure.tight_layout()
    figure.savefig(output_path, dpi=500, transparent=True)
    plt.close(figure)


def plot_grain_grid(
    grain_grid: np.ndarray,
    output_path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(8, 8))

    image = axis.imshow(
        grain_grid,
        origin="lower",
        interpolation="nearest",
        cmap="nipy_spectral",
    )

    axis.set_title("Unique grain IDs")
    axis.set_xlabel("Element-grid x index")
    axis.set_ylabel("Element-grid y index")

    colorbar = figure.colorbar(image, ax=axis)
    colorbar.set_label("Unique grain ID")

    figure.tight_layout()
    figure.savefig(output_path, dpi=300, transparent=True)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Calculate smoothed grain-boundary inclination directly from "
            "a unique-grains NPZ file."
        )
    )

    parser.add_argument(
        "input_npz",
        type=Path,
        help="NPZ written by identify_unique_grains.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for outputs. Default: directory containing input NPZ.",
    )
    parser.add_argument(
        "--loop-times",
        type=int,
        default=5,
        help="VECTOR linear smoothing iterations. Default: 5.",
    )
    parser.add_argument(
        "--cpus",
        type=int,
        default=1,
        help="CPU count passed to VECTOR smoothing. Default: 1.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=72,
        help="Number of inclination bins spanning [0, 180). Default: 72.",
    )
    parser.add_argument(
        "--coordinate-tolerance",
        type=float,
        default=None,
        help=(
            "Tolerance used when grouping element-center coordinates into "
            "grid rows/columns. Default: automatically selected."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose VECTOR smoothing diagnostics.",
    )
    parser.add_argument(
        "--debug-quiver",
        action="store_true",
        help=(
            "Write a grain-map debug plot with the dx/dy vectors used for "
            "inclination shown as quiver arrows."
        ),
    )
    parser.add_argument(
        "--debug-max-arrows",
        type=int,
        default=2000,
        help="Maximum quiver arrows in the debug plot. Default: 2000.",
    )

    args = parser.parse_args()
    logger = setup_logger(args.verbose)

    if not args.input_npz.is_file():
        parser.error(f"Input NPZ does not exist: {args.input_npz}")

    if args.loop_times < 1:
        parser.error("--loop-times must be at least 1.")

    if args.bins < 1:
        parser.error("--bins must be at least 1.")

    output_dir = args.output_dir or args.input_npz.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = args.input_npz.stem

    logger.info("Reading %s", args.input_npz)

    with np.load(args.input_npz, allow_pickle=False) as data:
        required = {
            "unique_grains",
            "element_center_x",
            "element_center_y",
        }

        missing = required.difference(data.files)
        if missing:
            raise KeyError(
                "The NPZ is missing required arrays: "
                f"{sorted(missing)}. Regenerate it with the updated "
                "identify_unique_grains.py script."
            )

        unique_grains = data["unique_grains"]
        center_x = data["element_center_x"]
        center_y = data["element_center_y"]

        actual_time = (
            float(data["actual_time"])
            if "actual_time" in data.files
            else None
        )

    grain_grid, x_coords, y_coords = reconstruct_grain_grid(
        unique_grains=unique_grains,
        center_x=center_x,
        center_y=center_y,
        coordinate_tolerance=args.coordinate_tolerance,
    )

    logger.info(
        "Reconstructed grain grid: ny=%d, nx=%d, unique IDs=%d",
        grain_grid.shape[0],
        grain_grid.shape[1],
        int(np.max(grain_grid)),
    )

    np.save(output_dir / f"{stem}_grain_grid.npy", grain_grid)

    plot_grain_grid(
        grain_grid,
        output_dir / f"{stem}_grain_grid.png",
    )

    vector_field, boundary_sites = calculate_inclination(
        grain_grid=grain_grid,
        loop_times=args.loop_times,
        cpus=args.cpus,
        verbose=args.verbose,
    )

    (
        angles_deg,
        valid_boundary_sites,
        dx_values,
        dy_values,
    ) = inclination_angles_from_gradient_field(
        vector_field=vector_field,
        boundary_sites=boundary_sites,
    )

    centers_deg, frequency, counts = inclination_distribution(
        angles_deg,
        args.bins,
    )

    csv_path = output_dir / f"{stem}_inclination_distribution_360deg.csv"
    write_inclination_csv(
        centers_deg,
        frequency,
        counts,
        csv_path,
    )

    plot_title = (
        "Directed grain-boundary inclination distribution\n"
        "Gradient direction: 0–360 degrees"
    )
    if actual_time is not None:
        plot_title += f"\ntime = {actual_time:.8g}"

    plot_path = output_dir / f"{stem}_inclination_polar_360deg.png"
    plot_inclination_polar(
        centers_deg,
        frequency,
        plot_path,
        plot_title,
    )

    if args.debug_quiver:
        quiver_path = output_dir / f"{stem}_DEBUG_inclination_quiver.png"

        plot_inclination_quiver_debug(
            grain_grid=grain_grid,
            boundary_sites=valid_boundary_sites,
            dx_values=dx_values,
            dy_values=dy_values,
            x_centers=x_coords,
            y_centers=y_coords,
            output_path=quiver_path,
            max_arrows=args.debug_max_arrows,
            normalize=True,
        )

        logger.info("Wrote inclination quiver debug plot: %s", quiver_path)

    logger.info("Boundary sites found: %d", boundary_sites.shape[0])
    logger.info("Boundary sites used:  %d", angles_deg.size)
    logger.info("Wrote CSV: %s", csv_path)
    logger.info("Wrote polar plot: %s", plot_path)
    logger.info("Wrote grain grid: %s", output_dir / f"{stem}_grain_grid.npy")


if __name__ == "__main__":
    main()
