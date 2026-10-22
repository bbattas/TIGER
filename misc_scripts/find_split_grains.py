#!/usr/bin/env python3
"""
find_split_grains.py

Detect disconnected components carrying the same elemental grain ID in
2D ExodusII files, with optional x/y periodic-boundary connectivity.

Requirements
------------
numpy
matplotlib
netCDF4
ExodusBasics.py

Examples
--------
Nonperiodic analysis:

    python find_split_grains.py --time 50

Periodic in both x and y:

    python find_split_grains.py --time 50 --periodic

Periodic analysis with diagnostics:

    python find_split_grains.py \
        --time 50 \
        --name polycrystal \
        --periodic \
        --debug \
        --show-exterior \
        --label-components \
        -vv

Specify the periodic edge-matching tolerance:

    python find_split_grains.py \
        --time 50 \
        --periodic \
        --periodic-tolerance 1e-8

Periodic implementation
-----------------------
The script identifies exterior element edges on xmin, xmax, ymin, and ymax
in the undeformed/reference mesh.

For x periodicity, xmin and xmax edges are matched using their y endpoint
coordinates. For y periodicity, ymin and ymax edges are matched using
their x endpoint coordinates.

Elements owning matched edge pairs are added as neighbors in the element
adjacency graph. The same-ID flood fill therefore crosses periodic seams.

This assumes a rectangular, conforming periodic mesh whose opposite
boundary edges have corresponding tangential coordinates. The code does
not guess connections for unmatched edges.
"""

from __future__ import annotations

import argparse
import glob
import logging
import sys
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

# Allows plotting on headless systems and compute clusters.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Polygon

from vector.ExodusBasics import ExodusBasics


LOGGER = logging.getLogger("find_split_grains")
INVALID_GRAIN_ID = np.iinfo(np.int64).min


@dataclass(frozen=True)
class BoundaryEdge:
    """One exterior edge on a named rectangular-domain side."""

    edge: tuple[int, int]
    element: int
    side: str
    tangent_low: float
    tangent_high: float


@dataclass(frozen=True)
class PeriodicLink:
    """Pair of corresponding edges/elements across a periodic boundary."""

    axis: str
    side_a: str
    side_b: str
    edge_a: tuple[int, int]
    edge_b: tuple[int, int]
    element_a: int
    element_b: int
    mismatch: float


@dataclass
class PeriodicResult:
    """Results from periodic-boundary discovery and matching."""

    enabled: bool
    tolerance: float | None
    bounds: tuple[float, float, float, float]
    links: list[PeriodicLink]
    boundary_edges: dict[str, list[BoundaryEdge]]
    unmatched_edges: dict[str, list[BoundaryEdge]]

    @property
    def x_link_count(self) -> int:
        return sum(link.axis == "x" for link in self.links)

    @property
    def y_link_count(self) -> int:
        return sum(link.axis == "y" for link in self.links)

    @property
    def unmatched_count(self) -> int:
        return sum(len(edges) for edges in self.unmatched_edges.values())


@dataclass
class AnalysisResult:
    """Results from grain connected-component analysis."""

    grain_ids: np.ndarray
    adjacency: list[set[int]]
    component_by_element: np.ndarray
    components_by_grain: dict[int, list[np.ndarray]]
    duplicate_grains: dict[int, list[np.ndarray]]
    invalid_elements: np.ndarray


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Find grain IDs occurring in multiple disconnected regions of a "
            "2D ExodusII element block, optionally accounting for x/y "
            "periodic boundary conditions."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    input_group = parser.add_argument_group("input selection")
    input_group.add_argument(
        "--directory",
        type=Path,
        default=Path("."),
        help="Directory in which to search for Exodus files.",
    )
    input_group.add_argument(
        "--name",
        default=None,
        help=(
            "Optional partial filename. Files matching '*NAME*.e*' are "
            "selected instead of the default '*.e'."
        ),
    )
    input_group.add_argument(
        "--pattern",
        default=None,
        help=(
            "Explicit glob pattern relative to --directory. This overrides "
            "--name and the default '*.e' pattern."
        ),
    )

    analysis_group = parser.add_argument_group("analysis")
    analysis_group.add_argument(
        "--time",
        type=float,
        required=True,
        help="Requested physical time; the nearest Exodus timestep is used.",
    )
    analysis_group.add_argument(
        "--variable",
        default="unique_grains",
        help="Elemental variable containing integer grain identifiers.",
    )
    analysis_group.add_argument(
        "--block",
        type=int,
        default=1,
        help="One-based Exodus element-block number, corresponding to connectN.",
    )
    analysis_group.add_argument(
        "--integer-tolerance",
        type=float,
        default=1.0e-5,
        help=(
            "Maximum difference allowed between a stored grain value and "
            "its nearest integer."
        ),
    )
    analysis_group.add_argument(
        "--include-id",
        type=int,
        action="append",
        default=None,
        help=(
            "Only analyze this grain ID. May be supplied multiple times. "
            "By default, all valid IDs are analyzed."
        ),
    )
    analysis_group.add_argument(
        "--exclude-id",
        type=int,
        action="append",
        default=None,
        help=(
            "Exclude this grain ID from duplicate detection. May be supplied "
            "multiple times; useful for background or void IDs."
        ),
    )
    analysis_group.add_argument(
        "--use-static-coordinates",
        action="store_true",
        help=(
            "Use original coordx/coordy for plotting rather than coordinates "
            "displaced at the selected timestep. Periodic matching always "
            "uses the original reference coordinates."
        ),
    )

    periodic_group = parser.add_argument_group("periodic boundaries")
    periodic_group.add_argument(
        "--periodic",
        action="store_true",
        help=(
            "Enable periodic connectivity in both x and y. Corresponding "
            "edges on xmin/xmax and ymin/ymax are treated as neighbors."
        ),
    )
    periodic_group.add_argument(
        "--periodic-tolerance",
        type=float,
        default=None,
        help=(
            "Absolute coordinate tolerance for identifying and matching "
            "periodic edges. By default, 1e-8 times the largest domain span "
            "is used."
        ),
    )
    periodic_group.add_argument(
        "--allow-unmatched-periodic-edges",
        action="store_true",
        help=(
            "Continue periodic analysis when opposite boundary edges cannot "
            "all be matched. Without this option, unmatched periodic edges "
            "cause that file's analysis to fail."
        ),
    )

    plot_group = parser.add_argument_group("plotting and output")
    plot_group.add_argument(
        "--output-directory",
        type=Path,
        default=Path("split_grain_results"),
        help="Directory for PNG images and debug reports.",
    )
    plot_group.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Resolution of saved PNG figures.",
    )
    plot_group.add_argument(
        "--line-width",
        type=float,
        default=0.55,
        help="Width of grain-boundary lines.",
    )
    plot_group.add_argument(
        "--show-exterior",
        action="store_true",
        help="Draw exterior domain edges in addition to grain interfaces.",
    )
    plot_group.add_argument(
        "--label-components",
        action="store_true",
        help="Label disconnected duplicate components on the main plot.",
    )

    diagnostics_group = parser.add_argument_group(
        "logging and diagnostics"
    )
    diagnostics_group.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase logging detail. Use -v for INFO or -vv for DEBUG.",
    )
    diagnostics_group.add_argument(
        "-d", "--debug",
        action="store_true",
        help=(
            "Run consistency tests and save a diagnostic subplot figure "
            "and detailed text report for each file."
        ),
    )

    return parser


def configure_logging(verbosity: int) -> None:
    if verbosity >= 2:
        level = logging.DEBUG
    elif verbosity == 1:
        level = logging.INFO
    else:
        level = logging.WARNING

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def discover_files(args: argparse.Namespace) -> list[Path]:
    directory = args.directory.expanduser().resolve()

    if not directory.is_dir():
        raise NotADirectoryError(
            f"Input directory does not exist: {directory}"
        )

    if args.pattern:
        pattern = args.pattern
    elif args.name:
        pattern = f"*{args.name}*.e*"
    else:
        pattern = "*.e"

    full_pattern = str(directory / pattern)
    files = sorted(
        Path(filename)
        for filename in glob.glob(full_pattern)
        if Path(filename).is_file()
    )

    LOGGER.info("Glob pattern: %s", full_pattern)
    LOGGER.info("Found %d candidate file(s)", len(files))
    return files


def corner_count(element_type: str, nodes_per_element: int) -> int:
    """
    Return the number of polygon corner nodes for a supported 2D element.

    Higher-order TRI and QUAD connectivity is assumed to list corner nodes
    before midside and interior nodes.
    """

    normalized = element_type.upper().strip()

    if "TRI" in normalized:
        if nodes_per_element < 3:
            raise ValueError(
                f"{element_type!r} has fewer than three connectivity entries."
            )
        return 3

    if "QUAD" in normalized:
        if nodes_per_element < 4:
            raise ValueError(
                f"{element_type!r} has fewer than four connectivity entries."
            )
        return 4

    raise NotImplementedError(
        f"Unsupported element type {element_type!r}. This implementation "
        "supports TRI* and QUAD* 2D elements."
    )


def element_edges(
    corner_connectivity: np.ndarray,
) -> Iterable[tuple[int, int, int]]:
    """Yield element index and endpoint node IDs for every polygon edge."""

    for element_index, nodes in enumerate(corner_connectivity):
        number_of_nodes = len(nodes)

        for local_index in range(number_of_nodes):
            node_a = int(nodes[local_index])
            node_b = int(nodes[(local_index + 1) % number_of_nodes])
            yield element_index, node_a, node_b


def canonical_edge(node_a: int, node_b: int) -> tuple[int, int]:
    """Return an orientation-independent edge key."""

    return (
        (node_a, node_b)
        if node_a < node_b
        else (node_b, node_a)
    )


def build_edge_ownership(
    corner_connectivity: np.ndarray,
) -> dict[tuple[int, int], list[int]]:
    """Map each undirected mesh edge to its owning element or elements."""

    edge_owners: dict[tuple[int, int], list[int]] = defaultdict(list)

    for element_index, node_a, node_b in element_edges(
        corner_connectivity
    ):
        if node_a == node_b:
            raise ValueError(
                f"Element {element_index} has a zero-length topological "
                f"edge at node {node_a}."
            )

        edge_owners[canonical_edge(node_a, node_b)].append(element_index)

    return dict(edge_owners)


def build_element_adjacency(
    number_of_elements: int,
    edge_owners: dict[tuple[int, int], list[int]],
) -> list[set[int]]:
    """Build ordinary element adjacency from shared complete edges."""

    adjacency = [set() for _ in range(number_of_elements)]

    for owners in edge_owners.values():
        for first_index, element_a in enumerate(owners):
            for element_b in owners[first_index + 1 :]:
                adjacency[element_a].add(element_b)
                adjacency[element_b].add(element_a)

    return adjacency


def default_periodic_tolerance(
    x: np.ndarray,
    y: np.ndarray,
) -> float:
    """
    Calculate a conservative default absolute coordinate tolerance.
    """

    x_span = float(np.max(x) - np.min(x))
    y_span = float(np.max(y) - np.min(y))
    largest_span = max(x_span, y_span)

    if largest_span <= 0.0:
        raise ValueError("The mesh has a zero-sized x/y domain.")

    coordinate_scale = max(
        float(np.max(np.abs(x))),
        float(np.max(np.abs(y))),
        largest_span,
        1.0,
    )

    floating_point_floor = (
        100.0 * np.finfo(float).eps * coordinate_scale
    )
    relative_tolerance = 1.0e-8 * largest_span

    return max(floating_point_floor, relative_tolerance)


def identify_periodic_boundary_edges(
    edge_owners: dict[tuple[int, int], list[int]],
    x: np.ndarray,
    y: np.ndarray,
    tolerance: float,
) -> tuple[
    dict[str, list[BoundaryEdge]],
    tuple[float, float, float, float],
]:
    """
    Classify exterior edges lying on xmin, xmax, ymin, or ymax.

    Only edges with exactly one ordinary owner are exterior candidates.
    Both edge endpoints must lie on the relevant rectangular boundary.
    """

    xmin = float(np.min(x))
    xmax = float(np.max(x))
    ymin = float(np.min(y))
    ymax = float(np.max(y))

    boundaries: dict[str, list[BoundaryEdge]] = {
        "xmin": [],
        "xmax": [],
        "ymin": [],
        "ymax": [],
    }

    for edge, owners in edge_owners.items():
        if len(owners) != 1:
            continue

        node_a, node_b = edge
        xa = float(x[node_a])
        xb = float(x[node_b])
        ya = float(y[node_a])
        yb = float(y[node_b])
        element = int(owners[0])

        if abs(xa - xmin) <= tolerance and abs(xb - xmin) <= tolerance:
            boundaries["xmin"].append(
                BoundaryEdge(
                    edge=edge,
                    element=element,
                    side="xmin",
                    tangent_low=min(ya, yb),
                    tangent_high=max(ya, yb),
                )
            )

        if abs(xa - xmax) <= tolerance and abs(xb - xmax) <= tolerance:
            boundaries["xmax"].append(
                BoundaryEdge(
                    edge=edge,
                    element=element,
                    side="xmax",
                    tangent_low=min(ya, yb),
                    tangent_high=max(ya, yb),
                )
            )

        if abs(ya - ymin) <= tolerance and abs(yb - ymin) <= tolerance:
            boundaries["ymin"].append(
                BoundaryEdge(
                    edge=edge,
                    element=element,
                    side="ymin",
                    tangent_low=min(xa, xb),
                    tangent_high=max(xa, xb),
                )
            )

        if abs(ya - ymax) <= tolerance and abs(yb - ymax) <= tolerance:
            boundaries["ymax"].append(
                BoundaryEdge(
                    edge=edge,
                    element=element,
                    side="ymax",
                    tangent_low=min(xa, xb),
                    tangent_high=max(xa, xb),
                )
            )

    for side in boundaries:
        boundaries[side].sort(
            key=lambda item: (
                item.tangent_low,
                item.tangent_high,
                item.element,
            )
        )

    return boundaries, (xmin, xmax, ymin, ymax)


def periodic_edge_mismatch(
    edge_a: BoundaryEdge,
    edge_b: BoundaryEdge,
) -> float:
    """
    Maximum tangential endpoint difference between two boundary edges.
    """

    return max(
        abs(edge_a.tangent_low - edge_b.tangent_low),
        abs(edge_a.tangent_high - edge_b.tangent_high),
    )


def match_periodic_sides(
    side_a_edges: list[BoundaryEdge],
    side_b_edges: list[BoundaryEdge],
    axis: str,
    tolerance: float,
) -> tuple[
    list[PeriodicLink],
    list[BoundaryEdge],
    list[BoundaryEdge],
]:
    """
    Strictly match two opposite sides using tangential edge coordinates.

    Pairing is one-to-one. Each edge is assigned to the currently closest
    unmatched opposite edge, provided that the endpoint mismatch is within
    tolerance.
    """

    available_b = set(range(len(side_b_edges)))
    links: list[PeriodicLink] = []
    unmatched_a: list[BoundaryEdge] = []

    # Edges with an exact counterpart are handled first. Sorting by the
    # best available mismatch reduces greedy-order sensitivity.
    candidates: list[
        tuple[float, int, list[tuple[float, int]]]
    ] = []

    for index_a, edge_a in enumerate(side_a_edges):
        edge_candidates = sorted(
            (
                periodic_edge_mismatch(edge_a, edge_b),
                index_b,
            )
            for index_b, edge_b in enumerate(side_b_edges)
        )

        best = (
            edge_candidates[0][0]
            if edge_candidates
            else float("inf")
        )
        candidates.append((best, index_a, edge_candidates))

    candidates.sort(key=lambda item: (item[0], item[1]))

    for _, index_a, edge_candidates in candidates:
        edge_a = side_a_edges[index_a]
        selected_index_b: int | None = None
        selected_mismatch = float("inf")

        for mismatch, index_b in edge_candidates:
            if mismatch > tolerance:
                break

            if index_b in available_b:
                selected_index_b = index_b
                selected_mismatch = mismatch
                break

        if selected_index_b is None:
            unmatched_a.append(edge_a)
            continue

        edge_b = side_b_edges[selected_index_b]
        available_b.remove(selected_index_b)

        links.append(
            PeriodicLink(
                axis=axis,
                side_a=edge_a.side,
                side_b=edge_b.side,
                edge_a=edge_a.edge,
                edge_b=edge_b.edge,
                element_a=edge_a.element,
                element_b=edge_b.element,
                mismatch=selected_mismatch,
            )
        )

    unmatched_b = [
        side_b_edges[index]
        for index in sorted(available_b)
    ]

    return links, unmatched_a, unmatched_b


def add_periodic_connectivity(
    adjacency: list[set[int]],
    edge_owners: dict[tuple[int, int], list[int]],
    reference_x: np.ndarray,
    reference_y: np.ndarray,
    tolerance: float,
) -> PeriodicResult:
    """
    Discover x/y periodic edge pairs and add their elements as neighbors.
    """

    boundaries, bounds = identify_periodic_boundary_edges(
        edge_owners,
        reference_x,
        reference_y,
        tolerance,
    )

    x_links, unmatched_xmin, unmatched_xmax = match_periodic_sides(
        boundaries["xmin"],
        boundaries["xmax"],
        axis="x",
        tolerance=tolerance,
    )

    y_links, unmatched_ymin, unmatched_ymax = match_periodic_sides(
        boundaries["ymin"],
        boundaries["ymax"],
        axis="y",
        tolerance=tolerance,
    )

    links = x_links + y_links

    for link in links:
        adjacency[link.element_a].add(link.element_b)
        adjacency[link.element_b].add(link.element_a)

    unmatched = {
        "xmin": unmatched_xmin,
        "xmax": unmatched_xmax,
        "ymin": unmatched_ymin,
        "ymax": unmatched_ymax,
    }

    return PeriodicResult(
        enabled=True,
        tolerance=tolerance,
        bounds=bounds,
        links=links,
        boundary_edges=boundaries,
        unmatched_edges=unmatched,
    )


def empty_periodic_result(
    x: np.ndarray,
    y: np.ndarray,
) -> PeriodicResult:
    return PeriodicResult(
        enabled=False,
        tolerance=None,
        bounds=(
            float(np.min(x)),
            float(np.max(x)),
            float(np.min(y)),
            float(np.max(y)),
        ),
        links=[],
        boundary_edges={
            "xmin": [],
            "xmax": [],
            "ymin": [],
            "ymax": [],
        },
        unmatched_edges={
            "xmin": [],
            "xmax": [],
            "ymin": [],
            "ymax": [],
        },
    )


def convert_grain_values(
    raw_values: np.ndarray,
    integer_tolerance: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert finite, near-integer values to int64 grain IDs.
    """

    raw_values = np.asarray(raw_values, dtype=float).reshape(-1)
    rounded = np.rint(raw_values)

    valid = np.isfinite(raw_values)
    valid &= np.abs(raw_values - rounded) <= integer_tolerance

    grain_ids = np.full(
        raw_values.shape,
        INVALID_GRAIN_ID,
        dtype=np.int64,
    )
    grain_ids[valid] = rounded[valid].astype(np.int64)

    return grain_ids, np.flatnonzero(~valid)


def connected_components_by_grain(
    grain_ids: np.ndarray,
    adjacency: list[set[int]],
    include_ids: set[int] | None = None,
    exclude_ids: set[int] | None = None,
) -> AnalysisResult:
    """
    Flood-fill connected components while crossing only same-ID links.
    """

    exclude_ids = exclude_ids or set()
    number_of_elements = len(grain_ids)

    component_by_element = np.full(
        number_of_elements,
        -1,
        dtype=np.int64,
    )
    visited = np.zeros(number_of_elements, dtype=bool)

    components_by_grain: dict[int, list[np.ndarray]] = defaultdict(list)
    next_component_id = 0

    for seed in range(number_of_elements):
        if visited[seed]:
            continue

        grain_id = int(grain_ids[seed])

        if grain_id == INVALID_GRAIN_ID:
            visited[seed] = True
            continue

        if include_ids is not None and grain_id not in include_ids:
            visited[seed] = True
            continue

        if grain_id in exclude_ids:
            visited[seed] = True
            continue

        queue: deque[int] = deque([seed])
        visited[seed] = True
        members: list[int] = []

        while queue:
            element = queue.popleft()
            members.append(element)

            for neighbor in adjacency[element]:
                if visited[neighbor]:
                    continue

                if int(grain_ids[neighbor]) != grain_id:
                    continue

                visited[neighbor] = True
                queue.append(neighbor)

        member_array = np.asarray(members, dtype=np.int64)
        component_by_element[member_array] = next_component_id
        components_by_grain[grain_id].append(member_array)
        next_component_id += 1

    components_by_grain = dict(components_by_grain)

    duplicate_grains = {
        grain_id: components
        for grain_id, components in components_by_grain.items()
        if len(components) > 1
    }

    return AnalysisResult(
        grain_ids=grain_ids,
        adjacency=adjacency,
        component_by_element=component_by_element,
        components_by_grain=components_by_grain,
        duplicate_grains=duplicate_grains,
        invalid_elements=np.flatnonzero(
            grain_ids == INVALID_GRAIN_ID
        ),
    )


def edge_segment(
    edge: tuple[int, int],
    x: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    node_a, node_b = edge

    return np.asarray(
        [
            [x[node_a], y[node_a]],
            [x[node_b], y[node_b]],
        ],
        dtype=float,
    )


def extract_boundary_segments(
    edge_owners: dict[tuple[int, int], list[int]],
    grain_ids: np.ndarray,
    periodic_result: PeriodicResult,
    x: np.ndarray,
    y: np.ndarray,
    include_exterior: bool,
) -> tuple[list[np.ndarray], list[np.ndarray], int, int]:
    """
    Extract physical and periodic grain-interface line segments.

    If two periodic counterpart elements have different IDs, both domain-edge
    segments are drawn as interfaces.
    """

    interface_segments: list[np.ndarray] = []
    exterior_segments: list[np.ndarray] = []
    nonmanifold_count = 0
    periodic_interface_pairs = 0

    for edge, owners in edge_owners.items():
        segment = edge_segment(edge, x, y)

        if len(owners) == 1:
            if include_exterior:
                exterior_segments.append(segment)
            continue

        if len(owners) > 2:
            nonmanifold_count += 1

        owner_ids = {
            int(grain_ids[element])
            for element in owners
        }

        if len(owner_ids) > 1:
            interface_segments.append(segment)

    if periodic_result.enabled:
        for link in periodic_result.links:
            grain_a = int(grain_ids[link.element_a])
            grain_b = int(grain_ids[link.element_b])

            if grain_a != grain_b:
                periodic_interface_pairs += 1
                interface_segments.append(
                    edge_segment(link.edge_a, x, y)
                )
                interface_segments.append(
                    edge_segment(link.edge_b, x, y)
                )

    return (
        interface_segments,
        exterior_segments,
        nonmanifold_count,
        periodic_interface_pairs,
    )


def make_polygons(
    corner_connectivity: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
) -> list[Polygon]:
    polygons: list[Polygon] = []

    for nodes in corner_connectivity:
        vertices = np.column_stack((x[nodes], y[nodes]))
        polygons.append(Polygon(vertices, closed=True))

    return polygons


def duplicate_color_information(
    duplicate_grains: dict[int, list[np.ndarray]],
) -> tuple[list[int], ListedColormap, BoundaryNorm]:
    duplicate_ids = sorted(duplicate_grains)

    if not duplicate_ids:
        cmap = ListedColormap([(0.85, 0.85, 0.85, 1.0)])
        norm = BoundaryNorm([-0.5, 0.5], cmap.N)
        return duplicate_ids, cmap, norm

    base_cmap = plt.get_cmap("turbo")
    samples = np.linspace(0.05, 0.95, len(duplicate_ids))
    cmap = ListedColormap(
        [base_cmap(value) for value in samples]
    )
    boundaries = (
        np.arange(len(duplicate_ids) + 1, dtype=float) - 0.5
    )
    norm = BoundaryNorm(boundaries, cmap.N)

    return duplicate_ids, cmap, norm


def duplicate_plot_values(
    number_of_elements: int,
    duplicate_grains: dict[int, list[np.ndarray]],
) -> tuple[np.ndarray, list[int], ListedColormap, BoundaryNorm]:
    duplicate_ids, cmap, norm = duplicate_color_information(
        duplicate_grains
    )

    values = np.full(number_of_elements, np.nan, dtype=float)

    for color_index, grain_id in enumerate(duplicate_ids):
        for component in duplicate_grains[grain_id]:
            values[component] = color_index

    return values, duplicate_ids, cmap, norm


def add_polygon_collection(
    axis: plt.Axes,
    polygons: list[Polygon],
    face_values: np.ndarray,
    cmap,
    norm=None,
    background_color: str = "0.88",
) -> PatchCollection:
    """
    Add element polygons with NaN values displayed in a neutral color.
    """

    local_cmap = cmap.copy()
    local_cmap.set_bad(background_color)

    collection = PatchCollection(
        polygons,
        linewidth=0.0,
        edgecolor="none",
        cmap=local_cmap,
        norm=norm,
    )
    collection.set_array(
        np.ma.masked_invalid(
            np.asarray(face_values, dtype=float)
        )
    )

    axis.add_collection(collection)
    axis.autoscale_view()
    axis.set_aspect("equal", adjustable="box")

    return collection


def add_boundary_lines(
    axis: plt.Axes,
    interface_segments: list[np.ndarray],
    exterior_segments: list[np.ndarray],
    line_width: float,
) -> None:
    if interface_segments:
        axis.add_collection(
            LineCollection(
                interface_segments,
                colors="black",
                linewidths=line_width,
                zorder=4,
            )
        )

    if exterior_segments:
        axis.add_collection(
            LineCollection(
                exterior_segments,
                colors="0.25",
                linewidths=max(line_width, 0.4),
                zorder=5,
            )
        )


def component_centers(
    components: list[np.ndarray],
    corner_connectivity: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
) -> list[tuple[float, float]]:
    element_x = x[corner_connectivity].mean(axis=1)
    element_y = y[corner_connectivity].mean(axis=1)

    return [
        (
            float(np.mean(element_x[elements])),
            float(np.mean(element_y[elements])),
        )
        for elements in components
    ]


def save_main_figure(
    output_path: Path,
    source_name: str,
    requested_time: float,
    actual_time: float,
    step: int,
    polygons: list[Polygon],
    result: AnalysisResult,
    periodic_result: PeriodicResult,
    corner_connectivity: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    interface_segments: list[np.ndarray],
    exterior_segments: list[np.ndarray],
    line_width: float,
    dpi: int,
    label_components: bool,
) -> None:
    figure, axis = plt.subplots(
        figsize=(11, 8.5),
        constrained_layout=True,
    )

    (
        plot_values,
        duplicate_ids,
        duplicate_cmap,
        duplicate_norm,
    ) = duplicate_plot_values(
        len(polygons),
        result.duplicate_grains,
    )

    collection = add_polygon_collection(
        axis,
        polygons,
        plot_values,
        cmap=duplicate_cmap,
        norm=duplicate_norm,
        background_color="0.93",
    )

    add_boundary_lines(
        axis,
        interface_segments,
        exterior_segments,
        line_width,
    )

    if duplicate_ids:
        colorbar = figure.colorbar(
            collection,
            ax=axis,
            ticks=np.arange(len(duplicate_ids)),
            fraction=0.035,
            pad=0.02,
        )
        colorbar.ax.set_yticklabels(
            [str(value) for value in duplicate_ids]
        )
        colorbar.set_label("Duplicated unique_grains ID")

    if label_components:
        for grain_id, components in result.duplicate_grains.items():
            centers = component_centers(
                components,
                corner_connectivity,
                x,
                y,
            )

            for component_number, center in enumerate(centers, 1):
                axis.text(
                    center[0],
                    center[1],
                    f"{grain_id}:{component_number}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    fontweight="bold",
                    color="white",
                    bbox={
                        "boxstyle": "round,pad=0.15",
                        "facecolor": "black",
                        "edgecolor": "none",
                        "alpha": 0.65,
                    },
                    zorder=8,
                )

    periodic_label = (
        f"periodic x/y; links={len(periodic_result.links)}"
        if periodic_result.enabled
        else "nonperiodic"
    )

    axis.set_title(
        f"{source_name}\n"
        f"requested time={requested_time:g}, "
        f"selected time={actual_time:g}, step={step}; "
        f"{periodic_label}; "
        f"duplicated IDs={len(result.duplicate_grains)}"
    )
    axis.set_xlabel("x")
    axis.set_ylabel("y")

    if duplicate_ids:
        status_text = (
            "Duplicated IDs: "
            + ", ".join(str(value) for value in duplicate_ids)
        )
    else:
        status_text = "No disconnected duplicate grain IDs detected."

    if periodic_result.enabled:
        status_text += (
            f"\nPeriodic pairs: x={periodic_result.x_link_count}, "
            f"y={periodic_result.y_link_count}; "
            f"unmatched edges={periodic_result.unmatched_count}"
        )

    axis.text(
        0.01,
        0.01,
        status_text,
        transform=axis.transAxes,
        fontsize=8,
        ha="left",
        va="bottom",
        bbox={
            "facecolor": "white",
            "edgecolor": "0.5",
            "alpha": 0.85,
        },
        zorder=10,
    )

    figure.savefig(output_path, dpi=dpi)
    plt.close(figure)


def periodic_debug_segments(
    periodic_result: PeriodicResult,
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[
    dict[str, list[np.ndarray]],
    list[np.ndarray],
]:
    """Create matched-side and unmatched-edge diagnostic segments."""

    side_segments = {
        "xmin": [],
        "xmax": [],
        "ymin": [],
        "ymax": [],
    }

    for link in periodic_result.links:
        side_segments[link.side_a].append(
            edge_segment(link.edge_a, x, y)
        )
        side_segments[link.side_b].append(
            edge_segment(link.edge_b, x, y)
        )

    unmatched_segments: list[np.ndarray] = []

    for edges in periodic_result.unmatched_edges.values():
        for boundary_edge in edges:
            unmatched_segments.append(
                edge_segment(boundary_edge.edge, x, y)
            )

    return side_segments, unmatched_segments


def save_debug_figure(
    output_path: Path,
    source_name: str,
    actual_time: float,
    polygons: list[Polygon],
    result: AnalysisResult,
    periodic_result: PeriodicResult,
    reference_x: np.ndarray,
    reference_y: np.ndarray,
    interface_segments: list[np.ndarray],
    exterior_segments: list[np.ndarray],
    line_width: float,
    dpi: int,
) -> None:
    """
    Save a four-panel diagnostic figure.
    """

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(15, 11),
        constrained_layout=True,
    )

    # Panel 1: all raw integer grain IDs.
    valid_ids = result.grain_ids[
        result.grain_ids != INVALID_GRAIN_ID
    ]
    unique_ids = np.unique(valid_ids)

    raw_values = np.full(
        len(result.grain_ids),
        np.nan,
        dtype=float,
    )
    raw_lookup = {
        int(grain_id): index
        for index, grain_id in enumerate(unique_ids)
    }

    for element, grain_id in enumerate(result.grain_ids):
        grain_id = int(grain_id)

        if grain_id in raw_lookup:
            raw_values[element] = raw_lookup[grain_id]

    raw_cmap = plt.get_cmap(
        "turbo",
        max(len(unique_ids), 1),
    )
    add_polygon_collection(
        axes[0, 0],
        polygons,
        raw_values,
        cmap=raw_cmap,
        background_color="0.8",
    )
    add_boundary_lines(
        axes[0, 0],
        interface_segments,
        exterior_segments,
        line_width,
    )
    axes[0, 0].set_title(
        f"Raw grain field: {len(unique_ids)} IDs"
    )
    axes[0, 0].set_xlabel("x")
    axes[0, 0].set_ylabel("y")

    # Panel 2: duplicated IDs only.
    (
        duplicate_values,
        duplicate_ids,
        duplicate_cmap,
        duplicate_norm,
    ) = duplicate_plot_values(
        len(polygons),
        result.duplicate_grains,
    )

    duplicate_collection = add_polygon_collection(
        axes[0, 1],
        polygons,
        duplicate_values,
        cmap=duplicate_cmap,
        norm=duplicate_norm,
        background_color="0.92",
    )
    add_boundary_lines(
        axes[0, 1],
        interface_segments,
        exterior_segments,
        line_width,
    )
    axes[0, 1].set_title("Duplicates after periodic merging")
    axes[0, 1].set_xlabel("x")
    axes[0, 1].set_ylabel("y")

    if duplicate_ids:
        colorbar = figure.colorbar(
            duplicate_collection,
            ax=axes[0, 1],
            ticks=np.arange(len(duplicate_ids)),
            fraction=0.04,
            pad=0.02,
        )
        colorbar.ax.set_yticklabels(
            [str(value) for value in duplicate_ids]
        )
        colorbar.set_label("Duplicated grain ID")

    # Panel 3: each remaining duplicate component has a unique color.
    component_values = np.full(
        len(polygons),
        np.nan,
        dtype=float,
    )
    component_index = 0

    for grain_id in duplicate_ids:
        for component in result.duplicate_grains[grain_id]:
            component_values[component] = component_index
            component_index += 1

    component_cmap = plt.get_cmap(
        "gist_ncar",
        max(component_index, 1),
    )
    add_polygon_collection(
        axes[1, 0],
        polygons,
        component_values,
        cmap=component_cmap,
        background_color="0.92",
    )
    add_boundary_lines(
        axes[1, 0],
        interface_segments,
        exterior_segments,
        line_width,
    )
    axes[1, 0].set_title(
        f"Remaining disconnected components: {component_index}"
    )
    axes[1, 0].set_xlabel("x")
    axes[1, 0].set_ylabel("y")

    # Panel 4: periodic matching on the reference mesh.
    periodic_axis = axes[1, 1]

    if periodic_result.enabled:
        side_segments, unmatched_segments = periodic_debug_segments(
            periodic_result,
            reference_x,
            reference_y,
        )

        side_styles = {
            "xmin": ("tab:blue", "xmin matched"),
            "xmax": ("tab:cyan", "xmax matched"),
            "ymin": ("tab:green", "ymin matched"),
            "ymax": ("tab:olive", "ymax matched"),
        }

        for side, segments in side_segments.items():
            if not segments:
                continue

            color, label = side_styles[side]
            periodic_axis.add_collection(
                LineCollection(
                    segments,
                    colors=color,
                    linewidths=2.5,
                    label=label,
                )
            )

        if unmatched_segments:
            periodic_axis.add_collection(
                LineCollection(
                    unmatched_segments,
                    colors="red",
                    linewidths=4.0,
                    label="unmatched",
                )
            )

        xmin, xmax, ymin, ymax = periodic_result.bounds
        x_padding = max((xmax - xmin) * 0.02, 1.0e-12)
        y_padding = max((ymax - ymin) * 0.02, 1.0e-12)

        periodic_axis.set_xlim(xmin - x_padding, xmax + x_padding)
        periodic_axis.set_ylim(ymin - y_padding, ymax + y_padding)
        periodic_axis.set_aspect("equal", adjustable="box")
        periodic_axis.set_xlabel("reference x")
        periodic_axis.set_ylabel("reference y")
        periodic_axis.set_title(
            "Periodic edge-pair validation\n"
            f"x pairs={periodic_result.x_link_count}, "
            f"y pairs={periodic_result.y_link_count}, "
            f"unmatched={periodic_result.unmatched_count}"
        )

        if any(side_segments.values()) or unmatched_segments:
            periodic_axis.legend(
                fontsize=7,
                loc="best",
            )
    else:
        periodic_axis.text(
            0.5,
            0.5,
            "Periodic analysis disabled",
            transform=periodic_axis.transAxes,
            ha="center",
            va="center",
            fontsize=14,
        )
        periodic_axis.set_axis_off()

    figure.suptitle(
        f"Connectivity diagnostics: {source_name}, "
        f"selected time={actual_time:g}",
        fontsize=14,
    )
    figure.savefig(output_path, dpi=dpi)
    plt.close(figure)


def validate_analysis(
    result: AnalysisResult,
    edge_owners: dict[tuple[int, int], list[int]],
    periodic_result: PeriodicResult,
) -> list[str]:
    """Run graph, component, and periodic-link consistency checks."""

    messages: list[str] = []
    number_of_elements = len(result.grain_ids)

    if len(result.adjacency) != number_of_elements:
        raise AssertionError(
            "Adjacency length does not match element count."
        )

    asymmetric_links = []

    for element, neighbors in enumerate(result.adjacency):
        for neighbor in neighbors:
            if element not in result.adjacency[neighbor]:
                asymmetric_links.append((element, neighbor))

    if asymmetric_links:
        raise AssertionError(
            f"Adjacency is asymmetric at "
            f"{len(asymmetric_links)} links."
        )

    messages.append(
        "PASS: complete element adjacency graph is symmetric"
    )

    assigned_members: list[int] = []

    for grain_id, components in result.components_by_grain.items():
        for component in components:
            if len(component) == 0:
                raise AssertionError(
                    f"Grain {grain_id} has an empty component."
                )

            if not np.all(
                result.grain_ids[component] == grain_id
            ):
                raise AssertionError(
                    f"Component for grain {grain_id} contains another ID."
                )

            assigned_members.extend(component.tolist())

    if len(assigned_members) != len(set(assigned_members)):
        raise AssertionError(
            "At least one element belongs to multiple components."
        )

    messages.append(
        "PASS: components are nonempty and nonoverlapping"
    )

    crossing_same_id_links = 0

    for element, neighbors in enumerate(result.adjacency):
        for neighbor in neighbors:
            if neighbor <= element:
                continue

            if result.grain_ids[element] != result.grain_ids[neighbor]:
                continue

            component_a = result.component_by_element[element]
            component_b = result.component_by_element[neighbor]

            if (
                component_a >= 0
                and component_b >= 0
                and component_a != component_b
            ):
                crossing_same_id_links += 1

    if crossing_same_id_links:
        raise AssertionError(
            f"{crossing_same_id_links} same-ID neighbor links cross "
            "component assignments."
        )

    messages.append(
        "PASS: every analyzed same-ID neighbor pair belongs to one component"
    )

    nonmanifold_count = sum(
        len(owners) > 2
        for owners in edge_owners.values()
    )
    messages.append(
        f"{'WARNING' if nonmanifold_count else 'PASS'}: "
        f"non-manifold mesh edges={nonmanifold_count}"
    )

    if periodic_result.enabled:
        seen_periodic_edges: set[
            tuple[str, tuple[int, int]]
        ] = set()

        for link in periodic_result.links:
            if link.element_b not in result.adjacency[link.element_a]:
                raise AssertionError(
                    "A periodic link is absent from adjacency."
                )

            if link.element_a not in result.adjacency[link.element_b]:
                raise AssertionError(
                    "A periodic link is not symmetric."
                )

            for side, edge in (
                (link.side_a, link.edge_a),
                (link.side_b, link.edge_b),
            ):
                key = (side, edge)

                if key in seen_periodic_edges:
                    raise AssertionError(
                        f"Periodic edge {edge} on {side} was matched twice."
                    )

                seen_periodic_edges.add(key)

            if link.mismatch > float(periodic_result.tolerance):
                raise AssertionError(
                    "A periodic edge pair exceeds the matching tolerance."
                )

        messages.append(
            "PASS: periodic links are symmetric, one-to-one, and within "
            "the requested tolerance"
        )
        messages.append(
            f"{'WARNING' if periodic_result.unmatched_count else 'PASS'}: "
            f"unmatched periodic edges={periodic_result.unmatched_count}"
        )

    return messages


def run_synthetic_self_tests() -> list[str]:
    """
    Test ordinary components and periodic side matching.
    """

    messages: list[str] = []

    # Component test: grain 7 is disconnected; grain 9 is connected.
    grain_ids = np.asarray([7, 7, 7, 7, 9, 9], dtype=np.int64)
    adjacency = [
        {1},
        {0, 4},
        {3},
        {2, 5},
        {1, 5},
        {3, 4},
    ]

    ordinary_result = connected_components_by_grain(
        grain_ids,
        adjacency,
    )

    assert 7 in ordinary_result.duplicate_grains
    assert len(ordinary_result.duplicate_grains[7]) == 2
    assert 9 not in ordinary_result.duplicate_grains

    messages.append(
        "PASS: synthetic disconnected-ID test identified grain 7"
    )
    messages.append(
        "PASS: synthetic connected-ID test did not flag grain 9"
    )

    # Periodic matching test.
    xmin_edges = [
        BoundaryEdge((0, 1), 0, "xmin", 0.0, 1.0),
        BoundaryEdge((1, 2), 1, "xmin", 1.0, 2.0),
    ]
    xmax_edges = [
        BoundaryEdge((3, 4), 2, "xmax", 0.0, 1.0),
        BoundaryEdge((4, 5), 3, "xmax", 1.0, 2.0),
    ]

    links, unmatched_a, unmatched_b = match_periodic_sides(
        xmin_edges,
        xmax_edges,
        axis="x",
        tolerance=1.0e-12,
    )

    assert len(links) == 2
    assert not unmatched_a
    assert not unmatched_b
    assert {link.element_a for link in links} == {0, 1}
    assert {link.element_b for link in links} == {2, 3}

    messages.append(
        "PASS: synthetic periodic-edge test created two exact "
        "one-to-one pairs"
    )

    # Ensure periodic graph links merge equal-ID regions.
    periodic_ids = np.asarray([12, 12], dtype=np.int64)
    periodic_adjacency = [set(), set()]
    periodic_adjacency[0].add(1)
    periodic_adjacency[1].add(0)

    merged_result = connected_components_by_grain(
        periodic_ids,
        periodic_adjacency,
    )

    assert len(merged_result.components_by_grain[12]) == 1
    assert 12 not in merged_result.duplicate_grains

    messages.append(
        "PASS: synthetic periodic link merged equal-ID boundary regions"
    )

    return messages


def boundary_edge_description(edge: BoundaryEdge) -> str:
    return (
        f"side={edge.side}, element={edge.element}, "
        f"edge={edge.edge}, tangent="
        f"[{edge.tangent_low:.17g}, {edge.tangent_high:.17g}]"
    )


def write_debug_report(
    output_path: Path,
    source_file: Path,
    variable: str,
    block: int,
    requested_time: float,
    selected_time: float,
    step: int,
    element_type: str,
    result: AnalysisResult,
    periodic_result: PeriodicResult,
    edge_owners: dict[tuple[int, int], list[int]],
    interface_count: int,
    periodic_interface_pairs: int,
    nonmanifold_count: int,
    validation_messages: list[str],
    self_test_messages: list[str],
) -> None:
    lines = [
        "Split-grain connectivity diagnostic report",
        "=" * 44,
        f"Source file: {source_file}",
        f"Variable: {variable}",
        f"Element block: {block}",
        f"Element type: {element_type}",
        f"Requested time: {requested_time:.17g}",
        f"Selected time: {selected_time:.17g}",
        f"Selected zero-based timestep: {step}",
        (
            "Absolute time difference: "
            f"{abs(selected_time - requested_time):.17g}"
        ),
        "",
        "Mesh statistics",
        "---------------",
        f"Elements: {len(result.grain_ids)}",
        f"Unique mesh edges: {len(edge_owners)}",
        (
            "Shared mesh edges: "
            f"{sum(len(v) >= 2 for v in edge_owners.values())}"
        ),
        (
            "Exterior mesh edges: "
            f"{sum(len(v) == 1 for v in edge_owners.values())}"
        ),
        f"Non-manifold mesh edges: {nonmanifold_count}",
        f"Plotted interface segments: {interface_count}",
        (
            "Periodic different-ID interface pairs: "
            f"{periodic_interface_pairs}"
        ),
        (
            "Invalid/noninteger variable elements: "
            f"{len(result.invalid_elements)}"
        ),
        "",
        "Periodic-boundary analysis",
        "--------------------------",
        f"Enabled: {periodic_result.enabled}",
        f"Domain bounds: {periodic_result.bounds}",
        f"Matching tolerance: {periodic_result.tolerance}",
        f"x periodic pairs: {periodic_result.x_link_count}",
        f"y periodic pairs: {periodic_result.y_link_count}",
        f"Unmatched periodic edges: {periodic_result.unmatched_count}",
    ]

    if periodic_result.enabled:
        for side in ("xmin", "xmax", "ymin", "ymax"):
            lines.append(
                f"{side} boundary edges: "
                f"{len(periodic_result.boundary_edges[side])}; "
                f"unmatched: "
                f"{len(periodic_result.unmatched_edges[side])}"
            )

        lines.extend(
            [
                "",
                "Periodic links",
                "--------------",
            ]
        )

        for link_number, link in enumerate(
            periodic_result.links,
            start=1,
        ):
            lines.append(
                f"{link_number}: axis={link.axis}, "
                f"{link.side_a} element={link.element_a} edge={link.edge_a} "
                f"<-> {link.side_b} element={link.element_b} "
                f"edge={link.edge_b}; mismatch={link.mismatch:.17g}"
            )

        if periodic_result.unmatched_count:
            lines.extend(
                [
                    "",
                    "Unmatched periodic edges",
                    "------------------------",
                ]
            )

            for side in ("xmin", "xmax", "ymin", "ymax"):
                for edge in periodic_result.unmatched_edges[side]:
                    lines.append(boundary_edge_description(edge))

    lines.extend(
        [
            "",
            "Synthetic self-tests",
            "--------------------",
            *self_test_messages,
            "",
            "Dataset consistency checks",
            "--------------------------",
            *validation_messages,
            "",
            "Component summary",
            "-----------------",
        ]
    )

    for grain_id in sorted(result.components_by_grain):
        components = result.components_by_grain[grain_id]
        sizes = [len(component) for component in components]
        status = (
            "DUPLICATED"
            if len(components) > 1
            else "connected"
        )

        lines.append(
            f"ID {grain_id}: components={len(components)}, "
            f"sizes={sizes}, status={status}"
        )

    lines.extend(
        [
            "",
            "Duplicated grain IDs",
            "--------------------",
        ]
    )

    if result.duplicate_grains:
        for grain_id in sorted(result.duplicate_grains):
            components = result.duplicate_grains[grain_id]

            lines.append(
                f"ID {grain_id}: {len(components)} disconnected "
                f"components; element counts="
                f"{[len(component) for component in components]}"
            )

            for component_number, component in enumerate(
                components,
                start=1,
            ):
                lines.append(
                    f"  component {component_number}: "
                    f"element indices={component.tolist()}"
                )
    else:
        lines.append("None detected.")

    if len(result.invalid_elements):
        lines.extend(
            [
                "",
                "Invalid/noninteger element indices",
                "----------------------------------",
                str(result.invalid_elements.tolist()),
            ]
        )

    output_path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def safe_time_tag(value: float) -> str:
    return (
        f"{value:.8g}"
        .replace("-", "m")
        .replace("+", "p")
        .replace(".", "p")
    )


def analyze_file(
    path: Path,
    args: argparse.Namespace,
) -> None:
    LOGGER.info("Analyzing %s", path)

    with ExodusBasics(str(path)) as exodus:
        times = np.asarray(
            exodus.time(),
            dtype=float,
        ).reshape(-1)

        if times.size == 0:
            raise ValueError("The Exodus file has no timesteps.")

        if not np.all(np.isfinite(times)):
            raise ValueError(
                "The Exodus time vector contains nonfinite values."
            )

        step = int(np.argmin(np.abs(times - args.time)))
        actual_time = float(times[step])

        LOGGER.info(
            "Requested time=%g; selected step=%d at time=%g",
            args.time,
            step,
            actual_time,
        )

        element_variables = exodus.elem_varnames()

        if args.variable not in element_variables:
            raise KeyError(
                f"{args.variable!r} is not an elemental variable. "
                f"Available elemental variables: {element_variables}"
            )

        connectivity = np.asarray(
            exodus.connectivity(
                which=args.block,
                zero_based=True,
            ),
            dtype=np.int64,
        )

        if connectivity.ndim != 2:
            raise ValueError(
                "Expected a two-dimensional connectivity array; "
                f"found shape {connectivity.shape}."
            )

        element_type = exodus.element_type(args.block)
        number_of_corners = corner_count(
            element_type,
            connectivity.shape[1],
        )
        corner_connectivity = connectivity[:, :number_of_corners]

        raw_grain_values = np.asarray(
            exodus.elem_var_at_step(
                args.variable,
                step,
                eb=args.block,
            )
        ).reshape(-1)

        if len(raw_grain_values) != len(connectivity):
            raise ValueError(
                f"Variable length {len(raw_grain_values)} does not "
                f"match connectivity length {len(connectivity)}."
            )

        # Reference coordinates are always used for periodic matching.
        reference_x, reference_y = exodus.coords_xy()
        reference_x = np.asarray(reference_x, dtype=float)
        reference_y = np.asarray(reference_y, dtype=float)

        if args.use_static_coordinates:
            plot_x = reference_x.copy()
            plot_y = reference_y.copy()
        else:
            plot_x, plot_y = exodus.coords_xy_at_step(step)
            plot_x = np.asarray(plot_x, dtype=float)
            plot_y = np.asarray(plot_y, dtype=float)

    # Exodus file is now closed; remaining operations use NumPy arrays.
    if corner_connectivity.size:
        minimum_node = int(corner_connectivity.min())
        maximum_node = int(corner_connectivity.max())

        if minimum_node < 0 or maximum_node >= len(reference_x):
            raise IndexError(
                "Connectivity references nodes outside the coordinate "
                f"arrays: node range={minimum_node}..{maximum_node}, "
                f"available=0..{len(reference_x) - 1}."
            )

    grain_ids, invalid_elements = convert_grain_values(
        raw_grain_values,
        args.integer_tolerance,
    )

    if len(invalid_elements):
        examples = [
            (
                int(index),
                float(raw_grain_values[index]),
            )
            for index in invalid_elements[:10]
        ]

        LOGGER.warning(
            "%s contains %d nonfinite or noninteger grain values; "
            "examples=%s",
            path.name,
            len(invalid_elements),
            examples,
        )

    edge_owners = build_edge_ownership(corner_connectivity)
    adjacency = build_element_adjacency(
        len(corner_connectivity),
        edge_owners,
    )

    if args.periodic:
        periodic_tolerance = (
            args.periodic_tolerance
            if args.periodic_tolerance is not None
            else default_periodic_tolerance(
                reference_x,
                reference_y,
            )
        )

        periodic_result = add_periodic_connectivity(
            adjacency,
            edge_owners,
            reference_x,
            reference_y,
            periodic_tolerance,
        )

        LOGGER.info(
            "%s periodic matching: x pairs=%d, y pairs=%d, "
            "unmatched=%d, tolerance=%g",
            path.name,
            periodic_result.x_link_count,
            periodic_result.y_link_count,
            periodic_result.unmatched_count,
            periodic_tolerance,
        )

        if periodic_result.unmatched_count:
            message = (
                f"{path.name} has "
                f"{periodic_result.unmatched_count} unmatched periodic "
                "boundary edges. The opposite sides may be nonconforming, "
                "the domain may not be rectangular, or "
                "--periodic-tolerance may need adjustment."
            )

            if args.allow_unmatched_periodic_edges:
                LOGGER.warning(message)
            else:
                raise RuntimeError(
                    message
                    + " Use --allow-unmatched-periodic-edges to continue "
                    "with only the successfully matched pairs."
                )
    else:
        periodic_result = empty_periodic_result(
            reference_x,
            reference_y,
        )

    include_ids = (
        set(args.include_id)
        if args.include_id
        else None
    )
    exclude_ids = set(args.exclude_id or [])

    result = connected_components_by_grain(
        grain_ids,
        adjacency,
        include_ids=include_ids,
        exclude_ids=exclude_ids,
    )

    (
        interface_segments,
        exterior_segments,
        nonmanifold_count,
        periodic_interface_pairs,
    ) = extract_boundary_segments(
        edge_owners,
        grain_ids,
        periodic_result,
        plot_x,
        plot_y,
        include_exterior=args.show_exterior,
    )

    if nonmanifold_count:
        LOGGER.warning(
            "%s contains %d edge(s) owned by more than two elements.",
            path.name,
            nonmanifold_count,
        )

    polygons = make_polygons(
        corner_connectivity,
        plot_x,
        plot_y,
    )

    args.output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    time_tag = safe_time_tag(actual_time)
    periodic_tag = "_periodic" if args.periodic else ""
    base_name = (
        f"{path.stem}_split_grains_t{time_tag}{periodic_tag}"
    )

    main_output = args.output_directory / f"{base_name}.png"

    save_main_figure(
        output_path=main_output,
        source_name=path.name,
        requested_time=args.time,
        actual_time=actual_time,
        step=step,
        polygons=polygons,
        result=result,
        periodic_result=periodic_result,
        corner_connectivity=corner_connectivity,
        x=plot_x,
        y=plot_y,
        interface_segments=interface_segments,
        exterior_segments=exterior_segments,
        line_width=args.line_width,
        dpi=args.dpi,
        label_components=args.label_components,
    )

    LOGGER.warning(
        "%s: selected time=%g; periodic=%s; duplicated IDs=%s; "
        "plot=%s",
        path.name,
        actual_time,
        args.periodic,
        sorted(result.duplicate_grains),
        main_output,
    )

    if args.debug:
        self_test_messages = run_synthetic_self_tests()

        validation_messages = validate_analysis(
            result,
            edge_owners,
            periodic_result,
        )

        debug_figure = (
            args.output_directory
            / f"{base_name}_debug.png"
        )
        save_debug_figure(
            output_path=debug_figure,
            source_name=path.name,
            actual_time=actual_time,
            polygons=polygons,
            result=result,
            periodic_result=periodic_result,
            reference_x=reference_x,
            reference_y=reference_y,
            interface_segments=interface_segments,
            exterior_segments=exterior_segments,
            line_width=args.line_width,
            dpi=args.dpi,
        )

        debug_report = (
            args.output_directory
            / f"{base_name}_debug.txt"
        )
        write_debug_report(
            output_path=debug_report,
            source_file=path,
            variable=args.variable,
            block=args.block,
            requested_time=args.time,
            selected_time=actual_time,
            step=step,
            element_type=element_type,
            result=result,
            periodic_result=periodic_result,
            edge_owners=edge_owners,
            interface_count=len(interface_segments),
            periodic_interface_pairs=periodic_interface_pairs,
            nonmanifold_count=nonmanifold_count,
            validation_messages=validation_messages,
            self_test_messages=self_test_messages,
        )

        LOGGER.info("Debug figure: %s", debug_figure)
        LOGGER.info("Debug report: %s", debug_report)


def validate_arguments(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> None:
    if args.block < 1:
        parser.error("--block must be at least 1.")

    if args.integer_tolerance < 0:
        parser.error("--integer-tolerance cannot be negative.")

    if (
        args.periodic_tolerance is not None
        and args.periodic_tolerance <= 0
    ):
        parser.error("--periodic-tolerance must be positive.")

    if args.periodic_tolerance is not None and not args.periodic:
        parser.error(
            "--periodic-tolerance requires --periodic."
        )

    if (
        args.allow_unmatched_periodic_edges
        and not args.periodic
    ):
        parser.error(
            "--allow-unmatched-periodic-edges requires --periodic."
        )

    if args.dpi <= 0:
        parser.error("--dpi must be positive.")

    if args.line_width < 0:
        parser.error("--line-width cannot be negative.")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    validate_arguments(parser, args)
    configure_logging(args.verbose)

    try:
        files = discover_files(args)
    except Exception:
        LOGGER.exception("Could not search for input files.")
        return 2

    if not files:
        LOGGER.error(
            "No Exodus files matched the requested selection in %s.",
            args.directory,
        )
        return 2

    failures = 0

    for path in files:
        try:
            analyze_file(path, args)
        except Exception:
            failures += 1
            LOGGER.exception("Analysis failed for %s", path)

    successes = len(files) - failures

    LOGGER.warning(
        "Finished: %d succeeded, %d failed, %d total.",
        successes,
        failures,
        len(files),
    )

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
