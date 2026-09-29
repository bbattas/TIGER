#!/usr/bin/env python3
"""
Identify connected, unique grains from reused Exodus elemental phase fields.

Example:
    python identify_unique_grains.py simulation.e \
        --time 125.0 \
        --threshold 0.5

Output:
    simulation_unique_grains_step000042.npz

The .npz file contains:
    unique_grains       Integer grain ID per Exodus element. 0 = background.
    element_ids         1-based global Exodus element IDs matching unique_grains.
    dominant_op_index   Index into op_names for the winning gr* variable; -1 = background.
    op_names            Names of phase-field elemental variables used.
    requested_time      Time requested with --time.
    actual_time         Exodus time selected.
    step                Zero-based Exodus timestep index.
    block_ids           Exodus element-block number for each element.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
from netCDF4 import Dataset


class UnionFind:
    def __init__(self, n: int):
        self.parent = np.arange(n, dtype=np.int64)
        self.rank = np.zeros(n, dtype=np.int8)

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        root_a = self.find(a)
        root_b = self.find(b)

        if root_a == root_b:
            return

        if self.rank[root_a] < self.rank[root_b]:
            root_a, root_b = root_b, root_a

        self.parent[root_b] = root_a

        if self.rank[root_a] == self.rank[root_b]:
            self.rank[root_a] += 1


def decode_name_array(char_array) -> list[str]:
    """Decode Exodus char[N, name_length] name arrays."""
    array = np.asarray(char_array)
    names = []

    for row in array:
        if row.dtype.kind == "S":
            name = b"".join(row.tolist()).decode("utf-8", "ignore").strip()
        else:
            pieces = []
            for value in row:
                if isinstance(value, bytes):
                    pieces.append(value.decode("utf-8", "ignore"))
                else:
                    pieces.append(str(value))
            name = "".join(pieces).strip()

        names.append(name)

    return names


def decode_attribute(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", "ignore").strip()
    return str(value).strip()


def natural_name_key(name: str):
    """Sort gr2 before gr10."""
    pieces = re.split(r"(\d+)", name)
    return [int(piece) if piece.isdigit() else piece.lower() for piece in pieces]


def local_faces(element_type: str) -> list[tuple[int, ...]]:
    """
    Return corner-node indices for element faces/edges.

    For 2D elements, a face is an edge.
    For 3D elements, a face is a cell face.

    QUAD8/QUAD9 and TRI6 are supported because their corner nodes follow
    the standard Exodus ordering.
    """
    element_type = element_type.upper()

    if element_type.startswith("QUAD"):
        return [(0, 1), (1, 2), (2, 3), (3, 0)]

    if element_type.startswith("TRI"):
        return [(0, 1), (1, 2), (2, 0)]

    if element_type.startswith("HEX"):
        return [
            (0, 1, 2, 3),
            (4, 5, 6, 7),
            (0, 1, 5, 4),
            (1, 2, 6, 5),
            (2, 3, 7, 6),
            (3, 0, 4, 7),
        ]

    if element_type.startswith("TET"):
        return [
            (0, 1, 2),
            (0, 1, 3),
            (1, 2, 3),
            (0, 2, 3),
        ]

    raise ValueError(
        f"Unsupported Exodus element type {element_type!r}. "
        "This script currently supports TRI*, QUAD*, TET*, and HEX* blocks."
    )


def read_nodal_variable(
    dataset: Dataset,
    variable_number: int,
    step: int,
) -> np.ndarray:
    """
    Read one Exodus nodal variable at a single zero-based timestep.

    Exodus nodal variable numbers are 1-based:
        vals_nod_var1, vals_nod_var2, ...
    """
    variable_name = f"vals_nod_var{variable_number}"

    if variable_name not in dataset.variables:
        raise KeyError(
            f"Expected Exodus nodal variable {variable_name!r} was not found."
        )

    return np.asarray(
        dataset.variables[variable_name][step, :],
        dtype=np.float64,
    )


def nodal_values_to_element_values(
    nodal_values: np.ndarray,
    connectivity: np.ndarray,
) -> np.ndarray:
    """
    Average nodal values over each element.

    Raw Exodus connectivity conventionally uses 1-based node IDs, while
    NumPy indexing is zero-based.  The returned array has one value per
    element in this block.
    """
    node_indices = np.asarray(connectivity, dtype=np.int64) - 1

    if node_indices.min() < 0 or node_indices.max() >= nodal_values.size:
        raise ValueError(
            "Connectivity appears incompatible with nodal variable length. "
            "Expected 1-based Exodus node IDs."
        )

    return np.mean(nodal_values[node_indices], axis=1)

def element_centers_from_connectivity(
    connectivity: np.ndarray,
    x_nodes: np.ndarray,
    y_nodes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute geometric element centers by averaging the coordinates of
    all nodes belonging to each Exodus element.

    Exodus connectivity is conventionally 1-based; NumPy is 0-based.
    """
    node_indices = np.asarray(connectivity, dtype=np.int64) - 1

    center_x = np.mean(x_nodes[node_indices], axis=1)
    center_y = np.mean(y_nodes[node_indices], axis=1)

    return center_x, center_y


def identify_unique_grains(
    exodus_filename: Path,
    requested_time: float,
    threshold: float,
    op_regex: str,
):
    with Dataset(exodus_filename, mode="r") as dataset:
        dataset.set_auto_maskandscale(False)

        if "time_whole" not in dataset.variables:
            raise KeyError("The Exodus file does not contain the time_whole variable.")

        times = np.asarray(dataset.variables["time_whole"][:], dtype=np.float64)

        if times.size == 0:
            raise ValueError("The Exodus file contains no timesteps in time_whole.")

        step = int(np.argmin(np.abs(times - requested_time)))
        actual_time = float(times[step])

        if "name_nod_var" not in dataset.variables:
            raise KeyError(
                "The Exodus file has no name_nod_var array. "
                "This script requires nodal gr* phase-field variables."
            )

        nodal_variable_names = decode_name_array(
            dataset.variables["name_nod_var"][:]
        )

        pattern = re.compile(op_regex)
        op_entries = [
            (index + 1, name)
            for index, name in enumerate(nodal_variable_names)
            if name and pattern.fullmatch(name)
        ]
        op_entries.sort(key=lambda item: natural_name_key(item[1]))

        if not op_entries:
            raise ValueError(
                f"No nodal variables matched regex {op_regex!r}. "
                f"Available nodal variables: {nodal_variable_names}"
            )

        op_variable_numbers = [entry[0] for entry in op_entries]
        op_names = [entry[1] for entry in op_entries]

        block_numbers = []
        block_connectivity = []
        block_types = []
        block_offsets = []

        offset = 0
        block_number = 1

        while f"connect{block_number}" in dataset.variables:
            connection_variable = dataset.variables[f"connect{block_number}"]
            connectivity = np.asarray(connection_variable[:], dtype=np.int64)

            if connectivity.ndim != 2:
                raise ValueError(
                    f"connect{block_number} should be 2D, got shape {connectivity.shape}."
                )

            element_type = decode_attribute(
                getattr(connection_variable, "elem_type", "")
            )

            if not element_type:
                raise ValueError(
                    f"connect{block_number} has no elem_type attribute; "
                    "cannot safely determine element adjacency."
                )

            # Validate supported topology before beginning segmentation.
            local_faces(element_type)

            block_count = connectivity.shape[0]

            block_numbers.append(block_number)
            block_connectivity.append(connectivity)
            block_types.append(element_type)
            block_offsets.append(offset)

            offset += block_count
            block_number += 1

        if not block_numbers:
            raise ValueError("No Exodus connect# element-block variables were found.")

        total_elements = offset
        dominant_op_index = np.full(total_elements, -1, dtype=np.int32)
        block_ids = np.empty(total_elements, dtype=np.int32)

        element_center_x = np.empty(total_elements, dtype=np.float64)
        element_center_y = np.empty(total_elements, dtype=np.float64)

        x_nodes = np.asarray(dataset.variables["coordx"][:], dtype=np.float64)
        y_nodes = np.asarray(dataset.variables["coordy"][:], dtype=np.float64)

        # Read each requested nodal phase field once for the selected timestep.
        nodal_phase_values = [
            read_nodal_variable(dataset, variable_number, step)
            for variable_number in op_variable_numbers
        ]

        # Convert every nodal phase field to element-centered values by averaging
        # over each element's nodes, then identify the dominant OP per element.
        for block_number, connectivity, block_offset in zip(
            block_numbers, block_connectivity, block_offsets
        ):
            count = connectivity.shape[0]
            phase_values = np.empty((len(op_variable_numbers), count), dtype=np.float64)

            for local_op_index, nodal_values in enumerate(nodal_phase_values):
                phase_values[local_op_index] = nodal_values_to_element_values(
                    nodal_values,
                    connectivity,
                )

            winner = np.argmax(phase_values, axis=0)
            winner_value = phase_values[winner, np.arange(count)]

            # Elements without a sufficiently strong phase-field value are background.
            active = np.isfinite(winner_value) & (winner_value >= threshold)

            center_x, center_y = element_centers_from_connectivity(
                connectivity,
                x_nodes,
                y_nodes,
            )

            element_slice = slice(block_offset, block_offset + count)
            element_center_x[element_slice] = center_x
            element_center_y[element_slice] = center_y
            dominant_op_index[element_slice] = np.where(active, winner, -1)
            block_ids[element_slice] = block_number

        # Merge same-phase elements that share a complete edge (2D) or face (3D).
        union_find = UnionFind(total_elements)
        face_owner: dict[tuple[int, ...], int] = {}

        for connectivity, element_type, block_offset in zip(
            block_connectivity, block_types, block_offsets
        ):
            faces = local_faces(element_type)

            for local_element_index, nodes in enumerate(connectivity):
                global_element_index = block_offset + local_element_index

                if dominant_op_index[global_element_index] < 0:
                    continue

                for face in faces:
                    face_key = tuple(sorted(int(nodes[node_index]) for node_index in face))

                    previous_element = face_owner.get(face_key)

                    if previous_element is None:
                        face_owner[face_key] = global_element_index
                    elif (
                        dominant_op_index[previous_element]
                        == dominant_op_index[global_element_index]
                    ):
                        union_find.union(previous_element, global_element_index)

        # Convert union-find roots to compact, globally unique grain IDs.
        unique_grains = np.zeros(total_elements, dtype=np.int64)
        root_to_grain_id: dict[int, int] = {}
        next_grain_id = 1

        for element_index in range(total_elements):
            if dominant_op_index[element_index] < 0:
                continue

            root = union_find.find(element_index)

            if root not in root_to_grain_id:
                root_to_grain_id[root] = next_grain_id
                next_grain_id += 1

            unique_grains[element_index] = root_to_grain_id[root]

    return {
        "unique_grains": unique_grains,
        "element_ids": np.arange(1, total_elements + 1, dtype=np.int64),
        "dominant_op_index": dominant_op_index,
        "op_names": np.asarray(op_names),
        "requested_time": np.asarray(requested_time, dtype=np.float64),
        "actual_time": np.asarray(actual_time, dtype=np.float64),
        "step": np.asarray(step, dtype=np.int64),
        "block_ids": block_ids,
        "element_center_x": element_center_x,
        "element_center_y": element_center_y,
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Create unique connected grain IDs from reused Exodus elemental gr* "
            "phase-field variables."
        )
    )
    parser.add_argument("exodus_file", type=Path, help="Input ExodusII file.")
    parser.add_argument(
        "--time",
        type=float,
        required=True,
        help="Requested physical time; the nearest Exodus timestep is used.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help=(
            "Minimum dominant phase-field value needed to classify an element "
            "as part of a grain. Default: 0.5."
        ),
    )
    parser.add_argument(
        "--op-regex",
        default=r"gr\d+",
        help=(
            "Regular expression matched against elemental variable names. "
            "Default: gr\\d+"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output .npz filename. Defaults to a name based on the Exodus file.",
    )
    args = parser.parse_args()

    if not args.exodus_file.is_file():
        parser.error(f"Input file does not exist: {args.exodus_file}")

    result = identify_unique_grains(
        exodus_filename=args.exodus_file,
        requested_time=args.time,
        threshold=args.threshold,
        op_regex=args.op_regex,
    )

    if args.output is None:
        stem = args.exodus_file.with_suffix("")
        step = int(result["step"])
        output = Path(f"{stem}_unique_grains_step{step:06d}.npz")
    else:
        output = args.output

    if output.suffix.lower() != ".npz":
        output = output.with_suffix(".npz")

    np.savez_compressed(output, **result)

    print(f"Requested time: {float(result['requested_time']):.16g}")
    print(f"Selected time:  {float(result['actual_time']):.16g}")
    print(f"Selected step:  {int(result['step'])}")
    print(f"Elements:       {result['unique_grains'].size}")
    print(f"Unique grains:  {int(result['unique_grains'].max())}")
    print(f"Wrote:          {output}")


if __name__ == "__main__":
    main()
