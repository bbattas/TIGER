#!/usr/bin/env python3
"""
Convert a directory of MATLAB phase-field outputs into one ExodusII file.

Expected input filenames:
    anything_nn0
    anything_nn10
    anything_nn20.mat

The filename must end in nn{step}, optionally followed by ".mat".

Each input file must contain a 2D variable named "microstructure".
Each matrix entry becomes one QUAD4 element, with:

    Exodus elemental variable: unique_grains
    Exodus time:               step * dt
    Element dimensions:        dx = dy = 1

Element numbering follows NumPy/MATLAB matrix layout after loading:
rows correspond to increasing y and columns correspond to increasing x.

Example:
    python mat_series_to_exodus.py ./results output.e --dt 0.0025

Dependencies:
    pip install numpy scipy netCDF4 h5py
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
from netCDF4 import Dataset
from scipy.io import loadmat


STEP_PATTERN = re.compile(r"nn(\d+)(?:\.mat)?$", re.IGNORECASE)
LEN_STRING = 33
LEN_LINE = 81


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert MATLAB microstructure files ending in nn{step} "
            "into one ExodusII time-series file."
        )
    )
    parser.add_argument(
        "input_directory",
        type=Path,
        help="Directory containing the MATLAB files.",
    )
    parser.add_argument(
        "output_file",
        type=Path,
        help="Output ExodusII filename, for example output.e.",
    )
    parser.add_argument(
        "--dt",
        type=float,
        required=True,
        help="Simulation timestep size. Exodus time is dt * nn{step}.",
    )
    parser.add_argument(
        "--variable",
        default="microstructure",
        help="MATLAB variable to read (default: microstructure).",
    )
    parser.add_argument(
        "--exodus-variable",
        default="unique_grains",
        help="Elemental Exodus variable name (default: unique_grains).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output file if it already exists.",
    )
    return parser.parse_args()


def discover_input_files(directory: Path) -> list[tuple[int, Path]]:
    """Find files ending in nn{step} or nn{step}.mat and sort by step."""
    if not directory.is_dir():
        raise NotADirectoryError(f"Input directory does not exist: {directory}")

    matches: list[tuple[int, Path]] = []

    for path in directory.iterdir():
        if not path.is_file():
            continue

        match = STEP_PATTERN.search(path.name)
        if match:
            matches.append((int(match.group(1)), path))

    matches.sort(key=lambda item: item[0])

    if not matches:
        raise FileNotFoundError(
            f"No files ending in nn{{step}} or nn{{step}}.mat found in {directory}"
        )

    steps = [step for step, _ in matches]
    duplicates = sorted({step for step in steps if steps.count(step) > 1})
    if duplicates:
        raise ValueError(
            "More than one input file was found for these steps: "
            + ", ".join(map(str, duplicates))
        )

    return matches


def load_ascii_matrix(path: Path, variable_name: str) -> np.ndarray:
    """
    Load a named matrix from an Octave text-format file.

    Supports files containing multiple variables, for example:

        # name: microstructure
        # type: matrix
        # rows: 100
        # columns: 100
        ...
        # name: grain_area
        # type: matrix
        ...
    """
    with path.open("r", encoding="utf-8") as stream:
        lines = stream.readlines()

    available_variables = []
    variable_start = None

    # Find the requested variable's '# name:' section.
    for index, line in enumerate(lines):
        stripped = line.strip()

        if stripped.startswith("# name:"):
            found_name = stripped.split(":", 1)[1].strip()
            available_variables.append(found_name)

            if found_name == variable_name:
                variable_start = index
                break

    if variable_start is None:
        raise KeyError(
            f"{path}: variable {variable_name!r} was not found. "
            f"Available variables: {available_variables}"
        )

    matrix_type = None
    number_of_rows = None
    number_of_columns = None
    data_start = None

    # Read metadata belonging only to the requested variable.
    for index in range(variable_start + 1, len(lines)):
        stripped = lines[index].strip()

        # Reaching another variable means the requested variable's
        # metadata/data section has ended.
        if stripped.startswith("# name:"):
            break

        if stripped.startswith("# type:"):
            matrix_type = stripped.split(":", 1)[1].strip()

        elif stripped.startswith("# rows:"):
            number_of_rows = int(stripped.split(":", 1)[1].strip())

        elif stripped.startswith("# columns:"):
            number_of_columns = int(
                stripped.split(":", 1)[1].strip()
            )

            # Matrix values begin after the columns metadata line.
            data_start = index + 1
            break

    if matrix_type != "matrix":
        raise TypeError(
            f"{path}: variable {variable_name!r} has type "
            f"{matrix_type!r}; expected 'matrix'."
        )

    if (
        number_of_rows is None
        or number_of_columns is None
        or data_start is None
    ):
        raise ValueError(
            f"{path}: incomplete matrix metadata for "
            f"{variable_name!r}."
        )

    data_rows = []

    for line in lines[data_start:]:
        stripped = line.strip()

        if not stripped:
            continue

        # Stop when the next variable begins.
        if stripped.startswith("# name:"):
            break

        # Ignore any other comment lines.
        if stripped.startswith("#"):
            continue

        values = np.fromstring(stripped, sep=" ", dtype=np.float64)

        if values.size != number_of_columns:
            raise ValueError(
                f"{path}: variable {variable_name!r} expected "
                f"{number_of_columns} columns, but a row contains "
                f"{values.size} values."
            )

        data_rows.append(values)

        if len(data_rows) == number_of_rows:
            break

    if len(data_rows) != number_of_rows:
        raise ValueError(
            f"{path}: variable {variable_name!r} declares "
            f"{number_of_rows} rows, but only "
            f"{len(data_rows)} were read."
        )

    return np.vstack(data_rows)



def load_matlab_variable(path: Path, variable_name: str) -> np.ndarray:
    """
    Load a variable from either:
      * MATLAB v4/v5/v6/v7 through scipy.io.loadmat
      * MATLAB v7.3/HDF5 through h5py

    appendmat=False is important because input files may have no .mat suffix.
    """
    scipy_error = None
    hdf5_error = None

    # MATLAB v4 through v7.2 binary formats.
    try:
        contents = loadmat(
            str(path),
            variable_names=[variable_name],
            appendmat=False,
            squeeze_me=True,
        )

        if variable_name not in contents:
            available = sorted(
                key for key in contents if not key.startswith("__")
            )
            raise KeyError(
                f"{path}: variable {variable_name!r} was not found. "
                f"Available variables: {available}"
            )

        array = np.asarray(contents[variable_name])

    except (NotImplementedError, ValueError, OSError) as exc:
        scipy_error = exc

        # MATLAB v7.3/HDF5 format.
        try:
            import h5py

            with h5py.File(path, "r") as matlab_file:
                if variable_name not in matlab_file:
                    raise KeyError(
                        f"{path}: variable {variable_name!r} was not found. "
                        f"Available variables: {sorted(matlab_file.keys())}"
                    )

                array = np.asarray(matlab_file[variable_name])

                # MATLAB v7.3 reverses dimension ordering in HDF5.
                if array.ndim == 2:
                    array = array.T

        except (ImportError, OSError) as exc:
            hdf5_error = exc

            # Octave/MATLAB ASCII matrix format.
            try:
                array = load_ascii_matrix(path, variable_name)
            except Exception as ascii_error:
                raise RuntimeError(
                    f"Could not read {path} as binary MAT, HDF5 MAT, "
                    f"or an ASCII matrix.\n"
                    f"scipy.io.loadmat error: {scipy_error}\n"
                    f"h5py error: {hdf5_error}\n"
                    f"ASCII error: {ascii_error}"
                ) from ascii_error

    array = np.squeeze(np.asarray(array))

    if array.ndim != 2:
        raise ValueError(
            f"{path}: {variable_name!r} must be two-dimensional; "
            f"found shape {array.shape}."
        )

    if not np.issubdtype(array.dtype, np.number):
        raise TypeError(
            f"{path}: {variable_name!r} is not numeric."
        )

    if np.iscomplexobj(array):
        raise TypeError(
            f"{path}: {variable_name!r} contains complex values."
        )

    if not np.all(np.isfinite(array)):
        raise ValueError(
            f"{path}: {variable_name!r} contains NaN or infinite values."
        )

    return np.asarray(array, dtype=np.float64)


def make_mesh(
    number_of_rows: int,
    number_of_columns: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Construct a unit-spaced structured QUAD4 mesh.

    Each microstructure matrix entry is one element:
      rows    -> y direction
      columns -> x direction
    """
    node_columns = number_of_columns + 1
    node_rows = number_of_rows + 1

    x_grid, y_grid = np.meshgrid(
        np.arange(node_columns, dtype=np.float64),
        np.arange(node_rows, dtype=np.float64),
        indexing="xy",
    )

    coord_x = x_grid.ravel(order="C")
    coord_y = y_grid.ravel(order="C")

    connectivity = np.empty(
        (number_of_rows * number_of_columns, 4),
        dtype=np.int32,
    )

    element = 0
    for row in range(number_of_rows):
        for column in range(number_of_columns):
            # Exodus node numbers are one-based.
            lower_left = row * node_columns + column + 1
            lower_right = lower_left + 1
            upper_left = lower_left + node_columns
            upper_right = upper_left + 1

            # Counterclockwise QUAD4 connectivity.
            connectivity[element, :] = (
                lower_left,
                lower_right,
                upper_right,
                upper_left,
            )
            element += 1

    return coord_x, coord_y, connectivity


def write_text_row(variable, row: int, text: str) -> None:
    """Write a null-padded string into an Exodus character array."""
    encoded = text.encode("utf-8")

    if len(encoded) >= LEN_STRING:
        raise ValueError(
            f"Exodus name {text!r} is too long. Maximum length is "
            f"{LEN_STRING - 1} bytes."
        )

    characters = np.full(LEN_STRING, b"\x00", dtype="S1")
    characters[: len(encoded)] = np.frombuffer(encoded, dtype="S1")
    variable[row, :] = characters


def create_exodus_file(
    output_path: Path,
    frames: list[tuple[int, Path]],
    timestep_size: float,
    matlab_variable: str,
    exodus_variable: str,
) -> None:
    first_step, first_path = frames[0]
    first_data = load_matlab_variable(first_path, matlab_variable)

    number_of_rows, number_of_columns = first_data.shape
    number_of_elements = number_of_rows * number_of_columns
    number_of_nodes = (number_of_rows + 1) * (number_of_columns + 1)

    coord_x, coord_y, connectivity = make_mesh(
        number_of_rows,
        number_of_columns,
    )

    # NETCDF3_64BIT_OFFSET is broadly compatible with Exodus readers.
    with Dataset(
        output_path,
        mode="w",
        format="NETCDF3_64BIT_OFFSET",
    ) as exodus:
        # Global Exodus attributes.
        exodus.title = "MATLAB microstructure time series"
        exodus.api_version = np.float32(7.22)
        exodus.version = np.float32(7.22)
        exodus.floating_point_word_size = np.int32(8)
        exodus.file_size = np.int32(1)
        exodus.maximum_name_length = np.int32(LEN_STRING - 1)

        # Dimensions.
        exodus.createDimension("len_string", LEN_STRING)
        exodus.createDimension("len_line", LEN_LINE)
        exodus.createDimension("four", 4)
        exodus.createDimension("time_step", None)
        exodus.createDimension("num_dim", 2)
        exodus.createDimension("num_nodes", number_of_nodes)
        exodus.createDimension("num_elem", number_of_elements)
        exodus.createDimension("num_el_blk", 1)
        exodus.createDimension("num_el_in_blk1", number_of_elements)
        exodus.createDimension("num_nod_per_el1", 4)
        exodus.createDimension("num_elem_var", 1)

        # Time.
        time_whole = exodus.createVariable(
            "time_whole",
            "f8",
            ("time_step",),
        )

        # Coordinates.
        coordx = exodus.createVariable("coordx", "f8", ("num_nodes",))
        coordy = exodus.createVariable("coordy", "f8", ("num_nodes",))
        coordx[:] = coord_x
        coordy[:] = coord_y

        coordinate_names = exodus.createVariable(
            "coor_names",
            "S1",
            ("num_dim", "len_string"),
        )
        write_text_row(coordinate_names, 0, "x")
        write_text_row(coordinate_names, 1, "y")

        # Explicit node and element number maps.
        node_map = exodus.createVariable(
            "node_num_map",
            "i4",
            ("num_nodes",),
        )
        element_map = exodus.createVariable(
            "elem_num_map",
            "i4",
            ("num_elem",),
        )
        node_map[:] = np.arange(1, number_of_nodes + 1, dtype=np.int32)
        element_map[:] = np.arange(
            1,
            number_of_elements + 1,
            dtype=np.int32,
        )

        # One element block containing all QUAD4 elements.
        block_status = exodus.createVariable(
            "eb_status",
            "i4",
            ("num_el_blk",),
        )
        block_status[:] = np.array([1], dtype=np.int32)

        block_ids = exodus.createVariable(
            "eb_prop1",
            "i4",
            ("num_el_blk",),
        )
        block_ids.setncattr("name", "ID")
        block_ids[:] = np.array([1], dtype=np.int32)

        block_names = exodus.createVariable(
            "eb_names",
            "S1",
            ("num_el_blk", "len_string"),
        )
        write_text_row(block_names, 0, "microstructure_block")

        connect = exodus.createVariable(
            "connect1",
            "i4",
            ("num_el_in_blk1", "num_nod_per_el1"),
        )
        connect.setncattr("elem_type", "QUAD4")
        connect[:, :] = connectivity

        # Declare one elemental result variable.
        elemental_names = exodus.createVariable(
            "name_elem_var",
            "S1",
            ("num_elem_var", "len_string"),
        )
        write_text_row(elemental_names, 0, exodus_variable)

        elemental_truth_table = exodus.createVariable(
            "elem_var_tab",
            "i4",
            ("num_el_blk", "num_elem_var"),
        )
        elemental_truth_table[:, :] = np.array([[1]], dtype=np.int32)

        unique_grains = exodus.createVariable(
            "vals_elem_var1eb1",
            "f8",
            ("time_step", "num_el_in_blk1"),
        )

        # Write all frames in ascending nn{step} order.
        for frame_index, (step, input_path) in enumerate(frames):
            if frame_index == 0:
                data = first_data
            else:
                data = load_matlab_variable(
                    input_path,
                    matlab_variable,
                )

            if data.shape != first_data.shape:
                raise ValueError(
                    f"{input_path}: shape {data.shape} does not match "
                    f"the first frame's shape {first_data.shape}."
                )

            time_whole[frame_index] = timestep_size * step
            unique_grains[frame_index, :] = data.ravel(order="C")

            print(
                f"Wrote frame {frame_index + 1}/{len(frames)}: "
                f"step={step}, time={timestep_size * step:g}, "
                f"file={input_path.name}"
            )


def main() -> int:
    arguments = parse_arguments()

    if arguments.dt <= 0:
        print("Error: --dt must be greater than zero.", file=sys.stderr)
        return 2

    output_path = arguments.output_file.resolve()

    if output_path.exists() and not arguments.overwrite:
        print(
            f"Error: output file already exists: {output_path}\n"
            "Use --overwrite to replace it.",
            file=sys.stderr,
        )
        return 2

    try:
        frames = discover_input_files(arguments.input_directory)

        # Avoid accidentally treating the output as a future input.
        frames = [
            (step, path)
            for step, path in frames
            if path.resolve() != output_path
        ]

        if not frames:
            raise FileNotFoundError(
                "No input files remain after excluding the output file."
            )

        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_path.exists():
            output_path.unlink()

        create_exodus_file(
            output_path=output_path,
            frames=frames,
            timestep_size=arguments.dt,
            matlab_variable=arguments.variable,
            exodus_variable=arguments.exodus_variable,
        )

    except Exception as error:
        # Remove an incomplete result.
        if output_path.exists():
            try:
                output_path.unlink()
            except OSError:
                pass

        print(f"Error: {error}", file=sys.stderr)
        return 1

    print(f"Successfully created: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
