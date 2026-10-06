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


def load_ascii_cell_matrices(
    path: Path,
    variable_name: str,
) -> list[np.ndarray]:
    """
    Load all matrices from a named Octave ASCII cell array.

    Expected structure:

        # name: eta
        # type: cell
        # rows: 1
        # columns: 2
        # name: <cell-element>
        # type: matrix
        # rows: 161
        # columns: 161
        ...
    """
    with path.open("r", encoding="utf-8") as stream:
        lines = stream.readlines()

    variable_start = None

    for index, line in enumerate(lines):
        stripped = line.strip()

        if stripped.startswith("# name:"):
            found_name = stripped.split(":", 1)[1].strip()

            if found_name == variable_name:
                variable_start = index
                break

    if variable_start is None:
        raise KeyError(
            f"{path}: cell variable {variable_name!r} was not found."
        )

    cell_type = None
    cell_rows = None
    cell_columns = None
    search_start = None

    for index in range(variable_start + 1, len(lines)):
        stripped = lines[index].strip()

        if stripped.startswith("# type:"):
            cell_type = stripped.split(":", 1)[1].strip()

        elif stripped.startswith("# rows:"):
            cell_rows = int(stripped.split(":", 1)[1].strip())

        elif stripped.startswith("# columns:"):
            cell_columns = int(stripped.split(":", 1)[1].strip())
            search_start = index + 1
            break

    if cell_type != "cell":
        raise TypeError(
            f"{path}: {variable_name!r} has type {cell_type!r}; "
            "expected an Octave cell array."
        )

    if (
        cell_rows is None
        or cell_columns is None
        or search_start is None
    ):
        raise ValueError(
            f"{path}: incomplete cell metadata for {variable_name!r}."
        )

    expected_cell_count = cell_rows * cell_columns
    matrices: list[np.ndarray] = []
    index = search_start

    while index < len(lines) and len(matrices) < expected_cell_count:
        stripped = lines[index].strip()

        if stripped != "# name: <cell-element>":
            index += 1
            continue

        matrix_type = None
        matrix_rows = None
        matrix_columns = None
        data_start = None
        index += 1

        while index < len(lines):
            stripped = lines[index].strip()

            if stripped.startswith("# type:"):
                matrix_type = stripped.split(":", 1)[1].strip()

            elif stripped.startswith("# rows:"):
                matrix_rows = int(
                    stripped.split(":", 1)[1].strip()
                )

            elif stripped.startswith("# columns:"):
                matrix_columns = int(
                    stripped.split(":", 1)[1].strip()
                )
                data_start = index + 1
                break

            index += 1

        if matrix_type != "matrix":
            raise TypeError(
                f"{path}: cell element {len(matrices)} in "
                f"{variable_name!r} has type {matrix_type!r}; "
                "expected 'matrix'."
            )

        if (
            matrix_rows is None
            or matrix_columns is None
            or data_start is None
        ):
            raise ValueError(
                f"{path}: incomplete metadata for cell element "
                f"{len(matrices)} in {variable_name!r}."
            )

        data_rows = []
        index = data_start

        while index < len(lines) and len(data_rows) < matrix_rows:
            stripped = lines[index].strip()

            if not stripped:
                index += 1
                continue

            if stripped.startswith("#"):
                raise ValueError(
                    f"{path}: cell element {len(matrices)} ended "
                    f"after {len(data_rows)} rows; expected "
                    f"{matrix_rows}."
                )

            values = np.fromstring(
                stripped,
                sep=" ",
                dtype=np.float64,
            )

            if values.size != matrix_columns:
                raise ValueError(
                    f"{path}: cell element {len(matrices)}, row "
                    f"{len(data_rows)}, expected {matrix_columns} "
                    f"columns but found {values.size}."
                )

            data_rows.append(values)
            index += 1

        if len(data_rows) != matrix_rows:
            raise ValueError(
                f"{path}: cell element {len(matrices)} declares "
                f"{matrix_rows} rows, but only "
                f"{len(data_rows)} were read."
            )

        matrix = np.vstack(data_rows)

        if not np.all(np.isfinite(matrix)):
            raise ValueError(
                f"{path}: cell element {len(matrices)} in "
                f"{variable_name!r} contains NaN or infinite values."
            )

        matrices.append(matrix)

    if len(matrices) != expected_cell_count:
        raise ValueError(
            f"{path}: {variable_name!r} declares "
            f"{expected_cell_count} cell elements, but only "
            f"{len(matrices)} were read."
        )

    return matrices


def element_grid_to_nodal_grid(
    element_values: np.ndarray,
) -> np.ndarray:
    """
    Map an (ny, nx) element-grid field onto the script's
    (ny + 1, nx + 1) node grid.

    Existing values are placed on corresponding lower-left nodes.
    The last row and column are extended to the outer boundary.
    """
    return np.pad(
        element_values,
        pad_width=((0, 1), (0, 1)),
        mode="edge",
    )


def validate_eta_fields(
    fields: list[np.ndarray],
    path: Path,
) -> list[np.ndarray]:
    validated = []

    for index, field in enumerate(fields):
        field = np.squeeze(np.asarray(field))

        if field.ndim != 2:
            raise ValueError(
                f"{path}: eta cell {index} must be two-dimensional; "
                f"found shape {field.shape}."
            )

        if not np.issubdtype(field.dtype, np.number):
            raise TypeError(
                f"{path}: eta cell {index} is not numeric."
            )

        if np.iscomplexobj(field):
            raise TypeError(
                f"{path}: eta cell {index} contains complex values."
            )

        field = np.asarray(field, dtype=np.float64)

        if not np.all(np.isfinite(field)):
            raise ValueError(
                f"{path}: eta cell {index} contains NaN or "
                "infinite values."
            )

        validated.append(field)

    if not validated:
        raise ValueError(f"{path}: eta contains no grain fields.")

    return validated


def load_eta_fields(path: Path) -> list[np.ndarray]:
    """
    Load the matrices stored in the MATLAB/Octave cell variable 'eta'.

    Supports:
      * MATLAB binary MAT files readable by scipy
      * MATLAB v7.3 HDF5 MAT files
      * Octave ASCII files
    """
    with path.open("rb") as stream:
        signature = stream.read(128)

    errors = []

    # Octave text files should be sent directly to the ASCII parser.
    if signature.lstrip().startswith(b"#"):
        try:
            return validate_eta_fields(
                load_ascii_cell_matrices(path, "eta"),
                path,
            )
        except Exception as error:
            raise RuntimeError(
                f"{path}: failed to read eta from the Octave "
                f"ASCII file: {error}"
            ) from error

    # Try ordinary MATLAB binary formats.
    try:
        contents = loadmat(
            str(path),
            variable_names=["eta"],
            appendmat=False,
            squeeze_me=True,
            struct_as_record=False,
        )

        if "eta" not in contents:
            available = sorted(
                name for name in contents
                if not name.startswith("__")
            )
            raise KeyError(
                f"eta was not found; available variables: {available}"
            )

        eta = np.asarray(contents["eta"])

        if eta.dtype != object:
            raise TypeError(
                f"eta is not a MATLAB cell array; "
                f"found dtype {eta.dtype} and shape {eta.shape}"
            )

        # MATLAB cell arrays use column-major ordering.
        fields = [
            np.asarray(cell)
            for cell in eta.ravel(order="F")
        ]

        return validate_eta_fields(fields, path)

    except Exception as error:
        errors.append(f"scipy.io.loadmat: {error}")

    # Try MATLAB v7.3, which uses HDF5.
    try:
        import h5py

        with h5py.File(path, "r") as matlab_file:
            if "eta" not in matlab_file:
                raise KeyError(
                    "eta was not found; available variables: "
                    f"{sorted(matlab_file.keys())}"
                )

            eta_dataset = matlab_file["eta"]
            references = np.asarray(eta_dataset)

            fields = []

            # Reverse HDF5's representation of MATLAB dimensions.
            for reference in references.T.ravel(order="C"):
                if not reference:
                    raise ValueError(
                        "eta contains an empty cell."
                    )

                field = np.asarray(matlab_file[reference])

                if field.ndim == 2:
                    field = field.T

                fields.append(field)

        return validate_eta_fields(fields, path)

    except Exception as error:
        errors.append(f"h5py: {error}")

    raise RuntimeError(
        f"{path}: could not read eta. The file does not appear to "
        "be a supported Octave ASCII, MATLAB binary, or MATLAB "
        "v7.3 file.\n" + "\n".join(errors)
    )


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
    # first_eta = load_ascii_cell_matrices(first_path, "eta")
    first_eta = load_eta_fields(first_path)

    number_of_rows, number_of_columns = first_data.shape
    number_of_grains = len(first_eta)

    if number_of_grains == 0:
        raise ValueError(
            f"{first_path}: eta does not contain any grain fields."
        )

    for grain_index, grain_data in enumerate(first_eta):
        if grain_data.shape != first_data.shape:
            raise ValueError(
                f"{first_path}: eta cell {grain_index} has shape "
                f"{grain_data.shape}, but microstructure has shape "
                f"{first_data.shape}."
            )

    number_of_elements = number_of_rows * number_of_columns
    number_of_nodes = ((number_of_rows + 1) * (number_of_columns + 1))

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
        exodus.createDimension("num_nod_var", number_of_grains)

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

        # Declare the nodal grain order-parameter variables:
        # gr0, gr1, ..., grN.
        nodal_names = exodus.createVariable(
            "name_nod_var",
            "S1",
            ("num_nod_var", "len_string"),
        )

        nodal_variables = []

        for grain_index in range(number_of_grains):
            grain_name = f"gr{grain_index}"
            write_text_row(nodal_names, grain_index, grain_name)

            nodal_variable = exodus.createVariable(
                f"vals_nod_var{grain_index + 1}",
                "f8",
                ("time_step", "num_nodes"),
            )
            nodal_variables.append(nodal_variable)

        # Write all frames in ascending nn{step} order.
        for frame_index, (step, input_path) in enumerate(frames):
            if frame_index == 0:
                data = first_data
                eta_fields = first_eta
            else:
                data = load_matlab_variable(
                    input_path,
                    matlab_variable,
                )
                # eta_fields = load_ascii_cell_matrices(
                #     input_path,
                #     "eta",
                # )
                eta_fields = load_eta_fields(input_path)

            if data.shape != first_data.shape:
                raise ValueError(
                    f"{input_path}: microstructure shape "
                    f"{data.shape} does not match the first frame's "
                    f"shape {first_data.shape}."
                )

            if len(eta_fields) != number_of_grains:
                raise ValueError(
                    f"{input_path}: eta contains {len(eta_fields)} "
                    f"grain fields, but the first frame contains "
                    f"{number_of_grains}."
                )

            time_value = timestep_size * step
            time_whole[frame_index] = time_value

            # Elemental variable.
            unique_grains[frame_index, :] = data.ravel(order="C")

            # Nodal variables.
            for grain_index, eta_field in enumerate(eta_fields):
                if eta_field.shape != first_data.shape:
                    raise ValueError(
                        f"{input_path}: eta cell {grain_index} has "
                        f"shape {eta_field.shape}; expected "
                        f"{first_data.shape}."
                    )

                nodal_grid = element_grid_to_nodal_grid(eta_field)

                nodal_variables[grain_index][frame_index, :] = (
                    nodal_grid.ravel(order="C")
                )

            print(
                f"Wrote frame {frame_index + 1}/{len(frames)}: "
                f"step={step}, time={time_value:g}, "
                f"grains={number_of_grains}, "
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
