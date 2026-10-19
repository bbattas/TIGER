#!/usr/bin/env python3

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat


FRAME_RE = re.compile(r"nn(\d+)\.mat$", re.IGNORECASE)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot microstructure and two eta fields from MATLAB files."
    )

    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument(
        "--frame",
        type=int,
        help="Frame number to plot, for example 300.",
    )
    selection.add_argument(
        "--all",
        action="store_true",
        help="Plot every matching nn{frame}.mat file.",
    )

    parser.add_argument(
        "--prefix",
        default="",
        help=(
            "Filename text before 'nn'. For example, use "
            "'incl_dependence_bicr_high_Lg' for "
            "'incl_dependence_bicr_high_Lgnn300.mat'. "
            "If omitted, any prefix is accepted."
        ),
    )
    parser.add_argument(
        "--directory",
        type=Path,
        default=Path("."),
        help="Directory containing the MAT files (default: current directory).",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("."),
        help="Directory for generated PNG files (default: current directory).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Output resolution in DPI (default: 200).",
    )

    return parser.parse_args()


def frame_from_filename(path):
    match = FRAME_RE.search(path.name)
    if not match:
        raise ValueError(
            f"Filename does not end with nn{{frame}}.mat: {path.name}"
        )
    return int(match.group(1))


def find_files(args):
    directory = args.directory.expanduser().resolve()

    if not directory.is_dir():
        raise FileNotFoundError(f"Directory does not exist: {directory}")

    if args.frame is not None:
        pattern = f"{args.prefix}nn{args.frame}.mat" if args.prefix else f"*nn{args.frame}.mat"
    else:
        pattern = f"{args.prefix}nn*.mat" if args.prefix else "*nn*.mat"

    matches = []
    for path in directory.glob(pattern):
        if not path.is_file():
            continue

        match = FRAME_RE.search(path.name)
        if match:
            matches.append(path)

    matches.sort(key=frame_from_filename)

    if not matches:
        raise FileNotFoundError(
            f"No files matching '{pattern}' were found in {directory}"
        )

    if args.frame is not None and len(matches) > 1:
        names = "\n  ".join(path.name for path in matches)
        raise RuntimeError(
            f"Multiple files were found for frame {args.frame}:\n  {names}\n"
            "Specify --prefix to select one file series."
        )

    return matches


def unwrap_matlab_cell(value):
    """Remove singleton object-array wrappers produced by scipy.loadmat."""
    while isinstance(value, np.ndarray) and value.dtype == object and value.size == 1:
        value = value.flat[0]
    return np.asarray(value).squeeze()


def require_2d(array, name):
    array = np.real_if_close(np.asarray(array).squeeze())

    if array.ndim != 2:
        raise ValueError(
            f"Expected '{name}' to be two-dimensional, but its shape is "
            f"{array.shape}."
        )

    if np.iscomplexobj(array):
        raise ValueError(f"'{name}' contains complex values and cannot be plotted directly.")

    return array


def extract_eta_maps(mat_data, microstructure):
    eta_key = next((key for key in ("eta", "etas") if key in mat_data), None)

    if eta_key is None:
        raise KeyError("The MAT file contains neither an 'eta' nor an 'etas' variable.")

    eta = np.asarray(mat_data[eta_key])

    # MATLAB cell array, such as a 1x2 eta cell.
    if eta.dtype == object:
        cells = list(eta.ravel())

        if len(cells) < 2:
            raise ValueError(
                f"'{eta_key}' contains only {len(cells)} cell(s); two are required."
            )

        eta0 = require_2d(unwrap_matlab_cell(cells[0]), f"{eta_key}[0]")
        eta1 = require_2d(unwrap_matlab_cell(cells[1]), f"{eta_key}[1]")
        return eta_key, eta0, eta1

    # Also support a numeric 3D array whose first or last dimension indexes eta.
    eta = np.squeeze(eta)

    if eta.ndim != 3:
        raise ValueError(
            f"Expected '{eta_key}' to be a cell array or 3D numeric array, "
            f"but its shape is {eta.shape}."
        )

    if eta.shape[0] >= 2 and eta.shape[1:] == microstructure.shape:
        eta0, eta1 = eta[0], eta[1]
    elif eta.shape[-1] >= 2 and eta.shape[:2] == microstructure.shape:
        eta0, eta1 = eta[..., 0], eta[..., 1]
    else:
        raise ValueError(
            f"Could not identify the component dimension of '{eta_key}' "
            f"with shape {eta.shape}."
        )

    return (
        eta_key,
        require_2d(eta0, f"{eta_key}[0]"),
        require_2d(eta1, f"{eta_key}[1]"),
    )


def load_fields(path):
    try:
        mat_data = loadmat(path, squeeze_me=False, struct_as_record=False)
    except NotImplementedError as error:
        raise RuntimeError(
            f"{path.name} appears to use MATLAB v7.3/HDF5 format. "
            "This script uses scipy.io.loadmat, which supports MAT formats "
            "through MATLAB v7.2."
        ) from error

    if "microstructure" not in mat_data:
        visible_keys = sorted(key for key in mat_data if not key.startswith("__"))
        raise KeyError(
            f"{path.name} has no 'microstructure' variable. "
            f"Variables found: {visible_keys}"
        )

    microstructure = require_2d(
        mat_data["microstructure"],
        "microstructure",
    )
    eta_key, eta0, eta1 = extract_eta_maps(mat_data, microstructure)

    for name, field in ((f"{eta_key}[0]", eta0), (f"{eta_key}[1]", eta1)):
        if field.shape != microstructure.shape:
            raise ValueError(
                f"{name} has shape {field.shape}, but microstructure has "
                f"shape {microstructure.shape}."
            )

    return microstructure, eta_key, eta0, eta1


def plot_frame(path, output_directory, dpi):
    frame = frame_from_filename(path)
    microstructure, eta_key, eta0, eta1 = load_fields(path)

    fields = (
        ("Microstructure", microstructure),
        (f"{eta_key}[0]", eta0),
        (f"{eta_key}[1]", eta1),
    )

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15, 4.8),
        constrained_layout=True,
    )

    for axis, (label, field) in zip(axes, fields):
        ny, nx = field.shape

        image = axis.imshow(
            field,
            origin="lower",
            extent=(0, nx - 1, 0, ny - 1),
            interpolation="nearest",
            aspect="equal",
        )

        axis.set_title(label)
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        fig.colorbar(image, ax=axis, shrink=0.85)

    fig.suptitle(f"Frame {frame}", fontsize=14)

    output_directory.mkdir(parents=True, exist_ok=True)
    output_path = output_directory / f"frame_{frame}.png"

    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"Created {output_path}")


def main():
    args = parse_args()
    files = find_files(args)

    for path in files:
        try:
            plot_frame(
                path=path,
                output_directory=args.output_directory.expanduser().resolve(),
                dpi=args.dpi,
            )
        except Exception as error:
            if args.all:
                print(f"ERROR processing {path.name}: {error}")
            else:
                raise


if __name__ == "__main__":
    main()
