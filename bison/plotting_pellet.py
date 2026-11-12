#!/usr/bin/env python3
"""Plot scalar Exodus block-2 results as quadrilateral PNGs."""

import argparse
import logging
import re
import sys
from pathlib import Path

import matplotlib

# Select a noninteractive backend unless the user wants to view figures.
if "--view" not in sys.argv:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import Normalize
import numpy as np

from MultiExodusReader_V2 import MultiExodusReader_V2


CONNECT_NUM = 2
LOG = logging.getLogger(__name__)


def positive_int(text):
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return value


def positive_float(text):
    value = float(text)
    if not np.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return value


def parse_args():
    p = argparse.ArgumentParser(
        description="Plot Exodus results for a specified scalar variable.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    log = p.add_argument_group("General")
    log.add_argument(
        "-v", "--verbose", action="count", default=0,
        help="Increase verbosity (-v, -vv, -vvv).",
    )
    log.add_argument(
        "-s", "--subdirs", action="store_true",
        help="Also search exactly one directory level below the current directory.",
    )
    log.add_argument(
        "--input", "-i", default=None, metavar="PATTERN",
        help="Only process .e files whose filenames contain this string.",
    )
    log.add_argument(
        "--output-dir", type=Path, default=Path("pics"),
        help="Directory for PNG files; ignored with --view.",
    )

    tim = p.add_argument_group("Target frame selection (choose one)")
    grp = tim.add_mutually_exclusive_group(required=True)
    grp.add_argument(
        "-t", "--time", type=float,
        help="Choose the timestep whose time_whole is closest to this value.",
    )
    grp.add_argument(
        "--full", action="store_true",
        help="Use the final Exodus timestep as the target/end frame.",
    )

    out = p.add_argument_group("Timestepping")
    out.add_argument(
        "--frame-mode", choices=("single", "all", "sequence"), default="single",
        help="Plot only the target, every frame through the target, or an "
             "evenly spaced sequence through the target.",
    )
    out.add_argument(
        "--nframes", "-f", type=positive_int, default=40,
        help="Requested number of sequence frames, including both endpoints "
             "when at least two timesteps are available.",
    )

    plot = p.add_argument_group("Plotting")
    plot.add_argument(
        "--var", "-p", required=True,
        help="Scalar variable to plot.",
    )
    plot.add_argument(
        "--view", action="store_true",
        help="Display figures instead of saving PNG files.",
    )
    plot.add_argument(
        "--dpi", type=positive_int, default=300,
        help="Saved-image resolution.",
    )
    plot.add_argument(
        "--aspect", type=positive_float, default=2.0,
        help="Plot-box height divided by width.",
    )
    plot.add_argument(
        "--minimal", action="store_true",
        help="Remove the title, axes, and colorbar.",
    )
    plot.add_argument(
        "--no-axes", action="store_true",
        help="Remove X and Y axes.",
    )
    plot.add_argument(
        "--no-colorbar", action="store_true",
        help="Remove the colorbar.",
    )
    plot.add_argument(
        "--no-title", action="store_true",
        help="Remove the title containing the time value.",
    )

    args = p.parse_args()

    if args.time is not None and not np.isfinite(args.time):
        p.error("--time must be finite")
    if not args.var.strip():
        p.error("--var cannot be empty")
    if args.frame_mode == "sequence" and args.nframes < 2:
        p.error("--nframes must be at least 2 in sequence mode")

    if args.minimal:
        args.no_axes = True
        args.no_colorbar = True
        args.no_title = True

    return args


def find_files(pattern, subdirs):
    root = Path.cwd()
    files = list(root.glob("*.e"))
    if subdirs:
        files.extend(root.glob("*/*.e"))

    return sorted(
        (path for path in files
         if path.is_file() and (pattern is None or pattern in path.name)),
        key=lambda path: str(path.relative_to(root)),
    )


def unique_base(path, used):
    """Disambiguate equal filenames found in different directories."""
    root = Path.cwd()
    base = path.stem

    if base in used:
        parent = "_".join(path.relative_to(root).parts[:-1]) or "current"
        base = f"{base}_{parent}"

    candidate = base
    suffix = 2
    while candidate in used:
        candidate = f"{base}_{suffix}"
        suffix += 1

    used.add(candidate)
    return candidate


def filename_part(text):
    """Make a variable name safe to include in a filename."""
    result = re.sub(r"[^A-Za-z0-9_.-]+", "_", text.strip())
    return result.strip("._") or "variable"


def select_indices(times, args):
    """Select zero-based positions in the recorded Exodus time array."""
    if args.full:
        target = len(times) - 1
    else:
        target = int(np.argmin(np.abs(times - args.time)))

    if args.frame_mode == "single":
        return [target]
    if args.frame_mode == "all":
        return list(range(target + 1))

    # There cannot be more distinct frames than recorded timesteps.
    count = min(args.nframes, target + 1)
    indices = np.linspace(0, target, count)
    return sorted({int(round(index)) for index in indices})


def read_frame(reader, variable, time, nodal_names):
    """Read scalar element values and, if available, deformed quad coordinates."""
    x, y, _z, values = reader.get_data_at_time(variable, time)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    values = np.asarray(values, dtype=float).reshape(-1)

    if (x.ndim != 2 or x.shape[1] != 4
            or y.shape != x.shape or values.size != x.shape[0]):
        raise ValueError(
            f"At t={time:g}, expected block-2 quads and one scalar per "
            f"quad; got x={x.shape}, y={y.shape}, values={values.shape}"
        )

    # Do not modify arrays potentially owned by the reader.
    x = x.copy()
    y = y.copy()

    for displacement_name, coordinates in (("disp_x", x), ("disp_y", y)):
        if displacement_name not in nodal_names:
            continue

        _dx, _dy, _dz, displacement = reader.get_data_at_time(
            displacement_name, time, full_nodal=True
        )
        displacement = np.asarray(displacement, dtype=float)

        if displacement.shape != coordinates.shape:
            raise ValueError(
                f"{displacement_name} shape {displacement.shape} does not "
                f"match quad coordinates {coordinates.shape} at t={time:g}"
            )
        coordinates += displacement

    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError(f"Non-finite coordinates at t={time:g}")

    return x, y, values


def plot_frame(x, y, values, time, args, limits, norm, destination):
    vertices = np.stack((x, y), axis=-1)
    fig, ax = plt.subplots(figsize=(6, 9))

    try:
        collection = PolyCollection(
            vertices,
            array=np.ma.masked_invalid(values),
            cmap="viridis",
            norm=norm,
            edgecolors="none",
        )
        ax.add_collection(collection)
        ax.set_xlim(limits[0], limits[1])
        ax.set_ylim(limits[2], limits[3])
        ax.set_box_aspect(args.aspect)

        if args.no_axes:
            ax.set_axis_off()
        else:
            ax.set_xlabel("x")
            ax.set_ylabel("y")

        if not args.no_title:
            ax.set_title(f"{args.var}, t={time / 86400:.2f} days")

        if not args.no_colorbar:
            fig.colorbar(collection, ax=ax, label=args.var)

        if args.view:
            plt.show()
        else:
            fig.savefig(destination, dpi=args.dpi, bbox_inches="tight")
    finally:
        plt.close(fig)


def process_file(path, base, args):
    # A literal filename prevents this reader from collecting other simulations.
    reader = MultiExodusReader_V2(str(path), connect_num=CONNECT_NUM)

    try:
        if len(reader.exodus_readers) != 1:
            raise ValueError(
                f"Expected one Exodus file; found {len(reader.exodus_readers)}"
            )

        times = np.asarray(reader.global_times, dtype=float)
        if times.ndim != 1 or times.size == 0 or not np.all(np.isfinite(times)):
            raise ValueError("No usable recorded Exodus times")

        indices = select_indices(times, args)
        nodal_names = set(reader.exodus_readers[0].nodal_var_names)

        # Scan selected frames for fixed spatial and color limits. This makes
        # images from the same file directly comparable across time.
        xmin = ymin = vmin = np.inf
        xmax = ymax = vmax = -np.inf

        for index in indices:
            x, y, values = read_frame(
                reader, args.var, float(times[index]), nodal_names
            )
            xmin = min(xmin, float(x.min()))
            xmax = max(xmax, float(x.max()))
            ymin = min(ymin, float(y.min()))
            ymax = max(ymax, float(y.max()))

            finite = values[np.isfinite(values)]
            if finite.size:
                vmin = min(vmin, float(finite.min()))
                vmax = max(vmax, float(finite.max()))

        if not np.isfinite(vmin):
            raise ValueError(f"No finite {args.var} values in selected frames")

        xpad = 0.02 * (xmax - xmin or 1.0)
        ypad = 0.02 * (ymax - ymin or 1.0)
        limits = (
            xmin - xpad, xmax + xpad,
            ymin - ypad, ymax + ypad,
        )

        # Matplotlib expands equal limits automatically for constant data.
        norm = Normalize(vmin=vmin, vmax=vmax)
        LOG.info(
            "%s: %d frame(s), time range %g–%g, value range %g–%g",
            path, len(indices), times[indices[0]], times[indices[-1]],
            vmin, vmax,
        )

        for index in indices:
            time = float(times[index])
            x, y, values = read_frame(reader, args.var, time, nodal_names)
            destination = (
                args.output_dir
                / f"{base}_{filename_part(args.var)}_{index:04d}.png"
            )
            plot_frame(x, y, values, time, args, limits, norm, destination)
            LOG.info("%s: t=%g -> %s", path, time,
                     "display" if args.view else destination)

    finally:
        for exodus_reader in reader.exodus_readers:
            exodus_reader.mesh.close()


def main():
    args = parse_args()
    level = logging.WARNING if args.verbose == 0 else (
        logging.INFO if args.verbose == 1 else logging.DEBUG
    )
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")

    files = find_files(args.input, args.subdirs)
    if not files:
        LOG.warning("No matching .e files found")
        return

    if not args.view:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    used_bases = set()
    for path in files:
        base = unique_base(path, used_bases)
        try:
            process_file(path, base, args)
        except Exception:
            LOG.exception("Could not process %s", path)


if __name__ == "__main__":
    main()
