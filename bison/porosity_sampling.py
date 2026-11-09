#!/usr/bin/env python3
"""Plot porosity and sample a horizontal line in Exodus block 2."""

import argparse
import csv
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Save figures without requiring a display.
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import Normalize
import numpy as np
from tqdm import tqdm

from MultiExodusReader_V2 import MultiExodusReader_V2


# Replace these examples with your desired times, in Exodus time units.
# TARGET_TIMES = np.array([0.0, 10368000], dtype=float)
# TARGET_TIMES = np.linspace(0.0, 10368000, 11)

VARIABLE = "porosity"
CONNECT_NUM = 2
LOG = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot porosity and export horizontal-line samples from .e files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    inputs = parser.add_argument_group("Input selection")
    inputs.add_argument(
        "--contains",
        default="",
        help="Only process .e filenames containing this text (default: all).",
    )
    inputs.add_argument(
        "-s", "--subdirs",
        action="store_true",
        help="Also search directories exactly one level below the current directory.",
    )
    inputs.add_argument(
        "--tmax",
        type=float,
        default=10368000,
        help="Max time (3%% is 10368000 and 12%% is 58950720).",
    )
    inputs.add_argument(
            "--tsteps",
            type=float,
            default=11,
            help="Number of linspace steps from 0-tmax including 0.",
        )

    measurements = parser.add_argument_group("Measurements")
    measurements.add_argument(
        "--line-y",
        type=float,
        default=0.0225,
        help="Y coordinate of the horizontal sampling line.",
    )
    measurements.add_argument(
        "--samples",
        type=int,
        default=20,
        help="Number of line samples, including both endpoints.",
    )

    outputs = parser.add_argument_group("Plots and output")
    outputs.add_argument(
        "--color-max",
        type=float,
        default=1.0,
        help="Upper end of the fixed porosity color scale, whose lower end is 0 ",
    )
    outputs.add_argument(
        "--output-dir",
        type=Path,
        default=Path("porosity_output"),
        help="Directory for CSVs and PNGs.",
    )
    outputs.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show detailed logging instead of progress bars.",
    )

    args = parser.parse_args()
    if args.samples < 2:
        parser.error("--samples must be at least 2")
    if not np.isfinite(args.line_y):
        parser.error("--line-y must be finite")
    if not np.isfinite(args.color_max) or args.color_max <= 0:
        parser.error("--color-max must be finite and positive")
    # Create times here:
    args.TARGET_TIMES = np.linspace(0,args.tmax,args.tsteps)
    if args.TARGET_TIMES.ndim != 1 or not np.all(np.isfinite(args.TARGET_TIMES)):
        parser.error("TARGET_TIMES must be a one-dimensional array of finite numbers")
    return args


def find_files(contains, subdirs):
    root = Path.cwd()
    files = list(root.glob("*.e"))
    if subdirs:
        files.extend(root.glob("*/*.e"))
    return sorted(
        (path for path in files if path.is_file()
         and contains in path.name),
        key=lambda path: str(path.relative_to(root)),
    )


def unique_base(path, used):
    """Keep the Exodus filename visible; disambiguate duplicate names by path."""
    root = Path.cwd()
    stem = path.stem
    base = stem
    if base in used:
        parent = "_".join(path.relative_to(root).parts[:-1]) or "current"
        base = f"{stem}_{parent}"
    candidate = base
    suffix = 2
    while candidate in used:
        candidate = f"{base}_{suffix}"
        suffix += 1
    used.add(candidate)
    return candidate


def select_frames(recorded_times, targets):
    """Return (target_time, actual_time) pairs in target order."""
    times = np.asarray(recorded_times, dtype=float)
    if times.size == 0 or not np.all(np.isfinite(times)):
        raise ValueError("No usable recorded times")

    selected = []
    used_indices = set()
    for target in targets:
        if target < times.min() or target > times.max():
            LOG.info("Skipping target %g: outside recorded time range", target)
            continue

        index = int(np.argmin(np.abs(times - target)))
        if index in used_indices:
            LOG.info(
                "Skipping target %g: closest frame at %g already selected",
                target, times[index],
            )
            continue

        used_indices.add(index)
        selected.append((float(target), float(times[index])))
    return selected

def read_deformed_frame(reader, time, nodal_names):
    """Return porosity and quad coordinates at one recorded time."""
    x, y, _z, values = reader.get_data_at_time(VARIABLE, time)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    values = np.asarray(values, dtype=float).reshape(-1)

    if (x.ndim != 2 or x.shape[1] != 4
            or y.shape != x.shape or values.size != x.shape[0]):
        raise ValueError(
            f"Unexpected data shapes at t={time:g}: "
            f"x={x.shape}, y={y.shape}, values={values.shape}"
        )

    # Make copies: avoid changing coordinates held by the reader.
    x = x.copy()
    y = y.copy()

    for variable, coordinate in (("disp_x", x), ("disp_y", y)):
        if variable not in nodal_names:
            continue

        _nx, _ny, _nz, displacement = reader.get_data_at_time(
            variable, time, full_nodal=True
        )
        displacement = np.asarray(displacement, dtype=float)
        if displacement.shape != coordinate.shape:
            raise ValueError(
                f"{variable} shape {displacement.shape} does not match "
                f"coordinate shape {coordinate.shape} at t={time:g}"
            )
        coordinate += displacement

    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError(f"Non-finite deformed coordinates at t={time:g}")

    return x, y, values


def quad_line_intervals(x, y, values, line_y):
    """
    Find positive-length intersections of a horizontal line with convex quads.

    Returns (left_x, right_x, element_value) tuples. Exodus quad nodes are
    assumed to be ordered around each element's perimeter.
    """
    intervals = []

    for xs, ys, value in zip(x, y, values):
        crossings = []
        for i in range(4):
            j = (i + 1) % 4
            x0, y0 = float(xs[i]), float(ys[i])
            x1, y1 = float(xs[j]), float(ys[j])

            if y0 == y1:
                if y0 == line_y:
                    crossings.extend((x0, x1))
            elif min(y0, y1) <= line_y <= max(y0, y1):
                fraction = (line_y - y0) / (y1 - y0)
                crossings.append(x0 + fraction * (x1 - x0))

        if crossings:
            left, right = min(crossings), max(crossings)
            if right > left:  # Exclude a line that only touches one vertex.
                intervals.append((left, right, float(value)))

    return intervals


def sample_line(x, y, values, line_y, count):
    intervals = quad_line_intervals(x, y, values, line_y)
    if not intervals:
        raise ValueError(f"Block does not cross y={line_y:g}")

    left = min(item[0] for item in intervals)
    right = max(item[1] for item in intervals)
    sample_x = np.linspace(left, right, count)
    sample_values = np.full(count, np.nan)

    # Permit tiny roundoff at shared quad edges, without extending the domain.
    tolerance = 1e-12 * max(1.0, right - left)
    for i, position in enumerate(sample_x):
        for start, end, value in intervals:
            if start - tolerance <= position <= end + tolerance:
                sample_values[i] = value
                break

    return sample_x, sample_values

def fixed_plot_limits(reader, frames, nodal_names, buffer_fraction=0.02):
    """Find buffered bounds from the first and last selected frames."""
    endpoint_times = (min(actual for _target, actual in frames),
                      max(actual for _target, actual in frames))

    bounds = []
    for time in dict.fromkeys(endpoint_times):  # Read once if both are equal.
        x, y, _values = read_deformed_frame(reader, time, nodal_names)
        bounds.append((x.min(), x.max(), y.min(), y.max()))

    xmin = min(bound[0] for bound in bounds)
    xmax = max(bound[1] for bound in bounds)
    ymin = min(bound[2] for bound in bounds)
    ymax = max(bound[3] for bound in bounds)

    xpad = buffer_fraction * (xmax - xmin or 1.0)
    ypad = buffer_fraction * (ymax - ymin or 1.0)
    return (xmin - xpad, xmax + xpad, ymin - ypad, ymax + ypad)


def save_plot(x, y, values, time, color_max, destination, limits):
    vertices = np.stack((x, y), axis=-1)
    fig, ax = plt.subplots(figsize=(6, 9))

    collection = PolyCollection(
        vertices,
        array=np.ma.masked_invalid(values),
        cmap="viridis",
        norm=Normalize(vmin=0, vmax=color_max),
        edgecolors="none",
    )
    ax.add_collection(collection)
    ax.set_xlim(limits[0], limits[1])
    ax.set_ylim(limits[2], limits[3])
    ax.set_box_aspect(2)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"{VARIABLE}, t={time/86400:.2f} days")
    fig.colorbar(collection, ax=ax, label=VARIABLE)
    fig.savefig(destination, dpi=200, bbox_inches="tight")
    plt.close(fig)


def process_file(path, base, args, show_frame_progress):
    # Passing one literal filename keeps this reader instance confined to
    # one simulation, even though its constructor accepts a glob pattern.
    reader = MultiExodusReader_V2(str(path), connect_num=CONNECT_NUM)

    try:
        if len(reader.exodus_readers) != 1:
            raise ValueError(f"Expected one Exodus file; found {len(reader.exodus_readers)}")

        frames = select_frames(reader.global_times, args.TARGET_TIMES)
        if not frames:
            LOG.warning("%s: no frames selected", path)
            return

        nodal_names = set(reader.exodus_readers[0].nodal_var_names)
        limits = fixed_plot_limits(reader, frames, nodal_names)
        LOG.info("%s: fixed plot limits = %s", path, limits)

        csv_path = args.output_dir / f"{base}_line.csv"
        with csv_path.open("w", newline="") as output:
            writer = csv.writer(output)
            writer.writerow(["x", VARIABLE, "time"])

            iterator = enumerate(frames)
            if show_frame_progress:
                iterator = tqdm(iterator, total=len(frames), desc=path.name)

            for frame_id, (target, actual) in iterator:
                x, y, values = read_deformed_frame(reader, actual, nodal_names)

                png_path = args.output_dir / f"{base}_{frame_id:04d}.png"
                save_plot(x, y, values, actual, args.color_max, png_path, limits)

                try:
                    sample_x, sample_values = sample_line(
                        x, y, values, args.line_y, args.samples
                    )
                except ValueError as exc:
                    LOG.warning("%s at t=%g: %s", path, actual, exc)
                    continue

                for position, value in zip(sample_x, sample_values):
                    writer.writerow([
                        float(position),
                        "" if np.isnan(value) else float(value),
                        actual,
                    ])
                LOG.info(
                    "%s: target=%g, actual=%g, plot=%s",
                    path, target, actual, png_path.name,
                )

        LOG.info("%s: CSV=%s", path, csv_path)
    finally:
        for exodus_reader in reader.exodus_readers:
            exodus_reader.mesh.close()


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s: %(message)s",
    )

    files = find_files(args.contains, args.subdirs)
    if not files:
        LOG.warning("No matching .e files found")
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    used_bases = set()

    file_iterator = files
    if not args.verbose and len(files) > 1:
        file_iterator = tqdm(files, desc="Exodus files")

    for path in file_iterator:
        base = unique_base(path, used_bases)
        try:
            process_file(
                path, base, args,
                show_frame_progress=not args.verbose and len(files) == 1,
            )
        except Exception:
            LOG.exception("Could not process %s", path)


if __name__ == "__main__":
    main()
