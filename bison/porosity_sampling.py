#!/usr/bin/env python3
"""Plot porosity and X_Zr and sample a horizontal line in Exodus block 2."""

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


VARIABLES = ("porosity", "X_Zr")
CONNECT_NUM = 2
LOG = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot variables and export horizontal-line samples from .e files.",
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
        help="Maximum target time, in Exodus time units.",
    )
    inputs.add_argument(
        "--tsteps",
        type=int,
        default=11,
        help="Number of target times from 0 to --tmax, including endpoints.",
    )
    inputs.add_argument(
        "--skip0",
        action="store_true",
        help="Never select the time=0 frame; use the earliest later frame for target 0.",
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
        help="Upper end of the porosity color scale; its lower end is 0.",
    )
    outputs.add_argument(
        "--skip-plots",
        action="store_true",
        help="Write CSVs without creating porosity or X_Zr PNGs.",
    )
    outputs.add_argument(
        "--output-dir",
        type=Path,
        default=Path("porosity_output"),
        help="Directory for CSVs and, unless skipped, PNGs.",
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
    if not np.isfinite(args.tmax):
        parser.error("--tmax must be finite")
    if args.tsteps < 1:
        parser.error("--tsteps must be at least 1")

    args.target_times = np.linspace(0, args.tmax, args.tsteps)
    return args


def find_files(contains, subdirs):
    root = Path.cwd()
    files = list(root.glob("*.e"))
    if subdirs:
        files.extend(root.glob("*/*.e"))
    return sorted(
        (path for path in files if path.is_file() and contains in path.name),
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


def select_frames(recorded_times, targets, skip_zero=False):
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

        if skip_zero and target == 0 and np.any(times == 0):
            # Target zero moves forward, never backward to a negative time.
            candidates = np.flatnonzero(times > 0)
            if candidates.size == 0:
                LOG.warning("Skipping target 0: no recorded time greater than 0")
                continue
            index = int(candidates[0])
        else:
            candidates = np.flatnonzero(times != 0) if skip_zero else np.arange(times.size)
            if candidates.size == 0:
                LOG.warning("Skipping target %g: no eligible recorded times", target)
                continue
            index = int(candidates[np.argmin(np.abs(times[candidates] - target))])

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
    """Return deformed quad coordinates and element values for both variables."""
    x = y = None
    values_by_name = {}

    for name in VARIABLES:
        variable_x, variable_y, _z, values = reader.get_data_at_time(name, time)
        variable_x = np.asarray(variable_x, dtype=float)
        variable_y = np.asarray(variable_y, dtype=float)
        values = np.asarray(values, dtype=float).reshape(-1)

        if (
            variable_x.ndim != 2
            or variable_x.shape[1] != 4
            or variable_y.shape != variable_x.shape
            or values.size != variable_x.shape[0]
        ):
            raise ValueError(
                f"Unexpected {name} data shapes at t={time:g}: "
                f"x={variable_x.shape}, y={variable_y.shape}, "
                f"values={values.shape}"
            )

        if x is None:
            x, y = variable_x.copy(), variable_y.copy()
        elif (
            variable_x.shape != x.shape
            or not np.array_equal(variable_x, x)
            or not np.array_equal(variable_y, y)
        ):
            raise ValueError(f"{name} uses different quad coordinates at t={time:g}")

        values_by_name[name] = values

    for name, coordinate in (("disp_x", x), ("disp_y", y)):
        if name not in nodal_names:
            continue

        _nx, _ny, _nz, displacement = reader.get_data_at_time(
            name, time, full_nodal=True
        )
        displacement = np.asarray(displacement, dtype=float)
        if displacement.shape != coordinate.shape:
            raise ValueError(
                f"{name} shape {displacement.shape} does not match "
                f"coordinate shape {coordinate.shape} at t={time:g}"
            )
        coordinate += displacement

    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError(f"Non-finite deformed coordinates at t={time:g}")

    return x, y, values_by_name


def quad_line_intervals(x, y, values_by_name, line_y):
    """
    Find positive-length intersections of a horizontal line with convex quads.

    Returns (left_x, right_x, values) tuples. Quad nodes are assumed to be
    ordered around each element's perimeter.
    """
    intervals = []
    value_columns = [values_by_name[name] for name in VARIABLES]

    for element_index, (xs, ys) in enumerate(zip(x, y)):
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
            if right > left:
                element_values = tuple(
                    float(column[element_index]) for column in value_columns
                )
                intervals.append((left, right, element_values))

    return intervals


def sample_line(x, y, values_by_name, line_y, count):
    intervals = quad_line_intervals(x, y, values_by_name, line_y)
    if not intervals:
        raise ValueError(f"Block does not cross y={line_y:g}")

    left = min(item[0] for item in intervals)
    right = max(item[1] for item in intervals)
    sample_x = np.linspace(left, right, count)
    sample_values = np.full((count, len(VARIABLES)), np.nan)

    # Permit tiny roundoff at shared quad edges, without extending the domain.
    tolerance = 1e-12 * max(1.0, right - left)
    for i, position in enumerate(sample_x):
        for start, end, element_values in intervals:
            if start - tolerance <= position <= end + tolerance:
                sample_values[i] = element_values
                break

    return sample_x, sample_values


def fixed_plot_limits(reader, frames, nodal_names, buffer_fraction=0.02):
    """Find buffered spatial bounds from the first and last selected frames."""
    endpoint_times = (
        min(actual for _target, actual in frames),
        max(actual for _target, actual in frames),
    )

    bounds = []
    for time in dict.fromkeys(endpoint_times):
        x, y, _values = read_deformed_frame(reader, time, nodal_names)
        bounds.append((x.min(), x.max(), y.min(), y.max()))

    xmin = min(bound[0] for bound in bounds)
    xmax = max(bound[1] for bound in bounds)
    ymin = min(bound[2] for bound in bounds)
    ymax = max(bound[3] for bound in bounds)

    xpad = buffer_fraction * (xmax - xmin or 1.0)
    ypad = buffer_fraction * (ymax - ymin or 1.0)
    return xmin - xpad, xmax + xpad, ymin - ypad, ymax + ypad


def save_plot(x, y, values, name, time, color_max, destination, limits):
    vertices = np.stack((x, y), axis=-1)
    fig, ax = plt.subplots(figsize=(6, 9))

    # Porosity retains its configured fixed scale. X_Zr uses the range
    # of finite values in this frame, shown by its colorbar.
    if name == "porosity":
        norm = Normalize(vmin=0, vmax=color_max)
    else:
        finite = values[np.isfinite(values)]
        if finite.size:
            lower, upper = float(finite.min()), float(finite.max())
            if lower == upper:
                upper = lower + 1.0
            norm = Normalize(vmin=lower, vmax=upper)
        else:
            norm = Normalize(vmin=0, vmax=1)

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
    ax.set_box_aspect(2)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"{name}, t={time / 86400:.2f} days")
    fig.colorbar(collection, ax=ax, label=name)
    fig.savefig(destination, dpi=200, bbox_inches="tight")
    plt.close(fig)


def process_file(path, base, args, show_frame_progress):
    # Use one literal filename so this reader handles one simulation.
    reader = MultiExodusReader_V2(str(path), connect_num=CONNECT_NUM)

    try:
        if len(reader.exodus_readers) != 1:
            raise ValueError(
                f"Expected one Exodus file; found {len(reader.exodus_readers)}"
            )

        # frames = select_frames(reader.global_times, args.target_times)
        frames = select_frames(reader.global_times, args.target_times, skip_zero=args.skip0)
        if not frames:
            LOG.warning("%s: no frames selected", path)
            return

        exodus_reader = reader.exodus_readers[0]
        available = set(exodus_reader.nodal_var_names) | set(
            exodus_reader.elem_var_names
        )
        missing = [name for name in VARIABLES if name not in available]
        if missing:
            raise ValueError(
                f"Missing Exodus variable(s) in block {CONNECT_NUM}: "
                f"{', '.join(missing)}"
            )

        nodal_names = set(exodus_reader.nodal_var_names)
        limits = None
        if not args.skip_plots:
            limits = fixed_plot_limits(reader, frames, nodal_names)
            LOG.info("%s: fixed spatial plot limits = %s", path, limits)

        csv_path = args.output_dir / f"{base}_line.csv"
        bounds_path = args.output_dir / f"{base}_bounds.csv"

        with (
            csv_path.open("w", newline="") as output,
            bounds_path.open("w", newline="") as bounds_output,
        ):
            writer = csv.writer(output)
            writer.writerow(["x", *VARIABLES, "time"])

            bounds_writer = csv.writer(bounds_output)
            bounds_writer.writerow(["time", "min_x", "max_x", "min_y", "max_y"])

            iterator = enumerate(frames)
            if show_frame_progress:
                iterator = tqdm(iterator, total=len(frames), desc=path.name)

            for frame_id, (target, actual) in iterator:
                x, y, values_by_name = read_deformed_frame(
                    reader, actual, nodal_names
                )

                bounds_writer.writerow([
                    actual,
                    float(x.min()),
                    float(x.max()),
                    float(y.min()),
                    float(y.max()),
                ])

                if not args.skip_plots:
                    for name in VARIABLES:
                        png_path = (
                            args.output_dir / f"{base}_{name}_{frame_id:04d}.png"
                        )
                        save_plot(
                            x, y, values_by_name[name], name, actual,
                            args.color_max, png_path, limits,
                        )

                try:
                    sample_x, sample_values = sample_line(
                        x, y, values_by_name, args.line_y, args.samples
                    )
                except ValueError as exc:
                    LOG.warning("%s at t=%g: %s", path, actual, exc)
                    continue

                for position, row_values in zip(sample_x, sample_values):
                    writer.writerow([
                        float(position),
                        *(
                            "" if np.isnan(value) else float(value)
                            for value in row_values
                        ),
                        actual,
                    ])

                LOG.info("%s: target=%g, actual=%g", path, target, actual)

        LOG.info("%s: CSV=%s; bounds CSV=%s", path, csv_path, bounds_path)
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
