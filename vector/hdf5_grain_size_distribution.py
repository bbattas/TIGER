#!/usr/bin/env python3
"""
hdf5_grain_size_distribution.py

Read one or more HDF5 files produced by vector_exodus_to_hdf5_vectorized.py
and compute the normalized grain size distribution at every stored frame.

Grain size metric:
    Equivalent circular radius  R = sqrt(area_px / pi)
    where area_px is the number of pixels belonging to a grain in P0.

Normalization (mirrors compare_grain_size_distribution.py):
    R_norm = R / <R>
    where <R> = mean(R) over all non-background grains at that frame.

Histogram binning:
    bin_width = 0.16,  range = [-0.5, 3.5]   (same defaults as reference)
    frequency normalised so that sum(freq * bin_width) = 1.0

Optional statistics overlay (default ON, disable with --no-stats):
    Mean and median of R/<R> plotted as vertical lines with values in legend.

Outputs (per HDF5 file):
    <out_dir>/<stem>_raw_grain_sizes.csv        — frame, time, grain_id, area_px, radius_px
    <out_dir>/<stem>_normalized_grain_sizes.csv — frame, time, grain_id, R_norm
    <out_dir>/plots/<stem>/frame_XXXX.png       — per-frame histogram (if --plot)

Usage examples:
    python hdf5_grain_size_distribution.py
    python hdf5_grain_size_distribution.py --plot -v
    python hdf5_grain_size_distribution.py --name-filter run42 --plot --out-dir results/
    python hdf5_grain_size_distribution.py --bins 30 --bin-width 0.10 -vv
    python hdf5_grain_size_distribution.py --plot --no-stats
"""

import argparse
import csv
import logging
import time
from pathlib import Path

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — safe on HPC nodes
import matplotlib.pyplot as plt
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Logging helpers  (mirrors vector_exodus_to_hdf5_vectorized.py style)
# ---------------------------------------------------------------------------

def setup_logging(verbosity: int) -> logging.Logger:
    level = logging.WARNING
    if verbosity >= 2:
        level = logging.DEBUG
    elif verbosity >= 1:
        level = logging.INFO
    logging.basicConfig(level=level, format="%(message)s")
    return logging.getLogger("GSD")


def tf(ti, log, extra=None):
    msg = (extra or "") + f"Time: {(time.perf_counter() - ti):.4g}s"
    log.warning(msg)


def vtf(ti, log, extra=None):
    msg = (extra or "") + f"Time: {(time.perf_counter() - ti):.4g}s"
    log.info(msg)


def progress(iterable, desc=None, *, verbose=0, **kwargs):
    """Wrap iterable in tqdm only when not in verbose mode."""
    if verbose > 0:
        return iterable
    return tqdm(iterable, desc=desc, **kwargs)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Compute and plot the normalized grain size distribution over time "
            "from HDF5 files produced by vector_exodus_to_hdf5_vectorized.py."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---- General ----
    gen = p.add_argument_group("General")
    gen.add_argument(
        "-v", "--verbose", action="count", default=0,
        help="Increase verbosity (-v = INFO, -vv = DEBUG).",
    )
    gen.add_argument(
        "--h5-dir", type=str, default=".", metavar="DIR",
        dest="h5_dir",
        help="Directory to search for .h5 files.",
    )
    gen.add_argument(
        "--name-filter", type=str, default=None, metavar="STR",
        dest="name_filter",
        help=(
            "Optional partial filename string; only .h5 files whose name "
            "contains this substring will be processed."
        ),
    )

    # ---- Output ----
    out = p.add_argument_group("Output")
    out.add_argument(
        "-o", "--out-dir", type=str, default="gsd_output", metavar="DIR",
        dest="out_dir",
        help="Directory for CSV outputs and plots.",
    )

    # ---- Plotting ----
    plot = p.add_argument_group("Plotting")
    plot.add_argument(
        "-p", "--plot", action="store_true", default=False,
        help="Save a per-frame normalized grain size distribution plot.",
    )
    plot.add_argument(
        "--no-stats", action="store_true", default=False,
        dest="no_stats",
        help=(
            "Disable mean/median vertical lines and values on distribution "
            "plots. Stats are shown by default when --plot is enabled."
        ),
    )

    # ---- Distribution parameters ----
    dist = p.add_argument_group("Distribution parameters")
    dist.add_argument(
        "--bin-width", type=float, default=0.16, metavar="W",
        dest="bin_width",
        help="Histogram bin width for the normalized size distribution.",
    )
    dist.add_argument(
        "--x-min", type=float, default=-0.5, metavar="X",
        dest="x_min",
        help="Lower bound of the normalized size axis (R/<R>).",
    )
    dist.add_argument(
        "--x-max", type=float, default=3.5, metavar="X",
        dest="x_max",
        help="Upper bound of the normalized size axis (R/<R>).",
    )

    return p.parse_args()


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def find_h5_files(h5_dir: str, name_filter: str | None,
                  log: logging.Logger) -> list[Path]:
    """
    Return sorted list of .h5 files in h5_dir, optionally filtered by
    a substring of the filename.
    """
    base = Path(h5_dir).resolve()
    if not base.is_dir():
        raise SystemExit(f"--h5-dir '{h5_dir}' is not a directory.")

    files = sorted(base.glob("*.h5"))

    if name_filter is not None:
        files = [f for f in files if name_filter in f.name]
        log.info(f"Name filter '{name_filter}' applied.")

    if not files:
        desc = f"matching '{name_filter}' " if name_filter else ""
        raise SystemExit(f"No .h5 files {desc}found in '{base}'.")

    log.warning(f"Found {len(files)} .h5 file(s) to process:")
    for f in files:
        log.warning(f"  {f.name}")

    return files


# ---------------------------------------------------------------------------
# HDF5 reading
# ---------------------------------------------------------------------------

def read_frames(h5_path: Path, log: logging.Logger) -> list[dict]:
    """
    Read all frames from an HDF5 file.

    Returns a list of dicts sorted by frame index:
        {"frame_num": int, "time": float, "P0": np.ndarray (nx, ny, int32)}
    """
    frames = []
    with h5py.File(h5_path, "r") as hf:
        if "frames" not in hf:
            raise RuntimeError(
                f"'{h5_path.name}' has no 'frames' group — "
                "is this the correct HDF5 format?"
            )
        frame_keys = sorted(hf["frames"].keys())
        log.info(f"  {h5_path.name}: {len(frame_keys)} frame(s) found.")

        for key in frame_keys:
            fg        = hf["frames"][key]
            frame_num = int(key.split("_")[-1])    # frame_0000 -> 0
            time_val  = float(fg["time"][()])
            P0        = fg["P0"][()]                # (nx, ny) int32

            log.debug(
                f"    {key}: time={time_val:.6g}, "
                f"P0 shape={P0.shape}, "
                f"grain ID range=[{int(np.nanmin(P0))}, {int(np.nanmax(P0))}]"
            )
            frames.append({
                "frame_num": frame_num,
                "time":      time_val,
                "P0":        P0,
            })

    frames.sort(key=lambda d: d["frame_num"])
    return frames


# ---------------------------------------------------------------------------
# Grain size calculation
# ---------------------------------------------------------------------------

def compute_grain_sizes(P0: np.ndarray,
                        log: logging.Logger) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute per-grain pixel area and equivalent circular radius from P0.

    P0 stores integer grain IDs.  Grain ID 0 and NaN are treated as
    background and excluded.

    Returns
    -------
    grain_ids : np.ndarray  (N,)  int32
    areas     : np.ndarray  (N,)  float64   pixel counts
    """
    # Flatten and remove NaN fill pixels
    flat = P0.astype(np.float64).ravel()
    flat = flat[~np.isnan(flat)].astype(np.int32)

    ids, counts = np.unique(flat, return_counts=True)

    # Remove background (grain ID == 0)
    mask   = ids != 0
    ids    = ids[mask]
    counts = counts[mask].astype(np.float64)

    log.debug(
        f"    compute_grain_sizes: {len(ids)} grains, "
        f"total px={int(counts.sum())}, "
        f"mean area={counts.mean():.1f} px"
    )
    return ids, counts


# ---------------------------------------------------------------------------
# Normalization  (mirrors compare_grain_size_distribution.py)
# ---------------------------------------------------------------------------

def compute_radii(areas: np.ndarray) -> np.ndarray:
    """Equivalent circular radius: R = sqrt(area / pi)."""
    return np.sqrt(areas / np.pi)


def normalize_radii(radii: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Return R_norm = R / <R>  and the mean radius <R>.
    Zero-radius entries are excluded from the mean calculation.
    """
    nonzero = radii[radii > 0.0]
    if len(nonzero) == 0:
        return np.zeros_like(radii), 0.0
    mean_r = float(np.mean(nonzero))
    return radii / mean_r, mean_r


def build_distribution(
    R_norm: np.ndarray,
    bin_width: float,
    x_min: float,
    x_max: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Histogram R_norm values into fixed bins and normalise so that
    sum(freq * bin_width) = 1.0  (probability density).

    Matches the binning logic in compare_grain_size_distribution.py [2].

    Returns
    -------
    bin_centers : np.ndarray
    frequency   : np.ndarray  (normalised)
    """
    bin_num = round((abs(x_min) + abs(x_max)) / bin_width)
    bin_centers = np.linspace(
        x_min + bin_width / 2,
        x_max - bin_width / 2,
        bin_num,
    )
    freq = np.zeros(bin_num)

    # Only include grains whose normalised size falls within [x_min, x_max)
    valid = R_norm[(R_norm >= x_min) & (R_norm < x_max)]
    for r in valid:
        idx = int((r - x_min) / bin_width)
        if 0 <= idx < bin_num:
            freq[idx] += 1

    total = np.sum(freq * bin_width)
    if total > 0:
        freq = freq / total

    return bin_centers, freq


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

def write_raw_csv(filepath: Path, rows: list[dict],
                  log: logging.Logger) -> None:
    """
    Write raw grain sizes CSV.

    Columns: frame, time, grain_id, area_px, radius_px
    """
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "time", "grain_id", "area_px", "radius_px"])
        for row in rows:
            writer.writerow([
                row["frame"],
                f"{row['time']:.8g}",
                row["grain_id"],
                f"{row['area_px']:.4f}",
                f"{row['radius_px']:.6f}",
            ])
    log.warning(f"  Raw grain sizes CSV written:        {filepath}")


def write_normalized_csv(filepath: Path, rows: list[dict],
                         log: logging.Logger) -> None:
    """
    Write normalized grain sizes CSV.

    Columns: frame, time, grain_id, R_norm
    """
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "time", "grain_id", "R_norm"])
        for row in rows:
            writer.writerow([
                row["frame"],
                f"{row['time']:.8g}",
                row["grain_id"],
                f"{row['R_norm']:.8f}",
            ])
    log.warning(f"  Normalized grain sizes CSV written: {filepath}")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_normalized_distribution(
    bin_centers: np.ndarray,
    freq: np.ndarray,
    R_norm: np.ndarray,
    time_val: float,
    frame_num: int,
    grain_count: int,
    mean_r: float,
    plot_dir: Path,
    x_min: float,
    x_max: float,
    show_stats: bool,
    log: logging.Logger,
) -> None:
    """
    Save a single normalised grain size distribution histogram as a PNG.

    Filename : frame_XXXX.png  inside plot_dir.
    Title    : carries the simulation time so frames can be assembled into
               a video.

    Parameters
    ----------
    bin_centers : bin centre positions for the distribution curve
    freq        : normalised frequency values (area-under-curve = 1)
    R_norm      : per-grain normalised radii — used for mean/median lines
    show_stats  : if True, overlay mean and median as vertical lines with
                  values annotated in the legend
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(bin_centers, freq, color="steelblue", linewidth=2,
            label=f"Simulation  (N={grain_count})")

    # --- Optional statistics overlay (default ON, disabled by --no-stats) ---
    if show_stats:
        valid      = R_norm[R_norm > 0.0]
        mean_norm  = float(np.mean(valid))
        median_norm = float(np.median(valid))

        ax.axvline(
            mean_norm,
            color="red", linestyle="--", linewidth=1.5,
            label=f"Mean:   {mean_norm:.4f}",
        )
        ax.axvline(
            median_norm,
            color="orange", linestyle=":", linewidth=1.5,
            label=f"Median: {median_norm:.4f}",
        )

    ax.set_xlabel(r"$R / \langle R \rangle$", fontsize=14)
    ax.set_ylabel("Frequency (density)", fontsize=14)
    ax.set_xlim([x_min, x_max])
    ax.set_ylim(bottom=0)
    ax.tick_params(labelsize=12)
    ax.legend(fontsize=12)
    ax.set_title(
        f"Normalised Grain Size Distribution\n"
        f"t = {time_val:.4g}   "
        f"$\\langle R \\rangle$ = {mean_r:.3f} px",
        fontsize=13,
    )

    fig.tight_layout()
    fname = plot_dir / f"frame_{frame_num:04d}.png"
    fig.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close(fig)

    log.info(f"    Plot saved: {fname.name}")


# ---------------------------------------------------------------------------
# Per-file orchestration
# ---------------------------------------------------------------------------

def process_h5_file(
    h5_path: Path,
    args: argparse.Namespace,
    out_dir: Path,
    log: logging.Logger,
) -> None:
    """
    Full pipeline for a single HDF5 file:
      1. Read frames
      2. Compute grain sizes and normalization for every frame
      3. Write raw and normalised CSVs
      4. Optionally write per-frame plots
    """
    ti   = time.perf_counter()
    stem = h5_path.stem

    log.warning(f"\nProcessing: {h5_path.name}")

    # --- Read ---
    frames = read_frames(h5_path, log)
    log.info(f"  {len(frames)} frame(s) loaded.")

    raw_rows  = []
    norm_rows = []

    # Per-file plot subdirectory — uses stem so multiple files never collide
    plot_dir = out_dir / "plots" / stem
    if args.plot:
        plot_dir.mkdir(parents=True, exist_ok=True)

    # --- Per-frame processing ---
    for frame in progress(
        frames,
        desc=f"  {stem}",
        verbose=args.verbose,
        total=len(frames),
    ):
        fnum     = frame["frame_num"]
        time_val = frame["time"]
        P0       = frame["P0"]

        tif = time.perf_counter()

        # Grain sizes
        grain_ids, areas = compute_grain_sizes(P0, log)
        radii            = compute_radii(areas)
        R_norm, mean_r   = normalize_radii(radii)

        log.info(
            f"  frame {fnum:04d}  t={time_val:.4g}  "
            f"grains={len(grain_ids)}  <R>={mean_r:.3f} px"
        )

        # Accumulate CSV rows
        for gid, area, rad, rn in zip(grain_ids, areas, radii, R_norm):
            raw_rows.append({
                "frame":     fnum,
                "time":      time_val,
                "grain_id":  int(gid),
                "area_px":   float(area),
                "radius_px": float(rad),
            })
            norm_rows.append({
                "frame":    fnum,
                "time":     time_val,
                "grain_id": int(gid),
                "R_norm":   float(rn),
            })

        # Optional plot
        if args.plot:
            bin_centers, freq = build_distribution(
                R_norm,
                bin_width=args.bin_width,
                x_min=args.x_min,
                x_max=args.x_max,
            )
            plot_normalized_distribution(
                bin_centers, freq,
                R_norm=R_norm,
                show_stats=not args.no_stats,
                time_val=time_val,
                frame_num=fnum,
                grain_count=len(grain_ids),
                mean_r=mean_r,
                plot_dir=plot_dir,
                x_min=args.x_min,
                x_max=args.x_max,
                log=log,
            )

        vtf(tif, log, extra=f"    frame {fnum:04d} processed: ")

    # --- Write CSVs ---
    write_raw_csv(
        out_dir / f"{stem}_raw_grain_sizes.csv",
        raw_rows,
        log,
    )
    write_normalized_csv(
        out_dir / f"{stem}_normalized_grain_sizes.csv",
        norm_rows,
        log,
    )

    tf(ti, log, extra=f"  {stem} total: ")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ti   = time.perf_counter()
    args = parse_args()
    log  = setup_logging(args.verbose)

    log.info("hdf5_grain_size_distribution.py  —  starting")
    log.info(f"Arguments: {args}")

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    log.warning(f"Output directory: {out_dir}")

    h5_files = find_h5_files(args.h5_dir, args.name_filter, log)

    for h5_path in h5_files:
        try:
            process_h5_file(h5_path, args, out_dir, log)
        except Exception as e:
            log.exception(f"ERROR processing {h5_path.name}: {e}")

    tf(ti, log, extra="Total runtime: ")


if __name__ == "__main__":
    main()
