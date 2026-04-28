#!/usr/bin/env python3
"""
fmiso_distribution.py

Standalone script to compute f_miso for all grain pairs in a
misorientation CSV or Parquet file, then plot the PDF histogram
and CDF of the resulting distribution.

Usage:
    python fmiso_distribution.py -m <misorientation_file> [options]

Options:
    -m, --misorientation   Path or glob to CSV / Parquet file (required)
    --miso-format          'auto' | 'csv' | 'parquet'  (default: auto)
    --batch-size           Rows per processing batch    (default: 100000)
    --bins                 Histogram bin count          (default: 60)
    --dpi                  Plot DPI                     (default: 150)
    --out-dir              Output directory             (default: .)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────

THETA_MAX_DEG = 62.0
THETA_MAX_RAD = np.deg2rad(THETA_MAX_DEG)


# ─────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Compute f_miso distribution from a misorientation file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("-m", "--misorientation", required=True,
                   help="Path or glob to misorientation CSV or Parquet file.")
    p.add_argument("--miso-format", choices=("auto", "csv", "parquet"),
                   default="auto", dest="miso_format")
    p.add_argument("--batch-size", type=int, default=100_000,
                   dest="batch_size",
                   help="Number of rows to process per batch.")
    p.add_argument("--bins", type=int, default=60,
                   help="Number of histogram bins.")
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument("--out-dir", type=Path, default=Path("."), dest="out_dir")
    return p.parse_args()


# ─────────────────────────────────────────────
#  File resolution
# ─────────────────────────────────────────────

def find_misorientation_file(partial: str, fmt: str = "auto") -> tuple[Path, str]:
    exact = Path(partial)
    if exact.is_file():
        return exact, _detect_format(exact, fmt)

    cwd = Path.cwd()

    def _candidates(ext: str) -> list[Path]:
        pattern = partial if partial.endswith(ext) else f"{partial}*{ext}"
        found = sorted(cwd.glob(pattern)) + sorted(cwd.glob(f"*/{pattern}"))
        return [p for p in found if p.is_file()]

    if fmt in ("parquet", "auto"):
        hits = _candidates(".parquet")
        if hits:
            return hits[0], "parquet"

    if fmt in ("csv", "auto"):
        hits = _candidates(".csv")
        if hits:
            return hits[0], "csv"

    raise FileNotFoundError(
        f"No misorientation file found matching '{partial}' (format='{fmt}')."
    )


def _detect_format(path: Path, hint: str) -> str:
    if hint in ("csv", "parquet"):
        return hint
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return "parquet"
    if suffix == ".csv":
        return "csv"
    raise ValueError(
        f"Cannot detect format from extension '{suffix}'. "
        "Use --miso-format csv or --miso-format parquet."
    )


# ─────────────────────────────────────────────
#  Vectorised f_miso (operates on numpy arrays)
# ─────────────────────────────────────────────

def compute_fmiso_vectorised(
    angle_deg: np.ndarray,
    ax_x: np.ndarray,
    ax_y: np.ndarray,
    ax_z: np.ndarray,
) -> np.ndarray:
    """
    Vectorised version of compute_fmiso_single.
    All inputs are 1-D arrays of the same length.
    Returns f_miso array with values in [0.3, ~1.0].
    """
    # --- normalise axis ---
    norm = np.sqrt(ax_x**2 + ax_y**2 + ax_z**2)
    degenerate = norm < 1e-12

    # Safe normalisation (degenerate rows get a dummy axis)
    norm_safe = np.where(degenerate, 1.0, norm)
    nx = ax_x / norm_safe
    ny = ax_y / norm_safe
    nz = ax_z / norm_safe

    # --- polar / azimuth ---
    polar   = np.arccos(np.clip(nz, -1.0, 1.0))
    azimuth = np.arctan2(ny, nx)
    azimuth = np.where(azimuth < 0.0, azimuth + 2.0 * np.pi, azimuth)

    # --- ang_energy ---
    theta_rad  = np.deg2rad(angle_deg)
    ratio      = theta_rad / THETA_MAX_RAD
    ratio      = np.clip(ratio, 0.0, None)

    ang_energy = np.where(
        ratio <= 0.0,
        0.0,
        np.where(
            ratio >= 1.0,
            1.0,
            ratio * (1.0 - np.log(np.clip(ratio, 1e-300, None))),
        ),
    )
    ang_energy = np.minimum(ang_energy, 1.0)

    # --- ax_energy ---
    ax_energy = (
        np.abs(np.cos(polar)) ** 0.4
        + np.abs(np.cos(azimuth / 2.0)) ** 0.4
    )
    ax_energy = np.minimum(ax_energy, 1.0)

    # --- f_miso ---
    f_miso = 0.3 + 0.7 * (ang_energy * ax_energy)

    # Degenerate pairs → minimum energy
    f_miso = np.where(degenerate, 0.3, f_miso)

    return f_miso.astype(np.float64)


# ─────────────────────────────────────────────
#  Batch loader & compute
# ─────────────────────────────────────────────

REQUIRED_COLS = ["i", "j", "angle_deg", "ax_x", "ax_y", "ax_z"]


def iter_batches_csv(path: Path, batch_size: int):
    """Yield DataFrames in chunks from a CSV file."""
    for chunk in pd.read_csv(path, chunksize=batch_size):
        chunk.columns = [c.strip().lower() for c in chunk.columns]
        yield chunk


def iter_batches_parquet(path: Path, batch_size: int):
    """Yield DataFrames in row-group batches from a Parquet file."""
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)
    buffer: list[pd.DataFrame] = []
    buffer_len = 0

    for rg_idx in range(pf.metadata.num_row_groups):
        rg_df = pf.read_row_group(
            rg_idx, columns=REQUIRED_COLS
        ).to_pandas()
        buffer.append(rg_df)
        buffer_len += len(rg_df)

        if buffer_len >= batch_size:
            combined = pd.concat(buffer, ignore_index=True)
            buffer, buffer_len = [], 0
            # Yield sub-batches if the combined block is very large
            for start in range(0, len(combined), batch_size):
                yield combined.iloc[start : start + batch_size]

    if buffer:
        combined = pd.concat(buffer, ignore_index=True)
        for start in range(0, len(combined), batch_size):
            yield combined.iloc[start : start + batch_size]


def compute_all_fmiso(path: Path, fmt: str, batch_size: int) -> np.ndarray:
    """
    Stream through the file in batches and accumulate all f_miso values.
    Returns a single 1-D float64 array.
    """
    iterator = (
        iter_batches_csv(path, batch_size)
        if fmt == "csv"
        else iter_batches_parquet(path, batch_size)
    )

    all_fmiso: list[np.ndarray] = []
    total_rows = 0

    for batch_idx, df in enumerate(iterator):
        missing = set(REQUIRED_COLS) - set(df.columns)
        if missing:
            raise ValueError(f"Batch {batch_idx}: missing columns {missing}")

        fmiso = compute_fmiso_vectorised(
            df["angle_deg"].to_numpy(dtype=np.float64),
            df["ax_x"].to_numpy(dtype=np.float64),
            df["ax_y"].to_numpy(dtype=np.float64),
            df["ax_z"].to_numpy(dtype=np.float64),
        )
        all_fmiso.append(fmiso)
        total_rows += len(fmiso)
        print(f"  Batch {batch_idx:>4d}: {len(fmiso):>10,} pairs  "
              f"(running total: {total_rows:,})", flush=True)

    if not all_fmiso:
        raise RuntimeError("No data found in misorientation file.")

    return np.concatenate(all_fmiso)


# ─────────────────────────────────────────────
#  Plotting
# ─────────────────────────────────────────────

def plot_histogram(fmiso: np.ndarray, bins: int, out_dir: Path, dpi: int):
    fig, ax = plt.subplots(constrained_layout=True) #figsize=(7, 4),

    ax.hist(
        fmiso,
        bins=bins,
        range=(0.3, 1.0),
        density=True,
        color="steelblue",
        edgecolor="white",
        linewidth=0.4,
    )

    ax.set_xlabel("$f_{miso}$", fontsize=13)
    ax.set_ylabel("Probability density", fontsize=12)
    ax.set_title(
        f"$f_{{miso}}$ distribution  "
        f"(n = {len(fmiso):,}  |  "
        f"mean = {fmiso.mean():.4f}  |  "
        f"median = {np.median(fmiso):.4f})",
        fontsize=11,
    )
    ax.set_xlim(0.3, 1.0)
    ax.grid(axis="y", linewidth=0.4, alpha=0.6)

    out_path = out_dir / "fmiso_histogram.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_cdf(fmiso: np.ndarray, out_dir: Path, dpi: int):
    sorted_vals = np.sort(fmiso)
    cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)

    fig, ax = plt.subplots(constrained_layout=True) #figsize=(7, 4),
    ax.plot(sorted_vals, cdf, color="darkorange", linewidth=1.6)

    ax.set_xlabel("$f_{miso}$", fontsize=13)
    ax.set_ylabel("Cumulative probability", fontsize=12)
    ax.set_title(
        f"$f_{{miso}}$ CDF  (n = {len(fmiso):,})",
        fontsize=11,
    )
    ax.set_xlim(0.3, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(linewidth=0.4, alpha=0.6)

    out_path = out_dir / "fmiso_cdf.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────

def main():
    args = parse_args()

    # Locate file
    try:
        miso_path, miso_fmt = find_misorientation_file(
            args.misorientation, args.miso_format
        )
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Misorientation file : {miso_path}")
    print(f"Format              : {miso_fmt}")
    print(f"Batch size          : {args.batch_size:,}")
    print()

    # Compute
    print("Computing f_miso ...")
    fmiso = compute_all_fmiso(miso_path, miso_fmt, args.batch_size)
    print(f"\nDone.  Total pairs: {len(fmiso):,}")
    print(f"  min  = {fmiso.min():.4f}")
    print(f"  max  = {fmiso.max():.4f}")
    print(f"  mean = {fmiso.mean():.4f}")
    print(f"  std  = {fmiso.std():.4f}")
    print()

    # Plot
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print("Saving plots ...")
    plot_histogram(fmiso, args.bins, args.out_dir, args.dpi)
    plot_cdf(fmiso, args.out_dir, args.dpi)


if __name__ == "__main__":
    main()
