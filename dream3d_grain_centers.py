"""
dream3d_grain_centers.py

Reads one or more DREAM.3D grain structure .txt files (matched via glob),
computes the centroid of each grain (FeatureId != 0), and writes a
space-delimited output file with columns:
  3D:  x y z
  2D:  x y

Usage:
  python dream3d_grain_centers.py -i "data/sample_*.txt"
  python dream3d_grain_centers.py -i "data/sample_*.txt" -o my_output.txt
"""

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 1. Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute grain centroids from a DREAM.3D grain structure txt file."
    )
    parser.add_argument(
        "-i", "--input",
        required=True,
        help=(
            "Input file path or glob pattern (e.g. 'data/sample_*.txt'). "
            "Quote the pattern to prevent shell expansion."
        ),
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help=(
            "Output file path. If omitted, defaults to <input_stem>_grain_ctrs.txt "
            "in the same directory as the input file. "
            "Ignored when multiple input files are matched."
        ),
    )
    return parser.parse_args()


def resolve_input_files(pattern: str) -> list[Path]:
    """
    Expand input to a glob pattern and return matching Path objects.

    - If input ends with '.txt': searches for '*<input>'
      e.g. 'test.txt'  -> '*test.txt'
    - Otherwise: searches for '*<input>*.txt'
      e.g. 'tiny'      -> '*tiny*.txt'
    """
    if pattern.endswith(".txt"):
        glob_pattern = f"*{pattern}"
    else:
        glob_pattern = f"*{pattern}*.txt"

    matches = sorted(glob.glob(glob_pattern, recursive=False))
    if not matches:
        print(f"[ERROR] No files matched the pattern: {glob_pattern}", file=sys.stderr)
        sys.exit(1)
    return [Path(m) for m in matches]


def default_output_path(input_path: Path) -> Path:
    """Return <stem>_grain_ctrs.txt next to the input file."""
    return input_path.parent / (input_path.stem + "_grain_ctrs.txt")


# ---------------------------------------------------------------------------
# 2. Header parsing
# ---------------------------------------------------------------------------

def parse_header(filepath: Path) -> tuple[int, dict]:
    """
    Scan lines that start with '#' at the top of the file.

    Returns
    -------
    skip_rows : int
        Number of lines to skip when reading data (all '#' lines).
    meta : dict
        Parsed key/value pairs from the header (keys upper-cased, e.g. 'Z_DIM').
    """
    meta = {}
    skip_rows = 0

    with open(filepath, "r") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped.startswith("#"):
                break
            skip_rows += 1
            # Parse lines like:  # KEY: value
            content = stripped.lstrip("#").strip()
            if ":" in content:
                key, _, value = content.partition(":")
                meta[key.strip().upper()] = value.strip()

    return skip_rows, meta


# ---------------------------------------------------------------------------
# 3. 2-D detection
# ---------------------------------------------------------------------------

def detect_2d(meta: dict, z_col: np.ndarray | None = None) -> bool:
    """
    Return True if the dataset is 2-D.

    Primary check  : Z_DIM in header is 0 or 1.
    Secondary check: all z coordinate values are identical (fallback if
                     Z_DIM is missing or ambiguous).
    """
    z_dim_str = meta.get("Z_DIM", "").strip()
    if z_dim_str:
        try:
            z_dim = int(z_dim_str)
            if z_dim <= 1:
                return True
            # Z_DIM > 1 is definitively 3-D; skip secondary check
            return False
        except ValueError:
            pass  # fall through to coordinate check

    # Fallback: check if all z values are the same
    if z_col is not None and len(z_col) > 0:
        return bool(np.all(z_col == z_col[0]))

    return False


# ---------------------------------------------------------------------------
# 4. Data loading  (optimised for very large files)
# ---------------------------------------------------------------------------

# Column indices in the data block (0-based, after header lines are skipped)
# phi1  PHI  phi2  x  y  z  FeatureId  PhaseId  Symmetry
#   0    1    2    3  4  5      6          7         8

_COL_NAMES  = ["x", "y", "z", "feature_id"]
_COL_INDICES = [3, 4, 5, 6]


def load_data(filepath: Path, skip_rows: int) -> pd.DataFrame:
    """
    Read only the x, y, z, and FeatureId columns from the data block.

    Uses pandas read_csv with explicit usecols for minimal memory footprint.
    float32 for coordinates, int32 for FeatureId.
    """
    df = pd.read_csv(
        filepath,
        sep=r"\s+",
        header=None,
        skiprows=skip_rows,
        usecols=_COL_INDICES,
        names=_COL_NAMES,
        dtype={
            "x":          np.float32,
            "y":          np.float32,
            "z":          np.float32,
            "feature_id": np.int32,
        },
        engine="c",          # fastest parser
        memory_map=True,     # avoids reading entire file into RAM at once
    )
    return df


# ---------------------------------------------------------------------------
# 5. Centroid computation  (fully vectorised)
# ---------------------------------------------------------------------------

def compute_centroids(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the mean x, y, z position for each FeatureId.

    FeatureId == 0 is excluded (DREAM.3D 'bad data' marker).
    Returns a DataFrame sorted by FeatureId with columns x, y, z.
    """
    # Drop unassigned / bad-data voxels
    df = df[df["feature_id"] != 0]

    # groupby + mean is internally vectorised via Cython/numpy in pandas
    centroids = (
        df.groupby("feature_id", sort=True)[["x", "y", "z"]]
        .mean()
        .reset_index(drop=True)
    )
    return centroids


# ---------------------------------------------------------------------------
# 6. Output writing
# ---------------------------------------------------------------------------

def write_output(centroids: pd.DataFrame, output_path: Path, is_2d: bool) -> None:
    """Write space-delimited centroid file with a header row."""
    if is_2d:
        cols   = ["x", "y"]
        header = "x y"
    else:
        cols   = ["x", "y", "z"]
        header = "x y z"

    with open(output_path, "w") as fh:
        fh.write(header + "\n")
        # numpy savetxt is fast for large arrays; convert to float64 for output
        np.savetxt(
            fh,
            centroids[cols].values.astype(np.float64),
            fmt="%.6f",
            delimiter=" ",
        )

    print(f"  -> Written: {output_path}  ({len(centroids)} grains)")


# ---------------------------------------------------------------------------
# 7. Main
# ---------------------------------------------------------------------------

def process_file(input_path: Path, output_path: Path) -> None:
    print(f"Processing: {input_path}")

    # --- Header ---
    skip_rows, meta = parse_header(input_path)
    print(f"  Header lines skipped : {skip_rows}")
    if "NUM_FEATURES" in meta or "Num_Features".upper() in meta:
        n_feat = meta.get("NUM_FEATURES", meta.get("NUM_FEATURES", "?"))
        print(f"  Num_Features (header): {n_feat}")

    # --- Load ---
    df = load_data(input_path, skip_rows)
    print(f"  Voxels loaded        : {len(df):,}")

    # --- 2-D detection ---
    is_2d = detect_2d(meta, z_col=df["z"].values)
    print(f"  Mode                 : {'2-D' if is_2d else '3-D'}")

    # --- Centroids ---
    centroids = compute_centroids(df)
    print(f"  Grains found         : {len(centroids)}")

    # --- Write ---
    write_output(centroids, output_path, is_2d)


def main():
    args = parse_args()

    input_files = resolve_input_files(args.input)
    multiple    = len(input_files) > 1

    if multiple and args.output:
        print(
            "[WARNING] --output is ignored when multiple input files are matched; "
            "each file gets its own default output name.",
            file=sys.stderr,
        )

    for input_path in input_files:
        if multiple or args.output is None:
            output_path = default_output_path(input_path)
        else:
            output_path = Path(args.output)

        process_file(input_path, output_path)

    print("Done.")


if __name__ == "__main__":
    main()
