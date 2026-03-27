#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Find a txt file by partial name and print unique Euler angle triplets."
    )
    p.add_argument(
        "--input", "-i",
        help="Partial filename string to search for with pattern *input*",
    )
    p.add_argument(
        "--subdir",
        default=".",
        help="Directory to search in (default: current directory)",
    )
    return p.parse_args()


def find_file(partial: str, subdir: str = ".") -> str:
    pattern = str(Path(subdir) / f"*{partial}*")
    matches = sorted(glob.glob(pattern))

    if not matches:
        raise FileNotFoundError(f"No files matched pattern: {pattern}")
    if len(matches) > 1:
        print("Multiple matches found, using the first one:", file=sys.stderr)
        for m in matches:
            print(f"  {m}", file=sys.stderr)

    return matches[0]


def read_unique_eulers(filename: str) -> pd.DataFrame:
    # Read whitespace-delimited data, ignoring lines that start with '#'
    df = pd.read_csv(
        filename,
        comment="#",
        delim_whitespace=True,
        header=None,
        usecols=[0, 1, 2],
        names=["phi1", "PHI", "phi2"],
    )

    unique_df = df.drop_duplicates().reset_index(drop=True)
    return unique_df


def main() -> None:
    args = parse_args()

    try:
        filename = find_file(args.input, args.subdir)
        print(f"Using file: {filename}\n")

        unique_eulers = read_unique_eulers(filename)

        print("Unique Euler angle triplets (phi1, PHI, phi2):")
        print(" ")
        print("# phi1     PHI     phi2")
        for row in unique_eulers.itertuples(index=False):
            print(f"{row.phi1:.6f} {row.PHI:.6f} {row.phi2:.6f}")

        print(f"\nTotal unique triplets: {len(unique_eulers)}")

    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
