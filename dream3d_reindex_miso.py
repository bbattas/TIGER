#!/usr/bin/env python3
"""
Shift DREAM3D EBSD FeatureId column (7th) so indexing starts at 0.

- Preserves header lines that start with '#'
- Preserves original whitespace formatting in data lines (only changes the FeatureId token)
- Input selection:
    * If -i/--input is an existing file path, uses it directly
    * Otherwise, treats it as a name fragment and searches for *<name>*.txt
"""

from __future__ import annotations

import argparse
import glob
import os
import re
from pathlib import Path
from typing import List, Tuple


WS_SPLIT_KEEP = re.compile(r"(\s+)")  # split but keep whitespace separators


def resolve_input(name_or_path: str) -> Path:
    p = Path(name_or_path)
    if p.is_file():
        return p

    # Treat as name fragment; search in current directory
    matches = [Path(m) for m in glob.glob(f"*{name_or_path}*.txt")]
    matches = [m for m in matches if m.is_file()]

    if not matches:
        raise FileNotFoundError(f"No matches for pattern '*{name_or_path}*.txt' in {os.getcwd()}")

    if len(matches) > 1:
        mlist = "\n".join(f"  - {m}" for m in sorted(matches))
        raise RuntimeError(
            f"Ambiguous input: {len(matches)} files match '*{name_or_path}*.txt'.\n"
            f"Be more specific or pass a full path with -i.\nMatches:\n{mlist}"
        )

    return matches[0]


def default_output_path(inp: Path) -> Path:
    # e.g., 2D_5x5.txt -> 2D_5x5_fid0.txt
    return inp.with_name(f"{inp.stem}_fid0{inp.suffix}")


def split_with_whitespace(line: str) -> Tuple[List[str], List[str]]:
    """
    Return (tokens, seps) where:
      - tokens are the non-whitespace chunks
      - seps are the whitespace chunks between tokens
    Reconstruction rule:
      out = tokens[0] + seps[0] + tokens[1] + seps[1] + ...
    """
    parts = WS_SPLIT_KEEP.split(line.rstrip("\n"))
    tokens: List[str] = []
    seps: List[str] = []
    # parts alternates: token, sep, token, sep, ...
    for i, part in enumerate(parts):
        if part == "":
            continue
        if i % 2 == 0:
            # token position (non-whitespace)
            if part.strip() != "":
                tokens.append(part)
        else:
            # separator position (whitespace)
            seps.append(part)

    # Ensure seps length is tokens-1 (common case). If not, we'll handle in reconstruction.
    return tokens, seps


def reconstruct_line(tokens: List[str], seps: List[str]) -> str:
    if not tokens:
        return ""
    out = [tokens[0]]
    for i in range(1, len(tokens)):
        sep = seps[i - 1] if i - 1 < len(seps) else " "
        out.append(sep)
        out.append(tokens[i])
    return "".join(out)


def is_data_line(line: str) -> bool:
    s = line.strip()
    return bool(s) and (not s.startswith("#"))


def main() -> None:
    ap = argparse.ArgumentParser(description="Shift DREAM3D EBSD FeatureId so it starts at 0.")
    ap.add_argument(
        "-i",
        "--input",
        required=True,
        help="Input file path OR name fragment to match '*name*.txt' in the current directory.",
    )
    ap.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output file path. Default: <input_stem>_fid0.txt",
    )
    args = ap.parse_args()

    inp = resolve_input(args.input)
    out = Path(args.output) if args.output else default_output_path(inp)

    lines = inp.read_text(encoding="utf-8", errors="replace").splitlines(True)

    # First pass: collect FeatureIds from data lines
    feature_ids: List[int] = []
    parsed: List[Tuple[bool, str, List[str], List[str]]] = []
    # tuple: (is_data, original_line, tokens, seps)

    for line in lines:
        if not is_data_line(line):
            parsed.append((False, line, [], []))
            continue

        tokens, seps = split_with_whitespace(line)
        if len(tokens) < 7:
            raise ValueError(f"Data line has fewer than 7 columns:\n{line}")

        # FeatureId is 7th column => index 6
        try:
            fid = int(tokens[6])
        except ValueError:
            # Sometimes it might be written as "2.0" etc.
            fid = int(float(tokens[6]))

        feature_ids.append(fid)
        parsed.append((True, line, tokens, seps))

    if not feature_ids:
        raise ValueError("No data lines found (nothing to shift).")

    offset = min(feature_ids)
    # If already starts at 0, offset=0 and we leave it unchanged
    print(f"Input:  {inp}")
    print(f"Output: {out}")
    print(f"FeatureId min = {offset} -> shifting by -{offset}")

    # Second pass: write output
    out_lines: List[str] = []
    for is_data, original, tokens, seps in parsed:
        if not is_data:
            out_lines.append(original.rstrip("\n") + "\n")
            continue

        fid = int(float(tokens[6]))
        tokens[6] = str(fid - offset)
        new_line = reconstruct_line(tokens, seps)
        out_lines.append(new_line + "\n")

    out.write_text("".join(out_lines), encoding="utf-8")
    print("Done.")


if __name__ == "__main__":
    main()
