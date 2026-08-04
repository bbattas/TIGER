import argparse
import glob
import math
from pathlib import Path


BASE_DECIMALS = 6


def get_output_decimals(multiplier: float) -> int:
    """Reduce decimal places based on the order of magnitude of the multiplier."""
    if multiplier <= 0:
        return BASE_DECIMALS
    reduction = int(math.log10(multiplier))
    return max(0, BASE_DECIMALS - reduction)


def main():
    parser = argparse.ArgumentParser(
        description="Rescale all numeric values in a whitespace-delimited txt file."
    )
    parser.add_argument(
        "-i", "--input",
        required=True,
        help="Partial filename string used to locate the input file (glob: *input*)"
    )
    parser.add_argument(
        "-m", "--multiplier",
        type=float,
        default=1000.0,
        help="Value to multiply all numbers by (default: 1000)"
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output filename (default: <input_base>_rescaled.txt)"
    )
    args = parser.parse_args()

    # --- File discovery ---
    matches = glob.glob(f"*{args.input}*")
    if len(matches) == 0:
        print(f"Error: no files found matching '*{args.input}*'")
        return
    if len(matches) > 1:
        print(f"Error: multiple files matched '*{args.input}*'. Be more specific:")
        for m in matches:
            print(f"  {m}")
        return
    input_path = Path(matches[0])

    # --- Read file ---
    with open(input_path, "r") as f:
        lines = [line.rstrip("\n") for line in f if line.strip()]

    header = lines[0]
    rows = []
    for line in lines[1:]:
        tokens = line.split()
        try:
            rows.append([float(t) for t in tokens])
        except ValueError as e:
            print(f"Error parsing line '{line}': {e}")
            return

    # --- Scale values ---
    scaled_rows = [
        [val * args.multiplier for val in row]
        for row in rows
    ]

    # --- Decimal places ---
    decimals = get_output_decimals(args.multiplier)
    fmt = f".{decimals}f"

    # --- Output filename ---
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.with_name(input_path.stem + "_rescaled.txt")

    # --- Write output ---
    with open(output_path, "w") as f:
        f.write(header + "\n")
        for row in scaled_rows:
            f.write(" ".join(format(val, fmt) for val in row) + "\n")

    print(f"Done. Rescaled file written to: {output_path}")
    print(f"  Multiplier : {args.multiplier}")
    print(f"  Decimal places : {decimals} (reduced from {BASE_DECIMALS})")


if __name__ == "__main__":
    main()
