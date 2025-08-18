#!/usr/bin/env python3
import argparse
import glob
import json
from pathlib import Path
import sys
from typing import List, Tuple, Dict, Any

JSON_NAME = "voronoi_meta.json"

def resolve_input_path(pattern: str) -> Path:
    if any(ch in pattern for ch in ["*", "?", "["]):
        matches = glob.glob(pattern)
    else:
        matches = glob.glob(f"*{pattern}*")
    matches = [m for m in matches if Path(m).is_file()]
    if not matches:
        sys.exit(f"[error] No files found matching pattern: {pattern!r}")
    if len(matches) > 1:
        msg = "\n".join(f"  - {m}" for m in matches)
        sys.exit(
            "[error] Multiple files match your pattern. Please be more specific "
            f"or pass a glob with wildcards.\nMatches:\n{msg}"
        )
    return Path(matches[0]).resolve()

def ensure_txt_suffix(name: str) -> str:
    p = Path(name)
    if p.suffix.lower() != ".txt":
        p = p.with_suffix(".txt")
    return str(p)

def read_points(input_path: Path) -> Tuple[str, List[Tuple[float, float]]]:
    points: List[Tuple[float, float]] = []
    header = None
    with input_path.open("r", encoding="utf-8") as fin:
        for line in fin:
            stripped = line.strip()
            if stripped == "":
                continue
            header = stripped
            break
        if header is None:
            sys.exit(f"[error] Input file {input_path} appears to be empty.")
        for line in fin:
            stripped = line.strip()
            if stripped == "" or stripped.startswith("#"):
                continue
            parts = stripped.split()
            if len(parts) < 2:
                continue
            try:
                x = float(parts[0]); y = float(parts[1])
            except ValueError:
                continue
            points.append((x, y))
    return header, points

# def write_filtered(header: str, filtered_points: List[Tuple[float, float]], output_path: Path) -> None:
#     with output_path.open("w", encoding="utf-8") as fout:
#         fout.write(header.strip() + "\n")
#         for x, y in filtered_points:
#             fout.write(f"{x} {y}\n")
def write_filtered(header, filtered_points, output_path: Path, decimals: int | None = None) -> None:
    with output_path.open("w", encoding="utf-8") as fout:
        fout.write(header.strip() + "\n")
        if decimals is None:
            for x, y in filtered_points:
                fout.write(f"{x} {y}\n")
        else:
            print('Decimals = ',decimals)
            # Note the single \n below
            # fmt = f"{{:.{decimals}f}} {{:.{decimals}f}}\n"
            # for x, y in filtered_points:
            #     fout.write(fmt.format(x, y))
            for x, y in filtered_points:
                fout.write(f"{x:.{decimals}f} {y:.{decimals}f}\n")

def filter_points(
    points: List[Tuple[float, float]],
    max_x: float,
    max_y: float,
    min_x: float | None = None,
    min_y: float | None = None
) -> List[Tuple[float, float]]:
    low_x = float("-inf") if min_x is None else min_x
    low_y = float("-inf") if min_y is None else min_y
    return [(x, y) for (x, y) in points if (low_x <= x <= max_x and low_y <= y <= max_y)]

def plot_points(
    all_points: List[Tuple[float, float]],
    kept_points: List[Tuple[float, float]],
    max_x: float,
    max_y: float,
    min_x: float | None = None,
    min_y: float | None = None,
    title: str = "Point filtering by bounds",
    save_path: str | None = None
) -> None:
    import matplotlib.pyplot as plt
    if all_points:
        xa, ya = zip(*all_points)
    else:
        xa, ya = [], []
    if kept_points:
        xk, yk = zip(*kept_points)
    else:
        xk, yk = [], []

    plt.figure()
    if len(xa) > 0:
        plt.scatter(xa, ya, s=10, c="black", label="original")

    if min_x is not None:
        plt.axvline(min_x, linestyle="--", linewidth=1, color="gray", label=f"x = {min_x}")
    if min_y is not None:
        plt.axhline(min_y, linestyle="--", linewidth=1, color="gray", label=f"y = {min_y}")

    plt.axvline(max_x, linestyle="--", linewidth=1, color="gray", label=f"x = {max_x}")
    plt.axhline(max_y, linestyle="--", linewidth=1, color="gray", label=f"y = {max_y}")

    if len(xk) > 0:
        plt.scatter(xk, yk, s=36, facecolors="none", edgecolors="red", linewidths=1.2, label="kept")

    plt.gca().set_aspect("equal", adjustable="datalim")
    plt.xlabel("x"); plt.ylabel("y")
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle=":", linewidth=0.8)

    if save_path:
        plt.savefig(str(save_path), dpi=300, bbox_inches="tight")
        print(f"[info] Saved plot to {save_path}")
    try:
        plt.show()
    except Exception as e:
        print(f"[warn] Could not display plot ({e!r}). If needed, use --save-plot to save to a file.")

def base_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Filter (x, y) points by min/max bounds; optional plotting; optional JSON config."
    )
    # Make all options non-required here; we'll enforce presence after merging JSON+CLI.
    p.add_argument("--input_pattern","-i", help="Substring or glob for the input file.")
    p.add_argument("--max-x","-x", type=float, help="Maximum allowed x (inclusive).")
    p.add_argument("--max-y","-y", type=float, help="Maximum allowed y (inclusive).")
    p.add_argument("--min-x", type=float, default=None, help="Minimum allowed x (inclusive).")
    p.add_argument("--min-y", type=float, default=None, help="Minimum allowed y (inclusive).")
    p.add_argument("-o","--output", help="Output filename ('.txt' added automatically if omitted).")
    p.add_argument("--plot", action="store_true", help="Show plot of original/kept points and bound lines.")
    p.add_argument("--save-plot", metavar="PATH", help="Save plot to file (implies --plot).")
    p.add_argument("--json","-j", action="store_true",
                   help=f"Use {JSON_NAME} if present; create it if missing. CLI values override JSON.")
    p.add_argument("--decimals","-d", type=int, default=None,
                   help="Number of decimal places to write in the output file (default: 5).")
    return p

def parser_defaults_dict(p: argparse.ArgumentParser) -> Dict[str, Any]:
    # Pull argparse defaults into a dict we can merge with JSON/CLI
    defaults = {}
    for a in p._actions:
        if a.dest != "help":
            defaults[a.dest] = a.default
    return defaults

def load_json_config() -> Dict[str, Any] | None:
    path = Path(JSON_NAME)
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError("JSON root must be an object")
        return data
    except Exception as e:
        sys.exit(f"[error] Failed to read {JSON_NAME}: {e}")

def save_json_config(cfg: Dict[str, Any]) -> None:
    # Never store the json flag itself
    cfg = {k: v for k, v in cfg.items() if k != "json"}
    try:
        with Path(JSON_NAME).open("w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, sort_keys=True)
        print(f"[info] Wrote {JSON_NAME}")
    except Exception as e:
        sys.exit(f"[error] Failed to write {JSON_NAME}: {e}")

def merge_config(cli: Dict[str, Any], json_cfg: Dict[str, Any] | None, defaults: Dict[str, Any]) -> Dict[str, Any]:
    # Start with defaults, then JSON, then CLI (CLI wins on non-None/non-False values)
    cfg = dict(defaults)
    if json_cfg:
        cfg.update({k: v for k, v in json_cfg.items() if k in cfg})
    # Apply CLI overrides: treat None/False/"" as "not provided" except booleans explicitly True
    for k, v in cli.items():
        if k not in cfg:
            continue
        if isinstance(v, bool):
            if v:  # only override if True, to avoid stomping JSON with False-by-default
                cfg[k] = v
        else:
            if v is not None:
                cfg[k] = v
    return cfg

def require_fields(cfg: Dict[str, Any], fields: List[str]) -> None:
    missing = [f for f in fields if cfg.get(f) in (None, "")]
    if missing:
        sys.exit(f"[error] Missing required option(s): {', '.join(missing)} "
                 f"(provide via CLI or {JSON_NAME})")

def shift_points(points, offset_x: float = 0.0, offset_y: float = 0.0):
    return [(x - offset_x, y - offset_y) for (x, y) in points]

def main():
    p = base_parser()
    args = p.parse_args()
    defaults = parser_defaults_dict(p)
    cli_dict = vars(args)

    json_cfg = None
    if args.json:
        json_cfg = load_json_config()
        if json_cfg is None:
            # Create a starter JSON using CLI where provided, else defaults
            starter = dict(defaults)
            starter.update({k: v for k, v in cli_dict.items() if k in starter and v is not None})
            # Drop the json flag from the file
            starter.pop("json", None)
            save_json_config(starter)
            json_cfg = starter  # and use it immediately

    # Merge precedence: defaults < JSON < CLI
    merged = merge_config(cli_dict, json_cfg, defaults)

    # Enforce required fields after merging
    require_fields(merged, ["input_pattern", "max_x", "max_y"])

    # Sanity checks on bounds
    if merged["min_x"] is not None and merged["max_x"] is not None and merged["min_x"] > merged["max_x"]:
        sys.exit("[error] --min-x cannot be greater than --max-x")
    if merged["min_y"] is not None and merged["max_y"] is not None and merged["min_y"] > merged["max_y"]:
        sys.exit("[error] --min-y cannot be greater than --max-y")

    # Resolve paths / generate output name
    input_path = resolve_input_path(merged["input_pattern"])
    if merged.get("output"):
        output_path = Path(ensure_txt_suffix(merged["output"])).resolve()
    else:
        stem = input_path.stem
        name_parts = [stem]
        if merged["min_x"] is not None: name_parts.append(f"minx{merged['min_x']}")
        if merged["min_y"] is not None: name_parts.append(f"miny{merged['min_y']}")
        name_parts.append(f"maxx{merged['max_x']}")
        name_parts.append(f"maxy{merged['max_y']}")
        output_path = input_path.with_name("_".join(name_parts) + ".txt").resolve()

    # Read/filter/write
    header, all_points = read_points(input_path)
    kept_points = filter_points(
        all_points,
        max_x=float(merged["max_x"]),
        max_y=float(merged["max_y"]),
        min_x=(None if merged["min_x"] is None else float(merged["min_x"])),
        min_y=(None if merged["min_y"] is None else float(merged["min_y"]))
    )
    # write_filtered(header, kept_points, output_path)
    # shift_x = args.min_x if args.min_x is not None else 0.0
    # shift_y = args.min_y if args.min_y is not None else 0.0
    shift_x = float(merged["min_x"]) if merged["min_x"] is not None else 0.0
    shift_y = float(merged["min_y"]) if merged["min_y"] is not None else 0.0
    out_points = shift_points(kept_points, shift_x, shift_y)
    # write_filtered(header, out_points, output_path)
    write_filtered(header, out_points, output_path, decimals=merged.get("decimals"))

    kept = len(kept_points)
    removed = len(all_points) - kept

    # Print summary + count of centers in output (explicit)
    print(f"[info] Input:  {input_path}")
    print(f"[info] Output: {output_path}")
    print(f"[info] Kept {kept} rows; removed {removed} rows.")
    print(f"[info] Centers in output: {kept}")
    if shift_x != 0.0 or shift_y != 0.0:
        print(f"[info] Output coordinates were shifted by (-{shift_x}, -{shift_y}).")

    # Plot if requested (save-plot implies plot)
    if merged.get("plot") or merged.get("save_plot"):
        title = f"{input_path.name} → {output_path.name}"
        plot_points(
            all_points=all_points,
            kept_points=kept_points,
            max_x=float(merged["max_x"]),
            max_y=float(merged["max_y"]),
            min_x=(None if merged["min_x"] is None else float(merged["min_x"])),
            min_y=(None if merged["min_y"] is None else float(merged["min_y"])),
            title=title,
            save_path=merged.get("save_plot")
        )

if __name__ == "__main__":
    main()
