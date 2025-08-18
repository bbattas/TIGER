#!/usr/bin/env python3
import argparse
import glob
import json
from pathlib import Path
import sys
from typing import List, Tuple, Dict, Any
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class default_vals:
    L = 1.15382e-6
    gbe = 4.60748e6
    kappa = 2.07337e7
    m = 4.5e6


def base_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Check the parameter space for varying levels of anisotropy."
    )
    p.add_argument("--gbe", type=float, default=default_vals.gbe, help="Iso GB Energy.")
    p.add_argument("--kappa", type=float, default=default_vals.kappa, help="Kappa.")
    p.add_argument("--m","-m", type=float, default=default_vals.m, help="FE constant m.")
    # Make all options non-required here; we'll enforce presence after merging JSON+CLI.
    # p.add_argument("--input_pattern","-i", help="Substring or glob for the input file.")
    # p.add_argument("--max-x","-x", type=float, help="Maximum allowed x (inclusive).")
    # p.add_argument("--max-y","-y", type=float, help="Maximum allowed y (inclusive).")
    # p.add_argument("--min-x", type=float, default=None, help="Minimum allowed x (inclusive).")
    # p.add_argument("--min-y", type=float, default=None, help="Minimum allowed y (inclusive).")
    # p.add_argument("-o","--output", help="Output filename ('.txt' added automatically if omitted).")
    p.add_argument("--plot", action="store_true", help="Show plot of original/kept points and bound lines.")
    p.add_argument("--save-plot", metavar="PATH", help="Save plot to file (implies --plot).")
    # p.add_argument("--json","-j", action="store_true",
    #                help=f"Use {JSON_NAME} if present; create it if missing. CLI values override JSON.")
    # p.add_argument("--decimals","-d", type=int, default=None,
    #                help="Number of decimal places to write in the output file (default: 5).")
    return p

# def parser_defaults_dict(p: argparse.ArgumentParser) -> Dict[str, Any]:
#     # Pull argparse defaults into a dict we can merge with JSON/CLI
#     defaults = {}
#     for a in p._actions:
#         if a.dest != "help":
#             defaults[a.dest] = a.default
#     return defaults

# def load_json_config() -> Dict[str, Any] | None:
#     path = Path(JSON_NAME)
#     if not path.exists():
#         return None
#     try:
#         with path.open("r", encoding="utf-8") as f:
#             data = json.load(f)
#         if not isinstance(data, dict):
#             raise ValueError("JSON root must be an object")
#         return data
#     except Exception as e:
#         sys.exit(f"[error] Failed to read {JSON_NAME}: {e}")

# def save_json_config(cfg: Dict[str, Any]) -> None:
#     # Never store the json flag itself
#     cfg = {k: v for k, v in cfg.items() if k != "json"}
#     try:
#         with Path(JSON_NAME).open("w", encoding="utf-8") as f:
#             json.dump(cfg, f, indent=2, sort_keys=True)
#         print(f"[info] Wrote {JSON_NAME}")
#     except Exception as e:
#         sys.exit(f"[error] Failed to write {JSON_NAME}: {e}")

# def merge_config(cli: Dict[str, Any], json_cfg: Dict[str, Any] | None, defaults: Dict[str, Any]) -> Dict[str, Any]:
#     # Start with defaults, then JSON, then CLI (CLI wins on non-None/non-False values)
#     cfg = dict(defaults)
#     if json_cfg:
#         cfg.update({k: v for k, v in json_cfg.items() if k in cfg})
#     # Apply CLI overrides: treat None/False/"" as "not provided" except booleans explicitly True
#     for k, v in cli.items():
#         if k not in cfg:
#             continue
#         if isinstance(v, bool):
#             if v:  # only override if True, to avoid stomping JSON with False-by-default
#                 cfg[k] = v
#         else:
#             if v is not None:
#                 cfg[k] = v
#     return cfg

# def require_fields(cfg: Dict[str, Any], fields: List[str]) -> None:
#     missing = [f for f in fields if cfg.get(f) in (None, "")]
#     if missing:
#         sys.exit(f"[error] Missing required option(s): {', '.join(missing)} "
#                  f"(provide via CLI or {JSON_NAME})")

def poly_g(g2):
    a1 = -3.0944
    a2 = -1.8169
    a3 = 10.323
    a4 = -8.1819
    a5 = 2.0033
    poly = (((a1 * g2 + a2) * g2 + a3) * g2 + a4) * g2 + a5
    return poly

def main():
    p = base_parser()
    args = p.parse_args()

    a_range = np.linspace(0,1,100)
    anis = np.stack([1 - a_range, 1 + a_range], axis=-1)
    MIN, MAX = 0, 1

    gbe = args.gbe * anis # (n,2)
    g = gbe / (np.sqrt(args.kappa * args.m))
    g2 = g * g
    polyg = poly_g(g2)
    gamma = 1 / polyg
    # gamma_limit_min = 0.5
    # gamma_limit_max = 40


    df = pd.DataFrame({
        "a": a_range,
        "gbe_min": gbe[:, MIN], "gbe_max": gbe[:, MAX],
        "g_min": g[:, MIN], "g_max": g[:, MAX],
        "g2_min": g2[:, MIN], "g2_max": g2[:, MAX],
        "polyg_min": polyg[:, MIN], "polyg_max": polyg[:, MAX],
        "gamma_min": gamma[:, MIN], "gamma_max": gamma[:, MAX],
        # "k_min":   k_band[:, MIN],   "k_max":   k_band[:, MAX],
        # "sig_min": sigma_band[:, MIN],"sig_max": sigma_band[:, MAX],
    })

    # fig, ax = plt.subplots()
    # ax.fill_between(df["a"], df["gbe_min"], df["gbe_max"], alpha=0.3, label="GBE band")
    # ax.plot(df["a"], 0.5*(df["gbe_min"]+df["gbe_max"]), label="GBE center")
    # ax.set_xlabel("Anisotropy Magnitude +/-")
    # ax.set_ylabel("GBE")
    # ax.legend()
    # plt.show()
    fig, ax = plt.subplots()
    ax.fill_between(df["a"], df["gamma_min"], df["gamma_max"], alpha=0.3, label="Gamma band")
    ax.plot(df["a"], 0.5*(df["gamma_min"]+df["gamma_max"]), label="Gamma center")

    ax.set_xlabel("Anisotropy Magnitude +/-")
    ax.set_ylabel("Gamma")
    ax.legend()
    plt.show()




if __name__ == "__main__":
    main()
