#!/usr/bin/env python3
import argparse
import numpy as np
import matplotlib.pyplot as plt

from ExodusBasics import ExodusBasics
import myInput
import PACKAGE_MP_Linear as smooth


def map_to_grid(xc, yc, val, *, tol):
    """
    Same idea as vector_inclination.map_to_grid: quantize centers to a grid.
    """
    xc = np.asarray(xc, float)
    yc = np.asarray(yc, float)
    val = np.asarray(val)

    kx = np.round(xc / tol) * tol
    ky = np.round(yc / tol) * tol

    ux = np.unique(kx)
    uy = np.unique(ky)

    x_centers = ux
    y_centers = uy

    # IMPORTANT: P0 is indexed [i,j] = [y_index, x_index] in image convention
    P0 = np.full((len(uy), len(ux)), np.nan, dtype=float)

    x_index = {v: j for j, v in enumerate(ux)}
    y_index = {v: i for i, v in enumerate(uy)}

    for xk, yk, v in zip(kx, ky, val):
        i = y_index[yk]
        j = x_index[xk]
        P0[i, j] = v

    return P0, x_centers, y_centers


def auto_tol_from_centers(xc, yc):
    """
    Pick a reasonable quantization tolerance based on median spacing.
    """
    xc = np.sort(np.unique(np.asarray(xc, float)))
    yc = np.sort(np.unique(np.asarray(yc, float)))
    dx = np.median(np.diff(xc)) if len(xc) > 1 else 1.0
    dy = np.median(np.diff(yc)) if len(yc) > 1 else 1.0
    h = min(dx, dy)
    # Quantize much finer than spacing, but not absurdly small
    return max(h * 1e-3, 1e-12), dx, dy


def gb_mask_exact(P):
    """
    VECTOR-style boundary: any neighbor differs, with periodic BCs.
    P is 2D.
    """
    ny, nx = P.shape
    out = np.zeros((ny, nx), dtype=bool)
    for i in range(ny):
        for j in range(nx):
            ip, im, jp, jm = myInput.periodic_bc(ny, nx, i, j)  # note: we pass (ny,nx)
            c = P[i, j]
            if (P[ip, j] - c) != 0 or (P[im, j] - c) != 0 or (P[i, jp] - c) != 0 or (P[i, jm] - c) != 0:
                out[i, j] = True
    return out


def gb_mask_labels_int(L):
    """
    Robust boundary: same logic but on integer labels.
    """
    ny, nx = L.shape
    out = np.zeros((ny, nx), dtype=bool)
    for i in range(ny):
        for j in range(nx):
            ip, im, jp, jm = myInput.periodic_bc(ny, nx, i, j)
            c = L[i, j]
            if (L[ip, j] != c) or (L[im, j] != c) or (L[i, jp] != c) or (L[i, jm] != c):
                out[i, j] = True
    return out


def plot_overlay(P0, mask, out_png, title):
    plt.figure(figsize=(10, 8))
    plt.imshow(P0, origin="lower", aspect="equal")
    ii, jj = np.where(mask)
    # Correct mapping for imshow: x=col=j, y=row=i
    plt.scatter(jj, ii, s=0.2)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exo", help="Path to .e Exodus file")
    ap.add_argument("--eb", type=int, default=1, help="Element block id (default 1)")
    ap.add_argument("--var", default="unique_grains", help="Element variable name (default unique_grains)")
    ap.add_argument("--step", type=int, default=0, help="Timestep index (0-based)")
    ap.add_argument("--tol", type=float, default=None, help="Quantization tol for map_to_grid (default auto)")
    ap.add_argument("--round_labels", action="store_true", help="Round P0 to nearest int for tests")
    ap.add_argument("--stem", default="GBDBG", help="Output file stem")
    ap.add_argument("--run_vector_get_all_gb", action="store_true",
                    help="Instantiate VECTOR linear_class and call get_all_gb_list (no smoothing)")
    args = ap.parse_args()

    with ExodusBasics(args.exo) as exo:
        xc, yc = exo.element_centers_xy(method="mean")
        ug = exo.elem_var_at_step(args.var, step=args.step, eb=args.eb)

    if args.tol is None:
        tol, dx, dy = auto_tol_from_centers(xc, yc)
    else:
        tol = args.tol
        dx = dy = np.nan

    P0, x_centers, y_centers = map_to_grid(xc, yc, ug, tol=tol)

    ny, nx = P0.shape
    n_nan = np.isnan(P0).sum()
    print("\n=== GRID / MAPPING ===")
    print(f"P0 shape (ny,nx) = {P0.shape}")
    print(f"tol = {tol:g} (auto dx~{dx}, dy~{dy})")
    print(f"NaNs in P0 = {n_nan} ({n_nan / P0.size:.3%})")

    finite = P0[np.isfinite(P0)]
    if finite.size == 0:
        raise SystemExit("P0 is all NaNs — mapping failed.")

    nearest = np.rint(finite)
    max_dev = np.max(np.abs(finite - nearest))
    frac_nonint = np.mean(np.abs(finite - nearest) > 1e-6)

    print("\n=== LABEL HEALTH ===")
    print(f"finite cells = {finite.size}")
    print(f"max |label - round(label)| = {max_dev:g}")
    print(f"fraction non-integer (>1e-6 from nearest int) = {frac_nonint:.3%}")
    print(f"min label = {finite.min():g}, max label = {finite.max():g}")

    # VECTOR-style mask on raw floats
    mask_exact = gb_mask_exact(P0)

    # Rounded labels mask
    L = np.rint(P0).astype(int)
    mask_int = gb_mask_labels_int(L)

    print("\n=== GB MASK STATS ===")
    print(f"mask_exact boundary fraction = {mask_exact.mean():.3%}")
    print(f"mask_int   boundary fraction = {mask_int.mean():.3%}")

    # Check contiguity
    u = np.unique(L[np.isfinite(P0)])
    u_sorted = np.sort(u)
    contiguous = np.all(u_sorted == np.arange(u_sorted[0], u_sorted[0] + len(u_sorted)))
    print("\n=== ID CONTIGUITY ===")
    print(f"unique labels count = {len(u_sorted)}")
    print(f"labels start at {u_sorted[0]}, end at {u_sorted[-1]}")
    print(f"contiguous sequence? {contiguous}")
    if u_sorted[0] != 1:
        print("WARNING: VECTOR get_all_gb_list assumes labels start at 1 (it indexes int(label-1)).")
    if not contiguous:
        print("WARNING: labels are not contiguous; VECTOR per-grain binning can misbehave.")

    # Plots (correct overlay)
    plot_overlay(P0, mask_exact, f"{args.stem}_overlay_mask_exact.png",
                 f"Overlay: VECTOR-style exact (!=0), tol={tol:g}")
    plot_overlay(P0, mask_int, f"{args.stem}_overlay_mask_int.png",
                 f"Overlay: integer labels (robust), tol={tol:g}")

    # Optional: call VECTOR's get_all_gb_list directly to compare counts
    if args.run_vector_get_all_gb:
        ng = int(np.nanmax(L))
        R = np.zeros((ny, nx, 2))
        cores = 1
        loop_times = 1
        sc = smooth.linear_class(ny, nx, ng, cores, loop_times, L.astype(float), R, verification_system=False)
        sites_by_grain = sc.get_all_gb_list()
        sites_total = sum(len(s) for s in sites_by_grain)
        print("\n=== VECTOR get_all_gb_list() ===")
        print(f"ng used = {ng}")
        print(f"total sites (sum over grains) = {sites_total}")

        # Plot those sites with correct mapping
        mask_vec = np.zeros((ny, nx), dtype=bool)
        for gsites in sites_by_grain:
            for (i, j) in gsites:
                mask_vec[int(i), int(j)] = True
        plot_overlay(P0, mask_vec, f"{args.stem}_overlay_mask_vector_get_all_gb.png",
                     "Overlay: VECTOR get_all_gb_list() sites")

    print("\nWrote:")
    print(f"  {args.stem}_overlay_mask_exact.png")
    print(f"  {args.stem}_overlay_mask_int.png")
    if args.run_vector_get_all_gb:
        print(f"  {args.stem}_overlay_mask_vector_get_all_gb.png")


if __name__ == "__main__":
    main()
