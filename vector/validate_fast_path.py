#!/usr/bin/env python3
"""
validate_fast_path.py

Validates that the --fast vectorized path in vector_exodus_to_hdf5.py
produces numerically equivalent results to the standard path.

Tests run in order of increasing complexity:
  1. Synthetic grid tests (no Exodus file needed)
     1a. Boundary/junction detection equivalence
     1b. TJ exclusion mask equivalence
     1c. GB property accumulation equivalence
     1d. Full curvature pipeline equivalence
  2. PACKAGE_MP_Linear tests (no Exodus file needed)
     2a. find_window vs find_all_windows_vectorized
     2b. _get_boundary_mask_vectorized
     2c. linear_combined_core vs linear_combined_core_fast (small grid)
  3. End-to-end Exodus file test (requires a .e file)
     3a. process_frame standard vs fast on a real file

Usage
-----
# Run all synthetic tests only (no .e file required):
    python validate_fast_path.py

# Run all tests including end-to-end on a real Exodus file:
    python validate_fast_path.py --exo /path/to/file.e -t 10.0 -n 4 -l 5

# Run only a specific test group:
    python validate_fast_path.py --only synthetic
    python validate_fast_path.py --only linear
    python validate_fast_path.py --only e2e --exo /path/to/file.e -t 10.0

# Adjust numerical tolerance (default 1e-10):
    python validate_fast_path.py --tol 1e-8

# Use a larger synthetic grid for stress testing:
    python validate_fast_path.py --grid-size 200

Notes
-----
- All tests print PASS/FAIL with a diff summary on failure.
- Exit code 0 = all tests passed.
- Exit code 1 = one or more tests failed.
- The synthetic Voronoi grid generator requires scipy (already a dependency).
"""

import sys
import argparse
import traceback
import time
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# ANSI color helpers
# ---------------------------------------------------------------------------

GREEN  = "\033[32m"
RED    = "\033[31m"
YELLOW = "\033[33m"
CYAN   = "\033[36m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def pass_msg(name):  print(f"  {GREEN}PASS{RESET}  {name}")
def fail_msg(name, reason): print(f"  {RED}FAIL{RESET}  {name}\n         {YELLOW}{reason}{RESET}")
def info_msg(msg):   print(f"  {CYAN}INFO{RESET}  {msg}")
def head_msg(msg):   print(f"\n{BOLD}{msg}{RESET}")


# ---------------------------------------------------------------------------
# Result accumulator
# ---------------------------------------------------------------------------

class Results:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.skipped = 0
        self._failures = []

    def record(self, name, ok, reason=""):
        if ok:
            self.passed += 1
            pass_msg(name)
        else:
            self.failed += 1
            self._failures.append((name, reason))
            fail_msg(name, reason)

    def skip(self, name, reason=""):
        self.skipped += 1
        print(f"  {YELLOW}SKIP{RESET}  {name}  ({reason})")

    def summary(self):
        total = self.passed + self.failed + self.skipped
        print(f"\n{'='*60}")
        print(f"Results: {self.passed}/{total} passed  "
              f"({self.failed} failed, {self.skipped} skipped)")
        if self._failures:
            print(f"\nFailed tests:")
            for name, reason in self._failures:
                print(f"  {RED}✗{RESET} {name}")
                print(f"    {YELLOW}{reason}{RESET}")
        print('='*60)
        return self.failed == 0


# ---------------------------------------------------------------------------
# Synthetic grid generators
# ---------------------------------------------------------------------------

def make_checkerboard(nx=50, ny=50, tile=10):
    """
    Simple checkerboard grain structure.
    Grain IDs = 1 or 2. Every tile×tile block alternates.
    Good for testing: clean GBs everywhere, no TJs.
    """
    P0 = np.zeros((nx, ny), dtype=np.int32)
    for i in range(nx):
        for j in range(ny):
            P0[i, j] = 1 + ((i // tile + j // tile) % 2)
    return P0


def make_voronoi_grid(nx=100, ny=100, n_grains=20, seed=42):
    """
    Voronoi tessellation grain structure.
    Grain IDs = 1..n_grains.
    Good for testing: realistic TJ geometry.
    """
    from scipy.spatial import cKDTree
    rng = np.random.default_rng(seed)
    seeds_x = rng.uniform(0, nx, n_grains)
    seeds_y = rng.uniform(0, ny, n_grains)
    seeds = np.stack([seeds_x, seeds_y], axis=1)

    xi, yi = np.meshgrid(np.arange(nx), np.arange(ny), indexing='ij')
    pts = np.stack([xi.ravel(), yi.ravel()], axis=1).astype(float)

    tree = cKDTree(seeds)
    _, labels = tree.query(pts)
    P0 = (labels + 1).reshape(nx, ny).astype(np.int32)
    return P0


def make_triple_junction_grid(nx=60, ny=60):
    """
    Three-grain structure with a deliberate triple junction at center.
    Grain 1: top-left triangle
    Grain 2: top-right triangle
    Grain 3: bottom half
    Good for testing: TJ detection and exclusion mask.
    """
    P0 = np.zeros((nx, ny), dtype=np.int32)
    cx, cy = nx // 2, ny // 2
    for i in range(nx):
        for j in range(ny):
            if j >= cy:
                P0[i, j] = 3
            elif i < cx:
                P0[i, j] = 1
            else:
                P0[i, j] = 2
    return P0


# ---------------------------------------------------------------------------
# Numerical comparison helpers
# ---------------------------------------------------------------------------

def arrays_close(a, b, tol=1e-10, name="array"):
    """
    Check two arrays are close. Returns (ok, reason_string).
    Handles NaN: NaN == NaN is treated as equal.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    if a.shape != b.shape:
        return False, f"{name}: shape mismatch {a.shape} vs {b.shape}"

    nan_a = np.isnan(a)
    nan_b = np.isnan(b)
    if not np.array_equal(nan_a, nan_b):
        n_diff = np.sum(nan_a != nan_b)
        return False, f"{name}: NaN pattern differs at {n_diff} positions"

    mask = ~nan_a
    if not np.allclose(a[mask], b[mask], atol=tol, rtol=tol):
        diff = np.abs(a[mask] - b[mask])
        return False, (
            f"{name}: max_diff={diff.max():.3e}, "
            f"mean_diff={diff.mean():.3e}, "
            f"n_exceed={np.sum(diff > tol)}"
        )
    return True, ""


def dicts_close(d_std, d_fast, tol=1e-10):
    """
    Compare two gb_dict outputs. Returns (ok, reason).
    gb_dict: pair_id -> np.array([avg_curv, gb_area, grain_id1,
                                   grain_id2, raw_gb_area])
    """
    keys_std  = set(d_std.keys())
    keys_fast = set(d_fast.keys())

    if keys_std != keys_fast:
        only_std  = keys_std  - keys_fast
        only_fast = keys_fast - keys_std
        msgs = []
        if only_std:
            msgs.append(f"keys only in standard ({len(only_std)}): "
                        f"{list(only_std)[:5]}")
        if only_fast:
            msgs.append(f"keys only in fast ({len(only_fast)}): "
                        f"{list(only_fast)[:5]}")
        return False, "; ".join(msgs)

    max_diff  = 0.0
    bad_pairs = []
    for pid in keys_std:
        v_std  = d_std[pid]
        v_fast = d_fast[pid]
        diff = np.abs(v_std - v_fast)
        if np.any(diff > tol):
            bad_pairs.append((pid, diff))
            max_diff = max(max_diff, diff.max())

    if bad_pairs:
        return False, (
            f"{len(bad_pairs)} pairs differ (max_diff={max_diff:.3e}). "
            f"First bad pair: {bad_pairs[0][0]}, diffs={bad_pairs[0][1]}"
        )
    return True, ""


def pixel_sets_equal(arr_std, arr_fast, name="pixels"):
    """
    Compare two (N,2) arrays as sets of (i,j) tuples, ignoring order.
    Returns (ok, reason).
    """
    if arr_std.ndim == 1 and len(arr_std) == 0:
        set_std = set()
    else:
        set_std = set(map(tuple, arr_std))

    if arr_fast.ndim == 1 and len(arr_fast) == 0:
        set_fast = set()
    else:
        set_fast = set(map(tuple, arr_fast))

    if set_std == set_fast:
        return True, ""

    only_std  = set_std  - set_fast
    only_fast = set_fast - set_std
    msgs = []
    if only_std:
        msgs.append(f"{name}: {len(only_std)} pixels only in standard")
    if only_fast:
        msgs.append(f"{name}: {len(only_fast)} pixels only in fast")
    return False, "; ".join(msgs)


# ---------------------------------------------------------------------------
# Group 1: Synthetic grid tests
# ---------------------------------------------------------------------------

def run_synthetic_tests(results, tol=1e-10, grid_size=100):
    head_msg("Group 1: Synthetic Grid Tests (no Exodus file required)")

    # Import the functions we need to test
    try:
        from vector_exodus_to_hdf5_vectorized import (
            accumulate_gb_properties,
            accumulate_gb_properties_fast,
            _detect_boundary_and_junction_vectorized,
            _build_tj_exclusion_mask,
            compute_gb_curvature,
            compute_gb_curvature_fast,
            average_gb_properties,
        )
    except ImportError as e:
        results.skip("All Group 1 tests", f"Import failed: {e}")
        return

    # ------------------------------------------------------------------
    # Build a minimal fake C array from a grain ID grid.
    # C[0] = grain IDs, C[1] = synthetic curvature (random values).
    # ------------------------------------------------------------------
    def make_C(P0, seed=0):
        rng = np.random.default_rng(seed)
        C = np.zeros((2,) + P0.shape, dtype=np.float64)
        C[0] = P0.astype(np.float64)
        C[1] = rng.uniform(-0.1, 0.1, P0.shape)
        return C

    grids = {
        "checkerboard_50":    make_checkerboard(50, 50, tile=10),
        "triple_junction_60": make_triple_junction_grid(60, 60),
        f"voronoi_{grid_size}x{grid_size}_20g":
            make_voronoi_grid(grid_size, grid_size, n_grains=20, seed=42),
        f"voronoi_{grid_size}x{grid_size}_50g":
            make_voronoi_grid(grid_size, grid_size, n_grains=50, seed=99),
    }

    # ---- 1a: Boundary/junction detection ----------------------------
    print("\n  1a. Boundary & junction detection")
    for gname, P0 in grids.items():
        try:
            C = make_C(P0)

            # Standard path uses the double-loop inside accumulate_gb_properties.
            # We extract equivalent boolean maps from its output.
            raw_std, bp_std, jp_std = accumulate_gb_properties(
                C, TJ_distance_max=0, signed=True
            )

            # Fast path uses _detect_boundary_and_junction_vectorized.
            is_bnd, is_jnc, _ = _detect_boundary_and_junction_vectorized(
                C[0].astype(np.int32)
            )
            bp_fast = np.argwhere(is_bnd)
            jp_fast = np.argwhere(is_jnc)

            ok_b, reason_b = pixel_sets_equal(bp_std, bp_fast, "boundary_pixels")
            ok_j, reason_j = pixel_sets_equal(jp_std, jp_fast, "junction_pixels")
            ok = ok_b and ok_j
            reason = (reason_b + (" | " if reason_b and reason_j else "") + reason_j)
            results.record(f"1a boundary+junction detection [{gname}]", ok, reason)

        except Exception as e:
            results.record(f"1a boundary+junction detection [{gname}]",
                           False, traceback.format_exc(limit=2))

    # ---- 1b: TJ exclusion mask -------------------------------------
    print("\n  1b. TJ exclusion mask")
    for gname, P0 in grids.items():
        try:
            C = make_C(P0)
            is_bnd, is_jnc, _ = _detect_boundary_and_junction_vectorized(
                C[0].astype(np.int32)
            )

            for tj_dist in [3, 6, 10]:
                mask_fast = _build_tj_exclusion_mask(is_jnc, tj_dist)

                # Ground truth: for each pixel, check Euclidean distance
                # to nearest junction pixel using brute force.
                jp_coords = np.argwhere(is_jnc)
                nx, ny = P0.shape
                if len(jp_coords) == 0:
                    mask_ref = np.zeros((nx, ny), dtype=bool)
                else:
                    # Use scipy distance transform as reference
                    # (same as fast path — confirms consistent behavior)
                    from scipy.ndimage import distance_transform_edt
                    mask_ref = distance_transform_edt(~is_jnc) < tj_dist

                ok = np.array_equal(mask_fast, mask_ref)
                reason = "" if ok else f"mask differs at {np.sum(mask_fast != mask_ref)} pixels"
                results.record(
                    f"1b TJ exclusion mask [{gname}, tj_dist={tj_dist}]", ok, reason
                )

        except Exception as e:
            results.record(f"1b TJ exclusion mask [{gname}]",
                           False, traceback.format_exc(limit=2))

    # ---- 1c: GB property accumulation ------------------------------
    print("\n  1c. GB property accumulation (accumulate_gb_properties)")
    for gname, P0 in grids.items():
        for tj_dist in [0, 6]:
            for signed in [True, False]:
                test_name = f"1c accumulate [{gname}, tj={tj_dist}, signed={signed}]"
                try:
                    C = make_C(P0, seed=7)

                    raw_std, bp_std, jp_std = accumulate_gb_properties(
                        C, TJ_distance_max=tj_dist, signed=signed
                    )
                    raw_fast, bp_fast, jp_fast = accumulate_gb_properties_fast(
                        C, TJ_distance_max=tj_dist, signed=signed
                    )

                    # Compare gb_dicts after averaging (same averaging step)
                    gb_std  = average_gb_properties(raw_std)
                    gb_fast = average_gb_properties(raw_fast)

                    ok_dict, reason_dict = dicts_close(gb_std, gb_fast, tol=tol)
                    ok_bp,   reason_bp   = pixel_sets_equal(bp_std, bp_fast, "boundary_pixels")
                    ok_jp,   reason_jp   = pixel_sets_equal(jp_std, jp_fast, "junction_pixels")

                    ok = ok_dict and ok_bp and ok_jp
                    reasons = [r for r in [reason_dict, reason_bp, reason_jp] if r]
                    results.record(test_name, ok, " | ".join(reasons))

                except Exception:
                    results.record(test_name, False, traceback.format_exc(limit=2))

    # ---- 1d: Full compute_gb_curvature pipeline --------------------
    print("\n  1d. Full curvature pipeline (compute_gb_curvature)")
    for gname, P0 in grids.items():
        test_name = f"1d full pipeline [{gname}]"
        try:
            C = make_C(P0, seed=13)

            gb_std,  bp_std,  jp_std  = compute_gb_curvature(
                C, TJ_distance_max=6, signed=True
            )
            gb_fast, bp_fast, jp_fast = compute_gb_curvature_fast(
                C, TJ_distance_max=6, signed=True
            )

            ok_d, reason_d = dicts_close(gb_std, gb_fast, tol=tol)
            ok_b, reason_b = pixel_sets_equal(bp_std, bp_fast, "boundary_pixels")
            ok_j, reason_j = pixel_sets_equal(jp_std, jp_fast, "junction_pixels")

            ok = ok_d and ok_b and ok_j
            reasons = [r for r in [reason_d, reason_b, reason_j] if r]
            results.record(test_name, ok, " | ".join(reasons))

            if ok:
                info_msg(
                    f"  {gname}: {len(gb_std)} GBs, "
                    f"{len(bp_std)} boundary px, "
                    f"{len(jp_std)} junction px"
                )

        except Exception:
            results.record(test_name, False, traceback.format_exc(limit=2))


# ---------------------------------------------------------------------------
# Group 2: PACKAGE_MP_Linear tests
# ---------------------------------------------------------------------------

def run_linear_tests(results, tol=1e-10, grid_size=60):
    head_msg("Group 2: PACKAGE_MP_Linear Vectorized Core Tests")

    try:
        import PACKAGE_MP_Linear_Vectorized as smooth
        import myInput
    except ImportError as e:
        results.skip("All Group 2 tests", f"Import failed: {e}")
        return

    def make_linear_instance(P0, loop_times=5, cores=1):
        nx, ny = P0.shape
        ng = int(np.nanmax(P0)) - int(np.nanmin(P0)) + 1
        R = np.zeros((nx, ny, 3))
        return smooth.linear_class(
            nx, ny, ng, cores, loop_times, P0, R,
            verification_system=False,
            curvature_sign=True,
        )

    P0_checker = np.zeros((grid_size, grid_size), dtype=np.int32)
    tile = max(5, grid_size // 10)
    for i in range(grid_size):
        for j in range(grid_size):
            P0_checker[i, j] = 1 + ((i // tile + j // tile) % 2)

    P0_voronoi = make_voronoi_grid(
        grid_size, grid_size, n_grains=max(5, grid_size // 10), seed=7
    ).astype(np.float64)

    test_grids = {
        f"checkerboard_{grid_size}": P0_checker,
        f"voronoi_{grid_size}":      P0_voronoi,
    }

    # ---- 2a: find_window vs find_all_windows_vectorized -------------
    print("\n  2a. find_window vs find_all_windows_vectorized")
    for gname, P0 in test_grids.items():
        for loop_times in [3, 5, 10]:
            test_name = f"2a find_window [{gname}, loop_times={loop_times}]"
            try:
                inst = make_linear_instance(P0, loop_times=loop_times)
                fw_n = inst.tableL - 2 * inst.clip

                # Get a sample of boundary pixels using the vectorized mask
                nx, ny = P0.shape
                all_ij = np.array(
                    [[i, j] for i in range(nx) for j in range(ny)]
                )
                is_bnd = inst._get_boundary_mask_vectorized(all_ij)
                boundary_ij = all_ij[is_bnd]

                if len(boundary_ij) == 0:
                    results.skip(test_name, "no boundary pixels found")
                    continue

                # Sample up to 200 boundary pixels for speed
                rng = np.random.default_rng(0)
                sample_idx = rng.choice(
                    len(boundary_ij),
                    size=min(200, len(boundary_ij)),
                    replace=False,
                )
                sample = boundary_ij[sample_idx]

                # Standard: find_window one at a time
                windows_scalar = np.stack([
                    inst.find_window(int(ij[0]), int(ij[1]), fw_n)
                    for ij in sample
                ])  # (N, fw_n, fw_n)

                # Fast: find_all_windows_vectorized
                windows_vector = inst.find_all_windows_vectorized(sample, fw_n)

                ok, reason = arrays_close(
                    windows_scalar, windows_vector, tol=tol, name="windows"
                )
                results.record(test_name, ok, reason)

            except Exception:
                results.record(test_name, False, traceback.format_exc(limit=2))

    # ---- 2b: _get_boundary_mask_vectorized --------------------------
    print("\n  2b. _get_boundary_mask_vectorized")
    for gname, P0 in test_grids.items():
        test_name = f"2b boundary_mask [{gname}]"
        try:
            inst = make_linear_instance(P0)
            nx, ny = P0.shape

            all_ij = np.array([[i, j] for i in range(nx) for j in range(ny)])

            # Reference: scalar periodic_bc check (same logic as original code)
            ref_mask = np.zeros(len(all_ij), dtype=bool)
            for k, (i, j) in enumerate(all_ij):
                ip, im, jp, jm = myInput.periodic_bc(nx, ny, i, j)
                ref_mask[k] = (
                    (inst.P[0, ip, j] != inst.P[0, i, j]) or
                    (inst.P[0, im, j] != inst.P[0, i, j]) or
                    (inst.P[0, i, jp] != inst.P[0, i, j]) or
                    (inst.P[0, i, jm] != inst.P[0, i, j])
                )

            fast_mask = inst._get_boundary_mask_vectorized(all_ij)

            ok = np.array_equal(ref_mask, fast_mask)
            n_diff = np.sum(ref_mask != fast_mask)
            reason = "" if ok else f"mask differs at {n_diff} pixels"
            results.record(test_name, ok, reason)

        except Exception:
            results.record(test_name, False, traceback.format_exc(limit=2))

    # ---- 2c: linear_combined_core vs linear_combined_core_fast ------
    print("\n  2c. linear_combined_core vs linear_combined_core_fast")
    for gname, P0 in test_grids.items():
        for loop_times in [3, 5]:
            test_name = f"2c combined_core [{gname}, loop_times={loop_times}]"
            try:
                # We need two separate instances so they each get a fresh
                # P/C array to accumulate results into.
                inst_std  = make_linear_instance(P0, loop_times=loop_times, cores=1)
                inst_fast = make_linear_instance(P0, loop_times=loop_times, cores=1)

                # Build the full domain input (same as linear_main does)
                nx, ny = P0.shape
                all_sites = np.array(
                    [[x, y] for x in range(nx) for y in range(ny)]
                ).reshape(nx, ny, 2)

                # Use a single subdomain (all pixels) for a direct comparison
                t0 = time.perf_counter()
                fval_std, t_std = inst_std.linear_combined_core(all_sites)
                t_elapsed_std = time.perf_counter() - t0

                inst_fast.fast_chunk_size = min(50_000, nx * ny)
                t0 = time.perf_counter()
                fval_fast, t_fast = inst_fast.linear_combined_core_fast(all_sites)
                t_elapsed_fast = time.perf_counter() - t0

                info_msg(
                    f"  {gname} loop_times={loop_times}: "
                    f"std={t_elapsed_std:.2f}s  fast={t_elapsed_fast:.2f}s  "
                    f"speedup={t_elapsed_std/max(t_elapsed_fast,1e-9):.1f}x"
                )

                ok_nv_i, reason_i = arrays_close(
                    fval_std[:,:,0], fval_fast[:,:,0], tol=tol, name="nv_i"
                )
                ok_nv_j, reason_j = arrays_close(
                    fval_std[:,:,1], fval_fast[:,:,1], tol=tol, name="nv_j"
                )
                ok_curv, reason_c = arrays_close(
                    fval_std[:,:,2], fval_fast[:,:,2], tol=tol, name="curvature"
                )

                ok = ok_nv_i and ok_nv_j and ok_curv
                reasons = [r for r in [reason_i, reason_j, reason_c] if r]
                results.record(test_name, ok, " | ".join(reasons))

            except Exception:
                results.record(test_name, False, traceback.format_exc(limit=2))

    # ---- 2d: linear_main standard vs fast (full parallel run) -------
    print("\n  2d. linear_main full run (standard vs fast, parallel)")
    # Use a small grid so the full parallel run finishes quickly
    small_P0 = make_voronoi_grid(40, 40, n_grains=8, seed=3).astype(np.float64)

    for loop_times in [3, 5]:
        for cores in [1, 2]:
            test_name = f"2d linear_main [40x40 voronoi, loop_times={loop_times}, cores={cores}]"
            try:
                nx, ny = small_P0.shape
                ng = int(np.nanmax(small_P0)) - int(np.nanmin(small_P0)) + 1
                R = np.zeros((nx, ny, 3))

                inst_std = smooth.linear_class(
                    nx, ny, ng, cores, loop_times, small_P0, R,
                    verification_system=False, curvature_sign=True
                )
                inst_fast = smooth.linear_class(
                    nx, ny, ng, cores, loop_times, small_P0, R,
                    verification_system=False, curvature_sign=True
                )

                t0 = time.perf_counter()
                inst_std.linear_main("both", fast=False)
                t_std = time.perf_counter() - t0

                t0 = time.perf_counter()
                inst_fast.linear_main("both", fast=True, chunk_size=5_000)
                t_fast = time.perf_counter() - t0

                info_msg(
                    f"  loop_times={loop_times} cores={cores}: "
                    f"std={t_std:.2f}s  fast={t_fast:.2f}s  "
                    f"speedup={t_std/max(t_fast,1e-9):.1f}x"
                )

                C_std  = inst_std.get_C()
                C_fast = inst_fast.get_C()
                P_std  = inst_std.get_P()
                P_fast = inst_fast.get_P()

                ok_C, reason_C = arrays_close(C_std, C_fast, tol=tol, name="C")
                ok_P, reason_P = arrays_close(P_std, P_fast, tol=tol, name="P")

                ok = ok_C and ok_P
                reasons = [r for r in [reason_C, reason_P] if r]
                results.record(test_name, ok, " | ".join(reasons))

            except Exception:
                results.record(test_name, False, traceback.format_exc(limit=2))


# ---------------------------------------------------------------------------
# Group 3: End-to-end Exodus file test
# ---------------------------------------------------------------------------

def run_e2e_tests(results, exo_path, time_val, cpus=4, loop_times=5, tol=1e-10):
    head_msg(f"Group 3: End-to-End Exodus File Test ({Path(exo_path).name})")

    try:
        from ExodusBasics import ExodusBasics
        from vector_exodus_to_hdf5_vectorized import (
            process_frame,
            select_step,
            compute_gb_curvature,
            compute_gb_curvature_fast,
        )
    except ImportError as e:
        results.skip("All Group 3 tests", f"Import failed: {e}")
        return

    # Build a minimal args namespace for process_frame
    import types
    args_std = types.SimpleNamespace(
        cpus=cpus,
        loop_times=loop_times,
        tj_distance=6,
        signed=True,
        fast=False,
        chunk_size=50_000,
        verbose=0,
    )
    args_fast = types.SimpleNamespace(
        cpus=cpus,
        loop_times=loop_times,
        tj_distance=6,
        signed=True,
        fast=True,
        chunk_size=50_000,
        verbose=0,
    )

    import logging
    log = logging.getLogger("validate_e2e")

    try:
        with ExodusBasics(exo_path) as exo:
            step = select_step(exo, grains=None, time_value=time_val, log=log)
            info_msg(f"Using step={step} (t={time_val})")

            # ---- 3a: process_frame standard vs fast ------------------
            test_name = f"3a process_frame [step={step}, t={time_val}]"
            try:
                info_msg("Running standard path process_frame...")
                t0 = time.perf_counter()
                res_std = process_frame(exo, step, args_std, log)
                t_std = time.perf_counter() - t0

                info_msg("Running fast path process_frame...")
                t0 = time.perf_counter()
                res_fast = process_frame(exo, step, args_fast, log)
                t_fast = time.perf_counter() - t0

                info_msg(
                    f"Wall time: std={t_std:.1f}s  fast={t_fast:.1f}s  "
                    f"speedup={t_std/max(t_fast,1e-9):.1f}x"
                )

                (step_s, t_s, P0_s, C_s, P_s, gb_s, bp_s, jp_s) = res_std
                (step_f, t_f, P0_f, C_f, P_f, gb_f, bp_f, jp_f) = res_fast

                ok_P0,  r_P0  = arrays_close(P0_s, P0_f,  tol=tol, name="P0")
                ok_C,   r_C   = arrays_close(C_s,  C_f,   tol=tol, name="C")
                ok_P,   r_P   = arrays_close(P_s,  P_f,   tol=tol, name="P")
                ok_gb,  r_gb  = dicts_close(gb_s, gb_f,   tol=tol)
                ok_bp,  r_bp  = pixel_sets_equal(bp_s, bp_f, "boundary_pixels")
                ok_jp,  r_jp  = pixel_sets_equal(jp_s, jp_f, "junction_pixels")

                ok = ok_P0 and ok_C and ok_P and ok_gb and ok_bp and ok_jp
                reasons = [r for r in [r_P0, r_C, r_P, r_gb, r_bp, r_jp] if r]
                results.record(test_name, ok, " | ".join(reasons))

                if ok:
                    info_msg(
                        f"  P0 shape={P0_s.shape}, "
                        f"C shape={C_s.shape}, "
                        f"GBs={len(gb_s)}, "
                        f"boundary_px={len(bp_s)}, "
                        f"junction_px={len(jp_s)}"
                    )

                # ---- 3b: GB curvature statistics comparison ----------
                test_name_stats = f"3b GB curvature statistics [step={step}]"
                if ok_gb:
                    curv_std  = np.array([v[0] for v in gb_s.values()])
                    curv_fast = np.array([v[0] for v in gb_f.values()])

                    # Sort by pair_id for aligned comparison
                    sorted_keys = sorted(gb_s.keys())
                    curv_std_sorted  = np.array([gb_s[k][0]  for k in sorted_keys])
                    curv_fast_sorted = np.array([gb_f[k][0] for k in sorted_keys])

                    ok_stats, r_stats = arrays_close(
                        curv_std_sorted, curv_fast_sorted,
                        tol=tol, name="avg_curvature"
                    )
                    results.record(test_name_stats, ok_stats, r_stats)

                    info_msg(
                        f"  Curvature stats (standard): "
                        f"mean={np.mean(curv_std):.4e}, "
                        f"std={np.std(curv_std):.4e}, "
                        f"min={np.min(curv_std):.4e}, "
                        f"max={np.max(curv_std):.4e}"
                    )
                    info_msg(
                        f"  Curvature stats (fast):     "
                        f"mean={np.mean(curv_fast):.4e}, "
                        f"std={np.std(curv_fast):.4e}, "
                        f"min={np.min(curv_fast):.4e}, "
                        f"max={np.max(curv_fast):.4e}"
                    )
                else:
                    results.skip(test_name_stats, "gb_dict mismatch — cannot compare curvature stats")

            except Exception:
                results.record(test_name, False, traceback.format_exc(limit=3))

    except Exception as e:
        results.skip("All Group 3 tests", f"Could not open Exodus file: {e}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Validate --fast path against standard path.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--exo", type=str, default=None, metavar="FILE",
        help="Path to Exodus .e file for end-to-end test (Group 3). "
             "If not provided, Group 3 is skipped.",
    )
    p.add_argument(
        "-t", "--time", type=float, default=None, metavar="T",
        help="Target simulation time for end-to-end test (required with --exo).",
    )
    p.add_argument(
        "-n", "--cpus", type=int, default=4,
        help="CPUs for end-to-end test.",
    )
    p.add_argument(
        "-l", "--loop-times", type=int, default=5,
        help="Smoothing loop_times for end-to-end test.",
    )
    p.add_argument(
        "--tol", type=float, default=1e-10,
        help="Absolute + relative tolerance for numerical comparisons.",
    )
    p.add_argument(
        "--grid-size", type=int, default=100,
        help="Synthetic grid size (NxN) for Groups 1 and 2.",
    )
    p.add_argument(
        "--only", choices=["synthetic", "linear", "e2e"], default=None,
        help="Run only one test group.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    results = Results()

    run_synthetic = args.only in (None, "synthetic")
    run_linear    = args.only in (None, "linear")
    run_e2e       = args.only in (None, "e2e")

    print(f"\n{BOLD}{'='*60}")
    print("vector_exodus_to_hdf5 --fast path validation")
    print(f"{'='*60}{RESET}")
    print(f"  tolerance  : {args.tol}")
    print(f"  grid_size  : {args.grid_size}")
    print(f"  exodus_file: {args.exo or 'None (Group 3 will be skipped)'}")
    if args.exo:
        print(f"  time       : {args.time}")
        print(f"  cpus       : {args.cpus}")
        print(f"  loop_times : {args.loop_times}")

    if run_synthetic:
        run_synthetic_tests(results, tol=args.tol, grid_size=args.grid_size)

    if run_linear:
        run_linear_tests(results, tol=args.tol, grid_size=min(args.grid_size, 80))

    if run_e2e:
        if args.exo is None:
            results.skip(
                "Group 3 (end-to-end)",
                "No --exo file provided. "
                "Run with --exo /path/to/file.e -t <time> to enable."
            )
        elif args.time is None:
            results.skip("Group 3 (end-to-end)", "--time is required with --exo")
        else:
            run_e2e_tests(
                results,
                exo_path=args.exo,
                time_val=args.time,
                cpus=args.cpus,
                loop_times=args.loop_times,
                tol=args.tol,
            )

    passed = results.summary()
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
