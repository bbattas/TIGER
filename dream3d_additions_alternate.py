'''Adds an external void phase to extend a Dream3D.txt file for use in MOOSE
Uses Bridson's algorithm (Poisson Disk Sampling) with multiple random seeds
for pore placement, ensuring good spatial coverage of the full domain.

Returns:
    out.txt file for ebsd reader in MOOSE
    .log file with full run details and metrics
    .png plot(s) for visualization
'''
from PIL import Image
import glob
import os
import cv2
import argparse
import re
import numpy as np
from itertools import product
import csv
import io
import math
import time
import logging
from datetime import datetime
from scipy.spatial import KDTree
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


# ─────────────────────────────────────────────────────────────────────────────
# ARGUMENT PARSING
# ─────────────────────────────────────────────────────────────────────────────

def parseArgs():
    '''Parse command line arguments

    Returns:
        cl_args
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', '-d', choices=['x', 'y', 'z'], default='x',
                        help='OUTDATED: Only does x direction. Coordinate direction (x,y,z) '
                             'to add the phi volume. Defaults to +x')
    parser.add_argument('--planes', '-n', type=int, default=0,
                        help='Number of element planes of phi to add. Default = 0')
    parser.add_argument('--input', '-i', type=str,
                        help='Name of Dream3D txt file to glob.glob(*__*.txt) find and read.')
    parser.add_argument('--out', '-o', type=str,
                        help='Name of output txt file. If not specified will use '
                             '[inputName]_plusVoid.txt')
    parser.add_argument('--pores', '-p', type=int, default=0,
                        help='Number of pores to add. Default = 0')
    parser.add_argument('--volume', '-v', type=int, default=5,
                        help='Volume percentage (1-100) to make the pores. Default = 5')
    parser.add_argument('--scale', '-s', type=int, default=1,
                        help='Multiplier to apply to all dimensions to scale the domain '
                             '(default=1)')
    parser.add_argument('--spacing', '-x', type=float, default=1,
                        help='Fraction of the average pore radius to use as the minimum '
                             'physical/numerical separation buffer between pores (default=1)')
    parser.add_argument('--phase3', action='store_true',
                        help='Make internal porosity a third phase, separate from external.')
    parser.add_argument('--round', action='store_false',
                        help='Disable rounding of x,y,z columns to clean up floating point '
                             'noise. Enabled by default. Precision is derived from actual '
                             'coordinate values, not a hardcoded decimal count.')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducible pore placement. Optional.')
    parser.add_argument('--noplot', action='store_true',
                        help='Suppress generation of visualization plots.')
    parser.add_argument('--seeds', type=int, default=None,
                        help='Number of initial Bridson seed points distributed across the '
                             'domain to improve spatial coverage. If not specified, defaults '
                             'to ceil(sqrt(N_pores)). Use --seeds 1 to reproduce the single-'
                             'seed behavior. Capped at N_pores - 1.')
    parser.add_argument('--relax', type=float, default=0.9,
                        help='Fraction to multiply min_sep by on each Bridson retry '
                            'if domain exhaustion occurs. Default=0.9. '
                            'Must be between 0 and 1.')
    parser.add_argument('--attempts', type=int, default=5,
                        help='Max number of attempts for relaxing the min_sep to retry.')
    cl_args = parser.parse_args()
    return cl_args


# ─────────────────────────────────────────────────────────────────────────────
# LOGGING SETUP
# ─────────────────────────────────────────────────────────────────────────────

def setup_logger(log_path):
    '''Set up logging to both file and terminal.

    Args:
        log_path: Full path for the log file.

    Returns:
        logger instance
    '''
    logger = logging.getLogger('dream3d_additions')
    logger.setLevel(logging.DEBUG)
    logger.handlers = []  # Clear any existing handlers

    fmt = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S')

    # File handler
    fh = logging.FileHandler(log_path, mode='w')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # Stream (terminal) handler
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    return logger


def log_section(logger, title):
    '''Log a clearly delimited section header.'''
    bar = '=' * 70
    logger.info(bar)
    logger.info(f'  {title}')
    logger.info(bar)


def log_warning_block(logger, lines):
    '''Log a highly visible warning block to both terminal and file.'''
    bar = '!' * 70
    logger.warning(bar)
    logger.warning('\x1b[31;1m  ***  WARNING *** \x1b[0m')
    for line in lines:
        logger.warning(f'  {line}')
    logger.warning(bar)


# ─────────────────────────────────────────────────────────────────────────────
# COORDINATE PRECISION HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def decimal_places_needed(coords_array):
    '''Determine the number of decimal places needed to represent coordinate
    values without truncation. Derived from the actual coordinate values in
    the data rather than the step size alone, so that half-step offsets
    (cell centers) are handled correctly.

    Example: step=0.250 produces cell-center coords like 0.125, 0.375 etc.,
    which need 3 decimal places. Inspecting 0.125 directly is unambiguous
    and requires no assumptions about mesh construction.

    Args:
        coords_array: 1D array of coordinate values to inspect.

    Returns:
        Integer number of decimal places needed, capped at 10.
    '''
    max_places = 0
    for val in np.unique(coords_array):
        s = f'{val:.10f}'.rstrip('0')
        if '.' in s:
            places = len(s.split('.')[1])
            if places > max_places:
                max_places = places
    return min(max_places, 10)


def fmt_coord(val, ndecimals):
    '''Round a coordinate value to ndecimals places and return as a clean
    string with no floating point noise.'''
    return str(round(float(val), ndecimals))


# ─────────────────────────────────────────────────────────────────────────────
# MESH HEADER CLASS
# ─────────────────────────────────────────────────────────────────────────────

class header_vals:
    def __init__(self, x, y, z, feature_id):
        self.dim = 3
        self.feature_id_max = int(max(feature_id))
        self.xyz_params(x, y, z)

    def xyz_params(self, x, y, z):
        '''Calculate mesh min/max and element size in all 3 directions.'''
        # x
        self.xu = np.unique(x)
        if len(self.xu) == 1:
            self.dim -= 1
            self.dx = 0.0
            self.xmin = 0.0
            self.xmax = 0.0
            self.ctr_xmax = 0.0
        else:
            self.dx = self.xu[1] - self.xu[0]
            self.xmin = min(self.xu) - 0.5 * self.dx
            self.xmax = max(self.xu) + 0.5 * self.dx
            self.ctr_xmax = max(self.xu)
        # y
        self.yu = np.unique(y)
        if len(self.yu) == 1:
            self.dim -= 1
            self.dy = 0.0
            self.ymin = 0.0
            self.ymax = 0.0
            self.ctr_ymax = 0.0
        else:
            self.dy = self.yu[1] - self.yu[0]
            self.ymin = min(self.yu) - 0.5 * self.dy
            self.ymax = max(self.yu) + 0.5 * self.dy
            self.ctr_ymax = max(self.yu)
        # z
        self.zu = np.unique(z)
        if len(self.zu) == 1:
            self.dim -= 1
            self.dz = 0.0
            self.zmin = 0.0
            self.zmax = 0.0
            self.ctr_zmax = 0.0
        else:
            self.dz = self.zu[1] - self.zu[0]
            self.zmin = min(self.zu) - 0.5 * self.dz
            self.zmax = max(self.zu) + 0.5 * self.dz
            self.ctr_zmax = max(self.zu)


# ─────────────────────────────────────────────────────────────────────────────
# VOLUME / RADIUS CALCULATION
# ─────────────────────────────────────────────────────────────────────────────

def vol_per_sphere(volume_solid, dims):
    '''Compute random per-pore volumes and radii that sum to the target
    volume fraction.'''
    min_vol = 1 / (1.5 * cl_args.pores)
    a = np.random.rand(cl_args.pores)
    a = (a / a.sum() * (1 - min_vol * cl_args.pores))
    weights = a + min_vol
    volumes = weights * volume_solid * cl_args.volume / 100
    if dims == 2:
        r = (volumes / math.pi) ** (1 / 2)
    elif dims == 3:
        r = (3 * volumes / (4 * math.pi)) ** (1 / 3)
    return r


# ─────────────────────────────────────────────────────────────────────────────
# DISTANCE HELPER
# ─────────────────────────────────────────────────────────────────────────────

def distance(pt1, pt2):
    return math.sqrt((pt1[0] - pt2[0]) ** 2 +
                     (pt1[1] - pt2[1]) ** 2 +
                     (pt1[2] - pt2[2]) ** 2)


# ─────────────────────────────────────────────────────────────────────────────
# BRIDSON PORE PLACEMENT
# ─────────────────────────────────────────────────────────────────────────────

def bridson_generate_centers(radii, mesh, min_sep, n_seeds, logger,
                 attempt=1, max_relax_attempts=5, relax_factor=0.5):
    '''Place pore centers using Bridson's Poisson Disk Sampling algorithm
    with multiple random seed points for improved domain coverage.

    Multiple seeds are placed first, distributed randomly across the full
    safe domain and checked against each other for overlap. They are added
    to the active list simultaneously so Bridson propagation fans out from
    multiple origins, preventing the clustering-in-one-region problem that
    occurs with a single seed and large spacing values.

    Args:
        radii:               Array of pore radii from vol_per_sphere.
        mesh:                header_vals object describing the domain.
        min_sep:             Physical/numerical buffer distance between pores.
        n_seeds:             Number of initial seed points to distribute.
        logger:              Logger instance.
        attempt:             Current relaxation attempt number (internal use).
        max_relax_attempts:  Max number of min_sep relaxation retries.

    Returns:
        coords_arr:  np.ndarray of shape (N_placed, 3) — unscaled pore centers.
        min_sep:     The min_sep value actually used (may differ if relaxed).
        stats:       Dict of placement diagnostic counters.
    '''
    n_pores = len(radii)
    dim = mesh.dim
    k_candidates = 30  # Bridson standard

    # ── Background grid ──────────────────────────────────────────────────────
    # Cell size set by smallest exclusion radius so each cell holds at most
    # one point, enabling O(1) neighbor lookup.
    r_min = float(np.min(radii))
    cell_size = (r_min + min_sep) / math.sqrt(dim)
    if cell_size <= 0:
        cell_size = float(np.min(radii)) / math.sqrt(dim)

    def grid_shape(lo, hi):
        return max(1, int(math.ceil((hi - lo) / cell_size)) + 1)

    gx = grid_shape(0, mesh.xmax)
    gy = grid_shape(0, mesh.ymax)
    gz = grid_shape(0, mesh.zmax) if dim == 3 else 1

    # Grid maps cell index -> placed pore index, -1 = empty
    grid = -np.ones((gx, gy, gz), dtype=int)

    def to_cell(pt):
        cx = int(pt[0] / cell_size)
        cy = int(pt[1] / cell_size)
        cz = int(pt[2] / cell_size) if dim == 3 else 0
        cx = min(cx, gx - 1)
        cy = min(cy, gy - 1)
        cz = min(cz, gz - 1)
        return cx, cy, cz

    # ── Safe domain bounds (pre-shrunk per pore) ─────────────────────────────
    # Pore centers are only ever sampled within this shrunken domain, making
    # boundary overlap structurally impossible without any rejection check.
    def safe_bounds(r_n):
        buf = r_n + min_sep
        x_lo, x_hi = buf, mesh.xmax - buf
        y_lo, y_hi = buf, mesh.ymax - buf
        z_lo = buf               if dim == 3 else 0.0
        z_hi = mesh.zmax - buf   if dim == 3 else 0.0
        return (x_lo, x_hi), (y_lo, y_hi), (z_lo, z_hi)

    def domain_valid(r_n):
        (x_lo, x_hi), (y_lo, y_hi), (z_lo, z_hi) = safe_bounds(r_n)
        if x_hi <= x_lo or y_hi <= y_lo:
            return False
        if dim == 3 and z_hi <= z_lo:
            return False
        return True

    def random_in_safe(r_n):
        (x_lo, x_hi), (y_lo, y_hi), (z_lo, z_hi) = safe_bounds(r_n)
        px = np.random.uniform(x_lo, x_hi)
        py = np.random.uniform(y_lo, y_hi)
        pz = np.random.uniform(z_lo, z_hi) if dim == 3 else 0.0
        return np.array([px, py, pz])

    def in_safe_domain(pt, r_n):
        (x_lo, x_hi), (y_lo, y_hi), (z_lo, z_hi) = safe_bounds(r_n)
        if not (x_lo <= pt[0] <= x_hi and y_lo <= pt[1] <= y_hi):
            return False
        if dim == 3 and not (z_lo <= pt[2] <= z_hi):
            return False
        return True

    def neighbours_clear(pt, r_n, coords):
        '''Check neighboring grid cells only (O(1)) for overlap.'''
        cx, cy, cz = to_cell(pt)
        search = 2
        for ddx in range(-search, search + 1):
            for ddy in range(-search, search + 1):
                for ddz in range((-search if dim == 3 else 0),
                                 (search + 1 if dim == 3 else 1)):
                    nx, ny, nz = cx + ddx, cy + ddy, cz + ddz
                    if 0 <= nx < gx and 0 <= ny < gy and 0 <= nz < gz:
                        idx = grid[nx, ny, nz]
                        if idx >= 0:
                            min_dist = radii[idx] + r_n + min_sep
                            if distance(pt, coords[idx]) < min_dist:
                                return False
        return True

    def sample_annulus(origin, r_inner, r_outer):
        '''Sample a random point in the annulus [r_inner, r_outer] around origin.'''
        if dim == 2:
            angle = np.random.uniform(0, 2 * math.pi)
            r = np.random.uniform(r_inner, r_outer)
            return np.array([origin[0] + r * math.cos(angle),
                             origin[1] + r * math.sin(angle),
                             0.0])
        else:
            v = np.random.randn(3)
            v /= np.linalg.norm(v)
            r = np.random.uniform(r_inner, r_outer)
            return origin + r * v

    # ── Statistics counters ───────────────────────────────────────────────────
    stats = {
        'candidates_generated':  0,
        'rejected_domain':       0,
        'rejected_overlap':      0,
        'active_exhaustions':    0,
        'n_seeds_requested':     n_seeds,
        'n_seeds_placed':        0,
        'seed_coords':           [],
        'min_sep_used':          min_sep,
        'attempt':               attempt,
    }

    # ── Validate every pore fits in its safe domain ───────────────────────────
    for i, r in enumerate(radii):
        if not domain_valid(r):
            raise ValueError(
                f'Pore {i} radius {r:.4f} + min_sep {min_sep:.4f} is too large '
                f'to fit in the domain even once. Reduce pore count, volume '
                f'fraction, or min_sep.')

    # ── Sort pores largest-first ──────────────────────────────────────────────
    # Seeds consume the first n_seeds entries so the largest pores are spread
    # across the domain first, which is the most conservative choice.
    coords = []
    order = np.argsort(radii)[::-1]
    placed_order = []  # maps placed index -> original radii index
    active = []

    pore_queue = list(order)  # full queue; seeds will pop from the front

    # ── Place seed pores ──────────────────────────────────────────────────────
    # Each seed is placed via random_in_safe (uniform over the full shrunk
    # domain) and checked against already-placed seeds. This is the key
    # difference from single-seed Bridson: propagation starts simultaneously
    # from n_seeds spread-out origins, covering the whole domain.
    seeds_to_place = min(n_seeds, len(pore_queue))
    for s in range(seeds_to_place):
        target_idx = pore_queue[0]
        target_r   = radii[target_idx]

        placed = False
        for _ in range(k_candidates):
            candidate = random_in_safe(target_r)
            stats['candidates_generated'] += 1

            # Seeds must also clear already-placed seeds
            if coords and not neighbours_clear(candidate, target_r, coords):
                stats['rejected_overlap'] += 1
                continue

            # Valid seed placement
            placed_local_idx = len(coords)
            coords.append(candidate)
            placed_order.append(target_idx)
            cx, cy, cz = to_cell(candidate)
            grid[cx, cy, cz] = placed_local_idx
            active.append(placed_local_idx)
            pore_queue.pop(0)
            stats['n_seeds_placed'] += 1
            stats['seed_coords'].append(candidate.copy())
            placed = True
            break

        if not placed:
            # Log a warning but continue — fewer seeds still better than one
            logger.warning(
                f'Seed {s + 1}/{seeds_to_place} could not be placed after '
                f'{k_candidates} attempts (domain too constrained at this '
                f'min_sep). Continuing with {stats["n_seeds_placed"]} seeds.')

    if stats['n_seeds_placed'] == 0:
        raise RuntimeError(
            'No seed pores could be placed. The domain is too small for the '
            'requested pore size and min_sep combination.')

    # ── Main Bridson while-loop ───────────────────────────────────────────────
    # Unchanged from single-seed version. Active list now starts with
    # n_seeds entries so propagation fans out from multiple origins.
    while active and pore_queue:
        active_pick = active[np.random.randint(len(active))]
        origin   = coords[active_pick]
        origin_r = radii[placed_order[active_pick]]

        target_idx = pore_queue[0]
        target_r   = radii[target_idx]

        r_inner = origin_r + target_r + min_sep
        r_outer = 2.0 * r_inner

        found = False
        for _ in range(k_candidates):
            candidate = sample_annulus(origin, r_inner, r_outer)
            stats['candidates_generated'] += 1

            if not in_safe_domain(candidate, target_r):
                stats['rejected_domain'] += 1
                continue

            if not neighbours_clear(candidate, target_r, coords):
                stats['rejected_overlap'] += 1
                continue

            # Valid — place it
            placed_local_idx = len(coords)
            coords.append(candidate)
            placed_order.append(target_idx)
            cx, cy, cz = to_cell(candidate)
            grid[cx, cy, cz] = placed_local_idx
            active.append(placed_local_idx)
            pore_queue.pop(0)
            found = True
            break

        if not found:
            active.remove(active_pick)
            stats['active_exhaustions'] += 1

    # ── Handle domain exhaustion ──────────────────────────────────────────────
    n_placed = len(coords)
    if n_placed < n_pores:
        if attempt <= max_relax_attempts:
            relaxed_sep = min_sep * relax_factor
            log_warning_block(logger, [
                f'Bridson domain exhaustion on attempt {attempt}/{max_relax_attempts}.',
                f'Pores placed: {n_placed} / {n_pores} requested.',
                f'Original min_sep:  {min_sep:.6f}',
                f'Relaxed min_sep:   {relaxed_sep:.6f}',
                f'Retrying placement with relaxed separation...',
            ])
            return bridson_generate_centers(
                radii, mesh, relaxed_sep, n_seeds, logger,
                attempt=attempt + 1,
                max_relax_attempts=max_relax_attempts,
                relax_factor=relax_factor)
        else:
            raise RuntimeError(
                f'Bridson placement failed after {max_relax_attempts} relaxation '
                f'attempts. Only {n_placed}/{n_pores} pores could be placed. '
                f'Consider reducing pore count, volume fraction, or min_sep.')

    # Re-order output to match original radii indexing
    reorder = np.argsort(placed_order)
    coords_arr = np.array(coords)[reorder]
    stats['min_sep_used'] = min_sep

    return coords_arr, min_sep, stats


# ─────────────────────────────────────────────────────────────────────────────
# PLOTTING — 2D
# ─────────────────────────────────────────────────────────────────────────────

def generate_plots_2d(data_scaled, mesh, pore_ctrs_scaled, rads_scaled,
                      phi_txt, out_stem, logger, effective_spacing):
    '''Generate a 2D plot of grain structure with pores overlaid.

    All inputs are already in scaled coordinates so no further multiplication
    is applied inside this function.

    Args:
        data_scaled:      Full mesh data array with x/y/z already scaled.
        mesh:             header_vals object (unscaled, used for labels only).
        pore_ctrs_scaled: Pore centers in scaled coordinates.
        rads_scaled:      Pore radii in scaled coordinates.
        phi_txt:          List of external void plane rows (may be empty).
        out_stem:         Base filename (no extension) for saving.
        logger:           Logger instance.
    '''
    fig, ax = plt.subplots(figsize=(8, 8))

    x_grain     = data_scaled[:, 3]
    y_grain     = data_scaled[:, 4]
    feature_ids = data_scaled[:, 6]
    phase_ids   = data_scaled[:, 7]

    grain_mask = phase_ids == 1
    pore_mask  = phase_ids != 1

    sc = ax.scatter(x_grain[grain_mask], y_grain[grain_mask],
                    c=feature_ids[grain_mask], cmap='tab20',
                    s=4, linewidths=0, zorder=1, label='Grain')
    if np.any(pore_mask):
        ax.scatter(x_grain[pore_mask], y_grain[pore_mask],
                   color='steelblue', s=4, linewidths=0, zorder=2,
                   label='Internal Pore (mesh)')
    plt.colorbar(sc, ax=ax, label='Feature ID', shrink=0.75)

    if phi_txt:
        # phi_txt coords are already scaled at construction time
        phi_arr = np.array([[float(r[3]), float(r[4])] for r in phi_txt])
        ax.scatter(phi_arr[:, 0], phi_arr[:, 1],
                   color='dimgray', s=4, linewidths=0, zorder=1,
                   label='External Void (phi)')

    # Pore circles — already scaled, no multiplication needed
    for i, (ctr, r) in enumerate(zip(pore_ctrs_scaled, rads_scaled)):
        circle = mpatches.Circle(
            (ctr[0], ctr[1]),
            radius=r,
            facecolor='steelblue', edgecolor='navy',
            linewidth=0.8, alpha=0.45, zorder=3,
            label='Pore circles' if i == 0 else None)
        ax.add_patch(circle)

    ax.set_aspect('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'Grain Structure + Pores\n'
             f'{cl_args.pores} pores, {cl_args.volume}% vol, '
             f'spacing={effective_spacing:.3f}'
             + (f' (relaxed from {cl_args.spacing})'
                if abs(effective_spacing - cl_args.spacing) > 1e-9 else ''))
    ax.legend(loc='upper right', markerscale=3, fontsize=8)

    plot_path = out_stem + '_2D.png'
    fig.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f'2D plot saved: {plot_path}')


# ─────────────────────────────────────────────────────────────────────────────
# PLOTTING — 3D
# ─────────────────────────────────────────────────────────────────────────────

WIREFRAME_THRESHOLD = 50


def _draw_bounding_box(ax, xmax, ymax, zmax):
    '''Draw a wireframe bounding box on a 3D axis.'''
    corners = np.array([[0, 0, 0], [xmax, 0, 0], [xmax, ymax, 0],
                         [0, ymax, 0], [0, 0, zmax], [xmax, 0, zmax],
                         [xmax, ymax, zmax], [0, ymax, zmax]])
    edges = [(0, 1), (1, 2), (2, 3), (3, 0),
             (4, 5), (5, 6), (6, 7), (7, 4),
             (0, 4), (1, 5), (2, 6), (3, 7)]
    for a, b in edges:
        ax.plot3D(*zip(corners[a], corners[b]), color='black',
                  linewidth=0.6, alpha=0.5)


def _draw_pores_3d(ax, pore_ctrs, rads, use_scatter, cmap='plasma'):
    '''Draw pores on a 3D axis as wireframe spheres or scaled scatter dots.

    Automatically switches to scatter dots above WIREFRAME_THRESHOLD pores
    to keep render time and visual clarity manageable.

    Args:
        ax:          Matplotlib 3D axis.
        pore_ctrs:   Pore centers array (scaled).
        rads:        Pore radii array (scaled).
        use_scatter: If True, use scatter dots; else use wireframe spheres.
        cmap:        Colormap name for color-coding pores.
    '''
    colors = plt.get_cmap(cmap)(np.linspace(0, 1, len(rads)))

    if use_scatter:
        s_vals = (np.array(rads) / np.max(rads) * 120) + 20
        ax.scatter(pore_ctrs[:, 0], pore_ctrs[:, 1], pore_ctrs[:, 2],
                   s=s_vals, c=np.linspace(0, 1, len(rads)),
                   cmap=cmap, depthshade=True, alpha=0.75, zorder=3)
    else:
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 12)
        xs = np.outer(np.cos(u), np.sin(v))
        ys = np.outer(np.sin(u), np.sin(v))
        zs = np.outer(np.ones_like(u), np.cos(v))
        for i, (ctr, r) in enumerate(zip(pore_ctrs, rads)):
            ax.plot_wireframe(ctr[0] + r * xs,
                              ctr[1] + r * ys,
                              ctr[2] + r * zs,
                              color=colors[i], alpha=0.4,
                              rstride=2, cstride=2, linewidth=0.4)


def _add_phi_box(ax, mesh, planes, scale):
    '''Render the external void plane region as a semi-transparent box.
    Coordinates are scaled to match the rest of the plot.'''
    x0 = mesh.ctr_xmax * scale
    x1 = (mesh.xmax + planes * mesh.dx) * scale
    y0, y1 = 0, mesh.ymax * scale
    z0, z1 = 0, mesh.zmax * scale
    verts = [
        [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0)],
        [(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)],
        [(x0, y0, z0), (x0, y0, z1), (x0, y1, z1), (x0, y1, z0)],
        [(x1, y0, z0), (x1, y0, z1), (x1, y1, z1), (x1, y1, z0)],
        [(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)],
        [(x0, y1, z0), (x1, y1, z0), (x1, y1, z1), (x0, y1, z1)],
    ]
    poly = Poly3DCollection(verts, alpha=0.15, facecolor='gray',
                            edgecolor='dimgray', linewidth=0.5)
    ax.add_collection3d(poly)


VIEWS = [
    ('XY (Top)',    90,  0),
    ('XZ (Front)',   0,  0),
    ('Isometric',   30, 45),
]


def _make_3d_figure(pore_ctrs_scaled, rads_scaled, mesh, scale, use_scatter,
                    include_phi, planes, title_suffix, effective_spacing):
    '''Build a 3-panel 3D figure. All geometry is in scaled coordinates.'''
    fig = plt.figure(figsize=(18, 6))
    fig.suptitle(f'{cl_args.pores} pores, {cl_args.volume}% vol, '
             f'spacing={effective_spacing:.3f}'
             + (f' (relaxed from {cl_args.spacing})'
                if abs(effective_spacing - cl_args.spacing) > 1e-9 else '')
             + f' — {title_suffix}', fontsize=11)

    xmax_s = mesh.xmax * scale
    ymax_s = mesh.ymax * scale
    zmax_s = mesh.zmax * scale

    for col, (view_label, elev, azim) in enumerate(VIEWS):
        ax = fig.add_subplot(1, 3, col + 1, projection='3d')
        _draw_bounding_box(ax, xmax_s, ymax_s, zmax_s)
        _draw_pores_3d(ax, pore_ctrs_scaled, rads_scaled, use_scatter)
        if include_phi and planes > 0:
            _add_phi_box(ax, mesh, planes, scale)
        ax.view_init(elev=elev, azim=azim)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_title(view_label, fontsize=9)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return fig


def generate_plots_3d(mesh, pore_ctrs_scaled, rads_scaled, planes, scale,
                      out_stem, logger, effective_spacing):
    '''Generate two 3D figures (without and with external void region),
    each showing three views. All geometry passed in is already scaled.

    Args:
        mesh:             header_vals object (unscaled domain description).
        pore_ctrs_scaled: Pore centers in scaled coordinates.
        rads_scaled:      Pore radii in scaled coordinates.
        planes:           Number of external void planes.
        scale:            Scale factor (passed through to phi box and bbox).
        out_stem:         Base filename (no extension).
        logger:           Logger instance.
    '''
    n_pores = len(rads_scaled)
    use_scatter = n_pores > WIREFRAME_THRESHOLD
    render_mode = 'scatter dots' if use_scatter else 'wireframe spheres'
    logger.info(f'3D rendering mode: {render_mode} '
                f'(threshold = {WIREFRAME_THRESHOLD} pores)')

    fig1 = _make_3d_figure(pore_ctrs_scaled, rads_scaled, mesh, scale,
                           use_scatter, include_phi=False, planes=planes,
                           title_suffix='Internal Pores Only',
                           effective_spacing=effective_spacing)
    path1 = out_stem + '_3D_nophi.png'
    fig1.savefig(path1, dpi=150, bbox_inches='tight')
    plt.close(fig1)
    logger.info(f'3D plot (no phi) saved: {path1}')

    fig2 = _make_3d_figure(pore_ctrs_scaled, rads_scaled, mesh, scale,
                           use_scatter, include_phi=True, planes=planes,
                           title_suffix='Internal Pores + External Void Region',
                           effective_spacing=effective_spacing)
    path2 = out_stem + '_3D_withphi.png'
    fig2.savefig(path2, dpi=150, bbox_inches='tight')
    plt.close(fig2)
    logger.info(f'3D plot (with phi) saved: {path2}')


# ─────────────────────────────────────────────────────────────────────────────
# FAST OUTPUT WRITER
# ─────────────────────────────────────────────────────────────────────────────

def write_output(out_name, header, data, phi_txt, cl_args,
                 coord_decimals_scaled, logger):
    '''Write output file using numpy.savetxt for fast bulk writing.

    Header rows (few lines) are written as plain text first.
    Body and phi rows are assembled as a NumPy array and written
    via np.savetxt, which is substantially faster than csv.writer
    for large row counts.

    Args:
        out_name:              Output file path.
        header:                List of header row token lists.
        data:                  NumPy array of mesh body data (N x 9).
        phi_txt:               List of external void row token lists (may be empty).
        cl_args:               Parsed command line arguments.
        coord_decimals_scaled: Number of decimal places for scaled coordinates.
        logger:                Logger instance.
    '''
    coord_fmt = f'%.{coord_decimals_scaled}f'
    body_fmt  = ['%d', '%d', '%d', coord_fmt, coord_fmt, coord_fmt,
                 '%d', '%d', '%d']

    # ── Write header lines (small — plain text is fine) ───────────────────────
    with open(out_name, 'w') as f:
        for row in header:
            f.write(' '.join(row) + '\n')

    # ── Build body array ──────────────────────────────────────────────────────
    # Scale coordinate columns and collect integer columns in one shot.
    # All columns are kept as float64 so np.column_stack works cleanly;
    # the '%d' fmt entries in body_fmt drop the decimals on write.
    body_arr = np.column_stack([
        data[:, 0],                                              # col 0 int
        data[:, 1],                                              # col 1 int
        data[:, 2],                                              # col 2 int
        np.round(data[:, 3] * cl_args.scale, coord_decimals_scaled),  # x scaled
        np.round(data[:, 4] * cl_args.scale, coord_decimals_scaled),  # y scaled
        np.round(data[:, 5] * cl_args.scale, coord_decimals_scaled),  # z scaled
        data[:, 6],                                              # feature_id int
        data[:, 7],                                              # phase int
        data[:, 8],                                              # index int
    ])

    # ── Append body (and phi if present) via np.savetxt ──────────────────────
    with open(out_name, 'ab') as f:
        np.savetxt(f, body_arr, fmt=body_fmt, delimiter=' ')
        if phi_txt:
            # phi_txt is a list of string-token lists; convert to float array
            phi_arr = np.array(phi_txt, dtype=float)
            np.savetxt(f, phi_arr, fmt=body_fmt, delimiter=' ')

    logger.info(f'Output written: {out_name}  '
                f'({len(body_arr) + len(phi_txt)} rows)')


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':

    cl_args = parseArgs()

    if cl_args.relax <= 0 or cl_args.relax >= 1:
        raise ValueError(f'--relax must be between 0 and 1 exclusive, got {cl_args.relax}')

    # ── Random seed ───────────────────────────────────────────────────────────
    if cl_args.seed is not None:
        np.random.seed(cl_args.seed)

    # ── Find input file ───────────────────────────────────────────────────────
    txt_names = []
    if cl_args.input is None:
        searchName = '*.txt'
    elif '.txt' in cl_args.input:
        searchName = '*' + cl_args.input
    elif '/' in cl_args.input and '.txt' not in cl_args.input:
        searchName = cl_args.input + '*.txt'
    else:
        searchName = '*' + cl_args.input + '*.txt'
    for file in glob.glob(searchName):
        txt_names.append(file)
    txt_file = txt_names[0]

    # ── Output naming ─────────────────────────────────────────────────────────
    run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_stem     = txt_file.rsplit('.', 1)[0]
    run_stem      = f'{base_stem}_{run_timestamp}'

    if cl_args.out is None:
        out_name = base_stem + '_plusVoid.txt'
    elif '.txt' in cl_args.out:
        out_name = cl_args.out
    else:
        out_name = cl_args.out + '.txt'

    log_path = run_stem + '.log'
    logger   = setup_logger(log_path)

    # ── Determine number of Bridson seeds ─────────────────────────────────────
    # Done early so it can be logged in Section A.
    if cl_args.pores > 0:
        if cl_args.seeds is None:
            n_seeds = math.ceil(math.sqrt(cl_args.pores))
            seeds_auto = True
        else:
            n_seeds = cl_args.seeds
            seeds_auto = False

        # Cap at n_pores - 1 so at least one pore is placed via propagation
        if n_seeds >= cl_args.pores:
            n_seeds = max(1, cl_args.pores - 1)
            logger.warning(f'--seeds capped to {n_seeds} (must be < n_pores).')

    # ── Section A: Run Configuration ─────────────────────────────────────────
    log_section(logger, 'A — RUN CONFIGURATION')
    logger.info(f'Timestamp:        {run_timestamp}')
    logger.info(f'Input file:       {txt_file}')
    logger.info(f'Output file:      {out_name}')
    logger.info(f'Log file:         {log_path}')
    logger.info(f'Random seed:      {cl_args.seed}')
    logger.info(f'Pores requested:  {cl_args.pores}')
    logger.info(f'Volume fraction:  {cl_args.volume}%')
    logger.info(f'Spacing (--x):    {cl_args.spacing}  (fraction of mean radius)')
    logger.info(f'Relax factor:     {cl_args.relax}  (min_sep multiplier on retry)')
    logger.info(f'Scale:            {cl_args.scale}')
    logger.info(f'Ext void planes:  {cl_args.planes}  (direction: {cl_args.dir})')
    logger.info(f'Phase3:           {cl_args.phase3}')
    logger.info(f'Rounding:         {cl_args.round}')
    logger.info(f'No plots:         {cl_args.noplot}')
    if cl_args.pores > 0:
        logger.info(f'Bridson seeds:    {n_seeds}  '
                    f'({"auto: ceil(sqrt(N))" if seeds_auto else "user specified"})')

    # ── Read input file ───────────────────────────────────────────────────────
    with open(txt_file) as f:
        full_data = [r.split() for r in f]
    header = []
    body   = []
    for row in full_data:
        if '#' in row[0]:
            header.append(row)
        else:
            body.append(row)

    data = np.asarray(body, dtype=float)

    # ── Determine coordinate precision from actual data values ────────────────
    # Inspect raw coordinate values directly so half-step cell-center offsets
    # (e.g. step=0.250 -> coords at 0.125, 0.375 ...) are captured correctly.
    active_axes = []
    if len(np.unique(data[:, 3])) > 1:
        active_axes.append(data[:, 3])
    if len(np.unique(data[:, 4])) > 1:
        active_axes.append(data[:, 4])
    if len(np.unique(data[:, 5])) > 1:
        active_axes.append(data[:, 5])
    all_coords = np.concatenate([ax[ax != 0] for ax in active_axes]) \
                 if active_axes else np.array([1.0])

    coord_decimals        = decimal_places_needed(all_coords)
    coord_decimals_scaled = decimal_places_needed(all_coords * cl_args.scale)

    # ── Optional rounding to clean up floating point noise ───────────────────
    if cl_args.round:
        logger.info(f'Rounding x,y,z to {coord_decimals} decimal places '
                    f'(derived from actual coordinate values in data).')
        data[:, 3] = np.round(data[:, 3], coord_decimals)
        data[:, 4] = np.round(data[:, 4], coord_decimals)
        data[:, 5] = np.round(data[:, 5], coord_decimals)

    mesh = header_vals(data[:, 3], data[:, 4], data[:, 5], data[:, 6])
    dim  = mesh.dim

    logger.info(f'Detected dimension:         {dim}D')
    logger.info(f'Coord precision (unscaled): {coord_decimals} decimal places')
    logger.info(f'Coord precision (scaled):   {coord_decimals_scaled} decimal places')
    logger.info(f'Domain x: [{mesh.xmin:.{coord_decimals}f}, '
                f'{mesh.xmax:.{coord_decimals}f}]  dx={mesh.dx:.{coord_decimals}f}')
    logger.info(f'Domain y: [{mesh.ymin:.{coord_decimals}f}, '
                f'{mesh.ymax:.{coord_decimals}f}]  dy={mesh.dy:.{coord_decimals}f}')
    if dim == 3:
        logger.info(f'Domain z: [{mesh.zmin:.{coord_decimals}f}, '
                    f'{mesh.zmax:.{coord_decimals}f}]  dz={mesh.dz:.{coord_decimals}f}')
    logger.info(f'Number of grains (feature_id_max): {mesh.feature_id_max}')
    logger.info(f'Total mesh points: {len(data)}')

    # ── External void planes ──────────────────────────────────────────────────
    phi_txt = []
    if cl_args.planes > 0:
        if 'x' in cl_args.dir:
            new_x = np.asarray([mesh.ctr_xmax + (n + 1) * mesh.dx
                                  for n in range(cl_args.planes)])
            new_y = mesh.yu
            new_z = mesh.zu
        new_coords = list(product(new_x, new_y, new_z))
        f_num = mesh.feature_id_max + 1
        for s in new_coords:
            phi_txt.append(['0.0', '0.0', '0.0',
                             fmt_coord(s[0] * cl_args.scale, coord_decimals_scaled),
                             fmt_coord(s[1] * cl_args.scale, coord_decimals_scaled),
                             fmt_coord(s[2] * cl_args.scale, coord_decimals_scaled),
                             str(f_num), '2', '43'])

        new_xmax = mesh.xmax + cl_args.planes * mesh.dx
        for row in header:
            if len(row) > 1:
                if 'X_MAX' in row[1]:
                    row[2] = str(new_xmax)
                if 'X_DIM' in row[1]:
                    row[2] = str(float(row[2]) + cl_args.planes)
                if 'STEP' in row[1] or 'MAX' in row[1]:
                    row[2] = str(cl_args.scale * float(row[2]))
        logger.info(f'External void: {len(phi_txt)} points added '
                    f'({cl_args.planes} planes in +{cl_args.dir} direction).')

    # ── Pore placement ────────────────────────────────────────────────────────
    pore_ctrs        = None
    rads             = None
    pore_ctrs_scaled = None
    rads_scaled      = None

    if cl_args.pores > 0:

        volume_solid = mesh.xmax * mesh.ymax
        if dim == 3:
            volume_solid *= mesh.zmax

        rads    = vol_per_sphere(volume_solid, dim)
        min_sep = float(np.average(rads)) * cl_args.spacing

        # ── Section B: Volume & Radius Calculations ───────────────────────────
        log_section(logger, 'B — VOLUME & RADIUS CALCULATIONS')
        target_vol = volume_solid * cl_args.volume / 100
        logger.info(f'Solid volume (domain):      {volume_solid:.6f}')
        logger.info(f'Target pore volume:         {target_vol:.6f}  ({cl_args.volume}%)')
        logger.info(f'Mean pore radius:           {np.mean(rads):.6f}')
        logger.info(f'Std  pore radius:           {np.std(rads):.6f}')
        logger.info(f'Min  pore radius:           {np.min(rads):.6f}')
        logger.info(f'Max  pore radius:           {np.max(rads):.6f}')
        logger.info(f'min_sep (spacing x mean_r): {min_sep:.6f}')
        logger.info(f'Per-pore radii array:\n  {np.round(rads, 6)}')

        for label, r_ref in [('smallest pore', np.min(rads)),
                              ('largest pore',  np.max(rads))]:
            buf = r_ref + min_sep
            logger.info(f'Safe domain ({label}, r={r_ref:.4f}): '
                        f'x=[{buf:.4f}, {mesh.xmax - buf:.4f}]  '
                        f'y=[{buf:.4f}, {mesh.ymax - buf:.4f}]' +
                        (f'  z=[{buf:.4f}, {mesh.zmax - buf:.4f}]'
                         if dim == 3 else ''))

        # ── Bridson placement (all in unscaled mesh coordinates) ──────────────
        t_start = time.time()
        pore_ctrs, used_min_sep, bstats = bridson_generate_centers(
            rads, mesh, min_sep, n_seeds, logger, max_relax_attempts=cl_args.attempts,
            relax_factor=cl_args.relax)
        t_end = time.time()

        # ── Scale pore geometry ONCE here ─────────────────────────────────────
        # All downstream code (logging, output, plots) uses these scaled values.
        pore_ctrs_scaled = pore_ctrs * cl_args.scale
        rads_scaled      = np.array(rads) * cl_args.scale

        # ── Section C: Placement Details ──────────────────────────────────────
        log_section(logger, 'C — BRIDSON PLACEMENT DETAILS')
        logger.info(f'Pores requested:          {cl_args.pores}')
        logger.info(f'Pores placed:             {len(pore_ctrs)}')
        logger.info(f'Seeds requested:          {bstats["n_seeds_requested"]}  '
                    f'({"auto" if seeds_auto else "user specified"})')
        logger.info(f'Seeds placed:             {bstats["n_seeds_placed"]}')
        logger.info('Seed coordinates (unscaled):')
        for si, sc_ in enumerate(bstats['seed_coords']):
            logger.info(f'  Seed {si + 1}: {np.round(sc_, 6)}')
        logger.info(f'min_sep originally:       {min_sep:.6f}')
        logger.info(f'min_sep used:             {used_min_sep:.6f}')
        if bstats['attempt'] > 1:
            logger.info(f'Relaxation attempts:      {bstats["attempt"] - 1}')
        logger.info(f'Candidates generated:     {bstats["candidates_generated"]}')
        logger.info(f'Rejected (domain):        {bstats["rejected_domain"]}')
        logger.info(f'Rejected (overlap):       {bstats["rejected_overlap"]}')
        logger.info(f'Active list exhaustions:  {bstats["active_exhaustions"]}')
        logger.info(f'Placement wall time:      {t_end - t_start:.3f} s')
        logger.info(f'Pore centers (unscaled):\n{np.round(pore_ctrs, 6)}')
        logger.info(f'Pore radii (unscaled):\n{np.round(rads, 6)}')
        logger.info(f'Pore centers (scaled):\n'
                    f'{np.round(pore_ctrs_scaled, coord_decimals_scaled)}')
        logger.info(f'Pore radii (scaled):\n'
                    f'{np.round(rads_scaled, coord_decimals_scaled)}')

        # MOOSE-ready block
        moose_r    = np.round(rads_scaled,        coord_decimals_scaled)
        moose_ctrs = np.round(pore_ctrs_scaled.T, coord_decimals_scaled)
        logger.info('MOOSE IC block (copy-paste ready):')
        logger.info(f'  R:  {list(moose_r)}')
        logger.info(f'  x:  {list(moose_ctrs[0])}')
        logger.info(f'  y:  {list(moose_ctrs[1])}')
        if dim == 3:
            logger.info(f'  z:  {list(moose_ctrs[2])}')

        print('Centers (unscaled):')
        print(pore_ctrs)
        print('Radii (unscaled):')
        print(rads)
        print('Centers (scaled):')
        print(pore_ctrs_scaled)
        print('Radii (scaled):')
        print(rads_scaled)
        print('Or for MOOSE IC')
        print('R: ', moose_r)
        print(moose_ctrs)

        # ── Assign pore phase to mesh points (vectorized via KDTree) ──────────
        # Build the tree once over all mesh point coordinates. For each pore,
        # query_ball_point returns the indices of all points within loop_rad
        # of the pore center — identical to the old distance() <= loop_rad
        # check but executed in compiled C rather than a Python loop.
        pore_phase = int(3) if cl_args.phase3 else int(2)
        pts  = data[:, 3:6]          # (N, 3) coordinate array — built once
        tree = KDTree(pts)           # O(N log N) build, done once

        t_assign_start = time.time()
        for pore in range(len(rads)):
            ctr      = pore_ctrs[pore]
            loop_rad = rads[pore]
            pore_id  = mesh.feature_id_max + 2 + pore
            idxs = tree.query_ball_point(ctr, loop_rad)  # C-level spatial query
            data[idxs, 6] = pore_id
            data[idxs, 7] = pore_phase
        t_assign_end = time.time()
        logger.info(f'Pore assignment wall time: {t_assign_end - t_assign_start:.3f} s')

    # ── Section D: Grain Structure Metrics ───────────────────────────────────
    log_section(logger, 'D — GRAIN STRUCTURE METRICS')
    phase_col   = data[:, 7]
    n_pore_pts  = int(np.sum(phase_col == (3 if cl_args.phase3 else 2)))
    actual_frac = n_pore_pts / len(data) * 100 if cl_args.pores > 0 else 0.0
    logger.info(f'Number of grains:         {mesh.feature_id_max}')
    logger.info(f'Total mesh points:        {len(data)}')
    logger.info(f'Points assigned to pores: {n_pore_pts}')
    logger.info(f'Actual pore vol fraction: {actual_frac:.3f}%  '
                f'(target: {cl_args.volume}%)')

    # ── Section E: Output Summary ─────────────────────────────────────────────
    log_section(logger, 'E — OUTPUT SUMMARY')
    logger.info(f'Output file:           {out_name}')
    logger.info(f'External void planes:  {cl_args.planes > 0}  '
                f'({cl_args.planes} planes, +{cl_args.dir})')
    logger.info(f'Phase3 internal pores: {cl_args.phase3}')

    # ── Write output (vectorized body rebuild + np.savetxt) ───────────────────
    t_write_start = time.time()
    write_output(out_name, header, data, phi_txt, cl_args,
                 coord_decimals_scaled, logger)
    t_write_end = time.time()
    logger.info(f'Write wall time: {t_write_end - t_write_start:.3f} s')

    logger.info('Done.')
    print('Done')

    # ── Plots ─────────────────────────────────────────────────────────────────
    if not cl_args.noplot and cl_args.pores > 0:
        data_scaled = data.copy()
        data_scaled[:, 3] = np.round(data[:, 3] * cl_args.scale,
                                      coord_decimals_scaled)
        data_scaled[:, 4] = np.round(data[:, 4] * cl_args.scale,
                                      coord_decimals_scaled)
        data_scaled[:, 5] = np.round(data[:, 5] * cl_args.scale,
                                      coord_decimals_scaled)

        effective_spacing = used_min_sep / float(np.average(rads))

        if dim == 2:
            generate_plots_2d(data_scaled, mesh, pore_ctrs_scaled, rads_scaled,
                               phi_txt, run_stem, logger, effective_spacing=effective_spacing)
        else:
            generate_plots_3d(mesh, pore_ctrs_scaled, rads_scaled,
                               cl_args.planes, cl_args.scale, run_stem, logger,
                               effective_spacing=effective_spacing)
