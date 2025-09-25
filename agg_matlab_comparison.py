from MultiExodusReader import MultiExodusReader
from MultiExodusReaderDerivs import MultiExodusReaderDerivs
# import multiprocessing as mp
# from VolumeScripts import *

import subprocess
from joblib import Parallel, delayed

import matplotlib.tri as mtri
from matplotlib.tri import Triangulation, LinearTriInterpolator
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PolyCollection
from matplotlib.lines import Line2D
# from matplotlib.collections import PatchCollection
# from mpl_toolkits.mplot3d.art3d import Poly3DCollection
# from mpl_toolkits.mplot3d import Axes3D
import matplotlib

from scipy.spatial import ConvexHull, Delaunay
import numpy as np
import time
# from time import time
import os
import glob
import pandas as pd
import math
import sys
import tracemalloc
import logging
import argparse
from pathlib import Path
import re
from enum import Enum
from tqdm import tqdm
import fnmatch

from scipy.spatial import cKDTree
import networkx as nx

import pyarrow as pa
import pyarrow.parquet as pq

# CL Argument Parser
parser = argparse.ArgumentParser()
parser.add_argument('--verbose', '-v', action='count', default=0,
                    help='Increase verbosity: -v for INFO, -vv for DEBUG.')
parser.add_argument('--plot','-p',action='store_true',
                            help='Save the plots in pics/, default=False')
parser.add_argument('--level','-c',type=float,default=0.99,
                    help='Contour value for plotting single contours.')
args = parser.parse_args()



# LOGGING
def configure_logging(args):
    if args.verbose == 2:
        logging.basicConfig(level=logging.DEBUG, format='%(message)s')
    elif args.verbose == 1:
        logging.basicConfig(level=logging.INFO, format='%(message)s')
    else:
        logging.basicConfig(level=logging.WARNING, format='%(message)s')

# Configure logging based on verbosity level
configure_logging(args)

# Logging functions
pt = logging.warning
verb = logging.info
db = logging.debug

# Example usage
verb('Verbose Logging Enabled')
db('Debug Logging Enabled')
verb('''INFO: Comparing specific PF agg bicrystal results to MATLAB Moelans code '''
     '''version of the same.  in pf/ex/agg/02_matlab_comparison.
     SHOULD BE RUN IN THAT PARENT FOLDER!''')
db('''WARNING: This script assumes PF elements are quad4.''')
db(' ')
# pt('This is a warning.')
db(f'Command-line arguments: {args}')
db(' ')




cwd = os.getcwd()

imdir = 'pics'
# if args.plot:
if not os.path.isdir(imdir):
    verb('Making picture directory: '+imdir)
    os.makedirs(imdir)




# ███████╗██╗   ██╗███╗   ██╗ ██████╗████████╗██╗ ██████╗ ███╗   ██╗███████╗
# ██╔════╝██║   ██║████╗  ██║██╔════╝╚══██╔══╝██║██╔═══██╗████╗  ██║██╔════╝
# █████╗  ██║   ██║██╔██╗ ██║██║        ██║   ██║██║   ██║██╔██╗ ██║███████╗
# ██╔══╝  ██║   ██║██║╚██╗██║██║        ██║   ██║██║   ██║██║╚██╗██║╚════██║
# ██║     ╚██████╔╝██║ ╚████║╚██████╗   ██║   ██║╚██████╔╝██║ ╚████║███████║
# ╚═╝      ╚═════╝ ╚═╝  ╚═══╝ ╚═════╝   ╚═╝   ╚═╝ ╚═════╝ ╚═╝  ╚═══╝╚══════╝


# For sorting to deal with no leading zeros
def natural_sort_key(s, _nsre=re.compile('([0-9]+)')):
    '''Sorts the file names naturally to account for lack of leading zeros
    use this function in listname.sort(key=natural_sort_key)

    Args:
        s: files/iterator
        _nsre: _description_. Defaults to re.compile('([0-9]+)').

    Returns:
        Sorted data
    '''
    return [int(text) if text.isdigit() else text.lower()
            for text in _nsre.split(s)]

def _family_from_exodus(e_name: str) -> str:
    """
    Infer 'm2'/'m4'/'m6' from an Exodus path like '02_pf_m2/02_pf_m2_out.e'.
    """
    m = re.search(r'pf_(m\d+)', str(e_name))
    if not m:
        raise ValueError(f"Could not infer family from exodus path: {e_name}")
    return m.group(1)

def _time_token(t_val) -> str:
    """
    Format a time stamp for filenames.
    - If it's an integer (5, 10, 15, 20) → '05','10','15','20'
    - Otherwise (e.g., 7.5) → '07.5' (width 5 with 1 decimal)
    """
    t = float(t_val)
    if t.is_integer():
        return f"{int(t):02d}"         # '05', '10', '15', '20'
    return f"{t:05.1f}"                # '07.5', '12.0', etc.

def matlab_csv_for(e_name: str, t_val, base_dir="01_matlab/results") -> Path:
    """
    Build the MATLAB CSV path that matches the Exodus family and time.
    Example: '01_matlab/results/matlab_m2_05s.csv'
    """
    family = _family_from_exodus(e_name)   # 'm2', 'm4', 'm6', ...
    tstr   = _time_token(t_val)
    return Path(base_dir) / f"matlab_{family}_{tstr}s.csv"


def find_files():
    """
    Check for existence of MATLAB CSV and Phase-Field Exodus result files.

    Returns
    -------
    matlab_files : list of str
        List of found MATLAB CSV file paths.
    exodus_files : list of str
        List of found Exodus (.e) file paths.

    Raises
    ------
    FileNotFoundError
        If any required file is missing.
    """
    # --- Define required files ---
    matlab_files = [
        os.path.join("01_matlab", "results", "matlab_m2_15s.csv"),
        os.path.join("01_matlab", "results", "matlab_m4_15s.csv"),
        os.path.join("01_matlab", "results", "matlab_m6_15s.csv"),
    ]
    exodus_files = [
        os.path.join("02_pf_m2", "02_pf_m2_out.e"),
        os.path.join("03_pf_m4", "03_pf_m4_out.e"),
        os.path.join("04_pf_m6", "04_pf_m6_out.e"),
    ]

    # --- Check for missing files ---
    missing = [f for f in matlab_files + exodus_files if not os.path.isfile(f)]

    if missing:
        msg = "Missing required files:\n" + "\n".join(f"  - {f}" for f in missing)
        raise FileNotFoundError(msg)

    return matlab_files, exodus_files


def read_matlab(filename):
    """
    Read MATLAB-exported CSV file and reconstruct 2D grids.

    Parameters
    ----------
    filename : str or Path
        Path to the CSV file generated by the MATLAB export function.
        The CSV must have columns: 'x', 'y', 'eta1', 'eta2'.

    Returns
    -------
    x : ndarray of shape (N, N)
        2D array of x-coordinate values (column-major order from MATLAB).
    y : ndarray of shape (N, N)
        2D array of y-coordinate values (column-major order from MATLAB).
    eta1_grid : ndarray of shape (N, N)
        2D array of eta1 values reconstructed to match MATLAB's layout.
    eta2_grid : ndarray of shape (N, N)
        2D array of eta2 values reconstructed to match MATLAB's layout.

    Notes
    -----
    This function assumes a square grid of size N x N with N = 801 (was 161)
    and reshapes using column-major order ('F') so that the data
    matches MATLAB's memory layout.
    """
    df = pd.read_csv(filename)
    N = 801#161
    eta1_grid = df['eta1'].to_numpy().reshape(N, N, order='F')  # MATLAB is column-major
    eta2_grid = df['eta2'].to_numpy().reshape(N, N, order='F')
    x = df['x'].to_numpy().reshape(N, N, order='F')
    y = df['y'].to_numpy().reshape(N, N, order='F')
    return x, y, eta1_grid, eta2_grid


def plot_matlab_debug(filename,level=args.level):
    """
    Load a MATLAB CSV and plot eta1 as a colormap over the x-y grid.

    Parameters
    ----------
    filename : str or Path
        Path to the CSV file containing x, y, eta1, eta2 data.
    """
    x, y, gr0, _ = read_matlab(filename)

    plt.figure()
    # Use pcolormesh for proper x/y mapping
    mesh = plt.pcolormesh(x, y, gr0, shading='auto', cmap='viridis')
    cbar = plt.colorbar(mesh)
    cbar.set_label(r'$\eta_0$', fontsize=12)
    plt.contour(x, y, gr0, levels=[level], colors='red', linewidths=2)
    plt.xlim(x.min(), x.max())
    plt.ylim(y.min(), y.max())

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"gr0 from {filename}")
    plt.axis('equal')
    plt.tight_layout()
    plt.show()

def grid_to_quads(x, y, c):
    """
    Convert a node-centered rectilinear grid (x,y,c) of shape (N,N) into QUAD4 arrays.

    Parameters
    ----------
    x, y : ndarray, shape (N, N)
        Node coordinates (as returned by read_matlab), uniformly or non-uniformly spaced.
    c : ndarray, shape (N, N)
        Field values at the same nodes.

    Returns
    -------
    x4, y4, c4 : ndarray, shape (n_elem, 4)
        Quad corner coordinates/values ordered [LL, LR, UR, UL] per element.
        Here n_elem = (N-1) * (N-1).
    """
    if x.shape != y.shape or x.shape != c.shape:
        raise ValueError("x, y, c must all have the same (N, N) shape")

    # Corners for each cell
    # LL(i,j) = (i, j)
    LLx = x[:-1, :-1];  LLy = y[:-1, :-1];  LLc = c[:-1, :-1]
    # LR(i,j) = (i, j+1)
    LRx = x[:-1, 1:];   LRy = y[:-1, 1:];   LRc = c[:-1, 1:]
    # UR(i,j) = (i+1, j+1)
    URx = x[1:,  1:];   URy = y[1:,  1:];   URc = c[1:,  1:]
    # UL(i,j) = (i+1, j)
    ULx = x[1:,  :-1];  ULy = y[1:,  :-1];  ULc = c[1:,  :-1]

    # Stack corners in consistent CCW order: [LL, LR, UR, UL]
    x4 = np.stack([LLx, LRx, URx, ULx], axis=-1).reshape(-1, 4)
    y4 = np.stack([LLy, LRy, URy, ULy], axis=-1).reshape(-1, 4)
    c4 = np.stack([LLc, LRc, URc, ULc], axis=-1).reshape(-1, 4)

    return x4, y4, c4


def closest_frame(MF,target):
    t_list = MF.global_times
    idx = min(range(len(t_list)), key=lambda i: abs(t_list[i] - target))
    closest_time = t_list[idx]
    return closest_time, idx


def format_elapsed_time(start_time):
    # Get the current time
    end_time = time.perf_counter()
    # Calculate elapsed time in seconds
    elapsed_time = end_time - start_time
    # Convert elapsed time to hours, minutes, and seconds
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    # seconds = int(elapsed_time % 60)
    seconds = (elapsed_time % 60)
    # Return formatted elapsed time as a string
    # return f"{hours:02}:{minutes:02}:{seconds:02}"
    return f"{hours:02}:{minutes:02}:{seconds:05.2f}"

def polygon_area(coords):
    x, y = coords[:,0], coords[:,1]
    return 0.5*np.abs(np.dot(x, np.roll(y,-1)) - np.dot(y, np.roll(x,-1)))


# Make sure exodus is in quad4 if it isnt already
def element_order(xk):
    """
    Infer per-element node count from xk shape (n_elem, k).
    Returns k (e.g., 4, 8, 9, ...).
    """
    if xk.ndim != 2:
        raise ValueError(f"xk must be (n_elem, k); got shape {xk.shape}")
    return xk.shape[1]

def _corners_by_extremes(px, py):
    """
    Given all nodes of one element (px,py) shape (k,), pick 4 corner indices
    using diagonal extremes and return them ordered [LL, LR, UR, UL].
    Robust for convex quads incl. second-order (8/9 node) quads.
    """
    s1 = px + py           # diagonal NE<->SW
    s2 = px - py           # diagonal SE<->NW

    i_ll = int(np.argmin(s1))  # lower-left  (min x+y)
    i_ur = int(np.argmax(s1))  # upper-right (max x+y)
    i_lr = int(np.argmax(s2))  # lower-right (max x-y)
    i_ul = int(np.argmin(s2))  # upper-left  (min x-y)

    idx = np.array([i_ll, i_lr, i_ur, i_ul], dtype=int)

    # If any duplicates (degenerate/ties), fall back to angle sort around centroid
    if len(np.unique(idx)) < 4:
        cx, cy = px.mean(), py.mean()
        ang = np.arctan2(py - cy, px - cx)
        # unique points only (keep first occurrence)
        pts = np.column_stack((px, py))
        _, uniq_idx = np.unique(pts, axis=0, return_index=True)
        ux, uy = px[uniq_idx], py[uniq_idx]
        uang = np.arctan2(uy - cy, ux - cx)
        order = np.argsort(uang)                   # CCW
        if order.size < 4:
            raise ValueError("Cannot identify 4 unique corners for this element.")
        cand = uniq_idx[order][:4]                 # take 4 hull-ish extremes
        # rotate so first is LL (min x+y), and ensure CCW
        start = np.argmin((px[cand] + py[cand]))
        idx = np.roll(cand, -start)
        # enforce CCW
        area = 0.5*np.sum(
            px[idx]*py[np.roll(idx,-1)] - py[idx]*px[np.roll(idx,-1)]
        )
        if area < 0:
            idx = idx[[0,3,2,1]]                   # reverse to make CCW

    return idx  # [LL, LR, UR, UL]

def ensure_quad4(xk, yk, ck):
    """
    Coerce any per-element node list to QUAD4 corners in [LL, LR, UR, UL] order.

    Parameters
    ----------
    xk, yk, ck : (n_elem, k) arrays
        Node coordinates/values for each element. k may be 4, 8, 9, ...

    Returns
    -------
    x4, y4, c4 : (n_elem, 4) arrays
        Corner-only arrays in consistent order [LL, LR, UR, UL].
    """
    if not (xk.shape == yk.shape == ck.shape):
        raise ValueError("xk, yk, ck must have identical shape (n_elem, k)")

    nE, k = xk.shape
    if k == 4:
        # Optionally re-order to [LL,LR,UR,UL] for safety:
        idxs = np.zeros((nE, 4), dtype=int)
        for e in range(nE):
            idxs[e] = _corners_by_extremes(xk[e], yk[e])
        x4 = np.take_along_axis(xk, idxs, axis=1)
        y4 = np.take_along_axis(yk, idxs, axis=1)
        c4 = np.take_along_axis(ck, idxs, axis=1)
        return x4, y4, c4

    # k != 4 → pick corners
    idxs = np.zeros((nE, 4), dtype=int)
    for e in range(nE):
        idxs[e] = _corners_by_extremes(xk[e], yk[e])

    x4 = np.take_along_axis(xk, idxs, axis=1)
    y4 = np.take_along_axis(yk, idxs, axis=1)
    c4 = np.take_along_axis(ck, idxs, axis=1)
    return x4, y4, c4


def _dedupe_nodes(x4, y4, tol=1e-12):
    """
    x4, y4: (n_elem, 4) arrays of quad corner coordinates
    Returns:
      XY: (n_nodes,2) unique node coords
      idx4: (n_elem,4) integer node indices into XY for each quad corner
    """
    # stack all corners, round to a tolerance so identical corners match exactly
    pts = np.column_stack([x4.ravel(), y4.ravel()])
    pts_key = np.round(pts / tol)  # integer-ish keys
    # Build mapping unique -> index
    _, inv, counts = np.unique(pts_key, axis=0, return_inverse=True, return_counts=True)
    XY = np.zeros((counts.size, 2))
    # The unique coords are the first appearance of each key
    unique_keys, unique_idx = np.unique(inv, return_index=True)
    XY[unique_keys] = pts[unique_idx]
    idx = inv.reshape(x4.shape)  # (n_elem, 4)
    return XY, idx

def quads_to_tris(idx4):
    """
    idx4: (n_elem,4) node indices for [n0, n1, n2, n3]
          assumed ordered around the quad (e.g., LL, LR, UR, UL)
    Returns triangles array (n_tris, 3) with two tris per quad.
    """
    # nE = idx4.shape[0]
    # Split each quad into (n0,n1,n2) and (n0,n2,n3) (consistent CCW)
    t1 = idx4[:, [0,1,2]]
    t2 = idx4[:, [0,2,3]]
    tris = np.vstack([t1, t2])
    return tris

def extract_iso_contour_from_quads(x4, y4, c4, level):
    """
    x4, y4, c4: (n_elem,4) arrays
    level: isocontour value for c
    Returns: list of (N_i,2) arrays of contour polylines
    """
    XY, idx4 = _dedupe_nodes(x4, y4, tol=1e-12)
    tris = quads_to_tris(idx4)
    # Interpolate c to nodes by averaging element-corner contributions
    # (since corners are shared, a simple average is fine)
    n_nodes = XY.shape[0]
    acc_c = np.zeros(n_nodes)
    acc_w = np.zeros(n_nodes)

    flat_idx = idx4.ravel()
    flat_c   = c4.ravel()
    np.add.at(acc_c, flat_idx, flat_c)
    np.add.at(acc_w, flat_idx, 1.0)
    c_nodes = acc_c / np.maximum(acc_w, 1)

    tri = mtri.Triangulation(XY[:,0], XY[:,1], triangles=tris)
    cs = plt.tricontour(tri, c_nodes, levels=[level])
    # paths = cs.collections[0].get_paths()
    paths = cs.allsegs[0]
    plt.close()

    # contours = [p.vertices for p in paths if len(p.vertices) >= 3]
    contours = [np.column_stack([seg[:,0], seg[:,1]]) for seg in paths if len(seg) >= 3]
    return contours



def plotting_helper(x4, y4, c4):
    XY, idx4 = _dedupe_nodes(x4, y4)
    tris = quads_to_tris(idx4)

    # average element-corner values to nodes
    n_nodes = XY.shape[0]
    acc_c = np.zeros(n_nodes)
    acc_w = np.zeros(n_nodes)
    np.add.at(acc_c, idx4.ravel(), c4.ravel())
    np.add.at(acc_w, idx4.ravel(), 1.0)
    c_nodes = acc_c / np.maximum(acc_w, 1.0)

    tri = mtri.Triangulation(XY[:, 0], XY[:, 1], triangles=tris)
    return XY, tri, c_nodes

def draw_iso_contour(ax, tri, c_nodes, level):
    cs = ax.tricontour(tri, c_nodes, levels=[level], colors='red', linewidths=2)
    return cs


def plot_combined(i, e_name, c_level=args.level, t_vals=(5, 10, 15, 20),
                  matlab_base="01_matlab/results"):
    for t_val in t_vals:
        MF = MultiExodusReaderDerivs(e_name)
        ti, idx = closest_frame(MF, t_val)

        # ---- Exodus ----
        xe, ye, ze, clist = MF.get_full_vars_at_time(['gr0'], ti)
        ck = clist[0]                # (n_elem, k) with k = 4 or 8 or 9
        x4e, y4e, c4e = ensure_quad4(xe, ye, ck)   # <-- coerce to QUAD4
        XY_e, tri_e, c_nodes_e = plotting_helper(x4e, y4e, c4e)
        verb(f"Exodus NaN ratio: {np.isnan(c_nodes_e).mean():.3f}")

        # ---- MATLAB ----
        m_path = matlab_csv_for(e_name, t_val, base_dir=matlab_base)
        xm, ym, cm, _ = read_matlab(m_path)
        x4m, y4m, c4m = grid_to_quads(xm, ym, cm)
        XY_m, tri_m, c_nodes_m = plotting_helper(x4m, y4m, c4m)
        verb(f"Matlab NaN ratio: {np.isnan(c_nodes_m).mean():.3f}")

        # ---- Plot ----
        fig, ax = plt.subplots(ncols=2, sharex=True, sharey=True, figsize=(8, 3.5))

        tpc = ax[0].tripcolor(tri_e, c_nodes_e, shading="gouraud",
                              cmap='viridis', vmin=0, vmax=1)
        fig.colorbar(tpc, ax=ax[0], label=r"$\eta_0$")
        draw_iso_contour(ax[0], tri_e, c_nodes_e, c_level)
        ax[0].set_title(f"Exodus @ {t_val}s")

        tpcm = ax[1].tripcolor(tri_m, c_nodes_m, shading="gouraud",
                               cmap='viridis', vmin=0, vmax=1)
        fig.colorbar(tpcm, ax=ax[1], label=r"$\eta_0$")
        draw_iso_contour(ax[1], tri_m, c_nodes_m, c_level)
        ax[1].set_title(f"MATLAB @ {t_val}s")

        # unified limits from both datasets
        xmin = min(XY_e[:,0].min(), XY_m[:,0].min())
        xmax = max(XY_e[:,0].max(), XY_m[:,0].max())
        ymin = min(XY_e[:,1].min(), XY_m[:,1].min())
        ymax = max(XY_e[:,1].max(), XY_m[:,1].max())
        for a in ax:
            a.set_xlim(xmin, xmax)
            a.set_ylim(ymin, ymax)
            a.set_aspect("equal")
            a.set_xlabel("x")
        ax[0].set_ylabel("y")
        ax[1].set_ylabel("y")

        fig.tight_layout()
        out = imdir + f'/P0{i+1}a_combined_gr0_contour_{t_val:02d}s.png'
        plt.savefig(out, dpi=500, transparent=True)  # no transparency while debugging
        plt.close(fig)
        verb(f"Wrote {out}")

        # ---- Stacked Contour ----
        extent = (xmin, xmax, ymin, ymax)
        series = [(tri_e, c_nodes_e), (tri_m, c_nodes_m)]
        labels = ["Exodus", "MATLAB"]

        plot_contour_overlay(
            series=series,
            level=c_level,
            labels=labels,
            colors=["crimson", "dodgerblue"],
            extent=extent,
            title=None,#f"gr0 contour @ {t_val}s (Exodus vs MATLAB)",
            out=imdir + f'/P0{i+1}b_stacked_gr0_contour_{t_val:02d}s.png'
            )

        out2c = plot_two_level_contours(
            i=i,
            tri_e=tri_e, c_nodes_e=c_nodes_e,
            tri_m=tri_m, c_nodes_m=c_nodes_m,
            levels=(0.01, 0.99),
            extent=extent,
            t_val=t_val,
            out_dir=imdir,
            fname_stub="gr0"
        )
        verb(f"Wrote {out2c}")


def plot_contour_overlay(series, level, labels=None, colors=None,
                         extent=None, title=None, out=None, ax=None):
    """
    Plot one overlay figure with multiple iso-contours (no colormap).

    Parameters
    ----------
    series : list[tuple]
        Each item is (tri, c_nodes), e.g. [(tri_e, c_nodes_e), (tri_m, c_nodes_m)].
    level : float
        Contour value.
    labels : list[str], optional
        Legend labels for each contour series (len == len(series)).
    colors : list[str], optional
        Matplotlib colors for each contour series (len == len(series)).
    extent : tuple(xmin, xmax, ymin, ymax), optional
        Axes limits to enforce. If None, inferred from all series.
    title : str, optional
        Figure title.
    out : str or Path, optional
        If given, save to this path.
    ax : matplotlib.axes.Axes, optional
        If provided, draw on this axes; otherwise create a new figure.

    Returns
    -------
    fig, ax : the figure and axes used.
    """
    # Defaults
    n = len(series)
    if labels is None:
        labels = [f"set {i+1}" for i in range(n)]
    if colors is None:
        colors = ["crimson", "dodgerblue", "seagreen", "orange"][:n]

    # Figure/axes
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
        created_fig = True
    else:
        fig = ax.figure

    # Draw contours
    for (tri, c_nodes), lab, col in zip(series, labels, colors):
        ax.tricontour(tri, c_nodes, levels=[level], colors=[col], linewidths=2)#, label=lab)

    # Limits/aspect
    if extent is None:
        xs = []; ys = []
        for tri, _ in series:
            xs.append(tri.x); ys.append(tri.y)
        xmin = min(x.min() for x in xs); xmax = max(x.max() for x in xs)
        ymin = min(y.min() for y in ys); ymax = max(y.max() for y in ys)
        extent = (xmin, xmax, ymin, ymax)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    if title: ax.set_title(title)

    # Legend
    handles = [Line2D([0], [0], color=c, lw=2, label=l) for c, l in zip(colors, labels)]
    ax.legend(handles=handles, loc="best", frameon=True)

    fig.tight_layout()

    if out:
        fig.savefig(out, dpi=500, transparent=True)  # avoid transparency while debugging
    return fig, ax


def plot_two_level_contours(i,
                            tri_e, c_nodes_e,
                            tri_m, c_nodes_m,
                            levels=(0.01, 0.99),
                            extent=None,
                            t_val=None,
                            out_dir="pics",
                            fname_stub="gr0"):
    """
    Make a 2-column figure (Exodus | MATLAB) showing only two iso-contours.

    Parameters
    ----------
    i : int
        Index for output naming (P0{i+1}...).
    tri_e, c_nodes_e : Triangulation, ndarray
        Exodus triangulation and nodal field.
    tri_m, c_nodes_m : Triangulation, ndarray
        MATLAB triangulation and nodal field.
    levels : tuple(float, float)
        Two contour levels to draw (default: (0.01, 0.99)).
    extent : (xmin, xmax, ymin, ymax) or None
        Axes limits. If None, inferred from both triangulations.
    t_val : int or float or None
        Time value for titles/filename tag. If None, no time in title.
    out_dir : str
        Directory to save the figure.
    fname_stub : str
        Text to include in filename (e.g., 'gr0', 'gr1').

    Returns
    -------
    out_path : str
        Saved file path.
    """
    # Colors (level 0.01, level 0.99)
    lvl_colors = ("dodgerblue", "crimson")
    lvl_labels = (rf"$\eta_0={levels[0]}$", rf"$\eta_0={levels[1]}$")

    # Create 2-column figure
    fig, ax = plt.subplots(ncols=2, sharex=True, sharey=True, figsize=(8, 3.5))

    # Exodus (left)
    ax[0].tricontour(tri_e, c_nodes_e, levels=levels, colors=lvl_colors, linewidths=2)
    ax[0].set_title(f"Exodus{'' if t_val is None else f' @ {t_val}s'}")

    # MATLAB (right)
    ax[1].tricontour(tri_m, c_nodes_m, levels=levels, colors=lvl_colors, linewidths=2)
    ax[1].set_title(f"MATLAB{'' if t_val is None else f' @ {t_val}s'}")

    # Limits/aspect
    if extent is None:
        xs = np.concatenate([tri_e.x, tri_m.x])
        ys = np.concatenate([tri_e.y, tri_m.y])
        extent = (xs.min(), xs.max(), ys.min(), ys.max())
    for a in ax:
        a.set_xlim(extent[0], extent[1])
        a.set_ylim(extent[2], extent[3])
        a.set_aspect("equal")
        a.set_xlabel("x")
    ax[0].set_ylabel("y")
    ax[1].set_ylabel("y")

    # Legend: two colored lines (one per level)
    handles = [Line2D([0], [0], color=c, lw=2, label=lab)
               for c, lab in zip(lvl_colors, lvl_labels)]
    ax[1].legend(handles=handles, loc="best", frameon=True)

    fig.tight_layout()

    # Filename like: P0{i+1}c_twoContours_gr0_05s.png
    if t_val is None:
        ttag = "NA"
    else:
        # zero-pad ints; for floats you can adjust as needed
        ttag = f"{int(round(float(t_val))):02d}"
    out_path = os.path.join(out_dir, f"P0{i+1}c_IW_{fname_stub}_{ttag}s.png")
    fig.savefig(out_path, dpi=500, transparent=True)
    plt.close(fig)
    return out_path





# ███╗   ███╗ █████╗ ██╗███╗   ██╗
# ████╗ ████║██╔══██╗██║████╗  ██║
# ██╔████╔██║███████║██║██╔██╗ ██║
# ██║╚██╔╝██║██╔══██║██║██║╚██╗██║
# ██║ ╚═╝ ██║██║  ██║██║██║ ╚████║
# ╚═╝     ╚═╝╚═╝  ╚═╝╚═╝╚═╝  ╚═══╝

#IF IN MAIN PROCESS
if __name__ == "__main__":
    tracemalloc.start()
    all_ti = time.perf_counter()
    # Read the files
    try:
        matlab_paths, exodus_paths = find_files()
        verb("✅ All files found.")
        verb(f'  MATLAB CSVs: {matlab_paths}')
        verb(f'  Exodus files: {exodus_paths}')
    except FileNotFoundError as e:
        verb("❌ File check failed!")
        verb(e)
    printnames = ['m = 2', 'm = 4', 'm = 6']
    #
    # for i,(m_name,e_name) in enumerate(zip(matlab_paths,exodus_paths)):
    for i,e_name in enumerate(exodus_paths):
        pt(' ')#\x1b[31;1m
        pt('\033[1m\033[96m'+'File '+str(i+1)+'/3: '+'\x1b[0m'+str(printnames[i]))
        plot_combined(i,e_name,c_level=args.level,t_vals=(5, 10, 15))
    #
    pt(' ')
    pt(f'Done Everything: {format_elapsed_time(all_ti)}')
    current, peak =  tracemalloc.get_traced_memory()
    pt('Memory after everything (current, peak): '+str(round(current/1048576,1))+', '+str(round(peak/1048576,1))+' MB')
    pt(' ')

    quit()
