from MultiExodusReader import MultiExodusReader
from MultiExodusReaderDerivs import MultiExodusReaderDerivs
# import multiprocessing as mp
# from VolumeScripts import *

import subprocess
from joblib import Parallel, delayed

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PolyCollection
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
import re
from enum import Enum
from tqdm import tqdm
import fnmatch

from scipy.spatial import cKDTree
import networkx as nx


# Defaults for the variables
# WARNING: manually listed nodal vs elemental options for now! should try to add that to MER later
class default_vals:
    cpus = 2
    n_frames = 300
    cutoff = 0.0
    bins = 20

var = 'unique_grains'


# CL Argument Parser
parser = argparse.ArgumentParser()
parser.add_argument('--verbose', '-v', action='count', default=0,
                    help='Increase verbosity: -v for INFO, -vv for DEBUG.')
parser.add_argument('--cpus','-n',type=int, default=default_vals.cpus,
                            help='How many cpus, default='+str(default_vals.cpus))
parser.add_argument('--bins','-b',type=int, default=default_vals.bins,
                            help='How many bins for histogram, default='+str(default_vals.bins))
parser.add_argument('--dim','-d',type=int,default=2, choices=[2,3],
                            help='Dimensions for grain size calculation (Default=2)')
parser.add_argument('--subdirs','-s',action='store_true',
                            help='Run in all subdirectories (vs CWD), default=False')
parser.add_argument('--esd',action='store_true',
                            help='''Return ESD (vol or area based if 3D/2D) instead of grain
                            area/volume, default=False''')
parser.add_argument('--sequence',action='store_true',
                            help='Time as a sequence, default=False')
parser.add_argument('--n_frames','-f',type=int, default=default_vals.n_frames,
                            help='''How many frames for if sequence is true, '''
                            '''default='''+str(default_vals.n_frames))
parser.add_argument('--cutoff','-c',type=int, default=default_vals.cutoff,
                            help='''What time to stop at, if 0.0 uses all data. '''
                            '''default='''+str(default_vals.cutoff))
parser.add_argument('--exo','-e',action='store_false',
                            help='Look for and use Exodus files instead of Nemesis, default=True')
parser.add_argument('--skip', nargs='+', required=False, help='List of text flags to skip')
parser.add_argument('--only', nargs='+', required=False, help='List of text flags to use')
cl_args = parser.parse_args()



# LOGGING
def configure_logging(args):
    if args.verbose == 2:
        logging.basicConfig(level=logging.DEBUG, format='%(message)s')
    elif args.verbose == 1:
        logging.basicConfig(level=logging.INFO, format='%(message)s')
    else:
        logging.basicConfig(level=logging.WARNING, format='%(message)s')

# Configure logging based on verbosity level
configure_logging(cl_args)

# Logging functions
pt = logging.warning
verb = logging.info
db = logging.debug

# Example usage
verb('Verbose Logging Enabled')
db('Debug Logging Enabled')
db('''INFO: This is set up to read nemesis (and maybe exodus too?) files and just use
   unique_grains and the element volume/area to determine the average grain size (ESD)
   with some amount of statistics.''')
db('''WARNING: This script assumes elements are the same size in x/y(/z).''')
db(' ')
# pt('This is a warning.')
db(f'Command-line arguments: {cl_args}')
pt(' ')


cwd = os.getcwd()

imdir = 'pics'
if not os.path.isdir(imdir):
    db('Making picture directory: '+imdir)
    os.makedirs(imdir)

# times_files = np.load('times_files.npy')
# times = times_files[:,0].astype(float)
# t_step = times_files[:,2].astype(int)



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



# def filter_filenames(filenames, skip_flags):
#     """Filter filenames based on the skip_flags and return the filtered list."""
#     if skip_flags is None:
#         # If no skip flags are provided, return the filenames as-is
#         return filenames

#     filtered_filenames = []
#     for filename in filenames:
#         # Check if any skip_flag is in the filename, if not, keep it
#         if not any(flag in filename for flag in skip_flags):
#             filtered_filenames.append(filename)
#     return filtered_filenames


def filter_filenames(filenames, only_flags=None, skip_flags=None):
    """
    Filter filenames based on the only_flags and skip_flags.

    - If only_flags is provided, only include filenames that contain at least one of the flags.
    - If skip_flags is provided, exclude filenames that contain any of the flags.
    """
    # If `only_flags` is specified, filter filenames to include only those matching any flag
    if only_flags:
        filenames = [filename for filename in filenames if any(flag in filename for flag in only_flags)]

    # If `skip_flags` is specified, filter out filenames matching any flag
    if skip_flags:
        filenames = [filename for filename in filenames if not any(flag in filename for flag in skip_flags)]

    return filenames



def find_files():
    """
    Find files with extension '.e*' in the current directory or subdirectories.


    Returns:
        list: Sorted list of file names with '.e*' extension.

    Raises:
        ValueError: If no files matching the pattern '*.e*' are found.
    """
    e_names = []
    if cl_args.subdirs:
        for dir_n in glob.glob('*/', recursive=True):
            if cl_args.exo:
                e_files_in_subdir = glob.glob(dir_n + '*.e')
                if e_files_in_subdir:
                    first_file = e_files_in_subdir[0]
                    # trimmed_file = first_file.split('.e', 1)[0] + '.e*'
                    trimmed_file = first_file #+ '.e*'
                    e_names.append(trimmed_file)
            else:
                e_files_in_subdir = [x.rsplit('.',1)[0]+"*" for x in glob.glob(dir_n + "*.e.*")]
                # e_files_in_subdir = glob.glob(dir_n + '*.e*')
                if e_files_in_subdir:
                    first_file = e_files_in_subdir[0]
                    # trimmed_file = first_file.split('.e', 1)[0] + '.e*'
                    trimmed_file = first_file #+ '.e*'
                    e_names.append(trimmed_file)
    else:
        if cl_args.exo:
            e_files_in_subdir = glob.glob('*.e')
            if e_files_in_subdir:
                first_file = e_files_in_subdir[0]
                # trimmed_file = first_file.split('.e', 1)[0] + '.e*'
                trimmed_file = first_file #+ '.e*'
                e_names.append(trimmed_file)
        else:
            e_files_in_dir = [x.rsplit('.',1)[0]+"*" for x in glob.glob("*.e.*")]
            if e_files_in_dir:
                first_file = e_files_in_dir[0]
                trimmed_file = first_file #.split('.e', 1)[0] + '.e*'
                e_names.append(trimmed_file)
    if not e_names:
        raise ValueError('No files found matching *.e*, make sure to specify subdirectories or not')
    # Skip if specified any flags for files to ignore
    e_names = filter_filenames(e_names, only_flags=cl_args.only, skip_flags=cl_args.skip)
    e_names.sort(key=natural_sort_key)
    verb('Files to use: ')
    verb(e_names)
    verb(' ')
    return e_names


def time_info(MF):
    '''Returns list of times for each frame from MultiExodusReader object

    Args:
        MF: MultiExodusReader object with * for all .e*files

    Raises:
        ValueError: Sequence cl_arg value issue (T/F)

    Returns:
        idx_frames: List of frame iterations/numbers
        t_frames: List of time values associated with each frame (idx_frames)
    '''
    times = MF.global_times
    if cl_args.sequence == True:
        # if cl_args.n_frames < len(times):
        t_max = times[-1]
        # t_max = max(times)
        t_frames =  np.linspace(0.0,t_max,cl_args.n_frames)
        idx_frames = [ np.where(times-t_frames[i] == min(times-t_frames[i],key=abs) )[0][0] for i in range(cl_args.n_frames) ]
        idx_frames = list( map(int, idx_frames) )
        # For sequence output use the actual time of each frame not the seq time
        t_frames = [times[n] for n in idx_frames]
        # else:
        #     t_frames = times
        #     idx_frames = range(len(times))
    elif cl_args.sequence == False:
        t_frames = times
        idx_frames = range(len(times))
    else:
        raise ValueError('sequence has to be True or False, not: ' + str(cl_args.sequence))

    if cl_args.cutoff != 0:
        verb("Cutting End Time to ",cl_args.cutoff)
        t_frames = [x for x in t_frames if x <= cl_args.cutoff]
        idx_frames = range(len(t_frames))

    t_frames_array = np.asarray(t_frames)
    # tot_frames = len(idx_frames)
    return idx_frames, t_frames_array

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

def plotit(pltx, plty, pltc, cname, times,i):
    coords = np.asarray([ np.asarray([x_val,y_val]).T for (x_val,y_val) in zip(pltx, plty) ])
    fig, ax = plt.subplots()
    p = PolyCollection(coords, cmap=matplotlib.cm.coolwarm, alpha=1)#,edgecolor='k'
    p.set_array(np.array(pltc) )
    ax.add_collection(p)
    fig.colorbar(p, label=cname)
    ax.set_xlim([np.amin(pltx),np.amax(pltx)])
    ax.set_ylim([np.amin(plty),np.amax(plty)])
    ax.set_aspect('equal')
    timestring = 't = ' + str(times[i])
    ax.set_title(timestring)
    fig.savefig(imdir+'/'+str(cname)+'_'+str(i)+'.png',dpi=500,transparent=True )
    if cl_args.verbose == 2:
        plt.show()
    else:
        plt.close()

def unique_spacing(*coords):
    du = []
    for n in coords:
        # Step 1: Extract unique values
        unique_values = np.unique(n)
        # Check if unique_values has only one element
        if len(unique_values) == 1:
            # Skip this n and move to the next item
            continue
        # Step 2: Sort the unique values (optional here, as np.unique returns a sorted array)
        sorted_unique_values = np.sort(unique_values)
        # Step 3: Compute differences between consecutive unique values
        differences = np.diff(sorted_unique_values)
        # Step 4: Calculate the average distance
        average_distance = np.mean(differences)
        du.append(average_distance)
        # Check if du is empty (all coords had only one unique value)
        if not du:
            raise ValueError("All input coordinates have only one unique value.")
        # Check if all items in du are approximately equal
        if len(du) > 1 and not np.allclose(du, du[0]):
            pt('\x1b[31;1m'+'WARNING:'+'\x1b[0m'+' dx/dy/dz not all equal.')
    return np.mean(du)

def t0_setup(MF,t_frames):
    x,y,z,c = MF.get_opmax_at_time_alt(t_frames[0])
    dx = unique_spacing(x,y,z)
    distance_threshold = dx * 1.1 #1.41 is diag in 2D, 1.73 next diag in 3D
    num_grs = max(c)
    feature_numbers = identify_features(x, y, z, c, distance_threshold)
    num_feats = max(feature_numbers)
    return distance_threshold, dx, num_grs, num_feats


def identify_features(x, y, z, c, distance_threshold):
    """
    Identifies unique features in a 3D point cloud based on spatial proximity and shared 'c' values.
    Assigns feature number 0 to all points where c == 0, and numbers other features starting from 1.

    Parameters:
    - x, y, z: 1D numpy arrays of coordinates.
    - c: 1D numpy array of integer values associated with each point.
    - distance_threshold: Maximum distance to consider points as neighbors.

    Returns:
    - feature_numbers: 1D numpy array of feature numbers assigned to each point.
    """
    # Combine coordinates into points
    points = np.vstack((x, y, z)).T

    num_points = len(points)

    # Initialize feature_numbers array with zeros
    feature_numbers = np.zeros(num_points, dtype=int)

    # Indices where c == 0 and c != 0
    # c_zero_indices = np.where(c == 0)[0]
    c_nonzero_indices = np.where(c != 0)[0]

    # If there are no nonzero c values, return feature_numbers as is
    if len(c_nonzero_indices) == 0:
        return feature_numbers  # All features are 0

    # Build cKDTree for nonzero c points
    points_nonzero = points[c_nonzero_indices]
    c_nonzero = c[c_nonzero_indices]

    tree = cKDTree(points_nonzero)

    # Initialize the graph
    G = nx.Graph()
    num_nonzero_points = len(points_nonzero)
    G.add_nodes_from(range(num_nonzero_points))

    # Build edges between neighboring nodes with the same 'c' value
    for idx in range(num_nonzero_points):
        point = points_nonzero[idx]
        # Find indices of neighbors within the distance threshold
        neighbor_indices = tree.query_ball_point(point, r=distance_threshold)
        for neighbor_idx in neighbor_indices:
            if neighbor_idx > idx and c_nonzero[neighbor_idx] == c_nonzero[idx]:
                G.add_edge(idx, neighbor_idx)

    # Find connected components
    connected_components = list(nx.connected_components(G))

    # Assign feature numbers starting from 1
    for feature_num, component in enumerate(connected_components, start=1):
        for idx in component:
            original_idx = c_nonzero_indices[idx]
            feature_numbers[original_idx] = feature_num

    # c == 0 points remain assigned to feature number 0

    return feature_numbers


def grain_sizes(dx,feature_numbers,diams=False):
    counts = np.bincount(feature_numbers)
    if cl_args.dim == 2:
        mesharea = dx * dx
        featvols = counts * mesharea
        if diams:
            diameters = 2 * np.sqrt(featvols / np.pi)
            return diameters
    elif cl_args.dim == 3:
        meshvol = dx * dx * dx
        featvols = counts * meshvol
        if diams:
            diameters = 2 * ((featvols / ((4 / 3) * np.pi)) ** (1 / 3))
            return diameters
    else:
        raise ValueError(f'Command line dimensions should be 2 or 3, is: {cl_args.dim}')
    return featvols


# def para_feature_volume(file,time,dx,distance_threshold):
#     MF = GrainMultiExodusReader(file)
#     x,y,z,c = MF.get_opmax_at_time_alt(time)
#     feature_numbers = identify_features(x, y, z, c, distance_threshold)
#     featvols = grain_sizes(dx,feature_numbers,cl_args.esd)
#     # if not isinstance(featvols, list):
#     #     featvols = featvols.tolist()
#     result = np.concatenate(([time], featvols))
#     return result


def out_name(file_name):
    """
    Generate an output file name based on the input file name and dimensionality.

    Args:
        file_name (str): The input file name.

    Returns:
        str: The generated output file name.

    Raises:
        ValueError: If the dimension is not 2 or 3.
    """
    suffix = '_grainsize.csv'
    # Beginning based on subdir
    if cl_args.subdirs:
        outBase = file_name.split('/')[0]
    else:
        outBase = os.path.split(os.getcwd())[-1]
    return outBase + suffix


def save_to_csv(volumes, output_filename):
    """
    Save feature volumes to a sorted CSV file with appropriate headers.

    Args:
        volumes (list): List of lists where each sublist corresponds to [time, featvols].
        output_filename (str): Name of the output CSV file.
    """
    # Extract times and feature volumes
    data_dict = {}
    for volume in volumes:
        time = volume[0]
        feature_vols = volume[1:]  # exclude time
        data_dict[time] = feature_vols

    # Create DataFrame
    df = pd.DataFrame.from_dict(data_dict, orient='index')
    df.index.name = "time"
    df = df.sort_index().reset_index()  # Sort by the time index and reset it

    # Add headers: time, grain## where ## is a two-digit number
    max_columns = df.shape[1] - 1  # Exclude 'time'
    headers = ["time", "phi"] + [f"grain{str(i).zfill(2)}" for i in range(1, max_columns)]
    df.columns = headers

    # Save DataFrame to CSV
    df.to_csv(output_filename, index=False)
    verb(f"Results saved to {output_filename}")



def gradient_structured(x, y, f):
    """
    Compute ∂f/∂x and ∂f/∂y on a rectilinear element-centre grid.

    Parameters
    ----------
    x, y : 1-D arrays (n_points,)
        Element-centre coords (must be structured).
    f    : 1-D array  (n_points,)
        Scalar at each element centre.

    Returns
    -------
    dfdx, dfdy : 1-D arrays (n_points,)
    """
    # ➊  Identify distinct grid lines
    xu = np.unique(np.round(x, 12))
    yu = np.unique(np.round(y, 12))
    nx, ny = len(xu), len(yu)

    # ➋  Reshape to (ny, nx) image using row-major ordering
    #     (y changes fastest ⇒ sort-index trick)
    idx_y = np.argsort(y)               # get rows together
    x2d   = x[idx_y].reshape(ny, nx)
    f2d   = f[idx_y].reshape(ny, nx)

    # ➌  Finite differences (np.gradient order = rows, cols)
    dy = np.diff(yu).mean()             # uniform spacing
    dx = np.diff(xu).mean()
    dfdy, dfdx = np.gradient(f2d, dy, dx)

    # ➍  Flatten in the original element order
    dfdx_flat = dfdx.ravel()[np.argsort(idx_y)]
    dfdy_flat = dfdy.ravel()[np.argsort(idx_y)]
    return dfdx_flat, dfdy_flat


def gradient_unstructured_full(x, y, f, k=6):
    """
    Least-squares ∇f on an arbitrary point cloud.

    Returns dfdx, dfdy as 1-D arrays the same size as x.
    """
    coords = np.column_stack((x, y))
    tree   = cKDTree(coords)

    grads  = np.empty((len(x), 2))
    for i, (xi, yi, fi) in enumerate(zip(x, y, f)):
        # first neighbour returned is the point itself → skip it
        _, idx = tree.query([xi, yi], k=k+1)
        nbrs   = idx[1:]

        dx = x[nbrs] - xi
        dy = y[nbrs] - yi
        df = f[nbrs] - fi

        # Solve [dx dy]·[a b]^T ≈ df  ⇒ [a b] = gradient
        A         = np.column_stack((dx, dy))   # (k, 2)
        g, *_     = np.linalg.lstsq(A, df, rcond=None)
        grads[i]  = g

    return grads[:, 0], grads[:, 1]


def gradient_unstructured(x, y, f, *, k=6,
                          exclude_value=None, atol=1e-12,
                          external_mask=None, fill=np.nan):
    """
    Least-squares ∇f on an arbitrary point cloud *with optional exclusion*.

    Parameters
    ----------
    x, y : 1-D arrays (n_points,)
        Element-centre coordinates.
    f    : 1-D array (n_points,)
        Scalar at each point.
    k    : int, default 6
        Number of neighbours for the local plane fit.
    exclude_value : float or None, default None
        If not None, any point with |f - exclude_value| < atol
        is ignored as a neighbour.  Its own gradient is returned
        as *fill*.
    atol : float, default 1e-12
        Absolute tolerance when comparing to *exclude_value*.
    external_mask : boolean array or None
        Alternate way to specify “good” points.  Must be same
        length as *x*.  If given, *exclude_value* is ignored.
    fill : float, default np.nan
        Value assigned to gradients of excluded points.

    Returns
    -------
    dfdx, dfdy : 1-D arrays (n_points,)
        Gradients at every input point; excluded locations hold *fill*.
    """
    n = len(x)
    coords = np.column_stack((x, y))

    # ------------------------------------------------------------
    # 1)  Determine which points are "good"
    # ------------------------------------------------------------
    if external_mask is not None:
        good = external_mask.astype(bool)
    elif exclude_value is not None:
        good = np.abs(f - exclude_value) > atol
    else:
        good = np.ones(n, dtype=bool)

    if good.sum() < k + 1:
        raise ValueError("Not enough good points to perform k-NN fits.")

    # ------------------------------------------------------------
    # 2)  Build KD-tree on *only* good points
    # ------------------------------------------------------------
    tree     = cKDTree(coords[good])
    good_ids = np.flatnonzero(good)          # map local→global indices

    # Prepare outputs
    dfdx = np.full(n, fill, dtype=float)
    dfdy = np.full(n, fill, dtype=float)

    # ------------------------------------------------------------
    # 3)  Loop over good points and fit local plane
    # ------------------------------------------------------------
    for local_i, global_i in enumerate(good_ids):
        xi, yi, fi = x[global_i], y[global_i], f[global_i]

        # query returns *k+1* because first neighbour is the point itself
        _, loc_idx = tree.query([xi, yi], k=k+1)
        nbr_idx    = loc_idx[1:]             # strip self

        # Map local neighbour ids back to global
        nbr_global = good_ids[nbr_idx]

        dx = x[nbr_global] - xi
        dy = y[nbr_global] - yi
        df = f[nbr_global] - fi

        A     = np.column_stack((dx, dy))    # (k, 2)
        g, *_ = np.linalg.lstsq(A, df, rcond=None)
        dfdx[global_i], dfdy[global_i] = g

    return dfdx, dfdy


def inc_f(theta,delta,theta0,thetapre=4):
    return 1 + delta * np.cos(thetapre*(theta + theta0))

def rose_curve(theta, n_bins=180, mode='probability'):
    """
    Return bin centres (rad) and bin heights for a rose diagram.

    mode = 'probability'  → heights sum to 1
           'density'      → area = 1    (like np.histogram(..., density=True))
           'count'        → raw counts
           'max'          → heights in [0,1] (divide by max)
    """
    theta = np.mod(theta, 2*np.pi)

    edges   = np.linspace(0, 2*np.pi, n_bins + 1)
    counts, _ = np.histogram(theta, bins=edges)

    if mode == 'probability':
        heights = counts / counts.sum()
    elif mode == 'density':
        widths  = np.diff(edges)
        heights = counts / (counts.sum() * widths)   # 1/(rad)
    elif mode == 'max':
        heights = counts / counts.max()
    else:                                           # 'count'
        heights = counts

    centres = (edges[:-1] + edges[1:]) / 2          # one centre per bin

    # close the curve so the last point meets the first
    centres  = np.hstack([centres,  centres[0]])
    heights  = np.hstack([heights,  heights[0]])
    return centres, heights


def pplot(incx,incy,lbl,t_frames,i):
    # Anisotropy Function
    full_deg = np.linspace(0, 360, 361)
    full_rad = np.deg2rad(full_deg)
    ref_inc = inc_f(full_rad,0.05,0,2)
    iso_inc = np.ones_like(ref_inc)
    # Inclination
    theta = np.arctan2(incy, incx)
    theta = np.mod(theta, 2*np.pi)
    ang, rad = rose_curve(theta, n_bins=cl_args.bins, mode='probability')
    # Weight aniso function
    ref_inc = ref_inc * ((max(rad) - min(rad))/2)
    iso_inc = iso_inc * ((max(rad) - min(rad))/2)
    # Plot
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.plot(full_rad, iso_inc, label='Isotropic Dist')
    ax.plot(full_rad, ref_inc, label='Aniso Function')
    ax.plot(ang, rad, label='Inclination Dist')
    ax.set_theta_zero_location('E')
    ax.set_theta_direction(1)
    ax.set_rlim(0)
    ax.set_rlabel_position(22.5)
    ax.grid(True)
    ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.15))
    timestring = 't = ' + str(t_frames[i])
    ax.set_title(timestring)
    fig.savefig(imdir+'/'+str(lbl)+'_'+str(i)+'.png',dpi=500,transparent=True )
    plt.close()




# ███╗   ███╗ █████╗ ██╗███╗   ██╗
# ████╗ ████║██╔══██╗██║████╗  ██║
# ██╔████╔██║███████║██║██╔██╗ ██║
# ██║╚██╔╝██║██╔══██║██║██║╚██╗██║
# ██║ ╚═╝ ██║██║  ██║██║██║ ╚████║
# ╚═╝     ╚═╝╚═╝  ╚═╝╚═╝╚═╝  ╚═══╝

#IF IN MAIN PROCESS
if __name__ == "__main__":
    tracemalloc.start()
    file_names = find_files()
    num_files = len(file_names)
    all_ti = time.perf_counter()
    for cnt,file_name in enumerate(file_names):
        pt(' ')#\x1b[31;1m
        pt('\033[1m\033[96m'+'File '+str(cnt+1)+'/'+str(num_files)+': '+'\x1b[0m'+str(file_name))
        verb('Initialization for file: '+str(file_name))
        init_ti = time.perf_counter()
        MF = MultiExodusReaderDerivs(file_name)
        idx_frames, t_frames = time_info(MF)
        varnames = ['inclination_vector_x','inclination_vector_y','ang_dist']
        MF.check_varlist(varnames)
        name_base = file_name.rsplit(".", 1)[0]
        csv_rows = []
        for i,ti in enumerate(tqdm(t_frames, desc='Timestepping')):
            x, y, z, clist, tx, ty, tz = MF.get_vars_at_time(varnames,ti,fullxy=True)
            v0 = clist[0]
            v1 = clist[1]
            # mask = (v0 != 0) | (v1 != 0)
            mask = clist[2] >= 0.0 # != -1
            clist_filtered1 = [arr[mask] for arr in clist]
            # also check magnitude
            ix, iy = clist_filtered1[:2]
            mag = np.sqrt(ix**2 + iy**2)
            mask2 = mag >= 0.75
            clist_filtered = [arr[mask2] for arr in clist_filtered1]
            incx = clist_filtered[0]
            incy = clist_filtered[1]
            adist = clist_filtered[2]
            pplot(incx,incy,'Inclination',t_frames,i)

            # Save a csv
            for j, (ix, iy, ad) in enumerate(zip(incx, incy, adist)):
                csv_rows.append({
                    "time_step":  i,
                    "time":       ti,
                    "index":      j,
                    "incx":       ix,
                    "incy":       iy,
                    "adist":      ad
                })


        #     # Plotting
        #     fx = tx[:, :4]
        #     fy = ty[:, :4]
        #     plotit(fx,fy,clist[0],'IncMat',t_frames,i)
            # for n,vname in enumerate(varnames):
            #     plotit(fx,fy,clist[n],vname,t_frames,i)
            #     print(" "+str(n))
        # x, y, z, clist, tx, ty, tz = MF.get_vars_at_time(varnames,t_frames[1],fullxy=True)
        # v0 = clist[0]
        # v1 = clist[1]
        # # mask = (v0 != 0) | (v1 != 0)
        # mask = clist[2] >= 0.0 # != -1
        # clist_filtered = [arr[mask] for arr in clist]
        # incx = clist_filtered[0]
        # incy = clist_filtered[1]
        # adist = clist_filtered[2]
        # pplot(incx,incy,'Inclination',t_frames,i)
        # Plot
        # fx = tx[:, :4]
        # fy = ty[:, :4]
        # plotit(fx[mask],fy[mask],adist,'ADist',t_frames,1)


        # theta = np.arctan2(incy, incx)           # −π … +π
        # theta = np.mod(theta, 2*np.pi)           # wrap into 0 … 2π
        # fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})

        # # example: three inclination data sets
        # data_sets = [theta]#, theta_set2, theta_set3]
        # labels    = ['slice A']#, 'slice B', 'slice C']

        # for theta_arr, lab in zip(data_sets, labels):
        #     ang, rad = rose_curve(theta_arr, n_bins=24, mode='probability')
        #     ax.plot(ang, rad, label=lab)           # connect the bin tops

        # # cosmetics (same as in your screenshot)
        # ax.set_theta_zero_location('E')            # 0° = up
        # ax.set_theta_direction(1)                 # clockwise
        # ax.set_rlim(0)                             # radial axis starts at 0
        # ax.set_rlabel_position(22.5)               # move r-tick labels so they don’t overlap
        # ax.grid(True)
        # ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.15))

        # plt.show()
        # plt.show()

        # # plt.figure()
        # # # plt.polar(theta_rad, inc_f(theta_rad,0,0), c='black', label='Iso')
        # # # plt.polar(theta_rad, inc_f(theta_rad,0.05,0,2), c='fuchsia', label='Aniso')

        # # plt.bar(edges[:-1], counts, width=widths, align='edge', bottom=0)

        # # # plt.ylim(0.75, 1.25)
        # # # plt.yticks([0.75,1,1.25])
        # # # plt.legend(loc='upper right')
        # # angle = np.deg2rad(67.5)
        # # plt.legend(loc="lower left",
        # #         bbox_to_anchor=(.5 + np.cos(angle)/2, .5 + np.sin(angle)/2))
        # # plt.savefig('P03_voronoi_aniso_polar',transparent=True)
        # # plt.close('all')

        # Save csv
        odf = pd.DataFrame(csv_rows)
        outcsv_name = name_base + "_inclination.csv"
        odf.to_csv(outcsv_name, index=False)

        pt(f'Done File {cnt+1}: {format_elapsed_time(init_ti)}')

    pt(' ')
    pt(f'Done Everything: {format_elapsed_time(all_ti)}')
    current, peak =  tracemalloc.get_traced_memory()
    pt('Memory after everything (current, peak): '+str(round(current/1048576,1))+', '+str(round(peak/1048576,1))+' MB')
    pt(' ')

    quit()
