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

import pyarrow as pa
import pyarrow.parquet as pq

import cv2


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
parser.add_argument('--time','-t',type=float,default=15,
                    help='Time to use for the single frame comparison.')
parser.add_argument('--level','-c',type=float,default=0.0,
                    help='Contour value for gr0 contour.')
parser.add_argument('--moelans',action='store_true',
                    help='Use moelans gr0^2/(gr0^2 * gr1^2) for contour, default=False')
parser.add_argument('--gr0',action='store_true',
                    help='Use gr0 for contour, default=False')
# parser.add_argument('--out','-o',type=str, default='Inclination',
#                                 help='Name of output')
# parser.add_argument('--cpus','-n',type=int, default=default_vals.cpus,
#                             help='How many cpus, default='+str(default_vals.cpus))
# parser.add_argument('--bins','-b',type=int, default=default_vals.bins,
#                             help='How many bins for histogram, default='+str(default_vals.bins))
# parser.add_argument('--dim','-d',type=int,default=2, choices=[2,3],
#                             help='Dimensions for grain size calculation (Default=2)')
parser.add_argument('--subdirs','-s',action='store_true',
                            help='Run in all subdirectories (vs CWD), default=False')
parser.add_argument('--plot','-p',action='store_true',
                            help='Save the contours as a plot in pics/, default=False')
parser.add_argument('--plotonly',action='store_true',
                            help='Skip the calculations and just plot gr0 with the overlaid contour, default=False')
parser.add_argument('--label',action='store_true',
                            help='Include the colorbar and label on the --plotonly plots, default=False')
# parser.add_argument('--save',action='store_true',
#                             help='Save the inclination data, default=False')
# parser.add_argument('--savename',type=str, default='inc_data',
#                                 help='If saving to Parquet, save dir name')
# parser.add_argument('--esd',action='store_true',
#                             help='''Return ESD (vol or area based if 3D/2D) instead of grain
#                             area/volume, default=False''')
# parser.add_argument('--sequence',action='store_true',
#                             help='Time as a sequence, default=False')
# parser.add_argument('--n_frames','-f',type=int, default=default_vals.n_frames,
#                             help='''How many frames for if sequence is true, '''
#                             '''default='''+str(default_vals.n_frames))
# parser.add_argument('--cutoff','-c',type=int, default=default_vals.cutoff,
#                             help='''What time to stop at, if 0.0 uses all data. '''
#                             '''default='''+str(default_vals.cutoff))
parser.add_argument('--exo','-e',action='store_false',
                            help='Look for and use Exodus files instead of Nemesis, default=True')
parser.add_argument('--skip', nargs='+', required=False, help='List of text flags to skip')
parser.add_argument('--only', nargs='+', required=False, help='List of text flags to use')
#
parser.add_argument('--inc', nargs='?', const='normal', choices=['normal','field'],
                    help='Measure GB inclination along the gr1-gr0=level contour. '
                         'Default is "normal" (unit normal from contour). '
                         'Use "field" to sample inclination_vector_x/y on the contour.')
parser.add_argument('--inc-n', type=int, default=360,
                    help='Resample the contour uniformly to this many points before measuring inclination (default=360).')
parser.add_argument('--inc-level', type=float, default=0.0,
                    help='Contour level for gr1-gr0 interface (default=0.0).')
# parser.add_argument('--inc-out', type=str, default='inclination.parquet',
#                     help='Output Parquet file for inclination samples (default=inclination.parquet).')
#
parser.add_argument(
    '--curve',
    action='store_true',
    help='Extra plotting: draw multi-level contours colored by level value.'
)
# Optional: customize levels
parser.add_argument(
    '--curve-min', type=float, default=0.1,
    help='Minimum contour level for --curve heatmap (default=0.1).'
)
parser.add_argument(
    '--curve-max', type=float, default=0.9,
    help='Maximum contour level for --curve heatmap (default=0.9).'
)
parser.add_argument(
    '--curve-n', type=int, default=9,
    help='Number of contour levels between min/max (default=9).'
)
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
db('''INFO: Recreating the single time plots of anisotropy magnitude from
   Lin's paper.''')
db('''WARNING: This script assumes elements are quad4.''')
db(' ')
# pt('This is a warning.')
db(f'Command-line arguments: {args}')
db(' ')
if args.level is None:
    pt('Contour threshold not specified, using [0.1, 0.5, 0.9]')
if not args.plot:
    verb('--plot not enabled, may cause errors if anything is wrongly saved to pics/')
pt(' ')



cwd = os.getcwd()

imdir = 'pics'
if args.plot or args.plotonly:
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
    if args.subdirs:
        for dir_n in glob.glob('*/', recursive=True):
            if args.exo:
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
        if args.exo:
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
    e_names = filter_filenames(e_names, only_flags=args.only, skip_flags=args.skip)
    e_names.sort(key=natural_sort_key)
    verb('Files to use: ')
    verb(e_names)
    verb(' ')
    return e_names

def closest_frame(MF,target=args.time):
    t_list = MF.global_times
    idx = min(range(len(t_list)), key=lambda i: abs(t_list[i] - target))
    closest_time = t_list[idx]
    return closest_time, idx


# def time_info(MF):
#     '''Returns list of times for each frame from MultiExodusReader object

#     Args:
#         MF: MultiExodusReader object with * for all .e*files

#     Raises:
#         ValueError: Sequence cl_arg value issue (T/F)

#     Returns:
#         idx_frames: List of frame iterations/numbers
#         t_frames: List of time values associated with each frame (idx_frames)
#     '''
#     times = MF.global_times
#     if cl_args.sequence == True:
#         # if cl_args.n_frames < len(times):
#         t_max = times[-1]
#         # t_max = max(times)
#         t_frames =  np.linspace(0.0,t_max,cl_args.n_frames)
#         idx_frames = [ np.where(times-t_frames[i] == min(times-t_frames[i],key=abs) )[0][0] for i in range(cl_args.n_frames) ]
#         idx_frames = list( map(int, idx_frames) )
#         # For sequence output use the actual time of each frame not the seq time
#         t_frames = [times[n] for n in idx_frames]
#         # else:
#         #     t_frames = times
#         #     idx_frames = range(len(times))
#     elif cl_args.sequence == False:
#         t_frames = times
#         idx_frames = range(len(times))
#     else:
#         raise ValueError('sequence has to be True or False, not: ' + str(cl_args.sequence))

#     if cl_args.cutoff != 0:
#         verb("Cutting End Time to ",cl_args.cutoff)
#         t_frames = [x for x in t_frames if x <= cl_args.cutoff]
#         idx_frames = range(len(t_frames))

#     t_frames_array = np.asarray(t_frames)
#     # tot_frames = len(idx_frames)
#     return idx_frames, t_frames_array

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

# def plotit(pltx, plty, pltc, cname, times,i):
#     coords = np.asarray([ np.asarray([x_val,y_val]).T for (x_val,y_val) in zip(pltx, plty) ])
#     fig, ax = plt.subplots()
#     p = PolyCollection(coords, cmap=matplotlib.cm.coolwarm, alpha=1)#,edgecolor='k'
#     p.set_array(np.array(pltc) )
#     ax.add_collection(p)
#     fig.colorbar(p, label=cname)
#     ax.set_xlim([np.amin(pltx),np.amax(pltx)])
#     ax.set_ylim([np.amin(plty),np.amax(plty)])
#     ax.set_aspect('equal')
#     timestring = 't = ' + str(times[i])
#     ax.set_title(timestring)
#     fig.savefig(imdir+'/'+str(cname)+'_'+str(i)+'.png',dpi=500,transparent=True )
#     if cl_args.verbose == 2:
#         plt.show()
#     else:
#         plt.close()

# def unique_spacing(*coords):
#     du = []
#     for n in coords:
#         # Step 1: Extract unique values
#         unique_values = np.unique(n)
#         # Check if unique_values has only one element
#         if len(unique_values) == 1:
#             # Skip this n and move to the next item
#             continue
#         # Step 2: Sort the unique values (optional here, as np.unique returns a sorted array)
#         sorted_unique_values = np.sort(unique_values)
#         # Step 3: Compute differences between consecutive unique values
#         differences = np.diff(sorted_unique_values)
#         # Step 4: Calculate the average distance
#         average_distance = np.mean(differences)
#         du.append(average_distance)
#         # Check if du is empty (all coords had only one unique value)
#         if not du:
#             raise ValueError("All input coordinates have only one unique value.")
#         # Check if all items in du are approximately equal
#         if len(du) > 1 and not np.allclose(du, du[0]):
#             pt('\x1b[31;1m'+'WARNING:'+'\x1b[0m'+' dx/dy/dz not all equal.')
#     return np.mean(du)

# def t0_setup(MF,t_frames):
#     x,y,z,c = MF.get_opmax_at_time_alt(t_frames[0])
#     dx = unique_spacing(x,y,z)
#     distance_threshold = dx * 1.1 #1.41 is diag in 2D, 1.73 next diag in 3D
#     num_grs = max(c)
#     feature_numbers = identify_features(x, y, z, c, distance_threshold)
#     num_feats = max(feature_numbers)
#     return distance_threshold, dx, num_grs, num_feats


# def identify_features(x, y, z, c, distance_threshold):
#     """
#     Identifies unique features in a 3D point cloud based on spatial proximity and shared 'c' values.
#     Assigns feature number 0 to all points where c == 0, and numbers other features starting from 1.

#     Parameters:
#     - x, y, z: 1D numpy arrays of coordinates.
#     - c: 1D numpy array of integer values associated with each point.
#     - distance_threshold: Maximum distance to consider points as neighbors.

#     Returns:
#     - feature_numbers: 1D numpy array of feature numbers assigned to each point.
#     """
#     # Combine coordinates into points
#     points = np.vstack((x, y, z)).T

#     num_points = len(points)

#     # Initialize feature_numbers array with zeros
#     feature_numbers = np.zeros(num_points, dtype=int)

#     # Indices where c == 0 and c != 0
#     # c_zero_indices = np.where(c == 0)[0]
#     c_nonzero_indices = np.where(c != 0)[0]

#     # If there are no nonzero c values, return feature_numbers as is
#     if len(c_nonzero_indices) == 0:
#         return feature_numbers  # All features are 0

#     # Build cKDTree for nonzero c points
#     points_nonzero = points[c_nonzero_indices]
#     c_nonzero = c[c_nonzero_indices]

#     tree = cKDTree(points_nonzero)

#     # Initialize the graph
#     G = nx.Graph()
#     num_nonzero_points = len(points_nonzero)
#     G.add_nodes_from(range(num_nonzero_points))

#     # Build edges between neighboring nodes with the same 'c' value
#     for idx in range(num_nonzero_points):
#         point = points_nonzero[idx]
#         # Find indices of neighbors within the distance threshold
#         neighbor_indices = tree.query_ball_point(point, r=distance_threshold)
#         for neighbor_idx in neighbor_indices:
#             if neighbor_idx > idx and c_nonzero[neighbor_idx] == c_nonzero[idx]:
#                 G.add_edge(idx, neighbor_idx)

#     # Find connected components
#     connected_components = list(nx.connected_components(G))

#     # Assign feature numbers starting from 1
#     for feature_num, component in enumerate(connected_components, start=1):
#         for idx in component:
#             original_idx = c_nonzero_indices[idx]
#             feature_numbers[original_idx] = feature_num

#     # c == 0 points remain assigned to feature number 0

#     return feature_numbers


# def grain_sizes(dx,feature_numbers,diams=False):
#     counts = np.bincount(feature_numbers)
#     if cl_args.dim == 2:
#         mesharea = dx * dx
#         featvols = counts * mesharea
#         if diams:
#             diameters = 2 * np.sqrt(featvols / np.pi)
#             return diameters
#     elif cl_args.dim == 3:
#         meshvol = dx * dx * dx
#         featvols = counts * meshvol
#         if diams:
#             diameters = 2 * ((featvols / ((4 / 3) * np.pi)) ** (1 / 3))
#             return diameters
#     else:
#         raise ValueError(f'Command line dimensions should be 2 or 3, is: {cl_args.dim}')
#     return featvols



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
    # suffix = '_grainsize.csv'
    # Beginning based on subdir
    if args.subdirs:
        outBase = file_name.split('/')[0]
    else:
        outBase = os.path.split(os.getcwd())[-1]
    return outBase


# def save_to_csv(volumes, output_filename):
#     """
#     Save feature volumes to a sorted CSV file with appropriate headers.

#     Args:
#         volumes (list): List of lists where each sublist corresponds to [time, featvols].
#         output_filename (str): Name of the output CSV file.
#     """
#     # Extract times and feature volumes
#     data_dict = {}
#     for volume in volumes:
#         time = volume[0]
#         feature_vols = volume[1:]  # exclude time
#         data_dict[time] = feature_vols

#     # Create DataFrame
#     df = pd.DataFrame.from_dict(data_dict, orient='index')
#     df.index.name = "time"
#     df = df.sort_index().reset_index()  # Sort by the time index and reset it

#     # Add headers: time, grain## where ## is a two-digit number
#     max_columns = df.shape[1] - 1  # Exclude 'time'
#     headers = ["time", "phi"] + [f"grain{str(i).zfill(2)}" for i in range(1, max_columns)]
#     df.columns = headers

#     # Save DataFrame to CSV
#     df.to_csv(output_filename, index=False)
#     verb(f"Results saved to {output_filename}")



def plot_slice_forCurvature(frame,x,y,z,c,out_namebase,cb_label=None):
        db('Plotting the sliced data')
        # Make pics subdirectory if it doesnt exist
        pic_directory = 'pics'
        if not os.path.isdir(pic_directory):
            db('Making picture directory: '+pic_directory)
            os.makedirs(pic_directory)
        cv_directory = pic_directory+'/cv_images'
        if not os.path.isdir(cv_directory):
            db('Making picture directory: '+cv_directory)
            os.makedirs(cv_directory)
        db('Plotting the slice as specified')
        # Take the average of the 4 corner values for c
        if hasattr(c[0], "__len__"):
            plotc = np.average(c, axis=1)
        else:
            plotc = c
        plt_x, plt_y = x,y#self.plt_xy(x,y,z)
        coords = np.asarray([ np.asarray([x_val,y_val]).T for (x_val,y_val) in zip(plt_x,plt_y) ])
        fig, ax = plt.subplots()
        p = PolyCollection(coords, cmap=matplotlib.cm.binary, alpha=1)#,edgecolor='k'
        p.set_array(np.array(plotc) )
        ax.add_collection(p)
        ax.set_xlim([np.amin(plt_x),np.amax(plt_x)])
        ax.set_ylim([np.amin(plt_y),np.amax(plt_y)])
        # ax.set_ylim([np.amin(plt_x),np.amax(plt_x)])
        #SET ASPECT RATIO TO EQUAL TO ENSURE IMAGE HAS SAME ASPECT RATIO AS ACTUAL MESH
        ax.set_aspect('equal')
        # ax.axis('off')
        # ax.set_yticklabels([])
        # ax.set_xticklabels([])
        plt.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)
        #ADD A COLORBAR, VALUE SET USING OUR COLORED POLYGON COLLECTION, [0,1]
        p.set_clim(0.0, 1.0)
        # p.set_clim(-0.8, 0.0)
        # if cb_label==None:
        #     p.set_clim(0.0, 1.0)
        #     fig.colorbar(p, label=self.var_to_plot)
        # else:
        #     fig.colorbar(p, label=cb_label)
        figname = cv_directory+'/'+out_namebase+'_cv2_gr0_frame'+str(frame)+'.png'
        fig.savefig(figname,dpi=500,transparent=True )
        plt.close()
        # if self.cl_args.debug:
        #     plt.show()
        # else:
        #     plt.close()
        # plt.show()
        return figname



# CHAT suggested
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

def polygon_area(coords):
    x, y = coords[:,0], coords[:,1]
    return 0.5*np.abs(np.dot(x, np.roll(y,-1)) - np.dot(y, np.roll(x,-1)))

def polygon_centroid(coords):
    x, y = coords[:,0], coords[:,1]
    x1, y1 = np.roll(x,-1), np.roll(y,-1)
    cross = x*y1 - y*x1
    A = 0.5*np.sum(cross)
    Cx = np.sum((x + x1)*cross) / (6*A)
    Cy = np.sum((y + y1)*cross) / (6*A)
    return np.array([Cx, Cy])

def radii_from_center(contour_xy, center_xy):
    v = contour_xy - center_xy
    return np.sqrt((v**2).sum(axis=1))


# def extract_interface_contour(x, y, tri_conn, gr0, level=0.5):
#     """
#     x, y: nodal coords (1D arrays, same length)
#     tri_conn: (ntri, 3) int array of triangle indices into x/y (if you have them)
#               If you don't have connectivity, build a Delaunay triangulation.
#     gr0: nodal values of the order parameter

#     Returns: a list of (N_i, 2) arrays of contour points. Use the largest by area.
#     """
#     # Triangulation
#     if tri_conn is None:
#         # Fallback: Delaunay on the fly
#         from matplotlib.tri import Triangulation
#         tri = Triangulation(x, y)
#     else:
#         tri = mtri.Triangulation(x, y, triangles=tri_conn)

#     cs = plt.tricontour(tri, gr0, levels=[level])
#     paths = cs.collections[0].get_paths()
#     plt.close()

#     contours = []
#     for p in paths:
#         v = p.vertices  # (N,2)
#         if len(v) >= 3:
#             contours.append(v)
#     return contours

def plot_contour_only(contours, center=None, radii_sample_step=0):
    """
    contours: list of (N_i,2) arrays from extract_iso_contour_from_quads
    center: (2,) centroid array (optional)
    radii_sample_step: if >0, draw every k-th radius vector from center to contour
    """
    fig, ax = plt.subplots()
    for poly in contours:
        ax.plot(poly[:,0], poly[:,1], lw=2)
    if center is not None:
        ax.plot(center[0], center[1], 'o', ms=6)
        if radii_sample_step and len(contours):
            main = max(contours, key=lambda c: c.shape[0])
            pts = main[::radii_sample_step]
            for p in pts:
                ax.plot([center[0], p[0]], [center[1], p[1]], alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Isocontour')
    plt.tight_layout()
    plt.show()


def plot_field_with_contour(x4, y4, c4, contours, outname, level, tris=None):
    """
    Quick visual to check the level set placement.
    - Builds a triangulation from the quads (same as in the extractor) if tris not given.
    - Shows a coarse field via tricontourf and overlays the polyline(s).
    """
    # Reconstruct node coords and nodal values (same averaging as extractor)
    # (Reuse the dedupe + averaging logic)
    # def _dedupe_nodes(x4, y4, tol=1e-12):
    #     pts = np.column_stack([x4.ravel(), y4.ravel()])
    #     keys = np.round(pts / tol)
    #     uniq, inv, first = np.unique(keys, axis=0, return_inverse=True, return_index=True)
    #     XY = pts[first]
    #     return XY, inv.reshape(x4.shape)

    # def quads_to_tris(idx4):
    #     t1 = idx4[:, [0,1,2]]
    #     t2 = idx4[:, [0,2,3]]
    #     return np.vstack([t1, t2])

    XY, idx4 = _dedupe_nodes(x4, y4)
    if tris is None:
        tris = quads_to_tris(idx4)

    n_nodes = XY.shape[0]
    acc_c = np.zeros(n_nodes)
    acc_w = np.zeros(n_nodes)
    np.add.at(acc_c, idx4.ravel(), c4.ravel())
    np.add.at(acc_w, idx4.ravel(), 1.0)
    c_nodes = acc_c / np.maximum(acc_w, 1)

    tri = mtri.Triangulation(XY[:,0], XY[:,1], triangles=tris)

    fig, ax = plt.subplots()
    # background field (coarse view just for sanity)
    cf = ax.tricontourf(tri, c_nodes, levels=20, alpha=0.6)
    for poly in contours:
        ax.plot(poly[:,0], poly[:,1], lw=2, c='red')
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('gr1 - gr0 with isocontour at 0.0')
    fig.colorbar(cf, ax=ax, label='gr0')
    plt.tight_layout()
    # plt.show()
    # plt.savefig(imdir+'/'+outname+'_contour_gr0_'+str(level)+'.png',transparent=True,dpi=500)
    plt.savefig(imdir+'/'+outname+'_contour_gr1-gr0_'+str(level)+'level.png',transparent=True,dpi=500)
    plt.close()


def plot_gr0_with_diff_contour_overlay(x4, y4, clist, outname, lvl=0.0, cmap='binary', add_bar=False):

    c4 = clist[1] - clist[0]
    gr0 = clist[0]
    # First calculate the contour
    contours = extract_iso_contour_from_quads(x4, y4, c4, level=lvl)  # adjust level
    main = max(contours, key=polygon_area)
    contours=[main]

    # Setup the triangulation plot for gr0
    XY, idx4 = _dedupe_nodes(x4, y4)
    tris = quads_to_tris(idx4)
    # average gr0 values onto nodes
    n_nodes = XY.shape[0]
    acc_c = np.zeros(n_nodes)
    acc_w = np.zeros(n_nodes)
    np.add.at(acc_c, idx4.ravel(), gr0.ravel())
    np.add.at(acc_w, idx4.ravel(), 1.0)
    c_nodes = acc_c / np.maximum(acc_w, 1)
    # Build triangle mesh thing
    tri = mtri.Triangulation(XY[:,0], XY[:,1], triangles=tris)

    # Now setup the plot
    fig, ax = plt.subplots()
    tpc = ax.tripcolor(tri, c_nodes, shading="gouraud", cmap=cmap)
    tpc.set_clim(0.0, 1.0) #force plot range 0 - 1 for niceness
    if add_bar:
        fig.colorbar(tpc, ax=ax, label="GR0")
    # Overlay the contour
    for poly in contours:
        ax.plot(poly[:,0], poly[:,1], lw=2, c='red')
    # No axis labels or extra space
    ax.set_xlim([np.amin(x4),np.amax(x4)])
    ax.set_ylim([np.amin(y4),np.amax(y4)])
    ax.set_aspect("equal")#, adjustable="box")
    plt.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)
    # ax.set_xlabel("x")
    # ax.set_ylabel("y")
    plt.tight_layout()
    plt.savefig(imdir+'/'+outname+'_gr0_contourOverlay.png',transparent=True,dpi=500)
    plt.close()


###########################################################################
###########################################################################
# INCLINATION
def _poly_arclength(P):
    d = np.linalg.norm(np.diff(np.vstack([P, P[0]]), axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(d[:-1])])
    return s, d.sum()

def resample_closed_polyline(P, n):
    """
    Uniform-arclength resample of closed polyline P (Nx2) to n points.
    """
    s, L = _poly_arclength(P)
    t = np.linspace(0, L, n+1)[:-1]  # n points on [0,L)
    # segment indices for each t
    seg = np.searchsorted(s[1:], t, side='right')
    # local parameter
    s0 = s[seg]
    s1 = s[seg+1]
    alpha = (t - s0) / np.maximum(s1 - s0, 1e-16)
    P0 = P[seg]
    P1 = P[(seg+1) % len(P)]
    Q = (1 - alpha)[:, None] * P0 + alpha[:, None] * P1
    return Q

def ensure_ccw(P):
    x, y = P[:,0], P[:,1]
    A2 = np.dot(x, np.roll(y,-1)) - np.dot(y, np.roll(x,-1))
    return P if A2 > 0 else P[::-1]

def polygon_centroid(P):
    x, y = P[:,0], P[:,1]
    x1, y1 = np.roll(x,-1), np.roll(y,-1)
    cross = x*y1 - y*x1
    A = 0.5*np.sum(cross)
    Cx = np.sum((x + x1)*cross) / (6*A)
    Cy = np.sum((y + y1)*cross) / (6*A)
    return np.array([Cx, Cy])

def normalize(V, eps=1e-16):
    n = np.linalg.norm(V, axis=1, keepdims=True)
    return V / np.maximum(n, eps)

def outward_normals_bisector(P):
    """
    Stable outward unit normals on closed polyline P (Nx2).
    - ensures CCW
    - tangent from *angle bisector* of adjacent edges (smooth)
    - rotates CW (-90°) for outward normal
    - flips any normals pointing inward (dot<0 with radial)
    """
    P = ensure_ccw(P)
    C = polygon_centroid(P)

    # edges: prev and next
    Eprev = P - np.roll(P, 1, axis=0)
    Enext = np.roll(P, -1, axis=0) - P
    eprev = normalize(Eprev)
    enext = normalize(Enext)

    # angle-bisector tangent (handles corners well)
    T = eprev + enext
    # if nearly 180° (T ~ 0), fall back to next edge direction
    mask = (np.linalg.norm(T, axis=1) < 1e-12)
    T[mask] = enext[mask]
    T = normalize(T)

    # rotate CW (-90°): [tx,ty] -> [ty, -tx]
    N = np.column_stack([T[:,1], -T[:,0]])

    # enforce outward
    radial = P - C
    flip = (np.einsum('ij,ij->i', N, radial) < 0)
    N[flip] *= -1.0
    return N

def contour_unit_normals(P):
    """
    Unit outward normals along closed polyline P (Nx2), assuming CCW order.
    Tangent ~ forward difference; normal = rotate(tangent) by +90deg.
    """
    # P = ensure_ccw(P)
    # forward difference tangent
    T = np.roll(P, -1, axis=0) - P
    # avoid zero-length
    lens = np.linalg.norm(T, axis=1, keepdims=True)
    T = T / np.maximum(lens, 1e-16)
    # rotate CCW tangent by +90° to get outward normal for CCW polygon
    N = np.column_stack([-T[:,1], T[:,0]])
    # already unit-length since T was unit
    return N

def contour_unit_normals_outward(P):
    """
    Unit **outward** normals along closed polyline P (Nx2),
    independent of whether P comes in CW or CCW.
    """
    P = ensure_ccw(P)                    # now P is CCW
    # unit tangent via forward difference
    T = np.roll(P, -1, axis=0) - P
    T /= np.maximum(np.linalg.norm(T, axis=1, keepdims=True), 1e-16)

    # rotate CW (-90°) to get outward normal for CCW contour:
    # rot_cw([tx,ty]) = [ty, -tx]
    N = np.column_stack([T[:,1], -T[:,0]])

    # make absolutely sure it's outward: flip if pointing toward centroid
    C = polygon_centroid(P)
    outward_check = np.einsum('ij,ij->i', N, P - C)  # dot(N, radial)
    N[outward_check < 0] *= -1.0

    return N


def outward_normals(P):
    """
    Unit outward normals for a *closed* polyline P (Nx2).
    Uses central-difference tangent and guarantees 'outward'
    by dotting with the centroid radial vector.
    """
    P = ensure_ccw(P)
    # central-difference tangent
    T = np.roll(P, -1, axis=0) - np.roll(P, 1, axis=0)
    T /= np.maximum(np.linalg.norm(T, axis=1, keepdims=True), 1e-16)
    # rotate CW (-90°): [tx,ty] -> [ty, -tx]
    N = np.column_stack([T[:,1], -T[:,0]])
    # flip to make sure it's truly outward
    C = polygon_centroid(P)
    radial = P - C
    flip = (np.einsum('ij,ij->i', N, radial) < 0)
    N[flip] *= -1.0
    return N


def angles_0_2pi(V):
    """
    Return angles in [0, 2π) from vectors V (Nx2), using atan2(y,x) mod 2π.
    """
    th = np.arctan2(V[:,1], V[:,0])
    th = np.mod(th, 2*np.pi)
    return th

def make_triangulation_from_quads(x4, y4):
    XY, idx4 = _dedupe_nodes(x4, y4)
    tris = quads_to_tris(idx4)
    tri = Triangulation(XY[:,0], XY[:,1], triangles=tris)
    return tri, XY, idx4

def sample_field_on_polyline(tri, nodal_vals, P):
    """
    Sample a nodal field (length = tri.x.size) on polyline points P (Nx2)
    using linear interpolation on the triangulation.
    """
    interp = LinearTriInterpolator(tri, nodal_vals)
    vals = interp(P[:,0], P[:,1])
    return np.asarray(vals)

def nodal_average_from_quads(idx4, c4):
    n_nodes = int(idx4.max())+1
    acc_c = np.zeros(n_nodes)
    acc_w = np.zeros(n_nodes)
    np.add.at(acc_c, idx4.ravel(), c4.ravel())
    np.add.at(acc_w, idx4.ravel(), 1.0)
    return acc_c / np.maximum(acc_w, 1)

def get_inclination_nodal_fields(x4, y4, ti, MF):
    """
    Returns: tri, nx_nodes, ny_nodes
    where nx_nodes, ny_nodes are nodal averages of inclination_vector_x/y.
    """
    # triangulation
    tri, XY, idx4 = make_triangulation_from_quads(x4, y4)
    # read inclination vectors at the same time
    # we pull full nodal arrays per element corner, then average to unique nodes
    _, _, _, V = MF.get_full_vars_at_time(['inclination_vector_x','inclination_vector_y'], ti)
    incx4 = V[0]  # shape (n_elem, 4)
    incy4 = V[1]  # shape (n_elem, 4)
    nx_nodes = nodal_average_from_quads(idx4, incx4)
    ny_nodes = nodal_average_from_quads(idx4, incy4)
    return tri, nx_nodes, ny_nodes



def debug_quiver(P, N, title="Contour with OUTWARD normals"):
    step = max(len(P)//32, 1)
    Q = P[::step]
    U = N[::step]

    fig, ax = plt.subplots()
    ax.plot(P[:,0], P[:,1], lw=2)

    # Key: make arrows big enough: scale < 1 makes longer arrows.
    ax.quiver(Q[:,0], Q[:,1], U[:,0], U[:,1],
              angles='xy', scale_units='xy', scale=0.1,  # ← long arrows
              width=0.004)

    ax.set_aspect('equal', adjustable='box')
    ax.set_title(title)
    plt.show()



# CONTOUR STACK
def _nodal_average_from_quads(idx4, c4):
    n_nodes = int(idx4.max()) + 1
    acc_c = np.zeros(n_nodes)
    acc_w = np.zeros(n_nodes)
    np.add.at(acc_c, idx4.ravel(), c4.ravel())
    np.add.at(acc_w, idx4.ravel(), 1.0)
    return acc_c / np.maximum(acc_w, 1)

def plot_multi_level_contours(x4, y4, c4, outname, title, levels, add_colorbar=True):
    """
    Draw a single plot with many contour levels, colored by the level value (viridis).
    Saves to pics/<outname>_<title>_curve_heatmap.png
    """
    # Build triangulation once from quads
    XY, idx4 = _dedupe_nodes(x4, y4)
    tris = quads_to_tris(idx4)
    tri = mtri.Triangulation(XY[:,0], XY[:,1], triangles=tris)

    # Average c4 from element corners to unique nodes
    c_nodes = _nodal_average_from_quads(idx4, c4)

    fig, ax = plt.subplots()
    # Draw contours; the colormap maps the contour *level* to color
    cs = ax.tricontour(tri, c_nodes, levels=levels, cmap='viridis')

    if add_colorbar:
        # Colorbar with ticks at the level values
        cbar = fig.colorbar(cs, ax=ax)
        cbar.set_label('Contour value')

    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim([np.amin(x4), np.amax(x4)])
    ax.set_ylim([np.amin(y4), np.amax(y4)])
    ax.set_title(title)
    plt.tight_layout()

    # Ensure pics/ exists
    if not os.path.isdir(imdir):
        os.makedirs(imdir)
    fig.savefig(os.path.join(imdir, f'{outname}_{title}_curve_heatmap.png'),
                dpi=500, transparent=True)
    plt.close(fig)



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
    results = []
    for cnt,file_name in enumerate(file_names):
        pt(' ')#\x1b[31;1m
        pt('\033[1m\033[96m'+'File '+str(cnt+1)+'/'+str(num_files)+': '+'\x1b[0m'+str(file_name))
        verb('Initialization for file: '+str(file_name))
        outbase = out_name(file_name)
        init_ti = time.perf_counter()
        MF = MultiExodusReaderDerivs(file_name)
        ti, idx = closest_frame(MF)
        # Read in the full data for gr0 and gr1
        x4, y4, z, clist = MF.get_full_vars_at_time(['gr0','gr1'],ti)
        # x4, y4, z, c4 = MF.get_data_at_time('gr0',ti,full_nodal=True)
        if args.moelans:
            c0 = clist[0]
            c1 = clist[1]
            c4 = c0 * c0 / (c0 * c0 + c1 * c1)
            args.level = 0.5
        elif args.gr0:
            c4 = clist[0]
        else:
            c4 = clist[1]-clist[0] # gr1-gr0

        # --- Optional curve heatmap (multi-level contours) ---
        if args.curve:
            levels = np.linspace(args.curve_min, args.curve_max, args.curve_n)

            # Plot for gr0
            plot_multi_level_contours(
                x4, y4, clist[0],
                outname=outbase,
                title='gr0',
                levels=levels,
                add_colorbar=True
            )

            # Plot for gr1
            plot_multi_level_contours(
                x4, y4, clist[1],
                outname=outbase,
                title='gr1',
                levels=levels,
                add_colorbar=True
            )

        # GR0 plot with gr1-gr0 contour overlaid only, skip calculations:
        if args.plotonly:
            plot_gr0_with_diff_contour_overlay(x4, y4, clist, outbase, lvl=args.level, cmap='binary', add_bar=args.label)
        else: # Do the calculations and plots
            if args.plot:
                plot_slice_forCurvature(idx,x4,y4,z,clist[0],outbase,cb_label=None)
            # 1) Extract
            if args.level is not None:
                levels = [args.level]
            else:
                levels = [0.1,0.5,0.9]
            for lvl in levels:
                contours = extract_iso_contour_from_quads(x4, y4, c4, level=lvl)  # adjust level
                main = max(contours, key=polygon_area)

                # 2) Center & radii (optional)
                center = polygon_centroid(main)
                radii  = np.sqrt(((main - center)**2).sum(axis=1))
                # print(f"mean radius = {radii.mean():.4f}")
                r_avg = radii.mean()
                n_sites = len(radii)

                metric = np.sum(np.abs(radii - r_avg)) / (r_avg * n_sites)
                # print("Normalized deviation:", metric)

                results.append({
                    "file_name": outbase,   # or use fname if you want full path/name
                    "file_num":  cnt + 1,
                    "time":      ti,
                    "contour":   lvl,
                    "r_avg":     r_avg,
                    "amag":      metric
                })

                # 4) Or overlay on the scalar field
                if args.plot:
                    # plot_contour_only(contours=[main], center=center, radii_sample_step=max(len(main)//24,1))
                    plot_field_with_contour(x4, y4, c4, contours=[main], outname=outbase, level=lvl)


            # --- Inclination measurement (if requested) ---
            if args.inc:
                # Use level for gr1-gr0 from args.inc_level (separate from args.level you use elsewhere)
                lvl_inc = args.inc_level
                # we already computed c4 = gr1-gr0 above; if not, do it
                # contours for inclination are computed at lvl_inc
                inc_contours = extract_iso_contour_from_quads(x4, y4, c4, level=lvl_inc)
                if not inc_contours:
                    pt('No interface contour found for inclination; skipping.')
                else:
                    P = max(inc_contours, key=polygon_area)   # (M,2)
                    # resample to uniform arclength to avoid clustering bias
                    P = resample_closed_polyline(P, args.inc_n)
                    if args.inc == 'normal':
                        N = contour_unit_normals(P)          # (M,2), CCW outward
                        theta = angles_0_2pi(N)
                        inc_df = pd.DataFrame({
                            'theta': theta,
                            'nx':    N[:,0],
                            'ny':    N[:,1],
                            'x':     P[:,0],
                            'y':     P[:,1],
                            'file_name': outbase,
                            'time':  ti,
                            'contour': lvl_inc,
                            'source': 'normal'
                        })
                        # TEST PLOT
                        # debug_quiver(P, N)
                    else:  # args.inc == 'field'
                        tri, nx_nodes, ny_nodes = get_inclination_nodal_fields(x4, y4, ti, MF)
                        nx = sample_field_on_polyline(tri, nx_nodes, P)
                        ny = sample_field_on_polyline(tri, ny_nodes, P)
                        # normalize just in case the stored vectors aren’t exactly unit
                        mag = np.maximum(np.sqrt(nx**2 + ny**2), 1e-16)
                        nxu, nyu = nx/mag, ny/mag
                        theta = angles_0_2pi(np.column_stack([nxu, nyu]))
                        inc_df = pd.DataFrame({
                            'theta': theta,
                            'nx':    nxu,
                            'ny':    nyu,
                            'x':     P[:,0],
                            'y':     P[:,1],
                            'file_name': outbase,
                            'time':  ti,
                            'contour': lvl_inc,
                            'source': 'field'
                        })

                    # --- Write one CSV per file/time ---
                    inc_outfile = f"{outbase}_inc_t{args.time}.csv" #ti for exact frame time
                    inc_df.to_csv(inc_outfile, index=False)
                    # pt(f"Wrote inclination CSV: {inc_outfile} ({len(inc_df)} rows)")

        pt(f'Done File {cnt+1}: {format_elapsed_time(init_ti)}')

    if not args.plotonly:
        # Save to csv
        df = pd.DataFrame(results)
        df.to_csv("aniso_magnitude.csv", index=False)
        # print("Saved", len(df), "rows to aniso_magnitude.csv")

    pt(' ')
    pt(f'Done Everything: {format_elapsed_time(all_ti)}')
    current, peak =  tracemalloc.get_traced_memory()
    pt('Memory after everything (current, peak): '+str(round(current/1048576,1))+', '+str(round(peak/1048576,1))+' MB')
    pt(' ')

    quit()
