from MultiExodusReader import MultiExodusReader
from MultiExodusReaderDerivs import MultiExodusReaderDerivs
# import multiprocessing as mp
# from VolumeScripts import *

import subprocess
from joblib import Parallel, delayed

import matplotlib.pyplot as plt
# from matplotlib.patches import Polygon
# from matplotlib.collections import PolyCollection
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

var = 'unique_grains'


# CL Argument Parser
parser = argparse.ArgumentParser()
parser.add_argument('--verbose', '-v', action='count', default=0,
                    help='Increase verbosity: -v for INFO, -vv for DEBUG.')
parser.add_argument('--cpus','-n',type=int, default=default_vals.cpus,
                            help='How many cpus, default='+str(default_vals.cpus))
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
        varnames = ['testout2','testoutgrad_x','testoutgrad_y']
        MF.check_varlist(varnames)

        # MF = GrainMultiExodusReader(file_name)
        # idx_frames, t_frames = time_info(MF) # frames and the time at each one
        # distance_threshold, dx, num_grs, num_feats = t0_setup(MF,t_frames)
        # verb(f'Done Initializing: {format_elapsed_time(init_ti)}')
        # # Calculaton
        # calc_ti = time.perf_counter()
        # # volumes = []
        # # for tstep in tqdm(t_frames):
        # #     step_ti = time.perf_counter()
        # #     volumes.append(para_feature_volume(MF,tstep,dx,distance_threshold))
        # #     pt(f'Done Calculating Step: {format_elapsed_time(step_ti)}')
        # # Run para_feature_volume in parallel
        # volumes = Parallel(n_jobs=cl_args.cpus)(
        #     delayed(para_feature_volume)(file_name, tstep, dx, distance_threshold)
        #     for tstep in tqdm(t_frames)
        # )
        # verb(' ')
        # verb(f'Done Calculating: {format_elapsed_time(calc_ti)}')
        # saveloc = out_name(file_name)
        # save_to_csv(volumes, saveloc)

        pt(f'Done File {cnt+1}: {format_elapsed_time(init_ti)}')

    pt(' ')
    pt(f'Done Everything: {format_elapsed_time(all_ti)}')
    current, peak =  tracemalloc.get_traced_memory()
    pt('Memory after everything (current, peak): '+str(round(current/1048576,1))+', '+str(round(peak/1048576,1))+' MB')
    pt(' ')

    quit()
