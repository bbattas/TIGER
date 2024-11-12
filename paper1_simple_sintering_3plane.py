from MultiExodusReader import MultiExodusReader
import multiprocessing as mp
# from VolumeScripts import *

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
# from matplotlib.patches import Polygon
# from matplotlib.collections import PolyCollection
# from matplotlib.collections import PatchCollection
# from mpl_toolkits.mplot3d.art3d import Poly3DCollection
# from mpl_toolkits.mplot3d import Axes3D
# import matplotlib

from scipy.spatial import ConvexHull, Delaunay
import numpy as np
import time
# from time import time
import os
import glob
# import pandas as pd
import math
import sys
import tracemalloc
import logging
import argparse
import re
from enum import Enum
from tqdm import tqdm
# import dask
# from dask import delayed, compute
# from dask.diagnostics import ProgressBar
import subprocess
from joblib import Parallel, delayed
# from tqdm_joblib import tqdm_joblib


# ASSUMES A QUARTER HULL!!!


pt = logging.warning
verb = logging.info
db = logging.debug

# Defaults for the variables
class default_vals:
    cpus = 2
    n_frames = 300
    cutoff = 0.0
    var_to_plot = 'phi'
    var_threshold = 0.5
    nodal_var_names = ['phi', 'wvac', 'wint', 'bnds', 'gr0', 'gr1', 'gr2', 'gr3']
    elem_var_names = ['T', 'unique_grains']
    # complexHull = True


# CL Argument Parser
parser = argparse.ArgumentParser()
parser.add_argument('--verbose','-v', action='store_true', help='Verbose output, default off')
parser.add_argument('--cpus','-n',type=int, default=default_vals.cpus,
                            help='How many cpus, default='+str(default_vals.cpus))
parser.add_argument('--time','-t', action='store_true',
                            help='Run time_file_make script, default off')
parser.add_argument('--var','-i',type=str, default=default_vals.var_to_plot,
                            help='What variable to plot/calc, default='+str(default_vals.var_to_plot))
parser.add_argument('--threshold','-m',type=float, default=default_vals.var_threshold,
                            help='''What value to set as the threshold on the variable for plot/calc, '''
                            '''default='''+str(default_vals.var_threshold))
parser.add_argument('--dim','-d',type=int,default=2, choices=[2,3],
                            help='Dimensions for grain size calculation (Default=2)')
parser.add_argument('--max_xy','-x',type=float,required=True,
                            help='''Max x&y for the 1/4 hull assuming center of symmetry is [max_xy,max_xy], '''
                            '''Required, Default=None''')
parser.add_argument('--max_z','-z',type=float,default=None,
                            help='''Max z for the 1/4 hull assuming center of symmetry is '''
                            '''[max_xy,max_xy,max_z], Required, Default=None''')
parser.add_argument('--subdirs','-s',action='store_true',
                            help='Run in all subdirectories (vs CWD), default=False')
parser.add_argument('--skip',action='store_true',help='Skip last timestep/file, currently not working?')
parser.add_argument('--plot','-p', action='store_true',
                            help='Show the calculation plot for debugging.')
parser.add_argument('--sequence',action='store_true',
                            help='Time as a sequence, default=False')
parser.add_argument('--n_frames','-f',type=int, default=default_vals.n_frames,
                            help='''How many frames for if sequence is true, '''
                            '''default='''+str(default_vals.n_frames))
parser.add_argument('--cutoff','-c',type=int, default=default_vals.cutoff,
                            help='''What time to stop at, if 0.0 uses all data. '''
                            '''default='''+str(default_vals.cutoff))
parser.add_argument('--complexHull',action='store_false',
                            help='For phi use UG to calculate centroids so its only internal porosity (default=True)')
cl_args = parser.parse_args()

# Custom validation
if cl_args.dim == 3 and cl_args.max_z is None:
    parser.error("When dim is 3, max_z must have an input other than None.")


# Toggle verbose
if cl_args.verbose == True:
    logging.basicConfig(level=logging.INFO,format='%(message)s')
elif cl_args.verbose == False:
    logging.basicConfig(level=logging.WARNING,format='%(message)s')
verb('Verbose Logging Enabled')
verb(cl_args)
verb(' ')


cwd = os.getcwd()



# ███████╗██╗   ██╗███╗   ██╗ ██████╗████████╗██╗ ██████╗ ███╗   ██╗███████╗
# ██╔════╝██║   ██║████╗  ██║██╔════╝╚══██╔══╝██║██╔═══██╗████╗  ██║██╔════╝
# █████╗  ██║   ██║██╔██╗ ██║██║        ██║   ██║██║   ██║██╔██╗ ██║███████╗
# ██╔══╝  ██║   ██║██║╚██╗██║██║        ██║   ██║██║   ██║██║╚██╗██║╚════██║
# ██║     ╚██████╔╝██║ ╚████║╚██████╗   ██║   ██║╚██████╔╝██║ ╚████║███████║
# ╚═╝      ╚═════╝ ╚═╝  ╚═══╝ ╚═════╝   ╚═╝   ╚═╝ ╚═════╝ ╚═╝  ╚═══╝╚══════╝


def time_file_make():
    full_time = os.path.abspath(os.path.expanduser('~/projects/TIGER/best_time_file_make.py'))
    command = ['python',full_time,'-n',str(cl_args.cpus)]
    if cl_args.verbose:
        command.append('-v')
    if cl_args.skip:
        command.append('--skip')
    verb('Running best_time_file_make.py with command:' + str(command))
    if cl_args.subdirs:
        for dir in sorted(glob.glob('*/')):
            if not dir.startswith('.') and not dir.startswith('pic'):
                os.chdir(cwd + "/" + dir)
                # Check for .e* files
                enames = glob.glob('*.e*')
                if enames:
                    pt('In subdir: ' + str(dir))
                    subprocess.run(command)
        os.chdir(cwd)
    else:
        subprocess.run(command)


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



def time_info(times_files):
    '''Returns list of times for each frame from MultiExodusReader object

    Args:
        MF: MultiExodusReader object with * for all .e*files

    Raises:
        ValueError: Sequence cl_arg value issue (T/F)

    Returns:
        idx_frames: List of frame iterations/numbers
        t_frames: List of time values associated with each frame (idx_frames)
    '''
    times = times_files[:,0].astype(float)
    if cl_args.sequence == True:
        if cl_args.n_frames < len(times):
            t_max = times[-1]
            # t_max = max(times)
            t_frames =  np.linspace(0.0,t_max,cl_args.n_frames)
            idx_frames = [ np.where(times-t_frames[i] == min(times-t_frames[i],key=abs) )[0][0] for i in range(cl_args.n_frames) ]
            idx_frames = list( map(int, idx_frames) )
        else:
            t_frames = times
            idx_frames = range(len(times))
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

def find_npy_files():
    """
    Find 'times_files.npy'' in the current directory or subdirectories.

    Returns:
        list: Sorted list of file names with 'times_files.npy' name.
        list: Sorted list of output names (directory name) for the outputs later.

    Raises:
        ValueError: If no files matching the pattern are found.
    """
    npy_names = []
    out_namebases = []
    if cl_args.subdirs:
        for dir_n in glob.glob('*/', recursive=True):
            npy_files_in_subdir = glob.glob(dir_n + 'times_files.npy')
            # e_files_in_subdir = glob.glob(dir_n + '*.e*')
            if npy_files_in_subdir:
                # first_file = npy_files_in_subdir[0]
                # # trimmed_file = first_file.split('.e', 1)[0] + '.e*'
                # trimmed_file = first_file #+ '.e*'
                npy_names.append(dir_n)
                out_namebases.append(os.path.basename(os.path.normpath(dir_n)))

    else:
        print('cwd only')
        npy_files_in_dir = glob.glob('times_files.npy')
        if npy_files_in_dir:
            # first_file = npy_files_in_dir[0]
            # trimmed_file = first_file #.split('.e', 1)[0] + '.e*'
            npy_names.append('-1')
            out_namebases.append(os.path.basename(os.path.normpath(cwd)))
    if not npy_names:
        raise ValueError('No files found matching times_files.npy, make sure to specify subdirectories or not')
    npy_names.sort(key=natural_sort_key)
    verb('Files to use: ')
    verb(npy_names)
    # verb(out_namebases)
    verb(' ')
    return npy_names, out_namebases


def out_name(dirname):
    """
    Generate an output file name based on the input file name and dimensionality.

    Args:
        dirname (str): The directory name.

    Returns:
        str: The generated output file name.

    Raises:
        ValueError: If the dimension is not 2 or 3.
    """
    # Ending based on dimensions
    if cl_args.dim == 2:
        suffix = '_areas.csv'
    elif cl_args.dim == 3:
        suffix = '_volumes.csv'
    else:
        raise ValueError('Output name needs CL --dim/-d to be 2 or 3 not '+str(cl_args.dim))
    # Beginning based on subdir
    if cl_args.subdirs:
        outBase = dirname.split('/')[0]
    else:
        outBase = os.path.split(os.getcwd())[-1]
    return outBase + suffix


def pore_in_hull(xyz_for_hull,void_ctr_xyz,tolerance=1e-12):
    """
    Check if points are inside the convex hull defined by a set of points.

    Args:
        xyz_for_hull (ndarray): Points defining the convex hull.
        void_ctr_xyz (ndarray): Points to check.
        tolerance (float, optional): Tolerance for point inclusion. Default is 1e-12.

    Returns:
        ndarray: Boolean array indicating which points are inside the hull.
    """
    hull = ConvexHull(xyz_for_hull)
    in_hull = np.all(np.add(np.dot(void_ctr_xyz, hull.equations[:,:-1].T),
                            hull.equations[:,-1]) <= tolerance, axis=1)
    # Output the boolean array of which void_ctr is in the hull for use
    return in_hull


def debug_plot(xyz_for_hull,void_ctr_xyz,tolerance=1e-12):
    """
    Plot the convex hull and the points inside it for debugging purposes.

    Args:
        xyz_for_hull (ndarray): Points defining the convex hull.
        void_ctr_xyz (ndarray): Points to check.
        tolerance (float): Tolerance for point inclusion.
    """
    hull = ConvexHull(xyz_for_hull)
    in_hull = np.all(np.add(np.dot(void_ctr_xyz, hull.equations[:,:-1].T),
                            hull.equations[:,-1]) <= tolerance, axis=1)
    # The void centers that are in the hull
    void_in_hull = void_ctr_xyz[in_hull]
    # Plot a scatterplot of the void mesh centers in the hull
    if cl_args.dim == 2:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for simplex in hull.simplices:
            # plt.plot(grain_ctr[simplex, 0], grain_ctr[simplex, 1], grain_ctr[simplex,2], 'r-')
            plt.plot(xyz_for_hull[simplex, 0], xyz_for_hull[simplex, 1], 'r-')
        # Now plot the void points
        ax.scatter(void_in_hull[:, 0], void_in_hull[:, 1],s=1,alpha=0.5)
        ax.set_aspect('equal')
        # plt.autoscale()
        plt.show()
    elif cl_args.dim == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111,projection='3d')
        for simplex in hull.simplices:
            # plt.plot(grain_ctr[simplex, 0], grain_ctr[simplex, 1], grain_ctr[simplex,2], 'r-')
            plt.plot(xyz_for_hull[simplex, 0], xyz_for_hull[simplex, 1], xyz_for_hull[simplex,2], 'r-')
        # Now plot the void points
        ax.scatter3D(void_in_hull[:, 0], void_in_hull[:, 1], void_in_hull[:, 2],s=0.01,alpha=0.5)
        plt.autoscale()
        plt.show()
    else:
        raise ValueError('CL --dim/-d needs to be 2 or 3, not '+str(cl_args.dim))
    # Output the boolean array of which void_ctr is in the hull for use
    return


class DimensionCase(Enum):
    """
    Enumeration to represent different dimensional calculation cases.
    """
    D2_NODAL = 1
    D2_ELEM = 2
    D3_NODAL = 3
    D3_ELEM = 4

def determine_dimension_case():
    """
    Determine the calculation case based on the variable (nodal/elemental) and dimension (2D/3D).

    Returns:
        DimensionCase: The determined dimension case.

    Raises:
        ValueError: If the dimension is not 2 or 3 or if the variable is not found.
    """
    if cl_args.var in default_vals.elem_var_names:
        if cl_args.dim == 2:
            verb('2D Elemental Calculations')
            return DimensionCase.D2_ELEM
        elif cl_args.dim == 3:
            verb('3D Elemental Calculations')
            return DimensionCase.D3_ELEM
        else:
            raise ValueError("Invalid dimension: must be 2 or 3.")
    elif cl_args.var in default_vals.nodal_var_names:
        if cl_args.dim == 2:
            verb('2D Nodal Calculations')
            return DimensionCase.D2_NODAL
        elif cl_args.dim == 3:
            verb('3D Nodal Calculations')
            return DimensionCase.D3_NODAL
        else:
            raise ValueError("Invalid dimension: must be 2 or 3.")
    else:
        raise ValueError(f"{cl_args.var} not found in either elem_var_names or nodal_var_names.")


# Uses min and max value of each coordinate (xyz) in each element to return center
def mesh_center_dir_independent(*args):
    """
    Calculate the center of each mesh element.

    Args:
        *args: Coordinate arrays for the mesh elements.

    Returns:
        ndarray: Array of mesh element centers.

    Raises:
        ValueError: If the number of dimensions is not 2 or 3.
    """
    min = []
    max = []
    for ar in args:
        min.append(np.amin(ar,axis=1))
        max.append(np.amax(ar,axis=1))
    if len(args) == 2:
        db('2D Mesh center calc')
        mesh_ctr = np.asarray([min[0][:] + (max[0][:] - min[0][:])/2,
                                min[1][:] + (max[1][:] - min[1][:])/2 ]).T
    # 3D
    elif len(args) == 3:
        db('3D mesh center calc')
        mesh_ctr = np.asarray([min[0][:] + (max[0][:] - min[0][:])/2,
                                min[1][:] + (max[1][:] - min[1][:])/2,
                                min[2][:] + (max[2][:] - min[2][:])/2 ]).T
    else:
        raise ValueError('mesh_center_quadElements needs 2 or 3 dimensions of x,y,z input')
    return mesh_ctr

# Uses min and max value of each coordinate in each element to return volume/area
def mesh_vol_dir_independent(*args):
    """
    Calculate the volume (or area) of each mesh element.

    Args:
        *args: Coordinate arrays for the mesh elements.

    Returns:
        ndarray: Array of mesh element volumes (or areas).

    Raises:
        ValueError: If the number of dimensions is not 2 or 3.
    """
    min = []
    max = []
    for ar in args:
        min.append(np.amin(ar,axis=1))
        max.append(np.amax(ar,axis=1))
    if len(args) == 2:
        db('2D Mesh vol calc')
        mesh_vol = np.asarray((max[0][:] - min[0][:])*
                                (max[1][:] - min[1][:]) )
    # 3D
    elif len(args) == 3:
        db('3D mesh vol calc')
        mesh_vol = np.asarray((max[0][:] - min[0][:])*
                                (max[1][:] - min[1][:])*
                                (max[2][:] - min[2][:]))
    else:
        raise ValueError('mesh_vol_dir_independent needs 2 or 3 dimensions of x,y,z input')
    return mesh_vol

def t0_opCount_headerBuild(prefix,times_files,idx_frames,t_frames):
    """
    Build the header for the output CSV file and return the number of OPs.

    Args:
        MF: Data source object with `get_data_at_time` method.
        t_frames (list): List of time frames.

    Returns:
        tuple: The OP count and CSV header list.
    """
    verb("Calculating initial frame...")
    MF = MultiExodusReader(prefix+times_files[idx_frames[0],1])
    x,y,z,c = MF.get_data_at_time(cl_args.var,t_frames[0])
    c_int = np.rint(c)
    op_0 = round(np.amax(c_int))+2
    csv_header = ["time", "internal_pore", "total_hull", "vol_density", "total_void"]
    for n in range(1,op_0):
        csv_header.append("Grain_"+str(n))
    return op_0, csv_header

def calculate_cents_on_max_plane(xug, yug, zug, cug, axis):
    """
    Calculate area centroids based on the specified max plane in the unique_grain data.

    Parameters:
        xug (ndarray): x-coordinates of unique grain data.
        yug (ndarray): y-coordinates of unique grain data.
        zug (ndarray): z-coordinates of unique grain data.
        cug (ndarray): grain order parameters corresponding to each point.
        axis (str): The axis to use for the max plane calculation ('x', 'y', or 'z').

    Returns:
        list: The centroid coordinates from the specified max plane.
    """
    # Select the appropriate axis data
    axis_data = {'x': xug, 'y': yug, 'z': zug}[axis]

    # Determine the overall max value along the chosen axis
    overall_max_value = np.max(axis_data)

    # Calculate the row-wise min and max along the chosen axis
    row_min = np.min(axis_data, axis=1)
    row_max = np.max(axis_data, axis=1)

    # Use vectorized comparison to check if overall_max_value is within each row's range
    max_plane_indices = (row_min <= overall_max_value) & (overall_max_value <= row_max)

    shortx = xug[max_plane_indices]
    shorty = yug[max_plane_indices]
    shortz = zug[max_plane_indices]
    shortc = np.rint(cug[max_plane_indices])

    short_ctr = mesh_center_dir_independent(shortx,shorty,shortz)
    short_vol = mesh_vol_dir_independent(shortx,shorty,shortz)
    # tempVols = np.sum(np.where(np.rint(shortc)==(-1),short_vol,0.0))
    centroid = [np.sum(np.where(shortc!=(-1),short_ctr[:,0] * short_vol,0.0)) / np.sum(np.where(shortc!=(-1),short_vol,0.0)),
                np.sum(np.where(shortc!=(-1),short_ctr[:,1] * short_vol,0.0)) / np.sum(np.where(shortc!=(-1),short_vol,0.0)),
                np.sum(np.where(shortc!=(-1),short_ctr[:,2] * short_vol,0.0)) / np.sum(np.where(shortc!=(-1),short_vol,0.0))]

    # Set cent based on the specified axis
    if axis == 'x':
        cents = [[cl_args.max_xy, centroid[1], 0], [cl_args.max_xy, centroid[1], cl_args.max_z]]
    elif axis == 'y':
        cents = [[centroid[0], cl_args.max_xy, 0] , [centroid[0], cl_args.max_xy, cl_args.max_z]]
    # elif axis == 'z':
    #     cents = [0, clargs.maxxy, centroid[2]]
    else:
        raise ValueError("Invalid axis specified. Use 'x' or 'y'.")

    return cents

def para_volume_calc(time_step,i,t_frames,dimension_case,times_files,prefix,op_max):
    """
    Perform area/volume calculations for a given time step assuming a quarter hull.

    Args:
        time_step (int): The time step index.
        i (int): The current frame index.
        t_frames (list): List of time frames.
        dimension_case (DimensionCase): The dimension calculation case.
        op_max (int): The number of OPs (from t0_opCount_headerBuild).

    Returns:
        list: Calculated volumes and other metrics for the given time step.

    Raises:
        ValueError: If elemental calculations are not set up or the dimension case is unknown.
    """
    db("Calculating frame",i+1)
    MF = MultiExodusReader(prefix+times_files[time_step,1])
    x,y,z,c = MF.get_data_at_time(cl_args.var,t_frames[time_step])
    match dimension_case:
        case DimensionCase.D2_NODAL:
            # db("Processing 2D Nodal case")
            # Phi mainly
            mesh_ctr = mesh_center_dir_independent(x,y)
            mesh_vol = mesh_vol_dir_independent(x,y)
            # Total volume of variable and not_variable (meshVolArray*variableValueArray)
            volumes = []
            volumes.append(np.sum(mesh_vol * c)) # Variable
            volumes.append(np.sum(mesh_vol * (1-c))) # Not Variable (all else)
            # Full Domain var and other assignments
            other_ctr = np.delete(mesh_ctr, np.where((c>=cl_args.threshold))[0], axis=0)
            var_ctr = np.delete(mesh_ctr, np.where((c<cl_args.threshold))[0], axis=0)
            var_vol_weighted = np.delete(mesh_vol*c, np.where((c<cl_args.threshold))[0], axis=0)
            # Calculate in the hull
            if cl_args.complexHull:
                # Calculate the centroids of the grains from UG for only internal porosity
                xug,yug,zug,cug = MF.get_data_at_time('unique_grains',t_frames[time_step])
                c_int = np.rint(cug)
                zeros = np.zeros_like(c_int)
                # Calculate individual UniqueGrains volumes and centroids
                grain_centroids = []
                for n in range(op_max):
                    tempVols = np.sum(np.where(c_int==(n-1),mesh_vol,zeros))
                    if tempVols > 0.0 and n > 0:
                        grain_centroids.append([ np.sum(np.where(c_int==(n-1),mesh_ctr[:,0] * mesh_vol,zeros)) / np.sum(np.where(c_int==(n-1),mesh_vol,zeros)),
                                                    np.sum(np.where(c_int==(n-1),mesh_ctr[:,1] * mesh_vol,zeros)) / np.sum(np.where(c_int==(n-1),mesh_vol,zeros))])
                # If there is only 1 or 0 grain centroids then skip this timestep and return -1
                if len(grain_centroids) < 2:
                    return -1
                # Alter grain centroids to account for quarter hull
                for n in range(len(grain_centroids)):
                    if grain_centroids[n][0] > grain_centroids[n][1]:
                        grain_centroids[n][0] = cl_args.max_xy
                    elif grain_centroids[n][1] > grain_centroids[n][0]:
                        grain_centroids[n][1] = cl_args.max_xy
                # Temp ctr using centroids and corner
                temp_ctr = np.append(grain_centroids,[[cl_args.max_xy,cl_args.max_xy]],axis=0)
            else:
                temp_ctr = np.append(other_ctr,[[cl_args.max_xy,cl_args.max_xy]],axis=0)
            # db('Only running on a manually defined box for hull!')
            # ytop = np.amax(y)
            # temp_ctr = np.array([[0, 0], [0, ytop], [cl_args.xcut, ytop], [cl_args.xcut, 0]])
            internal_var_vol_weighted = np.sum(var_vol_weighted[pore_in_hull(temp_ctr,var_ctr)])
            # Final calcs for output and mostly for debugging/checking calcs are correct
            total_hull_vol_weighted = volumes[1] + internal_var_vol_weighted
            per_tdens = (total_hull_vol_weighted - internal_var_vol_weighted) / total_hull_vol_weighted
            # Debug Plot
            if i == 0 and cl_args.plot:
                debug_plot(temp_ctr,var_ctr)
            return [t_frames[time_step], internal_var_vol_weighted, total_hull_vol_weighted, per_tdens] + volumes


        case DimensionCase.D2_ELEM:
            # db("Processing 2D Elemental case")
            c_int = np.rint(c)
            mesh_ctr = mesh_center_dir_independent(x,y)
            mesh_vol = mesh_vol_dir_independent(x,y)
            # Empties
            zeros = np.zeros_like(c_int)
            volumes = []
            grain_centroids = []
            # Calculate individual UniqueGrains volumes and centroids
            for n in range(op_max):
                volumes.append(np.sum(np.where(c_int==(n-1),mesh_vol,zeros)))
                if volumes[n] > 0.0 and n > 0:
                    grain_centroids.append([ np.sum(np.where(c_int==(n-1),mesh_ctr[:,0] * mesh_vol,zeros)) / np.sum(np.where(c_int==(n-1),mesh_vol,zeros)),
                                                np.sum(np.where(c_int==(n-1),mesh_ctr[:,1] * mesh_vol,zeros)) / np.sum(np.where(c_int==(n-1),mesh_vol,zeros))])
            # If there is only 1 or 0 grain centroids then skip this timestep and return -1
            if len(grain_centroids) < 2:
                return -1
            # Alter grain centroids to account for quarter hull
            for n in range(len(grain_centroids)):
                if grain_centroids[n][0] > grain_centroids[n][1]:
                    grain_centroids[n][0] = cl_args.max_xy
                elif grain_centroids[n][1] > grain_centroids[n][0]:
                    grain_centroids[n][1] = cl_args.max_xy
            # Element centers for grains and voids
            # grain_ctr = np.delete(mesh_ctr, np.where((c_int<0.0))[0], axis=0)
            # For if not using centroids for the convex hull
            # grain_vol = np.delete(mesh_vol, np.where((c_int<0.0))[0], axis=0)
            void_ctr = np.delete(mesh_ctr, np.where((c_int>=0.0))[0], axis=0)
            void_vol = np.delete(mesh_vol, np.where((c_int>=0.0))[0], axis=0)
            # For quarter hull add the corner point
            temp_ctr = np.append(grain_centroids,[[cl_args.max_xy,cl_args.max_xy]],axis=0)#grain_ctr
            internal_pore_vol = np.sum(void_vol[pore_in_hull(temp_ctr,void_ctr)])
            # For using whole grain and not centroids
            # grain_hull = np.sum(grain_vol[pore_in_hull(grain_ctr,grain_ctr,1e-12,point_plot_TF=False)])
            # Do other calcs
            total_hull_vol = sum(volumes[1:]) + internal_pore_vol
            per_tdens = (total_hull_vol - internal_pore_vol) / total_hull_vol
            # Debug Plot
            if i == 0 and cl_args.plot:
                debug_plot(temp_ctr,void_ctr)
            return [t_frames[time_step], internal_pore_vol, total_hull_vol, per_tdens] + volumes


        case DimensionCase.D3_NODAL:
            # db("Processing 3D Nodal case")
            # Phi mainly
            mesh_ctr = mesh_center_dir_independent(x,y,z)
            mesh_vol = mesh_vol_dir_independent(x,y,z)
            # Total volume of variable and not_variable (meshVolArray*variableValueArray)
            volumes = []
            volumes.append(np.sum(mesh_vol * c)) # Variable
            volumes.append(np.sum(mesh_vol * (1-c))) # Not Variable (all else)
            # Full Domain var and other assignments
            other_ctr = np.delete(mesh_ctr, np.where((c>=cl_args.threshold))[0], axis=0)
            var_ctr = np.delete(mesh_ctr, np.where((c<cl_args.threshold))[0], axis=0)
            var_vol_weighted = np.delete(mesh_vol*c, np.where((c<cl_args.threshold))[0], axis=0)
            # Calculate in the hull
            if cl_args.complexHull:
                # Calculate the centroids of the grains from UG for only internal porosity
                xug,yug,zug,cug = MF.get_data_at_time('unique_grains',t_frames[time_step])
                c_int = np.rint(cug)
                # Calculate x max and y max plane grain centroids
                xcents = calculate_cents_on_max_plane(xug, yug, zug, cug, 'x')
                ycents = calculate_cents_on_max_plane(xug, yug, zug, cug, 'y')

                # Define corner points based on cl_args
                corner_points = np.array([[cl_args.max_xy, cl_args.max_xy, 0],
                                        [cl_args.max_xy, cl_args.max_xy, cl_args.max_z]])

                # Combine xcents, ycents, and corner points into temp_ctr
                temp_ctr = np.vstack([xcents, ycents, corner_points])
                # print(temp_ctr)
            else:
                temp_ctr = np.append(other_ctr,[[cl_args.max_xy,cl_args.max_xy]],axis=0)
            internal_var_vol_weighted = np.sum(var_vol_weighted[pore_in_hull(temp_ctr,var_ctr)])
            # Final calcs for output and mostly for debugging/checking calcs are correct
            total_hull_vol_weighted = volumes[1] + internal_var_vol_weighted
            per_tdens = (total_hull_vol_weighted - internal_var_vol_weighted) / total_hull_vol_weighted
            # Debug plot on first frame if enabled
            if i == 0 and cl_args.plot:
                debug_plot(temp_ctr,var_ctr)
            return [t_frames[time_step], internal_var_vol_weighted, total_hull_vol_weighted, per_tdens] + volumes

        case DimensionCase.D3_ELEM:
            db("Processing 3D Elemental case")
            raise ValueError('Elemental calculations not set up yet!')

        case _:
            raise ValueError("Unknown dimension case")











# ███╗   ███╗ █████╗ ██╗███╗   ██╗
# ████╗ ████║██╔══██╗██║████╗  ██║
# ██╔████╔██║███████║██║██╔██╗ ██║
# ██║╚██╔╝██║██╔══██║██║██║╚██╗██║
# ██║ ╚═╝ ██║██║  ██║██║██║ ╚████║
# ╚═╝     ╚═╝╚═╝  ╚═╝╚═╝╚═╝  ╚═══╝

#IF IN MAIN PROCESS
if __name__ == "__main__":
    tracemalloc.start()
    # Run Time File Make if flagged to
    if cl_args.time:
        time_file_make()
    times_files_names, out_namebases = find_npy_files()
    num_files = len(times_files_names)
    all_ti = time.perf_counter()
    for cnt,file_name in enumerate(times_files_names):
        pt(' ')#\x1b[31;1m
        pt('\033[1m\033[96m'+'File '+str(cnt+1)+'/'+str(num_files)+': '+'\x1b[0m'+str(out_namebases[cnt]))
        # General stuff before calculation
        verb('Initialization for: '+str(out_namebases[cnt]))
        read_ti = time.perf_counter()
        if file_name == '-1':
            prefix = ''
        else:
            prefix = file_name
        time_file_name = prefix + 'times_files.npy'
        times_files = np.load(time_file_name) # Read the times_files.npy
        dimcase = determine_dimension_case() # Determine 2d/3d and nodal/elemental for calc
        idx_frames, t_frames = time_info(times_files) # frames and the time at each one
        op_max, csv_header = t0_opCount_headerBuild(prefix,times_files,idx_frames,t_frames) # Number of OPs and csv header for output
        verb('Initialization done: '+str(round(time.perf_counter()-read_ti,2))+'s')
        verb(' ')

        loop_ti = time.perf_counter()
        results = []
        # Serial calculation
        if cl_args.cpus == 1:
            verb('Running in serial')
            for i, frame in enumerate(tqdm(idx_frames, desc="Calculating")):
                results.append(para_volume_calc(frame,i,t_frames,dimcase,times_files,prefix,op_max))
        # Parallel calculation
        elif cl_args.cpus > 1:
            total_frames = len(idx_frames)
            verb(f'Running in parallel with {cl_args.cpus} CPUs')
            # Not perfect since it progresses tqdm at the start of each calculation not the end
            results = Parallel(n_jobs=cl_args.cpus)(delayed(para_volume_calc)(frame, i, t_frames, dimcase, times_files, prefix, op_max) for i, frame in enumerate(tqdm(idx_frames)))

            # # Use tqdm_joblib for the progress bar- its making extra bars
            # with tqdm_joblib(tqdm(total=total_frames, desc="Calculating")) as progress_bar:
            #     results = Parallel(n_jobs=cl_args.cpus)(
            #         delayed(para_volume_calc)(frame, i, t_frames, dimcase, times_files, prefix, op_max) for i, frame in enumerate(idx_frames)
            #     )

        else:
            maxCPU = mp.cpu_count()
            raise ValueError('Number of CPUs needs to be an integer between 1 and '+str(maxCPU)+
                             ', not '+str(cl_args.cpus))
        pt("Calculation Time: "+str(round(time.perf_counter()-loop_ti,2))+"s")
        verb("Aggregating data...")
        # Compile the data and save it as a csv
        results = [r for r in results if r != -1]
        out_volumes = np.asarray(results)
        # print(out_volumes)
        out_volumes = out_volumes[out_volumes[:, 0].astype(float).argsort()]
        verb('\n' + "Done Building Data")
        saveloc = out_name(prefix)
        np.savetxt(saveloc, np.asarray(out_volumes), delimiter=',', header=','.join(csv_header), comments='')
        current, peak =  tracemalloc.get_traced_memory()
        verb('Memory after calc (current, peak): '+str(round(current/1048576,1))+', '+str(round(peak/1048576,1))+'MB')
        verb(' ')

    quit()
quit()




