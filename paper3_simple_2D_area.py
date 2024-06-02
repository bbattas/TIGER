from MultiExodusReader import MultiExodusReader
import multiprocessing as mp
# from VolumeScripts import *

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PolyCollection
from matplotlib.collections import PatchCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D
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


pt = logging.warning
verb = logging.info
db = logging.debug

# Defaults for the variables
# WARNING: manually listed nodal vs elemental options for now! should try to add that to MER later
class default_vals:
    cpus = 2
    n_frames = 300
    cutoff = 0.0
    var_to_plot = 'phi'
    var_threshold = 0.5
    nodal_var_names = ['phi', 'wvac', 'wint', 'bnds', 'gr0', 'gr1']
    elem_var_names = ['T', 'unique_grains']
    xcut = 17000



# CL Argument Parser
parser = argparse.ArgumentParser()
parser.add_argument('--verbose','-v', action='store_true', help='Verbose output, default off')
parser.add_argument('--cpus','-n',type=int, default=default_vals.cpus,
                            help='How many cpus, default='+str(default_vals.cpus))
parser.add_argument('--var','-i',type=str, default=default_vals.var_to_plot,
                            help='What variable to plot/calc, default='+str(default_vals.var_to_plot))
parser.add_argument('--threshold','-t',type=float, default=default_vals.var_threshold,
                            help='''What value to set as the threshold on the variable for plot/calc, '''
                            '''default='''+str(default_vals.var_threshold))
parser.add_argument('--dim','-d',type=int,default=2, choices=[2,3],
                            help='Dimensions for grain size calculation (Default=2)')
parser.add_argument('--xcut','-x',type=float,default=default_vals.xcut,
                            help='Max x to include for hull/calculation, Default='+str(default_vals.xcut))
parser.add_argument('--subdirs','-s',action='store_true',
                            help='Run in all subdirectories (vs CWD), default=False')
parser.add_argument('--sequence',action='store_true',
                            help='Time as a sequence, default=False')
parser.add_argument('--n_frames','-f',type=int, default=default_vals.n_frames,
                            help='''How many frames for if sequence is true, '''
                            '''default='''+str(default_vals.n_frames))
parser.add_argument('--cutoff','-c',type=int, default=default_vals.cutoff,
                            help='''What time to stop at, if 0.0 uses all data. '''
                            '''default='''+str(default_vals.cutoff))

cl_args = parser.parse_args()



# Toggle verbose
if cl_args.verbose == True:
    logging.basicConfig(level=logging.INFO,format='%(message)s')
elif cl_args.verbose == False:
    logging.basicConfig(level=logging.WARNING,format='%(message)s')
verb('Verbose Logging Enabled')
verb(cl_args)
verb(' ')


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
            e_files_in_subdir = [x.rsplit('.',1)[0]+"*" for x in glob.glob(dir_n + "*.e.*")]
            # e_files_in_subdir = glob.glob(dir_n + '*.e*')
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
    e_names.sort(key=natural_sort_key)
    verb('Files to use: ')
    verb(e_names)
    verb(' ')
    return e_names


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
    # Ending based on dimensions
    if cl_args.dim == 2:
        suffix = '_areas.csv'
    elif cl_args.dim == 3:
        suffix = '_volumes.csv'
    else:
        raise ValueError('Output name needs CL --dim/-d to be 2 or 3 not '+str(cl_args.dim))
    # Beginning based on subdir
    if cl_args.subdirs:
        outBase = file_name.split('/')[0]
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


def debug_plot(xyz_for_hull,void_ctr_xyz,tolerance):
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
        ax.scatter(void_in_hull[:, 0], void_in_hull[:, 1],s=0.01,alpha=0.5)
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

def t0_opCount_headerBuild(MF,t_frames):
    """
    Build the header for the output CSV file and return the number of OPs.

    Args:
        MF: Data source object with `get_data_at_time` method.
        t_frames (list): List of time frames.

    Returns:
        tuple: The OP count and CSV header list.
    """
    verb("Calculating initial frame...")
    x,y,z,c = MF.get_data_at_time(cl_args.var,t_frames[0])
    c_int = np.rint(c)
    op_0 = round(np.amax(c_int))+2
    csv_header = ["time", "internal_pore", "total_hull", "vol_density", "total_void"]
    for n in range(1,op_0):
        csv_header.append("Grain_"+str(n))
    return op_0, csv_header

def para_volume_calc(time_step,i,t_frames,dimension_case,op_max):
    """
    Perform area/volume calculations for a given time step.

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
            # temp_ctr = np.append(other_ctr,[[max_xy,max_xy]],axis=0)
            db('Only running on a manually defined box for hull!')
            ytop = np.amax(y)
            temp_ctr = np.array([[0, 0], [0, ytop], [cl_args.xcut, ytop], [cl_args.xcut, 0]])
            internal_var_vol_weighted = np.sum(var_vol_weighted[pore_in_hull(temp_ctr,var_ctr)])
            # Final calcs for output and mostly for debugging/checking calcs are correct
            total_hull_vol_weighted = volumes[1] + internal_var_vol_weighted
            per_tdens = (total_hull_vol_weighted - internal_var_vol_weighted) / total_hull_vol_weighted
            # print(f'Frame {i}: updated progress')
            return [t_frames[time_step], internal_var_vol_weighted, total_hull_vol_weighted, per_tdens] + volumes

        case DimensionCase.D2_ELEM:
            db("Processing 2D Elemental case")
            raise ValueError('Elemental calculations not set up yet!')

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
            # temp_ctr = np.append(other_ctr,[[max_xy,max_xy]],axis=0)
            db('Only running on a manually defined box for hull!')
            ytop = np.amax(y)
            topz = np.amax(z)
            temp_ctr = np.array([[0, 0, 0], [0, ytop,0], [cl_args.xcut, ytop, 0], [cl_args.xcut, 0,0],
                                 [0, 0, topz], [0, ytop,topz], [cl_args.xcut, ytop, topz], [cl_args.xcut, 0,topz]])
            internal_var_vol_weighted = np.sum(var_vol_weighted[pore_in_hull(temp_ctr,var_ctr)])
            # Final calcs for output and mostly for debugging/checking calcs are correct
            total_hull_vol_weighted = volumes[1] + internal_var_vol_weighted
            per_tdens = (total_hull_vol_weighted - internal_var_vol_weighted) / total_hull_vol_weighted
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
    file_names = find_files()
    num_files = len(file_names)
    all_ti = time.perf_counter()
    for cnt,file_name in enumerate(file_names):
        pt(' ')#\x1b[31;1m
        pt('\033[1m\033[96m'+'File '+str(cnt+1)+'/'+str(num_files)+': '+'\x1b[0m'+str(file_name))
        # General stuff before calculation
        verb('Initialization for file: '+str(file_name))
        read_ti = time.perf_counter()
        MF = MultiExodusReader(file_name) # Read the files
        dimcase = determine_dimension_case() # Determine 2d/3d and nodal/elemental for calc
        idx_frames, t_frames = time_info(MF) # frames and the time at each one
        op_max, csv_header = t0_opCount_headerBuild(MF,t_frames) # Number of OPs and csv header for output
        verb('Initialization done: '+str(round(time.perf_counter()-read_ti,2))+'s')
        verb(' ')
        loop_ti = time.perf_counter()
        results = []
        # Serial calculation
        if cl_args.cpus == 1:
            verb('Running in serial')
            for i, frame in enumerate(tqdm(idx_frames, desc="Calculating")):
                results.append(para_volume_calc(frame,i,t_frames,dimcase,op_max))
            # compile and save the data
            pt("Calculation Time: "+str(round(time.perf_counter()-loop_ti,2))+"s")
            verb("Aggregating data...")
        # Parallel calculation
        elif cl_args.cpus > 1:
            total_frames = len(idx_frames)
            with mp.Pool(cl_args.cpus) as pool:
                jobs = [pool.apply_async(func=para_volume_calc, args=(frame, i, t_frames, dimcase, op_max)) for i, frame in enumerate(idx_frames)]
                pool.close()
                for job in tqdm(jobs):
                    results.append(job.get())
            pt("Total Pool Time: "+str(round(time.perf_counter()-loop_ti,2))+"s")
            verb("Aggregating data...")#Restructuring
        else:
            maxCPU = mp.cpu_count()
            raise ValueError('Number of CPUs needs to be an integer between 1 and '+str(maxCPU)+
                             ', not '+str(cl_args.cpus))
        # Compile the data and save it as a csv
        out_volumes = np.asarray(results)
        # print(out_volumes)
        out_volumes = out_volumes[out_volumes[:, 0].astype(float).argsort()]
        verb('\n' + "Done Building Data")
        saveloc = out_name(file_name)
        np.savetxt(saveloc, np.asarray(out_volumes), delimiter=',', header=','.join(csv_header), comments='')
        current, peak =  tracemalloc.get_traced_memory()
        verb('Memory after calc (current, peak): '+str(round(current/1048576,1))+', '+str(round(peak/1048576,1))+'MB')
        verb(' ')

    quit()
