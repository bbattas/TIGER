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
from matplotlib.colors import LogNorm

from scipy.spatial import ConvexHull, Delaunay
import numpy as np
import time
# from time import time
import os
import glob
import pandas as pd
import tracemalloc
import logging
import argparse
import re
from enum import Enum
from tqdm import tqdm


pt = logging.warning
verb = logging.info
db = logging.debug


# # Defaults for the variables
# # WARNING: manually listed nodal vs elemental options for now! should try to add that to MER later
# class default_vals:
#     cpus = 2
#     n_frames = 0
#     cutoff = 0.0
#     var_to_plot = 'phi'
#     var_threshold = 0.5
#     out_name = None
#     leg = None
#     # nodal_var_names = ['phi', 'wvac', 'wint', 'bnds', 'gr0', 'gr1']
#     # elem_var_names = ['T', 'unique_grains']

def argparser():
    class default_vals:
        cpus = 2
        n_frames = 0
        cutoff = 0.0
        var_to_plot = 'phi'
        var_threshold = 0.5
        out_name = None
        leg = None
    # CL Argument Parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose','-v', action='store_true', help='Verbose output, default off')
    parser.add_argument('--cpus','-n',type=int, default=default_vals.cpus,
                                help='How many cpus, default='+str(default_vals.cpus))
    parser.add_argument('--var','-i',type=str, default=default_vals.var_to_plot,
                                help='What variable to plot/calc, default='+str(default_vals.var_to_plot))
    parser.add_argument('--output','-o',type=str, default=default_vals.out_name,
                                help='Custom output image name, default='+str(default_vals.out_name))
    parser.add_argument('--legend','-l',type=str, default=default_vals.leg,
                                help='Custom colorbar legend name, default='+str(default_vals.leg))
    parser.add_argument('--log',action='store_true',
                                help='Use log scale for plotting, default=False')
    parser.add_argument('--scale',action='store_true',
                                help='Force plotting scale to [0-1], default=False')
    parser.add_argument('--exo','-e',action='store_true',
                                help='Look for and use Exodus files instead of Nemesis, default=False')
    parser.add_argument('--force','-p',action='store_true',
                                help='If parallel isnt working try this to use fork method on MultiPool, default=False')
    # parser.add_argument('--threshold','-t',type=float, default=default_vals.var_threshold,
    #                             help='''What value to set as the threshold on the variable for plot/calc, '''
    #                             '''default='''+str(default_vals.var_threshold))
    parser.add_argument('--dim','-d',type=int,default=2, choices=[2,3],
                                help='Dimensions for grain size calculation (Default=2)')
    parser.add_argument('--full_nodal',action='store_true',
                                help='Use full nodal output for plotting (instead of element average), default=False')
    parser.add_argument('--time',action='store_true',
                                help='Include time on each frame image, default=False')
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
    return cl_args

def verb_log():
    # Toggle verbose
    if cl_args.verbose == True:
        logging.basicConfig(level=logging.INFO,format='%(message)s')
    elif cl_args.verbose == False:
        logging.basicConfig(level=logging.WARNING,format='%(message)s')
    verb('Verbose Logging Enabled')
    verb(cl_args)
    verb(' ')






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


def picdir():
    '''Creates directory called 'pics' in CWD if it doesnt exist already.
    '''
    pic_directory = 'pics'
    if not os.path.isdir(pic_directory):
        db('Making picture directory: '+pic_directory)
        os.makedirs(pic_directory)

def time_info(MF):
    '''Returns list of times for each frame from MultiExodusReader object
    Set up to use a soft sequence (if frames >= timesteps doesnt use sequence)

    Args:
        MF: MultiExodusReader object with * for all .e*files

    Raises:
        ValueError: n_frames cl_arg value issue (T/F)

    Returns:
        idx_frames: List of frame iterations/numbers
        t_frames: List of time values associated with each frame (idx_frames)
    '''
    times = MF.global_times
    if cl_args.n_frames > 0:
        if cl_args.n_frames < len(times):
            t_max = times[-1]
            # t_max = max(times)
            t_frames =  np.linspace(0.0,t_max,cl_args.n_frames)
            idx_frames = [ np.where(times-t_frames[i] == min(times-t_frames[i],key=abs) )[0][0] for i in range(cl_args.n_frames) ]
            idx_frames = list( map(int, idx_frames) )
        else:
            t_frames = times
            idx_frames = range(len(times))
    elif cl_args.n_frames == 0:
        t_frames = times
        idx_frames = range(len(times))
    else:
        raise ValueError('n_frames has to be 0 for no sequence or an integer, not: ' + str(cl_args.n_frames))

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
    e_names.sort(key=natural_sort_key)
    verb('Files to use: ')
    verb(e_names)
    verb(' ')
    return e_names

def out_name_base(file_name):
    """
    Generate an figure frame file name based on the input file name and dimensionality.

    Args:
        file_name (str): The input file name.

    Returns:
        str: The generated output file name.

    Raises:
        ValueError: If the dimension is not 2 or 3.
    """
    if cl_args.output:
        return 'pics/' + str(cl_args.output)
    else:
        # Ending based on dimensions
        if cl_args.dim == 2:
            dim = '2D'
        elif cl_args.dim == 3:
            dim = '3D'
        else:
            raise ValueError('Output name needs CL --dim/-d to be 2 or 3 not '+str(cl_args.dim))
        # Beginning based on subdir
        if cl_args.subdirs:
            outBase = file_name.split('/')[0]
        else:
            outBase = os.path.split(os.getcwd())[-1]
        picName = 'pics/' + outBase + '_' + str(cl_args.var) + '_' + dim
        return picName

class DimensionCase(Enum):
    """
    Enumeration to represent different dimensional plotting cases.
    """
    D2_NODAL = 1
    D2_ELEM = 2
    D3_NODAL = 3
    D3_ELEM = 4

def determine_dimension_case():
    """
    Determine the plotting case based on the variable (nodal/elemental) and dimension (2D/3D).

    Returns:
        DimensionCase: The determined dimension case.

    Raises:
        ValueError: If the dimension is not 2 or 3 or if the variable is not found.
    """
    if cl_args.var in MF.exodus_readers[0].nodal_var_names and cl_args.full_nodal:
        if cl_args.dim == 2:
            verb('2D Nodal Calculations')
            verb('For now only setup to take the average of each element still')
            return DimensionCase.D2_NODAL
        elif cl_args.dim == 3:
            verb('3D Nodal Calculations')
            return DimensionCase.D3_NODAL
        else:
            raise ValueError("Invalid dimension: must be 2 or 3.")
    elif (cl_args.var in MF.exodus_readers[0].elem_var_names or cl_args.var in MF.exodus_readers[0].nodal_var_names) and cl_args.full_nodal == False:
        if cl_args.dim == 2:
            verb('2D Elemental Calculations')
            return DimensionCase.D2_ELEM
        elif cl_args.dim == 3:
            verb('3D Elemental Calculations')
            return DimensionCase.D3_ELEM
        else:
            raise ValueError("Invalid dimension: must be 2 or 3.")
    else:
        raise ValueError(f"{cl_args.var} not found in either elem_var_names or nodal_var_names.")


def plotting(time_step,i,t_frames,dimension_case,picName):
    x,y,z,c = MF.get_data_at_time(cl_args.var,t_frames[time_step])#,cl_args.full_nodal
    match dimension_case:
        case DimensionCase.D2_NODAL:
            db("Processing 2D Nodal case")
            raise ValueError('2D Full Nodal plotting is not set up yet! Run again without --full_nodal')
        case DimensionCase.D2_ELEM:
            db("Processing 2D Elemental case")
            coords = np.asarray([ np.asarray([x_val,y_val]).T for (x_val,y_val) in zip(x,y) ])
            fig, ax = plt.subplots()
            # Log scale
            if cl_args.log:
                db('Plotting on log scale')
                # Use LogNorm for log scale color mapping
                norm = LogNorm(vmin=np.min(c), vmax=np.max(c))
                p = PolyCollection(coords, cmap=plt.cm.coolwarm, alpha=1, norm=norm)
            else:
                db('Plotting on normal (not log) scale')
                p = PolyCollection(coords, cmap=matplotlib.cm.coolwarm, alpha=1)#,edgecolor='k'
            p.set_array(np.array(c))
            ax.add_collection(p)
            ax.set_xlim([np.amin(x),np.amax(x)])
            ax.set_ylim([np.amin(y),np.amax(y)])
            #SET ASPECT RATIO TO EQUAL TO ENSURE IMAGE HAS SAME ASPECT RATIO AS ACTUAL MESH
            ax.set_aspect('equal')
            # legend to either custom or the name of the thing plotted
            if cl_args.legend:
                cl_name = cl_args.legend
            else:
                cl_name = cl_args.var
            if cl_args.scale:
                p.set_clim(0.0, 1.0)
            fig.colorbar(p, label=cl_name)
            fig.savefig(picName+'_'+str(i)+'.png',dpi=500,transparent=True )
            plt.close()

        case DimensionCase.D3_NODAL:
            db("Processing 3D Nodal case")
            raise ValueError('3D plots not set up yet!')
        case DimensionCase.D3_ELEM:
            db("Processing 3D Elemental case")
            raise ValueError('3D plots not set up yet!')

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
    cl_args = argparser()
    verb_log()
    cwd = os.getcwd()
    file_names = find_files()
    num_files = len(file_names)
    if num_files > 0:
        picdir()
    all_ti = time.perf_counter()
    for cnt,file_name in enumerate(file_names):
        pt(' ')#\x1b[31;1m
        pt('\033[1m\033[96m'+'File '+str(cnt+1)+'/'+str(num_files)+': '+'\x1b[0m'+str(file_name))
        # General stuff before calculation
        verb('Initialization for file: '+str(file_name))
        read_ti = time.perf_counter()
        MF = MultiExodusReader(file_name) # Read the files
        dimcase = determine_dimension_case() # Determine 2d/3d and nodal/elemental for plotting
        idx_frames, t_frames = time_info(MF) # frames and the time at each one
        picName = out_name_base(file_name)

        verb('Initialization done: '+str(round(time.perf_counter()-read_ti,2))+'s')
        verb(' ')
        loop_ti = time.perf_counter()
        results = []
        # Serial calculation
        if cl_args.cpus == 1:
            verb('Running in serial')
            for i, frame in enumerate(tqdm(idx_frames, desc="Plotting")):
                results.append(plotting(frame,i,t_frames,dimcase,picName))
            # compile and save the data
            verb("Plotting Time: "+str(round(time.perf_counter()-loop_ti,2))+"s")
            verb("Aggregating data...")
        # Parallel calculation
        elif cl_args.cpus > 1:
            if cl_args.force:
                verb('Using Fork mode')
                mp.set_start_method('fork')
            total_frames = len(idx_frames)
            with mp.Pool(cl_args.cpus) as pool:
                jobs = [pool.apply_async(func=plotting, args=(frame, i, t_frames, dimcase, picName)) for i, frame in enumerate(idx_frames)]
                pool.close()
                for job in tqdm(jobs,desc='Plotting'):
                    results.append(job.get())
            verb("Total Pool Time: "+str(round(time.perf_counter()-loop_ti,2))+"s")

        else:
            maxCPU = mp.cpu_count()
            raise ValueError('Number of CPUs needs to be an integer between 1 and '+str(maxCPU)+
                             ', not '+str(cl_args.cpus))

        verb('\n' + "Done Plotting")
        current, peak =  tracemalloc.get_traced_memory()
        verb('Memory after calc (current, peak): '+str(round(current/1048576,1))+', '+str(round(peak/1048576,1))+'MB')
        verb(' ')

    quit()

