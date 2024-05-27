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


pt = logging.warning
verb = logging.info

# Defaults for the variables
class default_vals:
    cpus = 2
    n_frames = 300
    cutoff = 0.0
    var_to_plot = 'phi'
    var_threshold = 0.5


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




cwd = os.getcwd()
dirName = os.path.split(os.getcwd())[-1]

# times_files = np.load('times_files.npy')
# times = times_files[:,0].astype(float)
# t_step = times_files[:,2].astype(int)

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
        print("Cutting End Time to ",cl_args.cutoff)
        t_frames = [x for x in t_frames if x <= cl_args.cutoff]
        idx_frames = range(len(t_frames))

    # tot_frames = len(idx_frames)
    return idx_frames, t_frames

def find_files():
    e_names = []
    if cl_args.subdirs:
        for dir_n in glob.glob('*/', recursive=True):
            e_files_in_subdir = glob.glob(dir_n + '*.e*')
            if e_files_in_subdir:
                first_file = e_files_in_subdir[0]
                trimmed_file = first_file.split('.e', 1)[0] + '.e*'
                e_names.append(trimmed_file)
    else:
        e_files_in_dir = glob.glob('*.e*')
        if e_files_in_dir:
            first_file = e_files_in_dir[0]
            trimmed_file = first_file.split('.e', 1)[0] + '.e*'
            e_names.append(trimmed_file)
    if not e_names:
        raise ValueError('No files found matching *.e*, make sure to specify subdirectories or not')
    e_names.sort(key=natural_sort_key)
    return e_names

# GOT TO HERE BEFORE I STOPPED!!!






def pore_in_hull(xyz_for_hull,void_ctr_xyz,tolerance,point_plot_TF):
    # print(" Taking convex hull")
    # tic = time.perf_counter()
    hull = ConvexHull(xyz_for_hull)
    # toc = time.perf_counter()
    # print("     ",toc - tic)
    # print(" Doing in hull calculation")
    # get array of boolean values indicating in hull if True
    # tic = time.perf_counter()
    in_hull = np.all(np.add(np.dot(void_ctr_xyz, hull.equations[:,:-1].T),
                            hull.equations[:,-1]) <= tolerance, axis=1)
    # toc = time.perf_counter()
    # print("     ",toc - tic)

    # The void centers that are in the hull
    void_in_hull = void_ctr_xyz[in_hull]
    # Plot a scatterplot of the void mesh centers in the hull
    if point_plot_TF == True:
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
    # Output the boolean array of which void_ctr is in the hull for use
    return in_hull




def t0_opCount_headerBuild(idx_frames):
    print("Calculating initial frame...")
    read_ti = time.perf_counter()
    MF = MultiExodusReader(times_files[idx_frames[0],1])#.exodus_readers
    read_tf = time.perf_counter()
    print("  Finished reading initial frame:",round(read_tf-read_ti,2),"s")

    x,y,z,c = MF.get_data_at_time(var_to_plot,times[0])

    c_int = np.rint(c)

    op_0 = round(np.amax(c_int))+2
    csv_header = ["time", "internal_pore", "total_hull", "vol_density", "total_void"]
    for n in range(1,op_0):
        csv_header.append("Grain_"+str(n))
    return op_0, csv_header



def para_volume_calc(time_step,i,op_max):
    print("Calculating frame",i+1, "/",tot_frames)
    read_ti = time.perf_counter()
    MF = MultiExodusReader(times_files[time_step,1])#.exodus_readers
    read_tf = time.perf_counter()
    print("  Finished reading frame",i+1, ":",round(read_tf-read_ti,2),"s")

    x,y,z,c = MF.get_data_at_time(var_to_plot,times[time_step])# MF.get_data_at_time(var_to_plot,times[i])
    c_int = np.rint(c)

    mesh_ctr = np.asarray([ x[:, 0] + (x[:, 2] - x[:, 0])/2,
                            y[:, 0] + (y[:, 2] - y[:, 0])/2 ]).T

    mesh_vol = np.asarray((x[:, 2] - x[:, 0])*(y[:, 2] - y[:, 0]))

    zeros = np.zeros_like(c_int)
    volumes = []# np.zeros(round(np.amax(c_int))+2)
    # centroids = np.asarray([ volumes, volumes ]).T
    grain_centroids = []
    for n in range(op_max):
        volumes.append(np.sum(np.where(c_int==(n-1),mesh_vol,zeros)))
        if volumes[n] > 0.0 and n > 0:
            grain_centroids.append([ np.sum(np.where(c_int==(n-1),mesh_ctr[:,0] * mesh_vol,zeros)) / np.sum(np.where(c_int==(n-1),mesh_vol,zeros)),
                                        np.sum(np.where(c_int==(n-1),mesh_ctr[:,1] * mesh_vol,zeros)) / np.sum(np.where(c_int==(n-1),mesh_vol,zeros))])
    if quarter_hull == True:
        for n in range(len(grain_centroids)):
            if grain_centroids[n][0] > grain_centroids[n][1]:
                grain_centroids[n][0] = max_xy
            elif grain_centroids[n][1] > grain_centroids[n][0]:
                grain_centroids[n][1] = max_xy

    if len(grain_centroids) < 2:
        # print(grain_centroids)
        return -1
    # print(grain_centroids[0])
    # print(grain_centroids[0][1])
    grain_ctr = np.delete(mesh_ctr, np.where((c_int<0.0))[0], axis=0)
    # For if using centroids for the convex hull
    # grain_vol = np.delete(mesh_vol, np.where((c_int<0.0))[0], axis=0)
    void_ctr = np.delete(mesh_ctr, np.where((c_int>=0.0))[0], axis=0)
    void_vol = np.delete(mesh_vol, np.where((c_int>=0.0))[0], axis=0)

    # internal_pore_vol = np.sum(void_vol[pore_in_hull(grain_ctr,void_ctr,1e-12,point_plot_TF=False)])
    if quarter_hull == True:
        temp_ctr = np.append(grain_centroids,[[max_xy,max_xy]],axis=0)#grain_ctr
        internal_pore_vol = np.sum(void_vol[pore_in_hull(temp_ctr,void_ctr,1e-12,point_plot_TF=False)])
    else:
        internal_pore_vol = np.sum(void_vol[pore_in_hull(grain_ctr,void_ctr,1e-12,point_plot_TF=False)])
    # For if using centroids for the convex hull
    # grain_hull = np.sum(grain_vol[pore_in_hull(grain_ctr,grain_ctr,1e-12,point_plot_TF=False)])
    total_hull_vol = sum(volumes[1:]) + internal_pore_vol
    per_tdens = (total_hull_vol - internal_pore_vol) / total_hull_vol
    # print("Memory:",tracemalloc.get_traced_memory())
    print("  Finished calculating frame",i+1, ":",round(time.perf_counter()-read_tf,2),"s")
    # print([times[i], internal_pore_vol, total_hull_vol, per_tdens] + volumes)
    return [times[time_step], internal_pore_vol, total_hull_vol, per_tdens] + volumes





#IF IN MAIN PROCESS
if __name__ == "__main__":
    tracemalloc.start()
    file_names = find_files()
    print(file_names)
    quit()
    for file_name in file_names:
        # Read Exodus/Nemesis (not adaptive mesh)
        MF = MultiExodusReader(file_name)
    # Time info
    idx_frames, times = time_info(MF)
    # Calculate maximum number of OPs and csv header
    op_max, csv_header = t0_opCount_headerBuild(idx_frames)
    results = []

    if len(sys.argv) > 2:
        if "skip" in sys.argv[2]:
            print("NOTE: Skipping last file as indicated with 'skip' flag")
            print(" ")
            name_unq = name_unq[:-1]

    all_time_0 = time.perf_counter()
    if n_cpu == 1:
        print("Running in series")
        for i,frame in enumerate(idx_frames):
            results.append(para_volume_calc(frame, i, op_max ))
        # compile and save the data
        print("Total Time:",round(time.perf_counter()-all_time_0,2),"s")
        print("Aggregating data...")

    elif n_cpu > 1:
        print("Running in parallel")
        #CREATE A PROCESS POOL
        cpu_pool = mp.Pool(n_cpu)
        print(cpu_pool)
        all_time_0 = time.perf_counter()

        for i,frame in enumerate(idx_frames):
            results.append(cpu_pool.apply_async(para_volume_calc,args = (frame, i, op_max)))#, callback = log_result)
        # ex_files = [cpu_pool.map(para_time_build,args=(file,)) for file in name_unq  ]

        cpu_pool.close()
        cpu_pool.join()
        print(cpu_pool)
        print("Total Pool Time:",round(time.perf_counter()-all_time_0,2),"s")
        print("Aggregating data...")#Restructuring
        results = [r.get() for r in results]
        results = [r for r in results if r != -1]
    else:
        raise(ValueError("ERROR: n_cpu command line flag error"))
    # print(results)
    out_volumes = np.asarray(results)
    # print(out_volumes)
    out_volumes = out_volumes[out_volumes[:, 0].astype(float).argsort()]
    print('\n' + "Done Building Area Data")
    saveloc = '../' + dirName + '_areas.csv'
    np.savetxt(saveloc, np.asarray(out_volumes), delimiter=',', header=','.join(csv_header), comments='')
    current, peak =  tracemalloc.get_traced_memory()
    print("Memory Final (current, peak):",round(current/1048576,1), round(peak/1048576,1), "MB")
    # "volumes.csv",
    quit()

quit()



# np.savetxt("volumes.csv", np.asarray(out_volumes), delimiter=',', header=','.join(csv_header), comments='')

# plt.figure(2)
# plt.scatter(times[idx_frames],pore_array[:,0],label="Internal Pore")
# plt.scatter(times[idx_frames],pore_array[:,4],label="Neck 1")
# plt.scatter(times[idx_frames],pore_array[:,7],label="Neck 2")
# plt.xlabel("Time")
# plt.ylabel("Areas")
# plt.legend()
# plt.show()


# df = pd.DataFrame(columns=['time', 'pore_area','distance', 'g1x', 'g1y', 'g1area', 'g2x', 'g2y', 'g2area'],data=np.hstack((times[idx_frames,None], pore_array[:,:])))
# # print(df)
# # # pd.DataFrame(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
# # #                    columns=['a', 'b', 'c'])
# df.to_csv('../PoreArea.csv',index=False)
