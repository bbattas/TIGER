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

# This is the 3d_plane_data but for when there are too many nemesis/-s files to open
n_cpu = int(sys.argv[1])
var_to_plot = 'unique_grains' # OPs cant be plotted, needs to be elements not nodes
# z_plane = 10000#19688/2
sequence = False
n_frames = 40

#ADD OUTSIDE BOUNDS ERROR!!!!!!!!!!!!!!

#EXODUS FILE FOR RENDERING
#ANY CHARACTER(S) CAN BE PLACED IN PLACE OF THE *, EG. 2D/grain_growth_2D_graintracker_out.e.1921.0000 or 2D/grain_growth_2D_graintracker_out.e-s001
# filenames = '2D/grain_growth_2D_graintracker_out.e*'


times_files = np.load('times_files.npy')
times = times_files[:,0].astype(float)
t_step = times_files[:,2].astype(int)

#GETTING CLOSEST TIME STEP TO DESIRED SIMULATION TIME FOR RENDER --> TYPICALLY 200 FRAMES WITH 20 FPS GIVES A GOOD 10 S LONG VIDEO
# n_frames = 200
if sequence == True:
    t_max = times[-1]
    t_frames =  np.linspace(0.0,t_max,n_frames)
    idx_frames = [ np.where(times-t_frames[i] == min(times-t_frames[i],key=abs) )[0][0] for i in range(n_frames) ]
elif sequence == False:
    t_frames = times
    idx_frames = range(len(times))
else:
    raise ValueError('sequence has to be True or False, not: ' + str(sequence))

tot_frames = len(idx_frames)




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
        ax = fig.add_subplot(111,projection='3d')
        for simplex in hull.simplices:
            plt.plot(grain_ctr[simplex, 0], grain_ctr[simplex, 1], grain_ctr[simplex,2], 'r-')
        # Now plot the void points
        ax.scatter3D(void_in_hull[:, 0], void_in_hull[:, 1], void_in_hull[:, 2],s=0.01,alpha=0.5)
        plt.autoscale()
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

    x,y,z,c = MF.get_data_at_time(var_to_plot,times[i])
    c_int = np.rint(c)

    mesh_ctr = np.asarray([ x[:, 0] + (x[:, 2] - x[:, 0])/2,
                            y[:, 0] + (y[:, 2] - y[:, 0])/2,
                            z[:, 0] + (z[:, 4] - z[:, 0])/2]).T

    mesh_vol = np.asarray((x[:, 2] - x[:, 0])*(y[:, 2] - y[:, 0])*(z[:, 4] - z[:, 0]))

    zeros = np.zeros_like(c_int)
    volumes = []# np.zeros(round(np.amax(c_int))+2)
    centroids = np.asarray([ volumes, volumes, volumes ]).T

    for n in range(op_max):
        volumes.append(np.sum(np.where(c_int==(n-1),mesh_vol,zeros)))

    grain_ctr = np.delete(mesh_ctr, np.where((c_int<0.0))[0], axis=0)
    # For if using centroids for the convex hull
    # grain_vol = np.delete(mesh_vol, np.where((c_int<0.0))[0], axis=0)
    void_ctr = np.delete(mesh_ctr, np.where((c_int>=0.0))[0], axis=0)
    void_vol = np.delete(mesh_vol, np.where((c_int>=0.0))[0], axis=0)

    internal_pore_vol = np.sum(void_vol[pore_in_hull(grain_ctr,void_ctr,1e-12,point_plot_TF=False)])
    # For if using centroids for the convex hull
    # grain_hull = np.sum(grain_vol[pore_in_hull(grain_ctr,grain_ctr,1e-12,point_plot_TF=False)])
    total_hull_vol = sum(volumes[1:]) + internal_pore_vol
    per_tdens = (total_hull_vol - internal_pore_vol) / total_hull_vol
    # print([times[i], internal_pore_vol, total_hull_vol, per_tdens] + volumes)
    return [times[i], internal_pore_vol, total_hull_vol, per_tdens] + volumes





#IF IN MAIN PROCESS
if __name__ == "__main__":
    # Calculate maximum number of OPs and csv header
    op_max, csv_header = t0_opCount_headerBuild(idx_frames)
    #CREATE A PROCESS POOL
    cpu_pool = mp.Pool(n_cpu)
    print(cpu_pool)
    all_time_0 = time.perf_counter()
    results = []
    for i,frame in enumerate(idx_frames):
        results.append(cpu_pool.apply_async(para_volume_calc,args = (frame, i, op_max )))#, callback = log_result)
    # ex_files = [cpu_pool.map(para_time_build,args=(file,)) for file in name_unq  ]
    # print(ex_files)
    print("closing")
    cpu_pool.close()
    print("closed")
    cpu_pool.join()
    print("joined")
    print(cpu_pool)
    print("Total Pool Time:",round(time.perf_counter()-all_time_0,2),"s")
    print("Aggregating data...")#Restructuring
    results = [r.get() for r in results]
    # print(results)
    out_volumes = np.asarray(results)
    # print(out_volumes)
    out_volumes = out_volumes[out_volumes[:, 0].astype(float).argsort()]
    print('\n' + "Done Building Volume Data")
    np.savetxt("volumes.csv", np.asarray(out_volumes), delimiter=',', header=','.join(csv_header), comments='')

    quit()

quit()



np.savetxt("volumes.csv", np.asarray(out_volumes), delimiter=',', header=','.join(csv_header), comments='')

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
