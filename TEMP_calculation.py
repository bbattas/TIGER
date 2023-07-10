from MultiExodusReader import MultiExodusReader
import multiprocessing as mp
from CalculationsV2 import CalculationsV2
# from CalculationEngine import para_time_build

import json
import argparse
import logging
pt = logging.warning
verb = logging.info
# from logging import warning as pt
# from logging import info as verb
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
# from time import timecalc =
import os
import glob
import pandas as pd
import math
import sys
import tracemalloc

# This is the 2D conversion of parallel_3d_volume_from_timeFile
# 1st command line input is the number of cpus
# 2nd command line input is 'skip' if you want to skip the last unique file

# # if len(sys.argv) > 1:
# #     n_cpu = int(sys.argv[1])
# # else:
# n_cpu = 1
# var_to_plot = 'unique_grains' # OPs cant be plotted, needs to be elements not nodes
# # z_plane = 10000#19688/2
# sequence = True
# n_frames = 300
# cutoff = 0.0
# # Only for quarter structure hull adding the top right corner points and the centroid
# quarter_hull = True
# max_xy = 30000#1000
# #ADD OUTSIDE BOUNDS ERROR!!!!!!!!!!!!!!
# dirName = os.path.split(os.getcwd())[-1]

#EXODUS FILE FOR RENDERING
#ANY CHARACTER(S) CAN BE PLACED IN PLACE OF THE *, EG. 2D/grain_growth_2D_graintracker_out.e.1921.0000 or 2D/grain_growth_2D_graintracker_out.e-s001
# filenames = '2D/grain_growth_2D_graintracker_out.e*'
#IF IN MAIN PROCESS
# calc = CalculationEngine()
# print("Into main")
# pt(calc.cl_args)
#
# pt("pt test warning")
# verb("verbose test info")
#
# quit()

if __name__ == "__main__":
    print("__main__ Start")
    # xlist = []
    # x1 = np.asarray([0, 1, 2, 3, 4, 5, 6, 7])
    # x2 = np.asarray([10, 11, 12, 13, 14, 15, 16, 17])

    # xlist.append(x1)
    # xlist.append(x2)
    # x = np.vstack(xlist)
    # print(x)
    # print(" ")
    # mask = [0,3,7,4]
    # temp = []
    # test1 = x[:,0]
    # for n in mask:
    #     temp.append(x[:,n])
    #     test1 = test1 + x[:,n]
    # check = np.hstack(temp)

    # print(check)
    # print(test1)
    # test21 = [np.asarray([n1, n2, n3, n4]) for (n1,n2,n3,n4) in zip(x[:,0], x[:,3], x[:,7], x[:,4])]
    # print(test21)
    # test2 = np.asarray([np.asarray([n1, n2, n3, n4]) for (n1,n2,n3,n4) in zip(x[:,0], x[:,3], x[:,7], x[:,4])])

    # print(test2)






    calc = CalculationsV2()
    # print(calc.__dict__)
    print("Testing some shit:")

    calc_it = calc.get_frames()
    print(calc.file_names)
    frames = [0]
    read_ti = time.perf_counter()
    MF = MultiExodusReader(calc.file_names[0])
    # MF = MultiExodusReader('2D_NS_200iw_nemesis.e.12*')
    read_tf = time.perf_counter()
    print("  Finished reading files:",round(read_tf-read_ti,2),"s")

    # REMEMBERR IF NO MESH ADAPTIVITY CAN JUST OPEN THEM ALL!!!!
    for idx in frames:
        pt('Frame '+str(idx)+'/'+str(len(frames)))
        # MF = MultiExodusReader(calc.files[idx])
        x,y,z,c = MF.get_data_at_time(calc.var_to_plot,calc.times[idx],True)
        print('Got the data at the time')
        nx, ny, nz, nc = calc.plane_slice(x,y,z,c)
        calc.plot_slice(idx,nx,ny,nz,nc)


        # print(nx)
        # print(ny)
        # print(nz)
        # print(nc)
        # pc = np.average(nc, axis=1)
        # print(pc)

        # # plott
        # fig, ax = plt.subplots()
        # coords = np.asarray([ np.asarray([x_val,y_val]).T for (x_val,y_val) in zip(ny,nz) ])
        # # coords, vols = calc.mesh_center_quadElements(y,z)
        # print(coords)
        # #USE POLYCOLLECTION TO DRAW ALL POLYGONS DEFINED BY THE COORDINATES
        # p = PolyCollection(coords, cmap=matplotlib.cm.coolwarm, alpha=1)#,edgecolor='k'      #Edge color can be set if you want to show mesh

        # #COLOR THE POLYGONS WITH OUR VARIABLE
        # ## Map plot variable range to color range
        # c_min = np.amin(c)
        # c_max = np.amax(c)
        # colors = pc#(c - c_min)/(c_max-c_min)
        # #
        # p.set_array(np.array(colors) )

        # #ADD THE POLYGON COLLECTION TO AXIS --> THIS IS WHAT ACTUALLY PLOTS THE POLYGONS ON OUR WINDOW
        # ax.add_collection(p)

        # #FIGURE FORMATTING

        # #SET X AND Y LIMITS FOR FIGURE --> CAN USE x,y ARRAYS BUT MANUALLY SETTING IS EASIER
        # # ax.set_xlim([0,300])
        # # ax.set_ylim([0,300])
        # ax.set_xlim([np.amin(ny),np.amax(ny)])
        # ax.set_ylim([np.amin(nz),np.amax(nz)])
        # #SET ASPECT RATIO TO EQUAL TO ENSURE IMAGE HAS SAME ASPECT RATIO AS ACTUAL MESH
        # ax.set_aspect('equal')

        # #ADD A COLORBAR, VALUE SET USING OUR COLORED POLYGON COLLECTION
        # fig.colorbar(p,label="phi")
        # plt.show()






    # dict = {}
    # dict['cl_args'] = vars(calc.cl_args)
    # dict['params'] = {
    #     'adaptive_mesh':calc.adaptive_mesh,
    #     'test' : 7
    # }
    # dict['file_names'] = list(calc.file_names)
    # # dict['testvalue'] = list(calc.file_names)
    # print(calc.file_names)
    # print(dict)
    # with open('tiger_meta.json', 'w') as fp:
    #     json.dump(dict, fp, indent=4)

    pt("END OF THE MAIN")
    # with open('tiger_meta.json','r') as json_file:
    #     data = json.load(json_file)
    # print(data)
    # print(" ")
    # print(data['cl_args'])


    quit()
quit()






if __name__ == "__main__":
    print("__main__ Start")
    calc = CalculationEngine()
    # If it exited to run the parallel_times
    if calc.cl_args.parallel_times == 0:
        verb('Parallelizing with ' + str(calc.cl_args.cpu) + ' cpus')
        cpu_pool = mp.Pool(calc.cl_args.cpu)
        verb(cpu_pool)
        pool_t0 = time.perf_counter()
        results = []
        for i,file in enumerate(calc.file_names):
            results.append(cpu_pool.apply_async(para_time_build,args = (i, file, calc.len_files )))
        cpu_pool.close()
        cpu_pool.join()
        verb("Total Pool Time: "+str(round(time.perf_counter()-pool_t0))+"s")
        verb("Aggregating data...")#Restructuring
        verb(results[0])
        # verb(results[0].get())
        results = [r.get() for r in results]
        time_file_list = []
        for row1 in results:
            for row2 in row1:
                time_file_list.append(row2)
        times_files = np.asarray(time_file_list)
        times_files = times_files[times_files[:, 0].astype(float).argsort()]
        np.save('times_files.npy', times_files)
        verb('Done Building Time Data')
        sys.argv.append('--parallel-times=1')
        verb('Re-entering CalculationEngine')
        calc = CalculationEngine()


    pt("END OF THE MAIN")

    quit()
quit()
print(calc.cl_args.new_meta)
# if "--new-meta" in sys.argv:
# parser = argparse.ArgumentParser()
# parser.add_argument('--new-meta', action='store_true')
# parser.add_argument('--cpu','-n', default=1,type=int)
#
# args = parser.parse_args()
# print(args)
# print(args.new_meta)

times_files = np.load('large_2D_times_files.npy')
times = times_files[:,0].astype(float)
t_step = times_files[:,2].astype(int)


# quit()
# if "--cpu*" in sys.argv:
#     n_cpu =

if os.path.exists("test.json") and not calc.cl_args.new_meta:
    print("EXISTS")
elif calc.cl_args.new_meta or not os.path.exists("test.json"):
    if os.path.exists("test.json"):
        print("deleting old metadata and writing new")
    else:
        print("Writing new metadata")
else:
    print("problem")


dict = {}
dict['times_files'] = times_files.tolist()
# dict['times'] = list(times)
# dict['file_names'] = list(np.unique(times_files[:,1].astype(str)))
# dict['dimension'] = 2
print(dict)
# print(dict['dimension'])

with open('test.json', 'w') as fp:
    json.dump(dict, fp)

quit()
with open('test.json') as json_file:
    data = json.load(json_file)

print(data)
print(data['times_files'])
print(" ")
tf = np.asarray(data['times_files'])
print(tf)
print(tf[:,0].astype(float))
# print(dict['dimension'])
# print(dict['times'])
quit()

times_files = np.load('times_files.npy')
times = times_files[:,0].astype(float)
t_step = times_files[:,2].astype(int)

#GETTING CLOSEST TIME STEP TO DESIRED SIMULATION TIME FOR RENDER --> TYPICALLY 200 FRAMES WITH 20 FPS GIVES A GOOD 10 S LONG VIDEO
# n_frames = 200
if sequence == True:
    if n_frames < len(times):
        t_max = times[-1]
        # t_max = max(times)
        t_frames =  np.linspace(0.0,t_max,n_frames)
        idx_frames = [ np.where(times-t_frames[i] == min(times-t_frames[i],key=abs) )[0][0] for i in range(n_frames) ]
        idx_frames = list( map(int, idx_frames) )
    else:
        t_frames = times
        idx_frames = range(len(times))
elif sequence == False:
    t_frames = times
    idx_frames = range(len(times))
else:
    raise ValueError('sequence has to be True or False, not: ' + str(sequence))

if cutoff != 0:
    print("Cutting End Time to ",cutoff)
    t_frames = [x for x in t_frames if x <= cutoff]
    idx_frames = range(len(t_frames))

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
        print(len(grain_centroids))
        for n in range(len(grain_centroids)):
            if grain_centroids[n][0] > grain_centroids[n][1]:
                grain_centroids[n][0] = max_xy
            elif grain_centroids[n][1] > grain_centroids[n][0]:
                grain_centroids[n][1] = max_xy
    print(grain_centroids)
    print(grain_centroids[0])
    print(grain_centroids[0][1])
    grain_ctr = np.delete(mesh_ctr, np.where((c_int<0.0))[0], axis=0)
    # For if using centroids for the convex hull
    # grain_vol = np.delete(mesh_vol, np.where((c_int<0.0))[0], axis=0)
    void_ctr = np.delete(mesh_ctr, np.where((c_int>=0.0))[0], axis=0)
    void_vol = np.delete(mesh_vol, np.where((c_int>=0.0))[0], axis=0)

    # internal_pore_vol = np.sum(void_vol[pore_in_hull(grain_ctr,void_ctr,1e-12,point_plot_TF=False)])
    if quarter_hull == True:
        temp_ctr = np.append(grain_ctr,[[max_xy,max_xy]],axis=0)
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
