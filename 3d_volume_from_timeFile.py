from MultiExodusReader import MultiExodusReader
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

# This is the 3d_plane_data but for when there are too many nemesis/-s files to open
var_to_plot = 'unique_grains' # OPs cant be plotted, needs to be elements not nodes
# z_plane = 10000#19688/2
sequence = False
n_frames = 40

particleVolumes = False
particleCentroids = False
hullTF = True
plotTF = False

pic_directory = '../pics'

#ADD OUTSIDE BOUNDS ERROR!!!!!!!!!!!!!!

#EXODUS FILE FOR RENDERING
#ANY CHARACTER(S) CAN BE PLACED IN PLACE OF THE *, EG. 2D/grain_growth_2D_graintracker_out.e.1921.0000 or 2D/grain_growth_2D_graintracker_out.e-s001
# filenames = '2D/grain_growth_2D_graintracker_out.e*'

# for file in glob.glob("*.i"):
#     inputName = os",n,"/",file_len,": ",file.path.splitext(file)[0]
# print("Input File is: " + inputName + ".i")
# filenames = inputName + "_out.e*"#*
# print("   Output Files: " + filenames)
dirName = os.path.split(os.getcwd())[-1]

# if the ../pics directory doesnt exist, make it
if not os.path.isdir(pic_directory):
    os.makedirs(pic_directory)


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


# Define pore area array
# pore_area, neck_dist, neck gr1:x,y,area, neck gr2:x,y,area
# pore_array = np.empty((0,8))
out_volumes = []

#LOOP OVER EACH TIME STEP IN idx_frames
for (i,time_step) in enumerate(idx_frames):
    print( "Rendering frame no. ",i+1)

    #READ EXODUS FILE SERIES WITH MultiExodusReader
    tic = time.perf_counter()
    MF = MultiExodusReader(times_files[time_step,1])#.exodus_readers
    toc = time.perf_counter()
    print("File read time: ",toc - tic)
    # print("File is read")
    #GET A LIST OF SIMULATION TIME POINTS
    # times = np.append(times,MF.global_times[t_step[time_step]])
    # print("     Time",times)

    # #GENERATE Fmesh_area = IGURE WINDOW
    # fig, ax = plt.subplots()


    #GET X,Y,Z AND C (UNIQUE GRAINS VALUES) AT CURRENT TIME STEP
    #x,y,z,c = MF.get_data_at_time(var_to_plot,MF.global_times[t_step[time_step]] )#MF.global_times[time_step]               #Read coordinates and variable value --> Will be parallelized in future
    x,y,z,c = MF.get_data_at_time(var_to_plot,times[i])
    #x,y,z,c = MF.get_data_from_file_idx(var_to_plot,times[i],t_step[i])
    # print("data is pulled")


    # NEEDS TO SWITCH TO LISTS LIKE IN HULLTF
    if particleVolumes == True:
        c_int = np.rint(c)
        mesh_vol = np.asarray((x[:, 2] - x[:, 0])*(y[:, 2] - y[:, 0])*(z[:, 4] - z[:, 0]))
        zeros = np.zeros_like(c_int)

        volumes = np.zeros(round(np.amax(c_int))+2)
        #

        for n in range(len(volumes)):
            # print(n)
            volumes[n] = np.sum(np.where(c_int==(n-1),mesh_vol,zeros))




    # MAKE IT PART OF THE PARTICLEAREAS or check for dupes
    # NEEDS TO SWITCH TO LISTS LIKE IN HULLTF
    if particleCentroids == True:
        c_int = np.rint(c)
        mesh_ctr = np.asarray([ x[:, 0] + (x[:, 2] - x[:, 0])/2,
                                y[:, 0] + (y[:, 2] - y[:, 0])/2,
                                z[:, 0] + (z[:, 4] - z[:, 0])/2]).T

        mesh_vol = np.asarray((x[:, 2] - x[:, 0])*(y[:, 2] - y[:, 0])*(z[:, 4] - z[:, 0]))

        zeros = np.zeros_like(c_int)
        volumes = np.zeros(round(np.amax(c_int))+2)
        centroids = np.asarray([ volumes, volumes, volumes ]).T

        for n in range(len(volumes)):
            volumes[n] = np.sum(np.where(c_int==(n-1),mesh_vol,zeros))
            if volumes[n] > 0.0:
                # centroids[n] = np.asarray([ np.sum(np.where(c_int==(n-1),mesh_ctr[:,0],zeros) * np.where(c_int==(n-1),mesh_area,zeros)) / np.sum(np.where(c_int==(n-1),mesh_area,zeros)),
                #                             np.sum(np.where(c_int==(n-1),mesh_ctr[:,1],zeros) * np.where(c_int==(n-1),mesh_area,zeros)) / np.sum(np.where(c_int==(n-1),mesh_area,zeros))]).T
                centroids[n] = np.asarray([ np.sum(np.where(c_int==(n-1),mesh_ctr[:,0] * mesh_vol,zeros)) / np.sum(np.where(c_int==(n-1),mesh_vol,zeros)),
                                            np.sum(np.where(c_int==(n-1),mesh_ctr[:,1] * mesh_vol,zeros)) / np.sum(np.where(c_int==(n-1),mesh_vol,zeros)),
                                            np.sum(np.where(c_int==(n-1),mesh_ctr[:,2] * mesh_vol,zeros)) / np.sum(np.where(c_int==(n-1),mesh_vol,zeros)) ]).T



    if hullTF == True:
        if (particleVolumes != True) and (particleCentroids != True):
            c_int = np.rint(c)
            # print(" Taking mesh centers")
            tic = time.perf_counter()
            mesh_ctr = np.asarray([ x[:, 0] + (x[:, 2] - x[:, 0])/2,
                                    y[:, 0] + (y[:, 2] - y[:, 0])/2,
                                    z[:, 0] + (z[:, 4] - z[:, 0])/2]).T
            # toc = time.perf_counter()
            # print("     ",toc - tic)
            # print(" Taking mesh volumes")
            # tic = time.perf_counter()

            mesh_vol = np.asarray((x[:, 2] - x[:, 0])*(y[:, 2] - y[:, 0])*(z[:, 4] - z[:, 0]))

            # toc = time.perf_counter()
            # print("     ",toc - tic)

            zeros = np.zeros_like(c_int)
            volumes = []# np.zeros(round(np.amax(c_int))+2)
            centroids = np.asarray([ volumes, volumes, volumes ]).T
            # print(" Calculating volumes")
            # tic = time.perf_counter()
            for n in range(round(np.amax(c_int))+2):
                volumes.append(np.sum(np.where(c_int==(n-1),mesh_vol,zeros)))
            # toc = time.perf_counter()
            # print("     ",toc - tic)
                # volumes[n] = np.sum(np.where(c_int==(n-1),mesh_vol,zeros))

        grain_ctr = np.delete(mesh_ctr, np.where((c_int<0.0))[0], axis=0)
        void_ctr = np.delete(mesh_ctr, np.where((c_int>=0.0))[0], axis=0)
        void_vol = np.delete(mesh_vol, np.where((c_int>=0.0))[0], axis=0)

        # fig = plt.figure()
        # ax = fig.add_subplot(111,projection='3d')
        # ax.scatter(void_ctr[:,0],void_ctr[:,1],void_ctr[:,2],s=0.001)
        # # ax.scatter(grain_ctr[:,0],grain_ctr[:,1],grain_ctr[:,2],s=0.001)
        # plt.show()

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
        # tic1 = time.perf_counter()
        internal_pore_vol = np.sum(void_vol[pore_in_hull(grain_ctr,void_ctr,1e-12,point_plot_TF=False)])
        # toc1 = time.perf_counter()
        # print("Total internal pore calculaction: ",toc1 - tic1)
        out_volumes.append([times[i], internal_pore_vol] + volumes)#np.insert(volumes, 0, internal_pore_vol, axis=0)

        if i == 0:
            csv_header = ["time", "internal_pore", "total_void"]
            for n in range(1,len(volumes)):
                csv_header.append("Grain_"+str(n))
        # np.savetxt("volumes.csv", np.asarray(out_volumes), delimiter=",", header=','.join(csv_header), comments='')




    if plotTF == True:
        print("Plotting...")
        #GENERATE FIGURE WINDOW AND SUBPLOT AXIS
        fig = plt.figure()
        ax = fig.add_subplot(111,projection='3d')#'111'

        #CREATE A COLORMAP OBJECT (cw) THAT CONVERTS VARIABLE VALUE c INTO A RGBA VALUE
        c_min = np.amin(c)
        c_max = np.amax(c)
        #USING COLORMAP HSV. MATPLOTLIB HAS A LIST OF COLORMAPS YOU CAN USE, OR YOU CAN EVEN GENERATE YOUR OWN CUSTOM MAP
        cw = matplotlib.cm.ScalarMappable(cmap=matplotlib.cm.hsv)
        cw.set_array([c_min,c_max])

        #CONVERT VARIABLE VALUES TO RGBA VALUES
        C = cw.to_rgba(c)

        #LIST OF SURFACES, AND THEIR CORRESPONDING COLORS
        surfaces = []
        colors = []

        #GENERATE CORNER POINT COORDINATES FOR THE QUAD8 MESH POLYGONS
        coords = np.asarray([ np.asarray([x_val,y_val,z_val]).T for (x_val,y_val,z_val) in zip(x,y,z) ])

        #GENERATE THE 6 SIDES FOR EACH POLYGON, AND ASSIGN THE CELL VALUE OF THE POLYGON TO THE sides
        #THIS CODE IS VERY UNOPTIMIZED, AND WILL BE IMPROVED IN THE FUTURE
        for (i,side) in enumerate(coords):
            sides = [ [side[0],side[1],side[2],side[3]],
                             [side[4],side[5],side[6],side[7]],
                             [side[0],side[1],side[5],side[4]],
                             [side[3],side[2],side[6],side[7]],
                             [side[1],side[2],side[6],side[5]],
                             [side[4],side[7],side[3],side[0]] ]
            c_temp = [C[i] for j in range(6)]
            surfaces+=sides
            colors+=c_temp

        #CREATE A Poly3DCollection FROM OUR SURFACES
        P = Poly3DCollection(surfaces, facecolors=colors,   alpha=1.0)

        #PLOT THE POLY3DCOLLECTION ON OUR AXIS
        collection = ax.add_collection3d(P)

        #FIGURE FORMATTING SETTINGS
        ax.set_xlim([0,1000])                                                                   #You can use x and y arrays for setting this, but usually it is easier to manually set
        ax.set_ylim([0,1000])
        ax.set_zlim([0,1000])

        #CREATE COLORBAR FROM OUR COLORBAR OBJECT CW
        fig.colorbar(cw,label="Unique Grains")

        #TIGHT LAYOUT TO AUTOMATICALLY ADJUST BORDERS AND PADDING FOR BEST LOOKING IMAGE
        fig.tight_layout()

        fig.savefig(pic_directory+'/'+dirName+'_sliced_'+str(i)+'.png',dpi=500,transparent=True )#../

        #USE PLT.SHOW FOR AN INTERACTIVE 3D DISPLAY OF IMAGE. THIS CAN BE A BIT SLOW TO MANIPULATE DEPENDING ON EXODUS FILE SIZE
        plt.show()



    # #STORE FIGURE IN 2D FOLDER, AND THE NAME ENDS WITH THE INDEX OF THE RENDERED FRAME. DPI = 500 AND TRANSPARENT BACKGROUND
    # # fig.savefig('pics/2d_render_'+str(i)+'.png',dpi=500,transparent=True )
    # fig.savefig(pic_directory+'/'+dirName+'_sliced_'+str(i)+'.png',dpi=500,transparent=True )#../
    # #CLOSE FIGURE AFTER YOU ARE DONE WITH IT. OTHERWISE ALL GENERATED FIGURES WILL BE HELD IN MEMORY TILL SCRIPT FINISHES RUNNING
    # plt.close(fig)


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
