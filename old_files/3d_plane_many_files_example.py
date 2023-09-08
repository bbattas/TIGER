from MultiExodusReader import MultiExodusReader

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PolyCollection
import matplotlib
import numpy as np
from time import time
import os
import glob


# This is the 3d_plane_data but for when there are too many nemesis/-s files to open
var_to_plot = 'unique_grains' # OPs cant be plotted, needs to be elements not nodes
z_plane = 19688/2
sequence = True
n_frames = 10
particleCentroids = False


#EXODUS FILE FOR RENDERING
#ANY CHARACTER(S) CAN BE PLACED IN PLACE OF THE *, EG. 2D/grain_growth_2D_graintracker_out.e.1921.0000 or 2D/grain_growth_2D_graintracker_out.e-s001
# filenames = '2D/grain_growth_2D_graintracker_out.e*'

## Old version- to read the input file and determine the .e files from that,
##   not really relevant to the too many open files version this is
# for file in glob.glob("*.i"):
#     inputName = os.path.splitext(file)[0]
# print("Input File is: " + inputName + ".i")
# filenames = inputName + "_out.e*"#*
# print("   Output Files: " + filenames)
dirName = os.path.split(os.getcwd())[-1]

# if the pics directory doesnt exist, make it
if not os.path.isdir('pics'):
    os.makedirs('pics')


e_name = "*_out.e.*"#glob.glob("*_out.e.*") #first step
s_names = [x.rsplit('.',2)[0]+"*" for x in glob.glob("*_out.e-s*")] #after first step#x[:-8]

name_unq = np.unique(s_names)
name_unq = np.insert(name_unq, 0, e_name)
print("Files being used:")
print(name_unq[:4]," ...")
times_files = np.empty((0,3))

print("Building Time Data:")
file_len = len(name_unq)
for n,file in enumerate(name_unq):
    print("File ",n,"/",file_len,": ",file, end = "\r")
    MF = 0
    MF = MultiExodusReader(file)
    for i,time in enumerate(MF.global_times):
        times_files = np.append(times_files,[[time,file,i]],axis=0)

print('\n' + "Done Building Time Data")

# OPTIONAL- sort the array by times in case they weren't in order as they were read
# will be particularly useful if using continued/checkpointed runs later
times_files = times_files[times_files[:, 0].argsort()]

times = times_files[:,0].astype(float)
t_step = times_files[:,2].astype(int)


#GETTING CLOSEST TIME STEP TO DESIRED SIMULATION TIME FOR RENDER --> TYPICALLY 200 FRAMES WITH 20 FPS GIVES A GOOD 10 S LONG VIDEO
if sequence == True:
    t_max = times[-1]
    t_frames =  np.linspace(0.0,t_max,n_frames)
    idx_frames = [ np.where(times-t_frames[i] == min(times-t_frames[i],key=abs) )[0][0] for i in range(n_frames) ]
elif sequence == False:
    t_frames = times
    idx_frames = range(len(times))
else:
    raise ValueError('sequence has to be True or False, not: ' + str(sequence))



#LOOP OVER EACH TIME STEP IN idx_frames
for (i,time_step) in enumerate(idx_frames):
    print( "Rendering frame no. ",i+1)

    #READ EXODUS FILE SERIES WITH MultiExodusReader
    MF = MultiExodusReader(times_files[time_step,1])


    #GENERATE Fmesh_area = IGURE WINDOW
    fig, ax = plt.subplots()

    #GET X,Y,Z AND C (UNIQUE GRAINS VALUES) AT CURRENT TIME STEP
    x,y,z,c = MF.get_data_at_time(var_to_plot,MF.global_times[t_step[time_step]] )#MF.global_times[time_step]               #Read coordinates and variable value --> Will be parallelized in future

    # Trim X,Y,Z, and C values to just the relevant z_plane
    zmax = np.amax(z, axis=1)
    zmin = np.amin(z, axis=1)
    # Could also just use first and last points might be faster
    ind_z = np.where((z_plane <= zmax) & (z_plane >= zmin))
    x = x[ind_z][:]
    y = y[ind_z][:]
    z = z[ind_z][:]
    c = c[ind_z][:]

    zmax = np.amax(z, axis=1)
    zmin = np.amin(z, axis=1)

    # What fraction between the top and bottom of the given 8 coord box the plane of interest is
    # int_frac_z = (z_plane - zmin) / (zmax - zmin)

    # on_plane = np.where((z_plane = zmax) | (z_plane = zmin))
    on_plane = ((z_plane == zmax) | (z_plane == zmin))

    # Interpolate (not really) for the given z plane the coord/value you want
    # MANUAL for cubic mesh, eventually change to interp() if needed, but probably faster this way
    x = x[:, :4]
    y = y[:, :4]
    z = z_plane * np.ones_like(x)



    # MAKE IT PART OF THE PARTICLEAREAS or check for dupes
    if particleCentroids == True:
        c_int = np.rint(c)
        mesh_ctr = np.asarray([ x[:, 0] + (x[:, 2] - x[:, 0])/2, y[:, 0] + (y[:, 2] - y[:, 0])/2 ]).T

        mesh_area = np.asarray((x[:, 2] - x[:, 0])*(y[:, 2] - y[:, 0]))
        mesh_area = np.where(on_plane, mesh_area/2, mesh_area) #out_mesh_area

        zeros = np.zeros_like(c_int)
        areas = np.zeros(round(np.amax(c_int))+2)
        centroids = np.asarray([ areas, areas ]).T

        for n in range(len(areas)):
            areas[n] = np.sum(np.where(c_int==(n-1),mesh_area,zeros))
            if areas[n] > 0.0:
                centroids[n] = np.asarray([ np.sum(np.where(c_int==(n-1),mesh_ctr[:,0] * mesh_area,zeros)) / np.sum(np.where(c_int==(n-1),mesh_area,zeros)),
                                            np.sum(np.where(c_int==(n-1),mesh_ctr[:,1] * mesh_area,zeros)) / np.sum(np.where(c_int==(n-1),mesh_area,zeros))]).T






    #GENERATE COORDINATES ARRAY THAT STORES X AND Y POINTS TOGETHER
    coords = np.asarray([ np.asarray([x_val,y_val]).T for (x_val,y_val) in zip(x,y) ])

    #USE POLYCOLLECTION TO DRAW ALL POLYGONS DEFINED BY THE COORDINATES
    p = PolyCollection(coords, cmap=matplotlib.cm.coolwarm, alpha=1)#,edgecolor='k'      #Edge color can be set if you want to show mesh

    #COLOR THE POLYGONS WITH OUR VARIABLE
    ## Map plot variable range to color range
    c_min = np.amin(c)
    c_max = np.amax(c)
    colors = c#(c - c_min)/(c_max-c_min)
    #
    p.set_array(np.array(colors) )

    #ADD THE POLYGON COLLECTION TO AXIS --> THIS IS WHAT ACTUALLY PLOTS THE POLYGONS ON OUR WINDOW
    ax.add_collection(p)

    #FIGURE FORMATTING

    #SET X AND Y LIMITS FOR FIGURE --> CAN USE x,y ARRAYS BUT MANUALLY SETTING IS EASIER
    # ax.set_xlim([0,300])
    # ax.set_ylim([0,300])
    ax.set_xlim([np.amin(x),np.amax(x)])
    ax.set_ylim([np.amin(y),np.amax(y)])
    #SET ASPECT RATIO TO EQUAL TO ENSURE IMAGE HAS SAME ASPECT RATIO AS ACTUAL MESH
    ax.set_aspect('equal')

    #ADD A COLORBAR, VALUE SET USING OUR COLORED POLYGON COLLECTION
    fig.colorbar(p,label="Unique grains")

    #STORE FIGURE IN 2D FOLDER, AND THE NAME ENDS WITH THE INDEX OF THE RENDERED FRAME. DPI = 500 AND TRANSPARENT BACKGROUND
    # fig.savefig('pics/2d_render_'+str(i)+'.png',dpi=500,transparent=True )
    fig.savefig('pics/'+dirName+'_sliced_'+str(i)+'.png',dpi=500,transparent=True )#../
    #CLOSE FIGURE AFTER YOU ARE DONE WITH IT. OTHERWISE ALL GENERATED FIGURES WILL BE HELD IN MEMORY TILL SCRIPT FINISHES RUNNING
    plt.close(fig)
