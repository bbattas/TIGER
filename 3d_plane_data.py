from MultiExodusReader import MultiExodusReader

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PolyCollection
import matplotlib
import numpy as np
from time import time
import os
import glob

var_to_plot = 'unique_grains' # OPs cant be plotted, needs to be elements not nodes
z_plane = 100
sequence = False
n_frames = 200
particleAreas = False #GO BACK AND MAKE RELEVANT OR REMOVE
particleCentroids = True
overwriteCentroids = True
max_xy = 300
test = True
#ADD OUTSIDE BOUNDS ERROR!!!!!!!!!!!!!!

#EXODUS FILE FOR RENDERING
#ANY CHARACTER(S) CAN BE PLACED IN PLACE OF THE *, EG. 2D/grain_growth_2D_graintracker_out.e.1921.0000 or 2D/grain_growth_2D_graintracker_out.e-s001
# filenames = '2D/grain_growth_2D_graintracker_out.e*'
for file in glob.glob("*.i"):
    inputName = os.path.splitext(file)[0]
print("Input File is: " + inputName + ".i")
filenames = inputName + "_out.e*"
print("   Output Files: " + filenames)
dirName = os.path.split(os.getcwd())[-1]

# if the ../pics directory doesnt exist, make it
if not os.path.isdir('../pics'):
    os.makedirs('../pics')


#READ EXODUS FILE SERIES WITH MultiExodusReader
MF = MultiExodusReader(filenames)

#GET A LIST OF SIMULATION TIME POINTS
times = MF.global_times

#GETTING CLOSEST TIME STEP TO DESIRED SIMULATION TIME FOR RENDER --> TYPICALLY 200 FRAMES WITH 20 FPS GIVES A GOOD 10 S LONG VIDEO
# n_frames = 200
if sequence == True:
    t_max = times[-1]
    t_frames =  np.linspace(0,t_max,n_frames)
    idx_frames = [ np.where(times-t_frames[i] == min(times-t_frames[i],key=abs) )[0][0] for i in range(n_frames) ]
elif sequence == False:
    idx_frames = range(len(times))
else:
    raise ValueError('sequence has to be True or False, not: ' + str(sequence))

#LOOP OVER EACH TIME STEP IN idx_frames
for (i,time_step) in enumerate(idx_frames):
    print( "Rendering frame no. ",i+1)
    #GENERATE FIGURE WINDOW
    fig, ax = plt.subplots()

    #GET X,Y,Z AND C (UNIQUE GRAINS VALUES) AT CURRENT TIME STEP
    x,y,z,c = MF.get_data_at_time(var_to_plot,MF.global_times[time_step])               #Read coordinates and variable value --> Will be parallelized in future

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
    int_frac_z = (z_plane - zmin) / (zmax - zmin)

    # Interpolate for the given z plane the coord/value you want
    # NOT NECESSARY since element values are constant in the whole cube :(
    def interp(row,coord):
        return np.asarray([ coord[row][0] + int_frac_z[row]*(coord[row][4]-coord[row][0]),
                            coord[row][1] + int_frac_z[row]*(coord[row][5]-coord[row][1]),
                            coord[row][2] + int_frac_z[row]*(coord[row][6]-coord[row][2]),
                            coord[row][3] + int_frac_z[row]*(coord[row][7]-coord[row][3]) ])
    # INTERPOLATED values
    # MANUAL for cubic mesh, eventually change to interp() if needed, but probably faster this way
    x = x[:, :4]
    # print(x)
    # print(x[:, 0])
    y = y[:, :4]
    # print(int_y)
    z = z_plane * np.ones_like(x)
    # print(int_z)
    # int_c = np.asarray([ interp(row,newx) for row in range(len(int_z)) ])
    # c = c
    # print(int_c)
    if particleAreas == True:
        c_int = np.rint(c)
        mesh_area = np.asarray((x[:, 2] - x[:, 0])*(y[:, 2] - y[:, 0]))
        zeros = np.zeros_like(c_int)
        #
        areas = np.zeros(round(np.amax(c_int))+2)
        #
        # areas = np.asarray( np.sum(np.where(c_int==(n-1),mesh_area,zeros)) for n in range(round(np.amax(c_int))+2) )
        # print("Areas: ",areas)
        for n in range(len(areas)):
            # print(n)
            areas[n] = np.sum(np.where(c_int==(n-1),mesh_area,zeros))
        print("Particle Areas: ",areas)

    # MAKE IT PART OF THE PARTICLEAREAS or check for dupes
    if particleCentroids == True:
        c_int = np.rint(c)
        mesh_ctr = np.asarray([ x[:, 0] + (x[:, 2] - x[:, 0])/2, y[:, 0] + (y[:, 2] - y[:, 0])/2 ]).T
        # print(mesh_ctr)
        # print(mesh_ctr[0])
        # print(mesh_ctr[0,1])
        mesh_area = np.asarray((x[:, 2] - x[:, 0])*(y[:, 2] - y[:, 0]))
        zeros = np.zeros_like(c_int)
        areas = np.zeros(round(np.amax(c_int))+2)
        centroids = np.asarray([ areas, areas ]).T
        if test == True:
            unq, count = np.unique(mesh_ctr, axis=0, return_counts=True)
            u, indices = np.unique(mesh_ctr, axis=0, return_index=True)
            print(unq)
            print(count)
            print(" ")
            print(u)
            print(indices)

        centx = np.zeros_like(areas)
        for n in range(len(areas)):
            areas[n] = np.sum(np.where(c_int==(n-1),mesh_area,zeros))
            if areas[n] > 0.0:
                # centroids[n] = np.asarray([ np.sum(np.where(c_int==(n-1),mesh_ctr[:,0],zeros) * np.where(c_int==(n-1),mesh_area,zeros)) / np.sum(np.where(c_int==(n-1),mesh_area,zeros)),
                #                             np.sum(np.where(c_int==(n-1),mesh_ctr[:,1],zeros) * np.where(c_int==(n-1),mesh_area,zeros)) / np.sum(np.where(c_int==(n-1),mesh_area,zeros))]).T
                centroids[n] = np.asarray([ np.sum(np.where(c_int==(n-1),mesh_ctr[:,0] * mesh_area,zeros)) / np.sum(np.where(c_int==(n-1),mesh_area,zeros)),
                                            np.sum(np.where(c_int==(n-1),mesh_ctr[:,1] * mesh_area,zeros)) / np.sum(np.where(c_int==(n-1),mesh_area,zeros))]).T

        # For quarter cell- centroid is the boundary not where it would be calculated, plus midplane weird double op circles
        if overwriteCentroids == True:
            condx = centroids[1:,0]>centroids[1:,1] # where x>y
            condy = centroids[1:,0]<centroids[1:,1] # where y>x
            zero = np.zeros_like(centroids[1:,1])
            mid_grains = np.asarray([ [ max_xy, np.sum(np.where(condx,centroids[1:,1] * areas[1:],zero)) / np.sum(np.where(condx,areas[1:],zero))],
                                      [ np.sum(np.where(condy,centroids[1:,0] * areas[1:],zero)) / np.sum(np.where(condy,areas[1:],zero)), max_xy] ])
            print(mid_grains)

            # Quarter Cell Dividing Line
            slope = (mid_grains[1,1] - mid_grains[0,1]) / (mid_grains[1,0] - mid_grains[0,0])
            # print(slope)
            # print(mesh_ctr)
            # print(mesh_ctr[0])
            # print(mesh_ctr[0,1])
            # print(mesh_ctr[:,1] >= slope * (mesh_ctr[:,0] - mid_grains[0,0]) + mid_grains[0,1])
            pore_area = np.sum(np.where((c_int == -1) & (mesh_ctr[:,1] >= slope * (mesh_ctr[:,0] - mid_grains[0,0]) + mid_grains[0,1]),mesh_area,zeros))

            print("areas: ")
            print(areas)
            print(np.sum(areas))
            print(np.sum(mesh_area))
            print(pore_area)




    #GENERATE COORDINATES ARRAY THAT STORES X AND Y POINTS TOGETHER
    coords = np.asarray([ np.asarray([x_val,y_val]).T for (x_val,y_val) in zip(x,y) ])
    # coords = np.asarray([ np.asarray([x_val,y_val]).T for (x_val,y_val) in zip(int_x,int_y) ])
    # print(coords)
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
