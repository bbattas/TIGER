from MultiExodusReader import MultiExodusReader

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PolyCollection
import matplotlib
import numpy as np
from time import time
import os
import glob
import pandas as pd
import math

# This is the 3d_plane_data but for when there are too many nemesis/-s files to open
var_to_plot = 'unique_grains' # OPs cant be plotted, needs to be elements not nodes
z_plane = 19688/2
sequence = False
n_frames = 10
particleAreas = False #GO BACK AND MAKE RELEVANT OR REMOVE
particleCentroids = True
overwriteCentroids = True
max_xy = 30000
test = False
full_area = False

#ADD OUTSIDE BOUNDS ERROR!!!!!!!!!!!!!!

#EXODUS FILE FOR RENDERING
#ANY CHARACTER(S) CAN BE PLACED IN PLACE OF THE *, EG. 2D/grain_growth_2D_graintracker_out.e.1921.0000 or 2D/grain_growth_2D_graintracker_out.e-s001
# filenames = '2D/grain_growth_2D_graintracker_out.e*'

# for file in glob.glob("*.i"):
#     inputName = os.path.splitext(file)[0]
# print("Input File is: " + inputName + ".i")
# filenames = inputName + "_out.e*"#*
# print("   Output Files: " + filenames)
# dirName = os.path.split(os.getcwd())[-1]


e_name = [x.rsplit('.',1)[0]+"*" for x in glob.glob("*.e.*")]#"*_out.e.*"#glob.glob("*_out.e.*") #first step
s_names = [x.rsplit('.',2)[0]+"*" for x in glob.glob("*.e-s*")] #after first step#x[:-8]
#s_names = [x.rsplit('.',2)[0]+"*" for x in glob.glob("*.e*")]
# print(s_names)
# temp_names = [x for x in glob.glob("*_out.e-s*")]
# print(temp_names[0])
# print(temp_names[0].rsplit('.',2))
e_unq = np.unique(e_name)
# print(e_unq)
name_unq = np.unique(s_names)
#print(name_unq)
# print(name_unq[:5])
name_unq = np.insert(name_unq, 0, e_unq)
print("Files being used:")
print(name_unq[:4]," ...")
# print(name_unq," ...")
times_files = np.empty((0,3))

print("Building Time Data:")
file_len = len(name_unq)
for n,file in enumerate(name_unq):
    print("                                                       ", end = "\r")
    print("File ",n+1,"/",file_len,": ",file, end = "\r")
    # if n == 5:
    #     break
    MF = 0
    MF = MultiExodusReader(file)
    for i,time in enumerate(MF.global_times):
        times_files = np.append(times_files,[[time,file,i]],axis=0)
        # print(times)
print('\n' + "Done Building Time Data")

times_files = times_files[times_files[:, 0].astype(float).argsort()]
# print(times_files[:,0])# Time
# print(times_files[:,1])# Which file the time is from
times = times_files[:,0].astype(float)
t_step = times_files[:,2].astype(int)
# print(times_files)
# print(times_files[:,0])
# MF = MultiExodusReader(name_unq[0])
# print(MF.global_times[t_step[0]])￼
# print(MF.global_times[t_step[1]])
# print(times_files)
np.save('times_files.npy', times_files)
quit()
