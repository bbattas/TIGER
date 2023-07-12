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

e_name = [x.rsplit('.',1)[0]+"*" for x in glob.glob("*.e.*")]#"*_out.e.*"#glob.glob("*_out.e.*") #first step
s_names = [x.rsplit('.',2)[0]+"*" for x in glob.glob("*.e-s*")] #after first step#x[:-8]

e_unq = np.unique(e_name)
name_unq = np.unique(s_names)

if e_unq.size == 0:
    raise ValueError('No files found ending with "*.e.*"')
elif name_unq.size == 0:
    name_unq = e_unq
else:
    name_unq = np.insert(name_unq, 0, e_unq)

print("Files being used:")
print(name_unq[:4]," ...")
times_files = np.empty((0,3))

print("Building Time Data:")
file_len = len(name_unq)
for n,file in enumerate(name_unq):
    print("                                                       ", end = "\r")
    print("File ",n+1,"/",file_len,": ",file, end = "\r")
    # This line is for debugging to exit the loop at n==5 files and continue on
    # if n == 5:
    #     break
    MF = 0
    MF = MultiExodusReader(file).global_times
    for i,time in enumerate(MF): #.global_times
        times_files = np.append(times_files,[[time,file,i]],axis=0)
        # print(times)
print('\n' + "Done Building Time Data")

times_files = times_files[times_files[:, 0].astype(float).argsort()]
# print(times_files[:,0])# Time
# print(times_files[:,1])# Which file the time is from
# print(times_files[:,2])# Which dataset in the file the time is from

# times = times_files[:,0].astype(float)
# t_step = times_files[:,2].astype(int)

np.save('times_files.npy', times_files)
quit()
