from MultiExodusReader import MultiExodusReader
import multiprocessing as mp

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PolyCollection
import matplotlib
import numpy as np
#from time import time
import time
import os
import glob
import pandas as pd
import math
import sys


# n_cpu = 4
n_cpu = int(sys.argv[1])
# n_cpu = mp.cpu_count() - 2
# n_cpu = int(mp.cpu_count() / 2 - 1)



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
# times_files = np.empty((0,3))

print("Building Time Data:")
file_len = len(name_unq)


# FOR ADAPTIVE MESH- 1 cpu for all timesteps per .e-s* file
def para_time_build(unq_file_name,count):
    # print("                                                       ", end = "\r")
    print("File",count, "/",file_len,": ",unq_file_name)#, end = "\r"
    t0 = time.perf_counter()
    times_files = []#np.empty((0,3))
    MF = 0
    MF = MultiExodusReader(unq_file_name).global_times
    for i,time_val in enumerate(MF): #.global_times
        times_files.append([time_val,unq_file_name,i])
        # times_files = np.append(times_files,[[time,unq_file_name,i]],axis=0)
    print("   Finished file",count,": ",round(time.perf_counter()-t0,2),"s")
    # count = count+1
    return times_files

# MAKE A VERSION THAT RUNS IF len(unq_file_name) == 1 or 0 or whatever for not adaptive mesh


#IF IN MAIN PROCESS
if __name__ == "__main__":
    #CREATE A PROCESS POOL
    cpu_pool = mp.Pool(n_cpu)
    print(cpu_pool)
    results = []
    for i,file in enumerate(name_unq):
        results.append(cpu_pool.apply_async(para_time_build,args = (file, i )))#, callback = log_result)
    # ex_files = [cpu_pool.map(para_time_build,args=(file,)) for file in name_unq  ]
    # print(ex_files)

    cpu_pool.close()
    cpu_pool.join()
    print(cpu_pool)
    print("Aggregating data...")#Restructuring
    results = [r.get() for r in results]
    time_file_list = []
    for row1 in results:
        for row2 in row1:
            time_file_list.append(row2)

    times_files = np.asarray(time_file_list)

    print('\n' + "Done Building Time Data")

    times_files = times_files[times_files[:, 0].astype(float).argsort()]
    # print(times_files[:,0])# Time
    # print(times_files[:,1])# Which file the time is from
    # print(times_files[:,2])# Which dataset in the file the time is from

    # times = times_files[:,0].astype(float)
    # t_step = times_files[:,2].astype(int)

    print(times_files)

    np.save('times_files.npy', times_files)
    quit()

quit()
