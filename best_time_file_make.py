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
from tqdm import tqdm
import logging
import argparse
import dask
from dask import delayed, compute
from dask.diagnostics import ProgressBar


# You need the ulimit -n to be large enough for however many cpus you ran the job
# multiplied by the number of n_cpu you use for the pool here


pt = logging.warning
verb = logging.info
db = logging.debug

parser = argparse.ArgumentParser()
parser.add_argument('--verbose','-v', action='store_true', help='Verbose output, default off')
parser.add_argument('--cpus','-n',type=int, default=1,
                            help='How many cpus, default='+str(1))
parser.add_argument('--skip','-s',action='store_true',help='Skip last timestep/file, currently not working?')
cl_args = parser.parse_args()

n_cpu = cl_args.cpus

# Toggle verbose
if cl_args.verbose == True:
    logging.basicConfig(level=logging.INFO,format='%(message)s')
elif cl_args.verbose == False:
    logging.basicConfig(level=logging.WARNING,format='%(message)s')
verb('Verbose Logging Enabled')
verb(cl_args)
verb(' ')


# might need this to work on mac???
# mp.freeze_support()
# mp.set_start_method('spawn')


# Find the names
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

verb("Files being used:")
verb(str(name_unq[:4]) + " ...")
# times_files = np.empty((0,3))

verb("Building Time Data:")
file_len = len(name_unq)


# FOR ADAPTIVE MESH- 1 cpu for all timesteps per .e-s* file
def para_time_build(unq_file_name,count):
    # unq_file_name,count = args
    verb("File" + str(count+1) + "/" + str(file_len) + ": " + str(unq_file_name))#, end = "\r"
    t0 = time.perf_counter()
    times_files = []#np.empty((0,3))
    MF = 0
    MF = MultiExodusReader(unq_file_name).global_times
    for i,time_val in enumerate(MF): #.global_times
        times_files.append([time_val,unq_file_name,i])
        # times_files = np.append(times_files,[[time,unq_file_name,i]],axis=0)
    MF = 0
    verb("   Finished file" + str(count) + ": " + str(round(time.perf_counter()-t0,2)) + "s")
    # count = count+1
    return times_files


#IF IN MAIN PROCESS
if __name__ == "__main__":
    verb('In best_time_file_make.py __main__')
    # this was needed to make mac work now?
    mp.set_start_method('fork')
    # mp.set_start_method('spawn')

    if cl_args.skip:
        verb("NOTE: Skipping last file as indicated with 'skip' flag")
        verb(" ")
        name_unq = name_unq[:-1]

    # Start the actual time calculation
    loop_ti = time.perf_counter()
    results = []

    pt('Building time data')
    # Serial calculation
    if n_cpu == 1:
        verb('Running in serial')
        results = [para_time_build(file, i) for i, file in enumerate(tqdm(name_unq))] #name_unq, desc="Building time data"
        verb("Calculation Time: "+str(round(time.perf_counter()-loop_ti,2))+"s")
        # verb("Aggregating data...")
    # Parallel calculation with Dask
    else:
        verb(f'Running in parallel with {n_cpu} CPUs')
        tasks = [delayed(para_time_build)(file, i) for i, file in enumerate(name_unq)]
        with ProgressBar():
            results = compute(*tasks, num_workers=n_cpu)

        verb("Total Pool Time: "+str(round(time.perf_counter()-loop_ti,2))+"s")
    # verb("Aggregating data...") # Restructuring


    time_file_list = []
    for row1 in results:
        for row2 in row1:
            time_file_list.append(row2)

    times_files = np.asarray(time_file_list)

    verb('\n' + "Done Building Time Data")

    times_files = times_files[times_files[:, 0].astype(float).argsort()]
    # print(times_files[:,0])# Time
    # print(times_files[:,1])# Which file the time is from
    # print(times_files[:,2])# Which dataset in the file the time is from

    # times = times_files[:,0].astype(float)
    # t_step = times_files[:,2].astype(int)

    # print(times_files)

    np.save('times_files.npy', times_files)
    quit()

quit()
