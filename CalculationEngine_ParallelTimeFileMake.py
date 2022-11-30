import os
import glob
import numpy as np
import json
import argparse
import logging
import multiprocessing as mp
import time
from MultiExodusReader import MultiExodusReader


pt = logging.warning
verb = logging.info

def para_time_build(self,count,file_name,len_files):
    verb("File "+str(count+1)+ "/"+str(len_files)+": "+str(file_name))#, end = "\r"
    t0 = time.perf_counter()
    times_files = []
    MF = 0
    MF = MultiExodusReader(file_name).global_times
    for i,time_val in enumerate(MF):
        times_files.append([time_val,file_name,i])
    MF = 0
    verb("   Finished file "+str(count)+": "+str(round(time.perf_counter()-t0,2))+"s")
    return times_files
