import os
import glob
import numpy as np
import json
import argparse
import logging
import multiprocessing as mp
import time
from MultiExodusReader import MultiExodusReader
# from CalculationEngine_ParallelTimeFileMake import para_time_build


pt = logging.warning
verb = logging.info

def para_time_build(count,file_name,len_files):
    verb("File "+str(count+1)+ "/"+str(len_files)+": "+str(file_name))#, end = "\r"
    t0 = time.perf_counter()
    times_files = []
    MF = 0
    MF = MultiExodusReader(file_name).global_times
    for i,time_val in enumerate(MF):
        times_files.append([time_val,file_name,i])
    MF = 0
    verb("   Finished file "+str(count+1)+": "+str(round(time.perf_counter()-t0,2))+"s")
    return times_files

class CalculationEngine:
    def __init__(self): #,files
        # if os.path.exists("tiger_meta.json"):
        self.parse_cl_flags()
        self.set_logging(self.cl_args.verbose)
        verb('Command Line Flags: '+str(self.cl_args))

        self.get_meta(self.cl_args.new_meta)
        if self.cl_args.parallel_times == 0:
            # Exiting to run parallel times_files.npy making script
            return
        verb("continuing past the times files")


    def get_file_names_in_cwd(self):
        # Trim off the trailing CPU numbers on file names
        #   ex: nemesis.e.300.000 -> nemesis.e.300*
        #   ex: 2D_HYPRE_nemesis.e-s0002.300.000 -> nemesis.e-s0002*
        e_name = [x.rsplit('.',1)[0]+"*" for x in glob.glob("*.e.*")]
        s_names = [x.rsplit('.',2)[0]+"*" for x in glob.glob("*.e-s*")]
        e_unq = np.unique(e_name)
        name_unq = np.unique(s_names)
        if e_unq.size == 0:
            raise ValueError('No files found ending with "*.e.*"')
        elif name_unq.size == 0:
            name_unq = e_unq
        else:
            name_unq = np.insert(name_unq, 0, e_unq)
        self.file_names = name_unq
        return self.file_names

    def parse_cl_flags(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--new-meta', action='store_true')
        parser.add_argument('--cpu','-n', default=1,type=int)
        parser.add_argument('--verbose','-v',action='store_true')
        parser.add_argument('--new-times',action='store_true')
        parser.add_argument('--parallel-times',default=-1,type=int)
        parser.add_argument('--adaptive-mesh',action='store_true')
        parser.add_argument('--split',default=1,type=int)
        parser.add_argument('--sequence',action='store_true')
        parser.add_argument('--frames',default=100,type=int)
        parser.add_argument('--cutoff',default=0,type=int)
        parser.add_argument('--quarter-hull',action='store_true')
        parser.add_argument('--max-x',default=0,type=int)
        parser.add_argument('--max-y',default=0,type=int)
        parser.add_argument('--max-z',default=0,type=int)
        self.cl_args = parser.parse_args()
        return self.cl_args

    def set_logging(self,verbose_flag):
        if verbose_flag == True:
            logging.basicConfig(level=logging.INFO,format='%(message)s')
        elif verbose_flag == False:
            logging.basicConfig(level=logging.WARNING,format='%(message)s')
        verb('Verbose Logging Enabled')
        return

    def get_meta(self,rewrite_flag):
        if os.path.exists("tiger_meta.json") and not rewrite_flag:
            verb("Using existing tiger_meta.json")
        elif rewrite_flag or not os.path.exists("tiger_meta.json"):
            if os.path.exists("tiger_meta.json"):
                verb("Overwriting old tiger_meta.json with new")
            else:
                verb("Writing new tiger_meta.json")
            self.write_meta()
            if self.cl_args.parallel_times == 0:
                return
        else:
            pt("Error: tiger_meta.json failure?")
        return
        # with open('tiger_meta.json') as json_file:
        #     data = json.load(json_file)
        # self.meta = data
        # return self.meta

    def write_meta(self):
        if (self.cl_args.new_times == False or self.cl_args.parallel_times == 1) and os.path.exists("times_files.npy"):
            verb('Using existing times_files.npy to generate tiger_meta.json')
        else:#if self.cl_args.new_times == True:
            verb('Generating new times_files.npy')
            self.file_names = self.get_file_names_in_cwd()
            self.len_files = len(self.file_names)
            # Parallel Time File Make
            if self.cl_args.cpu > 1:
                verb('Exiting CalculationEngine to run parallel times_files.npy generation')
                self.cl_args.parallel_times = 0
                return
            # Serial Time File Make
            else:
                verb('Serialized')
                pool_t0 = time.perf_counter()
                results = []
                for i,file in enumerate(self.file_names):
                    results.append(para_time_build(i, file, self.len_files ))
                verb("Total Serial Time: "+str(round(time.perf_counter()-pool_t0))+"s")
                verb("Aggregating data...")#Restructuring
                time_file_list = []
                for row1 in results:
                    for row2 in row1:
                        time_file_list.append(row2)
                times_files = np.asarray(time_file_list)
                times_files = times_files[times_files[:, 0].astype(float).argsort()]
                np.save('times_files.npy', times_files)
                verb('Done Building Time Data')
        # Actually read the times_files.npy and pull info from it
        verb('Reading times_files.npy')
        times_files = np.load('times_files.npy')
        self.times = times_files[:,0].astype(float)
        self.file_names = times_files[:,1].astype(str)
        self.file_step = times_files[:,2].astype(int)
        if "*.e-s*" in self.file_names:
            verb('Adaptive Mesh = True')
            self.cl_args.adaptive_mesh = True


        return
