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
        print(__name__)
        self.parse_cl_flags()
        self.set_logging(self.cl_args.verbose)
        verb('Command Line Flags: '+str(self.cl_args))
        # if self.cl_args.cpu > 1:
        #     verb('Creating pool with ' + str(self.cl_args.cpu) + ' cpus')
        #     cpu_pool = mp.Pool(self.cl_args.cpu)
        #     verb(cpu_pool)
        # self.get_meta(self.cl_args.new_meta,cpu_pool)

        # if self.cl_args.cpu > 1:
        #     verb('closing pool')
        #     cpu_pool.close()
        #     cpu_pool.join()
        # self.get_file_names_in_cwd()

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
        parser.add_argument('--times',action='store_false')
        parser.add_argument('--adaptive-mesh',action='store_true')
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
                verb("Overwriting old metadata with new")
            else:
                verb("Writing new metadata")
            self.write_meta()
        else:
            pt("Error: tiger_meta.json failure?")
        return
        # with open('tiger_meta.json') as json_file:
        #     data = json.load(json_file)
        # self.meta = data
        # return self.meta

    def write_meta(self):
        if self.cl_args.times == True and os.path.exists("times_files.npy"):
            verb('Using existing times_files.npy to generate tiger_meta.json')
            times_files = np.load('times_files.npy')
            self.times = times_files[:,0].astype(float)
            self.file_names = times_files[:,1].astype(str)
            self.file_step = times_files[:,2].astype(int)
            if "*.e-s*" in self.file_names:
                verb('Adaptive Mesh = True')
                self.cl_args.adaptive_mesh = True

        elif self.cl_args.times == False:
            verb('Generating new times_files.npy')
            self.file_names = self.get_file_names_in_cwd()
            len_files = len(self.file_names)
            if "*.e-s*" in self.file_names:
                verb('Adaptive Mesh = True')
                self.cl_args.adaptive_mesh = True
            else:
                verb('Adaptive Mesh = False')
            # Parallel Time File Make
            if self.cl_args.cpu > 1:
                verb('Parallelizing with ' + str(self.cl_args.cpu) + ' cpus')
                cpu_pool = mp.Pool(self.cl_args.cpu)
                verb(cpu_pool)
                pool_t0 = time.perf_counter()
                results = []
                for i,file in enumerate(self.file_names):
                    results.append(cpu_pool.apply_async(para_time_build,args = (i, file, len_files )))
                cpu_pool.close()
                cpu_pool.join()
                verb("Total Pool Time: "+str(round(time.perf_counter()-pool_t0))+"s")
                verb("Aggregating data...")#Restructuring
                verb(results[0])
                # verb(results[0].get())
                outs = [r.get() for r in results]
                print(outs)
                time_file_list = []
                for row1 in results:
                    for row2 in row1:
                        time_file_list.append(row2)
                times_files = np.asarray(time_file_list)
                times_files = times_files[times_files[:, 0].astype(float).argsort()]
                verb('Done Building Time Data')
            # Serial Time File Make
            else:
                verb('Serialized')
                pool_t0 = time.perf_counter()
                results = []
                for i,file in enumerate(self.file_names):
                    results.append(para_time_build(i, file, len_files ))
                verb("Total Serial Time: "+str(round(time.perf_counter()-pool_t0))+"s")
                verb("Aggregating data...")#Restructuring
                time_file_list = []
                for row1 in results:
                    for row2 in row1:
                        time_file_list.append(row2)
                times_files = np.asarray(time_file_list)
                times_files = times_files[times_files[:, 0].astype(float).argsort()]
                verb('Done Building Time Data')
        return
