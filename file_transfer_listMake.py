#!/usr/bin/env python3
import os
import glob
from fnmatch import fnmatch
import numpy as np
import json
import argparse
import logging
import multiprocessing as mp
import time
import subprocess

pt = logging.warning
verb = logging.info

parser = argparse.ArgumentParser()
parser.add_argument('--verbose','-v',action='store_true')
# parser.add_argument('--inl',action='store_true')
parser.add_argument('--source',type=str)
# parser.add_argument('--dest',type=str)
# parser.add_argument('--new-meta', action='store_true')
cl_args = parser.parse_args()


if cl_args.verbose == True:
    logging.basicConfig(level=logging.INFO,format='%(message)s')
elif cl_args.verbose == False:
    logging.basicConfig(level=logging.WARNING,format='%(message)s')
verb('Verbose Logging Enabled')
verb(cl_args)


cwd = os.getcwd()

# if cl_args.dest == None:
#     pt("Setting destination to cwd: " + cwd)
#     dest = cwd
# else:
#     pt("Setting destination to : " + cl_args.dest)
#     dest = cl_args.dest
#     if dest[-1] == "/":
#         dest = dest[:-1]
#
# dest_tree = [dest + "/" + loc for loc in dest_tree]
#
print(cl_args.source)
if cl_args.source == None:
    pt("Setting source to cwd: " + cwd)
    source = cwd
else:
    pt("Setting destination to : " + cl_args.source)
    source = cl_args.source
    if source[-1] == "/":
        source = source[:-1]

# os.chdir(cwd + "/" + dir)

os.chdir(source)
pt("In source directory: " + source)
# Generate list of all to transfer
verb("Generating list of files to transfer")
tree = []

for dir in glob.glob("*/"):
    if not "checkpoint" in dir.lower() or not "cp" in dir.lower():
        verb("In " + dir)
        tree.append(dir)
        for subdir in glob.glob(dir+"*/"):
            if not "checkpoint" in subdir.lower() or not "cp" in subdir.lower():
                verb("   " + subdir)
                tree.append(subdir)
                for ssubdir in glob.glob(subdir+"*/"):
                    if not "checkpoint" in ssubdir.lower() or not "cp" in ssubdir.lower():
                        verb(" " + ssubdir)
                        tree.append(ssubdir)
                    else:
                            verb(" X " + ssubdir)
            else:
                verb(" X " + subdir)


pt("List of Directories: ")
pt(tree)
files = []
dest_tree = []

for dir in tree:
    if not dir.startswith('.'):
        os.chdir(source + "/" + dir)
        verb("In directory: " + dir)
        e_name = [x.rsplit('.',1)[0]+"*" for x in glob.glob("*.e.*")]#"*_out.e.*"#glob.glob("*_out.e.*") #first step
        s_names = [x.rsplit('.',2)[0]+"*" for x in glob.glob("*.e-s*")] #after first step#x[:-8]
        e_unq = np.unique(e_name).tolist()
        s_unq = np.unique(s_names).tolist()
        # temp_files =
        files.extend([cwd + "/" + dir + file for file in e_unq + s_unq])
        num_files = len(e_unq) + len(s_unq)
        if glob.glob(cwd + "/" + dir + "*.csv"):
            files.extend([cwd + "/" + dir + "*.csv"])
            num_files = num_files + 1
        if glob.glob(cwd + "/" + dir + "*.i"):
            files.extend([cwd + "/" + dir + "*.i"])
            num_files = num_files + 1
        verb(str(num_files) + " files in the directory")
        dest_tree.extend([dir] * num_files)
        # files.append(s_unq)
pt("File list finished")
verb("Files: ")
verb(files)
verb (" ")
# verb("Destination tree: ")
# verb(dest_tree)

os.chdir(source)
dict = {}
dict['files'] = files
dict['dest_tree'] = dest_tree
with open('rsync_list.json', 'w') as fp:
    json.dump(dict, fp)





quit()















quit()
#rsync -anv 3D_D10/*.e-s0002* rsync_test/

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
        parser.add_argument('--dim',default=0,type=int)
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
        self.meta = {}

        return

    def t0_properties(self):
        return
