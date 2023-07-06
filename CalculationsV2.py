import os
import glob
import numpy as np
import json
import argparse
import logging
import multiprocessing as mp
import time
from MultiExodusReader import MultiExodusReader
import subprocess

# Needs to have access to '~/projects/TIGER/parallel_time_file_make.py'



pt = logging.warning
verb = logging.info



class CalculationsV2:
    def __init__(self): #,files
        self.parse_cl_flags()
        self.set_logging(self.cl_args.verbose)
        verb('Command Line Flags: '+str(self.cl_args))
        self.times_files()
        self.get_meta(self.cl_args.new_meta)



    # Depreciated- not using it right now
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
            self.adaptive_mesh = False
        else:
            name_unq = np.insert(name_unq, 0, e_unq)
            self.adaptive_mesh = True
        self.file_names = name_unq
        return self.file_names


    def parse_cl_flags(self):
        # Should overwrite json metadata with these values
        parser = argparse.ArgumentParser()
        parser.add_argument('--verbose','-v',action='store_true',
                            help='Verbose output, default off')
        parser.add_argument('--new-meta','-j', action='store_true',
                            help='Write new metadata (tiger_meta.json), default off')
        parser.add_argument('--cpu','-n',type=int,
                            help='''Number of CPUs to use, default = 1.
                            If >1 will run parallel where applicable.''')
        parser.add_argument('--new-times','-t',action='store_true',
                            help='Write new times_files data, default off')
        parser.add_argument('--plane','-p',type=str,
                            help='Perform calcs and plotting on a single plane, ex: "x100", default None')

        self.cl_args = parser.parse_args()
        return self.cl_args


    def set_logging(self,verbose_flag):
        if verbose_flag == True:
            logging.basicConfig(level=logging.INFO,format='%(message)s')
        elif verbose_flag == False:
            logging.basicConfig(level=logging.WARNING,format='%(message)s')
        verb('Verbose Logging Enabled')
        return

    # Clean this up with better and and or so i have less of the same def multiple places
    def define_params(self,json_TF):
        # Default parameter/variable/input values:
        self.cpu = 1
        self.parallel = False
        self.plane_axis = None
        self.plane_coord = None
        # Writing without reading an existing tiger_meta.json
        if not json_TF:
            verb('Defining parameters without tiger_meta.json')
            # CPUs
            if not self.cl_args.cpu == None:
                self.cpu = self.cl_args.cpu
                self.parallel = True
            # planar params
            if not self.cl_args.plane == None:
                self.plane_axis = self.cl_args.plane[:1].lower()
                if not self.plane_axis in ['x','y','z']:
                    raise ValueError('\x1b[31;1m'+'ERROR:'+'\x1b[0m'+' --plane needs to begin with'
                                     +' x, y, or z followed by the coordinate on that axis to slice at.')
                self.plane_coord = int(self.cl_args.plane[1:])
            return
        # Reading the tiger_meta.json then overwriting those values with any cl_args specified
        else:
            verb('Defining parameters using tiger_meta.json but overwriting with any command line flags')
            # CPUs
            if self.cl_args.cpu == None:
                self.cpu = self.meta['params']['cpu']
                self.parallel = self.meta['params']['parallel']
            else:
                verb('  Overwriting cpu with flagged '+str(self.cl_args.cpu))
                self.cpu = self.cl_args.cpu
                if self.cpu > 1:
                    self.parallel = True
            # Planar params
            if self.cl_args.plane == None:
                self.plane_axis = self.meta['params']['plane_axis'].lower()
                if not self.plane_axis in ['x','y','z']:
                    raise ValueError('\x1b[31;1m'+'ERROR:'+'\x1b[0m'+' --plane needs to begin with'
                                     +' x, y, or z followed by the coordinate on that axis to slice at.')
                self.plane_coord = self.meta['params']['plane_coord']
            else:
                verb('  Overwriting plane with flagged '+str(self.cl_args.plane))
                self.plane_axis = self.cl_args.plane[:1].lower()
                if not self.plane_axis in ['x','y','z']:
                    raise ValueError('\x1b[31;1m'+'ERROR:'+'\x1b[0m'+' --plane needs to begin with'
                                     +' x, y, or z followed by the coordinate on that axis to slice at.')
                self.plane_coord = int(self.cl_args.plane[1:])
            return



    def get_meta(self,rewrite_flag):
        if os.path.exists("tiger_meta.json") and not rewrite_flag:
            verb("Using existing tiger_meta.json")
            with open('tiger_meta.json') as json_file:
                self.meta = json.load(json_file)
                # return self.meta
            self.define_params(True)
        elif rewrite_flag or not os.path.exists("tiger_meta.json"):
            if os.path.exists("tiger_meta.json"):
                verb("Overwriting old tiger_meta.json with new")
            else:
                verb("Writing new tiger_meta.json")
            self.define_params(False)
            self.write_meta()
            return
        else:
            pt('\x1b[31;1m'+'ERROR:'+'\x1b[0m'+' tiger_meta.json failure?')
        return


    # Create tiger_meta.json
    def write_meta(self):
        dict = {}
        dict['cl_args'] = vars(self.cl_args)
        dict['params'] = {
            'adaptive_mesh':self.adaptive_mesh,
            'len_files':self.len_files,
            'cpu':self.cpu,
            'parallel':self.parallel,
            'plane_axis':self.plane_axis,
            'plane_coord':self.plane_coord
        }
        self.meta = dict
        with open('tiger_meta.json', 'w') as fp:
            json.dump(dict, fp, indent=4)
        return


    # Write times_files.npy if doesnt exist and read it into self
    def times_files(self):
        if self.cl_args.new_times == False and os.path.exists("times_files.npy"):
            verb('Using existing times_files.npy')
        else:
            verb('Generating new times_files.npy')
            para_time_file_opt = os.path.expanduser('~/projects/TIGER/parallel_time_file_make.py')
            if os.path.exists(para_time_file_opt):
                verb('Found parallel_time_file_make.py')
            else:
                raise ValueError('\x1b[31;1m'+'ERROR:'+'\x1b[0m'+
                                 ' path not found for parallel_time_file_make.py')
            if self.cl_args.cpu == None:
                verb('Running parallel_time_file_make.py with 1 CPU')
                command = ['python',para_time_file_opt,'1']
            else:
                verb('Running parallel_time_file_make.py with '+str(self.cl_args.cpu)+' CPUs')
                command = ['python',para_time_file_opt,str(self.cl_args.cpu)]
            verb('    Command being run is: ' + str(command))
            subprocess.run(command)
        # Now read times files and add the parameters
        times_files = np.load('times_files.npy')
        self.times = times_files[:,0].astype(float)
        self.files = times_files[:,1].astype(str)
        self.t_steps = times_files[:,2].astype(int)
        self.file_names = np.unique(self.files)
        self.len_files = len(self.files)
        if "*.e-s*" in self.file_names:
            verb('Adaptive Mesh files detected')
            self.adaptive_mesh = True
        else:
            verb('No adaptive mesh files detected')
            self.adaptive_mesh = False
        return


    # # Serial Time File Build
    # def para_time_build(count,file_name,len_files):
    #     verb("File "+str(count+1)+ "/"+str(len_files)+": "+str(file_name))#, end = "\r"
    #     t0 = time.perf_counter()
    #     times_files = []
    #     MF = 0
    #     MF = MultiExodusReader(file_name).global_times
    #     for i,time_val in enumerate(MF):
    #         times_files.append([time_val,file_name,i])
    #     MF = 0
    #     verb("   Finished file "+str(count+1)+": "+str(round(time.perf_counter()-t0,2))+"s")
    #     return times_files
