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

# # Needs to have access to '~/projects/TIGER/parallel_time_file_make.py'
# can debug it in another script by running print(calc.__dict__)
# where: calc = CalculationsV2()


pt = logging.warning
verb = logging.info
db = logging.debug


class CalculationsV2:
    def __init__(self): #,files
        self.parse_cl_flags()
        self.set_logging()
        db('Command Line Flags: '+str(self.cl_args))
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
        parser.add_argument('--debug','-d',action='store_true',
                            help='Debug output, more verbose than verbose default off')
        parser.add_argument('--new-meta','-j', action='store_true',
                            help='Write new metadata (tiger_meta.json), default off')
        parser.add_argument('--cpu','-n',type=int,
                            help='''Number of CPUs to use, default = 1.
                            If >1 will run parallel where applicable.''')
        parser.add_argument('--new-times','-t',action='store_true',
                            help='Write new times_files data, default off')
        parser.add_argument('--plane','-p',type=str,
                            help='Perform calcs and plotting on a single plane, ex: "x100", default None')
        parser.add_argument('--seq','-s',type=int,
                            help='Number of frames if using a sequence, default = None.')
        parser.add_argument('--soft-seq',action='store_true',
                            help='If number of frames for --seq is more than the number of timesteps'
                            +' will use the timesteps instead of a sequence. default = False when '
                            +' no --seq specified, otherwise json default and only updates that when'
                            +' a new --seq is specified.')
        parser.add_argument('--var',type=str,
                            help='What to plot, unique_grains or an order parameter, default=None-> phi')
        self.cl_args = parser.parse_args()
        return self.cl_args


    def set_logging(self):
        if self.cl_args.debug:
            logging.basicConfig(level=logging.DEBUG,format='%(message)s')
        elif self.cl_args.verbose and not self.cl_args.debug:
            logging.basicConfig(level=logging.INFO,format='%(message)s')
        else:
            logging.basicConfig(level=logging.WARNING,format='%(message)s')
        db('Debug Logging Enabled: will output more messages in CalculationsV2 internal shit')
        db('    this is verbose plus even more, get ready for the screen to fill up...')
        verb('Verbose Logging Enabled')
        return


    # Clean this up with better and and or so i have less of the same def multiple places
    def define_params(self,json_TF):
        db('Defining Parameters in CalculationsV2')
        # Default parameter/variable/input values:
        self.cpu = 1
        self.parallel = False
        self.plane_axis = None
        self.plane_coord = None
        self.frames = None
        self.soft_seq = False
        self.var_to_plot = 'phi'
        # CPU
        if self.cl_args.cpu == None and json_TF == True:
            self.cpu = self.meta['params']['cpu']
            self.parallel = self.meta['params']['parallel']
        elif not self.cl_args.cpu == None:
            db('  Overwriting cpu with flagged '+str(self.cl_args.cpu))
            self.cpu = self.cl_args.cpu
            if self.cpu > 1:
                self.parallel = True
        # Planar Params
        if self.cl_args.plane == None and json_TF == True:
            if not self.meta['params']['plane_axis'] == None:
                db('Plane in json is: '+str(self.meta['params']['plane_axis']))
                db('at coordinate: '+str(self.meta['params']['plane_coord']))
                self.plane_axis = self.meta['params']['plane_axis'].lower()
                if not self.plane_axis in ['x','y','z']:
                    raise ValueError('\x1b[31;1m'+'ERROR:'+'\x1b[0m'+' --plane needs to begin with'
                                        +' x, y, or z followed by the coordinate on that axis to slice at.')
                self.plane_coord = self.meta['params']['plane_coord']
        elif not self.cl_args.plane == None:
            db('  Overwriting plane with flagged '+str(self.cl_args.plane))
            self.plane_axis = self.cl_args.plane[:1].lower()
            if not self.plane_axis in ['x','y','z']:
                raise ValueError('\x1b[31;1m'+'ERROR:'+'\x1b[0m'+' --plane needs to begin with'
                                    +' x, y, or z followed by the coordinate on that axis to slice at.')
            self.plane_coord = int(self.cl_args.plane[1:])
        # Frames
        if self.cl_args.seq == None and json_TF == True:
            self.frames = self.meta['params']['frames']
            self.soft_seq = self.meta['params']['soft_seq']
        elif not self.cl_args.seq == None:
            db('  Overwriting frames with flagged '+str(self.cl_args.seq))
            self.frames = self.cl_args.seq
            self.soft_seq = self.cl_args.soft_seq
        # Var to plot
        if self.cl_args.var == None and json_TF == True:
            self.frames = self.meta['params']['var_to_plot']
        elif not self.cl_args.var == None:
            db('  Overwriting var_to_plot with flagged '+str(self.cl_args.var))
            self.var_to_plot = self.cl_args.var
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
        dict['cl_args_dump-useless'] = vars(self.cl_args)
        dict['params'] = {
            'adaptive_mesh':self.adaptive_mesh,
            'len_files':self.len_files,
            'cpu':self.cpu,
            'parallel':self.parallel,
            'plane_axis':self.plane_axis,
            'plane_coord':self.plane_coord,
            'frames':self.frames,
            'soft_seq':self.soft_seq,
            'var_to_plot':self.var_to_plot
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
                db('Found parallel_time_file_make.py')
            else:
                raise ValueError('\x1b[31;1m'+'ERROR:'+'\x1b[0m'+
                                 ' path not found for parallel_time_file_make.py')
            if self.cl_args.cpu == None:
                db('Running parallel_time_file_make.py with 1 CPU')
                command = ['python',para_time_file_opt,'1']
            else:
                db('Running parallel_time_file_make.py with '+str(self.cl_args.cpu)+' CPUs')
                command = ['python',para_time_file_opt,str(self.cl_args.cpu)]
            db('    Command being run is: ' + str(command))
            subprocess.run(command)
        # Now read times files and add the parameters
        times_files = np.load('times_files.npy')
        self.times = times_files[:,0].astype(float)
        self.files = times_files[:,1].astype(str)
        self.t_steps = times_files[:,2].astype(int)
        self.file_names = np.unique(self.files)
        self.len_files = len(self.files)
        if "*.e-s*" in self.file_names:
            db('Adaptive Mesh files detected')
            self.adaptive_mesh = True
        else:
            db('No adaptive mesh files detected')
            self.adaptive_mesh = False
        return


    # Define the frames to iterate based on sequence or not
    def get_frames(self):
        db('Calculating frames to operate on based on sequence, number of timesteps, etc.')
        db('--soft-seq must be enabled in command line when you specify a new --seq if you want it'
           +' defaults to off, currently: '+str(self.soft_seq))
        if self.frames == None:
            verb('Using timesteps as frames')
            db('    No --seq framecount set')
            t_frames = self.times
            idx_frames = range(len(t_frames))
        elif self.soft_seq and self.frames<len(self.times):
            verb('--soft-seq: Using timesteps as frames')
            db('    Number of timesteps is less than --seq, so using timesteps instead')
            t_frames = self.times
            idx_frames = range(len(t_frames))
        else:
            verb('Using a sequence')
            db('    Setting --seq value as the number of frames')
            t_max = self.times.max() #self.times[-1]
            t_frames =  np.linspace(0.0,t_max,self.frames)
            idx_frames = [ np.where(self.times-t_frames[i] == min(self.times-t_frames[i],key=abs) )[0][0] for i in range(self.frames) ]
        return idx_frames,t_frames

            # if sequence == True:
            #     if n_frames < len(times):
            #         t_max = times[-1]
            #         t_frames =  np.linspace(0.0,t_max,n_frames)
            #         idx_frames = [ np.where(times-t_frames[i] == min(times-t_frames[i],key=abs) )[0][0] for i in range(n_frames) ]
            #     else:
            #         t_frames = times
            #         idx_frames = range(len(times))
            # elif sequence == False:
            #     t_frames = times
            #     idx_frames = range(len(times))

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

    def t0_properties(self):
        print("doing t0 tesst shit")
        return 7

