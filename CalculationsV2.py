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
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection


# # Needs to have access to '~/projects/TIGER/parallel_time_file_make.py'
# can debug it in another script by running print(calc.__dict__)
# where: calc = CalculationsV2()


pt = logging.warning
verb = logging.info
db = logging.debug


class CalculationsV2:
    def __init__(self): #,files
        self.timerReset()
        self.parse_cl_flags()
        self.set_logging()
        db('Command Line Flags: '+str(self.cl_args))
        self.checkForInput()
        self.times_files()
        self.get_meta(self.cl_args.new_meta)
        self.get_frames()
        self.timerMessage()




    # ██╗███╗   ██╗██╗████████╗██╗ █████╗ ██╗
    # ██║████╗  ██║██║╚══██╔══╝██║██╔══██╗██║
    # ██║██╔██╗ ██║██║   ██║   ██║███████║██║
    # ██║██║╚██╗██║██║   ██║   ██║██╔══██║██║
    # ██║██║ ╚████║██║   ██║   ██║██║  ██║███████╗
    # ╚═╝╚═╝  ╚═══╝╚═╝   ╚═╝   ╚═╝╚═╝  ╚═╝╚══════╝

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
        parser.add_argument('--plot',action='store_true',
                            help='Whether or not to plot results, default=False')
        parser.add_argument('--calc','-c',action='store_true',
                            help='Whether or not to run calculations, default=False')
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
        logging.getLogger('matplotlib.font_manager').disabled = True
        return


    def timerReset(self):
        self.timer = time.perf_counter()
        return


    def timerMessage(self):
        db("Time: "+str(round(time.perf_counter()-self.timer,2))+"s")
        self.timer = time.perf_counter()
        return


    # Check current directory for .i
    def checkForInput(self):
        if glob.glob("*.i"):
            for file in glob.glob("*.i"):
                self.outNameBase = file.rsplit('.',1)[0]
                db('  Input File found, is: ' + self.outNameBase + '.i')
            return self.outNameBase
        else:
            cwd = os.getcwd()
            self.outNameBase = cwd.rsplit('/',1)[1]
            db('  No Input File, using directory name: '+ self.outNameBase)
            return self.outNameBase

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
        self.plot = False
        self.calcs = False
        # json_TF is whether or not there was a json file read first
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
            self.var_to_plot = self.meta['params']['var_to_plot']
        elif not self.cl_args.var == None:
            db('  Overwriting var_to_plot with flagged '+str(self.cl_args.var))
            self.var_to_plot = self.cl_args.var
        # Plot or not
        if self.cl_args.plot == False and json_TF == True:
            self.plot = self.meta['params']['plot']
        elif self.cl_args.plot:
            db('  Overwriting plot with flagged '+str(self.cl_args.plot))
            self.plot = self.cl_args.plot
        # calculate or not
        if self.cl_args.calc == False and json_TF == True:
            self.calcs = self.meta['params']['calcs']
        elif self.cl_args.calc:
            db('  Overwriting plot with flagged '+str(self.cl_args.calc))
            self.calcs = self.cl_args.calc
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
            'var_to_plot':self.var_to_plot,
            'plot':self.plot,
            'calcs':self.calcs
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
            para_time_file_opt = os.path.expanduser('~/projects/TIGER/scripts/parallel_time_file_make.py')
            if os.path.exists(para_time_file_opt):
                db('Found parallel_time_file_make.py')
            else:
                raise ValueError('\x1b[31;1m'+'ERROR:'+'\x1b[0m'+
                                 ' path not found for parallel_time_file_make.py')
            if self.cl_args.cpu == None:
                db('Running parallel_time_file_make.py with 1 CPU')
                command = ['python',para_time_file_opt,'1']
                # db('Running time_file_make.py because parallel wasnt working on mac?')
                # command = ['python',para_time_file_opt.rsplit('/',1)[0]+'/time_file_make.py']
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
        self.idx_frames = idx_frames
        self.t_frames = t_frames
        return idx_frames,t_frames




    #  ██████╗ █████╗ ██╗      ██████╗██╗   ██╗██╗      █████╗ ████████╗██╗ ██████╗ ███╗   ██╗
    # ██╔════╝██╔══██╗██║     ██╔════╝██║   ██║██║     ██╔══██╗╚══██╔══╝██║██╔═══██╗████╗  ██║
    # ██║     ███████║██║     ██║     ██║   ██║██║     ███████║   ██║   ██║██║   ██║██╔██╗ ██║
    # ██║     ██╔══██║██║     ██║     ██║   ██║██║     ██╔══██║   ██║   ██║██║   ██║██║╚██╗██║
    # ╚██████╗██║  ██║███████╗╚██████╗╚██████╔╝███████╗██║  ██║   ██║   ██║╚██████╔╝██║ ╚████║
    #  ╚═════╝╚═╝  ╚═╝╚══════╝ ╚═════╝ ╚═════╝ ╚══════╝╚═╝  ╚═╝   ╚═╝   ╚═╝ ╚═════╝ ╚═╝  ╚═══╝

    # Calculate the mesh centers as [rows of [x y z] ] coordinates and mesh volume
    def mesh_center_quadElements(self,*args):
        db('Calculating the mesh centers for QUAD4(?) elements in 2D or 3D')
        db('Using the input x,y,z coordinates')
        # Refreshing the mesh_ctr value to empty before calculating for current time
        # self.mesh_ctr = None
        # self.mesh_vol = None
        # 2D
        if len(args) == 2 or args[2].nonzero()[0].size == 0:
            db('RUNNING 2D')
            mesh_ctr = np.asarray([args[0][:, 0] + (args[0][:, 2] - args[0][:, 0])/2,
                                        args[1][:, 0] + (args[1][:, 2] - args[1][:, 0])/2 ]).T
            mesh_vol = np.asarray((args[0][:, 2] - args[0][:, 0])*
                                       (args[1][:, 2] - args[1][:, 0]))
        # 3D
        elif len(args) == 3 or not args[2].nonzero()[0].size == 0:
            db('3D based on inputs')
            mesh_ctr = np.asarray([args[0][:, 0] + (args[0][:, 2] - args[0][:, 0])/2,
                                        args[1][:, 0] + (args[1][:, 2] - args[1][:, 0])/2,
                                        args[2][:, 0] + (args[2][:, 4] - args[2][:, 0])/2]).T
            mesh_vol = np.asarray((args[0][:, 2] - args[0][:, 0])*
                                       (args[1][:, 2] - args[1][:, 0])*
                                       (args[2][:, 4] - args[2][:, 0]))
        else:
            raise ValueError('mesh_center_quadElements needs 2 or 3 dimensions of x,y,z input')
        return mesh_ctr, mesh_vol


    # reorder the xyzc data based on planar slicing
    def masking_restructure(self,var,axis):
        # Which 4 points in the QUAD 8 series to use in what order
        if 'x' in axis:
            mask = [0,3,7,4]
        elif 'y' in axis:
            mask = [0,4,5,1]
        elif 'z' in axis:
            mask = [0,1,2,3]
        else:
            raise ValueError('Needs axis to be x y or z!')
        var_out = np.asarray([np.asarray([n1, n2, n3, n4]) for (n1,n2,n3,n4)
                              in zip(var[:,mask[0]], var[:,mask[1]], var[:,mask[2]], var[:,mask[3]])])
        return var_out


    # interpolate the nodal value onto a single plane based on fraction between two exteriors
    # pass arrays of coordinates here
    def plane_interpolate_nodal_quad(self,min,max,axis,plane_coord,var):
        int_frac = (plane_coord - min) / (max - min)
        if 'x' in axis:
            return np.asarray([ var[:,0] + int_frac[:]*(var[:,1]-var[:,0]),
                                var[:,3] + int_frac[:]*(var[:,2]-var[:,3]),
                                var[:,7] + int_frac[:]*(var[:,6]-var[:,7]),
                                var[:,4] + int_frac[:]*(var[:,5]-var[:,4]) ]).T
        elif 'y' in axis:
            return np.asarray([ var[:,0] + int_frac[:]*(var[:,3]-var[:,0]),
                                var[:,4] + int_frac[:]*(var[:,7]-var[:,4]),
                                var[:,5] + int_frac[:]*(var[:,6]-var[:,5]),
                                var[:,1] + int_frac[:]*(var[:,2]-var[:,1]) ]).T
        elif 'z' in axis:
            return np.asarray([ var[:,0] + int_frac[:]*(var[:,4]-var[:,0]),
                                var[:,1] + int_frac[:]*(var[:,5]-var[:,1]),
                                var[:,2] + int_frac[:]*(var[:,6]-var[:,2]),
                                var[:,3] + int_frac[:]*(var[:,7]-var[:,3]) ]).T
        else:
            raise ValueError('Axis needs to be x y or z')


    # Slice the xyz and c data onto a specific plane, and interpolate c
    # returns xyzc as 2d quad elements [[0 1 2 3] ... [0 1 2 3]]
    # Think of x plane (y = x and z = y), y plane (z = x and x = y), z plane (x = x and y = y)
    def plane_slice(self,x,y,z,c):
        # substituting variables based on right hand rule for axes
        if 'x' in self.plane_axis:
            db('Doing plane slice calculations along x-axis at value: '+str(self.plane_coord))
            db('  this means for visualization it will be using y as the x-axis and z as the y-axis')
            v1 = y
            v2 = z
            vs = x
        elif 'y' in self.plane_axis:
            db('Doing plane slice calculations along y-axis at value: '+str(self.plane_coord))
            db('  this means for visualization it will be using z as the x-axis and x as the y-axis')
            v1 = z
            v2 = x
            vs = y
        elif 'z' in self.plane_axis:
            db('Doing plane slice calculations along y-axis at value: '+str(self.plane_coord))
            db('  this means for visualization it will be using x as the x-axis and y as the y-axis')
            v1 = x
            v2 = y
            vs = z
        else:
            pt('\x1b[31;1m'+'ERROR:'+'\x1b[0m'+' plane slice needs axis to be x y or z!')
            return
        # Now trim full data down based on min and max
        # Calculate min/max of each mesh element in slicing direction
        vs_max = np.amax(vs, axis=1)
        vs_min = np.amin(vs, axis=1)
        ind_vs = np.where((self.plane_coord <= vs_max) & (self.plane_coord >= vs_min))
        v1 = v1[ind_vs][:]
        v2 = v2[ind_vs][:]
        vs = vs[ind_vs][:]
        c = c[ind_vs][:]
        # Calculate min/max of each mesh element in slicing direction in the sliced data
        vs_max = np.amax(vs, axis=1)
        vs_min = np.amin(vs, axis=1)
        # flag the specific ones of the shortened set that are ON a plane
        # c_on_plane = np.where((self.plane_coord == vs_max) | (self.plane_coord == vs_min))
        c_on_plane = ((self.plane_coord == vs_max) | (self.plane_coord == vs_min))
        # print(len(c))
        # print(len(c_on_plane))
        # print(sum(c_on_plane))
        new_c = self.plane_interpolate_nodal_quad(vs_min,vs_max,self.plane_axis,self.plane_coord,c)

        v1 = self.masking_restructure(v1,self.plane_axis)
        v2 = self.masking_restructure(v2,self.plane_axis)
        vs = self.plane_coord * np.ones_like(v1)
        # If i want to make them self.plane_x etc do it here
        if 'x' in self.plane_axis:
            return vs, v1, v2, new_c, c_on_plane
        elif 'y' in self.plane_axis:
            return v2, vs, v1, new_c, c_on_plane
        elif 'z' in self.plane_axis:
            return v1, v2, vs, new_c, c_on_plane

    # Using basic xyzc
    # calculate the area*(1-phi) to determine the effective grain area in the plane
    def c_area_in_slice(self,x,y,z,c,c_on_plane=None):
        if len(x[0]) == 8:
            db('starting from full 3D data')
            x, y, z, c, c_on_plane = self.plane_slice(x,y,z,c)
        elif len(x[0]) == 4:
            db('starting from sliced data')
        else:
            pt('\x1b[31;1m'+'ERROR:'+'\x1b[0m'+' data not in QUAD 4 or 8 unit pattern?')
        elem_c = np.average(c, axis=1)
        if 'phi' in self.var_to_plot:
            db('Converting value measured to 1-phi for c area calc')
            elem_c = 1 - elem_c
        if 'x' in self.plane_axis:
            db('Using y as the x-axis and z as the y-axis')
            plt_x = y
            plt_y = z
        elif 'y' in self.plane_axis:
            db('Using z as the x-axis and x as the y-axis')
            plt_x = z
            plt_y = x
        elif 'z' in self.plane_axis:
            db('Using x as the x-axis and y as the y-axis')
            plt_x = x
            plt_y = y
        else:
            pt('\x1b[31;1m'+'ERROR:'+'\x1b[0m'+' plane measuring c on not specified!')
        mesh_ctr, mesh_vol = self.mesh_center_quadElements(plt_x,plt_y)
        # instead of removing duplicates, just half the area of the doubled cells,
        # they should still have the same c value in both dupes
        mesh_vol = np.where(c_on_plane, mesh_vol/2, mesh_vol)
        area_weighted_c = elem_c*mesh_vol
        tot_area_c = np.sum(area_weighted_c)
        tot_area_mesh = np.sum(mesh_vol)
        return tot_area_c, tot_area_mesh





    # ██████╗ ██╗      ██████╗ ████████╗████████╗██╗███╗   ██╗ ██████╗
    # ██╔══██╗██║     ██╔═══██╗╚══██╔══╝╚══██╔══╝██║████╗  ██║██╔════╝
    # ██████╔╝██║     ██║   ██║   ██║      ██║   ██║██╔██╗ ██║██║  ███╗
    # ██╔═══╝ ██║     ██║   ██║   ██║      ██║   ██║██║╚██╗██║██║   ██║
    # ██║     ███████╗╚██████╔╝   ██║      ██║   ██║██║ ╚████║╚██████╔╝
    # ╚═╝     ╚══════╝ ╚═════╝    ╚═╝      ╚═╝   ╚═╝╚═╝  ╚═══╝ ╚═════╝


    def plot_slice(self,frame,x,y,z,c):
        db('Plotting the sliced data')
        # Make pics subdirectory if it doesnt exist
        pic_directory = 'pics'
        if not os.path.isdir(pic_directory):
            db('Making picture directory: '+pic_directory)
            os.makedirs(pic_directory)
        db('Plotting the slice as specified')
        # Take the average of the 4 corner values for c
        if hasattr(c[0], "__len__"):
            plotc = np.average(c, axis=1)
        else:
            plotc = c
        if 'x' in self.plane_axis:
            db('Plotting using y as the x-axis and z as the y-axis')
            plt_x = y
            plt_y = z
        elif 'y' in self.plane_axis:
            db('Plotting using z as the x-axis and x as the y-axis')
            plt_x = z
            plt_y = x
        elif 'z' in self.plane_axis:
            db('Plotting using x as the x-axis and y as the y-axis')
            plt_x = x
            plt_y = y
        else:
            pt('\x1b[31;1m'+'ERROR:'+'\x1b[0m'+' plot_slice needs axis to be x y or z!')
            return
        coords = np.asarray([ np.asarray([x_val,y_val]).T for (x_val,y_val) in zip(plt_x,plt_y) ])
        fig, ax = plt.subplots()
        p = PolyCollection(coords, cmap=matplotlib.cm.coolwarm, alpha=1)#,edgecolor='k'
        p.set_array(np.array(plotc) )
        ax.add_collection(p)
        ax.set_xlim([np.amin(plt_x),np.amax(plt_x)])
        ax.set_ylim([np.amin(plt_y),np.amax(plt_y)])
        #SET ASPECT RATIO TO EQUAL TO ENSURE IMAGE HAS SAME ASPECT RATIO AS ACTUAL MESH
        ax.set_aspect('equal')
        #ADD A COLORBAR, VALUE SET USING OUR COLORED POLYGON COLLECTION, [0,1]
        p.set_clim(0.0, 1.0)
        fig.colorbar(p, label=self.var_to_plot)
        fig.savefig(pic_directory+'/'+self.outNameBase+'_sliced_'+self.plane_axis+
                    str(self.plane_coord)+'_'+str(frame)+'.png',dpi=500,transparent=True )
        if self.cl_args.debug:
            plt.show()
        else:
            plt.close()
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


    # Doesnt work in here for parallel stuff
    # def getMER(self):
    #     self.MER = MultiExodusReader(self.file_names[0])
    #     print("Got it")
    #     print(self.MER)
    #     return

# # Has to reread input file(s) at the timestep to run from here in parallel
# def calc_parallelPlot(self,i,idx_frame):
#     para_t0 = time.perf_counter()
#     MF = MultiExodusReader(self.file_names[0])
#     x,y,z,c = MF.get_data_at_time(self.var_to_plot,self.times[idx_frame],True)
#     nx, ny, nz, nc = self.plane_slice(x,y,z,c)
#     self.plot_slice(i,nx,ny,nz,nc)
#     verb('  Finished plotting file '+str(i)+': '+str(round(time.perf_counter()-para_t0,2))+'s')
#     return idx_frame

