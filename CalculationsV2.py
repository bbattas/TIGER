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
import cv2
import math


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


    def timerMessage(self,message=None):
        # db("Time: "+str(round(time.perf_counter()-self.timer,2))+"s")
        if message == None:
            print("Time: "+str(round(time.perf_counter()-self.timer,2))+"s")
        else:
            db(str(message)+" (Time: "+str(round(time.perf_counter()-self.timer,2))+"s)")
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
        # Plane name for if it gets moved
        self.plane_coord_name = self.plane_coord
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
            para_time_file_opt = os.path.expanduser('~/projects/TIGER/parallel_time_file_make.py')
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


    # Check if plane is on the boundary between elements, if it is preferentially move it
    # positive direction by tolerance if it can, otherwise move negative by tol
    def adjust_plane(self,vs_max,vs_min):
        tol = 1e-6
        planeval = self.plane_coord
        dom_max = max(vs_max)
        dom_min = min(vs_min)
        c_on_plane = ((planeval == vs_max) | (planeval == vs_min))
        if np.any(c_on_plane):
            verb('MOVING Plane')
            db('Plane slicing is on boundary between elements, moving...')
            if (planeval + tol) < dom_max:
                db('Shifting plane + tol')
                self.plane_coord = planeval + tol
            elif (planeval - tol) > dom_min:
                db('Shifting plane - tol')
                self.plane_coord = planeval - tol
            else:
                raise ValueError('Plane coordinate is apparently within tolerance of domain min and max')
            # Check that it worked
            c_on_plane = ((self.plane_coord == vs_max) | (self.plane_coord == vs_min))
            if np.any(c_on_plane):
                raise ValueError('After shift, plane is still on boundary!')
            else:
                db('Plane now '+str(self.plane_coord))
                return
        else:
            db('Plane not on boundary, doesnt need moved')
            verb('NOT MOVING Plane')
            return



    # Slice the xyz and c data onto a specific plane, and interpolate c
    # returns xyzc as 2d quad elements [[0 1 2 3] ... [0 1 2 3]]
    # Think of x plane (y = x and z = y), y plane (z = x and x = y), z plane (x = x and y = y)
    def plane_slice(self,x,y,z,c,grads=False):
        # substituting variables based on right hand rule for axes
        v1, v2, vs, xyz_ref = self.plt_xyz(x,y,z)
        # Now trim full data down based on min and max
        # Calculate min/max of each mesh element in slicing direction
        vs_max = np.amax(vs, axis=1)
        vs_min = np.amin(vs, axis=1)
        # If we are on a mesh plane then we need to adjust the plane_coord by a tiny amount
        # to move off the plane
        self.adjust_plane(vs_max,vs_min)
        ind_vs = np.where((self.plane_coord <= vs_max) & (self.plane_coord >= vs_min))
        v1 = v1[ind_vs][:]
        v2 = v2[ind_vs][:]
        vs = vs[ind_vs][:]
        c = c[ind_vs][:]
        # Calculate min/max of each mesh element in slicing direction in the sliced data
        vs_max = np.amax(vs, axis=1)
        vs_min = np.amin(vs, axis=1)
        if grads:
            print('DEPRECIATED')
            dc = self.element_gradients(v1,v2,vs,c,xyz_ref)
            normdc = self.element_curvature(v1,v2,vs,c,xyz_ref)

        # Interpolate c based on where in the plane heightwise
        new_c = self.plane_interpolate_nodal_quad(vs_min,vs_max,self.plane_axis,self.plane_coord,c)

        v1 = self.masking_restructure(v1,self.plane_axis)
        v2 = self.masking_restructure(v2,self.plane_axis)
        vs = self.plane_coord * np.ones_like(v1)
        # If i want to make them self.plane_x etc do it here
        if grads:
            db('Outputting the dc/dxyz also from plane_slice')
            if 'x' in self.plane_axis:
                return vs, v1, v2, new_c, dc, normdc
            elif 'y' in self.plane_axis:
                return v2, vs, v1, new_c, dc, normdc
            elif 'z' in self.plane_axis:
                return v1, v2, vs, new_c, dc, normdc
        else:
            if 'x' in self.plane_axis:
                return vs, v1, v2, new_c
            elif 'y' in self.plane_axis:
                return v2, vs, v1, new_c
            elif 'z' in self.plane_axis:
                return v1, v2, vs, new_c


    # convert xyz into the x and y equivalent for whatever plane we are using
    def plt_xy(self,x,y,z):
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
            raise ValueError('Plane measuring c on not specified!')
        return plt_x, plt_y

    # convert xyz into the plane slicing relative xyz equivalent for whatever plane we are using
    def plt_xyz(self,x,y,z,ctr=None):
        if 'x' in self.plane_axis:
            db('Using y as the x-axis and z as the y-axis')
            plt_x = y
            plt_y = z
            plt_z = x
            mask = [1,2,0]
        elif 'y' in self.plane_axis:
            db('Using z as the x-axis and x as the y-axis')
            plt_x = z
            plt_y = x
            plt_z = y
            mask = [2,0,1]
        elif 'z' in self.plane_axis:
            db('Using x as the x-axis and y as the y-axis')
            plt_x = x
            plt_y = y
            plt_z = z
            mask = [0,1,2]
        else:
            raise ValueError('Plane measuring c on not specified!')
        if not ctr is None:
            var_out = np.asarray([np.asarray([n1, n2, n3]) for (n1,n2,n3)
                              in zip(ctr[:,mask[0]], ctr[:,mask[1]], ctr[:,mask[2]])])
            return plt_x, plt_y, plt_z, var_out
        else:
            return plt_x, plt_y, plt_z, mask


    # Using basic xyzc
    # calculate the area*(1-phi) to determine the effective grain area in the plane
    def c_area_in_slice(self,x,y,z,c,varname):
        if len(x[0]) == 8:
            db('starting from full 3D data')
            x, y, z, c = self.plane_slice(x,y,z,c)
        elif len(x[0]) == 4:
            db('starting from sliced data')
        else:
            pt('\x1b[31;1m'+'ERROR:'+'\x1b[0m'+' data not in QUAD 4 or 8 unit pattern?')
        elem_c = np.average(c, axis=1)
        if 'phi' in varname:
            db('Converting value measured to 1-phi for c area calc')
            elem_c = 1 - elem_c
        plt_x, plt_y = self.plt_xy(x,y,z)
        mesh_vol = self.mesh_vol_dir_independent(plt_x,plt_y)
        # instead of removing duplicates, just half the area of the doubled cells,
        # they should still have the same c value in both dupes
        # mesh_vol = np.where(c_on_plane, mesh_vol/2, mesh_vol)
        area_weighted_c = elem_c*mesh_vol
        tot_area_c = np.sum(area_weighted_c)
        tot_area_mesh = np.sum(mesh_vol)
        return tot_area_c, tot_area_mesh


    # calculate the d_var across the direction specified in each element
    def element_gradient_inDirection(self,var,xyz_ref):
        print('DEPRECIATED')
        d_var = []
        if type(xyz_ref) is int:
            xyz_ref = [xyz_ref]
        for ax in xyz_ref:
            # Which 4 points in the QUAD 8 series to use in what order
            # along actual x axis
            if ax == 0:
                mask0 = [0,3,7,4]
                mask1 = [1,2,6,5]
            # along actual y axis
            elif ax == 1:
                mask0 = [0,4,5,1]
                mask1 = [3,7,6,2]
            # along actual z axis
            elif ax == 2:
                mask0 = [0,1,2,3]
                mask1 = [4,5,6,7]
            else:
                raise ValueError('Needs axis reference to be 0, 1, or 2 !')
            var_0 = np.asarray([np.asarray([n1, n2, n3, n4]) for (n1,n2,n3,n4)
                                in zip(var[:,mask0[0]], var[:,mask0[1]], var[:,mask0[2]], var[:,mask0[3]])])
            var_1 = np.asarray([np.asarray([n1, n2, n3, n4]) for (n1,n2,n3,n4)
                                in zip(var[:,mask1[0]], var[:,mask1[1]], var[:,mask1[2]], var[:,mask1[3]])])
            avg_0 = np.average(var_0,axis=1)
            avg_1 = np.average(var_1,axis=1)
            d_var.append(avg_1-avg_0)
        return d_var


    # calcualtes the dcdx dcdy dcdz in the reference frame provided by xyz_mask
    # [0,1,2] is normal xyz, see self.plt_xyz() output mask
    def element_gradients(self,plt_x,plt_y,plt_z,c,xyz_mask):
        print('DEPRECIATED')
        dx = self.element_gradient_inDirection(plt_x,xyz_mask[0])
        dy = self.element_gradient_inDirection(plt_y,xyz_mask[1])
        dz = self.element_gradient_inDirection(plt_z,xyz_mask[2])
        dc = self.element_gradient_inDirection(c,xyz_mask)
        dcdxyz = np.asarray(dc) / np.asarray(dx + dy + dz)
        return dcdxyz

    # DIDNT WORK: need more z data to derrive
    def element_curvature(self,plt_x,plt_y,plt_z,c,xyz_mask):
        print('DEPRECIATED')
        # dc/dx,dc/dy,dc/dz
        dcdxyz = self.element_gradients(plt_x,plt_y,plt_z,c,xyz_mask)
        # dc/dx,dc/dy,dc/dz magnitude
        norm_sq = 0
        for n in range(len(dcdxyz)):
            norm_sq += dcdxyz[n]**2
        norm = np.sqrt(norm_sq)
        # Curvature starts:
        # grad c / mag(grad c)
        dc_dcnorm = dcdxyz / norm
        # Divergence of dc_dcnorm
        # dx = self.element_gradient_inDirection(plt_x,xyz_mask[0])
        # dy = self.element_gradient_inDirection(plt_y,xyz_mask[1])
        # dz = self.element_gradient_inDirection(plt_z,xyz_mask[2])
        # dc_dcnorm_x = self.element_gradient_inDirection(dc_dcnorm[0],xyz_mask[0])
        # dc_dcnorm_y = self.element_gradient_inDirection(dc_dcnorm[1],xyz_mask[1])
        # dc_dcnorm_z = self.element_gradient_inDirection(dc_dcnorm[2],xyz_mask[2])
        # curve_vec = np.asarray(dc_dcnorm_x + dc_dcnorm_y + dc_dcnorm_z) / np.asarray(dx + dy + dz)
        # print(curve_vec)
        return dc_dcnorm


    # output full 8 node elemental data for a planar slice
    # Keeps nodes in same orientation they come in
    def trim_data_to_plane(self,x,y,z,c,z_plane):
        # Calculate min/max of each mesh element in slicing direction
        z_max = np.amax(z, axis=1)
        z_min = np.amin(z, axis=1)
        if np.any(((z_plane == z_max) | (z_plane == z_min))):
            raise ValueError('trim_data_to_plane needs the plane to not be a mesh boundary plane')
        ind_z = np.where((z_plane < z_max) & (z_plane > z_min))
        x = x[ind_z][:]
        y = y[ind_z][:]
        z = z[ind_z][:]
        c = c[ind_z][:]
        return x, y, z, c


    def get_c_index(self,xyz_ctr, c, x, y, z):
        ind = (xyz_ctr == (x,y,z)).all(axis=1)
        row = c[ind]
        return row

    # pass it ctr and find corresponding X Y and teh C for that XY
    def undo_c_index(self,xy_ctr, C, X, Y):
        ind = [(X[:,0] == xy_ctr[0]), (Y[0] == xy_ctr[1])]
        c_val = C[tuple(ind)]
        return c_val

    # Uses min and max value of each coordinate (xyz) in each element to return center
    def mesh_center_dir_independent(self,*args):
        min = []
        max = []
        for ar in args:
            min.append(np.amin(ar,axis=1))
            max.append(np.amax(ar,axis=1))
        if len(args) == 2:
            db('RUNNING 2D')
            mesh_ctr = np.asarray([min[0][:] + (max[0][:] - min[0][:])/2,
                                   min[1][:] + (max[1][:] - min[1][:])/2 ]).T
        # 3D
        elif len(args) == 3:
            db('3D based on inputs')
            mesh_ctr = np.asarray([min[0][:] + (max[0][:] - min[0][:])/2,
                                   min[1][:] + (max[1][:] - min[1][:])/2,
                                   min[2][:] + (max[2][:] - min[2][:])/2 ]).T
        else:
            raise ValueError('mesh_center_quadElements needs 2 or 3 dimensions of x,y,z input')
        return mesh_ctr

    # Uses min and max value of each coordinate in each element to return volume/area
    def mesh_vol_dir_independent(self,*args):
        min = []
        max = []
        for ar in args:
            min.append(np.amin(ar,axis=1))
            max.append(np.amax(ar,axis=1))
        if len(args) == 2:
            db('RUNNING 2D')
            mesh_vol = np.asarray((max[0][:] - min[0][:])*
                                  (max[1][:] - min[1][:]) )
        # 3D
        elif len(args) == 3:
            db('3D based on inputs')
            mesh_vol = np.asarray((max[0][:] - min[0][:])*
                                  (max[1][:] - min[1][:])*
                                  (max[2][:] - min[2][:]))
        else:
            raise ValueError('mesh_vol_dir_independent needs 2 or 3 dimensions of x,y,z input')
        return mesh_vol


    def threeplane_dataReduce(self,x,y,z,c,z_planes):
        if not hasattr(z_planes, '__iter__'):
            z_planes = [z_planes]
        x_out = []
        y_out = []
        z_out = []
        c_out = []
        for i,pl in enumerate(z_planes):
            x_tmp, y_tmp, z_tmp, c_tmp = self.trim_data_to_plane(x,y,z,c,pl)
            x_out.append(x_tmp)
            y_out.append(y_tmp)
            z_out.append(z_tmp)
            c_out.append(c_tmp)
        return np.concatenate(x_out), np.concatenate(y_out), np.concatenate(z_out), np.concatenate(c_out)


    def threeplane_norm(self,dx,dy,dz):
        norm_sq = [0, 0, 0]
        for n in range(3):
            norm_sq[n] = dx[n]**2 + dy[n]**2 + dz[n]**2
        norm = np.sqrt(norm_sq)
        return norm


    # Feed real x y z
    # All the threeplane uses value = [[plane0],[plane1],[plane2]]
    # where each plane is the value = [[0 1 2 ... n], [0 1 2 ...n], ... [0 1 2 ... n]] for that plane
    def threeplane_curvature(self,x,y,z,c,full_out=False):
        db('Using the center of 3 planes of data to calculate shit')
        db('If using a surface plane without a mesh plane above and below it, wont work at the moment')
        # self.timerReset()
        # Full 3D mesh element centers
        mesh_ctr, mesh_vol = self.mesh_center_quadElements(x,y,z)
        # convert xyz into plot orientation
        plt_x,plt_y,plt_z,xyz_ref = self.plt_xyz(x,y,z)
        plt_ctr = np.asarray([np.asarray([n1, n2, n3]) for (n1,n2,n3)
                              in zip(mesh_ctr[:,xyz_ref[0]], mesh_ctr[:,xyz_ref[1]], mesh_ctr[:,xyz_ref[2]])])
        # Unique Mesh Element Coordinates (CHECKSUM THIS???)
        plt_x_u = np.unique(plt_ctr[:,0])
        plt_y_u = np.unique(plt_ctr[:,1])
        plt_z_u = np.unique(plt_ctr[:,2])
        # Calculate min/max of each mesh element in slicing direction
        z_max = np.amax(plt_z, axis=1)
        z_min = np.amin(plt_z, axis=1)
        # Adjust plane if it needs to shift off boundary
        self.adjust_plane(z_max,z_min)
        # find the closest (3) unique plt_z value
        ind_z_u_ctr = np.absolute(plt_z_u-self.plane_coord).argmin()
        ind_z_planes = [ind_z_u_ctr - 1, ind_z_u_ctr, ind_z_u_ctr + 1]
        # print(ind_z_planes)
        # print(plt_z_u[ind_z_planes])
        # Trim down the full 3D data to just that of the planes of interest combined into single arrays
        cutx,cuty,cutz,cutc = self.threeplane_dataReduce(plt_x,plt_y,plt_z,c,plt_z_u[ind_z_planes])
        cut_ctr = self.mesh_center_dir_independent(cutx, cuty, cutz)
        cut_c_avg = np.average(cutc,axis=1)
        # Make a meshgrid, sectioned by x,y on planes of z
        X,Z,Y = np.meshgrid(plt_x_u, plt_z_u[ind_z_planes], plt_y_u,  indexing='xy')
        # self.timerReset()
        # Convert the 3 plane c data to the meshgrid equivalent
        c_sort = np.array([self.get_c_index(cut_ctr,cut_c_avg,x,y,z) for (x,y,z) in zip(np.ravel(X), np.ravel(Y), np.ravel(Z))])
        C = c_sort.reshape(X.shape)
        # self.timerMessage()
        # Calculate Gradients
        # The reference to xyz_ref in gradients is wrong, theyre all in plt ref here
        # dx = np.gradient(X)[xyz_ref[0]]
        # dy = np.gradient(Y)[xyz_ref[1]]
        # dz = np.gradient(Z)[xyz_ref[2]]
        dx = np.gradient(X)[1]
        dy = np.gradient(Y)[2]
        dz = np.gradient(Z)[0]
        dc = np.gradient(C)
        # dcdx = np.asarray(dc[xyz_ref[0]]) / np.asarray(dx)
        # dcdy = np.asarray(dc[xyz_ref[1]]) / np.asarray(dy)
        # dcdz = np.asarray(dc[xyz_ref[2]]) / np.asarray(dz)
        dcdx = np.asarray(dc[1]) / np.asarray(dx)
        dcdy = np.asarray(dc[2]) / np.asarray(dy)
        dcdz = np.asarray(dc[0]) / np.asarray(dz)
        # Gradient Norms
        dc_norm = self.threeplane_norm(dcdx,dcdy,dcdz)
        # Curvature part: grad/grad.norm
        # This starts getting weird away from the relevant OP
        dcdx_norm = np.asarray(dcdx)/np.asarray(dc_norm)#[0]
        dcdy_norm = np.asarray(dcdy)/np.asarray(dc_norm)#[1]
        dcdz_norm = np.asarray(dcdz)/np.asarray(dc_norm)#[2]
        # Divergence of that!
        # dcdx_normdx = np.asarray(np.gradient(dcdx_norm)[xyz_ref[0]]) / np.asarray(dx)
        # dcdy_normdy = np.asarray(np.gradient(dcdy_norm)[xyz_ref[1]]) / np.asarray(dy)
        # dcdz_normdz = np.asarray(np.gradient(dcdz_norm)[xyz_ref[2]]) / np.asarray(dz)
        dcdx_normdx = np.asarray(np.gradient(dcdx_norm)[1]) / np.asarray(dx)
        dcdy_normdy = np.asarray(np.gradient(dcdy_norm)[2]) / np.asarray(dy)
        dcdz_normdz = np.asarray(np.gradient(dcdz_norm)[0]) / np.asarray(dz)
        curvature = dcdx_normdx + dcdy_normdy + dcdz_normdz
        # Output data for just the middle plane
        out_pltx, out_plty, out_pltz, out_c_full = self.threeplane_dataReduce(
            plt_x,plt_y,plt_z,c,plt_z_u[ind_z_u_ctr])
        out_c_avg = np.average(out_c_full,axis=1)
        # Calculate 2D mesh centers on that data
        out_ctr = self.mesh_center_dir_independent(out_pltx, out_plty)
        # Use those ctrs to generate curvature for middle plane in same format as the c_avg would be
        # self.timerReset()
        c_out = np.array([self.undo_c_index(xy,curvature[1],X[1],Y[1]) for (xy) in out_ctr])
        # self.timerMessage()
        curve_out = c_out.reshape(out_c_avg.shape)
        # If full_out == False then slice to a 4 point box on a plane instead of 8pt cube
        # This might be dated a bit now? better ways maybe?
        if not full_out:
            z_max = np.amax(out_pltz, axis=1)
            z_min = np.amin(out_pltz, axis=1)
            # Interpolate c based on where in the plane heightwise
            slice_c = self.plane_interpolate_nodal_quad(z_min,z_max,self.plane_axis,
                                                        self.plane_coord,out_c_full)
            out_pltx = self.masking_restructure(out_pltx,self.plane_axis)
            out_plty = self.masking_restructure(out_plty,self.plane_axis)
            out_pltz = self.plane_coord * np.ones_like(out_pltx)
            out_c_full = slice_c
        holdinglist = [out_pltx, out_plty, out_pltz]
        outlist = [0,0,0]
        for i,ref in enumerate(xyz_ref):
            outlist[ref] = holdinglist[i]
        # self.timerMessage('Curvature Meshgrid Done')
        return outlist[0], outlist[1], outlist[2], out_c_full, curve_out

    # Based on where gr# goes to 0 (is < e(=0.2))
    def delta_interface_func(self,c,limit=None):
        # EQ14 in Johnson/Voorhees 2014 (https://doi.org/10.1016/j.actamat.2013.12.012)
        # Element form check for c
        if hasattr(c[0], "__len__"):
            plotc = np.average(c, axis=1)
        else:
            plotc = c
        e = 0.2
        # delta = (1 + np.cos((np.pi * plotc)/e))/(2*e)
        delta = np.where(np.absolute(plotc)<e,(1 + np.cos((np.pi * plotc)/e))/(2*e),0)
        if limit is not None:
            delta = np.where(delta<=limit,delta,0)
        return delta

    # Full single OP curvature calculations
    # For the GB plane only, need to sum all the ops delta funcs and use that combined one only
    def MER_curvature_calcs(self,x,y,z,c,xyz_out=False):
        self.timerReset()
        cx,cy,cz,cc,cv = self.threeplane_curvature(x,y,z,c)
        delta_func = self.delta_interface_func(cc)#,1
        delta_cv = delta_func*cv
        sum_delta_cv = np.sum(delta_cv)
        self.timerMessage('Full Curvature Calculation')
        if xyz_out:
            return sum_delta_cv, delta_func, cv, cx, cy, cz, cc
        else:
            return sum_delta_cv, delta_func, cv


    # ACTUAL IMAGE CURVATURE
    def image_scaleFactor(self,bw_img_w_box,xrange,yrange=None):
        '''Reads the image of the GB plane OPs summed with a 0-1 white-black colorbar and
        based on the CV2 pixels in the image calculates the ratio between CV2 pixels and
        actual domain units (uses the box outlining the plot and the max x dimension)

        Args:
            bw_img_w_box: .png image of gr0+gr1 in B/W with a box around the domain (plot_slice_forCurvature output)
            xrange: Domain x range (max x - min x)
            yrange: Domain y range (not used currently). Defaults to None.

        Returns:
            scaling factor (nm/pixel), plot pixel x minimum value, plot pixel y minimum value
        '''
        # Pixels go 0 -> n for x but y is 0 v n down (0,0 is top left)
        # https://pyimagesearch.com/2021/01/20/opencv-getting-and-setting-pixels/
        src = cv2.imread(bw_img_w_box)#,-1
        gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)  # convert to grayscale
        blur = cv2.blur(gray, (3, 3))
        ret, thresh = cv2.threshold(blur, 50, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        n = 0
        if 0 in contours[n]:
            n += 1
        x,y = contours[n].T
        x_pixels = max(x[0]) - min(x[0])
        # Contour Testing Image
        # # src = cv2.imread('02_3grain_base_cv2_gb_x250_test_60.png')
        # cv2.drawContours(src, contours, n, (255,0,0), 1)
        # cv2.imwrite('scaling_domain_contour.jpg', src)
        return xrange/x_pixels, min(x[0]), min(y[0])

    def find_circle(self,a,b,c,fullcontour,returnXY=False):
        '''Calculates the circle passing through 3 points using the vertex method and
        the circle formulation process from http://paulbourke.net/geometry/circlesphere/

        Args:
            a: Point 1 (x,y)
            b: Point 2 (x,y) (middle of abc)
            c: Point 3 (x,y)
            fullcontour: The cv2 contour containing all the points
            returnXY: Whether or not to append the x_ctr and y_ctr. Defaults to False.

        Returns:
            r (radius), signage (-1 when the point is outside the contour, 1 inside and 0 on it)
        '''
        # Using the process from http://paulbourke.net/geometry/circlesphere/
        # slope of ab and bc
        tol = 1e-8
        if b[0] == a[0] and b[0] == c[0]:
            # Vertical line
            # print('vertical')
            if returnXY:
                return -1, 0, 0, 0
            else:
                return -1, 0
        # Calculate Slopes between the points
        m1 = (b[1] - a[1])/(b[0] - a[0])
        m2 = (c[1] - b[1])/(c[0] - b[0])
        if m2 == m1:
            # Horizontal line
            if returnXY:
                return -1, 0, 0, 0
            else:
                return -1, 0
        # Correction for when 2 points are stacked vertically to prevent the slope = nan
        if b[0] == a[0]:
            m1 = (b[1] - a[1])/(tol)
        if b[0] == c[0]:
            m2 = (c[1] - b[1])/(tol)
        # Calculate the center points based on where the lines perpendicular cross
        x_ctr = (m1*m2*(a[1] - c[1]) + m2*(a[0] + b[0]) - m1*(b[0] + c[0]))/(2*(m2 - m1))
        # y_ctr = -(1/m1)*(x_ctr - ((a[0] + b[0])/2) ) + ((a[1] + b[1])/2)
        y_ctr = (m1*(a[1] + b[1]) - m2*(b[1] + c[1]) + (a[0] - c[0]))/(2*(m1-m2))
        r = math.sqrt((a[0] - x_ctr)**2 + (a[1] - y_ctr)**2)
        # this returns -1 when the point is outside the contour, 1 inside and 0 on it
        signage = cv2.pointPolygonTest(fullcontour, (x_ctr,y_ctr), False)
        if returnXY:
            return r, signage, x_ctr, y_ctr
        else:
            return r, signage


    def curvature_fromImage(self,bw_img_w_box,xrange,yrange,nn):
        '''Calcaulates the curvature of the GB plane using CV2 contour

        Args:
            bw_img_w_box: .png image of gr0+gr1 in B/W with a box around the domain (plot_slice_forCurvature output)
            xrange: Domain x range (max-min)
            yrange: Domain y range (max-min)
            nn: Next Nearest parameter (1,2,3...) for +/- nn to select 3 points for a circle

        Returns:
            cv: Curvature (1/R) average value for the whole boundary
        '''
        filename = bw_img_w_box.rsplit('.',1)[0]
        frame = [int(s) for s in filename.split('_') if s.isdigit()]
        # frame = [int(x) for x in regex.findall(bw_img_w_box)]
        frame = frame[-1]
        scale, xmin, ymin = self.image_scaleFactor(bw_img_w_box,xrange)
        src = cv2.imread(bw_img_w_box)#,-1
        gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)  # convert to grayscale
        blur = cv2.medianBlur(gray,45)
        ret, thresh = cv2.threshold(blur, 50, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)#CHAIN_APPROX_SIMPLE)
        n = 0
        # print(contours)
        if 0 in contours[n]:
            n += 1
        x,y = contours[n].T
        if xmin in x[0] or ymin in y[0]:
            # print("Plot Boundary in second contour, using third")
            n +=1
            x,y = contours[n].T
        xy = [np.asarray([a,b]) for (a,b) in zip(x[0],y[0])]
        rad_loc = []
        for i in range(len(xy)):
            if (i + nn) > (len(xy) - 1):
                rad_loc.append(self.find_circle(xy[i-nn],xy[i],xy[i+nn-len(xy)],contours[n]))
            else:
                rad_loc.append(self.find_circle(xy[i-nn],xy[i],xy[i+nn],contours[n]))
        radii = [row[0] for row in rad_loc]
        sign = [row[1] for row in rad_loc]
        curvature = 1/(np.asarray(radii)*scale)
        curvature = np.where(np.asarray(radii)==-1,0,-1*np.asarray(sign)*curvature)
        cv = np.average(curvature)
        # # Testing Plot
        cm = plt.cm.get_cmap('RdYlBu')
        sc = plt.scatter((x-xmin)*scale, (y-ymin)*scale, c=curvature, vmin=min(curvature), vmax=max(curvature), s=35, cmap=cm)
        plt.colorbar(sc)
        plt.xlim([0,xrange])
        plt.ylim([0,yrange])
        plt.gca().set_aspect('equal')
        # plt.show()
        plt.savefig('pics/'+self.outNameBase+'_curvature_'+self.plane_axis+
                    str(self.plane_coord_name)+'_'+str(frame)+'.png',dpi=500,transparent=True )
        plt.close()
        return cv

    def gb_curvature(self,x,y,z,cgb,nn,frame):
        '''Caclulate the gb curvature and gb area on the specified plane of interest

        Args:
            x: Full 3D x data
            y: Full 3D y data
            z: Full 3D z data
            cgb: Combined gr0+gr1 full 3D data
            nn: Nearest Neighbor (1,2,3...) for curvature calculation
            frame: Frame number for the curvature image

        Returns:
            cv: Curvature (1/R) of gr0+gr1 on the plane
            gb_area: weighted area of gr0+gr1 on the plane
            tot_mesh_area: total mesh area on the plane
        '''
        cx,cy,cz,ccgb = self.plane_slice(x,y,z,cgb)
        gb_area, tot_mesh_area = self.c_area_in_slice(cx,cy,cz,ccgb,'gb')
        figname = self.plot_slice_forCurvature(str(frame),cx,cy,cz,ccgb,'gr0 + gr1')
        pltx,plty,pltz,xyzref = self.plt_xyz(cx,cy,cz)
        cv = self.curvature_fromImage(figname,pltx.max(),plty.max(),nn)
        return cv, gb_area, tot_mesh_area

    def rbm_distance_centroids(self,x,y,z,c):
        c_int = np.rint(c)
        zeros = np.zeros_like(c_int)
        mesh_ctr = self.mesh_center_dir_independent(x,y,z)
        mesh_vol = self.mesh_vol_dir_independent(x,y,z)
        # Setup Empty sets for the calculation
        volumes = []
        grain_centroids = []
        # Loop over the Unique Grains for gr0/gr1 (and not phi)
        for n in range(2):
            volumes.append(np.sum(np.where(c_int==(n),mesh_vol,zeros)))
            # if volumes[n] > 0.0:
            grain_centroids.append([np.sum(np.where(c_int==(n),mesh_ctr[:,0] * mesh_vol,zeros)) / np.sum(np.where(c_int==(n),mesh_vol,zeros)),
                                    np.sum(np.where(c_int==(n),mesh_ctr[:,1] * mesh_vol,zeros)) / np.sum(np.where(c_int==(n),mesh_vol,zeros)),
                                    np.sum(np.where(c_int==(n),mesh_ctr[:,2] * mesh_vol,zeros)) / np.sum(np.where(c_int==(n),mesh_vol,zeros))])
        # Calculate the distance between the two centroids
        dist = math.sqrt((grain_centroids[1][0]-grain_centroids[0][0])**2 +
                         (grain_centroids[1][1]-grain_centroids[0][1])**2 +
                         (grain_centroids[1][2]-grain_centroids[0][2])**2 )
        return dist

    # ██████╗ ██╗      ██████╗ ████████╗████████╗██╗███╗   ██╗ ██████╗
    # ██╔══██╗██║     ██╔═══██╗╚══██╔══╝╚══██╔══╝██║████╗  ██║██╔════╝
    # ██████╔╝██║     ██║   ██║   ██║      ██║   ██║██╔██╗ ██║██║  ███╗
    # ██╔═══╝ ██║     ██║   ██║   ██║      ██║   ██║██║╚██╗██║██║   ██║
    # ██║     ███████╗╚██████╔╝   ██║      ██║   ██║██║ ╚████║╚██████╔╝
    # ╚═╝     ╚══════╝ ╚═════╝    ╚═╝      ╚═╝   ╚═╝╚═╝  ╚═══╝ ╚═════╝


    def plot_slice(self,frame,x,y,z,c,cb_label=None):
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
        plt_x, plt_y = self.plt_xy(x,y,z)
        coords = np.asarray([ np.asarray([x_val,y_val]).T for (x_val,y_val) in zip(plt_x,plt_y) ])
        fig, ax = plt.subplots()
        p = PolyCollection(coords, cmap=matplotlib.cm.coolwarm, alpha=1)#,edgecolor='k'
        p.set_array(np.array(plotc) )
        ax.add_collection(p)
        ax.set_xlim([np.amin(plt_x),np.amax(plt_x)])
        ax.set_ylim([np.amin(plt_y),np.amax(plt_y)])
        # ax.set_ylim([np.amin(plt_x),np.amax(plt_x)])
        #SET ASPECT RATIO TO EQUAL TO ENSURE IMAGE HAS SAME ASPECT RATIO AS ACTUAL MESH
        ax.set_aspect('equal')
        #ADD A COLORBAR, VALUE SET USING OUR COLORED POLYGON COLLECTION, [0,1]
        # p.set_clim(0.0, 1.0)
        # p.set_clim(-0.8, 0.0)
        if cb_label==None:
            p.set_clim(0.0, 1.0)
            fig.colorbar(p, label=self.var_to_plot)
        else:
            fig.colorbar(p, label=cb_label)
        fig.savefig(pic_directory+'/'+self.outNameBase+'_sliced_'+self.plane_axis+
                    str(self.plane_coord_name)+'_'+str(frame)+'.png',dpi=500,transparent=True )
        if self.cl_args.debug:
            plt.show()
        else:
            plt.close()
        return

    def plot_slice_forCurvature(self,frame,x,y,z,c,cb_label=None):
        db('Plotting the sliced data')
        # Make pics subdirectory if it doesnt exist
        pic_directory = 'pics'
        if not os.path.isdir(pic_directory):
            db('Making picture directory: '+pic_directory)
            os.makedirs(pic_directory)
        cv_directory = pic_directory+'/cv_images'
        if not os.path.isdir(cv_directory):
            db('Making picture directory: '+cv_directory)
            os.makedirs(cv_directory)
        db('Plotting the slice as specified')
        # Take the average of the 4 corner values for c
        if hasattr(c[0], "__len__"):
            plotc = np.average(c, axis=1)
        else:
            plotc = c
        plt_x, plt_y = self.plt_xy(x,y,z)
        coords = np.asarray([ np.asarray([x_val,y_val]).T for (x_val,y_val) in zip(plt_x,plt_y) ])
        fig, ax = plt.subplots()
        p = PolyCollection(coords, cmap=matplotlib.cm.binary, alpha=1)#,edgecolor='k'
        p.set_array(np.array(plotc) )
        ax.add_collection(p)
        ax.set_xlim([np.amin(plt_x),np.amax(plt_x)])
        ax.set_ylim([np.amin(plt_y),np.amax(plt_y)])
        # ax.set_ylim([np.amin(plt_x),np.amax(plt_x)])
        #SET ASPECT RATIO TO EQUAL TO ENSURE IMAGE HAS SAME ASPECT RATIO AS ACTUAL MESH
        ax.set_aspect('equal')
        # ax.axis('off')
        # ax.set_yticklabels([])
        # ax.set_xticklabels([])
        plt.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)
        #ADD A COLORBAR, VALUE SET USING OUR COLORED POLYGON COLLECTION, [0,1]
        p.set_clim(0.0, 1.0)
        # p.set_clim(-0.8, 0.0)
        # if cb_label==None:
        #     p.set_clim(0.0, 1.0)
        #     fig.colorbar(p, label=self.var_to_plot)
        # else:
        #     fig.colorbar(p, label=cb_label)
        figname = cv_directory+'/'+self.outNameBase+'_cv2_gb_'+self.plane_axis+\
        str(self.plane_coord_name)+'_'+str(frame)+'.png'
        fig.savefig(figname,dpi=500,transparent=True )
        if self.cl_args.debug:
            plt.show()
        else:
            plt.close()
        return figname




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


