'''Adds an external void phase to extend a Dream3D.txt file for use in MOOSE

Returns:
    out.txt file for ebsd reader in Moose
'''
from PIL import Image
import glob
import os
import cv2
import argparse
import re
import numpy as np
from itertools import product
import csv
import math
import time


def parseArgs():
    '''Parse command line arguments

    Returns:
        cl_args
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir','-d',choices=['x','y','z'],default='x',
                                help='OUTDATED: Only does x direction.  Coordinate direction (x,y,z) to add the phi volume. Defaults to +x')
    parser.add_argument('--planes','-n',type=int, default=20,
                                help='Number of element planes of phi to add. Default = 20')
    parser.add_argument('--input','-i',type=str,
                                help='Name of Dream3D txt file to glob.glob(*__*.txt) find and read.')
    parser.add_argument('--out','-o',type=str,
                                help='Name of output txt file. If not specified will use [inputName]_plusVoid.txt')
    parser.add_argument('--pores','-p',type=int, default=5,
                                help='Number of pores to add. Default = 5')
    parser.add_argument('--volume','-v',type=int, default=5,
                                help='Volume percentage (1-100) to make the pores. Default = 5')
    parser.add_argument('--scale','-s',type=int, default=1,
                                help='Multiplier to apply to all dimensions to scale the domain (default=1)')
    parser.add_argument('--spacing','-x',type=float, default=1,
                                help='Fraction of the average pore radius to include as the minimum seperation between pores (default=1)')
    parser.add_argument('--phase3',action='store_true',
                                help='Make internal porosity a third phase, seperate from external.')
    cl_args = parser.parse_args()
    return cl_args

class header_vals:
    def __init__(self, x, y, z, feature_id):
        self.dim = 3
        self.feature_id_max = int(max(feature_id))
        self.xyz_params(x,y,z)

    def xyz_params(self,x,y,z):
        '''Calculate mesh min max and element size in all 3 directions

        Args:
            x: np array of x data
            y: np array of y data
            z: np array of z data
        '''
        # x coordinates
        self.xu = np.unique(x)
        if len(self.xu) == 1:
            self.dim = self.dim - 1
            self.dx = 0.0
            self.xmin = 0.0
            self.xmax = 0.0
            self.ctr_xmax = 0.0
        else:
            self.dx = self.xu[1] - self.xu[0]
            self.xmin = min(self.xu) - (0.5*self.dx)
            self.xmax = max(self.xu) + (0.5*self.dx)
            self.ctr_xmax = max(self.xu)
        # y coordinates
        self.yu = np.unique(y)
        if len(self.yu) == 1:
            self.dim = self.dim - 1
            self.dy = 0.0
            self.ymin = 0.0
            self.ymax = 0.0
            self.ctr_ymax = 0.0
        else:
            self.dy = self.yu[1] - self.yu[0]
            self.ymin = min(self.yu) - (0.5*self.dy)
            self.ymax = max(self.yu) + (0.5*self.dy)
            self.ctr_ymax = max(self.yu)
        # z coordinates
        self.zu = np.unique(z)
        if len(self.zu) == 1:
            self.dim = self.dim - 1
            self.dz = 0.0
            self.zmin = 0.0
            self.zmax = 0.0
            self.ctr_zmax = 0.0
        else:
            self.dz = self.zu[1] - self.zu[0]
            self.zmin = min(self.zu) - (0.5*self.dz)
            self.zmax = max(self.zu) + (0.5*self.dz)
            self.ctr_zmax = max(self.zu)


def vol_per_sphere(volume_solid,dims):
    min_vol = 1/(1.5*cl_args.pores) #0.055

    a = np.random.rand(cl_args.pores)
    a = (a/a.sum()*(1-min_vol*cl_args.pores))
    weights = a+min_vol
    volumes = weights*volume_solid*cl_args.volume/100
    r = []
    if dims == 2:
        r = (volumes / math.pi)**(1/2)
    elif dims == 3:
        r = (3 * volumes / (4 * math.pi))**(1/3)
    return r

def distance(pt_list1,pt_list2):
    dist = math.sqrt((pt_list1[0]-pt_list2[0])**2 +
                     (pt_list1[1]-pt_list2[1])**2 +
                     (pt_list1[2]-pt_list2[2])**2)
    return dist

def generate_centers(radii,mesh,min_sep,max_its=10000):
    coords = []
    it = 0
    for n in range(cl_args.pores):
        # print(n)
        loop = True
        while loop == True:
            # Random location
            ctr = [mesh.xmax, mesh.ymax, mesh.zmax] * np.random.rand(1,3)
            ctr = ctr[0] #remove the [[x y z]] so its just [x y z]
            # check if the pores overlap each other
            if n != 0:
                for i in range(0, n):
                    dist = distance(ctr,coords[i])
                    if dist < (radii[n] + radii[i] + min_sep):
                        if it < max_its:
                            it = it + 1
                            loop = True
                            break
                        else:
                            print("ERROR: MAX ITERATIONS REACHED - there will be pore overlap")
                    else:
                        loop = False
            else:
                loop = False
            # Check if the pores overlap a boundary
            # print(ctr)
            # print(ctr[0], radii[n])
            if (ctr[0] - radii[n] - min_sep < 0) or (ctr[0] + radii[n] + min_sep > mesh.xmax):
                # print('X Issue')
                it = it + 1
                loop = True
            if (ctr[1] - radii[n] - min_sep < 0) or (ctr[1] + radii[n] + min_sep > mesh.ymax):
                # print('Y Issue')
                it = it + 1
                loop = True
            if dim == 3:
                if (ctr[2] - radii[n] - min_sep < 0) or (ctr[2] + radii[n] + min_sep > mesh.zmax):
                    # print('Z Issue')
                    it = it + 1
                    loop = True
        # If it worked, save it
        coords.append(ctr)
    # print(it)
    return np.asarray(coords)



if __name__ == "__main__":
    print("__main__ Start")
    cl_args = parseArgs()

    # Find the input .txt file
    txt_names = []
    if cl_args.input is None:
        searchName = '*.txt'
    elif '.txt' in cl_args.input:
        searchName = '*'+cl_args.input
    elif '/' in cl_args.input and '.txt' not in cl_args.input:
        searchName = cl_args.input + '*.txt'
    else:
        searchName = '*'+cl_args.input + '*.txt'
    # print(searchName)
    for file in glob.glob(searchName):
        txt_names.append(file)
    txt_file = txt_names[0]

    # Read the txt file to a list
    with open(txt_file) as f:
        full_data = [r.split() for r in f]

    header = []
    body = []
    for row in full_data:
        if '#' in row[0]:
            header.append(row)#[1:]
        else:
            body.append(row)
    # ['phi1', 'PHI', 'phi2', 'x', 'y', 'z', 'FeatureId', 'PhaseId', 'Symmetry']
    data = np.asarray(body,dtype=float)
    # Measure mesh coordinates:
    mesh = header_vals(data[:,3],data[:,4],data[:,5],data[:,6])
    # print(vars(mesh))
    # print(" ")

    # Adding Phi
    # Define new coordinates to add
    if 'x' in cl_args.dir:
        new_x = np.asarray([mesh.ctr_xmax + (n+1)*mesh.dx for n in range(cl_args.planes)])
        new_y = mesh.yu
        new_z = mesh.zu
    new_coords = list(product(new_x,new_y,new_z))

    phi_txt = []
    f_num = mesh.feature_id_max + 1
    for set in new_coords:
        phi_txt.append(['0.0','0.0','0.0',str(cl_args.scale*set[0]),str(cl_args.scale*set[1]),str(cl_args.scale*set[2]),str(f_num),'2','43'])

    # Adjust the header values to include phi addition
    new_xmax = mesh.xmax + (cl_args.planes)*mesh.dx
    for row in header:
        if len(row)>1:
            # Assuming positive x direction for extra planes
            if 'X_MAX' in row[1]:
                row[2] = str(new_xmax)
            if 'X_DIM' in row[1]:
                row[2] = str(float(row[2]) + cl_args.planes)
            if 'STEP' in row[1] or 'MAX' in row[1]:
                row[2] = str( cl_args.scale * float(row[2]))


    # Porosity Internal
    if cl_args.pores > 0:
        # Calculate the solid volume
        dim = 3
        if mesh.zmax == 0.0:
            dim = 2
        volume_solid = mesh.xmax * mesh.ymax
        if dim == 3:
            volume_solid = volume_solid * mesh.zmax
        # print(volume_solid)
        # Generate Pore centers and radii
        # pore_ctrs = np.random.rand(cl_args.pores,3)
        rads = vol_per_sphere(volume_solid,dim)
        min_sep = np.average(rads) * cl_args.spacing
        pore_ctrs = generate_centers(rads,mesh,min_sep)
        print('Centers: ')
        print(pore_ctrs)
        print("Radii: ")
        print(rads)
        # Print for moose input:
        print('Or for MOOSE IC')
        print('R: ',np.round(cl_args.scale*rads,2))
        print(np.round(cl_args.scale*pore_ctrs.T,2))
        # for n in range(len(rads)):


        # Find and replace the gridpoints in the pores
        pore_phase = int(2)
        if cl_args.phase3:
            pore_phase = int(3)
        for pore in range(cl_args.pores):
            ctr = pore_ctrs[pore]
            loop_rad = rads[pore]
            pore_id = mesh.feature_id_max + 2 + pore
            for row in data:
                if distance([row[3],row[4],row[5]], ctr) <= loop_rad:
                    row[6] = pore_id
                    row[7] = pore_phase

    # Rebuild the data as a list so i can make the featureID and phaseID whole numbers
    body_list = []
    for row in data:
        # ['phi1', 'PHI', 'phi2', 'x', 'y', 'z', 'FeatureId', 'PhaseId', 'Symmetry']
        body_list.append([str(row[0]),str(row[1]),str(row[2]),
                          str(cl_args.scale*row[3]),str(cl_args.scale*row[4]),str(cl_args.scale*row[5]),
                          str(row[6]).split('.')[0],str(row[7]).split('.')[0],str(row[8]).split('.')[0]])

    # Output Naming
    if cl_args.out is None:
        out_name = txt_file.rsplit('.',1)[0] + '_plusVoid.txt'
    elif '.txt' in cl_args.out:
        out_name = cl_args.out
    else:
        out_name = cl_args.out + '.txt'

    # Write all the data to a new txt file
    with open(out_name,'w') as file:
        writer = csv.writer(file, delimiter=' ')
        # writer.writerows(full_data)
        writer.writerows(header)
        writer.writerows(body_list)
        writer.writerows(phi_txt)


    print('Done')





    # print(header)
    # print(body[:3])
    # print(data[:3])
    # if cl_args.out == None:
    #     cl_args.out = img_names[0].rsplit('.',1)[0]
    # print('Output named: '+cl_args.out)


# quit()
# quit()
