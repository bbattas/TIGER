'''Adds an external void phase to extend a Dream3D.txt file for use in MOOSE

Returns:
    out.gif or out.avi
'''
from PIL import Image
import glob
import os
import cv2
import argparse
import re
import numpy as np
from itertools import product


def parseArgs():
    '''Parse command line arguments

    Returns:
        cl_args
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir','-d',type=str,
                                help='Coordinate direction (x,y,z) to add the phi volume.')
    parser.add_argument('--planes','-n',type=str,
                                help='Number of element planes of phi to add.')
    parser.add_argument('--input','-i',type=str,
                                help='Name of Dream3D txt file to glob.glob(*__*.txt) find and read.')
    parser.add_argument('--out','-o',type=str,
                                help='Name of output txt file. If not specified will use [inputName]_plusVoid.txt')
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
            self.ymay = 0.0
            self.ctr_ymax = 0.0
        else:
            self.dy = self.yu[1] - self.yu[0]
            self.ymin = min(self.yu) - (0.5*self.dy)
            self.ymay = max(self.yu) + (0.5*self.dy)
            self.ctr_ymax = max(self.yu)
        # z coordinates
        self.zu = np.unique(z)
        if len(self.zu) == 1:
            self.dim = self.dim - 1
            self.dz = 0.0
            self.zmin = 0.0
            self.zmaz = 0.0
            self.ctr_zmax = 0.0
        else:
            self.dz = self.zu[1] - self.zu[0]
            self.zmin = min(self.zu) - (0.5*self.dz)
            self.zmaz = max(self.zu) + (0.5*self.dz)
            self.ctr_zmax = max(self.zu)



if __name__ == "__main__":
    print("__main__ Start")
    cl_args = parseArgs()

    # Find the input .txt file
    txt_names = []
    if cl_args.input is None:
        searchName = '*.txt'
    elif '.txt' in cl_args.input:
        searchName = '*'+cl_args.input
    else:
        searchName = '*'+cl_args.input + '*.txt'
    for file in glob.glob(searchName):
        txt_names.append(file)
    txt_file = txt_names[0]

    with open(txt_file) as f:
        # # Iterate through the file until the table starts
        # for line in f:
        #     if line.startswith('#'):
        #         break
        # Read the rest of the data, using spaces to split.
        full_data = [r.split() for r in f]
    header = []
    body = []
    for row in full_data:
        if '#' in row[0]:
            header.append(row[1:])
        else:
            body.append(row)
    # ['phi1', 'PHI', 'phi2', 'x', 'y', 'z', 'FeatureId', 'PhaseId', 'Symmetry']
    data = np.asarray(body,dtype=float)
    # Important columns
    # x = data[:,3]
    # y = data[:,4]
    # z = data[:,5]
    # feat = data[:,6]
    # print('X Unique')
    # print(np.unique(x))
    # print('y Unique')
    # print(np.unique(y))
    # print('z Unique')
    # print(np.unique(z))
    # print('FID Unique')
    # print(np.unique(feat))
    # print(" ")

    # Measure mesh coordinates:
    mesh = header_vals(data[:,3],data[:,4],data[:,5],data[:,6])
    print(vars(mesh))
    print(" ")


    # print(header)
    # print(body[:3])
    # print(data[:3])
    # if cl_args.out == None:
    #     cl_args.out = img_names[0].rsplit('.',1)[0]
    # print('Output named: '+cl_args.out)


# quit()
# quit()
