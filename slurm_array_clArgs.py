'''Compiles a list for cl value changes in a slurm array job for moose.
    Using it to test a variety of fission rates in a 2D sintering test.

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
import random
import pandas as pd

def parseArgs():
    '''Parse command line arguments

    Returns:
        cl_args
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--writelist','-w', action='store_true',
                                help='Create the .txt file, default off')
    parser.add_argument('--csv','-c', action='store_true',
                                help='Compile the voids/grains csv data, default off')
    parser.add_argument('--sort','-s', action='store_true',
                                help='Present a sorted list of the generated csv files, default off')
    parser.add_argument('--dim','-d',type=int,default=2, choices=[2,3],
                        help='Dimensions for grain size calculation (Default=2)')
    cl_args = parser.parse_args()
    return cl_args

if __name__ == "__main__":
    print("__main__ Start")
    cl_args = parseArgs()

    # Generate the txt file list of cl changes to account for the slurm array
    #  testing of multiple fission rates
    if cl_args.writelist:
        # Precision 2 is the {:.2e}
        fr10_9 = np.linspace(1e-10,1e-9,10)
        fr9_8 = np.linspace(1e-9,1e-8,10)[1:]
        # print(fr10_9)
        # print(fr9_8)
        frcomb = np.concatenate([fr10_9, fr9_8])
        # print(frcomb)
        frlist = []
        for n in frcomb:
            # frlist.append(str(n))
            frlist.append('{:.2e}'.format(n))

        # print(frlist)
        # frlist.sort(key=float)
        # print(frlist)
        outlist = []
        for n in frlist:
            strline = 'f_dot='+ n + ' Outputs/csv/file_base=fr_' + n + '_csv/fr_' + n
            # print(strline)
            outlist.append(strline)

        print(outlist)
        with open('fission_rates_slurmArray.txt','w') as file:
            for line in outlist:
                file.write(f"{line}\n")
        # writer = csv.writer(file, delimiter=',')
        # writer.writerows(outlist)


    # Compile the void and grain VPPs with the normal csv out
    if cl_args.csv:
        cwd = os.getcwd()
        for n in glob.glob('*/', recursive = True):
            # print(n)
            # fr_1.00e-10_csv/ is n, while base is fr_1.00e-10
            base_csv = n.rsplit('_',1)[0]
            # print(base_csv)
            # Names and files
            out_name = n + base_csv + '.csv'
            void_name = n + base_csv + '_voids_*.csv'
            void_files = sorted(glob.glob(void_name))
            grain_name = n + base_csv + '_grain_sizes_*.csv'
            grain_files = sorted(glob.glob(grain_name))
            output_name = base_csv + '_all.csv'
            # Voids
            vdf = pd.DataFrame(columns=['largest','total','bubbles'])
            for files in void_files:
                tempDF = pd.read_csv(files)
                large = tempDF['feature_volumes'].max()
                tot = tempDF['feature_volumes'].sum()
                bubbles = tot - large
                vdf.loc[len(vdf.index)] = [large, tot, bubbles]
            # Grains
            dfg = pd.DataFrame(columns=['avg_vol'])
            # print(grain_files)
            for files in grain_files:
                tempDF = pd.read_csv(files)
                avg_vol = tempDF['feature_volumes'].mean()
                # df.loc[df.shape[0]] = tempDF['feature_volumes'].sum()-maxPore
                dfg.loc[len(dfg.index)] = [avg_vol]
            # Grain Size from area/vol (ESD)
            if cl_args.dim == 2:
                dfg['grain_size'] = 2 * (dfg['avg_vol'] / math.pi)**(1/2)
            elif cl_args.dim == 3:
                dfg['grain_size'] = (6 * dfg['avg_vol'] / math.pi)**(1/3)
            else:
                dfg['grain_size'] = 0
            # Postprocessors and Time
            df = pd.read_csv(out_name)
            # Add void and grain stuff
            df = pd.concat([df,vdf,dfg], axis=1)
            df.to_csv(output_name)

    # Sort the existing compiled csv files (in CWD) into order by fission rate
    # Partially as a reference/example for plotting later
    if cl_args.sort:
        names = []
        for n in glob.glob('*_all.csv'):
            print(n)
            # This works as long as i only add to the name before the fission rate
            fr = n.rsplit('_')[-2]
            print(fr)
            names.append([n,fr])
        name_arr = np.asarray(names)
        print(name_arr)
        name_arr = name_arr[name_arr[:, 1].astype(float).argsort()]
        print(name_arr)

