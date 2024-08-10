#!/usr/bin/env python3
import os
import glob
import pandas as pd
from matplotlib import pyplot as plt
import argparse
import math

# # For sorting to deal with no leading zeros
# def natural_sort_key(s, _nsre=re.compile('([0-9]+)')):
#     '''Sorts the file names naturally to account for lack of leading zeros
#     use this function in listname.sort(key=natural_sort_key)

#     Args:
#         s: files/iterator
#         _nsre: _description_. Defaults to re.compile('([0-9]+)').

#     Returns:
#         Sorted data
#     '''
#     return [int(text) if text.isdigit() else text.lower()
#             for text in _nsre.split(s)]
def parseArgs():
    '''Parse command line arguments

    Returns:
        cl_args
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--force','-f',action='store_true',
                        help='Overwrite existing combined_csv if there is one')
    parser.add_argument('--grain','-g',action='store_true',
                        help='Calculate grain size from *grain__sizes*.csv')
    parser.add_argument('--dim','-d',type=int,default=2, choices=[2,3],
                        help='Dimensions for grain size calculation (Default=2)')
    parser.add_argument('--gname',type=str, default='grain_sizes',
                        help='Grain size VPP csv filename to *glob*. Default = grain_sizes')
    parser.add_argument('--all','-a',action='store_true',
                        help='Append a list of columns/PPs to the cut*.csv file from the *_out.csv')
    parser.add_argument('--plot','-p',action='store_true',
                        help='Include the plots at the end?')
    parser.add_argument('--input','-i',type=str,default=None,
                        help='Optional specify input name to *name* glob.glob search/use')
    cl_args = parser.parse_args()
    return cl_args



if __name__ == "__main__":
    print("__main__ Start")
    cl_args = parseArgs()

    bypass = False
    if glob.glob('cut_voids_*.csv'):
        bypass = True
    if cl_args.force == True:
        bypass = False

    if bypass == False:
        cwd = os.getcwd()
        if cl_args.input:
            inputsearch = "*" + cl_args.input + "*.i"
        else:
            inputsearch = "*.i"
        for file in glob.glob(inputsearch):
            inputName = os.path.splitext(file)[0]
        print("Input File is : " + inputName)
        # VOIDS
        postprocessorName = 'voids'

        outputName = inputName + "_out_" + postprocessorName + "_*.csv"

        fileDirectory = cwd + "/" + outputName
        print("The output being concatenated is : " + fileDirectory)

        csvfiles = sorted(glob.glob(fileDirectory))

        df = pd.DataFrame(columns=['largest','total','bubbles'])
        for files in csvfiles:
            tempDF = pd.read_csv(files)
            large = tempDF['feature_volumes'].max()
            tot = tempDF['feature_volumes'].sum()
            bubbles = tot - large
            # df.loc[df.shape[0]] = tempDF['feature_volumes'].sum()-maxPore
            df.loc[len(df.index)] = [large, tot, bubbles]

        # GRAIN SIZE(S)
        if cl_args.grain:
            dfg = pd.DataFrame(columns=['avg_vol'])
            grain_files = sorted(glob.glob(cwd + '/*' + cl_args.gname + "*_*.csv"))
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
            # Add to pore df
            df['grain_size'] = dfg['grain_size']
            df['avg_grain_vol'] = dfg['avg_vol']



        combinedName = "combined_" + postprocessorName + "_" + inputName + ".csv"
        df.to_csv(combinedName)
        print("The combined csv file is : " + combinedName)

        # Add time data
        cutdf = df#.loc[1].reset_index(drop=True)
        outdf = pd.read_csv(inputName + "_out.csv")
        main_pp_list = ['time', 'void_tracker', 'grain_tracker', 'runtime']
        for n in main_pp_list:
            if n in outdf.columns:
                cutdf[n] = outdf[n]
        # cutdf['time'] = outdf['time']
        # cutdf['void_tracker'] = outdf['void_tracker']
        # cutdf['grain_tracker'] = outdf['grain_tracker']
        # cutdf['runtime'] = outdf['runtime']
        all_pp_list = ['total_phi','total_rhoi','total_rhov']
        if cl_args.all:
            for n in all_pp_list:
                if n in outdf.columns:
                    cutdf[n] = outdf[n]

        cutName = "cut_" + postprocessorName + "_" + inputName + ".csv"
        cutdf.to_csv(cutName)
        print("The single cut down csv file is : " + cutName)
    else:
        for file in glob.glob('cut_*.csv'):
            cutdf = pd.read_csv(file)
    # Processing/Plotting
    # Extra Portion - combine the dfs into a new csv and graph to view

    # print(cutdf)
    if cl_args.plot:
        plt.figure(1)
        plt.scatter(cutdf.time,cutdf.largest)
        plt.xlabel("Time")
        plt.ylabel("External Void Volume")

        plt.figure(2)
        plt.scatter(cutdf.time,cutdf.total)
        plt.xlabel("Time")
        plt.ylabel("Total Void Volume")

        plt.figure(3)
        plt.scatter(cutdf.time,cutdf.bubbles)
        plt.xlabel("Time")
        plt.ylabel("Internal Void Volume")

        plt.show()

