#!/usr/bin/env python3
import os
import glob
import pandas as pd
from matplotlib import pyplot as plt
import argparse

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
    cl_args = parser.parse_args()
    return cl_args



if __name__ == "__main__":
    print("__main__ Start")
    cl_args = parseArgs()

    bypass = False
    if glob.glob('combined_voids_*.csv'):
        bypass = True
    if cl_args.force == True:
        bypass = False

    if bypass == False:
        cwd = os.getcwd()

        for file in glob.glob("*.i"):
            inputName = os.path.splitext(file)[0]
        print("Input File is : " + inputName)
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

        combinedName = "combined_" + postprocessorName + "_" + inputName + ".csv"
        df.to_csv(combinedName)
        print("The combined csv file is : " + combinedName)

        # Add time data
        cutdf = df#.loc[1].reset_index(drop=True)
        outdf = pd.read_csv(inputName + "_out.csv")
        cutdf['time'] = outdf['time']
        cutdf['void_tracker'] = outdf['void_tracker']

        cutName = "cut_" + postprocessorName + "_" + inputName + ".csv"
        cutdf.to_csv(cutName)
        print("The single cut down csv file is : " + cutName)
    else:
        for file in glob.glob('cut_*.csv'):
            cutdf = pd.read_csv(file)
    # Processing/Plotting
    # Extra Portion - combine the dfs into a new csv and graph to view

    # print(cutdf)

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
