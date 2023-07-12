'''Uses the json list of unique filenames from
file_transfer_listMake.py to rsync massively parallel
nemesis outputs in smaller more managable chunks

Returns:
    rsync command executed in terminal
'''
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


def parseArgs():
    '''Parse command line arguments

    Returns:
        cl_args
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose','-v',action='store_true')
    parser.add_argument('--inl',action='store_true')
    parser.add_argument('--source',type=str)
    parser.add_argument('--dest',type=str)
    parser.add_argument('--resume',type=str)
    cl_args = parser.parse_args()
    return cl_args


if __name__ == "__main__":
    print("__main__ Start")
    cl_args = parseArgs()

    pt = logging.warning
    verb = logging.info

    if cl_args.verbose == True:
        logging.basicConfig(level=logging.INFO,format='%(message)s')
    elif cl_args.verbose == False:
        logging.basicConfig(level=logging.WARNING,format='%(message)s')
    verb('Verbose Logging Enabled')
    verb(cl_args)


    cwd = os.getcwd()

    if cl_args.dest == None:
        pt("Setting destination to cwd: " + cwd)
        dest = cwd
        if dest[-1] == "/":
            dest = dest[:-1]
    else:
        pt("Setting destination to : " + cl_args.dest)
        dest = cl_args.dest
        if dest[-1] == "/":
            dest = dest[:-1]

    # dest_tree = [dest + "/" + loc for loc in dest_tree]

    if cl_args.source == None:
        pt("Need json file source location!")
        source = input("Input directory to transfer from: ")
        if source[-1] == "/":
            source = source[:-1]
    else:
        pt("Setting source to : " + cl_args.source)
        source = cl_args.source
        if source[-1] == "/":
            source = source[:-1]

    if cl_args.inl == True:
        verb("Adding ssh command to source")
        source = "battbran@hpclogin:" + source
        pt("Copying json to local")
        os.system("rsync" + " -a " + source +"/rsync_list.json " + dest)
        with open(dest+'/rsync_list.json') as json_file:
            dict = json.load(json_file)
    else:
        pt("Reading json list of files")
        with open(source+'/rsync_list.json') as json_file:
            dict = json.load(json_file)

    pt("Making destination directories")
    verb(np.unique(dict['dest_tree']))
    for dir in np.unique(dict['dest_tree']):
        if not os.path.isdir(dest + "/" + dir):
            os.makedirs(dest + "/" + dir)

    if cl_args.resume == None:
        pt("Starting transfer from the beginning")
        skip = 0
    else:
        pt("Starting transfer from: " + cl_args.resume)
        skip = 1


    for s,d in zip(dict['files'],dict['dest_tree']):
        # print(s, dest + "/" + d)
        end = dest + "/" + d
        if s.split("/", 2)[1] == "ibmstor":
            s = "/" + s.split("/", 2)[2]
        if cl_args.inl == True:
            command = "rsync" + " -av battbran@hpclogin:" + s + " " + end
        else:
            command = "rsync" + " -av " + s + " " + end
        verb(command)

        if skip == 1:
            if cl_args.resume in command:
                skip = 0
                pt("Starting the transfer with: " + command + " based on resume flag of: " + cl_args.resume)
            else:
                verb("Skipping: " + command)
        if skip == 0:
            verb(command)
            os.system(command)
    print('DONE!')
    # quit()
    # subprocess.call(["rsync", "-av", s, end])


# make the source battbran@hpclogin:/scratch/battbran/ whatever
