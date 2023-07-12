'''Generate .json list of unique filenames to be rsynced
assumes that there are a ton of nemesis files, ie very
many cpus

Returns:
    json list of unique filenames
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
    # parser.add_argument('--inl',action='store_true')
    parser.add_argument('--source',type=str)
    # parser.add_argument('--dest',type=str)
    # parser.add_argument('--new-meta', action='store_true')
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

    # if cl_args.dest == None:
    #     pt("Setting destination to cwd: " + cwd)
    #     dest = cwd
    # else:
    #     pt("Setting destination to : " + cl_args.dest)
    #     dest = cl_args.dest
    #     if dest[-1] == "/":
    #         dest = dest[:-1]
    #
    # dest_tree = [dest + "/" + loc for loc in dest_tree]
    #
    print(cl_args.source)
    if cl_args.source == None:
        pt("Setting source to cwd: " + cwd)
        source = cwd
    else:
        pt("Setting destination to : " + cl_args.source)
        source = cl_args.source
        if source[-1] == "/":
            source = source[:-1]

    # os.chdir(cwd + "/" + dir)

    os.chdir(source)
    pt("In source directory: " + source)
    # Generate list of all to transfer
    verb("Generating list of files to transfer")
    tree = []

    for dir in glob.glob("*/"):
        if not "checkpoint" in dir.lower() or not "cp" in dir.lower():
            verb("In " + dir)
            tree.append(dir)
            for subdir in glob.glob(dir+"*/"):
                if not "checkpoint" in subdir.lower() or not "cp" in subdir.lower():
                    verb("   " + subdir)
                    tree.append(subdir)
                    for ssubdir in glob.glob(subdir+"*/"):
                        if not "checkpoint" in ssubdir.lower() or not "cp" in ssubdir.lower():
                            verb(" " + ssubdir)
                            tree.append(ssubdir)
                        else:
                                verb(" X " + ssubdir)
                else:
                    verb(" X " + subdir)


    pt("List of Directories: ")
    pt(tree)
    files = []
    dest_tree = []

    for dir in tree:
        if not dir.startswith('.'):
            os.chdir(source + "/" + dir)
            verb("In directory: " + dir)
            e_name = [x.rsplit('.',1)[0]+"*" for x in glob.glob("*.e.*")]#"*_out.e.*"#glob.glob("*_out.e.*") #first step
            s_names = [x.rsplit('.',2)[0]+"*" for x in glob.glob("*.e-s*")] #after first step#x[:-8]
            e_unq = np.unique(e_name).tolist()
            s_unq = np.unique(s_names).tolist()
            # temp_files =
            files.extend([cwd + "/" + dir + file for file in e_unq + s_unq])
            num_files = len(e_unq) + len(s_unq)
            if glob.glob(cwd + "/" + dir + "*.csv"):
                files.extend([cwd + "/" + dir + "*.csv"])
                num_files = num_files + 1
            if glob.glob(cwd + "/" + dir + "*.i"):
                files.extend([cwd + "/" + dir + "*.i"])
                num_files = num_files + 1
            verb(str(num_files) + " files in the directory")
            dest_tree.extend([dir] * num_files)
            # files.append(s_unq)
    pt("File list finished")
    verb("Files: ")
    verb(files)
    verb (" ")
    # verb("Destination tree: ")
    # verb(dest_tree)

    os.chdir(source)
    dict = {}
    dict['files'] = files
    dict['dest_tree'] = dest_tree
    with open('rsync_list.json', 'w') as fp:
        json.dump(dict, fp)

    print('Done')
