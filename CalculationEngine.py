import os
import glob
import numpy as np
import json
import argparse

class CalculationEngine:
    def __init__(self): #,files
        # if os.path.exists("tiger_meta.json"):
        self.parse_cl_flags()

        print(self.cl_args)
        self.get_meta(self.cl_args.new_meta)
        # self.get_file_names_in_cwd()

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
        else:
            name_unq = np.insert(name_unq, 0, e_unq)
        self.file_names = name_unq
        return self.file_names

    def parse_cl_flags(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--new-meta', action='store_true')
        parser.add_argument('--cpu','-n', default=1,type=int)
        self.cl_args = parser.parse_args()
        return self.cl_args

    def get_meta(self,rewrite_flag):
        if os.path.exists("tiger_meta.json") and not rewrite_flag:
            print("EXISTS")
        elif rewrite_flag or not os.path.exists("test.json"):
            if os.path.exists("test.json"):
                print("deleting old metadata and writing new")
            else:
                print("Writing new metadata")
        else:
            print("problem")
