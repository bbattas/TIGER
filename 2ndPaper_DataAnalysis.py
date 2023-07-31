from MultiExodusReader import MultiExodusReader
import multiprocessing as mp
from CalculationsV2 import CalculationsV2

import os
import sys
import json
import time
import logging
pt = logging.warning
verb = logging.info
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import glob



# ███████╗██╗   ██╗███╗   ██╗ ██████╗████████╗██╗ ██████╗ ███╗   ██╗███████╗
# ██╔════╝██║   ██║████╗  ██║██╔════╝╚══██╔══╝██║██╔═══██╗████╗  ██║██╔════╝
# █████╗  ██║   ██║██╔██╗ ██║██║        ██║   ██║██║   ██║██╔██╗ ██║███████╗
# ██╔══╝  ██║   ██║██║╚██╗██║██║        ██║   ██║██║   ██║██║╚██╗██║╚════██║
# ██║     ╚██████╔╝██║ ╚████║╚██████╗   ██║   ██║╚██████╔╝██║ ╚████║███████║
# ╚═╝      ╚═════╝ ╚═╝  ╚═══╝ ╚═════╝   ╚═╝   ╚═╝ ╚═════╝ ╚═╝  ╚═══╝╚══════╝

def nondimen_time(df,surf_energy,Ds,T):
    '''Create a nondimensionalized time ['n_time'] based on the process used in the paper
        Currently, for unit cancelling we assume delta has nm units but is still just 1
    Args:
        df: Dataframe to scale ['time']
        surf_energy: Surface energy (our sigma_s) in eV/nm^2
        Ds: deltaDs or just our surface diffusivity in nm^2/s
        T: Temperature in K
    '''
    omega = 0.04092 #nm^3
    k_B = 8.617343e-5 #eV/K
    r_0 = 100 #nm
    scaling = (surf_energy * omega * Ds) / (k_B * T * (r_0 ** 4))
    df['n_time'] = scaling * df['time']
    return

def cr_r0(df):
    '''Calculate the effective contact radius, c*/r_0, as ['cr_r0']

    Args:
        df: Dataframe to use (has ['grain_area'])
    '''
    r_0 = 100 #nm
    df['cr_eff'] = np.sqrt(df.grain_area.div(math.pi))
    df['cr_r0'] = df.cr_eff / r_0
    return

def read_and_math_csv(csvname,surf_energy,Ds,T):
    df = pd.read_csv(csvname)
    df.sort_values(by="time").reset_index(drop=True, inplace=True)
    cr_r0(df)
    nondimen_time(df,surf_energy,Ds,T)
    return df


# ███╗   ███╗ █████╗ ██╗███╗   ██╗
# ████╗ ████║██╔══██╗██║████╗  ██║
# ██╔████╔██║███████║██║██╔██╗ ██║
# ██║╚██╔╝██║██╔══██║██║██║╚██╗██║
# ██║ ╚═╝ ██║██║  ██║██║██║ ╚████║
# ╚═╝     ╚═╝╚═╝  ╚═╝╚═╝╚═╝  ╚═══╝


if __name__ == "__main__":
    print("__main__ Start")
    short = True

    if short:
        # Using Pandas for shit
        filename = '*_calc_data.csv'
        for file in glob.glob(filename):
            csv_file = file
        # df = pd.read_csv(csv_file)
        # # csv_header = ['time', 'grain_area', 'tot_mesh_area','curvature']
        # df.sort_values(by="time").reset_index(drop=True, inplace=True)
        # cr_r0(df)
        # nondimen_time(df,9.86,1e11,1600)
        # print(df)
        df = read_and_math_csv(csv_file,9.86,1e11,1600)
        df['kappa_star'] = - df.curvature / df.grain_area
        df['kappa_star2'] = - df.delta_normalized_curvature / df.grain_area

        plt.figure(1)
        # plt.plot(df.cr_r0,-df.curvature,label='1')
        # plt.plot(df.cr_r0,-df.delta_normalized_curvature,label='2')
        plt.plot(df.cr_r0,df.kappa_star,label='3')
        plt.plot(df.cr_r0,df.kappa_star2,label='4')
        plt.xlabel('c*/r0')
        plt.ylabel('Curvature* x Ac')
        # plt.plot(df.n_time,df.cr_r0)
        # plt.xlabel('t*')
        # plt.ylabel('c*/r0')
        # plt.loglog()
        # plt.ylim([0.2,1.2])
        plt.legend()
        plt.show()

        sys.exit()



    print('__main__ DONE')
#     quit()
# quit()
