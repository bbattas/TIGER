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



# ███████╗██╗   ██╗███╗   ██╗ ██████╗████████╗██╗ ██████╗ ███╗   ██╗███████╗
# ██╔════╝██║   ██║████╗  ██║██╔════╝╚══██╔══╝██║██╔═══██╗████╗  ██║██╔════╝
# █████╗  ██║   ██║██╔██╗ ██║██║        ██║   ██║██║   ██║██╔██╗ ██║███████╗
# ██╔══╝  ██║   ██║██║╚██╗██║██║        ██║   ██║██║   ██║██║╚██╗██║╚════██║
# ██║     ╚██████╔╝██║ ╚████║╚██████╗   ██║   ██║╚██████╔╝██║ ╚████║███████║
# ╚═╝      ╚═════╝ ╚═╝  ╚═══╝ ╚═════╝   ╚═╝   ╚═╝ ╚═════╝ ╚═╝  ╚═══╝╚══════╝

# plot as specified in CalculationsV2 but in parallel
# needs calc and MF named as such
def parallelSlicePlot(i,idx_frame):
    para_t0 = time.perf_counter()
    x,y,z,c = MF.get_data_at_time(calc.var_to_plot,calc.times[idx_frame],True)
    nx, ny, nz, nc = calc.plane_slice(x,y,z,c)
    calc.plot_slice(i,nx,ny,nz,nc)
    verb('  Finished plotting file '+str(i)+': '+str(round(time.perf_counter()-para_t0,2))+'s')
    return

# Function to let me run calculation shit in parallel
def do_calculations(i,idx_frame):
    para_t0 = time.perf_counter()
    gr_area = calc.c_area_in_slice(*MF.get_data_at_time(calc.var_to_plot,calc.times[idx_frame],True))
    verb('  Finished calculating file '+str(i)+': '+str(round(time.perf_counter()-para_t0,2))+'s')
    return calc.times[idx_frame], gr_area



# ███╗   ███╗ █████╗ ██╗███╗   ██╗
# ████╗ ████║██╔══██╗██║████╗  ██║
# ██╔████╔██║███████║██║██╔██╗ ██║
# ██║╚██╔╝██║██╔══██║██║██║╚██╗██║
# ██║ ╚═╝ ██║██║  ██║██║██║ ╚████║
# ╚═╝     ╚═╝╚═╝  ╚═╝╚═╝╚═╝  ╚═══╝


if __name__ == "__main__":
    print("__main__ Start")
    calc = CalculationsV2()

    # Single nemesis file set (no adaptive mesh)
    # if len(calc.file_names) == 1:
    if not calc.adaptive_mesh:
        verb('Reading Exodus/Nemesis all at once, no adaptive mesh')
        MF = MultiExodusReader(calc.file_names[0])
    else:
        verb('Not set anything up to run with adaptive mesh yet')
        MF = None



    # Need definitions before this for some reason
    # Enable cpu pool if parallel enabled
    if calc.parallel:
        verb('Starting pool of '+str(calc.cpu)+' CPUs')
        cpu_pool = mp.Pool(calc.cpu)
        verb(cpu_pool)



    # Plot the current plane or whatever else later
    if calc.plot:
        print('Plotting')
        for i,idx in enumerate(calc.idx_frames):
            if calc.parallel:
                cpu_pool.apply_async(parallelSlicePlot,args = (i, idx ))
            else:
                parallelSlicePlot(i,idx)
        print('Done Plotting')


    # run calculations and do math
    if calc.calcs:
        print('Doing Calculations')
        para_results = []
        for i,idx in enumerate(calc.idx_frames):
            if calc.parallel:
                para_results.append(cpu_pool.apply_async(do_calculations,args = (i, idx )))
            else:
                 para_results.append(do_calculations(i,idx))

        if calc.parallel:
            para_results = [r.get() for r in para_results]

        print('Calculations Done')
        # output using np
        # # convert data to array
        # out_data = np.asarray(para_results)
        # # sort data by time
        # out_data = out_data[out_data[:, 0].astype(float).argsort()]
        # saveloc = calc.outNameBase + '_calc_data.csv'
        # csv_header = ['time', 'grain_area']
        # np.savetxt(saveloc, np.asarray(out_data), delimiter=',', header=','.join(csv_header), comments='')

        # Using Pandas for shit
        saveloc = calc.outNameBase + '_calc_data.csv'
        csv_header = ['time', 'grain_area']
        df = pd.DataFrame(para_results, columns = csv_header)
        df.sort_values(by="time").reset_index(drop=True, inplace=True)
        df['cr_eff'] = np.sqrt(df.grain_area.div(math.pi))
        df.to_csv(saveloc)


    if calc.parallel:
        cpu_pool.close()
        cpu_pool.join()

    verb('__main__ DONE')
    quit()
quit()
