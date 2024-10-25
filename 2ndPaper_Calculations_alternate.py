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
def do_calculations(i,idx_frame,all_op=False):
    para_t0 = time.perf_counter()
    if not all_op:
        x,y,z,c = MF.get_data_at_time(calc.var_to_plot,calc.times[idx_frame],True)
        op_area, tot_mesh_area = calc.c_area_in_slice(x,y,z,c,calc.var_to_plot)
        sum_del_cv, full_delta, full_cv = calc.MER_curvature_calcs(x,y,z,c)
        verb('  Finished calculating file '+str(i)+': '+str(round(time.perf_counter()-para_t0,2))+'s')
        return calc.times[idx_frame], op_area, tot_mesh_area, sum_del_cv
    else:
        allgrs = ['gr0','gr1']
        cgb = 0
        # Determine the number of grains
        numGrs = 0
        if 'gr2' not in MF.exodus_readers[0].nodal_var_names:
            numGrs = 2
        elif 'gr3' not in MF.exodus_readers[0].nodal_var_names:
            numGrs = 3
        elif 'gr3' in MF.exodus_readers[0].nodal_var_names:
            numGrs = 4
        else:
            raise ValueError('Number of grains not 2 3 or 4?')
        for grop in allgrs:
            x,y,z,c = MF.get_data_at_time(grop,calc.times[idx_frame],True)
            cgb += c
        cv, gb_area, tot_mesh_area = calc.gb_curvature(x,y,z,cgb,5,i,numGrs)
        # 2 Grain only: Rigid Body Motion
        ctr_dist = 0
        # if 'gr2' not in MF.exodus_readers[0].nodal_var_names:
        x,y,z,c = MF.get_data_at_time('unique_grains',calc.times[idx_frame],False)
        ctr_dist = calc.rbm_distance_centroids(x,y,z,c)
        print('  Finished calculating file '+str(i)+': '+str(round(time.perf_counter()-para_t0,2))+'s')
        return calc.times[idx_frame], gb_area, tot_mesh_area, cv, ctr_dist





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
        mp.set_start_method('fork')
        verb('set mp to fork since thats what Ubuntu uses and it works?')
        verb('Starting pool of '+str(calc.cpu)+' CPUs')
        cpu_pool = mp.Pool(calc.cpu)
        verb(cpu_pool)



    # Plot the current plane or whatever else later
    if calc.plot:
        print('Plotting')
        for i,idx in enumerate(calc.idx_frames):
            if calc.parallel:
                plots = cpu_pool.apply_async(parallelSlicePlot,args = (i, idx ))
            else:
                parallelSlicePlot(i,idx)
        if calc.parallel:
            plots.wait()
        print('Done Plotting')

    # run calculations and do math
    # if calc.calcs:
    #     print('Doing Calculations')
    #     para_results = []
    #     for i,idx in enumerate(calc.idx_frames):
    #         if calc.parallel:
    #             para_results.append(cpu_pool.apply_async(do_calculations,args = (i, idx, True )))
    #         else:
    #              para_results.append(do_calculations(i,idx))

    #     if calc.parallel:
    #         verb('Compiling results...')
    #         para_results = [r.get() for r in para_results]

    # run calculations and do math
    batch_size = 500
    csv_header = ['time', 'grain_area', 'tot_mesh_area','curvature','ctr_dist']
    if calc.calcs:
        print('Doing Calculations')
        if calc.parallel:
            # args_list = [(i, idx, True) for i, idx in enumerate(calc.idx_frames)]
            # para_results_async = cpu_pool.starmap_async(do_calculations, args_list)
            # verb('Compiling results...')
            # para_results = para_results_async.get()
            for i in range(0, len(calc.idx_frames), batch_size):
                batch_args = [(j, idx, True) for j, idx in enumerate(calc.idx_frames[i:i+batch_size])]
                para_results_async = cpu_pool.starmap_async(do_calculations, batch_args)
                print(f'Compiling batch {i // batch_size}')
                batch_results = para_results_async.get()
                # Save each batch
                batchloc = calc.outNameBase + '_calc_data_batch' + str(i).zfill(2) +'.csv'
                print('Saving Data: ',batchloc)
                df = pd.DataFrame(batch_results, columns = csv_header)
                df.sort_values(by="time").reset_index(drop=True, inplace=True)
                df.to_csv(batchloc)
                print(f'Finished saving batch {i // batch_size}')
                print(' ')
            # Compile batches
            print('Compiling batch csv files')
            try:
                # Use glob to find all batch CSV files
                batch_files_pattern = calc.outNameBase + '_calc_data_batch*.csv'
                batch_files = glob.glob(batch_files_pattern)

                if not batch_files:
                    print('No batch files found to combine.')
                else:
                    # Sort the batch files if necessary
                    batch_files.sort()

                    # Read each batch CSV file and store DataFrames in a list
                    dfs = []
                    for batch_file in batch_files:
                        print(f'Reading {batch_file}')
                        df = pd.read_csv(batch_file)
                        dfs.append(df)

                    # Concatenate all DataFrames
                    combined_df = pd.concat(dfs, ignore_index=True)

                    # Sort the combined DataFrame by 'time' if necessary
                    combined_df.sort_values(by="time", inplace=True)
                    combined_df.reset_index(drop=True, inplace=True)

                    # Save the combined DataFrame to a single CSV file
                    combined_csv_file = calc.outNameBase + '_calc_data.csv'
                    combined_df.to_csv(combined_csv_file, index=False)
                    print('Combined all batch files into:', combined_csv_file)

                    # # Optionally, delete the individual batch files
                    # for batch_file in batch_files:
                    #     os.remove(batch_file)
                    # print('Deleted individual batch files.')
            except Exception as e:
                print('An error occurred while combining batch files:', str(e))


        else:
            para_results = [do_calculations(i, idx) for i, idx in enumerate(calc.idx_frames)]
            saveloc = calc.outNameBase + '_calc_data.csv'
            print('Saving Data: ',saveloc)
            csv_header = ['time', 'grain_area', 'tot_mesh_area','curvature','ctr_dist']
            # calc.times[idx_frame], gb_area, tot_mesh_area, cv, ctr_dist
            df = pd.DataFrame(para_results, columns = csv_header)
            df.sort_values(by="time").reset_index(drop=True, inplace=True)
            # df['cr_eff'] = np.sqrt(df.grain_area.div(math.pi))
            df.to_csv(saveloc)
            print('Saved!')


        print('Calculations Done')


    if calc.parallel:
        verb('Closing pool')
        cpu_pool.close()
        cpu_pool.join()

    print('__main__ DONE')
#     quit()
# quit()
