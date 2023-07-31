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
def do_calculations(i,idx_frame,all_op=False):
    para_t0 = time.perf_counter()
    if not all_op:
        x,y,z,c = MF.get_data_at_time(calc.var_to_plot,calc.times[idx_frame],True)
        op_area, tot_mesh_area = calc.c_area_in_slice(x,y,z,c,calc.var_to_plot)
        sum_del_cv, full_delta, full_cv = calc.MER_curvature_calcs(x,y,z,c)
        verb('  Finished calculating file '+str(i)+': '+str(round(time.perf_counter()-para_t0,2))+'s')
        return calc.times[idx_frame], op_area, tot_mesh_area, sum_del_cv
    else:
        tot_gr_area = 0
        all_full_delta = []
        all_full_cv = []
        all_gr_ops = ['gr0','gr1','gr2','gr3']
        all_grs = [x for x in all_gr_ops if x in MF.exodus_readers[0].nodal_var_names]
        for grop in all_grs:
            x,y,z,c = MF.get_data_at_time(grop,calc.times[idx_frame],True)
            op_area, tot_mesh_area = calc.c_area_in_slice(x,y,z,c,grop)
            sum_del_cv, full_delta, full_cv, cx, cy, cz, cc = calc.MER_curvature_calcs(x,y,z,c,True)
            calc.plot_slice(str(grop)+'_curv_'+str(i),cx,cy,cz,full_cv,str(grop)+"_curvature")
            # Change this and just use it for tot_cr_area for more grains
            if 'phi' not in grop:
                tot_gr_area += op_area
                all_full_delta.append(full_delta)
                all_full_cv.append(full_cv)
        # No Cross Terms (delta_gr0*cv_gr1, etc)
        delta_cv = sum(all_full_delta)#sum([all_full_delta[n]*all_full_cv[n] for n in range(len(all_grs))])
        tot_delta_cv = np.sum(np.where((sum(all_full_delta)<=1) & (sum(all_full_delta)>0),delta_cv,0))
        tot_delta = np.sum(np.where(sum(all_full_delta)<=1,1,0))
        calc.plot_slice('delta_'+str(i),cx,cy,cz,delta_cv,'delta_total')
        calc.plot_slice('full_curvature_'+str(i),cx,cy,cz,np.where((sum(all_full_delta)<=1) & (sum(all_full_delta)>0),delta_cv,0),'curvature_total')
        # print(tot_delta_cv, tot_delta, tot_delta_cv/tot_delta)
        # With those cross terms
        # deltas = sum([np.where(all_full_delta[n]<=1,all_full_delta[n],0) for n in range(len(all_grs))])
        # cvs = sum([all_full_delta[n] for n in range(len(all_grs))])
        # tot_delta_cv = np.sum(deltas*cvs)
        print('  Finished calculating file '+str(i)+': '+str(round(time.perf_counter()-para_t0,2))+'s')
        return calc.times[idx_frame], tot_gr_area, tot_mesh_area, tot_delta_cv, tot_delta_cv/tot_delta




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
    if calc.calcs:
        print('Doing Calculations')
        para_results = []
        for i,idx in enumerate(calc.idx_frames):
            if calc.parallel:
                para_results.append(cpu_pool.apply_async(do_calculations,args = (i, idx, True )))
            else:
                 para_results.append(do_calculations(i,idx))

        if calc.parallel:
            verb('Compiling results...')
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
        print('Saving Data: ',saveloc)
        csv_header = ['time', 'grain_area', 'tot_mesh_area','curvature', 'delta_normalized_curvature']
        # calc.times[idx_frame], tot_gr_area, tot_mesh_area, tot_del_cv
        df = pd.DataFrame(para_results, columns = csv_header)
        df.sort_values(by="time").reset_index(drop=True, inplace=True)
        # df['cr_eff'] = np.sqrt(df.grain_area.div(math.pi))
        df.to_csv(saveloc)
        print('Saved!')


    if calc.parallel:
        verb('Closing pool')
        cpu_pool.close()
        cpu_pool.join()

    print('__main__ DONE')
#     quit()
# quit()
