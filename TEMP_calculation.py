from MultiExodusReader import MultiExodusReader
import multiprocessing as mp
from CalculationsV2 import CalculationsV2#, parallelPlot
# from CalculationEngine import para_time_build

import json
import argparse
import logging
pt = logging.warning
verb = logging.info
# from logging import warning as pt
# from logging import info as verb
# from VolumeScripts import *

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PolyCollection
from matplotlib.collections import PatchCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D
import matplotlib

from scipy.spatial import ConvexHull, Delaunay
import numpy as np
import time
# from time import timecalc =
import os
import glob
import pandas as pd
import math
import sys
import tracemalloc


# plot as specified but in parallel
# needs calc and MF named as such
def parallelPlot(i,idx_frame):
    para_t0 = time.perf_counter()
    x,y,z,c = MF.get_data_at_time(calc.var_to_plot,calc.times[idx_frame],True)
    nx, ny, nz, nc = calc.plane_slice(x,y,z,c)
    calc.plot_slice(i,nx,ny,nz,nc)
    verb('  Finished plotting file '+str(i)+': '+str(round(time.perf_counter()-para_t0,2))+'s')
    return


if __name__ == "__main__":
    print("__main__ Start")
    # xlist = []
    # x1 = np.asarray([0, 1, 2, 3, 4, 5, 6, 7])
    # x2 = np.asarray([10, 11, 12, 13, 14, 15, 16, 17])

    # xlist.append(x1)
    # xlist.append(x2)
    # x = np.vstack(xlist)
    # print(x)
    # print(" ")
    # mask = [0,3,7,4]
    # temp = []
    # test1 = x[:,0]
    # for n in mask:
    #     temp.append(x[:,n])
    #     test1 = test1 + x[:,n]
    # check = np.hstack(temp)

    # print(check)
    # print(test1)
    # test21 = [np.asarray([n1, n2, n3, n4]) for (n1,n2,n3,n4) in zip(x[:,0], x[:,3], x[:,7], x[:,4])]
    # print(test21)
    # test2 = np.asarray([np.asarray([n1, n2, n3, n4]) for (n1,n2,n3,n4) in zip(x[:,0], x[:,3], x[:,7], x[:,4])])

    # print(test2)

    # # MESHGRID TESTING
    # x_data = np.asarray([2, 2, 2, 3, 3, 3, 1, 1, 1, 4, 4, 4])
    # y_data = np.asarray([16, 64, 32, 64, 32, 16, 16, 32, 64, 32, 16, 64])
    # z_data = np.asarray([64, 31, 29, 78, 72, 63, 93, 40, 54, 35, 44, 3])
    # z_vals = np.asarray([2, 2, 2, 3, 3, 3, 1, 1, 1, 4, 4, 4])
    # # Sort coordinates and reshape in grid
    # idx1 = np.lexsort((y_data, x_data))
    # # print(idx1)
    # idx = np.lexsort((y_data, x_data)).reshape(4, 3)
    # # Plot
    # # print(idx)
    # # print(x_data[idx], y_data[idx], z_data[idx])
    # def tempctr(x,y):
    #     xy = np.asarray([x[:],y[:]]).T
    #     return xy

    # def tempctrz(x,y,z):
    #     xyz = np.asarray([x[:],y[:],z[:]]).T
    #     return xyz

    # ctr = tempctr(x_data,y_data)
    # ctrz = tempctrz(x_data,y_data,z_vals)
    # print(ctr)
    # print(" ")
    # print(ctr[2])

    # def get_c(xy_ctr, c, x, y):
    #     ind = (xy_ctr == (x,y)).all(axis=1)
    #     row = c[ind]
    #     return row
    # def get_c3(xy_ctr, c, x, y, z):
    #     ind = (xy_ctr == (x,y,z)).all(axis=1)
    #     row = c[ind]
    #     return row
    # print(get_c(ctr,z_data,2,32))

    # def xyc_to_array(xy_ctr,c):
    #     x_u = np.unique(xy_ctr[:,0])#[:,0]
    #     y_u = np.unique(xy_ctr[:,1])#mat[:,1]
    #     print(x_u)
    #     print(y_u)
    #     X,Y = np.meshgrid(x_u, y_u, indexing='xy')#specify indexing!!
    #     c_sort = np.array([get_c(xy_ctr,c,x,y) for (x,y) in zip(np.ravel(X), np.ravel(Y))])
    #     C = c_sort.reshape(X.shape)
    #     print(C)
    #     cdx,cdy = np.gradient(C)
    #     print(cdx)
    #     xdx,xdy= np.gradient(X)
    #     print(xdy)
    #     ydx,ydy= np.gradient(Y)
    #     print(xdy)
    #     print('dCdx')
    #     print(cdy/xdy)
    #     print('dCdy')
    #     print(cdx/ydx)
    #     dcdx = cdy/xdy
    #     dcdy = cdx/ydx

    #     return X, Y, C, dcdx, dcdy
    #     print(X)
    #     print(Y)
    #     print(C)
    #     plt.pcolormesh(X,Y,C)
    #     # plt.xlim(min(x), max(x))
    #     # plt.ylim(min(y), max(y))
    #     plt.show()

    # def xyzc_to_array(xy_ctr,c):
    #     x_u = np.unique(xy_ctr[:,0])#[:,0]
    #     y_u = np.unique(xy_ctr[:,1])#mat[:,1]
    #     z_u = np.unique(xy_ctr[:,2])
    #     print(x_u)
    #     print(y_u)
    #     print(z_u)
    #     X,Y,Z = np.meshgrid(x_u, y_u, z_u, indexing='xy')#specify indexing!!
    #     c_sort = np.array([get_c3(xy_ctr,c,x,y,z) for (x,y,z) in zip(np.ravel(X), np.ravel(Y), np.ravel(Z))])
    #     C = c_sort.reshape(X.shape)
    #     print(C)
    #     cdx,cdy = np.gradient(C)
    #     print(cdx)
    #     xdx,xdy= np.gradient(X)
    #     print(xdy)
    #     ydx,ydy= np.gradient(Y)
    #     print(xdy)
    #     print('dCdx')
    #     print(cdy/xdy)
    #     print('dCdy')
    #     print(cdx/ydx)
    #     dcdx = cdy/xdy
    #     dcdy = cdx/ydx

    #     return X, Y, C, dcdx, dcdy
    # X, Y, C, dcdx, dcdy = xyzc_to_array(ctrz,z_data)
    # print(dcdx)
    # print(dcdx.reshape(z_data.shape))
    # dCdxy = np.gradient(C, X, Y, axis=(1,2))
    # print(dCdxy)
    # quit()
# quit()
# exit()
    calc = CalculationsV2()
    # # print(calc.__dict__)
    # print("Testing some shit:")

    # calc_it = calc.get_frames()
    # # print(calc_it)
    # # frames = calc.frames
    read_ti = time.perf_counter()
    MF = MultiExodusReader(calc.file_names[0])
    # MF = MultiExodusReader('2D_NS_200iw_nemesis.e.12*')
    read_tf = time.perf_counter()
    print("  Finished reading files:",round(read_tf-read_ti,2),"s")

    # TEST
    # x,y,z,c = MF.get_data_at_time(calc.var_to_plot,calc.times[2],True)
    # nx, ny, nz, nc = calc.plane_slice(x,y,z,c,False)#, grads, norm
    # avg_c = np.average(nc, axis=1)
    # plt_x, plt_y = calc.plt_xy(nx,ny,nz)
    # mesh_ctr, mesh_vol = calc.mesh_center_quadElements(plt_x,plt_y)
    # gr_area = calc.c_area_in_slice(*MF.get_data_at_time(calc.var_to_plot,calc.times[1],True))
    # print(gr_area)

    # X, Y, C, dcdx, dcdy = xyc_to_array(mesh_ctr,avg_c)
    # plt.pcolormesh(X,Y,dcdz.reshape(X.shape),shading='nearest',cmap='coolwarm')
    # plt.colorbar()
    # calc.cl_args.debug = True
    # calc.plot_slice(1,nx,ny,nz,dcdz)
    # plt.show()
    # read_ti = time.perf_counter()
    # dx = calc.element_gradients(nx,ny,nz,nc)
    # read_tf = time.perf_counter()
    # print("  Finished doing gradients:",round(read_tf-read_ti,2),"s")
    results = []

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
            delta_phi = 0
            all_full_cv = []
            gb = []
            all_gr_ops = ['phi','gr0','gr1','gr2','gr3']
            all_grs = [x for x in all_gr_ops if x in MF.exodus_readers[0].nodal_var_names]
            # all_grs = ['gr0', 'gr1']
            for grop in all_grs:
                x,y,z,c = MF.get_data_at_time(grop,calc.times[idx_frame],True)
                op_area, tot_mesh_area = calc.c_area_in_slice(x,y,z,c,grop)
                sum_del_cv, full_delta, full_cv, cx, cy, cz, cc = calc.MER_curvature_calcs(x,y,z,c,True)
                calc.plot_slice(str(grop)+'_curv_'+str(i),cx,cy,cz,full_cv,str(grop)+"_curvature")
                # Change this and just use it for tot_cr_area for more grains
                # if 'phi' not in grop:
                all_full_cv.append(full_cv)
                if 'phi' in grop:
                    delta_phi = np.where((full_delta>0),0,1)
                if grop not in ['phi','gr2','gr3']:
                    gb.append(cc)
                    tot_gr_area += op_area
                    all_full_delta.append(full_delta)
                    # all_full_cv.append(full_cv)
            # No Cross Terms (delta_gr0*cv_gr1, etc)
            # delta_cv = sum([all_full_delta[n]*all_full_cv[n] for n in range(len(all_grs))])
            calc.plot_slice_forCurvature(str(i),cx,cy,cz,sum(gb),'gr0 + gr1')
            # delphi = np.where((delta_phi>0),0,1)
            # del01 = np.where(del0 & del1 & delphi, 1, 0)
            # calc.plot_slice('delta_gb_phi_'+str(i),cx,cy,cz,delphi,'delta_gb and phi')
            # calc.plot_slice('del_gb_phi_'+str(i),cx,cy,cz,del01,'delta_gb and phi')
            dels = [np.where((all_full_delta[n]>0) & (all_full_delta[n]<=1) & delta_phi,1,0) for n in range(len(all_full_delta))]
            # dels_phi = np.where(dels & delta_phi,1,0)
            deltot = np.where(sum(dels)>0,1,0)
            calc.plot_slice('all_gr_delta_'+str(i),cx,cy,cz,deltot,'delta_total all grs')
            # calc.plot_slice('all_gr_minus_phi_delta_'+str(i),cx,cy,cz,dels_phi,'delta_total all grs without phi')
            calc.plot_slice('delta_gr0_'+str(i),cx,cy,cz,del0,'delta_gr0')
            calc.plot_slice('delta_gr1_'+str(i),cx,cy,cz,del1,'delta_gr1')
            # delta_cv = sum([all_full_cv[n] for n in range(len(all_grs))])
            cv_tot = sum(all_full_cv)
            calc.plot_slice('delta_'+str(i),cx,cy,cz,deltot,'delta_total')
            tot_delta_cv = np.sum(cv_tot*deltot)
            tot_delta = np.sum(deltot)
            calc.plot_slice('curvature_'+str(i),cx,cy,cz,cv_tot*deltot,'delta*curvature_total')
            calc.plot_slice('full_curvature_'+str(i),cx,cy,cz,cv_tot,'curvature_total')
            # print(tot_delta_cv, tot_delta, tot_delta_cv/tot_delta)
            # With those cross terms
            # deltas = sum([np.where(all_full_delta[n]<=1,all_full_delta[n],0) for n in range(len(all_grs))])
            # cvs = sum([all_full_delta[n] for n in range(len(all_grs))])
            # tot_delta_cv = np.sum(deltas*cvs)
            print('  Finished calculating file '+str(i)+': '+str(round(time.perf_counter()-para_t0,2))+'s')
            return calc.times[idx_frame], tot_gr_area, tot_mesh_area, tot_delta_cv, tot_delta_cv/tot_delta

    results.append(do_calculations('test_60',calc.idx_frames[60],True))
    print(results)


    sys.exit()
    # gr0
    # x0,y0,z0,c0 = MF.get_data_at_time('gr0',calc.times[2],True)
    # cx0, cy0, cz0, cc0, cv0 = calc.threeplane_curvature(x0,y0,z0,c0)
    # d0 = calc.delta_interface_func(cc0,1)

    # sumcv0, outcv0, outx,outy,outz,outc = calc.MER_curvature_calcs(x0,y0,z0,c0,True)
    # print(sumcv0)
    # calc.plot_slice('TEST_c_gr0',cx0,cy0,cz0,cc0,'gr0')
    # calc.plot_slice('TEST_c_gr0_calc',outx,outy,outz,outc,'gr0')
    # calc.plot_slice('TEST_cv_gr0',cx0,cy0,cz0,cv0*d0,'gr0_curvature*delta')
    # calc.plot_slice('TEST_cv_gr0_calc',outx,outy,outz,outcv0,'gr0_curvature*delta')
    # sys.exit()
    # gr1
    x1,y1,z1,c1 = MF.get_data_at_time('gr1',calc.times[2],True)
    cx1, cy1, cz1, cc1, cv1 = calc.threeplane_curvature(x1,y1,z1,c1)
    d1 = calc.delta_interface_func(cc1)

    # gr2
    x2,y2,z2,c2 = MF.get_data_at_time('phi',calc.times[2],True)
    cx2, cy2, cz2, cc2, cv2 = calc.threeplane_curvature(x2,y2,z2,c2)
    d2 = calc.delta_interface_func(cc2)

    # calc.plot_slice(0,X,Y,Z,C)
    # calc.plot_slice(1,X[1],Y[1],Z[1],C[1])
    # calc.plot_slice(2,X[2],Y[2],Z[2],C[2])
    # print(min(cv0))
    # print(max(cv0))
    # calc.plot_slice('cv_gr0',cx0,cy0,cz0,np.where(cv0>=0.0,-1,cv0),'gr0_curvature')
    sys.exit()
    # # PLOTS
    temp_gb = np.where(((d20+d21+d22)<=1),(d20+d21+d22),0)
    temp_grs = np.where(((d20+d21)<=1),(d20+d21),0)
    calc.plot_slice('c_gr0',cx0,cy0,cz0,cc0,'gr0')
    calc.plot_slice('c_gr1',cx1,cy1,cz1,cc1,'gr1')
    calc.plot_slice('c_phi',cx2,cy2,cz2,cc2,'phi')
    calc.plot_slice('cv_gr0',cx0,cy0,cz0,cv0,'gr0_curvature')
    calc.plot_slice('cv_gr1',cx1,cy1,cz1,cv1,'gr1_curvature')
    calc.plot_slice('cv_phi',cx2,cy2,cz2,cv2,'phi_curvature')
    calc.plot_slice('cv_sum',cx2,cy2,cz2,cv0+cv1+cv2,'all_curvature')
    calc.plot_slice('cv_grsum',cx2,cy2,cz2,cv0+cv1,'allGr_curvature')
    calc.plot_slice('d_gr0',cx0,cy0,cz0,d20,'gr0_delta')
    calc.plot_slice('d_gr1',cx1,cy1,cz1,d21,'gr1_delta')
    calc.plot_slice('d_phi',cx2,cy2,cz2,d22,'phi_delta')
    calc.plot_slice('d_sum',cx2,cy2,cz2,d20+d21+d22,'all_delta')
    calc.plot_slice('d_grsum',cx2,cy2,cz2,d20+d21,'allGr_delta')
    calc.plot_slice('dcv_gr0',cx0,cy0,cz0,d20*cv0,'gr0_delta*cv')
    calc.plot_slice('dcv_gr1',cx1,cy1,cz1,d21*cv1,'gr1_delta*cv')
    calc.plot_slice('dcv_phi',cx2,cy2,cz2,d22*cv2,'phi_delta*cv')
    calc.plot_slice('d_all_cv_gr0',cx0,cy0,cz0,(d20+d21+d22)*cv0,'all_delta*gr0_cv')
    calc.plot_slice('d_all_cv_gr1',cx1,cy1,cz1,(d20+d21+d22)*cv1,'all_delta*gr1_cv')
    calc.plot_slice('d_all_cv_phi',cx2,cy2,cz2,(d20+d21+d22)*cv2,'all_delta*phi_cv')
    calc.plot_slice('d_all_cv_allgr',cx1,cy1,cz1,(d20+d21+d22)*(cv0+cv1),'all_delta*allgr_cv')
    calc.plot_slice('d_allmax1_cv_gr0',cx0,cy0,cz0,temp_gb*cv0,'all_delta_max1*gr0_cv')
    calc.plot_slice('d_allmax1_cv_gr1',cx1,cy1,cz1,temp_gb*cv1,'all_delta_max1*gr1_cv')
    calc.plot_slice('d_allmax1_cv_phi',cx2,cy2,cz2,temp_gb*cv2,'all_delta_max1*phi_cv')
    calc.plot_slice('d_allmax1_cv_phi',cx2,cy2,cz2,temp_gb*(cv0+cv1),'all_delta_max1*allgr_cv')
    calc.plot_slice('dcv_sumEachOP',cx0,cy0,cz0,(d20*cv0)+(d21*cv1)+(d22*cv2),'all delta*cv')
    calc.plot_slice('dcv_sumEachgr',cx0,cy0,cz0,(d20*cv0)+(d21*cv1),'allgr delta*cv')
    calc.plot_slice('dmax1cv_sumEachgr',cx0,cy0,cz0,(np.where(d20<=1,d20,0)*cv0)+(np.where(d21<=1,d21,0)*cv1),'allgr delta(max1)*cv')
    calc.plot_slice('dcv_sumAllgrMax1',cx0,cy0,cz0,(temp_grs*cv0)+(temp_grs*cv1),'allgr deltaSumMax1*cv')
    # calc.plot_slice(4,cx1,cy1,cz1,d20+d21+d22)
    # calc.plot_slice(5,cx1,cy1,cz1,d20+d22)
    # calc.plot_slice(6,cx1,cy1,cz1,d20)
    # calc.plot_slice(7,cx1,cy1,cz1,d21)
    # calc.plot_slice(8,cx1,cy1,cz1,d22)
    # calc.plot_slice(9,cx2,cy2,cz2,cc2)
    # calc.plot_slice(10,cx1,cy1,cz1,temp_gb)
    # calc.plot_slice(11,cx1,cy1,cz1,cv1)

        # plt.figure()
        # # plt.plot(Y[1][0],label='X')
        # plt.plot(Y[1][0],dcdy[1][0],label='dcdy')
        # plt.plot(Y[1][0],dc_norm[1][0],label='dcnorm')
        # plt.plot(Y[1][0],dcdy[1][0]/dc_norm[1][0],label='man dcdy_norm')
        # # plt.plot(Y[1][0],dcdy_norm[1][0],label='dcdy_norm')
        # plt.legend()
        # plt.show()
        # plt.pcolormesh(X[1],Y[1],dcdy[1],shading='nearest',cmap='coolwarm')
        # plt.colorbar()
        # plt.show()

    # calc.plot_slice(3,cx,cy,cz,cv)
    # tempcv = np.where(cv<0.0,0,1)
    # calc.plot_slice(4,cx,cy,cz,tempcv)
    # calc.plot_slice(5,cx,cy,cz,d1)
    # calc.plot_slice(6,cx,cy,cz,d2)
    # calc.plot_slice(7,cx,cy,cz,d2*cv)
    # calc.plot_slice(8,cx,cy,cz,d2*tempcv)
    # calc.plot_slice(9,cx,cy,cz,np.where((d2*cv)<0.0,0,d2*cv))
    # calc.plot_slice(10,cx,cy,cz,np.where((d2)<=1.0,d2,0))
    # calc.plot_slice(11,cx,cy,cz,np.where((d2)<5.0,d2*cv,0))
    # for n in range(len(C)):
    #     fig = plt.figure(n)
    #     plt.pcolormesh(X[n],Y[n],cv[n],shading='nearest',cmap='coolwarm')
    #     plt.colorbar()
    #     fig.savefig('pics'+'/'+calc.outNameBase+'_sliced_'+calc.plane_axis+
    #             str(calc.plane_coord_name)+'_curvature_'+str(n)+'.png',dpi=500,transparent=True )

    # calc.plot_slice(0,nx,ny,nz,nc)


    # calc.plot_slice(1,nx,ny,nz,grads[1])
    # calc.plot_slice(2,nx,ny,nz,grads[2])
    # calc.plot_slice(3,nx,ny,nz,norm[2])

    # quit()
    quit()
quit()
quit()
quit()
    # REMEMBERR IF NO MESH ADAPTIVITY CAN JUST OPEN THEM ALL!!!!
    # for i,idx in enumerate(calc_it[0]):
    #     pt('Frame '+str(i+1)+'/'+str(len(calc_it[0])))
    #     # MF = MultiExodusReader(calc.files[idx])
    #     x,y,z,c = MF.get_data_at_time(calc.var_to_plot,calc.times[idx],True)
    #     print('Got the data at the time')
    #     nx, ny, nz, nc = calc.plane_slice(x,y,z,c)
    #     calc.plot_slice(i,nx,ny,nz,nc)
    # calc.getMER()
    # for i,idx in enumerate(calc_it[0]):
    #     print(i)
    #     print(idx)
    #     print('and now run')
    #     calc.parallelPlot(i,idx)
    # print('Got this far')





    # cpu_pool = mp.Pool(calc.cpu)
    # pt(cpu_pool)
    # pool_t0 = time.perf_counter()
    # results = []
    # for i,idx in enumerate(calc_it[0]):
    #     # results.append(cpu_pool.apply_async(parallelPlot,args = (i, idx )))
    #     results.append(cpu_pool.apply_async(parallelPlot,args = (i, idx )))
    # cpu_pool.close()
    # cpu_pool.join()
    # pt("Total Pool Time: "+str(round(time.perf_counter()-pool_t0))+"s")
    # pt("Aggregating data...")#Restructuring
    # # pt(results[0])
    # results = [r.get() for r in results]
    # print(results)





        # # plott
        # fig, ax = plt.subplots()
        # coords = np.asarray([ np.asarray([x_val,y_val]).T for (x_val,y_val) in zip(ny,nz) ])
        # # coords, vols = calc.mesh_center_quadElements(y,z)
        # print(coords)
        # #USE POLYCOLLECTION TO DRAW ALL POLYGONS DEFINED BY THE COORDINATES
        # p = PolyCollection(coords, cmap=matplotlib.cm.coolwarm, alpha=1)#,edgecolor='k'      #Edge color can be set if you want to show mesh

        # #COLOR THE POLYGONS WITH OUR VARIABLE
        # ## Map plot variable range to color range
        # c_min = np.amin(c)
        # c_max = np.amax(c)
        # colors = pc#(c - c_min)/(c_max-c_min)
        # #
        # p.set_array(np.array(colors) )

        # #ADD THE POLYGON COLLECTION TO AXIS --> THIS IS WHAT ACTUALLY PLOTS THE POLYGONS ON OUR WINDOW
        # ax.add_collection(p)

        # #FIGURE FORMATTING

        # #SET X AND Y LIMITS FOR FIGURE --> CAN USE x,y ARRAYS BUT MANUALLY SETTING IS EASIER
        # # ax.set_xlim([0,300])
        # # ax.set_ylim([0,300])
        # ax.set_xlim([np.amin(ny),np.amax(ny)])
        # ax.set_ylim([np.amin(nz),np.amax(nz)])
        # #SET ASPECT RATIO TO EQUAL TO ENSURE IMAGE HAS SAME ASPECT RATIO AS ACTUAL MESH
        # ax.set_aspect('equal')

        # #ADD A COLORBAR, VALUE SET USING OUR COLORED POLYGON COLLECTION
        # fig.colorbar(p,label="phi")
        # plt.show()






    # dict = {}
    # dict['cl_args'] = vars(calc.cl_args)
    # dict['params'] = {
    #     'adaptive_mesh':calc.adaptive_mesh,
    #     'test' : 7
    # }
    # dict['file_names'] = list(calc.file_names)
    # # dict['testvalue'] = list(calc.file_names)
    # print(calc.file_names)
    # print(dict)
    # with open('tiger_meta.json', 'w') as fp:
    #     json.dump(dict, fp, indent=4)

#     pt("END OF THE MAIN")

#     quit()
# quit()






if __name__ == "__main__":
    print("__main__ Start")
    calc = CalculationEngine()
    # If it exited to run the parallel_times
    if calc.cl_args.parallel_times == 0:
        verb('Parallelizing with ' + str(calc.cl_args.cpu) + ' cpus')
        cpu_pool = mp.Pool(calc.cl_args.cpu)
        verb(cpu_pool)
        pool_t0 = time.perf_counter()
        results = []
        for i,file in enumerate(calc.file_names):
            results.append(cpu_pool.apply_async(para_time_build,args = (i, file, calc.len_files )))
        cpu_pool.close()
        cpu_pool.join()
        verb("Total Pool Time: "+str(round(time.perf_counter()-pool_t0))+"s")
        verb("Aggregating data...")#Restructuring
        verb(results[0])
        # verb(results[0].get())
        results = [r.get() for r in results]
        time_file_list = []
        for row1 in results:
            for row2 in row1:
                time_file_list.append(row2)
        times_files = np.asarray(time_file_list)
        times_files = times_files[times_files[:, 0].astype(float).argsort()]
        np.save('times_files.npy', times_files)
        verb('Done Building Time Data')
        sys.argv.append('--parallel-times=1')
        verb('Re-entering CalculationEngine')
        calc = CalculationEngine()


    pt("END OF THE MAIN")

    quit()
quit()
print(calc.cl_args.new_meta)
# if "--new-meta" in sys.argv:
# parser = argparse.ArgumentParser()
# parser.add_argument('--new-meta', action='store_true')
# parser.add_argument('--cpu','-n', default=1,type=int)
#
# args = parser.parse_args()
# print(args)
# print(args.new_meta)

times_files = np.load('large_2D_times_files.npy')
times = times_files[:,0].astype(float)
t_step = times_files[:,2].astype(int)


# quit()
# if "--cpu*" in sys.argv:
#     n_cpu =

if os.path.exists("test.json") and not calc.cl_args.new_meta:
    print("EXISTS")
elif calc.cl_args.new_meta or not os.path.exists("test.json"):
    if os.path.exists("test.json"):
        print("deleting old metadata and writing new")
    else:
        print("Writing new metadata")
else:
    print("problem")


dict = {}
dict['times_files'] = times_files.tolist()
# dict['times'] = list(times)
# dict['file_names'] = list(np.unique(times_files[:,1].astype(str)))
# dict['dimension'] = 2
print(dict)
# print(dict['dimension'])

with open('test.json', 'w') as fp:
    json.dump(dict, fp)

quit()
with open('test.json') as json_file:
    data = json.load(json_file)

print(data)
print(data['times_files'])
print(" ")
tf = np.asarray(data['times_files'])
print(tf)
print(tf[:,0].astype(float))
# print(dict['dimension'])
# print(dict['times'])
quit()

times_files = np.load('times_files.npy')
times = times_files[:,0].astype(float)
t_step = times_files[:,2].astype(int)

#GETTING CLOSEST TIME STEP TO DESIRED SIMULATION TIME FOR RENDER --> TYPICALLY 200 FRAMES WITH 20 FPS GIVES A GOOD 10 S LONG VIDEO
# n_frames = 200
if sequence == True:
    if n_frames < len(times):
        t_max = times[-1]
        # t_max = max(times)
        t_frames =  np.linspace(0.0,t_max,n_frames)
        idx_frames = [ np.where(times-t_frames[i] == min(times-t_frames[i],key=abs) )[0][0] for i in range(n_frames) ]
        idx_frames = list( map(int, idx_frames) )
    else:
        t_frames = times
        idx_frames = range(len(times))
elif sequence == False:
    t_frames = times
    idx_frames = range(len(times))
else:
    raise ValueError('sequence has to be True or False, not: ' + str(sequence))

if cutoff != 0:
    print("Cutting End Time to ",cutoff)
    t_frames = [x for x in t_frames if x <= cutoff]
    idx_frames = range(len(t_frames))

tot_frames = len(idx_frames)


def pore_in_hull(xyz_for_hull,void_ctr_xyz,tolerance,point_plot_TF):
    # print(" Taking convex hull")
    # tic = time.perf_counter()
    hull = ConvexHull(xyz_for_hull)
    # toc = time.perf_counter()
    # print("     ",toc - tic)
    # print(" Doing in hull calculation")
    # get array of boolean values indicating in hull if True
    # tic = time.perf_counter()
    in_hull = np.all(np.add(np.dot(void_ctr_xyz, hull.equations[:,:-1].T),
                            hull.equations[:,-1]) <= tolerance, axis=1)
    # toc = time.perf_counter()
    # print("     ",toc - tic)

    # The void centers that are in the hull
    void_in_hull = void_ctr_xyz[in_hull]
    # Plot a scatterplot of the void mesh centers in the hull
    if point_plot_TF == True:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for simplex in hull.simplices:
            # plt.plot(grain_ctr[simplex, 0], grain_ctr[simplex, 1], grain_ctr[simplex,2], 'r-')
            plt.plot(xyz_for_hull[simplex, 0], xyz_for_hull[simplex, 1], 'r-')
        # Now plot the void points
        ax.scatter(void_in_hull[:, 0], void_in_hull[:, 1],s=0.01,alpha=0.5)
        ax.set_aspect('equal')
        # plt.autoscale()
        plt.show()
    # Output the boolean array of which void_ctr is in the hull for use
    return in_hull




def t0_opCount_headerBuild(idx_frames):
    print("Calculating initial frame...")
    read_ti = time.perf_counter()
    MF = MultiExodusReader(times_files[idx_frames[0],1])#.exodus_readers
    read_tf = time.perf_counter()
    print("  Finished reading initial frame:",round(read_tf-read_ti,2),"s")

    x,y,z,c = MF.get_data_at_time(var_to_plot,times[0])

    c_int = np.rint(c)

    op_0 = round(np.amax(c_int))+2
    csv_header = ["time", "internal_pore", "total_hull", "vol_density", "total_void"]
    for n in range(1,op_0):
        csv_header.append("Grain_"+str(n))
    return op_0, csv_header



def para_volume_calc(time_step,i,op_max):
    print("Calculating frame",i+1, "/",tot_frames)
    read_ti = time.perf_counter()
    MF = MultiExodusReader(times_files[time_step,1])#.exodus_readers
    read_tf = time.perf_counter()
    print("  Finished reading frame",i+1, ":",round(read_tf-read_ti,2),"s")

    x,y,z,c = MF.get_data_at_time(var_to_plot,times[time_step])# MF.get_data_at_time(var_to_plot,times[i])
    c_int = np.rint(c)

    mesh_ctr = np.asarray([ x[:, 0] + (x[:, 2] - x[:, 0])/2,
                            y[:, 0] + (y[:, 2] - y[:, 0])/2 ]).T

    mesh_vol = np.asarray((x[:, 2] - x[:, 0])*(y[:, 2] - y[:, 0]))

    zeros = np.zeros_like(c_int)
    volumes = []# np.zeros(round(np.amax(c_int))+2)
    # centroids = np.asarray([ volumes, volumes ]).T
    grain_centroids = []
    for n in range(op_max):
        volumes.append(np.sum(np.where(c_int==(n-1),mesh_vol,zeros)))
        if volumes[n] > 0.0 and n > 0:
            grain_centroids.append([ np.sum(np.where(c_int==(n-1),mesh_ctr[:,0] * mesh_vol,zeros)) / np.sum(np.where(c_int==(n-1),mesh_vol,zeros)),
                                        np.sum(np.where(c_int==(n-1),mesh_ctr[:,1] * mesh_vol,zeros)) / np.sum(np.where(c_int==(n-1),mesh_vol,zeros))])
    if quarter_hull == True:
        print(len(grain_centroids))
        for n in range(len(grain_centroids)):
            if grain_centroids[n][0] > grain_centroids[n][1]:
                grain_centroids[n][0] = max_xy
            elif grain_centroids[n][1] > grain_centroids[n][0]:
                grain_centroids[n][1] = max_xy
    print(grain_centroids)
    print(grain_centroids[0])
    print(grain_centroids[0][1])
    grain_ctr = np.delete(mesh_ctr, np.where((c_int<0.0))[0], axis=0)
    # For if using centroids for the convex hull
    # grain_vol = np.delete(mesh_vol, np.where((c_int<0.0))[0], axis=0)
    void_ctr = np.delete(mesh_ctr, np.where((c_int>=0.0))[0], axis=0)
    void_vol = np.delete(mesh_vol, np.where((c_int>=0.0))[0], axis=0)

    # internal_pore_vol = np.sum(void_vol[pore_in_hull(grain_ctr,void_ctr,1e-12,point_plot_TF=False)])
    if quarter_hull == True:
        temp_ctr = np.append(grain_ctr,[[max_xy,max_xy]],axis=0)
        internal_pore_vol = np.sum(void_vol[pore_in_hull(temp_ctr,void_ctr,1e-12,point_plot_TF=False)])
    else:
        internal_pore_vol = np.sum(void_vol[pore_in_hull(grain_ctr,void_ctr,1e-12,point_plot_TF=False)])
    # For if using centroids for the convex hull
    # grain_hull = np.sum(grain_vol[pore_in_hull(grain_ctr,grain_ctr,1e-12,point_plot_TF=False)])
    total_hull_vol = sum(volumes[1:]) + internal_pore_vol
    per_tdens = (total_hull_vol - internal_pore_vol) / total_hull_vol
    # print("Memory:",tracemalloc.get_traced_memory())
    print("  Finished calculating frame",i+1, ":",round(time.perf_counter()-read_tf,2),"s")
    # print([times[i], internal_pore_vol, total_hull_vol, per_tdens] + volumes)
    return [times[time_step], internal_pore_vol, total_hull_vol, per_tdens] + volumes





#IF IN MAIN PROCESS
if __name__ == "__main__":
    tracemalloc.start()
    # Calculate maximum number of OPs and csv header
    op_max, csv_header = t0_opCount_headerBuild(idx_frames)
    results = []

    if len(sys.argv) > 2:
        if "skip" in sys.argv[2]:
            print("NOTE: Skipping last file as indicated with 'skip' flag")
            print(" ")
            name_unq = name_unq[:-1]

    all_time_0 = time.perf_counter()
    if n_cpu == 1:
        print("Running in series")
        for i,frame in enumerate(idx_frames):
            results.append(para_volume_calc(frame, i, op_max ))
        # compile and save the data
        print("Total Time:",round(time.perf_counter()-all_time_0,2),"s")
        print("Aggregating data...")

    elif n_cpu > 1:
        print("Running in parallel")
        #CREATE A PROCESS POOL
        cpu_pool = mp.Pool(n_cpu)
        print(cpu_pool)
        all_time_0 = time.perf_counter()

        for i,frame in enumerate(idx_frames):
            results.append(cpu_pool.apply_async(para_volume_calc,args = (frame, i, op_max)))#, callback = log_result)
        # ex_files = [cpu_pool.map(para_time_build,args=(file,)) for file in name_unq  ]

        cpu_pool.close()
        cpu_pool.join()
        print(cpu_pool)
        print("Total Pool Time:",round(time.perf_counter()-all_time_0,2),"s")
        print("Aggregating data...")#Restructuring
        results = [r.get() for r in results]

    else:
        raise(ValueError("ERROR: n_cpu command line flag error"))
    # print(results)
    out_volumes = np.asarray(results)
    # print(out_volumes)
    out_volumes = out_volumes[out_volumes[:, 0].astype(float).argsort()]
    print('\n' + "Done Building Area Data")
    saveloc = '../' + dirName + '_areas.csv'
    np.savetxt(saveloc, np.asarray(out_volumes), delimiter=',', header=','.join(csv_header), comments='')
    current, peak =  tracemalloc.get_traced_memory()
    print("Memory Final (current, peak):",round(current/1048576,1), round(peak/1048576,1), "MB")
    # "volumes.csv",
    quit()

quit()



# np.savetxt("volumes.csv", np.asarray(out_volumes), delimiter=',', header=','.join(csv_header), comments='')

# plt.figure(2)
# plt.scatter(times[idx_frames],pore_array[:,0],label="Internal Pore")
# plt.scatter(times[idx_frames],pore_array[:,4],label="Neck 1")
# plt.scatter(times[idx_frames],pore_array[:,7],label="Neck 2")
# plt.xlabel("Time")
# plt.ylabel("Areas")
# plt.legend()
# plt.show()


# df = pd.DataFrame(columns=['time', 'pore_area','distance', 'g1x', 'g1y', 'g1area', 'g2x', 'g2y', 'g2area'],data=np.hstack((times[idx_frames,None], pore_array[:,:])))
# # print(df)
# # # pd.DataFrame(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
# # #                    columns=['a', 'b', 'c'])
# df.to_csv('../PoreArea.csv',index=False)
