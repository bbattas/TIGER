# Test the nemesis changes that are causing the benchmark paper scripts to have problems
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.collections import PolyCollection
import logging
import argparse
import glob

from MultiExodusReader import MultiExodusReader

dim3 = False
# --------------------------------------------------------------------------------------
# Manually building bare minimum
# --------------------------------------------------------------------------------------

# Find the nemesis files:
e_name = [x.rsplit('.',1)[0]+"*" for x in glob.glob("*.n.*")]
files = np.unique(e_name)
print("Files being used:")
print(files[:4]," ...")
print(' ')

MF = MultiExodusReader(files[0])
times = MF.global_times
print(f"times = {times}")
for idx,t in enumerate(times):
    if idx > 5:
        break
    x,y,z,c = MF.get_data_at_time('phi',t,True)
    print('='*30)
    print(f'Step {idx}')
    if z.any():
        print('Nonzero Z!')
        dim3 = True
    print(f'x = {x}')
    print(' ')
    print(f'c = {c}')
    print(' ')
    if hasattr(c[0], "__len__"):
        plotc = np.average(c, axis=1)
    else:
        plotc = c
    print(f'plotting c = {plotc}')

    coords = np.asarray([ np.asarray([x_val,y_val]).T for (x_val,y_val) in zip(x,y) ])
    fig, ax = plt.subplots()
    p = PolyCollection(coords, cmap=matplotlib.cm.coolwarm, alpha=1)#,edgecolor='k'
    p.set_array(np.array(plotc) )
    ax.add_collection(p)
    ax.set_xlim([np.amin(x),np.amax(x)])
    ax.set_ylim([np.amin(y),np.amax(y)])
    ax.set_aspect('equal')
    p.set_clim(0.0, 1.0)
    fig.colorbar(p, label='phi')
    fig.savefig(f'debug_01_phi_{idx}.png',dpi=500,transparent=True)
    plt.close('all')

# That seems fine?
print('='*30)
print('='*30)
print('='*30)
print(' ')

# --------------------------------------------------------------------------------------
# Using times_files.npy with minimum
# --------------------------------------------------------------------------------------

times_files = np.load('times_files.npy')
times = times_files[:,0].astype(float)
files = times_files[:,1].astype(str)
t_steps = times_files[:,2].astype(int)
file_names = np.unique(files)

MF = MultiExodusReader(file_names[0])
print(f'Files = {file_names}')
print(f"times = {times}")
for idx,t in enumerate(times):
    if idx > 5:
        break
    x,y,z,c = MF.get_data_at_time('phi',t,True)
    if hasattr(c[0], "__len__"):
        plotc = np.average(c, axis=1)
    else:
        plotc = c

    coords = np.asarray([ np.asarray([x_val,y_val]).T for (x_val,y_val) in zip(x,y) ])
    fig, ax = plt.subplots()
    p = PolyCollection(coords, cmap=matplotlib.cm.coolwarm, alpha=1)#,edgecolor='k'
    p.set_array(np.array(plotc) )
    ax.add_collection(p)
    ax.set_xlim([np.amin(x),np.amax(x)])
    ax.set_ylim([np.amin(y),np.amax(y)])
    ax.set_aspect('equal')
    p.set_clim(0.0, 1.0)
    fig.colorbar(p, label='phi')
    fig.savefig(f'debug_02_phi_{idx}.png',dpi=500,transparent=True)
    plt.close('all')


print('='*30)
print('='*30)
print('='*30)
print(' ')

# --------------------------------------------------------------------------------------
# Assuming possible 3D:
# Rebuilding just the calcv2 parts for this here
# --------------------------------------------------------------------------------------


# Reconstructing based on calcv2

def plt_xyz(x,y,z,plane_axis,ctr=None):
    if 'x' in plane_axis:
        print('Using y as the x-axis and z as the y-axis')
        plt_x = y
        plt_y = z
        plt_z = x
        mask = [1,2,0]
    elif 'y' in plane_axis:
        print('Using z as the x-axis and x as the y-axis')
        plt_x = z
        plt_y = x
        plt_z = y
        mask = [2,0,1]
    elif 'z' in plane_axis:
        print('Using x as the x-axis and y as the y-axis')
        plt_x = x
        plt_y = y
        plt_z = z
        mask = [0,1,2]
    else:
        raise ValueError('Plane measuring c on not specified!')
    if not ctr is None:
        var_out = np.asarray([np.asarray([n1, n2, n3]) for (n1,n2,n3)
                            in zip(ctr[:,mask[0]], ctr[:,mask[1]], ctr[:,mask[2]])])
        return plt_x, plt_y, plt_z, var_out
    else:
        return plt_x, plt_y, plt_z, mask


def adjust_plane(zmin,zmax,zval):
    tol = 1e-6
    planeval = zval
    c_on_plane = ((planeval == zmax) | (planeval == zmin))
    if np.any(c_on_plane):
        planeval = planeval + tol
        print(f"Shifting plane from {zval} to {planeval}")
        c_on_plane = ((planeval == zmax) | (planeval == zmin))
        if np.any(c_on_plane):
            raise ValueError('After shift, plane is still on boundary!')
        else:
            return planeval
    else:
        return planeval

def plane_interpolate_nodal_quad(min,max,axis,plane_coord,var):
    int_frac = (plane_coord - min) / (max - min)
    if 'x' in axis:
        return np.asarray([ var[:,0] + int_frac[:]*(var[:,1]-var[:,0]),
                            var[:,3] + int_frac[:]*(var[:,2]-var[:,3]),
                            var[:,7] + int_frac[:]*(var[:,6]-var[:,7]),
                            var[:,4] + int_frac[:]*(var[:,5]-var[:,4]) ]).T
    elif 'y' in axis:
        return np.asarray([ var[:,0] + int_frac[:]*(var[:,3]-var[:,0]),
                            var[:,4] + int_frac[:]*(var[:,7]-var[:,4]),
                            var[:,5] + int_frac[:]*(var[:,6]-var[:,5]),
                            var[:,1] + int_frac[:]*(var[:,2]-var[:,1]) ]).T
    elif 'z' in axis:
        return np.asarray([ var[:,0] + int_frac[:]*(var[:,4]-var[:,0]),
                            var[:,1] + int_frac[:]*(var[:,5]-var[:,1]),
                            var[:,2] + int_frac[:]*(var[:,6]-var[:,2]),
                            var[:,3] + int_frac[:]*(var[:,7]-var[:,3]) ]).T
    else:
        raise ValueError('Axis needs to be x y or z')

def masking_restructure(var,axis):
        # Which 4 points in the QUAD 8 series to use in what order
        if 'x' in axis:
            mask = [0,3,7,4]
        elif 'y' in axis:
            mask = [0,4,5,1]
        elif 'z' in axis:
            mask = [0,1,2,3]
        else:
            raise ValueError('Needs axis to be x y or z!')
        var_out = np.asarray([np.asarray([n1, n2, n3, n4]) for (n1,n2,n3,n4)
                              in zip(var[:,mask[0]], var[:,mask[1]], var[:,mask[2]], var[:,mask[3]])])
        return var_out

def plane_slice(x,y,z,c,zval,plane_axis='z'):
    v1, v2, vs, xyz_ref = plt_xyz(x,y,z,plane_axis)
    vs_max = np.amax(vs, axis=1)
    vs_min = np.amin(vs, axis=1)
    plane_coord = adjust_plane(vs_min,vs_max,zval)
    ind_vs = np.where((plane_coord <= vs_max) & (plane_coord >= vs_min))
    v1 = v1[ind_vs][:]
    v2 = v2[ind_vs][:]
    vs = vs[ind_vs][:]
    c = c[ind_vs][:]
    vs_max = np.amax(vs, axis=1)
    vs_min = np.amin(vs, axis=1)
    # output vals
    new_c = plane_interpolate_nodal_quad(vs_min,vs_max,plane_axis,plane_coord,c)
    v1 = masking_restructure(v1,plane_axis)
    v2 = masking_restructure(v2,plane_axis)
    vs = plane_coord * np.ones_like(v1)
    if 'x' in plane_axis:
        return vs, v1, v2, new_c
    elif 'y' in plane_axis:
        return v2, vs, v1, new_c
    elif 'z' in plane_axis:
        return v1, v2, vs, new_c




# RUn it
if dim3:
    print('3D: Testing manual plane slice')
    times_files = np.load('times_files.npy')
    times = times_files[:,0].astype(float)
    files = times_files[:,1].astype(str)
    t_steps = times_files[:,2].astype(int)
    file_names = np.unique(files)

    MF = MultiExodusReader(file_names[0])
    # print(f'Files = {file_names}')
    # print(f"times = {times}")
    for idx,t in enumerate(times):
        if idx > 5:
            break
        xi,yi,zi,ci = MF.get_data_at_time('phi',t,True)
        x,y,z,c = plane_slice(xi,yi,zi,ci,150)
        if hasattr(c[0], "__len__"):
            plotc = np.average(c, axis=1)
        else:
            plotc = c

        coords = np.asarray([ np.asarray([x_val,y_val]).T for (x_val,y_val) in zip(x,y) ])
        fig, ax = plt.subplots()
        p = PolyCollection(coords, cmap=matplotlib.cm.coolwarm, alpha=1)#,edgecolor='k'
        p.set_array(np.array(plotc) )
        ax.add_collection(p)
        ax.set_xlim([np.amin(x),np.amax(x)])
        ax.set_ylim([np.amin(y),np.amax(y)])
        ax.set_aspect('equal')
        p.set_clim(0.0, 1.0)
        fig.colorbar(p, label='phi')
        fig.savefig(f'debug_03_phi_{idx}.png',dpi=500,transparent=True)
        plt.close('all')
else:
    print("2D, skipping plane slice")


print('='*30)
print('='*30)
print('='*30)
print(' ')


# --------------------------------------------------------------------------------------
# Assuming possible 3D:
# Doing the same xplane as in my test
# --------------------------------------------------------------------------------------
def plt_xy(x,y,z,plane_axis):
    if 'x' in plane_axis:
        # db('Using y as the x-axis and z as the y-axis')
        plt_x = y
        plt_y = z
    elif 'y' in plane_axis:
        # db('Using z as the x-axis and x as the y-axis')
        plt_x = z
        plt_y = x
    elif 'z' in plane_axis:
        # db('Using x as the x-axis and y as the y-axis')
        plt_x = x
        plt_y = y
    else:
        raise ValueError('Plane measuring c on not specified!')
    return plt_x, plt_y

if dim3:
    print('3D: Testing x250 plane slice')
    times_files = np.load('times_files.npy')
    times = times_files[:,0].astype(float)
    files = times_files[:,1].astype(str)
    t_steps = times_files[:,2].astype(int)
    file_names = np.unique(files)

    MF = MultiExodusReader(file_names[0])
    # print(f'Files = {file_names}')
    # print(f"times = {times}")
    for idx,t in enumerate(times):
        if idx > 5:
            break
        xi,yi,zi,ci = MF.get_data_at_time('phi',t,True)
        x,y,z,c = plane_slice(xi,yi,zi,ci,250,'x')
        plt_x, plt_y = plt_xy(x,y,z,'x')
        if hasattr(c[0], "__len__"):
            plotc = np.average(c, axis=1)
        else:
            plotc = c

        coords = np.asarray([ np.asarray([x_val,y_val]).T for (x_val,y_val) in zip(plt_x,plt_y) ])
        fig, ax = plt.subplots()
        p = PolyCollection(coords, cmap=matplotlib.cm.coolwarm, alpha=1)#,edgecolor='k'
        p.set_array(np.array(plotc) )
        ax.add_collection(p)
        ax.set_xlim([np.amin(plt_x),np.amax(plt_x)])
        ax.set_ylim([np.amin(plt_y),np.amax(plt_y)])
        ax.set_aspect('equal')
        p.set_clim(0.0, 1.0)
        fig.colorbar(p, label='phi')
        fig.savefig(f'debug_04_phi_{idx}.png',dpi=500,transparent=True)
        plt.close('all')
else:
    print("2D, skipping plane slice")


print('='*30)
print('='*30)
print('='*30)
print(' ')
