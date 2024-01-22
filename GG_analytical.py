#!/usr/bin/env python3
import os
import glob
import pandas as pd
from matplotlib import pyplot as plt
import argparse
import math
import numpy as np


def gb_mobility(T):
    '''Calculate gb mobility

    Args:
        T: Temp in K

    Returns:
        GB_Mobility nm^4/(eV*s)
    '''
    M_0 = 1.4759e9 # nm4/(eV s)
    Q = 2.77 #eV
    k_B = 8.617343e-5 #eV/K
    return M_0 * math.exp(-Q / (k_B * T))

def paper_gg_findTime(D, D_0, T):
    '''Using paper equation find the GG time for given grain size avg change

    Args:
        D: Average grain size (nm)
        D_0: Average initial grain size (nm)
        T: Temperature (K)

    Returns:
        time for GG (s)
    '''
    gb_energy = 9.86 #eV/nm2
    gb_mob = gb_mobility(T) #nm^4/(eV*s)
    alpha = 0.5 #2D - see 2024-01-17 notes
    t = (D**2 - D_0**2) / (2 * alpha * gb_mob * gb_energy)
    return t

def paper_gg_D(t, D_0, T):
    '''Using paper equation find the GG new avg size for given time

    Args:
        t: time (s)
        D_0: Average initial grain size (nm)
        T: Temeprature (K)

    Returns:
        Average grain size (nm)
    '''
    gb_energy = 9.86 #eV/nm2
    gb_mob = gb_mobility(T) #nm^4/(eV*s)
    alpha = 0.5 #2D - see 2024-01-17 notes
    D2 = (2 * alpha * gb_mob * gb_energy * t) + D_0**2
    return np.sqrt(D2)

def esd(area,dim=2):
    '''Equivalent circle or sphere diameter from area or volume

    Args:
        area: Area or Volume (nm2 or nm3)
        dim: Dimensions. Defaults to 2.

    Returns:
        Diameter (nm)
    '''
    if dim == 2:
        temp = abs(area/math.pi)
        return 2 * np.sqrt(temp)



def bison_gg_findTime(D, D_0, T):
    '''Using BISON GG formula find time for given grain size change.
    It works in area so i converted it to D like the other one.
    https://mooseframework.inl.gov/bison/modules/phase_field/Grain_Growth_Model.html

    Args:
        D: Grain size (nm)
        D_0: Initial grain size (nm)
        T: Temperature (K)

    Returns:
        time (s) for GG
    '''
    gb_energy = 9.86 #eV/nm2
    gb_mob = gb_mobility(T) #nm^4/(eV*s)
    area = math.pi * (D/2)**2
    t = ((D_0/2) - area) / (2 * math.pi * gb_mob * gb_energy)
    return t

def bison_gg_D(t, D_0, T):
    '''Using BISON GG formula find grain size for given time.
    It works in area so i converted it to D like the other one.
    https://mooseframework.inl.gov/bison/modules/phase_field/Grain_Growth_Model.html

    Args:
        t: time (s)
        D_0: Initial grain size
        T: Temperature (K)

    Returns:
        Grain size (nm)
    '''
    gb_energy = 9.86 #eV/nm2
    gb_mob = gb_mobility(T) #nm^4/(eV*s)
    area = (D_0/2) - ( 2 * math.pi * gb_mob * gb_energy * t)
    return esd(area)


def ainscough_gg_findTime(D, D_0, T):
    '''Using whay might actually be the BISON GG formula find time for given grain size change.
    From grainradiusaux (https://mooseframework.inl.gov/bison/source/auxkernels/GrainRadiusAux.html)
    https://doi.org/10.1016/0022-3115(73)90001-9

    Args:
        D: Grain size (nm)
        D_0: Initial grain size (nm)
        T: Temperature (K)

    Returns:
        time (hour)
    '''
    D = D * 1e-3
    D_0 = D_0 * 1e-3
    k = 5.24e7 * math.exp(-2.67e5 / (8.314 * T)) # um2/hour
    # k_nms = k * 3600 * 1e6 #nm2/s
    Dm = 2.23e3 * math.exp(-7620 / T) #um * 1e3 #um -> nm
    kt = Dm * (D_0 - D) + (Dm**2)*np.log(((Dm - D_0)/(Dm - D)))
    return kt/k


if __name__ == "__main__":
    print("__main__ Start")
    T = 1600
    t = np.linspace(0,50000,10000)
    D_0 = 8000 #nm
    D = np.linspace(8000, 12000, 1000)

    paper_t = paper_gg_findTime(D, D_0, T)
    bison_t = bison_gg_findTime(D, D_0, T)
    ains_t = ainscough_gg_findTime(D, D_0, T) * 3600

    paper_D = paper_gg_D(t,D_0,T)
    bison_D = bison_gg_D(t,D_0,T)


    plt.figure(1)
    plt.plot(D/1000,paper_t,label='Analytical')
    plt.plot(D/1000,bison_t,label='BISON')
    plt.plot(D/1000,ains_t,label='BISON- Ainscough')
    plt.xlabel('Grain Size (um)')
    plt.ylabel('Time (s)')
    plt.legend()
    plt.savefig('P01_timefromsize.png',transparent=True)

    plt.figure(2)
    plt.plot(t,paper_D,label='Analytical')
    plt.plot(t,bison_D,label='BISON')
    plt.ylabel('Grain Size (um)')
    plt.xlabel('Time (s)')
    plt.legend()
    plt.savefig('P02_sizefromtime.png',transparent=True)

    plt.show()
