import numpy as np
import math
from itertools import product
from itertools import combinations
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import os

def euler2quaternion(yaw, pitch, roll):
    """Convert euler angle into quaternion"""

    qx = np.cos(pitch/2.)*np.cos((yaw+roll)/2.)
    qy = np.sin(pitch/2.)*np.cos((yaw-roll)/2.)
    qz = np.sin(pitch/2.)*np.sin((yaw-roll)/2.)
    qw = np.cos(pitch/2.)*np.sin((yaw+roll)/2.)

    return [qx, qy, qz, qw]

def arrayeuler2quaternion(eangles):
    # eangles shape (..., 3) where last axis is [yaw, pitch, roll]
    yaw, pitch, roll = np.moveaxis(eangles, -1, 0)

    qx = np.cos(pitch/2.) * np.cos((yaw + roll)/2.)
    qy = np.sin(pitch/2.) * np.cos((yaw - roll)/2.)
    qz = np.sin(pitch/2.) * np.sin((yaw - roll)/2.)
    qw = np.cos(pitch/2.) * np.sin((yaw + roll)/2.)

    return np.stack([qx, qy, qz, qw], axis=-1)  # shape (..., 4)


def symquat(index, Osym = 24):
    """Convert one(index) symmetric matrix into a quaternion """

    q = np.zeros(4)

    if Osym == 24:
        SYM = np.array([[1, 0, 0,  0, 1, 0,  0, 0, 1],
                        [1, 0, 0,  0, -1, 0,  0, 0, -1],
                        [1, 0, 0,  0, 0, -1,  0, 1, 0],
                        [1, 0, 0,  0, 0, 1,  0, -1, 0],
                        [-1, 0, 0,  0, 1, 0,  0, 0, -1],
                        [-1, 0, 0,  0, -1, 0,  0, 0, 1],
                        [-1, 0, 0,  0, 0, -1,  0, -1, 0],
                        [-1, 0, 0,  0, 0, 1,  0, 1, 0],
                        [0, 1, 0, -1, 0, 0,  0, 0, 1],
                        [0, 1, 0,  0, 0, -1, -1, 0, 0],
                        [0, 1, 0,  1, 0, 0,  0, 0, -1],
                        [0, 1, 0,  0, 0, 1,  1, 0, 0],
                        [0, -1, 0,  1, 0, 0,  0, 0, 1],
                        [0, -1, 0,  0, 0, -1,  1, 0, 0],
                        [0, -1, 0, -1, 0, 0,  0, 0, -1],
                        [0, -1, 0,  0, 0, 1, -1, 0, 0],
                        [0, 0, 1,  0, 1, 0, -1, 0, 0],
                        [0, 0, 1,  1, 0, 0,  0, 1, 0],
                        [0, 0, 1,  0, -1, 0,  1, 0, 0],
                        [0, 0, 1, -1, 0, 0,  0, -1, 0],
                        [0, 0, -1,  0, 1, 0,  1, 0, 0],
                        [0, 0, -1, -1, 0, 0,  0, 1, 0],
                        [0, 0, -1,  0, -1, 0, -1, 0, 0],
                        [0, 0, -1,  1, 0, 0,  0, -1, 0]])
    elif Osym == 12:
        a = np.sqrt(3)/2
        SYM = np.array([[1,  0, 0,  0,   1, 0,  0, 0,  1],
                        [-0.5,  a, 0, -a, -0.5, 0,  0, 0,  1],
                        [-0.5, -a, 0,  a, -0.5, 0,  0, 0,  1],
                        [0.5,  a, 0, -a, 0.5, 0,  0, 0,  1],
                        [-1,  0, 0,  0,  -1, 0,  0, 0,  1],
                        [0.5, -a, 0,  a, 0.5, 0,  0, 0,  1],
                        [-0.5, -a, 0, -a, 0.5, 0,  0, 0, -1],
                        [1,  0, 0,  0,  -1, 0,  0, 0, -1],
                        [-0.5,  a, 0,  a, 0.5, 0,  0, 0, -1],
                        [0.5,  a, 0,  a, -0.5, 0,  0, 0, -1],
                        [-1,  0, 0,  0,   1, 0,  0, 0, -1],
                        [0.5, -a, 0, -a, -0.5, 0,  0, 0, -1]])

    if (1+SYM[index, 0]+SYM[index, 4]+SYM[index, 8]) > 0:
        q4 = np.sqrt(1+SYM[index, 0]+SYM[index, 4]+SYM[index, 8])/2
        q[0] = q4
        q[1] = (SYM[index, 7]-SYM[index, 5])/(4*q4)
        q[2] = (SYM[index, 2]-SYM[index, 6])/(4*q4)
        q[3] = (SYM[index, 3]-SYM[index, 1])/(4*q4)
    elif (1+SYM[index, 0]-SYM[index, 4]-SYM[index, 8]) > 0:
        q4 = np.sqrt(1+SYM[index, 0]-SYM[index, 4]-SYM[index, 8])/2
        q[0] = (SYM[index, 7]-SYM[index, 5])/(4*q4)
        q[1] = q4
        q[2] = (SYM[index, 3]+SYM[index, 1])/(4*q4)
        q[3] = (SYM[index, 2]+SYM[index, 6])/(4*q4)
    elif (1-SYM[index, 0]+SYM[index, 4]-SYM[index, 8]) > 0:
        q4 = np.sqrt(1-SYM[index, 0]+SYM[index, 4]-SYM[index, 8])/2
        q[0] = (SYM[index, 2]-SYM[index, 6])/(4*q4)
        q[1] = (SYM[index, 3]+SYM[index, 1])/(4*q4)
        q[2] = q4
        q[3] = (SYM[index, 7]+SYM[index, 5])/(4*q4)
    elif (1-SYM[index, 0]-SYM[index, 4]+SYM[index, 8]) > 0:
        q4 = np.sqrt(1-SYM[index, 0]-SYM[index, 4]+SYM[index, 8])/2
        q[0] = (SYM[index, 3]-SYM[index, 1])/(4*q4)
        q[1] = (SYM[index, 2]+SYM[index, 6])/(4*q4)
        q[2] = (SYM[index, 7]+SYM[index, 5])/(4*q4)
        q[3] = q4

    return q


def quat_Multi(q1, q2):
    """Return the product of two quaternion"""

    q = np.zeros(4)
    q[0] = q1[0]*q2[0] - q1[1]*q2[1] - q1[2]*q2[2] - q1[3]*q2[3]
    q[1] = q1[0]*q2[1] + q1[1]*q2[0] + q1[2]*q2[3] - q1[3]*q2[2]
    q[2] = q1[0]*q2[2] - q1[1]*q2[3] + q1[2]*q2[0] + q1[3]*q2[1]
    q[3] = q1[0]*q2[3] + q1[1]*q2[2] - q1[2]*q2[1] + q1[3]*q2[0]

    return q

def in_cubic_fz(m_axis):
    # print(m_axis)
    # check if the misoreintation axis in fundamental zone
    # three core axis
    axis_a = np.array([1,0,0])
    axis_b = np.array([1,1,0])/np.sqrt(2)
    axis_c = np.array([1,1,1])/np.sqrt(3)
    # if in fz
    judgement_0 = np.dot(np.cross(axis_a, axis_b), m_axis) >= 0
    judgement_1 = np.dot(np.cross(axis_b, axis_c), m_axis) >= 0
    judgement_2 = np.dot(np.cross(axis_c, axis_a), m_axis) >= 0
    # print(f"{judgement_0}, {judgement_1}, {judgement_2} = {judgement_0*judgement_1*judgement_2}")
    # print(np.sum(judgement_0*judgement_1*judgement_2))
    # print(np.squeeze(np.where(np.sum(judgement_0*judgement_1*judgement_2))))
    # print(' ')
    return judgement_0*judgement_1*judgement_2


def in_cubic_alt(m_axis):
    x, y, z = m_axis  # each is shape (2,)
    out = (0 <= z) & (z <= y) & (y <= x) & (x <= 1)
    return out


def quaternions_fz(q1, q2, symm2quat_matrix, Osym=24):
    """Return the misorientation of two quaternion"""

    q = np.zeros(4)
    misom = 2*np.pi
    axis = np.array([1, 0, 0])
    # print(f"q1: {q1}, q2: {q2}")
    for i in range(0, Osym):
        for j in range(0, Osym):
            # get misorientation quaternion q
            q1b = quat_Multi(symm2quat_matrix[i], q1)
            q2b = quat_Multi(symm2quat_matrix[j], q2)
            q2b[1:] = -q2b[1:]
            q = quat_Multi(q1b, q2b)
            # print(q)
            # get the q and inverse of q
            q_and_inverse = np.array([q,q])
            q_and_inverse[1,1:] = -q_and_inverse[1,1:]
            # get m_axis and inverse m_axis
            base = np.sqrt(1-q[0]*q[0])
            if base:
                axis_tmp = q_and_inverse[:,1:]/base
            else:
                axis_tmp = np.array([[1, 0, 0],[1, 0, 0]])
            # judge if the m_axis in fundamental zone or not
            in_cubic_fz_result = in_cubic_fz(axis_tmp.T)
            if not np.sum(in_cubic_fz_result):
                continue

            # find the index of m_axis in fundamental zone or not
            # true_index = np.squeeze(np.where(in_cubic_fz_result))
            true_index = int(np.flatnonzero(in_cubic_fz_result)[0])
            # print(f'Index: {true_index}')
            # find the minimal miso angle
            miso0 = 2*math.acos(round(q[0], 5))
            if miso0 > np.pi:
                miso0 = miso0 - 2*np.pi
            if abs(miso0) < misom:
                misom = abs(miso0)
                qmin = q_and_inverse[true_index]
                axis = axis_tmp[true_index]
            # MINE
            # print(miso0,q, axis)
    return misom, axis


def pre_operation_misorientation(Osym=24):
    # Create a quaternion matrix to show symmetry
    symm2quat_matrix = np.zeros((Osym, 4))
    for i in range(0, Osym):
        symm2quat_matrix[i, :] = symquat(i, Osym)
    return symm2quat_matrix


def para_angles(i,j,quats,symm2quat_matrix, Osym):
    qi = quats[i]
    qj = quats[j]
    theta, axis = quaternions_fz(qi, qj, symm2quat_matrix, Osym)
    # print(f'axis: {axis}')
    m_polar_angle = math.acos(round(axis[2],3))
    m_azimuth_angle = math.atan2(axis[1], axis[0]) + np.pi
    if m_polar_angle < 0 or m_polar_angle > np.pi or m_azimuth_angle < 0 or m_azimuth_angle > 2*np.pi:
        return i, j, theta, m_polar_angle, m_azimuth_angle, axis, 0
    else:
        return i, j, theta, m_polar_angle, m_azimuth_angle, axis, 1



# Osym = 24
# symm2quat_matrix = pre_operation_misorientation(Osym)
# q1 = euler2quaternion(5.454576, 2.617486, 5.379941)
# q2 = euler2quaternion(6.220170, 2.594327, 6.004859)
# print(f'q1: {q1}')
# print(f'q2: {q2}')
# print(' ')
# miso_min, axis_min = quaternions_fz(q1, q2, symm2quat_matrix, Osym=24)



# def main():
#     print("In Main")
#     Osym = 24
#     symm2quat_matrix = pre_operation_misorientation(Osym)
#     p1 = np.linspace(0,6.28,2)

#     eangles = np.array(list(product(p1,repeat=3)))
#     quat_array = arrayeuler2quaternion(eangles)
#     print(len(quat_array))
#     results = []
#     gnum = len(quat_array)
#     # for i in range((quat_array.shape[0])):
#     #     for j in range((quat_array.shape[0])):
#     #         if i != j:
#     #             results.append(para_angles(i,j,quat_array,symm2quat_matrix, Osym))
#     for i, j in tqdm(list(product(range(gnum),range(gnum))), desc="Total Progress"):
#         if i < j:
#             results.append(para_angles(i,j,quat_array,symm2quat_matrix, Osym))
#     print(results)



# PARALLEL
# globals inside each worker process
_QUATS = None
_SYMM = None
_OSYM = None

def _init_worker(quats, symm2quat_matrix, Osym):
    global _QUATS, _SYMM, _OSYM
    _QUATS = quats
    _SYMM = symm2quat_matrix
    _OSYM = Osym

def para_angles_pair(pair):
    i, j = pair
    qi = _QUATS[i]
    qj = _QUATS[j]
    theta, axis = quaternions_fz(qi, qj, _SYMM, _OSYM)

    m_polar_angle = math.acos(round(axis[2], 3))
    m_azimuth_angle = math.atan2(axis[1], axis[0]) + np.pi

    ok = 1 if (0 <= m_polar_angle <= np.pi and 0 <= m_azimuth_angle <= 2*np.pi) else 0
    return i, j, theta, m_polar_angle, m_azimuth_angle, axis[0], axis[1], axis[2], ok

def main():
    Osym = 24
    symm2quat_matrix = pre_operation_misorientation(Osym)

    n_pts = 6
    p1 = np.linspace(0, 6.28, n_pts)
    eangles = np.array(list(product(p1, repeat=3)))
    quat_array = arrayeuler2quaternion(eangles)

    gnum = len(quat_array)

    total = gnum * (gnum - 1) // 2
    workers = os.cpu_count() or 1
    chunksize = max(1, min(50, total // (workers * 50)))
    print(f"Workers: {workers}")
    print(f"Total: {total}")
    print(f"Chunksize: {chunksize}")

    pairs = combinations(range(gnum), 2)  # yields (i, j) with i<j lazily
    total = gnum * (gnum - 1) // 2

    results = []
    with ProcessPoolExecutor(
        initializer=_init_worker,
        initargs=(quat_array, symm2quat_matrix, Osym),
    ) as ex:
        for r in tqdm(ex.map(para_angles_pair, pairs, chunksize=chunksize), total=total, desc="Total Progress"):
            results.append(r)

    # SAVE DATA
    dtype = np.dtype([
        ("i", np.int32),
        ("j", np.int32),
        ("theta", np.float64),
        ("polar", np.float64),
        ("azimuth", np.float64),
        ("x", np.float64),
        ("y", np.float64),
        ("z", np.float64),
        ("ok", np.int8),
    ])
    arr = np.array(results, dtype=dtype)
    np.savez_compressed("pair_angles_"+str(n_pts)+"pt.npz", results=arr)

if __name__ == "__main__":
    main()
