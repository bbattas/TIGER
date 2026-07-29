# =======================================================================================
# IMPORTS
# =======================================================================================
import os
current_path = os.getcwd()  # Get current working directory
import numpy as np
from numpy import seterr
seterr(all='raise')  # Set numpy to raise exceptions on all floating point errors
import matplotlib.pyplot as plt
from tqdm import tqdm  # Progress bar for loops
import sys

# Add necessary paths for importing custom modules
# sys.path.append(current_path)  # Current directory
# sys.path.append(current_path+'/../../')  # Parent directory for main modules
import myInput  # Custom input handling module
import PACKAGE_MP_Linear as linear2d  # 2D linear multi-physics package
# import post_processing  # Post-processing utilities
# sys.path.append(current_path+'/../../examples/calculate_tangent/')  # Path for tangent calculation utilities
import logging
import argparse
from numba import njit, prange
import multiprocessing as mp
import csv

def parse_args():
    VALID_CPUS = (1, 2, 4, 8, 16, 32, 64, 128)
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose (DEBUG) logging"
    )
    parser.add_argument(
        "-n", "--cpus", type=int, default=4, choices=VALID_CPUS,
        help="Number of CPUs for smoothing/curvature calculations."
    )
    parser.add_argument(
        "-i", "--input", type=str, required=True,
        help="Path to the input .npy microstructure file."
    )
    parser.add_argument(
        "--case", type=str, default=None,
        help="Case name for output labeling. Defaults to the input filename stem."
    )
    parser.add_argument(
        "--grain-count", type=int, default=256,
        help="Total number of grains in the system."
    )
    parser.add_argument(
        "--frames", type=int, default=121,
        help="Number of time steps to analyze."
    )
    parser.add_argument(
        "--time-interval", type=int, default=30,
        help="Time interval between steps for velocity calculation."
    )
    parser.add_argument(
        "--min-area", type=int, default=50,
        help="Minimum GB area threshold for velocity analysis."
    )
    parser.add_argument(
        "--window-size", type=int, default=5,
        help="Sliding window size for anti-curvature filtering."
    )

    return parser.parse_args()

def setup_logger(verbose: bool) -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(message)s"
    )
    return logging.getLogger(__name__)


# =======================================================================================
# Functions via Lin
# =======================================================================================

# Optimized function to calculate volume change (dV) between time steps
@njit(parallel=True)
def compute_dV(npy_file_aniso_current, npy_file_aniso_next, pair_id_pair):
    """
    Calculate net volume change for a grain boundary between two time steps.

    Parameters:
    - npy_file_aniso_current: microstructure at current time step
    - npy_file_aniso_next: microstructure at next time step
    - pair_id_pair: [grain_id1, grain_id2] for the grain boundary

    Returns:
    - Net volume change (positive = grain1 growing into grain2)
    """
    # Count voxels where grain1 grows into grain2
    growth_direction1 = (npy_file_aniso_current == pair_id_pair[0]) & (npy_file_aniso_next == pair_id_pair[1])
    # Count voxels where grain2 grows into grain1
    growth_direction2 = (npy_file_aniso_current == pair_id_pair[1]) & (npy_file_aniso_next == pair_id_pair[0])
    return np.sum(growth_direction1) - np.sum(growth_direction2)

# Extended version that returns individual directional growth components
@njit(parallel=True)
def compute_dV_split(npy_file_aniso_current, npy_file_aniso_next, pair_id_pair):
    """
    Calculate volume change with separated directional components.

    Returns:
    - Net volume change, growth_direction1, growth_direction2
    """
    growth_direction1 = (npy_file_aniso_current == pair_id_pair[0]) & (npy_file_aniso_next == pair_id_pair[1])
    growth_direction2 = (npy_file_aniso_current == pair_id_pair[1]) & (npy_file_aniso_next == pair_id_pair[0])
    return np.sum(growth_direction1) - np.sum(growth_direction2), np.sum(growth_direction1), np.sum(growth_direction2)

def compute_necessary_info(key, time_interval,
                            GB_infomation_dict_list_one_step_one_key,
                            GBenergy_information_dict_list_one_step_one_key,
                            npy_file_aniso_current,
                            npy_file_aniso_next):
    """
    Compute GB velocity, curvature, energy, and anti-curvature behavior for one GB.

    Parameters:
    - key: GB identifier
    - time_interval: time between steps
    - GB_infomation_dict_list_one_step_one_key: GB curvature info
    - GBenergy_information_dict_list_one_step_one_key: GB energy info
    - npy_file_aniso_current/next: microstructures at current and next time steps

    Returns:
    - Dictionary with velocity, curvature, energy, and anti-curvature flag
    """
    # Calculate volume change and convert to velocity
    dV = compute_dV(npy_file_aniso_current, npy_file_aniso_next, GB_infomation_dict_list_one_step_one_key[6:8])
    velocity = dV / time_interval / (GB_infomation_dict_list_one_step_one_key[5] / 2)  # Normalize by area and time

    # Extract curvature and energy values
    current_curvature_value = GB_infomation_dict_list_one_step_one_key[4]
    current_eng = GBenergy_information_dict_list_one_step_one_key[4]

    result = {
        "key": key,
        "velocity": velocity,
        "current_curvature_value": current_curvature_value,
        "current_eng": current_eng,
        "is_anti_curvature": current_curvature_value * velocity < 0  # Anti-curvature: opposite signs
    }

    return result

def compute_necessary_info_split(key, time_interval,
                                GB_infomation_dict_list_one_step_one_key,
                                GBenergy_information_dict_list_one_step_one_key,
                                npy_file_aniso_current,
                                npy_file_aniso_next):
    """
    Extended version that includes directional growth components.
    """
    # Calculate volume change with directional split
    dV, dV_direction1, dV_direction2 = compute_dV_split(npy_file_aniso_current, npy_file_aniso_next,
                                                        GB_infomation_dict_list_one_step_one_key[6:8])
    velocity = dV / time_interval / (GB_infomation_dict_list_one_step_one_key[5] / 2)

    current_curvature_value = GB_infomation_dict_list_one_step_one_key[4]
    current_eng = GBenergy_information_dict_list_one_step_one_key[4]

    result = {
        "key": key,
        "velocity": velocity,
        "dV_direction1": dV_direction1,  # Growth of grain1 into grain2
        "dV_direction2": dV_direction2,  # Growth of grain2 into grain1
        "current_curvature_value": current_curvature_value,
        "current_eng": current_eng,
        "is_anti_curvature": current_curvature_value * velocity < 0
    }

    return result


def post_processing_get_line(i, j):
    """Get the row order of grain i and grain j in MisoEnergy.txt (i < j)
    This function calculates the index in a triangular matrix representation for grain boundary properties
    between two grains i and j.

    Args:
        i (int): First grain ID
        j (int): Second grain ID
    Returns:
        int: Index in triangular matrix representation
    """
    if i < j: return int(i+(j-1)*(j)/2)
    else: return int(j+(i-1)*(i)/2)


if __name__ == '__main__':
    args = parse_args()
    log = setup_logger(args.verbose)

    # Derive case name from input filename stem if not provided
    case_name = args.case if args.case is not None else os.path.splitext(os.path.basename(args.input))[0]

    # Filters
    CPUS          = args.cpus
    WINDOW_SIZE   = args.window_size
    GRAIN_COUNT   = args.grain_count
    FRAMES        = args.frames
    TIME_INTERVAL = args.time_interval
    MIN_CURVE     = 0
    MIN_AREA      = args.min_area
    TJD = 6

    data_file_folder = f"data/{case_name}/"

    npy_file_name_aniso = args.input
    npy_file_aniso = np.load(npy_file_name_aniso)
    npy_file_aniso = npy_file_aniso.astype(int)
    log.warning(f"The {case_name} data size is: {npy_file_aniso.shape}")
    log.warning("READING DATA DONE")

    # ISO energy is constant so we dont have to figure out how to build this yet
    npy_file_aniso_energy = np.ones_like(npy_file_aniso) # Load energy evolution
    log.warning(f"The {case_name} data size is: {npy_file_aniso_energy.shape}")
    log.warning("READING ENERGY DATA DONE")

    # Extract dimensions: time steps, spatial dimensions
    step_num, size_x, size_y, size_z = npy_file_aniso.shape

    # Make folder:
    os.makedirs(data_file_folder, exist_ok=True)


    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # =======================================================================================
    # GB Curvature Calculation
    # =======================================================================================
    log.warning(' ')
    log.warning('CURVATURE:')
    # Initialize parameters for GB analysis
    step_num = FRAMES #121  # Number of time steps to analyze
    grain_nums = GRAIN_COUNT #20000  # Total number of grains in the system
    GB_infomation_dict_list = []  # List to store GB information for each time step
    curvature_matrix_list = []  # List to store curvature matrices

    # Process each time step to extract GB information and calculate curvature
    for time_step in tqdm(range(step_num)):
        # Define file names for cached data to avoid recomputation
        data_file_name = f"2D_gb_dict_step{time_step}.npz"
        data_curvature_file_name = f"2D_curvature_step{time_step}.npz"

        # Try to load pre-computed data if available
        if os.path.exists(data_file_folder + data_file_name):
            npz_file = np.load(data_file_folder + data_file_name, allow_pickle=True)
            GB_infomation_dict = npz_file["GB_infomation_dict"]
            GB_infomation_dict_list.append(GB_infomation_dict.item())
        else:
            log.info(' Pre computed curvature does not exist, calculating.')
            os.makedirs(os.path.dirname(data_file_folder + data_file_name), exist_ok=True)
            # If pre-computed data doesn't exist, calculate from scratch
            current_microstructure = npy_file_aniso[time_step]

            # Calculate signed curvature using the linear solver
            if os.path.exists(data_file_folder + data_curvature_file_name):
                # Load pre-computed curvature matrix
                npz_file_curvature = np.load(data_file_folder + data_curvature_file_name)
                curvature_matrix = npz_file_curvature["curvature_matrix"]
                curvature_matrix_list.append(curvature_matrix)
            else:
                # Calculate curvature using the linear multi-physics solver
                cores = CPUS #128  # Number of CPU cores for parallel processing
                loop_times = WINDOW_SIZE  # Number of iterations for convergence
                R = np.zeros((size_x, size_y, 3))  # Initialize reference array

                # Create smoothing class instance for curvature calculation
                smoothing_class = linear2d.linear_class(
                    size_x, size_y, grain_nums, cores, loop_times,
                    current_microstructure[:, :, 0], R,
                    verification_system=False,
                    curvature_sign=True,  # Calculate signed curvature
                    id_offset=1 # FOR MCP WHERE WE SHIFT minimum to 1 instead of 0
                )
                smoothing_class.linear_main("curvature")  # Run curvature calculation
                C_ln = smoothing_class.get_C()  # Get curvature results
                curvature_matrix = C_ln[1, :]  # Extract curvature matrix
                curvature_matrix_list.append(curvature_matrix)
                # save curvature information
                np.savez(data_file_folder + data_curvature_file_name, curvature_matrix=curvature_matrix)
            log.info("Finish curvature calculation")


            # =======================================================================================
            # TRIPLE JUNCTION (TJ) AND QUADRUPLE JUNCTION (QJ) DETECTION
            # =======================================================================================

            # Identify triple and quadruple junctions to exclude them from GB analysis
            TJ_infomation_dict = dict()
            for index, result in np.ndenumerate(current_microstructure):
                i, j, k = index
                # Get periodic boundary conditions for neighbors
                ip, im, jp, jm = myInput.periodic_bc(size_x, size_y, i, j)

                # Check if current voxel is at a grain boundary
                if (((current_microstructure[ip, j, k] - current_microstructure[i, j, k]) != 0) or
                    ((current_microstructure[im, j, k] - current_microstructure[i, j, k]) != 0) or
                    ((current_microstructure[i, jp, k] - current_microstructure[i, j, k]) != 0) or
                    ((current_microstructure[i, jm, k] - current_microstructure[i, j, k]) != 0)):

                    central_site = int(current_microstructure[i, j, k])

                    # Get all neighboring grain IDs
                    neighboring_sites_list = np.array([
                        current_microstructure[ip, j, k], current_microstructure[i, jp, k],
                        current_microstructure[im, j, k], current_microstructure[i, jm, k]
                    ]).astype(int)

                    # Find unique neighboring grain IDs
                    neighboring_sites_set = set(neighboring_sites_list)
                    if central_site in neighboring_sites_set:
                        neighboring_sites_set.remove(central_site)  # Remove central grain ID
                    neighboring_sites_list_unque = list(neighboring_sites_set)

                    # If more than 1 neighboring grain, this is a junction point
                    if len(neighboring_sites_list_unque) > 1:
                        for m in range(len(neighboring_sites_list_unque)):
                            pair_id = post_processing_get_line(central_site, neighboring_sites_list_unque[m])
                            if pair_id in TJ_infomation_dict:
                                tmp = TJ_infomation_dict[pair_id]
                                tmp.append([i, j, k])
                                TJ_infomation_dict[pair_id] = tmp
                            else:
                                TJ_infomation_dict[pair_id] = [[i, j, k]]
            log.info("Finish TJ extraction")

            # =======================================================================================
            # GRAIN BOUNDARY CENTER CALCULATION
            # =======================================================================================

            # Calculate the center position and properties of each grain boundary
            GB_infomation_dict = dict()
            TJ_distance_max = TJD #6  # Maximum distance from TJ to exclude (in voxels)

            for index, result in np.ndenumerate(current_microstructure):
                i, j, k = index
                ip, im, jp, jm = myInput.periodic_bc(size_x, size_y, i, j)

                # Check if current voxel is at a grain boundary
                if (((current_microstructure[ip, j, k] - current_microstructure[i, j, k]) != 0) or
                    ((current_microstructure[im, j, k] - current_microstructure[i, j, k]) != 0) or
                    ((current_microstructure[i, jp, k] - current_microstructure[i, j, k]) != 0) or
                    ((current_microstructure[i, jm, k] - current_microstructure[i, j, k]) != 0)):

                    central_site = int(current_microstructure[i, j, k])
                    neighboring_sites_list = np.array([
                        current_microstructure[ip, j, k], current_microstructure[i, jp, k],
                        current_microstructure[im, j, k], current_microstructure[i, jm, k]
                    ]).astype(int)

                    neighboring_sites_set = set(neighboring_sites_list)
                    if central_site in neighboring_sites_set:
                        neighboring_sites_set.remove(central_site)
                    neighboring_sites_list_unque = list(neighboring_sites_set)

                    # Skip junction points (more than 2 grains meeting)
                    if len(neighboring_sites_list_unque) > 1:
                        continue

                    # Check distance from triple junctions to exclude nearby voxels
                    break_point = 0
                    pair_id = post_processing_get_line(central_site, neighboring_sites_list_unque[0])
                    if central_site < neighboring_sites_list_unque[0]:
                        pair_id_pair = [central_site, neighboring_sites_list_unque[0]]
                    else:
                        pair_id_pair = [neighboring_sites_list_unque[0], central_site]

                    # Initialize or update GB information
                    if pair_id in GB_infomation_dict:
                        GB_infomation_dict[pair_id][5] += 1  # Increment GB area count
                    else:
                        # [GB_count, sum_i, sum_j, sum_k, sum_curvature, area, grain_id1, grain_id2]
                        GB_infomation = np.array([0, 0, 0, 0, 0.0, 1, pair_id_pair[0], pair_id_pair[1]])
                        GB_infomation_dict[pair_id] = GB_infomation

                    # Check if this voxel is too close to a triple junction
                    if pair_id in TJ_infomation_dict:
                        for TJ_site in TJ_infomation_dict[pair_id]:
                            TJ_distance = np.linalg.norm(index - np.array(TJ_site))
                            if TJ_distance < TJ_distance_max:
                                break_point = 1
                                break
                    if break_point == 1:
                        continue

                    # Accumulate GB properties (position and curvature)
                    GB_infomation_dict[pair_id][0] += 1  # Increment valid voxel count
                    if central_site == pair_id_pair[0]:
                        # Add position and curvature with correct sign
                        GB_infomation_dict[pair_id][1:5] += np.array([i, j, k, curvature_matrix[i, j]])
                    else:
                        # Flip curvature sign for opposite grain orientation
                        GB_infomation_dict[pair_id][1:5] += np.array([i, j, k, -curvature_matrix[i, j]])


            # =============================================================================
            # POST-PROCESSING: REMOVE SMALL GBs AND CALCULATE AVERAGES
            # =============================================================================

            # Remove grain boundaries with no valid voxels after TJ filtering
            small_GB_list = []
            for key in GB_infomation_dict:
                if GB_infomation_dict[key][0] == 0:
                    small_GB_list.append(key)
            for s_index in range(len(small_GB_list)):
                GB_infomation_dict.pop(small_GB_list[s_index])

            # Calculate average position and curvature for each GB
            for key in GB_infomation_dict:
                GB_infomation_dict[key][1:5] = GB_infomation_dict[key][1:5] / GB_infomation_dict[key][0]

            log.info(f"Current GBs len at {time_step} steps: {len(GB_infomation_dict)}")

            # Save computed information for future use
            np.savez(data_file_folder + data_file_name, GB_infomation_dict=GB_infomation_dict)
            GB_infomation_dict_list.append(GB_infomation_dict)




    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # =============================================================================
    # GRAIN BOUNDARY ENERGY CALCULATION
    # =============================================================================
    log.warning(' ')
    log.warning('GB ENERGY CALC')
    # Similar to curvature calculation, but focused on extracting GB energy information
    step_num = FRAMES #121
    grain_nums = GRAIN_COUNT #20000
    GBenergy_information_dict_list = []  # Store GB energy information for each time step

    for time_step in tqdm(range(step_num)):
        data_file_name_GBenergy = f"2D_gbe_step{time_step}.npz"

        # Try to load pre-computed energy data
        if os.path.exists(data_file_folder + data_file_name_GBenergy):
            npz_file = np.load(data_file_folder + data_file_name_GBenergy, allow_pickle=True)
            GBenergy_information_dict = npz_file["GBenergy_information_dict"]
            GBenergy_information_dict_list.append(GBenergy_information_dict.item())
        else:
            log.info(' Pre computed GBE dict does not exist, calculating.')
            # Calculate energy information from scratch
            os.makedirs(os.path.dirname(data_file_folder + data_file_name_GBenergy), exist_ok=True)
            current_microstructure = npy_file_aniso[time_step]
            current_energy = npy_file_aniso_energy[time_step]  # Load corresponding energy data

            # Re-extract triple junction information for energy calculation
            TJ_infomation_dict = dict()
            for index, result in np.ndenumerate(current_microstructure):
                i, j, k = index
                ip, im, jp, jm = myInput.periodic_bc(size_x, size_y, i, j)

                if (((current_microstructure[ip, j, k] - current_microstructure[i, j, k]) != 0) or
                    ((current_microstructure[im, j, k] - current_microstructure[i, j, k]) != 0) or
                    ((current_microstructure[i, jp, k] - current_microstructure[i, j, k]) != 0) or
                    ((current_microstructure[i, jm, k] - current_microstructure[i, j, k]) != 0)):

                    central_site = int(current_microstructure[i, j, k])
                    neighboring_sites_list = np.array([
                        current_microstructure[ip, j, k], current_microstructure[i, jp, k],
                        current_microstructure[im, j, k], current_microstructure[i, jm, k]
                    ]).astype(int)

                    neighboring_sites_set = set(neighboring_sites_list)
                    if central_site in neighboring_sites_set:
                        neighboring_sites_set.remove(central_site)
                    neighboring_sites_list_unque = list(neighboring_sites_set)

                    # Identify triple and quadruple junctions
                    if len(neighboring_sites_list_unque) > 1:
                        for m in range(len(neighboring_sites_list_unque)):
                            pair_id = post_processing_get_line(central_site, neighboring_sites_list_unque[m])
                            if pair_id in TJ_infomation_dict:
                                tmp = TJ_infomation_dict[pair_id]
                                tmp.append([i, j, k])
                                TJ_infomation_dict[pair_id] = tmp
                            else:
                                TJ_infomation_dict[pair_id] = [[i, j, k]]
            log.info("Finish TJ extraction")

            # Calculate GB energy information
            GBenergy_information_dict = dict()
            TJ_distance_max = 6  # Maximum distance from TJ to exclude (in voxels)

            for index, result in np.ndenumerate(current_microstructure):
                i, j, k = index
                ip, im, jp, jm = myInput.periodic_bc(size_x, size_y, i, j)

                if (((current_microstructure[ip, j, k] - current_microstructure[i, j, k]) != 0) or
                    ((current_microstructure[im, j, k] - current_microstructure[i, j, k]) != 0) or
                    ((current_microstructure[i, jp, k] - current_microstructure[i, j, k]) != 0) or
                    ((current_microstructure[i, jm, k] - current_microstructure[i, j, k]) != 0)):

                    central_site = int(current_microstructure[i, j, k])
                    neighboring_sites_list = np.array([
                        current_microstructure[ip, j, k], current_microstructure[i, jp, k],
                        current_microstructure[im, j, k], current_microstructure[i, jm, k]
                    ]).astype(int)

                    neighboring_sites_set = set(neighboring_sites_list)
                    if central_site in neighboring_sites_set:
                        neighboring_sites_set.remove(central_site)
                    neighboring_sites_list_unque = list(neighboring_sites_set)

                    # Count the number of different neighboring grains in extended neighborhood
                    num_other_sites = 0
                    neighboring_sites_full_list = np.array([
                        current_microstructure[ip, jp, k], current_microstructure[ip, j, k],
                        current_microstructure[ip, jm, k], current_microstructure[i, jp, k],
                        current_microstructure[i, jm, k], current_microstructure[im, jp, k],
                        current_microstructure[im, j, k], current_microstructure[im, jm, k]
                    ]).astype(int)

                    for neigh_site in neighboring_sites_full_list:
                        if neigh_site != central_site:
                            num_other_sites += 1  # Count sites different from central grain

                    # Skip junction points
                    if len(neighboring_sites_list_unque) > 1:
                        continue

                    # Check distance from triple junctions
                    break_point = 0
                    pair_id = post_processing_get_line(central_site, neighboring_sites_list_unque[0])
                    if central_site < neighboring_sites_list_unque[0]:
                        pair_id_pair = [central_site, neighboring_sites_list_unque[0]]
                    else:
                        pair_id_pair = [neighboring_sites_list_unque[0], central_site]

                    # Initialize or update GB energy information
                    if pair_id in GBenergy_information_dict:
                        GBenergy_information_dict[pair_id][5] += 1  # Increment area count
                    else:
                        # [GB_count, sum_i, sum_j, sum_k, sum_energy, area, grain_id1, grain_id2]
                        GB_infomation = np.array([0, 0, 0, 0, 0.0, 1, pair_id_pair[0], pair_id_pair[1]])
                        GBenergy_information_dict[pair_id] = GB_infomation

                    # Check distance from triple junctions
                    if pair_id in TJ_infomation_dict:
                        for TJ_site in TJ_infomation_dict[pair_id]:
                            TJ_distance = np.linalg.norm(index - np.array(TJ_site))
                            if TJ_distance < TJ_distance_max:
                                break_point = 1
                                break
                    if break_point == 1:
                        continue

                    # Accumulate GB energy and position information
                    GBenergy_information_dict[pair_id][0] += 1  # Increment valid voxel count
                    # Add position and normalized energy (energy per neighboring grain)
                    GBenergy_information_dict[pair_id][1:5] += np.array([i, j, k, current_energy[i, j, k] / num_other_sites])

            # Remove small GBs and calculate averages
            small_GB_list = []
            for key in GBenergy_information_dict:
                if GBenergy_information_dict[key][0] == 0:
                    small_GB_list.append(key)
            for s_index in range(len(small_GB_list)):
                GBenergy_information_dict.pop(small_GB_list[s_index])

            # Calculate average position and energy for each GB
            for key in GBenergy_information_dict:
                GBenergy_information_dict[key][1:5] = GBenergy_information_dict[key][1:5] / GBenergy_information_dict[key][0]

            log.info(f"Current GBs len at {time_step} steps: {len(GBenergy_information_dict)}")

            # Save energy information
            np.savez(data_file_folder + data_file_name_GBenergy, GBenergy_information_dict=GBenergy_information_dict)
            GBenergy_information_dict_list.append(GBenergy_information_dict)


    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # =============================================================================
    # Main Velocity Loop
    # =============================================================================
    log.warning(' ')
    log.warning('VELOCITY (includes 00100 window filter)')
    # Analysis parameters
    time_interval = TIME_INTERVAL  # Time interval between steps
    curvature_limit = MIN_CURVE  # Minimum curvature threshold for analysis
    area_limit = MIN_AREA  # Minimum GB area threshold

    # Initialize storage arrays for all GB data
    GB_list_velocity_list = []  # Store velocities for each time step
    GB_list_curvature_list = []  # Store curvatures for each time step
    GB_list_GBenergy_list = []  # Store energies for each time step
    GB_list_area_list = []  # Store areas for each time step
    GB_list_dV_direction1 = []  # Store directional growth data
    GB_list_dV_direction2 = []

    # Track anti-curvature GBs specifically
    GB_id_focus = dict()  # Count of anti-curvature occurrences per GB
    GB_id_focus_detail = dict()  # Detailed time step records per GB
    GB_filter_kernel = [set(), set(), set(), set(), set()]  # 5-step sliding window filter
    dV_dict = []  # Store all computed results

    # Process each time step to calculate velocities and identify anti-curvature behavior
    for time_step in range(step_num):
        if time_step + 1 >= step_num:
            continue  # Skip last step (no next step for velocity calculation)

        # Initialize temporary storage for current time step
        GB_list_velocity_list_tmp = []
        GB_list_curvature_list_tmp = []
        GB_list_GBenergy_list_tmp = []
        GB_list_area_list_tmp = []
        GB_list_dV_direction1_tmp = []
        GB_list_dV_direction2_tmp = []
        key_set = set()

        # =============================================================================
        # FILTERING: Remove unsuitable GBs for analysis
        # =============================================================================

        # log.info(f"key num start with {len(GB_infomation_dict_list[time_step])}")
        # tmp_GB_infomation_dict_for_mp = {}
        # tmp_GBenergy_information_dict_for_mp = {}

        # # Filter GBs based on area, persistence, and curvature criteria
        # for key in GB_infomation_dict_list[time_step]:
        #     if (GB_infomation_dict_list[time_step][key][5] < area_limit or  # Too small area
        #         GB_infomation_dict_list[time_step + 1].get(key) is None or  # Disappears in next step
        #         abs(GB_infomation_dict_list[time_step][key][4]) < curvature_limit):  # Too low curvature
        #         continue
        #     tmp_GB_infomation_dict_for_mp[key] = GB_infomation_dict_list[time_step][key]
        #     tmp_GBenergy_information_dict_for_mp[key] = GBenergy_information_dict_list[time_step][key]

        # log.info(f"key num end with {len(tmp_GB_infomation_dict_for_mp)}")

        log.info(f"key num start with {len(GB_infomation_dict_list[time_step])}")
        # Diagnostic counters
        filtered_small_area = 0
        filtered_no_next_step = 0
        filtered_low_curvature = 0

        tmp_GB_infomation_dict_for_mp = {}
        tmp_GBenergy_information_dict_for_mp = {}

        for key in GB_infomation_dict_list[time_step]:
            area_val = GB_infomation_dict_list[time_step][key][5]
            curv_val = abs(GB_infomation_dict_list[time_step][key][4])
            next_exists = GB_infomation_dict_list[time_step + 1].get(key) is not None

            if area_val < area_limit:
                filtered_small_area += 1
                continue
            if not next_exists:
                filtered_no_next_step += 1
                continue
            if curv_val < curvature_limit:
                filtered_low_curvature += 1
                continue

            tmp_GB_infomation_dict_for_mp[key] = GB_infomation_dict_list[time_step][key]
            tmp_GBenergy_information_dict_for_mp[key] = GBenergy_information_dict_list[time_step][key]

        log.info(f"  filtered by small area (<{area_limit}): {filtered_small_area}")
        log.info(f"  filtered by no next step: {filtered_no_next_step}")
        log.info(f"  filtered by low curvature (<{curvature_limit}): {filtered_low_curvature}")
        log.info(f"key num end with {len(tmp_GB_infomation_dict_for_mp)}")
        log.info(' ')

        # =============================================================================
        # VELOCITY CALCULATION WITH CACHING
        # =============================================================================

        # Try to load pre-computed velocity data
        dV_dict_file_name = f"2D_dV_split_gbLimit{area_limit}_step{time_step}.npz"
        if os.path.exists(data_file_folder + dV_dict_file_name):
            npz_file = np.load(data_file_folder + dV_dict_file_name, allow_pickle=True)
            dV_dict_tmp = npz_file["dV_dict_tmp"]
            dV_dict_tmp = dV_dict_tmp.item()
        else:
            log.info(' Pre computed velocity data does not exist, calculating.')
            os.makedirs(os.path.dirname(data_file_folder + dV_dict_file_name), exist_ok=True)
            dV_dict_tmp = {}

        # Calculate or retrieve velocity data for each GB
        for key in tqdm(tmp_GB_infomation_dict_for_mp):
            if key in dV_dict_tmp:
                result = dV_dict_tmp[key]  # Use cached result
            else:
                # Compute new result
                result = compute_necessary_info_split(
                    key, time_interval,
                    tmp_GB_infomation_dict_for_mp[key],
                    tmp_GBenergy_information_dict_for_mp[key],
                    npy_file_aniso[time_step],
                    npy_file_aniso[time_step + 1]
                )
                dV_dict_tmp[key] = result

            # Store results for all GBs
            GB_list_velocity_list_tmp.append(result["velocity"])
            GB_list_curvature_list_tmp.append(result["current_curvature_value"])
            GB_list_GBenergy_list_tmp.append(result["current_eng"])
            GB_list_dV_direction1_tmp.append(result["dV_direction1"])
            GB_list_dV_direction2_tmp.append(result["dV_direction2"])

            # Calculate and store GB area
            current_GB_area = 0.5 * tmp_GB_infomation_dict_for_mp[key][0]
            GB_list_area_list_tmp.append(current_GB_area)

            # =============================================================================
            # ANTI-CURVATURE TRACKING
            # =============================================================================

            # Track GBs showing anti-curvature behavior
            if result["is_anti_curvature"]:
                key_set.add(key)
                if time_step >= step_num - 3:
                    continue  # Skip near end of simulation
                if key in GB_id_focus:
                    GB_id_focus[key] += 1  # Increment anti-curvature count
                    GB_id_focus_detail[key].append(time_step)
                else:
                    GB_id_focus[key] = 1  # First anti-curvature occurrence
                    GB_id_focus_detail[key] = [time_step]

        # Save computed data if not already cached
        if not os.path.exists(data_file_folder + dV_dict_file_name):
            np.savez(data_file_folder + dV_dict_file_name, dV_dict_tmp=dV_dict_tmp)
            # pass

        # Store results for current time step
        dV_dict.append(dV_dict_tmp)
        GB_list_velocity_list.append(GB_list_velocity_list_tmp)
        GB_list_curvature_list.append(GB_list_curvature_list_tmp)
        GB_list_GBenergy_list.append(GB_list_GBenergy_list_tmp)
        GB_list_area_list.append(GB_list_area_list_tmp)
        GB_list_dV_direction1.append(GB_list_dV_direction1_tmp)
        GB_list_dV_direction2.append(GB_list_dV_direction2_tmp)

        # =============================================================================
        # SLIDING WINDOW FILTER FOR ANTI-CURVATURE BEHAVIOR
        # =============================================================================

        # Update sliding window filter (5-step window)
        GB_filter_kernel[0:4] = GB_filter_kernel[1:]
        GB_filter_kernel[4] = key_set

        # Apply "00100" filter: anti-curvature at step 2, normal at steps 0,1,3,4
        # This filters out random/transient anti-curvature behavior
        filtered_set = GB_filter_kernel[2] - (GB_filter_kernel[0] | GB_filter_kernel[1] |
                                             GB_filter_kernel[3] | GB_filter_kernel[4])

        # Remove filtered GBs from tracking
        for key in filtered_set:
            GB_id_focus[key] -= 1
            GB_id_focus_detail[key].remove(time_step - 2)

        log.info(f"finish {time_step} with num of GBs {len(GB_list_velocity_list_tmp)}, and collected num of GBs {len(GB_id_focus)}")
        log.info(' ')

    # =============================================================================
    # FINAL FILTERING: Remove GBs with insufficient anti-curvature behavior
    # =============================================================================

    # Remove GBs that don't show consistent anti-curvature behavior
    GB_id_focus_copy = GB_id_focus.copy()
    for key in GB_id_focus_copy:
        if GB_id_focus_copy[key] <= 0:
            GB_id_focus.pop(key)
            GB_id_focus_detail.pop(key)




    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # =============================================================================
    # EXTRACT ANTI-CURVATURE AND NORMAL-CURVATURE GB DATA
    # =============================================================================

    # Initialize storage arrays for anti-curvature GBs
    GB_antic_list_velocity_AllList = []  # Velocity data for each time step
    GB_antic_list_curvature_AllList = []  # Curvature data for each time step
    GB_antic_list_GBenergy_AllList = []  # Energy data for each time step
    GB_antic_list_anticNum_AllList = []  # Anti-curvature occurrence count for each time step
    GB_antic_list_area_AllList = []  # Area data for each time step
    GB_antic_list_dV_normD_AllList = []  # Normal direction growth for each time step
    GB_antic_list_dV_antiD_AllList = []  # Anti-curvature direction growth for each time step

    # Flattened arrays combining all time steps for anti-curvature GBs
    GB_antic_list_velocity_list = []
    GB_antic_list_curvature_list = []
    GB_antic_list_GBenergy_list = []
    GB_antic_list_anticNum_list = []
    GB_antic_list_area_list = []
    GB_antic_list_dV_normD_list = []
    GB_antic_list_dV_antiD_list = []

    # Initialize storage arrays for normal-curvature GBs
    GB_normc_list_velocity_AllList = []
    GB_normc_list_curvature_AllList = []
    GB_normc_list_GBenergy_AllList = []
    GB_normc_list_area_AllList = []
    GB_normc_list_dV_normD_AllList = []
    GB_normc_list_dV_antiD_AllList = []

    # Flattened arrays for normal-curvature GBs
    GB_normc_list_velocity_list = []
    GB_normc_list_curvature_list = []
    GB_normc_list_GBenergy_list = []
    GB_normc_list_area_list = []
    GB_normc_list_dV_normD_list = []
    GB_normc_list_dV_antiD_list = []

    # Process each time step to extract anti-curvature and normal-curvature GB data
    for time_step in tqdm(range(step_num)):
        if time_step + 1 >= step_num:
            continue

        # Load pre-computed velocity data for current time step
        dV_dict_file_name = f"2D_dV_split_gbLimit{area_limit}_step{time_step}.npz"
        npz_file = np.load(data_file_folder + dV_dict_file_name, allow_pickle=True)
        dV_dict_tmp = npz_file["dV_dict_tmp"]
        dV_dict_tmp = dV_dict_tmp.item()

        # Initialize temporary storage for current time step
        tmp_GB_antic_list_velocity = []
        tmp_GB_antic_list_curvature = []
        tmp_GB_antic_list_GBenergy = []
        tmp_GB_antic_list_anticNum = []
        tmp_GB_antic_list_area = []
        tmp_GB_antic_list_dV_normD = []
        tmp_GB_antic_list_dV_antiD = []

        # =============================================================================
        # EXTRACT ANTI-CURVATURE GB DATA
        # =============================================================================

        for key in GB_id_focus_detail:
            if time_step in GB_id_focus_detail[key]:  # If this GB shows anti-curvature at this time step

                result = dV_dict_tmp[key]
                velocity = result["velocity"]
                current_curvature_value = result["current_curvature_value"]
                current_area = 0.5 * GB_infomation_dict_list[time_step][key][0]
                dV_direction1 = result["dV_direction1"]  # Growth of grain1 into grain2
                dV_direction2 = result["dV_direction2"]  # Growth of grain2 into grain1

                # Normalize signs: make curvature positive and adjust velocity/growth directions accordingly
                # This ensures consistent interpretation regardless of grain ID ordering
                if current_curvature_value < 0:
                    # If curvature is negative, flip signs to make analysis consistent
                    GB_antic_list_velocity_list.append(-velocity)
                    GB_antic_list_curvature_list.append(-current_curvature_value)
                    GB_antic_list_dV_normD_list.append(dV_direction2)  # Normal direction growth
                    GB_antic_list_dV_antiD_list.append(dV_direction1)  # Anti-curvature direction growth
                    tmp_GB_antic_list_velocity.append(-velocity)
                    tmp_GB_antic_list_curvature.append(-current_curvature_value)
                    tmp_GB_antic_list_dV_normD.append(dV_direction2)
                    tmp_GB_antic_list_dV_antiD.append(dV_direction1)
                else:
                    # Keep original signs
                    GB_antic_list_velocity_list.append(velocity)
                    GB_antic_list_curvature_list.append(current_curvature_value)
                    GB_antic_list_dV_normD_list.append(dV_direction1)
                    GB_antic_list_dV_antiD_list.append(dV_direction2)
                    tmp_GB_antic_list_velocity.append(velocity)
                    tmp_GB_antic_list_curvature.append(current_curvature_value)
                    tmp_GB_antic_list_dV_normD.append(dV_direction1)
                    tmp_GB_antic_list_dV_antiD.append(dV_direction2)

                # Store energy, anti-curvature count, and area data
                GB_antic_list_GBenergy_list.append(result["current_eng"])
                GB_antic_list_anticNum_list.append(GB_id_focus[key])  # Total anti-curvature occurrences
                GB_antic_list_area_list.append(current_area)
                tmp_GB_antic_list_GBenergy.append(result["current_eng"])
                tmp_GB_antic_list_anticNum.append(GB_id_focus[key])
                tmp_GB_antic_list_area.append(current_area)

        # Store anti-curvature data for current time step
        GB_antic_list_velocity_AllList.append(tmp_GB_antic_list_velocity)
        GB_antic_list_curvature_AllList.append(tmp_GB_antic_list_curvature)
        GB_antic_list_GBenergy_AllList.append(tmp_GB_antic_list_GBenergy)
        GB_antic_list_anticNum_AllList.append(tmp_GB_antic_list_anticNum)
        GB_antic_list_area_AllList.append(tmp_GB_antic_list_area)
        GB_antic_list_dV_normD_AllList.append(tmp_GB_antic_list_dV_normD)
        GB_antic_list_dV_antiD_AllList.append(tmp_GB_antic_list_dV_antiD)

        # =============================================================================
        # EXTRACT NORMAL-CURVATURE GB DATA
        # =============================================================================

        tmp_GB_normc_list_velocity = []
        tmp_GB_normc_list_curvature = []
        tmp_GB_normc_list_GBenergy = []
        tmp_GB_normc_list_area = []
        tmp_GB_normc_list_dV_normD = []
        tmp_GB_normc_list_dV_antiD = []

        for key in dV_dict_tmp:
            result = dV_dict_tmp[key]
            velocity = result["velocity"]
            current_curvature_value = result["current_curvature_value"]
            current_area = 0.5 * GB_infomation_dict_list[time_step][key][0]
            dV_direction1 = result["dV_direction1"]
            dV_direction2 = result["dV_direction2"]

            # Check if this GB shows normal curvature behavior (curvature and velocity same sign)
            if current_curvature_value * velocity > 0:  # Normal curvature behavior

                # Normalize signs similar to anti-curvature case
                if current_curvature_value < 0:
                    GB_normc_list_velocity_list.append(-velocity)
                    GB_normc_list_curvature_list.append(-current_curvature_value)
                    GB_normc_list_dV_normD_list.append(dV_direction2)
                    GB_normc_list_dV_antiD_list.append(dV_direction1)
                    tmp_GB_normc_list_velocity.append(-velocity)
                    tmp_GB_normc_list_curvature.append(-current_curvature_value)
                    tmp_GB_normc_list_dV_normD.append(dV_direction2)
                    tmp_GB_normc_list_dV_antiD.append(dV_direction1)
                else:
                    GB_normc_list_velocity_list.append(velocity)
                    GB_normc_list_curvature_list.append(current_curvature_value)
                    GB_normc_list_dV_normD_list.append(dV_direction1)
                    GB_normc_list_dV_antiD_list.append(dV_direction2)
                    tmp_GB_normc_list_velocity.append(velocity)
                    tmp_GB_normc_list_curvature.append(current_curvature_value)
                    tmp_GB_normc_list_dV_normD.append(dV_direction1)
                    tmp_GB_normc_list_dV_antiD.append(dV_direction2)

                # Store energy and area data
                GB_normc_list_GBenergy_list.append(result["current_eng"])
                GB_normc_list_area_list.append(current_area)
                tmp_GB_normc_list_GBenergy.append(result["current_eng"])
                tmp_GB_normc_list_area.append(current_area)

        # Store normal-curvature data for current time step
        GB_normc_list_velocity_AllList.append(tmp_GB_normc_list_velocity)
        GB_normc_list_curvature_AllList.append(tmp_GB_normc_list_curvature)
        GB_normc_list_GBenergy_AllList.append(tmp_GB_normc_list_GBenergy)
        GB_normc_list_area_AllList.append(tmp_GB_normc_list_area)
        GB_normc_list_dV_normD_AllList.append(tmp_GB_normc_list_dV_normD)
        GB_normc_list_dV_antiD_AllList.append(tmp_GB_normc_list_dV_antiD)




    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # =============================================================================
    # STATISTICAL ANALYSIS OF ANTI-CURVATURE BEHAVIOR
    # =============================================================================
    log.warning(' ')
    log.warning("ANTI-CURVATURE GB STATISTICS")
    log.warning("=" * 50)

    # Calculate total number of GB observations across all time steps
    total_GB_num = 0
    for time_step in range(step_num):
        step_GB = GB_infomation_dict_list[time_step].keys()
        total_GB_num += len(step_GB)

    # Calculate real total (filtered GBs only)
    total_GB_num_real = len(GB_antic_list_velocity_list) + len(GB_normc_list_velocity_list)

    # Print anti-curvature GB statistics
    log.warning(f"The number of the GBs showing anti-curvature during whole simulations is {len(GB_antic_list_velocity_list)}")
    log.warning(f"The ratio of anti-curvature GBs during whole simulations is {len(GB_antic_list_velocity_list)/total_GB_num*100:.2f}% in {total_GB_num} GBs")
    log.warning(f"The ratio of norm-curvature GBs during whole simulations is {len(GB_normc_list_velocity_list)/total_GB_num*100:.2f}% in {total_GB_num} GBs")
    log.warning(' ')

    log.warning("ANTI-CURVATURE GB STATISTICS (FILTERED DATA ONLY)")
    log.warning(f"The ratio of anti-curvature GBs during whole simulations is {len(GB_antic_list_velocity_list)/total_GB_num_real*100:.2f}% in {total_GB_num_real} GBs")
    log.warning(f"The ratio of norm-curvature GBs during whole simulations is {len(GB_normc_list_velocity_list)/total_GB_num_real*100:.2f}% in {total_GB_num_real} GBs")
    log.warning(' ')

    # =============================================================================
    # VOXEL-LEVEL ANALYSIS
    # =============================================================================

    log.warning("VOXEL-LEVEL STATISTICS (ALL GBs)")
    log.warning("=" * 40)

    normal_growth_voxel = 0
    antic_growth_voxel = 0

    # Calculate anti-curvature voxel fraction across all GBs
    for i in range(0, len(GB_list_curvature_list)):
        for j in range(len(GB_list_curvature_list[i])):
            # Determine normal vs anti-curvature growth based on curvature sign
            # When curvature < 0: direction2 is normal, direction1 is anti-curvature
            # When curvature >= 0: direction1 is normal, direction2 is anti-curvature
            if GB_list_curvature_list[i][j] < 0:
                normal_growth_voxel += GB_list_dV_direction2[i][j]
                antic_growth_voxel += GB_list_dV_direction1[i][j]
            elif GB_list_curvature_list[i][j] >= 0:
                normal_growth_voxel += GB_list_dV_direction1[i][j]
                antic_growth_voxel += GB_list_dV_direction2[i][j]

    total_voxels = antic_growth_voxel + normal_growth_voxel
    log.warning(f"The number of the normal voxels and anti-curvature voxels during whole simulations are {normal_growth_voxel} and {antic_growth_voxel}")
    log.warning(f"The ratio of anti-curvature voxels during whole simulations is {antic_growth_voxel/total_voxels*100:.2f}% in {total_voxels} voxels")
    log.warning(' ')

    log.warning("VOXEL-LEVEL STATISTICS (NORMAL-CURVATURE GBs ONLY)")
    log.warning("=" * 50)

    # Calculate voxel statistics for normal-curvature GBs only
    normal_growth_voxel = np.sum(GB_normc_list_dV_normD_list)
    antic_growth_voxel = np.sum(GB_normc_list_dV_antiD_list)
    total_voxels = antic_growth_voxel + normal_growth_voxel

    log.warning(f"The number of the normal voxels and anti-curvature voxels are {normal_growth_voxel} and {antic_growth_voxel}")
    log.warning(f"The ratio of anti-curvature voxels is {antic_growth_voxel/total_voxels*100:.2f}% in {total_voxels} voxels")


    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # =============================================================================
    # VOXEL DISTRIBUTION ANALYSIS AND CONFIDENCE FILTERING
    # =============================================================================
    log.warning(' ')
    log.warning('NORMAL/ANTIC 99% CONFIDENCE FILTERING')
    # Set up binning parameters for analyzing anti-curvature voxel fractions
    bin_interval = 0.01  # Bin width for histogram
    x_lim = [0.5, 1.0]  # Range: 50% to 100% anti-curvature voxels
    bin_number = int((x_lim[1] - x_lim[0]) / bin_interval)
    Antic_voxel_coordinate = np.arange(x_lim[0], x_lim[1], bin_interval) + bin_interval / 2

    # Initialize histograms for anti-curvature and normal-curvature GBs
    antic_voxel_num_bin = np.zeros(bin_number)
    normc_voxel_num_bin = np.zeros(bin_number)

    # Calculate voxel fraction distribution for anti-curvature GBs
    for index in range(len(GB_antic_list_dV_normD_list)):
        total_voxels = GB_antic_list_dV_normD_list[index] + GB_antic_list_dV_antiD_list[index]
        if total_voxels > 0:  # Avoid division by zero
            antic_fraction = GB_antic_list_dV_antiD_list[index] / total_voxels
            if x_lim[0] <= antic_fraction < x_lim[1]:  # Check if within range
                bin_idx = int((antic_fraction - x_lim[0]) // bin_interval)
                if 0 <= bin_idx < bin_number:  # Ensure valid bin index
                    antic_voxel_num_bin[bin_idx] += 1

    # Calculate voxel fraction distribution for normal-curvature GBs
    for index in range(len(GB_normc_list_dV_normD_list)):
        total_voxels = GB_normc_list_dV_normD_list[index] + GB_normc_list_dV_antiD_list[index]
        if total_voxels > 0:
            normc_fraction = GB_normc_list_dV_normD_list[index] / total_voxels
            if x_lim[0] <= normc_fraction < x_lim[1]:
                bin_idx = int((normc_fraction - x_lim[0]) // bin_interval)
                if 0 <= bin_idx < bin_number:
                    normc_voxel_num_bin[bin_idx] += 1

    # =============================================================================
    # PLOT NORMAL-CURVATURE VOXEL DISTRIBUTION
    # =============================================================================
    log.info('Plotting: Normal Curvature PDF (pre confidence filter)')
    os.makedirs(os.path.dirname("pics/"), exist_ok=True)

    figure_name_all_bin = f"pics/2D_normc_voxel_distribution_{case_name}_5d_afterBin.png"
    plt.figure()
    fig = plt.figure(figsize=(7, 7))
    plt.xlim([0.5, 1.0])
    plt.ylim([0, 0.5])
    plt.title("Normal-curvature GB voxel distribution", fontsize=16)

    # Plot normalized distribution
    if np.sum(normc_voxel_num_bin) > 0:
        plt.plot(Antic_voxel_coordinate, normc_voxel_num_bin / np.sum(normc_voxel_num_bin),
                '-', linewidth=2, label='Normal-curvature GBs')

    plt.xlabel("Fraction of voxels moving in curvature direction", fontsize=18)
    plt.ylabel("Probability density", fontsize=18)
    plt.legend()
    plt.savefig(figure_name_all_bin, dpi=400, bbox_inches='tight')

    # =============================================================================
    # CONFIDENCE-BASED FILTERING
    # =============================================================================

    confidence_factor = 0.99  # 99% confidence threshold for anti-curvature behavior

    # Initialize filtered arrays for anti-curvature GBs with high confidence
    update_GB_antic_list_velocity_AllList = []
    update_GB_antic_list_curvature_AllList = []
    update_GB_antic_list_GBenergy_AllList = []
    update_GB_antic_list_anticNum_AllList = []
    update_GB_antic_list_area_AllList = []
    update_GB_antic_list_dV_normD_AllList = []
    update_GB_antic_list_dV_antiD_AllList = []

    # Flattened arrays for high-confidence anti-curvature GBs
    update_GB_antic_list_velocity_list = []
    update_GB_antic_list_curvature_list = []
    update_GB_antic_list_GBenergy_list = []
    update_GB_antic_list_anticNum_list = []
    update_GB_antic_list_area_list = []
    update_GB_antic_list_dV_normD_list = []
    update_GB_antic_list_dV_antiD_list = []

    # Filter anti-curvature GBs based on confidence factor
    for index in range(len(GB_antic_list_velocity_list)):
        total_voxels = GB_antic_list_dV_normD_list[index] + GB_antic_list_dV_antiD_list[index]
        if total_voxels > 0:
            antic_fraction = GB_antic_list_dV_antiD_list[index] / total_voxels
            if antic_fraction > confidence_factor:  # High confidence anti-curvature
                update_GB_antic_list_velocity_list.append(GB_antic_list_velocity_list[index])
                update_GB_antic_list_curvature_list.append(GB_antic_list_curvature_list[index])
                update_GB_antic_list_GBenergy_list.append(GB_antic_list_GBenergy_list[index])
                update_GB_antic_list_anticNum_list.append(GB_antic_list_anticNum_list[index])
                update_GB_antic_list_area_list.append(GB_antic_list_area_list[index])
                update_GB_antic_list_dV_normD_list.append(GB_antic_list_dV_normD_list[index])
                update_GB_antic_list_dV_antiD_list.append(GB_antic_list_dV_antiD_list[index])

    # Filter time-step-wise data (currently empty initialization, would need similar filtering)
    for index_i in range(len(GB_antic_list_velocity_AllList)):
        tmp_update_lists = [[] for _ in range(7)]  # Initialize temporary lists

        for index_j in range(len(GB_antic_list_velocity_AllList[index_i])):
            total_voxels = (GB_antic_list_dV_normD_AllList[index_i][index_j] +
                          GB_antic_list_dV_antiD_AllList[index_i][index_j])
            if total_voxels > 0:
                antic_fraction = GB_antic_list_dV_antiD_AllList[index_i][index_j] / total_voxels
                if antic_fraction > confidence_factor:
                    # Add to filtered lists
                    tmp_update_lists[0].append(GB_antic_list_velocity_AllList[index_i][index_j])
                    tmp_update_lists[1].append(GB_antic_list_curvature_AllList[index_i][index_j])
                    tmp_update_lists[2].append(GB_antic_list_GBenergy_AllList[index_i][index_j])
                    tmp_update_lists[3].append(GB_antic_list_anticNum_AllList[index_i][index_j])
                    tmp_update_lists[4].append(GB_antic_list_area_AllList[index_i][index_j])
                    tmp_update_lists[5].append(GB_antic_list_dV_normD_AllList[index_i][index_j])
                    tmp_update_lists[6].append(GB_antic_list_dV_antiD_AllList[index_i][index_j])

        # Store filtered data for this time step
        update_GB_antic_list_velocity_AllList.append(tmp_update_lists[0])
        update_GB_antic_list_curvature_AllList.append(tmp_update_lists[1])
        update_GB_antic_list_GBenergy_AllList.append(tmp_update_lists[2])
        update_GB_antic_list_anticNum_AllList.append(tmp_update_lists[3])
        update_GB_antic_list_area_AllList.append(tmp_update_lists[4])
        update_GB_antic_list_dV_normD_AllList.append(tmp_update_lists[5])
        update_GB_antic_list_dV_antiD_AllList.append(tmp_update_lists[6])

    # =============================================================================
    # FILTER NORMAL-CURVATURE GBs WITH HIGH CONFIDENCE
    # =============================================================================

    # Similar filtering for normal-curvature GBs
    update_GB_normc_list_velocity_AllList = []
    update_GB_normc_list_curvature_AllList = []
    update_GB_normc_list_GBenergy_AllList = []
    update_GB_normc_list_area_AllList = []
    update_GB_normc_list_dV_normD_AllList = []
    update_GB_normc_list_dV_antiD_AllList = []

    update_GB_normc_list_velocity_list = []
    update_GB_normc_list_curvature_list = []
    update_GB_normc_list_GBenergy_list = []
    update_GB_normc_list_area_list = []
    update_GB_normc_list_dV_normD_list = []
    update_GB_normc_list_dV_antiD_list = []

    # Filter normal-curvature GBs based on confidence factor
    for index in range(len(GB_normc_list_velocity_list)):
        total_voxels = GB_normc_list_dV_normD_list[index] + GB_normc_list_dV_antiD_list[index]
        if total_voxels > 0:
            normc_fraction = GB_normc_list_dV_normD_list[index] / total_voxels
            if normc_fraction > confidence_factor:  # High confidence normal-curvature
                update_GB_normc_list_velocity_list.append(GB_normc_list_velocity_list[index])
                update_GB_normc_list_curvature_list.append(GB_normc_list_curvature_list[index])
                update_GB_normc_list_GBenergy_list.append(GB_normc_list_GBenergy_list[index])
                update_GB_normc_list_area_list.append(GB_normc_list_area_list[index])
                update_GB_normc_list_dV_normD_list.append(GB_normc_list_dV_normD_list[index])
                update_GB_normc_list_dV_antiD_list.append(GB_normc_list_dV_antiD_list[index])

    # Filter time-step-wise normal-curvature data (similar process)
    for index_i in range(len(GB_normc_list_velocity_AllList)):
        tmp_update_lists = [[] for _ in range(6)]  # Initialize temporary lists

        for index_j in range(len(GB_normc_list_velocity_AllList[index_i])):
            total_voxels = (GB_normc_list_dV_normD_AllList[index_i][index_j] +
                          GB_normc_list_dV_antiD_AllList[index_i][index_j])
            if total_voxels > 0:
                normc_fraction = GB_normc_list_dV_normD_AllList[index_i][index_j] / total_voxels
                if normc_fraction > confidence_factor:
                    # Add to filtered lists
                    tmp_update_lists[0].append(GB_normc_list_velocity_AllList[index_i][index_j])
                    tmp_update_lists[1].append(GB_normc_list_curvature_AllList[index_i][index_j])
                    tmp_update_lists[2].append(GB_normc_list_GBenergy_AllList[index_i][index_j])
                    tmp_update_lists[3].append(GB_normc_list_area_AllList[index_i][index_j])
                    tmp_update_lists[4].append(GB_normc_list_dV_normD_AllList[index_i][index_j])
                    tmp_update_lists[5].append(GB_normc_list_dV_antiD_AllList[index_i][index_j])

        # Store filtered data for this time step
        update_GB_normc_list_velocity_AllList.append(tmp_update_lists[0])
        update_GB_normc_list_curvature_AllList.append(tmp_update_lists[1])
        update_GB_normc_list_GBenergy_AllList.append(tmp_update_lists[2])
        update_GB_normc_list_area_AllList.append(tmp_update_lists[3])
        update_GB_normc_list_dV_normD_AllList.append(tmp_update_lists[4])
        update_GB_normc_list_dV_antiD_AllList.append(tmp_update_lists[5])


    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # =============================================================================
    # FILTERED STATISTICS ANALYSIS (HIGH-CONFIDENCE DATA)
    # =============================================================================
    log.warning(' ')
    log.warning('FILTERED STATISTICS')

    log.warning("VOXEL-LEVEL STATISTICS (HIGH-CONFIDENCE ANTI-CURVATURE GBs)")
    log.warning("=" * 60)

    # Calculate voxel statistics for high-confidence anti-curvature GBs
    normal_growth_voxel = np.sum(update_GB_antic_list_dV_normD_list)
    antic_growth_voxel = np.sum(update_GB_antic_list_dV_antiD_list)
    total_voxels = normal_growth_voxel + antic_growth_voxel

    log.warning(f"Normal voxels: {normal_growth_voxel}")
    log.warning(f"Anti-curvature voxels: {antic_growth_voxel}")
    log.warning(f"Total voxels: {total_voxels}")
    if total_voxels > 0:
        log.warning(f"Anti-curvature voxel fraction: {antic_growth_voxel/total_voxels*100:.2f}%")
    log.warning(' ')

    log.warning("VOXEL-LEVEL STATISTICS (HIGH-CONFIDENCE NORMAL-CURVATURE GBs)")
    log.warning("=" * 60)

    # Calculate voxel statistics for high-confidence normal-curvature GBs
    normal_growth_voxel = np.sum(update_GB_normc_list_dV_normD_list)
    antic_growth_voxel = np.sum(update_GB_normc_list_dV_antiD_list)
    total_voxels = normal_growth_voxel + antic_growth_voxel

    log.warning(f"Normal voxels: {normal_growth_voxel}")
    log.warning(f"Anti-curvature voxels: {antic_growth_voxel}")
    log.warning(f"Total voxels: {total_voxels}")
    if total_voxels > 0:
        log.warning(f"Normal-curvature voxel fraction: {normal_growth_voxel/total_voxels*100:.2f}%")
    log.warning(' ')

    log.warning("GB-LEVEL STATISTICS COMPARISON")
    log.warning("=" * 40)

    # Calculate total GB numbers for comparison
    total_GB_num = 0
    for time_step in range(step_num):
        step_GB = GB_infomation_dict_list[time_step].keys()
        total_GB_num += len(step_GB)

    total_GB_num_real = len(GB_antic_list_velocity_list) + len(GB_normc_list_velocity_list)

    log.warning("BEFORE CONFIDENCE FILTERING:")
    log.warning(f"Anti-curvature GBs: {len(GB_antic_list_dV_normD_list)} ({len(GB_antic_list_dV_normD_list)/total_GB_num_real*100:.2f}%)")
    log.warning(f"Normal-curvature GBs: {len(GB_normc_list_dV_normD_list)} ({len(GB_normc_list_dV_normD_list)/total_GB_num_real*100:.2f}%)")
    log.warning(f"Total filtered GBs: {total_GB_num_real}")
    log.warning(' ')

    log.warning("AFTER CONFIDENCE FILTERING (99% threshold):")
    log.warning(f"High-confidence anti-curvature GBs: {len(update_GB_antic_list_dV_antiD_list)} ({len(update_GB_antic_list_dV_antiD_list)/total_GB_num_real*100:.2f}%)")
    log.warning(f"High-confidence normal-curvature GBs: {len(update_GB_normc_list_dV_antiD_list)} ({len(update_GB_normc_list_dV_antiD_list)/total_GB_num_real*100:.2f}%)")
    log.warning(' ')

    log.warning("PERCENTAGE OF ORIGINAL TOTAL GBs:")
    log.warning(f"High-confidence anti-curvature GBs: {len(update_GB_antic_list_dV_antiD_list)/total_GB_num*100:.3f}%")
    log.warning(f"High-confidence normal-curvature GBs: {len(update_GB_normc_list_dV_antiD_list)/total_GB_num*100:.3f}%")
    log.warning(f"Original total GBs: {total_GB_num}")
    log.warning(' ')


    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    log.warning(' ')
    log.warning('VELOCITY VS CURVATURE')
    # =============================================================================
    # VELOCITY vs CURVATURE RELATIONSHIP ANALYSIS
    # =============================================================================

    # Create comprehensive scatter plot showing all GB data
    figure_name_all = f"pics/2D_velocity_signedcurvature_{case_name}_5d.png"
    plt.figure()
    fig = plt.figure(figsize=(5, 5))

    # Plot reference line at velocity = 0
    plt.plot([curvature_limit, 0.5], [0, 0], '-', color='grey', linewidth=2, alpha=0.7, label='V=0 reference')

    # Scatter plot: normal-curvature GBs (blue) and anti-curvature GBs (orange)
    plt.scatter(GB_normc_list_curvature_list, GB_normc_list_velocity_list,
                s=4, alpha=0.5, label='Normal-curvature GBs', color='C0')
    plt.scatter(GB_antic_list_curvature_list, GB_antic_list_velocity_list,
                s=4, alpha=0.5, color='C1', label='Anti-curvature GBs')

    plt.xlabel("Curvature κ", fontsize=20)
    plt.ylabel("Velocity (voxel/step)", fontsize=18)
    plt.xlim([curvature_limit, 0.1])
    plt.ylim([-0.3, 0.3])
    plt.legend()
    plt.title("GB Velocity vs Curvature", fontsize=16)
    plt.savefig(figure_name_all, dpi=400, bbox_inches='tight')

    # =============================================================================
    # DENSITY PLOT WITH STATISTICAL ANALYSIS
    # =============================================================================

    # Create 2D histogram for density visualization
    x_bins = np.linspace(0, 0.105, 40)  # Curvature bins
    y_bins = np.linspace(-0.4, 0.4, 40)  # Velocity bins

    # Combine all GB data for density analysis
    all_curvatures = GB_normc_list_curvature_list + GB_antic_list_curvature_list
    all_velocities = GB_normc_list_velocity_list + GB_antic_list_velocity_list

    hist, x_edges, y_edges = np.histogram2d(all_curvatures, all_velocities, bins=[x_bins, y_bins])

    # Compute bin centers for plotting
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2

    # =============================================================================
    # BINNED STATISTICAL ANALYSIS
    # =============================================================================

    # Set up fine binning for statistical analysis
    bin_interval = 0.002  # Fine bin width
    x_lim = [0, 0.1]  # Curvature range
    bin_number = int((x_lim[1] - x_lim[0]) / bin_interval)
    curvature_coordinate = np.arange(x_lim[0], x_lim[1], bin_interval) + bin_interval / 2

    # Combine data for analysis
    new_curvature_1Dlist_remove_0step = GB_normc_list_curvature_list + GB_antic_list_curvature_list
    new_velocity_1Dlist_remove_0step = GB_normc_list_velocity_list + GB_antic_list_velocity_list
    new_area_1Dlist_remove_0step = GB_normc_list_area_list + GB_antic_list_area_list
    new_GBenergy_1Dlist_remove_0step = GB_normc_list_GBenergy_list + GB_antic_list_GBenergy_list

    # Calculate binned statistics (mean and standard deviation)
    curvature_bin_counts = np.zeros(bin_number)
    velocity_bin_sums = np.zeros(bin_number)
    velocity_bin_squared_sums = np.zeros(bin_number)

    for index in range(len(new_curvature_1Dlist_remove_0step)):
        curvature_val = new_curvature_1Dlist_remove_0step[index]
        velocity_val = new_velocity_1Dlist_remove_0step[index]

        if abs(curvature_val) > x_lim[1]:  # Skip high curvature values
            continue

        bin_idx = int((curvature_val - x_lim[0]) // bin_interval)
        if 0 <= bin_idx < bin_number:
            curvature_bin_counts[bin_idx] += 1
            velocity_bin_sums[bin_idx] += velocity_val
            velocity_bin_squared_sums[bin_idx] += velocity_val ** 2

    # Calculate means and standard deviations
    velocity_bin_means = np.zeros(bin_number)
    velocity_bin_stds = np.zeros(bin_number)

    for i in range(bin_number):
        if curvature_bin_counts[i] > 0:
            velocity_bin_means[i] = velocity_bin_sums[i] / curvature_bin_counts[i]
            # Standard deviation calculation
            mean_squared = velocity_bin_means[i] ** 2
            mean_of_squares = velocity_bin_squared_sums[i] / curvature_bin_counts[i]
            velocity_bin_stds[i] = np.sqrt(mean_of_squares - mean_squared)

    # =============================================================================
    # ADVANCED DENSITY PLOT WITH LINEAR FITTING
    # =============================================================================

    figure_name_all = f"pics/2D_velocity_signedcurvature_hot_{case_name}_5d.png"
    plt.figure()
    fig = plt.figure(figsize=(5, 5))

    # Plot reference line
    plt.plot([curvature_limit, 0.5], [0, 0], '-', color='grey', linewidth=2, alpha=0.7)

    # Create density contour plot
    X, Y = np.meshgrid(x_centers, y_centers)
    hist.T[hist.T == 0] = 1  # Avoid log(0) issues

    # Plot density with log scale
    plt.contour(X, Y, np.log10(hist.T), levels=20, cmap='gray', alpha=0.1, vmin=0, vmax=2.8)
    ax2 = plt.contourf(X, Y, np.log10(hist.T), levels=20, cmap='coolwarm', alpha=0.9, vmin=0, vmax=2.8)

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap='coolwarm', norm=plt.Normalize(vmin=0, vmax=2.8))
    # cbar = plt.colorbar(sm)
    sm.set_array([])  # Required for manually created ScalarMappables
    cbar = plt.colorbar(sm, ax=plt.gca())  # <-- pass ax explicitly
    cbar.set_label(u"$\log_{10}(density)$", fontsize=20)

    # =============================================================================
    # STATISTICAL ERROR BARS AND LINEAR FITTING
    # =============================================================================

    # Plot error bars for bins with sufficient data
    valid_bins = curvature_bin_counts > 10  # Minimum 10 data points per bin
    plt.errorbar(curvature_coordinate[valid_bins], velocity_bin_means[valid_bins],
                yerr=velocity_bin_stds[valid_bins],
                fmt='o', color='k', linewidth=1, capsize=1, ecolor='black', markersize=2,
                label='Binned statistics')

    # Linear fit for all valid bins
    x_all = curvature_coordinate[valid_bins]
    y_all = velocity_bin_means[valid_bins]

    if len(x_all) > 1:  # Need at least 2 points for fitting
        p_all = np.polyfit(x_all, y_all, 1)  # Linear fit
        y_pred_all = np.polyval(p_all, x_all)
        r2_all = 1 - np.sum((y_all - y_pred_all)**2) / np.sum((y_all - np.mean(y_all))**2)
        print(f"All points fitting R²: {r2_all:.4f}")
        print(f"All points slope (mobility): {p_all[0]:.6f}")

        # Linear fit for low curvature region (κ < 0.03)
        mask_sub = x_all < 0.03
        x_sub = x_all[mask_sub]
        y_sub = y_all[mask_sub]

        if len(x_sub) > 1:
            p_sub = np.polyfit(x_sub, y_sub, 1)
            y_pred_sub = np.polyval(p_sub, x_sub)
            r2_sub = 1 - np.sum((y_sub - y_pred_sub)**2) / np.sum((y_sub - np.mean(y_sub))**2)
            print(f"Low curvature (κ<0.03) fitting R²: {r2_sub:.4f}")
            print(f"Low curvature slope (mobility): {p_sub[0]:.6f}")
        else:
            r2_sub = np.nan
            p_sub = [np.nan, np.nan]

        # Plot linear fits
        plt.plot(x_all, y_pred_all, '-', color='C1', linewidth=3,
                label=rf'All points (R$^2$={r2_all:.3f})')
        if len(x_sub) > 1:
            plt.plot(x_sub, y_pred_sub, '-', color='C2', linewidth=3,
                    label=rf'κ<0.03 (R$^2$={r2_sub:.3f})')

        plt.legend(loc='lower right')

    plt.xlabel("Curvature κ", fontsize=20)
    plt.ylabel("Velocity (voxel/step)", fontsize=20)
    plt.xlim([curvature_limit, 0.1])
    plt.ylim([-0.3, 0.3])
    plt.title("GB Velocity vs Curvature (Density Plot)", fontsize=14)
    plt.savefig(figure_name_all, dpi=400, bbox_inches='tight')

    # =============================================================================
    # SAVE VELOCITY + CURVATURE CSV (mirrors --final in vector_hdf5_processing)
    # =============================================================================

    csv_output_path = f"pics/velocity_curvature_density_{case_name}_data.csv"
    os.makedirs(os.path.dirname(csv_output_path), exist_ok=True)

    with open(csv_output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "curvature", "velocity", "area", "GBenergy", "group"
        ])
        for kappa, v, area, eng in zip(
            GB_normc_list_curvature_list,
            GB_normc_list_velocity_list,
            GB_normc_list_area_list,
            GB_normc_list_GBenergy_list,
        ):
            writer.writerow([kappa, v, area, eng, "normc"])
        for kappa, v, area, eng in zip(
            GB_antic_list_curvature_list,
            GB_antic_list_velocity_list,
            GB_antic_list_area_list,
            GB_antic_list_GBenergy_list,
        ):
            writer.writerow([kappa, v, area, eng, "antic"])

    log.warning(
        f"Velocity+curvature CSV saved: {csv_output_path}  "
        f"({len(GB_normc_list_curvature_list)} normc + "
        f"{len(GB_antic_list_curvature_list)} antic = "
        f"{len(new_curvature_1Dlist_remove_0step)} rows)"
    )
