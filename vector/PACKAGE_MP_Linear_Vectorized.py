#!/usr/bin/env python3

# -*- coding: utf-8 -*-
"""
Linear Smoothing Method Implementation for Interface Analysis

This module implements a linear smoothing method for calculating grain boundary
normal vectors and curvature in polycrystalline materials. The method uses
linear filtering with these features:

1. Linear Filtering:
   - Applies weighted averaging to smooth interface properties
   - Uses variable kernel sizes for multi-scale analysis
   - Preserves sharp features at grain junctions

2. Normal Vector Calculation:
   - Computes gradients from smoothed interface data
   - Handles multiple grain boundaries efficiently
   - Provides consistent normals at triple junctions

3. Curvature Calculation:
   - Uses second derivatives of smoothed data
   - Handles multiple length scales
   - Maintains numerical stability at interfaces

Key Features:
   - Parallel implementation for large datasets
   - Configurable smoothing parameters
   - Efficient sparse matrix operations
   - Error calculation against analytical solutions
   - Optional vectorized --fast path for large meshes (see linear_combined_core_fast)

Original Author: Lin Yang

This version added:
Fast path additions: vectorized core for large-scale simulations
"""

import os
current_path = os.getcwd() + '/'
import sys
sys.path.append(current_path)
import numpy as np
import math
import matplotlib.pyplot as plt
import myInput
import datetime
import multiprocessing as mp


class linear_class(object):
    """Linear smoothing algorithm implementation.

    This class implements linear smoothing to calculate normal vectors and curvature
    at grain boundaries. The algorithm uses a sliding window approach with weighted
    averaging to smooth boundaries and compute geometric properties.

    Attributes:
        P (ndarray): Phase field array storing microstructure and normal vectors
        C (ndarray): Array storing calculated curvature values
        nx (int): Number of grid points in x direction
        ny (int): Number of grid points in y direction
        ng (int): Number of grains
        cores (int): Number of CPU cores for parallel processing
        loop_times (int): Number of smoothing iterations
        clip (int): Number of boundary elements to ignore
        errors (float): Accumulated angle errors
        errors_per_site (float): Average error per boundary site
        running_time (float): Total computation time
        running_coreTime (float): Maximum core processing time
        fast_chunk_size (int): Boundary pixel batch size for fast vectorized mode
    """

    def __init__(self, nx, ny, ng, cores, loop_times, P0, R, clip=0,
                 verification_system=True, curvature_sign=False, id_offset=None):
        """Initialize the linear smoothing algorithm.

        Args:
            nx (int): Number of grid points in x direction
            ny (int): Number of grid points in y direction
            ng (int): Number of grains in the system
            cores (int): Number of CPU cores for parallel processing
            loop_times (int): Size of smoothing window
            P0 (ndarray): Initial microstructure configuration
            R (ndarray): Reference/analytical solution for validation
            clip (int): Number of boundary layers to ignore
            verification_system (bool): Enable validation against analytical solution
            curvature_sign (bool): Calculate signed curvature values
            id_offset: Grain ID offset; inferred from P0 if None
        """
        # Runtime tracking
        self.running_time = 0       # Total execution time
        self.running_coreTime = 0   # Core calculation time
        self.errors = 0             # Accumulated angle errors
        self.errors_per_site = 0    # Average error per GB site
        self.clip = clip            # Boundary clipping

        # Grid parameters
        self.nx = nx   # Number of sites in x axis
        self.ny = ny   # Number of sites in y axis
        self.ng = ng   # Number of grains
        self.R = R     # Reference normal vectors

        # Infer offset from data if not provided
        self.id_offset = int(np.nanmin(P0)) if id_offset is None else id_offset

        # Initialize result arrays
        self.P = np.zeros((3, nx, ny))   # Stores IC and normal vectors
        self.C = np.zeros((2, nx, ny))   # Stores curvature

        # Convert individual grain maps to single map
        if len(P0.shape) == 2:
            self.P[0, :, :] = np.array(P0)
            self.C[0, :, :] = np.array(P0)
        else:
            for i in range(0, np.shape(P0)[2]):
                self.P[0, :, :] += P0[:, :, i] * (i + 1)
                self.C[0, :, :] += P0[:, :, i] * (i + 1)

        # Parallel processing parameters
        self.cores = cores

        # Smoothing parameters
        self.loop_times = loop_times
        self.tableL = 2 * (loop_times + 1) + 1       # Table length for repeated calcs
        self.tableL_curv = 2 * (loop_times + 2) + 1
        self.halfL = loop_times + 1

        # Get smoothing matrices
        self.smoothed_vector_i, self.smoothed_vector_j = \
            myInput.output_linear_vector_matrix(self.loop_times, self.clip)

        self.verification_system = verification_system
        self.curvature_sign = curvature_sign

        # Fast path: chunk size for vectorized processing (set by linear_main)
        self.fast_chunk_size = 50_000

    # -----------------------------------------------------------------------
    # Accessors
    # -----------------------------------------------------------------------

    def get_P(self):
        """Get the phase field and normal vector results.

        Returns:
            ndarray: Matrix containing grain structure and normal vectors.
                     Shape (3, nx, ny):
                       P[0,:,:] — grain IDs
                       P[1:3,:,:] — normal vector components
        """
        return self.P

    def get_C(self):
        """Get the curvature calculation results.

        Returns:
            ndarray: Matrix containing curvature values. Shape (2, nx, ny):
                       C[0,:,:] — grain IDs
                       C[1,:,:] — curvature values
        """
        return self.C

    # -----------------------------------------------------------------------
    # Error / validation
    # -----------------------------------------------------------------------

    def get_errors(self):
        """Calculate error between calculated and reference normal vectors.

        Computes angular difference between calculated normal vectors and reference
        values at each grain boundary site.
        """
        ge_gbsites = self.get_gb_list()
        for gbSite in ge_gbsites:
            [gei, gej] = gbSite
            ge_dx, ge_dy = myInput.get_grad(self.P, gei, gej)
            self.errors += math.acos(
                round(abs(ge_dx * self.R[gei, gej, 0] + ge_dy * self.R[gei, gej, 1]), 5)
            )

        if len(ge_gbsites) > 0:
            self.errors_per_site = self.errors / len(ge_gbsites)
        else:
            self.errors_per_site = 0

    def get_curvature_errors(self):
        """Calculate error between calculated and reference curvature values.

        For each grain boundary site, computes difference between calculated
        curvature and reference value.
        """
        gce_gbsites = self.get_gb_list()
        for gbSite in gce_gbsites:
            [gcei, gcej] = gbSite
            self.errors += abs(self.R[gcei, gcej, 2] - self.C[1, gcei, gcej])

        if len(gce_gbsites) != 0:
            self.errors_per_site = self.errors / len(gce_gbsites)
        else:
            self.errors_per_site = 0

    # -----------------------------------------------------------------------
    # Visualization
    # -----------------------------------------------------------------------

    def get_2d_plot(self, init, algo):
        """Generate 2D visualization of microstructure with normal vectors.

        Args:
            init (str): Name of initial condition
            algo (str): Name of algorithm used
        """
        plt.subplots_adjust(wspace=0.2, right=1.8)
        plt.close()
        fig1 = plt.figure(1)
        fig_page = self.loop_times
        plt.title(f'{algo}-{init} \n loop = ' + str(fig_page))
        if fig_page < 10:
            String = '000' + str(fig_page)
        elif fig_page < 100:
            String = '00' + str(fig_page)
        elif fig_page < 1000:
            String = '0' + str(fig_page)
        elif fig_page < 10000:
            String = str(fig_page)
        plt.imshow(self.P[0, :, :], cmap='gray', interpolation='nearest')
        plt.xticks([])
        plt.yticks([])
        plt.savefig('BL_PolyGray_noArrows.png', dpi=1000, bbox_inches='tight')

        g2p_gbsites = self.get_gb_list()
        for gbSite in g2p_gbsites:
            [g2pi, g2pj] = gbSite
            g2p_dx, g2p_dy = myInput.get_grad(self.P, g2pi, g2pj)
            if g2pi > 200 and g2pi < 500:
                plt.arrow(g2pj, g2pi, 30 * g2p_dx, 30 * g2p_dy,
                          width=0.1, lw=0.1, alpha=0.8, color='navy')

    # -----------------------------------------------------------------------
    # Grain boundary site lists
    # -----------------------------------------------------------------------

    def get_gb_list(self, grainID=None):
        """Get list of grain boundary sites.

        Args:
            grainID (int): ID of grain to find boundaries for

        Returns:
            list: List of [i, j] coordinates of boundary sites
        """
        if grainID is None:
            grainID = self.id_offset
        ggn_gbsites = []
        for i in range(0, self.nx):
            for j in range(0, self.ny):
                ip, im, jp, jm = myInput.periodic_bc(self.nx, self.ny, i, j)
                if (((self.P[0, ip, j] - self.P[0, i, j]) != 0) or
                        ((self.P[0, im, j] - self.P[0, i, j]) != 0) or
                        ((self.P[0, i, jp] - self.P[0, i, j]) != 0) or
                        ((self.P[0, i, jm] - self.P[0, i, j]) != 0)) \
                        and self.P[0, i, j] == grainID:
                    ggn_gbsites.append([i, j])
        return ggn_gbsites

    def get_all_gb_list(self):
        gagn_gbsites = [[] for _ in range(int(self.ng))]
        for i in range(0, self.nx):
            for j in range(0, self.ny):
                ip, im, jp, jm = myInput.periodic_bc(self.nx, self.ny, i, j)
                if (((self.P[0, ip, j] - self.P[0, i, j]) != 0) or
                        ((self.P[0, im, j] - self.P[0, i, j]) != 0) or
                        ((self.P[0, i, jp] - self.P[0, i, j]) != 0) or
                        ((self.P[0, i, jm] - self.P[0, i, j]) != 0)):
                    gagn_gbsites[int(self.P[0, i, j] - self.id_offset)].append([i, j])
        return gagn_gbsites

    # -----------------------------------------------------------------------
    # Subdomain utilities
    # -----------------------------------------------------------------------

    def check_subdomain_and_nei(self, A):
        ca_length, ca_width = myInput.split_cores(self.cores)
        ca_area_cen = [int(A[0] / self.nx * ca_width), int(A[1] / self.ny * ca_length)]
        ca_area_nei = []
        ca_area_nei.append([int((ca_area_cen[0] - 1) % ca_width),
                            int((ca_area_cen[1] - 1) % ca_length)])
        ca_area_nei.append([int((ca_area_cen[0] - 1) % ca_width),
                            int(ca_area_cen[1])])
        ca_area_nei.append([int((ca_area_cen[0] - 1) % ca_width),
                            int((ca_area_cen[1] + 1) % ca_length)])
        ca_area_nei.append([int(ca_area_cen[0]),
                            int((ca_area_cen[1] + 1) % ca_length)])
        ca_area_nei.append([int((ca_area_cen[0] + 1) % ca_width),
                            int((ca_area_cen[1] + 1) % ca_length)])
        ca_area_nei.append([int((ca_area_cen[0] + 1) % ca_width),
                            int(ca_area_cen[1])])
        ca_area_nei.append([int((ca_area_cen[0] + 1) % ca_width),
                            int((ca_area_cen[1] - 1) % ca_length)])
        ca_area_nei.append([int(ca_area_cen[0]),
                            int((ca_area_cen[1] - 1) % ca_length)])
        return ca_area_cen, ca_area_nei

    # -----------------------------------------------------------------------
    # Window utilities — standard scalar path
    # -----------------------------------------------------------------------

    def find_window(self, i, j, fw_len):
        """Find the window matrix around a point for smoothing calculations.

        Args:
            i (int): x coordinate of center point
            j (int): y coordinate of center point
            fw_len (int): Width/height of window

        Returns:
            ndarray: Binary window matrix indicating grain membership
        """
        fw_half = int((fw_len - 1) / 2)
        window = np.zeros((fw_len, fw_len))

        for wi in range(fw_len):
            for wj in range(fw_len):
                global_x = (i - fw_half + wi) % self.nx
                global_y = (j - fw_half + wj) % self.ny
                if self.P[0, global_x, global_y] == self.P[0, i, j]:
                    window[wi, wj] = 1
                else:
                    window[wi, wj] = 0

        return window

    # -----------------------------------------------------------------------
    # Window utilities — fast vectorized path
    # -----------------------------------------------------------------------

    def find_all_windows_vectorized(self, boundary_ij: np.ndarray,
                                    fw_len: int) -> np.ndarray:
        """
        Vectorized replacement for looped find_window calls.

        Instead of calling find_window(i, j, fw_len) once per pixel, computes
        all windows simultaneously for a batch of boundary pixels using NumPy
        advanced indexing and broadcasting.

        Parameters
        ----------
        boundary_ij : np.ndarray, shape (N, 2)
            (i, j) coordinates of the boundary pixels to process.
        fw_len : int
            Window side length (same value passed to find_window).

        Returns
        -------
        np.ndarray, shape (N, fw_len, fw_len), dtype float64
            Binary window matrix for each pixel. windows[k, wi, wj] = 1.0
            if the pixel at offset (wi, wj) from pixel k belongs to the same
            grain as pixel k, else 0.0.

        Memory estimate
        ---------------
        N=50_000, fw_len=43:  50000 * 43 * 43 * 8 bytes ≈ 740 MB.
        Callers are responsible for chunking N to stay within memory budget.
        """
        fw_half = (fw_len - 1) // 2

        # Offset arrays: shape (fw_len,)
        di = np.arange(fw_len) - fw_half   # row offsets
        dj = np.arange(fw_len) - fw_half   # col offsets

        # Global row/col indices for every (pixel, window_row):
        #   gi: shape (N, fw_len) — each row is the N global row indices
        #       for one window row offset.
        gi = (boundary_ij[:, 0:1] + di[np.newaxis, :]) % self.nx   # (N, fw_len)
        gj = (boundary_ij[:, 1:2] + dj[np.newaxis, :]) % self.ny   # (N, fw_len)

        # Grain ID at the center of each boundary pixel: shape (N,)
        center_ids = self.P[0, boundary_ij[:, 0], boundary_ij[:, 1]]

        # Grain IDs at every window position:
        #   gi[:, :, np.newaxis] has shape (N, fw_len, 1)
        #   gj[:, np.newaxis, :] has shape (N, 1, fw_len)
        #   result: shape (N, fw_len, fw_len)
        P0_windows = self.P[0][gi[:, :, np.newaxis],
                               gj[:, np.newaxis, :]]

        # Binary window: 1.0 where the window pixel belongs to the same grain
        # as the center pixel.
        windows = (P0_windows == center_ids[:, np.newaxis, np.newaxis]
                   ).astype(np.float64)

        return windows   # shape (N, fw_len, fw_len)

    def _get_boundary_mask_vectorized(self, ij_array: np.ndarray) -> np.ndarray:
        """
        Vectorized boundary check for an array of (i, j) pixel coordinates.

        Replaces the per-pixel periodic_bc + neighbor comparison used inside
        the standard core loops.

        Parameters
        ----------
        ij_array : np.ndarray, shape (N, 2)
            Pixel coordinates to test.

        Returns
        -------
        np.ndarray bool, shape (N,)
            True where the pixel is on a grain boundary (any cardinal
            neighbor has a different grain ID).
        """
        i = ij_array[:, 0]
        j = ij_array[:, 1]
        ip = (i + 1) % self.nx
        im = (i - 1) % self.nx
        jp = (j + 1) % self.ny
        jm = (j - 1) % self.ny
        center = self.P[0, i, j]
        return (
            (self.P[0, ip, j] != center) |
            (self.P[0, im, j] != center) |
            (self.P[0, i, jp] != center) |
            (self.P[0, i, jm] != center)
        )

    # -----------------------------------------------------------------------
    # Curvature calculator (unchanged — operates on a single 5×5 matrix)
    # -----------------------------------------------------------------------

    def calculate_curvature(self, matrix):
        """Calculate curvature from normal vectors.

        Args:
            matrix (ndarray): Normal vector field (5×5 smoothed window)

        Returns:
            float: Local curvature value
        """
        I02 = matrix[0, 2]
        I11 = matrix[1, 1]
        I12 = matrix[1, 2]
        I13 = matrix[1, 3]
        I20 = matrix[2, 0]
        I21 = matrix[2, 1]
        I22 = matrix[2, 2]
        I23 = matrix[2, 3]
        I24 = matrix[2, 4]
        I31 = matrix[3, 1]
        I32 = matrix[3, 2]
        I33 = matrix[3, 3]
        I42 = matrix[4, 2]

        Ii  = (I32 - I12) / 2
        Ij  = (I23 - I21) / 2
        Imi = (I22 - I02) / 2
        Ipi = (I42 - I22) / 2
        Imj = (I22 - I20) / 2
        Ipj = (I24 - I22) / 2
        Imij = (I13 - I11) / 2
        Ipij = (I33 - I31) / 2
        Iii = (Ipi - Imi) / 2
        Ijj = (Ipj - Imj) / 2
        Iij = (Ipij - Imij) / 2

        if (Ii ** 2 + Ij ** 2) == 0:
            return 0

        if self.curvature_sign:
            return -(Ii ** 2 * Ijj - 2 * Ii * Ij * Iij + Ij ** 2 * Iii) \
                   / (Ii ** 2 + Ij ** 2) ** 1.5
        else:
            return abs(Ii ** 2 * Ijj - 2 * Ii * Ij * Iij + Ij ** 2 * Iii) \
                   / (Ii ** 2 + Ij ** 2) ** 1.5

    # -----------------------------------------------------------------------
    # Standard core functions (unchanged)
    # -----------------------------------------------------------------------

    def linear_curvature_core(self, core_input):
        """Core function for curvature calculation.

        Implements linear smoothing and calculates curvature using second
        derivatives of smoothed data.

        Args:
            core_input: Input data for this subdomain

        Returns:
            tuple: Calculated curvature values and timing information
        """
        core_stime = datetime.datetime.now()
        li, lj, lk = np.shape(core_input)
        fval = np.zeros((self.nx, self.ny, 1))

        corner1 = core_input[0, 0, :]
        corner3 = core_input[li - 1, lj - 1, :]

        core_area_cen, core_area_nei = self.check_subdomain_and_nei(corner1)
        if self.verification_system:
            print(f'the processor {core_area_cen} start...')

        for core_a in core_input:
            for core_b in core_a:
                i = core_b[0]
                j = core_b[1]
                ip, im, jp, jm = myInput.periodic_bc(self.nx, self.ny, i, j)
                if (((self.P[0, ip, j] - self.P[0, i, j]) != 0) or
                        ((self.P[0, im, j] - self.P[0, i, j]) != 0) or
                        ((self.P[0, i, jp] - self.P[0, i, j]) != 0) or
                        ((self.P[0, i, jm] - self.P[0, i, j]) != 0)):
                    window = self.find_window(i, j, self.tableL_curv - 2 * self.clip)
                    smoothed_matrix = myInput.output_smoothed_matrix(
                        window,
                        myInput.output_linear_smoothing_matrix(self.loop_times)
                    )[self.loop_times:-self.loop_times,
                      self.loop_times:-self.loop_times]
                    fval[i, j, 0] = self.calculate_curvature(smoothed_matrix)

        core_etime = datetime.datetime.now()
        if self.verification_system:
            print("my core time is " + str((core_etime - core_stime).total_seconds()))
        return (fval, (core_etime - core_stime).total_seconds())

    def linear_one_normal_vector_core(self, core_input):
        i = core_input[0]
        j = core_input[1]

        window = np.zeros((self.tableL, self.tableL))
        ip, im, jp, jm = myInput.periodic_bc(self.nx, self.ny, i, j)
        if (((self.P[0, ip, j] - self.P[0, i, j]) != 0) or
                ((self.P[0, im, j] - self.P[0, i, j]) != 0) or
                ((self.P[0, i, jp] - self.P[0, i, j]) != 0) or
                ((self.P[0, i, jm] - self.P[0, i, j]) != 0) or
                ((self.P[0, ip, jp] - self.P[0, i, j]) != 0) or
                ((self.P[0, im, jp] - self.P[0, i, j]) != 0) or
                ((self.P[0, ip, jm] - self.P[0, i, j]) != 0) or
                ((self.P[0, im, jm] - self.P[0, i, j]) != 0)):
            window = self.find_window(i, j, self.tableL - 2 * self.clip)

        return np.array([-np.sum(window * self.smoothed_vector_i),
                         np.sum(window * self.smoothed_vector_j)])

    def linear_normal_vector_core(self, core_input):
        """Core function for normal vector calculation.

        Implements linear smoothing and calculates interface normals using
        central differences.

        Args:
            core_input: Subset of points to process

        Returns:
            tuple: (Normal vector array, Computation time)
        """
        core_stime = datetime.datetime.now()
        li, lj, lk = np.shape(core_input)
        fval = np.zeros((self.nx, self.ny, 2))

        corner1 = core_input[0, 0, :]
        corner3 = core_input[li - 1, lj - 1, :]

        core_area_cen, core_area_nei = self.check_subdomain_and_nei(corner1)
        if self.verification_system:
            print(f'the processor {core_area_cen} start...')

        for core_a in core_input:
            for core_b in core_a:
                i = core_b[0]
                j = core_b[1]
                ip, im, jp, jm = myInput.periodic_bc(self.nx, self.ny, i, j)
                if (((self.P[0, ip, j] - self.P[0, i, j]) != 0) or
                        ((self.P[0, im, j] - self.P[0, i, j]) != 0) or
                        ((self.P[0, i, jp] - self.P[0, i, j]) != 0) or
                        ((self.P[0, i, jm] - self.P[0, i, j]) != 0)):
                    window = self.find_window(i, j, self.tableL - 2 * self.clip)
                    fval[i, j, 0] = -np.sum(window * self.smoothed_vector_i)
                    fval[i, j, 1] = np.sum(window * self.smoothed_vector_j)

        core_etime = datetime.datetime.now()
        if self.verification_system:
            print("my core time is " + str((core_etime - core_stime).total_seconds()))
        return (fval, (core_etime - core_stime).total_seconds())

    def linear_combined_core(self, core_input):
        """Standard (scalar) combined normal vector + curvature core.

        Processes each pixel in the subdomain sequentially using find_window.
        This is the original implementation, preserved unchanged for the
        standard (non-fast) path.

        Args:
            core_input: ndarray, shape (li, lj, 2) — subdomain pixel coords

        Returns:
            tuple: (fval, core_time_seconds)
                   fval shape (nx, ny, 3): [:,:,0]=nv_i, [:,:,1]=nv_j,
                   [:,:,2]=curvature
        """
        core_stime = datetime.datetime.now()
        li, lj, lk = np.shape(core_input)
        fval = np.zeros((self.nx, self.ny, 3))

        for core_a in core_input:
            for core_b in core_a:
                i = core_b[0]
                j = core_b[1]
                ip, im, jp, jm = myInput.periodic_bc(self.nx, self.ny, i, j)

                if (((self.P[0, ip, j] - self.P[0, i, j]) != 0) or
                        ((self.P[0, im, j] - self.P[0, i, j]) != 0) or
                        ((self.P[0, i, jp] - self.P[0, i, j]) != 0) or
                        ((self.P[0, i, jm] - self.P[0, i, j]) != 0)):

                    # Normal vector window (tableL size)
                    window_n = self.find_window(i, j, self.tableL - 2 * self.clip)
                    fval[i, j, 0] = -np.sum(window_n * self.smoothed_vector_i)
                    fval[i, j, 1] = np.sum(window_n * self.smoothed_vector_j)

                    # Curvature window (tableL_curv size)
                    window_c = self.find_window(i, j, self.tableL_curv - 2 * self.clip)
                    smoothed_matrix = myInput.output_smoothed_matrix(
                        window_c,
                        myInput.output_linear_smoothing_matrix(self.loop_times)
                    )[self.loop_times:-self.loop_times,
                      self.loop_times:-self.loop_times]
                    fval[i, j, 2] = self.calculate_curvature(smoothed_matrix)

        core_etime = datetime.datetime.now()
        return (fval, (core_etime - core_stime).total_seconds())

    # -----------------------------------------------------------------------
    # Fast vectorized core — new addition for --fast path
    # -----------------------------------------------------------------------

    def linear_combined_core_fast(self, core_input):
        """
        Chunked vectorized replacement for linear_combined_core.

        Key improvements over linear_combined_core:
        -----------------------------------------------
        1. Boundary detection is vectorized over the entire subdomain at once
           using _get_boundary_mask_vectorized, eliminating the per-pixel
           periodic_bc + neighbor comparison Python loop.

        2. find_window is replaced by find_all_windows_vectorized, called in
           chunks of self.fast_chunk_size pixels to bound peak memory usage.

        3. Normal vector dot products use np.einsum over entire chunks
           (no per-pixel loop).

        4. Curvature still has a per-pixel Python loop, but:
           - Only over confirmed boundary pixels (not all nx*ny).
           - Window data is pre-fetched as a batch before the loop.
           - The smoothing matrix is computed once per chunk, not per pixel.

        Memory budget per chunk (chunk_size=50_000, loop_times=20):
        ---------------------------------------------------------------
          windows_n : 50000 * 43 * 43 * 8 bytes ≈  740 MB  (transient)
          windows_c : 50000 * 45 * 45 * 8 bytes ≈  810 MB  (transient)
          Both are computed and deleted sequentially — never simultaneously.
          Reduce self.fast_chunk_size if OOM is observed.

        Parameters
        ----------
        core_input : np.ndarray, shape (li, lj, 2)
            Subdomain pixel coordinates — same format as linear_combined_core.

        Returns
        -------
        tuple : (fval, core_time_seconds)
            fval shape (nx, ny, 3):
              [:,:,0] = normal vector i-component
              [:,:,1] = normal vector j-component
              [:,:,2] = curvature
        """
        core_stime = datetime.datetime.now()

        li, lj, lk = np.shape(core_input)
        fval = np.zeros((self.nx, self.ny, 3))

        # ------------------------------------------------------------------
        # Step 1: Flatten subdomain to (M, 2) array of all pixel coordinates
        # ------------------------------------------------------------------
        all_ij = core_input.reshape(-1, 2)   # shape (li*lj, 2)

        # ------------------------------------------------------------------
        # Step 2: Vectorized boundary detection over the entire subdomain.
        #         Replaces the double for-loop + periodic_bc per pixel.
        # ------------------------------------------------------------------
        is_bnd = self._get_boundary_mask_vectorized(all_ij)
        boundary_ij = all_ij[is_bnd]         # shape (N_bnd, 2)

        if len(boundary_ij) == 0:
            core_etime = datetime.datetime.now()
            return (fval, (core_etime - core_stime).total_seconds())

        # Window sizes (same expressions as linear_combined_core)
        fw_n = self.tableL - 2 * self.clip          # normal vector window size
        fw_c = self.tableL_curv - 2 * self.clip     # curvature window size

        # Precompute the smoothing matrix once for the entire core call —
        # not once per pixel as in the original.
        smoothing_mat = myInput.output_linear_smoothing_matrix(self.loop_times)

        chunk_size = self.fast_chunk_size
        n_boundary = len(boundary_ij)

        # ------------------------------------------------------------------
        # Step 3: Chunked vectorized processing over boundary pixels.
        #
        #   Each chunk processes at most chunk_size boundary pixels.
        #   Within a chunk:
        #     - find_all_windows_vectorized fetches all windows at once.
        #     - np.einsum computes all normal vector dot products at once.
        #     - A Python loop over the chunk computes curvature, but only
        #       over boundary pixels (N_bnd << nx*ny at later timesteps)
        #       and with windows already in-hand as a NumPy array.
        # ------------------------------------------------------------------
        for start in range(0, n_boundary, chunk_size):
            chunk = boundary_ij[start: start + chunk_size]   # (K, 2)
            ci = chunk[:, 0]
            cj = chunk[:, 1]

            # ---- Normal vector windows ----
            # windows_n: shape (K, fw_n, fw_n)
            windows_n = self.find_all_windows_vectorized(chunk, fw_n)

            # Vectorized dot product with smoothing matrices.
            # smoothed_vector_i/j have shape (fw_n, fw_n).
            # Result shape: (K,)
            nv_i = -np.einsum('kwh,wh->k', windows_n, self.smoothed_vector_i)
            nv_j = np.einsum('kwh,wh->k', windows_n, self.smoothed_vector_j)

            fval[ci, cj, 0] = nv_i
            fval[ci, cj, 1] = nv_j

            # Explicitly free before the curvature allocation to avoid
            # holding both window arrays in memory simultaneously.
            del windows_n

            # ---- Curvature windows ----
            # windows_c: shape (K, fw_c, fw_c)
            windows_c = self.find_all_windows_vectorized(chunk, fw_c)

            # Per-pixel curvature loop over the chunk.
            # output_smoothed_matrix and calculate_curvature are not
            # trivially vectorizable without rewriting myInput, so we
            # retain a loop here — but it runs over K << nx*ny pixels
            # with windows already fetched as a contiguous array.
            for idx in range(len(chunk)):
                sm = myInput.output_smoothed_matrix(
                    windows_c[idx], smoothing_mat
                )[self.loop_times: -self.loop_times,
                  self.loop_times: -self.loop_times]
                fval[ci[idx], cj[idx], 2] = self.calculate_curvature(sm)

            del windows_c   # free before next chunk iteration

        core_etime = datetime.datetime.now()
        if self.verification_system:
            print("fast core time is " +
                  str((core_etime - core_stime).total_seconds()))

        return (fval, (core_etime - core_stime).total_seconds())

    def linear_combined_core_fast_dispatch(self, core_input):
        """
        Picklable dispatcher for linear_combined_core_fast.

        multiprocessing.Pool.apply_async requires all submitted callables to
        be picklable. Lambda functions and local closures are NOT picklable,
        so we cannot pass 'lambda inp: self.linear_combined_core_fast(inp,
        chunk_size=N)' directly to the pool.

        Instead, linear_main stores chunk_size on the instance
        (self.fast_chunk_size) before spawning workers, and this method
        reads it. Because the linear_class instance is forked into each
        worker process (not pickled on-the-fly), self.fast_chunk_size is
        available in every worker.

        Parameters
        ----------
        core_input : np.ndarray
            Same format as linear_combined_core.

        Returns
        -------
        tuple : same as linear_combined_core_fast
        """
        return self.linear_combined_core_fast(core_input)

    # -----------------------------------------------------------------------
    # Result callback
    # -----------------------------------------------------------------------

    def res_back(self, back_result):
        res_stime = datetime.datetime.now()
        (fval, core_time) = back_result
        if core_time > self.running_coreTime:
            self.running_coreTime = core_time

        if self.verification_system:
            print("res_back start...")

        if fval.shape[2] == 1:
            self.C[1, :, :] += fval[:, :, 0]
        elif fval.shape[2] == 2:
            self.P[1, :, :] += fval[:, :, 0]
            self.P[2, :, :] += fval[:, :, 1]
        elif fval.shape[2] == 3:
            self.P[1, :, :] += fval[:, :, 0]
            self.P[2, :, :] += fval[:, :, 1]
            self.C[1, :, :] += fval[:, :, 2]

        res_etime = datetime.datetime.now()
        if self.verification_system:
            print("my res time is " +
                  str((res_etime - res_stime).total_seconds()))

    # -----------------------------------------------------------------------
    # Main execution
    # -----------------------------------------------------------------------

    def linear_main(self, purpose="inclination", fast: bool = False,
                    chunk_size: int = 50_000):
        """Main execution function for linear smoothing algorithm.

        Controls the overall workflow including parallel processing setup,
        smoothing operations, normal vector calculation, and error calculation.

        Parameters
        ----------
        purpose : str
            Type of calculation: "inclination", "curvature", or "both".
        fast : bool
            If True, use the vectorized chunked core (linear_combined_core_fast)
            for purpose="both". Ignored for other purposes (fast cores are only
            implemented for the combined path).
            Default: False (standard scalar path).
        chunk_size : int
            Number of boundary pixels per vectorized batch in fast mode.
            Stored on the instance so that the picklable dispatcher can read it.
            Reduce if OOM is observed; increase for speed on memory-rich nodes.
            Default: 50_000  (≈740 MB peak per chunk at loop_times=20, fw=43).
        """
        starttime = datetime.datetime.now()

        # Store chunk_size on instance so linear_combined_core_fast_dispatch
        # can read it from forked worker processes without pickling issues.
        self.fast_chunk_size = chunk_size

        # Setup parallel processing
        pool = mp.Pool(processes=self.cores)
        main_lc, main_wc = myInput.split_cores(self.cores)

        # Split domain for parallel processing
        all_sites = np.array(
            [[x, y] for x in range(self.nx) for y in range(self.ny)]
        ).reshape(self.nx, self.ny, 2)
        multi_input = myInput.split_IC(all_sites, self.cores, 2, 0, 1)

        res_list = []

        if purpose == "inclination":
            for ki in range(main_wc):
                for kj in range(main_lc):
                    res_one = pool.apply_async(
                        func=self.linear_normal_vector_core,
                        args=(multi_input[ki][kj],),
                        callback=self.res_back)
                    res_list.append(res_one)

        elif purpose == "curvature":
            for ki in range(main_wc):
                for kj in range(main_lc):
                    res_one = pool.apply_async(
                        func=self.linear_curvature_core,
                        args=(multi_input[ki][kj],),
                        callback=self.res_back)
                    res_list.append(res_one)

        elif purpose == "both":
            # Select core function based on fast flag.
            # fast=True  → linear_combined_core_fast_dispatch (vectorized, chunked)
            # fast=False → linear_combined_core               (original scalar)
            if fast:
                target_core = self.linear_combined_core_fast_dispatch
            else:
                target_core = self.linear_combined_core

            for ki in range(main_wc):
                for kj in range(main_lc):
                    res_one = pool.apply_async(
                        func=target_core,
                        args=(multi_input[ki][kj],),
                        callback=self.res_back)
                    res_list.append(res_one)

        # Wait for all processes to complete
        pool.close()
        pool.join()

        if self.verification_system:
            print("core done!")

        # Calculate timing and errors
        endtime = datetime.datetime.now()
        self.running_time = (endtime - starttime).total_seconds()

        if purpose == "inclination":
            self.get_errors()
        elif purpose == "curvature":
            self.get_curvature_errors()
        elif purpose == "both":
            self.get_errors()
            self.get_curvature_errors()


# ---------------------------------------------------------------------------
# Module self-test
# ---------------------------------------------------------------------------

if __name__ == '__main__':

    nx, ny = 50, 50
    ng = 2
    max_iteration = 5
    radius = 20
    filename_save = (f"examples/curvature_calculation/"
                     f"BL_Curvature_R{radius}_Iteration_1_{max_iteration}")

    BL_errors = np.zeros(max_iteration)
    BL_runningTime = np.zeros(max_iteration)

    P0, R = myInput.Circle_IC(nx, ny, radius)

    for cores in [1]:
        for loop_times in range(4, max_iteration):
            test1 = linear_class(nx, ny, ng, cores, loop_times, P0, R)
            test1.linear_main("curvature")
            C_ln = test1.get_C()

            print('loop_times = ' + str(test1.loop_times))
            print('running_time = %.2f' % test1.running_time)
            print('running_core time = %.2f' % test1.running_coreTime)
            print('total_errors = %.2f' % test1.errors)
            print('per_errors = %.3f' % test1.errors_per_site)
            print()

            BL_errors[loop_times - 1] = test1.errors_per_site
            BL_runningTime[loop_times - 1] = test1.running_coreTime
