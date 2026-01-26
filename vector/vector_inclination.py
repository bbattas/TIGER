from ExodusBasics import ExodusBasics
from MultiExodusReaderDerivs import MultiExodusReaderDerivs
import PACKAGE_MP_Linear as smooth
import myInput

import glob
import time
import numpy as np
import argparse
import sys
import logging
import matplotlib.pyplot as plt
import math
from pathlib import Path



# Assumes uniform constant unchanging mesh, dx = dy and treats all index values as the coords

def parse_args():
    VALID_CPUS = (1, 2, 4, 8, 16, 32, 64, 128)
    p = argparse.ArgumentParser(
        description="Convert phase field GG results to inclination measurements,"
                    " using Lin's VECTOR smooting and inclination algorithms. "
                    "This code assumes uniform mesh elements.")

    # ---- General ----
    log = p.add_argument_group("General")
    log.add_argument("-v", "--verbose", action="count", default=0,
                   help="Increase verbosity (-v, -vv, -vvv).")
    log.add_argument('-n','--cpus',type=int,default=4, choices=VALID_CPUS,
                     help='Number of CPUs for smoothing/inclination calculations.'
                     'Allowed: powers of 2 (1, 2, 4, 8, 16, 32, 64, 128...). Default = 4')
    log.add_argument("-s", "--subdirs",action="store_true",
            help="Search for *.e files one level down (./*/.e). If not set, only search current directory.")


    # ---- Time selection ----
    tim = p.add_argument_group("Time (choose one)")
    grp = tim.add_mutually_exclusive_group(required=True)
    grp.add_argument("-g", "--grains", type=int,
                     help="Target grain count; chooses timestep with grain_tracker closest to this value.")
    grp.add_argument("-t", "--time", type=float,
                     help="Target time; chooses timestep with time_whole closest to this value.")

    # ---- CSV options ----
    cs = p.add_argument_group("CSV")
    cs.add_argument("--skip-csv", action="store_true",
               help="Skip computing/writing the polar histogram CSV output.")
    cs.add_argument("--csv-up", action="store_true",
               help="Write csv output up one directory from current (../file.csv).")

    # ---- Plot options ----
    plot = p.add_argument_group("Plotting")
    plot.add_argument('--plot','-p',action='store_true',
                            help='Save a polar plot of inclination, default=False')
    plot.add_argument('--debug-plot','-d',action='store_true',
                            help='Save the debugging plots, default=False.'
                            '-vv also enables this.')


    return p.parse_args()


def setup_logging(verbosity: int) -> logging.Logger:
    level = logging.WARNING
    if verbosity >= 2:
        level = logging.DEBUG
    elif verbosity >= 1:
        level = logging.INFO
    # log.warning("This prints by default")
    # log.info("This prints with -v")
    # log.debug("This prints with -vv")
    logging.basicConfig(level=level, format="%(message)s")
    return logging.getLogger("TIGER")


def tf(ti,log,extra=None):
    if extra is not None:
        log.warning(extra + f"Time: {(time.perf_counter()-ti):.4}s")
    else:
        log.warning(f"Time: {(time.perf_counter()-ti):.4}s")

def vtf(ti,log,extra=None):
    if extra is not None:
        log.info(extra + f"Time: {(time.perf_counter()-ti):.4}s")
    else:
        log.info(f"Time: {(time.perf_counter()-ti):.4}s")


def find_exodus_files(*, subdirs: bool = False, pattern: str = "*.e") -> list[Path]:
    """
    Find Exodus files in current directory or one-level-down subdirectories.
    Returns sorted list of Paths.
    """
    cwd = Path.cwd()

    if subdirs:
        files = sorted(cwd.glob(f"*/{pattern}"))
    else:
        files = sorted(cwd.glob(pattern))

    # Optional: ignore hidden dirs/files or non-files
    files = [p for p in files if p.is_file()]
    return files


def exodus_stem(exo_path: Path) -> str:
    """
    Convert '/a/b/file_out.e' -> 'file'
            './file.e'       -> 'file'
    """
    name = exo_path.name  # just filename
    if name.endswith(".e"):
        name = name[:-2]   # drop '.e'
    if name.endswith("_out"):
        name = name[:-4]   # drop '_out'
    return name


def closest_index(values: np.ndarray, target: float) -> int:
    """Return index of entry closest to target. Ties -> first occurrence."""
    values = np.asarray(values)
    return int(np.argmin(np.abs(values - target)))


def select_step(exo, *, grains: int | None, time_value: float | None, log: logging.Logger) -> int:
    """
    exo: an open ExodusBasics instance
    Returns: timestep index (0-based)
    """
    times = exo.time()

    if time_value is not None:
        step = closest_index(times, float(time_value))
        log.info(f"Frame selected by time: requested={time_value}, chosen step={step}, time={times[step]}")
        return step

    # grains path: require grain_tracker
    glo_names = exo.glo_varnames()
    if "grain_tracker" not in glo_names:
        raise RuntimeError(
            "You requested --grains, but this Exodus file has no global variable 'grain_tracker'. "
            f"Available global vars: {glo_names}"
        )

    gt = exo.glo_var_series("grain_tracker")
    # Often stored as float; treat as counts for matching
    gt_counts = np.rint(gt).astype(np.int64)

    step = closest_index(gt_counts, int(grains))
    log.info(f"Frame selected by grains: requested={grains}, chosen step={step}, grain_tracker={gt_counts[step]}")
    return step


def map_to_grid(xc, yc, val, *, tol=1e-6, fill_value=np.nan, reduce=None):
    """
    Map per-element values onto a structured 2D grid using xc,yc element "locations".
    - tol: quantization tolerance for floating noise (in same units as x/y)
    - reduce: None | 'max' | 'min' | 'sum' (handles collisions)
    """
    # Quantize coordinates to stable integer keys (avoids fragile rounding)
    kx = np.rint(xc / tol).astype(np.int64)
    ky = np.rint(yc / tol).astype(np.int64)

    # Unique axes + inverse indices (direct i/j mapping)
    ux, j = np.unique(kx, return_inverse=True)   # columns
    uy, i = np.unique(ky, return_inverse=True)   # rows

    # Convert axis keys back to coordinate values (for labels)
    x_centers = ux.astype(np.float64) * tol
    y_centers = uy.astype(np.float64) * tol

    P0 = np.full((len(y_centers), len(x_centers)), fill_value, dtype=np.float64 if np.isnan(fill_value) else val.dtype)

    if reduce is None:
        # If you *know* it's one element per cell, this is fastest
        P0[i, j] = val
    else:
        # Handle collisions safely
        if reduce == "max":
            P0[i, j] = -np.inf
            np.maximum.at(P0, (i, j), val)
        elif reduce == "min":
            P0[i, j] = np.inf
            np.minimum.at(P0, (i, j), val)
        elif reduce == "sum":
            P0[i, j] = 0.0
            np.add.at(P0, (i, j), val)
        else:
            raise ValueError("reduce must be None, 'max', 'min', or 'sum'")

    return P0, x_centers, y_centers

# From LIN
def get_normal_vector(P0,args,log):
    """Calculate normal vectors for grain boundaries in 2D microstructure.
    Uses smooth linear interpolation method.

    Args:
        P0 (ndarray): 2D microstructure array
        args (argparse): argparse options

    Returns:
        tuple: (smooth_field, boundary_sites, boundary_sites_by_grain) - Smoothed field and boundary locations
    """
    nx = P0.shape[0]
    ny = P0.shape[1]
    ng = np.max(P0)
    cores = args.cpus
    loop_times = 5
    R = np.zeros((nx,ny,2))
    verb = False
    if args.verbose >=1:
        verb = True
    smooth_class = smooth.linear_class(nx,ny,ng,cores,loop_times,P0,R,verification_system=verb)

    smooth_class.linear_main("inclination")
    P = smooth_class.get_P()
    sites = smooth_class.get_all_gb_list()
    sites_together = []
    for id in range(len(sites)): sites_together += sites[id]
    log.info(f"Total num of GB sites: {len(sites_together)}")

    return P, sites_together, sites


def get_normal_vector_slope(P, sites, para_name='Inc', bias=None, output=False):
    """Calculate and plot slope distribution of normal vectors in 2D.

    Args:
        P (ndarray): Smoothed field containing normal vector information
        sites (list): List of boundary site coordinates
        para_name (str): Parameter name for plot legend
        bias (float, optional): Optional bias to add to distribution

    Returns:
        float: Always returns 0 (used for tracking plot generation)
    """
    xLim = [0, 360]
    binValue = 10.01
    binNum = round((abs(xLim[0])+abs(xLim[1]))/binValue)
    xCor = np.linspace((xLim[0]+binValue/2),(xLim[1]-binValue/2),binNum)

    freqArray = np.zeros(binNum)
    degree = []
    for sitei in sites:
        [i,j] = sitei
        dx,dy = myInput.get_grad(P,i,j)
        degree.append(math.atan2(-dy, dx) + math.pi)
    for i in range(len(degree)):
        freqArray[int((degree[i]/math.pi*180-xLim[0])/binValue)] += 1
    freqArray = freqArray/sum(freqArray*binValue)

    # bias situation
    if bias is not None:
        freqArray = freqArray + bias
        freqArray = freqArray/sum(freqArray*binValue)
    # # Plot
    # plt.plot(xCor/180*math.pi, freqArray, linewidth=2, label=para_name)

    # Plot (close the loop so 0/360 connects)
    theta = xCor/180*math.pi          # bin centers in radians
    r = freqArray
    theta_closed = np.r_[theta, theta[0] + 2*np.pi]
    r_closed     = np.r_[r,     r[0]]
    if output:
        return theta_closed, r_closed
    else:
        plt.plot(theta_closed, r_closed, linewidth=2, label=para_name)
        return 0


def plot_inclination(args, log, P, sites, para_name, stem, bias=None):
    # SINGLE POLAR PLOT THEN CLOSE
    plt.close()  # Clear any existing plots to prevent interference
    fig = plt.figure(figsize=(5, 5))  # Square figure for balanced polar representation
    ax = fig.add_subplot(111, projection="polar")
    ax.set_thetagrids(np.arange(0.0, 360.0, 45.0),fontsize=16)  # 45-degree angular grid intervals
     # Angular Coordinate Configuration (Theta Axis)
    # Configure angular grid lines and labels for crystallographic orientation analysis
    ax.set_thetamin(0.0)    # Minimum angular range (0 degrees)
    ax.set_thetamax(360.0)  # Maximum angular range (360 degrees - full circle)
    # Radial Coordinate Configuration (R Axis)
    # Configure radial grid lines for probability density visualization
    ax.set_rgrids(np.arange(0, 0.01, 0.004))  # Radial grid lines every 0.004 units
    ax.set_rlabel_position(0.0)  # Position radial labels at 0-degree angle
    ax.set_rlim(0.0, 0.01)       # Radial axis limits for probability density range
    ax.set_yticklabels(['0', '4e-3', '8e-3'],fontsize=16)  # Custom radial axis labels
    # Enhanced Grid and Formatting Configuration
    ax.grid(True, linestyle="-", color="k", linewidth=0.5, alpha=0.5)  # Professional grid styling
    ax.set_axisbelow('True')  # Ensure grid lines appear behind data plots

    # Generate Normal Vector Slope Distribution Analysis
    # Compute and visualize inclination angle distributions with bias correction
    # Parameters: P - probability data, sites - boundary sites, timestep, method identifier
    slope_list = get_normal_vector_slope(P, sites, para_name)

    plt.legend(loc=(-0.12,-0.35),fontsize=16,ncol=3)
    figname = stem + '_inclination.png'
    plt.savefig(figname, dpi=400,bbox_inches='tight')
    plt.close(fig=fig)
    return


def debug_plot(P0,sites,stem):
    dfig = plt.figure(figsize=(10, 4))
    ax0 = dfig.add_subplot(1, 2, 1)
    ax1 = dfig.add_subplot(1, 2, 2)

    im = ax0.imshow(P0, origin='lower', aspect='auto')  # origin lower so y increases upward
    plt.colorbar(im,ax=ax0)
    ax0.set_aspect('equal')

    nx = P0.shape[0]
    ny = P0.shape[1]
    arr = np.array(sites)
    x = arr[:, 0]
    y = arr[:, 1]
    ax1.scatter(x, y, s=5, c='red')
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_aspect('equal')
    ax1.set_xlim([0,nx])
    ax1.set_ylim([0,ny])

    dfig.tight_layout()
    figname = stem + '_DEBUG_GB_plot.png'
    dfig.savefig(figname,dpi=500, transparent=True)
    plt.close(dfig)


def debug_quiver(
    P,
    sites,
    x_centers,
    y_centers,
    stem,
    *,
    get_grad=myInput.get_grad,
    normalize=True,
    scale=0.5,
    width=0.003,
    background=None,
    figsize=(7, 6),
    title="Normals from get_grad at boundary sites",
    ax=None,
    show=True,
):
    """
    Plot quiver arrows at given (i,j) sites using dx,dy from get_grad(P,i,j).

    Parameters
    ----------
    P : ndarray
        FIELD passed into get_grad. (Typically a 2D label map or a 3D P array with P[0,:,:] labels,
        depending on how get_grad is implemented in your repo.)
    sites : iterable of (i,j)
        Grid indices where you want vectors (e.g., boundary sites).
    x_centers : (nx,) array
        Physical x coordinate for each column j.
    y_centers : (ny,) array
        Physical y coordinate for each row i.
    get_grad : callable
        Function with signature get_grad(P, i, j) -> (dx, dy)
        e.g. myInput.get_grad
    normalize : bool
        Normalize (dx,dy) to unit vectors for direction-only arrows.
    background : 2D array or None
        Optional background to plot with pcolormesh; should be shape (ny,nx) if provided.

    Returns
    -------
    ax : matplotlib Axes
    """
    x_centers = np.asarray(x_centers)
    y_centers = np.asarray(y_centers)

    # Collect points and vectors
    xs, ys, us, vs = [], [], [], []
    for (i, j) in sites:
        dx, dy = get_grad(P, int(i), int(j))
        xs.append(x_centers[j])
        ys.append(y_centers[i])
        us.append(dx)
        vs.append(dy)

    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    us = np.asarray(us, dtype=float)
    vs = np.asarray(vs, dtype=float)

    # Optional normalize
    if normalize:
        mag = np.sqrt(us**2 + vs**2)
        nz = mag > 0
        us[nz] /= mag[nz]
        vs[nz] /= mag[nz]

    # Create axes if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # Optional background
    if background is not None:
        bg = np.asarray(background)
        X, Y = np.meshgrid(x_centers, y_centers)
        ax.pcolormesh(X, Y, bg, shading="nearest")

    ax.set_aspect("equal")
    ax.quiver(xs, ys, us, vs, angles="xy", scale_units="xy", scale=scale, width=width)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    plt.tight_layout()
    figname = stem + '_DEBUG_GB_quiver_plot.png'
    fig.savefig(figname,dpi=500, transparent=True)
    plt.close(fig)




def main():
    ti = time.perf_counter()
    args = parse_args()
    log = setup_logging(args.verbose)
    log.info('Setup:')
    log.info(f'Arguments: {args}')


    # exofile = glob.glob('*.e')[0]
    exo_files = find_exodus_files(subdirs=args.subdirs)
    if not exo_files:
        where = "subdirectories" if args.subdirs else "current directory"
        raise SystemExit(f"No .e files found in {where}.")

    log.info(' ')
    log.info('Exodus Files:')
    for ef in exo_files:
        log.info(f'  File: {ef}')
        log.info(f'  Namebase: {exodus_stem(ef)}')
        log.info(' ')

    for cnt,exofile in enumerate(exo_files):
        til = time.perf_counter()
        stem = exodus_stem(exofile)
        if len(exo_files) > 1:
            log.warning(' ')
        log.warning('\033[1m\033[96m'+'File '+str(cnt+1)+'/'+str(len(exo_files))+': '+'\x1b[0m'+str(stem))

        try:
            with ExodusBasics(exofile) as exo:
                # Pick the right timestep
                step = select_step(exo, grains=args.grains, time_value=args.time, log=log)
                # # Pull a couple reference values:
                # ug_max = exo.elem_var_at_step("unique_grains", step=0).max()
                # t = exo.time()[step]
                # Get data for inclination
                xc, yc = exo.element_centers_xy(method="mean")
                ug = exo.elem_var_at_step("unique_grains", step=step)
                vtf(ti,log,"End of Exodus ripping: ")
                log.info(' ')

                # VECTOR PORTION
                tiv = time.perf_counter()
                # Make Array for Lins approach
                # MIGHT need to adjust or rotate it?
                P0, xcen, ycen = map_to_grid(xc, yc, ug, tol=1e-12, fill_value=np.nan)
                # Normal vector
                P, sites, _ = get_normal_vector(P0,args,log)

                # Write histogram inclination to csv
                if not args.skip_csv:
                    theta_out, r_out = get_normal_vector_slope(P, sites, output=True)
                    theta_out = np.asarray(theta_out)
                    r_out = np.asarray(r_out)
                    if theta_out.shape != r_out.shape:
                        raise ValueError(f"theta and r must have same shape, got {theta_out.shape} vs {r_out.shape}")
                    prefix = '../' if args.csv_up else ''
                    out_name = prefix + stem + '_inc_hist.csv'
                    log.info(f'Writing histogram inclination points to {out_name}')
                    np.savetxt(
                        out_name,
                        np.column_stack((theta_out, r_out)),
                        delimiter=",",
                        header="theta,r",
                        comments=""
                    )

                # Debug Plots
                if args.verbose >= 2 or args.debug_plot:
                    debug_plot(P0,sites,stem)
                    # quiver_normals_xy(P, xcen, ycen, background=None, step=1, normalize=True)
                    debug_quiver(P,sites,xcen,ycen,stem,normalize=True,scale=0.3)

                vtf(tiv,log,"End of VECTOR calculations: ")
                log.info(' ')

                # Inclination Plot
                if args.plot:
                    log.info('Plotting single file inclination')
                    plot_inclination(args, log, P, sites, 'Inc', stem, bias=None)
                    log.info(' ')

                if len(exo_files) > 1:
                    tf(til,log,extra=f"File {cnt+1} ")




        except Exception as e:
            # argparse-style error output (clean for CLI)
            log.warning(f"ERROR: {e}")#, file=sys.stderr)
            sys.exit(2)

    # TOTAL end time
    if len(exo_files) > 1:
        log.warning(' ')
    tf(ti,log,extra="Total ")

if __name__ == "__main__":
    main()

