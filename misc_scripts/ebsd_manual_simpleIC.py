import numpy as np
from datetime import datetime
import argparse

# --- Your region / feature functions (keep/modify as you like) ----------------

def four_grains(X, Y, L=120.0):
    d1 = X > Y
    d2 = X < (L - Y)
    out = np.zeros_like(X, dtype=np.int32)
    # Example mapping (yours)
    out[(~d1) & (~d2)] = 2
    out[(~d1) & ( d2)] = 1
    out[( d1) & (~d2)] = 1 #3
    out[( d1) & ( d2)] = 2 #4
    return out

def circle_field(X, Y, cx=60.0, cy=60.0, r=30.0, inside_value=0, outside_value=-1):
    inside = (X - cx)**2 + (Y - cy)**2 <= r**2
    return np.where(inside, inside_value, outside_value)

def feat_to_euler(featID):
    if featID == 0:
        return 0.116174, 2.359454, 3.775561
    elif featID == 1 or featID == 3:
        return 5.333997, 1.028163, 5.158241
    elif featID == 2 or featID == 4:
        return 4.757033, 0.755838, 1.956608
    else:
        # fallback (or raise)
        return 0.0, 0.0, 0.0

def feat_to_euler2(featID,case=0):
    if case == 0: #01- 8, 02- 51, 12- 47
        if featID == 0:
            return np.radians(200.896956), np.radians(83.457538), np.radians(29.106038)
        elif featID == 1 or featID == 3:
            return np.radians(197.075987), np.radians(81.618801), np.radians(23.033282)
        elif featID == 2 or featID == 4:
            return np.radians(161.498961), np.radians(103.523997), np.radians(358.479164)
        else:
            return 0.0, 0.0, 0.0
    if case == 1:
        if featID == 0:
            return np.radians(200.896956), np.radians(83.457538), np.radians(29.106038)
        elif featID == 2 or featID == 4:
            return np.radians(197.075987), np.radians(81.618801), np.radians(23.033282)
        elif featID == 1 or featID == 3:
            return np.radians(161.498961), np.radians(103.523997), np.radians(358.479164)
        else:
            return 0.0, 0.0, 0.0
    elif case == 2:
        if featID == 1 or featID == 3:
            return np.radians(200.896956), np.radians(83.457538), np.radians(29.106038)
        elif featID == 0:
            return np.radians(197.075987), np.radians(81.618801), np.radians(23.033282)
        elif featID == 2 or featID == 4:
            return np.radians(161.498961), np.radians(103.523997), np.radians(358.479164)
        else:
            return 0.0, 0.0, 0.0
    elif case == 3:
        if featID == 2 or featID == 4:
            return np.radians(200.896956), np.radians(83.457538), np.radians(29.106038)
        elif featID == 1 or featID == 3:
            return np.radians(197.075987), np.radians(81.618801), np.radians(23.033282)
        elif featID == 0:
            return np.radians(161.498961), np.radians(103.523997), np.radians(358.479164)
        else:
            return 0.0, 0.0, 0.0
    else:
        return 0.0, 0.0, 0.0

def two_grains(X, Y, L=120.0):
    out = np.ones_like(X, dtype=np.int32)
    return out

def feat_to_euler2gr(featID,case=0):
    if case == 0: #01- 8
        if featID == 0:
            return np.radians(200.896956), np.radians(83.457538), np.radians(29.106038)
        elif featID == 1:
            return np.radians(197.075987), np.radians(81.618801), np.radians(23.033282)
        else:
            return 0.0, 0.0, 0.0
    elif case == 1: #01- 51
        if featID == 0:
            return np.radians(200.896956), np.radians(83.457538), np.radians(29.106038)
        elif featID == 1:
            return np.radians(161.498961), np.radians(103.523997), np.radians(358.479164)
        else:
            return 0.0, 0.0, 0.0
    else:
        return 0.0, 0.0, 0.0

# --- Writer ------------------------------------------------------------------

def write_dream3d_ebsd_txt(
    out_path: str,
    X: np.ndarray,
    Y: np.ndarray,
    feature_ids: np.ndarray,
    L: float,
    dx: float,
    phase_name: str = "Primary",
    symmetry: int = 43,
    phase_id: int = 1,
    fid_offset: int = 0,
    case: int = 0,
    bicr: bool = False
):
    """
    Writes a DREAM3D-like EBSD text file:
    phi1 PHI phi2 x y z FeatureId PhaseId Symmetry

    Assumes X,Y,feature_ids are same shape and represent cell centers.
    """

    if X.shape != Y.shape or X.shape != feature_ids.shape:
        raise ValueError("X, Y, and feature_ids must have the same shape")

    # Flatten in DREAM3D-like ordering: x varies fastest, then y
    x_flat = X.ravel(order="C")
    y_flat = Y.ravel(order="C")
    fid_flat = feature_ids.ravel(order="C").astype(np.int64)

    # If you have "void" / background IDs you want to exclude from feature counting, handle here:
    # Example: exclude -1
    valid_mask = fid_flat != -1

    unique_fids = np.unique(fid_flat[valid_mask])
    num_features = int(unique_fids.size)

    # Header values consistent with your sample:
    # - X_STEP is the spacing (dx)
    # - X_MIN/Y_MIN are 0
    # - X_MAX/Y_MAX are L
    # - X_DIM/Y_DIM are N
    N = X.shape[0]
    now = datetime.now().strftime("%a %b %d %H:%M:%S %Y")

    with open(out_path, "w") as f:
        f.write("# File written from Python Script\n")
        f.write(f"# DateTime: {now}\n")
        f.write(f"# X_STEP: {dx:.6f}\n")
        f.write(f"# Y_STEP: {dx:.6f}\n")
        f.write("# Z_STEP: 0.000000\n")
        f.write("#\n")
        f.write("# X_MIN: 0.000000\n")
        f.write("# Y_MIN: 0.000000\n")
        f.write("# Z_MIN: 0.000000\n")
        f.write("#\n")
        f.write(f"# X_MAX: {L:.6f}\n")
        f.write(f"# Y_MAX: {L:.6f}\n")
        f.write("# Z_MAX: 0.000000\n")
        f.write("#\n")
        f.write(f"# X_DIM: {N}\n")
        f.write(f"# Y_DIM: {N}\n")
        f.write("# Z_DIM: 0\n")
        f.write("#\n")
        f.write(f"# Phase_1: {phase_name}\n")
        f.write(f"# Symmetry_1: {symmetry}\n")
        f.write(f"# Features_1: {num_features}\n")
        f.write("#\n")
        f.write(f"# Num_Features: {num_features} \n")
        f.write("#\n")
        f.write("# phi1 PHI phi2 x y z FeatureId PhaseId Symmetry\n")

        # Data rows
        for x, y, fid in zip(x_flat, y_flat, fid_flat):
            if fid == -1:
                # If you want to skip background points entirely, continue.
                # If you want them written, remove this continue and decide on eulers for fid=-1.
                continue
            if bicr:
                ea, eb, ec = feat_to_euler2gr(int(fid),case=case)
            else:
                ea, eb, ec = feat_to_euler2(int(fid),case=case)
            f.write(
                f"{ea:.6f} {eb:.6f} {ec:.6f} "
                f"{x:.6f} {y:.6f} 0.000000 "
                f"{int(fid) + fid_offset} {phase_id} {symmetry}\n"
            )

# --- Main --------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--output", default="manual_ebsd.txt")
    ap.add_argument("--L", type=float, default=120.0, help="Domain max xy coordinate")
    ap.add_argument("--N", type=int, default=120, help="Number of elements")
    ap.add_argument("-r", type=float, default=30.0, help="Center grain radius")
    ap.add_argument("--fid_offset", type=int, default=0, help="Use 1 for 1-based FeatureId output")
    ap.add_argument("--case", type=int, default=0, help="[0,1,2,3] to change which grain has which orientation.")
    ap.add_argument("--bicr", action='store_true', help="Create a true bicrystal")
    args = ap.parse_args()

    L = args.L
    N = args.N
    dx = L / N

    coord_space = np.linspace(dx/2, L - dx/2, N)
    X, Y = np.meshgrid(coord_space, coord_space, indexing="xy")

    # Build feature IDs for each cell center
    if args.bicr:
        out = two_grains(X, Y, L=L)
    else:
        out = four_grains(X, Y, L=L)

    feature_ids = circle_field(X, Y, cx=L/2, cy=L/2, r=args.r, inside_value=0, outside_value=out).astype(np.int32)

    write_dream3d_ebsd_txt(
        out_path=args.output,
        X=X,
        Y=Y,
        feature_ids=feature_ids,
        L=L,
        dx=dx,
        fid_offset=args.fid_offset,
        case=args.case,
        bicr=args.bicr
    )

if __name__ == "__main__":
    main()
