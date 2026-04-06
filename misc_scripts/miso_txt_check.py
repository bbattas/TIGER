'''
python script.py euler_256.txt 5
optional extras: txt_path.txt num_grains_max
'''
import numpy as np
from scipy.spatial.transform import Rotation
import sys
import pandas as pd

# ── 1. Cubic symmetry operators (24) ─────────────────────────────────────────
SYM_MATRICES = np.array([
    [[1,0,0],[0,1,0],[0,0,1]],    [[1,0,0],[0,-1,0],[0,0,-1]],
    [[1,0,0],[0,0,-1],[0,1,0]],   [[1,0,0],[0,0,1],[0,-1,0]],
    [[-1,0,0],[0,1,0],[0,0,-1]],  [[-1,0,0],[0,-1,0],[0,0,1]],
    [[-1,0,0],[0,0,-1],[0,-1,0]], [[-1,0,0],[0,0,1],[0,1,0]],
    [[0,1,0],[-1,0,0],[0,0,1]],   [[0,1,0],[0,0,-1],[-1,0,0]],
    [[0,1,0],[1,0,0],[0,0,-1]],   [[0,1,0],[0,0,1],[1,0,0]],
    [[0,-1,0],[1,0,0],[0,0,1]],   [[0,-1,0],[0,0,-1],[1,0,0]],
    [[0,-1,0],[-1,0,0],[0,0,-1]], [[0,-1,0],[0,0,1],[-1,0,0]],
    [[0,0,1],[0,1,0],[-1,0,0]],   [[0,0,1],[1,0,0],[0,1,0]],
    [[0,0,1],[0,-1,0],[1,0,0]],   [[0,0,1],[-1,0,0],[0,-1,0]],
    [[0,0,-1],[0,1,0],[1,0,0]],   [[0,0,-1],[-1,0,0],[0,1,0]],
    [[0,0,-1],[0,-1,0],[-1,0,0]], [[0,0,-1],[1,0,0],[0,-1,0]],
], dtype=float)

# All 24x24 = 576 symmetry product quaternions, shape (576, 4) [x,y,z,w]
def build_sym_pairs():
    sym = Rotation.from_matrix(SYM_MATRICES)  # shape (24,)
    # All products s1 * s2.inv() — we apply s1 to qi and s2 to qj separately,
    # so we just store them as two flat arrays for broadcasting
    return sym  # will use directly

SYM_ROTS = Rotation.from_matrix(SYM_MATRICES)  # (24,)
SYM_QUAT = SYM_ROTS.as_quat()                  # (24, 4)  [x,y,z,w]


# ── 2. Quaternion multiply: (...,4) x (...,4) → (...,4)  [x,y,z,w] ──────────
def quat_mul(a, b):
    ax, ay, az, aw = a[...,0], a[...,1], a[...,2], a[...,3]
    bx, by, bz, bw = b[...,0], b[...,1], b[...,2], b[...,3]
    return np.stack([
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw,
        aw*bw - ax*bx - ay*by - az*bz,
    ], axis=-1)

def quat_inv(q):
    """Conjugate (assumes unit quaternions)."""
    inv = q.copy()
    inv[..., :3] *= -1
    return inv

def quat_normalize(q):
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


# ── 3. Read Euler file ────────────────────────────────────────────────────────
def read_euler_file(filepath):
    angles = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            try:
                angles.append([float(p) for p in parts[:3]])
            except ValueError:
                continue
    return np.array(angles)  # (N, 3) radians


# ── 4. Euler → quaternion array (N, 4) [x,y,z,w] ─────────────────────────────
def euler_to_quats(euler_angles):
    rots = Rotation.from_euler('ZXZ', euler_angles)
    return rots.as_quat()  # (N, 4)


# ── 5. Vectorized misorientation for ALL grain pairs ─────────────────────────
def compute_all_misorientations_vectorized(filepath, max_grain=None):
    euler_angles = read_euler_file(filepath)

    if max_grain is not None:
        euler_angles = euler_angles[:max_grain]

    N = len(euler_angles)
    print(f"Using {N} grains → {N*(N-1)//2} pairs")

    quats = euler_to_quats(euler_angles)  # (N, 4)
    quats = quat_normalize(quats)

    # Pre-apply all 24 symmetry ops to every grain: shape (24, N, 4)
    # s * q  for each symmetry s and each grain q
    sym_q = SYM_QUAT[:, np.newaxis, :]          # (24, 1, 4)
    grain_q = quats[np.newaxis, :, :]            # (1, N, 4)
    # broadcast multiply: (24, N, 4)
    sq = quat_normalize(quat_mul(sym_q, grain_q))  # (24, N, 4)

    # Build upper-triangle pair indices
    ii, jj = np.triu_indices(N, k=1)            # each shape (P,), P = N*(N-1)/2
    P = len(ii)

    # For each pair (i,j): we need to try all 24x24 sym combos
    # sq[:, ii, :] → (24, P, 4)  — symmetry-applied versions of grain i
    # sq[:, jj, :] → (24, P, 4)  — symmetry-applied versions of grain j

    sqi = sq[:, ii, :]   # (24, P, 4)
    sqj = sq[:, jj, :]   # (24, P, 4)

    # We need all 24x24 combos per pair.
    # Reshape to (24,1,P,4) and (1,24,P,4), then multiply → (24,24,P,4)
    sqi_exp = sqi[:, np.newaxis, :, :]   # (24, 1, P, 4)
    sqj_exp = sqj[np.newaxis, :, :, :]   # (1, 24, P, 4)

    # misorientation quaternion: qi_sym * qj_sym^{-1}
    mis_q = quat_normalize(quat_mul(sqi_exp, quat_inv(sqj_exp)))  # (24, 24, P, 4)

    # ── Fundamental zone check: 0 <= qz <= qy <= qx <= 1 ────────────────────
    qx = mis_q[..., 0]
    qy = mis_q[..., 1]
    qz = mis_q[..., 2]
    qw = mis_q[..., 3]

    in_fz     = (qz >= 0) & (qz <= qy) & (qy <= qx) & (qx <= 1.0)
    inv_in_fz = (-qz >= 0) & (-qz <= -qy) & (-qy <= -qx) & (-qx <= 1.0)
    valid     = in_fz | inv_in_fz                          # (24, 24, P)

    # Misorientation angle
    theta = 2.0 * np.arccos(np.clip(np.abs(qw), 0.0, 1.0))  # (24, 24, P)

    # Mask invalid combos with large angle so they never win argmin
    theta_masked = np.where(valid, theta, 2 * np.pi)        # (24, 24, P)

    # Best (minimum) angle over all 576 sym combos for each pair
    flat_theta = theta_masked.reshape(24*24, P)              # (576, P)
    flat_valid = valid.reshape(24*24, P)
    flat_in_fz = in_fz.reshape(24*24, P)
    flat_q     = mis_q.reshape(24*24, P, 4)

    best_idx   = np.argmin(flat_theta, axis=0)              # (P,)
    pair_idx   = np.arange(P)

    best_theta = flat_theta[best_idx, pair_idx]             # (P,)
    best_q     = flat_q[best_idx, pair_idx, :]              # (P, 4)
    best_in_fz = flat_in_fz[best_idx, pair_idx]             # (P,)

    # Flip sign if in inv_fz (conjugate = negate xyz)
    sign = np.where(best_in_fz[:, np.newaxis],
                    np.ones((P, 4)),
                    np.array([-1., -1., -1., -1.]))
    best_q = best_q * sign

    # ── Rotation axis (polar, azimuth) ───────────────────────────────────────
    axis  = best_q[:, :3]                                   # (P, 3)
    vnorm = np.linalg.norm(axis, axis=-1, keepdims=True)    # (P, 1)
    safe  = (vnorm[:, 0] > 1e-12)

    axis_n    = np.where(vnorm > 1e-12, axis / np.maximum(vnorm, 1e-30), 0.0)
    polar_ax  = np.where(safe, np.arccos(np.clip(axis_n[:, 2], -1, 1)), 0.0)
    azim_ax   = np.where(safe, np.arctan2(axis_n[:, 1], axis_n[:, 0]), 0.0)
    azim_ax   = np.where(azim_ax < 0, azim_ax + 2 * np.pi, azim_ax)

    return {
        'grain_i':    ii,
        'grain_j':    jj,
        'theta_rad':  best_theta,
        'theta_deg':  np.degrees(best_theta),
        'polar_ax':   polar_ax,
        'azim_ax':    azim_ax,
        'qmin':       best_q,
    }


# ── 6. getLineNum (mirrors C++) ───────────────────────────────────────────────
def get_line_num(grain_i, grain_j):
    if grain_i > grain_j:
        return grain_j + (grain_i - 1) * grain_i // 2
    else:
        return grain_i + (grain_j - 1) * grain_j // 2


# ── 7. Entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # filepath = sys.argv[1] if len(sys.argv) > 1 else "euler_256.txt"
    # res = compute_all_misorientations_vectorized(filepath)

    # # Query a specific pair
    # gi, gj = 0, 1
    # idx = get_line_num(gi, gj)
    # print(f"\nMisorientation grain {gi} vs {gj}:")
    # print(f"  theta      = {res['theta_deg'][idx]:.4f} deg")
    # print(f"  polar axis = {np.degrees(res['polar_ax'][idx]):.4f} deg")
    # print(f"  azim  axis = {np.degrees(res['azim_ax'][idx]):.4f} deg")
    # print(f"  qmin [x,y,z,w] = {res['qmin'][idx]}")

    # print("\nAll pairs:")
    # for k in range(len(res['grain_i'])):
    #     print(f"  ({res['grain_i'][k]:3d}, {res['grain_j'][k]:3d}) → {res['theta_deg'][k]:7.3f}°")
    # OUTPUT TO CSV
    filepath = sys.argv[1] if len(sys.argv) > 1 else "euler_256.txt"

    # Optional: pass a second argument to limit grain count, e.g. python script.py euler_256.txt 5
    max_grain = int(sys.argv[2]) if len(sys.argv) > 2 else None

    res = compute_all_misorientations_vectorized(filepath, max_grain=max_grain)

    df = pd.DataFrame({
        'i':       res['grain_i'],
        'j':       res['grain_j'],
        'degrees': res['theta_deg'],
        'radians': res['theta_rad']
    })

    out_path = "misorientation.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} pairs to '{out_path}'")
