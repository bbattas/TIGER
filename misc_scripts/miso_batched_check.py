'''
python script.py euler_256.txt 5
optional extras: txt_path.txt num_grains_max
'''
import numpy as np
from scipy.spatial.transform import Rotation
import sys
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import argparse

def get_args():
    parser = argparse.ArgumentParser(
        description="Compute pairwise misorientation angles from Euler angles file,"
        " in a batched approach for larger numbers of grains. Saves to parquet instead of csv"
    )
    parser.add_argument(
        "filepath",
        nargs="?",
        default="euler_256.txt",
        help="Path to the Euler angles .txt file (default: euler_256.txt)"
    )
    parser.add_argument(
        "--max-grains", "-n",
        type=int,
        default=None,
        dest="max_grain",
        help="Limit computation to the first N grains (default: all)"
    )
    parser.add_argument(
        "--chunk-size", "-c",
        type=int,
        default=50_000,
        dest="chunk_size",
        help="Number of pairs to process per chunk (default: 50000)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="misorientation.parquet",
        dest="out_path",
        help="Output Parquet file path (default: misorientation.parquet)"
    )

    args = parser.parse_args()
    return args

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


def canonicalize_quaternion_sign(q):
    """
    Force a unique sign convention for quaternion reporting.
    Since q and -q represent the same rotation, make w >= 0.
    """
    q = q.copy()
    mask = q[..., 3] < 0.0
    q[mask] *= -1.0
    return q


def pair_to_linear_index(i, j, N):
    """
    Index of pair (i, j) in the same order as np.triu_indices(N, k=1).
    """
    if i == j:
        raise ValueError("i and j must be different")
    if i > j:
        i, j = j, i
    return i * N - i * (i + 1) // 2 + (j - i - 1)


def build_pair_lookup(ii, jj):
    """
    Return a dict mapping (i, j) -> row index in the result arrays.
    """
    return {(int(i), int(j)): k for k, (i, j) in enumerate(zip(ii, jj))}

# ── 5. Batched misorientation for ALL grain pairs ─────────────────────────
def compute_all_misorientations_chunked(filepath, max_grain=None, chunk_size=50_000, out_path="misorientation.parquet"):
    euler_angles = read_euler_file(filepath)
    if max_grain is not None:
        euler_angles = euler_angles[:max_grain]

    N = len(euler_angles)
    print(f"Using {N} grains -> {N * (N - 1) // 2} pairs")

    quats = quat_normalize(euler_to_quats(euler_angles))
    sym_q = SYM_QUAT[:, np.newaxis, :]
    grain_q = quats[np.newaxis, :, :]
    sq = quat_normalize(quat_mul(sym_q, grain_q))  # (24, N, 4)

    ii, jj = np.triu_indices(N, k=1)
    P = len(ii)

    writer = None
    schema = pa.schema([
        ('i',         pa.int32()),
        ('j',         pa.int32()),
        ('angle_deg', pa.float32()),
        ('radians',   pa.float32()),
        ('ax_x',      pa.float32()),
        ('ax_y',      pa.float32()),
        ('ax_z',      pa.float32()),
    ])

    for start in range(0, P, chunk_size):
        end = min(start + chunk_size, P)
        ci, cj = ii[start:end], jj[start:end]
        Pc = end - start

        sqi = sq[:, ci, :]                        # (24, Pc, 4)
        sqj = sq[:, cj, :]
        sqi_exp = sqi[:, np.newaxis, :, :]        # (24, 1, Pc, 4)
        sqj_exp = sqj[np.newaxis, :, :, :]        # (1, 24, Pc, 4)

        mis_q = quat_normalize(quat_mul(sqi_exp, quat_inv(sqj_exp)))  # (24, 24, Pc, 4)

        qw = np.clip(np.abs(mis_q[..., 3]), 0.0, 1.0)
        theta = 2.0 * np.arccos(qw)

        flat_theta = theta.reshape(576, Pc)
        flat_q     = mis_q.reshape(576, Pc, 4)
        best_idx   = np.argmin(flat_theta, axis=0)
        pair_idx   = np.arange(Pc)

        best_theta = flat_theta[best_idx, pair_idx]
        best_q     = canonicalize_quaternion_sign(flat_q[best_idx, pair_idx])

        axis  = best_q[:, :3]
        vnorm = np.linalg.norm(axis, axis=-1, keepdims=True)
        safe  = vnorm[:, 0] > 1e-12
        axis_n = np.zeros_like(axis)
        axis_n[safe] = axis[safe] / vnorm[safe]

        batch = pa.table({
            'i':         pa.array(ci.astype(np.int32)),
            'j':         pa.array(cj.astype(np.int32)),
            'angle_deg': pa.array(best_theta * (180 / np.pi), type=pa.float32()),
            'radians':   pa.array(best_theta.astype(np.float32)),
            'ax_x':      pa.array(axis_n[:, 0].astype(np.float32)),
            'ax_y':      pa.array(axis_n[:, 1].astype(np.float32)),
            'ax_z':      pa.array(axis_n[:, 2].astype(np.float32)),
        }, schema=schema)

        if writer is None:
            writer = pq.ParquetWriter(out_path, schema, compression='snappy')
        writer.write_table(batch)
        print(f"  Written pairs {start}–{end-1}")

    if writer:
        writer.close()
    print(f"Done. Saved to '{out_path}'")


def get_pair_result(res, i, j):
    """
    Safe lookup for a specific grain pair.
    """
    if i == j:
        raise ValueError("i and j must be different")
    if i > j:
        i, j = j, i

    k = res["pair_lookup"][(i, j)]
    return {
        "i": i,
        "j": j,
        "theta_rad": res["theta_rad"][k],
        "theta_deg": res["theta_deg"][k],
        "polar_ax": res["polar_ax"][k],
        "azim_ax": res["azim_ax"][k],
        "qmin": res["qmin"][k],
        "row": k,
    }


# ── 6. getLineNum (mirrors C++) ───────────────────────────────────────────────
def get_line_num(grain_i, grain_j):
    if grain_i > grain_j:
        return grain_j + (grain_i - 1) * grain_i // 2
    else:
        return grain_i + (grain_j - 1) * grain_j // 2


# ── 7. Entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    '''
    python miso_txt_check.py euler_angles.txt [max_grains] [chunk_size]
    '''
    args = get_args()
    # filepath   = sys.argv[1] if len(sys.argv) > 1 else "euler_256.txt"
    # max_grain  = int(sys.argv[2]) if len(sys.argv) > 2 else None
    # chunk_size = int(sys.argv[3]) if len(sys.argv) > 3 else 50_000
    # out_path = "misorientation.parquet"

    compute_all_misorientations_chunked(
        args.filepath,
        max_grain=args.max_grain,
        chunk_size=args.chunk_size,
        out_path=args.out_path,
    )
