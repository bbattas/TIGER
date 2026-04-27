#!/usr/bin/env python3
"""
Generate Euler angle files for grain orientation simulations with controlled
misorientation distributions under cubic symmetry.

Usage:
    python grain_orientations.py -n 300 --dist mackenzie --sample uniform -v
    python grain_orientations.py -n 300 --dist mackenzie --sample normal --max-angle 6.2832
    python grain_orientations.py -n 300 --dist mackenzie --sample linear_inc --max-angle 3.1416
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations


# ---------------------------------------------------------------------------
# Cubic symmetry operators (24 proper rotations) as quaternions [w, x, y, z]
# ---------------------------------------------------------------------------
def cubic_symmetry_quaternions():
    """Return the 24 proper rotation quaternions for cubic (Oh) symmetry."""
    sq2 = np.sqrt(2) / 2
    ops = [
        [1, 0, 0, 0],
        [sq2,  sq2,  0,    0   ],
        [0,    1,    0,    0   ],
        [sq2, -sq2,  0,    0   ],
        [sq2,  0,    sq2,  0   ],
        [0,    0,    1,    0   ],
        [sq2,  0,   -sq2,  0   ],
        [sq2,  0,    0,    sq2 ],
        [0,    0,    0,    1   ],
        [sq2,  0,    0,   -sq2 ],
        [0.5,  0.5,  0.5,  0.5 ],
        [0.5, -0.5, -0.5, -0.5 ],
        [0.5, -0.5,  0.5,  0.5 ],
        [0.5,  0.5, -0.5, -0.5 ],
        [0.5,  0.5, -0.5,  0.5 ],
        [0.5, -0.5,  0.5, -0.5 ],
        [0.5, -0.5, -0.5,  0.5 ],
        [0.5,  0.5,  0.5, -0.5 ],
        [0,    sq2,  sq2,  0   ],
        [0,   -sq2,  sq2,  0   ],
        [0,    sq2,  0,    sq2 ],
        [0,    sq2,  0,   -sq2 ],
        [0,    0,    sq2,  sq2 ],
        [0,    0,    sq2, -sq2 ],
    ]
    return np.array(ops, dtype=np.float64)


CUBIC_OPS = cubic_symmetry_quaternions()


# ---------------------------------------------------------------------------
# Quaternion helpers
# ---------------------------------------------------------------------------
def quat_normalize(q):
    return q / np.linalg.norm(q)


def quat_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def quat_conjugate(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def euler_to_quat(phi1):
    half = phi1 / 2.0
    return np.array([np.cos(half), 0.0, 0.0, np.sin(half)])


def misorientation_angle_cubic(q1, q2):
    delta = quat_multiply(quat_conjugate(q1), q2)
    min_angle = np.inf
    for sym in CUBIC_OPS:
        q_sym = quat_multiply(sym, delta)
        for q_test in [q_sym, quat_conjugate(q_sym)]:
            w = np.clip(abs(q_test[0]), 0.0, 1.0)
            angle = 2.0 * np.degrees(np.arccos(w))
            if angle < min_angle:
                min_angle = angle
    return min_angle


# ---------------------------------------------------------------------------
# phi1 sampling strategies
# All functions take (n, max_angle) and return array of n phi1 values
# clipped/wrapped to [0, max_angle].
# ---------------------------------------------------------------------------
def sample_uniform(n, max_angle):
    """Uniform distribution on [0, max_angle]. Baseline / original behavior."""
    return np.random.uniform(0, max_angle, n)


def sample_linear_inc(n, max_angle):
    """
    Linearly increasing PDF: p(x) ~ x on [0, max_angle].
    More weight toward high phi1 values.
    Inverse CDF: x = max_angle * sqrt(u)
    """
    u = np.random.uniform(0, 1, n)
    return max_angle * np.sqrt(u)


def sample_linear_dec(n, max_angle):
    """
    Linearly decreasing PDF: p(x) ~ (max_angle - x) on [0, max_angle].
    More weight toward low phi1 values.
    Inverse CDF: x = max_angle * (1 - sqrt(1 - u))
    """
    u = np.random.uniform(0, 1, n)
    return max_angle * (1.0 - np.sqrt(1.0 - u))


def sample_normal(n, max_angle):
    """
    Normal (Gaussian) distribution centered at max_angle/2, with std =
    max_angle/6 so ~99.7% of mass falls within [0, max_angle].
    Values are clipped to [0, max_angle].
    """
    mu = max_angle / 2.0
    sigma = max_angle / 6.0
    samples = np.random.normal(mu, sigma, n)
    return np.clip(samples, 0, max_angle)


def sample_two_peak_normal(n, max_angle):
    """
    Bimodal: equal mixture of two Gaussians centered at max_angle/3
    and 2*max_angle/3, each with std = max_angle/10.
    Models a texture with two preferred orientation families.
    Values are clipped to [0, max_angle].
    """
    mu1 = max_angle / 3.0
    mu2 = 2.0 * max_angle / 3.0
    sigma = max_angle / 10.0
    n1 = n // 2
    n2 = n - n1
    s1 = np.random.normal(mu1, sigma, n1)
    s2 = np.random.normal(mu2, sigma, n2)
    samples = np.concatenate([s1, s2])
    np.random.shuffle(samples)
    return np.clip(samples, 0, max_angle)


def sample_sine(n, max_angle):
    """
    Sine-weighted PDF: p(x) ~ sin(pi * x / max_angle) on [0, max_angle].
    Peaks at max_angle/2, zero at boundaries. Smooth unimodal texture.
    Sampled via rejection sampling.
    """
    accepted = []
    while len(accepted) < n:
        batch = int((n - len(accepted)) * 3)
        x = np.random.uniform(0, max_angle, batch)
        u = np.random.uniform(0, 1, batch)
        mask = u < np.sin(np.pi * x / max_angle)
        accepted.extend(x[mask].tolist())
    return np.array(accepted[:n])


def sample_beta_low(n, max_angle):
    """
    Beta distribution (alpha=0.5, beta=2) scaled to [0, max_angle].
    Strongly right-skewed: most grains clustered near phi1=0.
    Useful for testing heavily textured near-zero orientations.
    """
    samples = np.random.beta(0.5, 2.0, n)
    return samples * max_angle


def sample_beta_high(n, max_angle):
    """
    Beta distribution (alpha=2, beta=0.5) scaled to [0, max_angle].
    Strongly left-skewed: most grains clustered near phi1=max_angle.
    Complement of beta_low.
    """
    samples = np.random.beta(2.0, 0.5, n)
    return samples * max_angle


def sample_uniform_discrete(n, max_angle, n_levels=8):
    """
    Discrete uniform: phi1 drawn from n_levels evenly spaced values in
    [0, max_angle]. Models a strongly quantized texture (e.g. rolled sheet).
    """
    levels = np.linspace(0, max_angle, n_levels, endpoint=False)
    indices = np.random.randint(0, n_levels, n)
    # Add small jitter so no two grains are exactly identical
    jitter = np.random.uniform(-max_angle / (2 * n_levels),
                                max_angle / (2 * n_levels), n)
    return np.clip(levels[indices] + jitter, 0, max_angle)


# Registry: name -> (function, description)
SAMPLE_REGISTRY = {
    "uniform":          (sample_uniform,
                         "Uniform on [0, max_angle] — original baseline"),
    "linear_inc":       (sample_linear_inc,
                         "Linearly increasing PDF — more weight at high phi1"),
    "linear_dec":       (sample_linear_dec,
                         "Linearly decreasing PDF — more weight at low phi1"),
    "normal":           (sample_normal,
                         "Gaussian centered at max_angle/2"),
    "two_peak_normal":  (sample_two_peak_normal,
                         "Bimodal Gaussian at max_angle/3 and 2*max_angle/3"),
    "sine":             (sample_sine,
                         "Sine-weighted PDF, peaks at max_angle/2"),
    "beta_low":         (sample_beta_low,
                         "Beta(0.5,2) — skewed toward low phi1"),
    "beta_high":        (sample_beta_high,
                         "Beta(2,0.5) — skewed toward high phi1"),
    "discrete":         (sample_uniform_discrete,
                         "Discrete uniform over 8 evenly spaced levels"),
}


def sample_phi1(n, sample_type, max_angle, verbose=False):
    fn, desc = SAMPLE_REGISTRY[sample_type]
    if verbose:
        print(f"[sample] Type: '{sample_type}' — {desc}")
        print(f"[sample] max_angle = {max_angle:.6f} rad "
              f"({np.degrees(max_angle):.2f} deg)")
    return fn(n, max_angle)


# ---------------------------------------------------------------------------
# Grain generation strategies
# ---------------------------------------------------------------------------
def mackenzie_distribution(n, sample_type, max_angle, verbose=False):
    if verbose:
        print("[mackenzie] Generating orientations...")
    return sample_phi1(n, sample_type, max_angle, verbose=verbose)


def textured_distribution(n, sample_type, max_angle,
                           low_angle_fraction=0.40, low_cutoff=15.0,
                           verbose=False):
    if verbose:
        print(f"[textured] Target: ≥{low_angle_fraction*100:.0f}% of pairs "
              f"< {low_cutoff}°")

    cluster_frac = min(np.sqrt(low_angle_fraction) * 1.05, 0.70)
    n_cluster = int(n * cluster_frac)
    n_random = n - n_cluster
    n_seeds = max(2, n_cluster // 8)
    seeds = sample_phi1(n_seeds, sample_type, max_angle, verbose=False)
    spread_rad = np.radians(low_cutoff / 2.0)

    cluster_phi1 = []
    for i in range(n_cluster):
        seed = seeds[i % n_seeds]
        offset = np.random.uniform(-spread_rad, spread_rad)
        cluster_phi1.append(np.clip(seed + offset, 0, max_angle))

    random_phi1 = sample_phi1(n_random, sample_type, max_angle,
                               verbose=False).tolist()
    phi1 = np.array(cluster_phi1 + random_phi1)
    np.random.shuffle(phi1)

    if verbose:
        print(f"[textured] {n_cluster} clustered + {n_random} random grains.")
    return phi1


# ---------------------------------------------------------------------------
# Misorientation computation
# ---------------------------------------------------------------------------
def compute_all_misorientations(phi1_array, verbose=False):
    n = len(phi1_array)
    quats = np.array([euler_to_quat(p) for p in phi1_array])
    n_pairs = n * (n - 1) // 2
    angles = np.empty(n_pairs, dtype=np.float64)

    if verbose:
        print(f"Computing misorientations for {n_pairs} pairs ({n} grains)...")

    idx = 0
    report_interval = max(1, n_pairs // 20)
    for i, j in combinations(range(n), 2):
        angles[idx] = misorientation_angle_cubic(quats[i], quats[j])
        idx += 1
        if verbose and idx % report_interval == 0:
            print(f"  {idx}/{n_pairs} pairs ({100*idx/n_pairs:.0f}%)",
                  end="\r")
    if verbose:
        print()
    return angles


# ---------------------------------------------------------------------------
# Plotting — includes phi1 distribution subplot so you can see the sampling
# ---------------------------------------------------------------------------
def plot_misorientation(phi1_array, angles, dist_name, sample_type,
                        max_angle, n_grains,
                        output_prefix="misorientation"):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        f"Distribution: {dist_name.capitalize()} | "
        f"Sample: {sample_type} | "
        f"max_angle: {np.degrees(max_angle):.1f}° | "
        f"{n_grains} grains",
        fontsize=12
    )

    # --- Plot 1: phi1 sampling distribution ---
    ax0 = axes[0]
    ax0.hist(np.degrees(phi1_array),
             bins=40, color="mediumseagreen", edgecolor="black",
             linewidth=0.5)
    ax0.set_xlabel("phi1 (degrees)")
    ax0.set_ylabel("Count")
    ax0.set_title("phi1 Sampling Distribution")
    ax0.set_xlim(0, np.degrees(max_angle))

    # --- Plot 2: Misorientation histogram ---
    ax1 = axes[1]
    bins = np.arange(0, 65, 2)
    ax1.hist(angles, bins=bins, color="steelblue", edgecolor="black",
             linewidth=0.5)
    ax1.set_xlabel("Misorientation Angle (degrees)")
    ax1.set_ylabel("Frequency")
    ax1.set_title("Misorientation Angle Histogram")
    ax1.set_xlim(0, 65)
    ax1.axvline(15, color="red", linestyle="--", linewidth=1.2,
                label="15° boundary")
    ax1.legend()
    frac_low = np.mean(angles < 15.0) * 100
    ax1.text(0.97, 0.95, f"{frac_low:.1f}% < 15°",
             transform=ax1.transAxes, ha="right", va="top",
             fontsize=10, color="red",
             bbox=dict(boxstyle="round,pad=0.3", fc="white",
                       ec="red", alpha=0.7))

    # --- Plot 3: CDF ---
    ax2 = axes[2]
    sorted_angles = np.sort(angles)
    cdf = np.arange(1, len(sorted_angles) + 1) / len(sorted_angles)
    ax2.plot(sorted_angles, cdf * 100, color="steelblue", linewidth=1.5)
    ax2.axvline(15, color="red", linestyle="--", linewidth=1.2, label="15°")
    ax2.axhline(frac_low, color="red", linestyle=":", linewidth=1.0)
    ax2.set_xlabel("Misorientation Angle (degrees)")
    ax2.set_ylabel("Cumulative Fraction (%)")
    ax2.set_title("Cumulative Distribution")
    ax2.set_xlim(0, 65)
    ax2.set_ylim(0, 100)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fname = f"{output_prefix}_{dist_name}_{sample_type}.png"
    plt.savefig(fname, dpi=150)
    print(f"[plot] Saved: {fname}")
    plt.show()


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------
def write_euler_file(phi1_array, filepath):
    with open(filepath, "w") as f:
        f.write("# phi1  PHI  phi2\n")
        for phi1 in phi1_array:
            f.write(f"{phi1:.6f} 0.0 0.0\n")
    print(f"[output] Wrote {len(phi1_array)} grain orientations to '{filepath}'")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # Build sample help string from registry
    sample_help = "phi1 sampling distribution. Options:\n" + "\n".join(
        f"  {k:20s}: {v[1]}" for k, v in SAMPLE_REGISTRY.items()
    )

    parser = argparse.ArgumentParser(
        description=(
            "Generate grain Euler angle files (v1, original structure) with\n"
            "multiple phi1 sampling options and configurable angle range."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("-n", "--ngrains", type=int, default=50,
                        help="Number of grains to generate.")
    parser.add_argument("--dist", choices=["mackenzie", "textured"],
                        default="mackenzie",
                        help="Misorientation distribution strategy.")
    parser.add_argument("--sample", choices=list(SAMPLE_REGISTRY.keys()),
                        default="uniform",
                        help=sample_help)
    parser.add_argument("--max-angle", type=float, default=2*np.pi,
                        help=(
                            "Maximum phi1 value in radians. "
                            "e.g. 6.2832 = 2*pi (default), "
                            "3.1416 = pi, 1.5708 = pi/2."
                        ))
    parser.add_argument("-o", "--output", default="euler_angles.txt",
                        help="Output filename for Euler angles.")
    parser.add_argument("--low-fraction", type=float, default=0.35,
                        help="Target fraction of pairs < 15° "
                             "(textured only, must be < 0.5).")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility.")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Print verbose progress output.")
    parser.add_argument("--no-plot", action="store_true",
                        help="Skip plotting.")

    args = parser.parse_args()

    if args.ngrains < 2:
        parser.error("--ngrains must be at least 2.")
    if args.low_fraction >= 0.5 or args.low_fraction <= 0.0:
        parser.error("--low-fraction must be in (0.0, 0.5).")
    if args.max_angle <= 0 or args.max_angle > 2 * np.pi:
        parser.error("--max-angle must be in (0, 2*pi].")
    if args.seed is not None:
        np.random.seed(args.seed)
        if args.verbose:
            print(f"[seed] Set to {args.seed}.")

    if args.verbose:
        print(f"[config] Grains={args.ngrains} | dist={args.dist} | "
              f"sample={args.sample} | max_angle={args.max_angle:.4f} rad | "
              f"output={args.output}")

    if args.dist == "mackenzie":
        phi1 = mackenzie_distribution(
            args.ngrains, args.sample, args.max_angle, verbose=args.verbose)
    else:
        phi1 = textured_distribution(
            args.ngrains, args.sample, args.max_angle,
            low_angle_fraction=args.low_fraction,
            low_cutoff=15.0, verbose=args.verbose)

    write_euler_file(phi1, args.output)

    angles = compute_all_misorientations(phi1, verbose=args.verbose)

    frac_low = np.mean(angles < 15.0) * 100
    print(f"[stats] Total pairs:           {len(angles)}")
    print(f"[stats] Mean misorientation:   {np.mean(angles):.2f}°")
    print(f"[stats] Median misorientation: {np.median(angles):.2f}°")
    print(f"[stats] Fraction < 15°:        {frac_low:.1f}%")
    if args.dist == "textured" and frac_low > 50.0:
        print("[warning] >50% of pairs are <15°. "
              "Try reducing --low-fraction.")

    if not args.no_plot:
        prefix = args.output.rsplit(".", 1)[0]
        plot_misorientation(phi1, angles, args.dist, args.sample,
                            args.max_angle, args.ngrains,
                            output_prefix=prefix)


if __name__ == "__main__":
    main()
