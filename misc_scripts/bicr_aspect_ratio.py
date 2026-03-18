import numpy as np
import argparse
import pandas as pd
from typing import Optional, Union, Iterable
from pathlib import Path
import glob
import re


def find_csv(
    file_part: Union[str, Path],
    subdir: Optional[Union[str, Path]] = None,
    *,
    recursive: bool = False,
    strict_single: bool = True,
) -> Path:
    """
    Find a CSV by partial name (or exact .csv), in cwd or a given subdir.

    - If `file_part` ends with ".csv": search for any file ending with that (e.g. "*foo.csv")
    - Else: search for any file containing that and ending in .csv (e.g. "*foo*.csv")

    Returns:
        Path to the match.

    Raises:
        FileNotFoundError: no matches
        FileExistsError: multiple matches and strict_single=True
    """
    base = Path(subdir) if subdir else Path.cwd()
    fp = Path(str(file_part))  # normalize inputs like 12 -> "12"
    token = fp.name  # ignore any accidental path parts in file_part

    pattern = f"*{token}" if token.lower().endswith(".csv") else f"*{token}*.csv"
    it = base.rglob(pattern) if recursive else base.glob(pattern)

    matches = sorted((p for p in it if p.is_file()), key=lambda p: p.name.lower())

    if not matches:
        # raise FileNotFoundError(f"No CSV matches pattern {pattern!r} in {str(base)!r}")
        return None

    if strict_single and len(matches) != 1:
        raise FileExistsError(
            f"Expected 1 match for {pattern!r} in {str(base)!r}, found {len(matches)}:\n"
            + "\n".join(str(m) for m in matches[:25])
            + ("\n..." if len(matches) > 25 else "")
        )

    return matches[0]


def x_at_level(df, xcol="x", ycol="contour", level=0.5, which="all"):
    """
    Return x location(s) where y(x) crosses 'level' using linear interpolation.

    Parameters
    ----------
    df : pd.DataFrame
    xcol, ycol : str
        Column names for x and the contour/field value.
    level : float
        Target contour value.
    which : {"all","first","last"}
        Which crossing(s) to return.

    Returns
    -------
    float or list[float] or None
    """
    d = df[[xcol, ycol]].dropna().sort_values(xcol)
    x = d[xcol].to_numpy()
    y = d[ycol].to_numpy()

    s = y - level

    # indices i where segment [i, i+1] crosses the level
    idx = np.where(s[:-1] * s[1:] < 0)[0]

    # also include exact hits (y == level) as "crossings"
    exact = np.where(s == 0)[0]
    xs = []

    # exact hits first
    xs.extend(x[exact].tolist())

    # interpolated crossings
    for i in idx:
        x0, x1 = x[i], x[i+1]
        y0, y1 = y[i], y[i+1]
        xi = x0 + (level - y0) * (x1 - x0) / (y1 - y0)
        xs.append(float(xi))

    if not xs:
        return None

    xs = sorted(xs)
    if which == "first":
        return xs[0]
    if which == "last":
        return xs[-1]
    return xs



def find_all_csv(
    file_part: Union[str, Path],
    subdir: Optional[Union[str, Path]] = None,
    *,
    recursive: bool = False
) -> Path:
    """
    Find a CSV by partial name (or exact .csv), in cwd or a given subdir.

    - If `file_part` ends with ".csv": search for any file ending with that (e.g. "*foo.csv")
    - Else: search for any file containing that and ending in .csv (e.g. "*foo*.csv")

    Returns:
        Path to the match.

    Raises:
        FileNotFoundError: no matches
        FileExistsError: multiple matches and strict_single=True
    """
    base = Path(subdir) if subdir else Path.cwd()
    fp = Path(str(file_part))  # normalize inputs like 12 -> "12"
    token = fp.name  # ignore any accidental path parts in file_part

    pattern = f"*{token}" if token.lower().endswith(".csv") else f"*{token}*.csv"
    it = base.rglob(pattern) if recursive else base.glob(pattern)

    matches = sorted((p for p in it if p.is_file()), key=lambda p: p.name.lower())
    nums = []
    for p in matches:
        m = re.search(r'_(\d+)$', p.stem)
        if not m:
            raise ValueError(f"Could not extract trailing number from: {p}")
        nums.append(int(m.group(1)))

    if not matches:
        # raise FileNotFoundError(f"No CSV matches pattern {pattern!r} in {str(base)!r}")
        return None, None

    return nums, matches


def p_to_dist(file, xcol="x", ycol="contour", level=0.5):
    df = pd.read_csv(file)
    xp = x_at_level(df, xcol=xcol, ycol=ycol, level=level, which="all")
    if xp is not None:
        if len(xp) == 2:
            dist = abs(xp[1] - xp[0])
            return dist
    return None


def p_to_rad(file, xcol="x", ycol="contour", level=0.5):
    df = pd.read_csv(file)
    xmin = df[xcol].min()
    xp = x_at_level(df, xcol=xcol, ycol=ycol, level=level, which="first")
    if xp is not None:
        dist = abs(xp - xmin)
        return dist
    return None



def aspect_ratio(subdir=None):
    # Time
    tf = find_csv('_out.csv',subdir=subdir)
    if tf is None:
        tf = find_csv('_vpp.csv',subdir=subdir)
    times = pd.read_csv(tf)['time'].to_numpy()

    idx, xfile = find_all_csv('horizontal',subdir=subdir)
    idy, yfile = find_all_csv('vertical',subdir=subdir)
    csv_out = []
    for a,b,xf,yf in zip(idx,idy,xfile,yfile):
        if a != b:
            # raise ValueError("idx and idy for vpp not the same?")
            print(f"idx and idy mismatch: {a}, {b}")
            continue
        t = times[a]
        xd = p_to_dist(xf,xcol='x',ycol='contour',level=0.5)
        yd = p_to_dist(yf,xcol='y',ycol='contour',level=0.5)
        if (xd is not None) and (yd is not None):
            ar = xd/yd
            csv_out.append({
                "time": t,
                "x": xd,
                "y": yd,
                "aspect": ar
            })
    odf = pd.DataFrame(csv_out)
    if subdir is not None:
        out_stem = subdir.rsplit("/", 1)[-1]
    else:
        out_stem = tf.stem.removesuffix("_out.csv")
    out_name = out_stem + "_aspect.csv"
    odf.to_csv(out_name, index=False)
    print(f'Saved {out_name} from {subdir}')


def radial(subdir=None):
    # Time
    tf = find_csv('_out.csv',subdir=subdir)
    if tf is None:
        tf = find_csv('_vpp.csv',subdir=subdir)
    times = pd.read_csv(tf)['time'].to_numpy()
    idr, rfile = find_all_csv('radial',subdir=subdir)
    csv_out = []
    for a,rf in zip(idr,rfile):
        t = times[a]
        rd = p_to_rad(rf,xcol='x',ycol='contour',level=0.5)
        if (rd is not None):
            r2 = rd*rd
            csv_out.append({
                "time": t,
                "x": rd,
                "r2": r2,
                "y": 1,
                "aspect": 1
            })
    odf = pd.DataFrame(csv_out)
    if subdir is not None:
        out_stem = subdir.rsplit("/", 1)[-1]
    else:
        out_stem = tf.stem.removesuffix("_out.csv")
    out_name = out_stem + "_aspect.csv"
    odf.to_csv(out_name, index=False)
    print(f'Saved {out_name} from {subdir}')



def subdirs_with_e_files(root="."):
    """
    Return a sorted list of subdirectory names under `root`
    that contain at least one file ending in '.e'.
    """
    root = Path(root)

    subdirs = {
        path.parent.name
        for path in root.rglob("*.e")
        if path.is_file() and path.parent != root
    }

    return sorted(subdirs)

def subdir_paths_with_e_files(root="."):
    """
    Return sorted relative subdirectory paths under `root`
    that contain at least one '.e' file.
    """
    root = Path(root)

    subdirs = {
        str(path.parent.relative_to(root))
        for path in root.rglob("*.e")
        if path.is_file() and path.parent != root
    }

    return sorted(subdirs)

def subdirs_with_e_files_one_level(root="."):
    """
    Return a sorted list of immediate subdirectory names under `root`
    that contain at least one file ending in '.e'.
    """
    root = Path(root)

    matches = []
    for child in root.iterdir():
        if child.is_dir():
            if any(f.is_file() and f.suffix == ".e" for f in child.iterdir()):
                matches.append(child.name)

    return sorted(matches)






pars = argparse.ArgumentParser(
        description="Measure the aspect ratio over time from LineValueSamplers")
pars.add_argument("-s", "--subdirs",action="store_true",
        help="Search for *.e files one level down (./*/.e). If not set, only search current directory.")
pars.add_argument("-r", "--radial",action="store_true",
        help="Output based on misorientation only, with one vpp called radial.")
args = pars.parse_args()



if args.subdirs:
    # dirnames = subdir_paths_with_e_files()
    dirnames = subdirs_with_e_files_one_level()
else:
    dirnames = [None]

for dir in dirnames:
    try:
        if args.radial:
            radial(subdir=dir)
        else:
            aspect_ratio(subdir=dir)
    except Exception as e:
        print(f'An error occurred in {dir}: ')
        print(str(e))
