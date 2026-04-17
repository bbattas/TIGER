import h5py
import numpy as np
from dataclasses import dataclass, field

@dataclass
class FMisoData:
    times:       np.ndarray          # (n_frames,)
    n_grains:    np.ndarray          # (n_frames,)
    pairs:       list                # list of (N_f, 2) int64 arrays
    fmiso:       list                # list of (N_f,) float64 arrays
    miso_deg:    list                # list of (N_f,) float64 arrays
    polar:       list                # list of (N_f,) float64 arrays
    azimuth:     list                # list of (N_f,) float64 arrays
    frame_keys:  list = field(default_factory=list)  # e.g. ['frame_0000', ...]

    def __len__(self):
        return len(self.times)

    def frame(self, idx: int) -> dict:
        """Return all data for a single frame as a dict."""
        return {
            "time":     self.times[idx],
            "n_grains": self.n_grains[idx],
            "pairs":    self.pairs[idx],
            "fmiso":    self.fmiso[idx],
            "miso_deg": self.miso_deg[idx],
            "polar":    self.polar[idx],
            "azimuth":  self.azimuth[idx],
        }

    def mean_fmiso(self) -> np.ndarray:
        """Per-frame mean f_miso as a uniform (n_frames,) array."""
        return np.array([f.mean() if len(f) > 0 else np.nan for f in self.fmiso])


def load_fmiso_h5(path: str) -> FMisoData:
    """
    Load all frames from an fmiso HDF5 file into a FMisoData object.

    Parameters
    ----------
    path : str
        Path to the .h5 file written by write_hdf5_frame().

    Returns
    -------
    FMisoData
    """
    times, n_grains = [], []
    all_pairs, all_fmiso, all_miso_deg, all_polar, all_azimuth = [], [], [], [], []
    frame_keys = []

    with h5py.File(path, "r") as f:
        keys = sorted(k for k in f.keys() if k.startswith("frame_"))
        for key in keys:
            grp = f[key]
            frame_keys.append(key)
            times.append(float(grp.attrs["time"]))
            n_grains.append(int(grp.attrs["n_grains"]))
            all_pairs.append(grp["pairs"][:])
            all_fmiso.append(grp["fmiso"][:])
            all_miso_deg.append(grp["misorientation_deg"][:])
            all_polar.append(grp["polar_rad"][:])
            all_azimuth.append(grp["azimuth_rad"][:])

    return FMisoData(
        times      = np.array(times),
        n_grains   = np.array(n_grains),
        pairs      = all_pairs,
        fmiso      = all_fmiso,
        miso_deg   = all_miso_deg,
        polar      = all_polar,
        azimuth    = all_azimuth,
        frame_keys = frame_keys,
    )
