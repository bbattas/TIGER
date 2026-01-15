from netCDF4 import Dataset
import numpy as np
import re

def _decode_names(char_2d) -> list[str]:
    """
    Decode Exodus 'name_*_var' arrays (char[*,*]) into Python strings.
    netCDF4 usually returns dtype 'S1'.
    """
    a = np.array(char_2d)
    out = []
    for row in a:
        # row is like [b'u', b'n', ...]
        out.append(b"".join(row).decode("utf-8", "ignore").strip())
    return out

class ExodusBasics:
    """
    Minimal, scalable ExodusII reader.
    Keeps the file open and only reads what you ask for.
    """

    def __init__(self, filename: str):
        self.filename = filename
        self.ds = None
        self._center_cache = {}  # (eb, method, tol) -> (xc, yc)

    def __enter__(self):
        self.ds = Dataset(self.filename, mode="r")
        # Avoid masked arrays + auto scaling overhead (usually what you want for Exodus)
        self.ds.set_auto_maskandscale(False)
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            if self.ds is not None:
                self.ds.close()
        finally:
            self.ds = None

    def _require_open(self):
        if self.ds is None:
            raise RuntimeError("File not open. Use `with ExodusBasics(...) as exo:`")

    # ---- basics ----
    def time(self) -> np.ndarray:
        # Reads just the time vector (small, even for many steps)
        return self.ds.variables["time_whole"][:]

    def coords_xy(self) -> tuple[np.ndarray, np.ndarray]:
        # Reads just the nodal coordinates (size = num_nodes)
        x = self.ds.variables["coordx"][:]
        y = self.ds.variables["coordy"][:]
        return x, y

    # ---- connectivity ----
    def connect_varnames(self) -> list[str]:
        """
        Return sorted connectivity variable names: ['connect1', 'connect2', ...]
        """
        names = [k for k in self.ds.variables.keys() if re.fullmatch(r"connect\d+", k)]
        names.sort(key=lambda s: int(s.replace("connect", "")))
        return names

    def connectivity(self, which: int | None = None, zero_based: bool = True) -> dict[str, np.ndarray] | np.ndarray:
        """
        Read connectivity array(s).

        Args:
            which: if None -> return dict of all connect# arrays
                   if int  -> return just connect{which} as an array
            zero_based: if True, subtract 1 (Exodus node ids are typically 1-based)

        Returns:
            dict[str, np.ndarray] or np.ndarray
        """
        if which is not None:
            name = f"connect{which}"
            if name not in self.ds.variables:
                raise KeyError(f"{name} not found. Available: {self.connect_varnames()}")
            arr = self.ds.variables[name][:]
            return (arr - 1) if zero_based else arr

        out = {}
        for name in self.connect_varnames():
            arr = self.ds.variables[name][:]
            out[name] = (arr - 1) if zero_based else arr
        return out

    def connectivity_meta(self) -> dict[str, dict]:
        """
        Return connectivity shapes/dims without reading full arrays.
        """
        meta = {}
        for name in self.connect_varnames():
            v = self.ds.variables[name]
            meta[name] = {
                "shape": tuple(v.shape),
                "dtype": str(v.dtype),
                "dimensions": tuple(v.dimensions),
            }
        return meta

    # ---- variable name tables ----
    def nodal_varnames(self) -> list[str]:
        self._require_open()
        if "name_nod_var" not in self.ds.variables:
            return []
        return _decode_names(self.ds.variables["name_nod_var"][:])

    def elem_varnames(self) -> list[str]:
        self._require_open()
        if "name_elem_var" not in self.ds.variables:
            return []
        return _decode_names(self.ds.variables["name_elem_var"][:])

    def var_kind(self, name: str) -> str:
        """
        Returns: 'nodal', 'element', or raises KeyError if not found.
        """
        self._require_open()
        if name in self.nodal_varnames():
            return "nodal"
        if name in self.elem_varnames():
            return "element"
        raise KeyError(f"Variable '{name}' not found in nodal or elemental name tables.")


    # ---- read one timestep (scales well) ----
    def nodal_var_at_step(self, name: str, step: int) -> np.ndarray:
        """
        Read nodal variable `name` at timestep index `step` (0-based).
        Returns array shape (num_nodes,)
        """
        self._require_open()
        names = self.nodal_varnames()
        if name not in names:
            raise KeyError(f"'{name}' not a nodal var. Available: {names}")

        idx = names.index(name) + 1  # Exodus vars are 1-indexed in vals_nod_var#
        vname = f"vals_nod_var{idx}"
        return self.ds.variables[vname][step, :]

    def elem_var_at_step(self, name: str, step: int, eb: int = 1) -> np.ndarray:
        """
        Read element variable `name` at timestep index `step` (0-based) for element block `eb`.
        Returns array shape (num_el_in_blk{eb},)
        """
        self._require_open()
        names = self.elem_varnames()
        if name not in names:
            raise KeyError(f"'{name}' not an element var. Available: {names}")

        idx = names.index(name) + 1  # Exodus vars are 1-indexed in vals_elem_var#eb#
        vname = f"vals_elem_var{idx}eb{eb}"
        if vname not in self.ds.variables:
            # Helpful error when multiple blocks exist and you picked the wrong one
            candidates = [k for k in self.ds.variables.keys() if k.startswith(f"vals_elem_var{idx}eb")]
            raise KeyError(f"{vname} not found. Available blocks for '{name}': {candidates}")

        return self.ds.variables[vname][step, :]

    def var_at_step(self, name: str, step: int, eb: int = 1) -> np.ndarray:
        """
        Convenience: auto-detect nodal vs element.
        For element vars, reads from element block `eb`.
        """
        kind = self.var_kind(name)
        if kind == "nodal":
            return self.nodal_var_at_step(name, step)
        else:
            return self.elem_var_at_step(name, step, eb=eb)

    def element_centers_xy(
        self,
        eb: int = 1,
        method: str = "mean",
        *,
        zero_based_connect: bool = True,
        cache: bool = True,
        quantize_tol: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute element 'center' x/y for element block `eb`.

        Args:
            eb: element block index (connect{eb})
            method:
                - "min"  : lower-left (min x, min y) corner-based position
                - "mean" : average of element node coordinates
                - "bbox" : 0.5*(min+max) bounding-box center (often a great default)
            zero_based_connect: if True, treat connectivity as 0-based indices
            cache: cache results (recommended; mesh is usually static)
            quantize_tol: if provided, snap centers to a grid by:
                          center = round(center/tol)*tol
                          Useful for stable gridding on noisy floats.

        Returns:
            (xc, yc) each shape (n_elem_in_block,)
        """
        self._require_open()
        method = method.lower().strip()
        if method not in {"min", "mean", "bbox"}:
            raise ValueError("method must be one of: 'min', 'mean', 'bbox'")

        key = (eb, method, quantize_tol, zero_based_connect)
        if cache and key in self._center_cache:
            return self._center_cache[key]

        x, y = self.coords_xy()
        conn = self.connectivity(which=eb, zero_based=zero_based_connect)

        # Gather element node coords (vectorized)
        x_e = x[conn]  # (n_elem, n_per_elem)
        y_e = y[conn]

        if method == "min":
            xc = x_e.min(axis=1)
            yc = y_e.min(axis=1)
        elif method == "mean":
            xc = x_e.mean(axis=1)
            yc = y_e.mean(axis=1)
        else:  # "bbox"
            xc = 0.5 * (x_e.min(axis=1) + x_e.max(axis=1))
            yc = 0.5 * (y_e.min(axis=1) + y_e.max(axis=1))

        if quantize_tol is not None:
            tol = float(quantize_tol)
            xc = np.rint(xc / tol) * tol
            yc = np.rint(yc / tol) * tol

        if cache:
            self._center_cache[key] = (xc, yc)

        return xc, yc


    # ---- global variables ----
    def glo_varnames(self) -> list[str]:
        self._require_open()
        if "name_glo_var" not in self.ds.variables:
            return []
        return _decode_names(self.ds.variables["name_glo_var"][:])

    def glo_var_at_step(self, name: str, step: int):
        """
        Read global variable `name` at timestep index `step` (0-based).
        Returns a scalar (usually float).
        """
        self._require_open()
        names = self.glo_varnames()
        if name not in names:
            raise KeyError(f"'{name}' not a global var. Available: {names}")

        idx = names.index(name)  # 0-based column in vals_glo_var
        return self.ds.variables["vals_glo_var"][step, idx]

    def glo_var_series(self, name: str) -> np.ndarray:
        """
        Read full time series of global variable `name`.
        Returns array shape (num_timesteps,).
        """
        self._require_open()
        names = self.glo_varnames()
        if name not in names:
            raise KeyError(f"'{name}' not a global var. Available: {names}")

        idx = names.index(name)
        return self.ds.variables["vals_glo_var"][:, idx]
