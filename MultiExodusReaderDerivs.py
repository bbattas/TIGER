import ExodusReader
import glob
import numpy as np

class MultiExodusReaderDerivs:
    def __init__(self,file_names):
        self.file_names = glob.glob(file_names)
        global_times = set()
        file_times = []
        exodus_readers = []
        for file_name in self.file_names:
            er = ExodusReader.ExodusReader(file_name)
            times = er.times
            # global_times.update(times[:])
            # changed to avoid unhashable MaskedConstant error
            global_times.update(np.ma.compressed(times[:]))
            exodus_readers+= [er]
            file_times+=[ [min(times),max(times)] ]
        self.dim = exodus_readers[0].dim
        global_times = list(global_times)
        global_times.sort()
        self.global_times = global_times
        self.exodus_readers = exodus_readers
        self.file_times = np.asarray(file_times)
        # self.get_grlist()
        # self.check_varlist()

    def get_data_from_file_idx(self,var_name,read_time,i,full_nodal=False):
        er = self.exodus_readers[i]
        x = er.x
        y = er.y
        z = er.z
        idx = np.where(read_time == er.times)[0][0]
        c = er.get_var_values(var_name,idx,full_nodal)
        return (x,y,z,c)

    def get_data_at_time(self,var_name,read_time,full_nodal=False):
        X = []
        Y = []
        Z = []
        C = []
        for (i,file_time) in enumerate(self.file_times):
            if ( file_time[0]<= read_time and file_time[1]>= read_time  ):
                x,y,z,c = self.get_data_from_file_idx(var_name,read_time,i,full_nodal)
                try:
                    X.append(x)
                    Y.append(y)
                    Z.append(z)
                    C.append(c)
                except:
                    # Is this actually what you want it to do?
                    X = x
                    Y = y
                    Z = z
                    C = c
            else:
                pass
        X = np.vstack(X)
        Y = np.vstack(Y)
        Z = np.vstack(Z)
        if full_nodal:
            C = np.vstack(C)
        else:
            C = np.hstack(C)
        return (X,Y,Z,C)


    def check_varlist(self, varlist, *, raise_error: bool = True) -> bool:
        """
        Verify that every entry in *varlist* exists in the Exodus file.

        Parameters
        ----------
        varlist : Iterable[str]
            Variable names the caller wants to use.
        raise_error : bool, default True
            If True, raise KeyError when a name is missing.
            If False, just return False and print a warning.

        Returns
        -------
        bool
            True  - all names are present
            False - one or more names are missing (only when raise_error=False)
        """
        er = self.exodus_readers[0]
        # Combine nodal + element variable names and remove duplicates
        available = set(er.nodal_var_names) | set(er.elem_var_names)

        # List comprehension keeps original order in case you want it
        missing = [name for name in varlist if name not in available]

        if missing:
            msg = (f"The following variables are not in this Exodus file: "
                   f"{', '.join(missing)}")
            if raise_error:
                raise KeyError(msg)
            else:
                print(f"WARNING: {msg}")
                return False
        return True


    def get_opmax_from_file_idx(self, read_time, i):
        er = self.exodus_readers[i]
        x = er.x
        y = er.y
        z = er.z
        idx = np.where(read_time == er.times)[0][0]
        c_vals = []
        for n, var in enumerate(self.varlist):
            vals = er.get_var_values(var, idx, full_nodal=True)
            # Compute the mean across each set (row)
            mean_vals = np.mean(vals, axis=1)
            c_vals.append(mean_vals)
        # Stack all variable arrays into a 2D array
        c_array = np.stack(c_vals, axis=0)  # Shape: (num_vars, num_points)
        # Find the index of the maximum value along the variables axis
        c = np.argmax(c_array, axis=0)  # Shape: (num_points,)
        return (x, y, z, c)

    def get_opmax_from_file_idx_alt(self, read_time, i):
        er = self.exodus_readers[i]
        x = er.x
        y = er.y
        z = er.z
        idx = np.where(read_time == er.times)[0][0]
        c_vals = []
        for n, var in enumerate(self.varlist):
            vals = er.get_var_values(var, idx, full_nodal=True)
            # Compute the mean across each set (row)
            mean_vals = np.mean(vals, axis=1)
            c_vals.append(mean_vals)
        # Stack all variable arrays into a 2D array
        c_array = np.stack(c_vals, axis=0)  # Shape: (num_vars, num_points)
        # Find the index of the maximum value along the variables axis
        c = np.argmax(c_array, axis=0)  # Shape: (num_points,)
        # Calc ctr coordinate as avg
        mean_x = np.mean(x, axis=1)
        mean_y = np.mean(y, axis=1)
        mean_z = np.mean(z, axis=1)
        return (mean_x, mean_y, mean_z, c)


    def get_opmax_at_time(self,read_time,full_nodal=False):
        X = []
        Y = []
        Z = []
        C = []
        for (i,file_time) in enumerate(self.file_times):
            if ( file_time[0]<= read_time and file_time[1]>= read_time  ):
                x,y,z,c = self.get_opmax_from_file_idx(read_time,i)
                try:
                    X.append(x)
                    Y.append(y)
                    Z.append(z)
                    C.append(c)
                except:
                    X = x
                    Y = y
                    Z = z
                    C = c
            else:
                pass
        X = np.vstack(X)
        Y = np.vstack(Y)
        Z = np.vstack(Z)
        if full_nodal:
            C = np.vstack(C)
        else:
            C = np.hstack(C)
        return (X,Y,Z,C)

    def get_grlist(self):
        er = self.exodus_readers[0]
        gr_list = [name for name in er.nodal_var_names if name.startswith('gr')]
        varlist = ['phi'] + gr_list
        self.varlist = varlist
        return self.varlist

    def get_opmax_at_time_alt(self, read_time, full_nodal=False):
        X = []
        Y = []
        Z = []
        C = []
        for i, file_time in enumerate(self.file_times):
            if file_time[0] <= read_time <= file_time[1]:
                mean_x, mean_y, mean_z, c = self.get_opmax_from_file_idx_alt(read_time, i)
                X.append(mean_x)
                Y.append(mean_y)
                Z.append(mean_z)
                C.append(c)
        # Concatenate the mean coordinates and c arrays
        X = np.concatenate(X)
        Y = np.concatenate(Y)
        Z = np.concatenate(Z)
        C = np.concatenate(C)
        return (X, Y, Z, C)


    # New functions for multiple c

    def get_vars_from_file_idx(self, varlist, read_time: float, i: int, fullxy=True):
        """
        Return mean-per-element coordinates and *all* requested variable
        values for the *i*-th Exodus file at the snapshot `read_time`.

        Returns
        -------
        mean_x, mean_y, mean_z : 1-D np.ndarray  (n_points,)
        vals                   : 2-D np.ndarray  (n_vars, n_points)
                                 where row k corresponds to self.varlist[k]
        """
        er  = self.exodus_readers[i]            # typed: ExodusReader
        idx = int(np.where(read_time == er.times)[0][0])

        # Coordinates – average the element-node coordinates so that we
        # have one (x,y,z) per element rather than per node
        mean_x = np.mean(er.x, axis=1)
        mean_y = np.mean(er.y, axis=1)
        mean_z = np.mean(er.z, axis=1)

        # Collect every variable’s mean value (same reduction you used)
        vals = []
        for var in varlist:
            raw = er.get_var_values(var, idx, full_nodal=True)
            if raw.ndim == 2:                  # nodal → average nodes
                vals.append(raw.mean(axis=1))  # shape (n_elem,)
            else:                              # element → already shape (n_elem,)
                vals.append(raw)
        vals = np.stack(vals, axis=0)

        if fullxy:
            return mean_x, mean_y, mean_z, vals, er.x, er.y, er.z
        else:
            return mean_x, mean_y, mean_z, vals

    def get_vars_at_time(self, varlist, read_time: float, fullxy=True):
        """
        Query every Exodus fragment that spans `read_time` and concatenate
        the results.

        Returns
        -------
        X, Y, Z : 1-D np.ndarray                  (total_points,)
        V       : 2-D np.ndarray                  (n_vars, total_points)
        """
        X, Y, Z  = [], [], []
        V_blocks = []           # list of (n_vars, n_local_points) arrays
        fx, fy, fz = [], [], []

        for i, window in enumerate(self.file_times):
            if window[0] <= read_time <= window[1]:
                if fullxy:
                    mx, my, mz, vals, xx, yy, zz = self.get_vars_from_file_idx(varlist, read_time, i, fullxy)
                    fx.append(xx)
                    fy.append(yy)
                    fz.append(zz)
                else:
                    mx, my, mz, vals = self.get_vars_from_file_idx(varlist, read_time, i, fullxy)
                X.append(mx)
                Y.append(my)
                Z.append(mz)
                V_blocks.append(vals)

        # Concatenate along the *point* axis
        X = np.concatenate(X)
        Y = np.concatenate(Y)
        Z = np.concatenate(Z)
        V = np.concatenate(V_blocks, axis=1)     # keep variable axis intact
        if fullxy:
            fx = np.concatenate(fx)
            fy = np.concatenate(fy)
            fz = np.concatenate(fz)
            return X, Y, Z, V, fx, fy, fz
        else:
            return X, Y, Z, V
