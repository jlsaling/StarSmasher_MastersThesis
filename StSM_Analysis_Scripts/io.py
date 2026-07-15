import numpy as np
import struct

def load_snap(filename, fields=None):
    """
    Load snapshot data from a file.
    
    Parameters
    ----------
    filename : str
        Path to the snapshot file
    fields : list of str, optional
        List of field names to load. If None, loads all fields.
        Available fields: 'x', 'y', 'z', 'vx', 'vy', 'vz', 'm', 'rho', 'u', 'mu', 'h', 'u_dot', 'temp'
    
    Returns
    -------
    dict
        Dictionary with field names as keys and arrays as values.
        Always includes 'time' key with the snapshot timestamp.
    
    Examples
    --------
    >>> data = load_snap('snap.txt', ['x', 'y', 'z', 'm'])
    >>> x, y, z, m = data['x'], data['y'], data['z'], data['m']
    >>> time = data['time']
    
    >>> data = load_snap('snap.txt', 'x')
    >>> x = data['x']
    >>> time = data['time']
    """
    
    # Define all available fields and their column indices
    all_fields = {
        'x': 0, 'y': 1, 'z': 2,
        'vx': 3, 'vy': 4, 'vz': 5,
        'm': 6, 'rho': 7, 'u': 8,
        'mu': 9, 'h': 10, 'u_dot': 11, 'temp': 12
    }
    
    # If no fields specified, load all
    if fields is None:
        fields = list(all_fields.keys())
    
    # Convert single field to list
    if isinstance(fields, str):
        fields = [fields]
    
    # Get column indices for requested fields
    cols = [all_fields[f] for f in fields]
    
    # Load the data
    data = np.genfromtxt(filename, usecols=cols, unpack=True)
    
    # Extract time
    time = None
    with open(filename, "r") as f:
        for line in f:
            if line.startswith("# time:"):
                # next line has the values
                next_line = next(f).strip("# \n")
                vals = next_line.split()
                time = float(vals[0])  # snapshot time
            if not line.startswith("#"):
                # first non-header line marks end of header
                break
    
    # Always return dictionary with time
    if len(fields) == 1:
        result = {fields[0]: data}
    else:
        result = dict(zip(fields, data))
    
    result['time'] = time
    return result

def load_mesa_profile(filepath):
    """
    Load a MESA stellar profile (.data) file into a dictionary.

    The MESA profile format has two sections:
      - Header (lines 1–3): global scalar values (e.g. model_number, star_age, Teff)
      - Profile (lines 5–end): per-zone tabular data (e.g. zone, mass, logT, ...)

    Returns a dict with two keys:
      'header'  -> dict mapping header column names to scalar values (str or float)
      'profile' -> dict mapping profile column names to 1-D numpy arrays

    Parameters
    ----------
    filepath : str
        Path to the MESA profile .data file.

    Example
    -------
    >>> data = load_mesa_profile("50_IAMS.data")
    >>> print(data['header']['star_age'])
    2944702.56
    >>> print(data['profile']['logT'][:5])
    [4.693 4.694 4.697 4.700 4.702]
    """
    with open(filepath, "r") as f:
        lines = f.readlines()

    # --- Header section (lines 0, 1, 2) ---
    header_names = lines[1].split()
    header_values_raw = lines[2].split()

    header = {}
    for name, val in zip(header_names, header_values_raw):
        try:
            header[name] = float(val)
        except ValueError:
            # strip surrounding quotes that MESA adds to strings
            header[name] = val.strip('"')

    # --- Profile section (lines 4, 5, 6+) ---
    # line 4 is blank; line 5 is column indices; line 6 is column names
    profile_names = lines[5].split()

    # all remaining lines are data rows (one per zone)
    data_rows = []
    for line in lines[6:]:
        stripped = line.strip()
        if stripped:
            data_rows.append([float(v) for v in stripped.split()])

    data_array = np.array(data_rows)   # shape: (n_zones, n_columns)

    profile = {
        name: data_array[:, i]
        for i, name in enumerate(profile_names)
    }

    return {"header": header, "profile": profile}


# -------------------- EXPERIMENTA BINARY READING -------------------------------


import numpy as np
import struct

# ── Unit parameters ───────────────────────────────────────────────
_MUNIT     = 1.9891e33
_RUNIT     = 6.9599e10
_GRAVCONST = 6.67390e-08
_BOLTZ     = 1.380658e-16

_T_CODE   = np.sqrt(_RUNIT**3 / (_GRAVCONST * _MUNIT))
_T_CONV   = _T_CODE / 86400
_RHO_CONV = _MUNIT / _RUNIT**3


# ── Correct Fortran record reader ──────────────────────────────────
def read_fortran_record(f):
    marker = f.read(4)
    if not marker:
        return None

    length = struct.unpack('=i', marker)[0]
    data = f.read(length)
    end_length = struct.unpack('=i', f.read(4))[0]

    if length != end_length:
        raise ValueError(f"Fortran record mismatch: {length} != {end_length}")

    return data


def load_snap_binary(filepath, fields=None):

    ALL_FIELDS = {
        'x','y','z','vx','vy','vz',
        'm','rho','u','mu','h','u_dot','temp',
        'time','ntot'
    }

    if fields is None:
        fields = ALL_FIELDS
    if isinstance(fields, str):
        fields = {fields}
    fields = set(fields)

    unknown = fields - ALL_FIELDS
    if unknown:
        raise ValueError(f"Unknown fields: {unknown}")

    with open(filepath, 'rb') as f:

        # ── HEADER ───────────────────────────────────────────────
        hdr = read_fortran_record(f)

        hdr_fmt = '=2i 5d 2i d i 3d 2i 3d'
        size = struct.calcsize(hdr_fmt)

        hdr_vals = struct.unpack(hdr_fmt, hdr[:size])

        ntot = hdr_vals[0]
        timetemp = hdr_vals[9]

        # ── PARTICLES ────────────────────────────────────────────
        part_fmt = '=17d i'
        part_size = struct.calcsize(part_fmt)

        x     = np.empty(ntot, dtype=np.float64)
        y     = np.empty(ntot, dtype=np.float64)
        z     = np.empty(ntot, dtype=np.float64)
        m     = np.empty(ntot, dtype=np.float64)
        h     = np.empty(ntot, dtype=np.float64)
        rho   = np.empty(ntot, dtype=np.float64)
        vx    = np.empty(ntot, dtype=np.float64)
        vy    = np.empty(ntot, dtype=np.float64)
        vz    = np.empty(ntot, dtype=np.float64)
        u     = np.empty(ntot, dtype=np.float64)
        u_dot = np.empty(ntot, dtype=np.float64)
        mu    = np.empty(ntot, dtype=np.float64)

        for j in range(ntot):
            rec = read_fortran_record(f)
            vals = struct.unpack('=17d i', rec)
        
            x[j] = vals[0]
            y[j] = vals[1]
            z[j] = vals[2]
            m[j] = vals[3]
            h[j] = vals[4]
            rho[j] = vals[5]
            vx[j] = vals[6]
            vy[j] = vals[7]
            vz[j] = vals[8]
            u[j] = vals[12]
            u_dot[j] = vals[13]
            mu[j] = vals[15]

        # ── derived temperature ────────────────────────────────
        temp = u * _GRAVCONST * _MUNIT / _RUNIT / (1.5 * _BOLTZ / mu)

    all_data = {
        'x': x,
        'y': y,
        'z': z,
        'vx': vx,
        'vy': vy,
        'vz': vz,
        'm': m,
        'rho': rho * _RHO_CONV,
        'u': u,
        'mu': mu,
        'h': h,
        'u_dot': u_dot,
        'temp': temp,
        'time': timetemp * _T_CONV,
        'ntot': ntot,
    }

    return {k: v for k, v in all_data.items() if k in fields}