# Analysis_Scripts

This package contains helper modules for working with stellar collision analysis data.

## Overview

The package is organized into these modules:

- `io.py` - snapshot and MESA profile loading utilities
- `recenter.py` - recentering, ordering, and velocity component calculations
- `physics.py` - mass, energy, binding, and binning helpers
- `plotting.py` - plotting utilities for radial profiles, particle maps, and diagnostic plots
- `diagnostics.py` - simple diagnostics and warning analysis support

## Using the package

This folder is now a Python package because it contains `__init__.py`.

### Option 1: Notebook in the parent folder
If your notebook is one level above `Analysis_Scripts`, import modules like this:

```python
from Analysis_Scripts.io import load_snap, load_mesa_profile
from Analysis_Scripts.recenter import re_center_and_order, get_vel_comp
from Analysis_Scripts.plotting import plot_radial_profile
from Analysis_Scripts.physics import mass_quantities, energy, bound_unbound
from Analysis_Scripts.diagnostics import compute_nearest_neighbor_distances
```

### Option 2: Notebook inside `Analysis_Scripts`
If your notebook sits inside the `Analysis_Scripts` folder, use direct imports:

```python
from io import load_snap
from recenter import re_center_and_order
from plotting import plot_radial_profile
```

### Option 3: Notebook somewhere else
If the notebook is not in the project folder, add the parent directory to `sys.path` first:

```python
import sys
sys.path.append(r'c:\Users\Julian\Documents\Uni\MA_StellarCollisions')
from Analysis_Scripts import plotting, io, recenter
```

## Example workflow

```python
from Analysis_Scripts.io import load_snap
from Analysis_Scripts.recenter import re_center_and_order
from Analysis_Scripts.plotting import plot_radial_profile

# Load a snapshot
data = load_snap('snap.txt', ['x', 'y', 'z', 'vx', 'vy', 'vz', 'm', 'u', 'rho'])

# Recenter and order the snapshot
snap = re_center_and_order(**data)

# Plot a radial profile
plot_radial_profile('ro_rho', [snap])
```

## Notes

- `__init__.py` exposes the core helpers for convenient package import.
- Each module imports only the dependencies it needs.
- If you use the code from another location, ensure the parent folder is on Python's import path.

## Module details

### `io.py`
- `load_snap(filename, fields=None)`
- `load_mesa_profile(filepath)`

### `recenter.py`
- `re_center_and_order(trace_stars=False, npart_1=None, **kwargs)`
- `get_vel_comp(snap)`

### `physics.py`
- `mass_quantities(m)`
- `energy(v, u, r, m_enc)`
- `bound_unbound(r, e)`
- `get_mass_based_edges(r, m, bin_mass=1.0, max_mass=None, return_mass_edges=False)`
- `bin_and_avg(X, Y, bin_edges)`

### `plotting.py`
- `plot_radial_profile_average(...)`
- `plot_radial_profile(...)`
- `plot_particles(...)`
- `plot_particles_hist2d(...)`
- `plot_map_slice(...)`
- `bound_unbound_plot(...)`
- `Munb_plot(...)`
- `energy_plot(...)`
- `energy_error(...)`
- `angmom_error(...)`
- `eorb_eint_ratio_plot(...)`

### `diagnostics.py`
- `compute_nearest_neighbor_distances(X, Y, Z)`
- `fort129_warning(path)`
