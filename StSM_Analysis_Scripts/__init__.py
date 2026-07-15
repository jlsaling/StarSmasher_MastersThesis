"""Analysis_Scripts package.

This package exposes the main analysis modules for snapshot loading,
re-centering, physics calculations, plotting, and diagnostics.
"""

from .io import load_snap, load_mesa_profile, load_snap_binary
from .recenter import re_center_and_order, get_vel_comp, re_center_and_order_exp
from .physics import mass_quantities, energy, bound_unbound, assign_bound_state, get_mass_based_edges, bin_and_avg
from .plotting import (
    plot_radial_profile_average,
    plot_radial_profile,
    plot_particles,
    plot_particles_hist2d,
    bound_unbound_plot,
    Munb_plot,
    energy_plot,
    energy_error,
    angmom_error,
    eorb_eint_ratio_plot,
    plot_map_slice,
)
from .diagnostics import compute_nearest_neighbor_distances, fort129_warning
