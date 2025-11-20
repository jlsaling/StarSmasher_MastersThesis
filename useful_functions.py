import numpy as np
import matplotlib.pyplot as plt
from astropy import constants as cons
from astropy import units

V_sun = 436.5 # km/s 

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
    dict or array
        If multiple fields requested: dictionary with field names as keys, arrays as values, and 'time' key
        If single field requested: just the array for that field (time not included)
    
    Examples
    --------
    >>> data = load_snap('snap.txt', ['x', 'y', 'z', 'm'])
    >>> x, y, z, m = data['x'], data['y'], data['z'], data['m']
    >>> time = data['time']

    TODO:
    --------
    - Return time of simulation
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
    
    # Handle single field case
    if len(fields) == 1:
        return data
    
    return dict(zip(fields, data))

def re_center_and_order(trace_stars=False, npart_1=None, **kwargs):
    """
    Takes highest density particle as center and reorders all quantities extracted from the ascii
    files, sorting from lower to higher radius.
    
    Parameters
    ----------
    **kwargs : dict
        Dictionary containing particle data. Required keys:
        - x, y, z : numpy.ndarray - Position of the particles
        Optional keys (will be reordered if present):
        - vx, vy, vz : numpy.ndarray - Velocity of the particles
        - m : numpy.ndarray - Mass of the particles
        - u : numpy.ndarray - Specific internal energy 
        - rho : numpy.ndarray - Density of the particles
        - mu, h, u_dot, temp : numpy.ndarray - Any other fields
    Returns
    -------
    dict
        Dictionary with recentered and reordered quantities (prefixed with 'ro_'):
        Always includes:
        - ro_x, ro_y, ro_z : Position reordered by radius
        - ro_r : Radius reordered
        Plus 'ro_' versions of all other input fields, and:
        - ro_v : Velocity norm (if vx, vy, vz present)
    
    Examples
    --------
    >>> data = load_snap('snap.txt', ['x', 'y', 'z', 'vx', 'vy', 'vz', 'm', 'u', 'rho'])
    >>> reordered = re_order(**data)
    >>> 
    >>> # Or load all fields and reorder everything
    >>> reordered = re_order(**load_snap('snap.ascii'))
    """
        
    max_d = np.argmax(kwargs['rho'])
    x, y, z = kwargs['x'], kwargs['y'], kwargs['z']
    vx, vy, vz = kwargs['vx'], kwargs['vy'], kwargs['vz']
   
    # find position of maximum density particle
    xcm = x[max_d]
    ycm = y[max_d]
    zcm = z[max_d]

    # find velocity of maximum density particle
    vcmx = vx[max_d]
    vcmy = vy[max_d]
    vcmz = vz[max_d]

    # re-center
    nw_x = (x-xcm)
    nw_y = (y-ycm)
    nw_z = (z-zcm)

    nw_vx = (vx-vcmx)
    nw_vy = (vy-vcmy)
    nw_vz = (vz-vcmz)

    r = np.sqrt(nw_x**2 + nw_y**2 + nw_z**2) # radius
    v = np.sqrt(nw_vx**2 + nw_vy**2 + nw_vz**2) # velocity
    
    indx = np.argsort(r) # Get sorting indices
    
    # Result dictionary with position and radius
    result = {
        'ro_x': nw_x[indx],
        'ro_y': nw_y[indx],
        'ro_z': nw_z[indx],
        'ro_vx': nw_vx[indx]*436.5,
        'ro_vy': nw_vy[indx]*436.5,
        'ro_vz': nw_vz[indx]*436.5,
        'ro_r': r[indx],
        'ro_v': v[indx]*436.5
    }
    
    # Reorder all other fields
    skip_fields = {'x', 'y', 'z', 'vx', 'vy', 'vz'}  # Already processed
    for key, value in kwargs.items():
        if key not in skip_fields:
            result[f'ro_{key}'] = value[indx]
            
    # Trace particles
    if trace_stars == True:
        ids = np.ones(len(kwargs['x']))
        ids[(npart_1+1):] = 2  # sets the id of particles that do not belong to star 1 to '2', leaves rest at 1 
        result['ro_ids'] = ids[indx]
    
    return result

def mass_quantities(m):
    """
    Computation of the total mass and enclosed mass profile.

    Parameters
    ----------
    m : numpy.ndarray
        Mass of the particles.  
  
    Returns
    -------
    mt : numpy.float64
        Total mass value.
    m_enc : numpy.ndarray
        Enclosed mass profile.
    """
    
    mt = np.sum(m)
    m_enc = np.cumsum(m)
    
    return mt, m_enc

def energy(v,u,r,m_enc):
    """
    Computation of the specific total energy of each particle. EXPECTS V IN KM / S NOW, NOT IN CODE UNITS !!!

    Parameters
    ----------
    v : numpy.ndarray
        Velocity norm of the particles reordered
        with the radial criteria.
    u : numpy.ndarray
        Specific internal energy of the particles reordered
        with the radial criteria.
    r : numpy.ndarray
        Radius of the particles reordered with the radial
        criteria.
    m_enc : numpy.ndarray
        Enclosed mass profile.
  
    Returns
    -------
    e : numpy.ndarray
        Specific total energy of the particles reordered
        with the radial criteria.
    """

    G = ((cons.G)/((units.R_sun.to(units.m)**3))*(units.M_sun.to(units.kg))*((1.8845e-2*86400)**2)).value
    
    e = (v/436.5)**2 + u - G*m_enc*(1/r) # EXPECTS V IN KM / S NOW, NOT IN CODE UNITS !!!
    
    return e

def bound_unbound(r,e):
    """
    Definition of the bound and unbound particles based
    on the energy calculation, asuming that the inner
    particles (R < 1.5 [R_sun]) are automatically bound.

    Parameters
    ----------
    r : numpy.ndarray
        Radius of the particles reordered with the radial
        criteria.
    e : numpy.ndarray
        Specific total energy of the particles reordered
        with the radial criteria.
  
    Returns
    -------
    bn : numpy.ndarray
        Index of the bound particles.
    un : numpy.ndarray
        Index of the unbound particles.
    """
    
    inner_r = np.where(r > 1.5)[0][0]
    
    un = np.where(e[inner_r:] > 0)[0] + inner_r
    
    bn_fn = np.where(e[inner_r:] <= 0)[0] + inner_r
    bn_in = np.where(r < r[inner_r])[0]
    bn = np.concatenate((bn_in,bn_fn))
    
    return bn, un

def bin_and_avg(X, Y, bin_edges):
    
    # Idea: Instead of radius based bins, I could do mass-based bins
    # So each bin = 1 enclosed solar mass and translate that into radius
    # then mass_edges = np.arange(0, maxMass + binSize, binSize)
    # find at which indices 'q' in the array m_enc this is true (first point where m_enc >= common_edges[i])
    # translate thos indices back into radius, so maybe data['ro_r'][q]
    # these are the new edges for which to bin
    
    # mass_edges = np.arange(0, maxMass + binSize, binSize)
    # 
    
    
    delta = np.diff(bin_edges)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    #valid = (~np.isnan(X)) & (~np.isnan(Y)) & (X != 0)

    sum_y, _ = np.histogram(X, bins=bin_edges, weights=Y)

    counts, _ = np.histogram(X, bins=bin_edges)
    sum_y[counts == 0] = np.nan
    
    avg_y = sum_y / counts

    return bin_centers, avg_y

def plot_radial_profile_average(y_ax_quant, snapshots, bin_edges=None, xlim=None, ylim=None, trace_stars=False, 
                       trace_bound=False, mass_binning=False, bin_mass=0.5, log=False, ax=None, snapshot_names=None, 
                       figsize=None, ylabel=None):
    '''
    y_ax_quant: The quantity to be plotted as radial profile, e.g. 'ro_rho' for density
    snapshots: array of re-centered and re-ordered snapshot data, e.g. [d_0479, d_0480, d_0481, ...]
    bin_edges: bin edges for radial binning (ignored if mass_binning=True)
    xlim: (upper,lower) 
    ylim: (upper,lower) 
    trace_stars: flag, if true then split the data into particles of star 1 and of star 2
    trace_bound: flag, if true then split data into bound and unbound particles
    mass_binning: if True, bin by enclosed mass instead of radius
    bin_mass: mass increment for each bin when mass_binning=True
    log: flag, if true use log scale for y-axis
    ax: matplotlib axes object, if None creates new figure
    snapshot_names: list of names for each snapshot (optional), e.g. ['d_0479', 'd_0480', 'd_0481']
    figsize: tuple (width, height) in inches, e.g. (10, 6)
    ylabel: custom y-axis label (optional, overrides default labels)
    
    Returns:
        ax: matplotlib axes object
    '''
    
    # Dictionary for default y-axis labels
    default_labels = {
        'ro_rho': r"Density $\rho$ [g / $\mathrm{cm}^3$]",
        'ro_u': r"Specific Internal Energy $u$ [erg / g]",
        'ro_h': r"Specific Enthalpy $h$ [erg / g]",
        'ro_temp': r"Temperature $T$ [K]",
        'ro_udot': r"Specific Internal Energy Change $du/dt$ [erg / (g s)]",
        'ro_v': r"Velocity $v$ [km / s]",
        'v_azimuthal': r"Azimuthal Velocity $v_\theta$ [km / s]",
        'v_radial': r"Radial Velocity $v_r$ [km / s]",
        'v_vertical': r"Vertical Velocity $v_z$ [km / s]",
        'R_cylindrical': r'Cylindrical Radius [$\mathrm{R}_\odot$]'
    }
    
    # Create axes if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    for i, snap in enumerate(snapshots):
        # Use provided name or default to enumeration
        snap_label = snapshot_names[i] if snapshot_names else f"snap_{i}"
         
        if mass_binning:
            radius_edges, mass_edges = get_mass_based_edges(snap['ro_r'], snap['ro_m'], bin_mass, return_mass_edges=True)
            bin_edges_to_use = radius_edges
            # Calculate mass bin centers for x-axis
            mass_bin_centers = 0.5 * (mass_edges[:-1] + mass_edges[1:])
            use_mass_xaxis = True
        else:
            bin_edges_to_use = bin_edges
            use_mass_xaxis = False
            
        if trace_stars:
            # Plot star 1 and star 2 separately
            for star_id, star_name in [(1, "S1"), (2, "S2")]:
                mask = snap['ro_ids'] == star_id
                
                if mass_binning:
                    # Need to recalculate edges for the masked data
                    radius_edges_masked, mass_edges_masked = get_mass_based_edges(
                        snap['ro_r'][mask], snap['ro_m'][mask], bin_mass, return_mass_edges=True)
                    X_radius, Y = bin_and_avg(snap['ro_r'][mask], snap[y_ax_quant][mask], radius_edges_masked)
                    mass_bin_centers_masked = 0.5 * (mass_edges_masked[:-1] + mass_edges_masked[1:])
                    X = mass_bin_centers_masked
                else:
                    X, Y = bin_and_avg(snap['ro_r'][mask], snap[y_ax_quant][mask], bin_edges_to_use)
                
                ax.plot(X, Y, label=f"{snap_label}, {star_name}", linewidth=2)
        
        elif trace_bound:
            # Plot bound and unbound particles separately
            _, M_enc = mass_quantities(snap['ro_m'])
            e = energy(snap['ro_v'], snap['ro_u'], snap['ro_r'], M_enc)
            bn, un = bound_unbound(snap['ro_r'], e)
            
            for mask, particle_type in [(bn, "bn"), (un, "un")]:
                if mass_binning:
                    radius_edges_masked, mass_edges_masked = get_mass_based_edges(
                        snap['ro_r'][mask], snap['ro_m'][mask], bin_mass, return_mass_edges=True)
                    X_radius, Y = bin_and_avg(snap['ro_r'][mask], snap[y_ax_quant][mask], radius_edges_masked)
                    mass_bin_centers_masked = 0.5 * (mass_edges_masked[:-1] + mass_edges_masked[1:])
                    X = mass_bin_centers_masked
                else:
                    X, Y = bin_and_avg(snap['ro_r'][mask], snap[y_ax_quant][mask], bin_edges_to_use)
                    
                ax.plot(X, Y, label=f"{snap_label}, {particle_type}", linewidth=2)
        
        else:
            if mass_binning:
                X_radius, Y = bin_and_avg(snap['ro_r'], snap[y_ax_quant], bin_edges_to_use)
                X = mass_bin_centers
            else:
                X, Y = bin_and_avg(snap['ro_r'], snap[y_ax_quant], bin_edges_to_use)
            
            ax.plot(X, Y, label=snap_label, linewidth=2)
    
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    if log:
        ax.set_yscale("log")
    
    # Set x-axis label based on binning method
    if mass_binning:
        ax.set_xlabel(r'Enclosed Mass [$\mathrm{M}_\odot$]')
    else:
        ax.set_xlabel(r'Radius [$\mathrm{R}_\odot$]')
    
    # Set y-axis label: custom > default > generic
    if ylabel:
        ax.set_ylabel(ylabel)
    elif y_ax_quant in default_labels:
        ax.set_ylabel(default_labels[y_ax_quant])
    else:
        ax.set_ylabel(y_ax_quant)  # Fallback to the variable name
    
    ax.legend()
    
    return ax

def get_mass_based_edges(r, m, bin_mass=1.0, max_mass=None, return_mass_edges=False):
    '''
    Get radius bin edges corresponding to equal enclosed mass bins
    
    Parameters:
    -----------
    r: radial positions of particles
    m: masses of particles
    bin_mass: mass increment for each bin
    max_mass: maximum enclosed mass (if None, uses total mass)
    return_mass_edges: if True, also return the mass edges
    
    Returns:
    --------
    radius_edges: radii corresponding to mass bin edges
    mass_edges: (optional) the mass bin edges if return_mass_edges=True
    '''
    
    # Sort by radius
    sort_idx = np.argsort(r)
    r_sorted = r[sort_idx]
    m_sorted = m[sort_idx]
    
    # Calculate enclosed mass
    m_enc = np.cumsum(m_sorted)
    
    # Define mass bin edges
    if max_mass is None:
        max_mass = m_enc[-1]
    
    mass_edges = np.arange(0, max_mass + bin_mass, bin_mass)
    
    # Find radii corresponding to each mass edge
    radius_edges = np.zeros(len(mass_edges))
    
    for i, mass_edge in enumerate(mass_edges):
        idx = np.searchsorted(m_enc, mass_edge)
        if idx >= len(r_sorted):
            radius_edges[i] = r_sorted[-1]
        else:
            radius_edges[i] = r_sorted[idx]
    
    if return_mass_edges:
        return radius_edges, mass_edges
    else:
        return radius_edges

def plot_radial_profile(y_ax_quant, snapshots, xlim=None, ylim=None, trace_stars=False, 
                       trace_bound=False, log=False, ax=None, snapshot_names=None, 
                       figsize=None, ylabel=None, color_by_mass=False, cmap='viridis', 
                       show_colorbar=True):
    '''
    y_ax_quant: The quantity to be plotted as radial profile, e.g. 'ro_rho' for density
    snapshots: array of re-centered and re-ordered snapshot data, e.g. [d_0479, d_0480, d_0481, ...]
    xlim: (upper,lower) 
    ylim: (upper,lower) 
    trace_stars: flag, if true then split the data into particles of star 1 and of star 2
    trace_bound: flag, if true then split data into bound and unbound particles
    log: flag, if true use log scale for y-axis
    ax: matplotlib axes object, if None creates new figure
    snapshot_names: list of names for each snapshot (optional), e.g. ['d_0479', 'd_0480', 'd_0481']
    figsize: tuple (width, height) in inches, e.g. (10, 6)
    ylabel: custom y-axis label (optional, overrides default labels)
    color_by_mass: if True, color scatter points by particle mass
    cmap: colormap name for mass coloring (default: 'viridis')
    show_colorbar: if True and color_by_mass is True, show colorbar
    
    Returns:
        ax: matplotlib axes object
    '''
    
    # Dictionary for default y-axis labels
    default_labels = {
        'ro_rho': r"Density $\rho$ [g / $\mathrm{cm}^3$]",
        'ro_u': r"Specific Internal Energy $u$ [erg / g]",
        'ro_h': r"Specific Enthalpy $h$ [erg / g]",
        'ro_temp': r"Temperature $T$ [K]",
        'ro_udot': r"Specific Internal Energy Change $du/dt$ [erg / (g s)]",
        'ro_v': r"Velocity $v$ [km / s]",
        'v_azimuthal': r"Azimuthal Velocity $v_\theta$ [km / s]",
        'v_radial': r"Radial Velocity $v_r$ [km / s]",
        'v_vertical': r"Vertical Velocity $v_z$ [km / s]",
        'R_cylindrical': r'Cylindrical Radius [$\mathrm{R}_\odot$]'
    }
    
    # Create axes if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    # Track scatter objects for colorbar
    scatter_objects = []
    
    for i, snap in enumerate(snapshots):
        # Use provided name or default to enumeration
        snap_label = snapshot_names[i] if snapshot_names else f"snap_{i}"
        
        if trace_stars:
            # Plot star 1 and star 2 separately
            for star_id, star_name in [(1, "S1"), (2, "S2")]:
                mask = snap['ro_ids'] == star_id
                if color_by_mass:
                    sc = ax.scatter(snap['ro_r'][mask], snap[y_ax_quant][mask], 
                                   c=snap['ro_m'][mask], cmap=cmap,
                                   label=f"{snap_label}, {star_name}", s=0.5)
                    scatter_objects.append(sc)
                else:
                    ax.scatter(snap['ro_r'][mask], snap[y_ax_quant][mask], 
                              label=f"{snap_label}, {star_name}", s=0.5)
        
        elif trace_bound:
            # Plot bound and unbound particles separately
            _, M_enc = mass_quantities(snap['ro_m'])
            e = energy(snap['ro_v'], snap['ro_u'], snap['ro_r'], M_enc)
            bn, un = bound_unbound(snap['ro_r'], e)
            
            for mask, particle_type in [(bn, "bn"), (un, "un")]:
                if color_by_mass:
                    sc = ax.scatter(snap['ro_r'][mask], snap[y_ax_quant][mask], 
                                   c=snap['ro_m'][mask], cmap=cmap,
                                   label=f"{snap_label}, {particle_type}", s=0.5)
                    scatter_objects.append(sc)
                else:
                    ax.scatter(snap['ro_r'][mask], snap[y_ax_quant][mask], 
                              label=f"{snap_label}, {particle_type}", s=0.5)
        
        else:
            # Plot all particles together
            if color_by_mass:
                sc = ax.scatter(snap['ro_r'], snap[y_ax_quant], 
                               c=snap['ro_m'], cmap=cmap,
                               label=snap_label, s=0.5)
                scatter_objects.append(sc)
            else:
                ax.scatter(snap['ro_r'], snap[y_ax_quant], label=snap_label, s=0.5)
    
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    if log:
        ax.set_yscale("log")
    
    ax.set_xlabel(r'Radius [$\mathrm{R}_\odot$]')
    
    # Set y-axis label: custom > default > generic
    if ylabel:
        ax.set_ylabel(ylabel)
    elif y_ax_quant in default_labels:
        ax.set_ylabel(default_labels[y_ax_quant])
    else:
        ax.set_ylabel(y_ax_quant)  # Fallback to the variable name
    
    ax.legend()
    
    # Add colorbar if requested and we have scatter objects
    if color_by_mass and show_colorbar and scatter_objects:
        cbar = plt.colorbar(scatter_objects[-1], ax=ax)
        cbar.set_label(r'Particle Mass [$\mathrm{M}_\odot$]')
    
    return ax

def plot_particles(snap, snap_label, trace_stars=False,xlim=None, ylim=None, figsize=None,trace_bound=False,color_by_density=False, 
                   ax=None, 
                   cmap='viridis', 
                   show_colorbar=True,
                   clim=None):
    
    '''
    Simple plotting of particles
    '''
    # Create axes if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    # Track scatter objects for colorbar
    scatter_objects = []
        
    if trace_stars:
        # Plot star 1 and star 2 separately
        for star_id, star_name in [(1, "S1"), (2, "S2")]:
            mask = snap['ro_ids'] == star_id
            if color_by_density:
                sc = ax.scatter(snap['ro_x'][mask], snap['ro_y'][mask], 
                                c=np.log10(snap['ro_rho'])[mask], cmap=cmap,
                                label=f"{snap_label}, {star_name}", s=0.5,alpha=0.8)
                scatter_objects.append(sc)
            else:
                ax.scatter(snap['ro_x'][mask], snap['ro_y'][mask], 
                            label=f"{snap_label}, {star_name}", s=0.5,alpha=0.8)
        
    elif trace_bound:
        # Plot bound and unbound particles separately
        _, M_enc = mass_quantities(snap['ro_m'])
        e = energy(snap['ro_v'], snap['ro_u'], snap['ro_r'], M_enc)
        bn, un = bound_unbound(snap['ro_r'], e)
            
        for mask, particle_type in [(bn, "bn"), (un, "un")]:
            if color_by_density:
                sc = ax.scatter(snap['ro_x'][mask], snap['ro_y'][mask], 
                                c=np.log10(snap['ro_rho'])[mask], cmap=cmap,
                                label=f"{snap_label}, {particle_type}", s=0.5, alpha=0.8)
                scatter_objects.append(sc)
            else:
                ax.scatter(snap['ro_x'][mask], snap['ro_y'][mask], 
                            label=f"{snap_label}, {particle_type}", s=0.5,alpha=0.8)
        
    else:
        # Plot all particles together
        if color_by_density:
            sc = ax.scatter(snap['ro_x'], snap['ro_y'], 
                            c=np.log10(snap['ro_rho']), cmap=cmap,
                            label=snap_label, s=0.5,alpha=0.8)
            scatter_objects.append(sc)
        else:
            ax.scatter(snap['ro_x'], snap['ro_y'], label=snap_label, s=0.5,alpha=0.8)
    
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    
    ax.set_xlabel(r'X [$\mathrm{R}_\odot$]')
    ax.set_ylabel(r'Y [$\mathrm{R}_\odot$]')
    ax.legend()
    
    # Add colorbar if requested and we have scatter objects
    if color_by_density and show_colorbar and scatter_objects:
        cbar = plt.colorbar(scatter_objects[-1], ax=ax)
        cbar.set_label(r'Density [g / $\mathrm{cm}^3$]')
    
    if color_by_density and clim and scatter_objects:
        for sc in scatter_objects:
            sc.set_clim(clim)
    
    return ax

def plot_particles_hist2d(snap, snap_label, trace_stars=False, xlim=None, ylim=None, figsize=None,
                   trace_bound=False, color_by_density=False, 
                   ax=None, 
                   cmap='viridis', 
                   show_colorbar=True,
                   clim=None,
                   bins=100,
                   use_histogram=True,
                   norm='log'):
    
    '''
    Plots particles in a 2d histogram
    
    Parameters:
    -----------
    bins: int or [int, int], number of bins for 2D histogram (default: 100)
    use_histogram: if True, use 2D histogram; if False, use scatter plot
    color_by_density: if True with histogram, shows column density (sum of densities)
    norm: 'log' or 'linear' for histogram color scaling
    '''
    import matplotlib.colors as mcolors
    from scipy.stats import binned_statistic_2d
    
    # Create axes if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    # Track image/scatter objects for colorbar
    plot_objects = []
    
    # Determine bin ranges if limits are provided
    if xlim and ylim:
        bin_range = [[xlim[0], xlim[1]], [ylim[0], ylim[1]]]
    else:
        bin_range = None
    
    # Set up normalization
    if norm == 'log':
        norm_obj = mcolors.LogNorm()
    else:
        norm_obj = None
        
    if trace_stars:
        # Plot star 1 and star 2 separately
        for star_id, star_name in [(1, "S1"), (2, "S2")]:
            mask = snap['ro_ids'] == star_id
            
            if use_histogram:
                if color_by_density:
                    # Sum densities in each bin (column density)
                    ret = binned_statistic_2d(snap['ro_x'][mask], snap['ro_y'][mask], 
                                            snap['ro_rho'][mask], 
                                            statistic='sum', bins=bins, range=bin_range)
                    
                    # Replace zeros/NaNs with small value for log plotting
                    column_density = ret.statistic.T
                    column_density[column_density == 0] = np.nan
                    
                    im = ax.imshow(column_density, 
                                 extent=[ret.x_edge[0], ret.x_edge[-1], ret.y_edge[0], ret.y_edge[-1]],
                                 origin='lower', cmap=cmap, aspect='auto', norm=norm_obj,
                                 interpolation='nearest')
                    plot_objects.append(im)
                else:
                    # Just count particles
                    h = ax.hist2d(snap['ro_x'][mask], snap['ro_y'][mask], 
                                 bins=bins, cmap=cmap, norm=norm_obj,
                                 range=bin_range, label=f"{snap_label}, {star_name}")
                    plot_objects.append(h[3])
            else:
                if color_by_density:
                    sc = ax.scatter(snap['ro_x'][mask], snap['ro_y'][mask], 
                                   c=np.log10(snap['ro_rho'][mask]), cmap=cmap,
                                   label=f"{snap_label}, {star_name}", s=0.5, alpha=0.8)
                    plot_objects.append(sc)
                else:
                    ax.scatter(snap['ro_x'][mask], snap['ro_y'][mask], 
                              label=f"{snap_label}, {star_name}", s=0.5, alpha=0.8)
        
    elif trace_bound:
        # Plot bound and unbound particles separately
        _, M_enc = mass_quantities(snap['ro_m'])
        e = energy(snap['ro_v'], snap['ro_u'], snap['ro_r'], M_enc)
        bn, un = bound_unbound(snap['ro_r'], e)
            
        for mask, particle_type in [(bn, "bn"), (un, "un")]:
            if use_histogram:
                if color_by_density:
                    ret = binned_statistic_2d(snap['ro_x'][mask], snap['ro_y'][mask], 
                                            snap['ro_rho'][mask], 
                                            statistic='sum', bins=bins, range=bin_range)
                    
                    column_density = ret.statistic.T
                    column_density[column_density == 0] = np.nan
                    
                    im = ax.imshow(column_density, 
                                 extent=[ret.x_edge[0], ret.x_edge[-1], ret.y_edge[0], ret.y_edge[-1]],
                                 origin='lower', cmap=cmap, aspect='auto', norm=norm_obj,
                                 interpolation='nearest')
                    plot_objects.append(im)
                else:
                    h = ax.hist2d(snap['ro_x'][mask], snap['ro_y'][mask], 
                                 bins=bins, cmap=cmap, norm=norm_obj,
                                 range=bin_range, label=f"{snap_label}, {particle_type}")
                    plot_objects.append(h[3])
            else:
                if color_by_density:
                    sc = ax.scatter(snap['ro_x'][mask], snap['ro_y'][mask], 
                                   c=np.log10(snap['ro_rho'][mask]), cmap=cmap,
                                   label=f"{snap_label}, {particle_type}", s=0.5, alpha=0.8)
                    plot_objects.append(sc)
                else:
                    ax.scatter(snap['ro_x'][mask], snap['ro_y'][mask], 
                              label=f"{snap_label}, {particle_type}", s=0.5, alpha=0.8)
        
    else:
        # Plot all particles together
        if use_histogram:
            if color_by_density:
                # Sum densities in each bin (column density)
                ret = binned_statistic_2d(snap['ro_x'], snap['ro_y'], 
                                        snap['ro_rho'], 
                                        statistic='sum', bins=bins, range=bin_range)
                
                column_density = ret.statistic.T
                column_density[column_density == 0] = np.nan
                
                im = ax.imshow(column_density, 
                             extent=[ret.x_edge[0], ret.x_edge[-1], ret.y_edge[0], ret.y_edge[-1]],
                             origin='lower', cmap=cmap, aspect='auto', norm=norm_obj,
                             interpolation='nearest')
                plot_objects.append(im)
            else:
                h = ax.hist2d(snap['ro_x'], snap['ro_y'], 
                             bins=bins, cmap=cmap, norm=norm_obj,
                             range=bin_range, label=snap_label)
                plot_objects.append(h[3])
        else:
            if color_by_density:
                sc = ax.scatter(snap['ro_x'], snap['ro_y'], 
                               c=np.log10(snap['ro_rho']), cmap=cmap,
                               label=snap_label, s=0.5, alpha=0.8)
                plot_objects.append(sc)
            else:
                ax.scatter(snap['ro_x'], snap['ro_y'], label=snap_label, s=0.5, alpha=0.8)
    
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    
    ax.set_xlabel(r'X [$\mathrm{R}_\odot$]')
    ax.set_ylabel(r'Y [$\mathrm{R}_\odot$]')
    
    if not use_histogram:
        ax.legend()
    
    # Add colorbar if requested and we have plot objects
    if show_colorbar and plot_objects:
        cbar = plt.colorbar(plot_objects[-1], ax=ax)
        if use_histogram and color_by_density:
            cbar.set_label(r'Column Density [g / $\mathrm{cm}^2$]')
        elif use_histogram:
            cbar.set_label('Particle Count')
        elif color_by_density:
            cbar.set_label(r'log$_{10}$(Density [g / $\mathrm{cm}^3$])')
    
    if clim and plot_objects:
        for obj in plot_objects:
            obj.set_clim(clim)
    
    return ax

def bound_unbound_plot(snapshots, snapshot_names=None, ax=None,xlim=None, ylim=None, figsize=None, onlybound=False):
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    for i, snap in enumerate(snapshots):
        # Use provided name or default to enumeration
        snap_label = snapshot_names[i] if snapshot_names else f"snap_{i}"
        
        Mt, M_enc = mass_quantities(snap['ro_m'])
        e = energy(snap['ro_v'], snap['ro_u'], snap['ro_r'], M_enc)
        bn, un = bound_unbound(snap['ro_r'], e)
        
        if onlybound: # Only plots the enclosed bound mass, useful for comparing many snapshots
            ax.plot(snap['ro_r'][bn],np.cumsum(snap['ro_m'][bn]), label=f"{snap_label}, bn", linewidth=2)
            percentage_bn = np.sum(snap['ro_m'][bn])*100/Mt
            percentage_un = np.sum(snap['ro_m'][un])*100/Mt
            print(f'Percentage bn for {snap_label}: {percentage_bn:.4f}')
            print(f'Percentage un for {snap_label}: {percentage_un:.4f}')

        else:
            for mask, particle_type in [(bn, "bn"), (un, "un")]:
                ax.plot(snap['ro_r'][mask],np.cumsum(snap['ro_m'][mask]), label=f"{snap_label}, {particle_type}", linewidth=2)
                percentage = np.sum(snap['ro_m'][mask])*100/Mt
                print(f'Percentage {particle_type} for {snap_label}: {percentage:.4f}')
            
            
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    
    ax.set_xlabel(r'Radius R [$\mathrm{R}_\odot$]')
    ax.set_ylabel(r'Enclosed Mass [$\mathrm{M}_\odot$]')
    ax.legend()
            
    return ax

def get_vel_comp(snap):
    '''
    Function that 
    A) Calculates the total angular momentum of the collision remnant
    B) Obtains the rotation axis from this by normalizing L / |L|
    C) Calculates the velocity components (radial, azimuthal, vertical)


    Example usage:  data = re_center_and_order(trace_stars=True, npart_1=99954,**load_snap(snap, fields))
                    data = get_vel_comp(data)
    '''
    
    # Calculate total angular momentum
    L_x = np.sum(snap['ro_m'] * (snap['ro_y']*snap['ro_vz'] - snap['ro_z']*snap['ro_vy']))
    L_y = np.sum(snap['ro_m'] * (snap['ro_z']*snap['ro_vx'] - snap['ro_x']*snap['ro_vz']))
    L_z = np.sum(snap['ro_m'] * (snap['ro_x']*snap['ro_vy'] - snap['ro_y']*snap['ro_vx']))
    
    L = np.array([L_x, L_y, L_z])
    
    # Normalize to get rotation axis unit vector
    L_mag = np.sqrt(L_x**2 + L_y**2 + L_z**2)
    L_unit = L / L_mag  # Direction of the rotation axis
    
    # Calculate velocity components for all particles
    
    # Stack position and velocity vectors (N x 3 arrays)
    positions = np.column_stack([snap['ro_x'], snap['ro_y'], snap['ro_z']])
    velocities = np.column_stack([snap['ro_vx'], snap['ro_vy'], snap['ro_vz']])
    
    # Component of position parallel to rotation axis (for all particles at once)
    r_parallel_mag = np.dot(positions, L_unit)  # (N,) array
    r_parallel = np.outer(r_parallel_mag, L_unit)  # (N, 3) array
    
    # Perpendicular distance vector from rotation axis
    r_perp = positions - r_parallel  # (N, 3)
    R_perp = np.linalg.norm(r_perp, axis=1)  # (N,) cylindrical radius
    
    # Radial unit vectors (avoiding division by zero)
    r_hat = np.zeros_like(r_perp)
    mask = R_perp > 1e-10  # Particles not on axis
    r_hat[mask] = r_perp[mask] / R_perp[mask, np.newaxis]
    
    # Azimuthal unit vectors (phi_hat = L_unit x r_hat)
    phi_hat = np.cross(L_unit, r_hat)
    
    # Project velocities onto the unit vectors (vectorized dot product)
    v_radial = np.sum(velocities * r_hat, axis=1)
    v_azimuthal = np.sum(velocities * phi_hat, axis=1)
    v_vertical = np.dot(velocities, L_unit)
    
    # Store results in the snap dictionary
    snap['L_total'] = L
    snap['L_unit'] = L_unit
    snap['v_radial'] = v_radial
    snap['v_azimuthal'] = v_azimuthal
    snap['v_vertical'] = v_vertical
    snap['R_cylindrical'] = R_perp
    
    return snap

def Munb_plot(filepaths, labels, ax=None, figsize=None):
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    for i,filename in enumerate(filepaths):
        
        label = labels[i]
    
        Munb_percent, time = np.genfromtxt(filename, usecols=[0,2], unpack=True)
        ax.plot(time, Munb_percent, label=f"{label}")
    
    ax.set_xlabel(r'Time [Days]')
    ax.set_ylabel("Unbound mass [%]")
    ax.legend()
    return ax
