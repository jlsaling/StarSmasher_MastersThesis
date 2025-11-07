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
    dict or tuple
        If multiple fields requested: dictionary with field names as keys and arrays as values
        If single field requested: just the array for that field
    
    Examples
    --------
    >>> data = load_snap('snap.txt', ['x', 'y', 'z', 'm'])
    >>> x, y, z, m = data['x'], data['y'], data['z'], data['m']
    
    >>> data = load_snap('snap.txt', ['rho', 'u'])
    >>> rho = data['rho']
    
    >>> x = load_snap('snap.txt', ['x'])  # Returns just the array
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
    
    # Handle single field case (genfromtxt returns 1D array, not tuple)
    if len(fields) == 1:
        return data
    
    # Return dictionary mapping field names to arrays
    return dict(zip(fields, data))

def re_center_and_order(trace_stars=False, npart_1=None, **kwargs):
    """
    Reordering of all quantities extracted from the ascii
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
        ## TODO: Implement flag 'trace_stars' that shows which particles originally belonged to which star, if set true then
                 need particle npart_1 number of star 1 as additional input
  
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
    >>> reordered = re_order(**load_snap('snap.txt'))
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
    
    # Initialize result dictionary with position and radius
    result = {
        'ro_x': nw_x[indx],
        'ro_y': nw_y[indx],
        'ro_z': nw_z[indx],
        'ro_r': r[indx],
        'ro_v': v[indx]
    }
    
    # Reorder all other fields
    skip_fields = {'x', 'y', 'z'}  # Already processed
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
    Computation of the specific total energy of each particle.

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
    
    e = v**2 + u - G*m_enc*(1/r)
    
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
    X_valid = X#[valid]
    Y_valid = Y#[v

    sum_y, _ = np.histogram(X_valid, bins=bin_edges, weights=Y_valid)

    #r_in = bin_edges[:-1]
    #r_out = bin_edges[1:]
    #shell_volumes = (4/3) * np.pi * (r_out**3 - r_in**3)
    #sum_y = sum_y #/ shell_volumes

    counts, _ = np.histogram(X_valid, bins=bin_edges)
    sum_y[counts == 0] = np.nan
    
    avg_y = sum_y / counts

    return bin_centers, avg_y

def plot_radial_profile_average(y_ax_quant, snapshots,bin_edges=None, xlim=None, ylim=None, trace_stars=False, 
                       trace_bound=False, mass_binning=False, bin_mass=0.5, log=False, ax=None, snapshot_names=None, 
                       figsize=None, ylabel=None):
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
        'ro_rho': r"Density [g / $\mathrm{cm}^3$]",
        'ro_u': r"Specific Internal Energy [erg / g]",
        'ro_h': r"Specific Enthalpy [erg / g]",
        'ro_temp': r"Temperature [K]",
        'ro_udot': r"Specific Internal Energy Change du/dt [erg / (g s)]"
    }
    
    # Create axes if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    # Track scatter objects for colorbar
    scatter_objects = []
    

    
    for i, snap in enumerate(snapshots):
        # Use provided name or default to enumeration
        snap_label = snapshot_names[i] if snapshot_names else f"snap_{i}"
         
        if mass_binning:

            # Then use with your existing function:
            radius_edges = get_mass_based_edges(snap['ro_r'], snap['ro_m'], bin_mass)
            bin_edges = radius_edges
            
        if trace_stars:
            # Plot star 1 and star 2 separately
            for star_id, star_name in [(1, "S1"), (2, "S2")]:
                mask = snap['ro_ids'] == star_id
                X, Y = bin_and_avg(snap['ro_r'][mask],snap[y_ax_quant][mask], bin_edges)
                
                ax.plot(X, Y, label=f"{snap_label}, {star_name}", color='#009E73', linewidth=2)
        
        elif trace_bound:
            # Plot bound and unbound particles separately
            _, M_enc = mass_quantities(snap['ro_m'])
            e = energy(snap['ro_v'], snap['ro_u'], snap['ro_r'], M_enc)
            bn, un = bound_unbound(snap['ro_r'], e)
            
            for mask, particle_type in [(bn, "bn"), (un, "un")]:
                X, Y = bin_and_avg(snap['ro_r'][mask],snap[y_ax_quant][mask], bin_edges)
                ax.plot(X,Y, label=f"{snap_label}, {particle_type}", color='#009E73', linewidth=2)
        
        else:
            X, Y = bin_and_avg(snap['ro_r'],snap[y_ax_quant], bin_edges)
            ax.plot(X, Y, label=snap_label, color='#009E73', linewidth=2)
    
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
    
    return ax
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
        'ro_rho': r"Density [g / $\mathrm{cm}^3$]",
        'ro_u': r"Specific Internal Energy [erg / g]",
        'ro_h': r"Specific Enthalpy [erg / g]",
        'ro_temp': r"Temperature [K]",
        'ro_udot': r"Specific Internal Energy Change du/dt [erg / (g s)]"
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

def get_mass_based_edges(r, m, bin_mass=1.0, max_mass=None):
    '''
    Get radius bin edges corresponding to equal enclosed mass bins
    
    Parameters:
    -----------
    r: radial positions of particles
    m: masses of particles
    bin_mass: mass increment for each bin
    max_mass: maximum enclosed mass (if None, uses total mass)
    
    Returns:
    --------
    radius_edges: radii corresponding to mass bin edges
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
    
    return radius_edges
