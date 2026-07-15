import numpy as np

from StSM_Analysis_Scripts import centers


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
        - time : float - Simulation time (passed through, not reordered)
    Returns
    -------
    dict
        Dictionary with recentered and reordered quantities (prefixed with 'ro_'):
        Always includes:
        - ro_x, ro_y, ro_z : Position reordered by radius
        - ro_r : Radius reordered
        - time : Simulation time (if present in input)
        Plus 'ro_' versions of all other input fields, and:
        - ro_v : Velocity norm (if vx, vy, vz present)
    
    Examples
    --------
    >>> data = load_snap('snap.txt', ['x', 'y', 'z', 'vx', 'vy', 'vz', 'm', 'u', 'rho'])
    >>> reordered = re_center_and_order(**data)
    >>> 
    >>> # Or load all fields and reorder everything
    >>> reordered = re_center_and_order(**load_snap('snap.ascii'))
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
    
    # Reorder all other fields (skip scalars like 'time')
    skip_fields = {'x', 'y', 'z', 'vx', 'vy', 'vz', 'time'}  # Already processed or scalar
    for key, value in kwargs.items():
        if key not in skip_fields:
            result[f'ro_{key}'] = value[indx]
    
    # Pass through time unchanged (if present)
    if 'time' in kwargs:
        result['time'] = kwargs['time']
            
    # Trace particles
    if trace_stars == True:
        ids = np.ones(len(kwargs['x']))
        ids[(npart_1+1):] = 2  # sets the id of particles that do not belong to star 1 to '2', leaves rest at 1 
        result['ro_ids'] = ids[indx]
    
    return result

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

#------------------ Experimental ------------------------

def re_center_and_order_exp(trace_stars=False, npart_1=None, mode='max density', **kwargs):
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
        - time : float - Simulation time (passed through, not reordered)
    Returns
    -------
    dict
        Dictionary with recentered and reordered quantities (prefixed with 'ro_'):
        Always includes:
        - ro_x, ro_y, ro_z : Position reordered by radius
        - ro_r : Radius reordered
        - time : Simulation time (if present in input)
        Plus 'ro_' versions of all other input fields, and:
        - ro_v : Velocity norm (if vx, vy, vz present)

    Examples
    --------
    >>> data = load_snap('snap.txt', ['x', 'y', 'z', 'vx', 'vy', 'vz', 'm', 'u', 'rho'])
    >>> reordered = re_center_and_order(**data)
    >>> 
    >>> # Or load all fields and reorder everything
    >>> reordered = re_center_and_order(**load_snap('snap.ascii'))
    """

    x, y, z = kwargs['x'], kwargs['y'], kwargs['z']
    vx, vy, vz = kwargs['vx'], kwargs['vy'], kwargs['vz']
    centers = center_definitions(**kwargs)

    if mode == 'max density':

        max_d = np.argmax(kwargs['rho'])

        # find position of maximum density particle
        x_cntr = x[max_d]
        y_cntr = y[max_d]
        z_cntr = z[max_d]
        # find velocity of maximum density particle
        v_cntr_x = vx[max_d]
        v_cntr_y = vy[max_d]
        v_cntr_z = vz[max_d]

    elif mode == 'center of mass': # takes the center of mass of the 50 particles with the highest density
        x_cntr, y_cntr, z_cntr, v_cntr_x, v_cntr_y, v_cntr_z = centers['center_of_mass']
    elif mode == 'center of density': # takes the center of density of the 50 particles with the highest density
        x_cntr, y_cntr, z_cntr, v_cntr_x, v_cntr_y, v_cntr_z = centers['center_of_density']
    elif mode == 'average position': # takes the average position of the 50 particles with the highest density
        x_cntr, y_cntr, z_cntr, v_cntr_x, v_cntr_y, v_cntr_z = centers['average_position']
    else:
        raise ValueError(f"Invalid mode '{mode}'. Choose from 'max density', 'center of mass', 'center of density', or 'average position'.")
    

    # re-center
    nw_x = (x-x_cntr)
    nw_y = (y-y_cntr)
    nw_z = (z-z_cntr)
    nw_vx = (vx-v_cntr_x)
    nw_vy = (vy-v_cntr_y)
    nw_vz = (vz-v_cntr_z)

    # also re-center the centers dictionary to reflect the new origin for later comparisons
    centers['center_of_mass'] = (centers['center_of_mass'][0] - x_cntr, centers['center_of_mass'][1] - y_cntr, centers['center_of_mass'][2] - z_cntr, centers['center_of_mass'][3] - v_cntr_x, centers['center_of_mass'][4] - v_cntr_y, centers['center_of_mass'][5] - v_cntr_z)
    centers['center_of_density'] = (centers['center_of_density'][0] - x_cntr, centers['center_of_density'][1] - y_cntr, centers['center_of_density'][2] - z_cntr, centers['center_of_density'][3] - v_cntr_x, centers['center_of_density'][4] - v_cntr_y, centers['center_of_density'][5] - v_cntr_z)
    centers['average_position'] = (centers['average_position'][0] - x_cntr, centers['average_position'][1] - y_cntr, centers['average_position'][2] - z_cntr, centers['average_position'][3] - v_cntr_x, centers['average_position'][4] - v_cntr_y, centers['average_position'][5] - v_cntr_z)

    r = np.sqrt(nw_x**2 + nw_y**2 + nw_z**2) # radius
    v = np.sqrt(nw_vx**2 + nw_vy**2 + nw_vz**2) # velocity

    indx = np.argsort(r) # Get sorting indices

    # Result dictionary with position and radius ||| Comment: I am not sure if the velocity conversion should be done here or rather outside the result dictionary, idk
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
    
    result['center_of_mass'] = centers['center_of_mass']
    result['center_of_density'] = centers['center_of_density']
    result['average_position'] = centers['average_position']

    # Reorder all other fields (skip scalars like 'time')
    skip_fields = {'x', 'y', 'z', 'vx', 'vy', 'vz', 'time'}  # Already processed or scalar
    for key, value in kwargs.items():
        if key not in skip_fields:
            result[f'ro_{key}'] = value[indx]

    # Pass through time unchanged (if present)
    if 'time' in kwargs:
        result['time'] = kwargs['time']
            
    # Trace particles
    if trace_stars == True:
        ids = np.ones(len(kwargs['x']))
        ids[(npart_1+1):] = 2  # sets the id of particles that do not belong to star 1 to '2', leaves rest at 1 
        result['ro_ids'] = ids[indx]

    return result


def center_definitions(**kwargs):
    """
    Calculate different definitions of the center of a particle distribution based on the highest density particles.

    Parameters
    ----------
    kwargs : dict
        Dictionary containing particle data, including positions ('x', 'y', 'z'), velocities ('vx', 'vy', 'vz'), 
        masses ('m'), and densities ('rho').

    Returns
    -------
    dict
        A dictionary containing the calculated centers and their velocities:
        - 'center_of_mass': Center of mass  (x_cm, y_cm, z_cm, v_cm_x, v_cm_y, v_cm_z)
        - 'center_of_density': Center of density (x_cd, y_cd, z_cd, v_cd_x, v_cd_y, v_cd_z)
        - 'average_position': Average (x_avg, y_avg, z_avg, v_avg_x, v_avg_y, v_avg_z)
    """

    rho_idx = np.argmax(kwargs['rho'])

    # take the 50 particles with the highest density
    rho_sorted = np.sort(kwargs['rho'])[::-1]
    top_50_indices = np.argsort(kwargs['rho'])[::-1][:50]

    x_50, y_50, z_50 = kwargs['x'][top_50_indices], kwargs['y'][top_50_indices], kwargs['z'][top_50_indices]
    vx_50, vy_50, vz_50 = kwargs['vx'][top_50_indices], kwargs['vy'][top_50_indices], kwargs['vz'][top_50_indices]
    m_50 = kwargs['m'][top_50_indices]

    # now A) calculate the center of mass position and velocity of these 50 particles
    # B) calculate a center of density position and velocity of these 50 particles
    # C) just average the positions and velocities of these 50 particles

    # A) Center of mass position and velocity
    total_mass = np.sum(m_50)
    x_cm = np.sum(x_50 * m_50) / total_mass
    y_cm = np.sum(y_50 * m_50) / total_mass
    z_cm = np.sum(z_50 * m_50) / total_mass 

    vx_cm = np.sum(vx_50 * m_50) / total_mass
    vy_cm = np.sum(vy_50 * m_50) / total_mass
    vz_cm = np.sum(vz_50 * m_50) / total_mass

    # B) Center of density position and velocity
    x_cd = np.sum(x_50 * kwargs['rho'][top_50_indices]) / np.sum(kwargs['rho'][top_50_indices])
    y_cd = np.sum(y_50 * kwargs['rho'][top_50_indices]) / np.sum(kwargs['rho'][top_50_indices])
    z_cd = np.sum(z_50 * kwargs['rho'][top_50_indices]) / np.sum(kwargs['rho'][top_50_indices])

    vx_cd = np.sum(vx_50 * kwargs['rho'][top_50_indices]) / np.sum(kwargs['rho'][top_50_indices])
    vy_cd = np.sum(vy_50 * kwargs['rho'][top_50_indices]) / np.sum(kwargs['rho'][top_50_indices])
    vz_cd = np.sum(vz_50 * kwargs['rho'][top_50_indices]) / np.sum(kwargs['rho'][top_50_indices])   

    # C) Average position and velocity
    x_avg = np.mean(x_50)
    y_avg = np.mean(y_50)
    z_avg = np.mean(z_50)

    vx_avg = np.mean(vx_50)
    vy_avg = np.mean(vy_50)
    vz_avg = np.mean(vz_50)


    return {
        'center_of_mass': (x_cm, y_cm, z_cm, vx_cm, vy_cm, vz_cm),
        'center_of_density': (x_cd, y_cd, z_cd, vx_cd, vy_cd, vz_cd),
        'average_position': (x_avg, y_avg, z_avg, vx_avg, vy_avg, vz_avg)
    }
