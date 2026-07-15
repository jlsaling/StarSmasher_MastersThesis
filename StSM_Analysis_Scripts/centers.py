
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
