import numpy as np
from astropy import constants as cons
from astropy import units


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

