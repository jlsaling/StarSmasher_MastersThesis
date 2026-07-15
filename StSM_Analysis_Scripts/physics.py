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
    
    e = 0.5*(v/436.5)**2 + u - G*m_enc*(1/r) # EXPECTS V IN KM / S NOW, NOT IN CODE UNITS !!!
    
    return e

'''
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
'''

#------------- Experimental new bound-unbound criterion -------------#


# ── Constants (computed once at module level) ──────────────────────────────────
# G in code units: [R_sun^3 / (M_sun * code_time^2)]
# Code time unit = 1.8845e-2 days = 1.8845e-2 * 86400 s
_CODE_TIME_S  = 1.8845e-2 * 86400          # code time unit in seconds
_G_CODE = ((cons.G)/((units.R_sun.to(units.m)**3))*(units.M_sun.to(units.kg))*((1.8845e-2*86400)**2)).value

# Velocity conversion: 1 code velocity unit = R_sun / code_time, converted to km/s
_KMS_TO_CODE  = 1.0 / 436.5

# Particles below this radius use a kinetic-only energy criterion because
# internal energy u is thermally dominated in the dense core and does not
# reflect actual escape ability.
_R_INNER_RSUN = 3.0                         # R_sun — tune if needed


def specific_energy(v_kms, u, r, m_enc):

    v_code = np.asarray(v_kms) * _KMS_TO_CODE
    return 0.5 * v_code**2 + np.asarray(u) - _G_CODE * np.asarray(m_enc) / np.asarray(r)


def specific_energy_kinetic(v_kms, r, m_enc):

    v_code = np.asarray(v_kms) * _KMS_TO_CODE
    return 0.5 * v_code**2 - _G_CODE * np.asarray(m_enc) / np.asarray(r)


# DEFINITELY GO OVER THIS ONCE MORE; IT IS REALLY IMPORTANT THAT THE RADIUS IS CHOSEN BASED ON THE MECHANICAL ENERGY CRITERION (EXCLUDING u)

'''
def find_r_inner(e, e_kin, r, m, m_enc, n_radial_bins=1000, crit_gradient=0.01, n_stable=3):
    """
    Find the safe inner radius iteratively by increasing the radius directly
    (not enclosed mass fraction, to avoid bias from variable particle masses)
    until the gradient of the unbound mass fraction stabilizes near zero for
    several consecutive steps. This marks the edge of the thermally-dominated
    core, where the kinetic-only criterion no longer changes the unbound mass
    significantly as we move outward. Returns the radius at the START of the
    stable plateau (not where the confirmation completes).

    Parameters
    ----------
    e             : np.ndarray  Full specific total energy [code units]
    e_kin         : np.ndarray  Kinetic-only specific energy [code units]
    r             : np.ndarray  Radial distance [R_sun], sorted ascending
    m             : np.ndarray  Particle masses [M_sun]
    m_enc         : np.ndarray  Enclosed mass profile [M_sun]
    n_radial_bins : int         Number of radius steps to scan (default 100)
    crit_gradient : float       |d(f_unb)/d(r)| threshold below which the
                                profile is considered flat [fractional mass
                                per R_sun] (default 0.01)
    n_stable      : int         Consecutive stable steps required before
                                accepting the plateau (default 3)

    Returns
    -------
    r_inner : float  Inner radius [R_sun], at the start of the plateau
    """
    m_tot  = m_enc[-1]
    r_grid = np.linspace(r[0], 50, n_radial_bins)
    dr     = r_grid[1] - r_grid[0]

    f_unb_prev           = None
    r_prev                = None
    r_inner_first_stable  = None
    stable_count          = 0

    for r_test in r_grid:
        inner_mask = r <= r_test
        is_unbound = np.where(inner_mask, e_kin > 0, e > 0)
        f_unb      = np.sum(m[is_unbound]) / m_tot

        if f_unb_prev is not None:
            gradient = (f_unb - f_unb_prev) / dr   # d(f_unb)/d(r)

            if abs(gradient) < crit_gradient:
                if stable_count == 0:
                    r_inner_first_stable = r_prev   # mark start of plateau
                stable_count += 1
            else:
                stable_count = 0   # reset — needs N *consecutive* stable steps

            if stable_count >= n_stable:
                print(f"Plateau detected at r = {r_test:.3f} R_sun "
                      f"({stable_count} stable steps, |grad| < {crit_gradient}) "
                      f"r_inner = {r_inner_first_stable:.3f} R_sun")
                return r_inner_first_stable

        f_unb_prev = f_unb
        r_prev     = r_test

    print(f" No stable plateau found, defaulting to r_inner = 1.5 R_sun")
    return 1.5

'''

def find_r_inner(e, e_kin, r, m, m_enc, step_size=0.01, crit_gradient=0.001, n_stable=3):
    """
    Find the safe inner radius iteratively by increasing the enclosed mass
    fraction used as the inner zone boundary, until the unbound mass fraction
    stabilizes (plateaus) for several consecutive steps. This marks the edge
    of the thermally-dominated core, where the kinetic-only criterion no
    longer changes the result significantly because we have moved past the
    region where internal energy dominates.

    Parameters
    ----------
    e            : np.ndarray  Full specific total energy [code units]
    e_kin        : np.ndarray  Kinetic-only specific energy [code units]
    r            : np.ndarray  Radial distance [R_sun], sorted ascending
    m            : np.ndarray  Particle masses [M_sun]
    m_enc        : np.ndarray  Enclosed mass profile [M_sun]
    step_size    : float       Step size in enclosed mass fraction (default 1%)
    crit_change  : float       Relative change threshold below which f_unb is
                               considered stable (default 5%)
    n_stable     : int         Number of consecutive stable steps required
                               before accepting the plateau (default 3)

    Returns
    -------
    r_inner : float  Inner radius [R_sun]
    """
    m_tot = m_enc[-1]

    f_unb_prev   = None
    r_inner_prev = None
    stable_count = 0

    for m_enc_frac in np.arange(m_enc[0], 1.0 + step_size, step_size): # try starting at  m_enc[0] for the core particle problem
        inner_mask = (m_enc / m_tot) <= m_enc_frac

        if not np.any(inner_mask):
            # Threshold not yet reached by even one particle (can happen
            # with very massive individual particles) — skip this step.
            continue
        
        is_unbound = np.where(inner_mask, e_kin > 0, e > 0)
        f_unb      = np.sum(m[is_unbound]) / m_tot
        r_inner    = r[inner_mask][-1]

        if f_unb_prev is not None:
            
            #denom = max(f_unb_prev, 1e-8) # probably should never be zero, but just in case I guess
            #relative_change = abs(f_unb - f_unb_prev) / denom
            # try with gradient again...
            gradient = np.abs(f_unb - f_unb_prev) / step_size

            if gradient < crit_gradient:
                if stable_count == 0:
                    r_inner_first_stable = r_inner_prev  # mark the start of the stable plateau
                stable_count += 1
            else:
                stable_count = 0   # reset — needs N *consecutive* stable steps

            if stable_count >= n_stable:
                # r_inner_first_stable marks the start of the stable plateau
                print(f" Plateau detected at M_enc/M_tot = {m_enc_frac:.2f} "
                      f"({stable_count} stable steps) → r_inner = {r_inner_first_stable:.3f} R_sun")
                print(f"Unbound fraction f_unb = {f_unb:.2f}")
                return r_inner_first_stable

        f_unb_prev   = f_unb
        r_inner_prev = r_inner

    print(f" No plateau found, defaulting to r_inner = 1.5 R_sun")
    return 1.5

def bound_unbound(v_kms, u, r, m,m_enc):
    """
    Split particles into bound and unbound based on specific total energy.

    Hybrid criterion:
    - Calls find_r_inner(...) to find the boundary of the thermally
      dominated core
    - Inner zone (r < find_r_inner): kinetic-only energy criterion,
      ignoring u because core internal energies are thermally dominated
      and do not reflect actual escape ability.
    - Outer zone (r >= find_r_inner): full energy criterion (kinetic +
      internal + gravitational).
    """

    r     = np.asarray(r)
    m_enc = np.asarray(m_enc)

    e     = specific_energy(v_kms, u, r, m_enc)
    e_kin = specific_energy_kinetic(v_kms, r, m_enc)
    inner_r = find_r_inner(e, e_kin, r, m, m_enc)  # Find the inner radius based on the energy profiles
    
    inner_mask = r < inner_r if inner_r <= 5.0 else r < 5  # Use the found inner radius, but cap it at 5 R_sun to avoid unreasonably large inner zones, especially during the merger process when "core" is not well defined
    is_unbound = np.where(inner_mask, e_kin > 0, e > 0)

    bound   = np.where(~is_unbound)[0]
    unbound = np.where( is_unbound)[0]

    return bound, unbound

def assign_bound_state(loaded_snap, npart_1, separation_threshold=2.0):
    """
    Assigns each particle a bound state relative to star 1 and star 2's COM (max rho particle).
    
    Parameters
    ----------
    loaded_snap     : dict with keys 'x','y','z','vx','vy','vz','m','u','rho'
    npart_1         : number of particles belonging to star 1
    separation_threshold : minimum COM separation in R_sun below which stars 
                           are considered merged (default: 2.0 R_sun)
    
    Returns
    -------
    bound_state : array of str, one per particle:
                  'bound_1'   - bound only to star 1
                  'bound_2'   - bound only to star 2
                  'bound_both'- bound to both
                  'unbound'   - bound to neither
    is_merged   : bool, True if the two COMs are closer than separation_threshold
    """
    ntot = len(loaded_snap['x'])
    star1_mask = np.arange(0, npart_1)
    star2_mask = np.arange(npart_1, ntot)

    x  = loaded_snap['x'];  y  = loaded_snap['y'];  z  = loaded_snap['z']
    vx = loaded_snap['vx']; vy = loaded_snap['vy']; vz = loaded_snap['vz']
    m  = loaded_snap['m']
    u  = loaded_snap['u']

    def get_com_maxrho(mask):
        idx = mask[np.argmax(loaded_snap['rho'][mask])]
        return (x[idx], y[idx], z[idx], vx[idx], vy[idx], vz[idx])

    # --- Get COMs ---
    com1 = get_com_maxrho(star1_mask)
    com2 = get_com_maxrho(star2_mask)

    # --- Merger check ---
    sep = np.sqrt(
        (com1[0] - com2[0])**2 +
        (com1[1] - com2[1])**2 +
        (com1[2] - com2[2])**2
    )
    is_merged = sep < separation_threshold
    if is_merged:
        print(f"WARNING: COM separation ({sep:.3f} R_sun) < threshold "
              f"({separation_threshold} R_sun). Stars are likely merged.")


    # Make this but with my new bound-unbound function

    # Particle velocities are in code units, so convert to km/s for the energy calculation (which expects km/s and then converts to code units internally lol)
    def reorder_wrt_COM(cx, cy, cz, cvx, cvy, cvz):
        """Specific energy of every particle relative to a given COM."""
        nw_x  = x  - cx
        nw_y  = y  - cy
        nw_z  = z  - cz
        nw_vx = (vx - cvx) * (1 / _KMS_TO_CODE)  # convert to km/s
        nw_vy = (vy - cvy) * (1 / _KMS_TO_CODE)  # convert to km/s
        nw_vz = (vz - cvz) * (1 / _KMS_TO_CODE)  # convert to km/s

        r = np.sqrt(nw_x**2 + nw_y**2 + nw_z**2)
        v = np.sqrt(nw_vx**2 + nw_vy**2 + nw_vz**2)

        # Sort by radius to compute enclosed mass
        indx  = np.argsort(r)
        ro_r  = r[indx]
        ro_v  = v[indx]
        ro_m  = m[indx]
        ro_u  = u[indx]

        m_enc = np.cumsum(ro_m)

        return ro_v, ro_u, ro_r, ro_m, m_enc, indx


    v_kms_1, u_1, r_1, m_1, m_enc_1, indx_1 = reorder_wrt_COM(*com1)
    v_kms_2, u_2, r_2, m_2, m_enc_2, indx_2 = reorder_wrt_COM(*com2)

    bound_idx_1, unbound_idx_1 = bound_unbound(v_kms_1, u_1, r_1, m_1, m_enc_1)
    bound_idx_2, unbound_idx_2 = bound_unbound(v_kms_2, u_2, r_2, m_2, m_enc_2)

    # Map indices (in the reordered frame) back to boolean masks in the
    # ORIGINAL particle order, so masks from star 1 and star 2 align.
    bound_to_1 = np.zeros(ntot, dtype=bool)
    bound_to_1[indx_1[bound_idx_1]] = True

    bound_to_2 = np.zeros(ntot, dtype=bool)
    bound_to_2[indx_2[bound_idx_2]] = True

    # Implement comparison to mechanically unbound particles (neglecing u)
    bound_mech_idx_1, unbound_mech_idx_1 = bound_unbound(v_kms_1, u_1*0.0, r_1, m_1, m_enc_1)
    bound_mech_idx_2, unbound_mech_idx_2 = bound_unbound(v_kms_2, u_2*0.0, r_2, m_2, m_enc_2)

    bound_mech_to_1 = np.zeros(ntot, dtype=bool)
    bound_mech_to_1[indx_1[bound_mech_idx_1]] = True

    bound_mech_to_2 = np.zeros(ntot, dtype=bool)
    bound_mech_to_2[indx_2[bound_mech_idx_2]] = True


    # --- Assign labels ---
    bound_state = np.empty(ntot, dtype=object)
    bound_state[ bound_to_1 &  bound_to_2] = 'bound_both'
    bound_state[ bound_to_1 & ~bound_to_2] = 'bound_1'
    bound_state[~bound_to_1 &  bound_to_2] = 'bound_2'
    bound_state[~bound_to_1 & ~bound_to_2] = 'unbound'
    bound_state[~bound_mech_to_1 & ~bound_mech_to_2] = 'unbound_both_mech'

    

    percent_unb = np.sum(m[~bound_to_1 & ~bound_to_2])*100/np.sum(m)

    # --- Summary ---
    print(f"COM separation     :                        {sep:.4f} R_sun")
    print(f"Merged             :                        {is_merged}")
    print(f"Bound to star 1    :                        {np.sum(bound_to_1)}")
    print(f"Bound to star 2    :                        {np.sum(bound_to_2)}")
    print(f"Bound to both      :                        {np.sum(bound_to_1 & bound_to_2)}")
    print(f"Unbound            :                        {np.sum(~bound_to_1 & ~bound_to_2)}")
    print(f"Mass percentage unbound            :        {np.sum(m[~bound_to_1 & ~bound_to_2])*100/np.sum(m):.4f}")
    print(f"Mass percentage unbound (mechanical):        {np.sum(m[~bound_mech_to_1 & ~bound_mech_to_2])*100/np.sum(m):.4f}")
    print(f"Mass percentage unbound w.r.t star 1 com  : {np.sum(m[~bound_to_1])*100/np.sum(m):.4f}")
    print(f"Mass percentage unbound w.r.t star 2 com  : {np.sum(m[~bound_to_2])*100/np.sum(m):.4f}")

    return percent_unb, bound_state, is_merged

#----------------------------- End ----------------------------------#

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

