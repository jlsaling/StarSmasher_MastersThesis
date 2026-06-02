import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler
import matplotlib.colors as mcolors
from scipy.stats import binned_statistic_2d
from physics import mass_quantities, energy, bound_unbound, get_mass_based_edges


def plot_radial_profile_average(y_ax_quant, snapshots, bin_edges=None, xlim=None, ylim=None, trace_stars=False, 
                       trace_bound=False, mass_binning=False, bin_mass=0.5, log=False,xlog=False, ax=None, snapshot_names=None, 
                       figsize=None, ylabel=None, show_time=True, custom_colors=None):
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
        'ro_h': r"Smoothing Length h [$\mathrm{R}_\odot$]",
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
    
    if custom_colors:
        my_cycler = (cycler(color=custom_colors))
        plt.rc('axes', prop_cycle=my_cycler)

    for i, snap in enumerate(snapshots):
        # Use provided name or default to enumeration
        if show_time and 'time' in snap:
            time_str = f"t={snap['time']:.2f} d"
            if snapshot_names:
                snap_label = f"{snapshot_names[i]} ({time_str})"
            else:
                snap_label = f"snap_{i} ({time_str})"
        else:
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
    if xlog:
        ax.set_xscale("log")
    
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

def plot_radial_profile(y_ax_quant, snapshots, xlim=None, ylim=None, trace_stars=False, 
                       trace_bound=False, log=False, xlog=False ,ax=None, snapshot_names=None, 
                       figsize=None, ylabel=None, color_by_mass=False, cmap='viridis', 
                       show_colorbar=True, show_time=True):
    '''
    y_ax_quant: The quantity to be plotted as radial profile, e.g. 'ro_rho' for density
    snapshots: array of re-centered and re-ordered snapshot data, e.g. [d_0479, d_0480, d_0481, ...]
    xlim: (upper,lower) 
    ylim: (upper,lower) 
    trace_stars: flag, if true then split the data into particles of star 1 and of star 2
    trace_bound: flag, if true then split data into bound and unbound particles
    log: flag, if true use log scale for y-axis
    xlog: flag, if true use log scale for x-axis
    ax: matplotlib axes object, if None creates new figure
    snapshot_names: list of names for each snapshot (optional), e.g. ['d_0479', 'd_0480', 'd_0481']
    figsize: tuple (width, height) in inches, e.g. (10, 6)
    ylabel: custom y-axis label (optional, overrides default labels)
    color_by_mass: if True, color scatter points by particle mass
    cmap: colormap name for mass coloring (default: 'viridis')
    show_colorbar: if True and color_by_mass is True, show colorbar
    show_time: if True, display simulation time in legend labels (default: True)
    
    Returns:
        ax: matplotlib axes object
    '''
    
    # Dictionary for default y-axis labels
    default_labels = {
        'ro_rho': r"Density $\rho$ [g / $\mathrm{cm}^3$]",
        'ro_u': r"Specific Internal Energy $u$ [erg / g]",
        'ro_h': r"Smoothing Length h [$\mathrm{R}_\odot$]",
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
        # Build label with optional time
        if show_time and 'time' in snap:
            time_str = f"t={snap['time']:.2f} d"
            if snapshot_names:
                snap_label = f"{snapshot_names[i]} ({time_str})"
            else:
                snap_label = f"snap_{i} ({time_str})"
        else:
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
    if xlog:
        ax.set_xscale("log")
    
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

def bound_unbound_plot(snapshots, snapshot_names=None, ax=None,xlim=None, ylim=None, figsize=None, onlybound=False, no_u=False):
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    for i, snap in enumerate(snapshots):
        # Use provided name or default to enumeration
        snap_label = snapshot_names[i] if snapshot_names else f"snap_{i}"
        
        Mt, M_enc = mass_quantities(snap['ro_m'])
        
        if no_u:
            e = energy(snap['ro_v'], snap['ro_u']*0.0, snap['ro_r'], M_enc) # if no_u is set, only consider mechanically unbound

        else:
            e = energy(snap['ro_v'], snap['ro_u'], snap['ro_r'], M_enc)

        bn, un = bound_unbound(snap['ro_r'], e)
        outer_bound_mask = (np.cumsum(snap['ro_m'][bn])/np.sum(snap['ro_m'][bn])) >= 0.99 
        r_encl = (snap['ro_r'][bn][outer_bound_mask])[0]

        if onlybound: # Only plots the enclosed bound mass, useful for comparing many snapshots
            ax.plot(snap['ro_r'][bn],np.cumsum(snap['ro_m'][bn]), label=f"{snap_label}, bn", linewidth=2, linestyle="-.",alpha=0.8)
            percentage_bn = np.sum(snap['ro_m'][bn])*100/Mt
            percentage_un = np.sum(snap['ro_m'][un])*100/Mt
            print(f'Percentage bn for {snap_label}: {percentage_bn:.4f}')
            print(f'Percentage un for {snap_label}: {percentage_un:.4f}')
            print(f"Radius enclosing 99% of bound mass for {snap_label}: {r_encl:.2f} Rsun")

        else:
            for mask, particle_type in [(bn, "bn"), (un, "un")]:
                ax.plot(snap['ro_r'][mask],np.cumsum(snap['ro_m'][mask]), label=f"{snap_label}, {particle_type}", linewidth=2, linestyle="-.", alpha=0.8)
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

def energy_plot(filepaths, labels, option,ax=None, figsize=None, no_norm=False, log=True):

    '''
    filepaths: array with strings / filepaths
    labels: array with strings
    option: valid are "E_tot", "E_pot", "E_kin", "E_int"
    '''
    unit_time = 1.8445e-02
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    for i,filename in enumerate(filepaths):
        print(filename)
        label = labels[i]
        t, E_tot, E_pot, E_kin, E_int, L = np.genfromtxt(filename,dtype="float",usecols=(0,4,1,2,3,6),unpack=True)
        
        if no_norm:
            ax.set_ylabel(r'$E$')
            if option=="E_tot":
                ax.plot(t*unit_time,E_tot,label=r'$E_\mathrm{tot}$, '+label)
            elif option=="E_pot":
                ax.plot(t*unit_time,E_pot,label=r'$E_\mathrm{pot}$, '+label)
            elif option=="E_kin":
                ax.plot(t*unit_time,E_kin,label=r'$E_\mathrm{kin}$, '+label)
            elif option=="E_int":
                ax.plot(t*unit_time,E_int,label=r'$E_\mathrm{int}$, '+label)
            elif option=="L":
                ax.plot(t*unit_time,L,label=r'$L$, '+label)
                ax.set_ylabel(r'$L$')
            else:
                print("Wrong option!")
                
        else:
            ax.set_ylabel(r'$E/E_0$')
            if option=="E_tot":
                ax.plot(t*unit_time,E_tot/E_tot[0],label=r'$E_\mathrm{tot}$, '+label)
            elif option=="E_pot":
                ax.plot(t*unit_time,E_pot/E_pot[0],label=r'$E_\mathrm{pot}$, '+label)
            elif option=="E_kin":
                ax.plot(t*unit_time,E_kin/E_kin[0],label=r'$E_\mathrm{kin}$, '+label)
            elif option=="E_int":
                ax.plot(t*unit_time,E_int/E_int[0],label=r'$E_\mathrm{int}$, '+label)
            elif option=="L":
                ax.plot(t*unit_time,L/L[0],label=r'$L$, '+label)
                ax.set_ylabel(r'$L/L_0$')
            else:
                print("Wrong option!")

    if log:
        ax.set_yscale('log')
    ax.set_xlabel('Time (days)')
    ax.legend()
        
    return ax

def energy_error(filepaths, labels, ax=None, figsize=None):
    '''
    filepaths: array with strings / filepaths
    labels: array with strings
    '''
    unit_time = 1.8445e-02
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    for i, filename in enumerate(filepaths):
        label = labels[i]
        t, E_tot = np.genfromtxt(filename, dtype="float", usecols=(0, 4), unpack=True)
        
        # Find first non-NaN value
        E_0_idx = np.where(~np.isnan(E_tot))[0]
        
        if len(E_0_idx) == 0:
            print(f"Warning: All values are NaN in {filename}, skipping...")
            continue
        
        E_0 = E_tot[E_0_idx[0]]
        
        ax.plot(t * unit_time, np.abs((E_tot - E_0) / E_0), label=label)
        
    ax.set_yscale('log')
    ax.set_ylabel(r'$|E-E_0|/E_0$')
    ax.set_xlabel('Time (days)')
    ax.legend()
        
    return ax
        
def angmom_error(filepaths, labels, ax=None, figsize=None):

    '''
    filepaths: array with strings / filepaths
    labels: array with strings
    '''
    unit_time = 1.8445e-02
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    for i,filename in enumerate(filepaths):
        
        label = labels[i]
        t, L = np.genfromtxt(filename,dtype="float",usecols=(0,6),unpack=True)
        
        ax.plot(t*unit_time, np.abs((L - L[0])/L[0]), label=label)
        
    ax.set_yscale('log')
    ax.set_ylabel(r'$|(L-L_0)/L_0|$')
    ax.set_xlabel('Time (days)')
    ax.legend()
        
    return ax

## Test plot of Eorb/Eint vs time

def eorb_eint_ratio_plot(filepaths, labels,ax=None, figsize=None):

    '''
    filepaths: array with strings / filepaths
    labels: array with strings
    option: valid are "E_tot", "E_pot", "E_kin", "E_int"
    '''
    unit_time = 1.8445e-02
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    for i,filename in enumerate(filepaths):
        print(filename)
        label = labels[i]
        t, E_tot, E_pot, E_kin, E_int = np.genfromtxt(filename,dtype="float",usecols=(0,4,1,2,3),unpack=True)
        E_orb = E_pot + E_kin
        ax.plot(t*unit_time,np.abs(E_int/E_orb),label=r'$E_\mathrm{int} / E_\mathrm{orb}$, '+label)
  
        
    ax.set_yscale('log')
    ax.set_ylabel(r'$E_\mathrm{int} / E_\mathrm{orb}$')
    ax.set_xlabel('Time (days)')
    ax.legend()
        
    return ax

def plot_map_slice(snap, snap_label, color_by = None ,zlim=0.5,xlim=None, ylim=None, figsize=None,
                   ax=None, 
                   cmap='viridis', 
                   show_colorbar=True,
                   clim=None,
                   unordered = False):
    
    # Create axes if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    # Track scatter objects for colorbar
    scatter_objects = []
    
    if unordered:
        
        default_labels = {
        'rho': r"Density $\rho$ [g / $\mathrm{cm}^3$]",
        'u': r"Specific Internal Energy $u$ [erg / g]",
        'h': r"Smoothing Length h [$\mathrm{R}_\odot$]",
        'm': r"Mass m [$\mathrm{M}_\odot$]",
        'temp': r"Temperature $T$ [K]",
        'udot': r"Specific Internal Energy Change $du/dt$ [erg / (g s)]",
        'v': r"Velocity $v$ [km / s]"
        }
        
        X = snap['x']
        Y = snap['y']
        Z = snap['z']
        snap['v'] = np.sqrt( (snap['vx'])**2 + (snap['vy'])**2 + (snap['vz'])**2   )*436.5 # for km/s
        
    else:
        
        default_labels = {
        'ro_rho': r"Density $\rho$ [g / $\mathrm{cm}^3$]",
        'ro_u': r"Specific Internal Energy $u$ [erg / g]",
        'ro_h': r"Smoothing Length h [$\mathrm{R}_\odot$]",
        'ro_m': r"Mass m [$\mathrm{M}_\odot$]",
        'ro_temp': r"Temperature $T$ [K]",
        'ro_udot': r"Specific Internal Energy Change $du/dt$ [erg / (g s)]",
        'ro_v': r"Velocity $v$ [km / s]",
        'v_azimuthal': r"Azimuthal Velocity $v_\theta$ [km / s]",
        'v_radial': r"Radial Velocity $v_r$ [km / s]",
        'v_vertical': r"Vertical Velocity $v_z$ [km / s]",
        'R_cylindrical': r'Cylindrical Radius [$\mathrm{R}_\odot$]'
        }
        
        X = snap['ro_x']
        Y = snap['ro_y']
        Z = snap['ro_z']
        
    orbital_plane_mask = np.abs(Z) <= zlim
    
    if color_by:
  
        sc = ax.scatter(X[orbital_plane_mask], Y[orbital_plane_mask], 
                        c=snap[color_by][orbital_plane_mask], cmap=cmap,
                        label=snap_label, s=0.5,alpha=0.8)
        scatter_objects.append(sc)
    
    else:
        
        sc = ax.scatter(X[orbital_plane_mask],Y[orbital_plane_mask], 
                        label=snap_label, s=0.5,alpha=0.8)
        scatter_objects.append(sc)
        
    
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    
    ax.set_xlabel(r'X [$\mathrm{R}_\odot$]')
    ax.set_ylabel(r'Y [$\mathrm{R}_\odot$]')
    ax.legend()
    
    # Add colorbar if requested and we have scatter objects
    if color_by and scatter_objects:
        cbar = plt.colorbar(scatter_objects[-1], ax=ax)
        cbar.set_label(default_labels[color_by])
    
    if clim and scatter_objects:
        for sc in scatter_objects:
            sc.set_clim(clim)
    
    return ax

