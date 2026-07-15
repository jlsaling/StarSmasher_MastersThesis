import numpy as np
from scipy.spatial import KDTree

def compute_nearest_neighbor_distances(X,Y,Z):
    
    # Combine the x,y,z positions into a single array of shape (N,3)
    positions = np.vstack((X,Y,Z)).T
    
    # Create a KDTree from the positions
    tree = KDTree(positions)
    
    # Query the tree for the nearest neighbor distances
    d_nn, _ = tree.query(positions, k=2)  # k=2 because the nearest neighbor is the point itself
    d_nn = d_nn[:,1]  # We take the second column which contains the distance to the nearest neighbor
    
    return d_nn

def fort129_warning(path):

    # Read the file
    data = np.loadtxt(path)
    
    # Column 1 (index 1) is the particle index 'i'
    particle_indices = data[:, 1].astype(int)
    
    # Count unique particles
    unique_particles = np.unique(particle_indices)
    n_unique = len(unique_particles)
    n_total_warnings = len(particle_indices)
    
    print(f"Total warnings: {n_total_warnings}")
    print(f"Unique particles affected: {n_unique}")
    print(f"Average warnings per particle: {n_total_warnings/n_unique:.1f}")
    
    # Show which particles (first 20 if many)
    if n_unique <= 20:
        print(f"\nParticle indices: {unique_particles}")
    else:
        print(f"\nFirst 20 particle indices: {unique_particles[:20]}")
        print(f"Last 20 particle indices: {unique_particles[-20:]}")

    # Check if they're concentrated in a region
    if data.shape[1] >= 8:  # Has x, y, z columns
        x = data[:, 5]
        y = data[:, 6]
        z = data[:, 7]
        r = np.sqrt(x**2 + y**2 + z**2)
        
        print(f"\nRadial distribution:")
        print(f"  Min r: {r.min():.3f}")
        print(f"  Max r: {r.max():.3f}")
        print(f"  Mean r: {r.mean():.3f}")
        
        # Group by radius
        print(f"\nParticles by region:")
        print(f"  r < 5: {np.sum(r < 5)}")
        print(f"  5 < r < 10: {np.sum((r >= 5) & (r < 10))}")
        print(f"  r > 10: {np.sum(r >= 10)}")

    return unique_particles