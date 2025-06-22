import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from sklearn import datasets
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform

# You'll need to install these:
# pip install ripser persim
try:
    from ripser import ripser
    from persim import plot_diagrams
    RIPSER_AVAILABLE = True
except ImportError as e:
    print(f"Import error: {e}")
    print("Ripser not installed. Run: pip install ripser persim")
    RIPSER_AVAILABLE = False

# --- 1. Load and Prepare the Data ---

# Load the Iris dataset
iris = datasets.load_iris()
X = iris.data  # The 4D data
y = iris.target # The species labels (0, 1, 2)
target_names = iris.target_names

# Reduce dimensionality from 4D to 2D using PCA
pca = PCA(n_components=2)
points_2d = pca.fit_transform(X)

print("Iris Dataset Analysis with Ripser")
print("=================================")
print(f"Original data shape: {X.shape}")
print(f"Reduced data shape: {points_2d.shape}")
print(f"Species: {target_names}")

if not RIPSER_AVAILABLE:
    print("\nPlease install ripser and persim to run this analysis:")
    print("pip install ripser persim")
    exit()

# --- 2. Compute Persistent Homology with Ripser ---

def compute_persistent_homology(data, max_dimension=2, threshold=2.0):
    """
    Compute persistent homology using Ripser
    
    Parameters:
    - data: point cloud data
    - max_dimension: maximum homology dimension to compute
    - threshold: maximum filtration value
    """
    print(f"\nComputing persistent homology...")
    print(f"Max dimension: H_{max_dimension}")
    print(f"Max filtration value: {threshold}")
    
    # Ripser computation
    result = ripser(
        data, 
        maxdim=max_dimension,  # Compute H_0, H_1, ..., H_max_dimension
        thresh=threshold,      # Maximum radius to consider
        metric='euclidean'     # Distance metric
    )
    
    return result

# Compute persistent homology
ph_result = compute_persistent_homology(points_2d, max_dimension=2, threshold=1.5)

# Extract persistence diagrams
diagrams = ph_result['dgms']
print(f"\nPersistence diagrams computed:")
for i, dgm in enumerate(diagrams):
    print(f"H_{i}: {len(dgm)} features")

# --- 3. Betti Number Analysis ---

def compute_betti_numbers_at_radius(diagrams, radius):
    """
    Compute Betti numbers at a specific radius
    
    Betti numbers count topological features:
    - β₀: number of connected components  
    - β₁: number of loops
    - β₂: number of voids/cavities
    """
    betti_numbers = []
    
    for dim, dgm in enumerate(diagrams):
        # Count features that are alive at this radius
        # A feature is alive if: birth <= radius < death
        alive_features = 0
        for birth, death in dgm:
            if birth <= radius < death:
                alive_features += 1
        betti_numbers.append(alive_features)
    
    return betti_numbers

def analyze_betti_evolution(diagrams, max_radius=1.5, num_points=100):
    """
    Analyze how Betti numbers change with increasing radius
    """
    radii = np.linspace(0, max_radius, num_points)
    betti_evolution = []
    
    for radius in radii:
        betti_nums = compute_betti_numbers_at_radius(diagrams, radius)
        betti_evolution.append(betti_nums)
    
    betti_evolution = np.array(betti_evolution)
    return radii, betti_evolution

# Compute Betti number evolution
radii, betti_evolution = analyze_betti_evolution(diagrams)

print("\nBetti Numbers Analysis:")
print("=" * 30)
print("At different radii:")
test_radii = [0.1, 0.3, 0.5, 0.7, 1.0]
for r in test_radii:
    betti = compute_betti_numbers_at_radius(diagrams, r)
    print(f"ε = {r:0.1f}: β₀={betti[0]:2d}, β₁={betti[1]:2d}, β₂={betti[2]:2d}")

# --- 4. Create Comprehensive Visualization ---

def create_ripser_analysis_plot():
    """
    Create a comprehensive plot showing:
    1. Original data with species colors
    2. Persistence diagrams  
    3. Betti number evolution
    4. Interactive slider to see complex at different radii
    """
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Data plot with species colors
    ax1 = plt.subplot(2, 3, 1)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    for i, name in enumerate(target_names):
        species_points = points_2d[y == i]
        ax1.scatter(species_points[:, 0], species_points[:, 1], 
                   c=colors[i], label=name, alpha=0.7, s=50)
    ax1.set_title('Iris Dataset (PCA Projection)', fontsize=12)
    ax1.set_xlabel('PC1')
    ax1.set_ylabel('PC2')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Persistence diagrams
    ax2 = plt.subplot(2, 3, 2)
    plot_diagrams(diagrams, show=False, ax=ax2)
    ax2.set_title('Persistence Diagrams', fontsize=12)
    
    # 3. Betti number evolution
    ax3 = plt.subplot(2, 3, 3)
    max_dim = len(diagrams)
    for dim in range(max_dim):
        ax3.plot(radii, betti_evolution[:, dim], 
                label=f'β_{dim} (H_{dim})', linewidth=2, marker='o', markersize=3)
    ax3.set_xlabel('Filtration Parameter ε')
    ax3.set_ylabel('Betti Numbers')
    ax3.set_title('Betti Number Evolution', fontsize=12)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Persistence barcode (custom implementation)
    ax4 = plt.subplot(2, 3, 4)
    
    def plot_barcodes(diagrams, ax):
        """Create a custom barcode plot"""
        colors = ['red', 'blue', 'green', 'purple']
        y_pos = 0
        
        for dim, dgm in enumerate(diagrams):
            for birth, death in dgm:
                if death == np.inf:
                    death = 1.5  # Cap infinite bars
                ax.plot([birth, death], [y_pos, y_pos], 
                       color=colors[dim % len(colors)], linewidth=2, alpha=0.7)
                y_pos += 1
        
        ax.set_xlabel('Filtration Parameter')
        ax.set_ylabel('Features')
        ax.set_title('Persistence Barcodes')
        
        # Add legend
        legend_elements = [plt.Line2D([0], [0], color=colors[i], lw=2, 
                          label=f'H_{i}') for i in range(len(diagrams))]
        ax.legend(handles=legend_elements)
    
    plot_barcodes(diagrams, ax4)
    
    # 5. Complex visualization (interactive)
    ax5 = plt.subplot(2, 3, 5)
    
    # Plot data points
    for i, name in enumerate(target_names):
        species_points = points_2d[y == i]
        ax5.scatter(species_points[:, 0], species_points[:, 1], 
                   c=colors[i], label=name, alpha=0.8, s=50, zorder=3)
    
    ax5.set_title('Vietoris-Rips Complex (Interactive)', fontsize=12)
    ax5.set_xlabel('PC1')
    ax5.set_ylabel('PC2')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Pre-calculate distance matrix
    dist_matrix = squareform(pdist(points_2d, 'euclidean'))
    drawn_artists = []
    
    def update_complex(radius):
        # Clear previous edges and circles
        for artist in drawn_artists:
            artist.remove()
        drawn_artists.clear()
        
        # Draw edges
        edge_indices = np.argwhere(dist_matrix < 2 * radius)
        for i, j in edge_indices:
            if i < j:
                p1, p2 = points_2d[i], points_2d[j]
                line, = ax5.plot([p1[0], p2[0]], [p1[1], p2[1]], 
                               'k-', alpha=0.3, lw=0.5, zorder=1)
                drawn_artists.append(line)
        
        # Draw circles
        for i in range(len(points_2d)):
            circle = plt.Circle(points_2d[i], radius, color='gray', 
                               alpha=0.1, fill=True, zorder=2)
            ax5.add_artist(circle)
            drawn_artists.append(circle)
        
        # Update Betti numbers in title
        current_betti = compute_betti_numbers_at_radius(diagrams, radius)
        betti_str = ", ".join([f"β_{i}={b}" for i, b in enumerate(current_betti)])
        ax5.set_title(f'Vietoris-Rips Complex (ε={radius:.2f})\n{betti_str}', fontsize=12)
        
        fig.canvas.draw_idle()
    
    # 6. Summary statistics
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    # Calculate some summary statistics
    total_features = sum(len(dgm) for dgm in diagrams)
    max_persistence = []
    for dim, dgm in enumerate(diagrams):
        if len(dgm) > 0:
            persistences = dgm[:, 1] - dgm[:, 0]
            max_pers = np.max(persistences[persistences < np.inf])
            max_persistence.append((dim, max_pers))
    
    summary_text = f"""
    PERSISTENT HOMOLOGY SUMMARY
    ═══════════════════════════
    
    Dataset: Iris (n={len(points_2d)})
    Total Features: {total_features}
    
    Homology Groups:
    • H₀: {len(diagrams[0])} connected components
    • H₁: {len(diagrams[1])} loops  
    • H₂: {len(diagrams[2])} cavities
    
    Most Persistent Features:
    """
    
    for dim, pers in max_persistence:
        summary_text += f"• H_{dim}: {pers:.3f}\n    "
    
    summary_text += f"""
    
    Key Insights:
    • Data has {len(target_names)} natural clusters
    • Topology reflects species structure
    • Persistent features indicate
      stable geometric patterns
    """
    
    ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, 
             fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    # Add slider for interactive complex visualization
    plt.subplots_adjust(bottom=0.15)
    ax_slider = plt.axes([0.1, 0.05, 0.65, 0.03])
    slider = Slider(ax_slider, 'Radius ε', 0, 1.5, valinit=0.3)
    
    def slider_update(val):
        update_complex(slider.val)
    
    slider.on_changed(slider_update)
    
    # Initialize with default radius
    update_complex(0.3)
    
    plt.tight_layout()
    return fig, slider

# --- 5. Run the Analysis ---

if __name__ == "__main__":
    print("\nCreating comprehensive visualization...")
    fig, slider = create_ripser_analysis_plot()
    
    print("\nAnalysis complete!")
    print("=" * 50)
    print("Interactive Features:")
    print("• Slide to change filtration radius")
    print("• Watch Betti numbers change in real-time")
    print("• Compare with persistence diagrams")
    print("• Observe topological feature evolution")
    
    plt.show() 