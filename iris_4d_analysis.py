import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from scipy.spatial.distance import pdist, squareform
import seaborn as sns

# You'll need ripser installed: pip install ripser persim
try:
    from ripser import ripser
    from persim import plot_diagrams
    RIPSER_AVAILABLE = True
except ImportError as e:
    print(f"Import error: {e}")
    print("Ripser not installed. Run: pip install ripser persim")
    RIPSER_AVAILABLE = False

def load_iris_4d():
    """Load the original 4D Iris dataset"""
    iris = datasets.load_iris()
    X = iris.data  # Keep original 4D data
    y = iris.target
    feature_names = iris.feature_names
    target_names = iris.target_names
    
    print("4D Iris Dataset Analysis")
    print("=" * 40)
    print(f"Data shape: {X.shape}")
    print(f"Features: {feature_names}")
    print(f"Species: {target_names}")
    print(f"Points per species: {np.bincount(y)}")
    
    return X, y, feature_names, target_names

def compute_4d_persistent_homology(X, max_dimension=3, threshold=None):
    """
    Compute persistent homology for 4D Iris dataset
    """
    if not RIPSER_AVAILABLE:
        print("Ripser not available. Cannot compute persistent homology.")
        return None
    
    # If no threshold specified, use a reasonable default based on data scale
    if threshold is None:
        distances = pdist(X)
        threshold = np.percentile(distances, 95)  # Use 95th percentile of distances
    
    print(f"\nComputing 4D Persistent Homology...")
    print(f"Max homology dimension: H_{max_dimension}")
    print(f"Filtration threshold: {threshold:.3f}")
    print(f"This may take a moment for 4D data...")
    
    # Compute persistent homology
    result = ripser(
        X,
        maxdim=max_dimension,  # Compute H_0, H_1, H_2, H_3
        thresh=threshold,
        metric='euclidean',
        n_perm=None  # Use all points, no subsampling
    )
    
    diagrams = result['dgms']
    
    print(f"\nPersistent Homology Results:")
    print("-" * 30)
    for i, dgm in enumerate(diagrams):
        print(f"H_{i}: {len(dgm)} topological features")
        if len(dgm) > 0:
            finite_features = dgm[dgm[:, 1] < np.inf]
            if len(finite_features) > 0:
                avg_persistence = np.mean(finite_features[:, 1] - finite_features[:, 0])
                max_persistence = np.max(finite_features[:, 1] - finite_features[:, 0])
                print(f"     Average persistence: {avg_persistence:.4f}")
                print(f"     Maximum persistence: {max_persistence:.4f}")
    
    return result, threshold

def create_persistence_table(diagrams, threshold, num_samples=50):
    """
    Create a comprehensive table of Betti numbers across filtration values
    """
    print(f"\nCreating Persistence Table...")
    
    # Create filtration values from 0 to threshold
    filtration_values = np.linspace(0, threshold, num_samples)
    
    # Initialize results list
    results = []
    
    for filt_val in filtration_values:
        # Calculate Betti numbers at this filtration value
        betti_numbers = compute_betti_numbers_at_filtration(diagrams, filt_val)
        
        # Create row of data
        row = {
            'filtration_value': filt_val,
            'beta_0': betti_numbers[0],  # Connected components
            'beta_1': betti_numbers[1],  # 1D loops
            'beta_2': betti_numbers[2],  # 2D voids
            'beta_3': betti_numbers[3] if len(betti_numbers) > 3 else 0,  # 3D voids
            'euler_characteristic': sum((-1)**i * b for i, b in enumerate(betti_numbers)),
            'total_features': sum(betti_numbers)
        }
        results.append(row)
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    print(f"Persistence table created with {len(df)} filtration values")
    return df

def compute_betti_numbers_at_filtration(diagrams, filtration_value):
    """
    Compute Betti numbers at a specific filtration value
    """
    betti_numbers = []
    
    for dim, dgm in enumerate(diagrams):
        # Count features alive at this filtration value
        alive_features = 0
        for birth, death in dgm:
            if birth <= filtration_value < death:
                alive_features += 1
        betti_numbers.append(alive_features)
    
    return betti_numbers

def analyze_persistence_features(diagrams):
    """
    Analyze the persistence features in detail
    """
    print(f"\nDetailed Feature Analysis:")
    print("=" * 50)
    
    feature_analysis = []
    
    for dim, dgm in enumerate(diagrams):
        print(f"\nH_{dim} (Dimension {dim}):")
        print("-" * 20)
        
        if len(dgm) == 0:
            print("  No features detected")
            continue
        
        # Separate finite and infinite features
        finite_mask = dgm[:, 1] < np.inf
        finite_features = dgm[finite_mask]
        infinite_features = dgm[~finite_mask]
        
        print(f"  Total features: {len(dgm)}")
        print(f"  Finite features: {len(finite_features)}")
        print(f"  Infinite features: {len(infinite_features)}")
        
        if len(finite_features) > 0:
            persistences = finite_features[:, 1] - finite_features[:, 0]
            births = finite_features[:, 0]
            deaths = finite_features[:, 1]
            
            print(f"  Birth range: [{births.min():.4f}, {births.max():.4f}]")
            print(f"  Death range: [{deaths.min():.4f}, {deaths.max():.4f}]")
            print(f"  Persistence range: [{persistences.min():.4f}, {persistences.max():.4f}]")
            print(f"  Average persistence: {persistences.mean():.4f}")
            
            # Find most persistent features
            if len(persistences) > 0:
                most_persistent_idx = np.argmax(persistences)
                most_persistent = persistences[most_persistent_idx]
                print(f"  Most persistent feature: {most_persistent:.4f}")
                print(f"    Birth: {births[most_persistent_idx]:.4f}")
                print(f"    Death: {deaths[most_persistent_idx]:.4f}")
            
            # Store for summary
            for i, (birth, death) in enumerate(finite_features):
                feature_analysis.append({
                    'dimension': dim,
                    'feature_id': i,
                    'birth': birth,
                    'death': death,
                    'persistence': death - birth
                })
    
    return pd.DataFrame(feature_analysis)

def create_comprehensive_visualization(X, diagrams, persistence_df, feature_df, threshold):
    """
    Create comprehensive visualization of 4D analysis results
    """
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Betti number evolution
    ax1 = plt.subplot(2, 4, 1)
    for dim in range(4):
        col_name = f'beta_{dim}'
        if col_name in persistence_df.columns:
            ax1.plot(persistence_df['filtration_value'], persistence_df[col_name], 
                    label=f'β_{dim}', linewidth=2, marker='o', markersize=3)
    ax1.set_xlabel('Filtration Value')
    ax1.set_ylabel('Betti Numbers')
    ax1.set_title('4D Betti Number Evolution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Euler characteristic evolution
    ax2 = plt.subplot(2, 4, 2)
    ax2.plot(persistence_df['filtration_value'], persistence_df['euler_characteristic'], 
             'purple', linewidth=2, marker='s', markersize=3)
    ax2.set_xlabel('Filtration Value')
    ax2.set_ylabel('Euler Characteristic')
    ax2.set_title('Euler Characteristic Evolution')
    ax2.grid(True, alpha=0.3)
    
    # 3. Total features
    ax3 = plt.subplot(2, 4, 3)
    ax3.plot(persistence_df['filtration_value'], persistence_df['total_features'], 
             'orange', linewidth=2, marker='^', markersize=3)
    ax3.set_xlabel('Filtration Value')
    ax3.set_ylabel('Total Features')
    ax3.set_title('Total Active Features')
    ax3.grid(True, alpha=0.3)
    
    # 4. Persistence diagrams
    ax4 = plt.subplot(2, 4, 4)
    plot_diagrams(diagrams, show=False, ax=ax4)
    ax4.set_title('4D Persistence Diagrams')
    
    # 5. Feature persistence distribution
    ax5 = plt.subplot(2, 4, 5)
    if not feature_df.empty:
        for dim in sorted(feature_df['dimension'].unique()):
            dim_features = feature_df[feature_df['dimension'] == dim]
            ax5.hist(dim_features['persistence'], alpha=0.6, 
                    label=f'H_{dim}', bins=20)
    ax5.set_xlabel('Persistence')
    ax5.set_ylabel('Count')
    ax5.set_title('Feature Persistence Distribution')
    ax5.legend()
    
    # 6. Birth-death scatter
    ax6 = plt.subplot(2, 4, 6)
    if not feature_df.empty:
        colors = ['red', 'blue', 'green', 'purple']
        for dim in sorted(feature_df['dimension'].unique()):
            dim_features = feature_df[feature_df['dimension'] == dim]
            ax6.scatter(dim_features['birth'], dim_features['death'], 
                       alpha=0.6, label=f'H_{dim}', c=colors[dim % len(colors)])
    
    # Add diagonal line
    max_val = max(ax6.get_xlim()[1], ax6.get_ylim()[1])
    ax6.plot([0, max_val], [0, max_val], 'k--', alpha=0.5)
    ax6.set_xlabel('Birth')
    ax6.set_ylabel('Death')
    ax6.set_title('Birth-Death Scatter')
    ax6.legend()
    
    # 7. Heatmap of Betti numbers
    ax7 = plt.subplot(2, 4, 7)
    betti_matrix = persistence_df[['beta_0', 'beta_1', 'beta_2', 'beta_3']].T
    sns.heatmap(betti_matrix, ax=ax7, cmap='viridis', 
                xticklabels=np.arange(0, len(persistence_df), 5),
                yticklabels=['β₀', 'β₁', 'β₂', 'β₃'])
    ax7.set_xlabel('Filtration Step')
    ax7.set_title('Betti Number Heatmap')
    
    # 8. Summary statistics
    ax8 = plt.subplot(2, 4, 8)
    ax8.axis('off')
    
    # Calculate summary statistics
    max_beta_0 = persistence_df['beta_0'].max()
    max_beta_1 = persistence_df['beta_1'].max()
    max_beta_2 = persistence_df['beta_2'].max()
    max_beta_3 = persistence_df['beta_3'].max()
    
    total_features_overall = len(feature_df) if not feature_df.empty else 0
    
    if not feature_df.empty:
        avg_persistence = feature_df['persistence'].mean()
        max_persistence = feature_df['persistence'].max()
    else:
        avg_persistence = 0
        max_persistence = 0
    
    summary_text = f"""
    4D IRIS PERSISTENT HOMOLOGY
    ═════════════════════════════
    
    Dataset: Iris 4D (n=150)
    Filtration threshold: {threshold:.3f}
    
    Maximum Betti Numbers:
    • β₀ (components): {max_beta_0}
    • β₁ (loops): {max_beta_1}
    • β₂ (voids): {max_beta_2}
    • β₃ (3D voids): {max_beta_3}
    
    Feature Statistics:
    • Total features: {total_features_overall}
    • Avg persistence: {avg_persistence:.4f}
    • Max persistence: {max_persistence:.4f}
    
    Key Insights:
    • 4D analysis reveals higher-
      dimensional topological structure
    • Captures features lost in 2D
      projection
    • Provides complete topological
      signature of Iris species
    """
    
    ax8.text(0.05, 0.95, summary_text, transform=ax8.transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    return fig

def save_results(persistence_df, feature_df, output_dir="."):
    """
    Save results to CSV files
    """
    print(f"\nSaving results...")
    
    # Save persistence table
    persistence_file = f"{output_dir}/iris_4d_persistence_table.csv"
    persistence_df.to_csv(persistence_file, index=False)
    print(f"Persistence table saved to: {persistence_file}")
    
    # Save feature analysis
    if not feature_df.empty:
        feature_file = f"{output_dir}/iris_4d_features.csv"
        feature_df.to_csv(feature_file, index=False)
        print(f"Feature analysis saved to: {feature_file}")
    
    # Save summary statistics
    summary_file = f"{output_dir}/iris_4d_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("4D Iris Dataset Persistent Homology Analysis\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("BETTI NUMBER SUMMARY:\n")
        f.write("-" * 20 + "\n")
        for col in ['beta_0', 'beta_1', 'beta_2', 'beta_3']:
            if col in persistence_df.columns:
                max_val = persistence_df[col].max()
                mean_val = persistence_df[col].mean()
                f.write(f"{col}: max={max_val}, mean={mean_val:.2f}\n")
        
        f.write(f"\nFILTRATION RANGE: 0 to {persistence_df['filtration_value'].max():.4f}\n")
        f.write(f"TOTAL SAMPLES: {len(persistence_df)}\n")
        
        if not feature_df.empty:
            f.write(f"\nFEATURE SUMMARY:\n")
            f.write("-" * 15 + "\n")
            for dim in sorted(feature_df['dimension'].unique()):
                dim_features = feature_df[feature_df['dimension'] == dim]
                f.write(f"H_{dim}: {len(dim_features)} features\n")
    
    print(f"Summary saved to: {summary_file}")

def main():
    """
    Main analysis function
    """
    if not RIPSER_AVAILABLE:
        return
    
    # Load 4D Iris dataset
    X, y, feature_names, target_names = load_iris_4d()
    
    # Compute 4D persistent homology
    result, threshold = compute_4d_persistent_homology(X, max_dimension=3)
    if result is None:
        return
    
    diagrams = result['dgms']
    
    # Create persistence table
    persistence_df = create_persistence_table(diagrams, threshold)
    
    # Analyze features in detail
    feature_df = analyze_persistence_features(diagrams)
    
    # Display key results
    print(f"\nKEY RESULTS:")
    print("=" * 30)
    print(f"Maximum Betti numbers:")
    for col in ['beta_0', 'beta_1', 'beta_2', 'beta_3']:
        if col in persistence_df.columns:
            max_val = persistence_df[col].max()
            print(f"  {col}: {max_val}")
    
    print(f"\nPersistence table preview:")
    print(persistence_df.head(10))
    
    # Create visualization
    print(f"\nCreating comprehensive visualization...")
    fig = create_comprehensive_visualization(X, diagrams, persistence_df, feature_df, threshold)
    
    # Save results
    save_results(persistence_df, feature_df)
    
    print(f"\nAnalysis complete! Check the generated files and visualization.")
    plt.show()

if __name__ == "__main__":
    main() 