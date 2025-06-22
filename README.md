# 🌸 Iris Dataset Persistent Homology Dashboard

An interactive web dashboard for exploring persistent homology analysis of the famous Iris dataset using topological data analysis (TDA).

## 🎯 What This Dashboard Shows

- **4D Persistent Homology Analysis** of the original Iris dataset
- **Betti Number Evolution** showing how topological features change
- **Interactive Persistence Diagrams** with birth/death times
- **Feature Analysis** with detailed barcode plots
- **Comprehensive Data Tables** with filtering and export capabilities

## 🔬 Key Discoveries

Our 4D analysis reveals topological features invisible in 2D projections:

- **β₀ (Connected Components)**: Max 149 components
- **β₁ (Loops)**: Max 5 loops detected 
- **β₂ (2D Voids)**: 4 cavity features found (lost in 2D projection!)
- **β₃ (3D Voids)**: None detected

## 🚀 Try the Dashboard

**Live Demo**: [Your Streamlit URL will go here]

## 📊 Dashboard Features

### 4D Analysis View
- Interactive Betti number evolution plots
- Persistence diagrams with hover details
- Euler characteristic analysis
- Summary metrics cards

### Feature Details View  
- Persistence barcode visualization
- Feature distribution histograms
- Most persistent features ranking

### Data Tables View
- Filterable persistence evolution table
- Feature details with dimension filtering
- CSV export functionality

### Comparison View
- 4D vs 2D analysis comparison
- Visual differences highlighting

## 🛠️ Technical Details

- **Framework**: Streamlit + Plotly
- **TDA Library**: Ripser for persistent homology computation
- **Dataset**: Iris (150 points, 4 features, 3 species)
- **Analysis**: Vietoris-Rips complex filtration

## 📁 Repository Contents

- `streamlit_dashboard.py` - Main dashboard application
- `iris_4d_persistence_table.csv` - Betti numbers across filtration values
- `iris_4d_features.csv` - Individual topological features
- `iris_4d_analysis.py` - Analysis script for generating data
- `requirements.txt` - Python dependencies

## 🧬 Scientific Significance

This analysis demonstrates how **dimensionality reduction loses important topological information**:

- 2D PCA projection: Convenient but topologically incomplete
- 4D original space: Reveals true topological signature of species separation
- Detection of H₂ features: Higher-dimensional geometric patterns in species clusters

## 🎓 About Persistent Homology

Persistent homology tracks how topological features (components, loops, voids) evolve across different scales, providing a mathematical fingerprint of data shape and structure.

## 💻 Local Development

```bash
# Clone the repository
git clone https://github.com/yourusername/iris-persistent-homology-dashboard.git
cd iris-persistent-homology-dashboard

# Install dependencies
pip install -r requirements.txt

# Run the dashboard
streamlit run streamlit_dashboard.py
```

## 📚 Learn More

- [Ripser Library](https://ripser.scikit-tda.org/)
- [Topological Data Analysis](https://en.wikipedia.org/wiki/Topological_data_analysis)
- [Persistent Homology Introduction](https://gudhi.inria.fr/introduction/)

---

*Created with ❤️ for exploring the beautiful intersection of topology and data science* 