import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# Page configuration
st.set_page_config(
    page_title="🌸 Iris Persistent Homology Dashboard",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
    }
    .stMetric {
        background: #ffffff;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load pre-computed persistent homology data"""
    data = {}
    
    # Try to load persistence table
    if os.path.exists('iris_4d_persistence_table.csv'):
        data['persistence_table'] = pd.read_csv('iris_4d_persistence_table.csv')
    
    # Try to load features
    if os.path.exists('iris_4d_features.csv'):
        data['features'] = pd.read_csv('iris_4d_features.csv')
    
    # Try to load 2D comparison data if available
    if os.path.exists('iris_2d_persistence_table.csv'):
        data['persistence_2d'] = pd.read_csv('iris_2d_persistence_table.csv')
    
    return data

def create_betti_evolution_plot(df):
    """Create interactive Betti number evolution plot"""
    fig = go.Figure()
    
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
    betti_cols = ['beta_0', 'beta_1', 'beta_2', 'beta_3']
    labels = ['β₀ (Components)', 'β₁ (Loops)', 'β₂ (2D Voids)', 'β₃ (3D Voids)']
    
    for i, (col, label) in enumerate(zip(betti_cols, labels)):
        if col in df.columns:
            fig.add_trace(go.Scatter(
                x=df['filtration_value'],
                y=df[col],
                mode='lines+markers',
                name=label,
                line=dict(color=colors[i], width=3),
                marker=dict(size=6),
                hovertemplate=f"<b>{label}</b><br>" +
                             "Filtration: %{x:.3f}<br>" +
                             "Count: %{y}<br>" +
                             "<extra></extra>"
            ))
    
    fig.update_layout(
        title="🔄 Betti Number Evolution",
        xaxis_title="Filtration Parameter (ε)",
        yaxis_title="Betti Numbers",
        hovermode='x unified',
        template='plotly_white',
        height=500,
        showlegend=True
    )
    
    return fig

def create_persistence_diagram(features_df):
    """Create interactive persistence diagram"""
    if features_df.empty:
        return go.Figure()
    
    fig = go.Figure()
    
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
    dim_labels = ['H₀', 'H₁', 'H₂', 'H₃']
    
    for dim in sorted(features_df['dimension'].unique()):
        dim_data = features_df[features_df['dimension'] == dim]
        
        fig.add_trace(go.Scatter(
            x=dim_data['birth'],
            y=dim_data['death'],
            mode='markers',
            name=dim_labels[dim],
            marker=dict(
                color=colors[dim],
                size=8,
                opacity=0.7
            ),
            hovertemplate=f"<b>{dim_labels[dim]}</b><br>" +
                         "Birth: %{x:.3f}<br>" +
                         "Death: %{y:.3f}<br>" +
                         "Persistence: %{customdata:.3f}<br>" +
                         "<extra></extra>",
            customdata=dim_data['persistence']
        ))
    
    # Add diagonal line
    max_val = max(features_df['death'].max(), features_df['birth'].max())
    fig.add_trace(go.Scatter(
        x=[0, max_val],
        y=[0, max_val],
        mode='lines',
        name='Diagonal',
        line=dict(color='black', dash='dash'),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    fig.update_layout(
        title="📊 Persistence Diagram",
        xaxis_title="Birth",
        yaxis_title="Death",
        template='plotly_white',
        height=500,
        showlegend=True
    )
    
    return fig

def create_feature_barcode(features_df):
    """Create persistence barcode plot"""
    if features_df.empty:
        return go.Figure()
    
    fig = go.Figure()
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
    
    y_pos = 0
    for dim in sorted(features_df['dimension'].unique()):
        dim_data = features_df[features_df['dimension'] == dim].sort_values('birth')
        
        for _, row in dim_data.iterrows():
            fig.add_trace(go.Scatter(
                x=[row['birth'], row['death']],
                y=[y_pos, y_pos],
                mode='lines',
                line=dict(color=colors[dim], width=4),
                showlegend=False,
                hovertemplate=f"H_{dim}<br>" +
                             f"Birth: {row['birth']:.3f}<br>" +
                             f"Death: {row['death']:.3f}<br>" +
                             f"Persistence: {row['persistence']:.3f}<br>" +
                             "<extra></extra>"
            ))
            y_pos += 1
    
    fig.update_layout(
        title="📈 Persistence Barcode",
        xaxis_title="Filtration Parameter",
        yaxis_title="Features",
        template='plotly_white',
        height=600,
        showlegend=False
    )
    
    return fig

def create_summary_metrics(persistence_df, features_df):
    """Create summary metrics for the dashboard"""
    metrics = {}
    
    if not persistence_df.empty:
        # Maximum Betti numbers
        for i in range(4):
            col = f'beta_{i}'
            if col in persistence_df.columns:
                metrics[f'max_beta_{i}'] = persistence_df[col].max()
        
        # Total features over time
        metrics['max_total_features'] = persistence_df['total_features'].max()
        
        # Euler characteristic range
        if 'euler_characteristic' in persistence_df.columns:
            metrics['min_euler'] = persistence_df['euler_characteristic'].min()
            metrics['max_euler'] = persistence_df['euler_characteristic'].max()
    
    if not features_df.empty:
        # Feature statistics
        metrics['total_features'] = len(features_df)
        metrics['avg_persistence'] = features_df['persistence'].mean()
        metrics['max_persistence'] = features_df['persistence'].max()
        
        # Features by dimension
        for dim in features_df['dimension'].unique():
            metrics[f'h{dim}_features'] = len(features_df[features_df['dimension'] == dim])
    
    return metrics

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1 style="color: white; margin: 0;">🌸 Iris Dataset Persistent Homology Dashboard</h1>
        <p style="color: white; margin: 0.5rem 0 0 0;">Interactive analysis of topological features in 4D Iris data</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data
    data = load_data()
    
    if not data:
        st.error("⚠️ No data files found! Please run the analysis scripts first:")
        st.code("""
        # Run this first to generate the data:
        python iris_4d_analysis.py
        """)
        return
    
    # Sidebar controls
    st.sidebar.header("🎛️ Dashboard Controls")
    
    # Analysis type selection
    analysis_type = st.sidebar.selectbox(
        "Select Analysis",
        ["4D Analysis", "Feature Details", "Data Tables", "Comparison"]
    )
    
    if analysis_type == "4D Analysis":
        show_main_analysis(data)
    elif analysis_type == "Feature Details":
        show_feature_details(data)
    elif analysis_type == "Data Tables":
        show_data_tables(data)
    elif analysis_type == "Comparison":
        show_comparison(data)

def show_main_analysis(data):
    """Show main 4D analysis dashboard"""
    persistence_df = data.get('persistence_table', pd.DataFrame())
    features_df = data.get('features', pd.DataFrame())
    
    if persistence_df.empty:
        st.error("No persistence data available")
        return
    
    # Summary metrics
    metrics = create_summary_metrics(persistence_df, features_df)
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "🔗 Max Connected Components", 
            value=metrics.get('max_beta_0', 'N/A'),
            help="Maximum number of connected components (β₀)"
        )
    
    with col2:
        st.metric(
            "🔄 Max Loops", 
            value=metrics.get('max_beta_1', 'N/A'),
            help="Maximum number of 1D loops (β₁)"
        )
    
    with col3:
        st.metric(
            "🕳️ Max 2D Voids", 
            value=metrics.get('max_beta_2', 'N/A'),
            help="Maximum number of 2D voids (β₂)"
        )
    
    with col4:
        st.metric(
            "📦 Max 3D Voids", 
            value=metrics.get('max_beta_3', 'N/A'),
            help="Maximum number of 3D voids (β₃)"
        )
    
    # Main plots
    col1, col2 = st.columns(2)
    
    with col1:
        # Betti evolution
        fig_betti = create_betti_evolution_plot(persistence_df)
        st.plotly_chart(fig_betti, use_container_width=True)
        
        # Euler characteristic
        if 'euler_characteristic' in persistence_df.columns:
            fig_euler = px.line(
                persistence_df, 
                x='filtration_value', 
                y='euler_characteristic',
                title="🔢 Euler Characteristic Evolution",
                template='plotly_white'
            )
            fig_euler.update_traces(line_color='purple', line_width=3)
            st.plotly_chart(fig_euler, use_container_width=True)
    
    with col2:
        # Persistence diagram
        if not features_df.empty:
            fig_pd = create_persistence_diagram(features_df)
            st.plotly_chart(fig_pd, use_container_width=True)
            
            # Total features evolution
            if 'total_features' in persistence_df.columns:
                fig_total = px.line(
                    persistence_df,
                    x='filtration_value',
                    y='total_features',
                    title="📊 Total Active Features",
                    template='plotly_white'
                )
                fig_total.update_traces(line_color='orange', line_width=3)
                st.plotly_chart(fig_total, use_container_width=True)

def show_feature_details(data):
    """Show detailed feature analysis"""
    features_df = data.get('features', pd.DataFrame())
    
    if features_df.empty:
        st.error("No feature data available")
        return
    
    st.header("🔍 Feature Details")
    
    # Feature barcode
    fig_barcode = create_feature_barcode(features_df)
    st.plotly_chart(fig_barcode, use_container_width=True)
    
    # Feature statistics by dimension
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Persistence Distribution")
        fig_hist = px.histogram(
            features_df,
            x='persistence',
            color='dimension',
            nbins=30,
            title="Distribution of Feature Persistence",
            template='plotly_white'
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    
    with col2:
        st.subheader("🎯 Features by Dimension")
        dim_counts = features_df['dimension'].value_counts().sort_index()
        fig_pie = px.pie(
            values=dim_counts.values,
            names=[f'H_{i}' for i in dim_counts.index],
            title="Feature Count by Homology Dimension"
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    # Most persistent features
    st.subheader("🏆 Most Persistent Features")
    top_features = features_df.nlargest(10, 'persistence')
    
    for _, feature in top_features.iterrows():
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Dimension", f"H_{int(feature['dimension'])}")
        with col2:
            st.metric("Birth", f"{feature['birth']:.4f}")
        with col3:
            st.metric("Death", f"{feature['death']:.4f}")
        with col4:
            st.metric("Persistence", f"{feature['persistence']:.4f}")

def show_data_tables(data):
    """Show data tables"""
    st.header("📋 Data Tables")
    
    tab1, tab2 = st.tabs(["Persistence Table", "Feature Details"])
    
    with tab1:
        persistence_df = data.get('persistence_table', pd.DataFrame())
        if not persistence_df.empty:
            st.subheader("🔄 Persistence Evolution Table")
            
            # Filter controls
            col1, col2 = st.columns(2)
            with col1:
                min_filt = st.slider(
                    "Minimum Filtration Value",
                    min_value=float(persistence_df['filtration_value'].min()),
                    max_value=float(persistence_df['filtration_value'].max()),
                    value=float(persistence_df['filtration_value'].min()),
                    step=0.01
                )
            with col2:
                max_filt = st.slider(
                    "Maximum Filtration Value",
                    min_value=float(persistence_df['filtration_value'].min()),
                    max_value=float(persistence_df['filtration_value'].max()),
                    value=float(persistence_df['filtration_value'].max()),
                    step=0.01
                )
            
            # Filter data
            filtered_df = persistence_df[
                (persistence_df['filtration_value'] >= min_filt) &
                (persistence_df['filtration_value'] <= max_filt)
            ]
            
            st.dataframe(
                filtered_df,
                use_container_width=True,
                height=400
            )
            
            # Download button
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name="filtered_persistence_table.csv",
                mime="text/csv"
            )
        else:
            st.error("No persistence table data available")
    
    with tab2:
        features_df = data.get('features', pd.DataFrame())
        if not features_df.empty:
            st.subheader("🎯 Individual Features")
            
            # Filter by dimension
            selected_dims = st.multiselect(
                "Select Homology Dimensions",
                options=sorted(features_df['dimension'].unique()),
                default=sorted(features_df['dimension'].unique()),
                format_func=lambda x: f"H_{x}"
            )
            
            if selected_dims:
                filtered_features = features_df[features_df['dimension'].isin(selected_dims)]
                
                st.dataframe(
                    filtered_features.sort_values('persistence', ascending=False),
                    use_container_width=True,
                    height=400
                )
                
                # Download button
                csv = filtered_features.to_csv(index=False)
                st.download_button(
                    label="📥 Download Features CSV",
                    data=csv,
                    file_name="filtered_features.csv",
                    mime="text/csv"
                )
            else:
                st.warning("Please select at least one dimension")
        else:
            st.error("No feature data available")

def show_comparison(data):
    """Show comparison between different analyses"""
    st.header("⚖️ Analysis Comparison")
    
    persistence_4d = data.get('persistence_table', pd.DataFrame())
    persistence_2d = data.get('persistence_2d', pd.DataFrame())
    
    if persistence_4d.empty:
        st.error("No 4D data available")
        return
    
    if not persistence_2d.empty:
        st.success("🎯 Comparing 4D vs 2D Analysis")
        
        # Side by side comparison
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("4D Analysis")
            fig_4d = create_betti_evolution_plot(persistence_4d)
            fig_4d.update_layout(title="4D Betti Evolution")
            st.plotly_chart(fig_4d, use_container_width=True)
        
        with col2:
            st.subheader("2D Analysis")
            fig_2d = create_betti_evolution_plot(persistence_2d)
            fig_2d.update_layout(title="2D Betti Evolution")
            st.plotly_chart(fig_2d, use_container_width=True)
        
        # Comparison metrics
        st.subheader("📊 Comparison Metrics")
        
        metrics_4d = create_summary_metrics(persistence_4d, data.get('features', pd.DataFrame()))
        metrics_2d = create_summary_metrics(persistence_2d, pd.DataFrame())
        
        comparison_data = {
            'Metric': ['Max β₀', 'Max β₁', 'Max β₂', 'Max β₃'],
            '4D Analysis': [
                metrics_4d.get('max_beta_0', 0),
                metrics_4d.get('max_beta_1', 0),
                metrics_4d.get('max_beta_2', 0),
                metrics_4d.get('max_beta_3', 0)
            ],
            '2D Analysis': [
                metrics_2d.get('max_beta_0', 0),
                metrics_2d.get('max_beta_1', 0),
                metrics_2d.get('max_beta_2', 0),
                0  # No β₃ in 2D
            ]
        }
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True)
        
    else:
        st.info("💡 Run the 2D analysis to enable comparison")
        st.code("""
        # Run this to generate 2D comparison data:
        python ripser_analysis.py
        """)

if __name__ == "__main__":
    main() 