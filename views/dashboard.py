import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, MeanShift, estimate_bandwidth
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA
from streamlit_folium import folium_static
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

from utils import (
    try_read_csv, 
    convert_columns_to_numeric, 
    detect_geo_columns, 
    create_cluster_map,
    create_heatmap
)

def show():
    # Advanced CSS Styling
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }
    
    h1, h2, h3 {
        font-family: 'Inter', sans-serif;
        letter-spacing: -0.02em;
    }
    
    .main-title {
        font-size: 2.25rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 0.5rem;
        line-height: 1.2;
    }
    
    .subtitle {
        font-size: 1rem;
        color: #64748b;
        font-weight: 400;
        margin-bottom: 2.5rem;
        line-height: 1.5;
    }
    
    .section-header {
        font-size: 1.25rem;
        font-weight: 600;
        color: #1e293b;
        margin: 2.5rem 0 1.25rem 0;
        padding-bottom: 0.75rem;
        border-bottom: 1px solid #e2e8f0;
    }
    
    .info-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 0.75rem;
        padding: 1.25rem;
        margin: 1rem 0;
    }
    
    .info-card strong {
        color: #334155;
        font-weight: 600;
    }
    
    .success-alert {
        background: #f0fdf4;
        border: 1px solid #bbf7d0;
        border-radius: 0.5rem;
        padding: 0.875rem 1.125rem;
        color: #166534;
        font-size: 0.9rem;
        margin: 1rem 0;
    }
    
    .warning-alert {
        background: #fef3c7;
        border: 1px solid #fde68a;
        border-radius: 0.5rem;
        padding: 0.875rem 1.125rem;
        color: #92400e;
        font-size: 0.9rem;
        margin: 1rem 0;
    }
    
    .stMetric {
        background: white;
        padding: 1.25rem;
        border-radius: 0.75rem;
        border: 1px solid #e2e8f0;
        box-shadow: 0 1px 2px 0 rgb(0 0 0 / 0.05);
    }
    
    .stMetric:hover {
        border-color: #cbd5e1;
        box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1);
        transition: all 0.2s ease;
    }
    
    .stButton > button {
        background: #0f172a;
        color: white;
        border: none;
        border-radius: 0.5rem;
        padding: 0.625rem 1.5rem;
        font-weight: 500;
        font-size: 0.9rem;
        transition: all 0.2s ease;
        letter-spacing: -0.01em;
    }
    
    .stButton > button:hover {
        background: #1e293b;
        box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1);
        transform: translateY(-1px);
    }
    
    .stSelectbox > div > div {
        border-radius: 0.5rem;
        border: 1px solid #e2e8f0;
    }
    
    .stMultiSelect > div > div {
        border-radius: 0.5rem;
        border: 1px solid #e2e8f0;
    }
    
    div[data-testid="stExpander"] {
        border: 1px solid #e2e8f0;
        border-radius: 0.75rem;
        background: white;
    }
    
    div[data-testid="stExpander"] > div:first-child {
        background: #f8fafc;
        border-radius: 0.75rem 0.75rem 0 0;
    }
    
    .welcome-box {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #a855f7 100%);
        padding: 3.5rem 2.5rem;
        border-radius: 1rem;
        text-align: center;
        margin: 2rem 0 3rem 0;
        box-shadow: 0 10px 40px rgba(139, 92, 246, 0.2);
    }
    
    .welcome-title {
        margin: 0 0 1rem 0;
        font-weight: 700;
        font-size: 2rem;
        color: white;
    }
    
    .welcome-subtitle {
        font-size: 1.05rem;
        color: rgba(255,255,255,0.95);
        margin: 0;
    }
    
    .features-section {
        margin-top: 2.5rem;
    }
    
    .features-title {
        color: #1e293b;
        font-weight: 600;
        margin-bottom: 1.5rem;
        font-size: 1.25rem;
    }
    
    .features-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
        gap: 1.25rem;
    }
    
    .feature-box {
        background: white;
        padding: 1.75rem;
        border-radius: 0.75rem;
        border: 1px solid #e2e8f0;
    }
    
    .feature-emoji {
        font-size: 2rem;
        margin-bottom: 0.75rem;
    }
    
    .feature-name {
        font-weight: 600;
        font-size: 1.1rem;
        margin-bottom: 0.5rem;
    }
    
    .feature-text {
        color: #64748b;
        font-size: 0.9rem;
        line-height: 1.5;
        margin: 0;
    }
    
    [data-testid="stSidebar"] {
        background: #f8fafc;
    }
    
    [data-testid="stSidebar"] h3 {
        font-size: 0.95rem;
        font-weight: 600;
        color: #1e293b;
        margin-top: 1.5rem;
        margin-bottom: 0.75rem;
    }
    
    .dataframe {
        border: 1px solid #e2e8f0 !important;
        border-radius: 0.5rem;
        font-size: 0.85rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<div class="main-title">Data Clustering Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Advanced machine learning clustering with comprehensive visualization and evaluation tools</div>', unsafe_allow_html=True)
    
    df = None
    
    # Sidebar
    if 'main_df' in st.session_state:
        df = st.session_state['main_df']
        if st.sidebar.button("Load Different Dataset"):
            del st.session_state['main_df']
            st.rerun()
    else:
        st.sidebar.markdown("### Data Upload")
        uploaded_file = st.sidebar.file_uploader("Select CSV file", type=["csv"], label_visibility="collapsed")
        if uploaded_file:
            try:
                df = try_read_csv(uploaded_file)
                df.columns = df.columns.str.strip()
                st.session_state['main_df'] = df
                st.success("Dataset loaded successfully")
                st.rerun()
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
    
    if df is not None:
        # Dataset Overview
        st.markdown('<div class="section-header">Dataset Overview</div>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Records", f"{len(df):,}", help="Total number of observations")
        with col2:
            st.metric("Features", df.shape[1], help="Number of variables")
        with col3:
            memory_kb = df.memory_usage(deep=True).sum() / 1024
            st.metric("Memory", f"{memory_kb:.1f} KB", help="Dataset size in memory")
        with col4:
            missing = df.isnull().sum().sum()
            missing_pct = (missing / (df.shape[0] * df.shape[1]) * 100) if df.size > 0 else 0
            st.metric("Missing", f"{missing_pct:.1f}%", help="Percentage of missing values")
        
        with st.expander("Preview Dataset"):
            st.dataframe(df.head(10), use_container_width=True, height=350)
        
        # Data Processing
        df_numeric = convert_columns_to_numeric(df)
        geo_info = detect_geo_columns(df)
        has_geo = geo_info['lat'] is not None and geo_info['lon'] is not None
        
        if df_numeric.shape[1] == 0:
            st.markdown('<div class="warning-alert">No numeric columns detected. Please ensure your dataset contains numerical values for clustering analysis.</div>', unsafe_allow_html=True)
            st.stop()
        
        st.markdown(f'<div class="success-alert">Processed {df_numeric.shape[1]} numeric features successfully</div>', unsafe_allow_html=True)
        
        if has_geo:
            st.markdown(f'<div class="info-card">Geographic coordinates detected: <strong>{geo_info["lat"]}</strong> and <strong>{geo_info["lon"]}</strong></div>', unsafe_allow_html=True)
        
        # Feature Selection
        st.sidebar.markdown("---")
        st.sidebar.markdown("### Analysis Configuration")
        
        default_cols = df_numeric.columns.tolist()[:min(3, len(df_numeric.columns))]
        selected_columns = st.sidebar.multiselect(
            "Features for clustering",
            df_numeric.columns.tolist(),
            default=default_cols
        )
        
        if not selected_columns:
            st.markdown('<div class="warning-alert">Please select at least one feature to proceed with clustering analysis.</div>', unsafe_allow_html=True)
            st.stop()
        
        X = df_numeric[selected_columns].copy()
        
        # Preprocessing
        with st.spinner("Processing data..."):
            imputer = SimpleImputer(strategy="mean")
            X_imputed = imputer.fit_transform(X)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_imputed)
        
        # Algorithm Selection
        st.sidebar.markdown("### Clustering Algorithm")
        algo = st.sidebar.selectbox(
            "Select method",
            ["KMeans", "DBSCAN", "Hierarchical", "Mean Shift", "Grid-Based"],
            label_visibility="collapsed"
        )
        
        # Algorithm Configuration
        st.markdown(f'<div class="section-header">Algorithm Configuration: {algo}</div>', unsafe_allow_html=True)
        
        labels = None
        param_col1, param_col2 = st.columns([1, 2])
        
        if algo == "KMeans":
            with param_col1:
                k = st.slider("Number of clusters", 2, 10, 3)
            with param_col2:
                st.markdown("""
                <div class="info-card">
                <strong>K-Means Clustering</strong><br>
                Partitions data into k distinct clusters based on distance to centroids. 
                Best suited for datasets with spherical cluster shapes and balanced sizes.
                </div>
                """, unsafe_allow_html=True)
            model = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = model.fit_predict(X_scaled)
        
        elif algo == "DBSCAN":
            with param_col1:
                eps = st.slider("Epsilon", 0.1, 10.0, 0.5, 0.1)
                min_samples = st.slider("Minimum samples", 1, 20, 5)
            with param_col2:
                st.markdown("""
                <div class="info-card">
                <strong>DBSCAN</strong><br>
                Density-based algorithm that discovers clusters of arbitrary shape and identifies outliers.
                No need to specify the number of clusters beforehand.
                </div>
                """, unsafe_allow_html=True)
            model = DBSCAN(eps=eps, min_samples=min_samples)
            labels = model.fit_predict(X_scaled)
        
        elif algo == "Hierarchical":
            with param_col1:
                k = st.slider("Number of clusters", 2, 10, 3)
                linkage = st.selectbox("Linkage method", ["ward", "complete", "average", "single"])
            with param_col2:
                st.markdown("""
                <div class="info-card">
                <strong>Hierarchical Clustering</strong><br>
                Builds a hierarchy of clusters using bottom-up approach. 
                Useful for exploring data structure and finding optimal number of clusters.
                </div>
                """, unsafe_allow_html=True)
            model = AgglomerativeClustering(n_clusters=k, linkage=linkage)
            labels = model.fit_predict(X_scaled)
        
        elif algo == "Mean Shift":
            with param_col1:
                use_auto = st.checkbox("Auto-detect bandwidth", value=True)
                if not use_auto:
                    bandwidth = st.slider("Bandwidth", 0.1, 5.0, 1.0, 0.1)
            with param_col2:
                st.markdown("""
                <div class="info-card">
                <strong>Mean Shift</strong><br>
                Discovers clusters by finding modes in the feature space density.
                Automatically determines the number of clusters without prior specification.
                </div>
                """, unsafe_allow_html=True)
            
            if use_auto:
                bandwidth_val = estimate_bandwidth(X_scaled, quantile=0.2, n_samples=min(500, len(X_scaled)))
                if bandwidth_val > 0:
                    st.info(f"Detected bandwidth: {bandwidth_val:.3f}")
                    model = MeanShift(bandwidth=bandwidth_val)
                else:
                    st.warning("Auto-detection failed, using default bandwidth")
                    model = MeanShift(bandwidth=1.0)
            else:
                model = MeanShift(bandwidth=bandwidth)
            labels = model.fit_predict(X_scaled)
        
        elif algo == "Grid-Based":
            with param_col1:
                n_bins = st.slider("Grid divisions", 2, 20, 5)
                density_threshold = st.slider("Density threshold", 1, 20, 3)
            with param_col2:
                st.markdown("""
                <div class="info-card">
                <strong>Grid-Based Clustering</strong><br>
                Partitions feature space into a grid and groups based on density.
                Efficient for large datasets with faster computation time.
                </div>
                """, unsafe_allow_html=True)
            
            bins = [np.linspace(np.min(X_scaled[:, i]), np.max(X_scaled[:, i]), n_bins + 1) 
                    for i in range(X_scaled.shape[1])]
            grid_indices = np.array([np.digitize(X_scaled[:, i], bins[i]) 
                                    for i in range(X_scaled.shape[1])]).T
            
            grid_counts = defaultdict(int)
            for idx in grid_indices:
                grid_counts[tuple(idx)] += 1
            
            labels = np.full(X_scaled.shape[0], -1, dtype=int)
            cluster_id = 0
            dense_grids = [g for g, c in grid_counts.items() if c >= density_threshold]
            
            for g in dense_grids:
                mask = np.all(grid_indices == g, axis=1)
                labels[mask] = cluster_id
                cluster_id += 1
        
        # Results
        df_result = df_numeric.copy().reset_index(drop=True)
        df_result["cluster"] = labels
        
        st.session_state['df_result'] = df_result
        st.session_state['X_scaled'] = X_scaled
        st.session_state['selected_features'] = selected_columns
        st.session_state['scaler'] = scaler
        
        # Cluster Distribution
        st.markdown('<div class="section-header">Clustering Results</div>', unsafe_allow_html=True)
        
        unique_labels = np.unique(labels)
        cluster_counts = pd.Series(labels).value_counts().sort_index()
        
        col_viz1, col_viz2 = st.columns([2.5, 1])
        
        with col_viz1:
            # Set style
            sns.set_style("whitegrid")
            fig_dist, ax_dist = plt.subplots(figsize=(10, 5), facecolor='white')
            
            # Color palette
            palette = ['#0ea5e9', '#8b5cf6', '#ec4899', '#f59e0b', '#10b981', '#ef4444']
            bar_colors = [palette[i % len(palette)] for i in range(len(cluster_counts))]
            
            bars = ax_dist.bar(range(len(cluster_counts)), cluster_counts.values, 
                              color=bar_colors, edgecolor='none', alpha=0.85, width=0.7)
            
            ax_dist.set_xlabel('Cluster', fontsize=10, color='#475569', fontweight='500')
            ax_dist.set_ylabel('Count', fontsize=10, color='#475569', fontweight='500')
            ax_dist.set_title('Distribution of Data Points per Cluster', 
                            fontsize=12, color='#1e293b', fontweight='600', pad=15)
            
            ax_dist.set_xticks(range(len(cluster_counts)))
            ax_dist.set_xticklabels([f'C{i}' if i != -1 else 'Noise' for i in cluster_counts.index], 
                                    fontsize=9, color='#64748b')
            
            ax_dist.spines['top'].set_visible(False)
            ax_dist.spines['right'].set_visible(False)
            ax_dist.spines['left'].set_color('#cbd5e1')
            ax_dist.spines['bottom'].set_color('#cbd5e1')
            ax_dist.tick_params(colors='#94a3b8', which='both')
            ax_dist.grid(axis='y', alpha=0.15, linestyle='-', linewidth=0.8)
            ax_dist.set_axisbelow(True)
            
            for bar in bars:
                height = bar.get_height()
                ax_dist.text(bar.get_x() + bar.get_width()/2., height,
                           f'{int(height)}', ha='center', va='bottom', 
                           fontweight='600', fontsize=9, color='#475569')
            
            plt.tight_layout()
            st.pyplot(fig_dist)
            plt.close(fig_dist)
        
        with col_viz2:
            st.markdown("**Summary Statistics**")
            for label in unique_labels:
                count = np.sum(labels == label)
                pct = (count / len(labels)) * 100
                label_text = "Noise" if label == -1 else f"Cluster {label}"
                st.metric(label_text, f"{count:,}", f"{pct:.1f}%")
        
        with st.expander("Clustered Dataset"):
            st.dataframe(df_result.head(25), use_container_width=True)
            csv_bytes = df_result.to_csv(index=False).encode('utf-8')
            st.download_button("Export to CSV", csv_bytes, "clustered_data.csv", "text/csv")
        
        # Evaluation
        non_noise = [l for l in unique_labels if l != -1]
        
        if len(non_noise) > 1:
            st.markdown('<div class="section-header">Model Evaluation</div>', unsafe_allow_html=True)
            
            eval_col1, eval_col2, eval_col3 = st.columns(3)
            
            try:
                sil = silhouette_score(X_scaled, labels)
                eval_col1.metric("Silhouette Score", f"{sil:.4f}", 
                               help="Range: -1 to 1. Higher values indicate better-defined clusters.")
            except:
                eval_col1.metric("Silhouette Score", "N/A")
            
            try:
                dbi = davies_bouldin_score(X_scaled, labels)
                eval_col2.metric("Davies-Bouldin", f"{dbi:.4f}",
                               help="Lower values indicate better cluster separation.")
            except:
                eval_col2.metric("Davies-Bouldin", "N/A")
            
            try:
                chi = calinski_harabasz_score(X_scaled, labels)
                eval_col3.metric("Calinski-Harabasz", f"{chi:.2f}",
                               help="Higher values indicate better-defined clusters.")
            except:
                eval_col3.metric("Calinski-Harabasz", "N/A")
        
        # PCA Visualization
        st.markdown('<div class="section-header">Cluster Visualization</div>', unsafe_allow_html=True)
        
        n_components = min(2, X_scaled.shape[1], X_scaled.shape[0])
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_scaled)
        
        fig, ax = plt.subplots(figsize=(12, 6.5), facecolor='white')
        
        palette_scatter = ['#0ea5e9', '#8b5cf6', '#ec4899', '#f59e0b', '#10b981', '#ef4444']
        
        for idx, label in enumerate(unique_labels):
            mask = labels == label
            color = palette_scatter[idx % len(palette_scatter)] if label != -1 else '#94a3b8'
            label_name = f"Cluster {label}" if label != -1 else "Noise"
            
            if n_components == 1:
                ax.scatter(X_pca[mask, 0], np.zeros_like(X_pca[mask, 0]), 
                          c=color, label=label_name, alpha=0.7, s=60, edgecolors='white', linewidth=0.5)
            else:
                ax.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                          c=color, label=label_name, alpha=0.7, s=60, edgecolors='white', linewidth=0.5)
        
        ax.set_xlabel("Principal Component 1", fontsize=10, color='#475569', fontweight='500')
        if n_components > 1:
            ax.set_ylabel("Principal Component 2", fontsize=10, color='#475569', fontweight='500')
        ax.set_title(f"PCA Projection of {algo} Clustering", 
                    fontsize=12, color='#1e293b', fontweight='600', pad=15)
        
        ax.legend(frameon=True, fancybox=False, shadow=False, fontsize=9, 
                 loc='best', framealpha=0.95, edgecolor='#e2e8f0')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#cbd5e1')
        ax.spines['bottom'].set_color('#cbd5e1')
        ax.tick_params(colors='#94a3b8')
        ax.grid(True, alpha=0.15, linestyle='-', linewidth=0.8)
        ax.set_axisbelow(True)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        
        # Geographic Visualization
        if has_geo:
            st.markdown('<div class="section-header">Geographic Distribution</div>', unsafe_allow_html=True)
            
            df_with_clusters = df.copy()
            df_with_clusters['cluster'] = labels
            
            tab1, tab2 = st.tabs(["Cluster Map", "Density Heatmap"])
            
            with tab1:
                cluster_map = create_cluster_map(
                    df_with_clusters,
                    geo_info['lat'],
                    geo_info['lon'],
                    'cluster',
                    geo_info['location']
                )
                if cluster_map:
                    folium_static(cluster_map, width=1200, height=600)
            
            with tab2:
                heatmap = create_heatmap(df_with_clusters, geo_info['lat'], geo_info['lon'])
                if heatmap:
                    folium_static(heatmap, width=1200, height=600)
    
    else:
        # Welcome Screen
        st.markdown("""
        <div class="welcome-box">
            <h1 class="welcome-title">Data Clustering Analysis Platform</h1>
            <p class="welcome-subtitle">Upload your dataset via sidebar to begin analysis</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('<div class="features-section">', unsafe_allow_html=True)
        st.markdown('<h3 class="features-title">Key Features</h3>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div class="feature-box">
                <div class="feature-emoji">📊</div>
                <h4 class="feature-name" style="color: #6366f1;">Multiple Algorithms</h4>
                <p class="feature-text">K-Means, DBSCAN, Hierarchical, Mean Shift, and Grid-Based methods</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="feature-box">
                <div class="feature-emoji">📈</div>
                <h4 class="feature-name" style="color: #8b5cf6;">Model Evaluation</h4>
                <p class="feature-text">Silhouette Score, Davies-Bouldin Index, and Calinski-Harabasz metrics</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="feature-box">
                <div class="feature-emoji">🗺️</div>
                <h4 class="feature-name" style="color: #a855f7;">Geographic Maps</h4>
                <p class="feature-text">Interactive maps and heatmaps with automatic coordinate detection</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown("""
            <div class="feature-box">
                <div class="feature-emoji">💾</div>
                <h4 class="feature-name" style="color: #ec4899;">Export Results</h4>
                <p class="feature-text">Download clustered data in CSV format for further analysis</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    show()