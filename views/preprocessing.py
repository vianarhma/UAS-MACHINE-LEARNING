import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns

from utils import try_read_csv, convert_columns_to_numeric

def show():
    st.title("🔧 Data Preprocessing")
    st.markdown("Persiapan dan transformasi data untuk analisis clustering")
    st.divider()
    
    df = None
    
    # File Upload Section
    st.subheader("Upload Dataset")
    
    if 'preprocess_df' in st.session_state:
        df = st.session_state['preprocess_df']
        st.success("Data berhasil dimuat")
        if st.button("Upload file berbeda"):
            del st.session_state['preprocess_df']
            st.rerun()
    else:
        uploaded_file = st.file_uploader("Pilih file CSV", type=["csv"])
        if uploaded_file:
            try:
                df = try_read_csv(uploaded_file)
                df.columns = df.columns.str.strip()
                st.session_state['preprocess_df'] = df
                st.success("File berhasil diupload")
                st.rerun()
            except Exception as e:
                st.error(f"Error membaca file: {e}")
    
    if df is not None:
        # Data Overview
        st.divider()
        st.subheader("Ringkasan Data")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Baris", f"{len(df):,}")
        col2.metric("Total Kolom", df.shape[1])
        col3.metric("Missing Values", df.isnull().sum().sum())
        col4.metric("Duplikat", df.duplicated().sum())
        
        with st.expander("Lihat preview data"):
            st.dataframe(df.head(10), use_container_width=True)
        
        with st.expander("Informasi kolom"):
            col_info = pd.DataFrame({
                'Kolom': df.columns,
                'Tipe Data': df.dtypes.values,
                'Non-Null': df.count().values,
                'Missing': df.isnull().sum().values,
                'Unique': [df[col].nunique() for col in df.columns]
            })
            st.dataframe(col_info, use_container_width=True)
        
        st.divider()
        
        # Missing Values Handling
        st.subheader("Penanganan Missing Values")
        
        missing_cols = df.columns[df.isnull().any()].tolist()
        
        if len(missing_cols) > 0:
            st.warning(f"Ditemukan {len(missing_cols)} kolom dengan missing values")
            
            fig, ax = plt.subplots(figsize=(10, 4))
            missing_data = df.isnull().sum()[df.isnull().sum() > 0].sort_values(ascending=False)
            ax.bar(range(len(missing_data)), missing_data.values, color='steelblue', edgecolor='black')
            ax.set_xticks(range(len(missing_data)))
            ax.set_xticklabels(missing_data.index, rotation=45, ha='right')
            ax.set_ylabel('Jumlah Missing')
            ax.set_title('Missing Values per Kolom')
            ax.grid(axis='y', alpha=0.3)
            st.pyplot(fig)
            plt.close(fig)
            
            handle_method = st.selectbox(
                "Metode penanganan:",
                ["Impute dengan Mean", "Impute dengan Median", "Impute dengan Mode", "Hapus Baris"]
            )
            
            if st.button("Terapkan", key="missing"):
                df_clean = df.copy()
                
                if handle_method == "Hapus Baris":
                    df_clean = df_clean.dropna()
                    st.success(f"Berhasil menghapus {len(df) - len(df_clean)} baris")
                else:
                    df_numeric = convert_columns_to_numeric(df_clean)
                    
                    if handle_method == "Impute dengan Mean":
                        imputer = SimpleImputer(strategy='mean')
                    elif handle_method == "Impute dengan Median":
                        imputer = SimpleImputer(strategy='median')
                    else:
                        imputer = SimpleImputer(strategy='most_frequent')
                    
                    numeric_cols = df_numeric.columns.tolist()
                    if len(numeric_cols) > 0:
                        df_clean[numeric_cols] = imputer.fit_transform(df_clean[numeric_cols])
                        st.success(f"Missing values berhasil diisi pada {len(numeric_cols)} kolom")
                
                st.session_state['preprocess_df'] = df_clean
                st.rerun()
        else:
            st.success("Tidak ada missing values")
        
        st.divider()
        
        # Duplicates Handling
        st.subheader("Penanganan Duplikat")
        
        duplicates = df.duplicated().sum()
        
        if duplicates > 0:
            st.warning(f"Ditemukan {duplicates} baris duplikat")
            
            if st.button("Hapus Duplikat"):
                df_clean = df.drop_duplicates()
                st.success(f"Berhasil menghapus {duplicates} baris duplikat")
                st.session_state['preprocess_df'] = df_clean
                st.rerun()
        else:
            st.success("Tidak ada duplikat")
        
        st.divider()
        
        # Data Scaling
        st.subheader("Scaling & Normalisasi")
        
        df_numeric = convert_columns_to_numeric(df)
        
        if df_numeric.shape[1] > 0:
            st.info(f"Terdeteksi {df_numeric.shape[1]} kolom numerik")
            
            scale_cols = st.multiselect(
                "Pilih kolom untuk di-scale:",
                df_numeric.columns.tolist(),
                default=df_numeric.columns.tolist()[:3] if len(df_numeric.columns) >= 3 else df_numeric.columns.tolist()
            )
            
            if len(scale_cols) > 0:
                scaling_method = st.selectbox(
                    "Metode scaling:",
                    ["StandardScaler (Z-score)", "MinMaxScaler (0-1)", "RobustScaler (Median & IQR)"]
                )
                
                col_viz1, col_viz2 = st.columns(2)
                
                with col_viz1:
                    st.markdown("**Before Scaling**")
                    fig, ax = plt.subplots(figsize=(8, 4))
                    df_numeric[scale_cols].boxplot(ax=ax)
                    ax.set_ylabel('Value')
                    ax.grid(axis='y', alpha=0.3)
                    plt.xticks(rotation=45, ha='right')
                    st.pyplot(fig)
                    plt.close(fig)
                
                if scaling_method == "StandardScaler (Z-score)":
                    scaler = StandardScaler()
                elif scaling_method == "MinMaxScaler (0-1)":
                    scaler = MinMaxScaler()
                else:
                    scaler = RobustScaler()
                
                df_scaled = df.copy()
                df_scaled[scale_cols] = scaler.fit_transform(df_numeric[scale_cols])
                
                with col_viz2:
                    st.markdown("**After Scaling**")
                    fig, ax = plt.subplots(figsize=(8, 4))
                    df_scaled[scale_cols].boxplot(ax=ax)
                    ax.set_ylabel('Value')
                    ax.grid(axis='y', alpha=0.3)
                    plt.xticks(rotation=45, ha='right')
                    st.pyplot(fig)
                    plt.close(fig)
                
                with st.expander("Lihat statistik"):
                    col_stat1, col_stat2 = st.columns(2)
                    
                    with col_stat1:
                        st.markdown("**Original**")
                        st.dataframe(df_numeric[scale_cols].describe())
                    
                    with col_stat2:
                        st.markdown("**After Scaling**")
                        st.dataframe(df_scaled[scale_cols].describe())
                
                if st.button("Terapkan Scaling", key="scaling"):
                    st.session_state['preprocess_df'] = df_scaled
                    st.session_state['scaled_columns'] = scale_cols
                    st.session_state['scaler_method'] = scaling_method
                    st.success(f"Scaling berhasil diterapkan pada {len(scale_cols)} kolom")
                    st.rerun()
            else:
                st.warning("Pilih minimal 1 kolom untuk di-scale")
        else:
            st.error("Tidak ada kolom numerik")
        
        st.divider()
        
        # Correlation Matrix
        st.subheader("Correlation Matrix")
        
        if df_numeric.shape[1] > 1:
            corr_matrix = df_numeric.corr()
            
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                       square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
            ax.set_title('Correlation Matrix')
            st.pyplot(fig)
            plt.close(fig)
            
            st.markdown("**Korelasi Tinggi (> 0.7)**")
            
            corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    if abs(corr_val) > 0.7:
                        corr_pairs.append({
                            'Kolom 1': corr_matrix.columns[i],
                            'Kolom 2': corr_matrix.columns[j],
                            'Correlation': round(corr_val, 3)
                        })
            
            if len(corr_pairs) > 0:
                df_corr = pd.DataFrame(corr_pairs)
                st.dataframe(df_corr, use_container_width=True)
                st.info("Korelasi tinggi dapat menyebabkan multicollinearity")
            else:
                st.success("Tidak ada korelasi tinggi antar fitur")
        else:
            st.warning("Minimal 2 kolom numerik diperlukan")
        
        st.divider()
        
        # Download Section
        st.subheader("Download Hasil")
        
        csv_bytes = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Preprocessed Data (CSV)",
            data=csv_bytes,
            file_name="preprocessed_data.csv",
            mime="text/csv",
            use_container_width=True
        )
        
        # Summary
        with st.expander("Lihat ringkasan preprocessing"):
            col_sum1, col_sum2 = st.columns(2)
            
            with col_sum1:
                st.markdown("**Status Preprocessing:**")
                st.write("✓ Data loading")
                st.write("✓ Missing values check")
                st.write("✓ Duplicate detection")
                st.write("✓ Data scaling")
                st.write("✓ Correlation analysis")
            
            with col_sum2:
                st.markdown("**Data Final:**")
                st.write(f"Baris: {len(df):,}")
                st.write(f"Kolom: {df.shape[1]}")
                st.write(f"Kolom Numerik: {df_numeric.shape[1]}")
                st.write(f"Missing: {df.isnull().sum().sum()}")
                st.write(f"Duplikat: {df.duplicated().sum()}")
        
        st.success("Data siap untuk clustering. Lanjut ke halaman Dashboard.")
    
    else:
        # Welcome Section
        st.info("Upload file CSV untuk memulai preprocessing")
        
        st.markdown("### Panduan Preprocessing")
        
        tab1, tab2 = st.tabs(["Missing Values", "Scaling"])
        
        with tab1:
            st.markdown("""
            **Metode Handling Missing Values:**
            
            - **Mean**: Cocok untuk data dengan distribusi normal
            - **Median**: Lebih robust terhadap outlier
            - **Mode**: Untuk data kategorikal
            - **Hapus Baris**: Jika missing values sedikit
            """)
        
        with tab2:
            st.markdown("""
            **Metode Scaling:**
            
            - **StandardScaler**: Normalisasi menggunakan mean dan standard deviation (data normal)
            - **MinMaxScaler**: Scale ke range 0-1 (data bounded)
            - **RobustScaler**: Menggunakan median dan IQR (data dengan outlier)
            
            Scaling penting untuk algoritma berbasis distance seperti K-Means dan DBSCAN.
            """)

if __name__ == "__main__":
    show()