import streamlit as st

def show():
    st.title("📊 Dashboard Clustering")
    st.markdown("Platform analisis data dengan berbagai algoritma clustering")
    st.divider()
    
    # About Section
    st.subheader("Tentang Aplikasi")
    st.info("""
    Dashboard Analisis Clustering adalah aplikasi web interaktif untuk melakukan analisis clustering 
    pada data dengan mudah dan cepat. Aplikasi ini mendukung berbagai algoritma clustering yang 
    dapat disesuaikan dengan kebutuhan analisis Anda.
    """)
    
    # Features Section
    st.subheader("Fitur Utama")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**📁 Upload Data Fleksibel**")
        st.caption("Upload file CSV dengan berbagai format delimiter. Sistem akan otomatis mendeteksi format file Anda.")
        
        st.write("")
        
        st.markdown("**⬇️ Export Hasil**")
        st.caption("Download hasil clustering dalam format CSV untuk analisis lebih lanjut atau integrasi dengan tools lain.")
    
    with col2:
        st.markdown("**🤖 5 Algoritma Clustering**")
        st.caption("Pilih dari KMeans, DBSCAN, Hierarchical, Mean Shift, dan Grid-Based sesuai karakteristik data Anda.")
        
        st.write("")
        
        st.markdown("**🔧 Tahapan Detail**")
        st.caption("Pelajari langkah demi langkah setiap algoritma clustering untuk pemahaman yang lebih mendalam.")
    
    with col3:
        st.markdown("**📊 Visualisasi Interaktif**")
        st.caption("Visualisasi hasil clustering dengan PCA 2D yang mudah dipahami dalam bentuk scatter plot berwarna.")
        
        st.write("")
        
        st.markdown("**📈 Evaluasi Lengkap**")
        st.caption("Evaluasi performa clustering dengan 3 metrik: Silhouette Score, Davies-Bouldin Index, dan Calinski-Harabasz Index.")
    
    st.divider()
    
    # Algorithms Section
    st.subheader("Algoritma yang Tersedia")
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["KMeans", "DBSCAN", "Hierarchical", "Mean Shift", "Grid-Based"])
    
    with tab1:
        col_a, col_b = st.columns([2, 1])
        with col_a:
            st.markdown("**KMeans Clustering**")
            st.caption("Jenis: Partitional Clustering berbasis centroid")
            st.write("")
            st.markdown("""
            KMeans adalah algoritma clustering yang membagi data menjadi k cluster berdasarkan 
            jarak ke centroid terdekat. Algoritma ini iteratif dan berusaha meminimalkan 
            within-cluster sum of squares.
            
            **Cara Kerja:**
            - Inisialisasi k centroid secara random
            - Assign setiap point ke centroid terdekat
            - Update centroid sebagai mean dari points
            - Ulangi sampai konvergen
            """)
        with col_b:
            st.info("**Parameter:**\n- k: Jumlah cluster\n- max_iter: Iterasi maks\n- tol: Threshold konvergensi")
            st.success("**Best For:**\n- Dataset besar\n- Cluster spherical\n- Cluster seimbang")
    
    with tab2:
        col_a, col_b = st.columns([2, 1])
        with col_a:
            st.markdown("**DBSCAN**")
            st.caption("Jenis: Density-Based Spatial Clustering")
            st.write("")
            st.markdown("""
            DBSCAN mengelompokkan points yang berdekatan dengan density tinggi dan menandai 
            points di area low-density sebagai outlier. Tidak perlu menentukan jumlah cluster di awal.
            
            **Cara Kerja:**
            - Identifikasi core points (≥ MinPts dalam radius EPS)
            - Hubungkan core points yang berdekatan
            - Assign border points ke cluster terdekat
            - Mark sisanya sebagai noise
            """)
        with col_b:
            st.info("**Parameter:**\n- EPS: Radius neighborhood\n- MinPts: Min points dalam EPS")
            st.success("**Best For:**\n- Cluster arbitrary shape\n- Data dengan noise\n- Varying density")
    
    with tab3:
        col_a, col_b = st.columns([2, 1])
        with col_a:
            st.markdown("**Hierarchical (Agglomerative)**")
            st.caption("Jenis: Hierarchical Clustering bottom-up")
            st.write("")
            st.markdown("""
            Hierarchical clustering membangun hierarki cluster dengan menggabungkan cluster 
            terdekat secara iteratif. Menghasilkan dendrogram yang bisa di-cut di level manapun.
            
            **Cara Kerja:**
            - Mulai dengan n cluster (setiap point = 1 cluster)
            - Gabungkan 2 cluster terdekat
            - Update distance matrix
            - Ulangi sampai 1 cluster
            - Cut dendrogram untuk mendapat k cluster
            """)
        with col_b:
            st.info("**Parameter:**\n- n_clusters: Jumlah cluster\n- linkage: Metode jarak")
            st.success("**Best For:**\n- Dendogram visualization\n- Small-medium dataset\n- Flexible clustering")
    
    with tab4:
        col_a, col_b = st.columns([2, 1])
        with col_a:
            st.markdown("**Mean Shift**")
            st.caption("Jenis: Mode-Seeking Clustering berbasis density")
            st.write("")
            st.markdown("""
            Mean Shift mencari mode (peak) dalam density distribution tanpa perlu menentukan 
            jumlah cluster. Setiap point bergerak ke arah density tertinggi sampai konvergen.
            
            **Cara Kerja:**
            - Setiap point menjadi kandidat centroid
            - Hitung mean shift vector untuk setiap point
            - Geser point ke arah mean
            - Ulangi sampai konvergen
            - Merge points yang konvergen ke mode sama
            """)
        with col_b:
            st.info("**Parameter:**\n- bandwidth: Radius kernel\n- kernel: Kernel function")
            st.success("**Best For:**\n- Unknown jumlah cluster\n- Cluster non-convex\n- Robust to outliers")
    
    with tab5:
        col_a, col_b = st.columns([2, 1])
        with col_a:
            st.markdown("**Grid-Based Clustering**")
            st.caption("Jenis: Grid-Based Clustering berbasis density")
            st.write("")
            st.markdown("""
            Grid-Based membagi ruang data menjadi grid cells dan mengelompokkan grid yang padat. 
            Sangat cepat karena complexity tidak bergantung pada jumlah data points.
            
            **Cara Kerja:**
            - Bagi ruang menjadi grid cells
            - Assign points ke grid berdasarkan koordinat
            - Hitung density setiap grid
            - Identifikasi dense grids
            - Gabungkan adjacent dense grids jadi cluster
            """)
        with col_b:
            st.info("**Parameter:**\n- n_bins: Jumlah grid\n- density_threshold: Min points")
            st.success("**Best For:**\n- Very large datasets\n- Fast clustering needed\n- Multi-dimensional data")
    
    st.divider()
    
    # Evaluation Metrics
    st.subheader("Metrik Evaluasi Clustering")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Silhouette Score**")
        st.caption("Range: -1 hingga 1")
        st.write("""
        Mengukur seberapa mirip objek dengan cluster sendiri dibanding cluster lain. 
        Nilai tinggi menunjukkan cluster yang terpisah dengan baik.
        """)
        st.success("Semakin tinggi semakin baik")
    
    with col2:
        st.markdown("**Davies-Bouldin Index**")
        st.caption("Range: 0 hingga ∞")
        st.write("""
        Mengukur rata-rata similarity antara setiap cluster dengan cluster yang paling mirip. 
        Nilai rendah menunjukkan cluster yang terpisah baik.
        """)
        st.success("Semakin rendah semakin baik")
    
    with col3:
        st.markdown("**Calinski-Harabasz Index**")
        st.caption("Range: 0 hingga ∞")
        st.write("""
        Mengukur rasio between-cluster dispersion vs within-cluster dispersion. 
        Nilai tinggi menunjukkan cluster yang kompak dan terpisah.
        """)
        st.success("Semakin tinggi semakin baik")
    
    st.divider()
    
    # Usage Guide
    st.subheader("Cara Penggunaan")
    
    steps = [
        ("Upload Data", "Klik 'Browse files' di sidebar dan pilih file CSV Anda. Sistem akan otomatis mendeteksi delimiter."),
        ("Pilih Fitur", "Pilih kolom numerik yang akan digunakan untuk clustering dari dropdown menu."),
        ("Pilih Algoritma & Atur Parameter", "Pilih algoritma clustering dan sesuaikan parameter sesuai kebutuhan menggunakan slider."),
        ("Analisis Hasil", "Lihat visualisasi, statistik cluster, dan metrik evaluasi untuk memahami hasil clustering."),
        ("Download Hasil", "Klik tombol download untuk menyimpan hasil clustering dalam format CSV.")
    ]
    
    for i, (title, desc) in enumerate(steps, 1):
        with st.container():
            col_num, col_text = st.columns([0.5, 9.5])
            with col_num:
                st.markdown(f"**{i}**")
            with col_text:
                st.markdown(f"**{title}**")
                st.caption(desc)
            st.write("")
    
    st.divider()
    
    # Data Format
    st.subheader("Format Data yang Didukung")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Tipe File**")
        st.markdown("""
        - **Format**: CSV (Comma-Separated Values)
        - **Delimiter**: Koma (,), Titik-koma (;), atau Spasi ( )
        - **Encoding**: UTF-8, ASCII
        """)
    
    with col2:
        st.markdown("**Persyaratan Data**")
        st.markdown("""
        - Minimal 1 kolom numerik
        - Data non-numerik akan diabaikan
        - Missing values akan di-impute dengan mean
        - Data otomatis di-standardisasi (Z-score)
        """)
    
    st.divider()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; padding: 20px; color: #666;'>
        <h4>Dashboard Clustering Analytics</h4>
        <p>Powered by Streamlit, Scikit-learn, Pandas & Matplotlib</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    show()