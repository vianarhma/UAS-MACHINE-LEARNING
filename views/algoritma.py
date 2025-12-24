import streamlit as st

def show():
    st.title("📖 Tahapan Algoritma Clustering")
    st.markdown("Memahami cara kerja setiap algoritma clustering step-by-step")
    st.divider()
    
    # Algorithm Selection
    algo_choice = st.selectbox(
        "Pilih algoritma untuk dipelajari:",
        ["KMeans", "DBSCAN", "Hierarchical (Agglomerative)", "Mean Shift", "Grid-Based"]
    )
    
    st.divider()
    
    # === KMEANS ===
    if algo_choice == "KMeans":
        st.subheader("KMeans Clustering")
        
        col_info1, col_info2, col_info3 = st.columns(3)
        col_info1.metric("Tipe", "Partitional")
        col_info2.metric("Kompleksitas", "O(n·k·i·d)")
        col_info3.metric("Best For", "Spherical clusters")
        
        st.write("")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("**Langkah-langkah:**")
            
            with st.container():
                st.markdown("**1. Inisialisasi Centroid**")
                st.caption("Pilih k titik secara random dari dataset sebagai centroid awal. Bisa menggunakan metode KMeans++ untuk inisialisasi yang lebih baik.")
            
            st.write("")
            
            with st.container():
                st.markdown("**2. Assignment Step**")
                st.caption("Setiap data point diassign ke centroid terdekat berdasarkan jarak Euclidean:")
                st.code("distance = √(Σ(xi - ci)²)", language="python")
            
            st.write("")
            
            with st.container():
                st.markdown("**3. Update Centroid**")
                st.caption("Hitung centroid baru sebagai rata-rata dari semua data points dalam setiap cluster:")
                st.code("centroid_new = Σ(points) / n_points", language="python")
            
            st.write("")
            
            with st.container():
                st.markdown("**4. Convergence Check**")
                st.caption("Ulangi step 2-3 sampai centroid tidak berubah signifikan (< tolerance) atau mencapai iterasi maksimum.")
            
            st.write("")
            
            with st.container():
                st.markdown("**5. Final Clustering**")
                st.caption("Setiap data point mendapat label cluster final berdasarkan centroid terdekat.")
        
        with col2:
            with st.container():
                st.markdown("**Parameter Penting**")
                st.markdown("""
                - **k**: Jumlah cluster
                - **max_iter**: Iterasi maksimum
                - **tol**: Threshold konvergensi
                - **init**: Metode inisialisasi
                """)
            
            st.success("**Kelebihan:**\n- Cepat & scalable\n- Mudah diimplementasi\n- Efisien untuk big data")
            st.warning("**Kekurangan:**\n- Perlu tentukan k\n- Sensitif terhadap outlier\n- Hanya untuk cluster spherical")
    
    # === DBSCAN ===
    elif algo_choice == "DBSCAN":
        st.subheader("DBSCAN (Density-Based Spatial Clustering)")
        
        col_info1, col_info2, col_info3 = st.columns(3)
        col_info1.metric("Tipe", "Density-Based")
        col_info2.metric("Kompleksitas", "O(n·log n)")
        col_info3.metric("Best For", "Arbitrary shape")
        
        st.write("")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("**Langkah-langkah:**")
            
            with st.container():
                st.markdown("**1. Tentukan Parameter**")
                st.caption("Set nilai EPS (radius neighborhood) dan MinPts (minimum points untuk menjadi core point).")
            
            st.write("")
            
            with st.container():
                st.markdown("**2. Identifikasi Core Points**")
                st.caption("Point p adalah core point jika dalam radius EPS terdapat minimal MinPts points (termasuk p sendiri).")
            
            st.write("")
            
            with st.container():
                st.markdown("**3. Form Clusters**")
                st.caption("Hubungkan core points yang saling berada dalam radius EPS. Semua core points yang terhubung membentuk 1 cluster.")
            
            st.write("")
            
            with st.container():
                st.markdown("**4. Assign Border Points**")
                st.caption("Points yang bukan core point tapi berada dalam EPS dari core point dijadikan border point dan masuk ke cluster tersebut.")
            
            st.write("")
            
            with st.container():
                st.markdown("**5. Noise Detection**")
                st.caption("Points yang bukan core point dan tidak dalam radius EPS dari core point manapun diberi label -1 (noise/outlier).")
        
        with col2:
            with st.container():
                st.markdown("**Parameter Penting**")
                st.markdown("""
                - **EPS**: Radius neighborhood
                - **MinPts**: Min points dalam EPS
                - **metric**: Distance metric
                """)
            
            st.success("**Kelebihan:**\n- Deteksi noise/outlier\n- Cluster shape arbitrary\n- Tidak perlu tentukan k")
            st.warning("**Kekurangan:**\n- Sensitif terhadap parameter\n- Sulit untuk varying density\n- Kurang bagus untuk high-dim")
    
    # === HIERARCHICAL ===
    elif algo_choice == "Hierarchical (Agglomerative)":
        st.subheader("Hierarchical Agglomerative Clustering")
        
        col_info1, col_info2, col_info3 = st.columns(3)
        col_info1.metric("Tipe", "Hierarchical")
        col_info2.metric("Kompleksitas", "O(n³)")
        col_info3.metric("Best For", "Dendrogram viz")
        
        st.write("")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("**Langkah-langkah:**")
            
            with st.container():
                st.markdown("**1. Initialization**")
                st.caption("Setiap data point dimulai sebagai cluster tersendiri (n clusters untuk n data points).")
            
            st.write("")
            
            with st.container():
                st.markdown("**2. Compute Distance Matrix**")
                st.caption("Hitung jarak antara setiap pasangan cluster menggunakan linkage criterion yang dipilih (single, complete, average, atau ward).")
            
            st.write("")
            
            with st.container():
                st.markdown("**3. Merge Closest Clusters**")
                st.caption("Gabungkan 2 cluster terdekat menjadi 1 cluster baru. Linkage methods:")
                st.markdown("""
                - **Single**: min distance antar points
                - **Complete**: max distance antar points
                - **Average**: rata-rata distance
                - **Ward**: minimize variance
                """)
            
            st.write("")
            
            with st.container():
                st.markdown("**4. Update Distance Matrix**")
                st.caption("Update distance matrix dengan menghitung jarak cluster baru terhadap cluster lainnya.")
            
            st.write("")
            
            with st.container():
                st.markdown("**5. Repeat & Cut Dendrogram**")
                st.caption("Ulangi step 2-4 sampai semua points dalam 1 cluster. Kemudian cut dendrogram di level tertentu untuk mendapat k clusters.")
        
        with col2:
            with st.container():
                st.markdown("**Parameter Penting**")
                st.markdown("""
                - **n_clusters**: Jumlah cluster akhir
                - **linkage**: Metode perhitungan jarak
                - **distance_threshold**: Threshold cut
                """)
            
            st.success("**Kelebihan:**\n- Menghasilkan dendrogram\n- Tidak perlu k di awal\n- Deterministik")
            st.warning("**Kekurangan:**\n- Kompleksitas tinggi O(n³)\n- Tidak scalable\n- Merge tidak bisa di-undo")
    
    # === MEAN SHIFT ===
    elif algo_choice == "Mean Shift":
        st.subheader("Mean Shift Clustering")
        
        col_info1, col_info2, col_info3 = st.columns(3)
        col_info1.metric("Tipe", "Mode-Seeking")
        col_info2.metric("Kompleksitas", "O(n²)")
        col_info3.metric("Best For", "Auto discovery")
        
        st.write("")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("**Langkah-langkah:**")
            
            with st.container():
                st.markdown("**1. Initialize**")
                st.caption("Setiap data point menjadi kandidat centroid. Set bandwidth (radius untuk kernel density estimation).")
            
            st.write("")
            
            with st.container():
                st.markdown("**2. Compute Mean Shift Vector**")
                st.caption("Untuk setiap point, hitung mean shift vector menuju rata-rata weighted points dalam bandwidth:")
                st.code("m(x) = (Σ K(xi - x) · xi) / Σ K(xi - x)", language="python")
            
            st.write("")
            
            with st.container():
                st.markdown("**3. Shift Points**")
                st.caption("Geser setiap point ke arah mean shift vector (menuju density mode/peak). Points akan bergerak menuju area dengan density tertinggi.")
            
            st.write("")
            
            with st.container():
                st.markdown("**4. Convergence Check**")
                st.caption("Ulangi step 2-3 sampai points konvergen (shift vector mendekati 0 atau perubahan sangat kecil).")
            
            st.write("")
            
            with st.container():
                st.markdown("**5. Merge Close Modes**")
                st.caption("Points yang konvergen ke mode yang sama (dalam bandwidth) dikelompokkan sebagai 1 cluster. Jumlah cluster ditentukan otomatis!")
        
        with col2:
            with st.container():
                st.markdown("**Parameter Penting**")
                st.markdown("""
                - **bandwidth**: Radius kernel
                - **kernel**: Kernel function
                - **bin_seeding**: Speed optimization
                """)
            
            st.success("**Kelebihan:**\n- Otomatis tentukan cluster\n- Cluster shape arbitrary\n- Robust terhadap outlier")
            st.warning("**Kekurangan:**\n- Lambat untuk data besar\n- Sensitif terhadap bandwidth\n- Mahal secara komputasi")
    
    # === GRID-BASED ===
    elif algo_choice == "Grid-Based":
        st.subheader("Grid-Based Clustering")
        
        col_info1, col_info2, col_info3 = st.columns(3)
        col_info1.metric("Tipe", "Grid-Based")
        col_info2.metric("Kompleksitas", "O(n)")
        col_info3.metric("Best For", "Large datasets")
        
        st.write("")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("**Langkah-langkah:**")
            
            with st.container():
                st.markdown("**1. Create Grid Structure**")
                st.caption("Bagi ruang data menjadi grid cells dengan ukuran tetap. Jumlah grid = n_bins per dimensi.")
            
            st.write("")
            
            with st.container():
                st.markdown("**2. Assign Points to Grid**")
                st.caption("Setiap data point diassign ke grid cell berdasarkan koordinatnya:")
                st.code("grid_index = floor((point - min) / grid_size)", language="python")
            
            st.write("")
            
            with st.container():
                st.markdown("**3. Compute Grid Density**")
                st.caption("Hitung jumlah points dalam setiap grid cell sebagai measure of density. Grid dengan banyak points = high density.")
            
            st.write("")
            
            with st.container():
                st.markdown("**4. Identify Dense Grids**")
                st.caption("Grid cells dengan density ≥ threshold dianggap sebagai dense grids (potential clusters). Grid dengan sedikit points diabaikan.")
            
            st.write("")
            
            with st.container():
                st.markdown("**5. Form Clusters**")
                st.caption("Gabungkan adjacent dense grids menjadi clusters. Points di non-dense grids diberi label -1 (noise).")
        
        with col2:
            with st.container():
                st.markdown("**Parameter Penting**")
                st.markdown("""
                - **n_bins**: Jumlah grid per dimensi
                - **density_threshold**: Min points per grid
                - **adjacency**: Definisi tetangga
                """)
            
            st.success("**Kelebihan:**\n- Sangat cepat O(n)\n- Scalable untuk big data\n- Handle multi-dimensional")
            st.warning("**Kekurangan:**\n- Grid size sensitif\n- Tidak bagus untuk sparse data\n- Hasil tergantung alignment")

if __name__ == "__main__":
    show()