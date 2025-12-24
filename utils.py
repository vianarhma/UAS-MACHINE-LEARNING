import pandas as pd
import numpy as np
import folium
from folium.plugins import MarkerCluster, HeatMap
import base64
import io

def try_read_csv(file_like):
    """
    Mencoba membaca CSV dengan berbagai delimiter
    """
    read_attempts = [
        {'sep': ';'}, 
        {'sep': ','}, 
        {'sep': r'\s+'}, 
        {'sep': None, 'engine': 'python'}
    ]
    
    # Reset file pointer jika memungkinkan
    if hasattr(file_like, 'seek'):
        file_like.seek(0)
    
    for params in read_attempts:
        try:
            if hasattr(file_like, 'seek'):
                file_like.seek(0)
            df_try = pd.read_csv(file_like, **params)
            # Validasi minimal 2 kolom
            if df_try.shape[1] <= 1:
                continue
            return df_try
        except Exception:
            continue
    
    # Jika semua gagal
    raise ValueError("Gagal mendeteksi separator CSV. Pastikan file dalam format CSV yang valid.")

def convert_columns_to_numeric(df):
    """
    Konversi kolom menjadi numerik dengan handling berbagai format angka
    """
    numeric_df = pd.DataFrame(index=df.index)
    
    for col in df.columns:
        # Convert to string dan strip whitespace
        s = df[col].astype(str).str.strip()
        
        # Coba konversi langsung
        conv1 = pd.to_numeric(s, errors='coerce')
        
        # Jika banyak NaN, coba strategi alternatif
        if conv1.isna().sum() > 0.6 * len(conv1):
            # Replace koma dengan titik, hapus spasi
            s2 = s.str.replace(',', '.', regex=False)
            s2 = s2.str.replace(' ', '', regex=False)
            # Hapus separator ribuan (titik sebelum 3 digit)
            s2 = s2.str.replace(r'(?<=\d)\.(?=\d{3}\b)', '', regex=True)
            
            conv2 = pd.to_numeric(s2, errors='coerce')
            conv = conv2 if conv2.isna().sum() < conv1.isna().sum() else conv1
        else:
            conv = conv1
        
        # Hanya tambahkan jika ada nilai valid
        if conv.notna().sum() > 0:
            numeric_df[col] = conv
    
    return numeric_df

def detect_geo_columns(df):
    """
    Deteksi kolom geografis (latitude, longitude, lokasi)
    Returns: dict dengan keys 'lat', 'lon', 'location'
    """
    geo_info = {'lat': None, 'lon': None, 'location': None}
    
    # Keywords untuk deteksi
    lat_keywords = ['lat', 'latitude', 'lintang', 'y', 'coord_y', 'koordinat_y']
    lon_keywords = ['lon', 'long', 'longitude', 'bujur', 'x', 'coord_x', 'koordinat_x']
    loc_keywords = ['location', 'lokasi', 'wilayah', 'daerah', 'kota', 'kabupaten', 
                    'provinsi', 'kecamatan', 'kelurahan', 'city', 'region', 'area', 
                    'place', 'tempat', 'nama_wilayah', 'nama_lokasi']
    
    # Mapping kolom ke lowercase
    df_cols_lower = {col.lower(): col for col in df.columns}
    
    # Deteksi Latitude
    for keyword in lat_keywords:
        if geo_info['lat']:
            break
        for col_lower, col_original in df_cols_lower.items():
            if keyword in col_lower:
                try:
                    values = pd.to_numeric(df[col_original], errors='coerce')
                    # Validasi: harus ada nilai valid dan dalam range latitude (-90 to 90)
                    if values.notna().sum() > 0:
                        valid_values = values.dropna()
                        if valid_values.min() >= -90 and valid_values.max() <= 90:
                            geo_info['lat'] = col_original
                            break
                except:
                    pass
    
    # Deteksi Longitude
    for keyword in lon_keywords:
        if geo_info['lon']:
            break
        for col_lower, col_original in df_cols_lower.items():
            if keyword in col_lower:
                try:
                    values = pd.to_numeric(df[col_original], errors='coerce')
                    # Validasi: harus ada nilai valid dan dalam range longitude (-180 to 180)
                    if values.notna().sum() > 0:
                        valid_values = values.dropna()
                        if valid_values.min() >= -180 and valid_values.max() <= 180:
                            geo_info['lon'] = col_original
                            break
                except:
                    pass
    
    # Deteksi kolom lokasi (text-based)
    for keyword in loc_keywords:
        if geo_info['location']:
            break
        for col_lower, col_original in df_cols_lower.items():
            if keyword in col_lower:
                # Cek apakah kolom berisi text (bukan numerik)
                if df[col_original].dtype == 'object' or df[col_original].dtype.name == 'category':
                    # Pastikan ada nilai non-null
                    if df[col_original].notna().sum() > 0:
                        geo_info['location'] = col_original
                        break
    
    return geo_info

def create_cluster_map(df, lat_col, lon_col, cluster_col, location_col=None):
    """
    Membuat peta interaktif dengan marker cluster
    """
    # Persiapan data
    df_map = df[[lat_col, lon_col, cluster_col]].copy()
    if location_col and location_col in df.columns:
        df_map['location_info'] = df[location_col]
    
    # Filter data valid
    df_map = df_map.dropna(subset=[lat_col, lon_col])
    
    if len(df_map) == 0:
        return None
    
    # Hitung center map
    center_lat = df_map[lat_col].mean()
    center_lon = df_map[lon_col].mean()
    
    # Buat base map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=5,
        tiles='OpenStreetMap'
    )
    
    # Color palette untuk cluster
    unique_clusters = sorted(df_map[cluster_col].unique())
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'lightred', 
              'beige', 'darkblue', 'darkgreen', 'cadetblue', 'darkpurple', 'pink', 
              'lightblue', 'lightgreen', 'gray', 'black', 'lightgray']
    cluster_colors = {cluster: colors[i % len(colors)] for i, cluster in enumerate(unique_clusters)}
    
    # Tambahkan marker untuk setiap cluster
    for cluster in unique_clusters:
        cluster_data = df_map[df_map[cluster_col] == cluster]
        
        # Tentukan nama dan warna cluster
        if cluster == -1:
            cluster_name = "Noise/Outlier"
            color = 'gray'
        else:
            cluster_name = f"Cluster {cluster}"
            color = cluster_colors[cluster]
        
        # Buat marker cluster group
        marker_cluster = MarkerCluster(name=cluster_name).add_to(m)
        
        # Tambahkan marker untuk setiap data point
        for idx, row in cluster_data.iterrows():
            # Build popup text
            popup_text = f"<b>{cluster_name}</b><br>"
            popup_text += f"Lat: {row[lat_col]:.4f}<br>"
            popup_text += f"Lon: {row[lon_col]:.4f}<br>"
            
            # Tambahkan info lokasi jika ada
            if location_col and 'location_info' in df_map.columns:
                popup_text += f"Lokasi: {row['location_info']}<br>"
            
            popup_text += f"Index: {idx}"
            
            # Tambahkan marker
            folium.Marker(
                location=[row[lat_col], row[lon_col]],
                popup=folium.Popup(popup_text, max_width=250),
                tooltip=cluster_name,
                icon=folium.Icon(color=color, icon='info-sign')
            ).add_to(marker_cluster)
    
    # Tambahkan layer control
    folium.LayerControl().add_to(m)
    
    return m

def create_heatmap(df, lat_col, lon_col):
    """
    Membuat heatmap density dari data geografis
    """
    # Filter data valid
    df_map = df[[lat_col, lon_col]].dropna()
    
    if len(df_map) == 0:
        return None
    
    # Hitung center map
    center_lat = df_map[lat_col].mean()
    center_lon = df_map[lon_col].mean()
    
    # Buat base map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=5,
        tiles='OpenStreetMap'
    )
    
    # Persiapan data untuk heatmap
    heat_data = [[row[lat_col], row[lon_col]] for idx, row in df_map.iterrows()]
    
    # Tambahkan heatmap layer
    HeatMap(
        heat_data, 
        radius=15,        # Radius setiap point
        blur=25,          # Blur effect
        max_zoom=13,      # Max zoom untuk heatmap
        min_opacity=0.3,  # Minimum opacity
        max_val=1.0       # Maximum value
    ).add_to(m)
    
    return m

def get_geo_statistics(df, lat_col, lon_col, cluster_col):
    """
    Mendapatkan statistik geografis per cluster
    """
    stats = df.groupby(cluster_col).agg({
        lat_col: ['mean', 'min', 'max', 'count'],
        lon_col: ['mean', 'min', 'max']
    }).round(4)
    
    # Rename columns untuk lebih readable
    stats.columns = ['Lat_Mean', 'Lat_Min', 'Lat_Max', 'Count', 'Lon_Mean', 'Lon_Min', 'Lon_Max']
    
    return stats

def analyze_location_clusters(df, location_col, cluster_col):
    """
    Analisis distribusi cluster per lokasi
    Returns: pivot table dan top locations
    """
    if location_col not in df.columns:
        return None, None
    
    # Group by lokasi dan cluster
    location_cluster = df.groupby([location_col, cluster_col]).size().reset_index(name='count')
    
    # Pivot table
    pivot = location_cluster.pivot_table(
        index=location_col,
        columns=cluster_col,
        values='count',
        fill_value=0
    )
    
    # Top locations (berdasarkan total count)
    top_locations = df[location_col].value_counts().head(10)
    
    return pivot, top_locations