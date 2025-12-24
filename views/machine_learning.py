import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, 
    classification_report, 
    confusion_matrix, 
    precision_score, 
    recall_score,
    f1_score
)

def show():
    # Header Section
    st.title("🤖 Machine Learning Model")
    st.markdown("Prediksi cluster menggunakan Random Forest Classifier")
    st.divider()
    
    # Check if clustering data exists
    if 'df_result' not in st.session_state:
        st.warning("Data clustering belum tersedia")
        st.info("Silakan lakukan clustering terlebih dahulu di halaman Dashboard")
        return
    
    # Load data from session
    df_result = st.session_state['df_result']
    X_scaled = st.session_state.get('X_scaled')
    selected_columns = st.session_state.get('selected_features', [])
    
    if X_scaled is None or len(selected_columns) == 0:
        st.error("Data preprocessing tidak ditemukan. Silakan lakukan clustering ulang.")
        return
    
    st.success(f"Data loaded: {len(df_result)} records • {df_result['cluster'].nunique()} clusters")
    
    # Cluster Interpretation Section
    st.subheader("Interpretasi Cluster")
    
    valid_clusters_df = df_result[df_result['cluster'] != -1]
    
    if valid_clusters_df.empty:
        st.warning("Hanya terdeteksi noise (-1). Training model tidak dapat dilakukan.")
        return
    
    # Calculate cluster statistics
    cluster_means = valid_clusters_df.groupby('cluster')[selected_columns].mean()
    cluster_means['score'] = cluster_means.mean(axis=1)
    sorted_clusters = cluster_means.sort_values('score').index.tolist()
    
    # Define cluster labels
    n_clusters = len(sorted_clusters)
    
    if n_clusters == 2:
        labels_text = ["RENDAH", "TINGGI"]
    elif n_clusters == 3:
        labels_text = ["RENDAH", "SEDANG", "TINGGI"]
    elif n_clusters == 4:
        labels_text = ["SANGAT RENDAH", "RENDAH", "TINGGI", "SANGAT TINGGI"]
    elif n_clusters == 5:
        labels_text = ["SANGAT RENDAH", "RENDAH", "SEDANG", "TINGGI", "SANGAT TINGGI"]
    else:
        labels_text = [f"Level {i+1}" for i in range(n_clusters)]
    
    cluster_mapping = {}
    
    # Display cluster labels in columns
    cols = st.columns(n_clusters)
    for i, cluster_id in enumerate(sorted_clusters):
        label = labels_text[i]
        cluster_mapping[cluster_id] = label
        
        with cols[i]:
            st.metric(
                label=f"Cluster {cluster_id}",
                value=label,
                delta=f"Score: {cluster_means.loc[cluster_id, 'score']:.2f}"
            )
    
    st.session_state['cluster_mapping'] = cluster_mapping
    
    with st.expander("Lihat detail statistik cluster"):
        st.dataframe(cluster_means.style.background_gradient(cmap='coolwarm', axis=1))
    
    st.divider()
    
    # Model Training Section
    st.subheader("Training Model")
    
    col1, col2 = st.columns(2)
    with col1:
        test_size = st.slider("Persentase data testing", 10, 50, 20) / 100
    with col2:
        n_estimators = st.slider("Jumlah decision trees", 50, 500, 100, step=50)
    
    if st.button("Train Model", type="primary", use_container_width=True):
        with st.spinner("Training model..."):
            try:
                # Siapkan data
                X = X_scaled
                y = valid_clusters_df['cluster'].values
                
                # Cek jumlah sampel per cluster
                from collections import Counter
                cluster_counts = Counter(y)
                min_samples = min(cluster_counts.values())
                
                # Info tentang distribusi cluster
                st.info(f"📊 Distribusi cluster: {dict(cluster_counts)}")
                
                # Tentukan apakah bisa pakai stratify
                use_stratify = min_samples >= 2
                
                if not use_stratify:
                    st.warning(f"⚠️ Ada cluster dengan jumlah data terlalu sedikit (minimal: {min_samples}). Stratified split dinonaktifkan.")
                    
                    # Opsi untuk filter cluster kecil
                    if min_samples == 1:
                        col_opt1, col_opt2 = st.columns(2)
                        with col_opt1:
                            if st.checkbox("Filter cluster dengan <2 data", value=True):
                                # Filter hanya cluster yang punya >= 2 sampel
                                valid_mask = valid_clusters_df['cluster'].isin(
                                    [k for k, v in cluster_counts.items() if v >= 2]
                                )
                                
                                if valid_mask.sum() < len(valid_clusters_df):
                                    filtered_indices = valid_clusters_df[valid_mask].index
                                    X = X_scaled[valid_clusters_df.index.isin(filtered_indices)]
                                    y = valid_clusters_df.loc[valid_mask, 'cluster'].values
                                    
                                    st.success(f"✅ Data difilter: {len(y)} records dari {len(valid_clusters_df)}")
                                    use_stratify = True
                
                # Split data
                if use_stratify:
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size, random_state=42, stratify=y
                    )
                    st.info("✅ Menggunakan stratified split")
                else:
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size, random_state=42
                    )
                    st.info("ℹ️ Menggunakan random split (tanpa stratify)")
                
                # Train model
                rf_model = RandomForestClassifier(
                    n_estimators=n_estimators, 
                    random_state=42,
                    n_jobs=-1
                )
                rf_model.fit(X_train, y_train)
                
                y_pred = rf_model.predict(X_test)
                
                # Evaluasi
                acc = accuracy_score(y_test, y_pred)
                prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                
                st.session_state.update({
                    'ml_model': rf_model,
                    'X_test': X_test,
                    'y_test': y_test,
                    'y_pred': y_pred,
                    'accuracy': acc,
                    'precision': prec,
                    'recall': rec,
                    'f1_score': f1,
                    'trained_features': selected_columns.copy()  # Simpan fitur yang dipakai training
                })
                
                st.success(f"✅ Model berhasil di-train dengan akurasi {acc:.2%}")
                st.balloons()
                
            except Exception as e:
                st.error(f"❌ Error: {e}")
                import traceback
                with st.expander("Detail error"):
                    st.code(traceback.format_exc())
    
    # Display Model Results
    if 'ml_model' in st.session_state:
        st.divider()
        st.subheader("Evaluasi Model")
        
        acc = st.session_state['accuracy']
        prec = st.session_state['precision']
        rec = st.session_state['recall']
        f1 = st.session_state['f1_score']
        y_test = st.session_state['y_test']
        y_pred = st.session_state['y_pred']
        rf_model = st.session_state['ml_model']
        
        # Metrics display
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", f"{acc:.1%}")
        col2.metric("Precision", f"{prec:.1%}")
        col3.metric("Recall", f"{rec:.1%}")
        col4.metric("F1-Score", f"{f1:.1%}")
        
        st.write("")
        
        # Confusion Matrix & Classification Report
        tab1, tab2, tab3 = st.tabs(["Confusion Matrix", "Classification Report", "Feature Importance"])
        
        with tab1:
            fig_cm, ax = plt.subplots(figsize=(8, 6))
            cm = confusion_matrix(y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, 
                       cbar_kws={'label': 'Count'})
            ax.set_xlabel("Predicted", fontweight='bold')
            ax.set_ylabel("Actual", fontweight='bold')
            ax.set_title("Confusion Matrix")
            st.pyplot(fig_cm)
            plt.close(fig_cm)
        
        with tab2:
            report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            report_df = pd.DataFrame(report_dict).T
            st.dataframe(
                report_df.style.background_gradient(cmap='RdYlGn', subset=['precision', 'recall', 'f1-score']),
                use_container_width=True
            )
        
        with tab3:
            # BAGIAN YANG DIPERBAIKI - Feature Importance dengan handling fleksibel
            try:
                importance_values = rf_model.feature_importances_
                
                # Ambil nama fitur dari model atau dari selected_columns
                if hasattr(rf_model, 'feature_names_in_'):
                    feature_names = list(rf_model.feature_names_in_)
                else:
                    feature_names = selected_columns.copy()
                
                # Validasi dan sesuaikan panjang
                if len(feature_names) != len(importance_values):
                    st.warning(f"⚠️ Jumlah fitur ({len(feature_names)}) tidak sama dengan importance values ({len(importance_values)}). Menyesuaikan...")
                    min_length = min(len(feature_names), len(importance_values))
                    feature_names = feature_names[:min_length]
                    importance_values = importance_values[:min_length]
                
                # Buat DataFrame
                feature_importance = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': importance_values
                }).sort_values('Importance', ascending=False)
                
                # Plot
                fig_imp, ax = plt.subplots(figsize=(10, 6))
                ax.barh(feature_importance['Feature'], feature_importance['Importance'], 
                       color='steelblue', edgecolor='black')
                ax.set_xlabel('Importance', fontweight='bold')
                ax.set_title('Feature Importance', fontweight='bold', fontsize=14)
                ax.invert_yaxis()
                
                for i, (idx, row) in enumerate(feature_importance.iterrows()):
                    ax.text(row['Importance'], i, f" {row['Importance']:.3f}", 
                           va='center', fontsize=9)
                
                plt.tight_layout()
                st.pyplot(fig_imp)
                plt.close(fig_imp)
                
                # Tampilkan tabel feature importance
                with st.expander("📊 Lihat detail Feature Importance"):
                    st.dataframe(
                        feature_importance.style.background_gradient(cmap='YlOrRd', subset=['Importance']),
                        use_container_width=True
                    )
                
            except Exception as e:
                st.error(f"Error saat membuat Feature Importance: {e}")
                st.info("Model ini mungkin tidak mendukung feature importance atau terjadi kesalahan dalam data.")
        
        st.divider()
        
        # Prediction Section
        st.subheader("Prediksi Data Baru")
        
        # Gunakan fitur yang sama dengan saat training
        trained_features = st.session_state.get('trained_features', selected_columns)
        
        # Validasi: pastikan trained_features ada
        if not trained_features:
            st.error("❌ Fitur training tidak ditemukan. Silakan train model ulang.")
            return
        
        st.info(f"🎯 Model menggunakan {len(trained_features)} fitur: {', '.join(trained_features)}")
        
        inputs = {}
        cols = st.columns(min(len(trained_features), 3))
        
        for i, col_name in enumerate(trained_features):
            col_idx = i % len(cols)
            with cols[col_idx]:
                # Pastikan kolom ada di dataframe
                if col_name not in df_result.columns:
                    st.warning(f"⚠️ Fitur '{col_name}' tidak ditemukan di data")
                    continue
                
                mean_val = float(df_result[col_name].mean())
                min_val = float(df_result[col_name].min())
                max_val = float(df_result[col_name].max())
                
                inputs[col_name] = st.number_input(
                    col_name,
                    value=mean_val,
                    min_value=min_val,
                    max_value=max_val,
                    format="%.2f"
                )
        
        if st.button("Prediksi", use_container_width=True, type="primary"):
            try:
                # Buat DataFrame dengan urutan fitur yang sama seperti saat training
                input_df = pd.DataFrame([inputs])[trained_features]
                
                # Validasi: cek jumlah fitur
                if len(input_df.columns) != len(trained_features):
                    st.error(f"❌ Jumlah fitur tidak sesuai! Expected: {len(trained_features)}, Got: {len(input_df.columns)}")
                    return
                
                # Transform data
                if 'scaler' in st.session_state:
                    input_scaled = st.session_state['scaler'].transform(input_df)
                else:
                    input_scaled = input_df.values
                
                # Validasi shape sebelum prediksi
                expected_features = rf_model.n_features_in_
                if input_scaled.shape[1] != expected_features:
                    st.error(f"❌ Model mengharapkan {expected_features} fitur, tapi menerima {input_scaled.shape[1]} fitur")
                    st.info(f"Fitur yang diharapkan: {list(trained_features)}")
                    return
                
                # Prediksi
                pred = rf_model.predict(input_scaled)[0]
                proba = rf_model.predict_proba(input_scaled)[0]
                
                cluster_mapping = st.session_state.get('cluster_mapping', {})
                human_label = cluster_mapping.get(pred, "Unknown")
                
                st.write("")
                col_res1, col_res2 = st.columns([1, 2])
                
                with col_res1:
                    st.metric(
                        label="Hasil Prediksi",
                        value=human_label,
                        delta=f"Cluster {pred}"
                    )
                
                with col_res2:
                    prob_data = []
                    for i, prob in enumerate(proba):
                        label = cluster_mapping.get(i, f"Cluster {i}")
                        prob_data.append({
                            'Label': f"{label} (C{i})",
                            'Probability': prob
                        })
                    
                    prob_df = pd.DataFrame(prob_data).sort_values('Probability', ascending=False)
                    
                    fig_prob, ax = plt.subplots(figsize=(8, 4))
                    ax.barh(prob_df['Label'], prob_df['Probability'], 
                           color='steelblue', edgecolor='black')
                    ax.set_xlabel('Probability', fontweight='bold')
                    ax.set_xlim(0, 1)
                    
                    for i, (idx, row) in enumerate(prob_df.iterrows()):
                        ax.text(row['Probability'], i, f" {row['Probability']:.1%}", 
                               va='center', fontsize=9, fontweight='bold')
                    
                    plt.tight_layout()
                    st.pyplot(fig_prob)
                    plt.close(fig_prob)
                
                with st.expander("Data input"):
                    st.dataframe(input_df.T.rename(columns={0: 'Nilai'}))
                
                st.success("Prediksi berhasil")
                
            except Exception as e:
                st.error(f"Error: {e}")

if __name__ == "__main__":
    show()