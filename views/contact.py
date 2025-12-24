import streamlit as st

def show():
    st.title("📧 Contact Me")
    st.markdown("Mari terhubung dan berkolaborasi bersama")
    st.divider()
    
    # Profile Section
    col_photo1, col_photo2, col_photo3 = st.columns([1, 2, 1])
    with col_photo2:
        try:
            st.image("views/fotovia.jpeg", width=200, use_container_width=False)
        except:
            st.markdown("""
            <div style='text-align: center;'>
                <div style='width: 200px; height: 200px; margin: 0 auto; background: #667eea; 
                     border-radius: 50%; display: flex; align-items: center; justify-content: center;'>
                    <span style='font-size: 80px;'>👤</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### Oktaviana Rahma Dhani")
        st.caption("Data Scientist | Machine Learning Enthusiast")
        st.caption("📍 Semarang, Indonesia")
    
    st.divider()
    
    # About Developer
    st.subheader("Tentang Developer")
    st.info("""
    Dashboard Analisis Clustering ini dibuat sebagai bagian dari Ujian Akhir Semester. 
    Aplikasi ini mendemonstrasikan implementasi berbagai algoritma clustering dengan visualisasi 
    interaktif menggunakan Python, Streamlit, dan Scikit-learn.
    """)
    
    st.divider()
    
    # Contact Cards
    st.subheader("Connect With Me")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**📧 Email**")
        st.caption("Hubungi saya via email untuk diskusi lebih lanjut")
        st.link_button("vianarhma@gmail.com", "mailto:vianarhma@gmail.com", use_container_width=True)
    
    with col2:
        st.markdown("**💼 LinkedIn**")
        st.caption("Mari terhubung secara profesional")
        st.link_button("Oktaviana Rahma Dhani", "https://www.linkedin.com/in/oktaviana-rahma-dhani-7a43a2291", use_container_width=True)
    
    with col3:
        st.markdown("**💻 GitHub**")
        st.caption("Lihat portfolio dan project lainnya")
        st.link_button("github.com/vianarhma", "https://github.com/vianarhma", use_container_width=True)
    
    st.divider()
    
    # Additional Info
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.markdown("**🎓 Informasi Akademik**")
        st.markdown("""
        - **Universitas**: Universitas Muhammadiyah Semarang
        - **Program Studi**: S1 Sains Data
        - **Mata Kuliah**: Machine Learning
        - **Semester**: 5 (2024/2025)
        """)
    
    with col_b:
        st.markdown("**🛠️ Tech Stack**")
        st.markdown("""
        - **Language**: Python 3.11+
        - **Framework**: Streamlit
        - **ML Library**: Scikit-learn
        - **Visualization**: Matplotlib, Seaborn, Folium
        - **Data Processing**: Pandas, NumPy
        """)
    
    st.divider()
    
    # Feedback Form
    st.subheader("Kirim Pesan")
    st.info("Form ini untuk demonstrasi. Untuk mengirim pesan, silakan hubungi melalui email atau LinkedIn di atas.")
    
    with st.form("contact_form"):
        col_form1, col_form2 = st.columns(2)
        with col_form1:
            name = st.text_input("Nama Anda", placeholder="John Doe")
        with col_form2:
            email = st.text_input("Email Anda", placeholder="john@example.com")
        
        subject = st.text_input("Subject", placeholder="Topik pesan Anda")
        message = st.text_area("Pesan Anda", height=150, placeholder="Tuliskan pesan, saran, atau feedback Anda...")
        
        submitted = st.form_submit_button("Kirim Pesan", use_container_width=True, type="primary")
        
        if submitted:
            if name and email and message:
                st.success(f"Terima kasih **{name}**! Pesan Anda telah diterima dan akan segera saya balas.")
                st.balloons()
                
                with st.expander("Lihat ringkasan pesan"):
                    st.write(f"**Dari**: {name}")
                    st.write(f"**Email**: {email}")
                    if subject:
                        st.write(f"**Subject**: {subject}")
                    st.write(f"**Pesan**: {message}")
            else:
                st.error("Mohon isi semua field yang wajib (Nama, Email, dan Pesan)")
    
    st.divider()
    
    # Project Stats
    st.subheader("Project Stats")
    
    col1, col2, col3, col4 = st.columns(4)
    
    col1.metric("Algoritma", "5", help="KMeans, DBSCAN, Hierarchical, Mean Shift, Grid-Based")
    col2.metric("Metrik Evaluasi", "3", help="Silhouette, Davies-Bouldin, Calinski-Harabasz")
    col3.metric("Visualisasi", "10+", help="PCA, Heatmap, Bar Chart, dll")
    col4.metric("Lines of Code", "2000+", help="Total baris kode Python")
    
    st.divider()
    
    # Footer
    st.markdown("""
    <div style='text-align: center; padding: 30px; color: #666;'>
        <h4>⭐ Suka dengan project ini?</h4>
        <p>Jangan lupa untuk memberikan star di GitHub dan share ke teman-teman!</p>
        <p style='margin-top: 20px; font-size: 13px; opacity: 0.8;'>
            © 2025 Dashboard Clustering Analytics | Made with ❤️ by Oktaviana Rahma Dhani
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    show()