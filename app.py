import streamlit as st
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb

# --- 1. KONFIGURASI HALAMAN & CSS ---
st.set_page_config(
    page_title="Stunting Prediction AI",
    page_icon="👶",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS untuk tampilan UI yang modern
st.markdown("""
    <style>
    /* Mengubah font dan background dasar */
    .main {
        background-color: #f0f2f6;
    }
    
    /* Styling Judul */
    h1 {
        color: #2c3e50;
        font-family: 'Helvetica Neue', sans-serif;
    }
    
    /* Styling Tombol */
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        border-radius: 12px;
        height: 55px;
        font-size: 18px;
        font-weight: bold;
        border: none;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #45a049;
        transform: translateY(-2px);
        box-shadow: 0 6px 8px rgba(0,0,0,0.15);
    }

    /* Styling Kartu Hasil */
    .result-card {
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        margin-top: 20px;
        color: white;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        animation: fadeIn 0.8s;
    }
    
    /* Animasi sederhana */
    @keyframes fadeIn {
        from {opacity: 0; transform: translateY(20px);}
        to {opacity: 1; transform: translateY(0);}
    }

    /* Info Box styling */
    .info-box {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #3498db;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        margin-bottom: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# --- 2. FUNGSI LOAD MODEL ---
@st.cache_resource
def load_components():
    try:
        # Load Model XGBoost
        model = joblib.load('model_xgb.pkl')
        
        # Load Scaler
        scaler = joblib.load('scaler.pkl')
        
        # Load Feature Names (Urutan kolom yang benar)
        feature_names = joblib.load('feature_names.pkl')
        
        # Load Label Encoder Target (Normal, Stunted, dll)
        # File bernama 'gender_encoder.pkl' tapi isinya adalah target classes
        target_decoder = joblib.load('gender_encoder.pkl')
        
        return model, scaler, feature_names, target_decoder
    except Exception as e:
        st.error(f"Error loading files: {e}")
        return None, None, None, None

# Memuat model
model, scaler, feature_names, target_decoder = load_components()

# --- 3. SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2966/2966334.png", width=80)
    st.title("AI Kesehatan Anak")
    st.markdown("Aplikasi ini membantu orang tua mendeteksi potensi stunting sejak dini menggunakan kecerdasan buatan (XGBoost).")
    
    st.markdown("### ℹ️ Panduan")
    st.info("""
    1. Masukkan **Umur** (bulan).
    2. Pilih **Jenis Kelamin**.
    3. Masukkan **Tinggi & Berat**.
    4. Klik **Analisis**.
    """)
    st.markdown("---")
    st.caption("Developed with ❤️ for Health Tech")

# --- 4. AREA UTAMA ---
col_main, col_spacing = st.columns([5,1])
with col_main:
    st.title("👶 Deteksi Status Gizi Balita")
    st.markdown("#### Monitor pertumbuhan anak Anda dengan presisi.")

# Layout 2 Kolom: Kiri (Input), Kanan (Hasil)
col_input, col_result = st.columns([1, 1.2], gap="large")

with col_input:
    st.markdown("### 📝 Data Anak")
    with st.container(border=True):
        # Input Umur
        umur = st.number_input("Umur (bulan)", min_value=0, max_value=120, value=12, step=1)
        
        # Input Gender dengan Icon
        gender_opt = st.radio("Jenis Kelamin", ["Laki-laki 👦", "Perempuan 👧"], horizontal=True)
        
        # Layout Tinggi & Berat Sebelahan
        c1, c2 = st.columns(2)
        with c1:
            tinggi = st.number_input("Tinggi (cm)", min_value=30.0, max_value=200.0, value=75.0, step=0.1)
        with c2:
            berat = st.number_input("Berat (kg)", min_value=2.0, max_value=100.0, value=9.0, step=0.1)
        
        st.markdown("<br>", unsafe_allow_html=True)
        predict_btn = st.button("🔍 Analisis Sekarang")

# --- 5. LOGIKA PREDIKSI ---
with col_result:
    if predict_btn:
        if model is None:
            st.error("Gagal memuat model. Pastikan file .pkl tersedia.")
        else:
            # A. PREPROCESSING
            # 1. Encoding Jenis Kelamin Manual
            # Sesuai feature 'Jenis Kelamin_Perempuan': 1 = Perempuan, 0 = Laki-laki
            is_female = 1 if "Perempuan" in gender_opt else 0
            
            # 2. Menyusun Data Frame
            input_data = {
                'Umur (bulan)': [umur],
                'Tinggi Badan (cm)': [tinggi],
                'Berat Badan (kg)': [berat],
                'Jenis Kelamin_Perempuan': [is_female]
            }
            df = pd.DataFrame(input_data)
            
            # 3. Memastikan urutan kolom sesuai feature_names.pkl
            try:
                df = df[feature_names]
            except KeyError:
                st.error("Format fitur tidak cocok. Cek file feature_names.pkl.")
                st.stop()
            
            # 4. Scaling
            scaled_data = scaler.transform(df)
            
            # B. PREDIKSI
            try:
                # XGBoost predict return index (0,1,2,3)
                prediction_idx = model.predict(scaled_data)
                
                # Mengubah index kembali ke teks (Normal, Stunted, dll)
                hasil_label = target_decoder.inverse_transform(prediction_idx)[0]
                
                # C. MENAMPILKAN HASIL UI
                st.markdown("### 📊 Hasil Analisis")
                
                # Logika Warna & Pesan
                label_check = hasil_label.lower()
                
                if "severely" in label_check:
                    bg_color = "linear-gradient(135deg, #ff416c, #ff4b2b)"
                    icon = "🚨"
                    title_text = "Severely Stunted"
                    desc = "Pertumbuhan anak sangat terhambat. Segera konsultasikan ke Dokter Spesialis Anak."
                elif "stunted" in label_check:
                    bg_color = "linear-gradient(135deg, #fce38a, #f38181)" # Orange tone
                    icon = "⚠️"
                    title_text = "Stunted (Pendek)"
                    desc = "Tinggi badan anak di bawah standar. Perbaiki asupan nutrisi dan gizi."
                elif "normal" in label_check:
                    bg_color = "linear-gradient(135deg, #11998e, #38ef7d)" # Green tone
                    icon = "✅"
                    title_text = "Normal"
                    desc = "Hebat! Pertumbuhan anak sangat baik dan sesuai usianya."
                elif "tall" in label_check:
                    bg_color = "linear-gradient(135deg, #2193b0, #6dd5ed)" # Blue tone
                    icon = "✨"
                    title_text = "Tinggi (Tall)"
                    desc = "Anak memiliki tinggi badan di atas rata-rata usianya."
                else:
                    bg_color = "#95a5a6"
                    icon = "ℹ️"
                    title_text = hasil_label
                    desc = "Status gizi tercatat."

                # Render Kartu Hasil
                st.markdown(f"""
                <div class="result-card" style="background: {bg_color};">
                    <div style="font-size: 50px; margin-bottom: 10px;">{icon}</div>
                    <h2 style="color: white; margin: 0;">{title_text}</h2>
                    <hr style="border-top: 1px solid rgba(255,255,255,0.4); margin: 15px 0;">
                    <p style="font-size: 16px; font-weight: 500;">{desc}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Menampilkan Data Ringkas
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(f"""
                <div class="info-box">
                    <b>Ringkasan Data:</b><br>
                    • Usia: {umur} Bulan<br>
                    • BMI Sederhana: {berat / ((tinggi/100)**2):.2f}
                </div>
                """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Terjadi kesalahan prediksi: {e}")
                
    else:
        # Tampilan Awal Kosong (Placeholder)
        st.info("👈 Masukkan data di sebelah kiri untuk melihat hasil prediksi.")
        st.markdown("""
        <div style="text-align: center; opacity: 0.5;">
            <img src="https://cdn-icons-png.flaticon.com/512/3050/3050525.png" width="150">
            <p>Menunggu Data...</p>
        </div>
        """, unsafe_allow_html=True)
