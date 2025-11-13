import streamlit as st
import pandas as pd
import time
from backend import (
    load_model_artifacts, 
    fetch_nasa_data, 
    apply_xgboost_filter,
    run_monte_carlo, 
    plot_simulation_viz
)

st.set_page_config(page_title="Kalkulator Tabrakan Asteroid", page_icon="☄️", layout="wide")

# --- SETUP ---
MODEL_PATH = 'models/asteroid_model.pkl'
SCALER_PATH = 'models/data_scaler.pkl'
METADATA_PATH = 'models/model_metadata.json'

@st.cache_resource
def load_models():
    return load_model_artifacts(MODEL_PATH, SCALER_PATH, METADATA_PATH)

model, scaler, metadata = load_models()

try:
    API_KEY = st.secrets["NASA_API_KEY"]
except (KeyError, FileNotFoundError):
    st.error("Kunci API NASA tidak ditemukan di .streamlit/secrets.toml")
    st.stop()

@st.cache_data(ttl=3600)
def get_filtered_data(api_key, pages):
    if model is None: return pd.DataFrame()
    raw_data = fetch_nasa_data(api_key, pages=pages)
    if raw_data.empty: return pd.DataFrame()
    return apply_xgboost_filter(raw_data, model, scaler, metadata)

# --- UI ---
st.title("☄️ Kalkulator Probabilitas Tabrakan Asteroid")
st.markdown("Alur Kerja: **1. Filter AI (XGBoost)** → **2. Simulasi Fisika (Monte Carlo)**")

if model is None:
    st.error("⚠️ Model ML belum dimuat. Pastikan file .pkl ada di folder 'models/'.")
    st.stop()

col1, col2 = st.columns([1, 2])

# --- KOLOM KIRI: FILTERING ---
with col1:
    st.header("1. Filter & Pindai")
    st.info("Memindai data live NASA untuk mencari kandidat berbahaya.")
    
    num_pages = st.slider("Halaman Data (1 hal ≈ 20 asteroid)", 1, 15, 3)
    
    if st.button("Mulai Pemindaian"):
        with st.spinner("Memproses data orbit..."):
            filtered = get_filtered_data(API_KEY, num_pages)
            st.session_state['filtered_data'] = filtered
            
            if not filtered.empty:
                st.success(f"Ditemukan {len(filtered)} asteroid berpotensi bahaya.")
            else:
                st.warning("Tidak ada ancaman ditemukan dalam sampel ini.")

# --- KOLOM KANAN: SIMULASI ---
with col2:
    st.header("2. Simulasi Tabrakan")
    
    if 'filtered_data' not in st.session_state or st.session_state['filtered_data'].empty:
        st.markdown("*Menunggu hasil pemindaian...*")
    else:
        data = st.session_state['filtered_data']
        
        # Tampilkan Tabel
        st.dataframe(
            data[['name', 'risk_probability', 'orbit_uncertainty', 'min_orbit_intersection']],
            column_config={
                "risk_probability": st.column_config.ProgressColumn("Skor Risiko AI", format="%.2f", min_value=0, max_value=1),
                "orbit_uncertainty": st.column_config.NumberColumn("Uncertainty (0-9)"),
                "min_orbit_intersection": st.column_config.NumberColumn("MOID (AU)")
            },
            use_container_width=True
        )
        
        st.divider()
        
        # Panel Simulasi
        col_sim_1, col_sim_2 = st.columns(2)
        with col_sim_1:
            target_asteroid = st.selectbox("Pilih Target Simulasi:", data['name'])
        with col_sim_2:
            sim_count = st.selectbox("Jumlah Klon:", [1000, 5000, 10000, 50000])
            
        if st.button(f"Hitung Probabilitas Tabrakan: {target_asteroid}"):
            details = data[data['name'] == target_asteroid]
            
            with st.spinner("Menjalankan Monte Carlo..."):
                start = time.time()
                results, prob = run_monte_carlo(details, sim_count)
                elapsed = time.time() - start
                
            # Hasil
            st.success(f"Selesai dalam {elapsed:.2f} detik.")
            st.metric("Probabilitas Tabrakan Fisik", f"{prob*100:.4f} %")
            
            if prob > 0:
                st.error(f"PERINGATAN: {int(prob*sim_count)} klon menabrak Bumi!")
            else:
                st.success("Tidak ada tabrakan terdeteksi dalam simulasi ini.")
                
            # Grafik
            moid_val = details['min_orbit_intersection'].values[0]
            fig = plot_simulation_viz(results, moid_val, target_asteroid)
            st.plotly_chart(fig, use_container_width=True)