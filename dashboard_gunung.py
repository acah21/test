import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib

st.set_page_config(page_title="Rekomendasi Gunung", layout="wide")
st.title("Sistem Rekomendasi Gunung Pulau Jawa")

# -------------------------------
# Load dataset, model, scaler
# -------------------------------
data = pd.read_csv("dataset_gunung_final.csv")
model = load_model("mlp_gunung_model.h5")
scaler = joblib.load("scaler_gunung.save")

# -------------------------------
# Sidebar input user
# -------------------------------
st.sidebar.header("Filter Preferensi")

# Pilih kategori
kategori = st.sidebar.selectbox("Pilih kategori gunung:", data['kategori'].unique())

# Filter dataset berdasarkan kategori
hasil_filter = data[data['kategori'] == kategori].copy()

# Pilih fitur MLP (numerik + dummy kategori)
fitur_num = ['elevation_m', 'hiking_duration_hours', 'distance_km', 'Elevation_gain']
fitur_cat = [col for col in data.columns if col.startswith(('difficulty_level_', 'recommended_for_'))]
fitur_model = fitur_num + fitur_cat

# Pastikan semua numeric untuk MLP
X = hasil_filter[fitur_model].astype(np.float32)

# Terapkan scaler (sama seperti saat training)
X_scaled = scaler.transform(X)

# -------------------------------
# Fungsi prediksi skor
# -------------------------------
def predict_score(X_input):
    return model.predict(X_input).flatten()

hasil_filter['Skor'] = predict_score(X_scaled)

# -------------------------------
# Tampilkan hasil rekomendasi
# -------------------------------
st.subheader("Hasil Rekomendasi Gunung")
st.dataframe(
    hasil_filter.sort_values('Skor', ascending=False).reset_index(drop=True)
)

# -------------------------------
# Optional: visualisasi top 5 gunung
# -------------------------------
top5 = hasil_filter.sort_values('Skor', ascending=False).head(5)
st.subheader("Top 5 Gunung")
st.bar_chart(data=top5.set_index('Name')['Skor'])
