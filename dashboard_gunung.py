import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib

st.set_page_config(page_title="Rekomendasi Gunung", layout="wide")
st.title("Sistem Rekomendasi Gunung Pulau Jawa")

# -------------------------------
# Load dataset & model
# -------------------------------
data = pd.read_csv("dataset_gunung_final.csv")
model = load_model("mlp_gunung_model.h5")
scaler = joblib.load("scaler_gunung.save")

# -------------------------------
# Input user interaktif
# -------------------------------
st.sidebar.header("Filter Preferensi")

# Contoh input: kategori dan fitur numerik
kategori = st.sidebar.selectbox("Pilih kategori gunung:", data['kategori'].unique())

# Untuk fitur numerik (misal 5 fitur)
fitur_model = ['fitur1','fitur2','fitur3','fitur4','fitur5']
input_data = {}
for f in fitur_model:
    min_val = float(data[f].min())
    max_val = float(data[f].max())
    val = st.sidebar.slider(f"{f}:", min_val, max_val, float((min_val+max_val)/2))
    input_data[f] = val

# -------------------------------
# Filter dataset berdasarkan kategori
# -------------------------------
hasil_filter = data[data['kategori'] == kategori].copy()

# -------------------------------
# Buat dataframe untuk prediksi
# -------------------------------
X_user = hasil_filter[fitur_model].astype(np.float32)

# Terapkan scaler (yang sama dengan saat training)
X_scaled = scaler.transform(X_user)

# -------------------------------
# Fungsi prediksi skor
# -------------------------------
def predict_score(X_input):
    scores = model.predict(X_input).flatten()
    return scores

# Tambahkan kolom skor
hasil_filter['Skor'] = predict_score(X_scaled)

# -------------------------------
# Tampilkan hasil rekomendasi
# -------------------------------
st.subheader("Rekomendasi Gunung Berdasarkan Preferensi")
st.write(hasil_filter.sort_values('Skor', ascending=False).reset_index(drop=True))

# Optional: tampilkan kolom penting saja
# st.write(hasil_filter.sort_values('Skor', ascending=False)[['nama', 'lokasi', 'tinggi', 'Skor']])
