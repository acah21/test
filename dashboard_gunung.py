import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.models import load_model
import joblib

st.set_page_config(page_title="Rekomendasi Gunung Pulau Jawa", layout="wide")
st.title("Sistem Rekomendasi Gunung Pulau Jawa")

# -------------------------------
# Load dataset, model, scaler
# -------------------------------
data = pd.read_csv("dataset_gunung_final.csv")

# Load MLP model
mlp_model = load_model("mlp_gunung_model.h5")

# Load scaler
scaler = joblib.load("scaler_gunung.save")

# -------------------------------
# Sidebar input user
# -------------------------------
st.sidebar.header("Preferensi Pendakian")

kategori = st.sidebar.selectbox("Pilih kategori gunung:", data['kategori'].unique())
difficulty = st.sidebar.selectbox("Pilih level kesulitan:", data['difficulty_level'].unique())
recommended_for = st.sidebar.selectbox("Direkomendasikan untuk:", data['recommended_for'].unique())

# Fitur numerik input (slider berdasarkan range dataset)
st.sidebar.subheader("Sesuaikan fitur gunung (opsional)")
elevation = st.sidebar.slider("Elevation (m)", float(data['elevation_m'].min()), float(data['elevation_m'].max()), float(data['elevation_m'].mean()))
hiking_duration = st.sidebar.slider("Hiking duration (hours)", float(data['hiking_duration_hours'].min()), float(data['hiking_duration_hours'].max()), float(data['hiking_duration_hours'].mean()))
distance = st.sidebar.slider("Distance (km)", float(data['distance_km'].min()), float(data['distance_km'].max()), float(data['distance_km'].mean()))
elevation_gain = st.sidebar.slider("Elevation gain", float(data['Elevation_gain'].min()), float(data['Elevation_gain'].max()), float(data['Elevation_gain'].mean()))

# -------------------------------
# Filter dataset berdasarkan input user
# -------------------------------
filtered = data[
    (data['kategori'] == kategori) &
    (data['difficulty_level'] == difficulty) &
    (data['recommended_for'] == recommended_for)
].copy()

if filtered.empty:
    st.warning("Tidak ada gunung yang sesuai filter. Silakan ubah preferensi.")
else:
    # -------------------------------
    # Preprocessing
    # -------------------------------
    # Fitur numerik
    features_num = ['elevation_m','hiking_duration_hours','distance_km','Elevation_gain']
    # Fitur kategori di-encode
    features_cat = ['difficulty_level','recommended_for']
    filtered_encoded = pd.get_dummies(filtered, columns=features_cat)

    # Pastikan semua fitur numerik di-scale
    filtered_encoded[features_num] = scaler.transform(filtered_encoded[features_num])

    # Buat matrix fitur akhir untuk MLP
    mlp_features = features_num + [col for col in filtered_encoded.columns if col.startswith(('difficulty_level_','recommended_for_'))]
    X_mlp = filtered_encoded[mlp_features].astype(np.float32)

    # -------------------------------
    # Prediksi skor MLP
    # -------------------------------
    filtered['mlp_score'] = mlp_model.predict(X_mlp).flatten()

    # -------------------------------
    # CBF similarity
    # -------------------------------
    # Gunakan fitur yang sama seperti MLP
    similarity_matrix = cosine_similarity(X_mlp)
    def recommend_cbf_top(filtered_data, top_n=5):
        sim_scores = list(enumerate(similarity_matrix[0]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        top_indices = [i[0] for i in sim_scores[1:top_n+1]]  # exclude self
        return filtered_data.iloc[top_indices][['Name','Province','source_url','mlp_score']]

    cbf_recommendation = recommend_cbf_top(filtered, top_n=5)

    # -------------------------------
    # Tampilkan hasil
    # -------------------------------
    st.subheader("Top 5 Gunung Rekomendasi (CBF + MLP)")
    st.dataframe(cbf_recommendation.reset_index(drop=True))

    # Visualisasi skor MLP
    st.subheader("Skor MLP dari Gunung Tersaring")
    st.bar_chart(cbf_recommendation.set_index('Name')['mlp_score'])
