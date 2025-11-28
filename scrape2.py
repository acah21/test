import pandas as pd

df = pd.read_csv("dataset_gunung.csv")

# --- Pastikan kolom elevasi betul ---
elev_col = "elevation_m"

# --- Jika tidak ada distance, isi default 8 km ---
if "distance_km" not in df.columns:
    df["distance_km"] = 8  # asumsi rata-rata (boleh kamu ganti)

# --- Hitung elevation gain (anggap basecamp 0 mdpl) ---
df["Elevation_gain"] = df[elev_col]

# --- Rumus durasi: Naismith's Rule ---
def calculate_duration(distance_km, elevation_gain):
    return (distance_km / 5) + (elevation_gain / 600)

df["hiking_duration_hours"] = df.apply(
    lambda row: calculate_duration(
        row["distance_km"],
        row["Elevation_gain"]
    ),
    axis=1
)

# --- Tentukan tingkat kesulitan ---
def difficulty_level(duration, elevation_gain, distance_km):
    slope = elevation_gain / (distance_km * 1000)

    if duration < 4 and elevation_gain < 800:
        return "Easy"
    elif 4 <= duration <= 8 or 800 <= elevation_gain <= 1500:
        return "Moderate"
    else:
        return "Hard"

df["difficulty_level"] = df.apply(
    lambda row: difficulty_level(
        row["hiking_duration_hours"],
        row["Elevation_gain"],
        row["distance_km"]
    ),
    axis=1
)

# --- Kelompok rekomendasi ---
def recommended_group(difficulty):
    if difficulty == "Easy":
        return "Beginner"
    elif difficulty == "Moderate":
        return "Intermediate"
    else:
        return "Expert"

df["recommended_for"] = df["difficulty_level"].apply(recommended_group)

df.to_csv("dataset_gunung_final.csv", index=False)

print("Selesai! Semua kolom sudah terisi. File disimpan sebagai dataset_gunung_final.csv")
