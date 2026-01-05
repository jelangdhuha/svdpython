from fastapi import FastAPI
import pickle
import pandas as pd
import os
import math
from surprise import SVD, Dataset, Reader, dump

app = FastAPI()

# =========================================
# 1. KONFIGURASI PATH & FILE
# =========================================
# PERBAIKAN 1: Gunakan Path Absolut yang sudah terbukti berhasil
# Sesuaikan huruf besar/kecil dengan folder asli Anda
DATA_DIR = r"D:/Sistem Rekomendasi/filmku-app/app/data"

print(f"📂 PYTHON MEMBACA DARI: {DATA_DIR}")

MOVIES_PATH = os.path.join(DATA_DIR, "movies_fixed.csv")
# PERBAIKAN 2: Nama file diperbaiki (pakai 's' -> ratings_...)
RATINGS_PATH = os.path.join(DATA_DIR, "ratings_final_fixed.csv") 
MODEL_PATH = os.path.join(DATA_DIR, "model.pkl") # Simpan model di folder data juga agar rapi

# Variabel Global
movies_df = pd.DataFrame()
ratings_df = pd.DataFrame()
model = None

# =========================================
# 2. FUNGSI LOAD DATA & MODEL
# =========================================
def load_data_initial():
    global movies_df, ratings_df, model

    # A. Load Movies
    if os.path.exists(MOVIES_PATH):
        try:
            movies_df = pd.read_csv(MOVIES_PATH, dtype={'movie_id': int})
            print(f"✅ Movies Loaded: {len(movies_df)} judul.")
        except Exception as e:
            print(f"❌ Error loading movies: {e}")

    # B. Load Ratings
    if os.path.exists(RATINGS_PATH):
        try:
            ratings_df = pd.read_csv(RATINGS_PATH, dtype={'user_id': int, 'movie_id': int})
            print(f"✅ Ratings Loaded: {len(ratings_df)} baris.")
        except Exception as e:
            print(f"❌ Error loading ratings: {e}")

    # C. Load Model
    # PERBAIKAN 3: Logika Else dihapus agar model tidak ter-reset
    if os.path.exists(MODEL_PATH):
        try:
            with open(MODEL_PATH, "rb") as f:
                model = pickle.load(f)
            print("✅ Model SVD Loaded (Siap Prediksi).")
        except:
            print("⚠️ Model rusak, membuat model baru (kosong).")
            model = SVD()
    else:
        print("⚠️ Model belum ada, membuat model baru (kosong).")
        model = SVD()

# Jalankan saat start
load_data_initial()


# =========================================
# 3. ENDPOINT: RELOAD DATA
# =========================================
@app.get("/system/reload-data")
def reload_data():
    load_data_initial()
    return {"status": "success", "message": "Data CSV & Model berhasil dimuat ulang."}


# =========================================
# 4. ENDPOINT: RETRAIN
# =========================================
@app.get("/system/retrain")
def retrain_model():
    global model, ratings_df
    try:
        if not os.path.exists(RATINGS_PATH):
            return {"status": "error", "message": "File ratings belum ada. Sync Laravel dulu."}

        df_train = pd.read_csv(RATINGS_PATH)

        # Cek kolom wajib
        required_cols = {'user_id', 'movie_id', 'rating'}
        if not required_cols.issubset(df_train.columns):
            return {"status": "error", "message": f"CSV Rating harus punya kolom: {required_cols}"}

        if df_train.empty:
            return {"status": "error", "message": "Data rating kosong."}

        # Training
        reader = Reader(rating_scale=(0, 5))
        data = Dataset.load_from_df(df_train[['user_id', 'movie_id', 'rating']], reader)
        trainset = data.build_full_trainset()

        new_model = SVD()
        new_model.fit(trainset)

        # Simpan Model
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(new_model, f)
        
        # Update Memory
        model = new_model
        ratings_df = df_train
        
        return {"status": "success", "message": f"Retrain Selesai! Dilatih dengan {len(df_train)} data rating."}

    except Exception as e:
        return {"status": "error", "message": str(e)}


# =========================================
# 5. ENDPOINT: REKOMENDASI (DENGAN FALLBACK)
# =========================================
@app.get("/recommend/{user_id}")
def recommend(user_id: int, top_n: int = 10):
    global model, movies_df, ratings_df

    if movies_df.empty:
        return [] # Jangan return string error, return list kosong agar Laravel tidak crash
    
    recommendations = []

    # ------------------------------------
    # CARA 1: PREDIKSI AI (PERSONAL)
    # ------------------------------------
    try:
        # Cari film yg sudah ditonton
        watched_ids = []
        if not ratings_df.empty and 'user_id' in ratings_df.columns:
            watched_data = ratings_df[ratings_df['user_id'] == user_id]
            watched_ids = watched_data['movie_id'].tolist()

        # Kandidat = Semua - Sudah Ditonton
        all_movie_ids = movies_df['movie_id'].unique()
        candidates = [m for m in all_movie_ids if m not in watched_ids]

        # Prediksi
        if candidates and model:
            preds = []
            for mid in candidates:
                try:
                    est_score = model.predict(user_id, mid).est
                    # Hapus NaN
                    if isinstance(est_score, float) and (math.isnan(est_score) or math.isinf(est_score)):
                        est_score = 0
                    preds.append((mid, est_score))
                except:
                    continue
            
            # Urutkan Top N
            top_preds = sorted(preds, key=lambda x: x[1], reverse=True)[:top_n]
            
            # Ambil detail film jika skor > 0
            top_ids = [p[0] for p in top_preds if p[1] > 0]
            
            if top_ids:
                result = movies_df[movies_df['movie_id'].isin(top_ids)].copy()
                score_map = {mid: round(score, 2) for mid, score in top_preds}
                result["predicted_rating"] = result['movie_id'].map(score_map)
                
                recommendations = result.sort_values("predicted_rating", ascending=False).fillna("").to_dict(orient="records")

    except Exception as e:
        print(f"⚠️ Error AI: {e}")

    # ------------------------------------
    # CARA 2: FALLBACK (POPULARITY)
    # ------------------------------------
    # PERBAIKAN 4: Jika AI kosong/gagal, tampilkan film rating tertinggi global
    if not recommendations:
        print(f"⚠️ User {user_id}: Hasil AI kosong. Menggunakan Top Global.")
        if 'rating' in movies_df.columns:
            # Ambil Top 10 berdasarkan rating asli di database
            top_global = movies_df.sort_values('rating', ascending=False).head(top_n).copy()
            top_global["predicted_rating"] = top_global["rating"] 
            recommendations = top_global.fillna("").to_dict(orient="records")

    return recommendations

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8001, reload=True)