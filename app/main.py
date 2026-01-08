from fastapi import FastAPI
import pickle
import pandas as pd
import os
import math
from surprise import SVD, Dataset, Reader

app = FastAPI()

DATA_DIR = r"D:/Sistem Rekomendasi/filmku-app/app/data"

MOVIES_PATH = os.path.join(DATA_DIR, "movies_fixed.csv")
RATINGS_PATH = os.path.join(DATA_DIR, "ratings_final_fixed.csv")
MODEL_PATH = os.path.join(DATA_DIR, "model.pkl")

movies_df = pd.DataFrame()
ratings_df = pd.DataFrame()
model = None


def load_data_initial():
    global movies_df, ratings_df, model

    if os.path.exists(MOVIES_PATH):
        movies_df = pd.read_csv(MOVIES_PATH, dtype={'movie_id': int})

    # B. Load Ratings
    if os.path.exists(RATINGS_PATH):
        ratings_df = pd.read_csv(RATINGS_PATH, dtype={'user_id': int, 'movie_id': int})

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


load_data_initial()


# =========================================
# 3. ENDPOINT: RELOAD DATA
# =========================================
@app.get("/system/reload-data")
def reload_data():
    load_data_initial()
    return {"status": "success", "message": "Data reloaded"}


# =========================================
# 4. ENDPOINT: RETRAIN
# =========================================
@app.get("/system/retrain")
def retrain_model():
    global model, ratings_df

    if not os.path.exists(RATINGS_PATH):
        return {"status": "error", "message": "Ratings file not found"}

    df_train = pd.read_csv(RATINGS_PATH)

        # Cek kolom wajib
        required_cols = {'user_id', 'movie_id', 'rating'}
        if not required_cols.issubset(df_train.columns):
            return {"status": "error", "message": f"CSV Rating harus punya kolom: {required_cols}"}

    if df_train.empty:
        return {"status": "error", "message": "Ratings data empty"}

    reader = Reader(rating_scale=(0, 5))
    data = Dataset.load_from_df(df_train[['user_id', 'movie_id', 'rating']], reader)
    trainset = data.build_full_trainset()

    new_model = SVD()
    new_model.fit(trainset)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(new_model, f)

    model = new_model
    ratings_df = df_train

    return {
        "status": "success",
        "message": f"Retrain completed with {len(df_train)} ratings"
    }


# =========================================
# 5. ENDPOINT: REKOMENDASI (DENGAN FALLBACK)
# =========================================
@app.get("/recommend/{user_id}")
def recommend(user_id: int, top_n: int = 10):
    global model, movies_df, ratings_df

    if movies_df.empty:
        return []

    recommendations = []

    # ------------------------------------
    # CARA 1: PREDIKSI AI (PERSONAL)
    # ------------------------------------
    try:
        # Cari film yg sudah ditonton
        watched_ids = []
        if not ratings_df.empty:
            watched_ids = ratings_df[ratings_df['user_id'] == user_id]['movie_id'].tolist()

        # Kandidat = Semua - Sudah Ditonton
        all_movie_ids = movies_df['movie_id'].unique()
        candidates = [m for m in all_movie_ids if m not in watched_ids]

        # Prediksi
        if candidates and model:
            preds = []
            for mid in candidates:
                try:
                    est = model.predict(user_id, mid).est
                    if isinstance(est, float) and (math.isnan(est) or math.isinf(est)):
                        est = 0
                    preds.append((mid, est))
                except:
                    continue

            top_preds = sorted(preds, key=lambda x: x[1], reverse=True)[:top_n]
            top_ids = [mid for mid, score in top_preds if score > 0]

            if top_ids:
                result = movies_df[movies_df['movie_id'].isin(top_ids)].copy()
                score_map = {mid: round(score, 2) for mid, score in top_preds}
                result["predicted_rating"] = result['movie_id'].map(score_map)
                recommendations = (
                    result
                    .sort_values("predicted_rating", ascending=False)
                    .fillna("")
                    .to_dict(orient="records")
                )

    except:
        pass

    if not recommendations and 'rating' in movies_df.columns:
        top_global = movies_df.sort_values('rating', ascending=False).head(top_n).copy()
        top_global["predicted_rating"] = top_global["rating"]
        recommendations = top_global.fillna("").to_dict(orient="records")

    return recommendations


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8001, reload=True)
