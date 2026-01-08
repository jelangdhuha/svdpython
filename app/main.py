from fastapi import FastAPI
import pickle
import pandas as pd
import os
import math

from surprise import SVD, Dataset, Reader
from surprise.model_selection import cross_validate

app = FastAPI()

# =========================================
# PATH KONFIGURASI
# =========================================
DATA_DIR = r"D:/Sistem Rekomendasi/filmku-app/app/data"

MOVIES_PATH = os.path.join(DATA_DIR, "movies_fixed.csv")
RATINGS_PATH = os.path.join(DATA_DIR, "ratings_final_fixed.csv")
MODEL_PATH = os.path.join(DATA_DIR, "model.pkl")

# =========================================
# GLOBAL VARIABLE
# =========================================
movies_df = pd.DataFrame()
ratings_df = pd.DataFrame()
model = None


# =========================================
# LOAD DATA & MODEL
# =========================================
def load_data_initial():
    global movies_df, ratings_df, model

    # Load Movies
    if os.path.exists(MOVIES_PATH):
        movies_df = pd.read_csv(MOVIES_PATH, dtype={"movie_id": int})
        print(f"✅ Movies loaded: {len(movies_df)} data")
    else:
        print("⚠️ Movies file tidak ditemukan")

    # Load Ratings
    if os.path.exists(RATINGS_PATH):
        ratings_df = pd.read_csv(
            RATINGS_PATH,
            dtype={"user_id": int, "movie_id": int}
        )
        print(f"✅ Ratings loaded: {len(ratings_df)} data")
    else:
        print("⚠️ Ratings file tidak ditemukan")

    # Load Model
    if os.path.exists(MODEL_PATH):
        try:
            with open(MODEL_PATH, "rb") as f:
                model = pickle.load(f)
            print("✅ Model SVD Loaded (Siap Prediksi)")
        except Exception as e:
            print("⚠️ Model rusak, membuat model baru:", e)
            model = SVD()
    else:
        print("⚠️ Model belum ada, membuat model baru")
        model = SVD()


load_data_initial()

# =========================================
# ENDPOINT: RELOAD DATA
# =========================================
@app.get("/system/reload-data")
def reload_data():
    load_data_initial()
    return {
        "status": "success",
        "message": "Data & model reloaded"
    }


# =========================================
# ENDPOINT: RETRAIN MODEL
# =========================================
@app.get("/system/retrain")
def retrain_model():
    global model, ratings_df

    if not os.path.exists(RATINGS_PATH):
        return {"status": "error", "message": "Ratings file not found"}

    df_train = pd.read_csv(RATINGS_PATH)

    required_cols = {"user_id", "movie_id", "rating"}
    if not required_cols.issubset(df_train.columns):
        return {
            "status": "error",
            "message": f"CSV Rating harus punya kolom: {required_cols}"
        }

    if df_train.empty:
        return {"status": "error", "message": "Ratings data empty"}

    reader = Reader(rating_scale=(0, 5))
    data = Dataset.load_from_df(
        df_train[["user_id", "movie_id", "rating"]],
        reader
    )

    # =====================================
    # PARAMETER YANG AKAN DIEKSPLORASI
    # =====================================
    param_grid = [
        {"n_factors": 50,  "n_epochs": 20, "lr_all": 0.005, "reg_all": 0.02},
        {"n_factors": 100, "n_epochs": 20, "lr_all": 0.005, "reg_all": 0.02},
        {"n_factors": 100, "n_epochs": 30, "lr_all": 0.005, "reg_all": 0.02},
        {"n_factors": 100, "n_epochs": 30, "lr_all": 0.002, "reg_all": 0.05},
    ]

    print("\n" + "=" * 60)
    print("MULAI EKSPLORASI PARAMETER SVD")

    best_rmse = float("inf")
    best_params = None
    history = []

    # =====================================
    # LOOP EVALUASI PARAMETER
    # =====================================
    for params in param_grid:
        algo = SVD(**params)

        results = cross_validate(
            algo,
            data,
            measures=["RMSE", "MAE"],
            cv=3,
            verbose=False
        )

        avg_rmse = results["test_rmse"].mean()
        avg_mae = results["test_mae"].mean()

        history.append({
            **params,
            "rmse": round(avg_rmse, 4),
            "mae": round(avg_mae, 4)
        })

        print(
            f"RMSE={avg_rmse:.4f} | MAE={avg_mae:.4f} | params={params}"
        )

        if avg_rmse < best_rmse:
            best_rmse = avg_rmse
            best_params = params

    print("-" * 60)
    print(f"BEST RMSE : {best_rmse:.4f}")
    print(f"BEST PARAMS : {best_params}")
    print("=" * 60 + "\n")

    # =====================================
    # TRAIN MODEL FINAL (BEST PARAM)
    # =====================================
    trainset = data.build_full_trainset()

    final_model = SVD(**best_params)
    final_model.fit(trainset)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(final_model, f)

    model = final_model
    ratings_df = df_train

    return {
        "status": "success",
        "message": "Retrain & tuning completed",
        "best_rmse": round(best_rmse, 4),
        "best_params": best_params,
        "history": history
    }


# =========================================
# ENDPOINT: REKOMENDASI
# =========================================
@app.get("/recommend/{user_id}")
def recommend(user_id: int, top_n: int = 10):
    global model, movies_df, ratings_df

    if movies_df.empty:
        return []

    recommendations = []

    # =============================
    # REKOMENDASI PERSONAL (SVD)
    # =============================
    try:
        watched_ids = []
        if not ratings_df.empty:
            watched_ids = ratings_df[
                ratings_df["user_id"] == user_id
            ]["movie_id"].tolist()

        all_movie_ids = movies_df["movie_id"].unique()
        candidates = [m for m in all_movie_ids if m not in watched_ids]

        if candidates and model:
            preds = []

            for movie_id in candidates:
                try:
                    est = model.predict(user_id, movie_id).est
                    if math.isnan(est) or math.isinf(est):
                        est = 0
                    preds.append((movie_id, est))
                except:
                    continue

            top_preds = sorted(
                preds,
                key=lambda x: x[1],
                reverse=True
            )[:top_n]

            top_ids = [mid for mid, score in top_preds if score > 0]

            if top_ids:
                result = movies_df[
                    movies_df["movie_id"].isin(top_ids)
                ].copy()

                score_map = {
                    mid: round(score, 2)
                    for mid, score in top_preds
                }

                result["predicted_rating"] = result["movie_id"].map(score_map)

                recommendations = (
                    result
                    .sort_values("predicted_rating", ascending=False)
                    .fillna("")
                    .to_dict(orient="records")
                )

    except Exception as e:
        print("⚠️ Error rekomendasi:", e)

    # =============================
    # FALLBACK: TOP GLOBAL
    # =============================
    if not recommendations and "rating" in movies_df.columns:
        top_global = movies_df.sort_values(
            "rating",
            ascending=False
        ).head(top_n)

        top_global["predicted_rating"] = top_global["rating"]
        recommendations = top_global.fillna("").to_dict(orient="records")

    return recommendations


# =========================================
# RUN SERVER
# =========================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8001,
        reload=True
    )
