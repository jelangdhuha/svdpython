import pandas as pd
import pickle
from surprise import Dataset, Reader, SVD

# =============================
# LOAD DATA
# =============================
ratings = pd.read_csv("data/ratings_final_fixed.csv")
movies = pd.read_csv("data/movies_fixed.csv")  

reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(
    ratings[['user_id', 'item_id', 'rating']],
    reader
)

# Menggunakan hyperparameter yang lebih ketat dari gambar untuk mencegah overfitting
algo_eval = SVD(n_epochs=40, lr_all=0.005, reg_all=0.05)

print("\n📊 Sedang melakukan Cross-Validation (Cek Error)...")
results = cross_validate(algo_eval, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

print("\n" + "="*30)
print(f"🏆 RATA-RATA RMSE: {results['test_rmse'].mean():.4f}")
print(f"🏆 RATA-RATA MAE : {results['test_mae'].mean():.4f}")
print("="*30)


# =============================
# TRAIN MODEL
# =============================
# model = SVD(n_factors=100, n_epochs=20)
trainset = data.build_full_trainset()
model = SVD(n_epochs=40, lr_all=0.005, reg_all=0.05)
model.fit(trainset)

# =============================
# SAVE MODEL & MOVIES
# =============================
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

movies.to_csv("data/movies_clean.csv", index=False)

print("✅ Model & movies siap")
