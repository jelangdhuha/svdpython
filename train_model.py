import pandas as pd
import pickle
from surprise import Dataset, Reader, SVD

# =============================
# LOAD DATA
# =============================
ratings = pd.read_csv("data/ratings_final.csv")
movies = pd.read_csv("data/movies_fixed.csv")  # DIPAKAI UNTUK MAPPING

reader = Reader(rating_scale=(1, 10))
data = Dataset.load_from_df(
    ratings[['user_id', 'item_id', 'rating']],
    reader
)

trainset = data.build_full_trainset()

# =============================
# TRAIN MODEL
# =============================
model = SVD(n_factors=100, n_epochs=20)
model.fit(trainset)

# =============================
# SAVE MODEL & MOVIES
# =============================
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

movies.to_csv("data/movies_clean.csv", index=False)

print("✅ Model & movies siap")
