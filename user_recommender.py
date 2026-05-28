import pandas as pd
import numpy as np
from surprise import SVD, Dataset, Reader
from surprise.model_selection import cross_validate
from data_merge import df_user_interactions

# ============================================================
# ONE row per (user, product) → no duplicates across train/test
# ============================================================
df_surprise = (
    df_user_interactions
    .groupby(["user_id", "product_id"], as_index=False)
    .agg(rating_score=("rating_score", "mean"))
)
df_surprise["rating_score"] = df_surprise["rating_score"].round(1)

reader = Reader(rating_scale=(1, 5))


# ============================================================
# TRAIN / TEST SPLIT — within each user
# ============================================================
def split_by_user(df, test_size=0.2, min_interactions=2, random_state=100):
    np.random.seed(random_state)
    train_indices, test_indices = [], []

    for _, group in df.groupby("user_id"):
        indices = group.index.tolist()
        n = len(indices)
        if n < min_interactions:
            train_indices.extend(indices)
        else:
            np.random.shuffle(indices)
            n_test = max(1, int(n * test_size))
            test_indices.extend(indices[:n_test])
            train_indices.extend(indices[n_test:])

    return train_indices, test_indices


train_idx, test_idx = split_by_user(df_surprise)

df_train = df_surprise.loc[train_idx].reset_index(drop=True)
df_test  = df_surprise.loc[test_idx].reset_index(drop=True)

common_users = set(df_train["user_id"].unique()) & set(df_test["user_id"].unique())
df_train_filtered = df_train[df_train["user_id"].isin(common_users)]
df_test_filtered  = df_test[df_test["user_id"].isin(common_users)]

print(f"Train size:   {len(df_train_filtered)}")
print(f"Test size:    {len(df_test_filtered)}")
print(f"Common users: {len(common_users)}")

if len(common_users) == 0:
    raise ValueError("No common users — check data volume / min_interactions.")


# ============================================================
# TRAIN
# ============================================================
train_data = Dataset.load_from_df(
    df_train_filtered[["user_id", "product_id", "rating_score"]], reader
)
trainset = train_data.build_full_trainset()

model = SVD(n_factors=20, n_epochs=20, random_state=100,
            lr_all=0.005, reg_all=0.02)
model.fit(trainset)

all_products = df_surprise["product_id"].unique()


# ============================================================
# RECOMMEND (predict on products unseen in train)
# ============================================================
def recommend_products(user_id, df_train, num_recommendations=10):
    seen = set(df_train[df_train["user_id"] == user_id]["product_id"])
    preds = [
        (pid, model.predict(user_id, pid).est)
        for pid in all_products if pid not in seen
    ]
    preds.sort(key=lambda x: x[1], reverse=True)
    return pd.Series(dict(preds[:num_recommendations]))


user_recommendations = {
    uid: recommend_products(uid, df_train_filtered)
    for uid in common_users
}
print(f"Recommendations generated for {len(user_recommendations)} users")


# ============================================================
# EVALUATE
# ============================================================
def precision_at_k(recs, df_test, k=6, threshold=4.0):
    precs = []
    for uid, r in recs.items():
        top_k = r.head(k).index
        relevant = set(df_test[
            (df_test["user_id"] == uid) &
            (df_test["rating_score"] >= threshold)
        ]["product_id"])
        if not relevant:
            continue
        precs.append(sum(p in relevant for p in top_k) / k)
    return np.mean(precs) if precs else 0.0


def recall_at_k(recs, df_test, k=6, threshold=4.0):
    recalls = []
    for uid, r in recs.items():
        top_k = r.head(k).index
        relevant = set(df_test[
            (df_test["user_id"] == uid) &
            (df_test["rating_score"] >= threshold)
        ]["product_id"])
        if not relevant:
            continue
        recalls.append(sum(p in relevant for p in top_k) / len(relevant))
    return np.mean(recalls) if recalls else 0.0


p6 = precision_at_k(user_recommendations, df_test_filtered, k=6)
r6 = recall_at_k(user_recommendations, df_test_filtered, k=6)
f1 = 2 * p6 * r6 / (p6 + r6) if (p6 + r6) > 0 else 0.0

print(f"\nPrecision@6: {p6:.4f}")
print(f"Recall@6:    {r6:.4f}")
print(f"F1@6:        {f1:.4f}")


# ============================================================
# CROSS VALIDATION (full data, separate model)
# ============================================================
full_data = Dataset.load_from_df(
    df_surprise[["user_id", "product_id", "rating_score"]], reader
)
results = cross_validate(
    SVD(n_factors=10, n_epochs=20, random_state=100),
    full_data, measures=["RMSE"], cv=5, verbose=True,
)