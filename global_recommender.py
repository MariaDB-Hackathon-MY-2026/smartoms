import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sqlalchemy import text
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix
from dotenv import load_dotenv
from config import DATABASE_URL

load_dotenv()

# ============================================================
# DATABASE CONNECTION
# ============================================================

engine = create_engine(DATABASE_URL)

def query_to_df(query, params=None):
    return pd.read_sql(query, engine, params=params)

# ============================================================
# LOAD DATA FROM MARIADB
# ============================================================

print("Loading data from MariaDB...")

# Interaction data
df = query_to_df("""
    SELECT 
        oi.order_item_id,
        oi.order_id,
        oi.product_id,
        oi.quantity,
        oi.rating_score,
        oi.unit_price,
        oi.category,
        o.user_id
    FROM orderitem oi
    JOIN orders o ON oi.order_id = o.order_id
""")
print(f"Interactions loaded: {len(df)}")

# Product features PRE-COMPUTED by MariaDB
df_db_features = query_to_df("SELECT * FROM v_product_features")
print(f"Product features from DB: {df_db_features.shape}")

# Popularity rankings PRE-COMPUTED by MariaDB
df_popularity = query_to_df("""
    SELECT * FROM v_product_popularity 
    ORDER BY overall_popularity_rank
""")
print(f"Popularity rankings from DB: {len(df_popularity)}")

product_ids = df["product_id"].unique()
n_products = len(product_ids)
print(f"Products: {n_products}, Users: {df['user_id'].nunique()}")


# ============================================================
# BUILD FEATURE MATRIX (MariaDB features + Python SVD)
# ============================================================

feature_frames = []
db_features = df_db_features.set_index("product_id")

# ----------------------------------------------------------
# 1. CATEGORY (one-hot, weighted) — from DB
# ----------------------------------------------------------
category_dummies = pd.get_dummies(
    db_features["category"], prefix="cat"
) * 5.0
feature_frames.append(category_dummies)
print(f"1. Category features: {category_dummies.shape[1]} cols")

# ----------------------------------------------------------
# 2. DB-COMPUTED FEATURES (from v_product_features view)
# ----------------------------------------------------------
db_numeric = db_features[[
    "avg_rating", "num_ratings", "unique_buyers",
    "total_quantity", "avg_price", "weighted_rating",
    "buyer_percentile", "category_rating_percentile",
    "pct_5star", "pct_4plus", "pct_2minus",
    "rating_std", "avg_quantity_per_order", "bulk_purchase_pct"
]].fillna(0)
feature_frames.append(db_numeric)
print(f"2. DB-computed features: {db_numeric.shape[1]} cols")

# ----------------------------------------------------------
# 3. PRICE TIER (one-hot from DB-computed tier)
# ----------------------------------------------------------
price_tier_dummies = pd.get_dummies(
    db_features["price_tier"], prefix="ptier"
) * 2.0

feature_frames.append(price_tier_dummies)
print(f"3. Price tier features: {price_tier_dummies.shape[1]} cols")

# ----------------------------------------------------------
# 4. CROSS FEATURES (category × key metrics)
# ----------------------------------------------------------
cross_features = pd.DataFrame(index=db_features.index)
for cat in category_dummies.columns:
    cat_name = cat.replace("cat_", "")
    mask = (db_features["category"] == cat_name).astype(float)
    cross_features[f"cross_{cat_name}_avgprice"] = (
        mask * db_features["avg_price"]
    )
    cross_features[f"cross_{cat_name}_avgrating"] = (
        mask * db_features["avg_rating"]
    )
cross_features = cross_features.fillna(0)
feature_frames.append(cross_features)
print(f"4. Cross features: {cross_features.shape[1]} cols")

# ----------------------------------------------------------
# 5. SVD EMBEDDINGS (computed in Python — ML operation)
# ----------------------------------------------------------
user_ids = df["user_id"].unique()
user_to_idx = {u: i for i, u in enumerate(user_ids)}
prod_to_idx = {p: i for i, p in enumerate(product_ids)}

rows = df["user_id"].map(user_to_idx).values
cols = df["product_id"].map(prod_to_idx).values
vals = df["rating_score"].values

user_product_matrix = csr_matrix(
    (vals, (rows, cols)),
    shape=(len(user_ids), len(product_ids))
)

n_components = min(20, n_products - 1, len(user_ids) - 1)
svd = TruncatedSVD(n_components=n_components, random_state=42)
product_embeddings = svd.fit_transform(user_product_matrix.T)

svd_df = pd.DataFrame(
    product_embeddings * 0.3,
    index=product_ids,
    columns=[f"svd_{i}" for i in range(n_components)]
)
feature_frames.append(svd_df)
print(f"5. SVD features: {n_components} cols "
      f"(variance: {svd.explained_variance_ratio_.sum():.3f})")


# ============================================================
# COMBINE & NORMALIZE
# ============================================================

df_product_features = pd.concat(feature_frames, axis=1, join="inner")
df_product_features = df_product_features.fillna(0)

print(f"\n{'='*50}")
print(f"FINAL FEATURE MATRIX: {df_product_features.shape}")
print(f"{'='*50}")

scaler = StandardScaler()
features_scaled = scaler.fit_transform(df_product_features)
features_normalized = normalize(features_scaled, norm="l2")


# ============================================================
# COSINE SIMILARITY MATRIX
# ============================================================

cosine_sim = cosine_similarity(features_normalized)

mask = ~np.eye(cosine_sim.shape[0], dtype=bool)
print(f"\nSimilarity distribution:")
print(f"  Mean: {cosine_sim[mask].mean():.4f}")
print(f"  Std:  {cosine_sim[mask].std():.4f}")
print(f"  Min:  {cosine_sim[mask].min():.4f}")
print(f"  Max:  {cosine_sim[mask].max():.4f}")

# Within vs cross category check
categories = db_features.loc[df_product_features.index, "category"]
for cat in categories.unique():
    cat_mask = (categories == cat).values
    within = cosine_sim[np.ix_(cat_mask, cat_mask)]
    within_no_diag = within[~np.eye(within.shape[0], dtype=bool)]
    cross = cosine_sim[np.ix_(cat_mask, ~cat_mask)]
    print(f"  {cat:>12}: within={within_no_diag.mean():.4f}, "
          f"cross={cross.mean():.4f}, "
          f"gap={within_no_diag.mean() - cross.mean():.4f}")

# Mappings
product_list = df_product_features.index.tolist()
product_to_sim_idx = {p: i for i, p in enumerate(product_list)}
sim_idx_to_product = {i: p for i, p in enumerate(product_list)}

# Product category lookup
product_category = db_features[["category"]]

# Popularity lookup (from DB)
pop_data = df_popularity.set_index("product_id")


# ============================================================
# COLD START (MariaDB stored procedure + fallback)
# ============================================================

# Category profiles for Python-based cold start fallback
category_profiles = {}
for cat in df["category"].unique():
    cat_products = product_category[
        product_category["category"] == cat
    ].index
    cat_indices = [
        product_to_sim_idx[p] for p in cat_products
        if p in product_to_sim_idx
    ]
    if cat_indices:
        category_profiles[cat] = features_normalized[cat_indices].mean(axis=0)


def recommend_cold_start(category, num_recommendations=10):
    try:
        with engine.connect() as conn:
            result = conn.execute(
                text("CALL sp_cold_start_recommendations(:cat, :lim)"),
                {"cat": category, "lim": num_recommendations}
            )
            df_results = pd.DataFrame(result.fetchall(), columns=result.keys())

        if not df_results.empty:
            return pd.Series(
                df_results["weighted_rating"].values,
                index=df_results["product_id"].values
            )

    except Exception as e:
        print(f"DB cold start failed ({e}), using Python fallback")

    # Python fallback using cosine similarity
    if category not in category_profiles:
        return recommend_popular(num_recommendations=num_recommendations)

    profile = category_profiles[category].reshape(1, -1)
    sim_scores = cosine_similarity(
        profile, features_normalized
    ).flatten()

    scored = [
        (sim_idx_to_product[i], sim_scores[i])
        for i in range(len(sim_scores))
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    scored = scored[:num_recommendations]

    return pd.Series(
        [s[1] for s in scored],
        index=[s[0] for s in scored]
    )


# ============================================================
# POPULARITY RECOMMENDATIONS (from MariaDB)
# ============================================================

def recommend_popular(user_id=None, df_interactions=None,
                      num_recommendations=10, category=None):
    """Popularity-based recommendations using MariaDB rankings."""
    if category:
        pop = query_to_df("""
            SELECT * FROM v_product_popularity 
            WHERE category = %s
            ORDER BY category_popularity_rank
        """, (category,))
    else:
        pop = query_to_df("""
            SELECT * FROM v_product_popularity 
            ORDER BY overall_popularity_rank
        """)

    if user_id is not None and df_interactions is not None:
        seen = set(
            df_interactions[
                df_interactions["user_id"] == user_id
            ]["product_id"]
        )
        pop = pop[~pop["product_id"].isin(seen)]

    pop = pop.head(num_recommendations)
    return pd.Series(
        pop["weighted_rating"].values,
        index=pop["product_id"].values
    )


# ============================================================
# USER PROFILE (improved v3 — positive/negative weighting)
# ============================================================

df_user_product = (
    df.groupby(["user_id", "product_id"], as_index=False)
      .agg(
          rating_score=("rating_score", "mean"),
          purchase_count=("order_id", "nunique"),
          total_quantity=("quantity", "sum"),
          avg_price=("unit_price", "mean"),
      )
      .merge(product_category.reset_index(), on="product_id", how="left")
)


def build_user_profile(user_id, df_interactions):
    """
    Build user profile with positive/negative weighting.
    High-rated + repeat-purchased items dominate the profile.
    Low-rated items push the profile away.
    """
    user_data = df_interactions[df_interactions["user_id"] == user_id]
    if user_data.empty:
        return None

    positive = np.zeros(features_normalized.shape[1])
    negative = np.zeros(features_normalized.shape[1])
    pos_weight = 0
    neg_weight = 0

    for _, row in user_data.iterrows():
        pid = row["product_id"]
        rating = row["rating_score"]

        if pid not in product_to_sim_idx:
            continue

        idx = product_to_sim_idx[pid]
        vec = features_normalized[idx]

        purchase_bonus = 1.0
        if "purchase_count" in row.index:
            purchase_bonus = 1.0 + np.log1p(row["purchase_count"])

        if rating >= 3.5:
            weight = ((rating - 2.0) / 3.0) * purchase_bonus
            positive += weight * vec
            pos_weight += weight
        elif rating <= 2.5:
            weight = ((3.0 - rating) / 3.0) * purchase_bonus
            negative += weight * vec
            neg_weight += weight

    if pos_weight > 0:
        positive /= pos_weight
    if neg_weight > 0:
        negative /= neg_weight

    profile = positive - 0.4 * negative
    norm = np.linalg.norm(profile)
    if norm > 0:
        profile = profile / norm

    return profile


# ============================================================
# RECOMMENDATIONS (with category boost + MMR diversity)
# ============================================================

def recommend_for_user(user_id, df_interactions, num_recommendations=10):
    """CBF recommendations with diversity re-ranking."""
    profile = build_user_profile(user_id, df_interactions)
    if profile is None:
        return pd.Series(dtype=float)

    content_scores = cosine_similarity(
        profile.reshape(1, -1), features_normalized
    ).flatten()

    user_data = df_interactions[df_interactions["user_id"] == user_id]
    seen = set(user_data["product_id"])

    user_cats = (
        user_data.groupby("category")["rating_score"]
        .mean()
    )

    candidates = []
    seen_pids = set()
    for i, score in enumerate(content_scores):
        pid = sim_idx_to_product[i]
        if pid in seen or pid in seen_pids:
            continue
        seen_pids.add(pid)

        prod_cat = product_category.loc[pid, "category"]

        # Category preference boost
        cat_boost = 1.0
        if prod_cat in user_cats.index:
            cat_boost = 1.0 + 0.5 * (user_cats[prod_cat] / 5.0)

        # Popularity boost from MariaDB rankings
        pop_boost = 1.0
        if pid in pop_data.index:
            pop_rank = pop_data.loc[pid, "buyer_percentile"]
            pop_boost = 1.0 + 0.1 * (1 - pop_rank)

        pop_penalty = 1.0
        if pid in pop_data.index:
            pop_rank = pop_data.loc[pid, "buyer_percentile"]
            pop_penalty = 0.9 + 0.1 * (1 - pop_rank)

        final_score = score * cat_boost * pop_penalty
        candidates.append((pid, final_score, prod_cat))

    candidates.sort(key=lambda x: x[1], reverse=True)

    # MMR diversity re-ranking
    selected = []
    remaining = candidates[:num_recommendations * 3]
    lambda_div = 0.35

    while len(selected) < num_recommendations and remaining:
        if not selected:
            selected.append(remaining.pop(0))
            continue

        selected_indices = [
            product_to_sim_idx[s[0]] for s in selected
        ]

        best_score = -1
        best_idx = 0

        for j, (pid, score, cat) in enumerate(remaining):
            pidx = product_to_sim_idx[pid]
            max_sim = max(
                cosine_sim[pidx][si] for si in selected_indices
            )
            mmr_score = (1 - lambda_div) * score - lambda_div * max_sim

            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = j

        selected.append(remaining.pop(best_idx))

    return pd.Series(
        [s[1] for s in selected],
        index=[s[0] for s in selected]
    )


def get_similar_products(product_id, num_recommendations=10):
    """Get similar products based on content similarity."""
    if product_id not in product_to_sim_idx:
        return pd.Series(dtype=float)

    idx = product_to_sim_idx[product_id]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores.sort(key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:num_recommendations + 1]

    return pd.Series(
        [s[1] for s in sim_scores],
        index=[sim_idx_to_product[s[0]] for s in sim_scores]
    )


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

def split_by_user(dataframe, test_size=0.2, min_interactions=2,
                  random_state=100):
    np.random.seed(random_state)
    train_idx, test_idx = [], []
    for _, group in dataframe.groupby("user_id"):
        indices = group.index.tolist()
        n = len(indices)
        if n < min_interactions:
            train_idx.extend(indices)
        else:
            np.random.shuffle(indices)
            n_test = max(1, int(n * test_size))
            test_idx.extend(indices[:n_test])
            train_idx.extend(indices[n_test:])
    return train_idx, test_idx


train_idx, test_idx = split_by_user(df_user_product)
df_train = df_user_product.loc[train_idx].reset_index(drop=True)
df_test = df_user_product.loc[test_idx].reset_index(drop=True)

common_users = (
    set(df_train["user_id"].unique()) &
    set(df_test["user_id"].unique())
)
df_train_f = df_train[df_train["user_id"].isin(common_users)]
df_test_f = df_test[df_test["user_id"].isin(common_users)]

print(f"\nTrain: {len(df_train_f)}, Test: {len(df_test_f)}, "
      f"Users: {len(common_users)}")


# ============================================================
# GENERATE RECOMMENDATIONS
# ============================================================

print("\nGenerating CBF recommendations...")
user_recommendations = {}
for i, uid in enumerate(common_users):
    recs = recommend_for_user(uid, df_train_f, num_recommendations=10)
    if not recs.empty:
        user_recommendations[uid] = recs
    # Progress indicator every 500 users
    if (i + 1) % 500 == 0:
        print(f"  ... {i + 1}/{len(common_users)} users")

print(f"CBF done: {len(user_recommendations)} users")

print("Generating popularity recommendations...")
pop_recommendations = {}

# Pre-convert df_popularity to dict for faster lookup
pop_dict = df_popularity.set_index("product_id")["weighted_rating"].to_dict()
pop_list = df_popularity["product_id"].tolist()

for i, uid in enumerate(common_users):
    # Get seen items for this user (fast set operation)
    seen = set(
        df_train_f[df_train_f["user_id"] == uid]["product_id"]
    )
    
    # Filter and take top 10 (fast list comprehension)
    recs = [
        (pid, pop_dict[pid]) 
        for pid in pop_list 
        if pid not in seen
    ][:10]
    
    if recs:
        pop_recommendations[uid] = pd.Series(
            [r[1] for r in recs],
            index=[r[0] for r in recs]
        )
    
    # Progress indicator every 500 users
    if (i + 1) % 500 == 0:
        print(f"  ... {i + 1}/{len(common_users)} users")

print(f"Popularity done: {len(pop_recommendations)} users")

# ============================================================
# EVALUATION METRICS
# ============================================================

def precision_at_k(recs, dft, k=6, threshold=4.0):
    precs = []
    for uid, r in recs.items():
        top_k = r.head(k).index.tolist()
        relevant = set(dft[
            (dft["user_id"] == uid) & (dft["rating_score"] >= threshold)
        ]["product_id"])
        if not relevant:
            continue
        hits = sum(1 for p in top_k if p in relevant)
        precs.append(hits / k)
    return np.mean(precs) if precs else 0.0


def recall_at_k(recs, dft, k=6, threshold=4.0):
    recalls = []
    for uid, r in recs.items():
        top_k = r.head(k).index.tolist()
        relevant = set(dft[
            (dft["user_id"] == uid) & (dft["rating_score"] >= threshold)
        ]["product_id"])
        if not relevant:
            continue
        hits = sum(1 for p in top_k if p in relevant)
        recalls.append(hits / len(relevant))
    return np.mean(recalls) if recalls else 0.0


def ndcg_at_k(recs, dft, k=6, threshold=4.0):
    ndcgs = []
    for uid, r in recs.items():
        top_k = r.head(k).index.tolist()
        relevant = set(dft[
            (dft["user_id"] == uid) & (dft["rating_score"] >= threshold)
        ]["product_id"])
        if not relevant:
            continue
        dcg = sum(
            1 / np.log2(i + 2)
            for i, p in enumerate(top_k) if p in relevant
        )
        idcg = sum(
            1 / np.log2(i + 2)
            for i in range(min(len(relevant), k))
        )
        if idcg > 0:
            ndcgs.append(dcg / idcg)
    return np.mean(ndcgs) if ndcgs else 0.0


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 60)
print("MODEL COMPARISON: CBF vs POPULARITY BASELINE")
print("=" * 60)
print(f"\n{'Metric':<15} {'Popularity':<15} {'CBF':<15} {'Winner'}")
print("-" * 60)

for k in [5, 10, 11]:
    p_pop = precision_at_k(pop_recommendations, df_test_f, k=k)
    p_cbf = precision_at_k(user_recommendations, df_test_f, k=k)
    winner = "CBF" if p_cbf > p_pop else "POP"
    print(f"P@{k:<12} {p_pop:<15.4f} {p_cbf:<15.4f} {winner}")

    r_pop = recall_at_k(pop_recommendations, df_test_f, k=k)
    r_cbf = recall_at_k(user_recommendations, df_test_f, k=k)
    winner = "CBF" if r_cbf > r_pop else "POP"
    print(f"R@{k:<12} {r_pop:<15.4f} {r_cbf:<15.4f} {winner}")

    n_pop = ndcg_at_k(pop_recommendations, df_test_f, k=k)
    n_cbf = ndcg_at_k(user_recommendations, df_test_f, k=k)
    winner = "CBF" if n_cbf > n_pop else "POP"
    print(f"NDCG@{k:<10} {n_pop:<15.4f} {n_cbf:<15.4f} {winner}")
    print()


# Sample recommendations
if user_recommendations:
    uid = list(user_recommendations.keys())[0]
    print(f"\nSample recs for user {uid}:")
    for pid, score in list(user_recommendations[uid].head(4).items()):
        cat = product_category.loc[pid, "category"]
        name = db_features.loc[pid, "product_name"] if "product_name" in db_features.columns else "N/A"
        print(f"  Product {pid:>6} [{cat:>12}] {name:>20}  "
              f"score: {score:.4f}")


# Cold start demo
print(f"\n{'='*50}")
print("COLD START (MariaDB Stored Procedure)")
print(f"{'='*50}")
for cat in df["category"].unique():
    print(f"\n'{cat}':")
    recs = recommend_cold_start(cat, 5)
    for pid, score in recs.items():
        name = db_features.loc[pid, "product_name"] if pid in db_features.index else "?"
        print(f"  Product {pid:>6} [{name:>15}]  score: {score:.4f}")