# hybrid_recommender.py
import pandas as pd
import numpy as np
from global_recommender import (
    recommend_for_user as cbf_recommend,
    recommend_cold_start,
    recommend_popular,
    df_user_product as df_cbf_interactions
)
from user_recommender import (
    recommend_products as svd_recommend,
    df_train_filtered as df_svd_train,
    common_users as svd_users
)


def hybrid_recommend(user_id, num_recommendations=10, alpha=0.6):
    """
    Hybrid = alpha * SVD + (1 - alpha) * CBF
    
    alpha adjusts based on user history:
      - New user (few interactions)  → lower alpha (lean on CBF)
      - Active user (many interactions) → higher alpha (lean on SVD)
    """
    
    # ── Check if user exists in SVD model ──────────────────
    has_svd = user_id in svd_users
    
    # ── Get user interaction count ─────────────────────────
    user_history = df_cbf_interactions[
        df_cbf_interactions["user_id"] == user_id
    ]
    n_interactions = len(user_history)
    
    # ── Cold start: no history at all ──────────────────────
    if n_interactions == 0:
        return {
            "recommendations": recommend_popular(num_recommendations=num_recommendations),
            "algorithm": "cold_start_popularity",
            "alpha": 0.0
        }
    
    # ── Adjust alpha based on history depth ────────────────
    #    Few interactions → trust CBF more
    #    Many interactions → trust SVD more
    if n_interactions < 5:
        alpha = 0.2    # Mostly CBF
    elif n_interactions < 15:
        alpha = 0.4    # Balanced
    elif n_interactions < 30:
        alpha = 0.6    # Lean SVD
    else:
        alpha = 0.8    # Heavy SVD
    
    # ── Get CBF recommendations ────────────────────────────
    cbf_recs = cbf_recommend(user_id, df_cbf_interactions, num_recommendations * 2)
    
    # ── Get SVD recommendations ────────────────────────────
    if has_svd:
        svd_recs = svd_recommend(user_id, df_svd_train, num_recommendations * 2)
    else:
        svd_recs = pd.Series(dtype=float)
    
    # ── Normalize scores to [0, 1] ─────────────────────────
    cbf_norm = normalize_scores(cbf_recs)
    svd_norm = normalize_scores(svd_recs)
    
    # ── Merge scores ───────────────────────────────────────
    all_products = set(cbf_norm.index) | set(svd_norm.index)
    
    combined = {}
    for pid in all_products:
        cbf_score = cbf_norm.get(pid, 0.0)
        svd_score = svd_norm.get(pid, 0.0)
        
        # Weighted combination
        combined[pid] = alpha * svd_score + (1 - alpha) * cbf_score
    
    # ── Sort and return top N ──────────────────────────────
    results = pd.Series(combined).sort_values(ascending=False).head(num_recommendations)
    
    # Determine which algorithm dominated
    if not has_svd:
        algo = "cbf_only"
    elif alpha >= 0.6:
        algo = "hybrid_svd_dominant"
    elif alpha <= 0.3:
        algo = "hybrid_cbf_dominant"
    else:
        algo = "hybrid_balanced"
    
    return {
        "recommendations": results,
        "algorithm": algo,
        "alpha": alpha,
        "n_interactions": n_interactions
    }


def normalize_scores(scores):
    """Min-max normalize to [0, 1]"""
    if scores.empty:
        return scores
    min_s = scores.min()
    max_s = scores.max()
    if max_s == min_s:
        return pd.Series(1.0, index=scores.index)
    return (scores - min_s) / (max_s - min_s)


# ============================================================
# EVALUATE HYBRID vs INDIVIDUAL
# ============================================================
if __name__ == "__main__":
    from global_recommender import df_test_f as cbf_test
    from user_recommender import df_test_filtered as svd_test
    
    # Use CBF test set (it has more columns)
    df_test = cbf_test
    
    print("Generating hybrid recommendations...")
    hybrid_recs = {}
    algo_counts = {}
    
    test_users = set(df_test["user_id"].unique())
    
    for i, uid in enumerate(test_users):
        result = hybrid_recommend(uid, num_recommendations=10)
        hybrid_recs[uid] = result["recommendations"]
        
        algo = result["algorithm"]
        algo_counts[algo] = algo_counts.get(algo, 0) + 1
        
        if (i + 1) % 500 == 0:
            print(f"  ... {i + 1}/{len(test_users)} users")
    
    print(f"\nHybrid done: {len(hybrid_recs)} users")
    print(f"Algorithm distribution:")
    for algo, count in sorted(algo_counts.items()):
        print(f"  {algo}: {count} users ({count/len(hybrid_recs)*100:.1f}%)")
    
    
    # ── Metrics ────────────────────────────────────────────
    def precision_at_k(recs, dft, k=6, threshold=4.0):
        precs = []
        for uid, r in recs.items():
            if isinstance(r, dict):
                r = r.get("recommendations", pd.Series(dtype=float))
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
            if isinstance(r, dict):
                r = r.get("recommendations", pd.Series(dtype=float))
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
            if isinstance(r, dict):
                r = r.get("recommendations", pd.Series(dtype=float))
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
    
    
    # ── Import individual model recommendations ───────────
    from global_recommender import user_recommendations as cbf_recs_all
    from user_recommender import user_recommendations as svd_recs_all
    
    print("\n" + "=" * 70)
    print("MODEL COMPARISON: CBF vs SVD vs HYBRID")
    print("=" * 70)
    print(f"\n{'Metric':<15} {'CBF':<12} {'SVD':<12} {'Hybrid':<12} {'Winner'}")
    print("-" * 70)
    
    for k in [5, 10]:
        p_cbf = precision_at_k(cbf_recs_all, df_test, k=k)
        p_svd = precision_at_k(svd_recs_all, df_test, k=k)
        p_hyb = precision_at_k(hybrid_recs, df_test, k=k)
        best = max(p_cbf, p_svd, p_hyb)
        winner = "CBF" if best == p_cbf else ("SVD" if best == p_svd else "Hybrid")
        print(f"P@{k:<12} {p_cbf:<12.4f} {p_svd:<12.4f} {p_hyb:<12.4f} {winner} ✅")
        
        r_cbf = recall_at_k(cbf_recs_all, df_test, k=k)
        r_svd = recall_at_k(svd_recs_all, df_test, k=k)
        r_hyb = recall_at_k(hybrid_recs, df_test, k=k)
        best = max(r_cbf, r_svd, r_hyb)
        winner = "CBF" if best == r_cbf else ("SVD" if best == r_svd else "Hybrid")
        print(f"R@{k:<12} {r_cbf:<12.4f} {r_svd:<12.4f} {r_hyb:<12.4f} {winner} ✅")
        
        n_cbf = ndcg_at_k(cbf_recs_all, df_test, k=k)
        n_svd = ndcg_at_k(svd_recs_all, df_test, k=k)
        n_hyb = ndcg_at_k(hybrid_recs, df_test, k=k)
        best = max(n_cbf, n_svd, n_hyb)
        winner = "CBF" if best == n_cbf else ("SVD" if best == n_svd else "Hybrid")
        print(f"NDCG@{k:<10} {n_cbf:<12.4f} {n_svd:<12.4f} {n_hyb:<12.4f} {winner} ✅")
        print()
    
    
    # ── Sample hybrid output ──────────────────────────────
    sample_uid = list(hybrid_recs.keys())[0]
    result = hybrid_recommend(sample_uid)
    print(f"\nSample hybrid for user {sample_uid}:")
    print(f"  Algorithm: {result['algorithm']}")
    print(f"  Alpha (SVD weight): {result['alpha']}")
    print(f"  Interactions: {result['n_interactions']}")
    for pid, score in result["recommendations"].head(5).items():
        print(f"  Product {pid:>5}  score: {score:.4f}")