# recommender_engine.py
import pandas as pd
import numpy as np
from config import DATABASE_URL
import os
from sqlalchemy import create_engine

from global_recommender import query_to_df

class RecommendationEngine:
    def __init__(self):
        self.is_loaded = False
    
    def load(self):
        """Call once at startup — heavy computation here"""
        print("Loading recommendation engine...")
        
        # Import your existing working code
        from global_recommender import (
            cosine_sim,
            features_normalized,
            product_to_sim_idx,
            sim_idx_to_product,
            product_category,
            category_profiles,
            build_user_profile,
            df_user_product,
            get_similar_products as _get_similar,
            recommend_cold_start as _cold_start,
            recommend_popular as _popular,
        )
        from user_recommender import (
            model as svd_model,
            all_products
        )
        
        # Store references
        self.cosine_sim = cosine_sim
        self.features_normalized = features_normalized
        self.product_to_sim_idx = product_to_sim_idx
        self.sim_idx_to_product = sim_idx_to_product
        self.product_category = product_category
        self.df_user_product = df_user_product
        self.svd_model = svd_model
        self.all_products = all_products
        self._get_similar = _get_similar
        self._cold_start = _cold_start
        self._popular = _popular
        self.build_user_profile = build_user_profile
        
        self.is_loaded = True
        print("Engine loaded successfully")
    
    def recommend(self, user_id, n=10):
        """Main entry point — hybrid recommendation"""
        if not self.is_loaded:
            raise RuntimeError("Engine not loaded")
        
        from hybrid_recommender import hybrid_recommend
        result = hybrid_recommend(user_id, num_recommendations=n)
        
        recs = result["recommendations"]
        product_ids = list(recs.index)
        
        # Fetch full product details from DB
        placeholders = ",".join(str(pid) for pid in product_ids)
        df_details = query_to_df(f"""
            SELECT 
                v.product_id,
                v.product_name,
                v.category,
                v.product_description,
                v.quantity_available,
                p.price,
                v.weighted_rating,
                v.avg_rating,
                v.num_ratings
            FROM v_product_popularity v
            JOIN product p ON v.product_id = p.product_id
            WHERE v.product_id IN ({placeholders})
        """).set_index("product_id")

        return {
            "user_id": int(user_id),
            "algorithm": result["algorithm"],
            "alpha": float(result["alpha"]),
            "recommendations": [
                {
                    "product_id": int(pid),
                    "score": float(recs[pid]),
                    "product_name": str(df_details.loc[pid, "product_name"]) if pid in df_details.index else f"Product {pid}",
                    "category": str(df_details.loc[pid, "category"]) if pid in df_details.index else "unknown",
                    "product_description": str(df_details.loc[pid, "product_description"]) if pid in df_details.index else None,
                    "quantity_available": int(df_details.loc[pid, "quantity_available"]) if pid in df_details.index else 0,
                    "price": float(df_details.loc[pid, "price"]) if pid in df_details.index else 0.0,
                    "weighted_rating": float(df_details.loc[pid, "weighted_rating"]) if pid in df_details.index else 0.0,
                    "avg_rating": float(df_details.loc[pid, "avg_rating"]) if pid in df_details.index else 0.0,
                    "num_ratings": int(df_details.loc[pid, "num_ratings"]) if pid in df_details.index else 0,
                }
                for pid in product_ids
            ]
        }
    
    def similar_products(self, product_id, n=5):
        """Content-based similar products"""
        results = self._get_similar(product_id, num_recommendations=n)
        return [
            {"product_id": int(pid), "similarity": float(score)}
            for pid, score in results.items()
        ]
    
    def cold_start(self, category, n=10):
        """For new users"""
        results = self._cold_start(category, num_recommendations=n)
        return [
            {"product_id": int(pid), "score": float(score)}
            for pid, score in results.items()
        ]
    
    def popular(self, category=None, n=10):
        """Popularity-based fallback"""
        results = self._popular(
            num_recommendations=n, category=category
        )
        return [
            {"product_id": int(pid), "score": float(score)}
            for pid, score in results.items()
        ]

    def close(self):
        # Cleanup logic goes here if needed in future
        print("Engine shutting down...")
        pass

# Singleton
engine = RecommendationEngine()