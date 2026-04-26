import random
import pandas as pd
from faker import Faker

faker = Faker()

# ─── Constants ──────────────────────────────────────────────────────
CATEGORIES = ["electronics", "clothing", "books"]

BIAS_PROFILES = {
    #              (selection_weight, rating_low, rating_high)
    "strong":      (0.60, 4.0, 5.0),
    "mild":        (0.28, 2.5, 4.0),
    "dislike":     (0.12, 1.0, 2.5),
}


# ─── User ───────────────────────────────────────────────────────────
def generate_user(n):
    users = []
    for _ in range(n):
        users.append({
            "name":       faker.name(),
            "email":      faker.email(),
            "password":   faker.password(
                length=12, special_chars=True, digits=True,
                upper_case=True, lower_case=True
            ),
            "created_at": faker.date_time_this_year(),
        })
    return pd.DataFrame(users)


# ─── Order ──────────────────────────────────────────────────────────
def generate_order(n, user_ids):
    orders = []
    for _ in range(n):
        orders.append({
            "user_id":      random.choice(user_ids),
            "order_date":   faker.date_time_this_year(),
            "order_status": random.choice(
                ["pending", "processing", "shipped", "delivered", "cancelled"]
            ),
            "total_amount": round(random.uniform(10, 500), 2),
        })
    return pd.DataFrame(orders)


# ─── Product ────────────────────────────────────────────────────────
def generate_product(n):

    electronics = [
        "Wireless Bluetooth Headphones", "USB-C Charging Cable", "4K Webcam",
        "Mechanical Keyboard", "Portable SSD 1TB", "Smart Watch Series 5",
        "Noise Cancelling Earbuds", "Gaming Mouse", "Tablet Stand",
        "Wireless Charger Pad", "Bluetooth Speaker", "Action Camera",
        "Laptop Stand", "USB Hub 7-in-1", "External Hard Drive"
    ]
    clothing = [
        "Cotton Crew Neck T-Shirt", "Slim Fit Jeans", "Running Sneakers",
        "Leather Wallet", "Aviator Sunglasses", "Canvas Backpack",
        "Wool Winter Scarf", "Sports Watch", "Denim Jacket",
        "Ankle Socks Pack", "Baseball Cap", "Crossbody Bag",
        "Yoga Leggings", "Summer Dress", "Winter Coat"
    ]
    books = [
        "Python Programming Guide", "Data Science Handbook", "Sci-Fi Novel Collection",
        "Cookbook: Easy Meals", "Mystery Thriller Bestseller", "Self-Help Journal",
        "Business Strategy 101", "History of Technology", "Design Patterns",
        "Fantasy Series Starter", "Travel Photography", "Meditation Manual",
        "Machine Learning Basics", "Financial Planning", "Science Encyclopedia"
    ]
    
    # Extend lists if n > 45 (cycle through with modulo)
    all_products = (electronics * ((n//3)+1))[:n//3] + \
                   (clothing * ((n//3)+1))[:n//3] + \
                   (books * ((n//3)+1))[:n//3]
    
    products = []
    for i in range(n):
        category_idx = i % 3
        product_name = all_products[i] if i < len(all_products) else f"Product {i}"
        
        products.append({
            "product_name":               product_name,
            "product_description":        f"High quality {['electronics','clothing','books'][category_idx]} item",
            "price":              round(random.uniform(10, 500), 2),
            "quantity_available": random.randint(0, 100),
            "created_at":         faker.date_time_this_year(),
        })
    return pd.DataFrame(products)


# ─── Per-User Bias ──────────────────────────────────────────────────
def generate_user_biases(user_ids):
    """
    Assign each user a *unique* random permutation of
    [strong, mild, dislike] across the 3 categories.

    Returns
    -------
    dict : { user_id: {"electronics": "strong",
                        "clothing":   "dislike",
                        "books":      "mild"}, … }
    """
    sentiments = ["strong", "mild", "dislike"]
    biases = {}
    for uid in user_ids:
        perm = random.sample(sentiments, k=3)
        biases[uid] = dict(zip(CATEGORIES, perm))
    return biases


# ─── Order Item (bias-aware) ────────────────────────────────────────
def generate_orderitem(n, order_ids, product_ids, user_order_map, user_biases):
    """
    Parameters
    ----------
    n              : total rows to generate
    order_ids      : list of valid order PKs
    product_ids    : list of valid product PKs (split into 3 equal segments)
    user_order_map : { order_id → user_id }
    user_biases    : output of generate_user_biases()
    """

    # ── split products into 3 category pools ──────────────────────
    k = len(product_ids) // 3
    category_pools = {
        "electronics": product_ids[:k],
        "clothing":    product_ids[k : 2 * k],
        "books":       product_ids[2 * k :],
    }

    # ── shared favourites (collaborative-filtering signal) ────────
    cluster_favorites = {
        cat: pool[:10] for cat, pool in category_pools.items()
    }
    sub_cluster_favs = {}
    for cat, pool in category_pools.items():
        sub_cluster_favs[(cat, 0)] = pool[:7]        # sub-group A
        sub_cluster_favs[(cat, 1)] = pool[3:10]      # sub-group B (overlapping)

    # ── pre-compute per-user weighted category lists ──────────────
    user_weights = {}
    for uid, bias_map in user_biases.items():
        cats, weights = [], []
        for cat in CATEGORIES:
            cats.append(cat)
            weights.append(BIAS_PROFILES[bias_map[cat]][0])
        user_weights[uid] = (cats, weights)

    # ── generate rows ─────────────────────────────────────────────
    orderitems = []
    for _ in range(n):
        order_id = random.choice(order_ids)
        user_id  = user_order_map[order_id]
        bias_map = user_biases[user_id]

        # 1) pick category (weighted by user bias) ─────────────────
        cats, weights = user_weights[user_id]
        chosen_cat    = random.choices(cats, weights=weights, k=1)[0]
        sentiment     = bias_map[chosen_cat]
        _, lo, hi     = BIAS_PROFILES[sentiment]

        pool     = category_pools[chosen_cat]
        favs     = cluster_favorites[chosen_cat]
        sub_id   = (user_id // 3) % 2
        sub_favs = sub_cluster_favs[(chosen_cat, sub_id)]

        # 2) pick product & base score ─────────────────────────────
        if sentiment == "strong":
            r = random.random()
            if r < 0.45:
                product_id = random.choice(favs)
                base_score = random.uniform(4.3, 5.0)
            elif r < 0.75:
                product_id = random.choice(sub_favs)
                base_score = random.uniform(4.0, 4.8)
            else:
                product_id = random.choice(pool)
                base_score = random.uniform(lo, hi)

        elif sentiment == "mild":
            r = random.random()
            if r < 0.30:
                product_id = random.choice(favs)
                base_score = random.uniform(3.2, 4.0)
            else:
                product_id = random.choice(pool)
                base_score = random.uniform(lo, hi)

        else:                                       # dislike
            product_id = random.choice(pool)
            base_score = random.uniform(lo, hi)

        rating_score = round(min(5.0, max(1.0, base_score)), 1)

        orderitems.append({
            "order_id":     order_id,
            "product_id":   product_id,
            "quantity":     random.randint(1, 10),
            "rating_score": rating_score,
            "unit_price":   round(random.uniform(10, 500), 2),
            "category":     chosen_cat,
        })

    return pd.DataFrame(orderitems)
