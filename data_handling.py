import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.dialects.mysql import insert
from faker_data import (
    generate_user,
    generate_order,
    generate_product,
    generate_user_biases,
    generate_orderitem,
)
from config import DATABASE_URL

engine = create_engine(
    DATABASE_URL,
    pool_size=10, max_overflow=20,
    pool_timeout=60, pool_recycle=3600,
)


# ─── helpers ────────────────────────────────────────────────────────
def insert_ignore(table, conn, keys, data_iter):
    """INSERT IGNORE for MySQL (skip duplicate rows)."""
    data = [dict(zip(keys, row)) for row in data_iter]
    stmt = insert(table.table).prefix_with("IGNORE")
    conn.execute(stmt, data)


def chunked_insert(df, table_name, engine, chunksize=5000, method="multi"):
    """Write a large DataFrame in chunks to avoid packet-size limits."""
    for start in range(0, len(df), chunksize):
        chunk = df.iloc[start : start + chunksize]
        chunk.to_sql(
            table_name, con=engine,
            if_exists="append", index=False,
            method=method,
        )
    print(f"{table_name}: {len(df)} rows inserted")


# ─── main pipeline ─────────────────────────────────────────────────
try:

    # 1. USERS
    df_users = generate_user(2000)
    # Schema note: the actual table name is `users`.
    # Using `user` here will diverge from the API CRUD path, which writes to `users`.
    df_users.to_sql(
        "user", con=engine,
        if_exists="append", index=False,
        method=insert_ignore,
    )
    print("users inserted")

    # Schema note: this should stay aligned with the real users table name.
    user_ids = pd.read_sql(
        "SELECT user_id FROM user", con=engine
    )["user_id"].tolist()


    # 2. PER-USER BIASES
    user_biases = generate_user_biases(user_ids)
    print(f"biases assigned to {len(user_biases)} users")


    # 3. PRODUCTS
    df_products = generate_product(500)
    df_products.to_sql(
        "product", con=engine,
        if_exists="append", index=False,
        method="multi",
    )
    print("products inserted")
    product_ids = pd.read_sql(
        "SELECT product_id FROM product", con=engine
    )["product_id"].tolist()


    # 4. ORDERS
    df_orders = generate_order(250_000, user_ids)
    chunked_insert(df_orders, "orders", engine)

    df_orders_db = pd.read_sql(
        "SELECT order_id, user_id FROM orders", con=engine
    )
    order_ids      = df_orders_db["order_id"].tolist()
    user_order_map = dict(zip(
        df_orders_db["order_id"],
        df_orders_db["user_id"],
    ))

    # 5. ORDER ITEMS  (bias-aware)
    df_orderitems = generate_orderitem(
        250_000,
        order_ids,
        product_ids,
        user_order_map,
        user_biases,
    )
    chunked_insert(df_orderitems, "orderitem", engine)

    # 6. Persist biases for later inspection
    bias_rows = [
        {"user_id": uid, "category": cat, "sentiment": sent}
        for uid, mapping in user_biases.items()
        for cat, sent in mapping.items()
    ]
    df_biases = pd.DataFrame(bias_rows)
    df_biases.to_sql(
        "user_bias", con=engine,
        if_exists="replace", index=False,
        method="multi",
    )
    print("user_bias table written (for debugging / analysis)")

finally:
    engine.dispose()
    print("\nDone — connection pool disposed.")
