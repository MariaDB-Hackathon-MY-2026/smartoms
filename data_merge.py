import pandas as pd
from sqlalchemy import create_engine
from load_data import (
    df_users,
    df_products,
    df_orders,
    df_orderitems
)
from config import DATABASE_URL

engine = create_engine(
    DATABASE_URL, pool_size=10, max_overflow=20,
    pool_timeout=60, pool_recycle=3600
)

# Merge table Orders, User and Product
# user_id + product_id + product_name + quantity + price + order_date
df_order_details = df_orderitems.merge(
    df_orders[["order_id", "user_id", "order_date"]],
    on="order_id"
)

df_order_details = df_order_details.merge(
    df_products[["product_id", "product_name", "price"]],
    on="product_id"
)


# Merge table User and Orders
# Shows customer order history
# user_id + name + order_id + order_date + total_amount
df_user_orders = df_orders.merge(
    df_users[["user_id", "name"]],
    on="user_id"
)


# Merge table related to Orders
# user_name + product_name + quantity + price + order_date
df_full_orders = df_orders.merge(
    df_users[["user_id", "name"]],
    on="user_id"
).merge(
    df_orderitems,
    on="order_id"
).merge(
    df_products[["product_id", "product_name", "price"]],
    on="product_id"
)


# Merge table Orders and Order Item for ML
# user + product + rating
df_user_interactions = df_orderitems.merge(
    df_orders[['order_id', 'user_id']],
    on='order_id'
)
print(df_user_interactions)
