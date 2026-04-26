import pandas as pd
from sqlalchemy import create_engine
from config import DATABASE_URL

engine = create_engine(DATABASE_URL, pool_size=10, max_overflow=20, pool_timeout=60, pool_recycle=3600)

# Reads data from database

# Users table note: the database table is `users`.
# Keep this pluralized so it stays consistent with API user creation.
query_users = "SELECT * FROM users"
df_users = pd.read_sql(query_users, engine)

# Product
query_products = "SELECT * FROM product"
df_products = pd.read_sql(query_products, engine)

# Orders
query_orders = "SELECT * FROM orders"
df_orders = pd.read_sql(query_orders, engine)

# Order Item
query_orderitems = "SELECT * FROM orderitem"
df_orderitems = pd.read_sql(query_orderitems, engine)

# Check for wrong values
# print(df_orders.isnull().sum())
