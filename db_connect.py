from sqlalchemy import create_engine
from dotenv import load_dotenv
import pandas as pd
from config import DATABASE_URL

load_dotenv()

engine = create_engine(DATABASE_URL)

def query_to_df(query, params=None):
    return pd.read_sql(query, engine, params=params)