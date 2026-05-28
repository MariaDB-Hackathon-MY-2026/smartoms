from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import redis
import json
import uvicorn
from dotenv import load_dotenv
from config import DATABASE_URL
from sqlalchemy import create_engine, text
from recommender_engine import RecommendationEngine
from pydantic import BaseModel
from typing import Optional

load_dotenv()

# SQLAlchemy engine 
db_engine = create_engine(DATABASE_URL, pool_size=10, max_overflow=20)

# Redis connection
r = redis.Redis(host='localhost', port=6379, db=0)

# Recommendation engine instance (will be initialized in lifespan)
rec_engine = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global rec_engine
    rec_engine = RecommendationEngine()
    
    print("Loading engine (this might take a moment)...")
    rec_engine.load()  # Removed 'await' - runs synchronously
    
    yield
    
    if rec_engine:
        rec_engine.close() # Removed 'await'

# Initialize FastAPI with lifespan
app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# SCHEMAS
# ============================================================

class UserCreate(BaseModel):
    name: str
    email: str
    password: str

class UserLogin(BaseModel):
    email: str
    password: str

class OrderCreate(BaseModel):
    user_id: int
    product_id: int
    quantity: int
    rating_score: float

class ViewCreate(BaseModel):
    user_id: int
    product_id: int

class ProductCreate(BaseModel):
    product_name: str
    category: str

# ============================================================
# ENDPOINTS
# ============================================================

@app.post("/api/users")
async def create_user(user: UserCreate):
    with db_engine.connect() as conn:
        result = conn.execute(
            text("""
                INSERT INTO users (name, email, password, created_at)
                VALUES (:name, :email, :password, NOW())
            """),
            {"name": user.name, "email": user.email, "password": user.password}
        )
        conn.commit()
        return {"user_id": result.lastrowid, "message": "User created"}

@app.post("/api/login")
async def login_user(user: UserLogin):
    with db_engine.connect() as conn:
        result = conn.execute(
            text("""
                SELECT user_id, name, email, created_at 
                FROM users 
                WHERE email = :email AND password = :password
            """),
            {"email": user.email, "password": user.password}
        )
        row = result.fetchone()
        
        if not row:
            raise HTTPException(status_code=401, detail="Invalid email or password")
            
        return {
            "user_id": row[0],
            "name": row[1],
            "email": row[2],
            "created_at": row[3].isoformat()
        }

@app.get("/api/recommendations/{user_id}")
async def get_recommendations(user_id: int, n: int = 10, category: str = None):
    result = rec_engine.recommend(user_id, n)
    
    # Filter by category if provided
    if category:
        result["recommendations"] = [
            r for r in result["recommendations"]
            if r["category"] == category
        ]
    
    return result

@app.get("/api/products")
async def get_products(category: str = None, limit: int = 20):
    with db_engine.connect() as conn:
        # JOIN the new 'product' table with the features view
        if category:
            query = text("""
                SELECT 
                    p.product_id, p.product_name, p.product_description, 
                    p.price, p.quantity_available,
                    pf.category, pf.avg_rating, pf.num_ratings
                FROM product p
                JOIN v_product_features pf ON p.product_id = pf.product_id
                WHERE pf.category = :cat 
                LIMIT :lim
            """)
            params = {"cat": category, "lim": limit}
        else:
            query = text("""
                SELECT 
                    p.product_id, p.product_name, p.product_description, 
                    p.price, p.quantity_available,
                    pf.category, pf.avg_rating, pf.num_ratings
                FROM product p
                JOIN v_product_features pf ON p.product_id = pf.product_id
                LIMIT :lim
            """)
            params = {"lim": limit}
            
        result = conn.execute(query, params)
        
        # Convert to list of dictionaries (so JSON can read it)
        products = [dict(row._mapping) for row in result]
        return {"products": products}

@app.get("/api/products/{product_id}")
async def get_product(product_id: int):
    with db_engine.connect() as conn:
        result = conn.execute(
            text("SELECT * FROM v_product_features WHERE product_id = :pid"),
            {"pid": product_id}
        )
        row = result.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Product not found")
        return dict(row._mapping)

@app.get("/api/users/{user_id}/history")
async def get_user_history(user_id: int):
    with db_engine.connect() as conn:
        result = conn.execute(
            text("""
                SELECT oi.product_id, p.product_name, oi.category,
                       oi.quantity, oi.rating_score, oi.unit_price,
                       o.order_date
                FROM orderitem oi
                JOIN orders o ON oi.order_id = o.order_id
                JOIN product p ON oi.product_id = p.product_id
                WHERE o.user_id = :uid
                ORDER BY o.order_date DESC
                LIMIT 50
            """),
            {"uid": user_id}
        )
        history = [dict(row._mapping) for row in result]
        return {"user_id": user_id, "history": history}

@app.post("/api/orders")
async def create_order(order: OrderCreate):
    with db_engine.connect() as conn:
        result = conn.execute(
            text("""
                INSERT INTO orders (user_id, order_date, order_status, total_amount)
                VALUES (:uid, NOW(), 'delivered', :amount)
            """),
            {"uid": order.user_id, "amount": order.quantity * 29.99}
        )
        order_id = result.lastrowid
        
        conn.execute(
            text("""
                INSERT INTO orderitem (order_id, product_id, quantity, 
                                       rating_score, unit_price, category)
                VALUES (:oid, :pid, :qty, :rating, 29.99,
                        (SELECT category FROM v_product_features 
                         WHERE product_id = :pid2 LIMIT 1))
            """),
            {
                "oid": order_id,
                "pid": order.product_id,
                "qty": order.quantity,
                "rating": order.rating_score,
                "pid2": order.product_id
            }
        )
        conn.commit()
        return {"order_id": order_id, "message": "Order created"}

@app.get("/api/trending")
async def get_trending(limit: int = 10):
    with db_engine.connect() as conn:
        result = conn.execute(
            text("""
                SELECT * FROM v_product_trending 
                ORDER BY trending_score DESC 
                LIMIT :lim
            """),
            {"lim": limit}
        )
        trending = [dict(row._mapping) for row in result]
        return {"trending": trending}

@app.get("/api/popular")
async def get_popular(category: str = None, limit: int = 10):
    with db_engine.connect() as conn:
        if category:
            query = text("""
                SELECT * FROM v_product_features 
                WHERE category = :cat
                ORDER BY popularity_score_0_100 DESC 
                LIMIT :lim
            """)
            result = conn.execute(query, {"cat": category, "lim": limit})
        else:
            query = text("""
                SELECT * FROM v_product_features 
                ORDER BY popularity_score_0_100 DESC 
                LIMIT :lim
            """)
            result = conn.execute(query, {"lim": limit})
            
        popular = [dict(row._mapping) for row in result]
        return {"popular": popular}

@app.get("/api/products/{product_id}/similar")
async def get_similar(product_id: int, n: int = 5):
    return {"similar": rec_engine.similar_products(product_id, n)}

@app.get("/api/cold-start/{category}")
async def get_cold_start(category: str, n: int = 10):
    return {"recommendations": rec_engine.cold_start(category, n)}

# ============================================================
# RUN THE SERVER
# ============================================================
if __name__ == "__main__":
    import uvicorn
    print("Starting FastAPI Server...")
    # This line actually starts the web server
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)