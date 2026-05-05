# SMARTOMS — Smart Order Management System

A hybrid recommendation system built with React, FastAPI, and MariaDB, featuring Content-Based Filtering (CBF) and Collaborative Filtering (CF) via SVD.

Team name: Sleeping Bag 
University: HELP university

---

## Table of Contents
- [Prerequisites](#prerequisites)
- [Project Structure](#project-structure)
- [Database Setup](#database-setup)
- [Backend Setup](#backend-setup)
- [Frontend Setup](#frontend-setup)
- [Environment Variables](#environment-variables)
- [Running the Application](#running-the-application)
- [Troubleshooting](#troubleshooting)

---

## Prerequisites

Make sure you have the following installed before proceeding:

|   Tool    |   Version   |       Download      |
|-----------|-------------|---------------------|
|   Python  |   3.10.11   | https://python.org  |
|  Node.js  |     18+     | https://nodejs.org  |
|  MariaDB  |    10.6+    | https://mariadb.org |
|    Git    | ----------- | https://git-scm.com |

---

## Project Structure

```
smartoms/
├── api/
│   └── __init__.py
├── src/                        # React frontend
│   ├── App.jsx
│   ├── App.css
│   └── main.jsx
├── main.py                     # FastAPI entry point
├── recommender_engine.py       # Recommendation engine loader
├── global_recommender.py       # CBF model
├── user_recommender.py         # SVD/CF model
├── hybrid_recommender.py       # Hybrid model
├── db_connect.py               # Database connection helper
├── data_handling.py            # Data processing scripts
├── data_merge.py               # Data merging utilities
├── faker_data.py               # Synthetic data generator
├── load_data.py                # Data loader
├── requirements.txt            # Python dependencies
├── package.json                # Node dependencies
├── vite.config.js              # Vite config
├── database.sql                # Full DB schema + seed data
├── .env.example                # Environment variable template
└── .gitignore
```

---

## Database Setup

**1. Start your MariaDB server**

**2. Create the database**
```sql
CREATE DATABASE oms;
```

**3. Import the schema and seed data**
```bash
mariadb -u root -p oms < smartoms.sql
```

**4. Verify the import**
```sql
USE oms;
SHOW TABLES;
```

You should see tables including `product`, `orders`, `orderitem`, `users`, and views like `v_product_features`, `v_product_popularity`.

---

## Backend Setup

**1. Clone the repository**
```bash
git clone https://github.com/chanxny-max/smartoms.git
cd smartoms
```

**2. Create and activate a virtual environment**
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python -m venv venv
source venv/bin/activate
```

**3. Install Python dependencies**
```bash
pip install -r requirements.txt
```

**4. Configure environment variables**
```bash
# Copy the example env file
cp .env.example .env

# Edit .env with your actual values
```

Open `.env` and fill in your database credentials and configuration (see [Environment Variables](#environment-variables)).

**5. Train the recommendation models**

The models are trained on startup when FastAPI launches via `main.py`, `recommender_engine.py` is the model loader class called during startup, but you can also run them individually:
```bash
python global_recommender.py    # Train CBF model
python user_recommender.py      # Train SVD/CF model
python hybrid_recommender.py    # Evaluate hybrid model
```

---

## Frontend Setup

**1. Install Node dependencies**
```bash
npm install
```

**2. Configure frontend environment**
```bash
cp .env.example .env.local
```

Set `VITE_API_URL` in `.env.local` to match your backend URL.

---

## Environment Variables

Copy `.env.example` to `.env` and configure the following:

```bash
cp .env.example .env
```

|     Variable    |            Description          |                     Example                   |
|-----------------|---------------------------------|-----------------------------------------------|
| `DATABASE_URL`  | Full database connection string | `mysql+pymysql://root:password@localhost/oms` |
|    `DB_HOST`    |          Database host          | `localhost`                                   |
|    `DB_PORT`    |          Database port          | `3306`                                        |
|    `DB_USER`    |        Database username        | `root`                                        |
|  `DB_PASSWORD`  |        Database password        | `yourpassword`                                |
|    `DB_NAME`    |          Database name          | `oms`                                         |
|    `API_HOST`   |           FastAPI host          | `0.0.0.0`                                     |
|    `API_PORT`   |           FastAPI port          | `8000`                                        |
|  `VITE_API_URL` |      Frontend API base URL      | `http://localhost:8000/api`                   |

---

## Running the Application

### Start the Backend

```bash
# Make sure your virtual environment is activated
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at: `http://localhost:8000`  
API docs (Swagger UI): `http://localhost:8000/docs`

### Start the Frontend

```bash
npm run dev
```

The frontend will be available at: `http://localhost:5173`

---

## API Endpoints

| Method |              Endpoint            |                          Description                         |
|--------|----------------------------------|--------------------------------------------------------------|
| `POST` |           `/api/users`           | Register a new user                                          |
| `POST` |           `/api/login`           | Login                                                        |
| `GET`  |           `/api/products`        | Get all products (supports `?category=` and `?limit=`)       |
| `GET`  | `/api/recommendations/{user_id}` | Get hybrid recommendations (supports `?n=` and `?category=`) |
| `GET`  |           `/api/popular`         | Get popular products (supports `?category=` and `?limit=`)   |
| `POST` |          `/api/orders`           | Create an order (feeds recommendation training data)         |

---

## Recommendation System

SMARTOMS uses a **Hybrid Recommendation System** that adapts based on user history:

| Interactions |       Algorithm      |              Behaviour           |
|--------------|----------------------|----------------------------------|
| 0            | Cold Start           | Returns most popular products    |
| < 5          | CBF Dominant (α=0.2) | Relies on content similarity     |
| 5–14         | Balanced (α=0.4)     | Equal weight CBF + SVD           |
| 15–29        | SVD Dominant (α=0.6) | Leans on collaborative filtering |
| 30+          | Heavy SVD (α=0.8)    | Strong collaborative filtering   |

---

## Troubleshooting

**Database connection error**
- Verify MariaDB is running
- Check `DATABASE_URL` in your `.env` file matches your credentials
- Ensure the `oms` database exists and `smartoms.sql` was imported

**`Table 'oms.products' doesn't exist`**
- Your table is named `product` not `products` — check any raw SQL queries

**Frontend shows `RM0.00` and no description**
- Ensure `v_product_features` and `v_product_popularity` views include `price`, `product_description`, and `quantity_available` via JOIN to the `product` table

**Recommendation engine slow to start**
- This is normal — the SVD matrix factorization and cosine similarity matrix are computed on startup. Subsequent requests are fast.

**Port already in use**
```bash
# Change backend port
uvicorn main:app --reload --port 8001

# Update VITE_API_URL in .env accordingly
VITE_API_URL=http://localhost:8001/api
```
