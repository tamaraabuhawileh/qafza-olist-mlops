# Olist MLOps Project

## Qafza MLOps Training 2026/2027 — Task 1

Task 1 focuses on building the data foundation for the Olist MLOps project. The goal is to load the **Brazilian E-Commerce Public Dataset by Olist** into a PostgreSQL database running locally with Docker, understand the relationships between the tables, and verify that the data can be queried and joined correctly.

The database created here will be used as the starting point for Task 2.

## Tech Stack

- Python
- PostgreSQL 16
- Docker / Docker Compose
- Pandas
- SQLAlchemy
- psycopg2

## Project Structure

```text
Dataset_Olist/
├── data/
│   └── raw/
├── sql/
│   ├── 01_create_tables.sql
│   └── 02_validation.sql
├── src/
│   └── ingest.py
├── .gitignore
├── .env.example
├── docker-compose.yml
├── requirements.txt
└── README.md
```

The raw Olist CSV files are not included in the repository.

## Dataset & Relationships

The dataset contains nine CSV files representing customers, orders, order items, products, sellers, payments, reviews, geolocation data, and product category translations.

The main relationships are:

```text
customers
    │ customer_id
    ▼
orders
    │ order_id
    ├──► order_items ──► products
    │         │
    │         └────────► sellers
    ├──► order_payments
    └──► order_reviews
```

Main keys:

| Key | Connects |
|---|---|
| `customer_id` | customers → orders |
| `order_id` | orders → items, payments, reviews |
| `product_id` | order_items → products |
| `seller_id` | order_items → sellers |

## ML Problem

The final ML problem is to predict whether an order will be delivered **late or on time**.

The target will later be derived using:

```text
order_delivered_customer_date
order_estimated_delivery_date
```

An order is considered late when its actual delivery date is later than its estimated delivery date.

Label creation, EDA, feature engineering, and modeling are not part of Task 1.

## Running the Project

### 1. Install dependencies

```cmd
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure the database

Create a `.env` file based on `.env.example`, then start PostgreSQL:

```cmd
docker compose up -d
```

### 3. Create the tables

```cmd
docker exec -i olist-postgres psql -U olist_user -d olist < sql\01_create_tables.sql
```

### 4. Add the data

Place the Olist CSV files in:

```text
data/raw/
```

Then run:

```cmd
python src\ingest.py
```

## Ingestion Result

All nine datasets were successfully loaded:

| Table | Rows |
|---|---:|
| customers | 99,441 |
| orders | 99,441 |
| order_items | 112,650 |
| products | 32,951 |
| sellers | 3,095 |
| order_payments | 103,886 |
| order_reviews | 99,224 |
| geolocation | 1,000,163 |
| product_category_translation | 71 |

The ingestion script uses chunked inserts for the larger tables. It also maps the original Olist product columns `product_name_lenght` and `product_description_lenght` to the corrected database column names.

## Validation

Database checks and example joins are available in:

```text
sql/02_validation.sql
```

Example join:

```sql
SELECT
    o.order_id,
    o.order_status,
    c.customer_city,
    c.customer_state
FROM orders AS o
JOIN customers AS c
    ON o.customer_id = c.customer_id
LIMIT 20;
```

This confirms that the relational tables can be queried and joined successfully.

## Task 1 Status

- [x] Olist data downloaded and understood
- [x] PostgreSQL running locally with Docker
- [x] Relational schema created
- [x] All datasets loaded into PostgreSQL
- [x] Table relationships verified
- [x] SQL queries and joins tested
- [x] Late vs. on-time prediction problem understood

**Task 1 completed.**

## Next Step

Task 2 will continue from this database and cover joining the tables into an ML dataset, creating the target, train/validation/test splitting, EDA, feature engineering, and initial model training. Each stage will save artifacts that can be reused by the following stage.
