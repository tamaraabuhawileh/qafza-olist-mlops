# Olist MLOps Project

## Qafza MLOps Training 2026/2027 — Task 1

### Get the Data Into a Database

This repository contains my implementation of **Task 1** of the Qafza MLOps Training 2026/2027.

The goal of this task is to establish the data foundation for the MLOps project by loading the **Brazilian E-Commerce Public Dataset by Olist** into a relational PostgreSQL database running locally with Docker.

The PostgreSQL database created in this task will serve as the data source for the subsequent stages of the project.

---

## Task Objectives

The objectives of Task 1 are to:

- Understand the Olist dataset and its relational structure
- Understand the future ML problem: predicting whether an order will be delivered late or on time
- Run PostgreSQL locally using Docker
- Design a relational database schema
- Ingest the Olist CSV datasets into PostgreSQL
- Validate the loaded data
- Query the database
- Test relationships between tables using SQL joins

> EDA, feature engineering, label creation, and model training are intentionally outside the scope of Task 1.

---

## Tech Stack

- Python
- PostgreSQL 16
- Docker / Docker Compose
- Pandas
- SQLAlchemy
- psycopg2
- SQL

---

## Project Structure

```text
Dataset_Olist/
│
├── data/
│   └── raw/
│       └── .gitkeep
│
├── sql/
│   ├── 01_create_tables.sql
│   └── 02_validation.sql
│
├── src/
│   ├── inspect_data.py
│   └── ingest.py
│
├── .gitignore
├── .env.example
├── docker-compose.yml
├── requirements.txt
└── README.md
```

The raw Olist CSV files are not committed to this repository.

---

## Dataset

The project uses the **Brazilian E-Commerce Public Dataset by Olist**.

The source dataset contains nine CSV files:

```text
olist_customers_dataset.csv
olist_geolocation_dataset.csv
olist_orders_dataset.csv
olist_order_items_dataset.csv
olist_order_payments_dataset.csv
olist_order_reviews_dataset.csv
olist_products_dataset.csv
olist_sellers_dataset.csv
product_category_name_translation.csv
```

After downloading the dataset, the files should be placed under:

```text
data/raw/
```

---

## Relational Structure

The Olist dataset consists of multiple related tables.

The main relationships used in this project are:

```text
customers
    │
    │ customer_id
    ▼
orders
    │
    │ order_id
    ├──────────────► order_items
    │                    │
    │                    ├── product_id ──► products
    │                    │
    │                    └── seller_id ───► sellers
    │
    ├──────────────► order_payments
    │
    └──────────────► order_reviews
```

Important identifiers include:

| Key | Relationship |
|---|---|
| `customer_id` | Customers → Orders |
| `order_id` | Orders → Order Items / Payments / Reviews |
| `product_id` | Order Items → Products |
| `seller_id` | Order Items → Sellers |

The `geolocation` table provides geographical information based on ZIP-code prefixes, while `product_category_translation` provides English translations for product category names.

---

## Machine Learning Problem

The later stages of this project will build a model to predict whether an order will be delivered **late or on time**.

Two important fields for creating the target are:

```text
order_delivered_customer_date
order_estimated_delivery_date
```

Conceptually:

```text
Actual delivery date > Estimated delivery date
→ Late

Actual delivery date <= Estimated delivery date
→ On Time
```

The target variable is **not created in Task 1**. Label creation and ML dataset preparation are part of the next stage of the project.

---

## Setup

### 1. Clone the Repository

```bash
git clone <repository-url>
cd Dataset_Olist
```

### 2. Create a Virtual Environment

On Windows:

```cmd
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install Dependencies

```cmd
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file based on `.env.example`.

Example:

```env
POSTGRES_DB=olist
POSTGRES_USER=olist_user
POSTGRES_PASSWORD=your_password
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
```

The real `.env` file is excluded from Git.

### 5. Add the Dataset

Place the nine Olist CSV files inside:

```text
data/raw/
```

---

## PostgreSQL with Docker

PostgreSQL 16 runs locally inside a Docker container.

Start the database:

```cmd
docker compose up -d
```

Verify that the container is running:

```cmd
docker ps
```

Connect to PostgreSQL:

```cmd
docker exec -it olist-postgres psql -U olist_user -d olist
```

Inside PostgreSQL, the tables can be listed using:

```sql
\dt
```

---

## Create the Database Schema

The relational schema is defined in:

```text
sql/01_create_tables.sql
```

From Windows Command Prompt:

```cmd
docker exec -i olist-postgres psql -U olist_user -d olist < sql\01_create_tables.sql
```

The schema defines the required:

- Tables
- Primary keys
- Composite keys
- Foreign keys
- Relationships between the Olist entities

---

## Data Ingestion

Run the ingestion pipeline:

```cmd
python src\ingest.py
```

The ingestion pipeline:

1. Reads each source CSV file
2. Applies required column mappings
3. Loads the tables in relational dependency order
4. Uses chunked inserts for large datasets
5. Inserts the data into PostgreSQL

The original files under `data/raw/` remain unchanged.

---

## Loaded Data

The completed ingestion produced the following table sizes:

| Table | Rows |
|---|---:|
| `customers` | 99,441 |
| `sellers` | 3,095 |
| `products` | 32,951 |
| `orders` | 99,441 |
| `order_items` | 112,650 |
| `order_payments` | 103,886 |
| `order_reviews` | 99,224 |
| `geolocation` | 1,000,163 |
| `product_category_translation` | 71 |

All nine source datasets were successfully loaded into PostgreSQL.

---

## Database Validation

Validation queries are stored in:

```text
sql/02_validation.sql
```

The validation covers:

- Table row counts
- Sample records
- Two-table joins
- Multi-table joins
- Delivery-related fields required for the future ML problem

### Example: Orders and Customers

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

This validates the relationship:

```text
customers.customer_id
        │
        ▼
orders.customer_id
```

### Example: Multi-Table Join

```sql
SELECT
    o.order_id,
    o.order_status,
    p.product_category_name,
    s.seller_city,
    s.seller_state,
    oi.price,
    oi.freight_value
FROM orders AS o
JOIN order_items AS oi
    ON o.order_id = oi.order_id
JOIN products AS p
    ON oi.product_id = p.product_id
JOIN sellers AS s
    ON oi.seller_id = s.seller_id
LIMIT 20;
```

This confirms that the relational tables can be combined successfully for later ML dataset construction.

---

## Implementation Challenges and Solutions

During implementation, two main ingestion issues were identified and resolved.

### 1. Large Database Inserts

The initial ingestion approach attempted to insert too many records in a single database operation, resulting in an oversized SQL statement.

The ingestion pipeline was updated to use chunked inserts:

```python
chunksize=2000
```

This allows large tables, including the geolocation dataset with more than one million rows, to be loaded reliably.

### 2. Product Column Naming

The original Olist products dataset contains the column names:

```text
product_name_lenght
product_description_lenght
```

The PostgreSQL schema uses the corrected names:

```text
product_name_length
product_description_length
```

The ingestion layer maps the source columns before insertion:

```python
df = df.rename(
    columns={
        "product_name_lenght": "product_name_length",
        "product_description_lenght": "product_description_length",
    }
)
```

This keeps the raw source files unchanged while maintaining cleaner column names in the database.

---

## Task 1 Completion

- [x] Olist dataset downloaded
- [x] Dataset tables and relationships understood
- [x] Late/on-time delivery problem understood
- [x] PostgreSQL running locally
- [x] PostgreSQL containerized using Docker
- [x] Relational database schema created
- [x] Primary and foreign key relationships defined
- [x] All nine source datasets ingested
- [x] Database queried successfully
- [x] Table row counts validated
- [x] SQL joins tested

**Task 1 completed successfully.**

---

## Next Stage — Task 2

Task 2 will build directly on the PostgreSQL database created in Task 1.

The next stage will consist of six ordered notebooks:

1. **Read and join the tables**  
   Build one ML table with one row per order.

2. **Create the labels**  
   Create the late/on-time delivery target and validate its distribution.

3. **Train / validation / test split**  
   Split the dataset before detailed analysis to avoid data leakage.

4. **Exploratory Data Analysis (EDA)**  
   Perform detailed analysis using the training split only.

5. **Feature engineering**  
   Build features, handle missing values, encode categorical variables, and persist fitted transformations.

6. **Train, tune, and evaluate**  
   Establish a baseline, train and tune the model, and perform the final evaluation.

Each stage will save its outputs as **artifacts**, allowing the next stage to consume the previous stage's results without recomputing them.

This structure will also make the workflow reproducible and prepare the project for later conversion from notebooks into production MLOps pipelines.
