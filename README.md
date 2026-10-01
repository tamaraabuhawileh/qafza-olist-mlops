# Olist MLOps Project

## Qafza MLOps Training 2026/2027 — Tasks 1 & 2

This project builds an end-to-end machine learning workflow using the **Brazilian E-Commerce Public Dataset by Olist**.

- **Task 1** builds the data foundation by loading the Olist datasets into PostgreSQL using Docker and validating the relational structure.
- **Task 2** transforms the relational data into an ML-ready dataset for predicting whether an order will be delivered **late or on time**, following a reproducible notebook and artifact-based workflow.

Dataset: [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)

---

## Tech Stack

- Python
- PostgreSQL 16
- Docker / Docker Compose
- Pandas
- SQLAlchemy
- psycopg2
- Jupyter Notebook
- scikit-learn
- XGBoost
- Joblib
- PyArrow
- Matplotlib

---

# Task 1 — Data Foundation

Task 1 focuses on building the data foundation for the project.

The goal is to load the Olist datasets into a PostgreSQL database running locally with Docker, understand the relationships between the tables, and verify that the data can be queried and joined correctly.

## Dataset & Relationships

![Olist relational schema](assets/olist_schema.png)

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

## Database Ingestion

All nine datasets were successfully loaded into PostgreSQL:

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

The ingestion script uses chunked inserts for the larger tables.

It also maps the original Olist product columns `product_name_lenght` and `product_description_lenght` to the corrected database column names.

## Database Validation

Database checks and example joins are available in:

```text
sql/02_validation.sql
```

Example:

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
- [x] Late vs. on-time prediction problem defined

**Task 1 completed.**

---

# Task 2 — From Relational Data to ML Pipeline

Task 2 builds on the PostgreSQL database created in Task 1.

The objective is to create a reproducible machine learning workflow that:

1. Joins the relational tables into an order-level ML dataset
2. Creates the late-delivery target
3. Splits the data into train, validation, and test sets
4. Performs EDA using the training data only
5. Builds and saves a reproducible feature-engineering pipeline
6. Trains, compares, tunes, and evaluates classification models

Each stage produces artifacts that are consumed by the following stage.

---

## ML Problem

The objective is to predict whether an order will be delivered **late or on time**.

The target is derived from:

```text
order_delivered_customer_date
order_estimated_delivery_date
```

An order is labeled:

```text
0 = On Time
1 = Late
```

A late order is one whose actual delivery date is later than its estimated delivery date.

Orders without sufficient delivery information are not assigned an artificial label.

---

# Task 2 Notebook Pipeline

Task 2 is organized into six notebooks.

```text
01_read_and_join.ipynb
        ↓
02_create_labels.ipynb
        ↓
03_train_val_test_split.ipynb
        ↓
04_eda.ipynb
        ↓
05_feature_engineering.ipynb
        ↓
06_train_tune_evaluate.ipynb
```

Each notebook has one main responsibility and saves artifacts for the next stage.

---

## Notebook 1 — Read and Join

`01_read_and_join.ipynb`

Reads the PostgreSQL tables created in Task 1 and constructs a single ML table with approximately one row per order.

One-to-many tables such as order items, payments, and reviews are aggregated to order level before joining to avoid duplicating orders.

Examples of aggregated information include:

- Number of items
- Number of unique products
- Number of sellers
- Item prices
- Freight value
- Payment information
- Product dimensions
- Product categories
- Seller information
- Review information

The resulting table is validated to ensure that `order_id` remains unique.

**Artifact:**

```text
artifacts/01_joined/ml_orders.parquet
```

---

## Notebook 2 — Create Labels

`02_create_labels.ipynb`

Creates the late-vs-on-time target.

Missing delivery dates are investigated before labeling. Most orders without an actual delivery date correspond to statuses such as shipped, canceled, unavailable, invoiced, or processing.

Only orders with sufficient information to determine the delivery outcome are retained for the supervised learning problem.

The target is stored as:

```text
is_late
```

where:

```text
0 = On Time
1 = Late
```

The label distribution is also inspected to identify class imbalance.

**Artifact:**

```text
artifacts/02_labeled/labeled_orders.parquet
```

---

## Notebook 3 — Train / Validation / Test Split

`03_train_val_test_split.ipynb`

The labeled dataset is split chronologically:

```text
70% Training
15% Validation
15% Test
```

A time-based split is used because the production scenario is chronological: a model learns from historical orders and predicts outcomes for future orders.

Observed late-delivery rates:

| Split | Late Rate |
|---|---:|
| Train | 9.03% |
| Validation | 5.34% |
| Test | 6.61% |

The changing late-delivery rate also demonstrates why a chronological split provides a more realistic evaluation than randomly mixing orders from different time periods.

The splits are validated to ensure:

- Every row belongs to exactly one split
- No `order_id` overlaps between splits
- Chronological order is preserved

**Artifacts:**

```text
artifacts/03_splits/train.parquet
artifacts/03_splits/validation.parquet
artifacts/03_splits/test.parquet
```

---

## Notebook 4 — Exploratory Data Analysis

`04_eda.ipynb`

Detailed EDA is performed using the **training split only**.

The validation and test sets are not used to make EDA or feature-engineering decisions.

The analysis covers:

- Dataset shape and data types
- Memory usage
- Missing values
- Numerical distributions
- Skewness and outliers
- Categorical cardinality
- Rare categories
- Target distribution
- Feature relationships with `is_late`
- Correlations and cross-tabs
- Monthly and weekday patterns
- Delivery seasonality
- Customer geography
- Seller geography
- ZIP-code information

The target is imbalanced, with late deliveries representing a minority of orders.

The EDA is used to guide feature selection and identify potential leakage.

Post-delivery information, review information, target-derived columns, and identifiers are not selected as prediction-time model features.

**Artifacts include:**

```text
artifacts/04_eda/
├── findings.txt
└── charts/
```

---

## Notebook 5 — Feature Engineering

`05_feature_engineering.ipynb`

Feature engineering converts the raw split datasets into model-ready feature tables.

Prediction-time features include order, customer, payment, product, seller, geographic, and purchase-time information.

Examples of engineered temporal features include:

```text
purchase_month
purchase_weekday
purchase_hour
purchase_weekend
estimated_delivery_days
```

Only information available at the defined prediction point is used.

Future information such as actual delivery dates and review outcomes is excluded from the model features.

### Preprocessing

Numerical features use:

- Median imputation
- Standard scaling

Categorical features use:

- Missing-value imputation
- One-hot encoding

The preprocessing pipeline is **fitted on the training set only**:

```text
Train       → fit_transform()
Validation  → transform()
Test        → transform()
```

This prevents information leakage.

The fitted preprocessing object is saved so that production inference can reuse exactly the same transformations without fitting them again.

**Artifacts:**

```text
artifacts/05_features/
├── train_features.parquet
├── validation_features.parquet
├── test_features.parquet
├── preprocessor.joblib
├── feature_list.json
└── input_features.json
```

---

## Notebook 6 — Train, Tune, Evaluate

`06_train_tune_evaluate.ipynb`

The final notebook trains and compares classification models.

A simple majority-class baseline is established first so that trained models have a clear benchmark to beat.

Candidate models include:

- Dummy Classifier baseline
- Logistic Regression
- Random Forest
- XGBoost
- Histogram Gradient Boosting
- LightGBM, when available

Because the target is imbalanced, model selection does not rely on accuracy alone.

The main evaluation metrics are:

- Precision
- Recall
- F1-score
- PR-AUC
- Accuracy as a supporting metric

For example, the initial Random Forest achieved high validation accuracy but extremely low recall for late deliveries, demonstrating why accuracy alone is misleading for this problem.

The strongest candidate models are tuned using the **validation split only**.

Hyperparameters and the classification threshold can be tuned to improve the trade-off between precision and recall.

The test set is used only once at the end, after the model configuration has been selected.

**Final artifacts:**

```text
artifacts/06_model/
├── final_model.joblib
├── model_config.json
└── results.json
```

---

# Leakage Prevention

Preventing data leakage is a key part of the workflow.

The model is designed around information available at prediction time.

Examples of columns excluded from model inputs include:

```text
order_delivered_customer_date
order_delivered_carrier_date
review_count
mean_review_score
min_review_score
max_review_score
is_late
delivery_label
```

Identifiers such as `order_id`, `customer_id`, and `customer_unique_id` are also excluded from model training.

Additionally:

- EDA uses training data only
- Preprocessing is fitted on training data only
- Validation is used for model selection and tuning
- Test is reserved for final evaluation

---

# Artifact-Based Workflow

A central goal of Task 2 is reproducibility.

```text
PostgreSQL
    ↓
01_joined
    ↓
02_labeled
    ↓
03_splits
    ↓
04_eda
    ↓
05_features
    ↓
06_model
```

Each notebook reads the previous stage's artifacts instead of repeating the entire workflow.

This makes the pipeline easier to reproduce, debug, and later convert into production ML code.

---

# Project Structure

```text
Dataset_Olist/
│
├── assets/
│   └── olist_schema.png
│
├── data/
│   └── raw/
│
├── notebooks/
│   ├── 01_read_and_join.ipynb
│   ├── 02_create_labels.ipynb
│   ├── 03_train_val_test_split.ipynb
│   ├── 04_eda.ipynb
│   ├── 05_feature_engineering.ipynb
│   └── 06_train_tune_evaluate.ipynb
│
├── artifacts/
│   ├── 01_joined/
│   ├── 02_labeled/
│   ├── 03_splits/
│   ├── 04_eda/
│   ├── 05_features/
│   └── 06_model/
│
├── sql/
│   ├── 01_create_tables.sql
│   └── 02_validation.sql
│
├── src/
│   └── ingest.py
│
├── .env.example
├── .gitignore
├── docker-compose.yml
├── requirements.txt
└── README.md
```

Raw Olist CSV files are not included in the repository.

---

# Running the Project

## 1. Create the Python environment

```cmd
python -m venv .venv

.venv\Scripts\activate

pip install -r requirements.txt
```

## 2. Configure PostgreSQL

Create a `.env` file based on `.env.example`.

Start PostgreSQL:

```cmd
docker compose up -d
```

## 3. Create the database tables

```cmd
docker exec -i olist-postgres psql -U olist_user -d olist < sql\01_create_tables.sql
```

## 4. Add the Olist data

Place the CSV files under:

```text
data/raw/
```

Then run:

```cmd
python src\ingest.py
```

## 5. Run the ML notebooks

Start Jupyter:

```cmd
jupyter notebook
```

Run the Task 2 notebooks in order:

```text
01_read_and_join.ipynb
02_create_labels.ipynb
03_train_val_test_split.ipynb
04_eda.ipynb
05_feature_engineering.ipynb
06_train_tune_evaluate.ipynb
```

---

# Project Status

### Task 1

- [x] PostgreSQL environment created with Docker
- [x] Relational schema created
- [x] Nine Olist datasets ingested
- [x] Relationships validated
- [x] SQL joins tested

### Task 2

- [x] Order-level ML table created
- [x] Late/on-time target created
- [x] Chronological train/validation/test split created
- [x] Training-only EDA completed
- [x] Feature engineering pipeline created
- [x] Fitted preprocessing objects saved
- [x] Baseline and candidate models trained
- [x] Imbalance-aware metrics evaluated
- [x] Model tuning performed
- [x] Final model and results saved

**Tasks 1 and 2 completed.**

---

## MLOps Principles Applied

This project introduces several practices that will support later productionization:

- Reproducible database setup
- Clear separation between pipeline stages
- Versionable intermediate artifacts
- Training-only fitting of preprocessing transformations
- Leakage-aware feature engineering
- Chronological evaluation
- Baseline comparison
- Imbalance-aware model evaluation
- Validation-based model selection
- Held-out test evaluation
- Persisted preprocessing and model objects

The resulting artifacts provide the foundation for converting the notebook workflow into a reusable training and inference pipeline.

## Task 3 — Inference Service (In Progress)

Task 3 builds an inference service around the model and fitted
preprocessor from Task 2. Training remains in the notebooks.

### Development setup

Run these commands from the repository root using Python 3.14.7.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements/dev.txt
.\.venv\Scripts\python.exe -m pip check
```

For runtime dependencies only, install `requirements/runtime.txt`.

### Project structure

- `app/`: API routes and request/response schemas.
- `config/`: inference configuration.
- `src/olist_inference/`: reusable inference modules.
- `tests/`: unit, data, model, and integration tests.
- `requirements/`: pinned runtime and development dependencies.
- `models/`: downloaded model cache.
- `notebooks/`: existing analysis and training notebooks.
- `artifacts/`: saved Task 2 datasets and fitted objects.

### Configuration

The default configuration is `config/inference.json`.
Relative paths are resolved against the project root.

To use another configuration in PowerShell:

```powershell
$env:OLIST_CONFIG = "config/inference.json"
```

Keep secrets and connection strings in environment variables.
Never commit `.env`.

The classification threshold is read from the saved
`artifacts/06_model/model_config.json`; its current value is 0.40.

### Progress

- Repository structure and configuration loader added.
- Runtime and development dependencies pinned.
- Saved preprocessor and model loaded in a fresh environment.
- Prediction parity, API, containers, and CI/CD are pending.