# Olist MLOps Project

## Qafza MLOps Training 2026/2027

This project develops an end-to-end machine learning workflow using the
**Brazilian E-Commerce Public Dataset by Olist**.

The project evolves through three phases:

```text
Phase 1                     Phase 2                      Phase 3
Data Foundation      →      ML Development       →      Inference Service
                         
PostgreSQL                   ML Dataset                   FastAPI
Docker                       EDA                          Validation
SQL                          Feature Engineering          Logging
Data Ingestion               Model Training               Testing
Validation                   Evaluation                   DVC
                             Saved Artifacts              Data Quality
```

The final machine learning objective is to predict whether an Olist order
will be delivered **late or on time**.

The project emphasizes reproducibility, leakage prevention, separation of
training and inference, artifact persistence, automated testing, and
production-oriented ML engineering.

---

# Project Phases

| Phase | Task | Objective | Status |
|---|---|---|---|
| Phase 1 | Task 1 | Build the relational data foundation | Completed |
| Phase 2 | Task 2 | Build and evaluate the ML pipeline | Completed |
| Phase 3 | Task 3 | Productionize model inference | In Progress |

---

# Tech Stack

### Data & Database

- PostgreSQL 16
- SQL
- Docker / Docker Compose
- Pandas
- SQLAlchemy
- psycopg2

### Machine Learning

- scikit-learn
- XGBoost
- LightGBM
- Joblib
- PyArrow
- Matplotlib
- Jupyter Notebook

### Inference & MLOps

- FastAPI
- Uvicorn
- Pytest
- Ruff
- DVC
- Great Expectations
- Python logging

---

# Dataset

The project uses the
[Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce).

The dataset contains approximately 100,000 orders from multiple Brazilian
marketplaces and includes information about:

- Customers
- Orders
- Order items
- Products
- Sellers
- Payments
- Reviews
- Geolocation
- Product category translations

Raw CSV files are not committed to the repository.

---

# Phase 1 — Data Foundation

## Task 1: PostgreSQL, Docker, and Data Ingestion

Phase 1 establishes the data foundation for the ML project.

The objective is to load the relational Olist datasets into PostgreSQL,
understand the relationships between the tables, and verify that the data can
be queried and joined reliably.

## Relational Schema

![Olist relational schema](assets/olist_schema.png)

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

| Key | Relationship |
|---|---|
| `customer_id` | customers → orders |
| `order_id` | orders → items, payments, reviews |
| `product_id` | order_items → products |
| `seller_id` | order_items → sellers |

## Database Ingestion

All nine Olist datasets were loaded into PostgreSQL.

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

The ingestion pipeline uses chunked inserts for larger datasets.

The original Olist product columns:

```text
product_name_lenght
product_description_lenght
```

are mapped to corrected database column names during ingestion.

## Database Validation

Database validation queries are available in:

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

These checks verify that the tables were loaded correctly and that the
relationships can be queried successfully.

## Phase 1 Output

Phase 1 produces a validated relational database that becomes the source for
the machine learning workflow in Phase 2.

### Task 1 Status

- [x] Olist datasets understood
- [x] PostgreSQL environment created with Docker
- [x] Relational schema created
- [x] Nine datasets ingested
- [x] Row counts validated
- [x] Relationships verified
- [x] SQL joins tested
- [x] ML problem defined

**Phase 1 completed.**

---

# Phase 2 — Machine Learning Development

## Task 2: From Relational Data to an ML Pipeline

Phase 2 transforms the PostgreSQL data into a reproducible machine learning
workflow.

The objective is to predict whether an order will be delivered:

```text
0 = On Time
1 = Late
```

The phase covers:

1. Relational table aggregation and joining
2. Target creation
3. Chronological data splitting
4. Training-only exploratory data analysis
5. Leakage-aware feature engineering
6. Reproducible preprocessing
7. Model training and comparison
8. Hyperparameter and threshold selection
9. Final evaluation
10. Artifact persistence

---

# ML Problem

The target is derived from:

```text
order_delivered_customer_date
order_estimated_delivery_date
```

An order is considered late when its actual delivery date occurs after its
estimated delivery date.

Orders without sufficient information to determine the delivery outcome are
not assigned an artificial label.

---

# Task 2 Pipeline

Task 2 is organized into six sequential notebooks:

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

Each notebook has one main responsibility and saves artifacts for the next
stage.

---

## Notebook 1 — Read and Join

`01_read_and_join.ipynb`

Reads the PostgreSQL tables created in Phase 1 and constructs an order-level
ML table.

One-to-many tables are aggregated before joining to avoid creating duplicate
orders.

Aggregated information includes:

- Item counts
- Unique product counts
- Seller counts
- Item prices
- Freight values
- Payment information
- Product dimensions
- Product categories
- Seller information
- Review information

The final dataset is validated to ensure that `order_id` remains unique.

### Output

```text
artifacts/01_joined/ml_orders.parquet
```

---

## Notebook 2 — Create Labels

`02_create_labels.ipynb`

Creates the late-delivery target.

Missing actual delivery dates are investigated before target creation.
Orders without enough information to determine their delivery outcome are
excluded from the supervised-learning population.

The target column is:

```text
is_late
```

where:

```text
0 = On Time
1 = Late
```

The target distribution is also inspected for class imbalance.

### Output

```text
artifacts/02_labeled/labeled_orders.parquet
```

---

## Notebook 3 — Train / Validation / Test Split

`03_train_val_test_split.ipynb`

The labeled data is divided chronologically:

```text
70% Training
15% Validation
15% Test
```

A chronological split better represents the intended production scenario:
the model learns from historical orders and predicts future orders.

Observed late-delivery rates:

| Split | Late Rate |
|---|---:|
| Train | 9.03% |
| Validation | 5.34% |
| Test | 6.61% |

The split is validated to ensure:

- Every order belongs to exactly one split
- No `order_id` overlap exists between splits
- Chronological ordering is preserved

### Outputs

```text
artifacts/03_splits/train.parquet
artifacts/03_splits/validation.parquet
artifacts/03_splits/test.parquet
```

---

## Notebook 4 — Exploratory Data Analysis

`04_eda.ipynb`

EDA is performed using the **training split only**.

Validation and test data are not used to make EDA or feature-engineering
decisions.

The analysis includes:

- Dataset shape and data types
- Memory usage
- Missing values
- Numerical distributions
- Skewness and outliers
- Categorical cardinality
- Rare categories
- Target distribution
- Feature relationships with `is_late`
- Correlations
- Monthly patterns
- Weekday patterns
- Delivery seasonality
- Customer geography
- Seller geography
- ZIP-code information

The analysis also identifies features that could introduce data leakage.

### Outputs

```text
artifacts/04_eda/
├── findings.txt
└── charts/
```

---

## Notebook 5 — Feature Engineering

`05_feature_engineering.ipynb`

Feature engineering converts the raw splits into model-ready feature tables.

Examples of engineered temporal features include:

```text
purchase_month
purchase_weekday
purchase_hour
purchase_weekend
estimated_delivery_days
```

Only information available at the defined prediction point is selected.

Future information such as actual delivery outcomes and review results is
excluded.

## Preprocessing

Numerical features use:

- Median imputation
- Standard scaling

Categorical features use:

- Missing-value imputation
- One-hot encoding

The preprocessing pipeline follows:

```text
Training     → fit_transform()
Validation   → transform()
Test         → transform()
```

The preprocessing object is fitted **only on the training set** and persisted
for later inference.

### Outputs

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

## Notebook 6 — Train, Tune, and Evaluate

`06_train_tune_evaluate.ipynb`

The final notebook trains and compares multiple classifiers.

Candidate models include:

- Dummy Classifier
- Logistic Regression
- Random Forest
- XGBoost
- Histogram Gradient Boosting
- LightGBM, when available

Because late deliveries represent a minority class, accuracy alone is not
used for model selection.

Important metrics include:

- Precision
- Recall
- F1-score
- PR-AUC
- Accuracy as a supporting metric

The experiments demonstrated why imbalance-aware metrics are necessary. For
example, a classifier may achieve high overall accuracy while detecting very
few late orders.

Model selection and threshold decisions use validation data.

The held-out test set is reserved for final evaluation after model decisions
have been made.

The final persisted classifier is an `XGBClassifier`.

### Final Outputs

```text
artifacts/06_model/
├── final_model.joblib
├── model_config.json
└── results.json
```

The saved classification threshold used by the inference service is:

```text
0.40
```

---

# Leakage Prevention

Leakage prevention is a central design principle of Phase 2.

Examples of information excluded from prediction-time model inputs include:

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

Identifiers such as:

```text
order_id
customer_id
customer_unique_id
```

are also excluded from model training.

Additional safeguards include:

- EDA uses training data only
- Preprocessing is fitted on training data only
- Validation data is used for model selection
- Test data is reserved for final evaluation
- Production inference loads the fitted preprocessing object instead of
  fitting a new one

---

# Phase 2 Artifact Pipeline

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

Each stage consumes artifacts from the previous stage instead of rebuilding
the entire workflow.

This makes the ML pipeline easier to reproduce, inspect, debug, and
productionize.

### Task 2 Status

- [x] Order-level ML table created
- [x] Late/on-time target created
- [x] Chronological train/validation/test split created
- [x] Training-only EDA completed
- [x] Leakage investigated
- [x] Feature-engineering pipeline created
- [x] Preprocessor fitted on training data only
- [x] Preprocessing artifacts saved
- [x] Baseline model created
- [x] Candidate models compared
- [x] Imbalance-aware metrics evaluated
- [x] Model tuning performed
- [x] Classification threshold selected
- [x] Final model persisted

**Phase 2 completed.**

---

# Phase 3 — Production Inference

## Task 3: From Notebooks to an Inference Service

Phase 3 converts the fitted Task 2 artifacts into reusable production-oriented
Python modules and an HTTP inference service.

A key architectural rule is:

> Training remains in the notebooks. The inference service loads fitted
> artifacts and never refits the model or preprocessing pipeline.

---

# Inference Architecture

```text
Client
  │
  ▼
FastAPI
  │
  ▼
Input Validation
  │
  ▼
Feature Engineering
  │
  ▼
Saved ColumnTransformer
  │
  ▼
Saved XGBClassifier
  │
  ▼
Probability
  │
  ▼
Threshold = 0.40
  │
  ▼
Late / On Time
```

Production logic is separated from the API layer so that the same inference
pipeline can be called from tests, scripts, or HTTP endpoints.

---

# Production Inference Modules

The reusable inference package is located under:

```text
src/olist_inference/
```

It separates responsibilities for:

- Configuration loading
- Artifact loading
- Input validation
- Feature engineering
- Prediction
- Logging
- Structured prediction-event logging

The service loads:

```text
artifacts/05_features/preprocessor.joblib
artifacts/05_features/feature_list.json
artifacts/05_features/input_features.json
artifacts/06_model/final_model.joblib
artifacts/06_model/model_config.json
```

No fitting occurs during prediction.

---

# Notebook-to-Service Parity

A production inference service must reproduce the transformations used during
model development.

A parity test was performed between direct Task 2 artifact inference and the
Task 3 service.

```text
Notebook probability: 0.44865238666534424
Service probability:  0.44865238666534424
Match: True
```

This confirms that the service reproduces the saved Task 2 inference pipeline.

---

# Input Validation

Incoming inference data is validated before prediction.

Checks include:

- Required timestamp columns
- Valid timestamp values
- Estimated delivery date must not precede purchase date
- Required model inputs
- Valid Brazilian customer state
- Invalid delivery-window detection

Missing numerical values supported by the training pipeline remain compatible
with the saved numerical imputers.

Invalid requests are rejected rather than silently producing a prediction.

---

# FastAPI Service

The model is exposed through FastAPI.

Available endpoints:

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Check service health |
| GET | `/model` | Return model/service information |
| POST | `/predict` | Predict one order |
| POST | `/predict/batch` | Predict multiple orders |

A prediction response contains information such as:

```text
prediction
is_late
probability
threshold
model_version
latency_ms
```

The API automatically provides interactive documentation through Swagger.

Start the service:

```powershell
python -m uvicorn app.main:app --reload
```

Then open:

```text
http://127.0.0.1:8000/docs
```

---

# Logging and Error Handling

The production code uses Python logging instead of `print()` statements.

Application logs capture:

- Service/model initialization
- Prediction requests
- Prediction results
- Probability
- Threshold
- Model version
- Prediction latency
- Errors and tracebacks

Prediction events are also stored in structured JSON Lines format for later
analysis and monitoring.

Runtime logs are excluded from Git.

---

# Automated Testing

Task 3 includes automated tests using `pytest`.

The tests cover:

- Feature engineering
- Raw-input validation
- Invalid input
- Artifact loading
- Classification threshold
- Model prediction
- Notebook-to-service parity
- FastAPI inference
- Invalid API requests
- Batch prediction

Run the complete suite with:

```powershell
python -m pytest -v
```

Code quality is checked using Ruff:

```powershell
python -m ruff check src app tests
```

The current test suite passes successfully.

---

# Data and Model Versioning with DVC

DVC is introduced in Phase 3 to separate large ML artifacts from normal Git
source-code versioning.

Important artifacts tracked through DVC include:

```text
artifacts/03_splits/
artifacts/05_features/
artifacts/06_model/
```

Git tracks the corresponding `.dvc` metadata while DVC manages the artifact
contents.

This creates a foundation for reproducible data and model versioning.

Typical commands include:

```powershell
python -m dvc status
python -m dvc push
python -m dvc pull
```

A DVC remote should be configured for environments that need to reproduce
the artifacts on another machine.

---

# Data Quality

The project includes two levels of input-quality work.

The inference package contains explicit production validation for required
fields and business rules.

Great Expectations was also introduced as the framework for formal
data-quality validation. Great Expectations `1.23.2` is compatible with the
Python 3.12 Task 3 environment.

Further expectation-suite integration remains part of the productionization
roadmap.

---

# Phase 3 Testing Flow

```text
Raw Request
     ↓
Schema / Input Validation
     ↓
Feature Engineering
     ↓
Saved Preprocessor
     ↓
Saved Model
     ↓
Prediction
     ↓
Logging
     ↓
HTTP Response
```

The same core predictor is used by both automated tests and the FastAPI
service.

---

# Project Structure

```text
Dataset_Olist/
│
├── app/
│   ├── __init__.py
│   └── main.py
│
├── assets/
│   └── olist_schema.png
│
├── config/
│   └── config.yaml
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
│   ├── 03_splits.dvc
│   ├── 04_eda/
│   ├── 05_features/
│   ├── 05_features.dvc
│   ├── 06_model/
│   └── 06_model.dvc
│
├── sql/
│   ├── 01_create_tables.sql
│   └── 02_validation.sql
│
├── src/
│   ├── ingest.py
│   └── olist_inference/
│       ├── __init__.py
│       ├── artifacts.py
│       ├── config.py
│       ├── features.py
│       ├── logging_config.py
│       ├── prediction_logger.py
│       ├── predictor.py
│       ├── schemas.py
│       └── validation.py
│
├── tests/
│   ├── integration/
│   │   └── test_api.py
│   ├── model/
│   │   ├── test_artifacts.py
│   │   └── test_predictor.py
│   ├── unit/
│   │   ├── test_features.py
│   │   └── test_validation.py
│   └── test_parity.py
│
├── .dvc/
├── .dvcignore
├── .env.example
├── .gitignore
├── docker-compose.yml
├── requirements.txt
└── README.md
```

The exact repository structure may evolve as later production components are
added.

---

# Running the Project

## 1. Clone the Repository

```powershell
git clone <repository-url>
cd Dataset_Olist
```

---

## 2. Create the Python Environment

Task 3 uses Python 3.12 to support the MLOps dependency stack.

```powershell
py -3.12 -m venv .venv312
.\.venv312\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

---

## 3. Configure Environment Variables

Create `.env` based on:

```text
.env.example
```

Do not commit the real `.env` file.

---

## 4. Start PostgreSQL

```powershell
docker compose up -d
```

---

## 5. Create the Database Tables

From PowerShell, the SQL file can be piped into the PostgreSQL container:

```powershell
Get-Content sql\01_create_tables.sql |
    docker exec -i olist-postgres psql -U olist_user -d olist
```

---

## 6. Add the Raw Olist Data

Place the original Olist CSV files under:

```text
data/raw/
```

Then run:

```powershell
python src\ingest.py
```

---

## 7. Run the Training Pipeline

Run the notebooks in order:

```text
01_read_and_join.ipynb
02_create_labels.ipynb
03_train_val_test_split.ipynb
04_eda.ipynb
05_feature_engineering.ipynb
06_train_tune_evaluate.ipynb
```

This produces the fitted preprocessing and model artifacts required by the
inference service.

If the required artifacts are managed through a configured DVC remote, they
can instead be restored with:

```powershell
python -m dvc pull
```

---

## 8. Run Automated Tests

```powershell
python -m pytest -v
```

---

## 9. Run Code Quality Checks

```powershell
python -m ruff check src app tests
```

---

## 10. Start the Inference API

```powershell
python -m uvicorn app.main:app --reload
```

Swagger documentation:

```text
http://127.0.0.1:8000/docs
```

Health endpoint:

```text
http://127.0.0.1:8000/health
```

Model information:

```text
http://127.0.0.1:8000/model
```

---

# End-to-End Project Flow

```text
                         OLIST DATA
                             │
                             ▼
                    ┌─────────────────┐
                    │     Phase 1     │
                    │ Data Foundation │
                    └─────────────────┘
                             │
                    PostgreSQL + Docker
                             │
                             ▼
                    Validated Relations
                             │
                             ▼
                    ┌─────────────────┐
                    │     Phase 2     │
                    │ ML Development  │
                    └─────────────────┘
                             │
                      Order-Level Data
                             │
                             ▼
                      Target Creation
                             │
                             ▼
                  Train / Validation / Test
                             │
                             ▼
                     Feature Engineering
                             │
                             ▼
                       Model Training
                             │
                             ▼
                       Saved Artifacts
                             │
                             ▼
                    ┌─────────────────┐
                    │     Phase 3     │
                    │    Inference    │
                    └─────────────────┘
                             │
                      Artifact Loading
                             │
                             ▼
                        Validation
                             │
                             ▼
                    Feature Engineering
                             │
                             ▼
                    Saved Preprocessor
                             │
                             ▼
                     XGBoost Classifier
                             │
                             ▼
                          FastAPI
                             │
                             ▼
                     Late / On Time
```

---

# MLOps Principles Applied

The project applies the following MLOps practices:

- Reproducible database setup
- Separation of data, training, and inference stages
- Artifact-based ML workflows
- Chronological evaluation
- Training-only preprocessing
- Leakage-aware feature engineering
- Persisted preprocessing objects
- Persisted model artifacts
- Configuration separated from application code
- No refitting during inference
- Notebook-to-production prediction parity
- Input validation
- Error handling
- Application and prediction logging
- Automated unit and integration testing
- Code-quality checks
- Data/model artifact versioning with DVC
- Production-oriented API design

---

# Project Status

## Phase 1 — Data Foundation

- [x] PostgreSQL environment
- [x] Docker database
- [x] Relational schema
- [x] Olist data ingestion
- [x] Data validation
- [x] SQL relationship testing

**Status: Completed**

## Phase 2 — ML Development

- [x] Order-level ML dataset
- [x] Target creation
- [x] Chronological split
- [x] Training-only EDA
- [x] Leakage analysis
- [x] Feature engineering
- [x] Reproducible preprocessing
- [x] Model comparison
- [x] Model tuning
- [x] Threshold selection
- [x] Final model persistence

**Status: Completed**

## Phase 3 — Production Inference

- [x] Modular inference package
- [x] Configuration management
- [x] Saved artifact loading
- [x] Production feature engineering
- [x] Input validation
- [x] Prediction service
- [x] Notebook-to-service parity
- [x] FastAPI service
- [x] Single prediction endpoint
- [x] Batch prediction endpoint
- [x] Health endpoint
- [x] Model information endpoint
- [x] Logging
- [x] Error handling
- [x] Structured prediction logging
- [x] Automated testing
- [x] API integration testing
- [x] Ruff code-quality checks
- [x] DVC artifact tracking
- [x] Great Expectations-compatible environment
- [ ] Great Expectations suites integrated into inference
- [ ] MLflow experiment tracking and model registry
- [ ] Dockerized inference service
- [ ] CI/CD pipeline
- [ ] Production monitoring and alerting

**Status: In Progress**

---

# Productionization Roadmap

The remaining Phase 3 work focuses on extending the current inference
foundation with:

```text
Great Expectations
        ↓
MLflow Tracking & Model Registry
        ↓
Dockerized Inference Service
        ↓
CI/CD
        ↓
Metrics & Drift Monitoring
        ↓
Production Deployment
```

These components build on the existing tested inference service without
changing the Task 2 training methodology.

---

# Summary

This project demonstrates the evolution of a machine learning system through
three phases:

**Phase 1 — Data Foundation**

Relational Olist data is ingested and validated using PostgreSQL and Docker.

**Phase 2 — ML Development**

The relational data is transformed into a leakage-aware, reproducible machine
learning pipeline for late-delivery prediction.

**Phase 3 — Production Inference**

The fitted Task 2 pipeline is converted into modular production code and
exposed through a tested FastAPI inference service, with configuration,
validation, logging, prediction parity, automated testing, and DVC artifact
versioning.

The project therefore progresses from **raw relational data → reproducible ML
development → production-oriented inference**, while maintaining separation
between training and serving.