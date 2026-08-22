\# Olist MLOps Project



\## Qafza MLOps Training 2026/2027 — Task 1



\### Get the Data Into a Database



This repository contains my implementation of \*\*Task 1\*\* of the Qafza MLOps Training 2026/2027.



The goal of this task is to establish the data foundation for the project by loading the Brazilian E-Commerce Public Dataset by Olist into a relational PostgreSQL database running locally with Docker.



This database will serve as the source for the subsequent stages of the MLOps project.



\---



\## Task Objective



The objectives of Task 1 are to:



\- Understand the Olist dataset and its relational structure

\- Understand the future ML problem: predicting whether an order will be delivered late or on time

\- Run PostgreSQL locally using Docker

\- Design the relational database schema

\- Ingest the Olist CSV datasets into PostgreSQL

\- Validate the loaded data

\- Query the database

\- Test relationships between tables using SQL joins



EDA, feature engineering, and model training are intentionally outside the scope of Task 1.



\---



\## Tech Stack



\- Python

\- PostgreSQL 16

\- Docker / Docker Compose

\- Pandas

\- SQLAlchemy

\- psycopg2

\- SQL



\---



\## Project Structure



```text

Dataset\_Olist/

│

├── data/

│   └── raw/

│       └── .gitkeep

│

├── sql/

│   ├── 01\_create\_tables.sql

│   └── 02\_validation.sql

│

├── src/

│   ├── inspect\_data.py

│   └── ingest.py

│

├── .gitignore

├── .env.example

├── docker-compose.yml

├── requirements.txt

└── README.md

```



The raw Olist CSV files are not committed to the repository.



\---



\## Dataset



The project uses the Brazilian E-Commerce Public Dataset by Olist.



The source contains nine CSV files:



```text

olist\_customers\_dataset.csv

olist\_geolocation\_dataset.csv

olist\_orders\_dataset.csv

olist\_order\_items\_dataset.csv

olist\_order\_payments\_dataset.csv

olist\_order\_reviews\_dataset.csv

olist\_products\_dataset.csv

olist\_sellers\_dataset.csv

product\_category\_name\_translation.csv

```



The files should be placed under:



```text

data/raw/

```



\---



\## Relational Structure



The main relationships are:



```text

customers

&#x20;   │

&#x20;   │ customer\_id

&#x20;   ▼

orders

&#x20;   │

&#x20;   │ order\_id

&#x20;   ├──────────────► order\_items

&#x20;   │                    │

&#x20;   │                    ├── product\_id ──► products

&#x20;   │                    │

&#x20;   │                    └── seller\_id ───► sellers

&#x20;   │

&#x20;   ├──────────────► order\_payments

&#x20;   │

&#x20;   └──────────────► order\_reviews

```



Important identifiers include:



| Key | Relationship |

|---|---|

| `customer\_id` | Customers → Orders |

| `order\_id` | Orders → Items / Payments / Reviews |

| `product\_id` | Order Items → Products |

| `seller\_id` | Order Items → Sellers |



\---



\## Machine Learning Problem



The later stages of this project will build a model to predict whether an order will be delivered \*\*late or on time\*\*.



The relevant delivery fields include:



```text

order\_delivered\_customer\_date

order\_estimated\_delivery\_date

```



Conceptually:



```text

Actual delivery > Estimated delivery

→ Late



Actual delivery <= Estimated delivery

→ On Time

```



The target is not created in Task 1. Label creation and ML dataset preparation are part of the next stage of the project.



\---



\## Database Setup



PostgreSQL 16 runs locally inside a Docker container.



Start the database with:



```bash

docker compose up -d

```



Verify that the container is running:



```bash

docker ps

```



Connect to PostgreSQL:



```bash

docker exec -it olist-postgres psql -U olist\_user -d olist

```



\---



\## Create the Database Schema



The relational schema is defined in:



```text

sql/01\_create\_tables.sql

```



On Windows Command Prompt:



```bash

docker exec -i olist-postgres psql -U olist\_user -d olist < sql\\01\_create\_tables.sql

```



The schema defines primary keys, composite keys, and foreign-key relationships between the Olist tables.



\---



\## Data Ingestion



Activate the virtual environment:



```bash

.venv\\Scripts\\activate

```



Install dependencies:



```bash

pip install -r requirements.txt

```



Run the ingestion pipeline:



```bash

python src\\ingest.py

```



The ingestion script reads the raw CSV files and loads them into their corresponding PostgreSQL tables.



Large datasets are inserted in chunks to avoid oversized database insert operations.



The raw source data is preserved unchanged.



\---



\## Loaded Data



The completed ingestion produced:



| Table | Rows |

|---|---:|

| customers | 99,441 |

| sellers | 3,095 |

| products | 32,951 |

| orders | 99,441 |

| order\_items | 112,650 |

| order\_payments | 103,886 |

| order\_reviews | 99,224 |

| geolocation | 1,000,163 |

| product\_category\_translation | 71 |



All nine source datasets were successfully loaded into PostgreSQL.



\---



\## Validation



Validation queries are stored in:



```text

sql/02\_validation.sql

```



They include:



\- Row-count validation

\- Reading sample records

\- Two-table joins

\- Multi-table joins

\- Inspection of fields relevant to the future delivery prediction problem



Example:



```sql

SELECT

&#x20;   o.order\_id,

&#x20;   o.order\_status,

&#x20;   c.customer\_city,

&#x20;   c.customer\_state

FROM orders AS o

JOIN customers AS c

&#x20;   ON o.customer\_id = c.customer\_id

LIMIT 20;

```



This validates the relationship:



```text

customers.customer\_id

&#x20;       ↓

orders.customer\_id

```



\---



\## Implementation Notes



During ingestion, two issues were identified and resolved.



\### Large Inserts



Loading large datasets in a single database operation resulted in oversized SQL inserts.



The ingestion pipeline was updated to use chunked inserts.



\### Product Column Naming



The original Olist products dataset contains:



```text

product\_name\_lenght

product\_description\_lenght

```



The database uses:



```text

product\_name\_length

product\_description\_length

```



The ingestion layer maps these source column names to the database schema while preserving the original raw CSV files.



\---



\## Task 1 Status



\- \[x] Olist dataset downloaded

\- \[x] Dataset tables and relationships understood

\- \[x] Late/on-time delivery problem understood

\- \[x] PostgreSQL running locally

\- \[x] PostgreSQL containerized using Docker

\- \[x] Relational database schema created

\- \[x] All nine source datasets ingested

\- \[x] Database queried successfully

\- \[x] Table row counts validated

\- \[x] SQL joins tested



\*\*Task 1 completed.\*\*



\---



\## Next Stage



Task 2 will build directly on the PostgreSQL database created in Task 1.



The next stage will cover:



1\. Reading and joining the database tables into one ML table

2\. Creating the late/on-time delivery label

3\. Creating train, validation, and test splits

4\. Performing EDA on the training data

5\. Feature engineering and persistence of fitted transformations

6\. Training, tuning, and evaluating the first model



Intermediate outputs will be persisted as artifacts so that each stage can consume the output of the previous stage reproducibly.

