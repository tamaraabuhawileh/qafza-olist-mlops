import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine import URL

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

load_dotenv()

DB_NAME = os.getenv("POSTGRES_DB")
DB_USER = os.getenv("POSTGRES_USER")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD")
DB_HOST = os.getenv("POSTGRES_HOST", "localhost")
DB_PORT = os.getenv("POSTGRES_PORT", "5432")


required_variables = {
    "POSTGRES_DB": DB_NAME,
    "POSTGRES_USER": DB_USER,
    "POSTGRES_PASSWORD": DB_PASSWORD,
}

missing_variables = [
    name
    for name, value in required_variables.items()
    if not value
]

if missing_variables:
    raise ValueError(
        "Missing required environment variables: "
        + ", ".join(missing_variables)
    )


DATABASE_URL = URL.create(
    drivername="postgresql+psycopg2",
    username=DB_USER,
    password=DB_PASSWORD,
    host=DB_HOST,
    port=int(DB_PORT),
    database=DB_NAME,
)

DATA_DIR = Path("data/raw")

engine = create_engine(DATABASE_URL)


# ------------------------------------------------------------
# Source CSV -> PostgreSQL table mapping
# ------------------------------------------------------------

TABLES = [
    ("olist_customers_dataset.csv", "customers"),
    ("olist_sellers_dataset.csv", "sellers"),
    ("olist_products_dataset.csv", "products"),
    ("olist_orders_dataset.csv", "orders"),
    ("olist_order_items_dataset.csv", "order_items"),
    ("olist_order_payments_dataset.csv", "order_payments"),
    ("olist_order_reviews_dataset.csv", "order_reviews"),
    ("olist_geolocation_dataset.csv", "geolocation"),
    (
        "product_category_name_translation.csv",
        "product_category_translation",
    ),
]


# ------------------------------------------------------------
# Data ingestion
# ------------------------------------------------------------

def load_table(csv_file: str, table_name: str) -> None:
    """Load one Olist CSV file into its PostgreSQL table."""

    file_path = DATA_DIR / csv_file

    print("=" * 70)
    print(f"Loading file : {csv_file}")
    print(f"Target table : {table_name}")

    if not file_path.exists():
        raise FileNotFoundError(
            f"File not found: {file_path}"
        )

    df = pd.read_csv(file_path)

    # The original Olist products dataset contains
    # misspelled column names ("lenght").
    # Map them to the corrected database schema names.
    if table_name == "products":
        df = df.rename(
            columns={
                "product_name_lenght":
                    "product_name_length",
                "product_description_lenght":
                    "product_description_length",
            }
        )

    print(f"Rows found   : {len(df):,}")
    print(f"Columns      : {len(df.columns)}")

    try:
        df.to_sql(
            table_name,
            engine,
            if_exists="append",
            index=False,
            chunksize=2000,
            method=None,
        )

        print(f"SUCCESS      : {table_name}")

    except Exception as exc:
        print(f"FAILED       : {table_name}")
        print(f"ERROR        : {exc}")
        raise


def main() -> None:
    """Load all Olist datasets into PostgreSQL."""

    for csv_file, table_name in TABLES:
        load_table(csv_file, table_name)

    print("=" * 70)
    print("DATA INGESTION COMPLETED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    main()