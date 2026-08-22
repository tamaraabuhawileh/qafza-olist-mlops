from pathlib import Path
import pandas as pd

DATA_DIR = Path("data/raw")

for file in DATA_DIR.glob("*.csv"):
    df = pd.read_csv(file)

    print("=" * 70)
    print(f"File: {file.name}")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")
    print("Columns:")
    
    for column in df.columns:
        print(f"  - {column}")

    print()