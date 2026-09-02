import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = Path("data") / "synthetic"
DATA_DIR.mkdir(parents=True, exist_ok=True)

NUMBER_OF_RECORDS = 400


# ============================================================
# FIND NEXT DATASET NUMBER
# ============================================================

existing_files = list(DATA_DIR.glob("production_data_*.csv"))

numbers = []

for file in existing_files:
    try:
        number = int(file.stem.split("_")[-1])
        numbers.append(number)
    except ValueError:
        pass

next_number = max(numbers, default=0) + 1

output_file = DATA_DIR / f"production_data_{next_number:03d}.csv"


# ============================================================
# GENERATE DATA
# ============================================================

np.random.seed(42 + next_number)

demand = np.random.randint(
    500,
    2000,
    NUMBER_OF_RECORDS
)

inventory = np.random.randint(
    50,
    500,
    NUMBER_OF_RECORDS
)

workers = np.random.randint(
    10,
    80,
    NUMBER_OF_RECORDS
)

working_hours = np.round(
    np.random.uniform(
        6,
        12,
        NUMBER_OF_RECORDS
    ),
    1
)

raw_material = np.random.randint(
    500,
    2500,
    NUMBER_OF_RECORDS
)


# ============================================================
# GENERATE PRODUCTION TARGET
# ============================================================

production = (
    0.75 * demand
    + 0.15 * inventory
    + 8 * workers
    + 35 * working_hours
    + 0.20 * raw_material
    + np.random.normal(
        0,
        20,
        NUMBER_OF_RECORDS
    )
)

production = np.maximum(
    production.round(),
    0
)


# ============================================================
# CREATE DATAFRAME
# ============================================================

df = pd.DataFrame({
    "Demand": demand,
    "Inventory": inventory,
    "Workers": workers,
    "Working_Hours": working_hours,
    "Raw_Material": raw_material,
    "Production": production
})


# ============================================================
# SAVE DATASET
# ============================================================

df.to_csv(
    output_file,
    index=False
)


# ============================================================
# DISPLAY INFORMATION
# ============================================================

print("=" * 60)
print("SYNTHETIC PRODUCTION DATASET GENERATED")
print("=" * 60)

print(f"Records : {len(df)}")
print(f"Columns : {len(df.columns)}")
print(f"Saved to: {output_file}")

print("\nColumns:")

for column in df.columns:
    print(f" - {column}")

print("\nFirst 5 records:")
print(df.head())

print("\nSynthetic dataset generation completed successfully!")