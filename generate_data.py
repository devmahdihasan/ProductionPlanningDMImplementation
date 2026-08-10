import numpy as np
import pandas as pd

# Make results reproducible
np.random.seed(42)

# Number of production records
n = 300

# Generate production-related data
demand = np.random.randint(400, 1501, n)
inventory = np.random.randint(30, 301, n)
workers = np.random.randint(10, 51, n)
working_hours = np.round(np.random.uniform(6, 12, n), 1)
raw_material = np.random.randint(500, 1801, n)

# Generate production quantity
production = (
    0.55 * demand
    - 0.20 * inventory
    + 5 * workers
    + 20 * working_hours
    + 0.25 * raw_material
    + np.random.normal(0, 25, n)
)

# Make sure production is positive and integer
production = np.maximum(production, 50).astype(int)

# Create DataFrame
data = pd.DataFrame({
    "Demand": demand,
    "Inventory": inventory,
    "Workers": workers,
    "Working_Hours": working_hours,
    "Raw_Material": raw_material,
    "Production": production
})

# Save dataset
data.to_csv("data/production_data.csv", index=False)

print("Production dataset created successfully!")
print(f"Number of records: {len(data)}")
print("\nFirst 10 records:")
print(data.head(10))