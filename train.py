import sys
from pathlib import Path

import pandas as pd
import joblib
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# CONFIGURATION
# ============================================================

SYNTHETIC_FEATURES = [
    "Demand",
    "Inventory",
    "Workers",
    "Working_Hours",
    "Raw_Material"
]

SYNTHETIC_TARGET = "Production"


REAL_FEATURES = [
    # Demand at the three production sites
    "DS1",
    "DS2",
    "DS3",

    # Current inventory at the three sites
    "I1",
    "I2",
    "I3",

    # Production constraints
    "U_Min123",
    "U_Max123",

    # Manufacturing, storage and shortage costs
    "MC123",
    "SC123",
    "SHC123",

    # Vehicle / distribution capacity
    "VC",

    # Inter-site distribution quantities
    "T12",
    "T13",
    "T21",
    "T23",
    "T31",
    "T32"
]

REAL_TARGET = "U1"


# ============================================================
# 1. READ COMMAND-LINE ARGUMENTS
# ============================================================

source = "synthetic"
dataset_argument = None


if "--source" in sys.argv:
    source_index = sys.argv.index("--source")

    try:
        source = sys.argv[source_index + 1].lower()
    except IndexError:
        print("ERROR: Please provide a source after --source")
        print("Example: python train.py --source synthetic")
        sys.exit(1)


if "--dataset" in sys.argv:
    dataset_index = sys.argv.index("--dataset")

    try:
        dataset_argument = sys.argv[dataset_index + 1]
    except IndexError:
        print("ERROR: Please provide a dataset path after --dataset")
        sys.exit(1)


if source not in ["synthetic", "real"]:
    print("ERROR: Source must be either 'synthetic' or 'real'.")
    sys.exit(1)


# ============================================================
# 2. SELECT DATASET
# ============================================================

if source == "synthetic":

    if dataset_argument:
        dataset_path = Path(dataset_argument)

    else:
        synthetic_dir = Path("data") / "synthetic"

        available_files = sorted(
            synthetic_dir.glob("production_data_*.csv")
        )

        if not available_files:
            print("=" * 60)
            print("ERROR")
            print("=" * 60)

            print("\nNo synthetic datasets found.")

            print("\nGenerate one first using:")
            print("python generate_data.py")

            sys.exit(1)

        # Use latest generated synthetic dataset
        dataset_path = available_files[-1]


else:

    if dataset_argument:
        dataset_path = Path(dataset_argument)

    else:
        dataset_path = (
            Path("data")
            / "real"
            / "Multi-site Production-Distribution Prediction.csv"
        )


# ============================================================
# 3. CHECK DATASET
# ============================================================

if not dataset_path.exists():
    print("=" * 60)
    print("ERROR")
    print("=" * 60)

    print("\nDataset not found:")
    print(dataset_path)

    sys.exit(1)


# ============================================================
# 4. LOAD DATASET
# ============================================================

if source == "synthetic":

    data = pd.read_csv(
        dataset_path
    )

else:

    data = pd.read_csv(
        dataset_path,
        sep=";",
        decimal=","
    )


# ============================================================
# 5. CLEAN REAL DATASET
# ============================================================

if source == "real":

    # Remove rows that are completely empty
    data = data.dropna(
        how="all"
    )

    # Keep only relevant columns
    required_columns = (
        REAL_FEATURES
        + [REAL_TARGET]
    )

    missing_columns = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:

        print("=" * 60)
        print("ERROR")
        print("=" * 60)

        print("\nMissing required columns:")

        for column in missing_columns:
            print(f" - {column}")

        sys.exit(1)

    data = data[
        required_columns
    ].copy()

    # Convert columns to numeric
    for column in required_columns:

        data[column] = pd.to_numeric(
            data[column],
            errors="coerce"
        )

    # Remove rows containing missing values
    data = data.dropna()


# ============================================================
# 6. DISPLAY DATASET INFORMATION
# ============================================================

print("=" * 60)
print("PRODUCTION PLANNING - DATA MINING")
print("=" * 60)

print(f"\nDataset source: {source.upper()}")

print(f"Dataset used: {dataset_path}")

print(f"Number of records: {len(data)}")

print("\nDataset preview:")

print(
    data.head()
)


# ============================================================
# 7. DEFINE FEATURES AND TARGET
# ============================================================

if source == "synthetic":

    features = SYNTHETIC_FEATURES
    target = SYNTHETIC_TARGET

else:

    features = REAL_FEATURES
    target = REAL_TARGET


X = data[
    features
]

y = data[
    target
]


print("\nFeatures:")

for feature in features:
    print(f" - {feature}")


print("\nTarget:")
print(target)


# ============================================================
# 8. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


print("\nTraining records:", len(X_train))

print("Testing records:", len(X_test))


# ============================================================
# 9. CREATE THREE MODELS
# ============================================================

models = {

    "Linear Regression":
        LinearRegression(),

    "Decision Tree Regression":
        DecisionTreeRegressor(
            max_depth=6,
            random_state=42
        ),

    "Random Forest Regression":
        RandomForestRegressor(
            n_estimators=100,
            max_depth=8,
            random_state=42
        )
}


# ============================================================
# 10. TRAIN AND EVALUATE MODELS
# ============================================================

results = []


for name, model in models.items():

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_test
    )


    mae = mean_absolute_error(
        y_test,
        predictions
    )


    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )


    r2 = r2_score(
        y_test,
        predictions
    )


    results.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    })


# ============================================================
# 11. DISPLAY RESULTS
# ============================================================

results_df = pd.DataFrame(
    results
)


print("\n")
print("=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)


print(
    results_df.to_string(
        index=False,
        formatters={
            "MAE": "{:.2f}".format,
            "RMSE": "{:.2f}".format,
            "R2": "{:.4f}".format
        }
    )
)


# ============================================================
# 12. FIND BEST MODEL
# ============================================================

best_model_row = results_df.loc[
    results_df["R2"].idxmax()
]

best_model_name = best_model_row[
    "Model"
]

best_model = models[
    best_model_name
]


# ============================================================
# 13. CREATE MODEL DIRECTORY
# ============================================================

model_dir = (
    Path("models")
    / source
)

model_dir.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 14. SAVE BEST MODEL
# ============================================================

model_path = (
    model_dir
    / "best_model.pkl"
)


joblib.dump(
    best_model,
    model_path
)


# ============================================================
# 15. SAVE MODEL RESULTS
# ============================================================

results_path = (
    model_dir
    / "model_results.csv"
)


results_df.to_csv(
    results_path,
    index=False
)


# ============================================================
# 16. SAVE MODEL METADATA
# ============================================================

metadata = {

    "source": source,

    "dataset": str(
        dataset_path
    ),

    "features": features,

    "target": target,

    "best_model": best_model_name,

    "records": len(data),

    "r2": float(
        best_model_row["R2"]
    ),

    "mae": float(
        best_model_row["MAE"]
    ),

    "rmse": float(
        best_model_row["RMSE"]
    )
}


metadata_path = (
    model_dir
    / "metadata.pkl"
)


joblib.dump(
    metadata,
    metadata_path
)


# ============================================================
# 17. DISPLAY BEST MODEL
# ============================================================

print("\n")
print("=" * 60)
print("BEST MODEL")
print("=" * 60)


print(
    "Model:",
    best_model_name
)

print(
    "R2 Score:",
    round(
        best_model_row["R2"],
        4
    )
)

print(
    "MAE:",
    round(
        best_model_row["MAE"],
        2
    )
)

print(
    "RMSE:",
    round(
        best_model_row["RMSE"],
        2
    )
)


# ============================================================
# 18. FINAL INFORMATION
# ============================================================

print("\n")
print("=" * 60)
print("TRAINING COMPLETED")
print("=" * 60)


print(
    f"Dataset source: {source.upper()}"
)

print(
    f"Dataset used: {dataset_path}"
)

print(
    f"Best model: {best_model_name}"
)

print(
    f"R2 Score: {best_model_row['R2']:.4f}"
)

print(
    f"\nModel saved to: {model_path}"
)

print(
    f"Results saved to: {results_path}"
)

print(
    f"Metadata saved to: {metadata_path}"
)