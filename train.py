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
# 1. SELECT DATASET
# ============================================================

# If a dataset path is provided from PowerShell:
#
#     python train.py data/production_data_001.csv
#
# then use that dataset.
#
# Otherwise, use the default dataset:
#
#     data/production_data.csv

if len(sys.argv) > 1:
    DATASET_PATH = sys.argv[1]
else:
    DATASET_PATH = "data/production_data.csv"


# Convert to Path for easier checking
dataset_path = Path(DATASET_PATH)


# ============================================================
# 2. CHECK DATASET
# ============================================================

if not dataset_path.exists():
    print("=" * 60)
    print("ERROR")
    print("=" * 60)

    print(f"\nDataset not found:")
    print(dataset_path)

    print("\nPlease check the dataset path.")

    print("\nExample:")
    print("python train.py data/production_data_001.csv")

    sys.exit(1)


# ============================================================
# 3. LOAD DATASET
# ============================================================

data = pd.read_csv(dataset_path)

print("=" * 60)
print("PRODUCTION PLANNING - DATA MINING")
print("=" * 60)

print("\nDataset loaded successfully!")

print(f"Dataset used: {dataset_path}")

print(f"Number of records: {len(data)}")

print("\nDataset preview:")
print(data.head())


# ============================================================
# 4. DEFINE FEATURES AND TARGET
# ============================================================

X = data[
    [
        "Demand",
        "Inventory",
        "Workers",
        "Working_Hours",
        "Raw_Material"
    ]
]

y = data["Production"]


print("\nFeatures:")
print(X.columns.tolist())

print("\nTarget:")
print("Production")


# ============================================================
# 5. TRAIN / TEST SPLIT
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
# 6. CREATE THREE MODELS
# ============================================================

models = {
    "Linear Regression": LinearRegression(),

    "Decision Tree Regression": DecisionTreeRegressor(
        max_depth=6,
        random_state=42
    ),

    "Random Forest Regression": RandomForestRegressor(
        n_estimators=100,
        max_depth=8,
        random_state=42
    )
}


# ============================================================
# 7. TRAIN AND EVALUATE MODELS
# ============================================================

results = []

for name, model in models.items():

    # Train
    model.fit(X_train, y_train)

    # Predict
    predictions = model.predict(X_test)

    # Evaluation
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
# 8. DISPLAY RESULTS
# ============================================================

results_df = pd.DataFrame(results)

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
# 9. FIND BEST MODEL
# ============================================================

best_model_row = results_df.loc[
    results_df["R2"].idxmax()
]

best_model_name = best_model_row["Model"]

best_model = models[best_model_name]


# ============================================================
# 10. SAVE BEST MODEL
# ============================================================

Path("models").mkdir(
    exist_ok=True
)

joblib.dump(
    best_model,
    "models/best_model.pkl"
)

print("\nBest model saved successfully!")

print("Saved as: models/best_model.pkl")


# ============================================================
# 11. DISPLAY BEST MODEL
# ============================================================

print("\n")
print("=" * 60)
print("BEST MODEL")
print("=" * 60)

print("Model:", best_model_row["Model"])

print(
    "R2 Score:",
    round(best_model_row["R2"], 4)
)

print(
    "MAE:",
    round(best_model_row["MAE"], 2)
)

print(
    "RMSE:",
    round(best_model_row["RMSE"], 2)
)


# ============================================================
# 12. SAVE MODEL COMPARISON RESULTS
# ============================================================

results_df.to_csv(
    "models/model_results.csv",
    index=False
)

print("\nModel comparison results saved!")

print(
    "Saved as: models/model_results.csv"
)


# ============================================================
# 13. FINAL INFORMATION
# ============================================================

print("\n")
print("=" * 60)
print("TRAINING COMPLETED")
print("=" * 60)

print(f"Dataset used: {dataset_path}")

print(f"Best model: {best_model_name}")

print(
    f"R2 Score: {best_model_row['R2']:.4f}"
)

print("\nReady for Streamlit prediction!")