import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import numpy as np




# ============================================================
# 1. LOAD DATASET
# ============================================================

data = pd.read_csv("data/production_data.csv")

print("=" * 60)
print("PRODUCTION PLANNING - DATA MINING")
print("=" * 60)

print("\nDataset loaded successfully!")
print(f"Number of records: {len(data)}")

print("\nDataset preview:")
print(data.head())


# ============================================================
# 2. DEFINE FEATURES AND TARGET
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
# 3. TRAIN / TEST SPLIT
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
# 4. CREATE THREE MODELS
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
# 5. TRAIN AND EVALUATE MODELS
# ============================================================

results = []

for name, model in models.items():

    # Train
    model.fit(X_train, y_train)

    # Predict
    predictions = model.predict(X_test)

    # Evaluation
    mae = mean_absolute_error(y_test, predictions)

    rmse = np.sqrt(
        mean_squared_error(y_test, predictions)
    )

    r2 = r2_score(y_test, predictions)

    results.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    })


# ============================================================
# 6. DISPLAY RESULTS
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
# 7. FIND BEST MODEL
# ============================================================

best_model_row = results_df.loc[
    results_df["R2"].idxmax()
]

# Save the best model
best_model_name = best_model_row["Model"]
best_model = models[best_model_name]

joblib.dump(
    best_model,
    "models/best_model.pkl"
)

print("\nBest model saved successfully!")
print("Saved as: models/best_model.pkl")

print("\n")
print("=" * 60)
print("BEST MODEL")
print("=" * 60)

print("Model:", best_model_row["Model"])
print("R2 Score:", round(best_model_row["R2"], 4))
print("MAE:", round(best_model_row["MAE"], 2))
print("RMSE:", round(best_model_row["RMSE"], 2))

# Save model comparison results
results_df.to_csv(
    "models/model_results.csv",
    index=False
)

print("\nModel comparison results saved!")
print("Saved as: models/model_results.csv")