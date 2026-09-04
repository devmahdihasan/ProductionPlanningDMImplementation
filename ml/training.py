from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor


# ============================================================
# MODEL DEFINITIONS
# ============================================================

def get_regression_models():
    return {
        "Linear Regression": LinearRegression(),

        "Decision Tree Regression": DecisionTreeRegressor(
            max_depth=6,
            random_state=42,
        ),

        "Random Forest Regression": RandomForestRegressor(
            n_estimators=100,
            max_depth=8,
            random_state=42,
        ),
    }


# ============================================================
# DATA CLEANING HELPERS
# ============================================================

def convert_numeric_like_columns(
    data: pd.DataFrame,
    exclude_columns: list[str] | None = None,
):
    cleaned_data = data.copy()

    exclude_columns = exclude_columns or []

    for column in cleaned_data.columns:

        if column in exclude_columns:
            continue

        if pd.api.types.is_numeric_dtype(
            cleaned_data[column]
        ):
            continue

        converted = pd.to_numeric(
            cleaned_data[column],
            errors="coerce",
        )

        original_non_null = (
            cleaned_data[column]
            .notna()
            .sum()
        )

        converted_non_null = (
            converted
            .notna()
            .sum()
        )

        if original_non_null == 0:
            continue

        conversion_ratio = (
            converted_non_null
            / original_non_null
        )

        if conversion_ratio >= 0.90:
            cleaned_data[column] = converted

    return cleaned_data


# ============================================================
# DATA PREPARATION
# ============================================================

def prepare_regression_data(
    data: pd.DataFrame,
    features: list[str],
    target: str,
):
    required_columns = features + [target]

    missing_columns = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing_columns)
        )

    prepared_data = data[
        required_columns
    ].copy()

    prepared_data = convert_numeric_like_columns(
        prepared_data,
        exclude_columns=[],
    )

    prepared_data[target] = pd.to_numeric(
        prepared_data[target],
        errors="coerce",
    )

    prepared_data = prepared_data.dropna(
        subset=[target]
    )

    prepared_data = prepared_data.dropna(
        how="all",
        subset=features,
    )

    if len(prepared_data) < 10:
        raise ValueError(
            "Not enough usable records for model training."
        )

    if prepared_data[target].nunique() < 2:
        raise ValueError(
            "The selected target column does not contain "
            "enough variation for regression."
        )

    X = prepared_data[features]
    y = prepared_data[target]

    return X, y, prepared_data


# ============================================================
# PREPROCESSOR
# ============================================================

def create_preprocessor(
    X: pd.DataFrame,
):
    numeric_features = (
        X.select_dtypes(
            include=["number"]
        )
        .columns
        .tolist()
    )

    categorical_features = [
        column
        for column in X.columns
        if column not in numeric_features
    ]

    transformers = []

    if numeric_features:
        numeric_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median"
                    ),
                ),
            ]
        )

        transformers.append(
            (
                "numeric",
                numeric_pipeline,
                numeric_features,
            )
        )

    if categorical_features:
        categorical_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="most_frequent"
                    ),
                ),

                (
                    "encoder",
                    OneHotEncoder(
                        handle_unknown="ignore"
                    ),
                ),
            ]
        )

        transformers.append(
            (
                "categorical",
                categorical_pipeline,
                categorical_features,
            )
        )

    if not transformers:
        raise ValueError(
            "No usable feature columns were found."
        )

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )

    return (
        preprocessor,
        numeric_features,
        categorical_features,
    )


# ============================================================
# TRAIN AND EVALUATE
# ============================================================

def train_regression_models(
    data: pd.DataFrame,
    features: list[str],
    target: str,
    test_size: float = 0.20,
    random_state: int = 42,
):
    if not features:
        raise ValueError(
            "Please select at least one feature column."
        )

    if target in features:
        raise ValueError(
            "The target column cannot also be used as a feature."
        )

    X, y, prepared_data = prepare_regression_data(
        data=data,
        features=features,
        target=target,
    )

    (
        preprocessor,
        numeric_features,
        categorical_features,
    ) = create_preprocessor(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )

    models = get_regression_models()

    trained_models = {}
    results = []

    for name, estimator in models.items():

        pipeline = Pipeline(
            steps=[
                (
                    "preprocessor",
                    preprocessor,
                ),
                (
                    "model",
                    estimator,
                ),
            ]
        )

        pipeline.fit(
            X_train,
            y_train,
        )

        predictions = pipeline.predict(
            X_test
        )

        mae = mean_absolute_error(
            y_test,
            predictions,
        )

        rmse = np.sqrt(
            mean_squared_error(
                y_test,
                predictions,
            )
        )

        r2 = r2_score(
            y_test,
            predictions,
        )

        trained_models[name] = pipeline

        results.append({
            "Model": name,
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
        })

    results_df = pd.DataFrame(
        results
    )

    best_model_row = results_df.loc[
        results_df["R2"].idxmax()
    ]

    best_model_name = (
        best_model_row["Model"]
    )

    best_model = trained_models[
        best_model_name
    ]

    return {
        "best_model": best_model,
        "best_model_name": best_model_name,
        "results": results_df,
        "features": features,
        "target": target,
        "records": len(prepared_data),
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "train_records": len(X_train),
        "test_records": len(X_test),
        "r2": float(
            best_model_row["R2"]
        ),
        "mae": float(
            best_model_row["MAE"]
        ),
        "rmse": float(
            best_model_row["RMSE"]
        ),
    }


# ============================================================
# SAVE TRAINING OUTPUT
# ============================================================

def save_training_result(
    training_result: dict,
    model_dir: Path,
    source: str,
    dataset_path: str,
):
    model_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_path = (
        model_dir
        / "best_model.pkl"
    )

    results_path = (
        model_dir
        / "model_results.csv"
    )

    metadata_path = (
        model_dir
        / "metadata.pkl"
    )

    joblib.dump(
        training_result["best_model"],
        model_path,
    )

    training_result[
        "results"
    ].to_csv(
        results_path,
        index=False,
    )

    metadata = {
        "source": source,
        "dataset": dataset_path,
        "features": training_result[
            "features"
        ],
        "target": training_result[
            "target"
        ],
        "best_model": training_result[
            "best_model_name"
        ],
        "records": training_result[
            "records"
        ],
        "numeric_features": training_result[
            "numeric_features"
        ],
        "categorical_features": training_result[
            "categorical_features"
        ],
        "r2": training_result["r2"],
        "mae": training_result["mae"],
        "rmse": training_result["rmse"],
    }

    joblib.dump(
        metadata,
        metadata_path,
    )

    return {
        "model_path": model_path,
        "results_path": results_path,
        "metadata_path": metadata_path,
    }