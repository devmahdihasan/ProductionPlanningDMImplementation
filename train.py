import sys
from pathlib import Path

import pandas as pd

from ml.training import (
    save_training_result,
    train_regression_models,
)


# ============================================================
# CONFIGURATION
# ============================================================

SYNTHETIC_FEATURES = [
    "Demand",
    "Inventory",
    "Workers",
    "Working_Hours",
    "Raw_Material",
]

SYNTHETIC_TARGET = "Production"


REAL_FEATURES = [
    "DS1",
    "DS2",
    "DS3",

    "I1",
    "I2",
    "I3",

    "U_Min123",
    "U_Max123",

    "MC123",
    "SC123",
    "SHC123",

    "VC",

    "T12",
    "T13",
    "T21",
    "T23",
    "T31",
    "T32",
]

REAL_TARGET = "U1"


# ============================================================
# READ COMMAND-LINE ARGUMENTS
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
    print(
        "ERROR: Source must be either "
        "'synthetic' or 'real'."
    )
    sys.exit(1)


# ============================================================
# SELECT DATASET
# ============================================================

if source == "synthetic":

    if dataset_argument:
        dataset_path = Path(
            dataset_argument
        )

    else:
        synthetic_dir = (
            Path("data")
            / "synthetic"
        )

        available_files = sorted(
            synthetic_dir.glob(
                "production_data_*.csv"
            )
        )

        if not available_files:

            print("=" * 60)
            print("ERROR")
            print("=" * 60)

            print(
                "\nNo synthetic datasets found."
            )

            print(
                "\nGenerate one first using:"
            )

            print(
                "python generate_data.py"
            )

            sys.exit(1)

        dataset_path = (
            available_files[-1]
        )


else:

    if dataset_argument:
        dataset_path = Path(
            dataset_argument
        )

    else:
        dataset_path = (
            Path("data")
            / "real"
            / "Multi-site Production-Distribution Prediction.csv"
        )


# ============================================================
# CHECK DATASET
# ============================================================

if not dataset_path.exists():

    print("=" * 60)
    print("ERROR")
    print("=" * 60)

    print("\nDataset not found:")
    print(dataset_path)

    sys.exit(1)


# ============================================================
# LOAD DATASET
# ============================================================

if source == "synthetic":

    data = pd.read_csv(
        dataset_path
    )

    features = SYNTHETIC_FEATURES
    target = SYNTHETIC_TARGET


else:

    data = pd.read_csv(
        dataset_path,
        sep=";",
        decimal=",",
    )

    # Remove completely empty rows
    data = data.dropna(
        how="all"
    )

    features = REAL_FEATURES
    target = REAL_TARGET


# ============================================================
# DISPLAY DATASET INFORMATION
# ============================================================

print("=" * 60)
print("PRODUCTION PLANNING - DATA MINING")
print("=" * 60)

print(
    f"\nDataset source: {source.upper()}"
)

print(
    f"Dataset used: {dataset_path}"
)

print(
    f"Number of raw records: {len(data)}"
)

print("\nDataset preview:")
print(
    data.head()
)

print("\nFeatures:")

for feature in features:
    print(
        f" - {feature}"
    )

print("\nTarget:")
print(
    target
)


# ============================================================
# TRAIN MODELS
# ============================================================

try:

    training_result = (
        train_regression_models(
            data=data,
            features=features,
            target=target,
            test_size=0.20,
            random_state=42,
        )
    )

except ValueError as error:

    print("=" * 60)
    print("ERROR")
    print("=" * 60)

    print(
        f"\n{error}"
    )

    sys.exit(1)


# ============================================================
# DISPLAY TRAIN / TEST INFORMATION
# ============================================================

print(
    "\nUsable records:",
    training_result["records"],
)

print(
    "Training records:",
    training_result["train_records"],
)

print(
    "Testing records:",
    training_result["test_records"],
)


print(
    "\nNumeric features:"
)

for feature in training_result[
    "numeric_features"
]:
    print(
        f" - {feature}"
    )


if training_result[
    "categorical_features"
]:

    print(
        "\nCategorical features:"
    )

    for feature in training_result[
        "categorical_features"
    ]:
        print(
            f" - {feature}"
        )


# ============================================================
# DISPLAY RESULTS
# ============================================================

results_df = training_result[
    "results"
]


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
            "R2": "{:.4f}".format,
        },
    )
)


# ============================================================
# SAVE MODEL OUTPUT
# ============================================================

model_dir = (
    Path("models")
    / source
)


saved_paths = save_training_result(
    training_result=training_result,
    model_dir=model_dir,
    source=source,
    dataset_path=str(
        dataset_path
    ),
)


# ============================================================
# DISPLAY BEST MODEL
# ============================================================

print("\n")
print("=" * 60)
print("BEST MODEL")
print("=" * 60)


print(
    "Model:",
    training_result[
        "best_model_name"
    ],
)

print(
    "R2 Score:",
    round(
        training_result["r2"],
        4,
    ),
)

print(
    "MAE:",
    round(
        training_result["mae"],
        2,
    ),
)

print(
    "RMSE:",
    round(
        training_result["rmse"],
        2,
    ),
)


# ============================================================
# FINAL INFORMATION
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
    "Best model:",
    training_result[
        "best_model_name"
    ],
)

print(
    f"R2 Score: "
    f"{training_result['r2']:.4f}"
)

print(
    "\nModel saved to:",
    saved_paths[
        "model_path"
    ],
)

print(
    "Results saved to:",
    saved_paths[
        "results_path"
    ],
)

print(
    "Metadata saved to:",
    saved_paths[
        "metadata_path"
    ],
)