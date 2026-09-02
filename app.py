import streamlit as st
import pandas as pd
import joblib
from pathlib import Path


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Production Planning",
    page_icon="🏭",
    layout="centered"
)


# ============================================================
# HEADER
# ============================================================

st.title("🏭 Production Planning System")

st.write(
    "Data Mining Based Production Prediction"
)

st.caption(
    "Compare regression models and predict production quantity "
    "using synthetic or externally sourced production-planning data."
)

st.divider()


# ============================================================
# DATA SOURCE SELECTION
# ============================================================

st.subheader("🗂️ Select Dataset Source")

dataset_option = st.radio(
    "Choose the dataset used for prediction:",
    [
        "Synthetic Dataset",
        "External Production Dataset"
    ],
    horizontal=True
)

source = (
    "synthetic"
    if dataset_option == "Synthetic Dataset"
    else "real"
)


# ============================================================
# MODEL PATHS
# ============================================================

model_dir = Path("models") / source

model_path = model_dir / "best_model.pkl"
results_path = model_dir / "model_results.csv"
metadata_path = model_dir / "metadata.pkl"


# ============================================================
# CHECK MODEL FILES
# ============================================================

required_files = [
    model_path,
    results_path,
    metadata_path
]

missing_files = [
    file
    for file in required_files
    if not file.exists()
]

if missing_files:

    st.error(
        f"No trained {source} model was found."
    )

    st.write(
        "Train the model first using:"
    )

    st.code(
        f"python train.py --source {source}"
    )

    st.stop()


# ============================================================
# LOAD MODEL, RESULTS AND METADATA
# ============================================================

model = joblib.load(
    model_path
)

results = pd.read_csv(
    results_path
)

metadata = joblib.load(
    metadata_path
)

best_model_row = results.loc[
    results["R2"].idxmax()
]


# ============================================================
# DATASET INFORMATION
# ============================================================

st.subheader("📁 Dataset Information")

if source == "synthetic":

    st.info(
        "This model was trained using a synthetically generated "
        "production-planning dataset created for academic demonstration."
    )

else:

    st.info(
        "This model was trained using an externally sourced "
        "multi-site production-distribution planning dataset."
    )


col1, col2 = st.columns(2)

with col1:

    st.metric(
        "Dataset Source",
        "Synthetic"
        if source == "synthetic"
        else "External"
    )

with col2:

    st.metric(
        "Records Used",
        metadata["records"]
    )


st.caption(
    f"Dataset: {metadata['dataset']}"
)

st.caption(
    f"Prediction target: {metadata['target']}"
)

st.divider()


# ============================================================
# MODEL PERFORMANCE
# ============================================================

st.subheader("📊 Model Performance")

st.write(
    "Three regression models were trained and evaluated "
    "using the selected dataset."
)

display_results = results.copy()

display_results["MAE"] = (
    display_results["MAE"]
    .round(2)
)

display_results["RMSE"] = (
    display_results["RMSE"]
    .round(2)
)

display_results["R2"] = (
    display_results["R2"]
    .round(4)
)

st.dataframe(
    display_results,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# BEST MODEL
# ============================================================

st.subheader("🏆 Best Performing Model")

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "Best Model",
        best_model_row["Model"]
    )

with col2:

    st.metric(
        "R² Score",
        f"{best_model_row['R2']:.4f}"
    )

with col3:

    st.metric(
        "MAE",
        f"{best_model_row['MAE']:.2f}"
    )


st.info(
    f"**{best_model_row['Model']}** achieved the highest "
    f"R² score for the selected dataset."
)

st.divider()


# ============================================================
# SYNTHETIC DATASET INPUTS
# ============================================================

if source == "synthetic":

    st.subheader("📝 Production Information")

    st.write(
        "Enter production-planning information to predict "
        "the expected production quantity."
    )

    demand = st.number_input(
        "Expected Demand",
        min_value=0,
        value=1000,
        step=50
    )

    inventory = st.number_input(
        "Current Inventory",
        min_value=0,
        value=100,
        step=10
    )

    workers = st.number_input(
        "Number of Workers",
        min_value=1,
        value=30,
        step=1
    )

    working_hours = st.number_input(
        "Working Hours per Day",
        min_value=1.0,
        max_value=24.0,
        value=9.0,
        step=0.5
    )

    raw_material = st.number_input(
        "Available Raw Material",
        min_value=0,
        value=1100,
        step=50
    )

    input_data = pd.DataFrame({
        "Demand": [demand],
        "Inventory": [inventory],
        "Workers": [workers],
        "Working_Hours": [working_hours],
        "Raw_Material": [raw_material]
    })


# ============================================================
# EXTERNAL DATASET INPUTS
# ============================================================

else:

    st.subheader("📝 Multi-site Production Information")

    st.write(
        "Enter production and distribution planning values "
        "to predict production at Site 1."
    )

    st.info(
        "The default values below represent one record from the "
        "external dataset. For that record, the actual Site 1 "
        "production value is **168 units**."
    )


    # ========================================================
    # DEMAND AND INVENTORY
    # ========================================================

    with st.expander(
        "📦 Demand & Inventory",
        expanded=True
    ):

        col1, col2 = st.columns(2)

        with col1:

            ds1 = st.number_input(
                "Demand — Site 1 (DS1)",
                value=117.0
            )

            ds2 = st.number_input(
                "Demand — Site 2 (DS2)",
                value=417.0
            )

            ds3 = st.number_input(
                "Demand — Site 3 (DS3)",
                value=227.0
            )

        with col2:

            i1 = st.number_input(
                "Inventory — Site 1 (I1)",
                value=168.0
            )

            i2 = st.number_input(
                "Inventory — Site 2 (I2)",
                value=136.0
            )

            i3 = st.number_input(
                "Inventory — Site 3 (I3)",
                value=277.0
            )


    # ========================================================
    # PRODUCTION CONSTRAINTS
    # ========================================================

    with st.expander(
        "⚙️ Production Constraints"
    ):

        col1, col2 = st.columns(2)

        with col1:

            u_min = st.number_input(
                "Minimum Production Limit",
                value=0.0
            )

        with col2:

            u_max = st.number_input(
                "Maximum Production Limit",
                value=300.0
            )


    # ========================================================
    # COSTS AND CAPACITY
    # ========================================================

    with st.expander(
        "💰 Costs & Capacity"
    ):

        col1, col2 = st.columns(2)

        with col1:

            mc = st.number_input(
                "Manufacturing Cost (MC123)",
                value=1.0
            )

            shc = st.number_input(
                "Shortage Cost (SHC123)",
                value=20.0
            )

        with col2:

            sc = st.number_input(
                "Storage Cost (SC123)",
                value=10.0
            )

            vc = st.number_input(
                "Vehicle Capacity (VC)",
                value=30.0
            )


    # ========================================================
    # INTER-SITE DISTRIBUTION
    # ========================================================

    with st.expander(
        "🚚 Inter-site Distribution"
    ):

        col1, col2 = st.columns(2)

        with col1:

            t12 = st.number_input(
                "Site 1 → Site 2 (T12)",
                value=0.0
            )

            t21 = st.number_input(
                "Site 2 → Site 1 (T21)",
                value=0.0
            )

            t31 = st.number_input(
                "Site 3 → Site 1 (T31)",
                value=0.0
            )

        with col2:

            t13 = st.number_input(
                "Site 1 → Site 3 (T13)",
                value=0.0
            )

            t23 = st.number_input(
                "Site 2 → Site 3 (T23)",
                value=0.0
            )

            t32 = st.number_input(
                "Site 3 → Site 2 (T32)",
                value=0.0
            )


    input_data = pd.DataFrame({
        "DS1": [ds1],
        "DS2": [ds2],
        "DS3": [ds3],

        "I1": [i1],
        "I2": [i2],
        "I3": [i3],

        "U_Min123": [u_min],
        "U_Max123": [u_max],

        "MC123": [mc],
        "SC123": [sc],
        "SHC123": [shc],

        "VC": [vc],

        "T12": [t12],
        "T13": [t13],
        "T21": [t21],
        "T23": [t23],
        "T31": [t31],
        "T32": [t32]
    })


# ============================================================
# PREDICTION
# ============================================================

if st.button(
    "🔮 Predict Production",
    use_container_width=True,
    type="primary"
):

    prediction = model.predict(
        input_data
    )[0]

    prediction = max(
        0,
        prediction
    )

    prediction = round(
        prediction
    )

    st.divider()

    st.subheader(
        "🎯 Prediction Result"
    )


    if source == "synthetic":

        st.success(
            f"🏭 **Predicted Production: "
            f"{prediction:,} units**"
        )

    else:

        st.success(
            f"🏭 **Predicted Production at Site 1: "
            f"{prediction:,} units**"
        )


    st.info(
        f"The prediction was generated using the trained "
        f"**{best_model_row['Model']}** model."
    )


    if source == "real":

        actual_example = 168
        difference = abs(
            prediction - actual_example
        )

        st.write(
            f"Example record actual value: "
            f"**{actual_example} units**"
        )

        st.write(
            f"Prediction difference for this example: "
            f"**{difference} units**"
        )

        st.caption(
            "This comparison applies only to the default example "
            "record. Overall model performance should be evaluated "
            "using MAE, RMSE, and R²."
        )

    else:

        st.caption(
            "This prediction is based on the synthetic "
            "production-planning model."
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Data Mining Lab Project • Production Planning"
)