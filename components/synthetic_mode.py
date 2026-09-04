from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


def render_synthetic_mode():
    source = "synthetic"

    model_dir = Path("models") / source

    model_path = model_dir / "best_model.pkl"
    results_path = model_dir / "model_results.csv"
    metadata_path = model_dir / "metadata.pkl"

    required_files = [
        model_path,
        results_path,
        metadata_path,
    ]

    missing_files = [
        file
        for file in required_files
        if not file.exists()
    ]

    if missing_files:
        st.error(
            "No trained synthetic model was found."
        )

        st.write(
            "Train the model first using:"
        )

        st.code(
            "python train.py --source synthetic"
        )

        return


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


    # ========================================================
    # DATASET INFORMATION
    # ========================================================

    st.subheader(
        "📁 Dataset Information"
    )

    st.info(
        "This model was trained using a synthetically generated "
        "production-planning dataset created for academic demonstration."
    )


    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Dataset Source",
            "Synthetic",
        )

    with col2:
        st.metric(
            "Records Used",
            metadata["records"],
        )


    st.caption(
        f"Dataset: {metadata['dataset']}"
    )

    st.caption(
        f"Prediction target: {metadata['target']}"
    )

    st.divider()


    # ========================================================
    # MODEL PERFORMANCE
    # ========================================================

    st.subheader(
        "📊 Model Performance"
    )

    st.write(
        "Three regression models were trained and evaluated "
        "using the synthetic dataset."
    )


    display_results = (
        results.copy()
    )

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
        hide_index=True,
    )


    # ========================================================
    # BEST MODEL
    # ========================================================

    st.subheader(
        "🏆 Best Performing Model"
    )


    col1, col2, col3 = st.columns(3)


    with col1:
        st.metric(
            "Best Model",
            best_model_row["Model"],
        )

    with col2:
        st.metric(
            "R² Score",
            f"{best_model_row['R2']:.4f}",
        )

    with col3:
        st.metric(
            "MAE",
            f"{best_model_row['MAE']:.2f}",
        )


    st.info(
        f"**{best_model_row['Model']}** achieved the highest "
        f"R² score for the synthetic dataset."
    )

    st.divider()


    # ========================================================
    # PRODUCTION INPUTS
    # ========================================================

    st.subheader(
        "📝 Production Information"
    )

    st.write(
        "Enter production-planning information "
        "to predict the expected production quantity."
    )


    demand = st.number_input(
        "Expected Demand",
        min_value=0,
        value=1000,
        step=50,
        key="synthetic_demand",
    )

    inventory = st.number_input(
        "Current Inventory",
        min_value=0,
        value=100,
        step=10,
        key="synthetic_inventory",
    )

    workers = st.number_input(
        "Number of Workers",
        min_value=1,
        value=30,
        step=1,
        key="synthetic_workers",
    )

    working_hours = st.number_input(
        "Working Hours per Day",
        min_value=1.0,
        max_value=24.0,
        value=9.0,
        step=0.5,
        key="synthetic_working_hours",
    )

    raw_material = st.number_input(
        "Available Raw Material",
        min_value=0,
        value=1100,
        step=50,
        key="synthetic_raw_material",
    )


    input_data = pd.DataFrame(
        {
            "Demand": [demand],
            "Inventory": [inventory],
            "Workers": [workers],
            "Working_Hours": [working_hours],
            "Raw_Material": [raw_material],
        }
    )


    # ========================================================
    # PREDICTION
    # ========================================================

    if st.button(
        "🔮 Predict Production",
        use_container_width=True,
        type="primary",
        key="synthetic_predict_button",
    ):

        prediction = model.predict(
            input_data
        )[0]

        prediction = max(
            0,
            prediction,
        )

        prediction = round(
            prediction
        )


        st.divider()

        st.subheader(
            "🎯 Prediction Result"
        )


        st.success(
            f"🏭 **Predicted Production: "
            f"{prediction:,} units**"
        )


        st.info(
            "The prediction was generated using "
            f"the trained **{best_model_row['Model']}** model."
        )


        st.caption(
            "This prediction is based on the synthetic "
            "production-planning model."
        )