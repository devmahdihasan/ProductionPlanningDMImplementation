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
    # 01 / DATASET
    # ========================================================

    st.markdown(
        """
        <div class="workflow-section">
            <span class="workflow-number">01 /</span>
            <span class="workflow-title">DATASET</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write(
        "Synthetic production-planning dataset generated specifically "
        "for demonstrating the regression workflow."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Source",
            "Synthetic",
        )

    with col2:
        st.metric(
            "Records",
            metadata["records"],
        )

    with col3:
        st.metric(
            "Target",
            metadata["target"],
        )

    st.caption(
        f"Dataset: {metadata['dataset']}"
    )

    st.divider()

    # ========================================================
    # 02 / MODEL COMPARISON
    # ========================================================

    st.markdown(
        """
        <div class="workflow-section">
            <span class="workflow-number">02 /</span>
            <span class="workflow-title">MODEL COMPARISON</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write(
        "Three regression models were trained and evaluated "
        "using the same synthetic dataset."
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
        hide_index=True,
    )

    st.markdown(
        """
        <div class="workflow-section">
            <span class="workflow-number">BEST /</span>
            <span class="workflow-title">SELECTED MODEL</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Model",
            best_model_row["Model"],
        )

    with col2:
        st.metric(
            "R²",
            f"{best_model_row['R2']:.4f}",
        )

    with col3:
        st.metric(
            "MAE",
            f"{best_model_row['MAE']:.2f}",
        )

    st.caption(
        "The model with the highest R² score is selected "
        "for prediction."
    )

    st.divider()

    # ========================================================
    # 03 / PRODUCTION INPUTS
    # ========================================================

    st.markdown(
        """
        <div class="workflow-section">
            <span class="workflow-number">03 /</span>
            <span class="workflow-title">PRODUCTION INPUTS</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write(
        "Enter production-planning values to estimate "
        "the expected production quantity."
    )

    col1, col2 = st.columns(2)

    with col1:
        demand = st.number_input(
            "Expected Demand",
            min_value=0,
            value=1000,
            step=50,
            key="synthetic_demand",
        )

        workers = st.number_input(
            "Number of Workers",
            min_value=1,
            value=30,
            step=1,
            key="synthetic_workers",
        )

        raw_material = st.number_input(
            "Available Raw Material",
            min_value=0,
            value=1100,
            step=50,
            key="synthetic_raw_material",
        )

    with col2:
        inventory = st.number_input(
            "Current Inventory",
            min_value=0,
            value=100,
            step=10,
            key="synthetic_inventory",
        )

        working_hours = st.number_input(
            "Working Hours per Day",
            min_value=1.0,
            max_value=24.0,
            value=9.0,
            step=0.5,
            key="synthetic_working_hours",
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
    # 04 / PREDICTION
    # ========================================================

    st.markdown(
        """
        <div class="workflow-section">
            <span class="workflow-number">04 /</span>
            <span class="workflow-title">PREDICTION</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button(
        "Generate production prediction",
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

        st.metric(
            "Predicted Production",
            f"{prediction:,} units",
        )

        st.caption(
            f"Prediction generated using {best_model_row['Model']} "
            "on the synthetic production-planning model."
        )