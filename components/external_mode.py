from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


def render_external_mode():
    source = "real"

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
            "No trained external model was found."
        )

        st.write(
            "Train the model first using:"
        )

        st.code(
            "python train.py --source real"
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
        "Externally sourced multi-site production-distribution "
        "planning dataset used for Site 1 production prediction."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Source",
            "External",
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
        "using the same external dataset."
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
    # 03 / PLANNING INPUTS
    # ========================================================

    st.markdown(
        """
        <div class="workflow-section">
            <span class="workflow-number">03 /</span>
            <span class="workflow-title">PLANNING INPUTS</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write(
        "Enter the multi-site production and distribution values "
        "used to estimate production at Site 1."
    )

    st.caption(
        "Default values represent one record from the external dataset. "
        "Its actual Site 1 production value is 168 units."
    )

    # ========================================================
    # DEMAND & INVENTORY
    # ========================================================

    with st.expander(
        "Demand & Inventory",
        expanded=True,
    ):

        col1, col2 = st.columns(2)

        with col1:
            ds1 = st.number_input(
                "Demand — Site 1 (DS1)",
                value=117.0,
                key="external_ds1",
            )

            ds2 = st.number_input(
                "Demand — Site 2 (DS2)",
                value=417.0,
                key="external_ds2",
            )

            ds3 = st.number_input(
                "Demand — Site 3 (DS3)",
                value=227.0,
                key="external_ds3",
            )

        with col2:
            i1 = st.number_input(
                "Inventory — Site 1 (I1)",
                value=168.0,
                key="external_i1",
            )

            i2 = st.number_input(
                "Inventory — Site 2 (I2)",
                value=136.0,
                key="external_i2",
            )

            i3 = st.number_input(
                "Inventory — Site 3 (I3)",
                value=277.0,
                key="external_i3",
            )

    # ========================================================
    # PRODUCTION CONSTRAINTS
    # ========================================================

    with st.expander(
        "Production Constraints"
    ):

        col1, col2 = st.columns(2)

        with col1:
            u_min = st.number_input(
                "Minimum Production Limit",
                value=0.0,
                key="external_u_min",
            )

        with col2:
            u_max = st.number_input(
                "Maximum Production Limit",
                value=300.0,
                key="external_u_max",
            )

    # ========================================================
    # COSTS & CAPACITY
    # ========================================================

    with st.expander(
        "Costs & Capacity"
    ):

        col1, col2 = st.columns(2)

        with col1:
            mc = st.number_input(
                "Manufacturing Cost (MC123)",
                value=1.0,
                key="external_mc",
            )

            shc = st.number_input(
                "Shortage Cost (SHC123)",
                value=20.0,
                key="external_shc",
            )

        with col2:
            sc = st.number_input(
                "Storage Cost (SC123)",
                value=10.0,
                key="external_sc",
            )

            vc = st.number_input(
                "Vehicle Capacity (VC)",
                value=30.0,
                key="external_vc",
            )

    # ========================================================
    # INTER-SITE DISTRIBUTION
    # ========================================================

    with st.expander(
        "Inter-site Distribution"
    ):

        col1, col2 = st.columns(2)

        with col1:
            t12 = st.number_input(
                "Site 1 → Site 2 (T12)",
                value=0.0,
                key="external_t12",
            )

            t21 = st.number_input(
                "Site 2 → Site 1 (T21)",
                value=0.0,
                key="external_t21",
            )

            t31 = st.number_input(
                "Site 3 → Site 1 (T31)",
                value=0.0,
                key="external_t31",
            )

        with col2:
            t13 = st.number_input(
                "Site 1 → Site 3 (T13)",
                value=0.0,
                key="external_t13",
            )

            t23 = st.number_input(
                "Site 2 → Site 3 (T23)",
                value=0.0,
                key="external_t23",
            )

            t32 = st.number_input(
                "Site 3 → Site 2 (T32)",
                value=0.0,
                key="external_t32",
            )

    # ========================================================
    # INPUT DATAFRAME
    # ========================================================

    input_data = pd.DataFrame(
        {
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
            "T32": [t32],
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
        key="external_predict_button",
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
            "Predicted Site 1 Production",
            f"{prediction:,} units",
        )

        actual_example = 168

        difference = abs(
            prediction - actual_example
        )

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Example Actual",
                f"{actual_example} units",
            )

        with col2:
            st.metric(
                "Difference",
                f"{difference} units",
            )

        st.caption(
            f"Prediction generated using {best_model_row['Model']}. "
            "The actual-value comparison applies only to the default example record."
        )