import streamlit as st
import pandas as pd
import joblib


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Production Planning",
    page_icon="🏭",
    layout="centered"
)


# ============================================================
# LOAD MODEL AND RESULTS
# ============================================================

model = joblib.load("models/best_model.pkl")
results = pd.read_csv("models/model_results.csv")


# Find the best model
best_model_row = results.loc[results["R2"].idxmax()]


# ============================================================
# HEADER
# ============================================================

st.title("🏭 Production Planning System")

st.write(
    "Data Mining Based Production Prediction"
)

st.caption(
    "Predict production quantity using historical "
    "production-related data."
)

st.divider()


# ============================================================
# MODEL PERFORMANCE
# ============================================================

st.subheader("📊 Model Performance")

st.write(
    "Three regression models were trained and evaluated "
    "using the same production dataset."
)

display_results = results.copy()

display_results["MAE"] = display_results["MAE"].round(2)
display_results["RMSE"] = display_results["RMSE"].round(2)
display_results["R2"] = display_results["R2"].round(4)

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
    f"R² score and was selected for production prediction."
)


st.divider()


# ============================================================
# INPUT SECTION
# ============================================================

st.subheader("📝 Production Information")

st.write(
    "Enter the current production-planning information "
    "to predict the production quantity."
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


# ============================================================
# PREDICTION
# ============================================================

if st.button(
    "🔮 Predict Production",
    use_container_width=True
):

    # Create input DataFrame
    input_data = pd.DataFrame({
        "Demand": [demand],
        "Inventory": [inventory],
        "Workers": [workers],
        "Working_Hours": [working_hours],
        "Raw_Material": [raw_material]
    })

    # Make prediction
    prediction = model.predict(input_data)[0]

    # Prevent negative prediction
    prediction = max(0, prediction)

    # Round prediction
    prediction = round(prediction)


    # ========================================================
    # DISPLAY RESULT
    # ========================================================

    st.divider()

    st.subheader("🎯 Prediction Result")

    st.success(
        f"🏭 **Predicted Production: {prediction:,} units**"
    )

    st.info(
        f"The prediction was generated using the trained "
        f"**{best_model_row['Model']}** model."
    )

    st.caption(
        "This prediction is intended to support production "
        "planning decisions based on historical data."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Data Mining Lab Project • Production Planning"
)