import streamlit as st


def render_header():
    st.title(
        "🏭 Production Planning System"
    )

    st.write(
        "Data Mining Based Production Prediction"
    )

    st.caption(
        "Compare regression models and predict production quantity "
        "using synthetic, externally sourced, or custom CSV data."
    )

    st.divider()