import streamlit as st


def render_dataset_selector():
    st.subheader(
        "🗂️ Select Dataset Source"
    )

    return st.radio(
        "Choose the dataset used for prediction:",
        [
            "Synthetic Dataset",
            "External Production Dataset",
            "Upload Custom CSV",
        ],
        horizontal=True,
    )