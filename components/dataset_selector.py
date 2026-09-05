import streamlit as st


def render_dataset_selector():
    st.markdown("### Choose your data source")

    return st.radio(
        "Dataset source",
        [
            "Synthetic Dataset",
            "External Production Dataset",
            "Upload Custom CSV",
        ],
        horizontal=True,
        label_visibility="collapsed",
    )