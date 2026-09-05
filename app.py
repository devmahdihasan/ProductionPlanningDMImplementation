import streamlit as st

from components.styles import inject_global_styles
from components.header import render_header
from components.footer import render_footer
from components.dataset_selector import render_dataset_selector
from components.custom_csv import render_custom_csv_mode
from components.synthetic_mode import render_synthetic_mode
from components.external_mode import render_external_mode


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Production Planning",
    page_icon="🏭",
    layout="centered",
)


# ============================================================
# GLOBAL STYLES
# ============================================================

inject_global_styles()


# ============================================================
# HEADER
# ============================================================

render_header()


# ============================================================
# DATASET SELECTION
# ============================================================

dataset_option = render_dataset_selector()


# ============================================================
# APPLICATION MODES
# ============================================================

if dataset_option == "Synthetic Dataset":
    render_synthetic_mode()

elif dataset_option == "External Production Dataset":
    render_external_mode()

elif dataset_option == "Upload Custom CSV":
    render_custom_csv_mode()


# ============================================================
# FOOTER
# ============================================================

render_footer()