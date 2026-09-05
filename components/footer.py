import streamlit as st


def render_footer():
    with st.container(key="app_footer"):
        st.divider()

        top_left, top_right = st.columns(
            [3, 1],
            vertical_alignment="center",
        )

        with top_left:
            st.caption("PRODUCTION PLANNING")

        with top_right:
            st.caption("REGRESSION / v1.0")

        bottom_left, bottom_right = st.columns(
            [2, 3],
            vertical_alignment="center",
        )

        with bottom_left:
            st.caption("Data Mining Lab Project")

        with bottom_right:
            st.caption(
                "Built by devmahdihasan · NextGrid Digital"
            )