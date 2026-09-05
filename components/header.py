import streamlit as st


def render_header():
    with st.container(key="app_header"):
        top_left, top_right = st.columns(
            [5, 1],
            vertical_alignment="center",
        )

        with top_left:
            st.caption(
                "PRODUCTION PLANNING / REGRESSION TOOL"
            )

        with top_right:
            st.caption("v1.0")

        st.markdown(
            """
            <div class="hero-heading">
                <div>Production decisions</div>
                <div>backed by data</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.write(
            "Compare regression models and generate production predictions "
            "from synthetic, external, or uploaded datasets"
        )