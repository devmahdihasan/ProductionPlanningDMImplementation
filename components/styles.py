import streamlit as st


def inject_global_styles():
    st.markdown(
        """
        <style>

        /* =========================================================
           PAGE
        ========================================================= */

        .stApp {
            background: #f6f6f3;
        }

        .block-container {
            max-width: 1040px;
            padding-top: 4.5rem;
            padding-bottom: 3rem;
        }


        /* =========================================================
           TYPOGRAPHY
        ========================================================= */

        html,
        body,
        [class*="css"] {
            font-family:
                Inter,
                ui-sans-serif,
                system-ui,
                -apple-system,
                BlinkMacSystemFont,
                "Segoe UI",
                sans-serif;
        }

        h1,
        h2,
        h3 {
            letter-spacing: -0.03em;
        }


        /* =========================================================
           APP HEADER
        ========================================================= */

        .st-key-app_header {
            margin-bottom: 1.6rem;
        }

        .st-key-app_header
        div[data-testid="stCaptionContainer"] {
            margin-bottom: 0.8rem;

            color: #78716c;

            font-size: 0.72rem;
            font-weight: 800;
            letter-spacing: 0.09em;
        }

        .st-key-app_header
        div[data-testid="stColumn"]:last-child
        div[data-testid="stCaptionContainer"] {
            text-align: right;

            color: #a8a29e;
            font-weight: 600;
        }

        .st-key-app_header .hero-heading {
            max-width: 760px;

            margin-top: 1rem;
            margin-bottom: 1rem;

            color: #171717;

            font-size: clamp(
                2.6rem,
                4.5vw,
                3.7rem
            );

            line-height: 1.05;
            font-weight: 800;
            letter-spacing: -0.03em;
        }

        .st-key-app_header p {
            max-width: 720px;

            margin-top: 0;

            color: #57534e;

            font-size: 1rem;
            line-height: 1.7;
        }


        /* =========================================================
           DATASET SELECTOR
        ========================================================= */

        div[role="radiogroup"] {
            display: grid;
            grid-template-columns: repeat(3, 1fr);

            gap: 0.75rem;

            margin-top: 0.7rem;
            margin-bottom: 2rem;
        }

        div[role="radiogroup"] label {
            background: #ffffff;

            border: 1px solid #dedbd5;
            border-radius: 14px;

            padding: 0.95rem 1rem;

            transition:
                border-color 0.15s ease,
                box-shadow 0.15s ease,
                transform 0.15s ease;
        }

        div[role="radiogroup"] label:hover {
            border-color: #f97316;

            box-shadow:
                0 8px 20px
                rgba(0, 0, 0, 0.04);

            transform: translateY(-1px);
        }

        div[role="radiogroup"] label:has(input:checked) {
            border-color: #f97316;

            background: #fff7ed;

            box-shadow:
                0 0 0 2px
                rgba(249, 115, 22, 0.10);
        }

        div[role="radiogroup"] label p {
            font-weight: 700;
            color: #292524;
        }


        /* =========================================================
           WORKFLOW SECTIONS
        ========================================================= */

        .workflow-section {
            display: flex;
            align-items: center;

            gap: 0.65rem;

            margin-top: 2.2rem;
            margin-bottom: 1rem;
        }

        .workflow-number {
            color: #f97316;

            font-size: 0.72rem;
            font-weight: 800;
            letter-spacing: 0.08em;
        }

        .workflow-title {
            color: #292524;

            font-size: 0.82rem;
            font-weight: 800;
            letter-spacing: 0.08em;
        }


        /* =========================================================
           FEATURE REVIEW HEADING
        ========================================================= */

        .feature-review-heading {
            margin-top: 1.5rem;
            margin-bottom: 0.35rem;

            color: #9a3412;

            font-size: 0.76rem;
            font-weight: 800;
            letter-spacing: 0.08em;
        }


        /* =========================================================
           FEATURE REVIEW PANEL
        ========================================================= */

        .st-key-feature_review_panel {
            margin-top: 0.75rem;
            margin-bottom: 1rem;

            padding: 0.3rem 0.5rem;

            background: #fff7ed;

            border: 1px solid #fdba74 !important;
            border-radius: 14px !important;
        }

        .st-key-feature_review_panel
        div[data-testid="stCheckbox"] {
            padding: 0.15rem 0;
        }

        .st-key-feature_review_panel
        div[data-testid="stCheckbox"] label {
            font-weight: 700;
            color: #9a3412;
        }

        .st-key-feature_review_panel
        div[data-testid="stCaptionContainer"] {
            color: #78716c;
            line-height: 1.5;
        }


        /* =========================================================
           REVIEW CHECKBOX ACCENT
        ========================================================= */

        .st-key-feature_review_panel
        div[data-testid="stCheckbox"]
        input:checked + div {
            border-color: #f97316 !important;
            background-color: #f97316 !important;
        }


        /* =========================================================
           REMOVE SUGGESTED BUTTON
        ========================================================= */

        .st-key-custom_remove_suggested
        button {
            min-height: 40px;

            background: #ffffff !important;

            border: 1px solid #fdba74 !important;
            border-radius: 10px !important;

            color: #9a3412 !important;

            font-size: 0.82rem;
            font-weight: 700;
        }

        .st-key-custom_remove_suggested
        button:hover {
            background: #ffedd5 !important;

            border-color: #f97316 !important;

            color: #9a3412 !important;
        }


        /* =========================================================
           METRIC CARDS
        ========================================================= */

        div[data-testid="stMetric"] {
            background: #ffffff;

            border: 1px solid #e3e0da;
            border-radius: 14px;

            padding: 0.85rem 1rem;
        }


        /* =========================================================
           ALERTS
        ========================================================= */

        div[data-testid="stAlert"] {
            border-radius: 14px;
        }


        /* =========================================================
           DATAFRAMES
        ========================================================= */

        div[data-testid="stDataFrame"] {
            border-radius: 14px;
            overflow: hidden;
        }


        /* =========================================================
           EXPANDERS
        ========================================================= */

        div[data-testid="stExpander"] {
            border-radius: 12px;
            overflow: hidden;
        }


        /* =========================================================
           BUTTONS
        ========================================================= */

        .stButton > button {
            min-height: 46px;

            border-radius: 12px;

            font-weight: 700;

            transition:
                background 0.15s ease,
                border-color 0.15s ease,
                box-shadow 0.15s ease,
                transform 0.15s ease;
        }

        .stButton > button:hover {
            transform: translateY(-1px);
        }

        .stButton > button[kind="primary"] {
            background: #f97316 !important;
            border-color: #f97316 !important;

            color: #171717 !important;
        }

        .stButton > button[kind="primary"]:hover {
            background: #ea580c !important;
            border-color: #ea580c !important;

            color: #171717 !important;

            box-shadow:
                0 8px 20px
                rgba(249, 115, 22, 0.18);
        }

        .stButton > button[kind="primary"]:focus {
            border-color: #f97316 !important;

            box-shadow:
                0 0 0 3px
                rgba(249, 115, 22, 0.18);
        }


        /* =========================================================
           FORM INPUTS
        ========================================================= */

        .stTextInput input,
        .stNumberInput input,
        .stSelectbox div[data-baseweb="select"] > div,
        .stMultiSelect div[data-baseweb="select"] > div {
            border-radius: 10px;
        }


        /* =========================================================
           DIVIDERS
        ========================================================= */

        hr {
            border: 0;
            border-top: 1px solid #dedbd5;
        }


        /* =========================================================
           APP FOOTER
        ========================================================= */

        .st-key-app_footer {
            margin-top: 3rem;
            padding-bottom: 0.5rem;
        }

        .st-key-app_footer hr {
            margin-bottom: 1.3rem;
        }

        .st-key-app_footer
        div[data-testid="stCaptionContainer"] {
            color: #8b8680;

            font-size: 0.78rem;
            line-height: 1.5;
        }

        .st-key-app_footer
        div[data-testid="stHorizontalBlock"]:first-of-type
        div[data-testid="stCaptionContainer"] {
            color: #292524;

            font-size: 0.72rem;
            font-weight: 800;
            letter-spacing: 0.08em;
        }

        .st-key-app_footer
        div[data-testid="stHorizontalBlock"]:first-of-type
        div[data-testid="stColumn"]:last-child
        div[data-testid="stCaptionContainer"] {
            color: #f97316;
            text-align: right;
        }

        .st-key-app_footer
        div[data-testid="stHorizontalBlock"]:last-of-type
        div[data-testid="stColumn"]:last-child
        div[data-testid="stCaptionContainer"] {
            text-align: right;
        }


        /* =========================================================
           HIDE STREAMLIT FOOTER
        ========================================================= */

        footer {
            visibility: hidden;
        }


        /* =========================================================
           MOBILE
        ========================================================= */

        @media (max-width: 760px) {

            .block-container {
                padding-top: 3.5rem;
            }

            .st-key-app_header {
                margin-bottom: 1.25rem;
            }

            .st-key-app_header .hero-heading {
                margin-top: 0.8rem;

                font-size: 2.3rem;
                line-height: 1.05;
            }

            .st-key-app_header p {
                font-size: 0.92rem;
                line-height: 1.65;
            }

            div[role="radiogroup"] {
                grid-template-columns: 1fr;
            }

            .st-key-feature_review_panel
            div[data-testid="stHorizontalBlock"] {
                gap: 0.25rem;
            }


            /* Footer mobile layout */

            .st-key-app_footer {
                margin-top: 2.5rem;
            }

            .st-key-app_footer
            div[data-testid="stHorizontalBlock"] {
                gap: 0.25rem;
            }

            .st-key-app_footer
            div[data-testid="stCaptionContainer"] {
                font-size: 0.72rem;
            }
        }

        </style>
        """,
        unsafe_allow_html=True,
    )