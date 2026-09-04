import io

import pandas as pd
import streamlit as st

from ml.training import (
    convert_numeric_like_columns,
    train_regression_models,
)


# ============================================================
# CSV READER
# ============================================================

def read_uploaded_csv(uploaded_file):
    file_bytes = uploaded_file.getvalue()

    formats = [
        {
            "sep": ",",
            "decimal": ".",
        },
        {
            "sep": ";",
            "decimal": ",",
        },
        {
            "sep": ";",
            "decimal": ".",
        },
        {
            "sep": "\t",
            "decimal": ".",
        },
        {
            "sep": "|",
            "decimal": ".",
        },
    ]

    successful_reads = []

    for csv_format in formats:

        try:

            data = pd.read_csv(
                io.BytesIO(file_bytes),
                sep=csv_format["sep"],
                decimal=csv_format["decimal"],
            )

            successful_reads.append(
                (
                    len(data.columns),
                    data,
                )
            )

        except Exception:
            continue


    if not successful_reads:

        raise ValueError(
            "The CSV format could not be detected."
        )


    successful_reads.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    return successful_reads[0][1]


# ============================================================
# DATA QUALITY ANALYSIS
# ============================================================

def analyze_feature_quality(
    data,
    features,
    target,
):
    warnings = []

    if not features:
        return warnings


    working_data = convert_numeric_like_columns(
        data[
            features + [target]
        ].copy()
    )


    if len(working_data) == 0:
        return warnings


    # ========================================================
    # TARGET QUALITY
    # ========================================================

    target_unique = (
        working_data[target]
        .dropna()
        .nunique()
    )


    if target_unique < 5:

        warnings.append(
            (
                f"Target `{target}` contains only "
                f"{target_unique} unique values. "
                "Regression may not be the most suitable "
                "technique for this target."
            )
        )


    # ========================================================
    # FEATURE QUALITY
    # ========================================================

    for feature in features:

        series = working_data[
            feature
        ]

        non_null = (
            series.dropna()
        )

        unique_count = (
            non_null.nunique()
        )

        missing_ratio = (
            series.isna().mean()
        )


        # ----------------------------------------------------
        # CONSTANT COLUMN
        # ----------------------------------------------------

        if unique_count <= 1:

            warnings.append(
                (
                    f"`{feature}` is constant or nearly empty "
                    "and will provide no useful predictive information."
                )
            )

            continue


        # ----------------------------------------------------
        # HIGH MISSING VALUES
        # ----------------------------------------------------

        if missing_ratio >= 0.40:

            warnings.append(
                (
                    f"`{feature}` contains "
                    f"{missing_ratio * 100:.1f}% missing values."
                )
            )


        # ----------------------------------------------------
        # ID-LIKE COLUMNS
        # ----------------------------------------------------

        feature_lower = (
            str(feature)
            .strip()
            .lower()
        )


        id_keywords = [
            "id",
            "index",
            "serial",
            "serial_no",
            "serial_number",
            "record_id",
            "row_id",
            "uuid",
        ]


        unique_ratio = (
            unique_count
            / max(
                len(non_null),
                1,
            )
        )


        looks_like_id_name = (
            feature_lower
            in id_keywords
            or feature_lower.endswith("_id")
            or feature_lower.startswith("id_")
        )


        if (
            looks_like_id_name
            and unique_ratio >= 0.80
        ):

            warnings.append(
                (
                    f"`{feature}` looks like an identifier. "
                    "Identifier columns usually should not be used "
                    "for prediction."
                )
            )


        # ----------------------------------------------------
        # HIGH-CARDINALITY CATEGORICAL
        # ----------------------------------------------------

        if not pd.api.types.is_numeric_dtype(
            series
        ):

            if (
                unique_count >= 20
                and unique_ratio >= 0.50
            ):

                warnings.append(
                    (
                        f"`{feature}` is a high-cardinality "
                        f"categorical feature with "
                        f"{unique_count} unique values. "
                        "This may create many encoded columns."
                    )
                )


    # ========================================================
    # TARGET LEAKAGE CHECK
    # ========================================================

    numeric_target = pd.to_numeric(
        working_data[target],
        errors="coerce",
    )


    for feature in features:

        numeric_feature = pd.to_numeric(
            working_data[
                feature
            ],
            errors="coerce",
        )


        valid_mask = (
            numeric_feature.notna()
            & numeric_target.notna()
        )


        if valid_mask.sum() < 10:
            continue


        if (
            numeric_feature[
                valid_mask
            ].nunique()
            < 2
        ):
            continue


        correlation = (
            numeric_feature[
                valid_mask
            ]
            .corr(
                numeric_target[
                    valid_mask
                ]
            )
        )


        if pd.isna(
            correlation
        ):
            continue


        if abs(correlation) >= 0.98:

            warnings.append(
                (
                    f"`{feature}` has an extremely strong "
                    f"correlation with the target "
                    f"({correlation:.3f}). "
                    "Check whether this feature may reveal "
                    "the target directly."
                )
            )


    return warnings


# ============================================================
# INITIALIZE SESSION STATE
# ============================================================

def initialize_custom_session_state():

    if "custom_training_result" not in st.session_state:
        st.session_state.custom_training_result = None

    if "custom_dataset_name" not in st.session_state:
        st.session_state.custom_dataset_name = None

    if "custom_features" not in st.session_state:
        st.session_state.custom_features = []

    if "custom_target" not in st.session_state:
        st.session_state.custom_target = None

    if "custom_training_signature" not in st.session_state:
        st.session_state.custom_training_signature = None


# ============================================================
# CUSTOM CSV MODE
# ============================================================

def render_custom_csv_mode():

    initialize_custom_session_state()


    st.subheader(
        "📤 Upload Your Dataset"
    )

    st.write(
        "Upload a CSV dataset, choose the target column, "
        "select the input features, and train regression models "
        "directly inside the application."
    )


    uploaded_file = st.file_uploader(
        "Upload CSV File",
        type=["csv"],
    )


    if uploaded_file is None:

        st.info(
            "Upload a CSV file to begin."
        )

        return


    # ========================================================
    # LOAD CSV
    # ========================================================

    try:

        custom_data = (
            read_uploaded_csv(
                uploaded_file
            )
        )

    except Exception as error:

        st.error(
            "The CSV file could not be read."
        )

        st.write(
            "The file may contain an unsupported format "
            "or malformed rows."
        )

        st.exception(
            error
        )

        return


    # ========================================================
    # CLEAN DATA
    # ========================================================

    custom_data = (
        custom_data.dropna(
            how="all"
        )
    )


    custom_data.columns = [
        str(column).strip()
        for column
        in custom_data.columns
    ]


    # ========================================================
    # RESET SESSION FOR NEW FILE
    # ========================================================

    if (
        st.session_state.custom_dataset_name
        != uploaded_file.name
    ):

        st.session_state.custom_training_result = None

        st.session_state.custom_training_signature = None

        st.session_state.custom_dataset_name = (
            uploaded_file.name
        )

        st.session_state.custom_features = []

        st.session_state.custom_target = None


    # ========================================================
    # DATASET OVERVIEW
    # ========================================================

    st.subheader(
        "📁 Dataset Overview"
    )


    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(
            "Rows",
            len(custom_data),
        )


    with col2:

        st.metric(
            "Columns",
            len(
                custom_data.columns
            ),
        )


    with col3:

        missing_values = int(
            custom_data
            .isna()
            .sum()
            .sum()
        )

        st.metric(
            "Missing Values",
            missing_values,
        )


    st.caption(
        f"File: {uploaded_file.name}"
    )


    # ========================================================
    # DATASET PREVIEW
    # ========================================================

    with st.expander(
        "🔍 Preview Dataset",
        expanded=True,
    ):

        st.dataframe(
            custom_data.head(
                20
            ),
            use_container_width=True,
        )


    # ========================================================
    # COLUMN INFORMATION
    # ========================================================

    with st.expander(
        "📋 Column Information"
    ):

        column_info = pd.DataFrame(
            {
                "Column":
                    custom_data.columns,

                "Data Type": [
                    str(
                        custom_data[
                            column
                        ].dtype
                    )
                    for column
                    in custom_data.columns
                ],

                "Missing Values": [
                    int(
                        custom_data[
                            column
                        ]
                        .isna()
                        .sum()
                    )
                    for column
                    in custom_data.columns
                ],

                "Unique Values": [
                    int(
                        custom_data[
                            column
                        ]
                        .nunique(
                            dropna=True
                        )
                    )
                    for column
                    in custom_data.columns
                ],
            }
        )


        st.dataframe(
            column_info,
            use_container_width=True,
            hide_index=True,
        )


    # ========================================================
    # VALIDATION
    # ========================================================

    if len(
        custom_data.columns
    ) < 2:

        st.error(
            "The dataset must contain at least two columns."
        )

        return


    if len(
        custom_data
    ) < 10:

        st.error(
            "The dataset must contain at least 10 usable rows."
        )

        return


    # ========================================================
    # TARGET SELECTION
    # ========================================================

    st.divider()

    st.subheader(
        "🎯 Select Prediction Target"
    )

    st.write(
        "Choose the numeric column that you want "
        "the regression models to predict."
    )


    numeric_candidate_data = (
        convert_numeric_like_columns(
            custom_data
        )
    )


    numeric_target_candidates = [
        column
        for column
        in numeric_candidate_data.columns
        if pd.api.types.is_numeric_dtype(
            numeric_candidate_data[
                column
            ]
        )
    ]


    if not numeric_target_candidates:

        st.error(
            "No usable numeric target columns were found. "
            "Regression requires a numeric prediction target."
        )

        return


    target = st.selectbox(
        "Target Column",
        numeric_target_candidates,
        index=(
            len(
                numeric_target_candidates
            ) - 1
        ),
    )


    # ========================================================
    # FEATURE SELECTION
    # ========================================================

    available_features = [
        column
        for column
        in custom_data.columns
        if column != target
    ]


    st.subheader(
        "🧩 Select Input Features"
    )

    st.write(
        "Choose the columns that should be used "
        "as model inputs."
    )


    selected_features = (
        st.multiselect(
            "Feature Columns",
            available_features,
            default=available_features,
        )
    )


    if not selected_features:

        st.warning(
            "Select at least one feature column."
        )


    # ========================================================
    # TRAINING SIGNATURE
    # ========================================================

    current_training_signature = (
        uploaded_file.name,
        target,
        tuple(
            sorted(
                selected_features
            )
        ),
    )


    # ========================================================
    # DATA PREPARATION INFO
    # ========================================================

    if selected_features:

        with st.expander(
            "🧹 Data Preparation Information"
        ):

            preview_data = (
                convert_numeric_like_columns(
                    custom_data[
                        selected_features
                        + [target]
                    ]
                )
            )


            numeric_features = [
                feature
                for feature
                in selected_features
                if pd.api.types.is_numeric_dtype(
                    preview_data[
                        feature
                    ]
                )
            ]


            categorical_features = [
                feature
                for feature
                in selected_features
                if feature
                not in numeric_features
            ]


            st.write(
                f"**Numeric features:** "
                f"{len(numeric_features)}"
            )


            if numeric_features:

                st.write(
                    ", ".join(
                        numeric_features
                    )
                )


            st.write(
                f"**Categorical features:** "
                f"{len(categorical_features)}"
            )


            if categorical_features:

                st.write(
                    ", ".join(
                        categorical_features
                    )
                )


            st.caption(
                "Missing numeric values are filled using the median. "
                "Missing categorical values are filled using the most "
                "frequent value. Categorical columns are automatically "
                "encoded before model training."
            )


    # ========================================================
    # DATA QUALITY WARNINGS
    # ========================================================

    if selected_features:

        quality_warnings = (
            analyze_feature_quality(
                data=custom_data,
                features=selected_features,
                target=target,
            )
        )


        if quality_warnings:

            with st.expander(
                f"⚠️ Data Quality Warnings "
                f"({len(quality_warnings)})",
                expanded=True,
            ):

                st.caption(
                    "These warnings do not automatically remove "
                    "any columns. Review them before training."
                )


                for warning in (
                    quality_warnings
                ):

                    st.warning(
                        warning
                    )


        else:

            st.success(
                "✅ No obvious data-quality problems were detected "
                "for the currently selected features."
            )


    # ========================================================
    # TRAIN MODELS
    # ========================================================

    st.divider()


    if st.button(
        "🚀 Train Regression Models",
        use_container_width=True,
        type="primary",
        disabled=not selected_features,
        key="custom_train_models",
    ):

        with st.spinner(
            "Training Linear Regression, "
            "Decision Tree Regression, "
            "and Random Forest Regression..."
        ):

            try:

                training_result = (
                    train_regression_models(
                        data=custom_data,
                        features=selected_features,
                        target=target,
                        test_size=0.20,
                        random_state=42,
                    )
                )


                st.session_state.custom_training_result = (
                    training_result
                )


                st.session_state.custom_features = (
                    selected_features
                )


                st.session_state.custom_target = (
                    target
                )


                st.session_state.custom_training_signature = (
                    current_training_signature
                )


            except Exception as error:

                st.session_state.custom_training_result = None

                st.session_state.custom_training_signature = None


                st.error(
                    "Model training failed."
                )


                st.exception(
                    error
                )


    # ========================================================
    # LOAD STORED RESULT
    # ========================================================

    training_result = (
        st.session_state.custom_training_result
    )


    stored_signature = (
        st.session_state.custom_training_signature
    )


    # ========================================================
    # CONFIGURATION CHANGED
    # ========================================================

    if (
        training_result is not None
        and stored_signature
        != current_training_signature
    ):

        training_result = None


        st.warning(
            "The target column or selected features have changed. "
            "Train the regression models again to update the results."
        )


    # ========================================================
    # TRAINING RESULTS
    # ========================================================

    if training_result is None:
        return


    st.divider()

    st.subheader(
        "📊 Model Performance"
    )


    col1, col2, col3 = (
        st.columns(3)
    )


    with col1:

        st.metric(
            "Usable Records",
            training_result[
                "records"
            ],
        )


    with col2:

        st.metric(
            "Training Records",
            training_result[
                "train_records"
            ],
        )


    with col3:

        st.metric(
            "Testing Records",
            training_result[
                "test_records"
            ],
        )


    custom_results = (
        training_result[
            "results"
        ].copy()
    )


    custom_results["MAE"] = (
        custom_results["MAE"]
        .round(2)
    )


    custom_results["RMSE"] = (
        custom_results["RMSE"]
        .round(2)
    )


    custom_results["R2"] = (
        custom_results["R2"]
        .round(4)
    )


    st.dataframe(
        custom_results,
        use_container_width=True,
        hide_index=True,
    )


    # ========================================================
    # BEST MODEL
    # ========================================================

    st.subheader(
        "🏆 Best Performing Model"
    )


    col1, col2, col3 = (
        st.columns(3)
    )


    with col1:

        st.metric(
            "Best Model",
            training_result[
                "best_model_name"
            ],
        )


    with col2:

        st.metric(
            "R² Score",
            f"{training_result['r2']:.4f}",
        )


    with col3:

        st.metric(
            "MAE",
            f"{training_result['mae']:.2f}",
        )


    st.info(
        f"**{training_result['best_model_name']}** "
        f"achieved the highest R² score "
        f"for this uploaded dataset."
    )


    # ========================================================
    # MODEL INTERPRETATION
    # ========================================================

    if training_result[
        "r2"
    ] < 0:

        st.warning(
            "The best model has a negative R² score. "
            "For this test split, the model performed worse "
            "than simply predicting the average target value."
        )


    elif training_result[
        "r2"
    ] < 0.30:

        st.warning(
            "The model explains only a small portion "
            "of the variation in the selected target."
        )


    elif training_result[
        "r2"
    ] >= 0.95:

        st.info(
            "The model achieved a very high R² score. "
            "Review the selected features for possible target "
            "leakage before assuming the result will generalize."
        )


    # ========================================================
    # PREDICTION
    # ========================================================

    st.divider()

    st.subheader(
        "🔮 Make a Prediction"
    )


    st.write(
        f"Enter input values to predict "
        f"**{training_result['target']}**."
    )


    trained_features = (
        training_result[
            "features"
        ]
    )


    trained_numeric_features = (
        training_result[
            "numeric_features"
        ]
    )


    trained_categorical_features = (
        training_result[
            "categorical_features"
        ]
    )


    prediction_values = {}


    # ========================================================
    # NUMERIC INPUTS
    # ========================================================

    if trained_numeric_features:

        with st.expander(
            "🔢 Numeric Inputs",
            expanded=True,
        ):

            for feature in (
                trained_numeric_features
            ):

                numeric_series = (
                    pd.to_numeric(
                        custom_data[
                            feature
                        ],
                        errors="coerce",
                    )
                )


                median_value = (
                    numeric_series.median()
                )


                if pd.isna(
                    median_value
                ):

                    median_value = 0.0


                prediction_values[
                    feature
                ] = st.number_input(
                    feature,
                    value=float(
                        median_value
                    ),
                    key=(
                        f"custom_numeric_"
                        f"{feature}"
                    ),
                )


    # ========================================================
    # CATEGORICAL INPUTS
    # ========================================================

    if trained_categorical_features:

        with st.expander(
            "🔤 Categorical Inputs",
            expanded=True,
        ):

            for feature in (
                trained_categorical_features
            ):

                options = (
                    custom_data[
                        feature
                    ]
                    .dropna()
                    .astype(str)
                    .unique()
                    .tolist()
                )


                options = sorted(
                    options
                )


                if not options:

                    options = [
                        "Unknown"
                    ]


                prediction_values[
                    feature
                ] = st.selectbox(
                    feature,
                    options,
                    key=(
                        f"custom_category_"
                        f"{feature}"
                    ),
                )


    # ========================================================
    # INPUT DATAFRAME
    # ========================================================

    custom_input_data = (
        pd.DataFrame(
            [
                {
                    feature:
                    prediction_values[
                        feature
                    ]
                    for feature
                    in trained_features
                }
            ]
        )
    )


    # ========================================================
    # PREDICT
    # ========================================================

    if st.button(
        "🎯 Predict Custom Target",
        use_container_width=True,
        key="custom_predict_button",
    ):

        try:

            custom_prediction = (
                training_result[
                    "best_model"
                ]
                .predict(
                    custom_input_data
                )[0]
            )


            st.success(
                f"🎯 **Predicted "
                f"{training_result['target']}: "
                f"{custom_prediction:,.2f}**"
            )


            st.info(
                "Prediction generated using "
                f"**{training_result['best_model_name']}**."
            )


        except Exception as error:

            st.error(
                "Prediction failed."
            )

            st.exception(
                error
            )