import io

import pandas as pd
import streamlit as st

from ml.training import (
    convert_numeric_like_columns,
    train_regression_models,
)


# ============================================================
# UI HELPERS
# ============================================================

def render_workflow_section(number, title):
    st.markdown(
        f"""
        <div class="workflow-section">
            <span class="workflow-number">{number} /</span>
            <span class="workflow-title">{title}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# CSV READER
# ============================================================

def read_uploaded_csv(uploaded_file):
    file_bytes = uploaded_file.getvalue()

    formats = [
        {"sep": ",", "decimal": "."},
        {"sep": ";", "decimal": ","},
        {"sep": ";", "decimal": "."},
        {"sep": "\t", "decimal": "."},
        {"sep": "|", "decimal": "."},
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
    suggested_removals = []
    removal_reasons = {}

    if not features:
        return warnings, suggested_removals, removal_reasons

    working_data = convert_numeric_like_columns(
        data[
            features + [target]
        ].copy()
    )

    if len(working_data) == 0:
        return warnings, suggested_removals, removal_reasons

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
        # CONSTANT / NEARLY EMPTY
        # ----------------------------------------------------

        if unique_count <= 1:
            message = (
                f"`{feature}` is constant or nearly empty "
                "and will provide no useful predictive information."
            )

            warnings.append(
                message
            )

            if feature not in suggested_removals:
                suggested_removals.append(
                    feature
                )

            removal_reasons[
                feature
            ] = (
                "Constant or nearly empty — "
                "adds no useful predictive information."
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
            message = (
                f"`{feature}` looks like an identifier. "
                "Identifier columns usually should not be used "
                "for prediction."
            )

            warnings.append(
                message
            )

            if feature not in suggested_removals:
                suggested_removals.append(
                    feature
                )

            removal_reasons[
                feature
            ] = (
                "Identifier-like column — "
                "usually not useful for prediction."
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

    return warnings, suggested_removals, removal_reasons


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

    if "custom_config_target" not in st.session_state:
        st.session_state.custom_config_target = None


# ============================================================
# CUSTOM CSV MODE
# ============================================================

def render_custom_csv_mode():
    initialize_custom_session_state()

    # ========================================================
    # 01 / UPLOAD
    # ========================================================

    render_workflow_section(
        "01",
        "UPLOAD",
    )

    st.write(
        "Upload a CSV dataset to train regression models "
        "directly inside the application."
    )

    uploaded_file = st.file_uploader(
        "CSV dataset",
        type=["csv"],
        key="custom_csv_upload",
    )

    if uploaded_file is None:
        st.caption(
            "Supported formats include comma, semicolon, "
            "tab, and pipe-separated CSV files."
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

        st.caption(
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
        st.session_state.custom_dataset_name = uploaded_file.name
        st.session_state.custom_features = []
        st.session_state.custom_target = None
        st.session_state.custom_config_target = None

        st.session_state.pop(
            "custom_normal_features",
            None,
        )

        st.session_state.pop(
            "custom_review_keep_state",
            None,
        )

    # ========================================================
    # DATASET OVERVIEW
    # ========================================================

    missing_values = int(
        custom_data
        .isna()
        .sum()
        .sum()
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
        st.metric(
            "Missing Values",
            missing_values,
        )

    st.caption(
        f"File: {uploaded_file.name}"
    )

    with st.expander(
        "Preview dataset",
        expanded=False,
    ):
        st.dataframe(
            custom_data.head(
                20
            ),
            use_container_width=True,
        )

    with st.expander(
        "Column information"
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
    # BASIC VALIDATION
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

    st.divider()

    # ========================================================
    # 02 / CONFIGURE
    # ========================================================

    render_workflow_section(
        "02",
        "CONFIGURE",
    )

    st.write(
        "Choose the prediction target and the input features "
        "used by the regression models."
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
        "Prediction Target",
        numeric_target_candidates,
        index=(
            len(
                numeric_target_candidates
            ) - 1
        ),
        key="custom_target_select",
    )

    available_features = [
        column
        for column
        in custom_data.columns
        if column != target
    ]

    # ========================================================
    # RESET CONFIG WHEN TARGET CHANGES
    # ========================================================

    if (
        st.session_state.custom_config_target
        != target
    ):
        st.session_state.custom_config_target = target

        st.session_state.pop(
            "custom_normal_features",
            None,
        )

        st.session_state.pop(
            "custom_review_keep_state",
            None,
        )

    # ========================================================
    # PRE-ANALYZE FEATURES
    # ========================================================

    (
        _,
        suggested_removals,
        removal_reasons,
    ) = analyze_feature_quality(
        data=custom_data,
        features=available_features,
        target=target,
    )

    suggested_removals = [
        feature
        for feature
        in suggested_removals
        if feature in available_features
    ]

    normal_candidates = [
        feature
        for feature
        in available_features
        if feature not in suggested_removals
    ]

    # ========================================================
    # NORMAL FEATURES
    # ========================================================

    if "custom_normal_features" not in st.session_state:
        st.session_state.custom_normal_features = (
            normal_candidates.copy()
        )

    selected_normal_features = st.multiselect(
        "Input Features",
        normal_candidates,
        key="custom_normal_features",
    )

    # ========================================================
    # REVIEW FEATURE STATE
    # ========================================================

    review_state = (
        st.session_state.get(
            "custom_review_keep_state",
            {},
        )
    )

    for feature in suggested_removals:
        if feature not in review_state:
            review_state[
                feature
            ] = True

    st.session_state.custom_review_keep_state = (
        review_state
    )

    # ========================================================
    # REVIEW PANEL
    # ========================================================

    selected_review_features = []

    if suggested_removals:
        st.markdown(
            """
            <div class="feature-review-heading">
                REVIEW / POTENTIALLY REMOVABLE FEATURES
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.caption(
            "These features were automatically flagged as "
            "clear candidates for removal. Keep any feature "
            "only if you have a specific reason to use it."
        )

        if st.button(
            "Remove all suggested features",
            key="custom_remove_suggested",
            use_container_width=False,
        ):
            for feature in suggested_removals:
                st.session_state.custom_review_keep_state[
                    feature
                ] = False

                st.session_state[
                    f"review_keep_{feature}"
                ] = False

            st.rerun()

        with st.container(
            border=True,
            key="feature_review_panel",
        ):
            for feature in suggested_removals:
                feature_col, reason_col = st.columns(
                    [1.6, 3.4],
                    vertical_alignment="center",
                )

                with feature_col:
                    keep_feature = st.checkbox(
                        feature,
                        value=st.session_state.custom_review_keep_state[
                            feature
                        ],
                        key=f"review_keep_{feature}",
                    )

                with reason_col:
                    st.caption(
                        removal_reasons.get(
                            feature,
                            "Review this feature before training.",
                        )
                    )

                st.session_state.custom_review_keep_state[
                    feature
                ] = keep_feature

                if keep_feature:
                    selected_review_features.append(
                        feature
                    )

    # ========================================================
    # COMBINED FEATURES
    # ========================================================

    selected_features = (
        selected_normal_features
        + selected_review_features
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
    # FEATURE INFORMATION
    # ========================================================

    numeric_features = []
    categorical_features = []

    if selected_features:
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

        with st.expander(
            "Data preparation details"
        ):
            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Numeric Features",
                    len(numeric_features),
                )

            with col2:
                st.metric(
                    "Categorical Features",
                    len(categorical_features),
                )

            if numeric_features:
                st.write(
                    "**Numeric:** "
                    + ", ".join(
                        numeric_features
                    )
                )

            if categorical_features:
                st.write(
                    "**Categorical:** "
                    + ", ".join(
                        categorical_features
                    )
                )

            st.caption(
                "Missing numeric values use median imputation. "
                "Missing categorical values use the most frequent value. "
                "Categorical features are automatically encoded."
            )

    st.divider()

    # ========================================================
    # 03 / VALIDATE
    # ========================================================

    render_workflow_section(
        "03",
        "VALIDATE",
    )

    st.write(
        "Review the selected data for common issues before training."
    )

    quality_warnings = []

    if selected_features:
        (
            quality_warnings,
            _,
            _,
        ) = analyze_feature_quality(
            data=custom_data,
            features=selected_features,
            target=target,
        )

    if quality_warnings:
        with st.expander(
            f"Data quality warnings ({len(quality_warnings)})",
            expanded=True,
        ):
            st.caption(
                "Warnings are advisory only. "
                "No columns are automatically removed."
            )

            for warning in quality_warnings:
                st.warning(
                    warning
                )

    elif selected_features:
        st.success(
            "No obvious data-quality problems were detected "
            "for the selected configuration."
        )

    else:
        st.caption(
            "Select at least one feature to run validation."
        )

    st.divider()

    # ========================================================
    # 04 / TRAIN MODELS
    # ========================================================

    render_workflow_section(
        "04",
        "TRAIN MODELS",
    )

    st.write(
        "Train Linear Regression, Decision Tree Regression, "
        "and Random Forest Regression using the selected configuration."
    )

    if st.button(
        "Train regression models",
        use_container_width=True,
        type="primary",
        disabled=not selected_features,
        key="custom_train_models",
    ):
        with st.spinner(
            "Training regression models..."
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

    if training_result is None:
        return

    # ========================================================
    # TRAINING SUMMARY
    # ========================================================

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Usable Records",
            training_result["records"],
        )

    with col2:
        st.metric(
            "Training Records",
            training_result["train_records"],
        )

    with col3:
        st.metric(
            "Testing Records",
            training_result["test_records"],
        )

    custom_results = (
        training_result["results"].copy()
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
            training_result["best_model_name"],
        )

    with col2:
        st.metric(
            "R²",
            f"{training_result['r2']:.4f}",
        )

    with col3:
        st.metric(
            "MAE",
            f"{training_result['mae']:.2f}",
        )

    st.caption(
        "The model with the highest R² score is selected "
        "for prediction."
    )

    # ========================================================
    # MODEL INTERPRETATION
    # ========================================================

    if training_result["r2"] < 0:
        st.warning(
            "The best model has a negative R² score. "
            "For this test split, the model performed worse "
            "than predicting the average target value."
        )

    elif training_result["r2"] < 0.30:
        st.warning(
            "The model explains only a small portion "
            "of the variation in the selected target."
        )

    elif training_result["r2"] >= 0.95:
        st.info(
            "The model achieved a very high R² score. "
            "Review the selected features for possible target leakage "
            "before assuming the result will generalize."
        )

    st.divider()

    # ========================================================
    # 05 / PREDICT
    # ========================================================

    render_workflow_section(
        "05",
        "PREDICT",
    )

    st.write(
        f"Enter new feature values to predict "
        f"**{training_result['target']}**."
    )

    trained_features = (
        training_result["features"]
    )

    trained_numeric_features = (
        training_result["numeric_features"]
    )

    trained_categorical_features = (
        training_result["categorical_features"]
    )

    prediction_values = {}

    # ========================================================
    # NUMERIC INPUTS
    # ========================================================

    if trained_numeric_features:
        with st.expander(
            "Numeric Inputs",
            expanded=True,
        ):
            for feature in trained_numeric_features:
                numeric_series = (
                    pd.to_numeric(
                        custom_data[feature],
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
            "Categorical Inputs",
            expanded=True,
        ):
            for feature in trained_categorical_features:
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
    # PREDICTION
    # ========================================================

    if st.button(
        "Generate prediction",
        use_container_width=True,
        type="primary",
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

            st.metric(
                f"Predicted {training_result['target']}",
                f"{custom_prediction:,.2f}",
            )

            st.caption(
                f"Prediction generated using "
                f"{training_result['best_model_name']}."
            )

        except Exception as error:
            st.error(
                "Prediction failed."
            )

            st.exception(
                error
            )