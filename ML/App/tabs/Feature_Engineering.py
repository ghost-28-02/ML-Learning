import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import seaborn as sns
from sklearn.preprocessing import (
    MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler, PowerTransformer, OneHotEncoder, LabelEncoder, OrdinalEncoder, KBinsDiscretizer
)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold


def render(df):
    st.subheader("🛠 Feature Engineering")
    st.write("Encode categorical variables, scale features, and define your target.")

    # Always override with session_state.df to ensure persistence
    if "df" not in st.session_state:
        st.warning("Please upload a dataset first.")
        return

    df = st.session_state.df.copy()  # use copy to avoid direct overwrite warnings

    # === 🎯 Target Column Selection ===
    st.markdown("##### 🎯 Select Target Column")

    if "target_col" not in st.session_state:
        st.session_state.target_col = df.columns[-1]  # default to last column

    st.session_state.target_col = st.selectbox(
        "Select the target/output column",
        options=df.columns,
        index=df.columns.get_loc(st.session_state.target_col)
    )

    target_col = st.session_state.target_col

    # === 📥 Feature Column Selection ===
    st.markdown("##### 📥 Select Feature Columns")

    default_features = [col for col in df.columns if col != target_col]

    feature_cols = st.multiselect(
        "Select feature/input columns",
        options=default_features,
        default=default_features
    )

    st.markdown("##### 📏 Feature Scaling")

    # Select numerical columns from selected features
    num_cols = df[feature_cols].select_dtypes(include=['int64', 'float64']).columns.tolist()

    if num_cols:
        selected_num = st.multiselect(
            "Select numerical columns to scale",
            options=num_cols,
            key="scale_columns"
        )

        scaler_type = st.selectbox(
            "Choose scaling method",
            options=["Normalization", "Standardization"],
            key="scaler_type"
        )

        normalization_method = None
        if scaler_type == "Normalization":
            normalization_method = st.selectbox(
                "Choose normalization technique",
                options=["MinMax", "Mean Normalization", "MaxAbs", "Robust"],
                key="norm_method"
            )

        if st.button("Apply Scaling"):
            try:
                if scaler_type == "Standardization":
                    scaler = StandardScaler()
                    df[selected_num] = scaler.fit_transform(df[selected_num])

                elif normalization_method == "Mean Normalization":
                    df[selected_num] = (
                        (df[selected_num] - df[selected_num].mean()) /
                        (df[selected_num].max() - df[selected_num].min())
                    )
                    st.success("✅ Mean Normalization applied to selected columns.")
                else:
                    if normalization_method == "MinMax":
                        scaler = MinMaxScaler()
                    elif normalization_method == "MaxAbs":
                        scaler = MaxAbsScaler()
                    elif normalization_method == "Robust":
                        scaler = RobustScaler()
                    else:
                        raise ValueError("Unknown normalization method selected.")

                    df[selected_num] = scaler.fit_transform(df[selected_num])
                    st.success(f"✅ {scaler_type} ({normalization_method}) applied to selected columns.")

                # Persist changes
                st.session_state.df = df

            except Exception as e:
                st.error(f"❌ Error during scaling: {e}")
    else:
        st.info("ℹ️ No numerical columns found in the selected features.")

    st.markdown("#### Missing Value Imputation")
    use_auto = st.checkbox("Use Auto Imputation (via GridSearchCV)", value=True)

    if use_auto:
        st.markdown("##### 🔍 Auto Imputation via GridSearchCV")

        if st.button("Run Auto Imputation with GridSearchCV"):
            try:
                X = df[feature_cols].copy()
                y = df[target_col].copy()

                num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
                cat_cols = X.select_dtypes(include='object').columns.tolist()

                # Coerce invalid numbers to NaN
                for col in num_cols:
                    X[col] = pd.to_numeric(X[col], errors='coerce')

                for col in cat_cols:
                    X[col] = X[col].astype(str).fillna("Missing")

                # Handle target variable type
                if y.dtype == 'O':
                    try:
                        y = pd.to_numeric(y, errors='coerce')
                    except Exception:
                        y = y.astype(str)

                # Define preprocessing transformers
                num_transformer = Pipeline([
                    ('imputer', SimpleImputer()),  # strategy will be tuned
                    ('scaler', StandardScaler())
                ])
                cat_transformer = Pipeline([
                    ('imputer', SimpleImputer(strategy='most_frequent')),  # strategy will be tuned
                    ('encoder', OneHotEncoder(handle_unknown='ignore'))
                ])

                preprocessor = ColumnTransformer([
                    ('num', num_transformer, num_cols),
                    ('cat', cat_transformer, cat_cols)
                ])

                # Pipeline with classifier
                pipe = Pipeline([
                    ('preprocessor', preprocessor),
                    ('model', RandomForestClassifier(n_estimators=10, random_state=0))
                ])

                # Parameter grid for imputer strategies
                param_grid = {
                    'preprocessor__num__imputer__strategy': ['mean', 'median'],
                    'preprocessor__cat__imputer__strategy': ['most_frequent', 'constant']
                }

                # Handle stratified CV with small classes
                min_class_size = y.value_counts().min()
                cv_folds = min(3, min_class_size)
                if cv_folds < 2:
                    raise ValueError("Not enough data to perform cross-validation. Ensure at least 2 samples per class.")

                grid = GridSearchCV(pipe, param_grid=param_grid, cv=StratifiedKFold(n_splits=cv_folds))
                grid.fit(X, y)

                # Extract best strategies
                best_num_strategy = grid.best_params_['preprocessor__num__imputer__strategy']
                best_cat_strategy = grid.best_params_['preprocessor__cat__imputer__strategy']

                st.success(f"✅ Best strategies found:\n\n- Numerical: `{best_num_strategy}`\n- Categorical: `{best_cat_strategy}`")

                # 🔁 Apply best strategies to original DataFrame
                for col in num_cols:
                    if best_num_strategy == 'mean':
                        df[col].fillna(df[col].mean(), inplace=True)
                    elif best_num_strategy == 'median':
                        df[col].fillna(df[col].median(), inplace=True)

                for col in cat_cols:
                    if best_cat_strategy == 'most_frequent':
                        df[col].fillna(df[col].mode()[0], inplace=True)
                    elif best_cat_strategy == 'constant':
                        df[col].fillna("Missing", inplace=True)

                # ✅ Save back to session state only once
                st.session_state.df = df
                st.success("✅ Data imputed using best strategies from GridSearchCV.")

            except Exception as e:
                st.error(f"❌ Error during auto imputation:\n\n{e}")


    else:
        st.markdown("##### 🧩 Categorical Missing Value Imputation")

        # Get categorical columns with missing values
        cat_cols_with_na = [col for col in df.select_dtypes(include='object').columns if df[col].isnull().any()]

        if cat_cols_with_na:
            selected_cat = st.multiselect(
                "Select categorical columns with missing values to fill",
                options=cat_cols_with_na,
                key="cat_impute_cols"
            )

            cat_methods = {}
            for col in selected_cat:
                method = st.selectbox(
                    f"Imputation method for `{col}`:",
                    options=["Mode (Most Frequent)", "Constant Value (e.g. 'Unknown')", "New Category (e.g. 'Missing')", "Drop Column"],
                    key=f"cat_method_{col}"
                )
                cat_methods[col] = method

            if st.button("Run Categorical Imputation"):
                for col, method in cat_methods.items():
                    if method == "Mode (Most Frequent)":
                        df[col].fillna(df[col].mode()[0], inplace=True)
                    elif method == "Constant Value (e.g. 'Unknown')":
                        df[col].fillna("Unknown", inplace=True)
                    elif method == "New Category (e.g. 'Missing')":
                        df[col].fillna("Missing", inplace=True)
                    elif method == "Drop Column":
                        df.drop(columns=[col], inplace=True)

                st.session_state.df = df
                st.success("✅ Categorical imputation completed.")

        # Numerical Imputation
        df = st.session_state.df
        st.markdown("##### 🧩 Numerical Missing Value Imputation")

        num_cols_with_na = [col for col in df.select_dtypes(include=['int64', 'float64']).columns if df[col].isnull().any()]

        if num_cols_with_na:
            selected_num = st.multiselect(
                "Select numerical columns to fill",
                options=num_cols_with_na,
                key="num_impute_cols"
            )

            num_methods = {}
            for col in selected_num:
                method = st.selectbox(
                    f"Imputation method for `{col}`:",
                    options=["Mean/Median Imputation", "Random Sampling", "KNN Imputation", "Drop Column"],
                    key=f"num_method_{col}"
                )
                num_methods[col] = method

            if st.button("Run Numerical Imputation"):
                try:
                    knn_needed_cols = []

                    for col, method in num_methods.items():
                        if method == "Mean/Median Imputation":
                            if abs(df[col].skew()) > 1:
                                df[col].fillna(df[col].median(), inplace=True)
                            else:
                                df[col].fillna(df[col].mean(), inplace=True)
                        elif method == "Random Sampling":
                            non_na_vals = df[col].dropna().values
                            if len(non_na_vals) > 0:
                                random_fill = np.random.choice(non_na_vals, size=df[col].isnull().sum())
                                df.loc[df[col].isnull(), col] = random_fill
                        elif method == "Drop Column":
                            df.drop(columns=[col], inplace=True)
                        elif method == "KNN Imputation":
                            knn_needed_cols.append(col)

                    # Perform KNN imputation if needed
                    if knn_needed_cols:
                        imputer = KNNImputer(n_neighbors=5)
                        df[knn_needed_cols] = imputer.fit_transform(df[knn_needed_cols])
                        st.success("✅ KNN Imputer applied to selected columns.")

                    st.session_state.df = df
                    st.success("✅ Numerical imputation completed.")

                except Exception as e:
                    st.error(f"❌ Error during numerical imputation: {e}")
        else:
            st.info("✅ No missing values in numerical columns.")

        
    st.markdown("##### 🔤 Categorical Encoding (Feature Columns)")

    cat_feat_cols = df[feature_cols].select_dtypes(include='object').columns.tolist()

    if cat_feat_cols:
        selected_cat_feats = st.multiselect(
            "Select categorical feature columns to encode",
            options=cat_feat_cols,
            key="cat_feat_encode_cols"
        )

        encoding_methods = {}
        custom_orders = {}

        for col in selected_cat_feats:
            encoding_type = st.selectbox(
                f"Encoding for `{col}`",
                options=["One-hot", "Ordinal encoding"],
                key=f"feat_encoding_{col}"
            )
            encoding_methods[col] = encoding_type

            if encoding_type == "Ordinal encoding":
                unique_vals = df[col].dropna().unique().tolist()
                custom_order = st.multiselect(
                    f"Custom order for `{col}`",
                    options=sorted(unique_vals),
                    default=sorted(unique_vals),
                    key=f"custom_order_{col}"
                )
                custom_orders[col] = custom_order

        if st.button("Apply Encoding to Selected Columns"):
            try:
                for col in selected_cat_feats:
                    enc_type = encoding_methods[col]

                    if enc_type == "One-hot":
                        dummies = pd.get_dummies(df[col], prefix=col, drop_first=True).astype(int)  # <-- Cast to int here
                        if not dummies.empty:
                            df = df.drop(columns=[col])
                            df = pd.concat([df, dummies], axis=1)
                            st.success(f"✅ One-hot encoding applied to `{col}`.")
                        else:
                            st.warning(f"⚠️ No unique values found in `{col}` for one-hot encoding.")
                    elif enc_type == "Ordinal encoding":
                        order = custom_orders.get(col)
                        if not order or set(order) != set(df[col].dropna().unique()):
                            st.error(f"⚠️ Incomplete custom order for `{col}`.")
                            continue
                        encoder = OrdinalEncoder(categories=[order])
                        df[col] = df[col].fillna("Missing")
                        df[col] = encoder.fit_transform(df[[col]])
                        st.success(f"✅ Ordinal encoding applied to `{col}` with custom order.")
                st.session_state.df = df
            except Exception as e:
                st.error(f"❌ Error applying encodings: {e}")
    else:
        st.info("✅ No categorical feature columns found.")

    # === 🔠 Label Encoding for Target Column ===
    st.markdown("##### 🎯 Encode Target Column (Label Encoding)")

    if df[target_col].dtype == 'object':
        if st.button("Apply Label Encoding to Target Column"):
            try:
                le = LabelEncoder()
                df[target_col] = le.fit_transform(df[target_col])
                st.session_state.df = df
                st.success(f"✅ Label encoding applied to target column `{target_col}`.")
            except Exception as e:
                st.error(f"❌ Error encoding target column: {e}")
    else:
        st.info("Target column is already numeric.")

    # === 🔧 Transformation Options ===
    st.markdown("### 🔧 Transformation Options")

    transform_option = st.radio(
        "Choose transformation type:",
        ["Function Transformation", "Power Transformation"],
        horizontal=True,
        key="transformation_type"
    )

    num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

    if not num_cols:
        st.info("ℹ️ No numerical columns available for transformation.")
    else:
        selected_cols = st.multiselect(
            "Select columns to transform",
            options=num_cols,
            key="transform_cols"
        )

        if transform_option == "Function Transformation":
            func_type = st.selectbox(
                "Choose function transformation",
                ["Log", "Square Root", "Square", "Reciprocal"],
                key="func_type"
            )

            if st.button("Apply Function Transformation"):
                try:
                    for col in selected_cols:
                        if func_type == "Log":
                            if (df[col] <= 0).any():
                                st.warning(f"⚠️ Skipping `{col}` due to non-positive values (log not defined).")
                                continue
                            df[col] = np.log1p(df[col])
                        elif func_type == "Square Root":
                            if (df[col] < 0).any():
                                st.warning(f"⚠️ Skipping `{col}` due to negative values (sqrt not defined).")
                                continue
                            df[col] = np.sqrt(df[col])
                        elif func_type == "Square":
                            df[col] = np.power(df[col], 2)
                        elif func_type == "Reciprocal":
                            if (df[col] == 0).any():
                                st.warning(f"⚠️ Skipping `{col}` due to zeros (division by zero).")
                                continue
                            df[col] = 1 / (df[col] + 1e-10)
                    st.session_state.df = df
                    st.success(f"✅ {func_type} applied to: {', '.join(selected_cols)}")
                except Exception as e:
                    st.error(f"❌ Error: {e}")

        elif transform_option == "Power Transformation":
            power_method = st.selectbox(
                "Choose power method",
                ["Yeo-Johnson", "Box-Cox"],
                key="power_method"
            )

            if st.button("Apply Power Transformation"):
                try:
                    if power_method == "Box-Cox":
                        for col in selected_cols:
                            if (df[col] <= 0).any():
                                raise ValueError(f"Box-Cox cannot be applied to non-positive values in '{col}'.")

                    method = "yeo-johnson" if power_method == "Yeo-Johnson" else "box-cox"
                    pt = PowerTransformer(method=method, standardize=True)
                    df[selected_cols] = pt.fit_transform(df[selected_cols])
                    st.session_state.df = df
                    st.success(f"✅ {power_method} transformation applied to: {', '.join(selected_cols)}")
                except Exception as e:
                    st.error(f"❌ Error: {e}")
        

    st.markdown("### 🔢 Encode Numerical Data (Discretization)")

    # Identify numerical columns
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

    if num_cols:
        selected_cols = st.multiselect(
            "Select numerical columns to discretize",
            options=num_cols,
            key="kbins_cols"
        )

        n_bins = st.number_input(
            "Enter number of bins",
            min_value=2,
            max_value=100,
            value=4,
            step=1,
            key="discretizer_n_bins"
        )

        strategy = st.selectbox(
            "Choose binning strategy",
            ["uniform", "quantile", "kmeans"],
            key="kbins_strategy"
        )

        encoding = st.selectbox(
            "Choose output encoding",
            ["ordinal", "onehot", "onehot-dense"],
            key="kbins_encoding"
        )

        if st.button("Apply KBinsDiscretizer"):
            try:
                discretizer = KBinsDiscretizer(
                    n_bins=n_bins,
                    encode=encoding,
                    strategy=strategy
                )
                transformed = discretizer.fit_transform(df[selected_cols])

                if encoding == "ordinal":
                    for idx, col in enumerate(selected_cols):
                        df[f"{col}_{strategy}_bin"] = transformed[:, idx]
                else:
                    transformed_df = pd.DataFrame(
                        transformed.toarray() if hasattr(transformed, "toarray") else transformed,
                        columns=discretizer.get_feature_names_out(selected_cols),
                        index=df.index
                    )
                    df = pd.concat([df, transformed_df], axis=1)

                st.session_state.df = df
                st.success(f"✅ KBinsDiscretizer ({strategy}, {encoding}) applied to: {', '.join(selected_cols)}")
                st.dataframe(df.tail())  # Preview result
            except Exception as e:
                st.error(f"❌ Error during discretization: {e}")
    else:
        st.info("ℹ️ No numerical columns available for discretization.")
    

    st.markdown("### 🚨 Outlier Detection & Handling")

    # Select numeric columns
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

    if num_cols:
        selected_cols = st.multiselect(
            "Select numerical columns for outlier detection",
            options=num_cols,
            key="outlier_cols"
        )

        detection_method = st.radio(
            "Choose outlier detection method",
            ["IQR (Interquartile Range)", "Z-Score", "Percentile"],
            horizontal=True,
            key="outlier_detection_method"
        )

        # Percentile-specific inputs
        if detection_method == "Percentile":
            lower_percentile = st.slider(
                "Lower percentile threshold", 0.0, 10.0, 1.0, step=0.5, key="lower_percentile"
            )
            upper_percentile = st.slider(
                "Upper percentile threshold", 90.0, 100.0, 99.0, step=0.5, key="upper_percentile"
            )

        # Handling options vary slightly for percentile
        if detection_method == "Percentile":
            handling_method = st.radio(
                "Choose percentile handling method",
                ["Trim (Remove Rows)", "Cap (Clip at Thresholds)"],
                horizontal=True,
                key="percentile_handling"
            )
        else:
            handling_method = st.radio(
                "Choose outlier handling method",
                ["Remove Rows", "Cap at Threshold (Winsorize)"],
                horizontal=True,
                key="standard_handling"
            )

        if st.button("Detect and Handle Outliers"):
            try:
                original_shape = df.shape
                df_copy = df.copy()

                for col in selected_cols:
                    if detection_method == "IQR (Interquartile Range)":
                        Q1 = df_copy[col].quantile(0.25)
                        Q3 = df_copy[col].quantile(0.75)
                        IQR = Q3 - Q1
                        lower_bound = Q1 - 1.5 * IQR
                        upper_bound = Q3 + 1.5 * IQR

                    elif detection_method == "Z-Score":
                        mean = df_copy[col].mean()
                        std = df_copy[col].std()
                        lower_bound = mean - 3 * std
                        upper_bound = mean + 3 * std

                    elif detection_method == "Percentile":
                        lower_bound = np.percentile(df_copy[col], lower_percentile)
                        upper_bound = np.percentile(df_copy[col], upper_percentile)

                    # === Handling ===
                    if detection_method == "Percentile":
                        if handling_method == "Trim (Remove Rows)":
                            df_copy = df_copy[(df_copy[col] >= lower_bound) & (df_copy[col] <= upper_bound)]
                        else:  # Cap
                            df_copy[col] = np.clip(df_copy[col], lower_bound, upper_bound)
                    else:
                        if handling_method == "Remove Rows":
                            df_copy = df_copy[(df_copy[col] >= lower_bound) & (df_copy[col] <= upper_bound)]
                        else:  # Cap
                            df_copy[col] = np.clip(df_copy[col], lower_bound, upper_bound)

                st.session_state.df = df_copy
                st.success(f"✅ Outlier handling complete. Rows before: {original_shape[0]}, after: {df_copy.shape[0]}")
                # 📊 Boxplot Before & After Outlier Handling
                st.markdown("### 📦 Boxplot Comparison (Before vs After)")

                before_df = df[selected_cols].copy()
                after_df = df_copy[selected_cols].copy()

                for col in selected_cols:
                    st.markdown(f"#### `{col}`")

                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown("**Before Handling**")
                        fig, ax = plt.subplots()
                        ax.boxplot(before_df[col].dropna())
                        ax.set_title(f"{col} (Before)")
                        st.pyplot(fig)

                    with col2:
                        st.markdown("**After Handling**")
                        fig, ax = plt.subplots()
                        ax.boxplot(after_df[col].dropna())
                        ax.set_title(f"{col} (After)")
                        st.pyplot(fig)
                st.dataframe(df_copy.head())

            except Exception as e:
                st.error(f"❌ Error during outlier handling: {e}")
    else:
        st.info("ℹ️ No numerical columns available for outlier handling.")


    target = df[target_col]


    st.markdown("### 🧮 Principal Component Analysis (PCA)")

    # Use the updated dataframe from session state
    df = st.session_state.df
    df = df.drop(columns=target_col,axis=1)

    # Get numeric columns dynamically
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

    if not num_cols:
        st.info("ℹ️ No numerical columns available for PCA.")
    else:
        selected_pca_cols = st.multiselect(
            "Select numerical columns for PCA",
            options=num_cols,
            key="pca_cols"
        )

        if selected_pca_cols:
            max_components = min(len(selected_pca_cols), df.shape[0])
            n_components = st.slider(
                "Number of principal components",
                min_value=1,
                max_value=max_components,
                value=2,
                key="pca_n_components"
            )

            if st.button("Run PCA"):
                try:
                    # Standardize selected columns
                    X = df[selected_pca_cols]
                    X_scaled = (X - X.mean()) / X.std()

                    # Apply PCA
                    pca = PCA(n_components=n_components)
                    principal_components = pca.fit_transform(X_scaled)

                    # Create new DataFrame with PCA components only
                    pca_columns = [f'PCA_{i+1}' for i in range(n_components)]
                    df_pca = pd.DataFrame(principal_components, columns=pca_columns, index=df.index)

                    # Update session state
                    st.session_state.df = df_pca

                    # Explained variance plot
                    st.markdown("#### 📈 Explained Variance")
                    fig, ax = plt.subplots()
                    ax.plot(range(1, n_components+1), pca.explained_variance_ratio_, marker='o')
                    ax.set_title("Explained Variance by Components")
                    ax.set_xlabel("Principal Component")
                    ax.set_ylabel("Variance Ratio")
                    st.pyplot(fig)

                    st.success(f"✅ PCA completed. Dataset now contains only {n_components} principal components.")

                    # 2D Scatter if possible
                    if n_components >= 2:
                        st.markdown("#### 🧭 PCA 2D Scatter Plot")
                        fig, ax = plt.subplots()
                        sns.scatterplot(x=principal_components[:, 0], y=principal_components[:, 1], ax=ax)
                        ax.set_xlabel("PCA 1")
                        ax.set_ylabel("PCA 2")
                        ax.set_title("PCA Projection (2D)")
                        st.pyplot(fig)
                    df = st.session_state.df
                except Exception as e:
                    st.error(f"❌ PCA failed: {e}")

    return df,target
