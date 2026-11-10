import streamlit as st
import pandas as pd
import numpy as np

def render(df=None):
    st.subheader("📋 Data Info")
    st.write("Explore your dataset: preview, structure, and missing values.")

    # Always use the latest df from session_state
    if "df" not in st.session_state:
        st.warning("No dataset loaded yet.")
        return

    df = st.session_state.df.copy()

    # Data preview
    st.markdown("###### Data Preview")
    mode = st.selectbox("Preview Mode", ("head", "sample", "tail"))

    with st.expander("Show Preview"):
        if mode == "head":
            st.dataframe(df.head())
        elif mode == "sample":
            st.dataframe(df.sample(5))
        elif mode == "tail":
            st.dataframe(df.tail())

    # DataFrame Info
    st.markdown("##### DataFrame Info")
    with st.expander("Show DataFrame Info"):
        info_df = pd.DataFrame({
            "Column": df.columns,
            "Data Type": df.dtypes.values,
            "Non-Nulls": df.notnull().sum().values,
            "Nulls": df.isnull().sum().values,
            "Unique Values": df.nunique().values
        })
        st.dataframe(info_df)

    # Missing Values Summary
    st.markdown("##### Null Values Summary")
    with st.expander("Show Null Summary"):
        null_count = df.isnull().sum()
        null_percent = (null_count / len(df)) * 100
        null_summary = pd.DataFrame({
            "Missing Values": null_count,
            "Missing %": null_percent.round(2)
        })
        null_summary = null_summary[null_summary["Missing Values"] > 0]

        if not null_summary.empty:
            st.dataframe(null_summary)
        else:
            st.success("🎉 No missing values found in the dataset.")

    # Drop Columns
    st.markdown("##### 🔻 Drop Columns")
    drop_mode = st.radio("Drop columns by:", ["Index", "Name"], horizontal=True)


    
    if drop_mode == "Index":
        drop_selection = st.multiselect(
            "Select columns to drop (by index)",
            options=list(range(df.shape[1])),
            format_func=lambda x: f"{x}: {df.columns[x]}",
            key="drop_by_index"
        )
        
        drop_selection = st.multiselect(
            "Select columns to drop (by name)",
            options=df.columns.tolist(),
            key="drop_by_name"
        )

    if st.button("Drop"):
        try:
            if drop_mode == "Index":
                dropped_columns = df.columns[drop_selection].tolist()
                df.drop(df.columns[drop_selection], axis=1, inplace=True)
            else:
                dropped_columns = drop_selection
                df.drop(columns=drop_selection, inplace=True)

            # Persist changes
            st.session_state.df = df
            st.success(f"✅ Dropped columns: {', '.join(dropped_columns)}")
        except Exception as e:
            st.error(f"❌ Error dropping columns: {e}")

    # Statistical Summary
    st.markdown("##### 📊 Advanced Preview")
    with st.expander("Show Statistical Summary (describe)"):
        desc_num = df.describe().T
        st.markdown("**Numeric Columns:**")
        st.dataframe(desc_num)

        # Only show categorical description if there are object columns
        if df.select_dtypes(include='object').columns.any():
            desc_cat = df.describe(include="object").T
            st.markdown("**Categorical Columns:**")
            st.dataframe(desc_cat)
        else:
            st.info("No categorical columns available for description.")