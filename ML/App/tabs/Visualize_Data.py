import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def render(csv):
    st.subheader("📊 Visualize Your Data")

    if "df" in st.session_state:
        df = st.session_state.df

        plot_col = st.selectbox("Select a column to visualize", df.columns)

        plot_type = st.selectbox("Select plot type", [
            "Histogram",
            "Box Plot",
            "Bar Chart",
            "Line Chart",
            "Pie Chart (for categorical)",
            "Correlation Heatmap"
        ])

        if plot_type != "Correlation Heatmap":
            fig, ax = plt.subplots()
            if plot_type == "Histogram":
                sns.histplot(df[plot_col], kde=True, ax=ax)
            elif plot_type == "Box Plot":
                sns.boxplot(x=df[plot_col], ax=ax)
            elif plot_type == "Bar Chart":
                df[plot_col].value_counts().plot(kind='bar', ax=ax)
            elif plot_type == "Line Chart":
                df[plot_col].plot(kind='line', ax=ax)
            elif plot_type == "Pie Chart (for categorical)":
                df[plot_col].value_counts().plot.pie(autopct='%1.1f%%', ax=ax)
                ax.set_ylabel("")
            st.pyplot(fig)

        else:  # Correlation heatmap
            numeric_cols = df.select_dtypes(include=['int64', 'float64'])
            if numeric_cols.shape[1] > 1:
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.heatmap(numeric_cols.corr(), annot=True, cmap="coolwarm", ax=ax)
                st.pyplot(fig)
            else:
                st.warning("Need at least 2 numerical columns for a correlation heatmap.")
    else:
        st.info("📁 Please load a dataset first in the Data Info tab.")
