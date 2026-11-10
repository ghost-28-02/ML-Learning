import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from tabs import Data_Info, Feature_Engineering, Model_Output, Models, Visualize_Data


st.title("🔍 Machine Learning App")
tabs = st.tabs(["📊 Visualize Data", "📋 Data Info", "🛠 Feature Engineering", "📦 Models", "📈 Model Output"])

st.sidebar.markdown("##### 📂 Data Gathering")

# File uploader in the sidebar
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type="csv")

# Optional manual CSV path input
csv_path = st.sidebar.text_input("Or enter CSV file path manually")

# Only load if df not already in session_state
if "df" not in st.session_state:
    if uploaded_file is not None:
        try:
            st.session_state.df = pd.read_csv(uploaded_file)
            st.sidebar.success("✅ File uploaded successfully.")
        except Exception as e:
            st.sidebar.error(f"❌ Error reading file: {e}")
    elif csv_path:
        try:
            st.session_state.df = pd.read_csv(csv_path)
            st.sidebar.success("✅ File loaded from path successfully.")
        except Exception as e:
            st.sidebar.error(f"❌ Failed to read file: {e}")

# Button to reset everything
if st.sidebar.button("🔄 Reset and Upload New File"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()

# Proceed only if df is loaded
if "df" in st.session_state:
    df = st.session_state.df

    with tabs[0]:
        Visualize_Data.render(df)

    with tabs[1]:
        Data_Info.render(df)

    with tabs[2]:
        feature,target = Feature_Engineering.render(df)

    with tabs[3]:
        Models.render(feature,target)

    with tabs[4]:
        Model_Output.render()

else:
    st.warning("Please upload a CSV file to get started.")