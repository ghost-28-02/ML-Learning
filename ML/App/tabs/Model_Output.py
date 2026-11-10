import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def render():
    st.subheader("📈 Model Output")
    st.write("View predictions and model evaluation metrics.")

    # Check if model has been trained
    if "model" not in st.session_state or st.session_state.model is None:
        st.warning("⚠️ No model has been trained yet. Please train a model in the 'Models' tab first.")
        return

    model = st.session_state.model
    feature = st.session_state.get("feature")
    target = st.session_state.get("target")

    if feature is None or feature.empty or target is None:
        st.warning("⚠️ Please provide valid feature columns and a target array.")
        return

    st.markdown("---")

    # === Model Predictions ===
    st.markdown("##### 🔮 Model Predictions")
    
    try:
        predictions = model.predict(feature)
        
        # Create a DataFrame with actual vs predicted values
        results_df = pd.DataFrame({
            "Actual": target,
            "Predicted": predictions,
            "Residual": target - predictions,
            "Absolute Error": np.abs(target - predictions)
        })

        # Display results table
        with st.expander("Show Predictions Table"):
            st.dataframe(results_df, use_container_width=True)

        # === Model Performance Metrics ===
        st.markdown("##### 📊 Model Performance Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        r2 = r2_score(target, predictions)
        mse = mean_squared_error(target, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(target, predictions)
        
        with col1:
            st.metric("R² Score", round(r2, 4))
        with col2:
            st.metric("Mean Squared Error (MSE)", round(mse, 4))
        with col3:
            st.metric("Root Mean Squared Error (RMSE)", round(rmse, 4))
        with col4:
            st.metric("Mean Absolute Error (MAE)", round(mae, 4))

        # === Visualization: Actual vs Predicted ===
        st.markdown("##### 📉 Actual vs Predicted Values")
        
        fig, ax = plt.subplots(figsize=(10, 5))
        x_axis = np.arange(len(target))
        ax.plot(x_axis, target, label="Actual", marker='o', alpha=0.7, linewidth=2)
        ax.plot(x_axis, predictions, label="Predicted", marker='s', alpha=0.7, linewidth=2)
        ax.set_xlabel("Sample Index")
        ax.set_ylabel("Value")
        ax.set_title("Actual vs Predicted Values")
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

        # === Visualization: Residual Plot ===
        st.markdown("##### 📊 Residual Plot")
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.scatter(predictions, results_df["Residual"], alpha=0.6, s=50)
        ax.axhline(y=0, color='r', linestyle='--', linewidth=2)
        ax.set_xlabel("Predicted Values")
        ax.set_ylabel("Residuals")
        ax.set_title("Residual Plot")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

        # === Visualization: Distribution of Residuals ===
        st.markdown("##### 📈 Residuals Distribution")
        
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.histplot(results_df["Residual"], kde=True, ax=ax, bins=30)
        ax.set_xlabel("Residuals")
        ax.set_ylabel("Frequency")
        ax.set_title("Distribution of Residuals")
        st.pyplot(fig)

        # === Feature Importance (if available) ===
        if hasattr(model, 'coef_'):
            st.markdown("##### 🎯 Model Coefficients")
            
            coef_df = pd.DataFrame({
                "Feature": feature.columns,
                "Coefficient": model.coef_
            }).sort_values("Coefficient", key=abs, ascending=False)
            
            fig, ax = plt.subplots(figsize=(10, max(5, len(coef_df) * 0.3)))
            ax.barh(coef_df["Feature"], coef_df["Coefficient"], color='steelblue')
            ax.set_xlabel("Coefficient Value")
            ax.set_title("Feature Coefficients")
            ax.grid(True, alpha=0.3, axis='x')
            st.pyplot(fig)
            
            with st.expander("Show Coefficients Table"):
                st.dataframe(coef_df, use_container_width=True)

        if hasattr(model, 'feature_importances_'):
            st.markdown("##### 🎯 Feature Importance")
            
            importance_df = pd.DataFrame({
                "Feature": feature.columns,
                "Importance": model.feature_importances_
            }).sort_values("Importance", ascending=False)
            
            fig, ax = plt.subplots(figsize=(10, max(5, len(importance_df) * 0.3)))
            ax.barh(importance_df["Feature"], importance_df["Importance"], color='coral')
            ax.set_xlabel("Importance Score")
            ax.set_title("Feature Importance")
            ax.grid(True, alpha=0.3, axis='x')
            st.pyplot(fig)
            
            with st.expander("Show Feature Importance Table"):
                st.dataframe(importance_df, use_container_width=True)

        # === Error Analysis ===
        st.markdown("##### 🔍 Error Analysis")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Max Error", round(results_df["Absolute Error"].max(), 4))
        with col2:
            st.metric("Min Error", round(results_df["Absolute Error"].min(), 4))
        with col3:
            st.metric("Mean Error", round(results_df["Absolute Error"].mean(), 4))

        # === Top Worst Predictions ===
        st.markdown("##### ⚠️ Worst Predictions")
        worst_predictions = results_df.nlargest(5, "Absolute Error")
        st.dataframe(worst_predictions, use_container_width=True)

    except Exception as e:
        st.error(f"❌ Error generating model output: {e}")
