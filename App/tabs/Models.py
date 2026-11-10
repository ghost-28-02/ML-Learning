from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd
import streamlit as st

def render(feature, target):
    st.subheader("📦 Models")
    st.write("Train a Linear Regression model on the selected features and target.")

    # Validate input
    if feature is None or feature.empty or target is None:
        st.warning("Please provide valid feature columns and a target array.")
        return

    if st.button("🚀 Train Linear Regression"):
        try:
            X = feature  # assumed to be a DataFrame
            y = target  # assumed to be np.array or Series

            model = LinearRegression()
            model.fit(X, y)
            preds = model.predict(X)

            # Store model and data in session state for Model_Output tab
            st.session_state.model = model
            st.session_state.feature = X
            st.session_state.target = y

            st.markdown("### 📊 Model Results")
            st.write("**R² Score:**", round(r2_score(y, preds), 4))
            st.write("**Mean Squared Error:**", round(mean_squared_error(y, preds), 4))

            # Show coefficients
            coef_df = pd.DataFrame({
                "Feature": X.columns,
                "Coefficient": model.coef_
            })
            st.write("**Coefficients:**")
            st.dataframe(coef_df)

            st.success("✅ Model trained successfully! Check the Model Output tab for detailed analysis.")

        except Exception as e:
            st.error(f"❌ Error training Linear Regression model: {e}")