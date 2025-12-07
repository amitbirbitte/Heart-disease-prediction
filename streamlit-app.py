import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


# -----------------------------
# 1. Load data
# -----------------------------
@st.cache_data
def load_data():
    # Make sure heart_.csv is in the same folder as this app.py
    df = pd.read_csv("heart_.csv")

    # 👇 If your target column is named something else (like 'output'),
    # change it here and below.
    # Example: target_col = "output"
    target_col = "target" if "target" in df.columns else "output"

    return df, target_col


# -----------------------------
# 2. Train model
# -----------------------------
@st.cache_resource
def train_model(df, target_col):
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = DecisionTreeClassifier(
        random_state=42,
        max_depth=4
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    return model, X.columns.tolist(), acc


# -----------------------------
# 3. UI: Sidebar inputs
# -----------------------------
def build_user_input(df, feature_cols):
    st.sidebar.header("🧍 Patient Details")

    user_data = {}
    for col in feature_cols:
        col_data = df[col]
        dtype = col_data.dtype

        # If the column has few unique values → show as selectbox
        unique_vals = sorted(col_data.dropna().unique())
        if len(unique_vals) <= 10:
            # Use select box for categorical-like features
            selected = st.sidebar.selectbox(
                f"{col}",
                unique_vals,
                index=0
            )
            user_data[col] = selected

        else:
            # Numeric range slider / number input
            min_val = float(col_data.min())
            max_val = float(col_data.max())
            mean_val = float(col_data.mean())

            if np.issubdtype(dtype, np.integer):
                val = st.sidebar.slider(
                    f"{col}",
                    int(min_val),
                    int(max_val),
                    int(mean_val)
                )
                user_data[col] = int(val)
            else:
                val = st.sidebar.slider(
                    f"{col}",
                    float(min_val),
                    float(max_val),
                    float(round(mean_val, 2))
                )
                user_data[col] = float(val)

    # Convert dict to single-row DataFrame
    input_df = pd.DataFrame([user_data])
    return input_df


# -----------------------------
# 4. Streamlit App Layout
# -----------------------------
def main():
    st.set_page_config(
        page_title="Heart Disease Prediction",
        page_icon="❤️",
        layout="centered"
    )

    st.title("❤️ Heart Disease Prediction App")
    st.write(
        """
        This app uses a Machine Learning model trained on clinical data  
        to **predict whether a person is at risk of heart disease**.
        """
    )

    # Load data + train model
    df, target_col = load_data()
    model, feature_cols, acc = train_model(df, target_col)

    # Show dataset preview expander
    with st.expander("📊 View dataset sample"):
        st.write(df.head())
        st.write(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
        st.write(f"Target column: `{target_col}`")

    st.markdown("---")

    # Sidebar inputs
    input_df = build_user_input(df, feature_cols)

    st.subheader("📝 Input Summary")
    st.write(input_df)

    # Predict button
    if st.button("🔮 Predict Heart Disease Risk"):
        prediction = model.predict(input_df)[0]

        # If model supports predict_proba, show confidence
        try:
            proba = model.predict_proba(input_df)[0]
            prob_negative = proba[0]
            prob_positive = proba[1]
        except Exception:
            prob_negative = prob_positive = None

        st.markdown("---")
        st.subheader("📌 Prediction Result")

        if prediction == 1:
            st.error("⚠️ The model predicts **Heart Disease: YES (at risk)**.")
            if prob_positive is not None:
                st.write(f"Model confidence (risk): **{prob_positive * 100:.2f}%**")
        else:
            st.success("✅ The model predicts **Heart Disease: NO (low risk)**.")
            if prob_negative is not None:
                st.write(f"Model confidence (no risk): **{prob_negative * 100:.2f}%**")

        st.caption(
            "⚕️ This is a machine learning-based estimation only and **not a medical diagnosis**."
        )

    st.markdown("---")
    st.caption("Built with Streamlit · Amit Birbitte — Heart Disease Prediction Project")


if __name__ == "__main__":
    main()
