import streamlit as st
import joblib
import numpy as np

# Load your trained model
model = joblib.load("breast_cancer_model.pkl")

st.title("🔬 Breast Cancer Prediction App")
st.write("This app predicts whether a tumor is **Benign** or **Malignant** based on 30 input features.")

# ---- Feature names (must be in the exact order used for training) ----
feature_names = [
    "Radius Mean", "Texture Mean", "Perimeter Mean", "Area Mean", "Smoothness Mean",
    "Compactness Mean", "Concavity Mean", "Concave Points Mean", "Symmetry Mean", "Fractal Dimension Mean",
    "Radius SE", "Texture SE", "Perimeter SE", "Area SE", "Smoothness SE",
    "Compactness SE", "Concavity SE", "Concave Points SE", "Symmetry SE", "Fractal Dimension SE",
    "Radius Worst", "Texture Worst", "Perimeter Worst", "Area Worst", "Smoothness Worst",
    "Compactness Worst", "Concavity Worst", "Concave Points Worst", "Symmetry Worst", "Fractal Dimension Worst"
]

# ---- Arrange inputs in 3 columns ----
inputs = []
cols = st.columns(3)
for i, name in enumerate(feature_names):
    with cols[i % 3]:  # distribute features across 3 columns
        val = st.number_input(name, min_value=0.0, format="%.5f")
        inputs.append(val)

# ---- Prediction ----
if st.button("Predict"):
    input_features = np.array(inputs).reshape(1, -1)  # shape must be (1, 30)
    prediction = model.predict(input_features)[0]

    if prediction == 1:
        st.error("⚠️ The tumor is **Malignant**")
    else:
        st.success("✅ The tumor is **Benign**")
