import streamlit as st
import numpy as np
import joblib
import os

st.title("Lung Disease Prediction – Naive Bayes")
st.write("### Input Features")

# Input fields
smoking = st.selectbox("Smoking", ["No", "Yes"])
peer = st.selectbox("Peer Pressure", ["No", "Yes"])

# Encode inputs
smoking_val = 1 if smoking == "Yes" else 0
peer_val = 1 if peer == "Yes" else 0

# Use a relative path to ensure model is found
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "lung_disease_nb_model.pkl")

# Load the model safely
try:
    model = joblib.load(model_path)
except FileNotFoundError:
    st.error(f"Model file not found at {model_path}. Make sure 'lung_disease_nb_model.pkl' is in the same folder as app.py.")
    st.stop()  # Stop execution if model is missing

# Prediction button
if st.button("Predict"):
    input_data = np.array([[smoking_val, peer_val]])
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("Lung Disease: YES")
    else:
        st.success("Lung Disease: NO")
