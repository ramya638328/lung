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

# Path to model
model_filename = "lung_disease_nb_model.pkl"

# Check if model exists
if os.path.isfile(model_filename):
    model = joblib.load(model_filename)
    model_found = True
else:
    st.warning(f"Model file not found: '{model_filename}'. Predictions will not work.")
    model_found = False

# Prediction button
if st.button("Predict"):
    if model_found:
        input_data = np.array([[smoking_val, peer_val]])
        prediction = model.predict(input_data)

        if prediction[0] == 1:
            st.error("Lung Disease: YES")
        else:
            st.success("Lung Disease: NO")
    else:
        st.error("Cannot predict because model file is missing.")
