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

# Correctly locate the pickle model
# Assumes 'lung_disease_nb_model.pkl' is in the same folder as app.py
model_filename = "lung_disease_nb_model.pkl"

if not os.path.isfile(model_filename):
    st.error(
        f"Model file not found: '{model_filename}'.\n"
        "Make sure 'lung_disease_nb_model.pkl' is uploaded to the same folder as app.py."
    )
    st.stop()

# Load the model
model = joblib.load(model_filename)

# Prediction
if st.button("Predict"):
    input_data = np.array([[smoking_val, peer_val]])
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("Lung Disease: YES")
    else:
        st.success("Lung Disease: NO")
