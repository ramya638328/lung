import streamlit as st
import numpy as np
import joblib

# Load trained model
model = joblib.load('lung_disease_nb_model.pkl')

st.title("Lung Disease Prediction – Naive Bayes")

st.write("### Input Features")

smoking = st.selectbox("Smoking", ["No", "Yes"])
peer = st.selectbox("Peer Pressure", ["No", "Yes"])

# Encode inputs
smoking_val = 1 if smoking == "Yes" else 0
peer_val = 1 if peer == "Yes" else 0

if st.button("Predict"):
    input_data = np.array([[smoking_val, peer_val]])
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("Lung Disease: YES")
    else:
        st.success("Lung Disease: NO")
