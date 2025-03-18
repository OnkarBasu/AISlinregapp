# -*- coding: utf-8 -*-
"""
Created on Tue Mar 18 01:03:46 2025

@author: abasu
"""

import streamlit as st
import pandas as pd
import pickle
import numpy as np
from datetime import datetime, timedelta
import time  # For loading animations

# Apply custom CSS for dark theme
st.markdown("""
    <style>
    body {
        background-color: #001f3f;  /* Dark Blue */
        color: white;
    }
    .stTextInput, .stSelectbox, .stNumberInput {
        background-color: #001f3f !important;
        color: white !important;
    }
    .stButton > button {
        background-color: #ffcc00 !important;
        color: black !important;
        font-size: 18px !important;
        font-weight: bold !important;
        border-radius: 10px !important;
    }
    .stButton > button:hover {
        background-color: #ff9900 !important;
        transform: scale(1.05);
    }
    </style>
""", unsafe_allow_html=True)

# Load the trained model
with open("eta_prediction_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

# Load the scaler
with open("scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

# Load the label encoders
with open("label_encoders.pkl", "rb") as encoders_file:
    label_encoders = pickle.load(encoders_file)

# Define the list of input features
features = ["SOG", "Acceleration", "COG", "Destination", 
            "Width_Length_Ratio", "Draught_Length_Ratio", "Navigational status"]

# Streamlit UI
st.title("🚢 Ship ETA Prediction App")
st.write("### 🌊 Enter the ship details below to predict the Estimated Time of Arrival (ETA).")

# Create input fields with sections
st.markdown("#### 📌 Ship Movement Features")
col1, col2 = st.columns(2)
with col1:
    sog = st.number_input("Speed Over Ground (SOG)", min_value=0.0, step=0.1)
    acceleration = st.number_input("Acceleration", min_value=-10.0, step=0.1)
    cog = st.number_input("Course Over Ground (COG)", min_value=0.0, max_value=360.0, step=0.1)

with col2:
    destination = st.selectbox("Destination", label_encoders["Destination"].classes_)
    width_length_ratio = st.number_input("Width to Length Ratio", min_value=0.0, step=0.01)
    draught_length_ratio = st.number_input("Draught to Length Ratio", min_value=0.0, step=0.01)
    navigational_status = st.selectbox("Navigational status", label_encoders["Navigational status"].classes_)

# Predict ETA button
if st.button("⚡ Predict ETA"):
    # Show a loading animation
    with st.spinner("🚢 Processing data and predicting ETA..."):
        time.sleep(2)  # Simulate a delay for effect

        # Encode categorical inputs
        destination_encoded = label_encoders["Destination"].transform([destination])[0]
        nav_status_encoded = label_encoders["Navigational status"].transform([navigational_status])[0]

        # Prepare input data
        input_data = np.array([[sog, acceleration, cog, destination_encoded, 
                                width_length_ratio, draught_length_ratio, nav_status_encoded]])

        # Scale the numerical features
        input_data[:, :5] = scaler.transform(input_data[:, :5])  

        # Predict ETA in seconds
        eta_seconds = model.predict(input_data)[0]

        # Convert seconds into a timestamp
        predicted_eta = datetime.now() + timedelta(seconds=eta_seconds)

    # Display the result with animation
    st.balloons()
    st.success(f"🕒 Predicted ETA: {predicted_eta.strftime('%Y-%m-%d %H:%M:%S')}")
