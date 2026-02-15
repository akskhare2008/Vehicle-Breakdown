import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the model
model_path = hf_hub_download(repo_id="akskhare/vehicle-breakdown-model", filename="gradient_boosting_vehicle_breakdown_v1.joblib")
model = joblib.load(model_path)

# Streamlit UI
st.title("Predictive Maintenance")
st.write("""
This application predicts whether an Engine is Faulty or Active.
Please enter the Engine Feature Details.
""")

# User input with valid variable names
engine_rpm = st.number_input("Engine RPM", min_value=50, max_value=2500)
lub_oil_pressure = st.number_input("Lube Oil Pressure", min_value=0.001, max_value=10.000)
fuel_pressure = st.number_input("Fuel Pressure", min_value=0.001, max_value=25.000)
coolant_pressure = st.number_input("Coolant Pressure", min_value=0.001, max_value=10.000)
lub_oil_temp = st.number_input("Lube Oil Temperature", min_value=50.000, max_value=150.000)
coolant_temp = st.number_input("Coolant Temperature", min_value=50.000, max_value=250.000)

# Assemble input into DataFrame with names matching the training columns
input_data = pd.DataFrame([{
    'Engine rpm': engine_rpm,
    'Lub oil pressure': lub_oil_pressure,
    'Fuel pressure': fuel_pressure,
    'Coolant pressure': coolant_pressure,
    'lub oil temp': lub_oil_temp,
    'Coolant temp': coolant_temp
}])

if st.button("Predict Engine Condition"):
    prediction = model.predict(input_data)[0]
    result = "Active (Normal)" if prediction == 0 else "Faulty"
    st.subheader("Prediction Result:")
    if prediction == 0:
        st.success(f"The model predicts: **{result}**")
    else:
        st.error(f"The model predicts: **{result}**")
