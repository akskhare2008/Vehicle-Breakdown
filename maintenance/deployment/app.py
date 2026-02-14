import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the model
model_path = hf_hub_download(repo_id="akskhare/vehicle-breakdown-model", filename="gradient_boosting_vehicle_breakdown_v1.joblib")
model = joblib.load(model_path)

# Streamlit UI for Tourism Packages Prediction
st.title("Predictive Maintenance")
st.write("""
This application predicts whether an Engine is Faulty or Active.
Please enter the Engine Feature Details.
""")

# User input
Engine rpm = st.number_input("Engine RPM", min_value=50, max_value=2500)
Lub oil pressure = st.number_input("Lube Oil Pressure", min_value=0.001, max_value=10.000)
Fuel pressure = st.number_input("Fuel Pressure", min_value=0.001, max_value=25.000)
Coolant pressure = st.number_input("Coolant Pressure", min_value=0.001, max_value=10.000)
lub oil temp = st.number_input("Lube Oil Temperature", min_value=50.000, max_value=150.000)
Coolant temp = st.number_input("Coolant Temperature", min_value=50.000, max_value=250.000)



# Assemble input into DataFrame with correct column names
input_data = pd.DataFrame([{
    'Engine RPM': Engine rpm,
    'Lube Oil Pressure': Lub oil pressure,
    'Fuel Pressure': Fuel pressure,
    'Coolant Pressure': Coolant pressure,
    'Lube Oil Temperature': lub oil temp,
    'Coolant Temperature': Coolant temp

}])

if st.button("Predict Engine Condition"): # Changed button text for clarity
    prediction = model.predict(input_data)[0]
    result = "Engine is " if prediction == 0 else "Engine is Faulty"
    st.subheader("Prediction Result:")
    st.success(f"The model predicts: **{result}**")
