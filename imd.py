import streamlit as st
import joblib
import numpy as np
from util import set_background

set_background('bgs/01.png')

# Explicitly set scikit-learn version
import sklearn
st.write(f"Using scikit-learn version: {sklearn.__version__}")

# Load models
def load_models():
    models = {}
    cities = ["Kolkata", "Agartala", "Patna", "Bhubaneshwar", "Gorakhpur", "Guwahati", "Lucknow", "Ranchi"]
    for city in cities:
        model = joblib.load(f"{city}_model.pkl")
        models[city] = model
    return models

models = load_models()

def predict(city, inputs):
    model = models[city]
    inputs_array = np.array(inputs).reshape(1, -1)  # Reshape inputs for prediction
    prediction = model.predict(inputs_array)
    return prediction[0]  # Return the prediction result

def main():
    st.title("Thunderstorm Prediction")

    city = st.radio("Select City", ("Kolkata", "Agartala", "Patna", "Bhubaneshwar", "Gorakhpur", "Guwahati", "Lucknow", "Ranchi"))

    if city:
        st.header(city)
        inputs = get_inputs()
        if inputs is not None:
            if st.button("Predict"):
                prediction = predict(city, inputs)
                st.write("Prediction:", prediction)

def get_inputs():
    st.subheader("Enter Input Data")
    sweat_index = st.number_input("SWEAT Index")
    k_index = st.number_input("K Index")
    totals_totals_index = st.number_input("Totals Totals Index")
    environmental_stability = st.number_input("Environmental Stability")
    moisture_indices = st.number_input("Moisture Indices")
    convective_potential = st.number_input("Convective Potential")
    temperature_pressure = st.number_input("Temperature Pressure")
    moisture_temperature_profiles = st.number_input("Moisture Temperature Profiles")
    
    return [sweat_index, k_index, totals_totals_index, environmental_stability, moisture_indices, convective_potential, temperature_pressure, moisture_temperature_profiles]

if __name__ == "__main__":
    main()
