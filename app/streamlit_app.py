import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt

st.title("Diabetes Prediction App")
st.write("Enter the patient details below:")

# Define the features in the exact order the model was trained on.
# The order is: Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin,
# BMI, DiabetesPedigreeFunction, Age.
features = {}

# For integer inputs:
features["Pregnancies"] = st.number_input("Enter Pregnancies", value=0, step=1)
features["Glucose"] = st.number_input("Enter Glucose", value=0, step=1)
features["BloodPressure"] = st.number_input("Enter BloodPressure", value=0, step=1)
features["SkinThickness"] = st.number_input("Enter SkinThickness", value=0, step=1)
features["Insulin"] = st.number_input("Enter Insulin", value=0, step=1)

# For float inputs:
features["BMI"] = st.number_input("Enter BMI", value=0.0, step=0.1)
features["DiabetesPedigreeFunction"] = st.number_input("Enter DiabetesPedigreeFunction", value=0.00, step=0.01)

# For integer input:
features["Age"] = st.number_input("Enter Age", value=0, step=1)

if st.button("Predict"):
    # Create a list of feature values in the exact order required.
    feature_names = [
        "Pregnancies",
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI",
        "DiabetesPedigreeFunction",
        "Age"
    ]
    features_list = [features[name] for name in feature_names]

    # Call the Flask API.
    url = "http://127.0.0.1:5000/predict"  # Change this if your API is hosted elsewhere.
    payload = {"features": features_list}
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
    except Exception as e:
        st.error(f"API request failed: {e}")
    else:
        result = response.json()
        prediction = result.get("prediction")
        probabilities = result.get("probabilities", [])
        shap_values = result.get("shap_values", [])

        st.subheader("Prediction")
        if prediction == 1:
            st.error("The patient is predicted to be Diabetic.")
        else:
            st.success("The patient is predicted to be Non-diabetic.")
        if probabilities:
            st.write("Prediction Probabilities:", probabilities)

        st.subheader("Feature Contributions (SHAP values)")
        # Create a DataFrame for displaying SHAP values.
        shap_df = pd.DataFrame({
            "Feature": feature_names,
            "Input": features_list,
            "SHAP Value": shap_values
        })
        st.dataframe(shap_df)

        # Plot a horizontal bar chart for SHAP values.
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(feature_names, shap_values, color='skyblue')
        ax.set_xlabel("SHAP Value")
        ax.set_title("Feature Contribution to Prediction")
        st.pyplot(fig)
