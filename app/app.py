# app.py
from flask import Flask, request, jsonify
import joblib 
import numpy as np
import shap

app = Flask(__name__)

# Load the saved model
model = joblib.load("app/model.pkl")

# If you used a scaler during training, load it.
# If not, you can remove or replace the scaling part.
try:
    scaler = joblib.load("app/scaler.pkl")

except FileNotFoundError:
    scaler = None
    print("Scaler file not found. Proceeding without scaling.")

# Prepare a SHAP explainer.
# If your model is tree-based, TreeExplainer is efficient.
# Otherwise, consider KernelExplainer.
try:
    explainer = shap.TreeExplainer(model)
    print("tree based")

except Exception as e:
    # In this fallback, we use KernelExplainer.
    # Note: KernelExplainer may be slower.
    # Here we create a simple background dataset.
    print("not tree based")
    dummy_data = np.zeros((1, 8))  # assuming 8 features for diabetes prediction
    explainer = shap.KernelExplainer(model.predict, dummy_data)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    features = data.get("features")
    
    if features is None or len(features) != 8:
        return jsonify({"error": "Please provide all 8 feature values."}), 400

    # Convert features to a numpy array and reshape for a single sample.
    input_array = np.array(features).reshape(1, -1)
    
    # Apply scaling if a scaler is available.
    if scaler is not None:
        input_array = scaler.transform(input_array)
    
    # Get the prediction and prediction probabilities.
    prediction = model.predict(input_array)[0]
    try:
        prediction_prob = model.predict_proba(input_array)[0].tolist()
    except AttributeError:
        prediction_prob = []

    # Compute SHAP values.
    shap_values = explainer.shap_values(input_array)
    # For binary classification, SHAP may return a list with two arrays.
    if isinstance(shap_values, list) and len(shap_values) == 2:
        # We choose the SHAP values for the positive class (index 1).
        shap_values = shap_values[1]
    shap_values = shap_values.tolist()[0]

    return jsonify({
        "prediction": int(prediction),
        "probabilities": prediction_prob,
        "shap_values": shap_values
    })

if __name__ == "__main__":
    app.run(debug=True)
