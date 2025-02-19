# Diabetes-prediction

## 📌 Project Overview
This project predicts whether a patient has diabetes based on medical features using machine learning models. Several algorithms were trained, including a neural network with PyTorch, to select the best model. The project is deployed using:
- **Flask API** (for backend predictions)
- **Streamlit App** (for user-friendly interaction)
- **SHAP** (for explainability of model predictions)

---

## 📊 Dataset
The dataset contains the following medical features:
- **Pregnancies**
- **Glucose**
- **BloodPressure**
- **SkinThickness**
- **Insulin**
- **BMI**
- **DiabetesPedigreeFunction**
- **Age**

The target variable (`Outcome`) indicates whether a patient has diabetes (1) or not (0).

---

## ⚙️ Installation
1. **Clone the Repository:**
   ```bash
   git clone https://github.com/Shay-Ostrovsky/Diabetes-prediction.git
   cd diabetes-prediction
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 🖥️ Running the Application

### 1️⃣ Run the Flask API
```bash
python app/app.py
```
This will start a local API server at `http://127.0.0.1:5000/predict`.

**Example API Request:**
```json
{
  "features": [2, 138, 62, 35, 0, 33.6, 0.127, 47]
}
```

### 2️⃣ Run the Streamlit App
```bash
streamlit run app/streamlit_app.py
```
This will open an interactive web app where users can enter feature values and get predictions with SHAP explanations.

---

## 📁 Repository Structure
```
diabetes-prediction/
├── README.md
├── requirements.txt
├── data/
│   └── diabetes.csv
├── notebooks/
│   └── Diabetes_Prediction_Notebook.ipynb
├── app/
│   ├── app.py
│   ├── streamlit_app.py
│   ├── model.pkl
│   └── scaler.pkl
```

---

## 📊 Model Training & Evaluation
- **8 different machine learning models** were trained.
- A **neural network with PyTorch** was also tested.
- The best-performing model was saved as `model.pkl`.
- **SHAP values** were used to explain model decisions.

---

## 🎯 Features
✅ **Machine Learning Pipeline** (Data Cleaning, Training, Evaluation)  
✅ **Flask API Deployment**  
✅ **Streamlit Web App for User Interaction**  
✅ **SHAP Explainability for Predictions**  
✅ **Visualizations of Model Performance**  
