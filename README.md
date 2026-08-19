# Diabetes Prediction System 🩺

A machine learning based web application that predicts the likelihood of diabetes using patient health parameters.

## 📌 Project Overview

The Diabetes Prediction System is a Flask-based machine learning application.

Users can enter health-related information and receive a prediction from a trained machine learning model.

The application also displays the model's estimated probability and model performance metrics.

## 🚀 Features

- Diabetes prediction using Machine Learning
- User-friendly Flask web interface
- Data preprocessing using imputation
- Feature scaling
- Prediction probability
- Model performance dashboard
- Accuracy, Precision, Recall and F1-Score
- Confusion Matrix
- Responsive UI
- Educational medical disclaimer

## 🧠 Input Parameters

The system uses the following parameters:

1. Pregnancies
2. Glucose
3. Blood Pressure
4. Skin Thickness
5. Insulin
6. BMI
7. Diabetes Pedigree Function
8. Age

## 🛠️ Technologies Used

- Python
- Flask
- NumPy
- Pandas
- Scikit-learn
- Joblib
- HTML
- CSS
- Machine Learning

## 📂 Project Structure

```text
Diabetes-Prediction-System/
│
├── app.py
│
├── dataset/
│   └── diabetes.csv
│
├── model/
│   ├── best_model.pkl
│   ├── scaler.pkl
│   └── imputer.pkl
│
├── templates/
│   ├── index.html
│   ├── result.html
│   ├── performance.html
│   ├── about.html
│   └── history.html
│
├── static/
│   └── css/
│       └── style.css
│
├── requirements.txt
├── README.md
└── .gitignore