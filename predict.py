import joblib
import numpy as np


# ==========================================
# LOAD TRAINED FILES
# ==========================================

model = joblib.load("model/best_model.pkl")
scaler = joblib.load("model/scaler.pkl")
imputer = joblib.load("model/imputer.pkl")

print("=" * 50)
print("DIABETES PREDICTION SYSTEM")
print("=" * 50)


# ==========================================
# TAKE USER INPUT
# ==========================================

print("\nEnter the following patient information:\n")

pregnancies = float(input("Pregnancies: "))
glucose = float(input("Glucose: "))
blood_pressure = float(input("BloodPressure: "))
skin_thickness = float(input("SkinThickness: "))
insulin = float(input("Insulin: "))
bmi = float(input("BMI: "))
diabetes_pedigree = float(
    input("DiabetesPedigreeFunction: ")
)
age = float(input("Age: "))


# ==========================================
# CREATE INPUT ARRAY
# ==========================================

input_data = np.array([[
    pregnancies,
    glucose,
    blood_pressure,
    skin_thickness,
    insulin,
    bmi,
    diabetes_pedigree,
    age
]])


# ==========================================
# HANDLE MISSING VALUES
# ==========================================

input_data = imputer.transform(input_data)


# ==========================================
# SCALE INPUT
# ==========================================

input_data = scaler.transform(input_data)


# ==========================================
# PREDICT
# ==========================================

prediction = model.predict(input_data)


# ==========================================
# DISPLAY RESULT
# ==========================================

print("\n" + "=" * 50)

if prediction[0] == 1:
    print("Prediction: DIABETES")
else:
    print("Prediction: NO DIABETES")

print("=" * 50)