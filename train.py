import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# ==============================
# PATHS
# ==============================

DATASET_PATH = "C:/Users/HP/Downloads/archive (10)/diabetes_project/dataset/diabetes.csv"

os.makedirs("model", exist_ok=True)


# ==============================
# LOAD DATASET
# ==============================

print("=" * 50)
print("DIABETES MODEL TRAINING")
print("=" * 50)

df = pd.read_csv(DATASET_PATH)

print("\nDataset loaded successfully!")
print("Shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())


# ==============================
# REMOVE DUPLICATES
# ==============================

duplicates = df.duplicated().sum()

print("\nDuplicate rows:", duplicates)

df = df.drop_duplicates()


# ==============================
# FEATURES AND TARGET
# ==============================

features = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age"
]

target = "Outcome"


X = df[features].copy()
y = df[target].copy()


# ==============================
# HANDLE INVALID ZERO VALUES
# ==============================

zero_columns = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI"
]

for column in zero_columns:
    X.loc[X[column] == 0, column] = float("nan")


# ==============================
# TRAIN TEST SPLIT
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ==============================
# IMPUTE MISSING VALUES
# ==============================

imputer = SimpleImputer(strategy="median")

X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)


# ==============================
# SCALE DATA
# ==============================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# ==============================
# MODELS
# ==============================

models = {

    "Logistic Regression":
        LogisticRegression(max_iter=1000),

    "Random Forest":
        RandomForestClassifier(
            n_estimators=200,
            random_state=42
        ),

    "KNN":
        KNeighborsClassifier(n_neighbors=5),

    "SVM":
        SVC(),

    "Gradient Boosting":
        GradientBoostingClassifier(
            random_state=42
        )
}


# ==============================
# TRAIN MODELS
# ==============================

results = []

best_model = None
best_name = ""
best_accuracy = 0


print("\n" + "=" * 50)
print("TRAINING MODELS")
print("=" * 50)


for name, model in models.items():

    print("\nTraining:", name)

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    print("Accuracy :", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall   :", round(recall, 4))
    print("F1 Score :", round(f1, 4))

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    })

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model
        best_name = name


# ==============================
# RESULTS
# ==============================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    "Accuracy",
    ascending=False
)

print("\n" + "=" * 50)
print("MODEL COMPARISON")
print("=" * 50)

print(results_df.to_string(index=False))


# ==============================
# BEST MODEL
# ==============================

print("\n" + "=" * 50)
print("BEST MODEL")
print("=" * 50)

print("Best Model:", best_name)
print("Best Accuracy:", round(best_accuracy, 4))


# ==============================
# BEST MODEL REPORT
# ==============================

best_predictions = best_model.predict(X_test)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        best_predictions,
        zero_division=0
    )
)

print("\nConfusion Matrix:")
print(
    confusion_matrix(
        y_test,
        best_predictions
    )
)


# ==============================
# SAVE FILES
# ==============================

joblib.dump(
    best_model,
    "model/best_model.pkl"
)

joblib.dump(
    scaler,
    "model/scaler.pkl"
)

joblib.dump(
    imputer,
    "model/imputer.pkl"
)

results_df.to_csv(
    "model/model_results.csv",
    index=False
)


print("\n" + "=" * 50)
print("TRAINING COMPLETED!")
print("=" * 50)

print("\nFiles created:")
print("1. model/best_model.pkl")
print("2. model/scaler.pkl")
print("3. model/imputer.pkl")
print("4. model/model_results.csv")