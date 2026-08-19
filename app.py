from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from database import (
    create_database,
    save_prediction,
    get_predictions
)


# =========================================================
# FLASK APPLICATION
# =========================================================

app = Flask(__name__)


# =========================================================
# BASE DIRECTORY
# =========================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)


# =========================================================
# MODEL PATHS
# =========================================================

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "best_model.pkl"
)

SCALER_PATH = os.path.join(
    BASE_DIR,
    "model",
    "scaler.pkl"
)

IMPUTER_PATH = os.path.join(
    BASE_DIR,
    "model",
    "imputer.pkl"
)


# =========================================================
# DATASET PATH
# =========================================================

DATASET_PATH = os.path.join(
    BASE_DIR,
    "diabetes.csv"
)


# =========================================================
# LOAD TRAINED MODEL
# =========================================================

model = joblib.load(MODEL_PATH)

scaler = joblib.load(SCALER_PATH)

imputer = joblib.load(IMPUTER_PATH)


# =========================================================
# CREATE DATABASE
# =========================================================

create_database()


# =========================================================
# FIND DATASET
# =========================================================

def find_dataset():

    # First check project folder

    if os.path.exists(DATASET_PATH):

        return DATASET_PATH


    # Search inside project folder

    for root, dirs, files in os.walk(BASE_DIR):

        if "diabetes.csv" in files:

            return os.path.join(
                root,
                "diabetes.csv"
            )


    return None


# =========================================================
# HOME PAGE
# =========================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# =========================================================
# ABOUT PAGE
# =========================================================

@app.route("/about")
def about():

    return render_template(
        "about.html"
    )


# =========================================================
# PREDICTION
# =========================================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    try:

        # -------------------------------------------------
        # GET FORM VALUES
        # -------------------------------------------------

        pregnancies = float(
            request.form["pregnancies"]
        )

        glucose = float(
            request.form["glucose"]
        )

        blood_pressure = float(
            request.form["blood_pressure"]
        )

        skin_thickness = float(
            request.form["skin_thickness"]
        )

        insulin = float(
            request.form["insulin"]
        )

        bmi = float(
            request.form["bmi"]
        )

        diabetes_pedigree = float(
            request.form["diabetes_pedigree"]
        )

        age = float(
            request.form["age"]
        )


        # -------------------------------------------------
        # CREATE INPUT ARRAY
        # -------------------------------------------------

        input_data = np.array([
            [
                pregnancies,
                glucose,
                blood_pressure,
                skin_thickness,
                insulin,
                bmi,
                diabetes_pedigree,
                age
            ]
        ])


        # -------------------------------------------------
        # IMPUTE
        # -------------------------------------------------

        input_data = imputer.transform(
            input_data
        )


        # -------------------------------------------------
        # SCALE
        # -------------------------------------------------

        input_data = scaler.transform(
            input_data
        )


        # -------------------------------------------------
        # PREDICTION
        # -------------------------------------------------

        prediction_value = model.predict(
            input_data
        )[0]


        # -------------------------------------------------
        # RESULT
        # -------------------------------------------------

        if prediction_value == 1:

            result = "Higher Risk of Diabetes"

            status = "risk"

        else:

            result = "Lower Risk of Diabetes"

            status = "safe"


        # -------------------------------------------------
        # SAVE PREDICTION
        # -------------------------------------------------

        save_prediction(

            pregnancies,

            glucose,

            blood_pressure,

            skin_thickness,

            insulin,

            bmi,

            diabetes_pedigree,

            age,

            result

        )


        # -------------------------------------------------
        # DISPLAY RESULT
        # -------------------------------------------------

        return render_template(

            "result.html",

            prediction=result,

            status=status

        )


    except Exception as e:

        return render_template(

            "result.html",

            prediction="Unable to make prediction",

            status="error",

            error=str(e)

        )


# =========================================================
# PREDICTION HISTORY
# =========================================================

@app.route("/history")
def history():

    predictions = get_predictions()

    return render_template(

        "history.html",

        predictions=predictions

    )


# =========================================================
# MODEL PERFORMANCE
# =========================================================

@app.route("/performance")
def performance():

    try:

        # -------------------------------------------------
        # FIND DATASET
        # -------------------------------------------------

        dataset_path = find_dataset()


        if dataset_path is None:

            return render_template(

                "model_performance.html",

                error=(
                    "diabetes.csv was not found. "
                    "Please place the dataset inside "
                    "the project folder."
                ),

                accuracy=None,

                precision=None,

                recall=None,

                f1=None,

                tn=0,

                fp=0,

                fn=0,

                tp=0,

                model_name=type(model).__name__

            )


        # -------------------------------------------------
        # LOAD DATASET
        # -------------------------------------------------

        df = pd.read_csv(
            dataset_path
        )


        # -------------------------------------------------
        # CHECK TARGET
        # -------------------------------------------------

        if "Outcome" not in df.columns:

            return render_template(

                "model_performance.html",

                error=(
                    "The dataset does not contain "
                    "an 'Outcome' column."
                ),

                accuracy=None,

                precision=None,

                recall=None,

                f1=None,

                tn=0,

                fp=0,

                fn=0,

                tp=0,

                model_name=type(model).__name__

            )


        # -------------------------------------------------
        # FEATURES
        # -------------------------------------------------

        X = df.drop(
            "Outcome",
            axis=1
        )


        # -------------------------------------------------
        # TARGET
        # -------------------------------------------------

        y = df["Outcome"]


        # -------------------------------------------------
        # TRAIN TEST SPLIT
        # -------------------------------------------------

        X_train, X_test, y_train, y_test = train_test_split(

            X,

            y,

            test_size=0.2,

            random_state=42,

            stratify=y

        )


        # -------------------------------------------------
        # IMPUTER
        # -------------------------------------------------

        X_test = imputer.transform(
            X_test
        )


        # -------------------------------------------------
        # SCALER
        # -------------------------------------------------

        X_test = scaler.transform(
            X_test
        )


        # -------------------------------------------------
        # PREDICTION
        # -------------------------------------------------

        y_pred = model.predict(
            X_test
        )


        # -------------------------------------------------
        # METRICS
        # -------------------------------------------------

        accuracy = accuracy_score(
            y_test,
            y_pred
        )

        precision = precision_score(
            y_test,
            y_pred,
            zero_division=0
        )

        recall = recall_score(
            y_test,
            y_pred,
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            y_pred,
            zero_division=0
        )


        # -------------------------------------------------
        # CONFUSION MATRIX
        # -------------------------------------------------

        cm = confusion_matrix(
            y_test,
            y_pred
        )


        if cm.shape == (2, 2):

            tn, fp, fn, tp = cm.ravel()

        else:

            tn = 0
            fp = 0
            fn = 0
            tp = 0


        # -------------------------------------------------
        # MODEL NAME
        # -------------------------------------------------

        model_name = type(
            model
        ).__name__


        # -------------------------------------------------
        # DISPLAY PERFORMANCE
        # -------------------------------------------------

        return render_template(

            "model_performance.html",

            error=None,

            accuracy=round(
                accuracy * 100,
                2
            ),

            precision=round(
                precision * 100,
                2
            ),

            recall=round(
                recall * 100,
                2
            ),

            f1=round(
                f1 * 100,
                2
            ),

            tn=tn,

            fp=fp,

            fn=fn,

            tp=tp,

            model_name=model_name

        )


    except Exception as e:

        return render_template(

            "model_performance.html",

            error=str(e),

            accuracy=None,

            precision=None,

            recall=None,

            f1=None,

            tn=0,

            fp=0,

            fn=0,

            tp=0,

            model_name=type(model).__name__

        )


# =========================================================
# RUN APPLICATION
# =========================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )