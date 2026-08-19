# ============================================================
# DIABETES PREDICTION SYSTEM
# ============================================================

import os

# ------------------------------------------------------------
# Reduce OpenBLAS / NumPy memory usage
# MUST be before numpy/sklearn imports
# ------------------------------------------------------------

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"


# ============================================================
# IMPORTS
# ============================================================

from flask import Flask, render_template, request

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "model"
)

DATASET_PATH = os.path.join(
    BASE_DIR,
    "dataset",
    "diabetes.csv"
)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "best_model.pkl"
)

SCALER_PATH = os.path.join(
    MODEL_DIR,
    "scaler.pkl"
)

IMPUTER_PATH = os.path.join(
    MODEL_DIR,
    "imputer.pkl"
)


# ============================================================
# LOAD MODEL
# ============================================================

model = None
scaler = None
imputer = None

MODEL_LOADED = False


try:

    print("=" * 60)
    print("Loading machine learning model...")
    print("=" * 60)

    model = joblib.load(
        MODEL_PATH
    )

    scaler = joblib.load(
        SCALER_PATH
    )

    imputer = joblib.load(
        IMPUTER_PATH
    )

    MODEL_LOADED = True

    print("Model loaded successfully.")
    print("Model:", MODEL_PATH)
    print("Scaler:", SCALER_PATH)
    print("Imputer:", IMPUTER_PATH)

except Exception as e:

    print("=" * 60)
    print("MODEL LOADING ERROR")
    print("=" * 60)

    print(e)

    print("=" * 60)


# ============================================================
# HOME
# ============================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# ============================================================
# PREDICTION
# ============================================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    try:

        # ----------------------------------------------------
        # Check model
        # ----------------------------------------------------

        if not MODEL_LOADED:

            return render_template(

                "result.html",

                prediction="Error",

                result_message=(
                    "The machine learning model "
                    "could not be loaded."
                ),

                probability=None

            )


        # ----------------------------------------------------
        # Get form values
        # ----------------------------------------------------

        pregnancies = float(
            request.form.get(
                "pregnancies",
                0
            )
        )

        glucose = float(
            request.form.get(
                "glucose",
                0
            )
        )

        blood_pressure = float(
            request.form.get(
                "blood_pressure",
                0
            )
        )

        skin_thickness = float(
            request.form.get(
                "skin_thickness",
                0
            )
        )

        insulin = float(
            request.form.get(
                "insulin",
                0
            )
        )

        bmi = float(
            request.form.get(
                "bmi",
                0
            )
        )

        diabetes_pedigree = float(
            request.form.get(
                "diabetes_pedigree",
                0
            )
        )

        age = float(
            request.form.get(
                "age",
                0
            )
        )


        # ----------------------------------------------------
        # Create input array
        # ----------------------------------------------------

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


        # ----------------------------------------------------
        # Imputation
        # ----------------------------------------------------

        input_data = imputer.transform(
            input_data
        )


        # ----------------------------------------------------
        # Scaling
        # ----------------------------------------------------

        input_data = scaler.transform(
            input_data
        )


        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        prediction = model.predict(
            input_data
        )[0]


        # ----------------------------------------------------
        # Probability
        # ----------------------------------------------------

        probability = None


        if hasattr(
            model,
            "predict_proba"
        ):

            probabilities = model.predict_proba(
                input_data
            )

            probability = float(
                probabilities[0][1] * 100
            )


        # ----------------------------------------------------
        # Result
        # ----------------------------------------------------

        if prediction == 1:

            result = "Positive"

            result_message = (
                "The model predicts a higher "
                "likelihood of diabetes based "
                "on the provided parameters."
            )

        else:

            result = "Negative"

            result_message = (
                "The model predicts a lower "
                "likelihood of diabetes based "
                "on the provided parameters."
            )


        # ----------------------------------------------------
        # Send result to HTML
        # ----------------------------------------------------

        return render_template(

            "result.html",

            prediction=result,

            result_message=result_message,

            probability=probability

        )


    except Exception as e:

        print()
        print("PREDICTION ERROR")
        print(e)
        print()


        return render_template(

            "result.html",

            prediction="Error",

            result_message=str(e),

            probability=None

        )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

@app.route("/performance")
def performance():

    try:

        # ----------------------------------------------------
        # Check model
        # ----------------------------------------------------

        if not MODEL_LOADED:

            return render_template(

                "performance.html",

                error="Machine learning model is not loaded.",

                accuracy=0,
                precision=0,
                recall=0,
                f1=0,

                tn=0,
                fp=0,
                fn=0,
                tp=0

            )


        # ----------------------------------------------------
        # Check dataset
        # ----------------------------------------------------

        print()
        print("Dataset path:")
        print(DATASET_PATH)


        if not os.path.exists(
            DATASET_PATH
        ):

            error_message = (

                "Dataset not found.\n\n"

                "Expected location:\n"

                + DATASET_PATH

            )


            return render_template(

                "performance.html",

                error=error_message,

                accuracy=0,
                precision=0,
                recall=0,
                f1=0,

                tn=0,
                fp=0,
                fn=0,
                tp=0

            )


        # ----------------------------------------------------
        # Read dataset
        # ----------------------------------------------------

        df = pd.read_csv(
            DATASET_PATH
        )


        print(
            "Dataset loaded:",
            df.shape
        )


        # ----------------------------------------------------
        # Feature columns
        # ----------------------------------------------------

        feature_columns = [

            "Pregnancies",
            "Glucose",
            "BloodPressure",
            "SkinThickness",
            "Insulin",
            "BMI",
            "DiabetesPedigreeFunction",
            "Age"

        ]


        # ----------------------------------------------------
        # Check columns
        # ----------------------------------------------------

        missing_columns = [

            column

            for column in feature_columns

            if column not in df.columns

        ]


        if missing_columns:

            return render_template(

                "performance.html",

                error=(
                    "Missing columns: "
                    + str(missing_columns)
                ),

                accuracy=0,
                precision=0,
                recall=0,
                f1=0,

                tn=0,
                fp=0,
                fn=0,
                tp=0

            )


        if "Outcome" not in df.columns:

            return render_template(

                "performance.html",

                error=(
                    "Outcome column not found "
                    "in diabetes.csv."
                ),

                accuracy=0,
                precision=0,
                recall=0,
                f1=0,

                tn=0,
                fp=0,
                fn=0,
                tp=0

            )


        # ----------------------------------------------------
        # X and y
        # ----------------------------------------------------

        X = df[
            feature_columns
        ]

        y = df[
            "Outcome"
        ]


        # ----------------------------------------------------
        # Impute
        # ----------------------------------------------------

        X_imputed = imputer.transform(
            X
        )


        # ----------------------------------------------------
        # Scale
        # ----------------------------------------------------

        X_scaled = scaler.transform(
            X_imputed
        )


        # ----------------------------------------------------
        # Predict
        # ----------------------------------------------------

        y_pred = model.predict(
            X_scaled
        )


        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        accuracy = accuracy_score(
            y,
            y_pred
        )

        precision = precision_score(
            y,
            y_pred,
            zero_division=0
        )

        recall = recall_score(
            y,
            y_pred,
            zero_division=0
        )

        f1 = f1_score(
            y,
            y_pred,
            zero_division=0
        )


        # ----------------------------------------------------
        # Confusion matrix
        # ----------------------------------------------------

        cm = confusion_matrix(
            y,
            y_pred
        )


        tn = int(
            cm[0][0]
        )

        fp = int(
            cm[0][1]
        )

        fn = int(
            cm[1][0]
        )

        tp = int(
            cm[1][1]
        )


        # ----------------------------------------------------
        # Display
        # ----------------------------------------------------

        return render_template(

            "performance.html",

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
            tp=tp

        )


    except Exception as e:

        print()
        print("PERFORMANCE ERROR")
        print(e)
        print()


        return render_template(

            "performance.html",

            error=str(e),

            accuracy=0,
            precision=0,
            recall=0,
            f1=0,

            tn=0,
            fp=0,
            fn=0,
            tp=0

        )


# ============================================================
# ABOUT
# ============================================================

@app.route("/about")
def about():

    return render_template(
        "about.html"
    )


# ============================================================
# HISTORY
# ============================================================

@app.route("/history")
def history():

    return render_template(
        "history.html"
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 60)
    print("       DIABETES PREDICTION SYSTEM")
    print("=" * 60)

    print()
    print("Project:")
    print(BASE_DIR)

    print()
    print("Dataset:")
    print(DATASET_PATH)

    print()
    print("Model loaded:")
    print(MODEL_LOADED)

    print()
    print("=" * 60)

    app.run(
        debug=True,
        threaded=False
    )