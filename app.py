# ============================================================
# DIABETES PREDICTION SYSTEM
# Flask Web Application
# ============================================================

# IMPORTANT:
# These environment variables MUST be set before
# importing numpy, pandas, sklearn or joblib.

import os

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
# FLASK APPLICATION
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


# ============================================================
# MODEL FILES
# ============================================================

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
# LOAD TRAINED MODEL
# ============================================================

model = None
scaler = None
imputer = None

MODEL_LOADED = False


try:

    print()
    print("=" * 60)
    print("Loading machine learning model...")
    print("=" * 60)

    model = joblib.load(
        MODEL_PATH
    )

    print("best_model.pkl loaded")

    scaler = joblib.load(
        SCALER_PATH
    )

    print("scaler.pkl loaded")

    imputer = joblib.load(
        IMPUTER_PATH
    )

    print("imputer.pkl loaded")

    MODEL_LOADED = True

    print()
    print("MODEL STATUS: SUCCESS")
    print("=" * 60)


except Exception as e:

    print()
    print("=" * 60)
    print("MODEL LOADING ERROR")
    print("=" * 60)

    print(e)

    print("=" * 60)

    MODEL_LOADED = False


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# ============================================================
# PREDICTION PAGE
# ============================================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    try:

        # ----------------------------------------------------
        # CHECK MODEL
        # ----------------------------------------------------

        if not MODEL_LOADED:

            return render_template(
                "result.html",

                prediction="Error",

                result_message=(
                    "Machine learning model "
                    "could not be loaded."
                )
            )


        # ----------------------------------------------------
        # GET USER INPUT
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
        # CREATE INPUT ARRAY
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
        # HANDLE MISSING VALUES
        # ----------------------------------------------------

        input_data = imputer.transform(
            input_data
        )


        # ----------------------------------------------------
        # SCALE DATA
        # ----------------------------------------------------

        input_data = scaler.transform(
            input_data
        )


        # ----------------------------------------------------
        # PREDICTION
        # ----------------------------------------------------

        prediction = model.predict(
            input_data
        )[0]


        # ----------------------------------------------------
        # RESULT
        # ----------------------------------------------------

        if prediction == 1:

            result = "Positive"

            result_message = (
                "The model predicts a higher "
                "risk of diabetes."
            )

        else:

            result = "Negative"

            result_message = (
                "The model predicts a lower "
                "risk of diabetes."
            )


        # ----------------------------------------------------
        # RETURN RESULT
        # ----------------------------------------------------

        return render_template(

            "result.html",

            prediction=result,

            result_message=result_message

        )


    except Exception as e:

        print()
        print("PREDICTION ERROR:")
        print(e)


        return render_template(

            "result.html",

            prediction="Error",

            result_message=str(e)

        )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

@app.route("/performance")
def performance():

    try:

        print()
        print("=" * 60)
        print("CALCULATING MODEL PERFORMANCE")
        print("=" * 60)


        # ----------------------------------------------------
        # CHECK MODEL
        # ----------------------------------------------------

        if not MODEL_LOADED:

            return render_template(

                "performance.html",

                error=(
                    "Machine learning model "
                    "is not loaded."
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
        # CHECK DATASET
        # ----------------------------------------------------

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

            print(error_message)


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
        # LOAD DATASET
        # ----------------------------------------------------

        print("Reading dataset...")

        df = pd.read_csv(
            DATASET_PATH
        )


        print(
            "Dataset loaded successfully."
        )

        print(
            "Dataset shape:",
            df.shape
        )

        print(
            "Dataset columns:"
        )

        print(
            df.columns.tolist()
        )


        # ----------------------------------------------------
        # EXPECTED FEATURES
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
        # CHECK FEATURES
        # ----------------------------------------------------

        missing_columns = [

            column

            for column in feature_columns

            if column not in df.columns

        ]


        if missing_columns:

            error_message = (

                "The following columns are missing "
                "from diabetes.csv:\n"

                + str(missing_columns)

            )


            print(error_message)


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
        # CHECK TARGET
        # ----------------------------------------------------

        if "Outcome" not in df.columns:

            error_message = (

                "The 'Outcome' column "
                "was not found in diabetes.csv."
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
        # FEATURES
        # ----------------------------------------------------

        X = df[
            feature_columns
        ]


        # ----------------------------------------------------
        # TARGET
        # ----------------------------------------------------

        y = df[
            "Outcome"
        ]


        # ----------------------------------------------------
        # HANDLE MISSING VALUES
        # ----------------------------------------------------

        print(
            "Applying imputer..."
        )

        X_imputed = imputer.transform(
            X
        )


        # ----------------------------------------------------
        # SCALE FEATURES
        # ----------------------------------------------------

        print(
            "Scaling features..."
        )

        X_scaled = scaler.transform(
            X_imputed
        )


        # ----------------------------------------------------
        # MODEL PREDICTION
        # ----------------------------------------------------

        print(
            "Generating predictions..."
        )

        y_pred = model.predict(
            X_scaled
        )


        # ----------------------------------------------------
        # ACCURACY
        # ----------------------------------------------------

        accuracy = accuracy_score(

            y,

            y_pred

        )


        # ----------------------------------------------------
        # PRECISION
        # ----------------------------------------------------

        precision = precision_score(

            y,

            y_pred,

            zero_division=0

        )


        # ----------------------------------------------------
        # RECALL
        # ----------------------------------------------------

        recall = recall_score(

            y,

            y_pred,

            zero_division=0

        )


        # ----------------------------------------------------
        # F1 SCORE
        # ----------------------------------------------------

        f1 = f1_score(

            y,

            y_pred,

            zero_division=0

        )


        # ----------------------------------------------------
        # CONFUSION MATRIX
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
        # PRINT RESULTS
        # ----------------------------------------------------

        print()
        print("MODEL PERFORMANCE")
        print("-" * 40)

        print(
            "Accuracy:",
            round(
                accuracy * 100,
                2
            ),
            "%"
        )

        print(
            "Precision:",
            round(
                precision * 100,
                2
            ),
            "%"
        )

        print(
            "Recall:",
            round(
                recall * 100,
                2
            ),
            "%"
        )

        print(
            "F1 Score:",
            round(
                f1 * 100,
                2
            ),
            "%"
        )

        print("-" * 40)

        print(
            "True Negative:",
            tn
        )

        print(
            "False Positive:",
            fp
        )

        print(
            "False Negative:",
            fn
        )

        print(
            "True Positive:",
            tp
        )

        print("=" * 60)


        # ----------------------------------------------------
        # SEND DATA TO HTML
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
        print("=" * 60)
        print("PERFORMANCE ERROR")
        print("=" * 60)

        print(
            str(e)
        )

        print("=" * 60)


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
# ABOUT PAGE
# ============================================================

@app.route("/about")
def about():

    return render_template(
        "about.html"
    )


# ============================================================
# HISTORY PAGE
# ============================================================

@app.route("/history")
def history():

    return render_template(
        "history.html"
    )


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 60)
    print("       DIABETES PREDICTION SYSTEM")
    print("=" * 60)

    print()
    print("Project directory:")
    print(BASE_DIR)

    print()
    print("Dataset:")
    print(DATASET_PATH)

    print()
    print("Model directory:")
    print(MODEL_DIR)

    print()
    print("Model loaded:")
    print(MODEL_LOADED)

    print()
    print("=" * 60)

    app.run(
        debug=True,
        threaded=False
    )