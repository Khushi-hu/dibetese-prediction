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

app = Flask(__name__)


# =========================================================
# BASE PROJECT DIRECTORY
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# =========================================================
# MODEL FILES
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
# LOAD MODEL
# =========================================================

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
imputer = joblib.load(IMPUTER_PATH)


# =========================================================
# FIND DATASET
# =========================================================

def find_dataset():

    possible_paths = [

        # Same folder as app.py
        os.path.join(
            BASE_DIR,
            "diabetes.csv"
        ),

        # archive (10) folder
        os.path.join(
            os.path.dirname(BASE_DIR),
            "diabetes.csv"
        ),

        # Parent folder
        os.path.join(
            os.path.dirname(
                os.path.dirname(BASE_DIR)
            ),
            "diabetes.csv"
        ),

        # Downloads
        os.path.join(
            os.path.expanduser("~"),
            "Downloads",
            "diabetes.csv"
        )

    ]


    # Check possible locations

    for path in possible_paths:

        if os.path.exists(path):

            return path


    # If not found, search project folder recursively

    for root, dirs, files in os.walk(BASE_DIR):

        if "diabetes.csv" in files:

            return os.path.join(
                root,
                "diabetes.csv"
            )


    return None


# =========================================================
# HOME
# =========================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# =========================================================
# ABOUT
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


        input_data = imputer.transform(
            input_data
        )


        input_data = scaler.transform(
            input_data
        )


        prediction = model.predict(
            input_data
        )[0]


        if prediction == 1:

            result = "Higher Risk of Diabetes"

            status = "risk"

        else:

            result = "Lower Risk of Diabetes"

            status = "safe"


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
# MODEL PERFORMANCE
# =========================================================

@app.route("/performance")
def performance():

    try:

        # Find dataset

        dataset_path = find_dataset()


        # Dataset not found

        if dataset_path is None:

            return f"""
            <h2>Dataset not found</h2>

            <p>
            The application searched for
            <b>diabetes.csv</b> inside:
            </p>

            <p>
            {BASE_DIR}
            </p>

            <p>
            Please locate your diabetes.csv file
            and place it inside the project folder.
            </p>
            """


        print(
            "Dataset found at:",
            dataset_path
        )


        # Load dataset

        df = pd.read_csv(
            dataset_path
        )


        # Check target

        if "Outcome" not in df.columns:

            return """
            <h2>Dataset Error</h2>

            <p>
            The dataset must contain a column
            named <b>Outcome</b>.
            </p>
            """


        # Features

        X = df.drop(
            "Outcome",
            axis=1
        )


        # Target

        y = df["Outcome"]


        # Train test split

        X_train, X_test, y_train, y_test = train_test_split(

            X,

            y,

            test_size=0.2,

            random_state=42,

            stratify=y

        )


        # Preprocessing

        X_test = imputer.transform(
            X_test
        )

        X_test = scaler.transform(
            X_test
        )


        # Prediction

        y_pred = model.predict(
            X_test
        )


        # Metrics

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


        # Confusion matrix

        cm = confusion_matrix(
            y_test,
            y_pred
        )


        if cm.shape == (2, 2):

            tn, fp, fn, tp = cm.ravel()

        else:

            tn = fp = fn = tp = 0


        # Model name

        model_name = type(
            model
        ).__name__


        # Render page

        return render_template(

            "model_performance.html",

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

        return f"""
        <h2>
        Error while calculating model performance
        </h2>

        <p>
        {e}
        </p>
        """


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )