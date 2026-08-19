import sqlite3
import os


# =========================================================
# DATABASE PATH
# =========================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

DATABASE = os.path.join(
    BASE_DIR,
    "predictions.db"
)


# =========================================================
# CREATE DATABASE
# =========================================================

def create_database():

    connection = sqlite3.connect(
        DATABASE
    )

    cursor = connection.cursor()


    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            pregnancies REAL NOT NULL,

            glucose REAL NOT NULL,

            blood_pressure REAL NOT NULL,

            skin_thickness REAL NOT NULL,

            insulin REAL NOT NULL,

            bmi REAL NOT NULL,

            diabetes_pedigree REAL NOT NULL,

            age REAL NOT NULL,

            prediction TEXT NOT NULL,

            created_at TIMESTAMP
            DEFAULT CURRENT_TIMESTAMP

        )
    """)


    connection.commit()

    connection.close()


# =========================================================
# SAVE PREDICTION
# =========================================================

def save_prediction(

    pregnancies,

    glucose,

    blood_pressure,

    skin_thickness,

    insulin,

    bmi,

    diabetes_pedigree,

    age,

    prediction

):

    connection = sqlite3.connect(
        DATABASE
    )

    cursor = connection.cursor()


    cursor.execute("""

        INSERT INTO predictions (

            pregnancies,

            glucose,

            blood_pressure,

            skin_thickness,

            insulin,

            bmi,

            diabetes_pedigree,

            age,

            prediction

        )

        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)

    """, (

        pregnancies,

        glucose,

        blood_pressure,

        skin_thickness,

        insulin,

        bmi,

        diabetes_pedigree,

        age,

        prediction

    ))


    connection.commit()

    connection.close()


# =========================================================
# GET HISTORY
# =========================================================

def get_predictions():

    connection = sqlite3.connect(
        DATABASE
    )

    cursor = connection.cursor()


    cursor.execute("""

        SELECT

            id,

            pregnancies,

            glucose,

            blood_pressure,

            skin_thickness,

            insulin,

            bmi,

            diabetes_pedigree,

            age,

            prediction,

            created_at

        FROM predictions

        ORDER BY id DESC

    """)


    data = cursor.fetchall()


    connection.close()


    return data