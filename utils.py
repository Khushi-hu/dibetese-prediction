import json
import pandas as pd


def load_config(config_path="config/config.json"):
    with open(config_path, "r") as file:
        config = json.load(file)

    return config


def load_dataset(dataset_path="dataset/diabetes.csv"):
    return pd.read_csv("C:/Users/HP/Downloads/archive (10)/diabetes_project/dataset/diabetes.csv")


def prepare_data(df, target_column, features):
    X = df[features]
    y = df[target_column]

    return X, y