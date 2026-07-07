import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor

FEATURE_COLUMNS = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
]
TARGET_COLUMN = "quality"
MODEL_PATH = Path(__file__).with_name("wine_quality_model.pkl")
DATA_PATH = Path(__file__).with_name("wine.csv")


def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


def load_or_train_model():
    if MODEL_PATH.exists():
        with MODEL_PATH.open("rb") as model_file:
            return pickle.load(model_file)

    data = load_data()
    X = data[FEATURE_COLUMNS]
    y = data[TARGET_COLUMN]

    model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X, y)

    with MODEL_PATH.open("wb") as model_file:
        pickle.dump(model, model_file)

    return model


def predict_quality(model, values: dict) -> int:
    feature_vector = np.array([[values[column] for column in FEATURE_COLUMNS]], dtype=float)
    raw_prediction = model.predict(feature_vector)[0]
    quality = int(round(float(raw_prediction)))
    return max(3, min(9, quality))


def main() -> None:
    st.title("Wine Quality Checker")
    st.write(
        "This app is designed to check the quality of a given wine. "
        "The user can input the wine features and receive a predicted quality score."
    )

    values = {
        "fixed acidity": st.number_input("Enter the fixed acidity of the wine:", min_value=0.0, max_value=20.0, step=0.1),
        "volatile acidity": st.number_input("Enter the volatile acidity of the wine:", min_value=0.0, max_value=5.0, step=0.1),
        "citric acid": st.number_input("Enter the citric acidity of the wine:", min_value=0.0, max_value=1.0, step=0.1),
        "residual sugar": st.number_input("Enter the residual sugar value for this wine:", min_value=0.0, max_value=15.0, step=0.1),
        "chlorides": st.number_input("Enter the chlorides value for this wine:", min_value=0.0, max_value=1.0, step=0.01),
        "free sulfur dioxide": st.number_input("Enter the free sulfur dioxide value for this wine:", min_value=0.0, max_value=72.0, step=1.0),
        "total sulfur dioxide": st.number_input("Enter the total sulfur dioxide value for this wine:", min_value=0.0, max_value=289.0, step=1.0),
        "density": st.number_input("Enter the density value for this wine:", min_value=0.0, max_value=1.5, step=0.001),
        "pH": st.number_input("Enter the pH value for this wine:", min_value=-5.0, max_value=10.0, step=0.01),
        "sulphates": st.number_input("Enter the sulphates value for this wine:", min_value=0.0, max_value=2.0, step=0.01),
        "alcohol": st.number_input("Enter the alcohol value for this wine:", min_value=0.0, max_value=15.0, step=0.1),
    }

    if st.button("Predict Quality"):
        model = load_or_train_model()
        prediction = predict_quality(model, values)
        st.write(f"The predicted quality of the wine is: {prediction}")


if __name__ == "__main__":
    main()
