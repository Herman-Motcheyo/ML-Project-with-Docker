import numpy as np
import pandas as pd
from app.model.load_model import load_model
from sklearn.datasets import load_wine

model = load_model()

class_names = load_wine().target_names

def predict_single(features: list) -> str:
    prediction = model.predict([features])[0]
    return class_names[prediction]  # retourne le nom de la classe

def predict_batch(df) -> pd.DataFrame:
    predictions = model.predict(df)
    predicted_labels = [class_names[pred] for pred in predictions]
    df["prediction"] = predicted_labels
    return df
