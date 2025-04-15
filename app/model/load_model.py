import joblib

from app.utils import load_params

params = load_params()
MODEL_PATH = params["MODEL_PATH"]

def load_model():
    return joblib.load(MODEL_PATH)
