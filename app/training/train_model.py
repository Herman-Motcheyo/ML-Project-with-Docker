import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import mlflow
from dotenv import load_dotenv

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score
)

from app.utils import load_params
from typing import Tuple, List


# Load environment variables from the .env file
load_dotenv()

# Securely set MLflow environment variables
os.environ["MLFLOW_TRACKING_URI"] = os.getenv("MLFLOW_TRACKING_URI", "")
os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME", "")
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD", "")

# Load paths from config
params = load_params()
DATA_PATH = params["DATA_PATH"]
MODEL_PATH = params["MODEL_PATH"]


def prepare_data() -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    Load and prepare the Wine dataset.

    Returns:
        X (pd.DataFrame): Features
        y (pd.Series): Labels
        class_labels (List[str]): List of target class names
    """
    data = load_wine()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name="target")
    class_labels = list(data.target_names)

    df = pd.concat([X, y], axis=1)
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    df.to_csv(DATA_PATH, index=False)

    return X, y, class_labels


def train_and_evaluate(X: pd.DataFrame, y: pd.Series, class_labels: List[str]) -> None:
    """
    Train a Gradient Boosting Classifier and log the model and metrics to MLflow.

    Args:
        X (pd.DataFrame): Feature dataset
        y (pd.Series): Target labels
        class_labels (List[str]): Target class names
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Initialize model
    model = GradientBoostingClassifier(random_state=42)

    # Set MLflow tracking and experiment
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment("Wine_Classifier_Experiment_with_docker")

    # Start MLflow run
    with mlflow.start_run():
        # Log model parameters
        mlflow.log_param("model_type", "GradientBoostingClassifier")
        mlflow.log_param("random_state", 42)

        # Train model
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)

        # Log accuracy
        accuracy = accuracy_score(y_test, y_pred)
        mlflow.log_metric("accuracy", accuracy)

        # Log full classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        for label, metrics in report.items():
            if isinstance(metrics, dict):
                for metric_name, value in metrics.items():
                    mlflow.log_metric(f"{label}_{metric_name}", value)

        # Generate and log confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(cm, display_labels=class_labels)
        disp.plot(cmap="Blues")
        plt.title("Confusion Matrix")
        plt.savefig("confusion_matrix.png")
        mlflow.log_artifact("confusion_matrix.png")
        #plt.show()

        # Save the trained model
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        joblib.dump(model, MODEL_PATH)

        # Log the model in MLflow
        mlflow.sklearn.log_model(model, artifact_path="model")

        print(f"Model saved to: {MODEL_PATH}")
        print("Model and metrics successfully logged to MLflow.")


def main() -> None:
    """Main execution function."""
    X, y, class_labels = prepare_data()
    train_and_evaluate(X, y, class_labels)


if __name__ == "__main__":
    main()
