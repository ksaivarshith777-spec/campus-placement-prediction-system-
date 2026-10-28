from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


# Project folders
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
MODELS_DIR = BASE_DIR / "models"

DATASET_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

DATA_PATH = DATASET_DIR / "student_placement_sample.csv"
MODEL_PATH = MODELS_DIR / "placement_model.joblib"


def create_sample_dataset(n_samples=1000, random_state=42):
    """Create synthetic data for testing the project pipeline."""
    rng = np.random.default_rng(random_state)

    cgpa = rng.uniform(5.0, 10.0, n_samples).round(2)
    backlogs = rng.integers(0, 5, n_samples)
    internships = rng.integers(0, 3, n_samples)
    projects = rng.integers(0, 6, n_samples)
    aptitude = rng.integers(30, 101, n_samples)
    communication = rng.integers(30, 101, n_samples)

    # Synthetic rule for demonstration only—not real placement evidence.
    score = (
        0.9 * (cgpa - 5)
        - 0.7 * backlogs
        + 0.9 * internships
        + 0.25 * projects
        + 0.025 * (aptitude - 30)
        + 0.02 * (communication - 30)
        + rng.normal(0, 1.0, n_samples)
    )

    placed = (score >= 3.2).astype(int)

    data = pd.DataFrame({
        "cgpa": cgpa,
        "backlogs": backlogs,
        "internships": internships,
        "projects": projects,
        "aptitude_score": aptitude,
        "communication_score": communication,
        "placed": placed,
    })

    data.to_csv(DATA_PATH, index=False)
    print(f"Synthetic sample dataset saved to: {DATA_PATH}")
    return data


def main():
    data = create_sample_dataset()

    features = [
        "cgpa",
        "backlogs",
        "internships",
        "projects",
        "aptitude_score",
        "communication_score",
    ]

    X = data[features]
    y = data["placed"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=1000)),
        ]),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=6, random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, random_state=42
        ),
        "SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", SVC(probability=True, random_state=42)),
        ]),
    }

    results = []
    trained_models = {}

    for name, model in models.items():
        start_time = time.perf_counter()
        model.fit(X_train, y_train)
        training_seconds = time.perf_counter() - start_time

        predictions = model.predict(X_test)

        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, predictions),
            "Precision": precision_score(
                y_test, predictions, zero_division=0
            ),
            "Recall": recall_score(
                y_test, predictions, zero_division=0
            ),
            "F1-score": f1_score(
                y_test, predictions, zero_division=0
            ),
            "Training time (seconds)": round(training_seconds, 4),
        })

        trained_models[name] = model

        print(f"\n--- {name} ---")
        print(classification_report(
            y_test,
            predictions,
            target_names=["Not Placed", "Placed"],
            zero_division=0,
        ))

    results_df = pd.DataFrame(results).sort_values(
        by="F1-score", ascending=False
    )

    print("\nMODEL COMPARISON")
    print(results_df.to_string(index=False))

    best_name = results_df.iloc[0]["Model"]
    best_model = trained_models[best_name]

    joblib.dump({
        "model": best_model,
        "features": features,
        "model_name": best_name,
        "data_type": "synthetic demonstration data",
    }, MODEL_PATH)

    results_df.to_csv(DATASET_DIR / "model_comparison.csv", index=False)

    print(f"\nBest model for this synthetic test: {best_name}")
    print(f"Saved model to: {MODEL_PATH}")
    print("Reminder: these results do not establish real-world placement accuracy.")


if __name__ == "__main__":
    main()