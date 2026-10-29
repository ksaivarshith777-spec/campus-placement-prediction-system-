from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

FEATURE_COLUMNS = [
    "cgpa",
    "backlogs",
    "internships",
    "projects",
    "aptitude_score",
    "communication_score",
]

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "placement_model.joblib"
DATASET_PATH = BASE_DIR / "dataset" / "student_placement_sample.csv"
MODEL_COMPARISON_PATH = BASE_DIR / "dataset" / "model_comparison.csv"


def load_model_bundle():
    """Load the demo model when present and return a safe fallback otherwise."""
    default = {
        "model": None,
        "features": FEATURE_COLUMNS,
        "model_name": "Demo model unavailable",
        "data_type": "synthetic demonstration data",
        "available": False,
        "message": "No trained model file was found. Run 'python train_model.py' to generate the demo model.",
    }

    if not MODEL_PATH.exists():
        return default

    try:
        saved_model = joblib.load(MODEL_PATH)
    except Exception as exc:  # pragma: no cover - defensive fallback for missing/corrupt files
        default["message"] = f"The trained model could not be opened: {exc}"
        return default

    if not isinstance(saved_model, dict) or "model" not in saved_model or "features" not in saved_model:
        default["message"] = "The saved model file is not in the expected format."
        return default

    features = saved_model["features"]
    if list(features) != FEATURE_COLUMNS:
        default["message"] = (
            "The saved model feature order does not match the prediction form. "
            "Expected: " + ", ".join(FEATURE_COLUMNS)
        )
        return default

    saved_model["available"] = True
    saved_model["message"] = "Demo model loaded successfully from the project folder. Results are synthetic and for demonstration only."
    return saved_model


MODEL_BUNDLE = load_model_bundle()
model = MODEL_BUNDLE.get("model")
features = MODEL_BUNDLE.get("features", FEATURE_COLUMNS)
model_name = MODEL_BUNDLE.get("model_name", "Demo model unavailable")
data_type = MODEL_BUNDLE.get("data_type", "synthetic demonstration data")
model_message = MODEL_BUNDLE.get("message", "No model available.")


def get_dataset_stats():
    """Read the current CSV and return summary statistics when available."""
    if not DATASET_PATH.exists():
        return {
            "student_count": None,
            "placed_count": None,
            "placement_rate": None,
            "average_cgpa": None,
            "average_aptitude": None,
            "average_communication": None,
            "data_label": data_type,
            "available": False,
            "message": "Dataset statistics are unavailable in the current project folder.",
        }

    data = pd.read_csv(DATASET_PATH)
    if "placed" in data.columns:
        placed_count = int(data["placed"].sum())
        student_count = int(len(data))
        placement_rate = round((placed_count / student_count) * 100, 2) if student_count else 0
        stats = {
            "student_count": student_count,
            "placed_count": placed_count,
            "placement_rate": placement_rate,
            "average_cgpa": round(float(data["cgpa"].mean()), 2),
            "average_aptitude": round(float(data["aptitude_score"].mean()), 2),
            "average_communication": round(float(data["communication_score"].mean()), 2),
            "data_label": data_type,
            "available": True,
            "message": "Dataset statistics are based on the project CSV and should be treated as demonstration data unless validated with real placement records.",
        }
    else:
        stats = {
            "student_count": None,
            "placed_count": None,
            "placement_rate": None,
            "average_cgpa": None,
            "average_aptitude": None,
            "average_communication": None,
            "data_label": data_type,
            "available": False,
            "message": "No placement label column was found in the CSV, so live summary statistics are unavailable.",
        }
    return stats


def get_model_performance():
    """Load the recorded comparison metrics from the CSV when present."""
    if not MODEL_COMPARISON_PATH.exists():
        return {
            "available": False,
            "message": "No model comparison file was found. Demo evaluation results will be displayed as unavailable.",
            "metrics": [],
        }

    performance = pd.read_csv(MODEL_COMPARISON_PATH)
    metrics = performance.to_dict(orient="records")
    return {
        "available": bool(metrics),
        "message": "Performance values are sourced from the project evaluation CSV and represent synthetic demonstration results only.",
        "metrics": metrics,
    }


@app.route("/")
def home():
    return render_template(
        "index.html",
        model_name=model_name,
        model_status=model_message,
        dashboard=get_dataset_stats(),
        model_performance=get_model_performance(),
    )


@app.route("/predict", methods=["POST"])
def predict():
    try:
        if model is None:
            raise ValueError("No trained model is available. Run 'python train_model.py' to generate the synthetic demo model before predicting.")

        missing_fields = [field for field in FEATURE_COLUMNS if field not in request.form or request.form[field] == ""]
        if missing_fields:
            raise ValueError("Please complete all required fields before predicting placement.")

        cgpa = float(request.form["cgpa"])
        backlogs = int(request.form["backlogs"])
        internships = int(request.form["internships"])
        projects = int(request.form["projects"])
        aptitude_score = float(request.form["aptitude_score"])
        communication_score = float(request.form["communication_score"])

        if not 0 <= cgpa <= 10:
            raise ValueError("CGPA must be between 0 and 10.")

        if backlogs < 0 or internships < 0 or projects < 0:
            raise ValueError("Backlogs, internships, and projects cannot be negative.")

        if not 0 <= aptitude_score <= 100:
            raise ValueError("Aptitude score must be between 0 and 100.")

        if not 0 <= communication_score <= 100:
            raise ValueError("Communication score must be between 0 and 100.")

        student_data = pd.DataFrame([
            {
                "cgpa": cgpa,
                "backlogs": backlogs,
                "internships": internships,
                "projects": projects,
                "aptitude_score": aptitude_score,
                "communication_score": communication_score,
            }
        ], columns=FEATURE_COLUMNS)

        prediction = int(model.predict(student_data)[0])

        if hasattr(model, "predict_proba"):
            probability = round(float(model.predict_proba(student_data)[0][1]) * 100, 2)
        else:
            probability = None

        return render_template(
            "index.html",
            model_name=model_name,
            model_status=model_message,
            dashboard=get_dataset_stats(),
            model_performance=get_model_performance(),
            prediction=prediction,
            probability=probability,
        )

    except (TypeError, ValueError, KeyError) as error:
        return render_template(
            "index.html",
            model_name=model_name,
            model_status=model_message,
            dashboard=get_dataset_stats(),
            model_performance=get_model_performance(),
            error=str(error),
        ), 400


if __name__ == "__main__":
    app.run(debug=True)
