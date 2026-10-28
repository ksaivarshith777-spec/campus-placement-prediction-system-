from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "placement_model.joblib"
DATASET_PATH = BASE_DIR / "dataset" / "student_placement_sample.csv"
MODEL_COMPARISON_PATH = BASE_DIR / "dataset" / "model_comparison.csv"

# Load the trained machine learning model once at startup.
saved_model = joblib.load(MODEL_PATH)
model = saved_model["model"]
features = saved_model["features"]
model_name = saved_model["model_name"]
data_type = saved_model.get("data_type", "Available dataset")


def get_dataset_stats():
    """Read the current CSV and return dashboard statistics when available."""
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
            "message": "No model comparison file was found. Results will be displayed as unavailable.",
            "metrics": [],
        }

    performance = pd.read_csv(MODEL_COMPARISON_PATH)
    metrics = performance.to_dict(orient="records")
    return {
        "available": bool(metrics),
        "message": "Performance values are sourced from the project evaluation CSV and may represent synthetic demonstration results.",
        "metrics": metrics,
    }


@app.route("/")
def home():
    return render_template(
        "index.html",
        model_name=model_name,
        dashboard=get_dataset_stats(),
        model_performance=get_model_performance(),
    )


@app.route("/predict", methods=["POST"])
def predict():
    try:
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
        ], columns=features)

        prediction = int(model.predict(student_data)[0])

        if hasattr(model, "predict_proba"):
            probability = round(float(model.predict_proba(student_data)[0][1]) * 100, 2)
        else:
            probability = None

        return render_template(
            "index.html",
            model_name=model_name,
            dashboard=get_dataset_stats(),
            model_performance=get_model_performance(),
            prediction=prediction,
            probability=probability,
        )

    except (TypeError, ValueError, KeyError) as error:
        return render_template(
            "index.html",
            model_name=model_name,
            dashboard=get_dataset_stats(),
            model_performance=get_model_performance(),
            error=str(error),
        ), 400


if __name__ == "__main__":
    app.run(debug=True)
