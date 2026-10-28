# Campus Placement Prediction System Using Machine Learning

## Project Overview
The Campus Placement Prediction System is a machine learning web application designed to estimate whether a student may be placed based on academic performance and skill-related inputs.

## Features
- Student placement prediction through a web form.
- Uses CGPA, backlogs, internships, projects, aptitude score, and communication score.
- Compares four machine learning algorithms.
- Displays model performance metrics.
- Provides a user-friendly web interface.
- Includes a warning that predictions are estimates, not guarantees.

## Technologies Used
- Python
- Flask
- HTML, CSS, and JavaScript
- Pandas
- Scikit-learn
- Joblib

## Project Structure
```text
campus-placement-prediction-system/
├── app.py
├── train_model.py
├── requirements.txt
├── dataset/
├── models/
├── static/
├── templates/
└── README.md
```

## How to Run

### 1. Clone the repository
```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd campus-placement-prediction-system
```

### 2. Create and activate a virtual environment
On Windows PowerShell:
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Train the model if required
```bash
python train_model.py
```

### 5. Start the application
```bash
python app.py
```

Open `http://127.0.0.1:5000` in your browser.

## Important Note
The current dataset contains synthetic/sample data. The reported evaluation metrics are demonstration results and do not establish real-world placement accuracy. Predictions should not be treated as guarantees of employment.

## Author
Kalluru Sai Varshith