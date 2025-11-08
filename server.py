y_reg = None
y_clf = None
tfidf = None
import os
from flask import Flask, request, jsonify, render_template
import json
import logging
from pathlib import Path

import joblib
import pandas as pd
from rapidfuzz import process, fuzz

app = Flask(__name__, static_folder="static", template_folder="templates")

# Configure logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "Networks outage log with sol.xlsx"
MODELS_DIR = BASE_DIR / "models"
DURATION_MODEL_PATH = MODELS_DIR / "duration_model.joblib"
SOLUTION_MODEL_PATH = MODELS_DIR / "solution_model.joblib"
METADATA_PATH = MODELS_DIR / "model_metadata.json"

raw_df = None
duration_model = None
solution_model = None
model_metadata = {}
valid_causes = []


def load_dataset() -> pd.DataFrame:
    logging.info("Loading dataset from Excel file")
    df = pd.read_excel(DATA_PATH)
    df.columns = df.columns.str.strip().str.lower()

    duration_col = "duration" if "duration" in df.columns else next(
        (c for c in df.columns if "duration" in c or "days" in c), None
    )
    if duration_col is None:
        raise ValueError("Duration column not found in dataset.")

    df["duration"] = pd.to_numeric(df[duration_col], errors="coerce")

    cause_col = "actual_cause" if "actual_cause" in df.columns else next(
        (c for c in df.columns if "cause" in c), None
    )
    if cause_col is None:
        raise ValueError("Cause column not found in dataset.")

    if "solution" in df.columns:
        df["solution_text"] = df["solution"]
    elif "solutions" in df.columns:
        df["solution_text"] = df["solutions"]
    else:
        df["solution_text"] = "Restart core routers, check ISP peering and DNS."

    if "country" not in df.columns:
        df["country"] = "unknown"
    if "region" not in df.columns:
        df["region"] = "unknown"

    df["actual_cause"] = df[cause_col].fillna("unknown").astype(str)
    df["country"] = df["country"].fillna("unknown").astype(str)
    df["region"] = df["region"].fillna("unknown").astype(str)
    df["solution_text"] = df["solution_text"].fillna(
        "Restart core routers, check ISP peering and DNS."
    ).astype(str)

    df = df.dropna(subset=["duration"])
    df = df[df["duration"] > 0]

    df["actual_cause"] = df["actual_cause"].str.strip()
    df["country"] = df["country"].str.strip()
    df["region"] = df["region"].str.strip()

    return df[["actual_cause", "country", "region", "duration", "solution_text"]].reset_index(drop=True)


def load_models() -> bool:
    global raw_df, duration_model, solution_model, model_metadata, valid_causes

    try:
        raw_df = load_dataset()
        valid_causes = sorted(raw_df["actual_cause"].str.lower().unique().tolist())

        if not DURATION_MODEL_PATH.exists() or not SOLUTION_MODEL_PATH.exists():
            raise FileNotFoundError("Trained model files are missing. Run train_models.py first.")

        duration_model = joblib.load(DURATION_MODEL_PATH)
        solution_model = joblib.load(SOLUTION_MODEL_PATH)

        if METADATA_PATH.exists():
            model_metadata = json.loads(METADATA_PATH.read_text())
            if not valid_causes:
                valid_causes = model_metadata.get("valid_causes", [])

        logging.info("Models and dataset loaded successfully")
        return True
    except Exception as exc:
        logging.exception("Failed to load models")
        return False

def fuzzy_match(value, valid_values, cutoff=70):
    """Fuzzy corrects the cause string for typos."""
    if not value or not isinstance(value, str):
        return value
    value = value.strip().lower()
    best = process.extractOne(value, valid_values, scorer=fuzz.WRatio)
    if best and best[1] >= cutoff:
        return best[0]
    else:
        logging.warning(f"No strong fuzzy match found for '{value}', using as-is.")
        return value

def normalise_text(value: str) -> str:
    return (value or "unknown").strip().lower() or "unknown"


def normalise_severity(value: str) -> str:
    cleaned = (value or "").strip().lower()
    allowed = {"low", "medium", "high", "critical"}
    return cleaned if cleaned in allowed else "medium"


def derive_severity_label(duration_value: float) -> str:
    if pd.isna(duration_value):
        return "medium"
    if duration_value <= 1:
        return "low"
    if duration_value <= 3:
        return "medium"
    if duration_value <= 7:
        return "high"
    return "critical"


def fuzzy_match(value: str, valid_values, cutoff: int = 70) -> str:
    if not value:
        return "unknown"
    candidate = value.strip().lower()
    if not valid_values:
        return candidate
    best_match = process.extractOne(candidate, valid_values, scorer=fuzz.WRatio)
    if best_match and best_match[1] >= cutoff:
        logging.info("Matched '%s' to '%s' (score=%s)", candidate, best_match[0], best_match[1])
        return best_match[0]
    logging.warning("No strong fuzzy match found for '%s', using as-is", candidate)
    return candidate


def prepare_input_row(cause: str, country: str, region: str, severity: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "cause_input": normalise_text(cause),
                "country_input": normalise_text(country),
                "region_input": normalise_text(region),
                "severity_input": normalise_severity(severity),
            }
        ]
    )


def filter_dataset(matched_cause: str, country: str = None, region: str = None) -> pd.DataFrame:
    if raw_df is None or raw_df.empty:
        return pd.DataFrame()

    subset = raw_df.copy()
    subset["actual_cause_lower"] = subset["actual_cause"].str.lower()

    cause_mask = subset["actual_cause_lower"].str.contains(matched_cause, case=False, na=False)
    if not cause_mask.any():
        for token in matched_cause.split():
            if len(token) > 3:
                token_mask = subset["actual_cause_lower"].str.contains(token, case=False, na=False)
                if token_mask.any():
                    cause_mask = token_mask
                    break
    if cause_mask.any():
        subset = subset[cause_mask]
    else:
        return pd.DataFrame()

    if country:
        subset = subset[subset["country"].str.lower().str.contains(country.lower(), na=False)]
    if region:
        subset = subset[subset["region"].str.lower().str.contains(region.lower(), na=False)]

    return subset.drop(columns=["actual_cause_lower"], errors="ignore")


def generate_solution_from_cause(cause: str, region: str = None) -> str:
    cause_lower = (cause or "").lower()
    area = region if region else "the affected area"

    if "fiber" in cause_lower or "cable" in cause_lower:
        return f"Repair the damaged fiber cable in {area} and reroute traffic via backup paths."
    if "power" in cause_lower or "electricity" in cause_lower:
        return f"Restore power sources (generators/UPS) and restart core network nodes."
    if "ddos" in cause_lower or "attack" in cause_lower:
        return f"Block attack traffic at the edge and enable DDoS mitigation services."
    if "cyber" in cause_lower or "hack" in cause_lower:
        return f"Isolate affected servers, block malicious IPs, and restore from clean backups."
    if "maintenance" in cause_lower:
        return f"Complete maintenance procedures and run verification tests."
    if "router" in cause_lower or "hardware" in cause_lower:
        return f"Reboot or replace the faulty equipment and verify configurations."
    if "government" in cause_lower or "legal" in cause_lower or "court" in cause_lower:
        return f"Coordinate with legal teams and restore services when authorized."
    if "protest" in cause_lower or "violence" in cause_lower:
        return f"Apply targeted access controls and keep emergency communications open."
    if "subsea" in cause_lower or "submarine" in cause_lower:
        return f"Coordinate with cable operator for subsea repair and route via backup cables."
    if "isp" in cause_lower or "peering" in cause_lower:
        return f"Contact ISPs to fix BGP/peering issues and restore routing."
    return f"Restart core routers in {area}, check ISP peering and DNS, and dispatch engineers if needed."


def predict_outcome(
    user_cause: str,
    user_country: str,
    user_region: str,
    user_severity: str,
) -> dict:
    matched_cause = fuzzy_match(user_cause, valid_causes)
    model_input = prepare_input_row(matched_cause, user_country, user_region, user_severity)

    predicted_duration = float(duration_model.predict(model_input)[0])
    predicted_duration = max(predicted_duration, 0.5)

    predicted_solution = solution_model.predict(model_input)[0]
    if isinstance(predicted_solution, (list, tuple)):
        predicted_solution = predicted_solution[0]
    predicted_solution = str(predicted_solution).strip()

    dataset_duration = None
    reference_count = 0
    dataset_slice = filter_dataset(matched_cause, user_country, user_region)
    if not dataset_slice.empty:
        dataset_duration = float(dataset_slice["duration"].median())
        predicted_duration = (predicted_duration + dataset_duration) / 2.0
        top_solution = dataset_slice["solution_text"].mode()
        if not top_solution.empty:
            predicted_solution = top_solution.iloc[0]
        reference_count = int(dataset_slice.shape[0])

    if not predicted_solution:
        predicted_solution = generate_solution_from_cause(matched_cause, user_region)

    severity_label = (user_severity or derive_severity_label(predicted_duration) or "medium").lower()
    context_area = user_region or user_country or "the affected area"

    progress_lookup = {"low": 32, "medium": 52, "high": 72, "critical": 88}
    timeline_progress = progress_lookup.get(severity_label, 52)

    base_steps = [
        f"Mobilise field operations for {context_area} and share the incident brief with the NOC.",
        "Restore primary links, validate traffic health, and keep customer comms flowing.",
        "Run post-incident review, document lessons, and schedule hardening tasks.",
    ]

    if severity_label in {"high", "critical"}:
        base_steps[0] = f"Activate crisis bridge and dispatch senior responders to {context_area} within 30 minutes."
        base_steps[1] = "Coordinate multi-region reroutes, verify latency, and confirm regulatory updates."
        base_steps[2] = "Publish executive summary, capture RCA, and update resilience playbooks."
    elif severity_label == "low":
        base_steps[0] = f"Notify on-call engineer covering {context_area} and collect supporting telemetry."
        base_steps[1] = "Roll out targeted fix, monitor recovery KPIs, and update stakeholders hourly."

    eta_days = max(1, int(round(predicted_duration)))
    eta_label = f"Target: {eta_days}d window"

    logging.info(
        "Prediction ready | cause=%s | duration=%.2f | severity=%s | references=%s",
        matched_cause,
        predicted_duration,
        severity_label,
        reference_count,
    )

    return {
        "duration_days": predicted_duration,
        "solution": predicted_solution,
        "severity": severity_label,
        "matched_cause": matched_cause,
        "reference_count": reference_count,
        "reference_median": dataset_duration,
        "timeline_progress": timeline_progress,
        "timeline_steps": base_steps,
        "eta_label": eta_label,
    }

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        logging.info("Received a request at /predict")

        # Retrieve form data
        cause = request.form.get('cause', '').strip()
        country = request.form.get('country', '').strip()
        region = request.form.get('region', '').strip()
        severity = request.form.get('severity', '').strip()
        
        logging.info(f"Form data - Cause: '{cause}', Country: '{country}', Region: '{region}', Severity: '{severity}'")

        # Validate required fields
        if not cause:
            logging.error("Cause is required but not provided")
            return jsonify({'error': 'Cause is required'}), 400

        if duration_model is None or solution_model is None:
            logging.error("Models not loaded")
            return jsonify({'error': 'Server not ready - models not loaded'}), 500
        if raw_df is None or raw_df.empty:
            logging.error("Dataset not loaded")
            return jsonify({'error': 'Server not ready - dataset not loaded'}), 500

        # Make predictions
        prediction = predict_outcome(
            user_cause=cause,
            user_country=country if country else None,
            user_region=region if region else None,
            user_severity=severity if severity else None
        )

        # Format the response
        response = {
            'repair_duration': f"{prediction['duration_days']:.1f} days",
            'recommended_solution': prediction['solution'],
            'severity_label': prediction['severity'],
            'matched_cause': prediction['matched_cause'],
            'reference_count': prediction['reference_count'],
            'timeline_progress': prediction['timeline_progress'],
            'timeline_steps': prediction['timeline_steps'],
            'eta_label': prediction['eta_label'],
        }
        if prediction.get('reference_median') is not None:
            response['reference_median'] = f"{prediction['reference_median']:.1f} days"
        
        logging.info(f"Sending response: {response}")
        
        # Return the predictions as JSON
        return jsonify(response), 200

    except Exception as e:
        logging.exception("An error occurred while processing the request")
        return jsonify({'error': f'Server error: {str(e)}'}), 500

if __name__ == '__main__':
    # Load models on startup
    if load_models():
        port = int(os.environ.get("PORT", 5000))
        debug_mode = os.environ.get("FLASK_DEBUG", "0") == "1"
        logging.info("Starting Flask server on port %s...", port)
        app.run(debug=debug_mode, host='0.0.0.0', port=port)
    else:
        logging.error("Failed to load models. Server not started.")