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


def ensure_models_loaded() -> bool:
    global duration_model, solution_model
    if duration_model is not None and solution_model is not None and raw_df is not None:
        return True
    return load_models()


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
        logging.info("Dataset loaded with %s rows and columns: %s", raw_df.shape[0], list(raw_df.columns))
        valid_causes = sorted(raw_df["actual_cause"].str.lower().unique().tolist())

        if not DURATION_MODEL_PATH.exists() or not SOLUTION_MODEL_PATH.exists():
            raise FileNotFoundError("Trained model files are missing. Run train_models.py first.")

        duration_model = joblib.load(DURATION_MODEL_PATH)
        solution_model = joblib.load(SOLUTION_MODEL_PATH)
        logging.info(
            "Models loaded: duration=%s | solution=%s",
            duration_model.__class__.__name__,
            solution_model.__class__.__name__,
        )

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

    exact_mask = subset["actual_cause_lower"] == matched_cause.lower()
    if exact_mask.any():
        subset = subset[exact_mask]
    else:
        cause_mask = subset["actual_cause_lower"].str.contains(matched_cause, case=False, na=False)
        if cause_mask.any():
            subset = subset[cause_mask]
        else:
            tokens = [token for token in matched_cause.split() if len(token) > 3]
            for token in tokens:
                token_mask = subset["actual_cause_lower"].str.contains(token, case=False, na=False)
                if token_mask.any():
                    subset = subset[token_mask]
                    break

    if subset.empty:
        return subset.drop(columns=["actual_cause_lower"], errors="ignore")

    if country:
        subset = subset[subset["country"].str.lower().str.contains(country.lower(), na=False)]
    if region:
        subset = subset[subset["region"].str.lower().str.contains(region.lower(), na=False)]

    return subset.drop(columns=["actual_cause_lower"], errors="ignore")


def generate_solution_from_cause(cause: str, region: str = None) -> str:
    """
    Rule-based solution generation for network outages.
    Each cause type gets a unique, actionable solution.
    """
    cause_lower = (cause or "").lower().strip()
    area = region if region else "the affected area"
    
    # Fiber/Cable Related
    if any(keyword in cause_lower for keyword in ["fiber", "cable cut", "fibre", "optic"]):
        return f"Dispatch fiber repair crew to {area}. Locate and splice damaged cable segments. Reroute traffic via backup fiber paths and test optical signal strength."
    
    # Power/Electricity Related
    elif any(keyword in cause_lower for keyword in ["power", "electricity", "generator", "ups", "battery"]):
        return f"Activate backup generators in {area}. Restore primary power feeds. Systematically restart core routers, switches, and transmission equipment in correct sequence."
    
    # DDoS/Cyber Attack Related
    elif any(keyword in cause_lower for keyword in ["ddos", "dos attack", "cyber attack", "flooding"]):
        return f"Enable DDoS mitigation at edge routers. Implement rate limiting and traffic scrubbing. Block attack source IPs and reroute legitimate traffic through clean pipes."
    
    # Malware/Hacking Related
    elif any(keyword in cause_lower for keyword in ["malware", "virus", "hack", "breach", "intrusion", "ransomware"]):
        return f"Isolate infected systems immediately. Scan and remove malware. Restore from verified clean backups. Patch vulnerabilities and rotate all credentials."
    
    # Hardware Failure
    elif any(keyword in cause_lower for keyword in ["hardware", "router", "switch", "equipment", "device failure"]):
        return f"Replace faulty hardware components in {area}. Deploy spare equipment from inventory. Restore configurations from backup and verify routing protocols."
    
    # Software/Configuration Issues
    elif any(keyword in cause_lower for keyword in ["software", "configuration", "upgrade", "patch", "bug"]):
        return f"Roll back problematic software changes. Restore last known good configuration. Test in isolated environment before redeploying to production."
    
    # Maintenance Related
    elif any(keyword in cause_lower for keyword in ["maintenance", "planned", "scheduled", "upgrade work"]):
        return f"Complete scheduled maintenance activities in {area}. Perform post-maintenance verification tests. Gradually restore services and monitor performance metrics."
    
    # Government/Legal Issues
    elif any(keyword in cause_lower for keyword in ["government", "legal", "court", "order", "directive", "regulation", "shutdown", "visit"]):
        return f"Engage legal and compliance teams. Document official directives. Maintain emergency services access. Prepare phased restoration plan pending authorization."
    
    # Civil Unrest/Violence
    elif any(keyword in cause_lower for keyword in ["protest", "riot", "violence", "unrest", "vandalism", "civil"]):
        return f"Coordinate with security authorities for {area}. Implement geo-fenced access controls. Protect critical infrastructure. Maintain emergency communication channels."
    
    # Subsea Cable Issues
    elif any(keyword in cause_lower for keyword in ["subsea", "submarine", "undersea", "sea cable"]):
        return f"Alert submarine cable consortium. Dispatch cable repair vessel with ROV equipment. Reroute international traffic via alternate cable systems. Estimate repair timeline based on fault location."
    
    # ISP/Peering Issues
    elif any(keyword in cause_lower for keyword in ["isp", "peering", "bgp", "routing", "upstream", "transit"]):
        return f"Contact upstream ISPs to resolve routing issues. Verify BGP peering sessions. Correct route advertisements. Clear dampened routes and monitor traffic flow normalization."
    
    # Weather Related
    elif any(keyword in cause_lower for keyword in ["weather", "storm", "flood", "lightning", "rain", "wind", "cyclone", "hurricane"]):
        return f"Assess weather damage to infrastructure in {area}. Deploy emergency repair teams. Protect equipment from water ingress. Restore power and connectivity to affected sites."
    
    # Fire Related
    elif any(keyword in cause_lower for keyword in ["fire", "burn", "smoke", "heat"]):
        return f"Coordinate with fire services. Evacuate personnel from {area}. Assess fire damage to network equipment. Replace burned components and restore from offsite backups."
    
    # Transmission Issues
    elif any(keyword in cause_lower for keyword in ["transmission", "backhaul", "microwave", "radio"]):
        return f"Inspect transmission links in {area}. Realign microwave dishes. Check radio frequencies for interference. Restore backup transmission paths."
    
    # DNS/Server Issues
    elif any(keyword in cause_lower for keyword in ["dns", "server", "database", "application"]):
        return f"Restart DNS/application servers. Verify database connectivity. Clear cache and test name resolution. Monitor query response times and error rates."
    
    # Capacity/Overload Issues
    elif any(keyword in cause_lower for keyword in ["capacity", "overload", "congestion", "traffic", "bandwidth"]):
        return f"Implement traffic shaping and QoS policies. Add capacity via additional circuits. Load balance across available paths. Identify and throttle heavy users if needed."
    
    # Construction/Excavation
    elif any(keyword in cause_lower for keyword in ["construction", "excavation", "dig", "road work", "drilling"]):
        return f"Coordinate with construction crews in {area}. Locate and repair damaged underground cables. Update cable route documentation. Install additional protection measures."
    
    # Animal/Environmental
    elif any(keyword in cause_lower for keyword in ["animal", "rodent", "bird", "nest", "tree"]):
        return f"Remove animal interference from equipment in {area}. Seal cable entry points. Trim vegetation near antennas. Install protective barriers."
    
    # Access/Physical Security
    elif any(keyword in cause_lower for keyword in ["access", "theft", "vandal", "security", "trespass", "sabotage"]):
        return f"Secure physical access to network sites in {area}. Replace stolen equipment. Enhance perimeter security. File police reports and insurance claims."
    
    # Air/Environmental  
    elif any(keyword in cause_lower for keyword in ["air", "temperature", "cooling", "hvac", "overheat"]):
        return f"Restore HVAC/cooling systems in {area}. Monitor equipment temperatures. Add temporary cooling if needed. Ensure proper ventilation."
    
    # Default fallback
    else:
        return f"Investigate root cause of outage in {area}. Deploy field engineers for on-site diagnostics. Implement appropriate corrective measures based on findings. Monitor restoration progress."


def build_recovery_steps(cause: str, severity: str, area: str, solution: str) -> list[str]:
    """
    Rule-based recovery timeline generation.
    Each cause type gets unique 3-step recovery process.
    """
    area_display = area or "the affected area"
    cause_lower = (cause or "").lower().strip()
    steps = []

    # Fiber/Cable Related
    if any(keyword in cause_lower for keyword in ["fiber", "cable cut", "fibre", "optic"]):
        steps = [
            f"Dispatch field fiber crew to {area_display} and initiate optical span testing.",
            "Splice or replace the damaged segment, then restore traffic via protected rings.",
            "Validate signal levels, update topology maps, and brief stakeholders on resolution.",
        ]
    
    # Power/Electricity Related
    elif any(keyword in cause_lower for keyword in ["power", "electric", "generator", "ups", "battery"]):
        steps = [
            f"Coordinate with facilities to restore stable power feeds across {area_display}.",
            "Bring core routers and aggregation layers back online with clean shutdown/startup checks.",
            "Audit UPS/generator runtime logs and schedule preventative maintenance follow-up.",
        ]
    
    # DDoS Attack Related
    elif any(keyword in cause_lower for keyword in ["ddos", "dos attack", "flooding"]):
        steps = [
            "Enable upstream DDoS scrubbing and block hostile prefixes at the edge.",
            "Rate-limit affected services, reroute trusted traffic, and monitor packet loss.",
            "File incident report, refresh mitigation playbooks, and coordinate with security ops.",
        ]
    
    # Malware/Hacking Related
    elif any(keyword in cause_lower for keyword in ["malware", "virus", "hack", "breach", "cyber", "intrusion", "ransomware"]):
        steps = [
            "Isolate compromised hosts, rotate credentials, and enable forensic data capture.",
            "Restore clean images, verify integrity checksums, and rejoin critical services cautiously.",
            "Coordinate post-incident hardening and communicate remediation progress.",
        ]
    
    # Government/Legal Issues
    elif any(keyword in cause_lower for keyword in ["government", "legal", "court", "order", "directive", "visit", "shutdown"]):
        steps = [
            "Engage legal/compliance teams to clarify scope of imposed restrictions.",
            "Maintain emergency services routing while preparing staged service restoration.",
            "Document timeline, notify impacted partners, and track regulatory follow-ups.",
        ]
    
    # Civil Unrest/Violence
    elif any(keyword in cause_lower for keyword in ["protest", "violence", "riot", "civil", "unrest", "vandalism"]):
        steps = [
            "Activate crisis response channel and establish secure comms with field teams.",
            "Implement geo-fenced controls while sustaining emergency connectivity.",
            "Debrief authorities, reassess risk zones, and plan staged service normalization.",
        ]
    
    # Subsea Cable Issues
    elif any(keyword in cause_lower for keyword in ["subsea", "submarine", "undersea", "sea cable"]):
        steps = [
            "Notify cable consortium, dispatch repair vessel, and reroute traffic via alternate paths.",
            "Monitor latency impacts, adjust transit agreements, and keep stakeholders informed.",
            "Review redundancy posture, update capacity models, and log lessons learned.",
        ]
    
    # ISP/Peering Issues
    elif any(keyword in cause_lower for keyword in ["isp", "peering", "bgp", "routing", "upstream", "transit"]):
        steps = [
            "Engage upstream peers to resolve routing anomalies and validate session health.",
            "Propagate corrected route policies, clear dampened prefixes, and confirm traffic stability.",
            "Capture BGP incident report and refine peering automation safeguards.",
        ]
    
    # Maintenance Related
    elif any(keyword in cause_lower for keyword in ["maintenance", "planned", "scheduled", "upgrade work"]):
        steps = [
            "Confirm maintenance window approvals and impacted circuits.",
            "Execute rollback or remediation tasks and validate service KPIs.",
            "Publish completion notice, capture lessons, and adjust future change plans.",
        ]
    
    # Hardware Failure
    elif any(keyword in cause_lower for keyword in ["hardware", "router", "switch", "equipment", "device"]):
        steps = [
            "Swap or reseat failed gear, applying golden configuration from source control.",
            "Perform end-to-end connectivity tests and re-enable routing adjacencies.",
            "Schedule root-cause review, log spares usage, and update asset database.",
        ]
    
    # Software/Configuration Issues
    elif any(keyword in cause_lower for keyword in ["software", "configuration", "upgrade", "patch", "bug"]):
        steps = [
            "Roll back problematic changes and restore last known good configuration.",
            "Test in isolated lab environment before redeploying to production.",
            "Document lessons learned and update change management procedures.",
        ]
    
    # Weather Related
    elif any(keyword in cause_lower for keyword in ["weather", "storm", "flood", "lightning", "rain", "wind", "cyclone"]):
        steps = [
            f"Assess weather damage to infrastructure in {area_display} and deploy emergency teams.",
            "Protect equipment from water ingress, restore power, and repair damaged sites.",
            "Monitor weather forecasts and prepare contingency plans for ongoing conditions.",
        ]
    
    # Fire Related
    elif any(keyword in cause_lower for keyword in ["fire", "burn", "smoke", "heat"]):
        steps = [
            f"Coordinate with fire services and evacuate personnel from {area_display}.",
            "Assess fire damage, replace burned equipment, and restore from offsite backups.",
            "Review fire safety systems and implement enhanced protection measures.",
        ]
    
    # Transmission Issues
    elif any(keyword in cause_lower for keyword in ["transmission", "backhaul", "microwave", "radio"]):
        steps = [
            f"Inspect transmission links in {area_display} and realign microwave dishes.",
            "Check radio frequencies for interference and restore backup paths.",
            "Validate signal quality and update link budget calculations.",
        ]
    
    # DNS/Server Issues
    elif any(keyword in cause_lower for keyword in ["dns", "server", "database", "application"]):
        steps = [
            "Restart DNS/application servers and verify database connectivity.",
            "Clear cache, test name resolution, and monitor query response times.",
            "Review server logs for errors and implement performance tuning.",
        ]
    
    # Capacity/Overload Issues
    elif any(keyword in cause_lower for keyword in ["capacity", "overload", "congestion", "traffic", "bandwidth"]):
        steps = [
            "Implement traffic shaping and QoS policies to manage congestion.",
            "Add capacity via additional circuits and load balance across paths.",
            "Analyze traffic patterns and plan for long-term capacity upgrades.",
        ]
    
    # Construction/Excavation
    elif any(keyword in cause_lower for keyword in ["construction", "excavation", "dig", "road work", "drilling"]):
        steps = [
            f"Coordinate with construction crews in {area_display} and locate damaged cables.",
            "Repair underground infrastructure and update cable route documentation.",
            "Install additional protection and warning markers at site.",
        ]
    
    # Animal/Environmental
    elif any(keyword in cause_lower for keyword in ["animal", "rodent", "bird", "nest", "tree"]):
        steps = [
            f"Remove animal interference from equipment in {area_display}.",
            "Seal cable entry points and trim vegetation near antennas.",
            "Install protective barriers and schedule regular site inspections.",
        ]
    
    # Access/Physical Security
    elif any(keyword in cause_lower for keyword in ["access", "theft", "vandal", "security", "trespass", "sabotage"]):
        steps = [
            f"Secure physical access to network sites in {area_display}.",
            "Replace stolen equipment and enhance perimeter security measures.",
            "File police reports, review security footage, and update access controls.",
        ]
    
    # Air/Cooling Issues
    elif any(keyword in cause_lower for keyword in ["air", "temperature", "cooling", "hvac", "overheat"]):
        steps = [
            f"Restore HVAC/cooling systems in {area_display} immediately.",
            "Monitor equipment temperatures and add temporary cooling if needed.",
            "Audit cooling capacity and plan for environmental upgrades.",
        ]
    
    # Default fallback
    else:
        steps = [
            f"Mobilise field operations for {area_display} and gather telemetry for triage.",
            "Apply recommended fix, monitor recovery KPIs, and keep customer comms active.",
            "Document root cause, update runbooks, and schedule resilience follow-ups.",
        ]

    # Adjust for severity
    if severity in {"high", "critical"}:
        steps[0] = f"🚨 URGENT: Activate war-room bridge and dispatch senior responders to {area_display} immediately."
        steps[2] += " Escalate progress updates to leadership every 30 minutes."

    return steps


def predict_outcome(
    user_cause: str,
    user_country: str,
    user_region: str,
    user_severity: str,
) -> dict:
    """
    Pure rule-based prediction system.
    Always generates unique solutions based on cause keywords.
    Uses dataset only for duration estimation and reference count.
    """
    # Step 1: Match cause to known causes
    matched_cause = fuzzy_match(user_cause, valid_causes)
    logging.info("User cause '%s' matched to '%s'", user_cause, matched_cause)
    
    # Step 2: Get historical data from dataset for duration estimation
    dataset_duration = None
    reference_count = 0
    context_area = user_region or user_country or "the affected area"
    
    dataset_slice = filter_dataset(matched_cause, user_country, user_region)
    if not dataset_slice.empty:
        dataset_duration = float(dataset_slice["duration"].median())
        reference_count = int(dataset_slice.shape[0])
        logging.info("Found %d historical cases in dataset with median duration %.2f days", 
                    reference_count, dataset_duration)
    
    # Step 3: Use ML model for duration prediction (but can be overridden by dataset)
    model_input = prepare_input_row(matched_cause, user_country, user_region, user_severity)
    predicted_duration = float(duration_model.predict(model_input)[0])
    predicted_duration = max(predicted_duration, 0.5)
    
    # Blend model prediction with dataset if available
    if dataset_duration is not None:
        predicted_duration = (predicted_duration + dataset_duration) / 2.0
        logging.info("Blended duration: model=%.2f, dataset=%.2f, final=%.2f", 
                    float(duration_model.predict(model_input)[0]), dataset_duration, predicted_duration)
    
    # Step 4: ALWAYS use rule-based solution generation (ignore ML model completely for solution)
    predicted_solution = generate_solution_from_cause(matched_cause, context_area)
    logging.info("Generated rule-based solution: '%s'", predicted_solution[:100])
    
    # Step 5: Determine severity
    severity_label = (user_severity or derive_severity_label(predicted_duration) or "medium").lower()
    
    # Step 6: Generate recovery timeline steps
    base_steps = build_recovery_steps(matched_cause, severity_label, context_area, predicted_solution)
    logging.info("Generated recovery steps: %s", [s[:60] + "..." if len(s) > 60 else s for s in base_steps])
    
    # Step 7: Calculate timeline progress and ETA
    progress_lookup = {"low": 32, "medium": 52, "high": 72, "critical": 88}
    timeline_progress = progress_lookup.get(severity_label, 52)
    
    eta_days = max(1, int(round(predicted_duration)))
    eta_label = f"Target: {eta_days}d window"

    logging.info(
        "✅ Prediction complete | cause=%s | duration=%.2f | severity=%s | references=%d",
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
    ensure_models_loaded()
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

        if not ensure_models_loaded():
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
    # Load models on startup for local dev; gunicorn workers will use ensure_models_loaded.
    ensure_models_loaded()
    port = int(os.environ.get("PORT", 5000))
    debug_mode = os.environ.get("FLASK_DEBUG", "0") == "1"
    logging.info("Starting Flask server on port %s...", port)
    app.run(debug=debug_mode, host='0.0.0.0', port=port)