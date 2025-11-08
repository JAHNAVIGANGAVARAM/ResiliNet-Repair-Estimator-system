# ResiliNet Repair Estimator

AI-assisted web app that predicts repair timelines and suggests next actions for network shutdown incidents. A Flask API serves RandomForest models trained on historical outage data, while an interactive frontend surfaces insights, timelines, and matched historical context.

## Contents

```
j/
├── server.py                          # Flask application (serves UI + /predict API)
├── train_models.py                    # Deterministic training pipeline
├── requirements.txt                   # Runtime dependencies
├── Procfile                           # Render/Gunicorn start command
├── README.md
├── templates/
│   └── index.html                     # Frontend markup
├── static/
│   ├── style.css                      # Styling + layout
│   └── script.js                      # Client-side interactions
├── models/                            # Auto-created after training
│   ├── duration_model.joblib
│   ├── solution_model.joblib
│   └── model_metadata.json
└── Networks outage log with sol.xlsx  # Training dataset (kept locally / in repo as needed)
```

## Quick Start (Local)

1. **Environment**
	```powershell
	python -m venv .venv
	.\.venv\Scripts\activate
	pip install -r requirements.txt
	```

2. **Train Models** (required on a fresh checkout)
	```powershell
	python train_models.py
	```
	The script cleans the Excel dataset, trains duration + action pipelines, and stores them in `models/` along with metadata.

3. **Run the App**
	```powershell
	python server.py
	```
	Visit `http://localhost:5000` and submit incident details to see predictions. The UI supports a demo scenario button if you want to preview behaviour quickly.

## Deploying to Render

1. Train locally and ensure `models/` contains the fresh `.joblib` files and `model_metadata.json` you want in production.
2. Review `requirements.txt`; keep only the packages you need at runtime (training-only extras can move to a secondary file if desired).
3. Commit the project (including the `models/` directory and dataset if you plan to load it at runtime) and push to GitHub/GitLab/Bitbucket.
4. On [render.com](https://render.com):
	- **New > Web Service**, connect the repo.
	- Runtime: Python 3.x, build command auto-detected from `requirements.txt`.
	- Start command: leave blank (Render reads `Procfile`) or set `gunicorn server:app` explicitly.
	- Add any environment variables; the app respects `PORT` automatically and enables debug only when `FLASK_DEBUG=1`.
5. Deploy and watch the logs. Once Render reports “Live,” open the service URL, confirm the homepage loads, and send a test request.

## API

- `GET /` – serves the interactive dashboard.
- `POST /predict` – form-encoded payload (`cause`, `country`, `region`, `severity`).
  Returns JSON fields such as `repair_duration`, `recommended_solution`, `severity_label`, `matched_cause`, reference counts, and timeline guidance consumed by the frontend.

## How It Works

- **Models**: RandomForest regression estimates repair duration, and RandomForest classification recommends actions. Outputs are blended with historical medians for stability.
- **Fuzzy Matching**: RapidFuzz normalises free-text causes, reducing sensitivity to typos before querying historical incidents.
- **Contextual Insights**: Server responds with matched cause, number of similar incidents, historical median duration, severity label, and step-by-step recovery prompts. The UI visualises these through cards, timeline progress, and toasts.
- **Assets**: All styling and JavaScript live under `static/`; Flask serves them alongside the template rendered from `templates/index.html`.

## Troubleshooting

- **“Server not ready – models not loaded”**: run `python train_models.py`; ensure `models/duration_model.joblib` and `models/solution_model.joblib` exist.
- **Dataset path errors**: confirm `Networks outage log with sol.xlsx` is in the project root or adjust `DATA_PATH` in `server.py`.
- **Port already in use**: set `PORT=5001 python server.py` (PowerShell: `$Env:PORT=5001; python server.py`). The app reads the `PORT` variable automatically.
- **Render build fails**: double-check `requirements.txt` (remove heavyweight or system-dependent packages if unused at runtime) and that `gunicorn` is listed.

## Common Tasks

- **Retrain with new data**: replace/update the Excel file, re-run `train_models.py`, and redeploy the refreshed `models/` directory.
- **Run only the API**: start `gunicorn server:app` locally; the template still serves the UI without extra configuration.
- **Extend to REST clients**: send JSON by wrapping values in `FormData` or adjust `/predict` to accept raw JSON payloads (Flask already has the response structure in place).
