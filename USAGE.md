(# digitaltwin — Usage)

This document explains how to run the backend (FastAPI + Postgres) and the Streamlit dashboard, and describes the API endpoints you can use for predictions, model metadata, and history.

Requirements
- Docker & Docker Compose (recommended)
- Or Python 3.10+ and the dependencies in `requirements.txt` and `frontend/requirements.txt`

Start with Docker Compose (recommended)

1. Build and start services:

```powershell
docker compose up --build -d
```

2. Tail logs if needed:

```powershell
docker compose logs -f web
docker compose logs -f dashboard
```

API endpoints

- GET /model-status
	- Returns whether a model is loaded and its type.
- POST /predict
	- Body: { "features": [f1, f2, ...] }
	- Returns a JSON prediction.
	- The server validates the number of features when possible and returns HTTP 400 if the length mismatches the model's expected input.
- GET /feature-columns
	- Returns an ordered list of feature names (so you can submit features in correct order).
- GET /model-info
	- Returns simple model metadata (feature importances if available).
- GET /history
	- Returns the digital twin history (from `digital_twin_history.csv`) as JSON.
- GET /items/
	- CRUD endpoints for example items stored in Postgres.

Curl examples

Check model status:

```powershell
curl -s http://127.0.0.1:8000/model-status | jq
```

Get feature columns:

```powershell
curl -s http://127.0.0.1:8000/feature-columns | jq
```

Get model info (feature importances):

```powershell
curl -s http://127.0.0.1:8000/model-info | jq
```

Download recent history (first 5 rows):

```powershell
curl -s http://127.0.0.1:8000/history | jq '.[0:5]'
```

Run prediction (example):

```powershell
curl -s -X POST http://127.0.0.1:8000/predict \
	-H "Content-Type: application/json" \
	-d '{"features": [0.1, 1.2, 3.4]}' | jq
```

Streamlit dashboard

The repository includes a Streamlit dashboard at `frontend/streamlit_app.py`. It displays:
- Model status and type
- Feature importances (if the loaded model exposes them)
- Digital twin history (plots: temperature over time, health score)
- A form to run predictions and see the last prediction

Run the dashboard locally (recommended for development):

```powershell
python -m venv .venv-frontend
.venv-frontend\Scripts\Activate.ps1
pip install -r frontend/requirements.txt
streamlit run frontend/streamlit_app.py
```

If you run Docker Compose, the dashboard service is started by the `dashboard` service and will be available at http://localhost:8501 (it uses `BACKEND_URL=http://web:8000`).

Notes on features & predictions

- Use `GET /feature-columns` to learn the correct order of features for `/predict`.
- The backend validates feature vector length against the model's `n_features_in_` when available and returns a helpful 400 error if they mismatch.
- The Streamlit app fetches `/model-info` to render feature importances and `/history` to display recent telemetry and health score charts.

Troubleshooting

- If `/predict` returns HTTP 400 complaining about feature length, call `/feature-columns` and supply the correct number and order of features.
- If `/history` returns 404, make sure `digital_twin_history.csv` exists in the repo root (it is included in the repository by default).

Security & production notes

- CORS is enabled for development convenience; restrict `allow_origins` in production.
- Pickled models are sensitive to library versions — ensure your environment matches the version used to save the model (scikit-learn pinned to 1.6.1 in `requirements.txt`).

If you'd like, I can also:
- Add a small UI panel in Streamlit to show the confusion matrix (if you provide a test set or we store predictions),
- Add an endpoint to compute model performance on an uploaded test CSV,
- Or embed the complete notebook visualizations directly into the dashboard.

---

Generated on: 2025-10-16

