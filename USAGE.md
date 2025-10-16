# digitaltwin - Usage

This document explains how to run the backend (FastAPI) and a simple React frontend that consumes the API. It also provides curl examples.

Prerequisites
- Docker and Docker Compose (recommended)
- Or Python 3.10+ and the dependencies in `requirements.txt`

Run with Docker Compose
1. Build and start services:

```powershell
docker compose up --build -d
```

2. View backend logs:

```powershell
docker compose logs -f web
```

3. The API will be available at http://localhost:8000

Useful endpoints
- GET /model-status  — returns whether a model is loaded
- POST /predict     — run a prediction; body: { "features": [0.1, 1.2, 3.4] }
- GET /items/ etc.  — CRUD items backed by the Postgres DB

Curl examples

Check model status:

```bash
curl -s http://127.0.0.1:8000/model-status | jq
```

Run prediction (example):

```bash
curl -s -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [0.1, 1.2, 3.4]}' | jq
```

If the model is not loaded the `/predict` endpoint will return HTTP 503.

Run the React frontend (development)

1. Change to the frontend folder and install dependencies:

```powershell
cd frontend
npm install
```

2. Start the dev server:

```powershell
npm run dev
```

The React app will run on http://localhost:5173 by default and calls the backend at http://localhost:8000.

Streamlit frontend (quick and easy)

1. Install the Streamlit frontend dependencies (recommended in a virtualenv):

```powershell
python -m venv .venv-frontend
.venv-frontend\Scripts\Activate.ps1
pip install -r frontend/requirements.txt
```

2. Run the Streamlit app:

```powershell
streamlit run frontend/streamlit_app.py
```

The dashboard will open in your browser (usually http://localhost:8501). Use the Refresh button to fetch active dashboard details such as model status and items count. Use Run Predict to send a features vector to `/predict`.

Production
- Build the frontend and serve it from a static host or integrate it into the Docker image. For local dev the bind-mount of `smart_home_model.pkl` allows replacing the model without rebuilding.

Notes
- CORS is enabled to allow a dev React server to call the API. Lock down origins before deploying to production.
