# FastAPI + Postgres (Docker) - digitaltwin

This small example shows how to run a FastAPI application that connects to a Postgres database using Docker for local development. The service uses SQLModel (SQLAlchemy + Pydantic) and creates tables at startup. It also loads a pickle model `smart_home_model.pkl` (if present) and exposes a `/predict` endpoint.

Files of interest
- `app/` - FastAPI application package
  - `main.py` - app, endpoints, and `/predict` that uses `smart_home_model.pkl`
  - `models.py` - SQLModel models
  - `database.py` - DB engine and session dependency with retry logic
- `Dockerfile` - container image for the web app
- `docker-compose.yml` - starts Postgres and the web service
- `requirements.txt` - Python dependencies
- `.dockerignore` - files to ignore when building the image

If you already have `smart_home_model.pkl` in the repo root, it will be included in the image and loaded automatically.

Quick start (Docker Compose) - PowerShell

1. Build and start Postgres + web in detached mode:

```powershell
docker compose up --build -d
```

2. Watch the web service logs (optional):

```powershell
docker compose logs -f web
```

3. Open the API docs in your browser:

http://127.0.0.1:8000/docs

4. To stop and remove containers:

```powershell
docker compose down
```

Run the app locally without Docker

1. Create a virtual environment and install dependencies:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Ensure Postgres is running (e.g., via Docker or a local install) and set the environment variable in PowerShell before running the server, e.g.:

```powershell
# Set database URL in PowerShell (example):
$env:DATABASE_URL = 'postgresql://postgres:postgres@localhost:5432/postgres'
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Testing the /predict endpoint

Use `curl`, `httpie`, `requests` or the OpenAPI docs. Example using `curl`:

```powershell
curl -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" -d '{"features": [0.1, 1.2, 3.4]}'
```

If the model file is missing, `/predict` will return 503 until the model is added.

Next steps

- Add Alembic for migrations instead of using `create_all()` at startup.
- Add health/readiness endpoints and readiness checks for orchestrators.
- Replace the simple pickled model with an artifact from a model registry for production.
