import os
import pickle
import traceback
from typing import List, Optional

try:
    import joblib  # type: ignore
except Exception:
    joblib = None

try:
    import cloudpickle  # type: ignore
except Exception:
    cloudpickle = None

from fastapi import FastAPI, Depends, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlmodel import select, Session

from .database import init_db, get_session
from .models import Item

app = FastAPI(title="digitaltwin-fastapi")

# Allow the React frontend (served on a different port during development) to
# call the API. In production you should tighten `allow_origins` to your
# frontend origin(s).
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# model will be loaded at startup if present
MODEL: Optional[object] = None


class PredictRequest(BaseModel):
    """Accepts a feature vector as a list of numbers."""
    features: List[float]


@app.on_event("startup")
def on_startup():
    # initialize DB (create tables) on startup
    init_db()

    # try to load ML model from repository root (smart_home_model.pkl)
    global MODEL
    candidate_paths = [
        os.path.join(os.getcwd(), "smart_home_model.pkl"),
        os.path.join(os.path.dirname(__file__), "..", "smart_home_model.pkl"),
        os.path.join(os.path.dirname(__file__), "..", "models", "smart_home_model.pkl"),
        "/app/smart_home_model.pkl",
    ]

    found = False
    for p in candidate_paths:
        p = os.path.normpath(p)
        if os.path.exists(p):
            # Try multiple loaders because the model may have been saved with
            # joblib or cloudpickle or with a different pickle protocol.
            loaded = None
            last_err: Optional[Exception] = None

            # 1) Try pickle (default)
            try:
                with open(p, "rb") as f:
                    loaded = pickle.load(f)
                print(f"Loaded ML model from {p} using pickle")
            except Exception as e_pickle:
                last_err = e_pickle

            # 2) Try joblib if available
            if loaded is None and joblib is not None:
                try:
                    loaded = joblib.load(p)
                    print(f"Loaded ML model from {p} using joblib")
                except Exception as e_joblib:
                    last_err = e_joblib

            # 3) Try cloudpickle if available
            if loaded is None and cloudpickle is not None:
                try:
                    with open(p, "rb") as f:
                        loaded = cloudpickle.load(f)
                    print(f"Loaded ML model from {p} using cloudpickle")
                except Exception as e_cloud:
                    last_err = e_cloud

            # 4) As a last resort, try pickle with latin1 encoding (Python2->3 compatibility)
            if loaded is None:
                try:
                    with open(p, "rb") as f:
                        loaded = pickle.load(f, encoding="latin1")
                    print(f"Loaded ML model from {p} using pickle with latin1 encoding")
                except Exception as e_final:
                    last_err = e_final

            if loaded is None:
                print(f"Failed to load model at {p}: {last_err}")
                traceback.print_exc()
                MODEL = None
                found = False
                # continue searching other candidates
            else:
                MODEL = loaded
                found = True
                break

    if not found:
        print("Model file not found in candidates; /predict will return 503")


@app.get('/model-status')
def model_status():
    """Return whether a model is loaded and its type (for debugging)."""
    if MODEL is None:
        return {"loaded": False}
    try:
        t = type(MODEL).__name__
        return {"loaded": True, "type": t}
    except Exception:
        return {"loaded": True, "type": "unknown"}


@app.get('/history')
def history():
    """Return the history CSV as JSON rows (list of dicts)."""
    csv_path = os.path.join(os.getcwd(), 'data', 'digital_twin_history.csv')
    if not os.path.exists(csv_path):
        raise HTTPException(status_code=404, detail='history file not found')
    try:
        import pandas as pd
        df = pd.read_csv(csv_path, parse_dates=['timestamp'])
        # convert to records and ensure timestamps are ISO strings
        records = df.to_dict(orient='records')
        for r in records:
            if isinstance(r.get('timestamp'), (pd.Timestamp,)):
                r['timestamp'] = r['timestamp'].isoformat()
        return records
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'failed to read history: {e}')


@app.get('/history.csv')
def history_csv():
    csv_path = os.path.join(os.getcwd(), 'data', 'digital_twin_history.csv')
    if not os.path.exists(csv_path):
        raise HTTPException(status_code=404, detail='history file not found')
    return FileResponse(csv_path, media_type='text/csv', filename='digital_twin_history.csv')


@app.post("/items/", response_model=Item)
def create_item(item: Item, session: Session = Depends(get_session)):
    session.add(item)
    session.commit()
    session.refresh(item)
    return item


@app.get("/items/", response_model=List[Item])
def list_items(session: Session = Depends(get_session)):
    items = session.exec(select(Item)).all()
    return items


@app.get("/items/{item_id}", response_model=Item)
def get_item(item_id: int, session: Session = Depends(get_session)):
    item = session.get(Item, item_id)
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")
    return item


@app.post("/predict")
def predict(req: PredictRequest):
    """Run prediction using the loaded ML model. Expects a list of numeric features."""
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Many scikit-learn models accept 2D arrays (n_samples, n_features)
    try:
        X = [req.features]
        # Validate feature length when model provides n_features_in_
        try:
            expected = getattr(MODEL, "n_features_in_", None)
            if expected is not None:
                if len(req.features) != int(expected):
                    raise HTTPException(
                        status_code=400,
                        detail=(
                            f"Invalid feature vector: model expects {int(expected)} features, "
                            f"but input has {len(req.features)} features."
                        ),
                    )
        except HTTPException:
            raise
        except Exception:
            # if querying n_features_in_ fails, continue and let predict raise if needed
            pass

        pred = MODEL.predict(X)
        # convert numpy types to native Python
        result = pred.tolist() if hasattr(pred, "tolist") else list(pred)
        return {"prediction": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
