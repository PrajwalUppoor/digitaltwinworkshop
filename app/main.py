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
import json
import pandas as pd
from fastapi.responses import FileResponse, JSONResponse
from fastapi import UploadFile, File as FAFile, Form
from sklearn.metrics import confusion_matrix, classification_report

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


@app.get('/feature-columns')
def feature_columns():
    """Return the feature columns used by the model (from file)."""
    cols = _load_feature_columns()
    if cols is None:
        return JSONResponse(status_code=404, content={"error": "feature_columns.json not found in expected locations"})
    return {"columns": cols}


def _load_feature_columns():
    """Try to load feature_columns.json from several likely locations and return list or None."""
    candidates = [
        os.path.join(os.getcwd(), 'feature_columns.json'),
        os.path.join(os.getcwd(), 'data', 'feature_columns.json'),
        '/app/feature_columns.json',
        '/app/data/feature_columns.json',
    ]
    for c in candidates:
        try:
            if os.path.exists(c):
                with open(c) as f:
                    return json.load(f)
        except Exception:
            continue
    return None


@app.get('/history')
def history():
    """Return digital twin history rows as JSON (from CSV if present)."""
    candidates = [
        os.path.join(os.getcwd(), 'digital_twin_history.csv'),
        os.path.join(os.getcwd(), 'data', 'digital_twin_history.csv'),
        '/app/data/digital_twin_history.csv',
    ]
    p = None
    for c in candidates:
        if os.path.exists(c):
            p = c
            break
    if p is None:
        return JSONResponse(status_code=404, content={"error": "digital_twin_history.csv not found in expected locations"})
    try:
        df = pd.read_csv(p)
        # convert timestamps to iso format if present
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp']).astype(str)
        return df.to_dict(orient='records')
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get('/model-info')
def model_info():
    """Return model metadata: feature importances and a simple confusion matrix
    if available. This is best-effort; returns what exists on the loaded model.
    """
    if MODEL is None:
        return JSONResponse(status_code=404, content={"error": "Model not loaded"})

    info = {}
    # feature importances
    try:
        import numpy as _np
        fi = getattr(MODEL, 'feature_importances_', None)
        if fi is not None:
            cols = _load_feature_columns()
            if cols is None or len(cols) != len(fi):
                # fallback to generic names if lengths mismatch
                cols = [f'feature_{i}' for i in range(len(fi))]
            info['feature_importances'] = dict(zip(cols, _np.round(_np.array(fi).tolist(), 4)))
    except Exception:
        pass

    # confusion matrix: if we have a stored test set or the model has attributes, skip for now
    # we leave room for future: users can upload a test set and we compute it here.

    return info


@app.post('/confusion')
async def confusion(file: UploadFile = FAFile(...), label_column: str = Form("ac_status")):
    """Compute confusion matrix and classification report from an uploaded CSV.

    Expects a CSV with the true labels in `label_column` and the feature columns
    matching `feature_columns.json`. Returns a JSON with the confusion matrix
    and a classification report (precision/recall/f1) when possible.
    """
    if MODEL is None:
        return JSONResponse(status_code=404, content={"error": "Model not loaded"})

    # load CSV into a DataFrame
    try:
        import io
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": f"Failed to read uploaded CSV: {e}"})

    # load feature columns
    cols = _load_feature_columns()
    if cols is None:
        return JSONResponse(status_code=400, content={"error": "feature_columns.json not found; cannot select features"})

    # ensure label column exists
    if label_column not in df.columns:
        return JSONResponse(status_code=400, content={"error": f"Label column '{label_column}' not found in uploaded CSV"})

    # ensure feature columns exist in df
    missing = [c for c in cols if c not in df.columns]
    if missing:
        return JSONResponse(status_code=400, content={"error": f"Missing feature columns in CSV: {missing}"})

    try:
        X = df[cols].values
        y_true = df[label_column].values
        # predict
        y_pred = MODEL.predict(X)
        # compute confusion matrix
        labels = sorted(list(set(list(y_true) + list(y_pred))))
        cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()
        creport = classification_report(y_true, y_pred, labels=labels, output_dict=True)
        return {
            "labels": labels,
            "confusion_matrix": cm,
            "classification_report": creport,
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Failed to compute confusion matrix: {e}"})


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

        # prediction
        pred = MODEL.predict(X)
        # convert numpy types to native Python
        result = pred.tolist() if hasattr(pred, "tolist") else list(pred)

        # probability/confidence when available
        prob = None
        try:
            if hasattr(MODEL, "predict_proba"):
                proba = MODEL.predict_proba(X)
                # choose probability of positive class if binary, else full vector
                prob = proba.tolist()
        except Exception:
            prob = None

        # include feature importances (if model exposes them)
        fi_map = None
        try:
            fi = getattr(MODEL, 'feature_importances_', None)
            if fi is not None:
                cols = _load_feature_columns()
                if cols is None or len(cols) != len(fi):
                    cols = [f'feature_{i}' for i in range(len(fi))]
                # convert to simple serializable mapping
                fi_map = dict(zip(cols, [float(x) for x in list(fi)]))
        except Exception:
            fi_map = None

        resp = {"prediction": result}
        if prob is not None:
            resp["probability"] = prob
        if fi_map is not None:
            resp["feature_importances"] = fi_map
        return resp
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
