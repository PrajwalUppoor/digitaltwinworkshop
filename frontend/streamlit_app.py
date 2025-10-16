import streamlit as st
import requests
import time

st.set_page_config(page_title="Digital Twin Dashboard", layout="wide")

st.title("Digital Twin Dashboard")

import os

default_backend = os.environ.get("BACKEND_URL", "http://web:8000")
base_url = st.text_input("Backend URL", value=default_backend)

st.write("This dashboard shows the model status and allows running a prediction. Use Refresh to fetch active dashboard details.")

# session storage for last fetches
if 'last_status' not in st.session_state:
    st.session_state['last_status'] = None
if 'last_prediction' not in st.session_state:
    st.session_state['last_prediction'] = None
if 'last_fetch_time' not in st.session_state:
    st.session_state['last_fetch_time'] = None

col1, col2 = st.columns([2, 1])

with col1:
    st.header("Model Status")
    if st.button("Refresh"):
        try:
            r = requests.get(f"{base_url}/model-status", timeout=5)
            r.raise_for_status()
            st.session_state['last_status'] = r.json()
            st.session_state['last_fetch_time'] = time.strftime('%Y-%m-%d %H:%M:%S')
        except Exception as e:
            st.error(f"Failed to fetch model status: {e}")

    if st.session_state['last_status'] is None:
        st.info("No status fetched yet. Click Refresh to query the backend.")
    else:
        st.write(f"Last fetched: {st.session_state['last_fetch_time']}")
        st.json(st.session_state['last_status'])
        if st.session_state['last_status'].get('loaded'):
            st.success(f"Model loaded: {st.session_state['last_status'].get('type')}")
        else:
            st.warning("Model not loaded")

    st.markdown("---")
    st.header("Active Dashboard Details")
    st.write("These are the active details refreshed when you press Refresh (example data).")
    # Example: show items count (if backend provides items)
    if st.button("Fetch items count"):
        try:
            r = requests.get(f"{base_url}/items/", timeout=5)
            r.raise_for_status()
            items = r.json()
            st.write(f"Items returned: {len(items)}")
        except Exception as e:
            st.error(f"Failed to fetch items: {e}")

    if st.button("Fetch history"):
        try:
            r = requests.get(f"{base_url}/history", timeout=10)
            r.raise_for_status()
            rows = r.json()
            st.write(f"History rows: {len(rows)}")
            # show last 10 rows as a table
            import pandas as pd
            df = pd.DataFrame(rows)
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            st.dataframe(df.tail(10))
        except Exception as e:
            st.error(f"Failed to fetch history: {e}")

with col2:
    st.header("Run a Prediction")
    features_input = st.text_input("Features (comma-separated)", value="0.1, 1.2, 3.4")
    if st.button("Run Predict"):
        try:
            features = [float(x.strip()) for x in features_input.split(',') if x.strip()]
        except Exception as e:
            st.error(f"Invalid features: {e}")
            features = None

        if features is not None:
            payload = {"features": features}
            try:
                r = requests.post(f"{base_url}/predict", json=payload, timeout=10)
                if r.status_code == 200:
                    st.session_state['last_prediction'] = r.json()
                    st.success("Prediction returned")
                    st.json(st.session_state['last_prediction'])
                else:
                    st.error(f"Prediction failed: {r.status_code} {r.text}")
            except Exception as e:
                st.error(f"Prediction request failed: {e}")

    st.markdown("---")
    st.header("Last Prediction")
    if st.session_state['last_prediction'] is None:
        st.info("No prediction run yet.")
    else:
        st.json(st.session_state['last_prediction'])

st.markdown("---")
st.caption("Tip: Run Streamlit with `streamlit run frontend/streamlit_app.py` and ensure the backend is at the URL above (default http://localhost:8000).")
