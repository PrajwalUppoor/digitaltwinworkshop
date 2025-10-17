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
            # fetch model info & feature columns
            try:
                rcols = requests.get(f"{base_url}/feature-columns", timeout=5)
                if rcols.status_code == 200:
                    st.session_state['feature_columns'] = rcols.json().get('columns', [])
                rinfo = requests.get(f"{base_url}/model-info", timeout=5)
                if rinfo.status_code == 200:
                    st.session_state['model_info'] = rinfo.json()
            except Exception:
                st.session_state['feature_columns'] = st.session_state.get('feature_columns', [])
                st.session_state['model_info'] = st.session_state.get('model_info', {})
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

    # show feature importances if available
    fi = st.session_state.get('model_info', {}).get('feature_importances')
    if fi:
        import pandas as pd
        import plotly.express as px
        df_fi = pd.DataFrame({'feature': list(fi.keys()), 'importance': list(fi.values())})
        df_fi = df_fi.sort_values('importance', ascending=True)
        st.markdown("### Feature importances")
        st.plotly_chart(px.bar(df_fi, x='importance', y='feature', orientation='h'), use_container_width=True)

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
            # quick plots: temperature over time and health score
            try:
                import plotly.express as px
                if 'temperature' in df.columns:
                    st.plotly_chart(px.line(df.tail(200), x='timestamp', y='temperature', title='Temperature (recent)'), use_container_width=True)
                if 'health_score' in df.columns:
                    st.plotly_chart(px.line(df.tail(200), x='timestamp', y='health_score', title='Health Score (recent)'), use_container_width=True)
            except Exception as e:
                st.info(f"Plotting skipped: {e}")
        except Exception as e:
            st.error(f"Failed to fetch history: {e}")

with col2:
    st.header("Run a Prediction")
    # fetch feature columns if we have them
    feature_cols = st.session_state.get('feature_columns') or []
    if not feature_cols:
        try:
            rcols = requests.get(f"{base_url}/feature-columns", timeout=5)
            if rcols.status_code == 200:
                feature_cols = rcols.json().get('columns', [])
                st.session_state['feature_columns'] = feature_cols
        except Exception:
            feature_cols = []

    st.write("Provide feature values (one field per feature). You can prefill from the latest history row.")
    # build a form with one input per feature
    values = {}
    if feature_cols:
        if 'prefill_latest' not in st.session_state:
            st.session_state['prefill_latest'] = False
        cols_ui = st.form(key='predict_form')
        with cols_ui:
            st.write("Features:")
            for c in feature_cols:
                default_val = ''
                # if user chose to prefill, fetch latest history and use values
                if st.session_state.get('prefill_latest'):
                    try:
                        r = requests.get(f"{base_url}/history", timeout=5)
                        if r.status_code == 200:
                            rows = r.json()
                            if rows:
                                latest = rows[-1]
                                if c in latest:
                                    default_val = str(latest[c])
                    except Exception:
                        default_val = ''
                values[c] = cols_ui.text_input(c, value=default_val)
            submit = cols_ui.form_submit_button('Run Predict')
            prefill = cols_ui.checkbox('Prefill from latest history', value=False)
            if prefill != st.session_state.get('prefill_latest'):
                st.session_state['prefill_latest'] = prefill
    else:
        st.info('Feature columns not available. You can still use the /predict API directly with JSON features list.')

    if 'submit' in locals() and submit:
        try:
            feature_values = [float(values[c]) for c in feature_cols]
        except Exception as e:
            st.error(f"Invalid feature inputs: {e}")
            feature_values = None

        if feature_values is not None:
            payload = {"features": feature_values}
            try:
                r = requests.post(f"{base_url}/predict", json=payload, timeout=10)
                if r.status_code == 200:
                    res = r.json()
                    st.session_state['last_prediction'] = res
                    st.success("Prediction returned")
                    st.json(res)
                    # show probability if present
                    if res.get('probability') is not None:
                        st.markdown('### Probability / Confidence')
                        st.write(res['probability'])
                    # show feature importances if returned
                    if res.get('feature_importances'):
                        import pandas as pd
                        import plotly.express as px
                        fi = res['feature_importances']
                        df_fi = pd.DataFrame({'feature': list(fi.keys()), 'importance': list(fi.values())})
                        df_fi = df_fi.sort_values('importance', ascending=True)
                        st.plotly_chart(px.bar(df_fi, x='importance', y='feature', orientation='h'), use_container_width=True)
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
    st.header("Confusion Matrix (upload test CSV)")
    st.write("Upload a CSV with true labels (default column 'ac_status') and feature columns matching `feature_columns.json`.")
    uploaded = st.file_uploader("Choose CSV file", type=['csv'])
    label_col = st.text_input("Label column name", value='ac_status')
    if st.button("Compute Confusion Matrix"):
        if uploaded is None:
            st.error("Please upload a CSV file first")
        else:
            try:
                files = {'file': (uploaded.name, uploaded.getvalue())}
                import requests
                # use multipart/form-data
                resp = requests.post(f"{base_url}/confusion", files=files, data={'label_column': label_col}, timeout=30)
                if resp.status_code != 200:
                    st.error(f"Failed: {resp.status_code} {resp.text}")
                else:
                    data = resp.json()
                    labels = data.get('labels', [])
                    cm = data.get('confusion_matrix', [])
                    import pandas as pd
                    import plotly.express as px
                    df_cm = pd.DataFrame(cm, index=[str(l) for l in labels], columns=[str(l) for l in labels])
                    st.plotly_chart(px.imshow(df_cm, text_auto=True, color_continuous_scale='Blues', title='Confusion Matrix'), use_container_width=True)
                    st.markdown("### Classification Report")
                    st.json(data.get('classification_report', {}))
            except Exception as e:
                st.error(f"Error computing confusion matrix: {e}")

st.markdown("---")
st.caption("Tip: Run Streamlit with `streamlit run frontend/streamlit_app.py` and ensure the backend is at the URL above (default http://localhost:8000).")
