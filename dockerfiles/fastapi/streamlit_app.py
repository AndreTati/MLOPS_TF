import os
import requests
import streamlit as st

FASTAPI_URL = os.getenv("FASTAPI_URL", "http://localhost:8800")

st.set_page_config(page_title="Dielectron Mass Predictor", layout="centered")

st.title("Dielectron Invariant Mass Prediction")
st.write("Ingresa las variables del par de electrones y obtén la predicción de la masa invariante M.")

with st.form(key="prediction_form"):
    E1 = st.number_input("E1", value=50.0, format="%.6f")
    px1 = st.number_input("px1", value=15.0, format="%.6f")
    py1 = st.number_input("py1", value=-8.0, format="%.6f")
    pz1 = st.number_input("pz1", value=40.0, format="%.6f")
    pt1 = st.number_input("pt1", value=17.0, format="%.6f")
    eta1 = st.number_input("eta1", value=1.2, format="%.6f")
    phi1 = st.number_input("phi1", value=0.5, format="%.6f")
    Q1 = st.number_input("Q1", value=-1.0, format="%.6f")
    E2 = st.number_input("E2", value=45.0, format="%.6f")
    px2 = st.number_input("px2", value=-12.0, format="%.6f")
    py2 = st.number_input("py2", value=5.0, format="%.6f")
    pz2 = st.number_input("pz2", value=-35.0, format="%.6f")
    pt2 = st.number_input("pt2", value=13.0, format="%.6f")
    eta2 = st.number_input("eta2", value=-0.8, format="%.6f")
    phi2 = st.number_input("phi2", value=-0.2, format="%.6f")
    Q2 = st.number_input("Q2", value=1.0, format="%.6f")

    submit_button = st.form_submit_button("Predecir masa M")

if submit_button:
    payload = {
        "E1": E1,
        "px1": px1,
        "py1": py1,
        "pz1": pz1,
        "pt1": pt1,
        "eta1": eta1,
        "phi1": phi1,
        "Q1": Q1,
        "E2": E2,
        "px2": px2,
        "py2": py2,
        "pz2": pz2,
        "pt2": pt2,
        "eta2": eta2,
        "phi2": phi2,
        "Q2": Q2,
    }

    try:
        response = requests.post(f"{FASTAPI_URL}/predict/", json=payload, timeout=15)
        response.raise_for_status()
        result = response.json()
        st.success(f"Predicción de masa invariante M: {result['prediction']:.6f}")
    except requests.RequestException as exc:
        st.error(f"Error al llamar a la API de FastAPI: {exc}")
        if exc.response is not None:
            st.json(exc.response.text)

st.markdown("---")
st.subheader("Variables del dataset")
st.write(
    "Este modelo usa las características del dataset `dielectron.csv` para predecir la masa invariante `M`. "
    "No se usan las columnas `Run` ni `Event` porque son identificadores."
)
