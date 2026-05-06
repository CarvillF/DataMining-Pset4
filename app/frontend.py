"""
Frontend Streamlit que consume la API FastAPI de predicción de precios.

Levantar (en una terminal aparte, desde la raíz del repo):
    1. `uvicorn src.api.main:app --reload`            # backend en :8000
    2. `streamlit run app/frontend.py`                # frontend en :8501

Variable de entorno opcional:
    API_URL → URL completa del endpoint de predicción
              (default: http://127.0.0.1:8000)
"""

from __future__ import annotations

import os
from datetime import datetime, time

import requests
import streamlit as st

# ─────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────────────────────

API_BASE = os.getenv("API_URL", "http://127.0.0.1:8000").rstrip("/")
PREDICT_URL = f"{API_BASE}/predict"
HEALTH_URL = f"{API_BASE}/health"

st.set_page_config(
    page_title="NYC Taxi Price Estimator",
    page_icon="🚕",
    layout="centered",
)


# ─────────────────────────────────────────────────────────────
# CATÁLOGOS (alineados con la OBT)
# ─────────────────────────────────────────────────────────────

BOROUGHS = ["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island", "EWR", "Unknown"]
SERVICES = ["yellow", "green"]
RATE_CODES = {
    "Standard rate": 1,
    "JFK": 2,
    "Newark": 3,
    "Nassau or Westchester": 4,
    "Negotiated fare": 5,
    "Group ride": 6,
}
VENDORS = [
    "Creative Mobile Technologies, LLC",
    "Curb Mobility, LLC",
    "Myle Technologies Inc",
    "Helix",
]
PAYMENT_TYPES = ["Flex Fare trip", "Cash", "No charge", "Dispute", "Unknown"]


# ─────────────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────────────

st.title("🚕 NYC Taxi — Estimador de Precio")
st.caption(
    "Calcula el precio estimado del viaje **antes** de iniciarlo. "
    "Modelo entrenado sobre datos NYC TLC 2015-2023 (Yellow + Green Taxi)."
)

# Barra lateral con estado de la API
with st.sidebar:
    st.header("⚙️ Estado del servicio")
    st.write(f"**API:** `{API_BASE}`")
    try:
        r = requests.get(HEALTH_URL, timeout=3)
        if r.ok:
            data = r.json()
            if data.get("model_loaded"):
                st.success(f"✅ Modelo cargado")
                st.caption(f"Versión: `{data.get('model_version', 'n/a')}`")
                st.caption(f"Features: {data.get('feature_count', 'n/a')}")
            else:
                st.warning("⚠️ API arriba pero sin modelo cargado.")
                if data.get("load_error"):
                    st.code(data["load_error"], language="text")
        else:
            st.error(f"API respondió {r.status_code}")
    except requests.RequestException as e:
        st.error(f"❌ API no responde: {e}")

st.divider()

# Formulario principal
with st.form("trip_form"):
    st.subheader("📋 Datos del viaje")

    col1, col2 = st.columns(2)
    with col1:
        pickup_date = st.date_input("Fecha de inicio", value=datetime.now().date())
        pickup_time = st.time_input("Hora de inicio", value=time(14, 30))
        trip_distance = st.number_input(
            "Distancia estimada (millas)",
            min_value=0.1, max_value=200.0, value=3.5, step=0.1,
            help="Distancia del viaje en millas. Disponible al hacer la solicitud.",
        )
        passenger_count = st.number_input(
            "Pasajeros",
            min_value=1, max_value=6, value=1,
        )

    with col2:
        rate_code_desc = st.selectbox(
            "Tarifa",
            list(RATE_CODES.keys()),
            help="JFK = $52 fijo. Newark agrega recargo aeroportuario.",
        )
        source_service = st.selectbox("Servicio", SERVICES)
        vendor_name = st.selectbox("Proveedor", VENDORS, index=1)
        payment_type_desc = st.selectbox("Pago previsto", PAYMENT_TYPES)

    st.subheader("📍 Origen y destino")
    c1, c2 = st.columns(2)
    with c1:
        pu_borough = st.selectbox("Borough origen", BOROUGHS)
        pu_location_id = st.number_input(
            "PU LocationID (TLC)", min_value=1, max_value=265, value=161,
            help="ID de zona TLC. 161 = Midtown Center (Manhattan).",
        )
        pu_zone = st.text_input("Zona origen (opcional)", value="Unknown")
    with c2:
        do_borough = st.selectbox("Borough destino", BOROUGHS)
        do_location_id = st.number_input(
            "DO LocationID (TLC)", min_value=1, max_value=265, value=237,
            help="237 = Upper East Side South (Manhattan).",
        )
        do_zone = st.text_input("Zona destino (opcional)", value="Unknown")

    submit = st.form_submit_button("💵 Estimar precio", type="primary", use_container_width=True)


# ─────────────────────────────────────────────────────────────
# LLAMADA A LA API
# ─────────────────────────────────────────────────────────────

if submit:
    payload = {
        "pickup_datetime": datetime.combine(pickup_date, pickup_time).isoformat(),
        "trip_distance": float(trip_distance),
        "passenger_count": int(passenger_count),
        "pu_location_id": int(pu_location_id),
        "do_location_id": int(do_location_id),
        "pu_borough": pu_borough,
        "do_borough": do_borough,
        "pu_zone": pu_zone or "Unknown",
        "do_zone": do_zone or "Unknown",
        "vendor_name": vendor_name,
        "rate_code_id": RATE_CODES[rate_code_desc],
        "rate_code_desc": rate_code_desc,
        "payment_type_desc": payment_type_desc,
        "source_service": source_service,
    }

    with st.spinner("Calculando..."):
        try:
            response = requests.post(PREDICT_URL, json=payload, timeout=15)
        except requests.RequestException as e:
            st.error(f"❌ No se pudo conectar a la API en {PREDICT_URL}\n\n{e}")
            st.stop()

        if response.status_code == 503:
            st.error(
                "⚠️ La API está arriba pero el modelo no está cargado. "
                "Entrena el modelo con `python -m src.models.train_model --mode sample` "
                "y reinicia el servidor."
            )
        elif not response.ok:
            st.error(f"Error {response.status_code}: {response.text}")
        else:
            data = response.json()
            est = data.get("estimated_total_amount")
            if est is None:
                st.error(f"Respuesta inesperada: {data}")
            else:
                st.success(f"## 💵 Precio estimado: **${est:,.2f} USD**")
                st.caption(f"Modelo: `{data.get('model_version', 'n/a')}`")
                with st.expander("Ver request enviado"):
                    st.json(payload)

# ─────────────────────────────────────────────────────────────
# INFO
# ─────────────────────────────────────────────────────────────

st.divider()
with st.expander("ℹ️ Sobre el modelo"):
    st.markdown(
        """
- **Target**: `total_amount` (USD).
- **Features**: 30+ variables espacio-temporales (cyclical encoding de hora /
  día / mes, flags de aeropuerto, rush hour, festivos, mismo borough,
  interacciones).
- **Sin leakage**: variables como `fare_amount`, `tip_amount`, `dropoff_*`,
  duración del viaje, velocidad promedio, `improvement_surcharge`, etc.
  están **explícitamente excluidas** porque sólo se conocen al cierre del viaje.
- **Splits**: train 2015-2023 / val 2024 / test 2025 (vistas SQL en Snowflake).
- **Algoritmo**: gradient boosting con log-transform del target (`log1p`/`expm1`).
        """
    )
