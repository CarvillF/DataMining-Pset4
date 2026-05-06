"""
API FastAPI que sirve el modelo de predicción de precios NYC Taxi.

Endpoints:
    GET  /health         → estado del servicio + flag de modelo cargado.
    POST /predict        → un solo viaje, retorna total_amount estimado.
    POST /predict/batch  → lista de viajes, retorna lista de predicciones.

Levantar el servidor (desde la raíz del repo):
    uvicorn src.api.main:app --reload

Variables de entorno opcionales:
    MODEL_PATH          → ruta al .pkl (default: models/price_model.pkl)
    FEATURE_CONFIG_PATH → ruta al feature_config.json
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field, field_validator

from src.models.predict_model import load_artifacts, predict
from src.utils.config import MODEL_PATH, MODELS_DIR, setup_logging

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# ESTADO COMPARTIDO (cargado en el startup)
# ─────────────────────────────────────────────────────────────

_state: dict = {
    "model": None,
    "feature_columns": None,
    "model_version": None,
    "load_error": None,
}


# ─────────────────────────────────────────────────────────────
# ESQUEMAS PYDANTIC
# ─────────────────────────────────────────────────────────────

# Valores permitidos (alineados con la SQL OBT)
_VALID_BOROUGHS = {
    "Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island", "EWR", "Unknown",
}
_VALID_RATE_CODES = {1, 2, 3, 4, 5, 6}
_VALID_SERVICES = {"yellow", "green"}


class TripInput(BaseModel):
    """
    Datos del viaje conocidos ANTES de iniciarlo.
    No incluyen variables de cierre (fare_amount, tip_amount, etc.) → cero leakage.
    """
    pickup_datetime: datetime = Field(..., description="Fecha y hora de inicio del viaje")
    trip_distance: float = Field(..., gt=0, le=200, description="Distancia estimada en millas")
    passenger_count: int = Field(..., ge=1, le=6, description="Cantidad de pasajeros (1-6)")

    pu_location_id: int = Field(..., ge=1, le=265, description="LocationID origen (TLC zones)")
    do_location_id: int = Field(..., ge=1, le=265, description="LocationID destino (TLC zones)")

    pu_borough: str = Field("Unknown", description="Borough de origen")
    do_borough: str = Field("Unknown", description="Borough de destino")
    pu_zone: str = Field("Unknown", description="Zona TLC de origen")
    do_zone: str = Field("Unknown", description="Zona TLC de destino")

    vendor_name: str = Field("Curb Mobility, LLC", description="Proveedor del viaje")
    rate_code_id: int = Field(1, description="ID de tarifa (1=Std, 2=JFK, 3=Newark, ...)")
    rate_code_desc: str = Field("Standard rate", description="Descripción de la tarifa")
    payment_type_desc: str = Field("Flex Fare trip", description="Tipo de pago (a priori)")

    source_service: str = Field("yellow", description="Servicio: yellow / green")

    @field_validator("pu_borough", "do_borough")
    @classmethod
    def _validate_borough(cls, v: str) -> str:
        return v if v in _VALID_BOROUGHS else "Unknown"

    @field_validator("rate_code_id")
    @classmethod
    def _validate_rate_code(cls, v: int) -> int:
        return v if v in _VALID_RATE_CODES else 1

    @field_validator("source_service")
    @classmethod
    def _validate_service(cls, v: str) -> str:
        v_low = v.lower()
        return v_low if v_low in _VALID_SERVICES else "yellow"


class PredictionOutput(BaseModel):
    estimated_total_amount: float = Field(..., description="Estimación de total_amount en USD")
    currency: str = "USD"
    model_version: Optional[str] = None


class BatchInput(BaseModel):
    trips: list[TripInput]


class BatchOutput(BaseModel):
    predictions: list[float]
    currency: str = "USD"
    model_version: Optional[str] = None


class HealthOutput(BaseModel):
    status: str
    model_loaded: bool
    model_version: Optional[str] = None
    feature_count: Optional[int] = None
    load_error: Optional[str] = None


# ─────────────────────────────────────────────────────────────
# CICLO DE VIDA: carga del modelo en startup
# ─────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    model_path = os.getenv("MODEL_PATH", str(MODEL_PATH))
    config_path = os.getenv(
        "FEATURE_CONFIG_PATH", str(MODELS_DIR / "feature_config.json"),
    )
    try:
        model, cfg = load_artifacts(model_path=model_path, config_path=config_path)
        _state["model"] = model
        _state["feature_columns"] = cfg["feature_columns"]
        _state["model_version"] = cfg.get("metadata", {}).get("model_version", "v1.0")
        logger.info(
            "Modelo cargado (v=%s | %d features) desde %s",
            _state["model_version"], len(_state["feature_columns"]), model_path,
        )
    except Exception as e:
        # No tirar el server: permitir levantarlo aún sin modelo
        # (útil en CI / dev antes de entrenar).
        _state["load_error"] = str(e)
        logger.warning("No se pudo cargar el modelo: %s", e)
    yield
    _state.clear()


app = FastAPI(
    title="API - Predicción de Precios NYC Taxi",
    version="1.0.0",
    description="Predice `total_amount` antes de iniciar el viaje. Cero leakage.",
    lifespan=lifespan,
)


# ─────────────────────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthOutput)
def health() -> HealthOutput:
    """Probe de salud + estado del modelo."""
    return HealthOutput(
        status="ok" if _state.get("model") is not None else "degraded",
        model_loaded=_state.get("model") is not None,
        model_version=_state.get("model_version"),
        feature_count=(
            len(_state["feature_columns"]) if _state.get("feature_columns") else None
        ),
        load_error=_state.get("load_error"),
    )


def _trips_to_dataframe(trips: list[TripInput]) -> pd.DataFrame:
    """Convierte una lista de TripInput a DataFrame con tipos correctos."""
    rows = [t.model_dump() for t in trips]
    df = pd.DataFrame(rows)
    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])
    return df


def _ensure_model_loaded():
    if _state.get("model") is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Modelo no cargado. Entrena con "
                "`python -m src.models.train_model --mode sample` y reinicia el servidor. "
                f"Detalle: {_state.get('load_error')}"
            ),
        )


@app.post("/predict", response_model=PredictionOutput)
def predict_one(trip: TripInput) -> PredictionOutput:
    """Predice el `total_amount` de un viaje individual."""
    _ensure_model_loaded()
    df = _trips_to_dataframe([trip])
    try:
        preds = predict(
            _state["model"], df,
            feature_columns=_state["feature_columns"],
            run_fe=True,
        )
    except Exception as e:
        logger.exception("Error en /predict")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error de inferencia: {e}",
        ) from e
    return PredictionOutput(
        estimated_total_amount=float(round(preds[0], 2)),
        model_version=_state.get("model_version"),
    )


@app.post("/predict/batch", response_model=BatchOutput)
def predict_batch(payload: BatchInput) -> BatchOutput:
    """Predice una lista de viajes en una sola llamada (más eficiente)."""
    _ensure_model_loaded()
    if not payload.trips:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="`trips` no puede estar vacío.",
        )
    df = _trips_to_dataframe(payload.trips)
    try:
        preds = predict(
            _state["model"], df,
            feature_columns=_state["feature_columns"],
            run_fe=True,
        )
    except Exception as e:
        logger.exception("Error en /predict/batch")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error de inferencia: {e}",
        ) from e
    return BatchOutput(
        predictions=[float(round(p, 2)) for p in preds],
        model_version=_state.get("model_version"),
    )
