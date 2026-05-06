"""
Smoke tests end-to-end del pipeline de modelado, predicción y API.

NO tocan Snowflake. Usan un DataFrame sintético que simula la OBT,
entrenan un Pipeline ligero (LightGBM o GBDT como fallback), serializan
los artefactos y verifican que la API y predict_model funcionan.

Cómo ejecutar:
    pytest tests/test_train_predict_api.py -v
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.features.build_features import (  # noqa: E402
    FEATURE_CONFIG,
    clean_dataframe,
    get_all_feature_columns,
    run_feature_engineering,
)
from src.models.train_model import (  # noqa: E402
    build_pipeline,
    evaluate_pipeline,
    save_artifacts,
    split_xy,
)
from src.models.predict_model import (  # noqa: E402
    load_artifacts,
    predict,
    prepare_input,
)


# ─────────────────────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────────────────────

def _has_lightgbm() -> bool:
    try:
        import lightgbm  # noqa: F401
        return True
    except ImportError:
        return False


@pytest.fixture
def synthetic_obt() -> pd.DataFrame:
    """
    DataFrame de ~1500 filas que simula la OBT post-limpieza-Snowflake.
    El target es una función (con ruido) de trip_distance, hora y aeropuerto,
    para que cualquier modelo razonable aprenda algo y el RMSE no sea 0.
    """
    rng = np.random.default_rng(42)
    n = 1500
    base = pd.Timestamp("2022-01-01")
    pickup = base + pd.to_timedelta(rng.integers(0, 60 * 60 * 24 * 365, size=n), unit="s")
    distance = rng.exponential(scale=4.0, size=n).clip(0.1, 100)
    rate_codes = rng.choice([1, 1, 1, 1, 2, 3], size=n)  # mayoría Standard
    boroughs = rng.choice(
        ["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"], size=n,
        p=[0.55, 0.2, 0.15, 0.07, 0.03],
    )

    # Target sintético: tarifa base + por milla + recargo aeropuerto + ruido
    base_fare = 3.0
    per_mile = 2.5
    airport_surcharge = np.where(np.isin(rate_codes, [2, 3]), 12.0, 0.0)
    noise = rng.normal(0, 1.5, n)
    total = base_fare + per_mile * distance + airport_surcharge + noise
    total = np.maximum(total, 3.0)

    df = pd.DataFrame({
        "pickup_datetime": pickup,
        "pu_location_id": rng.integers(1, 265, n),
        "do_location_id": rng.integers(1, 265, n),
        "vendor_name": rng.choice(["Curb Mobility, LLC", "Creative Mobile Technologies, LLC"], n),
        "rate_code_id": rate_codes,
        "rate_code_desc": np.where(
            rate_codes == 2, "JFK",
            np.where(rate_codes == 3, "Newark", "Standard rate"),
        ),
        "payment_type_desc": rng.choice(["Flex Fare trip", "Cash"], n),
        "passenger_count": rng.integers(1, 5, n),
        "trip_distance": distance,
        "total_amount": total,
        "month": pd.to_datetime(pickup).month,
        "year": pd.to_datetime(pickup).year,
        "source_service": rng.choice(["yellow", "green"], n),
        "pu_zone": rng.choice([f"Zone_{i}" for i in range(20)], n),
        "do_zone": rng.choice([f"Zone_{i}" for i in range(20)], n),
        "pu_borough": boroughs,
        "do_borough": rng.choice(
            ["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"], size=n,
        ),
        "pickup_date": pd.to_datetime(pickup).date,
        "pickup_hour": pd.to_datetime(pickup).hour,
        "day_of_week": pd.to_datetime(pickup).dayofweek,
    })
    return df


@pytest.fixture
def engineered_df(synthetic_obt) -> pd.DataFrame:
    """Sintético post clean + FE (igual que el flujo de producción)."""
    df = clean_dataframe(synthetic_obt)
    df = run_feature_engineering(df)
    return df.sort_values("pickup_datetime").reset_index(drop=True)


@pytest.fixture
def trained_pipeline(engineered_df, tmp_path):
    """Pipeline entrenado + artefactos serializados en tmp_path."""
    X, y, feature_columns = split_xy(engineered_df)

    # Usar un modelo rápido y sin deps externas si lightgbm no está
    model_name = "lightgbm" if _has_lightgbm() else "gbdt"
    overrides = {"n_estimators": 50} if model_name == "lightgbm" else {"n_estimators": 30}
    pipeline = build_pipeline(model_name, use_log_target=True, **overrides)
    pipeline.fit(X, y)

    metadata = {
        "model_name": model_name,
        "model_version": "test-v1",
        "n_train_rows": len(X),
        "n_features": len(feature_columns),
    }
    model_path = tmp_path / "price_model.pkl"
    save_artifacts(pipeline, feature_columns, metadata, model_path=model_path)
    return pipeline, feature_columns, model_path, tmp_path


# ─────────────────────────────────────────────────────────────
# TESTS: TRAIN_MODEL
# ─────────────────────────────────────────────────────────────

class TestBuildPipeline:
    def test_lightgbm_pipeline_builds(self):
        if not _has_lightgbm():
            pytest.skip("lightgbm no instalado")
        pipe = build_pipeline("lightgbm", use_log_target=True)
        assert "preprocessor" in pipe.named_steps
        assert "model" in pipe.named_steps

    def test_gbdt_pipeline_builds(self):
        pipe = build_pipeline("gbdt", use_log_target=True)
        assert pipe is not None

    def test_unsupported_model_raises(self):
        with pytest.raises(ValueError, match="Modelo no soportado"):
            build_pipeline("nonexistent_model")

    def test_log_target_is_applied(self):
        pipe = build_pipeline("gbdt", use_log_target=True)
        # El step "model" debe ser TransformedTargetRegressor
        from sklearn.compose import TransformedTargetRegressor
        assert isinstance(pipe.named_steps["model"], TransformedTargetRegressor)


class TestTrainEndToEnd:
    def test_pipeline_fits_and_predicts(self, engineered_df):
        X, y, _ = split_xy(engineered_df)
        model_name = "lightgbm" if _has_lightgbm() else "gbdt"
        overrides = {"n_estimators": 30}
        pipeline = build_pipeline(model_name, use_log_target=True, **overrides)
        pipeline.fit(X, y)
        preds = pipeline.predict(X)
        assert preds.shape == (len(X),)
        assert np.isfinite(preds).all()

    def test_evaluate_returns_metrics(self, engineered_df):
        X, y, _ = split_xy(engineered_df)
        pipeline = build_pipeline("gbdt", use_log_target=True, n_estimators=30)
        pipeline.fit(X, y)
        metrics = evaluate_pipeline(pipeline, X, y, label="train")
        assert "rmse" in metrics
        assert "mae" in metrics
        assert "r2" in metrics
        # Con un target sintético claramente correlacionado, RMSE debe ser bajo
        assert metrics["rmse"] < 10.0
        assert metrics["r2"] > 0.5

    def test_save_artifacts_writes_files(self, trained_pipeline):
        _, feature_columns, model_path, tmp_path = trained_pipeline
        assert model_path.exists()
        config_path = tmp_path / "feature_config.json"
        assert config_path.exists()
        with open(config_path) as f:
            cfg = json.load(f)
        assert cfg["target"] == FEATURE_CONFIG["target"]
        assert cfg["feature_columns"] == feature_columns


# ─────────────────────────────────────────────────────────────
# TESTS: PREDICT_MODEL
# ─────────────────────────────────────────────────────────────

class TestPredictModel:
    def test_load_artifacts_roundtrip(self, trained_pipeline):
        _, feature_columns, model_path, tmp_path = trained_pipeline
        config_path = tmp_path / "feature_config.json"
        model, cfg = load_artifacts(model_path=model_path, config_path=config_path)
        assert model is not None
        assert cfg["feature_columns"] == feature_columns

    def test_load_model_missing_raises(self, tmp_path):
        from src.models.predict_model import load_model
        with pytest.raises(FileNotFoundError):
            load_model(model_path=tmp_path / "no_existe.pkl")

    def test_prepare_input_aligns_columns(self, engineered_df, trained_pipeline):
        _, feature_columns, _, _ = trained_pipeline
        X = prepare_input(engineered_df, feature_columns, run_fe=False)
        assert list(X.columns) == feature_columns

    def test_prepare_input_creates_missing_columns(self, trained_pipeline):
        _, feature_columns, _, _ = trained_pipeline
        # DataFrame sin la mayoría de las features
        df = pd.DataFrame({
            "pickup_datetime": [pd.Timestamp("2024-01-15 14:30:00")],
            "trip_distance": [3.5],
            "passenger_count": [1],
        })
        X = prepare_input(df, feature_columns, run_fe=True)
        assert list(X.columns) == feature_columns
        assert len(X) == 1

    def test_predict_returns_array(self, engineered_df, trained_pipeline):
        pipeline, feature_columns, _, _ = trained_pipeline
        sample = engineered_df.head(5)
        preds = predict(pipeline, sample, feature_columns, run_fe=False)
        assert preds.shape == (5,)
        assert (preds >= 0).all()  # clip_negative=True por defecto

    def test_predict_on_raw_input(self, trained_pipeline):
        """Input mínimo (como el que vendría del frontend) debe funcionar."""
        pipeline, feature_columns, _, _ = trained_pipeline
        df = pd.DataFrame({
            "pickup_datetime": [pd.Timestamp("2024-06-15 14:30:00")],
            "trip_distance": [5.5],
            "passenger_count": [2],
            "pu_location_id": [161],
            "do_location_id": [237],
            "pu_borough": ["Manhattan"],
            "do_borough": ["Manhattan"],
            "pu_zone": ["Unknown"],
            "do_zone": ["Unknown"],
            "vendor_name": ["Curb Mobility, LLC"],
            "rate_code_id": [1],
            "rate_code_desc": ["Standard rate"],
            "payment_type_desc": ["Flex Fare trip"],
            "source_service": ["yellow"],
            "month": [6],
            "year": [2024],
            "pickup_hour": [14],
            "day_of_week": [5],
        })
        preds = predict(pipeline, df, feature_columns, run_fe=True)
        assert preds.shape == (1,)
        assert preds[0] > 0
        assert preds[0] < 200  # un viaje de 5.5 millas no debería costar > $200


# ─────────────────────────────────────────────────────────────
# TESTS: API (FastAPI con TestClient)
# ─────────────────────────────────────────────────────────────

class TestAPI:
    @pytest.fixture
    def api_client(self, trained_pipeline, monkeypatch):
        """
        Levanta la app FastAPI usando los artefactos del fixture trained_pipeline.
        Usa TestClient (sincrónico, sin servidor real).
        """
        from fastapi.testclient import TestClient

        _, _, model_path, tmp_path = trained_pipeline
        config_path = tmp_path / "feature_config.json"
        monkeypatch.setenv("MODEL_PATH", str(model_path))
        monkeypatch.setenv("FEATURE_CONFIG_PATH", str(config_path))

        # Importar después del setenv para que el lifespan use las rutas correctas
        from src.api import main as api_main
        # Resetear el estado por si otro test ya lo cargó
        api_main._state.clear()
        api_main._state.update({
            "model": None, "feature_columns": None,
            "model_version": None, "load_error": None,
        })

        with TestClient(api_main.app) as client:
            yield client

    def test_health_endpoint(self, api_client):
        r = api_client.get("/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "ok"
        assert data["model_loaded"] is True
        assert data["feature_count"] is not None

    def test_predict_endpoint(self, api_client):
        payload = {
            "pickup_datetime": "2024-06-15T14:30:00",
            "trip_distance": 5.5,
            "passenger_count": 2,
            "pu_location_id": 161,
            "do_location_id": 237,
            "pu_borough": "Manhattan",
            "do_borough": "Manhattan",
            "pu_zone": "Unknown",
            "do_zone": "Unknown",
            "vendor_name": "Curb Mobility, LLC",
            "rate_code_id": 1,
            "rate_code_desc": "Standard rate",
            "payment_type_desc": "Flex Fare trip",
            "source_service": "yellow",
        }
        r = api_client.post("/predict", json=payload)
        assert r.status_code == 200, r.text
        data = r.json()
        assert "estimated_total_amount" in data
        assert isinstance(data["estimated_total_amount"], float)
        assert data["estimated_total_amount"] > 0

    def test_predict_batch_endpoint(self, api_client):
        trip = {
            "pickup_datetime": "2024-06-15T14:30:00",
            "trip_distance": 5.5,
            "passenger_count": 2,
            "pu_location_id": 161,
            "do_location_id": 237,
            "pu_borough": "Manhattan",
            "do_borough": "Manhattan",
            "pu_zone": "Unknown",
            "do_zone": "Unknown",
            "vendor_name": "Curb Mobility, LLC",
            "rate_code_id": 1,
            "rate_code_desc": "Standard rate",
            "payment_type_desc": "Flex Fare trip",
            "source_service": "yellow",
        }
        r = api_client.post("/predict/batch", json={"trips": [trip, trip, trip]})
        assert r.status_code == 200, r.text
        data = r.json()
        assert len(data["predictions"]) == 3
        assert all(p > 0 for p in data["predictions"])

    def test_predict_validates_passenger_count(self, api_client):
        payload = {
            "pickup_datetime": "2024-06-15T14:30:00",
            "trip_distance": 5.5,
            "passenger_count": 9,  # inválido (>6)
            "pu_location_id": 161,
            "do_location_id": 237,
        }
        r = api_client.post("/predict", json=payload)
        assert r.status_code == 422  # Pydantic rechaza

    def test_predict_validates_trip_distance(self, api_client):
        payload = {
            "pickup_datetime": "2024-06-15T14:30:00",
            "trip_distance": -5.0,  # inválido (debe ser >0)
            "passenger_count": 2,
            "pu_location_id": 161,
            "do_location_id": 237,
        }
        r = api_client.post("/predict", json=payload)
        assert r.status_code == 422

    def test_predict_batch_empty_rejected(self, api_client):
        r = api_client.post("/predict/batch", json={"trips": []})
        assert r.status_code == 400
