"""
Inferencia: carga del Pipeline serializado y predicción sobre nuevos viajes.

El Pipeline guardado por `train_model.py` ya contiene preprocessor +
TransformedTargetRegressor (log1p / expm1) + el estimador final, así que el
único trabajo aquí es:

    1. Cargar el .pkl + el feature_config.json.
    2. Aplicar `run_feature_engineering` al input crudo (ya viene del usuario
       en formato OBT-row, sin componentes de cierre del viaje).
    3. Alinear columnas según `feature_columns` y llamar a `model.predict`.
    4. Recortar predicciones a >= 0 (las tarifas no son negativas).

Uso desde la API:

    >>> from src.models.predict_model import load_artifacts, predict
    >>> model, cfg = load_artifacts()
    >>> preds = predict(model, df_input, feature_columns=cfg["feature_columns"])
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional, Union

import joblib
import numpy as np
import pandas as pd

from src.features.build_features import run_feature_engineering
from src.utils.config import MODEL_PATH, MODELS_DIR

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# CARGA DE ARTEFACTOS
# ─────────────────────────────────────────────────────────────

def load_model(model_path: Optional[Union[str, Path]] = None):
    """
    Carga el Pipeline serializado.

    Args:
        model_path: ruta al .pkl. Si es None, usa `config.MODEL_PATH`.

    Returns:
        Pipeline de sklearn listo para `.predict()`.
    """
    path = Path(model_path) if model_path is not None else Path(MODEL_PATH)
    if not path.exists():
        raise FileNotFoundError(
            f"No se encontró modelo en {path}. "
            f"Ejecuta `python -m src.models.train_model --mode sample` primero."
        )
    logger.info("Cargando modelo desde %s", path)
    return joblib.load(path)


def load_feature_config(config_path: Optional[Union[str, Path]] = None) -> dict:
    """
    Carga la configuración de features (lista de columnas, target, metadata)
    persistida por `train_model.save_artifacts`.
    """
    if config_path is None:
        config_path = Path(MODELS_DIR) / "feature_config.json"
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(
            f"No se encontró feature_config.json en {config_path}."
        )
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_artifacts(
    model_path: Optional[Union[str, Path]] = None,
    config_path: Optional[Union[str, Path]] = None,
) -> tuple:
    """Conveniencia: carga modelo + config en un solo paso."""
    return load_model(model_path), load_feature_config(config_path)


# ─────────────────────────────────────────────────────────────
# PREPARACIÓN DE INPUT
# ─────────────────────────────────────────────────────────────

def prepare_input(
    raw_df: pd.DataFrame,
    feature_columns: list[str],
    run_fe: bool = True,
) -> pd.DataFrame:
    """
    Aplica feature engineering y alinea columnas al esquema esperado por el modelo.

    Args:
        raw_df: DataFrame con las columnas mínimas que vienen del usuario o de
                un nuevo extracto de Snowflake (post-limpieza estructural).
        feature_columns: lista exacta de columnas que el preprocessor espera.
        run_fe: si True (default), aplica `run_feature_engineering`. Si los
                inputs ya tienen las features creadas, pasar False.

    Returns:
        DataFrame con exactamente las columnas en `feature_columns`,
        en el mismo orden, listo para el preprocessor del Pipeline.
    """
    df = raw_df.copy()
    if run_fe:
        df = run_feature_engineering(df)

    # Crear columnas faltantes con NaN para que el imputador del pipeline
    # haga su trabajo, y descartar columnas que el modelo no espera.
    for col in feature_columns:
        if col not in df.columns:
            df[col] = np.nan
    return df[feature_columns]


# ─────────────────────────────────────────────────────────────
# PREDICCIÓN
# ─────────────────────────────────────────────────────────────

def predict(
    model,
    raw_df: pd.DataFrame,
    feature_columns: list[str],
    run_fe: bool = True,
    clip_negative: bool = True,
) -> np.ndarray:
    """
    Pipeline de inferencia end-to-end.

    Args:
        model: Pipeline cargado con `load_model`.
        raw_df: una o varias filas con campos crudos del viaje.
        feature_columns: del feature_config.json.
        run_fe: aplicar feature engineering antes (default True).
        clip_negative: si True, recorta predicciones < 0 a 0 (tarifa no negativa).

    Returns:
        np.ndarray con las predicciones de `total_amount` en USD.
    """
    X = prepare_input(raw_df, feature_columns, run_fe=run_fe)
    preds = model.predict(X)
    if clip_negative:
        preds = np.maximum(preds, 0.0)
    return np.asarray(preds, dtype=float)
