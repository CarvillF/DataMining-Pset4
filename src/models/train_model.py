"""
Pipeline de entrenamiento productivo para el modelo de precios NYC Taxi.

Diseño:
    1. Carga datos desde Snowflake (sample para iteración rápida o batches OOC).
    2. Aplica clean_dataframe + run_feature_engineering (mismo pipeline que notebooks).
    3. Construye un sklearn Pipeline = ColumnTransformer + TransformedTargetRegressor(log1p).
    4. Entrena modelo (LightGBM por defecto; XGBoost / CatBoost / GBDT / AdaBoost disponibles).
    5. (Opcional) Hyperparameter tuning con RandomizedSearchCV o Optuna sobre TimeSeriesSplit.
    6. Evalúa en val_set 2024 (Snowflake) usando RMSE.
    7. Guarda el Pipeline completo en models/price_model.pkl + feature_config.json.

Uso:
    # Entrenamiento rápido sobre sample (default LightGBM, log-target, sin tuning)
    python -m src.models.train_model --mode sample --sample-prob 1.0

    # Hyperparameter tuning sobre sample
    python -m src.models.train_model --mode tune --tuner randomized --n-iter 30
    python -m src.models.train_model --mode tune --tuner optuna     --n-iter 50

    # Entrenamiento Out-of-Core (warm-start tras sample)
    python -m src.models.train_model --mode oof --batch-size 500000
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Iterator, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline

# `src.data.ingestion` requiere snowflake-connector-python. Importamos perezosamente
# dentro de las funciones que tocan Snowflake para que el resto del módulo
# (build_pipeline, evaluate_pipeline, save_artifacts, etc.) sea utilizable
# en entornos sin Snowflake (tests unitarios, CI, inferencia).
from src.features.build_features import (
    FEATURE_CONFIG,
    build_preprocessor_for_trees,
    clean_dataframe,
    get_all_feature_columns,
    run_feature_engineering,
)
from src.utils.config import (
    DEFAULT_BATCH_SIZE,
    MODEL_PATH,
    MODELS_DIR,
    RANDOM_STATE,
    setup_logging,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# QUERIES POR DEFECTO
# ─────────────────────────────────────────────────────────────

DEFAULT_TRAIN_QUERY = "SELECT * FROM NYC_TAXI_P5.ANALYTICS.train_set"
DEFAULT_VAL_QUERY = "SELECT * FROM NYC_TAXI_P5.ANALYTICS.val_set"
DEFAULT_TEST_QUERY = "SELECT * FROM NYC_TAXI_P5.ANALYTICS.test_set"


# ─────────────────────────────────────────────────────────────
# CARGA Y PREPARACIÓN DE DATOS
# ─────────────────────────────────────────────────────────────

def load_and_prepare(query: str, sample_prob: float = 1.0) -> pd.DataFrame:
    """
    Descarga muestra de Snowflake → limpieza → feature engineering.

    Args:
        query: SQL contra train_set / val_set / test_set.
        sample_prob: % de filas a traer (TABLESAMPLE pushdown).

    Returns:
        DataFrame con todas las features ingeniadas, listo para `Pipeline.fit`.
    """
    from src.data.ingestion import fetch_sample  # lazy: requiere snowflake-connector
    logger.info("Cargando muestra desde Snowflake (sample_prob=%.3f%%)...", sample_prob)
    df_raw = fetch_sample(query, sample_prob=sample_prob)
    logger.info("Aplicando clean_dataframe + run_feature_engineering...")
    df_clean = clean_dataframe(df_raw)
    df_fe = run_feature_engineering(df_clean)
    logger.info(
        "Datos preparados: %d filas × %d cols (%.1f MB)",
        len(df_fe), df_fe.shape[1],
        df_fe.memory_usage(deep=True).sum() / 1e6,
    )
    return df_fe


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Devuelve X, y, lista de feature columns disponibles."""
    feature_columns = get_all_feature_columns(df)
    target = FEATURE_CONFIG["target"]
    if target not in df.columns:
        raise ValueError(f"Target '{target}' no presente en el DataFrame.")
    if not feature_columns:
        raise ValueError("Ninguna feature de FEATURE_CONFIG presente en el DataFrame.")
    return df[feature_columns], df[target], feature_columns


# ─────────────────────────────────────────────────────────────
# CONSTRUCCIÓN DE PIPELINES
# ─────────────────────────────────────────────────────────────

def _build_regressor(model_name: str, **overrides: Any):
    """
    Instancia el regresor según el nombre. Los overrides pisan los defaults.

    Modelos soportados (rúbrica obligatoria + extras):
        adaboost, gbdt, xgboost, lightgbm, catboost
    """
    name = model_name.lower()

    if name == "lightgbm":
        import lightgbm as lgb
        params = dict(
            n_estimators=800,
            learning_rate=0.05,
            num_leaves=127,
            max_depth=-1,
            min_child_samples=50,
            subsample=0.85,
            subsample_freq=1,
            colsample_bytree=0.85,
            reg_alpha=0.1,
            reg_lambda=1.0,
            n_jobs=-1,
            random_state=RANDOM_STATE,
            verbosity=-1,
        )
        params.update(overrides)
        return lgb.LGBMRegressor(**params)

    if name == "xgboost":
        import xgboost as xgb
        params = dict(
            n_estimators=600,
            learning_rate=0.05,
            max_depth=7,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.1,
            reg_lambda=1.0,
            tree_method="hist",
            n_jobs=-1,
            random_state=RANDOM_STATE,
            verbosity=0,
        )
        params.update(overrides)
        return xgb.XGBRegressor(**params)

    if name == "catboost":
        from catboost import CatBoostRegressor
        params = dict(
            iterations=800,
            learning_rate=0.05,
            depth=8,
            l2_leaf_reg=3.0,
            border_count=128,
            random_seed=RANDOM_STATE,
            verbose=0,
            allow_writing_files=False,
        )
        params.update(overrides)
        return CatBoostRegressor(**params)

    if name == "gbdt":
        from sklearn.ensemble import GradientBoostingRegressor
        params = dict(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            min_samples_leaf=50,
            random_state=RANDOM_STATE,
        )
        params.update(overrides)
        return GradientBoostingRegressor(**params)

    if name == "adaboost":
        from sklearn.ensemble import AdaBoostRegressor
        from sklearn.tree import DecisionTreeRegressor
        params = dict(
            n_estimators=200,
            learning_rate=0.1,
            loss="linear",
            random_state=RANDOM_STATE,
        )
        params.update(overrides)
        return AdaBoostRegressor(
            estimator=DecisionTreeRegressor(max_depth=4),
            **params,
        )

    raise ValueError(
        f"Modelo no soportado: '{model_name}'. "
        "Usa uno de: lightgbm, xgboost, catboost, gbdt, adaboost."
    )


def build_pipeline(
    model_name: str = "lightgbm",
    use_log_target: bool = True,
    **regressor_overrides: Any,
) -> Pipeline:
    """
    Construye el Pipeline = preprocessor (trees) + (TransformedTargetRegressor) + regresor.

    El TransformedTargetRegressor con log1p / expm1 estabiliza la varianza
    del target (total_amount está fuertemente sesgado a la derecha) y suele
    bajar 5-15% el RMSE en datasets de tarifas.
    """
    preprocessor = build_preprocessor_for_trees()
    regressor = _build_regressor(model_name, **regressor_overrides)

    if use_log_target:
        regressor = TransformedTargetRegressor(
            regressor=regressor,
            func=np.log1p,
            inverse_func=np.expm1,
            check_inverse=False,
        )

    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", regressor),
    ])


# ─────────────────────────────────────────────────────────────
# EVALUACIÓN
# ─────────────────────────────────────────────────────────────

def evaluate_pipeline(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    label: str = "val",
) -> dict[str, float]:
    """Calcula RMSE / MAE / R² de un pipeline ya entrenado."""
    pred = pipeline.predict(X)
    pred = np.maximum(pred, 0.0)  # las tarifas no pueden ser negativas
    rmse = float(np.sqrt(mean_squared_error(y, pred)))
    mae = float(mean_absolute_error(y, pred))
    r2 = float(r2_score(y, pred))
    metrics = {"rmse": rmse, "mae": mae, "r2": r2}
    logger.info(
        "[%s] RMSE=$%.4f | MAE=$%.4f | R²=%.4f",
        label.upper(), rmse, mae, r2,
    )
    return metrics


# ─────────────────────────────────────────────────────────────
# HYPERPARAMETER TUNING
# ─────────────────────────────────────────────────────────────

def _lgbm_param_dist() -> dict:
    """Espacio de búsqueda razonable para LightGBM (RandomizedSearchCV)."""
    return {
        "model__regressor__num_leaves": [31, 63, 127, 255],
        "model__regressor__learning_rate": [0.02, 0.03, 0.05, 0.08],
        "model__regressor__min_child_samples": [20, 50, 100, 200],
        "model__regressor__reg_alpha": [0.0, 0.1, 1.0, 5.0],
        "model__regressor__reg_lambda": [0.0, 1.0, 5.0, 10.0],
        "model__regressor__subsample": [0.7, 0.85, 1.0],
        "model__regressor__colsample_bytree": [0.7, 0.85, 1.0],
        "model__regressor__n_estimators": [400, 600, 800, 1200],
    }


def tune_with_randomized_search(
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str = "lightgbm",
    n_iter: int = 30,
    cv_splits: int = 3,
    use_log_target: bool = True,
) -> tuple[Pipeline, float, dict]:
    """
    Tuning con RandomizedSearchCV sobre TimeSeriesSplit.
    Asume que (X, y) están ordenados temporalmente para que el CV sea válido.
    """
    if model_name.lower() != "lightgbm":
        raise NotImplementedError(
            "RandomizedSearchCV configurado sólo para LightGBM. "
            "Para otros modelos usa --tuner optuna."
        )
    pipeline = build_pipeline(model_name, use_log_target=use_log_target)
    param_dist = _lgbm_param_dist()

    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=TimeSeriesSplit(n_splits=cv_splits),
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=1,
        refit=True,
    )
    logger.info(
        "RandomizedSearchCV: n_iter=%d | TimeSeriesSplit=%d | scoring=neg_RMSE",
        n_iter, cv_splits,
    )
    t0 = time.time()
    search.fit(X, y)
    elapsed = time.time() - t0

    best_rmse = -float(search.best_score_)
    logger.info(
        "Búsqueda completada en %.1fs | mejor RMSE CV = $%.4f",
        elapsed, best_rmse,
    )
    logger.info("Mejores hiperparámetros: %s", search.best_params_)
    return search.best_estimator_, best_rmse, dict(search.best_params_)


def tune_with_optuna(
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str = "lightgbm",
    n_trials: int = 50,
    cv_splits: int = 3,
    use_log_target: bool = True,
) -> tuple[Pipeline, float, dict]:
    """Tuning bayesiano con Optuna (mejor sample efficiency que RandomizedSearch)."""
    try:
        import optuna
    except ImportError as e:
        raise ImportError(
            "optuna no instalado. Ejecuta `pip install optuna` o usa --tuner randomized."
        ) from e

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    name = model_name.lower()
    cv = TimeSeriesSplit(n_splits=cv_splits)

    def objective(trial: "optuna.trial.Trial") -> float:
        if name == "lightgbm":
            overrides = dict(
                num_leaves=trial.suggest_int("num_leaves", 31, 255),
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
                min_child_samples=trial.suggest_int("min_child_samples", 10, 300),
                reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
                reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
                subsample=trial.suggest_float("subsample", 0.6, 1.0),
                colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
                n_estimators=trial.suggest_int("n_estimators", 300, 1500, step=100),
            )
        elif name == "xgboost":
            overrides = dict(
                max_depth=trial.suggest_int("max_depth", 4, 10),
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
                min_child_weight=trial.suggest_float("min_child_weight", 1, 10),
                reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
                reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
                subsample=trial.suggest_float("subsample", 0.6, 1.0),
                colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
                n_estimators=trial.suggest_int("n_estimators", 300, 1500, step=100),
            )
        elif name == "catboost":
            overrides = dict(
                depth=trial.suggest_int("depth", 4, 10),
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
                l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
                iterations=trial.suggest_int("iterations", 300, 1500, step=100),
            )
        else:
            raise NotImplementedError(f"Optuna search no definido para {name}.")

        rmses = []
        for tr_idx, va_idx in cv.split(X):
            X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
            y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
            pipe = build_pipeline(name, use_log_target=use_log_target, **overrides)
            pipe.fit(X_tr, y_tr)
            pred = np.maximum(pipe.predict(X_va), 0.0)
            rmses.append(float(np.sqrt(mean_squared_error(y_va, pred))))
        return float(np.mean(rmses))

    study = optuna.create_study(direction="minimize", study_name=f"{name}_pricing")
    logger.info("Optuna: model=%s | n_trials=%d | TimeSeriesSplit=%d", name, n_trials, cv_splits)
    t0 = time.time()
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    elapsed = time.time() - t0
    logger.info(
        "Optuna terminó en %.1fs | mejor RMSE CV = $%.4f",
        elapsed, study.best_value,
    )
    logger.info("Mejores hiperparámetros: %s", study.best_params)

    best_pipeline = build_pipeline(name, use_log_target=use_log_target, **study.best_params)
    best_pipeline.fit(X, y)
    return best_pipeline, float(study.best_value), dict(study.best_params)


# ─────────────────────────────────────────────────────────────
# OUT-OF-CORE (warm start sobre todos los lotes del train_set)
# ─────────────────────────────────────────────────────────────

def train_out_of_core(
    pipeline: Pipeline,
    train_query: str = DEFAULT_TRAIN_QUERY,
    batch_size: int = DEFAULT_BATCH_SIZE,
    n_estimators_per_batch: int = 30,
) -> Pipeline:
    """
    Refina un Pipeline ya entrenado en sample iterando sobre todos los batches
    del train_set en Snowflake, agregando árboles con warm-start.

    Sólo soportado cuando el regresor envuelto es LightGBM o XGBoost (init_model
    / xgb_model). Para otros modelos hace re-fit por batch (no recomendado).

    Args:
        pipeline: Pipeline ya fit en sample.
        train_query: SQL contra el train_set completo.
        batch_size: filas por batch.
        n_estimators_per_batch: árboles a agregar por batch.

    Returns:
        El mismo Pipeline con el regresor refinado.
    """
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]

    # Desempaquetar TransformedTargetRegressor si aplica
    if isinstance(model, TransformedTargetRegressor):
        log_target = True
        regressor = model.regressor_
    else:
        log_target = False
        regressor = model

    cls_name = regressor.__class__.__name__
    logger.info("Out-of-Core training | regresor=%s | log_target=%s", cls_name, log_target)

    from src.data.ingestion import fetch_data_in_batches  # lazy: requiere snowflake-connector

    total_rows = 0
    batch_num = 0
    for batch_df in fetch_data_in_batches(train_query, batch_size=batch_size):
        batch_num += 1
        batch_clean = clean_dataframe(batch_df)
        if len(batch_clean) == 0:
            logger.info("Batch %d vacío post-limpieza, saltando.", batch_num)
            continue
        batch_fe = run_feature_engineering(batch_clean)
        feature_columns = get_all_feature_columns(batch_fe)
        X_batch = batch_fe[feature_columns]
        y_batch = batch_fe[FEATURE_CONFIG["target"]]
        if log_target:
            y_batch = np.log1p(y_batch.values)

        X_proc = preprocessor.transform(X_batch)

        if cls_name == "LGBMRegressor":
            import lightgbm as lgb
            booster = getattr(regressor, "booster_", None)
            params = regressor.get_params()
            train_data = lgb.Dataset(X_proc, label=y_batch, free_raw_data=False)
            booster = lgb.train(
                {
                    "objective": "regression",
                    "metric": "rmse",
                    "learning_rate": params.get("learning_rate", 0.05),
                    "num_leaves": params.get("num_leaves", 63),
                    "min_data_in_leaf": params.get("min_child_samples", 50),
                    "lambda_l1": params.get("reg_alpha", 0.1),
                    "lambda_l2": params.get("reg_lambda", 1.0),
                    "feature_fraction": params.get("colsample_bytree", 0.85),
                    "bagging_fraction": params.get("subsample", 0.85),
                    "verbosity": -1,
                },
                train_data,
                num_boost_round=n_estimators_per_batch,
                init_model=booster,
            )
            regressor._Booster = booster

        elif cls_name == "XGBRegressor":
            import xgboost as xgb
            booster = regressor.get_booster() if hasattr(regressor, "_Booster") else None
            params = {
                "objective": "reg:squarederror",
                "learning_rate": regressor.get_params().get("learning_rate", 0.05),
                "max_depth": regressor.get_params().get("max_depth", 6),
                "tree_method": "hist",
                "verbosity": 0,
            }
            dtrain = xgb.DMatrix(X_proc, label=y_batch)
            booster = xgb.train(
                params, dtrain,
                num_boost_round=n_estimators_per_batch,
                xgb_model=booster,
            )
            regressor._Booster = booster

        else:
            logger.warning(
                "Out-of-Core no implementado para %s; sólo LightGBM/XGBoost. "
                "Saltando batches.", cls_name,
            )
            break

        total_rows += len(X_batch)
        logger.info(
            "Batch %d | filas=%d | acumulado=%d", batch_num, len(X_batch), total_rows,
        )

    logger.info("OOC completado. %d filas | %d batches", total_rows, batch_num)
    # Re-empaquetar
    if isinstance(pipeline.named_steps["model"], TransformedTargetRegressor):
        pipeline.named_steps["model"].regressor_ = regressor
    else:
        pipeline.named_steps["model"] = regressor
    return pipeline


# ─────────────────────────────────────────────────────────────
# PERSISTENCIA
# ─────────────────────────────────────────────────────────────

def save_artifacts(
    pipeline: Pipeline,
    feature_columns: list[str],
    metadata: dict,
    model_path: Optional[Path] = None,
) -> tuple[Path, Path]:
    """Serializa el Pipeline + un JSON con la lista de features y metadata."""
    if model_path is None:
        model_path = MODEL_PATH
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(pipeline, model_path)
    logger.info("Pipeline guardado en %s", model_path)

    config_path = model_path.parent / "feature_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "target": FEATURE_CONFIG["target"],
                "feature_columns": feature_columns,
                "feature_groups": {
                    k: v for k, v in FEATURE_CONFIG.items() if k != "target"
                },
                "metadata": metadata,
            },
            f, indent=2, default=str,
        )
    logger.info("Configuración guardada en %s", config_path)
    return model_path, config_path


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["sample", "tune", "oof"], default="sample",
                   help="sample: entrenamiento rápido; tune: HP search; oof: out-of-core (incluye sample warm-start).")
    p.add_argument("--model", default="lightgbm",
                   choices=["lightgbm", "xgboost", "catboost", "gbdt", "adaboost"],
                   help="Modelo a entrenar.")
    p.add_argument("--train-query", default=DEFAULT_TRAIN_QUERY)
    p.add_argument("--val-query", default=DEFAULT_VAL_QUERY)
    p.add_argument("--sample-prob", type=float, default=1.0,
                   help="%% de filas a traer del train_set para sample/tune.")
    p.add_argument("--val-sample-prob", type=float, default=2.0,
                   help="%% de filas del val_set 2024 para evaluación final.")
    p.add_argument("--no-log-target", action="store_true",
                   help="Desactiva el log1p/expm1 sobre total_amount.")
    p.add_argument("--tuner", choices=["randomized", "optuna"], default="optuna")
    p.add_argument("--n-iter", type=int, default=30, help="Trials de tuning.")
    p.add_argument("--cv-splits", type=int, default=3, help="Splits del TimeSeriesSplit.")
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--ooc-rounds", type=int, default=30, help="Árboles a agregar por batch en OOC.")
    p.add_argument("--model-path", type=str, default=str(MODEL_PATH))
    p.add_argument("--skip-snowflake", action="store_true",
                   help="Saltar val/test en Snowflake (entrenamiento sólo, útil para CI).")
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    setup_logging()

    use_log_target = not args.no_log_target
    metadata: dict[str, Any] = {
        "model_name": args.model,
        "use_log_target": use_log_target,
        "mode": args.mode,
        "tuner": args.tuner if args.mode == "tune" else None,
        "random_state": RANDOM_STATE,
    }

    # 1) Cargar y preparar el sample de entrenamiento
    df_train = load_and_prepare(args.train_query, sample_prob=args.sample_prob)
    df_train = df_train.sort_values("pickup_datetime").reset_index(drop=True)
    X_train, y_train, feature_columns = split_xy(df_train)
    metadata["n_train_rows"] = len(X_train)
    metadata["n_features"] = len(feature_columns)

    # 2) Entrenar / tunear según modo
    t0 = time.time()
    if args.mode == "tune":
        if args.tuner == "randomized":
            pipeline, cv_rmse, best_params = tune_with_randomized_search(
                X_train, y_train,
                model_name=args.model,
                n_iter=args.n_iter,
                cv_splits=args.cv_splits,
                use_log_target=use_log_target,
            )
        else:
            pipeline, cv_rmse, best_params = tune_with_optuna(
                X_train, y_train,
                model_name=args.model,
                n_trials=args.n_iter,
                cv_splits=args.cv_splits,
                use_log_target=use_log_target,
            )
        metadata["best_cv_rmse"] = cv_rmse
        metadata["best_params"] = best_params
    else:
        pipeline = build_pipeline(args.model, use_log_target=use_log_target)
        logger.info("Entrenando %s sobre %d filas...", args.model, len(X_train))
        pipeline.fit(X_train, y_train)

    metadata["fit_time_sec"] = round(time.time() - t0, 2)

    # 3) Refinamiento OOC si corresponde
    if args.mode == "oof":
        logger.info("Iniciando refinamiento Out-of-Core...")
        pipeline = train_out_of_core(
            pipeline,
            train_query=args.train_query,
            batch_size=args.batch_size,
            n_estimators_per_batch=args.ooc_rounds,
        )

    # 4) Métricas en train (sanity check)
    train_metrics = evaluate_pipeline(pipeline, X_train, y_train, label="train")
    metadata["train_metrics"] = train_metrics

    # 5) Métricas en val_set 2024 (Snowflake)
    if not args.skip_snowflake:
        try:
            df_val = load_and_prepare(args.val_query, sample_prob=args.val_sample_prob)
            X_val, y_val, _ = split_xy(df_val)
            val_metrics = evaluate_pipeline(pipeline, X_val, y_val, label="val_2024")
            metadata["val_metrics"] = val_metrics
        except Exception as e:
            logger.warning("No se pudo evaluar en val_set: %s", e)
            metadata["val_metrics"] = {"error": str(e)}

    # 6) Guardar artefactos
    metadata["model_version"] = time.strftime("%Y%m%d_%H%M%S")
    model_path, config_path = save_artifacts(
        pipeline, feature_columns, metadata,
        model_path=Path(args.model_path),
    )

    logger.info("=" * 60)
    logger.info("ENTRENAMIENTO COMPLETADO")
    logger.info("Modelo:        %s", args.model)
    logger.info("Pipeline:      %s", model_path)
    logger.info("Feature cfg:   %s", config_path)
    if "val_metrics" in metadata and "rmse" in metadata.get("val_metrics", {}):
        logger.info("RMSE val 2024: $%.4f", metadata["val_metrics"]["rmse"])
    logger.info("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
