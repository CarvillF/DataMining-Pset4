"""
Módulo para limpieza de datos y feature engineering.

Contiene las funciones prototipiadas en los notebooks 02 y 03,
refactorizadas como código modular de producción.

Flujo esperado:
    1. clean_dataframe()  →  aplica limpieza completa (leakage, outliers, nulos)
    2. run_feature_engineering()  →  crea features temporales, geográficas e interacciones
    3. build_preprocessor_for_trees() / build_preprocessor_for_linear()  →  pipeline sklearn

Uso desde train_model.py:
    from src.features.build_features import (
        clean_dataframe,
        run_feature_engineering,
        build_preprocessor_for_trees,
        FEATURE_CONFIG,
    )
"""

import pandas as pd
import numpy as np
from typing import Optional

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer


# ─────────────────────────────────────────────────────────────
# CONFIGURACIÓN CENTRAL DE FEATURES
# ─────────────────────────────────────────────────────────────

# Columnas que provocan Data Leakage: componentes del target
# o información solo disponible al FINALIZAR el viaje.
LEAKAGE_COLUMNS = [
    "fare_amount",
    "extra",
    "mta_tax",
    "tip_amount",
    "tolls_amount",
    "improvement_surcharge",
    "congestion_surcharge",
    "airport_fee",
    "tip_pct",
    "dropoff_datetime",
    "dropoff_date",
    "dropoff_hour",
    "trip_duration_min",
    "avg_speed_mph",
    "run_id",
    "ingested_at_utc",
]

# Definición de grupos de features para el ColumnTransformer
FEATURE_CONFIG = {
    "target": "total_amount",

    # Variables numéricas continuas → imputación mediana
    "numeric_features": [
        "trip_distance",
        "passenger_count",
        "pickup_hour_sin",
        "pickup_hour_cos",
        "day_of_week_sin",
        "day_of_week_cos",
        "month_sin",
        "month_cos",
        "distance_per_passenger",
    ],

    # IDs de localización → pasar directo (ordinales naturales para árboles)
    "location_features": [
        "pu_location_id",
        "do_location_id",
    ],

    # Flags binarios 0/1 → pasar directo
    "binary_features": [
        "is_weekend",
        "is_rush_hour",
        "is_night",
        "is_holiday",
        "is_airport_trip",
        "is_jfk",
        "is_newark",
        "same_borough",
        "is_inter_borough",
        "is_manhattan_origin",
        "is_long_trip",
        "is_short_trip",
        "rush_airport",
        "night_airport",
        "night_manhattan",
    ],

    # Categóricas de baja cardinalidad → OrdinalEncoder (para trees) / OHE (para linear)
    "cat_low_card": [
        "vendor_name",
        "rate_code_desc",
        "payment_type_desc",
        "pu_borough",
        "do_borough",
        "source_service",
    ],

    # Categóricas de alta cardinalidad → OrdinalEncoder con unknown=-1
    "cat_high_card": [
        "pu_zone",
        "do_zone",
    ],
}

# Festivos federales de NYC (formato MM-DD)
NYC_HOLIDAYS = {"01-01", "07-04", "11-11", "12-25", "12-31"}


# ─────────────────────────────────────────────────────────────
# LIMPIEZA (notebook 02)
# ─────────────────────────────────────────────────────────────

def remove_leakage_columns(
    df: pd.DataFrame,
    leakage_cols: Optional[list] = None,
) -> pd.DataFrame:
    """
    Elimina las columnas que constituyen Data Leakage.

    Args:
        df: DataFrame con columnas de la OBT.
        leakage_cols: Lista de columnas a eliminar. Por defecto usa LEAKAGE_COLUMNS.

    Returns:
        DataFrame sin columnas de leakage.
    """
    if leakage_cols is None:
        leakage_cols = LEAKAGE_COLUMNS
    cols_to_drop = [c for c in leakage_cols if c in df.columns]
    return df.drop(columns=cols_to_drop)


def apply_business_filters(
    df: pd.DataFrame,
    amount_min: float = 2.50,
    amount_max: float = 500.0,
    distance_max: float = 200.0,
    passenger_max: int = 6,
) -> pd.DataFrame:
    """
    Aplica filtros basados en lógica de negocio del dominio NYC Taxi.

    Args:
        df: DataFrame post-eliminación de leakage.
        amount_min: Mínimo de total_amount (tarifa mínima NYC = $2.50).
        amount_max: Máximo razonable de total_amount.
        distance_max: Máximo de trip_distance en millas.
        passenger_max: Máximo de pasajeros (capacidad de taxi NYC = 6).

    Returns:
        DataFrame filtrado.
    """
    mask = pd.Series(True, index=df.index)

    mask &= df["total_amount"] > amount_min
    mask &= df["total_amount"] <= amount_max

    if "trip_distance" in df.columns:
        mask &= df["trip_distance"] > 0
        mask &= df["trip_distance"] <= distance_max

    if "passenger_count" in df.columns:
        mask &= df["passenger_count"].between(1, passenger_max)

    if "pickup_datetime" in df.columns:
        mask &= df["pickup_datetime"].notna()

    return df[mask].copy()


def handle_nulls(df: pd.DataFrame) -> pd.DataFrame:
    """
    Maneja valores nulos con estrategias por tipo de columna.

    Estrategias:
        - passenger_count: mediana
        - rate_code_id: 1 (Standard rate)
        - Categóricas de texto: 'Unknown'
        - Geográficas: 'Unknown'

    Args:
        df: DataFrame filtrado.

    Returns:
        DataFrame sin nulos relevantes.
    """
    df = df.copy()

    if "passenger_count" in df.columns:
        df["passenger_count"] = df["passenger_count"].fillna(
            df["passenger_count"].median()
        )

    if "rate_code_id" in df.columns:
        df["rate_code_id"] = df["rate_code_id"].fillna(1)

    str_cols = [
        "rate_code_desc", "payment_type_desc", "vendor_name",
        "pu_zone", "do_zone", "pu_borough", "do_borough",
        "source_service", "service_type",
    ]
    for col in str_cols:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")

    return df


def group_rare_categories(
    df: pd.DataFrame,
    col: str,
    top_n: int = 50,
    other_label: str = "Other",
) -> pd.DataFrame:
    """
    Agrupa categorías poco frecuentes en 'Other' para reducir cardinalidad.

    Args:
        df: DataFrame.
        col: Columna categórica a agrupar.
        top_n: Número de categorías top a conservar.
        other_label: Etiqueta para las categorías agrupadas.

    Returns:
        DataFrame con la columna modificada.
    """
    if col not in df.columns:
        return df
    df = df.copy()
    top_cats = df[col].value_counts().nlargest(top_n).index
    df[col] = df[col].where(df[col].isin(top_cats), other=other_label)
    return df


def cast_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optimiza tipos de datos para reducir uso de memoria.

    Conversiones:
        - pickup_datetime → datetime64
        - Enteros seleccionados → int32
        - Flotantes → float32
        - Categóricas de baja cardinalidad → category

    Args:
        df: DataFrame a optimizar.

    Returns:
        DataFrame con tipos optimizados.
    """
    df = df.copy()

    if "pickup_datetime" in df.columns:
        df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"], errors="coerce")

    int_cols = [
        "passenger_count", "pu_location_id", "do_location_id",
        "rate_code_id", "pickup_hour", "day_of_week", "month", "year",
    ]
    for col in int_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype("int32")

    float_cols = ["trip_distance", "total_amount"]
    for col in float_cols:
        if col in df.columns:
            df[col] = df[col].astype("float32")

    low_card_cats = [
        "vendor_name", "rate_code_desc", "payment_type_desc",
        "pu_borough", "do_borough", "source_service", "service_type",
    ]
    for col in low_card_cats:
        if col in df.columns:
            df[col] = df[col].astype("category")

    return df


def clean_dataframe(
    df: pd.DataFrame,
    leakage_cols: Optional[list] = None,
    amount_min: float = 2.50,
    amount_max: float = 500.0,
) -> pd.DataFrame:
    """
    Pipeline completo de limpieza de datos.

    Encadena:
      1. remove_leakage_columns()
      2. apply_business_filters()
      3. handle_nulls()
      4. group_rare_categories() para zonas de alta cardinalidad
      5. cast_dtypes()

    Args:
        df: DataFrame crudo de la OBT de Snowflake.
        leakage_cols: Columnas a eliminar por leakage. Por defecto: LEAKAGE_COLUMNS.
        amount_min: Mínimo de total_amount válido.
        amount_max: Máximo de total_amount válido.

    Returns:
        DataFrame limpio listo para feature engineering.
    """
    df = remove_leakage_columns(df, leakage_cols)
    df = apply_business_filters(df, amount_min, amount_max)
    df = handle_nulls(df)

    for zone_col in ["pu_zone", "do_zone"]:
        df = group_rare_categories(df, zone_col, top_n=50)

    df = cast_dtypes(df)
    return df


# ─────────────────────────────────────────────────────────────
# FEATURE ENGINEERING (notebook 03)
# ─────────────────────────────────────────────────────────────

def create_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea variables temporales a partir de pickup_datetime.

    Nuevas columnas:
        - is_weekend (0/1)
        - is_rush_hour (0/1): 7-9 AM y 4-7 PM en días de semana
        - is_night (0/1): 8 PM - 6 AM (tarifa nocturna NYC)
        - is_holiday (0/1): días festivos nacionales en NYC
        - week_of_month (1-5)
        - pickup_hour_sin / pickup_hour_cos: encoding cíclico
        - day_of_week_sin / day_of_week_cos: encoding cíclico
        - month_sin / month_cos: encoding cíclico
    """
    df = df.copy()

    if "pickup_datetime" in df.columns:
        df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"], errors="coerce")
        dt = df["pickup_datetime"].dt

        if "pickup_hour" not in df.columns:
            df["pickup_hour"] = dt.hour
        if "day_of_week" not in df.columns:
            df["day_of_week"] = dt.dayofweek
        if "month" not in df.columns:
            df["month"] = dt.month

        # Festivos
        month_day = dt.strftime("%m-%d")
        df["is_holiday"] = month_day.isin(NYC_HOLIDAYS).astype("int8")

        # Semana del mes
        df["week_of_month"] = ((dt.day - 1) // 7 + 1).astype("int8")
    else:
        df["is_holiday"] = 0
        df["week_of_month"] = 1

    # Flags binarios
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype("int8")

    rush_am = df["pickup_hour"].between(7, 9)
    rush_pm = df["pickup_hour"].between(16, 19)
    weekday = ~df["is_weekend"].astype(bool)
    df["is_rush_hour"] = ((rush_am | rush_pm) & weekday).astype("int8")

    df["is_night"] = ((df["pickup_hour"] >= 20) | (df["pickup_hour"] < 6)).astype("int8")

    # Encoding cíclico (preserva continuidad periódica)
    df["pickup_hour_sin"] = np.sin(2 * np.pi * df["pickup_hour"] / 24)
    df["pickup_hour_cos"] = np.cos(2 * np.pi * df["pickup_hour"] / 24)

    df["day_of_week_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["day_of_week_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    return df


def create_geographic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea variables geográficas a partir de boroughs, zonas y rate_code.

    Nuevas columnas:
        - is_airport_trip (0/1)
        - is_jfk (0/1): tarifa fija $52
        - is_newark (0/1)
        - same_borough (0/1)
        - is_inter_borough (0/1)
        - is_manhattan_origin (0/1)
    """
    df = df.copy()

    AIRPORT_ZONES = {"JFK Airport", "LaGuardia Airport", "Newark Airport", "JFK", "LGA", "EWR"}

    airport_by_zone = pd.Series(False, index=df.index)
    if "pu_zone" in df.columns:
        airport_by_zone |= df["pu_zone"].isin(AIRPORT_ZONES)
    if "do_zone" in df.columns:
        airport_by_zone |= df["do_zone"].isin(AIRPORT_ZONES)

    airport_by_rate = pd.Series(False, index=df.index)
    if "rate_code_id" in df.columns:
        airport_by_rate = df["rate_code_id"].isin([2, 3])

    df["is_airport_trip"] = (airport_by_zone | airport_by_rate).astype("int8")

    df["is_jfk"] = 0
    if "rate_code_id" in df.columns:
        df["is_jfk"] = (df["rate_code_id"] == 2).astype("int8")

    df["is_newark"] = 0
    if "rate_code_id" in df.columns:
        df["is_newark"] = (df["rate_code_id"] == 3).astype("int8")

    if "pu_borough" in df.columns and "do_borough" in df.columns:
        df["same_borough"] = (df["pu_borough"] == df["do_borough"]).astype("int8")
        df["is_inter_borough"] = (~(df["pu_borough"] == df["do_borough"])).astype("int8")

    if "pu_borough" in df.columns:
        df["is_manhattan_origin"] = (df["pu_borough"] == "Manhattan").astype("int8")

    return df


def create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea ratios e interacciones entre variables existentes.

    Nuevas columnas:
        - distance_per_passenger: proxy de costo por persona
        - is_long_trip (0/1): viaje > 10 millas
        - is_short_trip (0/1): viaje < 1 milla
        - rush_airport: rush_hour AND airport
        - night_airport: is_night AND airport
        - night_manhattan: is_night AND manhattan_origin
    """
    df = df.copy()

    if "trip_distance" in df.columns and "passenger_count" in df.columns:
        df["distance_per_passenger"] = (
            df["trip_distance"] / df["passenger_count"].clip(lower=1)
        ).astype("float32")

    if "trip_distance" in df.columns:
        df["is_long_trip"] = (df["trip_distance"] > 10).astype("int8")
        df["is_short_trip"] = (df["trip_distance"] < 1).astype("int8")

    if "is_rush_hour" in df.columns and "is_airport_trip" in df.columns:
        df["rush_airport"] = (df["is_rush_hour"] & df["is_airport_trip"]).astype("int8")

    if "is_night" in df.columns and "is_airport_trip" in df.columns:
        df["night_airport"] = (df["is_night"] & df["is_airport_trip"]).astype("int8")

    if "is_night" in df.columns and "is_manhattan_origin" in df.columns:
        df["night_manhattan"] = (df["is_night"] & df["is_manhattan_origin"]).astype("int8")

    return df


def run_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pipeline completo de feature engineering.

    Encadena:
      1. create_temporal_features()
      2. create_geographic_features()
      3. create_interaction_features()

    Args:
        df: DataFrame limpio del pipeline clean_dataframe().

    Returns:
        DataFrame con todas las features generadas, listo para el ColumnTransformer.
    """
    df = create_temporal_features(df)
    df = create_geographic_features(df)
    df = create_interaction_features(df)
    return df


# ─────────────────────────────────────────────────────────────
# PREPROCESSORS (sklearn pipelines)
# ─────────────────────────────────────────────────────────────

def build_preprocessor_for_trees(
    numeric_features: Optional[list] = None,
    location_features: Optional[list] = None,
    binary_features: Optional[list] = None,
    cat_low_card: Optional[list] = None,
    cat_high_card: Optional[list] = None,
) -> ColumnTransformer:
    """
    Construye un ColumnTransformer optimizado para modelos de árbol y boosting.

    Estrategia por tipo:
        - Numéricas: imputación mediana (sin escalado; los árboles son invariantes a escala)
        - IDs de localización: passthrough
        - Binarias (0/1): passthrough
        - Categóricas baja cardinalidad: OrdinalEncoder (unknown=-1)
        - Categóricas alta cardinalidad: OrdinalEncoder (unknown=-1)

    Nota: XGBoost, LightGBM y CatBoost NO necesitan StandardScaler.
          Escalar variables antes de pasarlas a estos modelos no mejora
          el rendimiento y agrega cómputo innecesario.

    Args:
        Listas de nombres de columnas para cada tipo.
        Si son None, usa los valores de FEATURE_CONFIG.

    Returns:
        ColumnTransformer listo para .fit() / .transform().
    """
    if numeric_features is None:
        numeric_features = FEATURE_CONFIG["numeric_features"]
    if location_features is None:
        location_features = FEATURE_CONFIG["location_features"]
    if binary_features is None:
        binary_features = FEATURE_CONFIG["binary_features"]
    if cat_low_card is None:
        cat_low_card = FEATURE_CONFIG["cat_low_card"]
    if cat_high_card is None:
        cat_high_card = FEATURE_CONFIG["cat_high_card"]

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("encoder", OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )),
    ])

    transformers = []
    if numeric_features:
        transformers.append(("num", numeric_pipe, numeric_features))
    if location_features:
        transformers.append(("loc", "passthrough", location_features))
    if binary_features:
        transformers.append(("bin", "passthrough", binary_features))
    if cat_low_card:
        transformers.append(("cat_low", cat_pipe, cat_low_card))
    if cat_high_card:
        transformers.append(("cat_high", cat_pipe, cat_high_card))

    return ColumnTransformer(transformers=transformers, remainder="drop")


def build_preprocessor_for_linear(
    numeric_features: Optional[list] = None,
    binary_features: Optional[list] = None,
    cat_low_card: Optional[list] = None,
) -> ColumnTransformer:
    """
    Construye un ColumnTransformer para modelos lineales (baseline de regresión).

    Estrategia:
        - Numéricas: imputación mediana + StandardScaler
        - Binarias: passthrough
        - Categóricas: imputación + OneHotEncoder (drop='first' para evitar multicolinealidad)

    Args:
        Listas de nombres de columnas para cada tipo.
        Si son None, usa los valores de FEATURE_CONFIG.

    Returns:
        ColumnTransformer listo para .fit() / .transform().
    """
    if numeric_features is None:
        numeric_features = FEATURE_CONFIG["numeric_features"]
    if binary_features is None:
        binary_features = FEATURE_CONFIG["binary_features"]
    if cat_low_card is None:
        cat_low_card = FEATURE_CONFIG["cat_low_card"]

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("encoder", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
            drop="first",
        )),
    ])

    transformers = []
    if numeric_features:
        transformers.append(("num", numeric_pipe, numeric_features))
    if binary_features:
        transformers.append(("bin", "passthrough", binary_features))
    if cat_low_card:
        transformers.append(("cat", cat_pipe, cat_low_card))

    return ColumnTransformer(transformers=transformers, remainder="drop")


def get_all_feature_columns(df: pd.DataFrame) -> list:
    """
    Retorna la lista de todas las features disponibles en df
    según FEATURE_CONFIG, filtrando solo las que existen en el DataFrame.

    Útil para asegurar que el orden y existencia de columnas es correcto
    antes de pasarlas al ColumnTransformer.

    Args:
        df: DataFrame post-feature-engineering.

    Returns:
        Lista de nombres de columnas de features.
    """
    all_cols = (
        FEATURE_CONFIG["numeric_features"]
        + FEATURE_CONFIG["location_features"]
        + FEATURE_CONFIG["binary_features"]
        + FEATURE_CONFIG["cat_low_card"]
        + FEATURE_CONFIG["cat_high_card"]
    )
    return [c for c in all_cols if c in df.columns]
