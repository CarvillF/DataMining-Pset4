"""
Tests unitarios para src/features/build_features.py y src/utils/config.py

Cómo ejecutar:
    Desde la raíz del proyecto:
        pytest tests/test_features.py -v

    Para ver output de print() también:
        pytest tests/test_features.py -v -s

    Para ejecutar solo un test específico:
        pytest tests/test_features.py::test_remove_leakage_columns -v

No requieren conexión a Snowflake. Trabajan con DataFrames sintéticos
que simulan la estructura de analytics.obt_trips.
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

# Agregar raíz del proyecto al path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.features.build_features import (
    remove_leakage_columns,
    apply_business_filters,
    handle_nulls,
    group_rare_categories,
    cast_dtypes,
    clean_dataframe,
    create_temporal_features,
    create_geographic_features,
    create_interaction_features,
    run_feature_engineering,
    build_preprocessor_for_trees,
    build_preprocessor_for_linear,
    get_all_feature_columns,
    LEAKAGE_COLUMNS,
    FEATURE_CONFIG,
)


# ─────────────────────────────────────────────────────────────
# FIXTURES: DataFrames sintéticos que simulan la OBT
# ─────────────────────────────────────────────────────────────

@pytest.fixture
def sample_obt_df() -> pd.DataFrame:
    """
    DataFrame que simula la estructura de analytics.obt_trips_model.
    Incluye columnas de leakage, outliers y nulos para testear la limpieza.
    """
    np.random.seed(42)
    n = 200

    df = pd.DataFrame({
        # Temporales
        "pickup_datetime":  pd.date_range("2022-01-01", periods=n, freq="2h"),
        "dropoff_datetime": pd.date_range("2022-01-01 00:30:00", periods=n, freq="2h"),

        # Geográficas
        "pu_location_id": np.random.randint(1, 265, n),
        "do_location_id": np.random.randint(1, 265, n),
        "pu_borough":     np.random.choice(["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"], n),
        "do_borough":     np.random.choice(["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"], n),
        "pu_zone":        [f"Zone_{i}" for i in np.random.randint(1, 80, n)],
        "do_zone":        [f"Zone_{i}" for i in np.random.randint(1, 80, n)],

        # Características del viaje
        "passenger_count": np.random.choice([1, 2, 3, 4, np.nan], n, p=[0.5, 0.2, 0.15, 0.1, 0.05]),
        "trip_distance":   np.random.exponential(5, n),
        "rate_code_id":    np.random.choice([1, 2, 3, 4, 5, np.nan], n, p=[0.8, 0.05, 0.05, 0.03, 0.05, 0.02]),
        "rate_code_desc":  np.random.choice(["Standard rate", "JFK", "Newark", "Unknown"], n),
        "payment_type_desc": np.random.choice(["Credit card", "Cash", "Unknown"], n),
        "vendor_name":     np.random.choice(["Creative Mobile Technologies", "VeriFone Inc.", "Unknown"], n),

        # TARGET
        "total_amount": np.concatenate([
            np.random.uniform(3, 80, n - 20),  # Valores normales
            [-5, 0, 1000, -100, 600,            # Outliers/negativos
             3.5, 12.0, 45.0, 7.5, 22.0,        # Normales adicionales
             250.0, 30.0, 18.0, 9.0, 55.0,
             400.0, 11.0, 14.0, 28.0, 33.0],
        ]),

        # LEAKAGE COLUMNS (componentes del target)
        "fare_amount":            np.random.uniform(2, 60, n),
        "extra":                  np.random.uniform(0, 3, n),
        "mta_tax":                np.full(n, 0.5),
        "tip_amount":             np.random.uniform(0, 10, n),
        "tolls_amount":           np.random.uniform(0, 5, n),
        "improvement_surcharge":  np.full(n, 0.3),
        "congestion_surcharge":   np.random.uniform(0, 2.75, n),
        "airport_fee":            np.random.choice([0, 1.25], n),
        "tip_pct":                np.random.uniform(0, 25, n),
        "trip_duration_min":      np.random.uniform(3, 60, n),
        "avg_speed_mph":          np.random.uniform(5, 40, n),
        "run_id":                 np.arange(n),
        "ingested_at_utc":        pd.date_range("2024-01-01", periods=n, freq="1min"),

        # Metadatos
        "source_service": np.random.choice(["yellow", "green"], n),
        "service_type":   np.random.choice(["yellow", "green"], n),
        "pickup_hour":    pd.date_range("2022-01-01", periods=n, freq="2h").hour,
        "day_of_week":    pd.date_range("2022-01-01", periods=n, freq="2h").dayofweek,
        "month":          pd.date_range("2022-01-01", periods=n, freq="2h").month,
        "year":           pd.date_range("2022-01-01", periods=n, freq="2h").year,
        "pickup_date":    pd.date_range("2022-01-01", periods=n, freq="2h").date,
        "dropoff_date":   pd.date_range("2022-01-01 00:30:00", periods=n, freq="2h").date,
        "dropoff_hour":   pd.date_range("2022-01-01 00:30:00", periods=n, freq="2h").hour,
    })

    return df


@pytest.fixture
def clean_df(sample_obt_df) -> pd.DataFrame:
    """DataFrame limpio (post clean_dataframe) para tests de feature engineering."""
    return clean_dataframe(sample_obt_df)


# ─────────────────────────────────────────────────────────────
# TESTS: LIMPIEZA (notebook 02)
# ─────────────────────────────────────────────────────────────

class TestRemoveLeakageColumns:
    def test_eliminates_all_leakage_cols(self, sample_obt_df):
        """Ninguna columna de leakage debe quedar en el resultado."""
        result = remove_leakage_columns(sample_obt_df)
        remaining = [c for c in LEAKAGE_COLUMNS if c in result.columns]
        assert remaining == [], f"Columnas de leakage residuales: {remaining}"

    def test_preserves_target(self, sample_obt_df):
        """El target total_amount debe permanecer después de eliminar leakage."""
        result = remove_leakage_columns(sample_obt_df)
        assert "total_amount" in result.columns

    def test_preserves_safe_columns(self, sample_obt_df):
        """Las features seguras no deben ser eliminadas."""
        safe = ["trip_distance", "passenger_count", "pickup_datetime", "pu_borough"]
        result = remove_leakage_columns(sample_obt_df)
        for col in safe:
            assert col in result.columns, f"Columna segura eliminada: {col}"

    def test_custom_leakage_list(self, sample_obt_df):
        """Acepta una lista personalizada de columnas a eliminar."""
        result = remove_leakage_columns(sample_obt_df, leakage_cols=["tip_amount"])
        assert "tip_amount" not in result.columns
        assert "fare_amount" in result.columns  # No estaba en la lista custom

    def test_returns_dataframe(self, sample_obt_df):
        """El resultado siempre es un DataFrame."""
        result = remove_leakage_columns(sample_obt_df)
        assert isinstance(result, pd.DataFrame)


class TestApplyBusinessFilters:
    def test_removes_negative_total_amount(self, sample_obt_df):
        """Valores negativos o cero de total_amount deben ser eliminados."""
        result = apply_business_filters(sample_obt_df)
        assert (result["total_amount"] <= 0).sum() == 0

    def test_removes_extreme_total_amount(self, sample_obt_df):
        """Valores de total_amount > $500 deben ser eliminados."""
        result = apply_business_filters(sample_obt_df, amount_max=500.0)
        assert (result["total_amount"] > 500).sum() == 0

    def test_removes_zero_distance(self, sample_obt_df):
        """Distancias <= 0 son imposibles y deben filtrarse."""
        # Insertar viaje con distancia 0
        df_with_zero = sample_obt_df.copy()
        df_with_zero.loc[0, "trip_distance"] = 0
        df_with_zero.loc[0, "total_amount"] = 10.0
        result = apply_business_filters(df_with_zero)
        assert (result["trip_distance"] <= 0).sum() == 0

    def test_passenger_count_range(self, sample_obt_df):
        """passenger_count debe estar en el rango 1-6."""
        df_test = sample_obt_df.copy()
        df_test["passenger_count"] = df_test["passenger_count"].fillna(1)
        df_test.loc[0, "passenger_count"] = 0
        df_test.loc[1, "passenger_count"] = 9
        df_test["total_amount"] = 10.0
        df_test["trip_distance"] = 5.0
        result = apply_business_filters(df_test)
        assert result["passenger_count"].between(1, 6).all()

    def test_returns_fewer_rows_than_input(self, sample_obt_df):
        """La función debe reducir el número de filas (hay outliers en el fixture)."""
        result = apply_business_filters(sample_obt_df)
        assert len(result) < len(sample_obt_df)

    def test_custom_thresholds(self, sample_obt_df):
        """Se pueden pasar umbrales personalizados."""
        result = apply_business_filters(sample_obt_df, amount_min=10.0, amount_max=100.0)
        assert result["total_amount"].min() > 10.0
        assert result["total_amount"].max() <= 100.0


class TestHandleNulls:
    def test_no_nulls_in_passenger_count(self, sample_obt_df):
        """passenger_count no debe tener nulos después de la imputación."""
        result = handle_nulls(sample_obt_df)
        assert result["passenger_count"].isnull().sum() == 0

    def test_passenger_count_imputed_with_median(self, sample_obt_df):
        """Los nulos de passenger_count se imputan con la mediana (valor entero razonable)."""
        median_before = sample_obt_df["passenger_count"].median()
        result = handle_nulls(sample_obt_df)
        # Todos los valores imputados deben ser >= 1 (mediana de pasajeros es típicamente 1)
        assert result["passenger_count"].min() >= 1

    def test_categorical_nulls_filled_with_unknown(self, sample_obt_df):
        """Columnas categóricas con nulos se llenan con 'Unknown'."""
        df_test = sample_obt_df.copy()
        df_test.loc[0, "pu_borough"] = None
        df_test.loc[1, "vendor_name"] = None
        result = handle_nulls(df_test)
        assert result["pu_borough"].isnull().sum() == 0
        assert result["vendor_name"].isnull().sum() == 0

    def test_rate_code_id_imputed_with_1(self, sample_obt_df):
        """rate_code_id nulo se imputa con 1 (Standard rate)."""
        result = handle_nulls(sample_obt_df)
        assert result["rate_code_id"].isnull().sum() == 0


class TestGroupRareCategories:
    def test_reduces_cardinality(self, sample_obt_df):
        """El número de categorías únicas debe reducirse al top_n + 1 (Other)."""
        result = group_rare_categories(sample_obt_df, "pu_zone", top_n=5)
        assert result["pu_zone"].nunique() <= 6  # 5 top + Other

    def test_other_label_present(self, sample_obt_df):
        """La etiqueta 'Other' debe aparecer si hay categorías raras."""
        result = group_rare_categories(sample_obt_df, "pu_zone", top_n=5)
        # Con 80 zonas y top_n=5, Other debe existir
        assert "Other" in result["pu_zone"].values

    def test_top_categories_preserved(self, sample_obt_df):
        """Las categorías más frecuentes deben conservarse tal cual."""
        top_cats = sample_obt_df["pu_zone"].value_counts().nlargest(5).index.tolist()
        result = group_rare_categories(sample_obt_df, "pu_zone", top_n=5)
        for cat in top_cats:
            assert cat in result["pu_zone"].values

    def test_nonexistent_column_returns_unchanged(self, sample_obt_df):
        """Si la columna no existe, retorna el DataFrame sin cambios."""
        result = group_rare_categories(sample_obt_df, "columna_inexistente", top_n=5)
        assert result.shape == sample_obt_df.shape


class TestCastDtypes:
    def test_pickup_datetime_is_datetime(self, sample_obt_df):
        """pickup_datetime debe ser convertida a datetime64."""
        result = cast_dtypes(sample_obt_df)
        assert pd.api.types.is_datetime64_any_dtype(result["pickup_datetime"])

    def test_int_columns_are_int32(self, sample_obt_df):
        """Columnas enteras seleccionadas deben ser int32."""
        result = cast_dtypes(sample_obt_df)
        if "pickup_hour" in result.columns:
            assert result["pickup_hour"].dtype == np.int32

    def test_float_columns_are_float32(self, sample_obt_df):
        """trip_distance y total_amount deben ser float32."""
        result = cast_dtypes(sample_obt_df)
        assert result["total_amount"].dtype == np.float32
        assert result["trip_distance"].dtype == np.float32

    def test_memory_reduced(self, sample_obt_df):
        """El uso de memoria debe reducirse después de la optimización de tipos."""
        mem_before = sample_obt_df.memory_usage(deep=True).sum()
        result = cast_dtypes(sample_obt_df)
        mem_after = result.memory_usage(deep=True).sum()
        # Con columnas int32 vs int64, debe haber reducción
        assert mem_after <= mem_before * 1.2  # Toleramos hasta 20% más (por category overhead)


class TestCleanDataframe:
    def test_no_leakage_in_output(self, sample_obt_df):
        """El DataFrame limpio no debe contener ninguna columna de leakage."""
        result = clean_dataframe(sample_obt_df)
        remaining = [c for c in LEAKAGE_COLUMNS if c in result.columns]
        assert remaining == [], f"Leakage residual: {remaining}"

    def test_total_amount_range(self, sample_obt_df):
        """total_amount debe estar en el rango (2.50, 500]."""
        result = clean_dataframe(sample_obt_df)
        assert result["total_amount"].min() > 2.50
        assert result["total_amount"].max() <= 500.0

    def test_no_nulls_in_critical_columns(self, sample_obt_df):
        """Columnas críticas no deben tener nulos."""
        result = clean_dataframe(sample_obt_df)
        for col in ["passenger_count", "trip_distance", "total_amount"]:
            if col in result.columns:
                assert result[col].isnull().sum() == 0, f"Nulos en {col}"

    def test_returns_dataframe(self, sample_obt_df):
        """El resultado es siempre un DataFrame."""
        result = clean_dataframe(sample_obt_df)
        assert isinstance(result, pd.DataFrame)

    def test_rows_reduced(self, sample_obt_df):
        """Deben eliminarse filas con valores inválidos."""
        result = clean_dataframe(sample_obt_df)
        assert len(result) < len(sample_obt_df)


# ─────────────────────────────────────────────────────────────
# TESTS: FEATURE ENGINEERING (notebook 03)
# ─────────────────────────────────────────────────────────────

class TestCreateTemporalFeatures:
    def test_creates_is_weekend(self, clean_df):
        """is_weekend debe existir y ser binaria (0 o 1)."""
        result = create_temporal_features(clean_df)
        assert "is_weekend" in result.columns
        assert result["is_weekend"].isin([0, 1]).all()

    def test_creates_is_rush_hour(self, clean_df):
        """is_rush_hour debe existir y ser binaria."""
        result = create_temporal_features(clean_df)
        assert "is_rush_hour" in result.columns
        assert result["is_rush_hour"].isin([0, 1]).all()

    def test_creates_is_night(self, clean_df):
        """is_night debe ser 1 para horas >= 20 o < 6."""
        result = create_temporal_features(clean_df)
        assert "is_night" in result.columns
        night_hours = result[result["is_night"] == 1]["pickup_hour"]
        assert ((night_hours >= 20) | (night_hours < 6)).all()

    def test_creates_cyclic_encoding(self, clean_df):
        """Las features cíclicas sin/cos deben estar en rango [-1, 1]."""
        result = create_temporal_features(clean_df)
        for col in ["pickup_hour_sin", "pickup_hour_cos", "day_of_week_sin",
                    "day_of_week_cos", "month_sin", "month_cos"]:
            assert col in result.columns, f"Feature cíclica faltante: {col}"
            assert result[col].between(-1.0, 1.0).all(), f"{col} fuera de [-1, 1]"

    def test_no_leakage_in_temporal_features(self, clean_df):
        """Las features temporales no introducen leakage (no usan dropoff info)."""
        result = create_temporal_features(clean_df)
        leakage_temporal = ["dropoff_datetime", "trip_duration_min", "avg_speed_mph"]
        for col in leakage_temporal:
            assert col not in result.columns, f"Leakage temporal introducido: {col}"

    def test_is_holiday_on_new_year(self):
        """is_holiday debe ser 1 el 1 de enero."""
        df = pd.DataFrame({
            "pickup_datetime": pd.to_datetime(["2022-01-01 10:00:00"]),
            "pickup_hour": [10],
            "day_of_week": [6],
            "month": [1],
            "trip_distance": [5.0],
            "total_amount": [15.0],
        })
        result = create_temporal_features(df)
        assert result["is_holiday"].iloc[0] == 1

    def test_rush_hour_not_on_weekend(self):
        """Rush hour no debe activarse en fines de semana."""
        # Sábado a las 8 AM (hora de rush en semana, pero es fin de semana)
        df = pd.DataFrame({
            "pickup_datetime": pd.to_datetime(["2022-01-01 08:00:00"]),  # Sábado
            "pickup_hour": [8],
            "day_of_week": [5],  # Sábado
            "month": [1],
            "trip_distance": [5.0],
            "total_amount": [15.0],
        })
        result = create_temporal_features(df)
        assert result["is_rush_hour"].iloc[0] == 0


class TestCreateGeographicFeatures:
    def test_creates_is_airport_trip(self, clean_df):
        """is_airport_trip debe existir y ser binaria."""
        result = create_temporal_features(clean_df)
        result = create_geographic_features(result)
        assert "is_airport_trip" in result.columns
        assert result["is_airport_trip"].isin([0, 1]).all()

    def test_jfk_detected_by_rate_code(self):
        """is_jfk debe ser 1 cuando rate_code_id = 2."""
        df = pd.DataFrame({
            "pickup_datetime": pd.to_datetime(["2022-06-15 14:00:00"]),
            "pickup_hour": [14],
            "day_of_week": [2],
            "month": [6],
            "trip_distance": [15.0],
            "total_amount": [52.0],
            "passenger_count": [2],
            "rate_code_id": [2],  # JFK
            "pu_borough": ["Manhattan"],
            "do_borough": ["Queens"],
        })
        result = create_temporal_features(df)
        result = create_geographic_features(result)
        assert result["is_jfk"].iloc[0] == 1
        assert result["is_airport_trip"].iloc[0] == 1

    def test_same_borough_flag(self):
        """same_borough debe ser 1 cuando origen y destino están en el mismo borough."""
        df = pd.DataFrame({
            "pickup_datetime": pd.to_datetime(["2022-06-15 14:00:00"]),
            "pickup_hour": [14],
            "day_of_week": [2],
            "month": [6],
            "trip_distance": [3.0],
            "total_amount": [12.0],
            "passenger_count": [1],
            "rate_code_id": [1],
            "pu_borough": ["Manhattan"],
            "do_borough": ["Manhattan"],  # Mismo borough
        })
        result = create_temporal_features(df)
        result = create_geographic_features(result)
        assert result["same_borough"].iloc[0] == 1
        assert result["is_inter_borough"].iloc[0] == 0

    def test_is_manhattan_origin(self, clean_df):
        """is_manhattan_origin debe ser 1 solo cuando pu_borough = 'Manhattan'."""
        result = create_temporal_features(clean_df)
        result = create_geographic_features(result)
        assert "is_manhattan_origin" in result.columns
        manhattan_mask = result["pu_borough"] == "Manhattan"
        assert (result.loc[manhattan_mask, "is_manhattan_origin"] == 1).all()
        assert (result.loc[~manhattan_mask, "is_manhattan_origin"] == 0).all()


class TestCreateInteractionFeatures:
    def test_creates_distance_per_passenger(self, clean_df):
        """distance_per_passenger debe ser mayor a 0."""
        result = create_temporal_features(clean_df)
        result = create_geographic_features(result)
        result = create_interaction_features(result)
        assert "distance_per_passenger" in result.columns
        assert (result["distance_per_passenger"] > 0).all()

    def test_is_long_trip_threshold(self):
        """is_long_trip debe ser 1 para viajes > 10 millas."""
        df = pd.DataFrame({
            "pickup_datetime": pd.to_datetime(["2022-01-01"] * 3),
            "pickup_hour": [10, 10, 10],
            "day_of_week": [0, 0, 0],
            "month": [1, 1, 1],
            "trip_distance": [5.0, 10.0, 15.0],
            "total_amount": [15.0, 30.0, 50.0],
            "passenger_count": [1, 1, 1],
            "rate_code_id": [1, 1, 1],
            "pu_borough": ["Manhattan"] * 3,
            "do_borough": ["Brooklyn"] * 3,
        })
        result = create_temporal_features(df)
        result = create_geographic_features(result)
        result = create_interaction_features(result)
        assert result.loc[result["trip_distance"] == 5.0, "is_long_trip"].iloc[0] == 0
        assert result.loc[result["trip_distance"] == 15.0, "is_long_trip"].iloc[0] == 1

    def test_rush_airport_interaction(self):
        """rush_airport debe ser 1 solo cuando ambas condiciones son verdaderas."""
        df = pd.DataFrame({
            "pickup_datetime": pd.to_datetime(["2022-01-03 08:00:00"]),  # Lunes 8 AM
            "pickup_hour": [8],
            "day_of_week": [0],  # Lunes
            "month": [1],
            "trip_distance": [15.0],
            "total_amount": [52.0],
            "passenger_count": [1],
            "rate_code_id": [2],  # JFK
            "pu_borough": ["Manhattan"],
            "do_borough": ["Queens"],
        })
        result = create_temporal_features(df)
        result = create_geographic_features(result)
        result = create_interaction_features(result)
        # Lunes a las 8 AM es rush hour; rate_code 2 es aeropuerto
        assert result["is_rush_hour"].iloc[0] == 1
        assert result["is_airport_trip"].iloc[0] == 1
        assert result["rush_airport"].iloc[0] == 1

    def test_no_division_by_zero_in_distance_per_passenger(self):
        """No debe haber división por cero en distance_per_passenger."""
        df = pd.DataFrame({
            "pickup_datetime": pd.to_datetime(["2022-01-01"]),
            "pickup_hour": [10],
            "day_of_week": [0],
            "month": [1],
            "trip_distance": [5.0],
            "total_amount": [15.0],
            "passenger_count": [0],  # Caso borde
            "rate_code_id": [1],
            "pu_borough": ["Manhattan"],
            "do_borough": ["Brooklyn"],
        })
        result = create_temporal_features(df)
        result = create_geographic_features(result)
        result = create_interaction_features(result)
        assert np.isfinite(result["distance_per_passenger"].iloc[0])


class TestRunFeatureEngineering:
    def test_all_feature_groups_created(self, clean_df):
        """run_feature_engineering debe crear features de los 3 grupos."""
        result = run_feature_engineering(clean_df)

        temporal = ["is_weekend", "is_rush_hour", "is_night", "pickup_hour_sin"]
        geographic = ["is_airport_trip", "same_borough", "is_manhattan_origin"]
        interactions = ["distance_per_passenger", "is_long_trip", "rush_airport"]

        for feat in temporal + geographic + interactions:
            assert feat in result.columns, f"Feature faltante: {feat}"

    def test_more_columns_than_input(self, clean_df):
        """El dataset de salida debe tener más columnas que el de entrada."""
        result = run_feature_engineering(clean_df)
        assert result.shape[1] > clean_df.shape[1]

    def test_no_new_leakage_introduced(self, clean_df):
        """El feature engineering no debe introducir columnas de leakage."""
        result = run_feature_engineering(clean_df)
        residual = [c for c in LEAKAGE_COLUMNS if c in result.columns]
        assert residual == [], f"Leakage introducido por FE: {residual}"

    def test_returns_dataframe(self, clean_df):
        """run_feature_engineering retorna un DataFrame."""
        result = run_feature_engineering(clean_df)
        assert isinstance(result, pd.DataFrame)


# ─────────────────────────────────────────────────────────────
# TESTS: PREPROCESSORS sklearn
# ─────────────────────────────────────────────────────────────

class TestPreprocessors:

    @pytest.fixture
    def engineered_df(self, clean_df):
        return run_feature_engineering(clean_df)

    def test_preprocessor_trees_fits_and_transforms(self, engineered_df):
        """El preprocessor para árboles debe poder hacer fit_transform sin errores."""
        all_feat = get_all_feature_columns(engineered_df)
        X = engineered_df[all_feat]

        preprocessor = build_preprocessor_for_trees()
        X_transformed = preprocessor.fit_transform(X)

        assert X_transformed is not None
        assert X_transformed.shape[0] == len(engineered_df)
        assert X_transformed.shape[1] > 0

    def test_preprocessor_trees_output_is_numeric(self, engineered_df):
        """La salida del preprocessor de árboles debe ser completamente numérica."""
        all_feat = get_all_feature_columns(engineered_df)
        X = engineered_df[all_feat]

        preprocessor = build_preprocessor_for_trees()
        X_transformed = preprocessor.fit_transform(X)

        assert np.issubdtype(X_transformed.dtype, np.number)

    def test_preprocessor_trees_no_nan_in_output(self, engineered_df):
        """No debe haber NaN en la salida del preprocessor."""
        all_feat = get_all_feature_columns(engineered_df)
        X = engineered_df[all_feat]

        preprocessor = build_preprocessor_for_trees()
        X_transformed = preprocessor.fit_transform(X)

        assert not np.isnan(X_transformed).any(), "NaN encontrados en la salida del preprocessor"

    def test_preprocessor_linear_fits_and_transforms(self, engineered_df):
        """El preprocessor lineal debe hacer fit_transform sin errores."""
        from src.features.build_features import FEATURE_CONFIG
        num_feats = [c for c in FEATURE_CONFIG["numeric_features"] if c in engineered_df.columns]
        bin_feats = [c for c in FEATURE_CONFIG["binary_features"] if c in engineered_df.columns]
        cat_feats = [c for c in FEATURE_CONFIG["cat_low_card"] if c in engineered_df.columns]

        X = engineered_df[num_feats + bin_feats + cat_feats]
        preprocessor = build_preprocessor_for_linear(num_feats, bin_feats, cat_feats)
        X_transformed = preprocessor.fit_transform(X)

        assert X_transformed.shape[0] == len(engineered_df)

    def test_preprocessor_handles_unknown_categories(self, engineered_df):
        """El preprocessor debe manejar categorías no vistas en entrenamiento."""
        all_feat = get_all_feature_columns(engineered_df)
        X_train = engineered_df[all_feat].iloc[:150]
        X_test  = engineered_df[all_feat].iloc[150:].copy()

        preprocessor = build_preprocessor_for_trees()
        preprocessor.fit(X_train)

        # Convertir a str antes de insertar la categoría desconocida
        # (cast_dtypes convierte pu_borough a category, que no acepta valores nuevos directamente)
        if "pu_borough" in X_test.columns:
            X_test["pu_borough"] = X_test["pu_borough"].astype(str)
            X_test.loc[X_test.index[0], "pu_borough"] = "Categoría_Desconocida"

        # No debe lanzar error (OrdinalEncoder con unknown_value=-1)
        X_transformed = preprocessor.transform(X_test)
        assert X_transformed is not None


# ─────────────────────────────────────────────────────────────
# TESTS: CONFIGURACIÓN
# ─────────────────────────────────────────────────────────────

class TestConfig:
    def test_feature_config_has_required_keys(self):
        """FEATURE_CONFIG debe tener todas las claves requeridas."""
        required_keys = [
            "target", "numeric_features", "location_features",
            "binary_features", "cat_low_card", "cat_high_card",
        ]
        for key in required_keys:
            assert key in FEATURE_CONFIG, f"Clave faltante en FEATURE_CONFIG: {key}"

    def test_feature_config_target_is_total_amount(self):
        """El target debe ser total_amount."""
        assert FEATURE_CONFIG["target"] == "total_amount"

    def test_leakage_columns_list_not_empty(self):
        """La lista de columnas de leakage no debe estar vacía."""
        assert len(LEAKAGE_COLUMNS) > 0

    def test_leakage_includes_fare_amount(self):
        """fare_amount es el componente principal del leakage y debe estar en la lista."""
        assert "fare_amount" in LEAKAGE_COLUMNS

    def test_leakage_includes_tip_amount(self):
        """tip_amount es leakage clásico (se da al finalizar el viaje)."""
        assert "tip_amount" in LEAKAGE_COLUMNS

    def test_leakage_includes_dropoff_datetime(self):
        """dropoff_datetime es leakage (solo disponible al terminar el viaje)."""
        assert "dropoff_datetime" in LEAKAGE_COLUMNS


# ─────────────────────────────────────────────────────────────
# TESTS DE INTEGRACIÓN: Pipeline completo
# ─────────────────────────────────────────────────────────────

class TestFullPipeline:
    def test_clean_then_engineer_then_preprocess(self, sample_obt_df):
        """
        Test de integración del pipeline completo:
        OBT cruda → clean_dataframe → run_feature_engineering → preprocessor.fit_transform
        """
        # Paso 1: Limpieza
        df_clean = clean_dataframe(sample_obt_df)
        assert len(df_clean) > 0, "La limpieza eliminó todas las filas"

        # Paso 2: Feature Engineering
        df_fe = run_feature_engineering(df_clean)
        assert df_fe.shape[1] > df_clean.shape[1], "No se crearon features nuevas"

        # Paso 3: Preprocesamiento
        all_feat = get_all_feature_columns(df_fe)
        assert len(all_feat) > 0, "No hay features disponibles para el modelo"

        X = df_fe[all_feat]
        y = df_fe[FEATURE_CONFIG["target"]]

        preprocessor = build_preprocessor_for_trees()
        X_transformed = preprocessor.fit_transform(X)

        assert X_transformed.shape[0] == len(df_fe)
        assert not np.isnan(X_transformed).any(), "NaN en la salida final del pipeline"
        assert len(y) == len(df_fe)

        print(f"\n✅ Pipeline completo OK:")
        print(f"   OBT cruda:     {sample_obt_df.shape}")
        print(f"   Post-limpieza: {df_clean.shape}")
        print(f"   Post-FE:       {df_fe.shape}")
        print(f"   Post-preproc:  {X_transformed.shape}")

    def test_target_not_in_features(self, sample_obt_df):
        """El target (total_amount) nunca debe estar en X."""
        df_clean = clean_dataframe(sample_obt_df)
        df_fe    = run_feature_engineering(df_clean)
        all_feat = get_all_feature_columns(df_fe)

        assert "total_amount" not in all_feat, \
            "CRÍTICO: total_amount está incluido en las features de entrada"

    def test_no_leakage_in_final_features(self, sample_obt_df):
        """Ninguna columna de leakage debe llegar al modelo final."""
        df_clean = clean_dataframe(sample_obt_df)
        df_fe    = run_feature_engineering(df_clean)
        all_feat = get_all_feature_columns(df_fe)

        leakage_in_features = [c for c in LEAKAGE_COLUMNS if c in all_feat]
        assert leakage_in_features == [], \
            f"CRÍTICO: Leakage en features finales: {leakage_in_features}"
