"""
Módulo para la conexión a Snowflake y extracción de datos iterativa (Big Data).

Provee dos modos de extracción:
  - fetch_sample(): muestra representativa para EDA (notebooks 01-03)
  - fetch_data_in_batches(): iterador de chunks para Out-of-Core training (notebook 04 y src/models/)

Uso desde notebooks:
    from src.data.ingestion import fetch_sample, fetch_data_in_batches

Uso desde scripts de producción:
    from src.data.ingestion import fetch_data_in_batches
"""

import pandas as pd
import os
import logging
from typing import Iterator, Optional

import snowflake.connector
from snowflake.connector import DictCursor

from src.utils.config import get_snowflake_credentials

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# CONEXIÓN
# ─────────────────────────────────────────────────────────────

def get_snowflake_connection() -> snowflake.connector.SnowflakeConnection:
    """
    Establece y retorna un objeto de conexión activa a Snowflake
    usando las credenciales del archivo .env (via src/utils/config.py).

    Returns:
        Objeto de conexión de snowflake.connector.

    Raises:
        ConnectionError: Si faltan credenciales o la conexión falla.
    """
    creds = get_snowflake_credentials()

    # Validar que no haya credenciales vacías
    missing = [k for k, v in creds.items() if not v]
    if missing:
        raise ConnectionError(
            f"Faltan credenciales de Snowflake en .env: {missing}. "
            "Copia .env.example a .env y llena los valores."
        )

    try:
        conn = snowflake.connector.connect(
            user=creds["user"],
            password=creds["password"],
            account=creds["account"],
            warehouse=creds["warehouse"],
            database=creds["database"],
            schema=creds["schema"],
            # Configuraciones de rendimiento recomendadas
            client_session_keep_alive=True,
        )
        logger.info("✅ Conexión a Snowflake establecida.")
        return conn

    except Exception as e:
        raise ConnectionError(f"Error al conectar con Snowflake: {e}") from e


# ─────────────────────────────────────────────────────────────
# EXTRACCIÓN EN MUESTRA (para EDA y notebooks de exploración)
# ─────────────────────────────────────────────────────────────

def fetch_sample(query: str, sample_prob: float = 1.0) -> pd.DataFrame:
    """
    Extrae una muestra aleatoria desde Snowflake usando TABLESAMPLE.
    
    La cláusula TABLESAMPLE corre del lado del servidor (pushdown), por lo que
    NO descarga los ~20 GB completos a la memoria local. Solo trae sample_prob%
    de las filas a Python.

    Args:
        query: Query SQL base, ej: "SELECT * FROM analytics.train_set"
               No debe incluir punto y coma al final.
        sample_prob: Porcentaje de filas a extraer (0.1 = 0.1%, 1.0 = 1%, 100.0 = todo).
                     Para el EDA se recomienda entre 0.3% y 2%.

    Returns:
        pandas DataFrame con la muestra.

    Raises:
        ConnectionError: Si la conexión a Snowflake falla.
        ValueError: Si sample_prob está fuera de rango (0, 100].
    """
    if not (0 < sample_prob <= 100):
        raise ValueError(f"sample_prob debe estar entre 0 y 100, recibido: {sample_prob}")

    # Inyectar TABLESAMPLE justo antes de posibles cláusulas WHERE/ORDER
    # Envolvemos la query original como subquery para que TABLESAMPLE aplique correctamente
    sampled_query = f"""
        SELECT *
        FROM ({query}) AS base_query
        TABLESAMPLE ({sample_prob})
    """

    conn = get_snowflake_connection()
    try:
        logger.info(f"Extrayendo muestra ({sample_prob}%)...")
        df = pd.read_sql(sampled_query, conn)
        df.columns = df.columns.str.lower()
        logger.info(f"✅ Muestra cargada: {len(df):,} filas × {df.shape[1]} columnas")
        return df
    except Exception as e:
        logger.error(f"Error al extraer muestra: {e}")
        raise
    finally:
        conn.close()


# ─────────────────────────────────────────────────────────────
# EXTRACCIÓN EN LOTES (para Out-of-Core training)
# ─────────────────────────────────────────────────────────────

def fetch_data_in_batches(
    query: str,
    batch_size: int = 100_000,
) -> Iterator[pd.DataFrame]:
    """
    Extrae datos de Snowflake en lotes (chunks) mediante cursores,
    en lugar de cargar todo el DataFrame de una vez.

    Crucial para datasets de ~20 GB: evita MemoryError cargando
    batch_size filas a la vez y entregándolas al modelo iterativamente
    (Out-of-Core training con partial_fit o actualización incremental).

    Args:
        query: Query SQL completa a ejecutar contra Snowflake.
               Ejemplo: "SELECT * FROM analytics.train_set"
        batch_size: Número de filas por lote. Con 100k filas y ~30 cols
                    cada chunk pesa aproximadamente 50-100 MB en RAM.

    Yields:
        pandas DataFrames de `batch_size` filas cada uno.
        El último batch puede tener menos filas.

    Raises:
        ConnectionError: Si la conexión a Snowflake falla.

    Ejemplo de uso:
        >>> query = "SELECT * FROM analytics.train_set"
        >>> for batch_df in fetch_data_in_batches(query, batch_size=200_000):
        ...     X = batch_df.drop('total_amount', axis=1)
        ...     y = batch_df['total_amount']
        ...     model.partial_fit(X, y)
    """
    conn = get_snowflake_connection()

    try:
        cursor = conn.cursor()
        logger.info(f"Ejecutando query para extracción en lotes (batch_size={batch_size:,})...")
        cursor.execute(query)

        # Obtener nombres de columnas desde la descripción del cursor
        col_names = [desc[0].lower() for desc in cursor.description]

        batch_num = 0
        total_rows = 0

        while True:
            rows = cursor.fetchmany(batch_size)
            if not rows:
                break  # No quedan más datos

            batch_num += 1
            total_rows += len(rows)
            batch_df = pd.DataFrame(rows, columns=col_names)

            logger.info(f"Batch {batch_num}: {len(batch_df):,} filas | Total acumulado: {total_rows:,}")
            yield batch_df

        logger.info(f"✅ Extracción completada. Total: {total_rows:,} filas en {batch_num} batches.")

    except Exception as e:
        logger.error(f"Error durante la extracción en lotes: {e}")
        raise
    finally:
        cursor.close()
        conn.close()


# ─────────────────────────────────────────────────────────────
# UTILIDADES
# ─────────────────────────────────────────────────────────────

def test_connection() -> bool:
    """
    Prueba rápida de la conexión a Snowflake.
    Ejecuta una query trivial y verifica que regresa resultados.

    Returns:
        True si la conexión funciona, False en caso contrario.
    """
    try:
        conn = get_snowflake_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT CURRENT_VERSION()")
        version = cursor.fetchone()[0]
        print(f"✅ Conexión exitosa. Versión de Snowflake: {version}")
        conn.close()
        return True
    except Exception as e:
        print(f"❌ Error de conexión: {e}")
        return False


def count_rows(table_or_view: str) -> Optional[int]:
    """
    Retorna el número de filas de una tabla o vista en Snowflake.
    Útil para dimensionar los batches antes de ejecutar la extracción completa.

    Args:
        table_or_view: Nombre completo de la tabla/vista, ej: "analytics.train_set"

    Returns:
        Número de filas como entero, o None si hay error.
    """
    try:
        conn = get_snowflake_connection()
        cursor = conn.cursor()
        cursor.execute(f"SELECT COUNT(*) FROM {table_or_view}")
        count = cursor.fetchone()[0]
        conn.close()
        print(f"📊 {table_or_view}: {count:,} filas")
        return count
    except Exception as e:
        logger.error(f"Error contando filas de {table_or_view}: {e}")
        return None
