"""
Módulo utilitario para la recolección de variables de entorno y configuración central.

Centraliza toda la configuración del proyecto para que el resto de módulos
importen desde aquí en lugar de usar os.getenv() directamente.

Uso:
    from src.utils.config import get_snowflake_credentials, PROJECT_ROOT
"""

import os
import logging
from pathlib import Path
from dotenv import load_dotenv

# ─────────────────────────────────────────────────────────────
# RUTAS DEL PROYECTO
# ─────────────────────────────────────────────────────────────

# Raíz del proyecto (dos niveles arriba de src/utils/)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Directorios estándar
DATA_DIR        = PROJECT_ROOT / "data"
DATA_RAW        = DATA_DIR / "raw"
DATA_INTERIM    = DATA_DIR / "interim"
DATA_PROCESSED  = DATA_DIR / "processed"
MODELS_DIR      = PROJECT_ROOT / "models"
NOTEBOOKS_DIR   = PROJECT_ROOT / "notebooks"

# Crear directorios si no existen (útil al clonar el repo)
for _dir in [DATA_RAW, DATA_INTERIM, DATA_PROCESSED, MODELS_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# VARIABLES DE ENTORNO
# ─────────────────────────────────────────────────────────────

# Intentar cargar el .env desde la raíz del proyecto
_env_path = PROJECT_ROOT / ".env"
load_dotenv(dotenv_path=_env_path)


def get_snowflake_credentials() -> dict:
    """
    Retorna un diccionario con las credenciales de conexión a Snowflake
    leídas desde las variables de entorno (archivo .env).

    Variables requeridas en .env:
        SNOWFLAKE_USER, SNOWFLAKE_PASSWORD, SNOWFLAKE_ACCOUNT,
        SNOWFLAKE_WAREHOUSE, SNOWFLAKE_DATABASE, SNOWFLAKE_SCHEMA

    Returns:
        dict con las claves: user, password, account, warehouse, database, schema.

    Raises:
        EnvironmentError: Si alguna variable obligatoria no está definida.
    """
    creds = {
        "user":      os.getenv("SNOWFLAKE_USER"),
        "password":  os.getenv("SNOWFLAKE_PASSWORD"),
        "account":   os.getenv("SNOWFLAKE_ACCOUNT"),
        "warehouse": os.getenv("SNOWFLAKE_WAREHOUSE"),
        "database":  os.getenv("SNOWFLAKE_DATABASE"),
        "schema":    os.getenv("SNOWFLAKE_SCHEMA"),
    }

    missing = [k for k, v in creds.items() if not v]
    if missing:
        raise EnvironmentError(
            f"Variables de entorno faltantes en .env: {missing}. "
            f"Copia .env.example → .env y llena los valores reales."
        )

    return creds


# ─────────────────────────────────────────────────────────────
# PARÁMETROS DE MODELADO
# ─────────────────────────────────────────────────────────────

# Semilla global de reproducibilidad
RANDOM_STATE = 42

# Configuración de la extracción en lotes
DEFAULT_BATCH_SIZE = 100_000   # ~50-100 MB por batch con ~30 columnas

# Umbrales de limpieza (sincronizados con build_features.py)
AMOUNT_MIN    = 2.50
AMOUNT_MAX    = 500.0
DISTANCE_MAX  = 200.0
PASSENGER_MAX = 6

# Ruta del modelo serializado
MODEL_PATH = MODELS_DIR / "price_model.pkl"

# ─────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────

def setup_logging(level: int = logging.INFO) -> None:
    """
    Configura el logging básico del proyecto.
    Llama a esta función al inicio de cada script de producción.

    Args:
        level: Nivel de logging (logging.DEBUG, logging.INFO, etc.)
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
