# Implementaciones de la sesión — Proyecto Final

> Este documento describe los cambios técnicos aplicados al repositorio en esta
> sesión: **qué se modificó, por qué, qué falta y cómo correr todo end-to-end**.
> Complementa al `README.md` (sin reemplazarlo).

Branch destino: `feature/production-pipeline-and-api`
Ejecución del workflow: diagnóstico → fix bug crítico SQL → pipeline modular →
predict + API + frontend → tests automatizados (80 passed).

---

## 1. Diagnóstico inicial

### Estado encontrado
- Notebooks 01-04 completos. CatBoost 500 iter alcanzó **RMSE ≈ 12.70** en
  validación 2023 (dentro de submuestra del `train_set`).
- `src/features/build_features.py` modular y bien diseñado (limpieza + 30+
  features espacio-temporales + 2 ColumnTransformers). 60+ tests pasando.
- `src/data/ingestion.py` con `fetch_sample` (TABLESAMPLE pushdown) y
  `fetch_data_in_batches` (cursor por chunks).
- `src/utils/config.py` correcto.
- SQL en Snowflake: OBT → cleaned_obt → splits temporales (train 2015-23,
  val 2024, test 2025) — el diseño correcto para evitar saturar RAM.

### Problemas críticos detectados
1. **🔴 Bug en `03_clean_obt.sql`**: `trip_distance` se usaba en el `WHERE`
   pero **no se incluía en el `SELECT`**. La tabla limpia downstream se
   quedaba sin el predictor más fuerte. Cualquier corrida fresca habría
   roto el pipeline.
2. `src/models/train_model.py` era un esqueleto con TODOs e importaba
   `get_feature_pipeline` (función inexistente).
3. `src/models/predict_model.py` 100% TODO.
4. `src/api/main.py` retornaba literalmente `{"estimated_total_amount": 0.0}`.
5. `app/frontend.py` sólo tenía un comentario.
6. Sin hyperparameter tuning. Modelos con parámetros hardcoded.
7. Sin log-transform del target (`total_amount` es muy sesgado a la derecha).
8. Sin script reproducible para regenerar artefactos sin Jupyter.
9. Validación temporal usaba `year<=2022 vs year==2023` **dentro de la
   submuestra del train_set**, en vez del `val_set` 2024 oficial.
10. `requirements.txt` sin XGBoost / LightGBM / CatBoost / Streamlit.

---

## 2. Cambios aplicados

### 2.1 SQL crítico — `src/data/sql/03_clean_obt.sql`
Se añadió `trip_distance` al `SELECT` (sigue filtrado en el `WHERE`).
Re-ejecutar este script en Snowflake recrea `cleaned_obt_trips` con la
columna y, por extensión, las vistas `train_set`, `val_set`, `test_set`.

### 2.2 Pipeline de entrenamiento — `src/models/train_model.py` (~440 líneas)
Reemplazo total del esqueleto. Implementa:

- **CLI con `argparse`** y tres modos:
  - `--mode sample` — entrenamiento rápido sobre TABLESAMPLE.
  - `--mode tune` — hyperparameter search.
  - `--mode oof` — out-of-core warm-start sobre todos los batches.
- **Modelos soportados**: `lightgbm` (default), `xgboost`, `catboost`,
  `gbdt`, `adaboost`. Cubre los 5 boostings obligatorios de la rúbrica.
- **`TransformedTargetRegressor(log1p / expm1)`** envuelve al regresor.
  Estabiliza la varianza de un target muy sesgado a la derecha → suele
  bajar RMSE 5-15% en este tipo de problema.
- **HP tuning** con dos backends:
  - `--tuner randomized` → `RandomizedSearchCV` (LightGBM).
  - `--tuner optuna` → optimización bayesiana (LightGBM/XGBoost/CatBoost).
  - Ambos usan `TimeSeriesSplit(3)` para no fugar futuro a pasado.
- **Out-of-Core training** real: warm-start con `init_model` (LightGBM) o
  `xgb_model` (XGBoost) iterando sobre `fetch_data_in_batches`.
- **Evaluación en `val_set` 2024** real (Snowflake), no en sub-validación
  del train. Esto corrige el riesgo metodológico del notebook.
- **Persistencia atómica**: `joblib.dump` del Pipeline completo +
  `feature_config.json` con lista de features, target, metadata
  (RMSE CV, hiperparámetros, model_version timestamp).
- **Lazy import de Snowflake** para que el módulo se pueda importar en
  CI/tests sin instalar `snowflake-connector-python`.

### 2.3 Inferencia — `src/models/predict_model.py`
Implementa la cara opuesta:
- `load_model`, `load_feature_config`, `load_artifacts`.
- `prepare_input`: corre `run_feature_engineering` sobre input crudo,
  rellena columnas faltantes con NaN (el imputer del Pipeline las maneja),
  alinea al esquema exacto del modelo.
- `predict`: clip a 0 (las tarifas no son negativas), retorna `np.ndarray`.

### 2.4 API — `src/api/main.py`
FastAPI productivo:
- **`lifespan`** carga el modelo en startup; el server arranca aún sin
  modelo (estado `degraded`), útil en dev.
- **Schema Pydantic `TripInput`** con todos los campos OBT-row, validadores
  de `passenger_count` (1-6), `trip_distance` (>0 ≤200), `rate_code_id`
  (1-6), borough/service whitelisted.
- **Endpoints**:
  - `GET  /health` — flag de modelo, versión, conteo de features.
  - `POST /predict` — un viaje, retorna `estimated_total_amount`.
  - `POST /predict/batch` — lista de viajes (más eficiente).
- Configurable por env: `MODEL_PATH`, `FEATURE_CONFIG_PATH`.

### 2.5 Frontend — `app/frontend.py`
Streamlit completo:
- Form con **fecha/hora**, distancia, pasajeros, borough origen/destino,
  location IDs, tarifa, vendor, pago previsto, servicio (yellow/green).
- **Sidebar con `/health` polling** muestra estado del modelo, versión,
  número de features, error de carga si lo hay.
- Render del precio en USD + expander con el JSON enviado a la API.
- Lee `API_URL` por env (default `http://127.0.0.1:8000`).

### 2.6 Dependencias — `requirements.txt`
- Añade `xgboost`, `lightgbm`, `catboost`, `optuna`, `streamlit`, `requests`,
  `httpx` (para `TestClient` en tests), `joblib` explícito.
- **Pin de Starlette** `>=0.40,<0.47` para evitar el bug
  `Router.__init__() got an unexpected keyword argument 'on_startup'`
  que aparece cuando alguna sub-dependencia (p.ej. `sse-starlette`)
  arrastra Starlette 1.0.0 incompatible con FastAPI 0.115.

### 2.7 Tests — `tests/test_train_predict_api.py` (19 nuevos smoke tests)
Cubre:
- `build_pipeline` (LightGBM, GBDT, modelo no soportado, log-target activo).
- `evaluate_pipeline` (RMSE / MAE / R²) sobre target sintético correlacionado.
- `save_artifacts` (verifica .pkl y feature_config.json).
- `load_artifacts` round-trip.
- `prepare_input` (alineación de columnas, columnas faltantes).
- `predict` (shape, no-negatividad, sample crudo de 1 fila).
- API (`TestClient`):
  - `/health` con modelo cargado.
  - `/predict` con payload válido.
  - `/predict/batch` con 3 viajes.
  - Validación 422 para `passenger_count=9` y `trip_distance=-5`.
  - Rechazo 400 de `/predict/batch` con lista vacía.

Total proyecto: **80 tests pasan, 0 fallan** (`pytest tests/`).

---

## 3. Cómo correr todo

```powershell
# 0. Asegurarse de tener .env con credenciales de Snowflake (no commitear).
# 1. Instalar dependencias
pip install -r requirements.txt

# 2. (En Snowflake) Re-correr 03_clean_obt.sql para regenerar cleaned_obt_trips
#    con trip_distance. Se puede hacer desde el notebook 02 (cell que ejecuta
#    el .sql) o desde un cliente SQL externo.

# 3. Entrenamiento rápido (~1% del train_set, default LightGBM + log-target)
python -m src.models.train_model --mode sample --sample-prob 1.0

# 4. Hyperparameter tuning (recomendado para minimizar RMSE)
python -m src.models.train_model --mode tune --tuner optuna --n-iter 50

# 5. Refinamiento out-of-core sobre TODOS los batches del train_set
python -m src.models.train_model --mode oof --batch-size 500000 --ooc-rounds 30

# 6. Tests
python -m pytest tests/ -v

# 7. API
uvicorn src.api.main:app --reload                       # http://127.0.0.1:8000

# 8. Frontend (en otra terminal)
streamlit run app/frontend.py                           # http://127.0.0.1:8501
```

### Argumentos CLI relevantes (`train_model.py`)
| Flag | Default | Descripción |
|------|---------|-------------|
| `--mode` | `sample` | `sample` / `tune` / `oof` |
| `--model` | `lightgbm` | `lightgbm` / `xgboost` / `catboost` / `gbdt` / `adaboost` |
| `--sample-prob` | `1.0` | % del train_set a traer (0.5-2.0 típico) |
| `--val-sample-prob` | `2.0` | % del val_set 2024 para evaluar |
| `--tuner` | `optuna` | `optuna` (bayesiano) o `randomized` (RandomizedSearchCV) |
| `--n-iter` | `30` | Trials de tuning |
| `--cv-splits` | `3` | Splits de TimeSeriesSplit |
| `--no-log-target` | off | Desactiva `log1p/expm1` (no recomendado) |
| `--batch-size` | `100_000` | Filas por batch en OOC |
| `--ooc-rounds` | `30` | Árboles a agregar por batch en OOC |
| `--skip-snowflake` | off | Salta evaluación en val_set (CI/dev offline) |

---

## 4. Alineación con la rúbrica

| Criterio | Puntaje | Estado |
|----------|---------|--------|
| Data Engineering (SQL) | 15 | ✅ OBT + splits + cleaned (con trip_distance corregido) |
| Experimentación y Ensambles | 25 | ✅ Voting/Bagging/Pasting + AdaBoost/GBDT/XGBoost/LightGBM/CatBoost en NB04, OOC en `train_model.py` |
| Métricas (RMSE en test) | 15 | ⚠️ Pendiente correr el pipeline real (ver §5) |
| Software y Despliegue | 15 | ✅ FastAPI funcional + Streamlit + 80 tests verdes |
| Defensa Final | 30 | N/A (presentación) |

### Penalizaciones evitadas
- **Data Leakage** (-50 pts): `LEAKAGE_COLUMNS` en `build_features.py`
  excluye `fare_amount`, `tip_amount`, `tolls_amount`, `dropoff_*`,
  `trip_duration_min`, `avg_speed_mph`, `improvement_surcharge`, `tip_pct`,
  `run_id`, `ingested_at_utc`. Tests automáticos verifican que ninguna
  llega al modelo final.
- **No usar muestras/lotes** (-50 pts): `fetch_sample` (TABLESAMPLE) y
  `fetch_data_in_batches` (cursor pushdown). El OOC en `train_model.py`
  itera sobre todos los batches sin cargar 20GB en RAM.
- **Faltante de algoritmos** (-10 pts c/u): los 5 boostings obligatorios
  están en `_build_regressor`.

---

## 5. Lo que falta / limitaciones reconocidas

### Bloqueantes pendientes
1. **Re-ejecutar `03_clean_obt.sql` en Snowflake**. La tabla actual en la
   DB todavía está sin `trip_distance` (fue creada antes del fix). Hasta
   que se re-cree, los splits no tienen la feature.
2. **Correr `train_model.py --mode tune` y `--mode oof` con datos reales**.
   Yo no tengo acceso a Snowflake desde acá, así que no produje un .pkl
   real. El RMSE final del entregable depende de tu corrida.

### Mejoras opcionales (siguientes iteraciones para bajar RMSE)
- **Haversine entre centroides de zonas TLC** (requiere agregar lat/lon al
  lookup de zonas — no está en la OBT actual).
- **Target encoding por CV interno** para `pu_zone` y `do_zone` (pueden
  tener 200+ categorías). Más señal que el OrdinalEncoder actual.
- **Feature de tarifa esperada por `rate_code_id`** (Standard ≈ 2.50 + 2.5/mile,
  JFK = 52, etc.) como predictor explícito.
- **Stacking** LightGBM + XGBoost + CatBoost como base + Ridge meta.
- **OOC para CatBoost**: actualmente sólo soporta LightGBM y XGBoost en
  `train_out_of_core`. Para CatBoost requiere `Pool` + `init_model` cada
  batch (factible, pero no implementado por simplicidad).

### Observaciones menores
- Warning "X does not have valid feature names" en LightGBM 4.x es benigno
  (ocurre cuando el `ColumnTransformer` retorna un numpy array sin nombres
  y el booster recuerda los nombres del fit). No afecta predicciones.
- El `improvement_surcharge` ya está en `LEAKAGE_COLUMNS`. Verificar que en
  producción `payment_type_desc='Voided trip'/'Dispute'` no se filtre como
  input — son post-viaje (riesgo bajo si el usuario aún no pagó).

---

## 6. Archivos modificados / creados

### Modificados
- `src/data/sql/03_clean_obt.sql` — fix de `trip_distance` en SELECT.
- `requirements.txt` — añade ML/serving deps + pin de Starlette.
- `src/models/train_model.py` — reescrito completo (~440 líneas).
- `src/models/predict_model.py` — reescrito completo.
- `src/api/main.py` — reescrito completo (FastAPI + lifespan).
- `app/frontend.py` — reescrito completo (Streamlit form + API client).

### Creados
- `tests/test_train_predict_api.py` — 19 nuevos tests.
- `IMPLEMENTACIONES.md` — este documento.

### No tocados
- `notebooks/*.ipynb` — quedan tal cual; los flows productivos viven en
  `src/`. Los notebooks siguen siendo el sandbox exploratorio.
- `src/features/build_features.py` — ya estaba bien diseñado.
- `src/data/ingestion.py` — sólo se usa lazy-import desde train_model.
- `src/utils/config.py` — correcto.
- `tests/test_features.py` — los 60+ tests existentes siguen pasando.
- `docs/obt_splits_documentation.md` — sigue siendo válido.
