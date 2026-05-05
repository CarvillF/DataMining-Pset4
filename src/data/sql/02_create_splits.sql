-- ==========================================
-- 02: Separación Temporal (Data Splits)
-- ==========================================
-- Los conjuntos de entrenamiento, validación y prueba NO se crearán 
-- en la máquina local usando train_test_split. Se materializarán en Snowflake.

-- * Train: 2015-2023
-- * Validación: 2024
-- * Test: 2025

CREATE OR REPLACE VIEW NYC_TAXI_P5.ANALYTICS.train_set AS
SELECT * FROM NYC_TAXI_P5.ANALYTICS.cleaned_obt_trips 
WHERE EXTRACT(YEAR FROM pickup_datetime) BETWEEN 2015 AND 2023;

CREATE OR REPLACE VIEW NYC_TAXI_P5.ANALYTICS.val_set AS
SELECT * FROM NYC_TAXI_P5.ANALYTICS.cleaned_obt_trips 
WHERE EXTRACT(YEAR FROM pickup_datetime) = 2024;

CREATE OR REPLACE VIEW NYC_TAXI_P5.ANALYTICS.test_set AS
SELECT * FROM NYC_TAXI_P5.ANALYTICS.cleaned_obt_trips 
WHERE EXTRACT(YEAR FROM pickup_datetime) = 2025;
