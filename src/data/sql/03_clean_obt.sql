-- ==========================================
-- 03: Limpieza de la One Big Table (OBT) en Base de Datos
-- ==========================================
-- Se aplican reglas lógicas detectadas en el EDA, 
-- delegando el cómputo a Snowflake (Pushdown Computation) 
-- para evitar saturar la memoria local (Pandas) y creando 
-- una tabla limpia lista para el feature engineering.

CREATE OR REPLACE TABLE NYC_TAXI_P5.ANALYTICS.cleaned_obt_trips AS
SELECT
    pickup_datetime,
    pu_location_id,
    do_location_id,
    COALESCE(vendor_name, 'Unknown') AS vendor_name,
    COALESCE(rate_code_id, 1) AS rate_code_id,
    COALESCE(rate_code_desc, 'Unknown') AS rate_code_desc,
    COALESCE(payment_type_desc, 'Unknown') AS payment_type_desc,
    COALESCE(passenger_count, 1) AS passenger_count,
    total_amount,
    month,
    year,
    source_service,
    COALESCE(pu_zone, 'Unknown') AS pu_zone,
    COALESCE(pu_borough, 'Unknown') AS pu_borough,
    COALESCE(do_zone, 'Unknown') AS do_zone,
    COALESCE(do_borough, 'Unknown') AS do_borough,
    pickup_date,
    pickup_hour,
    day_of_week
FROM NYC_TAXI_P5.ANALYTICS.obt_trips_model
WHERE total_amount > 2.50 AND total_amount <= 500.0
  AND trip_distance > 0 AND trip_distance <= 200.0
  AND passenger_count BETWEEN 1 AND 6
  AND pickup_datetime IS NOT NULL;
