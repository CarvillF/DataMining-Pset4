-- ==========================================
-- 01: Construcción de la One Big Table (OBT)
-- ==========================================
-- Como el volumen de datos es ~20GB, NO exportaremos múltiples tablas a Python.
-- Toda la lógica de "joinear" o agregar debe correr en el clúster de Snowflake.

CREATE OR REPLACE TABLE analytics.obt_trips_model AS 
WITH base_trips AS (
    SELECT 
        -- Unificación de Fechas
        COALESCE(tpep_pickup_datetime, lpep_pickup_datetime) AS pickup_datetime,
        COALESCE(tpep_dropoff_datetime, lpep_dropoff_datetime) AS dropoff_datetime,
        
        PULocationID AS pu_location_id,
        DOLocationID AS do_location_id,
        
        -- Decodificación de Vendedores
        CASE 
            WHEN VendorID = 1 THEN 'Creative Mobile Technologies, LLC'
            WHEN VendorID = 2 THEN 'Curb Mobility, LLC'
            WHEN VendorID = 6 THEN 'Myle Technologies Inc'
            WHEN VendorID = 7 THEN 'Helix'
            ELSE 'Unknown'
        END AS vendor_name,
        
        CAST(RatecodeID AS INTEGER) AS rate_code_id,
        CASE 
            WHEN CAST(RatecodeID AS INTEGER) = 1 THEN 'Standard rate'
            WHEN CAST(RatecodeID AS INTEGER) = 2 THEN 'JFK'
            WHEN CAST(RatecodeID AS INTEGER) = 3 THEN 'Newark'
            WHEN CAST(RatecodeID AS INTEGER) = 4 THEN 'Nassau or Westchester'
            WHEN CAST(RatecodeID AS INTEGER) = 5 THEN 'Negotiated fare'
            WHEN CAST(RatecodeID AS INTEGER) = 6 THEN 'Group ride'
            ELSE 'Unknown'
        END AS rate_code_desc,
        
        CASE 
            WHEN payment_type = 1 THEN 'Flex Fare trip'
            WHEN payment_type = 2 THEN 'Cash'
            WHEN payment_type = 3 THEN 'No charge'
            WHEN payment_type = 4 THEN 'Dispute'
            WHEN payment_type = 5 THEN 'Unknown'
            WHEN payment_type = 6 THEN 'Voided trip'
            ELSE 'Other'
        END AS payment_type_desc,
        
        passenger_count,
        trip_distance,
        fare_amount,
        extra,
        mta_tax,
        tip_amount,
        tolls_amount,
        improvement_surcharge,
        total_amount,
        congestion_surcharge,
        Airport_fee AS airport_fee,
        
        run_id,
        ingested_at_utc,
        source_year,
        source_month,
        service_type
    FROM raw.trips_raw
),
enriched_trips AS (
    SELECT
        b.*,
        pu.Zone AS pu_zone,
        pu.Borough AS pu_borough,
        do.Zone AS do_zone,
        do.Borough AS do_borough
    FROM base_trips b
    LEFT JOIN raw.taxi_zone_lookup pu ON b.pu_location_id = pu.LocationID
    LEFT JOIN raw.taxi_zone_lookup do ON b.do_location_id = do.LocationID
)
SELECT 
    * EXCLUDE (source_month, source_year, service_type),
    source_month AS month,
    source_year AS year,
    service_type AS source_service,

    -- Componentes de Fecha/Hora
    TO_DATE(pickup_datetime) AS pickup_date,
    DATE_PART('hour', pickup_datetime) AS pickup_hour,
    TO_DATE(dropoff_datetime) AS dropoff_date,
    DATE_PART('hour', dropoff_datetime) AS dropoff_hour,
    DAYOFWEEK(pickup_datetime) AS day_of_week

FROM enriched_trips;