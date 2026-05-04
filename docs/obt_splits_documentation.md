# Documentación de Modelado de Datos: OBT y Splits

## 1. Columnas de la OBT (`obt_trips_model`)
La tabla final `obt_trips_model` se construye enriqueciendo los datos base y creando nuevas características temporales. Las columnas resultantes son:

### Datos base del viaje:
- `pickup_datetime`: Fecha y hora de inicio del viaje.
- `dropoff_datetime`: Fecha y hora de fin del viaje.
- `pu_location_id`: ID de la ubicación de origen.
- `do_location_id`: ID de la ubicación de destino.
- `vendor_name`: Nombre del proveedor (ej. Creative Mobile Technologies, Curb Mobility, etc.), mapeado a partir del `VendorID`.
- `rate_code_id`: ID del código de tarifa.
- `rate_code_desc`: Descripción de la tarifa (ej. Standard rate, JFK, Newark), mapeado a partir del ID.
- `payment_type_desc`: Descripción del método de pago (ej. Cash, Flex Fare trip, Dispute), mapeado a partir del ID.
- `passenger_count`: Cantidad de pasajeros.
- `trip_distance`: Distancia del viaje.

### Desglose de tarifas y cobros:
- `fare_amount`: Monto de la tarifa.
- `extra`: Cargos extra.
- `mta_tax`: Impuesto MTA.
- `tip_amount`: Monto de propina.
- `tolls_amount`: Monto de peajes.
- `improvement_surcharge`: Recargo por mejoras.
- `total_amount`: Monto total cobrado.
- `congestion_surcharge`: Recargo por congestión.
- `airport_fee`: Tarifa de aeropuerto.

### Datos de origen y auditoría (renombrados de la fuente original):
- `run_id`: ID de la ejecución de ingesta.
- `ingested_at_utc`: Marca de tiempo de ingesta.
- `month` (originalmente `source_month`).
- `year` (originalmente `source_year`).
- `source_service` (originalmente `service_type`).

### Datos de zonas enriquecidos (vía JOIN con `TAXI_ZONE_LOOKUP`):
- `pu_zone`: Nombre de la zona de origen.
- `pu_borough`: Distrito (Borough) de origen.
- `do_zone`: Nombre de la zona de destino.
- `do_borough`: Distrito (Borough) de destino.

### Nuevas características temporales extraídas:
- `pickup_date`: Fecha de inicio (sin hora).
- `pickup_hour`: Hora del día de inicio.
- `dropoff_date`: Fecha de fin (sin hora).
- `dropoff_hour`: Hora del día de fin.
- `day_of_week`: Día de la semana en que inició el viaje.

---

## 2. Separación de los datos por rangos de fechas (Splits)
Los datos se separan temporalmente tomando como base el año de la columna `pickup_datetime`. Se materializan como vistas (`VIEW`) en Snowflake de la siguiente manera:

- **Train Set (`train_set`):** Viajes ocurridos entre los años **2015 y 2023** (ambos inclusive).
- **Validation Set (`val_set`):** Viajes ocurridos en el año **2024**.
- **Test Set (`test_set`):** Viajes ocurridos en el año **2025**.
