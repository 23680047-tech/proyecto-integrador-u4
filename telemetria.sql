SELECT setseed(0.42);

CREATE TABLE parcelas AS
WITH muestras AS MATERIALIZED (
    SELECT
        parcela_id,
        round(100 + 200 * random(), 2) AS volumen_riego,
        round(50 + 100 * random(), 2) AS fertilizante_kg,
        CASE WHEN random() < 0.50 THEN 'Tratamiento_A'
             ELSE 'Tratamiento_B' END AS tipo_tratamiento,
        1.6 * random() - 0.8 AS ruido
    FROM range(1, 101) AS parcelas(parcela_id)
)
SELECT
    parcela_id,
    volumen_riego,
    fertilizante_kg,
    tipo_tratamiento,
    round(5.0 + 0.035 * volumen_riego + 0.020 * fertilizante_kg
          + CASE WHEN tipo_tratamiento = 'Tratamiento_B' THEN 1.8 ELSE 0.0 END
          + ruido, 2) AS rendimiento_ton
FROM muestras;

-- La guía pide temperatura para el mapa de calor. Se genera después
-- de los datos principales y se utiliza únicamente para exploración.
ALTER TABLE parcelas ADD COLUMN temperatura_c DOUBLE;
UPDATE parcelas SET temperatura_c = round(20 + 10 * random(), 2);
