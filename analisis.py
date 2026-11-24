"""Preparación y análisis estadístico de la telemetría del invernadero."""

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

PREDICTORES = ["volumen_riego", "fertilizante_kg"]
RESPUESTA = "rendimiento_ton"
GRUPO = "tipo_tratamiento"
TRATAMIENTOS = {"A": "Tratamiento_A", "B": "Tratamiento_B"}
ETIQUETAS = {
    "volumen_riego": "Riego",
    "fertilizante_kg": "Fertilizante",
    "temperatura_c": "Temperatura",
    "rendimiento_ton": "Rendimiento",
}


def cargar_parcelas():
    consulta = Path(__file__).with_name("telemetria.sql").read_text(encoding="utf-8")
    with duckdb.connect(":memory:", config={"threads": 1}) as conexion:
        conexion.execute(consulta)
        return conexion.execute("""
            SELECT parcela_id, volumen_riego, fertilizante_kg, tipo_tratamiento,
                   rendimiento_ton, temperatura_c
            FROM parcelas
            WHERE parcela_id BETWEEN 1 AND 100
              AND volumen_riego BETWEEN 100 AND 300
              AND fertilizante_kg BETWEEN 50 AND 150
              AND tipo_tratamiento IN ('Tratamiento_A', 'Tratamiento_B')
              AND rendimiento_ton > 0
              AND temperatura_c BETWEEN 20 AND 30
            QUALIFY row_number() OVER (PARTITION BY parcela_id ORDER BY parcela_id) = 1
            ORDER BY parcela_id
        """).df()


def correlaciones(datos):
    return datos[list(ETIQUETAS)].rename(columns=ETIQUETAS).corr(method="pearson")


def ajustar_ols(datos):
    matriz = sm.add_constant(datos[PREDICTORES], has_constant="add")
    return sm.OLS(datos[RESPUESTA], matriz).fit()


def linea_ols(datos, modelo):
    riego = np.linspace(datos["volumen_riego"].min(), datos["volumen_riego"].max(), 100)
    predictores = pd.DataFrame({
        "volumen_riego": riego,
        "fertilizante_kg": datos["fertilizante_kg"].mean(),
    })
    matriz = sm.add_constant(predictores, has_constant="add")
    prediccion = modelo.get_prediction(matriz).summary_frame(alpha=0.05)
    return riego, prediccion


def comparar_tratamientos(datos):
    a = datos.loc[datos[GRUPO] == TRATAMIENTOS["A"], RESPUESTA]
    b = datos.loc[datos[GRUPO] == TRATAMIENTOS["B"], RESPUESTA]
    prueba = stats.ttest_ind(b, a, equal_var=True, alternative="greater")
    intervalo = stats.ttest_ind(b, a, equal_var=True).confidence_interval(0.95)
    return a, b, prueba, intervalo
