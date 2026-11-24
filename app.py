"""Punto de entrada de Streamlit para el proyecto integrador de la Unidad 4."""

import numpy as np
import pandas as pd
import streamlit as st

from analisis import (
    GRUPO, RESPUESTA, TRATAMIENTOS, ajustar_ols, cargar_parcelas, comparar_tratamientos,
)
from graficas import cajas_tratamientos, dispersion_riego, histograma_residuos, mapa_correlaciones

st.set_page_config(page_title="Rendimiento agrícola · PI-U4", layout="wide")


@st.cache_data
def parcelas():
    return cargar_parcelas()


def main():
    datos = parcelas()
    st.title("Rendimiento agrícola en invernadero")
    st.caption(f"Proyecto integrador U4 · {len(datos)} parcelas simuladas en DuckDB · Semilla 0.42")
    opcion = st.sidebar.radio("Tratamiento", ("Todos", "A", "B"))
    filtrados = (
        datos if opcion == "Todos" else datos.loc[datos[GRUPO] == TRATAMIENTOS[opcion]]
    )

    modelo = ajustar_ols(filtrados)
    a, b, prueba, intervalo = comparar_tratamientos(datos)
    diferencia = b.mean() - a.mean()

    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric("Rendimiento promedio global", f"{datos[RESPUESTA].mean():.2f} t/ha")
    kpi2.metric("R² ajustado del modelo visible", f"{modelo.rsquared_adj:.1%}")
    kpi3.metric("Valor p unilateral global (B > A)", f"{prueba.pvalue:.7f}")
    st.caption(f"Vista de correlación, ajuste y residuos para {len(filtrados)} parcelas.")

    izquierda, derecha = st.columns(2)
    with izquierda:
        st.subheader("Correlación de Pearson")
        st.pyplot(mapa_correlaciones(filtrados))
        st.subheader("Rendimiento por tratamiento")
        st.pyplot(cajas_tratamientos(datos))
        st.caption("Comparación con ambos grupos. Los rombos muestran las medias y sus IC del 95 %.")
    with derecha:
        st.subheader("Riego y rendimiento")
        st.pyplot(dispersion_riego(filtrados, modelo))
        st.caption("La recta múltiple se muestra con el fertilizante fijo en su promedio.")
        st.subheader("Distribución de residuos")
        st.pyplot(histograma_residuos(modelo))

    st.header("Comparación de tratamientos")
    st.latex(r"H_0:mu_B-mu_Aleq0qquad H_1:mu_B-mu_A>0")
    st.write(
        f"A: {a.mean():.2f} t/ha (n = {len(a)}) · "
        f"B: {b.mean():.2f} t/ha (n = {len(b)}) · "
        f"Diferencia B − A: {diferencia:+.2f} t/ha"
    )
    st.write(
        f"t de Student: {prueba.statistic:.3f} (gl = {int(prueba.df)}) · "
        f"IC bilateral del 95 %: [{intervalo.low:.2f}, {intervalo.high:.2f}] t/ha"
    )
    decision = (
        "Se rechaza H₀. La simulación aporta evidencia de un mayor rendimiento con B."
        if prueba.pvalue < 0.05
        else "No se rechaza H₀. La simulación no aporta evidencia suficiente de un mayor rendimiento con B."
    )
    st.info(decision)
    st.caption("El efecto de B se programó en la simulación. Una decisión de campo requiere datos observados y costos.")

    with st.expander("Coeficientes del modelo y datos de la selección"):
        rmse = np.sqrt(np.mean(modelo.resid ** 2))
        st.write(f"R²: {modelo.rsquared:.3f} · RMSE del ajuste: {rmse:.3f} t/ha")
        coeficientes = pd.DataFrame({
            "Término": ["Constante", "Riego (m³/ha)", "Fertilizante (kg/ha)"],
            "Coeficiente": modelo.params.to_numpy(),
            "Valor p": modelo.pvalues.to_numpy(),
        })
        st.dataframe(coeficientes, hide_index=True)
        st.dataframe(filtrados, hide_index=True)


if __name__ == "__main__":
    main()
