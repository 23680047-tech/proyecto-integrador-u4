"""Figuras del dashboard y del informe."""

import numpy as np
import seaborn as sns
from matplotlib.figure import Figure
from scipy import stats

from analisis import GRUPO, RESPUESTA, TRATAMIENTOS, correlaciones, linea_ols

VERDE = "#116a51"
AZUL = "#345f9c"
GRIS = "#445663"


def lienzo():
    figura = Figure(figsize=(6.5, 4.05), layout="constrained")
    eje = figura.subplots()
    eje.spines[["top", "right"]].set_visible(False)
    return figura, eje


def mapa_correlaciones(datos):
    figura, eje = lienzo()
    sns.heatmap(
        correlaciones(datos), annot=True, fmt=".2f", vmin=-1, vmax=1, center=0,
        cmap="RdBu", linewidths=0.6, cbar_kws={"label": "r de Pearson"}, ax=eje,
    )
    eje.tick_params(axis="both", labelrotation=0)
    return figura


def dispersion_riego(datos, modelo):
    figura, eje = lienzo()
    riego, prediccion = linea_ols(datos, modelo)
    eje.scatter(
        datos["volumen_riego"], datos[RESPUESTA], s=19, alpha=0.65, color=AZUL,
        label="Parcelas",
    )
    eje.fill_between(
        riego, prediccion["mean_ci_lower"], prediccion["mean_ci_upper"],
        color=VERDE, alpha=0.18, label="IC del 95 % de la media",
    )
    eje.plot(riego, prediccion["mean"], color=VERDE, linewidth=2, label="Ajuste OLS")
    eje.set(xlabel="Riego (m³/ha)", ylabel="Rendimiento (t/ha)")
    eje.legend(frameon=False, fontsize=8)
    return figura


def cajas_tratamientos(datos):
    figura, eje = lienzo()
    a, b = TRATAMIENTOS.values()
    sns.boxplot(
        data=datos, x=GRUPO, y=RESPUESTA, order=[a, b], hue=GRUPO,
        palette={a: "#a8c2e1", b: "#86c7ab"}, width=0.48, legend=False, ax=eje,
    )
    for posicion, tratamiento in enumerate((a, b)):
        grupo = datos.loc[datos[GRUPO] == tratamiento, RESPUESTA]
        media = grupo.mean()
        margen = stats.t.ppf(0.975, len(grupo) - 1) * stats.sem(grupo)
        eje.errorbar(
            posicion + 0.30, media, yerr=margen, fmt="D", color=GRIS, capsize=4,
            markersize=5, label="Media e IC del 95 %" if posicion == 0 else None,
        )
    eje.set(xticks=[0, 1], xticklabels=["A · Estándar", "B · Bioestimulante"],
            xlabel="Tratamiento", ylabel="Rendimiento (t/ha)")
    eje.legend(frameon=False, fontsize=8, loc="upper left")
    return figura


def histograma_residuos(modelo):
    figura, eje = lienzo()
    residuos = modelo.resid
    media, desviacion = stats.norm.fit(residuos)
    dominio = np.linspace(residuos.min() - 0.3, residuos.max() + 0.3, 240)
    eje.hist(residuos, bins="auto", density=True, color="#90b9df", edgecolor="white")
    eje.plot(dominio, stats.gaussian_kde(residuos)(dominio), color=AZUL, label="Densidad KDE")
    eje.plot(dominio, stats.norm.pdf(dominio, media, desviacion), "--",
             color=VERDE, label="Normal ajustada")
    eje.axvline(0, color=GRIS, linewidth=0.8, alpha=0.55)
    eje.set(xlabel="Rendimiento observado − predicho (t/ha)", ylabel="Densidad")
    eje.legend(frameon=False, fontsize=8)
    return figura
