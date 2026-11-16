import argparse
import math
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

MATRICULA = 2118882          # semilla para generar r (una sola vez)
T_MIN, T_MAX = 10.0, 500.0   # rango de temperatura definido para el problema
TAMANOS = [5, 10, 15]        # bloque: numero de ciudades
N_REPLICAS = 10              # corridas por punto de diseno
MAX_ITER = 2000              # iteraciones del recocido en cada corrida
ALPHA_ENFRIAMIENTO = 0.995   # T(t) = T0 * alpha^t  (enfriamiento exponencial)
N_BOOTSTRAP = 2000           # remuestreos bootstrap
ALFA = 0.05                  # nivel de significancia (IC 95 %)
NOMBRES_NIVELES = ["Bajo", "Medio", "Alto"]

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")

def generar_niveles():
    random.seed(MATRICULA)
    r = random.random()          
    fracciones = [0.3 * r, 0.3 * r + 0.3, 0.3 * r + 0.6]
    temperaturas = [T_MIN + f * (T_MAX - T_MIN) for f in fracciones]
    return r, fracciones, temperaturas

def generar_instancia(n):
    rng = random.Random(MATRICULA * 100 + n)
    return [(rng.uniform(0, 100), rng.uniform(0, 100)) for _ in range(n)]


def longitud_ruta(ciudades, ruta):
    return sum(math.dist(ciudades[ruta[i]], ciudades[ruta[(i + 1) % len(ruta)]])
               for i in range(len(ruta)))


def movimiento_2opt(ruta, rng):
    i = rng.randint(1, len(ruta) - 3)
    k = rng.randint(i + 1, len(ruta) - 1)
    return ruta[:i] + ruta[i:k + 1][::-1] + ruta[k + 1:]


def recocido_simulado(ciudades, T0, rng, max_iter=MAX_ITER, alpha=ALPHA_ENFRIAMIENTO):
    ruta = list(range(len(ciudades)))
    rng.shuffle(ruta)
    costo_actual = longitud_ruta(ciudades, ruta)
    mejor = costo_actual

    for t in range(1, max_iter):
        T = max(T0 * (alpha ** t), 1e-12)
        candidata = movimiento_2opt(ruta, rng)
        costo_cand = longitud_ruta(ciudades, candidata)
        delta_E = costo_actual - costo_cand
        if delta_E > 0 or rng.random() < math.exp(delta_E / T):
            ruta, costo_actual = candidata, costo_cand
            mejor = min(mejor, costo_actual)
    return mejor

def correr_experimento(temperaturas):
    filas = []
    for n in TAMANOS:
        ciudades = generar_instancia(n)
        for j, (nombre, T0) in enumerate(zip(NOMBRES_NIVELES, temperaturas)):
            etiqueta = f"{nombre} ({T0:.2f})"
            for rep in range(N_REPLICAS):
                semilla = MATRICULA + 1000 * n + 100 * j + rep
                costo = recocido_simulado(ciudades, T0, random.Random(semilla))
                filas.append({"tamano": n, "nivel": nombre, "temperatura": etiqueta,
                              "T0": T0, "replica": rep + 1, "semilla": semilla,
                              "costo": costo})
    return pd.DataFrame(filas)

def anova_dos_factores(df):
    modelo = ols("costo ~ C(temperatura) * C(tamano)", data=df).fit()
    tabla = sm.stats.anova_lm(modelo, typ=2)
    tabla["eta_cuadrada"] = tabla["sum_sq"] / tabla["sum_sq"].sum()
    return modelo, tabla

def diagnosticos(modelo, df):
    lineas = []
    sh = stats.shapiro(modelo.resid)
    lineas.append(f"Shapiro-Wilk (residuos): W = {sh.statistic:.4f}, p = {sh.pvalue:.4g}")
    grupos = [g["costo"].values for _, g in df.groupby(["tamano", "temperatura"])]
    lv = stats.levene(*grupos, center="median")
    lineas.append(f"Levene (9 puntos de diseno): estadistico = {lv.statistic:.4f}, p = {lv.pvalue:.4g}")
    return "\n".join(lineas)

def bootstrap_mediana(x, rng, n_boot=N_BOOTSTRAP, alfa=ALFA):
    x = np.asarray(x, dtype=float)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    medianas = np.median(x[idx], axis=1)
    return (float(np.median(x)),
            float(np.percentile(medianas, 100 * alfa / 2)),
            float(np.percentile(medianas, 100 * (1 - alfa / 2))))


def tabla_bootstrap(df):
    rng = np.random.default_rng(MATRICULA)
    filas = []
    for (n, nivel), g in df.groupby(["tamano", "nivel"], sort=False):
        med, lo, hi = bootstrap_mediana(g["costo"].values, rng)
        filas.append({"tamano": n, "nivel": nivel, "T0": g["T0"].iloc[0],
                      "mediana_bootstrap": med, "IC95_inf": lo, "IC95_sup": hi})
    return pd.DataFrame(filas)

def seleccionar_optimos(df, boot):
    filas, texto = [], []
    for n in TAMANOS:
        sub = boot[boot["tamano"] == n].sort_values(["mediana_bootstrap", "T0"])
        mejor = sub.iloc[0]
        d = df[df["tamano"] == n]
        grupos = [d[d["nivel"] == nv]["costo"].values for nv in NOMBRES_NIVELES]
        todos_iguales = np.ptp(d["costo"].values) < 1e-9
        p_anova = 1.0 if todos_iguales else stats.f_oneway(*grupos).pvalue
        significativo = p_anova < ALFA
        filas.append({"tamano": n, "nivel_optimo": mejor["nivel"], "T_optima": mejor["T0"],
                      "mediana_bootstrap": mejor["mediana_bootstrap"],
                      "IC95_inf": mejor["IC95_inf"], "IC95_sup": mejor["IC95_sup"],
                      "p_ANOVA_una_via": p_anova,
                      "diferencia_entre_niveles_significativa": significativo})
        texto.append(f"--- n = {n}: ANOVA de una via entre niveles, p = {p_anova:.4g} "
                     f"({'significativo' if significativo else 'NO significativo'}) ---")
        if not todos_iguales:
            try:
                texto.append(str(pairwise_tukeyhsd(d["costo"], d["nivel"], alpha=ALFA)))
            except Exception as e:  # pragma: no cover
                texto.append(f"(Tukey no disponible: {e})")
        else:
            texto.append("Los 30 costos son identicos: no hay variabilidad entre niveles.")
    return pd.DataFrame(filas), "\n".join(texto)

def _media_ic(x):
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return x.mean(), 0.0
    return x.mean(), stats.sem(x) * stats.t.ppf(1 - ALFA / 2, len(x) - 1)

def grafica_efectos_principales(df, temperaturas, ruta):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    etiquetas = [f"{nv}\n({T:.2f})" for nv, T in zip(NOMBRES_NIVELES, temperaturas)]

    stats_t = [_media_ic(df[df["nivel"] == nv]["costo"]) for nv in NOMBRES_NIVELES]
    ax1.errorbar(range(3), [s[0] for s in stats_t], yerr=[s[1] for s in stats_t],
                 marker="o", capsize=5, color="tab:blue")
    ax1.set_xticks(range(3)); ax1.set_xticklabels(etiquetas)
    ax1.set_title("Efecto principal: temperatura inicial")
    ax1.set_xlabel("Nivel de T0"); ax1.set_ylabel("Costo medio (longitud del tour)")
    ax1.grid(True, alpha=0.4)

    stats_n = [_media_ic(df[df["tamano"] == n]["costo"]) for n in TAMANOS]
    ax2.errorbar(range(3), [s[0] for s in stats_n], yerr=[s[1] for s in stats_n],
                 marker="s", capsize=5, color="tab:green")
    ax2.set_xticks(range(3)); ax2.set_xticklabels([f"n = {n}" for n in TAMANOS])
    ax2.set_title("Efecto principal: tamano del problema")
    ax2.set_xlabel("Tamano de la instancia"); ax2.set_ylabel("Costo medio (longitud del tour)")
    ax2.grid(True, alpha=0.4)

    fig.suptitle("Grafica de efectos principales (media e IC 95 %)")
    fig.tight_layout(); fig.savefig(ruta, dpi=200); return fig


def grafica_interaccion(df, temperaturas, ruta):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))
    etiquetas = [f"{nv}\n({T:.2f})" for nv, T in zip(NOMBRES_NIVELES, temperaturas)]
    colores = {5: "tab:blue", 10: "tab:orange", 15: "tab:green"}
    for n in TAMANOS:
        d = df[df["tamano"] == n]
        st = [_media_ic(d[d["nivel"] == nv]["costo"]) for nv in NOMBRES_NIVELES]
        ax1.errorbar(range(3), [s[0] for s in st], yerr=[s[1] for s in st],
                     marker="o", capsize=4, label=f"n = {n}", color=colores[n])
        media_n = d["costo"].mean()
        rel = [_media_ic(d[d["nivel"] == nv]["costo"] / media_n) for nv in NOMBRES_NIVELES]
        ax2.errorbar(range(3), [s[0] for s in rel], yerr=[s[1] for s in rel],
                     marker="o", capsize=4, label=f"n = {n}", color=colores[n])
    for ax, tit, yl in [(ax1, "Interaccion temperatura x tamano (escala original)", "Costo medio"),
                        (ax2, "Misma interaccion, costo relativo a la media de cada n",
                         "Costo / media del tamano")]:
        ax.set_xticks(range(3)); ax.set_xticklabels(etiquetas)
        ax.set_title(tit, fontsize=10); ax.set_xlabel("Nivel de T0"); ax.set_ylabel(yl)
        ax.legend(); ax.grid(True, alpha=0.4)
    fig.tight_layout(); fig.savefig(ruta, dpi=200); return fig


def grafica_cajas(df, temperaturas, ruta):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    for ax, n in zip(axes, TAMANOS):
        d = df[df["tamano"] == n]
        ax.boxplot([d[d["nivel"] == nv]["costo"] for nv in NOMBRES_NIVELES],
                   tick_labels=[f"{nv}\n({T:.2f})" for nv, T in zip(NOMBRES_NIVELES, temperaturas)])
        ax.set_title(f"n = {n}"); ax.set_xlabel("Nivel de T0"); ax.set_ylabel("Costo")
        ax.grid(True, axis="y", alpha=0.4)
    fig.suptitle("Distribucion del costo por nivel de temperatura (10 replicas)")
    fig.tight_layout(); fig.savefig(ruta, dpi=200); return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mostrar", action="store_true", help="mostrar las graficas en pantalla")
    args = parser.parse_args()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    pd.set_option("display.width", 160); pd.set_option("display.max_columns", 20)

    r, fracciones, temperaturas = generar_niveles()
    print("=" * 70)
    print(f"Matricula (semilla)      : {MATRICULA}")
    print(f"r ~ U(0,1) (fijo)        : {r:.6f}")
    print(f"Rango de temperatura     : [{T_MIN}, {T_MAX}]")
    for nv, f, T in zip(NOMBRES_NIVELES, fracciones, temperaturas):
        print(f"  Nivel {nv:<5}: fraccion = {f:.6f}  ->  T0 = {T:.2f}")
    print("=" * 70)

    df = correr_experimento(temperaturas)
    df.to_csv(os.path.join(RESULTS_DIR, "datos_crudos.csv"), index=False)
    print(f"\nCorridas realizadas: {len(df)} (esperado: 90)")
    print("\nMedia y desv. est. por punto de diseno:")
    print(df.groupby(["tamano", "nivel"], sort=False)["costo"].agg(["mean", "std", "median"]).round(4))

    modelo, tabla = anova_dos_factores(df)
    tabla.to_csv(os.path.join(RESULTS_DIR, "anova_dos_factores.csv"))
    print("\n=== ANOVA de dos factores (tipo II) ===")
    print(tabla)
    diag = diagnosticos(modelo, df)
    print("\n=== Diagnosticos de supuestos ===\n" + diag)

    boot = tabla_bootstrap(df)
    boot.to_csv(os.path.join(RESULTS_DIR, "bootstrap_puntos_diseno.csv"), index=False)
    print(f"\n=== Bootstrap de la mediana ({N_BOOTSTRAP} remuestreos, IC 95 %) ===")
    print(boot.round(4))

    optimos, texto_tukey = seleccionar_optimos(df, boot)
    optimos.to_csv(os.path.join(RESULTS_DIR, "tabla_resumen_optimos.csv"), index=False)
    print("\n=== Comparacion entre niveles por tamano ===\n" + texto_tukey)
    print("\n=== TABLA RESUMEN: temperatura optima por tamano ===")
    print(optimos.round(4))

    with open(os.path.join(RESULTS_DIR, "diagnosticos.txt"), "w", encoding="utf-8") as f:
        f.write(diag + "\n\n" + texto_tukey + "\n")

    grafica_efectos_principales(df, temperaturas, os.path.join(RESULTS_DIR, "efectos_principales.png"))
    grafica_interaccion(df, temperaturas, os.path.join(RESULTS_DIR, "interaccion.png"))
    grafica_cajas(df, temperaturas, os.path.join(RESULTS_DIR, "cajas_por_tamano.png"))
    print(f"\nResultados y graficas guardados en: {os.path.abspath(RESULTS_DIR)}")
    if args.mostrar:
        plt.show()

if __name__ == "__main__":
    main()
