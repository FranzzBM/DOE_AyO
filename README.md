# Tarea 3 – Ajuste de Parámetros con Diseño de Experimentos (DOE)

**Algoritmia y Optimización – UANL, FCFM**
Francisco Blanco Morales · Matrícula 2118882 · Grupo 031

Ajuste de la temperatura inicial T₀ del Recocido Simulado (2-opt) sobre TSP, mediante un
**Diseño Factorial Completo (Opción A)**: 3 niveles de temperatura × 3 tamaños (n = 5, 10, 15) × 10 réplicas = 90 corridas.

## Estructura

```
DOE_AyO/
├── README.md
├── requirements.txt
├── src/
│   └── doe.py      # experimento, ANOVA, bootstrap y gráficas
├── results/                 # se genera al ejecutar (CSV, PNG, diagnósticos)
└── docs/
    └── Reporte_Tarea3_AyO_2118882.pdf
```

## Ejecución

```bash
pip install -r requirements.txt
python src/doe.py            # guarda todo en results/
python src/doe.py --mostrar  # además abre las gráficas en pantalla
```

## Diseño experimental

| Elemento | Valor |
|---|---|
| Semilla de r | `random.seed(2118882); r = random.random()` (una sola vez) |
| Rango de temperatura | [T_mín, T_máx] = [10, 500] |
| Niveles (fracciones) | bajo = 0.3·r, medio = 0.3·r + 0.3, alto = 0.3·r + 0.6 |
| Conversión | T = T_mín + fracción · (T_máx − T_mín) |
| Enfriamiento | exponencial, T(t) = T₀ · 0.995ᵗ, 2000 iteraciones |
| Costo reportado | longitud de la mejor ruta encontrada en la corrida |
| Instancias | una por tamaño, fija (semilla derivada de la matrícula) |
| Réplicas | 10 por punto de diseño, cada una con semilla distinta |
| Análisis | ANOVA de dos factores (tipo II), α = 0.05; bootstrap de la mediana (2000 remuestreos, IC 95 % por percentiles) |

## Archivos generados en `results/`

| Archivo | Contenido |
|---|---|
| `datos_crudos.csv` | Las 90 corridas (tamaño, nivel, T₀, réplica, semilla, costo) |
| `anova_dos_factores.csv` | Tabla ANOVA (suma de cuadrados, gl, F, p, η²) |
| `bootstrap_puntos_diseno.csv` | Mediana bootstrap e IC 95 % de los 9 puntos de diseño |
| `tabla_resumen_optimos.csv` | Temperatura seleccionada por tamaño, con mediana e IC |
| `diagnosticos.txt` | Shapiro-Wilk, Levene y comparaciones Tukey por tamaño |
| `efectos_principales.png` | Gráfica de efectos principales (temperatura y tamaño) |
| `interaccion.png` | Gráfica de interacción temperatura × tamaño |
| `cajas_por_tamano.png` | Diagramas de caja por tamaño (apoyo) |

## Regla de selección de la temperatura óptima

Para cada tamaño se elige el nivel de T₀ con **menor mediana bootstrap** del costo (empates: menor T₀).
El ANOVA de dos factores indica si la temperatura (o su interacción con el tamaño) tiene efecto
significativo; además se reporta un ANOVA de una vía y Tukey por tamaño para saber si la diferencia
entre niveles dentro de cada n es estadísticamente distinguible. Si no lo es, el nivel elegido
es solo el de menor mediana y debe interpretarse como estadísticamente equivalente a los demás.
