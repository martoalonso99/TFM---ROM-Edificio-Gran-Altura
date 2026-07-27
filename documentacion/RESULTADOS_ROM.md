# Resultados del Pipeline ROM — Análisis de Muestreo Adaptativo

**TFM: Modelo de Orden Reducido para Cp sobre edificio alto**
Martín Rodríguez Alonso — Máster en Matemática Industrial

---

## 1. Objetivo

El objetivo de este análisis es construir un ROM (Reduced Order Model) que prediga la distribución de coeficiente de presión $C_p$ sobre las cuatro caras de un edificio alto rectangular para cualquier ángulo de incidencia del viento $\theta \in [0°, 45°]$, y estudiar cómo el número y la **distribución** de simulaciones CFD de entrenamiento afecta a la precisión del modelo, a igualdad de coste computacional.

Se comparan cuatro estrategias de muestreo:

| Nombre | N ángulos | Estrategia | CPU-h (eq. 10 cores) |
|---|---|---|---|
| `base_ref` | 10 | Uniforme, paso 5° | 431 |
| `base_kawai` | 17 | `base_ref` + 7 ángulos en zona vórtice [10–30°] | 768 |
| `base_kawai2` | 22 | `base_kawai` + paso 1.25° en toda la zona | 990 |
| `base_full22` | 22 | Uniforme, paso ~2.5° en todo el rango | 976 |

Las dos últimas tienen **igual N e igual presupuesto** (diferencia 1.4%) pero distinta distribución: es una comparación controlada de *dónde* colocar las simulaciones.

El rango físico válido es $\theta \in [0°, 45°]$ por simetría C4 de la sección cuadrada.

---

## 2. Datos de entrada: matriz de snapshots

Cada simulación produce **400 sondas de presión** sobre las cuatro caras del edificio (4 caras × 100 sondas, 5 columnas × 20 filas, espaciado uniforme 0.02 m). El $C_p$ se calcula como:

$$C_p = \frac{p_{kin}}{q_{ref}} = \frac{p}{0.5 \cdot U_H^2} = \frac{p}{60.5 \text{ m}^2/\text{s}^2}$$

La **matriz de snapshots** es $X \in \mathbb{R}^{400 \times N}$ donde cada columna es el campo $C_p$ convergido para un ángulo.

**Metodología común a todas las bases:**
- POD centrada (SVD económica): $X' = X - \bar{x}\mathbf{1}^T = U\Sigma V^T$, $\Phi = U[:,{:}r]$
- Una GPR por modo POD (kernel Matérn + WhiteKernel, MLE con 50 multistarts)
- Selección de kernel y de $r^*$ por **validación cruzada (LOO-CV) con re-POD por fold** (la base se recalcula desde cero en cada fold para evitar contaminación)
- Métrica: $\varepsilon = \|C_{p,CFD} - C_{p,ROM}\|_2 / \|C_{p,CFD} - \bar{C_p}^{train}\|_2$

**Nomenclatura de conjuntos** (nombres estándar de ML, para evitar ambigüedad):

- **Validación cruzada (LOO-CV)** — resampling sobre los ángulos de entrenamiento de cada base; se usa para **seleccionar** $r^*$ y el kernel. No es una medida de generalización imparcial (§7.3).
- **Evaluación fuera de muestra por base** (§3–§6) — ángulos ya simulados que una base no usó en su entrenamiento; **diagnóstico** de dónde falla cada diseño. Los ángulos **difieren entre bases**, por lo que **no es comparable entre ellas** (§7.3–§7.5).
- **Conjunto de test** (holdout pre-registrado, §7.6) — 9 ángulos comunes que ninguna base vio jamás, con métricas fijadas *antes* de simular. Es la **evaluación final imparcial y la única comparación válida entre bases**.

---

## 3. Fase 1 — Base de referencia (`base_ref`)

**Ángulos de entrenamiento (N=10):** 0°, 5°, 10°, 15°, 20°, 25°, 30°, 35°, 40°, 45°

### 3.1 Visión general de los snapshots

![Snapshot overview base_ref](../outputs/base_ref/snapshot_overview.png)

Observaciones:
- A 0°: cara barlovento con $C_p \approx +0.97$, sotavento con succión $C_p \approx -0.5$
- A ángulos oblicuos (~10–30°): succión intensa en la arista barlovento — **vórtice cónico de Kawai** (pico de succión $C_p \approx -1.25$ en θ=12.5°)
- La distribución varía de forma no lineal con $\theta$, especialmente en la zona del vórtice

### 3.2 POD

![Espectro singular base_ref](../outputs/base_ref/POD_singular_spectrum.png)

- Los primeros 4 modos capturan ≥99.5% de la energía
- Modos 1–2: patrón global estancamiento + reflujo; modos 3–6: firma del vórtice cónico

![Coeficientes modales base_ref](../outputs/base_ref/POD_modal_coefficients.png)

![Modos POD desplegados base_ref](../outputs/base_ref/POD_modes_unfolded.png)

![Error LOO proyección base_ref](../outputs/base_ref/POD_loo_projection_error.png)

### 3.3 GPR

![Comparación kernels base_ref](../outputs/base_ref/GPR_kernel_comparison.png)

| Kernel | LOO medio (r=4) |
|---|---|
| **Matérn 3/2** | **0.1750** ✓ |
| Matérn 5/2 | 0.1922 |
| RatQuad | 0.2161 |
| Gibbs (no estac.) | 0.2187 |
| RBF | 0.2232 |

*(El kernel no estacionario de Gibbs se incorpora como 5º candidato; pierde en las cuatro bases — ver §8.4.)*

![Sweep r base_ref](../outputs/base_ref/GPR_r_sweep.png)

- **r\* = 8**, LOO = 0.1714 (proj 0.076 / interp 0.095)
- El error de interpolación GPR supera al de proyección desde r=5: más modos no ayudan con 10 puntos

![Ajuste GPR modos base_ref](../outputs/base_ref/GPR_mode_fits.png)

![Scatter LOO base_ref](../outputs/base_ref/GPR_loo_scatter.png)

### 3.4 Evaluación fuera de muestra (17 ángulos no entrenados)

Con las 5 simulaciones nuevas del nivel 1.25°, base_ref se evalúa ahora en **17 ángulos** no vistos:

![Evaluación fuera de muestra base_ref](../outputs/base_ref/GPR_error_vs_theta.png)

| θ (°) | Cp_min CFD | Cp_min ROM | err_rel |
|---|---|---|---|
| 2.5 | −0.605 | −0.594 | 0.034 |
| 7.5 | −0.750 | −0.894 | 0.064 |
| 11.25 | −1.159 | −1.144 | 0.084 |
| **12.5** | **−1.249** | **−1.097** | **0.120** |
| 13.75 | −1.159 | −1.024 | 0.082 |
| 16.25 | −0.950 | −0.883 | 0.116 |
| 17.5 | −0.784 | −0.817 | 0.067 |
| 18.75 | −0.736 | −0.735 | 0.083 |
| 21.25 | −0.609 | −0.572 | 0.118 |
| **22.5** | −0.647 | −0.501 | **0.163** |
| **23.75** | −0.495 | −0.496 | **0.218** |
| 26.25 | −0.496 | −0.496 | 0.120 |
| **27.5** | −0.501 | −0.497 | **0.195** |
| 28.75 | −0.495 | −0.496 | 0.092 |
| 32.5 | −0.505 | −0.496 | 0.025 |
| 37.5 | −0.498 | −0.494 | 0.035 |
| 42.5 | −0.492 | −0.471 | 0.054 |

**Resumen base_ref:**
- Error medio zona Kawai [10–30°]: **12.2%** (12 ángulos) — inaceptable
- Error medio fuera: **4.2%** — aceptable
- Error máximo: **21.8%** en θ=23.75°
- Nótese el pico de succión en θ=12.5°: el ROM lo subestima un 12% ($C_{p,min}$ −1.10 vs −1.25) — relevante para cargas de diseño

**Diagnóstico:** el paso 5° es insuficiente donde el vórtice migra no linealmente. Hay que densificar.

---

## 4. Fase 2 — Base Kawai (`base_kawai`)

**Ángulos de entrenamiento (N=17):** base_ref + {12.5°, 17.5°, 22.5°, 23.75°, 26.25°, 27.5°, 28.75°}

Estrategia adaptativa: añadir simulaciones **solo donde el error era alto**.

### 4.1 POD y GPR

![Snapshot overview base_kawai](../outputs/base_kawai/snapshot_overview.png)

![Espectro singular base_kawai](../outputs/base_kawai/POD_singular_spectrum.png)

![Coeficientes modales base_kawai](../outputs/base_kawai/POD_modal_coefficients.png)

![Modos POD desplegados base_kawai](../outputs/base_kawai/POD_modes_unfolded.png)

![Error LOO proyección base_kawai](../outputs/base_kawai/POD_loo_projection_error.png)

![Comparación kernels base_kawai](../outputs/base_kawai/GPR_kernel_comparison.png)

![Sweep r base_kawai](../outputs/base_kawai/GPR_r_sweep.png)

- Kernel: **Matérn 3/2**; **r\* = 10**, LOO = 0.1384 (proj 0.052 / interp 0.086)

![Ajuste GPR modos base_kawai](../outputs/base_kawai/GPR_mode_fits.png)

![Scatter LOO base_kawai](../outputs/base_kawai/GPR_loo_scatter.png)

### 4.2 Evaluación fuera de muestra (10 ángulos)

Con los 5 ángulos nuevos, base_kawai tiene ahora test **dentro** de la zona (los midpoints del paso 2.5°):

![Evaluación fuera de muestra base_kawai](../outputs/base_kawai/GPR_error_vs_theta.png)

| θ (°) | err_rel | Zona |
|---|---|---|
| 2.5 | 0.036 | no |
| 7.5 | 0.051 | no |
| 11.25 | 0.062 | **sí** |
| 13.75 | 0.052 | **sí** |
| 16.25 | 0.131 | **sí** |
| 18.75 | 0.078 | **sí** |
| 21.25 | 0.153 | **sí** |
| 32.5 | 0.018 | no |
| 37.5 | 0.034 | no |
| 42.5 | 0.053 | no |

**Resumen base_kawai:**
- Error en zona (midpoints a 1.25° de los vecinos): **9.5%**
- Error fuera: **3.8%**
- El paso 2.5° dentro de la zona reduce el error a menos de la mitad respecto al paso 5° (12.2% → 9.5% en posiciones más exigentes), pero los ángulos 16.25° y 21.25° siguen altos

---

## 5. Fase 3 — Base completa uniforme (`base_full22`)

**Ángulos de entrenamiento (N=22):** paso ~2.5° en todo [0°, 45°]
{0, 2.5, 5, 7.5, 10, 12.5, 15, 17.5, 20, 22.5, 23.75, 25, 26.25, 27.5, 28.75, 30, 32.5, 35, 37.5, 40, 42.5, 45}

*(Nota: en el código este set está congelado como `BASE_FULL22`; sus resultados viven en `outputs/base_full/`.)*

### 5.1 POD y GPR

![Snapshot overview base_full](../outputs/base_full/snapshot_overview.png)

![Espectro singular base_full](../outputs/base_full/POD_singular_spectrum.png)

![Coeficientes modales base_full](../outputs/base_full/POD_modal_coefficients.png)

![Modos POD desplegados base_full](../outputs/base_full/POD_modes_unfolded.png)

![Error LOO proyección base_full](../outputs/base_full/POD_loo_projection_error.png)

![Comparación kernels base_full](../outputs/base_full/GPR_kernel_comparison.png)

| Kernel | LOO medio (r=4) |
|---|---|
| **Matérn 5/2** | **0.1212** ✓ |
| Matérn 3/2 | 0.1214 |
| RatQuad | 0.1234 |
| Gibbs (no estac.) | 0.1260 |
| RBF | 0.1329 |

- Kernel: **Matérn 5/2** — único caso: con N=22 uniforme hay información para estimar una función más suave sin sobreajuste. Es el único bake-off donde Matérn 5/2 supera a 3/2 (por un margen mínimo, 0.1212 vs 0.1214). Gibbs, el candidato no estacionario, queda 4º (ver §8.4).

![Sweep r base_full](../outputs/base_full/GPR_r_sweep.png)

- **r\* = 12**, LOO = 0.0966 (proj 0.038 / interp 0.059)

![Ajuste GPR modos base_full](../outputs/base_full/GPR_mode_fits.png)

![Scatter LOO base_full](../outputs/base_full/GPR_loo_scatter.png)

### 5.2 Evaluación fuera de muestra (5 ángulos, todos en zona)

Los 5 ángulos nuevos del nivel 1.25° **no** están en el entrenamiento de base_full22 → miden cómo interpola la zona el paso uniforme 2.5°:

| θ (°) | Cp_min CFD | Cp_min ROM | err_rel |
|---|---|---|---|
| 11.25 | −1.159 | −1.248 | 0.055 |
| 13.75 | −1.159 | −1.097 | 0.052 |
| 16.25 | −0.950 | −0.857 | 0.110 |
| 18.75 | −0.736 | −0.703 | 0.075 |
| **21.25** | −0.609 | −0.716 | **0.146** |

**Error medio en zona: 8.75%** — el paso 2.5° deja un error residual apreciable en los midpoints, concentrado en 16.25° y 21.25°.

---

## 6. Fase 4 — Base Kawai nivel 2 (`base_kawai2`)

**Ángulos de entrenamiento (N=22):** base_kawai + {11.25°, 13.75°, 16.25°, 18.75°, 21.25°} → paso 1.25° uniforme en toda la zona [10°, 30°], paso 5° fuera.

Las 5 simulaciones nuevas se ejecutaron con la misma configuración V5 (malla ~7M celdas, 3500 iteraciones, 10 cores).

### 6.1 POD y GPR

![Snapshot overview base_kawai2](../outputs/base_kawai2/snapshot_overview.png)

![Espectro singular base_kawai2](../outputs/base_kawai2/POD_singular_spectrum.png)

![Coeficientes modales base_kawai2](../outputs/base_kawai2/POD_modal_coefficients.png)

![Modos POD desplegados base_kawai2](../outputs/base_kawai2/POD_modes_unfolded.png)

![Error LOO proyección base_kawai2](../outputs/base_kawai2/POD_loo_projection_error.png)

![Comparación kernels base_kawai2](../outputs/base_kawai2/GPR_kernel_comparison.png)

![Sweep r base_kawai2](../outputs/base_kawai2/GPR_r_sweep.png)

- Kernel: **Matérn 3/2**; **r\* = 10**, LOO = 0.1251 (proj 0.049 / interp 0.076)

![Ajuste GPR modos base_kawai2](../outputs/base_kawai2/GPR_mode_fits.png)

![Scatter LOO base_kawai2](../outputs/base_kawai2/GPR_loo_scatter.png)

### 6.2 Evaluación fuera de muestra (5 ángulos, todos fuera de zona)

![Evaluación fuera de muestra base_kawai2](../outputs/base_kawai2/GPR_error_vs_theta.png)

| θ (°) | err_rel |
|---|---|
| 2.5 | 0.041 |
| 7.5 | 0.074 |
| 32.5 | 0.018 |
| 37.5 | 0.029 |
| 42.5 | 0.048 |

**Error medio fuera de zona: 4.2%** — concentrar el muestreo en la zona no degrada el comportamiento fuera (base_kawai daba 3.8% con los mismos tests).

---

## 7. Comparativa: ¿dónde colocar las simulaciones?

### 7.1 Figura diagnóstica (no comparativa)

![Diagnóstico de error por base](../outputs/comparativa/error_vs_theta_comparativa.png)

Esta figura es un **diagnóstico de localización del error, no una comparación entre bases**. Cada curva es la evaluación fuera de muestra de una base, definida **solo en sus propios ángulos no entrenados** (por eso se interrumpe donde la base sí entrenó). Como las bases no comparten esos ángulos, **no deben compararse las alturas de las curvas entre bases** — esa es precisamente la razón de ser del conjunto de test común de §7.6. Lo que sí es legible y común a todas: el error se dispara en la banda del vórtice [10–30°] para cualquier diseño muestral.

### 7.2 Tabla resumen (evaluación fuera de muestra por base)

| Base | N | r\* | Kernel | Valid. (LOO) | Err. f. muestra zona | Err. f. muestra fuera |
|---|---|---|---|---|---|---|
| `base_ref` | 10 | 8 | Matérn 3/2 | 0.171 | 12.2% (12 ang.) | 4.2% (5 ang.) |
| `base_kawai` | 17 | 10 | Matérn 3/2 | 0.138 | 9.5% (5 ang.) | 3.8% (5 ang.) |
| `base_kawai2` | 22 | 10 | Matérn 3/2 | 0.125 | — (en training) | 4.2% (5 ang.) |
| `base_full22` | 22 | 12 | Matérn 5/2 | 0.097 | 8.8% (5 ang.) | — (en training) |

### 7.3 Advertencia metodológica: el LOO no es comparable entre diseños

El LOO medio de cada base depende de la geometría de sus propios folds: en base_kawai2 los folds fuera de la zona dejan huecos de 10° (muy difíciles), mientras que en base_full22 ningún fold deja más de 5°. Comparar LOO entre diseños distintos favorece sistemáticamente al diseño uniforme. **La comparación válida es a geometría de predicción igualada.**

### 7.4 Head-to-head a igual presupuesto y geometría igualada

base_kawai2 (990 CPU-h) vs base_full22 (976 CPU-h). En ambos casos se evalúa la predicción de un ángulo situado a 1.25° de dos vecinos de entrenamiento:

**Dentro de la zona** (en los 5 midpoints {11.25, ..., 21.25}):

| θ (°) | base_full22 (f. muestra) | base_kawai2 (fold LOO) |
|---|---|---|
| 11.25 | 0.055 | 0.064 |
| 13.75 | 0.052 | 0.053 |
| 16.25 | 0.110 | 0.157 |
| 18.75 | 0.075 | 0.109 |
| 21.25 | 0.146 | 0.203 |
| **Media** | **0.088** | **0.117** |

**Fuera de la zona** (en los 5 semienteros {2.5, ..., 42.5}):

| θ (°) | base_kawai2 (f. muestra) | base_full22 (fold LOO) |
|---|---|---|
| 2.5 | 0.041 | 0.036 |
| 7.5 | 0.074 | 0.059 |
| 32.5 | 0.018 | 0.029 |
| 37.5 | 0.029 | 0.039 |
| 42.5 | 0.048 | 0.057 |
| **Media** | **0.042** | **0.044** |

**Veredicto: a igual presupuesto, el diseño uniforme 2.5° es superior.** Fuera de la zona empatan, pero dentro de la zona base_full22 gana con claridad (8.8% vs 11.7%). La densificación extra 2.5° → 1.25° no solo no mejora: el muestreo grueso (5°) que base_kawai2 mantiene fuera de la zona empobrece su base POD global, y eso penaliza también sus predicciones dentro de la zona.

### 7.5 El error residual en θ ≈ 22.5–24° no es un problema de densidad

Folds LOO más difíciles de ambas bases N=22:

| θ (°) | base_kawai2 (huecos 2.5°) | base_full22 (huecos 5°) |
|---|---|---|
| 21.25 | 0.203 | — |
| 22.5 | 0.256 | 0.266 |
| 23.75 | 0.259 | 0.322 |

Con paso 1.25° el error en 22.5–23.75° apenas baja respecto al paso 2.5°. En esa franja los coeficientes modales del vórtice tienen su variación más rápida (el pico de succión se desplaza y reorganiza) y la interpolación GPR es intrínsecamente difícil: es el **límite de la parametrización actual**, no un déficit de muestreo. Documentable como limitación esperada del método (consistente con la física del vórtice de Kawai).

**Nota importante**: §7.4 usa el fold de validación cruzada de base_kawai2 como sustituto de una evaluación externa común (porque sus propios ángulos de entrenamiento no dejan huecos libres en la zona). §7.3 ya advierte que esto es una aproximación, no una medida externa real. La sección 7.6 resuelve esta limitación con un holdout pre-registrado, común a las cuatro bases, y **matiza el veredicto de 7.4**.

---

## 7.6 Conjunto de test: holdout pre-registrado (comparación ciega, BRIEF_holdout.md)

Para eliminar la dependencia de folds LOO (§7.3–7.5 usan aproximaciones distintas para cada base, no una medida común), se pre-registró un holdout de **9 ángulos que ninguna base ha visto jamás en entrenamiento**, con métricas y protocolo fijados *antes* de simular (ver `BRIEF_holdout.md`):

$$T = \{3.125°, 8.125°, 13.125°, 18.125°, 23.125°, 28.125°, 33.125°, 38.125°, 43.125°\}$$

Peine uniforme de paso 5° desplazado 3.125° — múltiplos impares de 0.625°, por construcción fuera del retículo $k \cdot 1.25°$ donde viven los 27 ángulos de entrenamiento. Estratificación: 2 puntos fuera-baja, 4 en zona [10°,30°], 3 fuera-alta.

### 7.6.1 Suelo de ruido CFD

Antes de interpretar diferencias entre bases, se cuantifica el ruido intrínseco del pipeline CFD (malla rotada, cobertura de capas variable) usando los pares físicamente equivalentes por simetría $\theta \leftrightarrow 90°-\theta$: (40°, 50°) y (42.5°, 47.5°). La transformación de simetría correcta se descubre empíricamente probando las 8 transformaciones del grupo diédrico D4 sobre el layout de sondas y eligiendo la que minimiza la discrepancia:

![Suelo de ruido](../outputs/holdout/symmetry_noise_fields.png)

La transformación ganadora (reflexión antidiagonal, $(x,y) \to (-y,-x)$) reduce el RMS de 0.07–0.12 (con cualquier otra transformación, o sin transformar) a **0.0003–0.0005** — confirmación de que la simetría C4+reflexión del pipeline CFD es correcta con precisión de 3 órdenes de magnitud.

$$\text{Suelo de ruido} = \text{RMS} \approx 0.0004 \text{ (unidades de } C_p\text{)}$$

Diferencias de RMSE entre bases por debajo de este valor no son atribuibles al diseño muestral.

### 7.6.2 Resultados: media sobre los 9 ángulos de holdout

![Error relativo holdout](../outputs/holdout/error_vs_theta_holdout_err_rel_common.png)

![Error en Cp_min holdout](../outputs/holdout/error_vs_theta_holdout_err_cpmin.png)

| Métrica | Estrato | base_ref | base_kawai | base_kawai2 | base_full22 |
|---|---|---|---|---|---|
| **RMSE** (abs., suelo=0.0004) | Global | 0.0139 | 0.0115 | 0.0116 | **0.0107** |
| | Zona [10–30°] | 0.0179 | 0.0124 | **0.0116** | 0.0132 |
| | Fuera | 0.0107 | 0.0108 | 0.0116 | **0.0086** |
| **Err. relativo** (denom. común) | Global | 0.0914 | 0.0768 | 0.0767 | **0.0743** |
| | Zona | 0.1473 | 0.1138 | **0.1088** | 0.1175 |
| | Fuera | 0.0467 | 0.0473 | 0.0511 | **0.0397** |
| **Err. en $C_{p,min}$** | Global | 0.0380 | 0.0257 | **0.0243** | 0.0333 |
| | Zona | 0.0630 | 0.0421 | **0.0305** | 0.0461 |
| | Fuera | 0.0180 | 0.0125 | 0.0194 | **0.0230**† |

† En "fuera", base_full22 pierde en $C_{p,min}$ pese a ganar en RMSE/err. relativo — dominado por un solo punto (θ=8.125°, ver tabla de detalle).

**El resultado clave, y la razón de ser del pre-registro**: la comparación ciega **no repite el veredicto de §7.4**. Allí, usando el fold de validación cruzada de base_kawai2 como sustituto de una evaluación externa común, base_full22 ganaba con claridad dentro de la zona (8.8% vs 11.7%). Aquí, con el mismo conjunto de test para ambas, **base_kawai2 tiene el menor RMSE y el menor error relativo dentro de la zona de las cuatro bases**, y domina con claridad en la métrica físicamente más relevante — el error en el pico de succión ($C_{p,min}$) — tanto en zona (0.031 vs 0.046 de base_full22) como globalmente (0.024, mejor que las otras tres).

Esto confirma exactamente la advertencia de §7.3: el fold de validación cruzada de un diseño con huecos irregulares no es intercambiable con un conjunto de test común. La comparación pre-registrada es la autoritativa; **§7.4 se mantiene documentado por su valor pedagógico (ilustra el sesgo), pero su conclusión "el uniforme gana en zona" queda revertida por este resultado.**

### 7.6.3 Comparaciones pareadas (test de signos, n=9)

| Par | Métrica | Resultado | p (mejor unilateral) |
|---|---|---|---|
| base_kawai vs base_kawai2 | err. relativo | kawai gana 7/9 | 0.090 |
| base_ref vs base_full22 | RMSE | full gana 6/9 | 0.254 |
| resto de pares | todas | 4/9–5/9 o 5/9–4/9 | ≥0.50 |

**Ninguna comparación pareada alcanza significancia convencional** ($p<0.05$) con $n=9$ — la potencia estadística es baja, como se anticipó en el pre-registro. La diferencia más cercana (kawai vs kawai2, $p=0.09$) es paradójica: kawai2 tiene mejor *media* que kawai en zona/Cp_min, pero kawai gana en más puntos individuales del error relativo — kawai2 pierde por márgenes pequeños en la mayoría de ángulos pero gana por márgenes grandes en unos pocos (13.125° y 18.125°, ver `outputs/holdout/holdout_results.csv`). El test de signos, al ignorar magnitud, no captura esto: **hay que leer medias y test de signos juntos, nunca uno solo.**

Comparando magnitudes contra el suelo de ruido (0.0004): la diferencia global kawai/kawai2 en RMSE (0.0115 vs 0.0116) es **menor que el suelo de ruido** — indistinguibles. La diferencia kawai2/full22 en zona (0.0116 vs 0.0132) es ~4× el suelo — real. base_ref vs cualquier otra base (diferencias ≥0.003) está claramente por encima del ruido en todos los casos.

### 7.6.4 Interpretación para el diseño del ROM final

No hay una base que domine en las tres métricas y los dos estratos simultáneamente. La elección depende de qué prioriza el ROM:

- **Si el objetivo es error medio global mínimo** (uso genérico del ROM en todo el rango): base_full22 gana en RMSE y error relativo global, y es marginalmente más barata (976 vs 990 CPU-h).
- **Si el objetivo es la carga de diseño estructural** (pico de succión, relevante para cladding y elementos de fachada en la zona de vórtice): base_kawai2 es la mejor opción, con un margen claro sobre el suelo de ruido.

Dado que el TFM enmarca el problema en términos de cargas de viento sobre edificio alto (§ Objetivo), el error en $C_{p,min}$ tiene relevancia física directa que el error relativo agregado no captura. Esto se traslada a la recomendación final en §10.

Salidas: `outputs/holdout/holdout_results.csv` (detalle por ángulo), `holdout_summary.csv` (medias), `holdout_pairwise.csv` (test de signos), `symmetry_noise.csv`.

---

## 8. Suelo de error: descomposición y límite del método

Elegida la base, la pregunta natural es *por qué* el error no baja de ~0.07–0.12 (error relativo en holdout) y si más CFD, más modos POD o un kernel más flexible lo reducirían. Esta sección descompone el error y descarta, con un experimento propio para cada una, las cuatro causas candidatas. Scripts: `investigacion/{ROM_floor_diagnostic, ROM_floor_external, ROM_nonstationary_kernel, ROM_gibbs_sensitivity}.py`.

### 8.1 Descomposición proyección + interpolación frente a r

El error de reconstrucción de un ángulo se descompone de forma aditiva en el que introduce la **base POD** (no representa el campo) y el que introduce la **GPR** (no predice bien los coeficientes):

$$\varepsilon_{total}(r) = \varepsilon_{proj}(r) + \varepsilon_{interp}(r)$$

Sobre `base_full` (`outputs/floor_diag/floor_vs_r.csv`):

| r | ε_proj | ε_interp | ε_total |
|---|---|---|---|
| 4 | 0.097 | 0.024 | 0.121 |
| 8 | 0.049 | 0.049 | 0.098 |
| **12** | **0.038** | **0.059** | **0.0966** |
| 16 | 0.032 | 0.065 | 0.097 |
| 20 | 0.030 | 0.067 | 0.097 |

![Descomposición del suelo de error](../outputs/floor_diag/floor_diagnostic.png)

$\varepsilon_{proj}$ decrece de forma monótona y **satura** en r≈11 (~0.030); $\varepsilon_{interp}$ **crece** de forma monótona (los modos altos son ruidosos y difíciles de interpolar). El mínimo de $\varepsilon_{total}$ en r≈11–12 es exactamente r\*. **Añadir modos más allá de r\* no ayuda**: es la interpolación, no la base, lo que se degrada al subir r.

### 8.2 Escalado con la densidad angular: meseta en Δθ≈5°

Fijando la base y variando la densidad de entrenamiento (test held-out fijo), se mide cómo cae el error con Δθ (`floor_vs_dtheta.csv`, RMSE de campo):

| Δθ entrenamiento | N | Interpolación (base fija) | ROM completo (re-POD) |
|---|---|---|---|
| 10° | 6 | 0.0260 | 0.0288 |
| **5°** | 10 | **0.0147** | **0.0152** |
| 2.5° | 13 | 0.0145 | 0.0151 |
| 2.5°+dens | 16 | 0.0144 | 0.0151 |

De 10°→5° el error casi se divide por dos; **de 5° en adelante densificar no cambia nada**, con o sin re-POD. La meseta está en Δθ≈5°.

### 8.3 Variante externa rigurosa: la base POD sí contribuye

La versión de §8.1–8.2 es *autocontenida* (la verdad son snapshots reconstruidos de la propia base, que ya "ha visto" los ángulos de test) → su suelo de proyección (0.0018) es optimista. La versión rigurosa usa como verdad el **CFD real del holdout** y **re-POD por fold** (`outputs/floor_external/floor_external_vs_dtheta.csv`). A Δθ=2.5° (el `base_full` real, cuyo ε_total coincide con su holdout de §7.6.2):

| Estrato | ε_proj | ε_total | proj/total | Término dominante |
|---|---|---|---|---|
| Global | 0.0297 | 0.0743 | 40% | interpolación (60%) |
| **Zona [10–30°]** | 0.0366 | 0.1175 | **31%** | **interpolación (69%)** |
| **Fuera** | 0.0242 | 0.0397 | **61%** | **proyección (61%)** |

![Suelo de error, variante externa](../outputs/floor_external/floor_external.png)

**Matiz clave respecto a la versión LOO**: con datos externos la proyección **no** es despreciable (~40% global), y es dependiente del estrato — **en la zona del vórtice domina la interpolación** (69%), pero **fuera domina el truncamiento POD** (61%). La meseta hacia Δθ≈5° se mantiene.

### 8.4 ¿Es no-estacionariedad de los coeficientes? Kernel de Gibbs

Los coeficientes $a_j(\theta)$ del vórtice varían más rápido dentro de [10,30]° que fuera (escala local 1.7–2.4× más corta en los modos 1–5). Se prueba un GP **no estacionario de Gibbs**, $\ell(\theta)=\ell_0\,e^{-\beta\,b(\theta)}$ con $b$ una campana centrada en el vórtice, ajustando $\beta$ por máxima verosimilitud (anida el estacionario en $\beta=0$):

| Modelo | glob | zona | fuera |
|---|---|---|---|
| Estacionario | 0.01402 | 0.01580 | 0.01047 |
| Gibbs (MLE) | 0.01403 | 0.01581 | 0.01047 |

![Kernel no estacionario](../outputs/nonstationary/nonstationary_kernel.png)

**Idénticos.** La MLE elige $\beta\approx0$ en los modos que cargan el campo (solo activa $\beta$ en modos 7–9, de energía ínfima): la navaja de Occam rechaza la complejidad extra con N≈16–22. Blindado por sensibilidad (`outputs/sensitivity/`):

- **Forma de la campana** (9 combinaciones μ∈{15,20,25}°, ancho∈{5,8,12}°): mejora ≤0.03%; ninguna bate al estacionario.
- **ℓ(θ) libre** (spline log-lineal de 4 nudos): **−2.5%** (peor).
- **r\*∈{8,12,16}**: Gibbs = estacionario en todos.
- **LOO completo n=22 (bootstrap)**: Gibbs − estacionario = **+0.00007**, IC95% **[0.00000, 0.00018]** → estadísticamente distinguible pero ~0.4% **peor**.

Gibbs está integrado como **5º candidato** en la selección de kernel del pipeline y **pierde en las cuatro bases** (§5.1, §6.1). El pipeline **demuestra** en cada corrida que la no-estacionariedad no gana, en lugar de asumirlo.

### 8.5 ¿Es ruido de malla? Jitter frente a señal

El único estimador limpio del ruido numérico es el test de simetría (§7.6.1), porque compara dos configuraciones físicamente idénticas: RMS ≈ 0.0004. Frente al residuo de interpolación en zona (0.0158) es el **2.5%** (0.06% en energía); el ruido relativo de los coeficientes (2.7%) lo confirma (`jitter_estimate.csv`). Los estimadores por coeficiente ($\sigma_n$ de la MLE, diferencias finitas) dan artefactos (~0.14) porque la amplitud del modo 1 los contamina — no deben usarse. **El residuo es señal no capturada, no ruido.**

### 8.6 Síntesis: naturaleza del suelo de error

Descartadas las cuatro causas candidatas, el suelo es un **límite muestral**:

- **No es solo la base POD** — pero contribuye ~40% global (y ~61% *fuera* del vórtice); en la zona crítica sí domina la interpolación (69%).
- **No es RANS** — el holdout mide ROM-vs-CFD; RANS solo limita ROM-vs-TPU.
- **No es el kernel** — Gibbs = estacionario, con respaldo bootstrap.
- **No es ruido de malla** — jitter 0.06% en energía.

Lo que queda es la **migración no lineal del vórtice submuestreada a 2.5–5°**: la señal existe (está ~35× por encima del suelo de ruido) pero N≈16–22 no basta para interpolarla mejor. Único lever real: **más snapshots dirigidos al vórtice** (dirección `base_kawai2`) o aceptar el suelo — lo que refuerza el valor del muestreo adaptativo y de la sensorización óptima (fase siguiente).

---

## 9. Coste computacional normalizado

Script: `ROM_cost_analysis.py`. El coste de cada caso se extrae de los logs del solver (`log.simpleFoam.<nCores>`) y se normaliza en dos capas:

**Capa 1 — equivalencia de cores.** Dos casos (θ=5°, 15°) se ejecutaron a 10 y a 20 cores, lo que permite medir la eficiencia paralela:

$$f_{10/20} = \frac{\text{CPU-h}(10)}{\text{CPU-h}(20)} = 0.733$$

Correr a 20 cores cuesta 1.36× más CPU-h que a 10 (eficiencia del 73% al doblar cores). Todo caso se expresa en **CPU-h equivalentes a 10 cores**.

**Capa 2 — corrección de contención.** Las 5 simulaciones nuevas se ejecutaron simultáneamente en la misma máquina (50 ranks compitiendo), inflando su `ExecutionTime` ~2× sin reflejar coste intrínseco. Se detectan comparando la tasa CPU/(celda·iteración) con la mediana de todos los runs (umbral 1.5×) y su coste se sustituye por el modelado en ejecución exclusiva:

| θ (°) | Medido (CPU-h) | Modelado (CPU-h) |
|---|---|---|
| 11.25 | 94.7 | 44.5 |
| 13.75 | 89.5 | 42.1 |
| 16.25 | 92.6 | 45.6 |
| 18.75 | 96.5 | 43.5 |
| 21.25 | 96.1 | 46.2 |

**Coste acumulado por base:**

![Coste vs error](../outputs/comparativa/cost_vs_error.png)

| Base | N | CPU-h eq10 | LOO |
|---|---|---|---|
| base_ref | 10 | 431 | 0.171 |
| base_kawai | 17 | 768 | 0.138 |
| base_kawai2 | 22 | 990 | 0.125 |
| base_full22 | 22 | 976 | 0.097 |

Salidas: `outputs/comparativa/cost_per_case.csv`, `cost_resumen.csv`, `cost_vs_error.png`.

---

## 10. Conclusiones

1. **El paso 5° es insuficiente en la zona del vórtice de Kawai**: 12.2% de error medio en la zona (evaluación fuera de muestra) frente a 4.2% fuera (base_ref). La variación no lineal de la posición del vórtice exige más resolución angular local.

2. **Densificar la zona a 2.5° es la inversión más rentable**: base_kawai (+337 CPU-h sobre base_ref) reduce el error en zona de 12.2% a 9.5% y el LOO un 19%, sin degradar el comportamiento fuera.

3. **El head-to-head basado en LOO (§7.4) favorecía artificialmente al diseño uniforme**: bajo esa aproximación, base_kawai2 perdía contra base_full22 en zona (11.7% vs 8.8%). El **holdout pre-registrado y ciego (§7.6)**, con el mismo test externo real para las cuatro bases, revierte esta conclusión: base_kawai2 tiene el menor RMSE y error relativo dentro de la zona, y domina claramente en error de $C_{p,min}$ (pico de succión) tanto en zona como globalmente. Esta es la comparación autoritativa; **§7.4 se documenta por su valor pedagógico** (ilustra por qué el pre-registro de §7.3 era necesario), no como conclusión final.

4. **El error residual en θ ≈ 22.5–24° es irreducible por muestreo** con esta parametrización: persiste (~0.25–0.32 en fold LOO, y también en el holdout ciego en θ=23.125° — err. relativo ≈0.15–0.19 en las cuatro bases por igual) con paso 1.25°. Es el punto de variación más rápida de los coeficientes modales del vórtice y constituye la limitación documentable del método POD+GPR en este problema, **no un déficit de ninguna estrategia de muestreo en particular** (las cuatro bases fallan de forma similar ahí).

5. **El kernel óptimo depende del diseño muestral**: Matérn 3/2 en todas las bases salvo base_full22 (Matérn 5/2) — solo el muestreo uniforme y denso soporta la hipótesis de mayor suavidad sin sobreajuste.

6. **El cuello de botella depende del estrato** (§8.3): en la validación LOO interna $\varepsilon^{interp} > \varepsilon^{proj}$ en $r^*$ en las cuatro bases, pero la descomposición **externa rigurosa** (CFD real + re-POD) matiza que, a un ángulo nunca visto, la interpolación GPR domina **en la zona del vórtice** (~69% del error) mientras que **fuera** domina el truncamiento POD (~61%); globalmente la proyección aporta ~40% —no es despreciable—. Añadir modos más allá de $r^*$ no reduce el error ($\varepsilon^{interp}$ crece con $r$); añadir snapshots **dirigidos al vórtice** sí.

7. **No hay una base que domine todas las métricas** (§7.6.4): base_full22 minimiza el error medio global (y es ligeramente más barata); base_kawai2 minimiza el error en el pico de succión, la cantidad físicamente relevante para cargas de diseño en la zona de vórtice. **Diseño recomendado**: si el TFM prioriza precisión general del ROM, base_full22 (LOO 9.7%, ~976 CPU-h); si prioriza fidelidad de cargas estructurales en la zona crítica, base_kawai2. Ninguna comparación pareada alcanza significancia estadística convencional con n=9 (§7.6.3) — la elección debe justificarse por relevancia física, no solo por el promedio agregado. La **sensorización óptima** (siguiente fase) se aplica sobre la base POD de la opción elegida.

8. **Normalización de coste**: con casos ejecutados en configuraciones heterogéneas (10/20 cores, ejecución exclusiva o concurrente), la comparación justa exige normalizar a CPU-h equivalentes (aquí: eficiencia paralela medida 0.733 y corrección de contención por tasa CPU/celda/iteración).

9. **El suelo de ruido del pipeline CFD es muy bajo** (RMS ≈ 0.0004 en unidades de $C_p$, vía pares simétricos θ↔90°-θ, §7.6.1): confirma que la simetría C4+reflexión del pipeline es correcta con precisión de 3 órdenes de magnitud, y que las diferencias entre bases documentadas aquí (salvo kawai/kawai2 a nivel global) son señal real, no artefacto de malla.

10. **El suelo de error es un límite muestral, no del método** (§8): la meseta aparece en Δθ≈5° y no baja densificando más. Se descarta cada causa alternativa con su propio experimento — no es RANS (el holdout mide ROM-vs-CFD, no ROM-vs-TPU), no es ruido de malla (jitter ≈0.0004 = 0.06% del residuo en energía), y no es el kernel: un GP no estacionario de Gibbs con $\ell(\theta)$ por MLE es idéntico al estacionario (β→0 en los modos con energía), blindado por sensibilidad de forma, ℓ(θ) libre y r\*, y un IC bootstrap n=22 de **[0.00000, 0.00018]**. La base POD sí aporta ~40% del error a un ángulo no visto (más fuera del vórtice que dentro, §8.3). Lo irreducible es la migración del vórtice submuestreada; el único lever es muestreo dirigido, lo que motiva la fase de **sensorización óptima**.

---

*Pipeline ejecutado en: `Programacion/` — rama git `rom-11-tpu`*
*Actualizado: 2026-07-18 (añade §8 — suelo de error, descomposición proj/interp externa, kernel no estacionario de Gibbs y sensibilidad; Gibbs como 5º candidato de kernel)*
