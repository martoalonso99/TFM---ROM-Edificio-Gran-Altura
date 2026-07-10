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
- Selección de kernel y de $r^*$ por **LOO-CV con re-POD por fold** (la base se recalcula desde cero en cada fold para evitar contaminación)
- Test externo: predicción en ángulos disponibles no usados en el entrenamiento
- Métrica: $\varepsilon = \|C_{p,CFD} - C_{p,ROM}\|_2 / \|C_{p,CFD} - \bar{C_p}^{train}\|_2$

---

## 3. Fase 1 — Base de referencia (`base_ref`)

**Ángulos de entrenamiento (N=10):** 0°, 5°, 10°, 15°, 20°, 25°, 30°, 35°, 40°, 45°

### 3.1 Visión general de los snapshots

![Snapshot overview base_ref](outputs/base_ref/snapshot_overview.png)

Observaciones:
- A 0°: cara barlovento con $C_p \approx +0.97$, sotavento con succión $C_p \approx -0.5$
- A ángulos oblicuos (~10–30°): succión intensa en la arista barlovento — **vórtice cónico de Kawai** (pico de succión $C_p \approx -1.25$ en θ=12.5°)
- La distribución varía de forma no lineal con $\theta$, especialmente en la zona del vórtice

### 3.2 POD

![Espectro singular base_ref](outputs/base_ref/POD_singular_spectrum.png)

- Los primeros 4 modos capturan ≥99.5% de la energía
- Modos 1–2: patrón global estancamiento + reflujo; modos 3–6: firma del vórtice cónico

![Coeficientes modales base_ref](outputs/base_ref/POD_modal_coefficients.png)

![Modos POD desplegados base_ref](outputs/base_ref/POD_modes_unfolded.png)

![Error LOO proyección base_ref](outputs/base_ref/POD_loo_projection_error.png)

### 3.3 GPR

![Comparación kernels base_ref](outputs/base_ref/GPR_kernel_comparison.png)

| Kernel | LOO medio (r=4) |
|---|---|
| **Matérn 3/2** | **0.1750** ✓ |
| Matérn 5/2 | 0.1922 |
| RatQuad | 0.2161 |
| RBF | 0.2232 |

![Sweep r base_ref](outputs/base_ref/GPR_r_sweep.png)

- **r\* = 8**, LOO = 0.1714 (proj 0.076 / interp 0.095)
- El error de interpolación GPR supera al de proyección desde r=5: más modos no ayudan con 10 puntos

![Ajuste GPR modos base_ref](outputs/base_ref/GPR_mode_fits.png)

![Scatter LOO base_ref](outputs/base_ref/GPR_loo_scatter.png)

### 3.4 Test externo (17 ángulos intermedios)

Con las 5 simulaciones nuevas del nivel 1.25°, base_ref se evalúa ahora en **17 ángulos** no vistos:

![Error test externo base_ref](outputs/base_ref/GPR_error_vs_theta.png)

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

![Snapshot overview base_kawai](outputs/base_kawai/snapshot_overview.png)

![Espectro singular base_kawai](outputs/base_kawai/POD_singular_spectrum.png)

![Coeficientes modales base_kawai](outputs/base_kawai/POD_modal_coefficients.png)

![Modos POD desplegados base_kawai](outputs/base_kawai/POD_modes_unfolded.png)

![Error LOO proyección base_kawai](outputs/base_kawai/POD_loo_projection_error.png)

![Comparación kernels base_kawai](outputs/base_kawai/GPR_kernel_comparison.png)

![Sweep r base_kawai](outputs/base_kawai/GPR_r_sweep.png)

- Kernel: **Matérn 3/2**; **r\* = 10**, LOO = 0.1384 (proj 0.052 / interp 0.086)

![Ajuste GPR modos base_kawai](outputs/base_kawai/GPR_mode_fits.png)

![Scatter LOO base_kawai](outputs/base_kawai/GPR_loo_scatter.png)

### 4.2 Test externo (10 ángulos)

Con los 5 ángulos nuevos, base_kawai tiene ahora test **dentro** de la zona (los midpoints del paso 2.5°):

![Error test externo base_kawai](outputs/base_kawai/GPR_error_vs_theta.png)

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

![Snapshot overview base_full](outputs/base_full/snapshot_overview.png)

![Espectro singular base_full](outputs/base_full/POD_singular_spectrum.png)

![Coeficientes modales base_full](outputs/base_full/POD_modal_coefficients.png)

![Modos POD desplegados base_full](outputs/base_full/POD_modes_unfolded.png)

![Error LOO proyección base_full](outputs/base_full/POD_loo_projection_error.png)

![Comparación kernels base_full](outputs/base_full/GPR_kernel_comparison.png)

- Kernel: **Matérn 5/2** — único caso: con N=22 uniforme hay información para estimar una función más suave sin sobreajuste

![Sweep r base_full](outputs/base_full/GPR_r_sweep.png)

- **r\* = 12**, LOO = 0.0966 (proj 0.038 / interp 0.059)

![Ajuste GPR modos base_full](outputs/base_full/GPR_mode_fits.png)

![Scatter LOO base_full](outputs/base_full/GPR_loo_scatter.png)

### 5.2 Test externo (5 ángulos, todos en zona)

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

![Snapshot overview base_kawai2](outputs/base_kawai2/snapshot_overview.png)

![Espectro singular base_kawai2](outputs/base_kawai2/POD_singular_spectrum.png)

![Coeficientes modales base_kawai2](outputs/base_kawai2/POD_modal_coefficients.png)

![Modos POD desplegados base_kawai2](outputs/base_kawai2/POD_modes_unfolded.png)

![Error LOO proyección base_kawai2](outputs/base_kawai2/POD_loo_projection_error.png)

![Comparación kernels base_kawai2](outputs/base_kawai2/GPR_kernel_comparison.png)

![Sweep r base_kawai2](outputs/base_kawai2/GPR_r_sweep.png)

- Kernel: **Matérn 3/2**; **r\* = 10**, LOO = 0.1251 (proj 0.049 / interp 0.076)

![Ajuste GPR modos base_kawai2](outputs/base_kawai2/GPR_mode_fits.png)

![Scatter LOO base_kawai2](outputs/base_kawai2/GPR_loo_scatter.png)

### 6.2 Test externo (5 ángulos, todos fuera de zona)

![Error test externo base_kawai2](outputs/base_kawai2/GPR_error_vs_theta.png)

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

### 7.1 Figura principal

![Comparativa 4 bases](outputs/comparativa/error_vs_theta_comparativa.png)

Las curvas se interrumpen donde una base no tiene ángulos de test (están en su entrenamiento).

### 7.2 Tabla resumen (test externo)

| Base | N | r\* | Kernel | LOO | Err. test zona | Err. test fuera |
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

| θ (°) | base_full22 (test ext.) | base_kawai2 (fold LOO) |
|---|---|---|
| 11.25 | 0.055 | 0.064 |
| 13.75 | 0.052 | 0.053 |
| 16.25 | 0.110 | 0.157 |
| 18.75 | 0.075 | 0.109 |
| 21.25 | 0.146 | 0.203 |
| **Media** | **0.088** | **0.117** |

**Fuera de la zona** (en los 5 semienteros {2.5, ..., 42.5}):

| θ (°) | base_kawai2 (test ext.) | base_full22 (fold LOO) |
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

---

## 8. Coste computacional normalizado

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

![Coste vs error](outputs/comparativa/cost_vs_error.png)

| Base | N | CPU-h eq10 | LOO |
|---|---|---|---|
| base_ref | 10 | 431 | 0.171 |
| base_kawai | 17 | 768 | 0.138 |
| base_kawai2 | 22 | 990 | 0.125 |
| base_full22 | 22 | 976 | 0.097 |

Salidas: `outputs/comparativa/cost_per_case.csv`, `cost_resumen.csv`, `cost_vs_error.png`.

---

## 9. Conclusiones

1. **El paso 5° es insuficiente en la zona del vórtice de Kawai**: 12.2% de error medio de test frente a 4.2% fuera (base_ref). La variación no lineal de la posición del vórtice exige más resolución angular local.

2. **Densificar la zona a 2.5° es la inversión más rentable**: base_kawai (+337 CPU-h sobre base_ref) reduce el error en zona de 12.2% a 9.5% y el LOO un 19%, sin degradar el comportamiento fuera.

3. **Densificar más allá de 2.5° no paga**: base_kawai2 (paso 1.25° en zona) pierde contra base_full22 a igual presupuesto tanto en LOO comparable como en el head-to-head a geometría igualada (11.7% vs 8.8% en zona). La cobertura uniforme 2.5° fuera de la zona enriquece la base POD global más de lo que aporta la densidad extra dentro.

4. **El error residual en θ ≈ 22.5–24° es irreducible por muestreo** con esta parametrización: persiste (~0.25–0.32 en fold LOO) con paso 1.25°. Es el punto de variación más rápida de los coeficientes modales del vórtice y constituye la limitación documentable del método POD+GPR en este problema.

5. **El kernel óptimo depende del diseño muestral**: Matérn 3/2 en todas las bases salvo base_full22 (Matérn 5/2) — solo el muestreo uniforme y denso soporta la hipótesis de mayor suavidad sin sobreajuste.

6. **El cuello de botella es siempre la interpolación GPR**, no el truncamiento POD: en las cuatro bases $\varepsilon^{interp} > \varepsilon^{proj}$ en $r^*$. Añadir modos más allá de $r^*$ no reduce el error; añadir snapshots sí.

7. **Diseño recomendado para el ROM final**: muestreo uniforme a paso 2.5° (base_full22, LOO 9.7%, ~976 CPU-h), con la advertencia de precisión reducida en 16–24°. La **sensorización óptima** (siguiente fase) se aplica sobre su base POD ($r^*=12$, $\Phi \in \mathbb{R}^{400\times12}$).

8. **Normalización de coste**: con casos ejecutados en configuraciones heterogéneas (10/20 cores, ejecución exclusiva o concurrente), la comparación justa exige normalizar a CPU-h equivalentes (aquí: eficiencia paralela medida 0.733 y corrección de contención por tasa CPU/celda/iteración).

---

*Pipeline ejecutado en: `Programacion/` — rama git `rom-11-tpu`*
*Actualizado: 2026-07-10 (incluye base_kawai2 y análisis de coste)*
