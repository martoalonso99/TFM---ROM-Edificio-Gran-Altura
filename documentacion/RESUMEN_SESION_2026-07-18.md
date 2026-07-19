# Resumen de trabajo — hasta 2026-07-18

> Documento de handoff para retomar en Claude Cowork y redactar el capítulo de resultados del ROM.
> Los números detallados viven en `documentacion/RESULTADOS_ROM.md` y en los CSV bajo `outputs/`.
> Rama git: `rom-11-tpu`. Último commit de código: `4737c30`.

---

## 0. TL;DR — qué cambió y por qué importa para el capítulo

1. **Se añadió un nivel de densidad de muestreo** (`base_kawai2`, paso 1.25° en la zona del vórtice) → 4 bases comparables a igual coste.
2. **Se pre-registró un holdout ciego de 9 ángulos** y su evaluación **revierte el veredicto** de la comparación por LOO: en la zona del vórtice y en el pico de succión gana el muestreo adaptativo (`base_kawai2`), no el uniforme.
3. **Se diagnosticó el "suelo de error"**: es un **límite muestral** (migración del vórtice submuestreada), NO de la base POD, NO de RANS, NO del kernel, NO de ruido numérico. Con respaldo estadístico (IC bootstrap).
4. **Se probó y descartó un kernel no estacionario (Gibbs)** con máxima verosimilitud, ahora **integrado como 5º candidato** en la selección de kernel del pipeline (se verifica en cada corrida, no se asume).
5. **Reestructuración del código a formato productivo**: `main.py` como entrada única, núcleo/evaluación/investigación separados, duplicación eliminada.

Ninguna de las conclusiones científicas previas se rompió: el recook completo (2.5 h) reprodujo **byte a byte** las selecciones de kernel y los LOO de las 4 bases.

---

## 1. Reestructuración de carpetas

### 1.1 Raíz del TFM (`TFM/`)
Antes: decenas de ficheros sueltos. Ahora:

```
TFM/
├── CLAUDE.md, main.tex/.aux/.log/.out, referencias.bib, Images/   ← memoria activa
├── notas/          ← handoff.md, FAQ_ROM.md, GPML_Expert_Analysis.md, plan_actuacion_main.md, gpml_ch*.txt
├── gestion/        ← Plan Proyecto TFM/ARUP, Simulation Status.xlsx, CaseA/CaseB.xls, Gmail status
├── presentacion/   ← slides_ROM.tex (beamer defensa)
├── legacy/         ← versiones antiguas de la memoria (TFM_ROM_Edificio.tex, TFM_memoria_corregido.tex, .docx, output/)
├── cheatsheets/    ← CFD/turbulencia/GPML/Overleaf
├── Papers/, Simulaciones/, OpenFOAM Manual/, Base Geometry and Results/
└── Programacion/   ← pipeline (ver 1.2)
```
`main.tex` solo depende de `Images/`; verificado que ningún `.tex`/`.bib` referenciaba los ficheros movidos. `CLAUDE.md` actualizado (rutas `notas/handoff.md`, mapa de ficheros).

### 1.2 Pipeline (`Programacion/`) — estructura final
```
Programacion/
├── main.py                    ← ENTRADA ÚNICA (CLI orquestador)
├── ROM_POD.py                 ← núcleo: lee CFD, POD (SVD centrada)
├── ROM_GPR.py                 ← núcleo: GPR por modo, LOO-CV re-POD, selección kernel (5 candidatos)
├── ROM_gibbs_kernel.py        ← núcleo: GP de Gibbs (no estacionario) compartido
├── ROM_comparativa.py         ← evaluación: cruza bases
├── ROM_cost_analysis.py       ← evaluación: coste CPU-h normalizado
├── ROM_holdout_eval.py        ← evaluación: holdout pre-registrado
├── ROM_test_intermediate.py   ← evaluación: test en ángulos no vistos
├── investigacion/             ← 6 scripts de investigación (llaman al núcleo)
│   ├── ROM_floor_diagnostic.py / ROM_floor_external.py
│   ├── ROM_nonstationary_kernel.py / ROM_nonstationary_external.py
│   ├── ROM_gibbs_sensitivity.py
│   └── ROM_symmetry_noise.py
├── outputs/                   ← resultados (npz/png gitignored; CSV versionados)
├── legacy/                    ← backups + outputs_2026-07-18/ (snapshot pre-Gibbs, gitignored)
├── documentacion/             ← RESULTADOS_ROM.md, BRIEF_*.md, este archivo, diagrama pipeline
├── utils/                     ← generate_new_cases.py, copy_stls.py, export_probes_csv.py
└── memoria/                   ← cap4_secciones_44_48.tex
```

Se **descartó** llevar los `ROM_*.py` a un paquete `rom/` anidado: el valor productivo (entrada única, sin duplicación, investigación separada) ya estaba capturado y no compensaba el riesgo de re-pathing.

---

## 2. Cambios en el código

### 2.1 `main.py` (nuevo) — orquestador CLI
Subcomandos que delegan en cada módulo vía `main(argv)` (sin duplicar lógica):
```
python main.py run       --angles <set> --outdir <dir>   # pod + gpr + evaluate
python main.py pod       --angles <set> --outdir <dir>
python main.py gpr       --outdir <dir>
python main.py evaluate  --outdir <dir>
python main.py compare | cost | holdout
python main.py investigate                                # lista los scripts de investigacion/
```
Sets de ángulos: `base_ref, base_kawai, base_kawai2, base_full22, base_full, tpu11, all24`.
**Ojo**: `base_full` es *dinámico* (todos los ≤45° disponibles = ahora 27); para reproducir el base_full documentado (N=22) usar `base_full22`.

### 2.2 Kernel de Gibbs integrado en la selección (nuevo `ROM_gibbs_kernel.py`)
- Maquinaria del GP no estacionario de Gibbs (longitud de escala variable `ℓ(θ)=ℓ₀·exp(−β·campana_vórtice)`, anida el estacionario con β=0), implementada a mano (numpy+scipy) porque sklearn no soporta `length_scale` no estacionario.
- Clase `GibbsModeGP` con interfaz estilo sklearn (`.predict(X, return_std=)`), centra `y` (normalize_y) para comparación justa.
- `ROM_GPR.KERNEL_NAMES` pasa de 4 a **5 candidatos**: `[Matern52, Matern32, RBF, RatQuad, Gibbs]`.
- Factory `fit_mode_gp(kernel_name, ...)` despacha entre sklearn (4 estacionarios, **código byte-idéntico al previo**) y Gibbs. Se usa en `loo_cv_full` y `fit_full_model`.

### 2.3 Duplicación eliminada
- `retrain_gprs` triplicado (comparativa/holdout_eval/test_intermediate) → ahora todos llaman a `fit_mode_gp`. También quedan Gibbs-safe.
- Maquinaria de Gibbs centralizada en `ROM_gibbs_kernel.py`; `ROM_nonstationary_kernel.py` la re-exporta (compat. con sus dependientes).

### 2.4 Otros
- `main(argv=None)` en `ROM_POD`, `ROM_GPR`, `ROM_test_intermediate` (para orquestación + retrocompatibilidad standalone).
- Scripts de `investigacion/` con shim `sys.path` + rutas basadas en `ROOT` (funcionan desde la subcarpeta).
- `HOLDOUT_SET` congelado en `ROM_POD.py`; `discover_angles()` lo **excluye por defecto** → ninguna base puede absorber el holdout como entrenamiento.
- `ROM_cost_analysis.py`: normalización de coste en 2 capas (eficiencia paralela `f_10/20=0.733` medida en casos duplicados + corrección de contención por tasa CPU/celda/iter, umbral 1.5× mediana).

---

## 3. Experimentos nuevos (contenido para el capítulo)

### 3.1 Cuarta base — `base_kawai2` (nivel de densidad 1.25°)
5 simulaciones CFD nuevas en {11.25, 13.75, 16.25, 18.75, 21.25}° → paso 1.25° uniforme en toda la zona [10,30]°. Da un par comparable a **igual N e igual presupuesto**:

| Base | N | Estrategia | r\* | Kernel | LOO | CPU-h eq10 |
|---|---|---|---|---|---|---|
| base_ref | 10 | uniforme 5° | 8 | Matérn 3/2 | 0.1714 | 431 |
| base_kawai | 17 | ref + densif. zona | 10 | Matérn 3/2 | 0.1384 | 768 |
| base_kawai2 | 22 | ref + paso 1.25° en zona | 10 | Matérn 3/2 | 0.1251 | ~1004 |
| base_full22 | 22 | uniforme 2.5° | 12 | Matérn 5/2 | 0.0966 | 976 |

### 3.2 Holdout pre-registrado (`BRIEF_holdout.md` + `ROM_holdout_eval.py`)
- **9 ángulos** {3.125, 8.125, …, 43.125}° — peine de paso 5° desplazado 3.125° (múltiplos impares de 0.625° → nunca en el retículo de entrenamiento). Estratificación 2/4/3 (fuera-baja / zona / fuera-alta). Métricas e hipótesis **fijadas antes de simular**.
- Métricas: RMSE global, error relativo con **denominador común** (media de 27 snapshots), error en $C_{p,min}$ (pico de succión).
- **Resultados (media sobre 9 ángulos):**

| Métrica | base_ref | base_kawai | base_kawai2 | base_full |
|---|---|---|---|---|
| RMSE global | 0.0139 | 0.0115 | 0.0116 | **0.0107** |
| RMSE zona [10-30] | 0.0179 | 0.0124 | **0.0116** | 0.0132 |
| Err. rel. zona | 0.1473 | 0.1138 | **0.1088** | 0.1175 |
| Err. $C_{p,min}$ global | 0.0380 | 0.0257 | **0.0243** | 0.0333 |
| Err. $C_{p,min}$ zona | 0.0630 | 0.0421 | **0.0305** | 0.0461 |

- **Hallazgo clave (revierte la Fase C):** con LOO, `base_full` parecía ganar en zona (8.8% vs 11.7%). Con el holdout ciego real, **`base_kawai2` tiene el menor RMSE/err.rel. en zona y domina en $C_{p,min}$** (cantidad físicamente relevante para cargas de diseño). `base_full` gana en error medio *global* y es marginalmente más barata. **Ninguna comparación pareada es significativa** (test de signos, n=9). → No hay base dominante; la elección depende de la métrica priorizada (global vs carga de diseño en la zona crítica).

### 3.3 Suelo de ruido CFD (`ROM_symmetry_noise.py`)
- Por simetría C4+reflexión, $C_p(\theta) = C_p(90°−\theta)$ salvo permutación de sondas. Se descubre **empíricamente** la transformación correcta probando las 8 del grupo D4 sobre pares ya simulados (40↔50, 42.5↔47.5).
- Ganadora: **reflexión antidiagonal**, RMS = **0.0004** (las otras 7 dan 0.6–0.8). → El pipeline respeta la simetría física a **3 órdenes de magnitud**; diferencias entre bases por debajo de ~0.0004 no son atribuibles al diseño muestral.

### 3.4 Diagnóstico del suelo de error (`ROM_floor_diagnostic.py` + `ROM_floor_external.py`)
Separa error de **proyección** (base POD) vs **interpolación** (GPR) y su escala con la densidad Δθ.
- **Autocontenido** (verdad = snapshots reconstruidos de la base): meseta en Δθ≈5° (RMSE 0.0260→0.0147→0.0145→0.0144 para Δθ 10/5/2.5/dens); suelo de proyección **0.0018**; barrido vs r: ε_proj satura ~0.030 (r≥11), ε_interp **crece** (0.003→0.067), ε_total mínimo 0.0966 en r≈12.
- **Externo riguroso** (verdad = CFD real del holdout, re-POD honesto): a Δθ=2.5°, ε_proj=0.0297, ε_total=0.0743 (= holdout de base_full, validación cruzada). La meseta hacia Δθ≈5° **se mantiene**. **Matiz importante para la memoria:** con datos reales el suelo de proyección es ~40% del error total (no el 12% autocontenido) → ε_proj < ε_interp **pero no "mucho menor"**; la base POD sí contribuye al error de un ángulo nunca visto.

### 3.5 Kernel no estacionario (`ROM_nonstationary_kernel.py` + `ROM_nonstationary_external.py`)
- GP de Gibbs con ℓ(θ) por MLE (β libre) vs estacionario (β=0). **Idénticos**: autocontenido 0.0140/0.0158/0.0105 (glob/zona/fuera) ambos; externo 0.0108/0.0122/0.0096 ambos. β solo se activa en modos 7–9 (energía ínfima).
- **Jitter vs señal:** jitter numérico (test de simetría, 4e-4) = 2.5% del residuo de interpolación (0.06% en energía). ⇒ el residuo es **señal no capturada, no ruido**.

### 3.6 Sensibilidad (`ROM_gibbs_sensitivity.py`) — blindaje del hallazgo anterior
- **3.1 Forma del kernel:** 9 campanas (μ∈{15,20,25}°, ancho∈{5,8,12}°) + ℓ(θ) **libre** (spline log-lineal 4 nudos). **Ninguno** bate al estacionario (umbral 5%; el libre incluso −2.5%).
- **3.2 Sensibilidad a r\*** ∈ {8,12,16}: Gibbs = estacionario en todos.
- **3.3 LOO completo n=22** (bootstrap): diferencia Gibbs−estacionario = **+0.00007**, IC95% [0.00000, 0.00018]. El 0 queda fuera → estadísticamente distinguible **pero Gibbs es ~0.5% peor** (efecto despreciable). Conclusión defendible: **no-estacionariedad inexplotable con N≈16–22 (Occam), con intervalo formal.**

### 3.7 Gibbs en la selección de kernel (bake-off de 5) — recook 2026-07-18
Con Gibbs ya como candidato, se re-corrieron las 4 bases:
- **Selecciones idénticas** a lo previo (Matérn 3/2 ×3, Matérn 5/2 en base_full; r\* y LOO byte-idénticos).
- Gibbs compite y **pierde** siempre. Ej. base_full (r=4): Matérn52 0.1212, Matérn32 0.1214, RatQuad 0.1234, **Gibbs 0.1260**, RBF 0.1329. base_ref: Matérn32 0.1750 gana, Gibbs 0.2187.
- Figuras `outputs/<base>/GPR_kernel_comparison.png` ahora muestran las 5 barras → **el pipeline demuestra en cada corrida que la no-estacionariedad no gana**, en vez de asumirlo.

---

## 4. Dónde están los datos (para análisis en Cowork)

| Contenido | Ruta |
|---|---|
| Resultados detallados + narrativa (Fases 1-4, holdout §7.6, coste, conclusiones) | `documentacion/RESULTADOS_ROM.md` |
| POD + GPR por base (npz) y figuras (png) | `outputs/{base_ref,base_kawai,base_kawai2,base_full}/` |
| Comparativa entre bases | `outputs/comparativa/*.csv` + `error_vs_theta_comparativa.png` |
| Coste normalizado | `outputs/comparativa/cost_{per_case,resumen}.csv` + `cost_vs_error.png` |
| Holdout (por ángulo, resumen, pareadas, ruido) | `outputs/holdout/holdout_*.csv`, `symmetry_noise.csv`, `error_vs_theta_holdout_*.png` |
| Suelo de error | `outputs/floor_diag/*.csv`, `outputs/floor_external/*.csv` |
| Kernel no estacionario + jitter | `outputs/nonstationary/*.csv`, `outputs/nonstationary_external/*.csv` |
| Sensibilidad Gibbs | `outputs/sensitivity/{shape_sweep,free_ell,rstar_sweep,loo_n22}.csv` |
| Pre-registro del holdout | `documentacion/BRIEF_holdout.md` |
| Snapshot pre-Gibbs (idéntico tras recook) | `legacy/outputs_2026-07-18/` |

---

## 5. Síntesis de hallazgos para el capítulo

1. **Muestreo adaptativo**: densificar la zona del vórtice a 2.5° es la inversión más rentable (base_kawai, +19% LOO). Más allá (1.25°, base_kawai2) no mejora el error medio global pero **sí el pico de succión** — el holdout ciego lo confirma y **revierte** la lectura por LOO.
2. **El LOO no es comparable entre diseños muestrales** (geometría de folds distinta); la comparación válida es un holdout externo con denominador común. Argumento metodológico fuerte.
3. **El suelo de error (~0.014) es un límite muestral**: señal de la migración del vórtice submuestreada a 2.5–5°. NO es de la base POD, NO de RANS, NO del kernel, NO de ruido numérico — cada alternativa se descartó con su propio experimento. Único lever real: más snapshots dirigidos al vórtice.
4. **No-estacionariedad inexplotable** (Gibbs = estacionario, β→0), con respaldo estadístico (IC bootstrap n=22). Refuerza el uso de Matérn estacionario.
5. **El cuello de botella es la interpolación GPR, no el truncamiento POD** en r\* — pero con datos externos reales la base POD sí aporta ~40% del error (matiz respecto a la versión autocontenida).
6. **El punto duro θ≈22.5–24°** persiste en todas las bases y con todos los kernels → es intrínseco a la reorganización del vórtice, no un déficit de una estrategia concreta.

---

## 6. Decisiones / pendientes

- **Elección de base para el ROM final**: `base_full22` (mejor error global, más barata) vs `base_kawai2` (mejor carga de diseño en la zona crítica). Depende de qué prioriza el capítulo; recomendable justificar por relevancia física, no solo por el promedio.
- **Sensorización óptima (QR pivoting / SSPOR)**: pendiente, se aplicaría sobre la base POD de la opción elegida (Φ ∈ ℝ^{400×r\*}).
- **Figuras de la memoria**: las `GPR_kernel_comparison.png` (5 kernels) están gitignored (regenerables); decidir si versionarlas o solo referenciarlas.
- **`base_full` a N=27**: se decidió mantener N=22 por consistencia; expandir cambiaría los números documentados y solaparía con base_kawai2.
