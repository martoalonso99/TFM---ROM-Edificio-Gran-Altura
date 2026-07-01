# Brief — Análisis paramétrico ROM con muestreo adaptativo

## Objetivo
Demostrar, con datos held-out, que un ROM con muestreo angular grueso (paso 5°) es
**insuficiente en la zona del vórtice de Kawai**, y que **densificar solo esa zona**
recupera la precisión de un muestreo denso global. Es decir, validar una estrategia
de **muestreo adaptativo**: refinar donde la física lo exige, no uniformemente.

## Contexto
- Repo `TFM/Programacion`, rama **`rom-11-tpu`**.
- **Lee `CLAUDE.md`** de la raíz: contiene las decisiones técnicas cerradas (POD centrada
  + SVD económica, LOO-CV con **re-POD por fold**, una GPR por modo, kernel Matérn,
  normalización θ/50, multistart MLE, r* por mínimo del LOO). **Respétalas.**
- Pipeline ya existente: `ROM_POD.py`, `ROM_GPR.py` (con flags `--angles`/`--suffix`),
  `ROM_test_intermediate.py` (test externo en ángulos intermedios).
- Resultado previo que confirma la zona: ROM con 11 ángulos → LOO 14.9%; error test en
  zona vórtice ≈14.4% (pico 26.4% en 27.5°); fuera del vórtice ≈4.2%. La **zona de Kawai
  es ≈[10°, 30°]**.

## Reglas globales (aplican a TODO el análisis y todas las figuras)
1. **Descartar los ángulos > 45°** (50° y 47.5°) por redundancia de simetría C4: el rango
   fundamental de una sección cuadrada es [0°, 45°]. Todo el análisis se restringe a ≤ 45°.
2. **Ejes con ángulos**: poner marcas (ticks/labels) **solo en la malla de referencia de 5°**
   (0, 5, 10, …, 45), **nunca en todos los ángulos densos**. La figura `GPR_r_sweep_*.png`
   tiene el eje X **aglomerado** (una marca por cada ángulo denso) — ese es exactamente el
   problema a evitar. Esto incluye el panel de "error por fold" de r_sweep: etiqueta solo
   los ángulos de referencia.
   > Nota: al descartar >45°, el set de referencia queda en 10 ángulos (0–45). Si prefieres
   > conservar 50° únicamente como marca del eje, ajústalo.
3. **Organizar las salidas** en la estructura de carpetas propuesta (abajo), no sueltas en
   `Programacion/`.

## Flujo de 3 fases

### Fase 1 — Base de referencia (paso 5°)
- Ángulos: `{0, 5, 10, 15, 20, 25, 30, 35, 40, 45}`.
- Generar: POD + GPR + LOO-CV completo (re-POD por fold) + **test externo** en los ángulos
  intermedios disponibles ≤45° (2.5, 7.5, 12.5, 17.5, 22.5, 23.75, 26.25, 27.5, 28.75,
  32.5, 37.5, 42.5).
- Salidas → `outputs/base_ref/`.

### Fase 2 — Identificar zona y densificar (Kawai)
- A partir de la **curva error-vs-θ de la Fase 1**, identificar la zona de alto error
  (esperado: vórtice de Kawai ≈[10°, 30°]).
- **Base nutrida** = referencia + ángulos finos disponibles **dentro de esa zona**
  (12.5, 17.5, 22.5, 23.75, 26.25, 27.5, 28.75).
- Regenerar POD + GPR + LOO + test externo (los ángulos de test que queden fuera de la base).
- Salidas → `outputs/base_kawai/`.

### Fase 3 — Comparar contra la base densa completa
- **Base densa completa** = todos los ángulos disponibles ≤45° (0–45 paso 2.5° + los 1.25°
  de refinamiento = 22 ángulos).
- Comparar las **tres bases** (ref, kawai, full) en: LOO medio, error test por zona
  (dentro/fuera de Kawai), r*, kernel elegido.
- Generar **figura comparativa** error-vs-θ con las tres curvas (ejes con ticks de referencia)
  + **tabla resumen** CSV.
- Salidas → `outputs/comparativa/`.

## Estructura de carpetas propuesta
```
Programacion/
├── ROM_POD.py
├── ROM_GPR.py
├── ROM_test_intermediate.py
├── CLAUDE.md  (referencia)
├── BRIEF_analisis_ROM.md  (este archivo)
└── outputs/
    ├── base_ref/     # 10 áng referencia (5°, ≤45°)
    │   ├── ROM_POD_basis.npz
    │   ├── ROM_GPR_results.npz
    │   ├── POD_*.png
    │   └── GPR_*.png
    ├── base_kawai/   # referencia + densificado en zona Kawai
    │   └── (mismos ficheros)
    ├── base_full/    # todos los ángulos densos ≤45°
    │   └── (mismos ficheros)
    └── comparativa/  # cruce entre las tres bases
        ├── error_vs_theta_comparativa.png
        └── resumen_comparativo.csv
```
- Parametriza el directorio de salida en los scripts (p. ej. `--outdir outputs/base_ref`)
  en lugar de sufijos en el nombre; queda más limpio.
- Añade `outputs/` al `.gitignore` (son regenerables), salvo que quieras versionar las figuras finales.

## Convenciones
- `resumen_comparativo.csv`: una fila por base con `base, n_angulos, r_star, kernel,
  LOO_medio, err_test_kawai, err_test_fuera`.
- Figuras: fondo blanco, PNG ≥150 dpi (o PDF vectorial), ejes con ticks de referencia (regla 2).
- Reporta **error relativo Y RMSE absoluto** en las tablas (el relativo se infla donde
  ‖Cp−mean‖ es pequeño; el RMSE evita malinterpretar picos como el de 27.5°).

## Entregables
1. Las 3 bases (`.npz`) + sus figuras, en `outputs/`.
2. Figura comparativa de las 3 curvas error-vs-θ.
3. `resumen_comparativo.csv`.
4. Commit + push en `rom-11-tpu`.

## No reinventar
- Decisiones técnicas: `CLAUDE.md`.
- `loo_cv_full()` (re-POD por fold) ya está implementado en `ROM_GPR.py`.
- El test externo ya está en `ROM_test_intermediate.py`; adáptalo a los 3 sets.
