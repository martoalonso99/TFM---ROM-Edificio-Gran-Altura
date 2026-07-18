# Brief para Claude Code — checkpoint + validación de los diagnósticos del suelo de error

> Repo: `Programacion/` · rama `rom-11-tpu`. Ejecuta desde `Programacion/`.
> Entorno: el del pipeline (numpy, scipy, scikit-learn, matplotlib).
> Regla: **Fase 0 (push de checkpoint) ANTES de tocar nada más.** No edites los scripts hasta cerrar la Fase 0.

---

## Contexto (qué llegó desde Cowork esta sesión)

Se añadieron dos scripts de diagnóstico y sus salidas:

- `ROM_floor_diagnostic.py` → `outputs/floor_diag/` (`floor_vs_dtheta.csv`, `floor_vs_r.csv`, `floor_diagnostic.png`)
- `ROM_nonstationary_kernel.py` → `outputs/nonstationary/` (`gibbs_vs_stationary.csv`, `jitter_estimate.csv`, `nonstationary_kernel.png`)

Ambos son **autocontenidos**: reconstruyen los 22 snapshots exactos desde `outputs/base_full/ROM_POD_basis.npz` (base de rango N−1) y NO usan los CFD externos del holdout. Esa es justamente la limitación que la Fase 2 corrige.

También se editó `CLAUDE.md` (en la **raíz `TFM/`**, que NO es un repo git → queda fuera de este push; versionarlo es decisión aparte).

---

## Fase 0 — Push de checkpoint (obligatoria, antes de editar)

Objetivo: dejar un punto de restauración con el estado actual **verificado íntegro** (los ficheros venían por sync de OneDrive; hay que confirmar que no están truncados).

```bash
cd Programacion

# 0.1 rama correcta
git rev-parse --abbrev-ref HEAD          # -> rom-11-tpu (si no: git checkout rom-11-tpu)
git status

# 0.2 INTEGRIDAD de los scripts (deben parsear y NO estar truncados)
python -c "import ast; ast.parse(open('ROM_floor_diagnostic.py').read()); ast.parse(open('ROM_nonstationary_kernel.py').read()); print('sintaxis OK')"
tail -n 3 ROM_nonstationary_kernel.py    # DEBE terminar en:  if __name__ == \"__main__\":\n    main()
tail -n 3 ROM_floor_diagnostic.py        # idem (main())

# 0.3 REPRODUCIR (valida ejecución + números; criterios en Fase 1)
python ROM_floor_diagnostic.py
python ROM_nonstationary_kernel.py

# 0.4 ver qué está gitignored (para no intentar versionar artefactos regenerables)
git check-ignore -v outputs/floor_diag/* outputs/nonstationary/* 2>/dev/null || true

# 0.5 STAGE: los scripts son obligatorios; los CSV tracked que cambiaron (corrida jul-15), también
git add ROM_floor_diagnostic.py ROM_nonstationary_kernel.py BRIEF_claude_code_diagnostics.md
git add -u outputs/                      # CSV ya trackeados modificados por el holdout jul-15
# Si outputs/floor_diag y outputs/nonstationary NO están gitignored y los quieres versionar:
#   git add outputs/floor_diag/*.csv outputs/nonstationary/*.csv

git status                               # REVISА el staging antes de commitear
```

Criterio de parada de la Fase 0: **no commitear** si (a) `ast.parse` falla, (b) algún `tail` no muestra `main()`, o (c) los scripts no reproducen los números de la Fase 1. En ese caso, el fichero llegó truncado por el sync → re-sincroniza/regenera antes de seguir.

```bash
git commit -m "ROM: diagnostico del suelo de error (floor + kernel no estacionario + jitter); brief"
git push origin rom-11-tpu
```

---

## Fase 1 — Criterios de aceptación (números esperados de la corrida autocontenida)

`ROM_floor_diagnostic.py`:
- (A) interpolación, base fija r*=12 — RMSE glob por densidad: **Δθ10 ≈ 0.0260 · Δθ5 ≈ 0.0147 · Δθ2.5 ≈ 0.0145 · +dens ≈ 0.0144** (meseta desde 5°).
- (B) ROM completo re-POD por nivel: 0.0288 / 0.0152 / 0.0151 / 0.0151 (r* = 5 / 9 / 12 / 12).
- (C) suelo de proyección (sin GPR): **0.0018**.
- Barrido vs r (de `ROM_GPR_results.npz`): ε_proj 0.365→0.030 (satura r≥11), ε_interp 0.003→0.067 (crece), ε_total mínimo **0.0966 en r≈12**.

`ROM_nonstationary_kernel.py`:
- (a) estacionario = Gibbs: **0.0140 / 0.0158 / 0.0105** (glob/zona/fuera), ambos. β MLE por modo ≈ `[0,0,0,0.48,0,0,4.02,4.02,4.02,0,0.23,0]` (β solo se activa en modos 7–9, energía ínfima).
- (b) jitter (simetría) = 4·10⁻⁴; residuo zona = 0.0158; **ratio 2.5% (0.06% energía)**; ruido relativo de coeficientes 2.7%.

Si los números difieren de forma no trivial, hay divergencia de datos/entorno → investigar antes de la Fase 2.

**Punto de validación conceptual (revisar, no solo ejecutar):** en (b) los estimadores de jitter por coeficientes (σ_n de la MLE, 2ª diferencia) dan artefactos (~0.14) porque la amplitud del modo 1 los contamina; el único estimador limpio es el test de simetría (compara dos configuraciones físicamente idénticas). Confirmar que se está de acuerdo con ese razonamiento (documentado en el docstring de `ROM_nonstationary_kernel.py`).

---

## Fase 2 — Variante rigurosa con holdout EXTERNO (después del checkpoint)

Motivo: en la versión autocontenida el "test" son ángulos de entrenamiento de `base_full` retenidos del GPR pero **presentes en la base POD** (Φ los ha visto) → el suelo de proyección (0.0018) es optimista y la verdad no es CFD independiente. La versión rigurosa usa los **9 ángulos del holdout pre-registrado** (θ = 3.125°…43.125°, nunca vistos) con su **CFD real**, y **re-POD por fold** (base solo con ángulos de entrenamiento).

Reutiliza los helpers existentes:
```python
from ROM_POD import HOLDOUT_SET, load_openfoam_snapshots   # load_..(angs)-> {"X": [400xN], ...}
from ROM_holdout_eval import load_cp_holdout                # load_cp_holdout(theta)-> Cp [400]
from ROM_GPR import make_kernel, N_RESTARTS, THETA_SCALE
```

Crea `ROM_floor_external.py` (o añade `--external` a `ROM_floor_diagnostic.py`) que, para cada malla de densidad Δθ ∈ {10°, 5°, 2.5°} (subconjuntos de los 22 ángulos de `base_full`):
1. **Re-POD solo con la malla de entrenamiento** (centrado + SVD); r* = min(12, N−1) o el que fije el LOO interno.
2. **ε_proj externo**: proyecta cada uno de los 9 CFD del holdout sobre la base de entrenamiento (mejor reconstrucción posible, sin GPR) → RMSE.
3. **ε_total externo**: entrena una GPR por modo (kernel del pipeline) sobre los coeficientes de entrenamiento, predice en los 9 ángulos externos, reconstruye el campo y compara con el CFD real. ε_interp = ε_total − ε_proj (o medido).
4. Denominador común para el error relativo = media de los 27 snapshots ≤45° (idéntico a `ROM_holdout_eval`).
5. Estratifica zona vórtice [10,30] vs fuera, y guarda CSV + figura en `outputs/floor_external/`.

Para el kernel no estacionario externo (`ROM_nonstationary_external.py` o flag):
- Re-POD con **todos** los ángulos de entrenamiento de `base_full` (r*=12), entrena estacionario vs Gibbs sobre esos coeficientes, predice en los **9 ángulos externos** y compara RMSE contra el CFD real (glob/zona/fuera).

**Criterio de aceptación (cualitativo, robusto):** deben mantenerse las tres conclusiones aunque los valores absolutos suban algo (CFD real + re-POD): (i) meseta hacia Δθ≈5°; (ii) ε_proj ≪ ε_interp (suelo de proyección muy por debajo del residuo); (iii) Gibbs ≈ estacionario. Reporta los números externos junto a los autocontenidos.

---

## Fase 3 — Chequeos de sensibilidad (para blindar (a) y (b))

1. **Forma del kernel de Gibbs**: repite (a) variando la campana (μ ∈ {15,20,25}°, ancho ∈ {5,8,12}°) y, además, con una ℓ(θ) **libre** (p. ej. log-ℓ como spline/GP de pocos nodos) para descartar que β→0 sea un artefacto de fijar la forma a priori. Reporta si algún ajuste bate al estacionario de forma significativa.
2. **r***: repite con r* ∈ {8, 12, 16}.
3. **Tamaño del test**: en la versión autocontenida, usa TODOS los ángulos fuera de malla disponibles como test (no solo 6) e informa un intervalo, porque comparaciones tan próximas (0.0140 vs 0.0140) están dentro del ruido con n=6.

---

## Fase 4 — Commit de resultados y reporte

```bash
git add ROM_floor_external.py ROM_nonstationary_external.py   # los que crees
git add -u outputs/                                            # CSV tracked
git commit -m "ROM: variante holdout externo + sensibilidad (Gibbs/forma, r*, n-test)"
git push origin rom-11-tpu
```

Devuélveme: (1) tabla números autocontenidos vs externos, (2) si las tres conclusiones se sostienen, (3) resultado de la sensibilidad de forma del Gibbs (¿algún ℓ(θ) libre bate al estacionario?), (4) cualquier discrepancia con la Fase 1.

---

## Notas

- `CLAUDE.md` (raíz `TFM/`) no está en este repo; si quieres versionar el estado del proyecto, inicializa/gestiona ese repo aparte.
- Los PNG suelen estar gitignored (regenerables); prioriza commitear **scripts** y, si procede, los **CSV** de resultados.
