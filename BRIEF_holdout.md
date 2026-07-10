# Pre-registro: holdout común para comparativa justa entre diseños muestrales

**Fecha de pre-registro:** 2026-07-10 (commit de este fichero, ANTES de ejecutar ninguna simulación de holdout)
**Contexto:** `RESULTADOS_ROM.md` §7.3 establece que el LOO-CV no es comparable entre diseños muestrales distintos (la geometría de los folds difiere). Este documento fija, antes de simular, el conjunto de test común, las métricas y el protocolo de evaluación. Cualquier desviación posterior debe documentarse explícitamente.

---

## 1. Conjunto de test (9 ángulos)

$$T = \{3.125°,\ 8.125°,\ 13.125°,\ 18.125°,\ 23.125°,\ 28.125°,\ 33.125°,\ 38.125°,\ 43.125°\}$$

**Justificación del diseño (peine desplazado):**
- Todos los ángulos de entrenamiento existentes (27 casos) viven en el retículo $k \cdot 1.25°$. Los puntos de $T$ son múltiplos impares de $0.625°$ → nunca coinciden con ningún ángulo de entrenamiento, presente o futuro sobre ese retículo.
- $T$ es un peine uniforme de paso 5° desplazado 3.125°: cobertura sistemática de $[0°, 45°]$ sin selección manual de puntos (cero cherry-picking; la franja difícil 22.5–24° queda cubierta por construcción, no por elección).
- Estratificación exactamente proporcional a las longitudes de región: 2 puntos en $[0°,10°)$, 4 en $[10°,30°]$ (zona vórtice), 3 en $(30°,45°]$.
- Distancias al entrenamiento interpretables: cada punto queda a 0.625° del grid paso-2.5° (base_full22; también base_kawai2 dentro de zona) y a 1.875° de los grids paso-5°.

## 2. Reglas de pureza

1. Los ángulos de $T$ **no entran jamás en ninguna base de entrenamiento** ROM mientras el capítulo comparativo no esté cerrado.
2. Implementación técnica: `HOLDOUT_SET` en `ROM_POD.py`; `discover_angles()` los excluye por defecto, de modo que ni la POD, ni el test externo (`ROM_test_intermediate.py`), ni la comparativa (`ROM_comparativa.py`) pueden absorberlos accidentalmente.
3. La evaluación del holdout se hace **exclusivamente** con un script dedicado (`ROM_holdout_eval.py`, a escribir) que aplica las métricas de §4 y nada más.
4. Una vez cerrado el capítulo, el ROM final de producción puede re-entrenarse incorporando $T$ (práctica estándar), dejando constancia.

## 3. Bases a evaluar

Las cuatro bases congeladas, cada una con su **pipeline completo** (su r\* y su kernel seleccionados por su propio LOO interno — no se igualan hiperparámetros entre bases):

| Base | N | r\* | Kernel |
|---|---|---|---|
| base_ref | 10 | 8 | Matérn 3/2 |
| base_kawai | 17 | 10 | Matérn 3/2 |
| base_kawai2 | 22 | 10 | Matérn 3/2 |
| base_full22 | 22 | 12 | Matérn 5/2 |

## 4. Métricas pre-registradas

Para cada base $b$ y cada $\theta^* \in T$:

1. **RMSE global** sobre las 400 sondas (absoluto, sin denominadores dependientes de la base).
2. **Error relativo con denominador común**: $\varepsilon_b(\theta^*) = \|C_{p,CFD} - C_{p,ROM}^{(b)}\|_2 \,/\, \|C_{p,CFD} - \bar{x}^{(27)}\|_2$, donde $\bar{x}^{(27)}$ es la media de los 27 snapshots de entrenamiento disponibles (idéntica para todas las bases).
3. **Error en pico de succión**: $|C_{p,min}^{CFD} - C_{p,min}^{ROM}|$ (cantidad crítica para cargas de diseño).

**Agregación:** media global sobre $T$, media por estrato (zona / fuera), y **comparaciones pareadas** por ángulo entre bases con test de signos (9 pares por pareja de bases; 8/9 → p≈0.02, 9/9 → p≈0.002, unilateral).

## 5. Suelo de ruido CFD (análisis previo a la interpretación)

Antes de interpretar diferencias entre bases, cuantificar el ruido del pipeline CFD (artefacto de malla rotada; cobertura de capas varía 69–98% según el ángulo) con los **pares simétricos existentes** vía la reflexión $\theta \leftrightarrow 90°-\theta$ (simetría de la sección cuadrada):

- Par (40°, 50°) y par (42.5°, 47.5°): reflejar el campo del segundo miembro (permutación de caras + espejo de columnas) y calcular el RMS de la diferencia con el primero.
- Regla de decisión: diferencias de error entre bases **por debajo del suelo de ruido no son atribuibles al diseño muestral** y se reportan como empate.

## 6. Logística de ejecución

- Configuración idéntica a los 27 casos: plantilla V5, `simpleFoam`, 3500 iteraciones, malla ~7M celdas.
- **Ejecución secuencial** (lección de la Fase C: 5 casos concurrentes inflaron ExecutionTime ~2×). Si se ejecuta a 20 cores para reducir el tiempo de reloj, documentarlo; el coste del holdout no computa en el presupuesto de diseño de las bases.
- Coste estimado: 9 × ~43 CPU-h ≈ **390 CPU-h eq10** (~39 h de reloj secuencial a 10 cores).
- Casos generados con `generate_new_cases.py --angles 3.125 8.125 13.125 18.125 23.125 28.125 33.125 38.125 43.125`.

## 7. Resultados esperados (hipótesis, registradas antes de simular)

- H1: base_full22 domina o empata en la media global sobre $T$.
- H2: base_kawai2 no supera a base_full22 dentro de la zona (consistente con Fase C §7.4).
- H3: base_ref pierde con claridad dentro de la zona (>10% error relativo medio).
- H4: fuera de la zona, las cuatro bases quedan dentro del suelo de ruido entre sí, salvo base_ref en $[0°,10°)$.

Si algún resultado contradice H1–H4, se reporta igualmente (el pre-registro existe precisamente para eso).
