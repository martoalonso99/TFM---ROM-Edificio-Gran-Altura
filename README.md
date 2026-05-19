# ROM paramétrico de Cp sobre edificio alto

TFM — Máster en Matemática Industrial  
Autor: Martín Rodríguez García

## Descripción

Modelo de orden reducido (ROM) que predice la distribución del coeficiente de presión $C_p$ sobre la superficie de un edificio alto rectangular para cualquier ángulo de incidencia del viento $\theta \in [0°, 50°]$.

Pipeline:

```
CFD (OpenFOAM 10) → snapshots → POD (SVD) → GPR (Kriging) → ROM validado vs TPU
```

Entrenado con 11 simulaciones CFD (θ = 0°, 5°, ..., 50°) y validado contra datos experimentales del **TPU Aerodynamic Database**.

## Estructura

```
Programacion/
├── ROM_POD.py          # Lee OpenFOAM, construye Cp_matrix[400x11], POD via SVD
├── ROM_GPR.py          # Compara kernels, sweep de r, modelo GPR completo
└── copy_stls.py        # Utilidad: copia STLs rotados a cada caso ROM
```

Los datos CFD (casos OpenFOAM, ficheros .mat, coordenadas de probes) viven en
`Simulaciones/` y no se versionan por tamaño.

## Requisitos

```
numpy
scipy
matplotlib
scikit-learn
```

## Uso

```bash
# 1. Extraer snapshots y calcular POD
python ROM_POD.py

# 2. Comparar kernels GPR y obtener r*
python ROM_GPR.py
```

## Edificio

- Dimensiones: B × D × H = 0.10 × 0.10 × 0.40 m (escala modelo, H/B = 4)
- Solver: simpleFoam (RANS estacionario, k-ω SST)
- Perfil ABL: power-law α = 0.25 (AIJ Cat IV), U_H = 11 m/s
- 400 probes body-fixed distribuidos uniformemente en las 4 caras laterales

## Referencias

- Manohar et al. (2018). *Data-Driven Sparse Sensor Placement*. IEEE CSM.
- Tominaga et al. (2008). *AIJ guidelines for CFD*. JWEIA.
- TPU Aerodynamic Database: http://www.wind.arch.t-kougei.ac.jp/system/eng/contents/code/tpu
