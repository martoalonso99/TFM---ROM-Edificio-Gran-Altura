"""
ROM_floor_external.py
======================
Variante rigurosa (Fase 2 de BRIEF_claude_code_diagnostics.md) de
ROM_floor_diagnostic.py.

Motivo: en la version autocontenida el "test" son angulos de entrenamiento
de base_full retenidos del GPR pero PRESENTES en la base POD (Phi los ha
visto) -> el suelo de proyeccion (0.0018) es optimista. Esta version usa:
  - CFD real de los 9 angulos del holdout pre-registrado (nunca vistos)
  - re-POD SOLO con los angulos de entrenamiento en cada nivel de densidad
    (Phi tampoco ha visto el holdout)

Para cada nivel Delta-theta en {10, 5, 2.5} (subconjuntos de los 22
angulos de base_full22):
  1. Re-POD solo con esa malla de entrenamiento (SVD centrada);
     r* = min(12, N_train - 1).
  2. eps_proj externo: proyecta el CFD real de cada uno de los 9 angulos
     del holdout sobre la base de entrenamiento (mejor reconstruccion
     posible SIN GPR) -> RMSE relativo.
  3. eps_total externo: una GPR por modo (kernel Matern52 -- el
     seleccionado por el pipeline para base_full) entrenada sobre los
     coeficientes de entrenamiento, prediccion en los 9 angulos externos,
     reconstruccion y comparacion con el CFD real.
  4. Denominador comun del error relativo = ||Cp - mean_27|| por angulo,
     con mean_27 = media de los 27 snapshots de entrenamiento <=45
     disponibles (identico a ROM_holdout_eval.py).
  5. Estratifica zona del vortice [10,30] vs fuera.

Salidas -> outputs/floor_external/:
  floor_external_vs_dtheta.csv, floor_external.png

Criterio de aceptacion (Fase 2, cualitativo -- deben mantenerse aunque
suban los valores absolutos frente a la version autocontenida):
  (i)   meseta de eps_total hacia Delta-theta ~= 5 grados
  (ii)  eps_proj << eps_interp (suelo de proyeccion muy por debajo del
        residuo de interpolacion)
  (iii) [Gibbs ~= estacionario: ver ROM_nonstationary_external.py]

Uso:  python ROM_floor_external.py
"""

from __future__ import annotations

import csv
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor

SCRIPT_DIR = Path(__file__).resolve().parent          # investigacion/
ROOT       = SCRIPT_DIR.parent                        # Programacion/
sys.path.insert(0, str(ROOT))

from ROM_POD import ANGLES, BASE_FULL22, HOLDOUT_SET, MAX_THETA, load_openfoam_snapshots   # noqa: E402
from ROM_holdout_eval import load_cp_holdout   # noqa: E402
from ROM_GPR import N_RESTARTS, THETA_SCALE, make_kernel   # noqa: E402

OUT_DIR    = ROOT / "outputs" / "floor_external"
RSTAR_MAX  = 12
KERNEL     = "Matern52"          # kernel seleccionado por el pipeline para base_full
VORTEX     = (10.0, 30.0)

LEVELS = {
    "dtheta_10":  [0, 10, 20, 30, 40, 45],
    "dtheta_5":   [0, 5, 10, 15, 20, 25, 30, 35, 40, 45],
    "dtheta_2p5": sorted(BASE_FULL22),
}


def fit_gpr(theta_tr: np.ndarray, y: np.ndarray) -> GaussianProcessRegressor:
    var = max(float(np.var(y)), 1e-8)
    gp = GaussianProcessRegressor(
        make_kernel(KERNEL, var),
        n_restarts_optimizer=N_RESTARTS,
        normalize_y=True,
        alpha=1e-10,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gp.fit(theta_tr, y)
    return gp


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    holdout_angles = sorted(HOLDOUT_SET)
    zona = np.array([VORTEX[0] <= t <= VORTEX[1] for t in holdout_angles])

    print("Cargando CFD real del holdout (9 angulos pre-registrados, nunca en training)...")
    Cp_holdout = np.column_stack([load_cp_holdout(t) for t in holdout_angles])  # (400, 9)

    print("Cargando denominador comun (27 snapshots <=45 deg, identico a ROM_holdout_eval)...")
    angles_27 = sorted(a for a in ANGLES if a <= MAX_THETA)
    data27  = load_openfoam_snapshots(angles_27)
    mean_27 = data27["X"].mean(axis=1)
    denom   = np.linalg.norm(Cp_holdout - mean_27[:, None], axis=0)  # (9,)

    print("Cargando CFD real de los 22 angulos de base_full22 (universo de entrenamiento)...")
    data22 = load_openfoam_snapshots(sorted(BASE_FULL22))
    X22, angles22 = data22["X"], data22["angles"]

    def col(theta: float) -> int:
        return int(np.where(np.round(angles22, 3) == theta)[0][0])

    rows = []
    print("=" * 78)
    print("HOLDOUT EXTERNO (9 ang., CFD real, re-POD por nivel de densidad)")
    print("=" * 78)
    print(f"{'Nivel':<14}{'N_tr':>5}{'r*':>4}{'proj_glob':>11}{'proj_zona':>10}"
          f"{'proj_fuera':>11}{'tot_glob':>10}{'tot_zona':>10}{'tot_fuera':>10}")

    for name, ang_list in LEVELS.items():
        tr_idx = [col(a) for a in ang_list]
        Xtr    = X22[:, tr_idx]
        th_tr  = (angles22[tr_idx] / THETA_SCALE).reshape(-1, 1)

        mb = Xtr.mean(axis=1)
        Xc = Xtr - mb[:, None]
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        r   = min(RSTAR_MAX, len(tr_idx) - 1)
        Phi = U[:, :r]
        A_tr = Phi.T @ Xc

        # eps_proj externo: proyecta el CFD REAL del holdout sobre la base de entrenamiento
        Xh_c       = Cp_holdout - mb[:, None]
        proj_field = mb[:, None] + Phi @ (Phi.T @ Xh_c)
        e_proj     = np.linalg.norm(Cp_holdout - proj_field, axis=0) / denom

        # eps_total externo: GPR por modo (kernel del pipeline), prediccion en los 9 externos
        th_ho      = (np.array(holdout_angles) / THETA_SCALE).reshape(-1, 1)
        alpha_pred = np.vstack([fit_gpr(th_tr, A_tr[j, :]).predict(th_ho) for j in range(r)])
        rom_field  = mb[:, None] + Phi @ alpha_pred
        e_total    = np.linalg.norm(Cp_holdout - rom_field, axis=0) / denom

        print(f"{name:<14}{len(tr_idx):>5}{r:>4}"
              f"{e_proj.mean():>11.4f}{e_proj[zona].mean():>10.4f}{e_proj[~zona].mean():>11.4f}"
              f"{e_total.mean():>10.4f}{e_total[zona].mean():>10.4f}{e_total[~zona].mean():>10.4f}")

        rows.append(dict(
            level=name, n_train=len(tr_idx), r_star=r,
            proj_glob=e_proj.mean(), proj_zona=e_proj[zona].mean(), proj_fuera=e_proj[~zona].mean(),
            total_glob=e_total.mean(), total_zona=e_total[zona].mean(), total_fuera=e_total[~zona].mean(),
        ))

    with open(OUT_DIR / "floor_external_vs_dtheta.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r_ in rows:
            w.writerow({k: (f"{v:.5f}" if isinstance(v, float) else v) for k, v in r_.items()})
    print(f"\n  -> {OUT_DIR.relative_to(ROOT)}/floor_external_vs_dtheta.csv")

    fig, ax = plt.subplots(figsize=(7, 5))
    dt = [10, 5, 2.5]
    ax.plot(dt, [r_["proj_glob"] for r_ in rows], "o--", color="gray",
            label="eps_proj externo (sin GPR)")
    ax.plot(dt, [r_["total_glob"] for r_ in rows], "s-", color="#1f4e79",
            label="eps_total externo (ROM completo)")
    ax.set_xlabel("Delta theta de entrenamiento (deg)")
    ax.set_ylabel("Error relativo (denom. comun, 9 ang. holdout real)")
    ax.set_title("Holdout externo: eps_proj vs eps_total por densidad")
    ax.invert_xaxis()
    ax.grid(alpha=.3)
    ax.legend(fontsize=9)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "floor_external.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {OUT_DIR.relative_to(ROOT)}/floor_external.png")


if __name__ == "__main__":
    main()
