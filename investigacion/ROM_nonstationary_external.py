"""
ROM_nonstationary_external.py
==============================
Variante rigurosa (Fase 2 de BRIEF_claude_code_diagnostics.md) de
ROM_nonstationary_kernel.py, punto (a): reutiliza EXACTAMENTE la misma
maquinaria de GP a mano (Gibbs vs estacionario, ver ROM_nonstationary_kernel)
pero sustituye los datos autocontenidos (snapshots reconstruidos desde
outputs/base_full/ROM_POD_basis.npz, test = angulos de training retenidos
del GPR) por:

  - Re-POD con TODOS los 22 angulos de entrenamiento de base_full22
    (CFD real, via load_openfoam_snapshots) -> r*=12.
  - Prediccion en los 9 angulos del holdout pre-registrado (nunca vistos
    por ninguna base) -> comparacion contra su CFD real
    (via ROM_holdout_eval.load_cp_holdout).

Compara RMSE glob/zona/fuera entre el GP estacionario (beta=0) y el GP de
Gibbs con l(theta) MLE (beta libre, anida el estacionario).

Salidas -> outputs/nonstationary_external/:
  gibbs_vs_stationary_external.csv

Criterio de aceptacion (Fase 2, cualitativo): Gibbs ~= estacionario debe
mantenerse (el MLE no encuentra ventaja en la no-estacionariedad tampoco
con el CFD real de test).

Uso:  python ROM_nonstationary_external.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent          # investigacion/
ROOT       = SCRIPT_DIR.parent                        # Programacion/
sys.path.insert(0, str(ROOT))

from ROM_POD import BASE_FULL22, HOLDOUT_SET, load_openfoam_snapshots   # noqa: E402
from ROM_holdout_eval import load_cp_holdout   # noqa: E402
from ROM_nonstationary_kernel import RSTAR, VORTEX, fit_gp, predict   # noqa: E402

OUT_DIR    = ROOT / "outputs" / "nonstationary_external"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    train_angles = sorted(BASE_FULL22)
    holdout_angles = sorted(HOLDOUT_SET)
    zona = np.array([VORTEX[0] <= t <= VORTEX[1] for t in holdout_angles])

    print("Cargando CFD real de entrenamiento (22 angulos de base_full22)...")
    data_tr = load_openfoam_snapshots(train_angles)
    Xtr, angles_tr = data_tr["X"], data_tr["angles"]

    print("Cargando CFD real del holdout (9 angulos, nunca en training)...")
    Cp_holdout = np.column_stack([load_cp_holdout(t) for t in holdout_angles])  # (400, 9)

    # Re-POD con TODOS los angulos de entrenamiento
    xbar = Xtr.mean(axis=1)
    Xc   = Xtr - xbar[:, None]
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    Phi = U[:, :RSTAR]
    A   = Phi.T @ Xc
    th_tr = angles_tr / 50.0
    th_ho = np.array(holdout_angles) / 50.0

    rmse = {"stat": [], "gibbs": []}
    beta_fit = []
    for j in range(RSTAR):
        y = A[j, :]
        gp_s = fit_gp(th_tr, y, beta_free=False)
        gp_g = fit_gp(th_tr, y, beta_free=True)
        beta_fit.append(gp_g["beta"])
        for tag, gp in (("stat", gp_s), ("gibbs", gp_g)):
            pred = np.array([predict(gp, t) for t in th_ho])
            rmse[tag].append(pred)

    def field_rmse(preds):
        e = []
        for m in range(len(holdout_angles)):
            ah = np.array([preds[j][m] for j in range(RSTAR)])
            recon = xbar + Phi @ ah
            e.append(np.sqrt(np.mean((Cp_holdout[:, m] - recon) ** 2)))
        return np.array(e)

    es, eg = field_rmse(rmse["stat"]), field_rmse(rmse["gibbs"])

    print("=" * 74)
    print(f"HOLDOUT EXTERNO (9 ang. CFD real, base fija r*={RSTAR}, "
          f"{len(train_angles)} ang. entrenamiento)")
    print("=" * 74)
    print(f"    {'modelo':<24}{'RMSE glob':>10}{'zona':>9}{'fuera':>9}")
    print(f"    {'GP estacionario (b=0)':<24}{es.mean():>10.4f}{es[zona].mean():>9.4f}{es[~zona].mean():>9.4f}")
    print(f"    {'GP Gibbs l(theta) MLE':<24}{eg.mean():>10.4f}{eg[zona].mean():>9.4f}{eg[~zona].mean():>9.4f}")
    print(f"    mejora zona {100*(1-eg[zona].mean()/es[zona].mean()):+.1f}% | "
          f"global {100*(1-eg.mean()/es.mean()):+.1f}%")
    print(f"    beta MLE por modo (0=eligio estacionario): {np.round(beta_fit[:6], 2)} ...")

    with open(OUT_DIR / "gibbs_vs_stationary_external.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "rmse_glob", "rmse_zona", "rmse_fuera"])
        w.writerow(["stationary", f"{es.mean():.5f}", f"{es[zona].mean():.5f}", f"{es[~zona].mean():.5f}"])
        w.writerow(["gibbs", f"{eg.mean():.5f}", f"{eg[zona].mean():.5f}", f"{eg[~zona].mean():.5f}"])
    print(f"\n  -> {OUT_DIR.relative_to(ROOT)}/gibbs_vs_stationary_external.csv")


if __name__ == "__main__":
    main()
