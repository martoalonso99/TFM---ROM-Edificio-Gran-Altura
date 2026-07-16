"""
ROM_floor_diagnostic.py
=======================
Diagnostico del SUELO de error del ROM: separa que parte del error de
prediccion es de la base POD (proyeccion) y que parte es de la interpolacion
GPR de los coeficientes modales, y como escala cada una con la densidad
angular de entrenamiento (Deltatheta) y con el numero de modos (r).

Objetivo: identificar en que punto densificar deja de mejorar y por que.

Metodo (autocontenido, solo usa outputs/base_full/ROM_POD_basis.npz):
  - Reconstruye los 22 snapshots exactos (la base guardada es de rango N-1).
  - (A) INTERPOLACION con BASE FIJA: fija Phi (POD de los 22, r*=12), toma los
        coeficientes verdaderos y entrena GPR sobre subconjuntos de angulos de
        densidad creciente (Deltatheta = 10, 5, 2.5 grados); evalua en un test
        FIJO de angulos held-out. Aisla el error de interpolacion.
  - (B) ROM COMPLETO con RE-POD por nivel (base recalculada por subconjunto).
  - (C) SUELO DE PROYECCION: mejor reconstruccion posible del test con la base
        fija (sin interpolar) -> cota inferior alcanzable por la base.

Tambien re-reporta el barrido e_proj/e_interp vs r ya guardado en
outputs/base_full/ROM_GPR_results.npz (sweep_mean_proj / sweep_mean_interp).

Salidas -> outputs/floor_diag/:
  floor_vs_dtheta.csv, floor_vs_r.csv, floor_diagnostic.png

NOTA (para re-correr en la maquina con datos completos):
  Esta version usa como "verdad" los snapshots reconstruidos de la propia base.
  Para un holdout externo estricto, sustituir el test por los CFD reales del
  conjunto pre-registrado (ver ROM_holdout_eval.load_cp_holdout) y fijar Phi a
  partir SOLO de los angulos de entrenamiento (re-POD por fold).

Uso:  python ROM_floor_diagnostic.py
"""
from __future__ import annotations
import csv, warnings
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C, WhiteKernel

SCRIPT_DIR = Path(__file__).resolve().parent
OUT_DIR    = SCRIPT_DIR / "outputs" / "floor_diag"
THETA_SCALE = 50.0
RSTAR       = 12
N_RESTARTS  = 15
TEST_ANGLES = [7.5, 12.5, 17.5, 22.5, 27.5, 37.5]   # held-out fijo
VORTEX      = (10.0, 30.0)


def make_kernel(nu: float, var: float):
    return (C(var, (1e-3, 1e3))
            * Matern(length_scale=0.3, length_scale_bounds=(0.05, 2.0), nu=nu)
            + WhiteKernel(1e-6 * var, (1e-10, 1e-2)))


def fit_gpr(theta_tr, y):
    var = max(float(np.var(y)), 1e-8)
    gp = GaussianProcessRegressor(make_kernel(2.5, var), n_restarts_optimizer=N_RESTARTS,
                                  normalize_y=True, alpha=1e-10)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gp.fit(theta_tr, y)
    return gp


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    d = np.load(str(SCRIPT_DIR / "outputs" / "base_full" / "ROM_POD_basis.npz"))
    mean, Phi_full, A_full = d["mean"], d["Phi"], d["A"]
    angles = np.round(d["angles"], 3)
    X = mean[:, None] + Phi_full @ A_full            # snapshots exactos (400 x 22)

    xbar = X.mean(axis=1); Xp = X - xbar[:, None]
    U, S, Vt = np.linalg.svd(Xp, full_matrices=False)
    Phi = U[:, :RSTAR]; A_true = Phi.T @ Xp          # base fija r*=12

    def idx(vals): return [int(np.where(angles == v)[0][0]) for v in vals]
    ti = idx(TEST_ANGLES)
    zona = np.array([VORTEX[0] <= t <= VORTEX[1] for t in TEST_ANGLES])

    levels = {
        "dtheta_10": [0, 10, 20, 30, 40, 45],
        "dtheta_5":  [0, 5, 10, 15, 20, 25, 30, 35, 40, 45],
        "dtheta_2p5_unif": [a for a in [0, 2.5, 5, 7.5, 10, 12.5, 15, 17.5, 20, 22.5,
                            25, 27.5, 30, 32.5, 35, 37.5, 40, 42.5, 45]
                            if a in angles and a not in TEST_ANGLES],
        "dtheta_2p5_dens": [a for a in angles if a not in TEST_ANGLES],
    }

    def field_rmse(Phi_b, bar_b, ahat):
        return np.array([np.sqrt(np.mean((X[:, k] - (bar_b + Phi_b @ ahat[:, m])) ** 2))
                         for m, k in enumerate(ti)])

    rows = []
    print("=" * 74)
    print("(A) INTERPOLACION, base fija r*=12 | (B) ROM completo re-POD por nivel")
    print("=" * 74)
    print(f"{'Nivel':<20}{'A_glob':>8}{'A_zona':>8}{'A_fuera':>8}"
          f"{'B_glob':>9}{'B_zona':>8}{'B_fuera':>8}{'r*':>4}")
    for name, tr_ang in levels.items():
        tr = idx([a for a in tr_ang if a in angles])
        th_tr = (angles[tr] / THETA_SCALE).reshape(-1, 1)
        th_te = (angles[ti] / THETA_SCALE).reshape(-1, 1)
        # (A) base fija
        ahA = np.vstack([fit_gpr(th_tr, A_true[j, tr]).predict(th_te) for j in range(RSTAR)])
        eA = field_rmse(Phi, xbar, ahA)
        # (B) re-POD
        Xtr = X[:, tr]; mb = Xtr.mean(axis=1); Xc = Xtr - mb[:, None]
        Ur, Sr, Vtr = np.linalg.svd(Xc, full_matrices=False)
        rr = min(RSTAR, len(tr) - 1); Phir = Ur[:, :rr]; Ar = Phir.T @ Xc
        ahB = np.vstack([fit_gpr(th_tr, Ar[j, :]).predict(th_te) for j in range(rr)])
        eB = field_rmse(Phir, mb, ahB)
        print(f"{name:<20}{eA.mean():>8.4f}{eA[zona].mean():>8.4f}{eA[~zona].mean():>8.4f}"
              f"{eB.mean():>9.4f}{eB[zona].mean():>8.4f}{eB[~zona].mean():>8.4f}{rr:>4}")
        rows.append(dict(level=name, n_train=len(tr),
                         A_glob=eA.mean(), A_zona=eA[zona].mean(), A_fuera=eA[~zona].mean(),
                         B_glob=eB.mean(), B_zona=eB[zona].mean(), B_fuera=eB[~zona].mean(), r_star=rr))

    # (C) suelo de proyeccion
    proj = np.array([np.sqrt(np.mean((X[:, k] - (xbar + Phi @ (Phi.T @ (X[:, k] - xbar)))) ** 2))
                     for k in ti])
    print(f"\n(C) Suelo de PROYECCION (base fija, sin GPR): glob={proj.mean():.4f} "
          f"zona={proj[zona].mean():.4f} fuera={proj[~zona].mean():.4f}")

    with open(OUT_DIR / "floor_vs_dtheta.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader()
        for r in rows: w.writerow({k: (f"{v:.5f}" if isinstance(v, float) else v) for k, v in r.items()})

    # barrido e_proj/e_interp vs r (ya guardado)
    g = np.load(str(SCRIPT_DIR / "outputs" / "base_full" / "ROM_GPR_results.npz"))
    sr, sp, si, st = g["sweep_r"], g["sweep_mean_proj"], g["sweep_mean_interp"], g["sweep_mean_total"]
    with open(OUT_DIR / "floor_vs_r.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["r", "e_proj", "e_interp", "e_total"])
        for i in range(len(sr)): w.writerow([int(sr[i]), f"{sp[i]:.4f}", f"{si[i]:.4f}", f"{st[i]:.4f}"])

    # figura
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    dt = [10, 5, 2.5, 2.5]
    ax[0].axhline(proj.mean(), ls="--", color="gray", label=f"suelo proyeccion ({proj.mean():.4f})")
    ax[0].plot(dt[:3], [rows[i]["A_glob"] for i in range(3)], "o-", color="#1f4e79", label="interp. (base fija)")
    ax[0].plot(dt[:3], [rows[i]["B_glob"] for i in range(3)], "s--", color="#d62728", label="ROM completo (re-POD)")
    ax[0].set_xlabel("Deltatheta de entrenamiento (deg)"); ax[0].set_ylabel("RMSE en test held-out")
    ax[0].set_title("Error vs densidad angular: meseta ~5 deg"); ax[0].invert_xaxis()
    ax[0].grid(alpha=.3); ax[0].legend(fontsize=8); ax[0].set_ylim(bottom=0)
    ax[1].plot(sr, sp, "o-", label="e_proj"); ax[1].plot(sr, si, "s-", label="e_interp")
    ax[1].plot(sr, st, "^-", color="k", label="e_total")
    ax[1].axvline(int(g["r_star"]), ls=":", color="gray", label=f"r*={int(g['r_star'])}")
    ax[1].set_xlabel("numero de modos r"); ax[1].set_ylabel("error LOO (relativo)")
    ax[1].set_title("Descomposicion proj/interp vs r"); ax[1].grid(alpha=.3); ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT_DIR / "floor_diagnostic.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  -> {OUT_DIR.relative_to(SCRIPT_DIR)}/  (floor_vs_dtheta.csv, floor_vs_r.csv, floor_diagnostic.png)")


if __name__ == "__main__":
    main()
