"""
ROM_sensor_placement.py — Sensorizacion optima (SSPOR, Manohar et al. 2018)
===========================================================================
Coloca sensores de presion optimos sobre la superficie del edificio mediante
QR con pivoteo de columnas sobre la base POD (Sparse Sensor Placement for
Optimal Reconstruction). Con solo r* sensores bien situados se reconstruye el
campo Cp completo (400 sondas) cerca del suelo de proyeccion de la base.

Metodo
------
Base POD  Phi in R^{n x r}  (n = 400 sondas candidatas, r = r* modos).
1. Seleccion QR:  Phi^T P = QR (pivoteo de columnas). Los primeros p indices
   de P son las p ubicaciones optimas (conjunto anidado: p = r, r+1, ...).
2. Reconstruccion (gappy POD): medido y = Cp[S] en las p sondas del conjunto S,
   se estiman los coeficientes modales por minimos cuadrados y se reconstruye
   el campo completo:
       a_hat = (Phi[S,:])^+ (y - mean[S])
       Cp_hat = mean + Phi @ a_hat
3. Se compara el error de reconstruccion frente a:
   - sensores aleatorios (Monte Carlo, banda media +- std),
   - sensores en malla uniforme,
   - el suelo de proyeccion (reconstruccion con las 400 sondas = proyeccion pura).
   Y se evalua tanto en los snapshots de entrenamiento como en el CONJUNTO DE
   TEST (holdout pre-registrado, 9 angulos nunca vistos).

Caveat fisico (para la memoria): el optimo matematico da r* sondas; sobre-
muestrear a ~2r* mejora el condicionamiento y la robustez frente a ruido /
sondas perdidas, a coste de mas instrumentacion. Todas las candidatas son
posiciones de sonda validas (malla 0.02 m), asi que la colocabilidad no es un
problema en este marco discreto; si lo seria en un modelo fisico real.

Salidas -> outputs/<base>/sensors/:
  sensor_indices.csv        (ranking QR: idx, cara, x, y, z)
  reconstruction_vs_p.csv   (p, err QR/random/uniforme, train/test, condicion)
  sensors_map.png           (posiciones optimas sobre las 4 caras desplegadas)
  reconstruction_curve.png  (error vs numero de sensores)
  condition_number.png      (condicion de Phi[S,:] vs p: QR vs aleatorio)

Uso:
  python ROM_sensor_placement.py --outdir outputs/base_kawai2
  python main.py sensors --outdir outputs/base_kawai2
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import qr

from ROM_POD import HOLDOUT_SET, Q_REF, identify_faces, load_openfoam_snapshots
from ROM_holdout_eval import load_cp_holdout

SCRIPT_DIR = Path(__file__).resolve().parent
RNG = np.random.default_rng(0)
N_RANDOM = 200          # sorteos Monte Carlo para el baseline aleatorio


# =======================================================================
#  SELECCION DE SENSORES
# =======================================================================

def qr_pivot_sensors(Phi: np.ndarray, p: int) -> np.ndarray:
    """
    Primeros p indices del pivoteo QR de Phi^T (SSPOR).
    Conjunto anidado: qr_pivot_sensors(Phi, p)[:q] == qr_pivot_sensors(Phi, q).
    """
    _, _, piv = qr(Phi.T, mode="economic", pivoting=True)
    return piv[:p]


def uniform_sensors(n: int, p: int) -> np.ndarray:
    """p indices repartidos uniformemente sobre las n sondas candidatas."""
    return np.unique(np.round(np.linspace(0, n - 1, p)).astype(int))


# =======================================================================
#  RECONSTRUCCION GAPPY POD
# =======================================================================

def reconstruct(Phi: np.ndarray, mean: np.ndarray, S: np.ndarray,
                X: np.ndarray) -> np.ndarray:
    """
    Reconstruye los campos completos X (n x m) a partir de las sondas S.
    Phi (n x r), mean (n,). Devuelve Xhat (n x m).
    """
    C = Phi[S, :]                       # (p x r)
    Cpinv = np.linalg.pinv(C)           # (r x p)
    A_hat = Cpinv @ (X[S, :] - mean[S][:, None])      # (r x m)
    return mean[:, None] + Phi @ A_hat


def rel_error(X: np.ndarray, Xhat: np.ndarray, mean: np.ndarray) -> float:
    """Error relativo medio sobre los m snapshots: ||x-xhat|| / ||x-mean||."""
    num = np.linalg.norm(X - Xhat, axis=0)
    den = np.linalg.norm(X - mean[:, None], axis=0)
    den = np.where(den < 1e-12, np.nan, den)
    return float(np.nanmean(num / den))


def projection_floor(Phi: np.ndarray, mean: np.ndarray, X: np.ndarray) -> float:
    """Error de proyeccion pura (todas las sondas) = mejor reconstruccion posible."""
    A = Phi.T @ (X - mean[:, None])
    Xproj = mean[:, None] + Phi @ A
    return rel_error(X, Xproj, mean)


# =======================================================================
#  MAIN
# =======================================================================

def main(argv=None):
    p = argparse.ArgumentParser(description="Sensorizacion optima QR (SSPOR)")
    p.add_argument("--outdir", default="outputs/base_kawai2",
                   help="Base ROM con ROM_GPR_results.npz (default: outputs/base_kawai2)")
    p.add_argument("--pmax", type=int, default=60, help="Maximo nº de sensores a barrer")
    args = p.parse_args(argv)

    outdir = SCRIPT_DIR / args.outdir
    sens_dir = outdir / "sensors"
    sens_dir.mkdir(parents=True, exist_ok=True)

    # --- Carga de la base POD ---
    d = np.load(str(outdir / "ROM_GPR_results.npz"), allow_pickle=False)
    Phi, mean = d["Phi"], d["mean"]
    r = int(d["r_star"])
    angles = d["angles"]
    probe_coords = d["probe_coords"]
    n = Phi.shape[0]
    print(f"\n{'='*66}")
    print(f"  Sensorizacion optima (SSPOR)  |  base: {outdir.name}")
    print(f"  n={n} sondas candidatas  |  r*={r} modos  |  kernel={str(d['best_kernel'])}")
    print(f"{'='*66}")

    # --- Verdad de campo: CFD real de entrenamiento y de test (holdout) ---
    print("  Cargando CFD de entrenamiento...")
    X_train = load_openfoam_snapshots(list(angles))["X"]        # (400, N_train)
    print("  Cargando CFD del holdout (conjunto de test)...")
    holdout = sorted(HOLDOUT_SET)
    X_test = np.column_stack([load_cp_holdout(t) for t in holdout])  # (400, 9)

    floor_train = projection_floor(Phi, mean, X_train)
    floor_test  = projection_floor(Phi, mean, X_test)
    print(f"  Suelo de proyeccion  train={floor_train:.4f}  test={floor_test:.4f}\n")

    # --- Barrido de p ---
    p_grid = sorted(set([r] + [8, 10, 12, 15, 20, 25, 30, 40, 50, args.pmax]))
    p_grid = [pp for pp in p_grid if 1 <= pp <= n]

    rows = []
    print(f"  {'p':>4} {'QR_tr':>7} {'QR_te':>7} {'rnd_tr':>7} {'unif_tr':>7} {'cond_QR':>9} {'cond_rnd':>9}")
    print(f"  {'-'*4} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*9} {'-'*9}")
    for pp in p_grid:
        S_qr = qr_pivot_sensors(Phi, pp)
        e_qr_tr = rel_error(X_train, reconstruct(Phi, mean, S_qr, X_train), mean)
        e_qr_te = rel_error(X_test,  reconstruct(Phi, mean, S_qr, X_test),  mean)
        cond_qr = float(np.linalg.cond(Phi[S_qr, :]))

        # baseline uniforme
        S_un = uniform_sensors(n, pp)
        e_un_tr = rel_error(X_train, reconstruct(Phi, mean, S_un, X_train), mean)

        # baseline aleatorio (Monte Carlo)
        e_rnd, cond_rnd = [], []
        for _ in range(N_RANDOM):
            S_rd = RNG.choice(n, size=pp, replace=False)
            e_rnd.append(rel_error(X_train, reconstruct(Phi, mean, S_rd, X_train), mean))
            cond_rnd.append(float(np.linalg.cond(Phi[S_rd, :])))
        e_rnd_med = float(np.median(e_rnd))
        e_rnd_p25, e_rnd_p75 = (float(v) for v in np.percentile(e_rnd, [25, 75]))
        cond_rnd_med = float(np.median(cond_rnd))

        rows.append(dict(p=pp, qr_train=e_qr_tr, qr_test=e_qr_te,
                         rnd_train_med=e_rnd_med, rnd_train_p25=e_rnd_p25,
                         rnd_train_p75=e_rnd_p75, unif_train=e_un_tr,
                         cond_qr=cond_qr, cond_rnd=cond_rnd_med))
        print(f"  {pp:>4} {e_qr_tr:>7.4f} {e_qr_te:>7.4f} {e_rnd_med:>7.4f} "
              f"{e_un_tr:>7.4f} {cond_qr:>9.1f} {cond_rnd_med:>9.1f}")

    # --- Ranking de sensores (para la memoria) ---
    S_full = qr_pivot_sensors(Phi, min(2 * r, n))
    faces = identify_faces(probe_coords[:, :, 0])
    face_of = np.empty(n, dtype=object)
    for fname, f in faces.items():
        face_of[f["idx"]] = fname

    with open(sens_dir / "sensor_indices.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rank", "probe_idx", "cara", "x", "y", "z"])
        for rank, idx in enumerate(S_full, 1):
            x, y, z = probe_coords[idx, :, 0]
            w.writerow([rank, int(idx), face_of[idx], f"{x:.4f}", f"{y:.4f}", f"{z:.4f}"])

    with open(sens_dir / "reconstruction_vs_p.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) + ["floor_train", "floor_test"])
        w.writeheader()
        for rw in rows:
            rw2 = {k: (f"{v:.5f}" if isinstance(v, float) else v) for k, v in rw.items()}
            rw2["floor_train"] = f"{floor_train:.5f}"
            rw2["floor_test"] = f"{floor_test:.5f}"
            w.writerow(rw2)

    # --- Resumen headline ---
    row_r = next(rw for rw in rows if rw["p"] == r)
    print(f"\n  Con r*={r} sensores optimos (QR):")
    print(f"    train {row_r['qr_train']:.4f} vs suelo {floor_train:.4f}  "
          f"(aleatorio mediana {row_r['rnd_train_med']:.4f})")
    print(f"    test  {row_r['qr_test']:.4f} vs suelo {floor_test:.4f}")
    print(f"    condicion Phi[S,:]  QR={row_r['cond_qr']:.1f}  aleatorio~{row_r['cond_rnd']:.1f}")

    # --- Figuras ---
    _plot_reconstruction(rows, floor_train, floor_test, r,
                         sens_dir / "reconstruction_curve.png")
    _plot_condition(rows, r, sens_dir / "condition_number.png")
    _plot_sensor_map(faces, mean, S_full, r, probe_coords,
                     sens_dir / "sensors_map.png")

    print(f"\n  Salidas -> {sens_dir.relative_to(SCRIPT_DIR)}/\n")


# =======================================================================
#  FIGURAS
# =======================================================================

def _plot_reconstruction(rows, floor_tr, floor_te, r, out_path):
    ps = np.array([rw["p"] for rw in rows])
    fig, ax = plt.subplots(figsize=(8, 5.2))
    ax.axvline(r, ls=":", color="gray", lw=1.2, label=f"r* = {r}")
    ax.axvline(2 * r, ls=":", color="#bbbbbb", lw=1.0, label=f"2·r* = {2*r}")
    ax.axhline(floor_tr, ls="--", color="#1f4e79", lw=1.2,
               label=f"suelo proyeccion (train) = {floor_tr:.3f}")

    rnd_med = np.array([rw["rnd_train_med"] for rw in rows])
    rnd_p25 = np.array([rw["rnd_train_p25"] for rw in rows])
    rnd_p75 = np.array([rw["rnd_train_p75"] for rw in rows])
    ax.fill_between(ps, np.maximum(rnd_p25, 1e-3), rnd_p75, color="#d62728", alpha=0.15)
    ax.plot(ps, rnd_med, "o-", color="#d62728", lw=1.3, ms=5,
            label="aleatorio (mediana, IQR)")
    ax.plot(ps, [rw["unif_train"] for rw in rows], "^--", color="#7f7f7f",
            lw=1.2, ms=5, label="malla uniforme")
    ax.plot(ps, [rw["qr_train"] for rw in rows], "s-", color="#2ca02c",
            lw=1.8, ms=6, label="QR optimo (train)")
    ax.plot(ps, [rw["qr_test"] for rw in rows], "D-", color="#ff7f0e",
            lw=1.6, ms=5, label="QR optimo (test/holdout)")

    ax.set_yscale("log")
    ax.set_xlabel("Numero de sensores p", fontsize=11)
    ax.set_ylabel("Error relativo de reconstruccion  (log)", fontsize=11)
    ax.set_title("Reconstruccion del campo Cp desde p sensores\n"
                 "(QR pivoting / SSPOR vs aleatorio vs uniforme)",
                 fontweight="bold", fontsize=11)
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=8.5, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out_path.name}")


def _plot_condition(rows, r, out_path):
    ps = [rw["p"] for rw in rows]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.axvline(r, ls=":", color="gray", lw=1.2, label=f"r* = {r}")
    ax.semilogy(ps, [rw["cond_qr"] for rw in rows], "s-", color="#2ca02c",
                lw=1.8, ms=6, label="QR optimo")
    ax.semilogy(ps, [rw["cond_rnd"] for rw in rows], "o-", color="#d62728",
                lw=1.3, ms=5, label="aleatorio (mediana)")
    ax.set_xlabel("Numero de sensores p", fontsize=11)
    ax.set_ylabel("Condicion de Phi[S,:]  (cond)", fontsize=11)
    ax.set_title("Condicionamiento de la matriz de medida\n"
                 "(menor = reconstruccion mas robusta)", fontweight="bold", fontsize=11)
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out_path.name}")


def _plot_sensor_map(faces, mean, S_full, r, probe_coords, out_path):
    """4 caras desplegadas: campo medio Cp de fondo + sensores optimos marcados."""
    S_primary = set(S_full[:r])            # r* primeros (circulos rellenos)
    S_extra = set(S_full[r:])              # sobre-muestreo r*..2r* (cuadrados)
    vmax = float(np.max(np.abs(mean)))
    fig, axes = plt.subplots(1, 4, figsize=(15, 5), sharey=True)

    for ax, (fname, f) in zip(axes, faces.items()):
        idx = f["idx"]; u = f["u"]; v = f["v"]
        ax.scatter(u, v, c=mean[idx], cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                   s=55, marker="s", edgecolors="none", alpha=0.85)
        for local, gi in enumerate(idx):
            if gi in S_primary:
                ax.scatter(u[local], v[local], s=130, facecolors="none",
                           edgecolors="k", linewidths=2.0, zorder=5)
            elif gi in S_extra:
                ax.scatter(u[local], v[local], s=90, marker="D",
                           facecolors="none", edgecolors="#333333",
                           linewidths=1.3, zorder=4)
        ax.set_title(fname, fontsize=10)
        ax.set_xlabel(f["ulabel"], fontsize=9)
        ax.set_aspect("equal", adjustable="box")
    axes[0].set_ylabel("z (m)", fontsize=9)

    handles = [
        plt.Line2D([], [], marker="o", ls="", mfc="none", mec="k", mew=2, ms=11,
                   label=f"sensores optimos (r*={r})"),
        plt.Line2D([], [], marker="D", ls="", mfc="none", mec="#333333", mew=1.3,
                   ms=9, label=f"sobre-muestreo (hasta 2·r*={2*r})"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=9,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Ubicaciones optimas de sensores (QR pivoting) sobre el campo Cp medio",
                 fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out_path.name}")


if __name__ == "__main__":
    main()
