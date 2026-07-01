"""
ROM_GPR.py
==========

Entrena modelos GPR (uno por modo POD) e implementa:

  Fase 1 — Comparacion de kernels via LOO-CV a r_compare (calculado automaticamente
    como el r que supera el 99.5% de energia acumulada, clamped a [3, N-2]):
    Compara Matern52, Matern32, RBF y RationalQuadratic.
    Selecciona el kernel con menor error LOO medio.

  Fase 2 — Sweep de r con el mejor kernel:
    Barre r = 1..r_max (= N_snap - 2, rango efectivo del fold LOO) y calcula
    el error LOO total, de proyeccion e interpolacion.
    El r* optimo minimiza el error LOO total.

  Fase 3 — Modelo completo con r* y mejor kernel:
    Entrena r* GPRs sobre los 11 snapshots completos.
    Genera banda de incertidumbre en theta denso.

Salidas (en Programacion/):
  - ROM_GPR_results.npz        : datos para ROM_validation_TPU.py
  - GPR_kernel_comparison.png
  - GPR_r_sweep.png
  - GPR_mode_fits.png
  - GPR_loo_scatter.png

Notas:
  - LOO-CV con re-POD por fold (critico para evitar contaminacion).
  - Una GPR independiente por modo (cokriging descartado con N=11).
  - Kernel: ConstantKernel * <base> + WhiteKernel (ruido regularizador).
  - Normalizacion: theta_norm = theta / 50 in [0, 1].

Autor: TFM - Marto | Mayo 2026
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    ConstantKernel as C,
    Matern,
    RBF,
    RationalQuadratic,
    WhiteKernel,
)


# =======================================================================
#  CONFIGURACION
# =======================================================================

SCRIPT_DIR  = Path(__file__).resolve().parent
POD_NPZ     = SCRIPT_DIR / "ROM_POD_basis.npz"
OUT_NPZ     = SCRIPT_DIR / "ROM_GPR_results.npz"

R_MIN       = 1      # inicio del sweep (siempre 1)
# R_MAX y R_COMPARE se calculan en main() a partir de N_snap:
#   R_MAX     = N_snap - 2  (rango efectivo del fold LOO con N-1 snapshots centrados)
#   R_COMPARE = r tal que energia acumulada >= 99.5%, clamped a [3, R_MAX]
N_RESTARTS  = 50     # restarts MLE por GPR (mayor robustez frente a optimos locales)
THETA_SCALE = 50.0   # theta_norm = theta / THETA_SCALE in [0, 1]

KERNEL_NAMES = ["Matern52", "Matern32", "RBF", "RatQuad"]

COLORS = {
    "Matern52": "#1f4e79",
    "Matern32": "#2e86ab",
    "RBF":      "#e84855",
    "RatQuad":  "#f4a261",
}


# =======================================================================
#  1. CARGA
# =======================================================================

def load_pod_basis(path: Path) -> dict:
    """Carga la base POD generada por ROM_POD.py."""
    if not path.is_file():
        raise FileNotFoundError(
            f"No existe {path}. Ejecuta ROM_POD.py primero."
        )
    data = np.load(str(path), allow_pickle=False)
    out = {k: data[k] for k in data.files}
    print(f"\n{'='*65}")
    print(f"  Base POD cargada desde: {path.name}")
    print(f"  Phi  : {out['Phi'].shape}   (modos espaciales)")
    print(f"  A    : {out['A'].shape}     (coeficientes modales)")
    print(f"  angles: {out['angles']}")
    print(f"{'='*65}\n")
    return out


# =======================================================================
#  2. FABRICA DE KERNELS
# =======================================================================

def make_kernel(name: str, var_y: float):
    """
    Crea un kernel fresco para cada ajuste GPR.
    Estructura: ConstantKernel * <base> + WhiteKernel

    var_y : varianza empirica del coeficiente modal a_j (escala la amplitud
            inicial y los bounds del ruido de forma adaptativa por modo).
    """
    var_y = max(var_y, 1e-8)
    amp = C(
        constant_value=var_y,
        constant_value_bounds=(1e-3 * var_y, 1e3 * var_y),
    )
    noise = WhiteKernel(
        noise_level=1e-6 * var_y,
        noise_level_bounds=(1e-12, 1e-2 * var_y),
    )

    if name == "Matern52":
        base = Matern(length_scale=0.3, length_scale_bounds=(0.05, 2.0), nu=2.5)
    elif name == "Matern32":
        base = Matern(length_scale=0.3, length_scale_bounds=(0.05, 2.0), nu=1.5)
    elif name == "RBF":
        base = RBF(length_scale=0.3, length_scale_bounds=(0.05, 2.0))
    elif name == "RatQuad":
        base = RationalQuadratic(
            length_scale=0.3, alpha=1.0,
            length_scale_bounds=(0.05, 2.0),
            alpha_bounds=(0.1, 20.0),
        )
    else:
        raise ValueError(f"Kernel desconocido: {name}")

    return amp * base + noise


# =======================================================================
#  3. LOO-CV COMPLETO CON RE-POD POR FOLD
# =======================================================================

def loo_cv_full(
    X: np.ndarray,
    angles: np.ndarray,
    r: int,
    kernel_name: str,
    n_restarts: int = N_RESTARTS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    LOO-CV completo: para cada fold k, recalcula la POD desde cero
    con los N-1 snapshots restantes, entrena r GPRs y predice Cp_k.

    La POD se recalcula en cada fold para evitar que el snapshot
    held-out contamine la base — sin esto el error seria artificialmente bajo.

    Returns
    -------
    err_total   : (N_snap,)          error relativo total LOO
    err_proj    : (N_snap,)          error de proyeccion (cota inferior)
    cp_pred_loo : (N_probes, N_snap) campo Cp predicho en cada fold
    """
    N_probes, N_snap = X.shape
    theta_norm = angles / THETA_SCALE

    err_total   = np.full(N_snap, np.nan)
    err_proj    = np.full(N_snap, np.nan)
    cp_pred_loo = np.full((N_probes, N_snap), np.nan)

    for k in range(N_snap):
        mask    = np.arange(N_snap) != k
        X_train = X[:, mask]                          # (N_probes, N_snap-1)
        th_train = theta_norm[mask].reshape(-1, 1)    # (N_snap-1, 1)
        th_test  = np.array([[theta_norm[k]]])        # (1, 1)

        # --- Re-POD ---
        mean_k = X_train.mean(axis=1)                 # (N_probes,)
        Xc     = X_train - mean_k[:, None]
        U_k, s_k, Vt_k = np.linalg.svd(Xc, full_matrices=False)
        Phi_r  = U_k[:, :r]                           # (N_probes, r)
        A_tr   = np.diag(s_k[:r]) @ Vt_k[:r, :]      # (r, N_snap-1)

        # --- Snapshot test centrado ---
        x_test = X[:, k] - mean_k                     # (N_probes,)
        denom  = np.linalg.norm(x_test)
        if denom < 1e-12:
            continue

        # --- Error de proyeccion (sin GPR) ---
        x_proj     = Phi_r @ (Phi_r.T @ x_test)
        err_proj[k] = np.linalg.norm(x_test - x_proj) / denom

        # --- GPR: un modelo por modo ---
        alpha_pred = np.zeros(r)
        for j in range(r):
            a_j   = A_tr[j, :]
            var_j = max(float(np.var(a_j)), 1e-8)
            gpr   = GaussianProcessRegressor(
                kernel=make_kernel(kernel_name, var_j),
                n_restarts_optimizer=n_restarts,
                normalize_y=True,
                alpha=1e-10,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                gpr.fit(th_train, a_j)
            alpha_pred[j] = float(gpr.predict(th_test))

        # --- Reconstruccion y error total ---
        cp_hat          = mean_k + Phi_r @ alpha_pred
        err_total[k]    = np.linalg.norm(x_test - Phi_r @ alpha_pred) / denom
        cp_pred_loo[:, k] = cp_hat

    return err_total, err_proj, cp_pred_loo


# =======================================================================
#  4. FASE 1 — COMPARACION DE KERNELS
# =======================================================================

def compare_kernels(X: np.ndarray, angles: np.ndarray, r_compare: int) -> dict:
    """LOO-CV a r_compare para todos los kernels. Devuelve dict con resultados."""
    print(f"{'='*65}")
    print(f"  Fase 1: comparacion de kernels  (r = {r_compare})")
    print(f"{'='*65}")

    results = {}
    for name in KERNEL_NAMES:
        print(f"  [{name}] ...", flush=True)
        err_t, err_p, _ = loo_cv_full(X, angles, r_compare, name)
        err_i = err_t - err_p
        results[name] = {
            "err_total":  err_t,
            "err_proj":   err_p,
            "err_interp": err_i,
            "mean_total":  float(np.nanmean(err_t)),
            "std_total":   float(np.nanstd(err_t)),
            "mean_proj":   float(np.nanmean(err_p)),
            "mean_interp": float(np.nanmean(err_i)),
        }
        print(f"         LOO medio: {results[name]['mean_total']:.4f}  "
              f"(proj: {results[name]['mean_proj']:.4f}  "
              f"interp: {results[name]['mean_interp']:.4f})")

    best = min(results, key=lambda n: results[n]["mean_total"])
    print(f"\n  >> Mejor kernel: {best}  "
          f"(LOO = {results[best]['mean_total']:.4f})\n")
    return results, best


# =======================================================================
#  5. FASE 2 — SWEEP DE r CON EL MEJOR KERNEL
# =======================================================================

def sweep_r(X: np.ndarray, angles: np.ndarray, kernel_name: str, r_max: int) -> dict:
    """
    LOO-CV completo para r = R_MIN..r_max con el kernel seleccionado.
    Devuelve dict indexado por r con errores y predicciones LOO.
    """
    print(f"{'='*65}")
    print(f"  Fase 2: sweep de r  (kernel = {kernel_name}, r_max = {r_max})")
    print(f"{'='*65}")

    sweep = {}
    for r in range(R_MIN, r_max + 1):
        print(f"  r = {r} ...", flush=True)
        err_t, err_p, cp_loo = loo_cv_full(X, angles, r, kernel_name)
        err_i = err_t - err_p
        sweep[r] = {
            "err_total":    err_t,
            "err_proj":     err_p,
            "err_interp":   err_i,
            "mean_total":   float(np.nanmean(err_t)),
            "mean_proj":    float(np.nanmean(err_p)),
            "mean_interp":  float(np.nanmean(err_i)),
            "cp_pred_loo":  cp_loo,
        }
        print(f"         LOO total: {sweep[r]['mean_total']:.4f}  "
              f"proj: {sweep[r]['mean_proj']:.4f}  "
              f"interp: {sweep[r]['mean_interp']:.4f}")

    r_star = min(sweep, key=lambda r: sweep[r]["mean_total"])
    print(f"\n  >> r* optimo: {r_star}  "
          f"(LOO total = {sweep[r_star]['mean_total']:.4f})\n")
    return sweep, r_star


# =======================================================================
#  6. FASE 3 — MODELO COMPLETO (entrenado sobre los 11 snapshots)
# =======================================================================

def fit_full_model(
    pod: dict,
    r_star: int,
    kernel_name: str,
) -> tuple[list, np.ndarray, np.ndarray, np.ndarray]:
    """
    Entrena r* GPRs sobre los 11 snapshots completos usando la base POD global.
    Devuelve predicciones en una malla densa de theta para visualizacion.

    Returns
    -------
    gprs        : lista de r* objetos GPR ajustados
    mu_dense    : (r*, 200)  medias GPR en theta denso
    std_dense   : (r*, 200)  desviaciones estandar GPR en theta denso
    theta_dense : (200,)     angulos de la malla densa [deg]
    """
    A      = pod["A"]          # (10, 11) coeficientes modales completos
    angles = pod["angles"]
    theta_norm  = angles / THETA_SCALE
    theta_dense = np.linspace(0.0, 1.0, 200)

    gprs      = []
    mu_dense  = np.zeros((r_star, 200))
    std_dense = np.zeros((r_star, 200))

    print(f"{'='*65}")
    print(f"  Fase 3: modelo completo  (r*={r_star}, kernel={kernel_name})")
    print(f"{'='*65}")

    for j in range(r_star):
        a_j   = A[j, :]
        var_j = max(float(np.var(a_j)), 1e-8)
        gpr   = GaussianProcessRegressor(
            kernel=make_kernel(kernel_name, var_j),
            n_restarts_optimizer=N_RESTARTS,
            normalize_y=True,
            alpha=1e-10,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gpr.fit(theta_norm.reshape(-1, 1), a_j)

        mu, std = gpr.predict(theta_dense.reshape(-1, 1), return_std=True)
        mu_dense[j]  = mu
        std_dense[j] = std
        gprs.append(gpr)

        k_params = gpr.kernel_
        print(f"  Modo {j+1:2d}: kernel ajustado = {k_params}")

    return gprs, mu_dense, std_dense, theta_dense * THETA_SCALE


# =======================================================================
#  7. VISUALIZACION
# =======================================================================

def plot_kernel_comparison(kernel_results: dict, r_compare: int, out_path: Path):
    """Bar chart del error LOO medio por kernel, desglosado por fold."""
    names  = KERNEL_NAMES
    n      = len(names)
    x      = np.arange(n)
    width  = 0.35

    means_t = [kernel_results[nm]["mean_total"]  for nm in names]
    means_p = [kernel_results[nm]["mean_proj"]   for nm in names]
    means_i = [kernel_results[nm]["mean_interp"] for nm in names]
    stds_t  = [kernel_results[nm]["std_total"]   for nm in names]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # Panel 1: barras error total con std entre folds
    bars = ax1.bar(x, means_t, color=[COLORS[nm] for nm in names],
                   alpha=0.85, edgecolor="white", linewidth=0.8)
    ax1.errorbar(x, means_t, yerr=stds_t, fmt="none",
                 color="black", capsize=5, linewidth=1.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(names)
    ax1.set_ylabel("Error LOO relativo medio")
    ax1.set_title(f"Error LOO total por kernel  (r = {r_compare})")
    ax1.grid(True, axis="y", alpha=0.3)
    for bar, val in zip(bars, means_t):
        ax1.text(bar.get_x() + bar.get_width() / 2, val + 0.001,
                 f"{val:.4f}", ha="center", va="bottom", fontsize=9)

    # Panel 2: descomposicion proyeccion + interpolacion
    b1 = ax2.bar(x - width/2, means_p, width, label="Proyeccion",
                 color="#4472c4", alpha=0.85, edgecolor="white")
    b2 = ax2.bar(x + width/2, means_i, width, label="Interpolacion GPR",
                 color="#ed7d31", alpha=0.85, edgecolor="white")
    ax2.set_xticks(x)
    ax2.set_xticklabels(names)
    ax2.set_ylabel("Error LOO relativo medio")
    ax2.set_title("Descomposicion: proyeccion vs interpolacion")
    ax2.legend()
    ax2.grid(True, axis="y", alpha=0.3)

    fig.suptitle(f"Comparacion de kernels GPR  (r = {r_compare}, N = {r_compare} modos)",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  -> {out_path.name}")


def plot_r_sweep(sweep: dict, best_kernel: str, r_star: int, angles: np.ndarray,
                 out_path: Path):
    """Curva en U: error LOO total, de proyeccion e interpolacion vs r."""
    r_vals    = sorted(sweep.keys())
    tot       = [sweep[r]["mean_total"]   for r in r_vals]
    proj      = [sweep[r]["mean_proj"]    for r in r_vals]
    interp    = [sweep[r]["mean_interp"]  for r in r_vals]

    # Error por fold para el r*
    err_folds = sweep[r_star]["err_total"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax1.plot(r_vals, tot,    "o-",  color="#1f4e79", lw=2,   ms=8,  label="Total LOO")
    ax1.plot(r_vals, proj,   "s--", color="#70ad47", lw=1.5, ms=7,  label="Proyeccion (cota inf.)")
    ax1.plot(r_vals, interp, "^:",  color="#ed7d31", lw=1.5, ms=7,  label="Interpolacion GPR")
    ax1.axvline(r_star, color="red", ls="--", lw=1.5, alpha=0.7,
                label=f"r* = {r_star}")
    ax1.set_xlabel("r (modos retenidos)")
    ax1.set_ylabel("Error LOO relativo medio")
    ax1.set_title(f"Sweep de r  (kernel = {best_kernel})")
    ax1.set_xticks(r_vals)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale("log")

    # Panel 2: error por fold en r*
    ax2.bar(np.arange(len(err_folds)), err_folds,
            color="#1f4e79", alpha=0.8, edgecolor="white")
    ax2.set_xticks(np.arange(len(err_folds)))
    ax2.set_xticklabels([f"{a:g}°" for a in angles], fontsize=9)
    ax2.set_xlabel("Angulo excluido theta")
    ax2.set_ylabel("Error LOO relativo")
    ax2.set_title(f"Error LOO por fold  (r* = {r_star})")
    ax2.grid(True, axis="y", alpha=0.3)
    ax2.axhline(np.nanmean(err_folds), color="red", ls="--", lw=1.5,
                label=f"Media = {np.nanmean(err_folds):.4f}")
    ax2.legend(fontsize=9)

    fig.suptitle(f"Seleccion de r*: curva LOO total vs r  (kernel = {best_kernel})",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  -> {out_path.name}")


def plot_gpr_mode_fits(
    pod: dict,
    r_star: int,
    best_kernel: str,
    mu_dense: np.ndarray,
    std_dense: np.ndarray,
    theta_dense: np.ndarray,
    out_path: Path,
):
    """
    Para cada modo j=1..r*:
      - Scatter de a_j(theta_k) sobre los 11 puntos de entrenamiento
      - Curva media GPR sobre theta denso [0, 50]
      - Banda de incertidumbre +/- 2 sigma
    """
    A      = pod["A"]
    angles = pod["angles"]
    energy = pod["energy"]
    sigma  = pod["sigma"]

    n_cols = min(3, r_star)
    n_rows = int(np.ceil(r_star / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(5.5 * n_cols, 3.5 * n_rows),
                             sharex=True)
    axes = np.atleast_1d(axes).ravel()

    for j in range(r_star):
        ax  = axes[j]
        a_j = A[j, :]
        mu  = mu_dense[j]
        std = std_dense[j]

        ax.fill_between(theta_dense, mu - 2*std, mu + 2*std,
                        alpha=0.20, color="#1f4e79", label="+/- 2 sigma")
        ax.plot(theta_dense, mu, "-", color="#1f4e79", lw=2, label="GPR media")
        ax.scatter(angles, a_j, color="#c00000", zorder=5, s=55,
                   label="Datos CFD")
        ax.axhline(0, color="black", lw=0.5, alpha=0.4)
        ax.set_title(
            f"Modo j={j+1}  |  sigma={sigma[j]:.3f}  |  E={energy[j]*100:.1f}%",
            fontsize=10,
        )
        ax.grid(True, alpha=0.3)
        ax.set_xticks(angles)
        if j >= r_star - n_cols:
            ax.set_xlabel("theta (deg)")
        if j % n_cols == 0:
            ax.set_ylabel("a_j(theta)")
        if j == 0:
            ax.legend(fontsize=8)

    for j in range(r_star, len(axes)):
        axes[j].axis("off")

    fig.suptitle(
        f"Ajuste GPR por modo  (r*={r_star}, kernel={best_kernel})\n"
        f"Banda = +/-2 sigma (95% credible interval)",
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  -> {out_path.name}")


def plot_loo_scatter(
    X: np.ndarray,
    angles: np.ndarray,
    cp_pred_loo: np.ndarray,
    r_star: int,
    best_kernel: str,
    out_path: Path,
):
    """
    Scatter Cp_CFD vs Cp_ROM_LOO para todos los folds.
    Cada punto es un probe en un angulo; coloreado por theta.
    La diagonal perfecta es Cp_ROM = Cp_CFD.
    """
    N_probes, N_snap = X.shape
    cmap = plt.cm.plasma
    norm = plt.Normalize(vmin=angles.min(), vmax=angles.max())

    fig, ax = plt.subplots(figsize=(7, 6.5))

    for k in range(N_snap):
        cp_true = X[:, k]
        cp_rom  = cp_pred_loo[:, k]
        valid   = ~np.isnan(cp_rom)
        ax.scatter(cp_true[valid], cp_rom[valid],
                   c=np.full(valid.sum(), angles[k]),
                   cmap=cmap, norm=norm,
                   s=8, alpha=0.55, edgecolors="none")

    # Diagonal de referencia
    all_vals = np.concatenate([X.ravel(), cp_pred_loo[~np.isnan(cp_pred_loo)].ravel()])
    lim = [all_vals.min() - 0.05, all_vals.max() + 0.05]
    ax.plot(lim, lim, "k--", lw=1.2, label="Cp_ROM = Cp_CFD")
    ax.set_xlim(lim)
    ax.set_ylim(lim)

    # RMSE global
    mask_all = ~np.isnan(cp_pred_loo)
    rmse = float(np.sqrt(np.mean((X[mask_all] - cp_pred_loo[mask_all])**2)))
    ax.text(0.04, 0.95, f"RMSE global = {rmse:.4f}",
            transform=ax.transAxes, fontsize=10,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.85)
    cbar.set_label("theta excluido (deg)")

    ax.set_xlabel("Cp CFD (referencia)")
    ax.set_ylabel("Cp ROM predicho (LOO)")
    ax.set_title(
        f"Scatter LOO: Cp_CFD vs Cp_ROM  (r*={r_star}, kernel={best_kernel})",
        fontweight="bold",
    )
    ax.legend(fontsize=9)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  -> {out_path.name}")


# =======================================================================
#  8. GUARDADO
# =======================================================================

def save_results(
    pod: dict,
    best_kernel: str,
    r_star: int,
    sweep: dict,
    mu_dense: np.ndarray,
    std_dense: np.ndarray,
    theta_dense: np.ndarray,
    out_path: Path,
):
    """
    Guarda los resultados para ROM_validation_TPU.py.
    El modelo completo (media POD + modos + curvas GPR densas) permite
    reconstruir Cp en cualquier theta sin re-entrenar.
    """
    np.savez(
        out_path,
        # Identificadores del modelo optimo
        best_kernel=best_kernel,
        r_star=r_star,
        # Base POD global (completa, no LOO)
        mean=pod["mean"],
        Phi=pod["Phi"][:, :r_star],
        sigma=pod["sigma"][:r_star],
        A=pod["A"][:r_star, :],
        angles=pod["angles"],
        probe_coords=pod["probe_coords"],
        Uref=pod["Uref"],
        q_ref=pod["q_ref"],
        # Curvas GPR densas para reconstruccion
        theta_dense=theta_dense,
        mu_dense=mu_dense,
        std_dense=std_dense,
        # Errores LOO del sweep (para tablas en la memoria)
        sweep_r=np.array(sorted(sweep.keys())),
        sweep_mean_total=np.array([sweep[r]["mean_total"] for r in sorted(sweep)]),
        sweep_mean_proj=np.array([sweep[r]["mean_proj"]  for r in sorted(sweep)]),
        sweep_mean_interp=np.array([sweep[r]["mean_interp"] for r in sorted(sweep)]),
    )
    print(f"\nResultados guardados en: {out_path.name}")
    print(f"  best_kernel : {best_kernel}")
    print(f"  r*          : {r_star}")


# =======================================================================
#  MAIN
# =======================================================================

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ROM_GPR: GPR por modo POD")
    p.add_argument(
        "--suffix", default="",
        help="Sufijo para ficheros NPZ de E/S, p.ej. _11ang (default: sin sufijo)",
    )
    return p.parse_args()


def main():
    args   = _parse_args()
    suffix = args.suffix

    pod_npz = SCRIPT_DIR / f"ROM_POD_basis{suffix}.npz"
    out_npz = SCRIPT_DIR / f"ROM_GPR_results{suffix}.npz"

    # 0. Carga
    pod    = load_pod_basis(pod_npz)
    # X_centered = Phi @ A  (A ya incluye los valores singulares: A = diag(sigma) @ Vt)
    X_full = pod["Phi"] @ pod["A"] + pod["mean"][:, None]   # (N_probes, N_snap)
    angles = pod["angles"]
    N_snap = X_full.shape[1]

    # Rango efectivo del fold LOO: N-1 snapshots centrados => rango <= N-2
    r_max = N_snap - 2

    # r_compare: r que supera el 99.5% de energia, clamped a [3, r_max]
    cum_e = pod["cum_energy"]
    r_compare = int(np.searchsorted(cum_e, 0.995)) + 1
    r_compare = int(np.clip(r_compare, 3, r_max))

    print(f"\n  Parametros calculados automaticamente:")
    print(f"    suffix    = '{suffix}'")
    print(f"    N_snap    = {N_snap}")
    print(f"    r_max     = {r_max}  (rango efectivo fold LOO = N-2)")
    print(f"    r_compare = {r_compare}  (energia acumulada >= 99.5%)\n")

    # 1. Comparacion de kernels
    kernel_results, best_kernel = compare_kernels(X_full, angles, r_compare)

    # 2. Sweep de r con el mejor kernel
    sweep, r_star = sweep_r(X_full, angles, best_kernel, r_max)

    # 3. Modelo completo con r* y mejor kernel
    gprs, mu_dense, std_dense, theta_dense = fit_full_model(pod, r_star, best_kernel)

    # 4. Figuras
    print(f"\n{'='*65}")
    print("  Generando figuras...")
    print(f"{'='*65}")

    plot_kernel_comparison(
        kernel_results, r_compare,
        SCRIPT_DIR / f"GPR_kernel_comparison{suffix}.png",
    )
    plot_r_sweep(
        sweep, best_kernel, r_star, angles,
        SCRIPT_DIR / f"GPR_r_sweep{suffix}.png",
    )
    plot_gpr_mode_fits(
        pod, r_star, best_kernel,
        mu_dense, std_dense, theta_dense,
        SCRIPT_DIR / f"GPR_mode_fits{suffix}.png",
    )
    plot_loo_scatter(
        X_full, angles,
        sweep[r_star]["cp_pred_loo"], r_star, best_kernel,
        SCRIPT_DIR / f"GPR_loo_scatter{suffix}.png",
    )

    # 5. Guardado
    save_results(
        pod, best_kernel, r_star, sweep,
        mu_dense, std_dense, theta_dense,
        out_npz,
    )

    print(f"\n{'='*65}")
    print(f"  GPR completado (suffix='{suffix}').")
    print(f"  Mejor kernel : {best_kernel}")
    print(f"  r*           : {r_star}")
    print(f"  Siguiente paso: ROM_test_intermediate.py  o  ROM_validation_TPU.py")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
