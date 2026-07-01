"""
ROM_test_intermediate.py
========================
Test externo del ROM-11: predice Cp en los angulos intermedios
(no vistos en entrenamiento) que tienen CFD de referencia.

Demuestra que el muestreo a 5 grados es insuficiente en la region del
vortice de Kawai (~10-30 grados), motivando la densificacion a 24 angulos.

Pipeline:
  ROM_GPR_results_11ang.npz  ->  re-entrenar r* GPRs  ->  predecir en 13 test
  CFD de referencia          ->  comparar  ->  error relativo por angulo

Salidas:
  - GPR_error_vs_theta_11ang.png   <- figura clave (motivacion de densificacion)
  - ROM_comparison_summary.csv     <- tabla resumen para la memoria

Uso:
  python ROM_test_intermediate.py [--suffix _11ang]
"""

from __future__ import annotations

import argparse
import csv
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor

# Reutilizar funciones del pipeline existente
from ROM_POD import (
    DATA_DIR, N_PROBES, Q_REF,
    _parse_probe_file, discover_angles, fmt_angle,
)
from ROM_GPR import make_kernel, N_RESTARTS, THETA_SCALE

SCRIPT_DIR   = Path(__file__).resolve().parent
TPU11_SET    = {0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0}
VORTEX_RANGE = (10.0, 30.0)   # zona del vortice de Kawai


# =======================================================================
#  1. CARGA Y RE-ENTRENAMIENTO
# =======================================================================

def load_gpr_model(suffix: str) -> dict:
    path = SCRIPT_DIR / f"ROM_GPR_results{suffix}.npz"
    if not path.is_file():
        raise FileNotFoundError(
            f"No existe {path}.\n"
            f"Ejecuta: python ROM_POD.py --angles tpu11 --suffix {suffix}\n"
            f"         python ROM_GPR.py --suffix {suffix}"
        )
    raw = np.load(str(path), allow_pickle=False)
    return {k: raw[k] for k in raw.files}


def retrain_gprs(gpr_data: dict) -> list:
    """
    Re-entrena r* GPRs sobre los N_train puntos de entrenamiento.
    Necesario porque save_results() guarda curvas densas pero no los objetos GPR.
    """
    A        = gpr_data["A"]           # (r*, N_train)
    angles   = gpr_data["angles"]      # (N_train,)
    r_star   = int(gpr_data["r_star"])
    kernel   = str(gpr_data["best_kernel"])

    theta_norm = (angles / THETA_SCALE).reshape(-1, 1)

    print(f"  Re-entrenando {r_star} GPRs ({kernel}) sobre {len(angles)} puntos...")
    gprs = []
    for j in range(r_star):
        a_j   = A[j, :]
        var_j = max(float(np.var(a_j)), 1e-8)
        gpr   = GaussianProcessRegressor(
            kernel=make_kernel(kernel, var_j),
            n_restarts_optimizer=N_RESTARTS,
            normalize_y=True,
            alpha=1e-10,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gpr.fit(theta_norm, a_j)
        gprs.append(gpr)
    print(f"  Listo.")
    return gprs


def predict_cp(gpr_data: dict, gprs: list, theta_test: float) -> np.ndarray:
    """Reconstruye Cp(x, theta_test) usando el ROM-11."""
    mean   = gpr_data["mean"]    # (400,)
    Phi    = gpr_data["Phi"]     # (400, r*)
    r_star = int(gpr_data["r_star"])

    th_norm = np.array([[theta_test / THETA_SCALE]])
    alpha   = np.array([float(gprs[j].predict(th_norm)) for j in range(r_star)])
    return mean + Phi @ alpha


# =======================================================================
#  2. IDENTIFICACION DE ANGULOS DE TEST
# =======================================================================

def find_test_angles(angles_all: list[float], angles_train: list[float]) -> list[float]:
    """Angulos en all24 que no estan en la base de entrenamiento."""
    train_set = set(angles_train)
    return [a for a in angles_all if a not in train_set]


# =======================================================================
#  3. EVALUACION DE ERRORES
# =======================================================================

def compute_test_errors(
    gpr_data: dict,
    gprs: list,
    test_angles: list[float],
) -> dict:
    """
    Para cada angulo de test:
      1. Lee CFD de OpenFOAM
      2. Predice con ROM-11
      3. Calcula error relativo normalizado por distancia al mean_train
    """
    mean_train = gpr_data["mean"]
    results    = {}

    print(f"\n  {'theta':>7}  {'Cp_max_CFD':>12}  {'Cp_max_ROM':>12}  {'error_rel':>10}  {'RMSE':>8}")
    print(f"  {'-'*7}  {'-'*12}  {'-'*12}  {'-'*10}  {'-'*8}")

    for theta in test_angles:
        tag   = f"theta_{fmt_angle(theta)}"
        fpath = (DATA_DIR / f"ROMCase_{tag}"
                 / "postProcessing" / "probes_buildingPressure" / "0" / "p")

        if not fpath.is_file():
            print(f"  {theta:>7.2f}  FICHERO NO ENCONTRADO — omitido")
            continue

        _, _, p_vals = _parse_probe_file(fpath)
        cp_cfd  = p_vals / Q_REF
        cp_rom  = predict_cp(gpr_data, gprs, theta)

        denom = np.linalg.norm(cp_cfd - mean_train)
        if denom < 1e-12:
            continue
        error_rel = float(np.linalg.norm(cp_cfd - cp_rom) / denom)
        rmse      = float(np.sqrt(np.mean((cp_cfd - cp_rom) ** 2)))

        results[theta] = {
            "cp_cfd":       cp_cfd,
            "cp_rom":       cp_rom,
            "error_rel":    error_rel,
            "rmse":         rmse,
            "cp_max_cfd":   float(cp_cfd.max()),
            "cp_max_rom":   float(cp_rom.max()),
            "cp_min_cfd":   float(cp_cfd.min()),
            "cp_min_rom":   float(cp_rom.min()),
        }
        print(f"  {theta:>7.2f}  {cp_cfd.max():>12.4f}  {cp_rom.max():>12.4f}  "
              f"{error_rel:>10.4f}  {rmse:>8.4f}")

    return results


# =======================================================================
#  4. FIGURA CLAVE
# =======================================================================

def plot_error_vs_theta(
    gpr_data: dict,
    test_results: dict,
    out_path: Path,
    gpr_data_24: dict | None = None,
):
    """
    Error relativo ROM-11 en los 13 angulos de test vs theta.
    Muestra el error concentrado en la zona del vortice de Kawai.
    """
    angles_test  = sorted(test_results)
    errors_test  = [test_results[a]["error_rel"] for a in angles_test]
    angles_train = list(gpr_data["angles"])

    # LOO medio del ROM-11 en r*
    sweep_r    = gpr_data["sweep_r"]
    sweep_tot  = gpr_data["sweep_mean_total"]
    r_star     = int(gpr_data["r_star"])
    idx_star   = int(np.searchsorted(sweep_r, r_star))
    loo_11_mean = float(sweep_tot[idx_star])

    fig, ax = plt.subplots(figsize=(11, 5.5))

    # Zona vortice de Kawai
    ax.axvspan(
        VORTEX_RANGE[0], VORTEX_RANGE[1],
        alpha=0.10, color="#d62728",
        label=f"Zona vortice Kawai [{VORTEX_RANGE[0]:.0f}°–{VORTEX_RANGE[1]:.0f}°]",
    )

    # Linea de referencia LOO-11
    ax.axhline(loo_11_mean, color="#1f4e79", ls="--", lw=1.6,
               label=f"LOO medio ROM-11 = {loo_11_mean:.3f}")

    # Linea de referencia LOO-24 (si disponible)
    if gpr_data_24 is not None:
        sr24  = gpr_data_24["sweep_r"]
        st24  = gpr_data_24["sweep_mean_total"]
        r24   = int(gpr_data_24["r_star"])
        i24   = int(np.searchsorted(sr24, r24))
        loo24 = float(st24[i24])
        ax.axhline(loo24, color="#70ad47", ls="--", lw=1.6,
                   label=f"LOO medio ROM-24 = {loo24:.3f}")

    # Marcas de angulos de entrenamiento en el eje
    ax.scatter(
        angles_train, np.zeros(len(angles_train)),
        marker="|", s=180, linewidths=1.8,
        color="#1f4e79", zorder=5, clip_on=False,
        label=f"Entrenamiento ROM-11 ({len(angles_train)} angulos)",
    )

    # Errores de test
    ax.plot(angles_test, errors_test, "-", color="#c00000", lw=1.2, alpha=0.5)
    sc = ax.scatter(angles_test, errors_test,
                    marker="s", s=85, color="#c00000",
                    zorder=6, label="Test externo ROM-11 (13 intermedios)")

    # Anotaciones de valor sobre cada punto
    for theta, err in zip(angles_test, errors_test):
        ax.annotate(
            f"{err:.3f}", (theta, err),
            textcoords="offset points", xytext=(0, 8),
            ha="center", fontsize=7.5, color="#c00000",
        )

    ax.set_xlabel("Angulo de incidencia $\\theta$ (deg)", fontsize=11)
    ax.set_ylabel(
        r"Error relativo  $\|Cp_\mathrm{CFD} - Cp_\mathrm{ROM}\|_2 \;/\; \|Cp_\mathrm{CFD} - \bar{Cp}\|_2$",
        fontsize=10,
    )
    ax.set_title(
        f"Test externo ROM-11: insuficiencia del muestreo a 5°\n"
        f"(r*={r_star}, kernel={str(gpr_data['best_kernel'])}, "
        f"N_train={len(angles_train)} angulos)",
        fontweight="bold", fontsize=11,
    )
    ax.set_xlim(-2, 52)
    ax.set_ylim(bottom=0)
    ax.set_xticks(range(0, 55, 5))
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  -> {out_path.name}")


# =======================================================================
#  5. CSV RESUMEN
# =======================================================================

def save_csv(
    gpr_data: dict,
    test_results: dict,
    gpr_data_24: dict | None,
    out_path: Path,
):
    r_star   = int(gpr_data["r_star"])
    kernel   = str(gpr_data["best_kernel"])
    sweep_r  = gpr_data["sweep_r"]
    sweep_t  = gpr_data["sweep_mean_total"]
    loo_11   = float(sweep_t[int(np.searchsorted(sweep_r, r_star))])

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([f"# ROM-11: r*={r_star}, kernel={kernel}, LOO_medio={loo_11:.4f}"])
        if gpr_data_24 is not None:
            r24    = int(gpr_data_24["r_star"])
            k24    = str(gpr_data_24["best_kernel"])
            sr24   = gpr_data_24["sweep_r"]
            st24   = gpr_data_24["sweep_mean_total"]
            loo24  = float(st24[int(np.searchsorted(sr24, r24))])
            w.writerow([f"# ROM-24: r*={r24}, kernel={k24}, LOO_medio={loo24:.4f}"])
        w.writerow([])
        w.writerow(["theta_test", "cp_max_cfd", "cp_max_rom",
                    "cp_min_cfd", "cp_min_rom", "error_rel", "rmse"])
        for theta in sorted(test_results):
            r = test_results[theta]
            w.writerow([
                f"{theta:.2f}",
                f"{r['cp_max_cfd']:.4f}", f"{r['cp_max_rom']:.4f}",
                f"{r['cp_min_cfd']:.4f}", f"{r['cp_min_rom']:.4f}",
                f"{r['error_rel']:.4f}",  f"{r['rmse']:.4f}",
            ])

    print(f"  -> {out_path.name}")


# =======================================================================
#  MAIN
# =======================================================================

def main():
    p = argparse.ArgumentParser(description="Test externo ROM-11 en angulos intermedios")
    p.add_argument("--suffix", default="_11ang",
                   help="Sufijo del modelo ROM-11 (default: _11ang)")
    args   = p.parse_args()
    suffix = args.suffix

    print(f"\n{'='*65}")
    print(f"  ROM Test Externo — ROM-11 en angulos intermedios")
    print(f"  Modelo: ROM_GPR_results{suffix}.npz")
    print(f"{'='*65}\n")

    # 1. Cargar modelo ROM-11
    gpr_data     = load_gpr_model(suffix)
    angles_train = list(gpr_data["angles"])
    r_star       = int(gpr_data["r_star"])
    kernel       = str(gpr_data["best_kernel"])

    print(f"  ROM-11: r*={r_star}, kernel={kernel}, N_train={len(angles_train)}")
    print(f"  Angulos entrenamiento: {[int(a) if a == int(a) else a for a in angles_train]}")

    # 2. Re-entrenar GPRs
    gprs = retrain_gprs(gpr_data)

    # 3. Descubrir todos los angulos disponibles e identificar los de test
    angles_all  = discover_angles(DATA_DIR)
    test_angles = find_test_angles(angles_all, angles_train)

    print(f"\n  Angulos disponibles total : {len(angles_all)}")
    print(f"  Angulos de test (intermedios): {len(test_angles)}")
    print(f"  {sorted(test_angles)}")

    # 4. Calcular errores de test
    print(f"\n{'='*65}")
    print(f"  Prediccion y error en angulos intermedios")
    print(f"{'='*65}")
    test_results = compute_test_errors(gpr_data, gprs, test_angles)

    if not test_results:
        print("  ERROR: No se encontraron ficheros CFD para los angulos de test.")
        return

    # 5. Cargar ROM-24 para linea de referencia (si existe)
    gpr_data_24 = None
    for candidate in ["ROM_GPR_results_24ang.npz", "ROM_GPR_results.npz"]:
        path_24 = SCRIPT_DIR / candidate
        if path_24.is_file():
            try:
                raw = np.load(str(path_24), allow_pickle=False)
                gpr_data_24 = {k: raw[k] for k in raw.files}
                print(f"\n  ROM-24 cargado ({candidate}): r*={int(gpr_data_24['r_star'])}")
            except Exception as e:
                print(f"\n  AVISO: no se pudo cargar {candidate}: {e}")
            break

    # 6. Figura clave
    print(f"\n{'='*65}")
    print(f"  Generando figuras...")
    print(f"{'='*65}")
    plot_error_vs_theta(
        gpr_data, test_results,
        SCRIPT_DIR / f"GPR_error_vs_theta{suffix}.png",
        gpr_data_24=gpr_data_24,
    )

    # 7. CSV resumen
    save_csv(
        gpr_data, test_results, gpr_data_24,
        SCRIPT_DIR / "ROM_comparison_summary.csv",
    )

    # 8. Resumen en consola
    errors_list  = [v["error_rel"] for v in test_results.values()]
    mean_err     = float(np.mean(errors_list))
    max_theta    = max(test_results, key=lambda t: test_results[t]["error_rel"])
    max_err      = test_results[max_theta]["error_rel"]

    # Error medio en zona del vortice vs fuera
    err_vortex   = [test_results[t]["error_rel"] for t in test_results
                    if VORTEX_RANGE[0] <= t <= VORTEX_RANGE[1]]
    err_outside  = [test_results[t]["error_rel"] for t in test_results
                    if not (VORTEX_RANGE[0] <= t <= VORTEX_RANGE[1])]

    sweep_r  = gpr_data["sweep_r"]
    loo_11   = float(gpr_data["sweep_mean_total"][int(np.searchsorted(sweep_r, r_star))])

    print(f"\n{'='*65}")
    print(f"  RESUMEN TEST EXTERNO ROM-11")
    print(f"{'='*65}")
    print(f"  N angulos test        : {len(test_results)}")
    print(f"  Error medio (todos)   : {mean_err:.4f}")
    print(f"  Error medio (vortice) : {np.mean(err_vortex):.4f}  "
          f"(theta en [{VORTEX_RANGE[0]:.0f},{VORTEX_RANGE[1]:.0f}] deg)")
    print(f"  Error medio (fuera)   : {np.mean(err_outside):.4f}")
    print(f"  Error maximo          : {max_err:.4f}  (theta = {max_theta} deg)")
    print(f"  LOO medio ROM-11      : {loo_11:.4f}  (entrenamiento interno)")
    if gpr_data_24 is not None:
        sr24   = gpr_data_24["sweep_r"]
        r24    = int(gpr_data_24["r_star"])
        loo_24 = float(gpr_data_24["sweep_mean_total"][int(np.searchsorted(sr24, r24))])
        print(f"  LOO medio ROM-24      : {loo_24:.4f}  (referencia)")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
