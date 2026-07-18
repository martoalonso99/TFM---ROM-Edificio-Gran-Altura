"""
ROM_test_intermediate.py
========================
Evalua el ROM entrenado (desde --outdir) en los angulos intermedios disponibles
que NO estan en su base de entrenamiento.

Para base_full no hay angulos de test externos; el script lo reporta y termina.
Para base_ref y base_kawai genera error relativo por angulo y figura comparativa.

Uso:
  python ROM_test_intermediate.py --outdir outputs/base_ref
  python ROM_test_intermediate.py --outdir outputs/base_kawai
  python ROM_test_intermediate.py --outdir outputs/base_full
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ROM_POD import (
    DATA_DIR, MAX_THETA, N_PROBES, Q_REF, REF_TICKS,
    _parse_probe_file, discover_angles, fmt_angle,
)
from ROM_GPR import fit_mode_gp, THETA_SCALE

SCRIPT_DIR   = Path(__file__).resolve().parent
VORTEX_RANGE = (10.0, 30.0)


# =======================================================================
#  CARGA Y RE-ENTRENAMIENTO
# =======================================================================

def load_gpr_model(outdir: Path) -> dict:
    path = outdir / "ROM_GPR_results.npz"
    if not path.is_file():
        raise FileNotFoundError(
            f"No existe {path}.\n"
            f"Ejecuta: python ROM_GPR.py --outdir {outdir.relative_to(SCRIPT_DIR)}"
        )
    raw = np.load(str(path), allow_pickle=False)
    return {k: raw[k] for k in raw.files}


def retrain_gprs(gpr_data: dict) -> list:
    A        = gpr_data["A"]
    angles   = gpr_data["angles"]
    r_star   = int(gpr_data["r_star"])
    kernel   = str(gpr_data["best_kernel"])
    theta_norm = (angles / THETA_SCALE).reshape(-1, 1)

    print(f"  Re-entrenando {r_star} GPRs ({kernel}) sobre {len(angles)} puntos...")
    gprs = []
    for j in range(r_star):
        gprs.append(fit_mode_gp(kernel, theta_norm, A[j, :]))
    print("  Listo.")
    return gprs


def predict_cp(gpr_data: dict, gprs: list, theta_test: float) -> np.ndarray:
    mean   = gpr_data["mean"]
    Phi    = gpr_data["Phi"]
    r_star = int(gpr_data["r_star"])
    th_n   = np.array([[theta_test / THETA_SCALE]])
    alpha  = np.array([float(gprs[j].predict(th_n)) for j in range(r_star)])
    return mean + Phi @ alpha


# =======================================================================
#  IDENTIFICACION DE ANGULOS DE TEST
# =======================================================================

def find_test_angles(gpr_data: dict) -> list[float]:
    """Angulos disponibles <=MAX_THETA no incluidos en el set de entrenamiento."""
    angles_all  = discover_angles(DATA_DIR)
    train_set   = set(gpr_data["angles"])
    return [a for a in angles_all if a <= MAX_THETA and a not in train_set]


# =======================================================================
#  EVALUACION DE ERRORES
# =======================================================================

def compute_test_errors(
    gpr_data: dict,
    gprs: list,
    test_angles: list[float],
) -> dict:
    mean_train = gpr_data["mean"]
    results    = {}

    print(f"\n  {'theta':>7}  {'Cp_max_CFD':>11}  {'Cp_max_ROM':>11}  {'err_rel':>8}  {'RMSE':>8}")
    print(f"  {'-'*7}  {'-'*11}  {'-'*11}  {'-'*8}  {'-'*8}")

    for theta in test_angles:
        fpath = (DATA_DIR / f"ROMCase_theta_{fmt_angle(theta)}"
                 / "postProcessing" / "probes_buildingPressure" / "0" / "p")
        if not fpath.is_file():
            print(f"  {theta:>7.2f}  [no encontrado]")
            continue

        _, _, p_vals = _parse_probe_file(fpath)
        cp_cfd  = p_vals / Q_REF
        cp_rom  = predict_cp(gpr_data, gprs, theta)

        denom = np.linalg.norm(cp_cfd - mean_train)
        if denom < 1e-12:
            continue
        err_rel = float(np.linalg.norm(cp_cfd - cp_rom) / denom)
        rmse    = float(np.sqrt(np.mean((cp_cfd - cp_rom) ** 2)))

        results[theta] = {
            "cp_cfd": cp_cfd, "cp_rom": cp_rom,
            "err_rel": err_rel, "rmse": rmse,
            "cp_max_cfd": float(cp_cfd.max()), "cp_max_rom": float(cp_rom.max()),
            "cp_min_cfd": float(cp_cfd.min()), "cp_min_rom": float(cp_rom.min()),
        }
        print(f"  {theta:>7.2f}  {cp_cfd.max():>11.4f}  {cp_rom.max():>11.4f}"
              f"  {err_rel:>8.4f}  {rmse:>8.4f}")

    return results


# =======================================================================
#  FIGURA
# =======================================================================

def plot_error_vs_theta(
    gpr_data: dict,
    test_results: dict,
    out_path: Path,
    base_label: str = "",
    ref_data: dict | None = None,
):
    angles_test = sorted(test_results)
    errors_test = [test_results[a]["err_rel"] for a in angles_test]
    angles_train = list(gpr_data["angles"])

    sweep_r    = gpr_data["sweep_r"]
    sweep_tot  = gpr_data["sweep_mean_total"]
    r_star     = int(gpr_data["r_star"])
    loo_mean   = float(sweep_tot[int(np.searchsorted(sweep_r, r_star))])

    fig, ax = plt.subplots(figsize=(11, 5.5))

    ax.axvspan(VORTEX_RANGE[0], VORTEX_RANGE[1], alpha=0.10, color="#d62728",
               label=f"Zona vortice Kawai [{VORTEX_RANGE[0]:.0f}–{VORTEX_RANGE[1]:.0f}°]")

    ax.axhline(loo_mean, color="#1f4e79", ls="--", lw=1.6,
               label=f"LOO medio {base_label} = {loo_mean:.3f}")

    if ref_data is not None:
        sr2 = ref_data["sweep_r"]
        st2 = ref_data["sweep_mean_total"]
        r2  = int(ref_data["r_star"])
        l2  = float(st2[int(np.searchsorted(sr2, r2))])
        ax.axhline(l2, color="#70ad47", ls="--", lw=1.4,
                   label=f"LOO medio referencia = {l2:.3f}")

    ax.scatter(angles_train, np.zeros(len(angles_train)),
               marker="|", s=180, linewidths=1.8, color="#1f4e79",
               zorder=5, clip_on=False,
               label=f"Entrenamiento {base_label} ({len(angles_train)} ang.)")

    ax.plot(angles_test, errors_test, "-", color="#c00000", lw=1.2, alpha=0.5)
    ax.scatter(angles_test, errors_test, marker="s", s=85,
               color="#c00000", zorder=6,
               label=f"Test externo {base_label} ({len(angles_test)} ang.)")

    for theta, err in zip(angles_test, errors_test):
        ax.annotate(f"{err:.3f}", (theta, err),
                    textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=7.5, color="#c00000")

    ax.set_xlabel("Angulo de incidencia θ (deg)", fontsize=11)
    ax.set_ylabel(
        r"Error relativo  $\|Cp_{CFD} - Cp_{ROM}\| / \|Cp_{CFD} - \bar{Cp}\|$",
        fontsize=10,
    )
    ax.set_title(
        f"Test externo {base_label}  (r*={r_star}, kernel={str(gpr_data['best_kernel'])}, "
        f"N_train={len(angles_train)})",
        fontweight="bold", fontsize=11,
    )
    ax.set_xlim(-2, MAX_THETA + 2)
    ax.set_ylim(bottom=0)
    ax.set_xticks(REF_TICKS)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  -> {out_path.name}")


# =======================================================================
#  CSV
# =======================================================================

def save_csv(gpr_data: dict, test_results: dict, out_path: Path):
    r_star  = int(gpr_data["r_star"])
    kernel  = str(gpr_data["best_kernel"])
    sr      = gpr_data["sweep_r"]
    loo     = float(gpr_data["sweep_mean_total"][int(np.searchsorted(sr, r_star))])

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([f"# r*={r_star}, kernel={kernel}, LOO_medio={loo:.4f}"])
        w.writerow([])
        w.writerow(["theta_test", "cp_max_cfd", "cp_max_rom",
                    "cp_min_cfd", "cp_min_rom", "err_rel", "rmse"])
        for theta in sorted(test_results):
            r = test_results[theta]
            w.writerow([f"{theta:.2f}", f"{r['cp_max_cfd']:.4f}", f"{r['cp_max_rom']:.4f}",
                        f"{r['cp_min_cfd']:.4f}", f"{r['cp_min_rom']:.4f}",
                        f"{r['err_rel']:.4f}", f"{r['rmse']:.4f}"])
    print(f"  -> {out_path.name}")


# =======================================================================
#  MAIN
# =======================================================================

def main(argv=None):
    p = argparse.ArgumentParser(description="Test externo ROM en angulos intermedios")
    p.add_argument("--outdir", default="outputs/base_ref",
                   help="Directorio del modelo ROM (default: outputs/base_ref)")
    args   = p.parse_args(argv)
    outdir = SCRIPT_DIR / args.outdir

    print(f"\n{'='*65}")
    print(f"  ROM Test Externo")
    print(f"  Modelo: {outdir.relative_to(SCRIPT_DIR) / 'ROM_GPR_results.npz'}")
    print(f"{'='*65}\n")

    gpr_data     = load_gpr_model(outdir)
    angles_train = list(gpr_data["angles"])
    base_label   = outdir.name

    print(f"  Base '{base_label}': r*={int(gpr_data['r_star'])}, "
          f"kernel={str(gpr_data['best_kernel'])}, N_train={len(angles_train)}")

    test_angles = find_test_angles(gpr_data)
    if not test_angles:
        print(f"\n  Sin angulos de test externos (todos los disponibles <=45 ya estan "
              f"en la base '{base_label}'). Solo LOO disponible.")
        return

    print(f"\n  Angulos de test: {len(test_angles)} -> {sorted(test_angles)}")
    gprs = retrain_gprs(gpr_data)

    print(f"\n{'='*65}")
    print(f"  Prediccion en angulos intermedios")
    print(f"{'='*65}")
    test_results = compute_test_errors(gpr_data, gprs, test_angles)

    if not test_results:
        print("  ERROR: No se encontraron ficheros CFD para los angulos de test.")
        return

    # Figura
    print(f"\n{'='*65}")
    print(f"  Generando figuras...")
    print(f"{'='*65}")
    plot_error_vs_theta(
        gpr_data, test_results,
        outdir / "GPR_error_vs_theta.png",
        base_label=base_label,
    )

    # CSV
    save_csv(gpr_data, test_results, outdir / "test_errors.csv")

    # Resumen
    errs   = [v["err_rel"] for v in test_results.values()]
    e_vort = [test_results[t]["err_rel"] for t in test_results
               if VORTEX_RANGE[0] <= t <= VORTEX_RANGE[1]]
    e_out  = [test_results[t]["err_rel"] for t in test_results
               if not (VORTEX_RANGE[0] <= t <= VORTEX_RANGE[1])]
    sr     = gpr_data["sweep_r"]
    loo    = float(gpr_data["sweep_mean_total"][
                   int(np.searchsorted(sr, int(gpr_data["r_star"])))])
    max_t  = max(test_results, key=lambda t: test_results[t]["err_rel"])

    print(f"\n{'='*65}")
    print(f"  RESUMEN  '{base_label}'")
    print(f"{'='*65}")
    print(f"  N test          : {len(test_results)}")
    print(f"  Error medio     : {np.mean(errs):.4f}")
    if e_vort:
        print(f"  Error zona Kawai: {np.mean(e_vort):.4f}  ({len(e_vort)} ang. en [{VORTEX_RANGE[0]:.0f},{VORTEX_RANGE[1]:.0f}]°)")
    if e_out:
        print(f"  Error fuera     : {np.mean(e_out):.4f}  ({len(e_out)} ang.)")
    print(f"  Error maximo    : {test_results[max_t]['err_rel']:.4f}  (theta={max_t}°)")
    print(f"  LOO medio       : {loo:.4f}")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
