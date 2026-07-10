"""
ROM_comparativa.py
==================
Cruza los resultados de las 3 bases (base_ref, base_kawai, base_full) y genera:
  - outputs/comparativa/error_vs_theta_comparativa.png  <- figura clave
  - outputs/comparativa/resumen_comparativo.csv         <- tabla para la memoria

Asume que los 3 ROMs ya estan entrenados:
  outputs/base_ref/ROM_GPR_results.npz
  outputs/base_kawai/ROM_GPR_results.npz
  outputs/base_full/ROM_GPR_results.npz

Uso:
  python ROM_comparativa.py
"""

from __future__ import annotations

import csv
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor

from ROM_POD import (
    DATA_DIR, MAX_THETA, N_PROBES, Q_REF, REF_TICKS,
    _parse_probe_file, discover_angles, fmt_angle,
)
from ROM_GPR import make_kernel, N_RESTARTS, THETA_SCALE

SCRIPT_DIR   = Path(__file__).resolve().parent
OUT_DIR      = SCRIPT_DIR / "outputs" / "comparativa"
VORTEX_RANGE = (10.0, 30.0)

BASES = [
    ("base_ref",    "#1f4e79", "o", "Base ref (10 ang, paso 5°)"),
    ("base_kawai",  "#d62728", "s", "Base Kawai (17 ang, ref + zona [10-30°])"),
    ("base_kawai2", "#ff7f0e", "D", "Base Kawai2 (22 ang, paso 1.25° en zona)"),
    ("base_full",   "#2ca02c", "^", "Base full (22 ang, paso ~2.5°)"),
]


# =======================================================================
#  HELPERS
# =======================================================================

def load_gpr(base_name: str) -> dict | None:
    path = SCRIPT_DIR / "outputs" / base_name / "ROM_GPR_results.npz"
    if not path.is_file():
        print(f"  AVISO: no encontrado {path} — omitiendo '{base_name}'")
        return None
    raw = np.load(str(path), allow_pickle=False)
    return {k: raw[k] for k in raw.files}


def retrain_gprs(gpr_data: dict) -> list:
    A        = gpr_data["A"]
    angles   = gpr_data["angles"]
    r_star   = int(gpr_data["r_star"])
    kernel   = str(gpr_data["best_kernel"])
    theta_n  = (angles / THETA_SCALE).reshape(-1, 1)
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
            gpr.fit(theta_n, a_j)
        gprs.append(gpr)
    return gprs


def predict_cp(gpr_data: dict, gprs: list, theta: float) -> np.ndarray:
    mean  = gpr_data["mean"]
    Phi   = gpr_data["Phi"]
    r     = int(gpr_data["r_star"])
    th_n  = np.array([[theta / THETA_SCALE]])
    alpha = np.array([float(gprs[j].predict(th_n)) for j in range(r)])
    return mean + Phi @ alpha


def compute_errors(gpr_data: dict, gprs: list) -> dict:
    """
    Evalua el ROM en todos los angulos disponibles <=45 NO en su training set.
    Devuelve dict theta -> dict con err_rel y rmse.
    """
    angles_all = discover_angles(DATA_DIR)
    train_set  = set(gpr_data["angles"])
    test_angs  = [a for a in angles_all if a <= MAX_THETA and a not in train_set]

    mean_train = gpr_data["mean"]
    results    = {}

    for theta in test_angs:
        fpath = (DATA_DIR / f"ROMCase_theta_{fmt_angle(theta)}"
                 / "postProcessing" / "probes_buildingPressure" / "0" / "p")
        if not fpath.is_file():
            continue
        _, _, p_vals = _parse_probe_file(fpath)
        cp_cfd  = p_vals / Q_REF
        cp_rom  = predict_cp(gpr_data, gprs, theta)
        denom   = np.linalg.norm(cp_cfd - mean_train)
        if denom < 1e-12:
            continue
        results[theta] = {
            "err_rel": float(np.linalg.norm(cp_cfd - cp_rom) / denom),
            "rmse":    float(np.sqrt(np.mean((cp_cfd - cp_rom) ** 2))),
        }

    return results


def loo_mean(gpr_data: dict) -> float:
    sr  = gpr_data["sweep_r"]
    st  = gpr_data["sweep_mean_total"]
    r   = int(gpr_data["r_star"])
    return float(st[int(np.searchsorted(sr, r))])


# =======================================================================
#  FIGURA COMPARATIVA
# =======================================================================

def plot_comparativa(base_results: dict, out_path: Path):
    """
    3 curvas de error test (una por base) vs theta.
    Ejes limpios con ticks de referencia (REF_TICKS).
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.axvspan(VORTEX_RANGE[0], VORTEX_RANGE[1], alpha=0.09, color="#d62728",
               label=f"Zona vortice Kawai [{VORTEX_RANGE[0]:.0f}–{VORTEX_RANGE[1]:.0f}°]")

    for base_name, color, marker, label in BASES:
        if base_name not in base_results:
            continue
        bd = base_results[base_name]
        test_res = bd["test_results"]

        if not test_res:
            # base_full: sin test — solo mostrar LOO como linea horizontal
            ax.axhline(bd["loo"], color=color, ls=":", lw=1.5,
                       label=f"{label} | LOO={bd['loo']:.3f} (sin test externo)")
            continue

        thetas = sorted(test_res)
        errs   = [test_res[t]["err_rel"] for t in thetas]
        loo    = bd["loo"]

        # Conectar solo test consecutivos: un hueco >10 deg significa que ahi
        # no hay medida (los angulos estan en training) y la linea enganharia
        seg_x, seg_y = [thetas[0]], [errs[0]]
        for x, y in zip(thetas[1:], errs[1:]):
            if x - seg_x[-1] > 10.0:
                ax.plot(seg_x, seg_y, "-", color=color, lw=1.3, alpha=0.6)
                seg_x, seg_y = [], []
            seg_x.append(x)
            seg_y.append(y)
        ax.plot(seg_x, seg_y, "-", color=color, lw=1.3, alpha=0.6)
        ax.scatter(thetas, errs, marker=marker, s=70, color=color, zorder=5,
                   label=f"{label} | LOO={loo:.3f}")

        # Marcar angulos de entrenamiento en el eje
        train_ang = bd["angles_train"]
        ax.scatter(train_ang, np.zeros(len(train_ang)),
                   marker="|", s=120, linewidths=1.5, color=color,
                   zorder=4, clip_on=False, alpha=0.5)

    ax.set_xlabel("Angulo de incidencia θ (deg)", fontsize=11)
    ax.set_ylabel(
        r"Error relativo  $\|Cp_{CFD} - Cp_{ROM}\| / \|Cp_{CFD} - \bar{Cp}\|$",
        fontsize=10,
    )
    ax.set_title(
        "Comparativa de estrategias de muestreo (test externo por base)\n"
        "(Muestreo adaptativo: densificar solo la zona del vortice de Kawai)",
        fontweight="bold", fontsize=11,
    )
    ax.set_xlim(-2, MAX_THETA + 2)
    ax.set_ylim(bottom=0)
    ax.set_xticks(REF_TICKS)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out_path.relative_to(SCRIPT_DIR)}")


# =======================================================================
#  CSV RESUMEN
# =======================================================================

def save_resumen(base_results: dict, out_path: Path):
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["base", "n_angulos", "r_star", "kernel",
                    "LOO_medio", "n_test_kawai", "err_test_kawai",
                    "n_test_fuera", "err_test_fuera"])

        for base_name, _, _, _ in BASES:
            if base_name not in base_results:
                continue
            bd       = base_results[base_name]
            test_res = bd["test_results"]

            e_kaw = [test_res[t]["err_rel"] for t in test_res
                     if VORTEX_RANGE[0] <= t <= VORTEX_RANGE[1]]
            e_out = [test_res[t]["err_rel"] for t in test_res
                     if not (VORTEX_RANGE[0] <= t <= VORTEX_RANGE[1])]

            w.writerow([
                base_name,
                bd["n_train"],
                bd["r_star"],
                bd["kernel"],
                f"{bd['loo']:.4f}",
                len(e_kaw),
                f"{np.mean(e_kaw):.4f}" if e_kaw else "nan",
                len(e_out),
                f"{np.mean(e_out):.4f}" if e_out else "nan",
            ])

    print(f"  -> {out_path.relative_to(SCRIPT_DIR)}")


# =======================================================================
#  MAIN
# =======================================================================

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*65}")
    print(f"  ROM Comparativa: base_ref vs base_kawai vs base_full")
    print(f"{'='*65}\n")

    base_results = {}

    for base_name, color, marker, label in BASES:
        print(f"\n--- {label} ---")
        gpr_data = load_gpr(base_name)
        if gpr_data is None:
            continue

        r_star  = int(gpr_data["r_star"])
        kernel  = str(gpr_data["best_kernel"])
        n_train = len(gpr_data["angles"])
        print(f"  r*={r_star}, kernel={kernel}, N_train={n_train}")

        print("  Re-entrenando GPRs...", end=" ", flush=True)
        gprs = retrain_gprs(gpr_data)
        print("listo")

        print("  Calculando errores de test...")
        test_res = compute_errors(gpr_data, gprs)
        print(f"  N test: {len(test_res)}")

        base_results[base_name] = {
            "gpr_data":     gpr_data,
            "test_results": test_res,
            "loo":          loo_mean(gpr_data),
            "r_star":       r_star,
            "kernel":       kernel,
            "n_train":      n_train,
            "angles_train": list(gpr_data["angles"]),
        }

    if not base_results:
        print("\nERROR: Ninguna base encontrada. Ejecuta primero ROM_POD.py + ROM_GPR.py para cada base.")
        return

    # Tabla resumen en consola
    print(f"\n{'='*65}")
    print(f"  RESUMEN COMPARATIVO")
    print(f"{'='*65}")
    print(f"  {'Base':<14} {'N':>4} {'r*':>4} {'Kernel':<10} {'LOO':>7} "
          f"{'Err_Kawai':>10} {'Err_fuera':>10}")
    print(f"  {'-'*14} {'-'*4} {'-'*4} {'-'*10} {'-'*7} {'-'*10} {'-'*10}")
    for base_name, _, _, _ in BASES:
        if base_name not in base_results:
            continue
        bd  = base_results[base_name]
        tr  = bd["test_results"]
        ek  = [tr[t]["err_rel"] for t in tr if VORTEX_RANGE[0] <= t <= VORTEX_RANGE[1]]
        eo  = [tr[t]["err_rel"] for t in tr if not (VORTEX_RANGE[0] <= t <= VORTEX_RANGE[1])]
        ek_s = f"{np.mean(ek):.4f}" if ek else "  (no test)"
        eo_s = f"{np.mean(eo):.4f}" if eo else "  (no test)"
        print(f"  {base_name:<14} {bd['n_train']:>4} {bd['r_star']:>4} "
              f"{bd['kernel']:<10} {bd['loo']:>7.4f} {ek_s:>10} {eo_s:>10}")
    print(f"{'='*65}\n")

    # Figuras y CSV
    print(f"  Generando salidas en {OUT_DIR.relative_to(SCRIPT_DIR)} ...")
    plot_comparativa(base_results, OUT_DIR / "error_vs_theta_comparativa.png")
    save_resumen(base_results, OUT_DIR / "resumen_comparativo.csv")

    print(f"\n  Listo.")


if __name__ == "__main__":
    main()
