"""
ROM_holdout_eval.py
===================
Evaluacion de las 4 bases (base_ref, base_kawai, base_kawai2, base_full)
sobre el holdout pre-registrado (BRIEF_holdout.md): 9 angulos que ninguna
base ha visto jamas en entrenamiento, elegidos ANTES de correr este script.

Metricas pre-registradas (BRIEF_holdout.md S4), calculadas por base y por
angulo de holdout:
  1. RMSE global (absoluto, 400 sondas)
  2. Error relativo con denominador COMUN: ||Cp_cfd - Cp_rom|| / ||Cp_cfd - mean_27||
     mean_27 = media de los 27 snapshots de entrenamiento disponibles
     (identica para las 4 bases -> comparacion sin trampas de escala)
  3. Error en el pico de succion |Cp_min_cfd - Cp_min_rom|

Agregacion: media global, media por estrato (zona vortice [10,30] vs
fuera), y comparaciones PAREADAS por angulo entre bases (test de signos).

Salidas -> outputs/holdout/:
  holdout_results.csv     (largo: base, theta, rmse, err_rel_common, err_cpmin, zona)
  holdout_summary.csv     (por base: medias global/zona/fuera)
  holdout_pairwise.csv    (por par de bases y metrica: victorias/9, p-valor signo)
  error_vs_theta_holdout.png

Requiere: las 4 bases ya entrenadas en outputs/<base>/ROM_GPR_results.npz,
y los 9 casos CFD del holdout ya simulados y post-procesados.

Uso:
  python ROM_holdout_eval.py
"""

from __future__ import annotations

import csv
import warnings
from itertools import combinations
from math import comb
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor

from ROM_POD import (
    ANGLES, DATA_DIR, HOLDOUT_SET, MAX_THETA, Q_REF, REF_TICKS,
    _parse_probe_file, fmt_angle, load_openfoam_snapshots,
)
from ROM_GPR import make_kernel, N_RESTARTS, THETA_SCALE

SCRIPT_DIR   = Path(__file__).resolve().parent
OUT_DIR      = SCRIPT_DIR / "outputs" / "holdout"
VORTEX_RANGE = (10.0, 30.0)

BASES = [
    ("base_ref",    "#1f4e79", "o", "Base ref (10 ang, paso 5°)"),
    ("base_kawai",  "#d62728", "s", "Base Kawai (17 ang, ref + zona [10-30°])"),
    ("base_kawai2", "#ff7f0e", "D", "Base Kawai2 (22 ang, paso 1.25° en zona)"),
    ("base_full",   "#2ca02c", "^", "Base full (22 ang, paso ~2.5°)"),
]

METRICS = ["rmse", "err_rel_common", "err_cpmin"]
METRIC_LABELS = {
    "rmse":           "RMSE global",
    "err_rel_common": "Error relativo (denom. comun)",
    "err_cpmin":      "Error en Cp_min",
}


# =======================================================================
#  CARGA Y RE-ENTRENAMIENTO (mismo patron que ROM_comparativa.py)
# =======================================================================

def load_gpr(base_name: str) -> dict | None:
    path = SCRIPT_DIR / "outputs" / base_name / "ROM_GPR_results.npz"
    if not path.is_file():
        print(f"  AVISO: no encontrado {path} — omitiendo '{base_name}'")
        return None
    raw = np.load(str(path), allow_pickle=False)
    return {k: raw[k] for k in raw.files}


def retrain_gprs(gpr_data: dict) -> list:
    A       = gpr_data["A"]
    angles  = gpr_data["angles"]
    r_star  = int(gpr_data["r_star"])
    kernel  = str(gpr_data["best_kernel"])
    theta_n = (angles / THETA_SCALE).reshape(-1, 1)
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
    mean = gpr_data["mean"]
    Phi  = gpr_data["Phi"]
    r    = int(gpr_data["r_star"])
    th_n = np.array([[theta / THETA_SCALE]])
    alpha = np.array([float(gprs[j].predict(th_n)) for j in range(r)])
    return mean + Phi @ alpha


def load_cp_holdout(theta: float) -> np.ndarray:
    fpath = (DATA_DIR / f"ROMCase_theta_{fmt_angle(theta)}"
             / "postProcessing" / "probes_buildingPressure" / "0" / "p")
    if not fpath.is_file():
        raise FileNotFoundError(f"No encontrado: {fpath}")
    _, _, p_vals = _parse_probe_file(fpath)
    return p_vals / Q_REF


# =======================================================================
#  EVALUACION POR BASE
# =======================================================================

def evaluate_base(base_name: str, gpr_data: dict, mean_27: np.ndarray,
                   holdout_angles: list[float]) -> dict:
    r_star = int(gpr_data["r_star"])
    kernel = str(gpr_data["best_kernel"])
    n_train = len(gpr_data["angles"])
    print(f"  {base_name}: r*={r_star}, kernel={kernel}, N_train={n_train}")

    print("    Re-entrenando GPRs...", end=" ", flush=True)
    gprs = retrain_gprs(gpr_data)
    print("listo")

    per_theta = {}
    for theta in holdout_angles:
        cp_cfd = load_cp_holdout(theta)
        cp_rom = predict_cp(gpr_data, gprs, theta)
        denom  = np.linalg.norm(cp_cfd - mean_27)

        rmse   = float(np.sqrt(np.mean((cp_cfd - cp_rom) ** 2)))
        err_rc = float(np.linalg.norm(cp_cfd - cp_rom) / denom) if denom > 1e-12 else float("nan")
        err_cm = float(abs(cp_cfd.min() - cp_rom.min()))

        per_theta[theta] = {"rmse": rmse, "err_rel_common": err_rc, "err_cpmin": err_cm}

    return {
        "base": base_name, "r_star": r_star, "kernel": kernel, "n_train": n_train,
        "per_theta": per_theta,
    }


# =======================================================================
#  AGREGACION Y COMPARACIONES PAREADAS
# =======================================================================

def sign_test_pvalue(k: int, n: int) -> float:
    """P-valor unilateral exacto de que >=k de n exitos ocurran por azar (p=0.5)."""
    return sum(comb(n, i) for i in range(k, n + 1)) / (2 ** n)


def summarize(base_results: dict, holdout_angles: list[float]) -> list[dict]:
    rows = []
    for base_name, _, _, _ in BASES:
        if base_name not in base_results:
            continue
        per_theta = base_results[base_name]["per_theta"]
        for metric in METRICS:
            vals_all  = [per_theta[t][metric] for t in holdout_angles]
            vals_zone = [per_theta[t][metric] for t in holdout_angles
                        if VORTEX_RANGE[0] <= t <= VORTEX_RANGE[1]]
            vals_out  = [per_theta[t][metric] for t in holdout_angles
                        if not (VORTEX_RANGE[0] <= t <= VORTEX_RANGE[1])]
            rows.append({
                "base": base_name, "metric": metric,
                "mean_global": float(np.mean(vals_all)),
                "mean_zona":   float(np.mean(vals_zone)) if vals_zone else float("nan"),
                "mean_fuera":  float(np.mean(vals_out)) if vals_out else float("nan"),
            })
    return rows


def pairwise_comparisons(base_results: dict, holdout_angles: list[float]) -> list[dict]:
    rows = []
    available = [b for b, _, _, _ in BASES if b in base_results]
    for base_a, base_b in combinations(available, 2):
        pt_a = base_results[base_a]["per_theta"]
        pt_b = base_results[base_b]["per_theta"]
        for metric in METRICS:
            wins_a = sum(1 for t in holdout_angles if pt_a[t][metric] < pt_b[t][metric])
            n = len(holdout_angles)
            p_a = sign_test_pvalue(wins_a, n)
            p_b = sign_test_pvalue(n - wins_a, n)
            rows.append({
                "base_a": base_a, "base_b": base_b, "metric": metric,
                "n": n, "wins_a": wins_a, "wins_b": n - wins_a,
                "p_a_better": f"{p_a:.4f}", "p_b_better": f"{p_b:.4f}",
            })
    return rows


# =======================================================================
#  FIGURA
# =======================================================================

def plot_error_vs_theta(base_results: dict, holdout_angles: list[float],
                        metric: str, out_path: Path):
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.axvspan(VORTEX_RANGE[0], VORTEX_RANGE[1], alpha=0.09, color="#d62728",
               label=f"Zona vortice Kawai [{VORTEX_RANGE[0]:.0f}–{VORTEX_RANGE[1]:.0f}°]")

    for base_name, color, marker, label in BASES:
        if base_name not in base_results:
            continue
        per_theta = base_results[base_name]["per_theta"]
        vals = [per_theta[t][metric] for t in holdout_angles]
        ax.plot(holdout_angles, vals, "-", color=color, lw=1.2, alpha=0.6)
        ax.scatter(holdout_angles, vals, marker=marker, s=90, color=color,
                   zorder=5, label=label)

    ax.set_xlabel("Angulo de incidencia θ (deg)", fontsize=11)
    ax.set_ylabel(METRIC_LABELS[metric], fontsize=11)
    ax.set_title(
        f"Holdout pre-registrado (N=9): {METRIC_LABELS[metric]} por base\n"
        "(ningun angulo de este conjunto se uso en ningun entrenamiento)",
        fontweight="bold", fontsize=11,
    )
    ax.set_xticks(sorted(set(REF_TICKS) | {int(t) for t in holdout_angles}))
    ax.set_xlim(-2, 47)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out_path.relative_to(SCRIPT_DIR)}")


# =======================================================================
#  MAIN
# =======================================================================

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    holdout_angles = sorted(HOLDOUT_SET)

    print(f"\n{'='*65}")
    print(f"  Evaluacion holdout pre-registrado (N={len(holdout_angles)})")
    print(f"  Angulos: {holdout_angles}")
    print(f"{'='*65}\n")

    angles_27 = [a for a in ANGLES if a <= MAX_THETA]
    print(f"  Cargando los {len(angles_27)} snapshots de entrenamiento <=45° (denominador comun)...")
    data_27 = load_openfoam_snapshots(angles_27)
    mean_27 = data_27["X"].mean(axis=1)
    print(f"  mean_27 calculada sobre {len(angles_27)} angulos.\n")

    base_results = {}
    for base_name, _, _, _ in BASES:
        gpr_data = load_gpr(base_name)
        if gpr_data is None:
            continue
        base_results[base_name] = evaluate_base(base_name, gpr_data, mean_27, holdout_angles)

    if not base_results:
        print("ERROR: ninguna base disponible.")
        return

    # --- CSV largo ---
    results_csv = OUT_DIR / "holdout_results.csv"
    with open(results_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["base", "theta", "zona", "rmse", "err_rel_common", "err_cpmin"])
        for base_name, res in base_results.items():
            for theta in holdout_angles:
                pt = res["per_theta"][theta]
                zona = int(VORTEX_RANGE[0] <= theta <= VORTEX_RANGE[1])
                w.writerow([base_name, theta, zona,
                           f"{pt['rmse']:.5f}", f"{pt['err_rel_common']:.5f}",
                           f"{pt['err_cpmin']:.5f}"])
    print(f"\n  -> {results_csv.relative_to(SCRIPT_DIR)}")

    # --- Resumen ---
    summary_rows = summarize(base_results, holdout_angles)
    summary_csv = OUT_DIR / "holdout_summary.csv"
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["base", "metric", "mean_global", "mean_zona", "mean_fuera"])
        for row in summary_rows:
            w.writerow([row["base"], row["metric"], f"{row['mean_global']:.5f}",
                       f"{row['mean_zona']:.5f}", f"{row['mean_fuera']:.5f}"])
    print(f"  -> {summary_csv.relative_to(SCRIPT_DIR)}")

    print(f"\n{'='*65}")
    print(f"  RESUMEN (media global sobre los 9 angulos de holdout)")
    print(f"{'='*65}")
    for metric in METRICS:
        print(f"\n  {METRIC_LABELS[metric]}:")
        print(f"    {'Base':<14} {'Global':>9} {'Zona':>9} {'Fuera':>9}")
        for row in summary_rows:
            if row["metric"] != metric:
                continue
            print(f"    {row['base']:<14} {row['mean_global']:>9.4f} "
                  f"{row['mean_zona']:>9.4f} {row['mean_fuera']:>9.4f}")

    # --- Comparaciones pareadas ---
    pairwise_rows = pairwise_comparisons(base_results, holdout_angles)
    pairwise_csv = OUT_DIR / "holdout_pairwise.csv"
    with open(pairwise_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["base_a", "base_b", "metric", "n", "wins_a", "wins_b",
                   "p_a_better", "p_b_better"])
        for row in pairwise_rows:
            w.writerow([row["base_a"], row["base_b"], row["metric"], row["n"],
                       row["wins_a"], row["wins_b"], row["p_a_better"], row["p_b_better"]])
    print(f"\n  -> {pairwise_csv.relative_to(SCRIPT_DIR)}")

    print(f"\n{'='*65}")
    print(f"  COMPARACIONES PAREADAS (test de signos, n=9 por par)")
    print(f"{'='*65}")
    for row in pairwise_rows:
        if row["metric"] != "err_rel_common":
            continue
        print(f"  {row['base_a']:<12} vs {row['base_b']:<12}: "
              f"{row['wins_a']}/{row['n']} - {row['wins_b']}/{row['n']}  "
              f"(p_a={row['p_a_better']}, p_b={row['p_b_better']})")

    # --- Figuras ---
    print(f"\n{'='*65}")
    print("  Generando figuras...")
    print(f"{'='*65}")
    for metric in METRICS:
        plot_error_vs_theta(base_results, holdout_angles, metric,
                           OUT_DIR / f"error_vs_theta_holdout_{metric}.png")

    print("\n  Listo.\n")


if __name__ == "__main__":
    main()
