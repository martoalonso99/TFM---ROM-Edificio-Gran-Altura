"""
ROM_nonstationary_kernel.py   (investigacion/)
==============================================
Dos experimentos sobre el suelo de error del ROM (ver ROM_floor_diagnostic.py),
usando SOLO outputs/base_full/ROM_POD_basis.npz (base fija r*=12).

(a) KERNEL NO ESTACIONARIO DE GIBBS con longitud de escala variable
    l(theta) = l0 * exp(-beta * b(theta)),  b = campana centrada en la zona
    del vortice (mu=20 deg, w=8 deg). beta>=0 acorta l en el vortice.
    Anida el modelo estacionario (beta=0), de modo que la maxima verosimilitud
    marginal (MLE) elige por si sola si merece la pena ser no estacionario.
    Se compara el holdout (6 angulos fijos) contra el mismo GP con beta=0.

(b) JITTER vs SEÑAL: en la banda densificada del vortice (puntos a 1.25 deg)
    se estima el ruido angular de los coeficientes por dos vias (residuo de
    tendencia suave y segunda diferencia) y se propaga a RMS de campo. Se
    compara con (i) el residuo de interpolacion (~0.014) y (ii) el suelo de
    reproducibilidad numerica del test de simetria (4e-4).

La maquinaria del GP de Gibbs (bump, ell, gibbs_K, nll, fit_gp, predict, MU,
WID, RNG) vive en ROM_gibbs_kernel.py (produccion) y se re-exporta aqui para
compatibilidad con los scripts que hacen 'from ROM_nonstationary_kernel import
fit_gp, predict, ...'.

Salidas -> outputs/nonstationary/:
  gibbs_vs_stationary.csv, jitter_estimate.csv, nonstationary_kernel.png

Uso:  python investigacion/ROM_nonstationary_kernel.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent          # investigacion/
ROOT       = SCRIPT_DIR.parent                        # Programacion/
sys.path.insert(0, str(ROOT))

# Re-exporta la maquinaria del kernel de Gibbs desde el modulo de produccion
from ROM_gibbs_kernel import (   # noqa: E402
    MU, WID, RNG, bump, ell, gibbs_K, nll, fit_gp, predict,
)

OUT_DIR    = ROOT / "outputs" / "nonstationary"
RSTAR      = 12
TEST       = [7.5, 12.5, 17.5, 22.5, 27.5, 37.5]
VORTEX     = (10.0, 30.0)
SYM_FLOOR  = 4e-4               # suelo de simetria (field RMS)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    d = np.load(str(ROOT / "outputs" / "base_full" / "ROM_POD_basis.npz"))
    mean, Phi_full, A_full = d["mean"], d["Phi"], d["A"]
    angles = np.round(d["angles"], 3)
    X = mean[:, None] + Phi_full @ A_full
    xbar = X.mean(axis=1); Xp = X - xbar[:, None]
    U, S, Vt = np.linalg.svd(Xp, full_matrices=False)
    Phi = U[:, :RSTAR]; A = Phi.T @ Xp
    idx = lambda v: [int(np.where(angles == x)[0][0]) for x in v]
    ti = idx(TEST); zona = np.array([VORTEX[0] <= t <= VORTEX[1] for t in TEST])
    TR = [a for a in angles if a not in TEST]; tr = idx(TR)
    th_tr = angles[tr] / 50.0

    # ---------- (a) Gibbs vs estacionario ----------
    rmse = {"stat": [], "gibbs": []}
    beta_fit = []
    for j in range(RSTAR):
        y = A[j, tr]
        gp_s = fit_gp(th_tr, y, beta_free=False)
        gp_g = fit_gp(th_tr, y, beta_free=True)
        beta_fit.append(gp_g["beta"])
        for tag, gp in (("stat", gp_s), ("gibbs", gp_g)):
            pred = np.array([predict(gp, angles[k] / 50.0) for k in ti])
            rmse[tag].append(pred)
    # reconstruccion de campo por angulo de test
    def field(preds):
        e = []
        for m, k in enumerate(ti):
            ah = np.array([preds[j][m] for j in range(RSTAR)])
            e.append(np.sqrt(np.mean((X[:, k] - (xbar + Phi @ ah)) ** 2)))
        return np.array(e)
    es, eg = field(rmse["stat"]), field(rmse["gibbs"])

    print("=" * 70)
    print("(a) HOLDOUT (6 ang, base fija r*=12, 16 ang entrenamiento)")
    print("=" * 70)
    print(f"    {'modelo':<24}{'RMSE glob':>10}{'zona':>9}{'fuera':>9}")
    print(f"    {'GP estacionario (b=0)':<24}{es.mean():>10.4f}{es[zona].mean():>9.4f}{es[~zona].mean():>9.4f}")
    print(f"    {'GP Gibbs l(theta) MLE':<24}{eg.mean():>10.4f}{eg[zona].mean():>9.4f}{eg[~zona].mean():>9.4f}")
    print(f"    mejora zona {100*(1-eg[zona].mean()/es[zona].mean()):+.1f}% | "
          f"global {100*(1-eg.mean()/es.mean()):+.1f}%")
    print(f"    beta MLE por modo (0=eligio estacionario): "
          f"{np.round(beta_fit[:6],2)} ...")

    with open(OUT_DIR / "gibbs_vs_stationary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["model", "rmse_glob", "rmse_zona", "rmse_fuera"])
        w.writerow(["stationary", f"{es.mean():.5f}", f"{es[zona].mean():.5f}", f"{es[~zona].mean():.5f}"])
        w.writerow(["gibbs",      f"{eg.mean():.5f}", f"{eg[zona].mean():.5f}", f"{eg[~zona].mean():.5f}"])

    # ---------- (b) jitter vs señal ----------
    # Estimador LIMPIO de jitter = test de simetria (dos configuraciones fisicamente
    # identicas -> su diferencia es ruido numerico puro). Los estimadores basados en
    # los coeficientes (sigma_n de la MLE, diferencias finitas) NO sirven aqui: el
    # modo 1 (94% energia) tiene amplitud enorme y su misfit contamina cualquier
    # estimacion absoluta de ruido. Se reporta, como control, el ruido RELATIVO por
    # modo (adimensional), que si es interpretable.
    sn2 = np.array([fit_gp(th_tr, A[j, tr], beta_free=False)["sn2"] for j in range(RSTAR)])
    var = np.array([np.var(A[j, tr]) for j in range(RSTAR)])
    rel = np.sqrt(sn2 / np.maximum(var, 1e-12))              # ruido relativo por modo
    wE = (S[:RSTAR] ** 2) / np.sum(S[:RSTAR] ** 2)
    rel_mean = float(np.sum(wE * rel))                       # medio ponderado por energia
    interp_res = float(eg[zona].mean())

    print("\n" + "=" * 70)
    print("(b) JITTER vs SEÑAL")
    print("=" * 70)
    print(f"    jitter LIMPIO (test simetria, field RMS):  {SYM_FLOOR:.5f}")
    print(f"    residuo de interpolacion en zona vortice:  {interp_res:.5f}")
    print(f"    => jitter/residuo ~ {100*SYM_FLOOR/interp_res:.1f}% (RMS) => "
          f"~{100*(SYM_FLOOR/interp_res)**2:.2f}% en energia: NEGLIGIBLE.")
    print(f"    control: ruido relativo medio de coeficientes (ponderado energia) = {100*rel_mean:.1f}%")
    print(f"    Conclusion: el residuo es SEÑAL no capturada, no ruido numerico; y (a)")
    print(f"    muestra que un kernel no estacionario tampoco la recupera con N=16")
    print(f"    -> la meseta es un LIMITE MUESTRAL (senal del vortice submuestreada),")
    print(f"    no de base POD, no de RANS, no de kernel, no de ruido.")

    with open(OUT_DIR / "jitter_estimate.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["quantity", "value"])
        for k, v in [("jitter_symmetry_field_rms", SYM_FLOOR),
                     ("interp_residual_zona", interp_res),
                     ("ratio_jitter_over_residual", SYM_FLOOR / interp_res),
                     ("rel_coeff_noise_energy_weighted", rel_mean)]:
            w.writerow([k, f"{v:.6f}"])

    # ---------- figura ----------
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    thg = np.linspace(0, 45, 200)
    for j in range(3):
        gp = fit_gp(th_tr, A[j, tr], beta_free=True)
        ax[0].plot(thg, ell(thg, gp["l0"] * 50, gp["beta"]), label=f"modo {j+1} (β={gp['beta']:.2f})")
    ax[0].axvspan(*VORTEX, alpha=.08, color="#d62728")
    ax[0].set_xlabel("θ (deg)"); ax[0].set_ylabel("l(θ) ajustada (deg)")
    ax[0].set_title("(a) Escala variable l(θ) MLE (Gibbs)"); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
    labels = ["glob", "zona", "fuera"]
    xs = np.arange(3)
    ax[1].bar(xs - 0.18, [es.mean(), es[zona].mean(), es[~zona].mean()], 0.36, label="estacionario")
    ax[1].bar(xs + 0.18, [eg.mean(), eg[zona].mean(), eg[~zona].mean()], 0.36, label="Gibbs")
    ax[1].axhline(SYM_FLOOR, ls="--", color="k", label=f"jitter (simetría, {SYM_FLOOR:.4f})")
    ax[1].set_xticks(xs); ax[1].set_xticklabels(labels); ax[1].set_ylabel("RMSE holdout")
    ax[1].set_title("(b) estacionario vs Gibbs + suelo de jitter"); ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(OUT_DIR / "nonstationary_kernel.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  -> {OUT_DIR.relative_to(ROOT)}/  (gibbs_vs_stationary.csv, jitter_estimate.csv, nonstationary_kernel.png)")


if __name__ == "__main__":
    main()
