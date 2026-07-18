"""
ROM_gibbs_sensitivity.py
=========================
Fase 3 de BRIEF_claude_code_diagnostics.md: chequeos de sensibilidad para
blindar las conclusiones (a) Gibbs~=estacionario y (b) jitter negligible
de ROM_nonstationary_kernel.py.

Tres analisis, todos sobre los datos AUTOCONTENIDOS (outputs/base_full/
ROM_POD_basis.npz, igual que ROM_nonstationary_kernel.py):

  3.1 FORMA DEL KERNEL DE GIBBS. Repite el experimento (a) (TRAIN=16,
      TEST=6, mismo particionado que ROM_nonstationary_kernel.py)
      variando la campana fija a priori (mu en {15,20,25} grados, ancho
      en {5,8,12} grados: 9 combinaciones) para comprobar que beta->0 no
      es un artefacto de haber fijado la forma. Ademas ajusta un l(theta)
      LIBRE (spline log-lineal de 4 nudos en {0,15,30,45} grados, sin
      forma impuesta a priori) via la misma MLE.

  3.2 SENSIBILIDAD A r*. Repite (a) (campana por defecto MU=20, WID=8)
      con r* en {8, 12, 16}.

  3.3 TAMAÑO DEL TEST. En vez de la particion fija TEST=6, LOO completo
      sobre los 22 angulos de base_full (n=22 folds: cada angulo se deja
      fuera una vez, re-fit con los otros 21, r*=12, campana por
      defecto), con intervalo bootstrap sobre Gibbs-estacionario --
      con n=6 diferencias tan pequeñas (0.0140 vs 0.0140) estan dentro
      del ruido.

Salidas -> outputs/sensitivity/:
  shape_sweep.csv, free_ell.csv, rstar_sweep.csv, loo_n22.csv

Uso:  python ROM_gibbs_sensitivity.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
from numpy.linalg import cholesky, solve
from scipy.optimize import minimize

SCRIPT_DIR = Path(__file__).resolve().parent          # investigacion/
ROOT       = SCRIPT_DIR.parent                        # Programacion/
sys.path.insert(0, str(ROOT))

from ROM_nonstationary_kernel import RNG, RSTAR, TEST, VORTEX, fit_gp, predict   # noqa: E402

OUT_DIR    = ROOT / "outputs" / "sensitivity"

MU_GRID    = [15.0, 20.0, 25.0]
WID_GRID   = [5.0, 8.0, 12.0]
RSTAR_GRID = [8, 12, 16]
KNOTS_DEG  = np.array([0.0, 15.0, 30.0, 45.0])
KNOTS_NORM = KNOTS_DEG / 50.0


def load_selfcontained():
    d = np.load(str(ROOT / "outputs" / "base_full" / "ROM_POD_basis.npz"))
    mean, Phi_full, A_full = d["mean"], d["Phi"], d["A"]
    angles = np.round(d["angles"], 3)
    X = mean[:, None] + Phi_full @ A_full
    return X, angles


def repod(X, angles, train_angles, r):
    idx = [int(np.where(angles == x)[0][0]) for x in train_angles]
    xbar = X[:, idx].mean(axis=1)
    Xc = X[:, idx] - xbar[:, None]
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    r_eff = min(r, len(idx) - 1)
    Phi = U[:, :r_eff]
    A = Phi.T @ Xc
    return Phi, A, xbar, r_eff, idx


# =======================================================================
#  Kernel de Gibbs con (mu, ancho) LIBRES como argumento (no fijos a 20/8)
# =======================================================================

def bump_mw(theta, mu, wid):
    return np.exp(-((np.asarray(theta, float) - mu) / wid) ** 2)


def ell_mw(theta, l0, beta, mu, wid):
    return l0 * np.exp(-beta * bump_mw(theta, mu, wid))


def gibbs_K_mw(ta, tb, l0, beta, sf2, mu, wid):
    la = ell_mw(ta, l0, beta, mu, wid)[:, None]
    lb = ell_mw(tb, l0, beta, mu, wid)[None, :]
    d2 = (ta[:, None] - tb[None, :]) ** 2
    pre = np.sqrt(2.0 * la * lb / (la ** 2 + lb ** 2))
    return sf2 * pre * np.exp(-d2 / (la ** 2 + lb ** 2))


def nll_mw(params, th, y, mu, wid):
    l0 = np.exp(params[0]); sf2 = np.exp(params[1]); sn2 = np.exp(params[2])
    beta = np.log1p(np.exp(params[3]))  # beta>=0 libre (softplus); forma fija por mu/wid
    K = gibbs_K_mw(th, th, l0, beta, sf2, mu, wid) + sn2 * np.eye(len(th))
    try:
        L = cholesky(K + 1e-12 * np.eye(len(th)))
    except np.linalg.LinAlgError:
        return 1e10
    a = solve(L.T, solve(L, y))
    return float(0.5 * y @ a + np.log(np.diag(L)).sum() + 0.5 * len(th) * np.log(2 * np.pi))


def fit_gp_mw(th, y, mu, wid):
    yv = float(np.var(y))
    best = None
    for _ in range(12):
        p0 = np.array([np.log(0.2 + 0.3 * RNG.random()), np.log(max(yv, 1e-8)),
                       np.log(1e-6 * max(yv, 1e-8) * (1 + RNG.random())),
                       RNG.normal(-1, 1)])
        bnds = [(np.log(0.03), np.log(3)), (np.log(1e-8), np.log(1e3)),
                (np.log(1e-12), np.log(1e-1)), (-6, 4)]
        r = minimize(nll_mw, p0, args=(th, y, mu, wid), method="L-BFGS-B", bounds=bnds)
        if best is None or r.fun < best.fun:
            best = r
    p = best.x
    l0 = np.exp(p[0]); sf2 = np.exp(p[1]); sn2 = np.exp(p[2])
    beta = np.log1p(np.exp(p[3]))
    return dict(l0=l0, sf2=sf2, sn2=sn2, beta=beta, th=th, y=y, mu=mu, wid=wid)


def predict_mw(gp, tstar):
    th, y = gp["th"], gp["y"]
    K = (gibbs_K_mw(th, th, gp["l0"], gp["beta"], gp["sf2"], gp["mu"], gp["wid"])
         + gp["sn2"] * np.eye(len(th)))
    ks = gibbs_K_mw(np.atleast_1d(tstar), th, gp["l0"], gp["beta"], gp["sf2"], gp["mu"], gp["wid"])
    L = cholesky(K + 1e-12 * np.eye(len(th)))
    a = solve(L.T, solve(L, y))
    return float((ks @ a).ravel()[0])


# =======================================================================
#  l(theta) LIBRE: spline log-lineal de 4 nudos, sin forma impuesta
# =======================================================================

def ell_free(theta, log_l_knots):
    logl = np.interp(theta, KNOTS_NORM, log_l_knots)
    return np.exp(logl)


def gibbs_K_free(ta, tb, log_l_knots, sf2):
    la = ell_free(ta, log_l_knots)[:, None]
    lb = ell_free(tb, log_l_knots)[None, :]
    d2 = (ta[:, None] - tb[None, :]) ** 2
    pre = np.sqrt(2.0 * la * lb / (la ** 2 + lb ** 2))
    return sf2 * pre * np.exp(-d2 / (la ** 2 + lb ** 2))


def nll_free(params, th, y):
    log_l_knots = params[:4]
    sf2 = np.exp(params[4]); sn2 = np.exp(params[5])
    K = gibbs_K_free(th, th, log_l_knots, sf2) + sn2 * np.eye(len(th))
    try:
        L = cholesky(K + 1e-12 * np.eye(len(th)))
    except np.linalg.LinAlgError:
        return 1e10
    a = solve(L.T, solve(L, y))
    return float(0.5 * y @ a + np.log(np.diag(L)).sum() + 0.5 * len(th) * np.log(2 * np.pi))


def fit_gp_free(th, y):
    yv = float(np.var(y))
    best = None
    for _ in range(16):
        p0 = np.concatenate([
            RNG.normal(np.log(0.3), 0.5, size=4),
            [np.log(max(yv, 1e-8)), np.log(1e-6 * max(yv, 1e-8) * (1 + RNG.random()))],
        ])
        bnds = ([(np.log(0.03), np.log(3))] * 4
                + [(np.log(1e-8), np.log(1e3)), (np.log(1e-12), np.log(1e-1))])
        r = minimize(nll_free, p0, args=(th, y), method="L-BFGS-B", bounds=bnds)
        if best is None or r.fun < best.fun:
            best = r
    p = best.x
    return dict(log_l_knots=p[:4], sf2=np.exp(p[4]), sn2=np.exp(p[5]), th=th, y=y)


def predict_free(gp, tstar):
    th, y = gp["th"], gp["y"]
    K = gibbs_K_free(th, th, gp["log_l_knots"], gp["sf2"]) + gp["sn2"] * np.eye(len(th))
    ks = gibbs_K_free(np.atleast_1d(tstar), th, gp["log_l_knots"], gp["sf2"])
    L = cholesky(K + 1e-12 * np.eye(len(th)))
    a = solve(L.T, solve(L, y))
    return float((ks @ a).ravel()[0])


# =======================================================================
#  3.1 -- barrido de forma + l(theta) libre  (TRAIN=16, TEST=6, r*=12)
# =======================================================================

def section_31(X, angles):
    print("=" * 78)
    print("3.1 -- FORMA DEL KERNEL DE GIBBS (TRAIN=16, TEST=6, r*=12)")
    print("=" * 78)

    train_angles = [a for a in angles if a not in TEST]
    Phi, A, xbar, r, tr = repod(X, angles, train_angles, RSTAR)
    ti   = [int(np.where(angles == x)[0][0]) for x in TEST]
    zona = np.array([VORTEX[0] <= t <= VORTEX[1] for t in TEST])
    th_tr = angles[tr] / 50.0
    th_te = angles[ti] / 50.0

    def field_rmse(preds_per_mode):
        e = []
        for m, k in enumerate(ti):
            ah = np.array([preds_per_mode[j][m] for j in range(r)])
            e.append(np.sqrt(np.mean((X[:, k] - (xbar + Phi @ ah)) ** 2)))
        return np.array(e)

    # baseline estacionario (no depende de la forma de la campana)
    preds_stat = [[predict(fit_gp(th_tr, A[j, :], beta_free=False), t) for t in th_te]
                 for j in range(r)]
    e_stat = field_rmse(preds_stat)
    print(f"  Baseline estacionario: glob={e_stat.mean():.4f} "
          f"zona={e_stat[zona].mean():.4f} fuera={e_stat[~zona].mean():.4f}\n")

    rows = []
    print(f"  {'mu':>5}{'wid':>6}{'glob':>9}{'zona':>9}{'fuera':>9}{'mejora_glob':>13}")
    for mu in MU_GRID:
        for wid in WID_GRID:
            preds = [[predict_mw(fit_gp_mw(th_tr, A[j, :], mu, wid), t) for t in th_te]
                    for j in range(r)]
            e = field_rmse(preds)
            improve = 100 * (1 - e.mean() / e_stat.mean())
            print(f"  {mu:>5.0f}{wid:>6.0f}{e.mean():>9.4f}{e[zona].mean():>9.4f}"
                  f"{e[~zona].mean():>9.4f}{improve:>12.1f}%")
            rows.append(dict(mu=mu, wid=wid, glob=e.mean(), zona=e[zona].mean(),
                             fuera=e[~zona].mean(), mejora_glob_pct=improve))

    # l(theta) libre
    preds_free = [[predict_free(fit_gp_free(th_tr, A[j, :]), t) for t in th_te] for j in range(r)]
    e_free = field_rmse(preds_free)
    improve_free = 100 * (1 - e_free.mean() / e_stat.mean())
    print(f"\n  l(theta) LIBRE (spline 4 nudos): glob={e_free.mean():.4f} "
          f"zona={e_free[zona].mean():.4f} fuera={e_free[~zona].mean():.4f}  "
          f"mejora_glob={improve_free:+.1f}%")

    best = max(rows, key=lambda r_: r_["mejora_glob_pct"])
    peor_de_los_dos = max(best["mejora_glob_pct"], improve_free)
    print(f"\n  Mejor combinacion fija: mu={best['mu']:.0f} wid={best['wid']:.0f} "
          f"(mejora {best['mejora_glob_pct']:+.1f}% sobre estacionario)")
    if peor_de_los_dos < 5.0:
        print("  Ningun ajuste (forma fija variada, ni forma libre) bate al "
              "estacionario de forma relevante (umbral 5%).")
    else:
        print("  ATENCION: al menos un ajuste supera el umbral de 5% de mejora -- revisar.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "shape_sweep.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["mu", "wid", "glob", "zona", "fuera", "mejora_glob_pct"])
        w.writeheader()
        for r_ in rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r_.items()})
    print(f"  -> {OUT_DIR.relative_to(ROOT)}/shape_sweep.csv")

    with open(OUT_DIR / "free_ell.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "glob", "zona", "fuera", "mejora_glob_pct"])
        w.writerow(["stationary_baseline", f"{e_stat.mean():.4f}", f"{e_stat[zona].mean():.4f}",
                   f"{e_stat[~zona].mean():.4f}", "0.0"])
        w.writerow(["free_ell_4knots", f"{e_free.mean():.4f}", f"{e_free[zona].mean():.4f}",
                   f"{e_free[~zona].mean():.4f}", f"{improve_free:.1f}"])
    print(f"  -> {OUT_DIR.relative_to(ROOT)}/free_ell.csv\n")


# =======================================================================
#  3.2 -- sensibilidad a r*  (TRAIN=16, TEST=6, campana por defecto)
# =======================================================================

def section_32(X, angles):
    print("=" * 78)
    print("3.2 -- SENSIBILIDAD A r*  (TRAIN=16, TEST=6, mu=20 wid=8)")
    print("=" * 78)
    train_angles = [a for a in angles if a not in TEST]
    ti   = [int(np.where(angles == x)[0][0]) for x in TEST]
    zona = np.array([VORTEX[0] <= t <= VORTEX[1] for t in TEST])

    rows = []
    print(f"  {'r*':>4}{'stat_glob':>11}{'gibbs_glob':>12}{'stat_zona':>11}{'gibbs_zona':>12}")
    for r_req in RSTAR_GRID:
        Phi, A, xbar, r_eff, tr = repod(X, angles, train_angles, r_req)
        th_tr = angles[tr] / 50.0
        th_te = angles[ti] / 50.0

        def field_rmse(preds_per_mode, Phi=Phi, xbar=xbar, r_eff=r_eff):
            e = []
            for m, k in enumerate(ti):
                ah = np.array([preds_per_mode[j][m] for j in range(r_eff)])
                e.append(np.sqrt(np.mean((X[:, k] - (xbar + Phi @ ah)) ** 2)))
            return np.array(e)

        preds_s = [[predict(fit_gp(th_tr, A[j, :], beta_free=False), t) for t in th_te]
                  for j in range(r_eff)]
        preds_g = [[predict(fit_gp(th_tr, A[j, :], beta_free=True), t) for t in th_te]
                  for j in range(r_eff)]
        es, eg = field_rmse(preds_s), field_rmse(preds_g)
        print(f"  {r_eff:>4}{es.mean():>11.4f}{eg.mean():>12.4f}"
              f"{es[zona].mean():>11.4f}{eg[zona].mean():>12.4f}")
        rows.append(dict(r_star=r_eff, stat_glob=es.mean(), gibbs_glob=eg.mean(),
                         stat_zona=es[zona].mean(), gibbs_zona=eg[zona].mean(),
                         stat_fuera=es[~zona].mean(), gibbs_fuera=eg[~zona].mean()))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "rstar_sweep.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r_ in rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r_.items()})
    print(f"  -> {OUT_DIR.relative_to(ROOT)}/rstar_sweep.csv\n")


# =======================================================================
#  3.3 -- LOO completo n=22  (r*=12, campana por defecto)
# =======================================================================

def section_33(X, angles):
    print("=" * 78)
    print("3.3 -- LOO COMPLETO n=22  (r*=12, mu=20 wid=8)")
    print("=" * 78)
    zona_flag = np.array([VORTEX[0] <= t <= VORTEX[1] for t in angles])

    rows = []
    for k, theta_k in enumerate(angles):
        train_angles = [a for i, a in enumerate(angles) if i != k]
        Phi, A, xbar, r_eff, tr = repod(X, angles, train_angles, RSTAR)
        th_tr = angles[tr] / 50.0
        t_te  = theta_k / 50.0

        alpha_s = np.array([predict(fit_gp(th_tr, A[j, :], beta_free=False), t_te) for j in range(r_eff)])
        alpha_g = np.array([predict(fit_gp(th_tr, A[j, :], beta_free=True), t_te) for j in range(r_eff)])
        e_s = float(np.sqrt(np.mean((X[:, k] - (xbar + Phi @ alpha_s)) ** 2)))
        e_g = float(np.sqrt(np.mean((X[:, k] - (xbar + Phi @ alpha_g)) ** 2)))
        rows.append(dict(theta=float(theta_k), zona=int(zona_flag[k]),
                         rmse_stat=e_s, rmse_gibbs=e_g, diff=e_g - e_s))
        print(f"    theta={theta_k:>6.2f}  stat={e_s:.4f}  gibbs={e_g:.4f}  diff={e_g - e_s:+.4f}")

    diffs = np.array([r_["diff"] for r_ in rows])
    stats = np.array([r_["rmse_stat"] for r_ in rows])
    gibbs = np.array([r_["rmse_gibbs"] for r_ in rows])
    zona_mask = np.array([bool(r_["zona"]) for r_ in rows])

    n_boot = 20000
    rng = np.random.default_rng(1)
    boot_means = np.array([rng.choice(diffs, size=len(diffs), replace=True).mean()
                           for _ in range(n_boot)])
    ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])

    print(f"\n  n={len(rows)}  RMSE medio estacionario={stats.mean():.4f}  Gibbs={gibbs.mean():.4f}")
    print(f"  Diferencia media (Gibbs-estacionario) = {diffs.mean():+.5f}  "
          f"IC95% bootstrap=[{ci_lo:+.5f}, {ci_hi:+.5f}]")
    print(f"  Zona: stat={stats[zona_mask].mean():.4f} gibbs={gibbs[zona_mask].mean():.4f} | "
          f"Fuera: stat={stats[~zona_mask].mean():.4f} gibbs={gibbs[~zona_mask].mean():.4f}")
    zero_in_ci = ci_lo <= 0 <= ci_hi
    print(f"  El 0 {'ESTA' if zero_in_ci else 'NO esta'} dentro del IC95% -> "
          f"{'no hay evidencia de diferencia' if zero_in_ci else 'hay diferencia significativa'}.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "loo_n22.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["theta", "zona", "rmse_stat", "rmse_gibbs", "diff"])
        w.writeheader()
        for r_ in rows:
            w.writerow({k: (f"{v:.5f}" if isinstance(v, float) else v) for k, v in r_.items()})
        w.writerow({})
        w.writerow(dict(theta="mean_diff", zona="", rmse_stat=f"{stats.mean():.5f}",
                        rmse_gibbs=f"{gibbs.mean():.5f}", diff=f"{diffs.mean():.5f}"))
        w.writerow(dict(theta="ci95_lo", zona="", rmse_stat="", rmse_gibbs="", diff=f"{ci_lo:.5f}"))
        w.writerow(dict(theta="ci95_hi", zona="", rmse_stat="", rmse_gibbs="", diff=f"{ci_hi:.5f}"))
    print(f"  -> {OUT_DIR.relative_to(ROOT)}/loo_n22.csv\n")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    X, angles = load_selfcontained()
    section_31(X, angles)
    section_32(X, angles)
    section_33(X, angles)
    print("Listo.")


if __name__ == "__main__":
    main()
