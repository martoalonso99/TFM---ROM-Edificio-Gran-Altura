"""
ROM_gibbs_kernel.py
===================
Kernel de Gibbs (GP no estacionario) compartido entre produccion e investigacion.

Maquinaria pura del proceso gaussiano de Gibbs con longitud de escala variable

    l(theta) = l0 * exp(-beta * bump(theta)),   bump = campana en la zona del vortice

que anida el modelo estacionario (beta = 0). La verosimilitud marginal (MLE)
elige por si sola si merece la pena ser no estacionario. Se implementa a mano
(numpy + scipy) porque scikit-learn no admite length_scale no estacionario.

Lo usan:
  - ROM_GPR.py (produccion): "Gibbs" como candidato mas en compare_kernels(),
    envuelto en GibbsModeGP para exponer la misma interfaz que un
    GaussianProcessRegressor de scikit-learn (.predict(X, return_std=...)).
  - investigacion/ROM_nonstationary_kernel.py y sus dependientes, que
    re-exportan fit_gp / predict / RNG / MU / WID desde aqui.

La clase GibbsModeGP centra y (normalize_y, como hace sklearn) para que la
comparacion de kernels en produccion sea a igualdad de preprocesado; las
funciones sueltas fit_gp / predict NO centran, para reproducir exactamente el
analisis de investigacion ya validado.
"""

from __future__ import annotations

import numpy as np
from numpy.linalg import cholesky, solve
from scipy.optimize import minimize

# Campana del vortice (grados) — fija por diseno del experimento no estacionario
MU, WID = 20.0, 8.0
# RNG seeded compartido: reproducibilidad identica a la corrida de investigacion
RNG = np.random.default_rng(0)


# =======================================================================
#  MATEMATICA DEL KERNEL DE GIBBS
# =======================================================================

def bump(theta):
    return np.exp(-((np.asarray(theta, float) - MU) / WID) ** 2)


def ell(theta, l0, beta):
    return l0 * np.exp(-beta * bump(theta))


def gibbs_K(ta, tb, l0, beta, sf2):
    la = ell(ta, l0, beta)[:, None]
    lb = ell(tb, l0, beta)[None, :]
    d2 = (ta[:, None] - tb[None, :]) ** 2
    pre = np.sqrt(2.0 * la * lb / (la ** 2 + lb ** 2))
    return sf2 * pre * np.exp(-d2 / (la ** 2 + lb ** 2))


def nll(params, th, y, beta_free):
    l0 = np.exp(params[0]); sf2 = np.exp(params[1]); sn2 = np.exp(params[2])
    beta = (np.log1p(np.exp(params[3])) if beta_free else 0.0)   # softplus >= 0
    K = gibbs_K(th, th, l0, beta, sf2) + sn2 * np.eye(len(th))
    try:
        L = cholesky(K + 1e-12 * np.eye(len(th)))
    except np.linalg.LinAlgError:
        return 1e10
    a = solve(L.T, solve(L, y))
    return float(0.5 * y @ a + np.log(np.diag(L)).sum() + 0.5 * len(th) * np.log(2 * np.pi))


def fit_gp(th, y, beta_free, n_restarts: int = 12):
    """Ajusta el GP de Gibbs por MLE con multistart. th, y son 1-D."""
    yv = float(np.var(y))
    best = None
    for _ in range(n_restarts):
        p0 = np.array([np.log(0.2 + 0.3 * RNG.random()),                     # l0 (en unidades theta/50)
                       np.log(max(yv, 1e-8)),                                # sf2
                       np.log(1e-6 * max(yv, 1e-8) * (1 + RNG.random())),    # sn2
                       RNG.normal(-1, 1)])                                   # beta pre-softplus
        bnds = [(np.log(0.03), np.log(3)), (np.log(1e-8), np.log(1e3)),
                (np.log(1e-12), np.log(1e-1)), (-6, 4)]
        r = minimize(nll, p0, args=(th, y, beta_free), method="L-BFGS-B", bounds=bnds)
        if best is None or r.fun < best.fun:
            best = r
    p = best.x
    l0 = np.exp(p[0]); sf2 = np.exp(p[1]); sn2 = np.exp(p[2])
    beta = (np.log1p(np.exp(p[3])) if beta_free else 0.0)
    return dict(l0=l0, sf2=sf2, sn2=sn2, beta=beta, th=th, y=y)


def predict(gp, tstar):
    """Media posterior en un unico tstar (escalar). Devuelve float."""
    th, y = gp["th"], gp["y"]
    K = gibbs_K(th, th, gp["l0"], gp["beta"], gp["sf2"]) + gp["sn2"] * np.eye(len(th))
    ks = gibbs_K(np.atleast_1d(tstar), th, gp["l0"], gp["beta"], gp["sf2"])
    L = cholesky(K + 1e-12 * np.eye(len(th)))
    a = solve(L.T, solve(L, y))
    return float((ks @ a).ravel()[0])


def predict_std(gp, tstar):
    """Desviacion estandar posterior en tstar (escalar). Devuelve float."""
    th = gp["th"]
    K = gibbs_K(th, th, gp["l0"], gp["beta"], gp["sf2"]) + gp["sn2"] * np.eye(len(th))
    L = cholesky(K + 1e-12 * np.eye(len(th)))
    ks = gibbs_K(np.atleast_1d(tstar), th, gp["l0"], gp["beta"], gp["sf2"])  # (1, N)
    v = solve(L, ks.ravel())
    kss = float(gp["sf2"])            # gibbs_K(t*, t*) = sf2 (pre=1, exp(0)=1)
    var = max(kss - float(v @ v), 0.0)
    return float(np.sqrt(var))


# =======================================================================
#  ADAPTADOR ESTILO SCIKIT-LEARN (para usar Gibbs como un kernel mas)
# =======================================================================

class GibbsModeGP:
    """
    Envuelve un GP de Gibbs ajustado exponiendo la interfaz de un
    GaussianProcessRegressor de scikit-learn:  .predict(X, return_std=False).

    Centra y (normalize_y) para igualar el preprocesado de los kernels
    estacionarios en la comparacion de produccion.
    """

    def __init__(self, gp: dict, y_mean: float, beta: float):
        self._gp = gp
        self._y_mean = y_mean
        self.beta = beta          # β MLE del modo (0 = eligio estacionario)

    def predict(self, X, return_std: bool = False):
        th = np.asarray(X, float).ravel()
        mu = np.array([predict(self._gp, t) for t in th]) + self._y_mean
        if not return_std:
            return mu
        std = np.array([predict_std(self._gp, t) for t in th])
        return mu, std

    def describe(self) -> str:
        g = self._gp
        return (f"Gibbs(l0={g['l0']:.3g}, beta={g['beta']:.3g}, "
                f"sf2={g['sf2']:.3g}, sn2={g['sn2']:.2g})")


def fit_gibbs_mode(th2d, y, n_restarts: int = 8, beta_free: bool = True) -> GibbsModeGP:
    """
    Ajusta un GP de Gibbs para un modo POD con interfaz sklearn.
    th2d : (N, 1) angulos normalizados theta/THETA_SCALE (como en ROM_GPR).
    """
    th = np.asarray(th2d, float).ravel()
    y = np.asarray(y, float).ravel()
    y_mean = float(y.mean())
    gp = fit_gp(th, y - y_mean, beta_free=beta_free, n_restarts=n_restarts)
    return GibbsModeGP(gp, y_mean, gp["beta"])
