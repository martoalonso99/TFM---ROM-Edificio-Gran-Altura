"""
ROM_POD.py
==========

Lee directamente los ficheros de probes de OpenFOAM (antes via MATLAB),
construye la matriz de snapshots Cp[400 x N] y realiza la descomposicion
POD via SVD. Integra lo que antes hacia ROM_PostProc_Batch.m.

Pipeline:
  OpenFOAM p files  ->  Cp_matrix[400xN]  ->  POD (SVD)  ->  ROM_POD_basis.npz

Salidas (en Programacion/):
  - ROM_POD_basis.npz            : base POD para ROM_GPR.py
  - snapshot_overview.png        : mapa de calor + perfiles Cp por angulo
  - POD_singular_spectrum.png    : espectro y energia acumulada
  - POD_modal_coefficients.png   : a_j(theta) por modo
  - POD_modes_unfolded.png       : modos espaciales sobre las 4 caras
  - POD_loo_projection_error.png : cota inferior del error LOO-CV

Notas:
  - p en OpenFOAM incompresible es presion cinematica [m^2/s^2].
    Cp = p / (0.5 * Uref^2) directamente, sin dividir por rho.
  - Los angulos se descubren automaticamente desde carpetas ROMCase_theta_NN.
  - N = len(ANGLES) snapshots -> rango efectivo <= N-1 tras centrado.
  - Distribucion uniforme de probes -> POD sin ponderacion de areas es valida.

Autor: Martín Rodríguez García
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# =======================================================================
#  CONFIGURACION
# =======================================================================

SCRIPT_DIR = Path(__file__).resolve().parent                # TFM/Programacion/
DATA_DIR   = SCRIPT_DIR.parent / "Simulaciones" / "ROM"    # TFM/Simulaciones/ROM/
OUT_NPZ    = SCRIPT_DIR / "ROM_POD_basis.npz"

# Fisica
UREF      = 11.0                  # m/s  - velocidad de referencia a z=H
Q_REF     = 0.5 * UREF ** 2      # m^2/s^2 - presion dinamica cinematica
N_PROBES  = 400


def discover_angles(data_dir: Path) -> list[float]:
    """
    Descubre los angulos ROM escaneando carpetas ROMCase_theta_* en data_dir.
    Soporta nombres enteros (theta_10) y decimales (theta_2p5, 'p' = punto decimal).
    Devuelve la lista ordenada de angulos (float) encontrados.
    """
    pattern = re.compile(r'^ROMCase_theta_([\dp]+)$')
    angles = []
    for entry in data_dir.iterdir():
        if entry.is_dir():
            m = pattern.match(entry.name)
            if m:
                angles.append(float(m.group(1).replace('p', '.')))
    if not angles:
        raise FileNotFoundError(
            f"No se encontraron carpetas ROMCase_theta_* en {data_dir}"
        )
    return sorted(angles)


def fmt_angle(theta: float) -> str:
    """Convierte angulo a string de nombre de carpeta. 2.5 -> '2p5', 10.0 -> '10'."""
    if theta == int(theta):
        return f"{int(theta):02d}"
    return str(theta).replace(".", "p")


ANGLES = discover_angles(DATA_DIR)

# POD
ENERGY_THRESHOLDS = [0.90, 0.95, 0.99, 0.999]
N_MODES_PLOT      = 6

# Geometria de caras (para visualizacion)
N_COLS_FACE   = 5
N_ROWS_FACE   = 20
N_PROBES_FACE = N_COLS_FACE * N_ROWS_FACE   # 100 por cara
N_FACES       = 4

REF_TICKS = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]   # ticks de referencia para ejes de angulo
MAX_THETA = 45.0                                        # rango fundamental seccion cuadrada (C4)

BASE_REF   = {0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0}
BASE_KAWAI = BASE_REF | {12.5, 17.5, 22.5, 23.75, 26.25, 27.5, 28.75}
# BASE_FULL: todos los angulos descubiertos <= MAX_THETA (calculado en main)


# =======================================================================
#  1. LECTURA DIRECTA DE OpenFOAM
# =======================================================================

def _parse_probe_file(fpath: Path) -> tuple[np.ndarray, float, np.ndarray]:
    """
    Parsea un fichero p de OpenFOAM (probes function object).

    Formato cabecera (una linea por probe):
        # Probe 0 (x y z)
        # Probe 1 (x y z)
        ...
        # Time

    Formato datos (una linea por iteracion SIMPLE):
        iteracion  p_0  p_1  ...  p_399

    Lee la ultima linea de datos = ultimo timestep convergido.

    Returns
    -------
    coords     : (N_PROBES, 3)  coordenadas xyz de cada probe
    t_final    : float           numero de iteracion del ultimo timestep
    p_vals     : (N_PROBES,)    presion cinematica [m^2/s^2] del ultimo timestep
    """
    coords         = np.zeros((N_PROBES, 3))
    last_data_line = None
    probe_re       = re.compile(r'#\s*Probe\s+(\d+)\s+\(([^)]+)\)')

    with open(fpath, "r") as fh:
        for line in fh:
            line_s = line.strip()
            if not line_s:
                continue
            if line_s.startswith("#"):
                m = probe_re.match(line_s)
                if m:
                    idx  = int(m.group(1))
                    vals = [float(v) for v in m.group(2).split()]
                    if idx < N_PROBES and len(vals) == 3:
                        coords[idx] = vals
            else:
                last_data_line = line_s

    if last_data_line is None:
        raise ValueError(f"No se encontraron datos en {fpath}")

    raw = np.fromstring(last_data_line, sep=" ")
    if len(raw) < N_PROBES + 1:
        raise ValueError(
            f"Columnas inesperadas ({len(raw)}) en {fpath.parent.parent.parent.parent.parent.name}"
        )

    return coords, float(raw[0]), raw[1 : N_PROBES + 1]


def load_openfoam_snapshots(angles: list | None = None) -> dict:
    """
    Lee los ficheros p de OpenFOAM para los casos ROM y construye
    la matriz de snapshots Cp[400 x N].

    Estructura de ficheros esperada:
        DATA_DIR/ROMCase_theta_XX/postProcessing/probes_buildingPressure/0/p

    Replica la logica de ROM_PostProc_Batch.m (MATLAB).
    """
    if angles is None:
        angles = ANGLES
    N_ANG        = len(angles)
    Cp_matrix    = np.zeros((N_PROBES, N_ANG))
    probe_coords = np.zeros((N_PROBES, 3, N_ANG))

    print(f"\n{'='*65}")
    print(f"  ROM PostProc  |  Uref = {UREF:.1f} m/s  |  q_ref = {Q_REF:.2f} m^2/s^2")
    print(f"{'='*65}")
    print(f"  {'Caso':<16} {'t_final':>8} {'Cp_max':>8} {'Cp_min':>8} {'Cp_medio':>10}")
    print(f"  {'-'*54}")

    for i, theta in enumerate(angles):
        tag   = f"theta_{fmt_angle(theta)}"
        fpath = (DATA_DIR / f"ROMCase_{tag}"
                 / "postProcessing" / "probes_buildingPressure" / "0" / "p")

        if not fpath.is_file():
            raise FileNotFoundError(
                f"No encontrado: {fpath}\n"
                f"Verifica que DATA_DIR apunta a la carpeta ROM correcta."
            )

        coords, t_final, p_vals = _parse_probe_file(fpath)
        probe_coords[:, :, i]   = coords
        Cp_matrix[:, i]         = p_vals / Q_REF

        cp = Cp_matrix[:, i]
        print(f"  {tag:<16} {t_final:>8.0f} {cp.max():>8.3f} {cp.min():>8.3f} {cp.mean():>10.3f}")

    print(f"  {'='*54}")

    # Validacion basica: Cp_max en theta=0 deberia ser ~0.94 (validado vs TPU)
    cp_max_ref = Cp_matrix[:, 0].max()
    print(f"\n  Check theta=0: Cp_max = {cp_max_ref:.3f}  (ref TPU: ~0.94)")
    if abs(cp_max_ref - 0.94) > 0.10:
        print("  AVISO: Cp_max fuera del rango esperado. Revisar Uref o datos.")
    print()

    return {
        "X":            Cp_matrix,
        "angles":       np.array(angles, dtype=float),
        "probe_coords": probe_coords,
        "Uref":         UREF,
        "q_ref":        Q_REF,
    }


# =======================================================================
#  2. POD via SVD CENTRADO
# =======================================================================

def pod_svd(X: np.ndarray) -> dict:
    """
    POD via SVD economico sobre la matriz centrada.

    Parameters
    ----------
    X : (N_probes, N_snap) array
        Cada columna es un snapshot Cp(theta_k).

    Returns
    -------
    dict con:
        mean       : (N_probes,)      media empirica sobre snapshots
        Phi        : (N_probes, r)    modos POD espaciales (columnas ortonormales)
        sigma      : (r,)             valores singulares decrecientes
        Vt         : (r, N_snap)      vectores singulares derechos transpuestos
        A          : (r, N_snap)      coeficientes modales = diag(sigma) @ Vt
        energy     : (r,)             fraccion de varianza de cada modo
        cum_energy : (r,)             energia acumulada
    """
    N_probes, N_snap = X.shape

    mean       = X.mean(axis=1)
    X_centered = X - mean[:, None]

    U, s, Vt = np.linalg.svd(X_centered, full_matrices=False)

    # Tras centrado, el ultimo valor singular es ~0 (rango efectivo <= N_snap-1)
    rank_eff = N_snap - 1
    Phi   = U[:, :rank_eff]
    sigma = s[:rank_eff]
    Vt    = Vt[:rank_eff, :]
    A     = sigma[:, None] * Vt     # (r, N_snap)

    energy     = sigma ** 2 / np.sum(sigma ** 2)
    cum_energy = np.cumsum(energy)

    return {
        "mean":       mean,
        "Phi":        Phi,
        "sigma":      sigma,
        "Vt":         Vt,
        "A":          A,
        "energy":     energy,
        "cum_energy": cum_energy,
    }


# =======================================================================
#  3. DIAGNOSTICO ENERGETICO
# =======================================================================

def report_energy(pod: dict, thresholds=ENERGY_THRESHOLDS) -> dict:
    print(f"{'='*65}")
    print("  Espectro POD y energia acumulada")
    print(f"{'='*65}")
    print(f"  {'Modo':>4}  {'sigma':>12}  {'energia':>10}  {'acumulada':>11}")
    print(f"  {'-'*4}  {'-'*12}  {'-'*10}  {'-'*11}")
    for j, (s, e, ce) in enumerate(
        zip(pod["sigma"], pod["energy"], pod["cum_energy"]), start=1
    ):
        print(f"  {j:>4d}  {s:>12.4e}  {e*100:>9.3f}%  {ce*100:>10.3f}%")
    print()

    r_eff = {}
    print("  r_eff por criterio energetico:")
    for thr in thresholds:
        idx = int(np.searchsorted(pod["cum_energy"], thr)) + 1
        r_eff[thr] = idx
        print(f"     E >= {thr*100:5.1f}%   =>   r = {idx}")
    print()
    return r_eff


# =======================================================================
#  4. LOO PROYECTIVO (cota inferior del error LOO-CV completo)
# =======================================================================

def loo_projection_error(X: np.ndarray, r_max: int | None = None) -> np.ndarray:
    """
    Para cada snapshot k: recalcula POD con los N-1 restantes, proyecta
    Cp_k sobre esa base y calcula el error relativo.

    Este es el limite inferior del error LOO-CV completo (asume GPR perfecto).
    La diferencia con el error real del ROM cuantifica el coste de interpolacion.

    Returns
    -------
    err : (N_snap, r_max) array
        err[k, r-1] = error relativo de proyeccion del snapshot k con r modos.
    """
    N_probes, N_snap = X.shape
    if r_max is None:
        r_max = N_snap - 2   # rango maximo con N-1 snapshots tras centrar

    err = np.full((N_snap, r_max), np.nan)

    for k in range(N_snap):
        mask    = np.arange(N_snap) != k
        X_train = X[:, mask]
        mean_k  = X_train.mean(axis=1)
        Xc      = X_train - mean_k[:, None]

        U_k, _, _ = np.linalg.svd(Xc, full_matrices=False)

        x_k   = X[:, k] - mean_k
        denom = np.linalg.norm(x_k)
        if denom < 1e-12:
            continue

        for r in range(1, r_max + 1):
            Phi_r  = U_k[:, :r]
            x_proj = Phi_r @ (Phi_r.T @ x_k)
            err[k, r - 1] = np.linalg.norm(x_k - x_proj) / denom

    return err


def report_loo_projection(err: np.ndarray, angles: np.ndarray):
    r_max = err.shape[1]
    print(f"{'='*65}")
    print("  Error LOO de proyeccion (cota inferior del error LOO-CV)")
    print(f"{'='*65}")
    print(f"  {'r':>3}  {'mean':>10}  {'max':>10}  {'arg max theta':>16}")
    print(f"  {'-'*3}  {'-'*10}  {'-'*10}  {'-'*16}")
    for r in range(1, r_max + 1):
        col     = err[:, r - 1]
        idx_max = int(np.nanargmax(col))
        print(f"  {r:>3d}  {np.nanmean(col):>9.4f}  {np.nanmax(col):>9.4f}  "
              f"theta = {angles[idx_max]:>5.0f}")
    print()


# =======================================================================
#  5. VISUALIZACION
# =======================================================================

def plot_snapshot_overview(X: np.ndarray, angles: np.ndarray, out_path: Path):
    """
    Mapa de calor de la matriz de snapshots + perfiles Cp por angulo.
    Eje vertical creciente: probe 0 (z=0.01 m) en la base, probe 399 arriba.
    Marcas de zona con labels de cada cara del edificio.
    """
    # Ordenacion de caras: 100 probes cada una, probes 0-99=Sotavento, etc.
    ZONE_BOUNDS = [0, 100, 200, 300, 400]
    ZONE_LABELS = ["Sotavento", "Barlovento", "Lateral+", "Lateral-"]
    ZONE_COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))

    # --- Panel 1: mapa de calor ---
    # origin="lower": probe 0 en la base, eje y creciente hacia arriba
    im = ax1.imshow(
        X, aspect="auto", origin="lower",
        cmap="RdBu_r", vmin=-2.5, vmax=1.1,
        extent=[angles[0] - 2.5, angles[-1] + 2.5, 0, N_PROBES],
        interpolation="nearest",
    )
    ax1.set_xticks(REF_TICKS)
    ax1.set_xticklabels([f"{a:g}" for a in REF_TICKS], fontsize=8)
    ax1.set_xlabel("theta (deg)")
    ax1.set_ylabel("Indice de probe")
    ax1.set_title(f"Matriz de snapshots Cp  [400 x {len(angles)}]")
    fig.colorbar(im, ax=ax1, shrink=0.85).set_label("Cp")

    # Lineas divisorias y labels de zona en panel 1 (dentro del heatmap)
    x_label = angles[0] - 1.5   # cerca del borde izquierdo del extent
    for i, (label, color) in enumerate(zip(ZONE_LABELS, ZONE_COLORS)):
        mid = (ZONE_BOUNDS[i] + ZONE_BOUNDS[i + 1]) / 2
        if i > 0:
            ax1.axhline(ZONE_BOUNDS[i], color="white", lw=1.0, ls="--", alpha=0.8)
        ax1.text(x_label, mid, label,
                 ha="left", va="center", fontsize=8, color="white",
                 fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.2", facecolor=color,
                           alpha=0.75, edgecolor="none"))

    # --- Panel 2: perfiles por angulo ---
    cmap_lines = plt.cm.plasma(np.linspace(0, 1, len(angles)))
    for i, (theta, color) in enumerate(zip(angles, cmap_lines)):
        ax2.plot(X[:, i], np.arange(N_PROBES), color=color, lw=1.0,
                 label=f"{theta:g}deg")
    ax2.set_xlabel("Cp")
    ax2.set_ylabel("Indice de probe")
    ax2.set_title("Perfiles Cp por angulo")
    ax2.axvline(0, color="black", lw=0.8, alpha=0.4)
    ax2.grid(True, alpha=0.25)
    ax2.legend(fontsize=7, ncol=2, loc="lower right")

    # Lineas divisorias y labels de zona en panel 2 (dentro del area de datos)
    trans2 = ax2.get_yaxis_transform()   # x en coords de ejes [0,1], y en datos
    for i, (label, color) in enumerate(zip(ZONE_LABELS, ZONE_COLORS)):
        mid = (ZONE_BOUNDS[i] + ZONE_BOUNDS[i + 1]) / 2
        if i > 0:
            ax2.axhline(ZONE_BOUNDS[i], color="gray", lw=1.0, ls="--", alpha=0.6)
        ax2.text(0.02, mid, label, transform=trans2,
                 ha="left", va="center", fontsize=8, color=color,
                 fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                           alpha=0.7, edgecolor="none"))

    fig.suptitle("ROM: extraccion de snapshots CFD (OpenFOAM)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out_path.name}")


def plot_singular_spectrum(pod: dict, out_path: Path):
    sigma = pod["sigma"]
    cum_e = pod["cum_energy"]
    r     = np.arange(1, len(sigma) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    ax1.semilogy(r, sigma, "o-", color="#1f4e79", lw=1.8, ms=7)
    ax1.set_xlabel("Indice del modo j")
    ax1.set_ylabel("Valor singular sigma_j")
    ax1.set_title("Espectro singular POD")
    ax1.grid(True, which="both", alpha=0.3)
    ax1.set_xticks(r)

    ax2.plot(r, cum_e * 100, "s-", color="#c00000", lw=1.8, ms=7)
    for thr in ENERGY_THRESHOLDS:
        ax2.axhline(thr * 100, color="gray", ls="--", alpha=0.4, lw=0.8)
        ax2.text(r[-1] + 0.2, thr * 100, f"{thr*100:.1f}%",
                 va="center", fontsize=8, color="gray")
    ax2.set_xlabel("Numero de modos retenidos r")
    ax2.set_ylabel("Energia acumulada E(r) [%]")
    ax2.set_title("Energia acumulada")
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(r)
    ax2.set_ylim([min(cum_e[0] * 100 - 5, 60), 101])

    fig.suptitle("POD: diagnostico de truncamiento energetico", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  -> {out_path.name}")


def plot_modal_coefficients(pod: dict, angles: np.ndarray, out_path: Path,
                            n_modes: int = N_MODES_PLOT):
    A       = pod["A"]
    n_modes = min(n_modes, A.shape[0])
    n_cols  = 3
    n_rows  = int(np.ceil(n_modes / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(11, 3.2 * n_rows), sharex=True)
    axes = np.atleast_2d(axes).ravel()

    for j in range(n_modes):
        ax = axes[j]
        ax.plot(angles, A[j, :], "o-", color="#1f4e79", lw=1.8, ms=7)
        ax.axhline(0, color="black", lw=0.5, alpha=0.5)
        ax.set_title(
            f"Modo j={j+1}  (sigma={pod['sigma'][j]:.3f}, "
            f"E={pod['energy'][j]*100:.1f}%)",
            fontsize=10,
        )
        ax.grid(True, alpha=0.3)
        ax.set_xticks(REF_TICKS)
        if j >= n_modes - n_cols:
            ax.set_xlabel("theta (deg)")
        if j % n_cols == 0:
            ax.set_ylabel("a_j(theta)")

    for j in range(n_modes, len(axes)):
        axes[j].axis("off")

    fig.suptitle("Coeficientes modales POD vs theta  (lo que GPR interpolara)",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  -> {out_path.name}")


def identify_faces(probe_coords_theta0: np.ndarray) -> dict:
    """
    Clasifica los 400 probes en las 4 caras del edificio usando
    las coordenadas body-fixed de theta=0.
    """
    x, y, z = (probe_coords_theta0[:, c] for c in range(3))
    faces = {}

    faces["Sotavento (x=+0.05)"] = {
        "idx": np.where(np.abs(x - 0.05) < 5e-3)[0],
        "u": y[np.abs(x - 0.05) < 5e-3],
        "v": z[np.abs(x - 0.05) < 5e-3],
        "ulabel": "y (m)", "vlabel": "z (m)",
    }
    faces["Barlovento (x=-0.05)"] = {
        "idx": np.where(np.abs(x + 0.05) < 5e-3)[0],
        "u": y[np.abs(x + 0.05) < 5e-3],
        "v": z[np.abs(x + 0.05) < 5e-3],
        "ulabel": "y (m)", "vlabel": "z (m)",
    }
    faces["Lateral+ (y=+0.05)"] = {
        "idx": np.where(np.abs(y - 0.05) < 5e-3)[0],
        "u": x[np.abs(y - 0.05) < 5e-3],
        "v": z[np.abs(y - 0.05) < 5e-3],
        "ulabel": "x (m)", "vlabel": "z (m)",
    }
    faces["Lateral- (y=-0.05)"] = {
        "idx": np.where(np.abs(y + 0.05) < 5e-3)[0],
        "u": x[np.abs(y + 0.05) < 5e-3],
        "v": z[np.abs(y + 0.05) < 5e-3],
        "ulabel": "x (m)", "vlabel": "z (m)",
    }

    total = sum(len(f["idx"]) for f in faces.values())
    if total != N_PROBES:
        print(f"  AVISO: clasificacion de caras suma {total} probes, no {N_PROBES}.")

    return faces


def plot_modes_unfolded(pod: dict, probe_coords: np.ndarray, out_path: Path,
                        n_modes: int = N_MODES_PLOT):
    Phi     = pod["Phi"]
    mean    = pod["mean"]
    n_modes = min(n_modes, Phi.shape[1])
    faces   = identify_faces(probe_coords[:, :, 0])
    n_rows  = n_modes + 1

    fig, axes = plt.subplots(n_rows, N_FACES,
                             figsize=(3.2 * N_FACES, 2.6 * n_rows),
                             sharex="col", sharey=True)

    field_names = ["Media"] + [f"Modo {j+1}" for j in range(n_modes)]
    fields      = [mean]   + [Phi[:, j]      for j in range(n_modes)]

    for i_row, (name, field) in enumerate(zip(field_names, fields)):
        cmap = "RdBu_r" if i_row > 0 else "viridis"
        vmax = np.max(np.abs(field)) if i_row > 0 else field.max()
        vmin = -vmax                  if i_row > 0 else field.min()

        for i_col, (fname, f) in enumerate(faces.items()):
            ax = axes[i_row, i_col]
            sc = ax.scatter(f["u"], f["v"], c=field[f["idx"]],
                            cmap=cmap, vmin=vmin, vmax=vmax,
                            s=18, marker="s", edgecolors="none")
            if i_row == 0:
                ax.set_title(fname, fontsize=9)
            if i_col == 0:
                ax.set_ylabel(f"{name}\n{f['vlabel']}", fontsize=9)
            if i_row == n_rows - 1:
                ax.set_xlabel(f["ulabel"], fontsize=9)
            ax.set_aspect("equal", adjustable="box")

        fig.colorbar(sc, ax=axes[i_row, :].tolist(), shrink=0.85, pad=0.01
                     ).ax.tick_params(labelsize=8)

    fig.suptitle("Modos POD desplegados sobre las 4 caras (Mean + primeros modos)",
                 fontweight="bold", y=1.00)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out_path.name}")


def plot_loo_projection(err: np.ndarray, angles: np.ndarray, out_path: Path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    r_axis = np.arange(1, err.shape[1] + 1)

    im = ax1.imshow(
        err.T, aspect="auto", origin="lower", cmap="magma_r",
        extent=[angles[0] - 2.5, angles[-1] + 2.5, 0.5, err.shape[1] + 0.5],
        interpolation="nearest",
    )
    ax1.set_xticks(REF_TICKS)
    ax1.set_yticks(r_axis)
    ax1.set_xlabel("theta excluido (deg)")
    ax1.set_ylabel("r (modos retenidos)")
    ax1.set_title("Error LOO de proyeccion por fold y r")
    fig.colorbar(im, ax=ax1, shrink=0.85).set_label("Error relativo")

    err_mean = np.nanmean(err, axis=0)
    err_max  = np.nanmax(err, axis=0)
    ax2.plot(r_axis, err_mean, "o-",  color="#1f4e79", lw=1.8, ms=7, label="Media LOO")
    ax2.plot(r_axis, err_max,  "s--", color="#c00000", lw=1.5, ms=6, label="Maximo LOO")
    ax2.set_xlabel("r (modos retenidos)")
    ax2.set_ylabel("Error relativo de proyeccion")
    ax2.set_title("Cota inferior del error LOO-CV completo")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_xticks(r_axis)
    ax2.set_yscale("log")

    fig.suptitle("LOO de proyeccion: error intrinseco a la base POD",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  -> {out_path.name}")


# =======================================================================
#  6. GUARDADO
# =======================================================================

def save_basis(pod: dict, angles: np.ndarray, probe_coords: np.ndarray,
               out_path: Path):
    np.savez(
        out_path,
        mean=pod["mean"],
        Phi=pod["Phi"],
        sigma=pod["sigma"],
        Vt=pod["Vt"],
        A=pod["A"],
        energy=pod["energy"],
        cum_energy=pod["cum_energy"],
        angles=angles,
        probe_coords=probe_coords,
        Uref=UREF,
        q_ref=Q_REF,
    )
    print(f"\nBase POD guardada en: {out_path.name}")
    print(f"  mean  : {pod['mean'].shape}")
    print(f"  Phi   : {pod['Phi'].shape}    (modos espaciales)")
    print(f"  sigma : {pod['sigma'].shape}  (valores singulares)")
    print(f"  A     : {pod['A'].shape}      (coeficientes modales)")


# =======================================================================
#  MAIN
# =======================================================================

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ROM_POD: POD de snapshots CFD")
    p.add_argument(
        "--angles",
        choices=["base_ref", "base_kawai", "base_full", "tpu11", "all24"],
        default="base_full",
        help=(
            "base_ref: {0,5,...,45} (10 ang, paso 5); "
            "base_kawai: ref + densificado zona vortice [10-30]; "
            "base_full: todos disponibles <=45 grados; "
            "tpu11/all24: legacy"
        ),
    )
    p.add_argument(
        "--outdir", default="outputs/base_full",
        help="Directorio de salida relativo a Programacion/ (default: outputs/base_full)",
    )
    return p.parse_args()


TPU11_SET = {0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0}


def main():
    args   = _parse_args()
    outdir = SCRIPT_DIR / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    if args.angles == "base_ref":
        angles_use = [a for a in ANGLES if a in BASE_REF]
    elif args.angles == "base_kawai":
        angles_use = [a for a in ANGLES if a in BASE_KAWAI]
    elif args.angles == "base_full":
        angles_use = [a for a in ANGLES if a <= MAX_THETA]
    elif args.angles == "tpu11":
        angles_use = [a for a in ANGLES if a in TPU11_SET]
    else:
        angles_use = ANGLES

    print(f"\n  Set '{args.angles}': {len(angles_use)} angulos")
    print(f"  Salida: {outdir.relative_to(SCRIPT_DIR)}")

    out_npz = outdir / "ROM_POD_basis.npz"

    # 1. Lectura directa de OpenFOAM
    data         = load_openfoam_snapshots(angles_use)
    X            = data["X"]
    angles       = data["angles"]
    probe_coords = data["probe_coords"]

    # 2. POD
    pod   = pod_svd(X)
    r_eff = report_energy(pod)

    # 3. LOO de proyeccion
    err_proj = loo_projection_error(X)
    report_loo_projection(err_proj, angles)

    # 4. Figuras
    print(f"{'='*65}")
    print("  Generando figuras...")
    print(f"{'='*65}")
    plot_snapshot_overview(X, angles,      outdir / "snapshot_overview.png")
    plot_singular_spectrum(pod,            outdir / "POD_singular_spectrum.png")
    plot_modal_coefficients(pod, angles,   outdir / "POD_modal_coefficients.png")
    plot_modes_unfolded(pod, probe_coords, outdir / "POD_modes_unfolded.png")
    plot_loo_projection(err_proj, angles,  outdir / "POD_loo_projection_error.png")

    # 5. Guardado
    save_basis(pod, angles, probe_coords, out_npz)

    print(f"\n{'='*65}")
    print(f"  POD completada: set='{args.angles}', N={len(angles_use)}")
    print(f"  Siguiente: python ROM_GPR.py --outdir {args.outdir}")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
