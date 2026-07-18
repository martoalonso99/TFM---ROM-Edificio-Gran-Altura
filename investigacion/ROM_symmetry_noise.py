"""
ROM_symmetry_noise.py
======================
Suelo de ruido CFD (BRIEF_holdout.md S5), previo a interpretar el holdout.

Por simetria C4 + reflexion de la seccion cuadrada, el campo Cp(theta) debe
coincidir con Cp(90-theta) salvo una permutacion de las 400 sondas (una
transformacion del grupo diedrico D4 sobre las coordenadas body-fixed).
En vez de derivar esa transformacion a mano (facil equivocar un signo o el
orden de columnas), se descubre EMPIRICAMENTE: se prueban las 8
transformaciones D4 sobre el layout de sondas conocido (identico al usado
en generate_new_cases.py) y se elige la que minimiza la discrepancia entre
los pares ya simulados (40,50) y (42.5,47.5).

La discrepancia resultante, bajo la transformacion ganadora, es el suelo de
ruido: diferencias de error entre bases del holdout por debajo de este
valor no son atribuibles al diseno muestral (son ruido de malla/CFD).

Salidas:
  outputs/holdout/symmetry_noise.csv
  outputs/holdout/symmetry_noise_fields.png   (Cp_A vs Cp_B(perm) por par)

Uso:
  python ROM_symmetry_noise.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent          # investigacion/
ROOT       = SCRIPT_DIR.parent                        # Programacion/
sys.path.insert(0, str(ROOT))

from ROM_POD import DATA_DIR, N_PROBES, Q_REF, _parse_probe_file, fmt_angle   # noqa: E402

OUT_DIR    = ROOT / "outputs" / "holdout"

# Pares fisicamente equivalentes por reflexion theta <-> 90-theta
PAIRS = [(40.0, 50.0), (42.5, 47.5)]

# Layout de sondas body-fixed (theta=0), identico a generate_new_cases.py:
# 4 caras x (5 columnas x 20 filas), orden: cara -> columna -> fila
PROBE_OFFSET = 0.0501
PROBE_COLS   = [-0.04, -0.02, 0.0, 0.02, 0.04]
PROBE_Z      = [round(0.01 + 0.02 * i, 5) for i in range(20)]


# =======================================================================
#  1. LAYOUT BODY-FIXED Y GRUPO D4
# =======================================================================

def body_fixed_layout() -> np.ndarray:
    """
    Reconstruye (x0, y0, z) para los 400 indices de sonda en el orden
    exacto en que se escriben en el fichero de probes (ver
    generate_new_cases.py: make_probe_content).
    """
    face_origins = [
        [(PROBE_OFFSET, y) for y in PROBE_COLS],    # cara 1: sotavento x=+0.0501
        [(-PROBE_OFFSET, y) for y in PROBE_COLS],   # cara 2: barlovento x=-0.0501
        [(x, PROBE_OFFSET) for x in PROBE_COLS],    # cara 3: lateral+ y=+0.0501
        [(x, -PROBE_OFFSET) for x in PROBE_COLS],   # cara 4: lateral- y=-0.0501
    ]
    coords = []
    for face in face_origins:
        for x0, y0 in face:
            for z in PROBE_Z:
                coords.append((x0, y0, z))
    return np.array(coords)   # (400, 3)


def d4_transforms() -> dict:
    """Los 8 elementos del grupo diedrico D4 sobre (x,y); z es invariante."""
    return {
        "identity":    lambda x, y: (x, y),
        "rot90":       lambda x, y: (-y, x),
        "rot180":      lambda x, y: (-x, -y),
        "rot270":      lambda x, y: (y, -x),
        "mirror_x":    lambda x, y: (x, -y),
        "mirror_y":    lambda x, y: (-x, y),
        "mirror_diag": lambda x, y: (y, x),
        "mirror_adiag": lambda x, y: (-y, -x),
    }


def build_permutation(layout: np.ndarray, transform) -> np.ndarray | None:
    """
    Para cada indice i, encuentra j tal que layout[j] == transform(layout[i]).
    Devuelve None si el layout no es cerrado bajo la transformacion dada
    (no deberia ocurrir: el grid de sondas es simetrico bajo D4 por
    construccion, salvo tolerancia numerica).
    """
    n = layout.shape[0]
    perm = np.full(n, -1, dtype=int)
    for i in range(n):
        x0, y0, z0 = layout[i]
        xt, yt = transform(x0, y0)
        d = np.sqrt((layout[:, 0] - xt) ** 2 + (layout[:, 1] - yt) ** 2
                    + (layout[:, 2] - z0) ** 2)
        j = int(np.argmin(d))
        if d[j] > 1e-6:
            return None
        perm[i] = j
    return perm


# =======================================================================
#  2. CARGA DE CAMPOS CP
# =======================================================================

def load_cp(theta: float) -> np.ndarray:
    fpath = (DATA_DIR / f"ROMCase_theta_{fmt_angle(theta)}"
             / "postProcessing" / "probes_buildingPressure" / "0" / "p")
    if not fpath.is_file():
        raise FileNotFoundError(f"No encontrado: {fpath}")
    _, _, p_vals = _parse_probe_file(fpath)
    return p_vals / Q_REF


# =======================================================================
#  3. BUSQUEDA DE LA TRANSFORMACION GANADORA
# =======================================================================

def find_best_transform(layout: np.ndarray, cp_pairs: list[tuple]) -> tuple[str, np.ndarray, dict]:
    """
    Prueba las 8 transformaciones D4 sobre los pares conocidos y elige la
    que minimiza el RMS medio combinado. Devuelve (nombre, permutacion,
    rms_por_par) de la ganadora.
    """
    results = {}
    for name, transform in d4_transforms().items():
        perm = build_permutation(layout, transform)
        if perm is None:
            print(f"  [{name}] layout no cerrado bajo esta transformacion — descartada")
            continue
        rms_list = []
        for cp_a, cp_b in cp_pairs:
            rms = float(np.sqrt(np.mean((cp_a - cp_b[perm]) ** 2)))
            rms_list.append(rms)
        rms_mean = float(np.mean(rms_list))
        results[name] = {"perm": perm, "rms_list": rms_list, "rms_mean": rms_mean}
        print(f"  [{name:<12}] RMS por par = {['%.4f' % r for r in rms_list]}  "
              f"media = {rms_mean:.4f}")

    best_name = min(results, key=lambda k: results[k]["rms_mean"])
    return best_name, results[best_name]["perm"], results


# =======================================================================
#  FIGURA
# =======================================================================

def plot_pairs(cp_pairs_thetas: list[tuple], cp_fields: list[tuple],
               perm: np.ndarray, out_path: Path):
    n = len(cp_pairs_thetas)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, (th_a, th_b), (cp_a, cp_b) in zip(axes, cp_pairs_thetas, cp_fields):
        cp_b_perm = cp_b[perm]
        lim = [min(cp_a.min(), cp_b_perm.min()) - 0.05,
               max(cp_a.max(), cp_b_perm.max()) + 0.05]
        ax.plot(lim, lim, "k--", lw=1.2, label="igualdad perfecta")
        ax.scatter(cp_a, cp_b_perm, s=10, alpha=0.5, color="#1f4e79")
        rms = float(np.sqrt(np.mean((cp_a - cp_b_perm) ** 2)))
        ax.set_xlabel(f"Cp CFD (θ={th_a:.1f}°)")
        ax.set_ylabel(f"Cp CFD (θ={th_b:.1f}°, permutado)")
        ax.set_title(f"θ={th_a:.1f}° vs θ={th_b:.1f}°  (RMS={rms:.4f})",
                     fontweight="bold", fontsize=10)
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle("Suelo de ruido CFD: pares simetricos θ ↔ 90°-θ",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out_path.relative_to(ROOT)}")


# =======================================================================
#  MAIN
# =======================================================================

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*65}")
    print("  Suelo de ruido CFD via pares simetricos (BRIEF_holdout.md S5)")
    print(f"{'='*65}\n")

    layout = body_fixed_layout()

    print("  Cargando pares...")
    cp_pairs = []
    for th_a, th_b in PAIRS:
        cp_a, cp_b = load_cp(th_a), load_cp(th_b)
        cp_pairs.append((cp_a, cp_b))
        print(f"    theta={th_a}°: Cp in [{cp_a.min():.3f}, {cp_a.max():.3f}]  "
              f"|  theta={th_b}°: Cp in [{cp_b.min():.3f}, {cp_b.max():.3f}]")

    print(f"\n  Probando las 8 transformaciones D4 sobre los pares conocidos:")
    best_name, best_perm, all_results = find_best_transform(layout, cp_pairs)

    print(f"\n  >> Transformacion ganadora: {best_name}")
    for (th_a, th_b), rms in zip(PAIRS, all_results[best_name]["rms_list"]):
        print(f"     ({th_a}°, {th_b}°): RMS = {rms:.4f}")

    noise_floor = all_results[best_name]["rms_mean"]
    print(f"\n  SUELO DE RUIDO (RMS medio) = {noise_floor:.4f}")
    print(f"  Regla de decision: diferencias de RMSE entre bases del holdout")
    print(f"  por debajo de {noise_floor:.4f} no son atribuibles al diseno muestral.\n")

    # CSV
    csv_path = OUT_DIR / "symmetry_noise.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["transform", "theta_a", "theta_b", "rms"])
        for name, res in all_results.items():
            for (th_a, th_b), rms in zip(PAIRS, res["rms_list"]):
                w.writerow([name, th_a, th_b, f"{rms:.4f}"])
        w.writerow([])
        w.writerow(["best_transform", best_name, "", ""])
        w.writerow(["noise_floor_rms", f"{noise_floor:.4f}", "", ""])
    print(f"  -> {csv_path.relative_to(ROOT)}")

    plot_pairs(PAIRS, cp_pairs, best_perm, OUT_DIR / "symmetry_noise_fields.png")

    print("\n  Listo.\n")


if __name__ == "__main__":
    main()
