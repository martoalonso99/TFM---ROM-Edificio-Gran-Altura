"""
ROM_cost_analysis.py
====================
Extrae el coste computacional de cada caso CFD desde los logs del solver
(log.simpleFoam.<nCores>) y lo normaliza para que los casos ejecutados con
distinto numero de cores sean comparables.

Normalizacion en dos capas:

1) Cores. Los casos duplicados (ejecutados a 10 y a 20 cores) miden la
   perdida de eficiencia paralela. Con ese factor, todo caso se expresa en
   "CPU-h equivalentes a 10 cores":

    cpu_h        = ExecutionTime * nCores / 3600
    cpu_h_eq10   = cpu_h                    (si nCores = 10)
    cpu_h_eq10   = cpu_h * f_10/20          (si nCores = 20)

   donde f_10/20 = media de cpu_h(10)/cpu_h(20) sobre los casos duplicados.

2) Contencion. Algunos casos se ejecutaron simultaneamente en la misma
   maquina (co-scheduling), inflando su ExecutionTime sin reflejar coste
   intrinseco. Se detectan comparando la tasa por unidad de trabajo
   rate = cpu_h_eq10 / (n_celdas * n_iters) con la mediana de todos los
   runs: si rate > CONTENTION_FACTOR * mediana, el run se marca como
   contended y su coste se sustituye por el coste modelado
   cpu_h_model = rate_mediana * n_celdas * n_iters (coste que habria
   tenido en ejecucion exclusiva).

Salidas:
  - outputs/comparativa/cost_per_case.csv    coste por caso y por run
  - outputs/comparativa/cost_vs_error.png    LOO vs CPU-h acumuladas por base
  - outputs/comparativa/cost_resumen.csv     coste acumulado por base

Uso:
  python ROM_cost_analysis.py
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ROM_POD import (
    BASE_REF, BASE_KAWAI, BASE_KAWAI2, BASE_FULL22, DATA_DIR, MAX_THETA,
)

SCRIPT_DIR = Path(__file__).resolve().parent
OUT_DIR    = SCRIPT_DIR / "outputs" / "comparativa"

RE_EXEC  = re.compile(r"ExecutionTime = ([\d.]+) s")
RE_TIME  = re.compile(r"^Time = (\d+)s?", re.MULTILINE)
RE_CELLS = re.compile(r"^    cells:\s+(\d+)", re.MULTILINE)

# Un run cuya tasa CPU/(celda*iter) supere este multiplo de la mediana se
# considera ejecutado con contencion (co-scheduling) y su coste se modela
CONTENTION_FACTOR = 1.5


# =======================================================================
#  PARSEO DE LOGS
# =======================================================================

def parse_theta(case_name: str) -> float:
    """ROMCase_theta_23p75 -> 23.75, ROMCase_theta_05 -> 5.0"""
    return float(case_name.split("theta_")[1].replace("p", "."))


def read_log_tail(path: Path, n_bytes: int = 20_000) -> str:
    with open(path, "rb") as f:
        f.seek(0, 2)
        f.seek(max(0, f.tell() - n_bytes))
        return f.read().decode("utf-8", errors="replace")


def read_cells(case_dir: Path) -> int:
    """Numero de celdas de la malla segun log.checkMesh (0 si no disponible)."""
    log = case_dir / "log.checkMesh"
    if not log.is_file():
        return 0
    m = RE_CELLS.search(log.read_text(encoding="utf-8", errors="replace"))
    return int(m.group(1)) if m else 0


def scan_runs() -> list[dict]:
    """Un registro por log de solver encontrado (un caso puede tener varios)."""
    runs = []
    for case_dir in sorted(DATA_DIR.glob("ROMCase_theta_*")):
        if not case_dir.is_dir():
            continue
        theta = parse_theta(case_dir.name)
        cells = read_cells(case_dir)
        for log in sorted(case_dir.glob("log.simpleFoam.*")):
            cores = int(log.suffix.lstrip("."))
            tail  = read_log_tail(log)
            m_exec = RE_EXEC.findall(tail)
            m_time = RE_TIME.findall(tail)
            if not m_exec:
                print(f"  AVISO: sin ExecutionTime en {log} — omitido")
                continue
            exec_s = float(m_exec[-1])
            runs.append({
                "theta":      theta,
                "case":       case_dir.name,
                "cores":      cores,
                "cells":      cells,
                "exec_s":     exec_s,
                "last_iter":  int(m_time[-1]) if m_time else -1,
                "cpu_h":      exec_s * cores / 3600.0,
            })
    return runs


# =======================================================================
#  NORMALIZACION
# =======================================================================

def efficiency_factor(runs: list[dict]) -> float:
    """
    Factor f_10/20 = cpu_h(10 cores) / cpu_h(20 cores), medido en los casos
    que se ejecutaron con ambas configuraciones. f < 1 refleja la perdida de
    eficiencia paralela al doblar los cores.
    """
    by_theta: dict[float, dict[int, float]] = {}
    for r in runs:
        by_theta.setdefault(r["theta"], {})[r["cores"]] = r["cpu_h"]

    factors = []
    for theta, cfg in sorted(by_theta.items()):
        if 10 in cfg and 20 in cfg:
            f = cfg[10] / cfg[20]
            factors.append(f)
            print(f"  theta={theta:>6.2f}: cpu_h(10)={cfg[10]:.1f}  "
                  f"cpu_h(20)={cfg[20]:.1f}  ->  f={f:.3f}")

    if not factors:
        print("  AVISO: sin casos duplicados — se asume eficiencia perfecta (f=1)")
        return 1.0
    return float(np.mean(factors))


def normalize_runs(runs: list[dict], f_1020: float) -> dict[float, float]:
    """
    Coste por caso en CPU-h equivalentes a 10 cores, con correccion de
    contencion. Si hay varios runs del mismo caso se promedian los valores
    normalizados (deberian coincidir).
    """
    # Capa 1: equivalencia de cores
    for r in runs:
        r["cpu_h_eq10"] = r["cpu_h"] * (f_1020 if r["cores"] == 20 else 1.0)

    # Capa 2: deteccion de contencion via tasa por unidad de trabajo
    rates = [r["cpu_h_eq10"] / (r["cells"] * r["last_iter"])
             for r in runs if r["cells"] > 0 and r["last_iter"] > 0]
    rate_med = float(np.median(rates)) if rates else 0.0

    contended = []
    for r in runs:
        work = r["cells"] * r["last_iter"]
        if work <= 0 or rate_med <= 0:
            r["contended"]  = False
            r["cpu_h_norm"] = r["cpu_h_eq10"]
            continue
        rate = r["cpu_h_eq10"] / work
        r["contended"] = rate > CONTENTION_FACTOR * rate_med
        r["cpu_h_norm"] = rate_med * work if r["contended"] else r["cpu_h_eq10"]
        if r["contended"]:
            contended.append(r)

    if contended:
        print(f"\n  Runs con contencion detectada (tasa > {CONTENTION_FACTOR}x mediana):")
        for r in contended:
            print(f"    theta={r['theta']:>6.2f} ({r['cores']} cores): "
                  f"medido {r['cpu_h_eq10']:.1f} CPU-h -> modelado {r['cpu_h_norm']:.1f} CPU-h")

    by_theta: dict[float, list[float]] = {}
    for r in runs:
        by_theta.setdefault(r["theta"], []).append(r["cpu_h_norm"])
    return {t: float(np.mean(v)) for t, v in by_theta.items()}


# =======================================================================
#  COSTE POR BASE + LOO
# =======================================================================

def base_angle_sets(cost: dict[float, float]) -> dict[str, set[float]]:
    """
    Sets congelados de cada base (el nombre coincide con su outputs/<base>/).
    'base_full' usa BASE_FULL22 porque el modelo almacenado en
    outputs/base_full se entreno con esos 22 angulos, no con todos los
    disponibles a dia de hoy.
    """
    available = {t for t in cost if t <= MAX_THETA}
    named = {
        "base_ref":    BASE_REF,
        "base_kawai":  BASE_KAWAI,
        "base_kawai2": BASE_KAWAI2,
        "base_full":   BASE_FULL22,
    }
    sets = {}
    for name, ang_set in named.items():
        if ang_set <= available:
            sets[name] = ang_set
        else:
            missing = sorted(ang_set - available)
            print(f"  AVISO: '{name}' incompleta (faltan casos {missing}) — omitida")
    return sets


def load_loo(base_name: str) -> float | None:
    path = SCRIPT_DIR / "outputs" / base_name / "ROM_GPR_results.npz"
    if not path.is_file():
        return None
    d      = np.load(str(path), allow_pickle=False)
    r_star = int(d["r_star"])
    idx    = int(np.searchsorted(d["sweep_r"], r_star))
    return float(d["sweep_mean_total"][idx])


# =======================================================================
#  FIGURA
# =======================================================================

def plot_cost_vs_error(resumen: list[dict], out_path: Path):
    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    colors  = {"base_ref": "#1f4e79", "base_kawai": "#d62728",
               "base_kawai2": "#ff7f0e", "base_full": "#2ca02c"}
    markers = {"base_ref": "o", "base_kawai": "s",
               "base_kawai2": "D", "base_full": "^"}

    xs = [r["cpu_h_total"] for r in resumen]
    ys = [r["loo"] for r in resumen]
    ax.plot(xs, ys, "-", color="gray", lw=1.2, alpha=0.6, zorder=1)

    for r in resumen:
        ax.scatter(r["cpu_h_total"], r["loo"],
                   s=140, color=colors[r["base"]], marker=markers[r["base"]],
                   zorder=5, label=f"{r['base']} (N={r['n_angulos']})")
        ax.annotate(f"  LOO={r['loo']:.3f}",
                    (r["cpu_h_total"], r["loo"]),
                    textcoords="offset points", xytext=(10, 5), fontsize=9)

    ax.set_xlabel("Coste CFD acumulado (CPU-h equivalentes a 10 cores)", fontsize=11)
    ax.set_ylabel("Error LOO medio", fontsize=11)
    ax.set_title("Coste computacional vs precision del ROM\n"
                 "(cada punto = una estrategia de muestreo)",
                 fontweight="bold", fontsize=11)
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

    print(f"\n{'='*65}")
    print("  Analisis de coste computacional CFD")
    print(f"{'='*65}\n")

    runs = scan_runs()
    print(f"  Runs encontrados: {len(runs)} "
          f"({len({r['theta'] for r in runs})} casos)\n")

    print("  Casos duplicados (miden eficiencia paralela):")
    f_1020 = efficiency_factor(runs)
    print(f"\n  Factor de normalizacion f_10/20 = {f_1020:.3f}")
    print(f"  (correr a 20 cores cuesta {1/f_1020:.2f}x mas CPU-h que a 10)\n")

    cost = normalize_runs(runs, f_1020)

    # --- CSV por caso ---
    per_case_csv = OUT_DIR / "cost_per_case.csv"
    with open(per_case_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["theta", "cores", "cells", "exec_time_s", "last_iter",
                    "cpu_h", "cpu_h_eq10", "contended", "cpu_h_norm"])
        for r in sorted(runs, key=lambda r: (r["theta"], r["cores"])):
            w.writerow([f"{r['theta']:.2f}", r["cores"], r["cells"],
                        f"{r['exec_s']:.0f}", r["last_iter"],
                        f"{r['cpu_h']:.2f}", f"{r['cpu_h_eq10']:.2f}",
                        int(r["contended"]), f"{r['cpu_h_norm']:.2f}"])
    print(f"  -> {per_case_csv.relative_to(SCRIPT_DIR)}")

    # --- Coste acumulado por base + LOO ---
    resumen = []
    print(f"\n  {'Base':<14} {'N':>4} {'CPU-h eq10':>12} {'LOO':>8}")
    print(f"  {'-'*14} {'-'*4} {'-'*12} {'-'*8}")
    for base_name, ang_set in base_angle_sets(cost).items():
        total = sum(cost[t] for t in ang_set)
        loo   = load_loo(base_name)
        print(f"  {base_name:<14} {len(ang_set):>4} {total:>12.1f} "
              f"{loo if loo is not None else float('nan'):>8.4f}")
        resumen.append({
            "base":        base_name,
            "n_angulos":   len(ang_set),
            "cpu_h_total": total,
            "loo":         loo,
        })

    resumen_csv = OUT_DIR / "cost_resumen.csv"
    with open(resumen_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["base", "n_angulos", "cpu_h_eq10_total", "LOO_medio"])
        for r in resumen:
            w.writerow([r["base"], r["n_angulos"], f"{r['cpu_h_total']:.1f}",
                        f"{r['loo']:.4f}" if r["loo"] is not None else "nan"])
    print(f"\n  -> {resumen_csv.relative_to(SCRIPT_DIR)}")

    plot_cost_vs_error([r for r in resumen if r["loo"] is not None],
                       OUT_DIR / "cost_vs_error.png")

    print("\n  Listo.\n")


if __name__ == "__main__":
    main()
