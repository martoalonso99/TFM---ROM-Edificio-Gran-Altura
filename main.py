"""
main.py — Punto de entrada unico del pipeline ROM
=================================================
Orquesta las etapas de PRODUCCION que bastan para predecir Cp en un angulo
nuevo, en el orden correcto:

    pod  ->  gpr  ->  evaluate            (+ compare / cost / holdout, transversales)

Cada subcomando delega en el modulo correspondiente (ROM_POD, ROM_GPR, ...)
llamando a su main(argv); no duplica logica. Los scripts de INVESTIGACION
(suelo de error, kernel no estacionario, sensibilidad, ruido de simetria)
viven en investigacion/ y se ejecutan aparte: 'investigate' los lista.

Ejemplos
--------
  python main.py run      --angles base_full --outdir outputs/base_full
  python main.py pod       --angles base_kawai2 --outdir outputs/base_kawai2
  python main.py gpr       --outdir outputs/base_kawai2
  python main.py evaluate  --outdir outputs/base_kawai2
  python main.py compare
  python main.py cost
  python main.py holdout
  python main.py investigate
"""

from __future__ import annotations

import argparse
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

# Conjuntos de angulos disponibles para --angles (deben coincidir con ROM_POD)
ANGLE_SETS = ["base_ref", "base_kawai", "base_kawai2", "base_full22",
              "base_full", "tpu11", "all24"]


# =======================================================================
#  SUBCOMANDOS  (import perezoso: cada uno carga solo lo que necesita)
# =======================================================================

def cmd_pod(args):
    import ROM_POD
    ROM_POD.main(["--angles", args.angles, "--outdir", args.outdir])


def cmd_gpr(args):
    import ROM_GPR
    ROM_GPR.main(["--outdir", args.outdir])


def cmd_evaluate(args):
    import ROM_test_intermediate
    ROM_test_intermediate.main(["--outdir", args.outdir])


def cmd_compare(args):
    import ROM_comparativa
    ROM_comparativa.main()


def cmd_cost(args):
    import ROM_cost_analysis
    ROM_cost_analysis.main()


def cmd_holdout(args):
    import ROM_holdout_eval
    ROM_holdout_eval.main()


def cmd_run(args):
    """Pipeline completo de una base: pod -> gpr -> evaluate."""
    print(f"\n### [1/3] POD  ({args.angles} -> {args.outdir})")
    cmd_pod(args)
    print(f"\n### [2/3] GPR  ({args.outdir})")
    cmd_gpr(args)
    print(f"\n### [3/3] Evaluacion externa  ({args.outdir})")
    cmd_evaluate(args)
    print(f"\n### Pipeline completo: {args.outdir}")


def cmd_investigate(args):
    inv = SCRIPT_DIR / "investigacion"
    scripts = sorted(inv.glob("ROM_*.py")) if inv.is_dir() else []
    if not scripts:
        print("No hay scripts en investigacion/.")
        return
    print("Scripts de investigacion (ejecutar directamente, no forman parte")
    print("del pipeline de produccion):\n")
    for f in scripts:
        print(f"  python investigacion/{f.name}")


# =======================================================================
#  CLI
# =======================================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="main.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="cmd", required=True, metavar="<comando>")

    sp = sub.add_parser("pod", help="POD de los snapshots CFD de una base")
    sp.add_argument("--angles", choices=ANGLE_SETS, default="base_full")
    sp.add_argument("--outdir", default="outputs/base_full")
    sp.set_defaults(func=cmd_pod)

    sg = sub.add_parser("gpr", help="GPR por modo + LOO-CV + seleccion de kernel")
    sg.add_argument("--outdir", default="outputs/base_full")
    sg.set_defaults(func=cmd_gpr)

    se = sub.add_parser("evaluate", help="test externo en angulos no vistos por la base")
    se.add_argument("--outdir", default="outputs/base_full")
    se.set_defaults(func=cmd_evaluate)

    sr = sub.add_parser("run", help="pipeline completo de una base (pod + gpr + evaluate)")
    sr.add_argument("--angles", choices=ANGLE_SETS, default="base_full")
    sr.add_argument("--outdir", default="outputs/base_full")
    sr.set_defaults(func=cmd_run)

    sub.add_parser("compare", help="comparativa de error entre bases").set_defaults(func=cmd_compare)
    sub.add_parser("cost", help="coste computacional normalizado por base").set_defaults(func=cmd_cost)
    sub.add_parser("holdout", help="evaluacion sobre el holdout pre-registrado").set_defaults(func=cmd_holdout)
    sub.add_parser("investigate", help="lista los scripts de investigacion/").set_defaults(func=cmd_investigate)

    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
