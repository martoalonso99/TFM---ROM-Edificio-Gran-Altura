"""
copy_stls.py
Copia los STLs rotados desde Base Geometry and Results a cada caso ROM.
Ejecutar desde cualquier directorio (las rutas son absolutas por defecto).

Uso:
    python copy_stls.py
    python copy_stls.py --stl_dir /ruta/a/stls --rom_dir /ruta/a/ROM
"""

import os
import shutil
import struct
import argparse
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent   # TFM/Programacion/

# ── Mapeo angulo -> nombre STL ───────────────────────────────────────────────
ANGLE_STL = {
     0: "building.stl",
     5: "building_05.stl",
    10: "building_10.stl",
    15: "building_15.stl",
    20: "building_20.stl",
    25: "building_25.stl",
    30: "building_30.stl",
    35: "building_35.stl",
    40: "building_40.stl",
    45: "building_45.stl",
    50: "building_50.stl",
}

def is_real_file(fp):
    """Devuelve True si el archivo esta descargado (no es placeholder OneDrive)."""
    try:
        with open(fp, "rb") as f:
            return len(f.read(10)) > 0
    except Exception:
        return False

def get_stl_info(fp):
    """Extrae numero de triangulos y primer vertice de un STL binario."""
    try:
        with open(fp, "rb") as f:
            data = f.read()
        n_tri = struct.unpack_from("<I", data, 80)[0]
        v1    = struct.unpack_from("<fff", data, 84 + 12)
        return n_tri, v1
    except Exception:
        return None, None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stl_dir",
        default=str(SCRIPT_DIR.parent / "Base Geometry and Results"),
        help="Carpeta con los STLs rotados",
    )
    parser.add_argument(
        "--rom_dir",
        default=str(SCRIPT_DIR.parent / "Simulaciones" / "ROM"),
        help="Carpeta raiz con los casos ROMCase_theta_XX",
    )
    args = parser.parse_args()

    stl_dir = os.path.realpath(args.stl_dir)
    rom_dir = os.path.realpath(args.rom_dir)

    print(f"STL source : {stl_dir}")
    print(f"ROM cases  : {rom_dir}")
    print()

    ok = 0
    skip = 0
    missing = 0

    for angle, stl_name in sorted(ANGLE_STL.items()):
        case_name = f"ROMCase_theta_{angle:02d}"
        case_dir  = os.path.join(rom_dir, case_name)
        src       = os.path.join(stl_dir, stl_name)
        dst       = os.path.join(case_dir, "constant", "triSurface", "building.stl")

        # Caso no existe -> saltar
        if not os.path.isdir(case_dir):
            print(f"  [{angle:02d}]  {case_name} no encontrado -- saltando")
            skip += 1
            continue

        # STL fuente no existe o es placeholder
        if not os.path.exists(src) or not is_real_file(src):
            print(f"  [{angle:02d}]  {stl_name} no disponible (placeholder o falta) -- saltando")
            missing += 1
            continue

        # Copiar
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with open(src, "rb") as f:
            data = f.read()
        with open(dst, "wb") as f:
            f.write(data)

        # Verificar
        n_tri, v1 = get_stl_info(dst)
        v1_str = f"({v1[0]:.2f}, {v1[1]:.2f}, {v1[2]:.2f})" if v1 else "?"
        print(f"  [{angle:02d}]  {stl_name} -> {case_name}  |  {n_tri} tri, v1={v1_str}  OK")
        ok += 1

    print()
    print(f"Resultado: {ok} copiados, {skip} casos no encontrados, {missing} STLs pendientes")

if __name__ == "__main__":
    main()
