"""
generate_new_cases.py
Genera ZIPs de casos OpenFOAM para los angulos intermedios del grid de 2.5 deg:
  2.5, 7.5, 12.5, 17.5, 22.5, 27.5, 32.5, 37.5, 42.5, 47.5 grados.

Complementa los 11 casos originales (0, 5, 10, ..., 50) para cubrir el espacio
parametrico con resolucion de 2.5 deg en [0, 50].

Convencion de nombre para angulos decimales: '.' -> 'p'
  Ejemplo: 2.5 -> ROMCase_theta_2p5
  Los enteros mantienen cero a la izquierda: 10 -> ROMCase_theta_10

Cada ZIP contiene un caso listo para correr con:
  - STL binario rotado (12 triangulos, vertices en mm, scale 0.001 en snappyHexMeshDict)
  - Fichero de probes rotado (400 probes body-fixed, mismo orden que los 11 casos base)
  - Resto de ficheros copiados del caso plantilla ROMCase_theta_20
  - Allrun.sh actualizado con el angulo correcto

Ordenacion de probes (body-fixed, consistente con casos existentes):
  Cara 1 (sotavento,  x=+0.0501): cols y in [-0.04,-0.02,0,+0.02,+0.04], filas z
  Cara 2 (barlovento, x=-0.0501): cols y in [-0.04,-0.02,0,+0.02,+0.04], filas z
  Cara 3 (lateral+,   y=+0.0501): cols x in [-0.04,-0.02,0,+0.02,+0.04], filas z
  Cara 4 (lateral-,   y=-0.0501): cols x in [-0.04,-0.02,0,+0.02,+0.04], filas z

Uso:
    python generate_new_cases.py
"""

import math
import os
import struct
import zipfile
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "Simulaciones" / "ROM"
TEMPLATE_CASE = "ROMCase_theta_20"

# Angulos intermedios: grid de 2.5 deg, excluyendo los ya simulados (multiplos de 5)
NEW_ANGLES = [2.5, 7.5, 12.5, 17.5, 22.5, 27.5, 32.5, 37.5, 42.5, 47.5]

B_HALF = 50.0    # mm (semilado seccion cuadrada; scale 0.001 -> 0.05 m)
H_MM = 400.0     # mm (altura edificio; scale 0.001 -> 0.40 m)

PROBE_OFFSET = 0.0501    # m (semilado 0.05 m + 0.0001 m setback)
PROBE_COLS = [-0.04, -0.02, 0.0, 0.02, 0.04]   # m
PROBE_Z = [round(0.01 + 0.02 * i, 5) for i in range(20)]  # m  (0.01 a 0.39)

# Ficheros estaticos a copiar del caso plantilla
COPY_FILES = [
    "0/k", "0/nut", "0/omega", "0/p", "0/U",
    "system/blockMeshDict",
    "system/controlDict",
    "system/decomposeParDict",
    "system/fvSchemes",
    "system/fvSolution",
    "system/snappyHexMeshDict",
    "system/surfaceFeaturesDict",
    "constant/transportProperties",
    "constant/turbulenceProperties",
]


def fmt_angle(theta: float) -> str:
    """
    Convierte un angulo a string para nombre de carpeta.
    Enteros: cero a la izquierda de dos digitos.  2.5 -> '2p5',  10 -> '10'
    Decimales: punto decimal reemplazado por 'p'.  7.5 -> '7p5', 12.5 -> '12p5'
    """
    if theta == int(theta):
        return f"{int(theta):02d}"
    return str(theta).replace(".", "p")


def rotate_xy(x: float, y: float, theta_deg: float) -> tuple[float, float]:
    c = math.cos(math.radians(theta_deg))
    s = math.sin(math.radians(theta_deg))
    return x * c - y * s, x * s + y * c


def make_probe_content(theta_deg: float) -> str:
    lines = [
        "        // Auto-generated probe locations for building pressure taps",
        "        probeLocations",
        "        (",
    ]
    face_origins = [
        [(PROBE_OFFSET, y) for y in PROBE_COLS],   # cara 1: sotavento
        [(-PROBE_OFFSET, y) for y in PROBE_COLS],  # cara 2: barlovento
        [(x, PROBE_OFFSET) for x in PROBE_COLS],   # cara 3: lateral+
        [(x, -PROBE_OFFSET) for x in PROBE_COLS],  # cara 4: lateral-
    ]
    for face in face_origins:
        for x0, y0 in face:
            xr, yr = rotate_xy(x0, y0, theta_deg)
            for z in PROBE_Z:
                lines.append(f"            ({xr:.5f} {yr:.5f} {z:.5f})")
    lines.append("        );")
    return "\n".join(lines)


def make_stl_binary(theta_deg: float) -> bytes:
    # 4 esquinas en mm (CCW vistas desde arriba en theta=0)
    c1 = rotate_xy(-B_HALF, -B_HALF, theta_deg)
    c2 = rotate_xy(+B_HALF, -B_HALF, theta_deg)
    c3 = rotate_xy(+B_HALF, +B_HALF, theta_deg)
    c4 = rotate_xy(-B_HALF, +B_HALF, theta_deg)
    corners = [c1, c2, c3, c4]

    triangles = []

    # 4 caras laterales (2 triangulos cada una), c1->c2->c3->c4->c1
    for i in range(4):
        A = corners[i]
        B = corners[(i + 1) % 4]
        dx, dy = B[0] - A[0], B[1] - A[1]
        length = math.sqrt(dx * dx + dy * dy)
        nx, ny = dy / length, -dx / length   # normal saliente: giro 90 CW de la arista
        n = (nx, ny, 0.0)
        triangles.append((n, (A[0], A[1], H_MM), (A[0], A[1], 0.0), (B[0], B[1], 0.0)))
        triangles.append((n, (B[0], B[1], H_MM), (A[0], A[1], H_MM), (B[0], B[1], 0.0)))

    # Cara inferior (z=0, normal=(0,0,-1))
    triangles.append(((0.0, 0.0, -1.0),
                      (c2[0], c2[1], 0.0), (c1[0], c1[1], 0.0), (c3[0], c3[1], 0.0)))
    triangles.append(((0.0, 0.0, -1.0),
                      (c1[0], c1[1], 0.0), (c4[0], c4[1], 0.0), (c3[0], c3[1], 0.0)))

    # Cara superior (z=H_MM, normal=(0,0,+1))
    triangles.append(((0.0, 0.0, 1.0),
                      (c2[0], c2[1], H_MM), (c3[0], c3[1], H_MM), (c1[0], c1[1], H_MM)))
    triangles.append(((0.0, 0.0, 1.0),
                      (c1[0], c1[1], H_MM), (c3[0], c3[1], H_MM), (c4[0], c4[1], H_MM)))

    header_str = f"Binary STL TFM ROM building theta={theta_deg} deg"
    header = header_str[:80].encode("ascii").ljust(80, b" ")
    data = header + struct.pack("<I", len(triangles))
    for n, v1, v2, v3 in triangles:
        data += struct.pack("<fff", *n)
        data += struct.pack("<fff", *v1)
        data += struct.pack("<fff", *v2)
        data += struct.pack("<fff", *v3)
        data += struct.pack("<H", 0)
    return data


def make_allrun(theta_deg: float) -> str:
    name = f"ROMCase_theta_{fmt_angle(theta_deg)}"
    return (
        "#!/bin/sh\n"
        f'echo "========================================================="\n'
        f'echo "   ROM Database - theta = {theta_deg} deg"\n'
        f'echo "========================================================="\n'
        "source /opt/openfoam10/etc/bashrc\n"
        "\n"
        "# 1. CLEAN\n"
        "foamListTimes -rm\n"
        "rm -rf constant/polyMesh processor* log.*\n"
        "\n"
        "# 2. BASE MESH\n"
        "blockMesh | tee log.blockMesh\n"
        "\n"
        "# 3. SURFACE FEATURES\n"
        "surfaceFeatures | tee log.surfaceFeatures\n"
        "\n"
        "# 4. SNAPPY HEX MESH\n"
        "snappyHexMesh -overwrite | tee log.snappy\n"
        "checkMesh -allGeometry -allTopology | tee log.checkMesh\n"
        "\n"
        "# 5. DECOMPOSE\n"
        "decomposePar | tee log.decomposePar\n"
        "\n"
        "# 6. SOLVE\n"
        "mpirun -np 10 potentialFoam -initialiseUBCs -parallel | tee log.potentialFoam.10\n"
        "mpirun -np 10 simpleFoam -parallel | tee log.simpleFoam.10\n"
        "\n"
        "# 7. RECONSTRUCT & CLEAN\n"
        "reconstructPar -latestTime | tee log.reconstructPar\n"
        "rm -rf processor*\n"
        "\n"
        f'echo "========================================================="\n'
        f'echo "   theta={theta_deg} DONE"\n'
        f'echo "========================================================="\n'
        "cd ..\n"
        f"zip -r {name} {name}/\n"
    )


def generate_case_zip(theta_deg: float, template_dir: Path, output_dir: Path) -> Path:
    case_name = f"ROMCase_theta_{fmt_angle(theta_deg)}"
    zip_path = output_dir / f"{case_name}.zip"

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        # Ficheros estaticos del plantilla
        for rel in COPY_FILES:
            src = template_dir / Path(rel.replace("/", os.sep))
            if src.exists():
                zf.write(src, f"{case_name}/{rel}")
            else:
                print(f"  WARNING: {rel} no encontrado en plantilla, omitido")

        # Especificos del angulo: fichero de probes
        zf.writestr(
            f"{case_name}/system/probesBuildingPoints.incl",
            make_probe_content(theta_deg),
        )

        # Especificos del angulo: STL
        zf.writestr(
            f"{case_name}/constant/triSurface/building.stl",
            make_stl_binary(theta_deg),
        )

        # Especificos del angulo: Allrun.sh
        zf.writestr(f"{case_name}/Allrun.sh", make_allrun(theta_deg))

    return zip_path


def verify_stl(theta_deg: float) -> tuple:
    """Sanity-check: primer triangulo del STL generado."""
    data = make_stl_binary(theta_deg)
    n_tri = struct.unpack_from("<I", data, 80)[0]
    n = struct.unpack_from("<fff", data, 84)
    v1 = struct.unpack_from("<fff", data, 84 + 12)
    v2 = struct.unpack_from("<fff", data, 84 + 24)
    v3 = struct.unpack_from("<fff", data, 84 + 36)
    return n_tri, n, v1, v2, v3


def main():
    template_dir = DATA_DIR / TEMPLATE_CASE
    output_dir = DATA_DIR

    print(f"Plantilla : {template_dir}")
    print(f"Salida    : {output_dir}")
    print(f"Angulos   : {NEW_ANGLES}")
    print()

    if not template_dir.exists():
        raise FileNotFoundError(f"Caso plantilla no encontrado: {template_dir}")

    for theta in NEW_ANGLES:
        print(f"theta = {theta}°  (carpeta: ROMCase_theta_{fmt_angle(theta)})")
        zip_path = generate_case_zip(theta, template_dir, output_dir)

        n_tri, n, v1, v2, v3 = verify_stl(theta)
        print(f"  STL    : {n_tri} triangulos | tri0 n=({n[0]:.3f},{n[1]:.3f},{n[2]:.3f})")
        print(f"           v1=({v1[0]:.2f},{v1[1]:.2f},{v1[2]:.0f})")

        probe_content = make_probe_content(theta)
        n_probes = probe_content.count("\n            (")
        print(f"  Probes : {n_probes} puntos")
        print(f"  ZIP    : {zip_path.name}  ({zip_path.stat().st_size // 1024} KB)")
        print()

    print("Listo. Copia los ZIPs al cluster y ejecuta Allrun.sh en cada carpeta.")


if __name__ == "__main__":
    main()
