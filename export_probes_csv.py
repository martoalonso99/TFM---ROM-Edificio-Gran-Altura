"""
export_probes_csv.py
Exporta las coordenadas de los 400 probes y sus valores de Cp a CSV,
uno por angulo. Cargable en ParaView con el filtro "Table to Points".

Uso en ParaView:
  1. File > Open > seleccionar probes_theta_XX.csv
  2. Aplicar filtro: Filters > Alphabetical > Table to Points
     - X Column: x,  Y Column: y,  Z Column: z
  3. Colorear por Cp
"""

from pathlib import Path
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
NPZ        = SCRIPT_DIR / "ROM_POD_basis.npz"
OUT_DIR    = SCRIPT_DIR / "probes_csv"
OUT_DIR.mkdir(exist_ok=True)

data         = np.load(str(NPZ))
probe_coords = data["probe_coords"]   # (400, 3, N_ang)
angles       = data["angles"]         # (N_ang,)
Cp_matrix    = data["Phi"] @ data["A"] + data["mean"][:, None]   # (400, N_ang)
UREF         = float(data["Uref"])

FACE_NAMES = (
    ["Sotavento"] * 100 +
    ["Barlovento"] * 100 +
    ["Lateral+"] * 100 +
    ["Lateral-"] * 100
)

for i, theta in enumerate(angles):
    coords = probe_coords[:, :, i]   # (400, 3)
    cp     = Cp_matrix[:, i]         # (400,)

    tag = str(theta).replace(".", "p").replace("p0", "")
    tag = f"{int(theta):02d}" if theta == int(theta) else tag

    out_path = OUT_DIR / f"probes_theta_{tag}.csv"
    with open(out_path, "w") as f:
        f.write("probe,face,x,y,z,Cp\n")
        for j in range(400):
            f.write(f"{j},{FACE_NAMES[j]},"
                    f"{coords[j,0]:.5f},{coords[j,1]:.5f},{coords[j,2]:.5f},"
                    f"{cp[j]:.5f}\n")

    print(f"  theta={theta:5.1f}  ->  {out_path.name}")

print(f"\nCSVs en: {OUT_DIR}")
print("En ParaView: Open CSV > filtro 'Table to Points' (x,y,z) > colorear por Cp")
