"""
plot_gpr_diagram.py
Diagrama del pipeline ROM_GPR: LOO-CV con re-POD paso a paso.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# ── Colores ──────────────────────────────────────────────────
UC3M   = "#005398"
GRAY   = "#646464"
GREEN  = "#006400"
ORANGE = "#C86400"
LIGHT  = "#B4D2F0"
WHITE  = "#FFFFFF"
RED    = "#8B0000"

fig, ax = plt.subplots(figsize=(18, 22))
ax.set_xlim(0, 18)
ax.set_ylim(0, 22)
ax.axis("off")
fig.patch.set_facecolor("#F8F9FA")

# ── Helpers ──────────────────────────────────────────────────
def box(ax, x, y, w, h, label, sublabel="", color=UC3M, fontsize=10, textcolor=WHITE, style="round,pad=0.1"):
    rect = FancyBboxPatch((x - w/2, y - h/2), w, h,
                          boxstyle=style, linewidth=1.5,
                          edgecolor=color, facecolor=color, zorder=3)
    ax.add_patch(rect)
    ax.text(x, y + (0.15 if sublabel else 0), label,
            ha="center", va="center", fontsize=fontsize,
            fontweight="bold", color=textcolor, zorder=4)
    if sublabel:
        ax.text(x, y - 0.22, sublabel,
                ha="center", va="center", fontsize=fontsize - 1.5,
                color=textcolor, zorder=4, style="italic")

def lightbox(ax, x, y, w, h, label, sublabel="", color=LIGHT, fontsize=9.5):
    rect = FancyBboxPatch((x - w/2, y - h/2), w, h,
                          boxstyle="round,pad=0.1", linewidth=1.5,
                          edgecolor=UC3M, facecolor=color, zorder=3)
    ax.add_patch(rect)
    ax.text(x, y + (0.15 if sublabel else 0), label,
            ha="center", va="center", fontsize=fontsize,
            fontweight="bold", color=UC3M, zorder=4)
    if sublabel:
        ax.text(x, y - 0.22, sublabel,
                ha="center", va="center", fontsize=fontsize - 1.5,
                color=GRAY, zorder=4)

def arrow(ax, x1, y1, x2, y2, color=UC3M, lw=1.8):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=lw, mutation_scale=14), zorder=5)

def label_arrow(ax, x, y, text, fontsize=8.5, color=GRAY):
    ax.text(x, y, text, ha="center", va="center",
            fontsize=fontsize, color=color, style="italic", zorder=6)

def section_bg(ax, x, y, w, h, label, color="#E8F0F8"):
    rect = FancyBboxPatch((x, y), w, h,
                          boxstyle="round,pad=0.15", linewidth=1.5,
                          edgecolor=UC3M, facecolor=color,
                          alpha=0.35, zorder=1)
    ax.add_patch(rect)
    ax.text(x + 0.2, y + h - 0.25, label,
            ha="left", va="top", fontsize=9, fontweight="bold",
            color=UC3M, zorder=2)

# ══════════════════════════════════════════════════════════════
# TÍTULO
# ══════════════════════════════════════════════════════════════
ax.text(9, 21.5, "Pipeline ROM_GPR: LOO-CV con re-POD",
        ha="center", va="center", fontsize=15,
        fontweight="bold", color=UC3M)
ax.text(9, 21.0, "Tres fases: comparativa de kernels → barrido de r → modelo final",
        ha="center", va="center", fontsize=10, color=GRAY)

# ══════════════════════════════════════════════════════════════
# BLOQUE 0 — DATOS DE ENTRADA
# ══════════════════════════════════════════════════════════════
section_bg(ax, 0.3, 19.2, 17.4, 1.5, "Datos de entrada", "#E8F0F8")

box(ax, 3.0, 19.95, 3.8, 0.85,
    "ROM_POD_basis.npz",
    "Phi(400×10), A(10×11), mean, angles",
    color=UC3M, fontsize=9.5)

lightbox(ax, 7.5, 19.95, 3.2, 0.85,
         "Normalizar ángulos",
         "θ̃ = θ/50  →  [0.0 ... 1.0]")

box(ax, 12.5, 19.95, 4.0, 0.85,
    "3 Fases de optimización",
    "kernel × r  →  modelo final",
    color=ORANGE, fontsize=9.5)

arrow(ax, 4.9, 19.95, 5.9, 19.95)
arrow(ax, 9.1, 19.95, 10.5, 19.95)

# ══════════════════════════════════════════════════════════════
# BLOQUE 1 — LOO-CV CON RE-POD (función central)
# ══════════════════════════════════════════════════════════════
section_bg(ax, 0.3, 7.8, 17.4, 11.1,
           "Función central: LOO-CV con re-POD  (se ejecuta para cada combinación kernel × r)", "#EEF5EE")

# Encabezado del loop
box(ax, 9.0, 18.55, 5.5, 0.75,
    "LOOP: fold k = 0, 1, ..., 10",
    "Para cada ángulo de test",
    color=GREEN, fontsize=10)

arrow(ax, 9.0, 19.53, 9.0, 18.93)

# ── Paso 1 ───────────────────────────────────────────────────
lightbox(ax, 4.5, 17.4, 6.5, 1.0,
         "PASO 1 — Retirar snapshot k",
         "X_train = mean + Phi @ A[:,train]  (400×10)  |  X_test = mean + Phi @ A[:,k]  (400,)")
arrow(ax, 9.0, 18.18, 9.0, 17.9)

# ── Paso 2 ───────────────────────────────────────────────────
lightbox(ax, 4.5, 16.1, 6.5, 1.0,
         "PASO 2 — Re-POD sobre 10 snapshots",
         "mean_k = media(X_train)  →  SVD(X_train − mean_k)  →  Phi_k(400×r),  A_k(r×10)")
arrow(ax, 4.5, 16.9, 4.5, 16.6)
arrow(ax, 4.5, 15.6, 4.5, 15.35)

# ── Paso 3 ───────────────────────────────────────────────────
lightbox(ax, 4.5, 14.95, 6.5, 0.75,
         "PASO 3 — Error de proyección (cota mínima sin GPR)",
         "α_proj = Phi_kᵀ(X_test − mean_k)   →   ε_proj = ‖X_test − (mean_k + Phi_k @ α_proj)‖ / ‖X_test − mean_k‖")

# ── Paso 4 ───────────────────────────────────────────────────
lightbox(ax, 4.5, 13.7, 6.5, 0.9,
         "PASO 4 — Entrenar r GPRs",
         "Para j=1..r:  GPR_j.fit(θ̃_train, A_k[j,:])   [10 pares × r GPRs]   optimización ML-II con 50 reinicios")
arrow(ax, 4.5, 14.58, 4.5, 14.15)
arrow(ax, 4.5, 13.25, 4.5, 13.0)

# ── Paso 5 ───────────────────────────────────────────────────
lightbox(ax, 4.5, 12.6, 6.5, 0.75,
         "PASO 5 — Predecir en θk",
         "α̂[j] = GPR_j.predict(θ̃_k)   →   Ĉp = mean_k + Phi_k @ α̂   →   ε_total = ‖X_test − Ĉp‖ / ‖X_test − mean_k‖")
arrow(ax, 4.5, 12.13, 4.5, 11.9)

# ── Paso 6 ───────────────────────────────────────────────────
lightbox(ax, 4.5, 11.5, 6.5, 0.75,
         "PASO 6 — Acumular errores",
         "ε_interp_k = ε_total_k − ε_proj_k")

# ── Flecha loop ──────────────────────────────────────────────
ax.annotate("", xy=(9.0, 18.18), xytext=(8.0, 11.5),
            arrowprops=dict(arrowstyle="-|>", color=GREEN,
                            lw=1.8, connectionstyle="arc3,rad=-0.3",
                            mutation_scale=14), zorder=5)
ax.text(10.8, 15.0, "k+1", ha="center", fontsize=9,
        color=GREEN, fontweight="bold", style="italic")

# ── Promedios ────────────────────────────────────────────────
arrow(ax, 4.5, 11.13, 4.5, 10.5)

lightbox(ax, 4.5, 10.1, 6.5, 0.65,
         "Promediar sobre los 11 folds",
         "ε_proj = mean(ε_proj_k)   |   ε_total = mean(ε_total_k)   |   ε_interp = ε_total − ε_proj")

# ── Outputs ──────────────────────────────────────────────────
arrow(ax, 4.5, 9.78, 4.5, 9.4)

box(ax, 2.5, 9.05, 2.2, 0.6,  "ε_proj",  color="#4A7C59", fontsize=10)
box(ax, 4.9, 9.05, 2.2, 0.6,  "ε_interp", color=ORANGE,   fontsize=10)
box(ax, 7.3, 9.05, 2.2, 0.6,  "ε_total",  color=RED,       fontsize=10)

# Separadores de salida
ax.plot([1.3, 1.3], [9.78, 9.05], color=UC3M, lw=1, ls="--", alpha=0.4, zorder=2)
ax.plot([7.7, 7.7], [9.78, 9.05], color=UC3M, lw=1, ls="--", alpha=0.4, zorder=2)

# ══════════════════════════════════════════════════════════════
# FASE 1 — Comparativa de kernels
# ══════════════════════════════════════════════════════════════
section_bg(ax, 0.3, 4.9, 5.3, 2.65, "FASE 1 — Comparativa kernels  (r = 6 fijo)", "#FFF3E0")

arrow(ax, 2.5, 8.75, 2.5, 7.55)

box(ax, 2.9, 7.2, 4.5, 0.6,
    "Ejecutar LOO-CV × 4 kernels",
    color=ORANGE, fontsize=9)

lightbox(ax, 2.9, 6.35, 4.5, 1.0,
         "Resultados:",
         "Matérn 3/2: 0.149 ✓   Matérn 5/2: 0.161\nRBF: 0.177              RatQuad: 0.178",
         fontsize=9)

box(ax, 2.9, 5.35, 4.5, 0.6,
    "→  Kernel ganador: Matérn 3/2",
    color=GREEN, fontsize=9)

# ══════════════════════════════════════════════════════════════
# FASE 2 — Barrido de r
# ══════════════════════════════════════════════════════════════
section_bg(ax, 6.2, 4.9, 5.3, 2.65, "FASE 2 — Barrido de r  (Matérn 3/2 fijo)", "#FFF3E0")

arrow(ax, 4.9, 8.75, 9.0, 7.55)

box(ax, 8.8, 7.2, 4.5, 0.6,
    "Ejecutar LOO-CV  r = 1..9",
    color=ORANGE, fontsize=9)

lightbox(ax, 8.8, 6.35, 4.5, 1.0,
         "ε_proj baja, ε_interp sube, suma ≈ cte desde r=6",
         "r=1: 0.330   r=6: 0.149   r=9: 0.149",
         fontsize=9)

box(ax, 8.8, 5.35, 4.5, 0.6,
    "→  r* = 9  (argmin LOO_total)",
    color=GREEN, fontsize=9)

# ══════════════════════════════════════════════════════════════
# FASE 3 — Modelo final
# ══════════════════════════════════════════════════════════════
section_bg(ax, 12.1, 4.9, 5.5, 2.65, "FASE 3 — Modelo final  (r*=9, Matérn 3/2)", "#FFF3E0")

arrow(ax, 7.3, 8.75, 14.5, 7.55)

box(ax, 14.6, 7.2, 4.5, 0.6,
    "Entrenar 9 GPRs con N=11",
    color=ORANGE, fontsize=9)

lightbox(ax, 14.6, 6.35, 4.5, 1.0,
         "Hiperparámetros ajustados por ML-II:",
         "Modos 1-4: ℓ=0.87..0.14, σn²≈0\nModos 5-9: ℓ=0.05 (límite), σn²>0",
         fontsize=8.5)

box(ax, 14.6, 5.35, 4.5, 0.6,
    "→  ROM_GPR_results.npz",
    color=GREEN, fontsize=9)

# ══════════════════════════════════════════════════════════════
# OUTPUT FINAL
# ══════════════════════════════════════════════════════════════
arrow(ax, 2.9, 5.05, 9.0, 4.35)
arrow(ax, 8.8, 5.05, 9.0, 4.35)
arrow(ax, 14.6, 5.05, 9.0, 4.35)

box(ax, 9.0, 3.9, 7.5, 0.7,
    "9 GPRs entrenados  →  Ĉp(x, θ*) = x̄ + Φ · [α̂₁(θ*),..., α̂₉(θ*)]",
    color=UC3M, fontsize=10)

# ── Nota final ───────────────────────────────────────────────
ax.text(9.0, 3.3,
        "Para predecir en θ* nuevo: normalizar θ̃* = θ*/50  →  cada GPR_j.predict(θ̃*)  →  reconstruir con Φ",
        ha="center", va="center", fontsize=9, color=GRAY, style="italic")

# ══════════════════════════════════════════════════════════════
plt.tight_layout()
plt.savefig("GPR_pipeline_diagram.png", dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.show()
print("Guardado: GPR_pipeline_diagram.png")
