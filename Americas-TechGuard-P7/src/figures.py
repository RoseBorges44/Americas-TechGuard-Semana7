"""figures.py — Gera todas as figuras de evidência a partir de outputs/arrays.npz."""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

OUT = os.path.join(os.path.dirname(__file__), "..", "outputs")
FIG = os.path.join(OUT, "figures")
os.makedirs(FIG, exist_ok=True)
WET = 0.03

A = np.load(os.path.join(OUT, "arrays.npz"))
R = json.load(open(os.path.join(OUT, "results.json")))
dem, imp, dr, hist = A["dem"], A["imp"], A["dr"], A["history"]

# Paleta de profundidade (estilo do artigo: tons de azul por faixa)
BINS = [0, 0.03, 0.05, 0.15, 0.50, 10.0]
COLORS = ["#f7fbff", "#c6dbef", "#6baed6", "#2171b5", "#08306b"]
cmap = ListedColormap(COLORS); norm = BoundaryNorm(BINS, cmap.N)

def depth_ax(ax, arr, title):
    im = ax.imshow(arr, cmap=cmap, norm=norm)
    ax.set_title(title, fontsize=9); ax.set_xticks([]); ax.set_yticks([])
    return im

# ---------- Fig 1: Área de estudo (terreno sintético Blumenau) ----------
fig, axs = plt.subplots(1, 3, figsize=(10, 3.4))
im0 = axs[0].imshow(dem * 15.0, cmap="terrain"); axs[0].set_title("DEM sintético (m) — vale do Itajaí")
plt.colorbar(im0, ax=axs[0], fraction=0.046)
im1 = axs[1].imshow(imp, cmap="OrRd"); axs[1].set_title("Impermeabilização (núcleo urbano)")
plt.colorbar(im1, ax=axs[1], fraction=0.046)
im2 = axs[2].imshow(dr, cmap="Blues"); axs[2].set_title("Bueiros / drenagem")
plt.colorbar(im2, ax=axs[2], fraction=0.046)
for a in axs: a.set_xticks([]); a.set_yticks([])
fig.suptitle("Figura 1 — Fatores espaciais de entrada (área de estudo estilizada, Blumenau/SC)", fontsize=10)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig1_area_estudo.png"), dpi=130); plt.close()

# ---------- Fig 2: Evolução temporal predito vs MIKE-surrogate ----------
ev = 2
pred, gt = A[f"pred_{ev}"], A[f"gt_{ev}"]
T = pred.shape[0]
steps = [int(T*0.25), int(T*0.45), int(T*0.65), T-1]
fig, axs = plt.subplots(2, len(steps), figsize=(11, 5.4))
for j, s in enumerate(steps):
    depth_ax(axs[0, j], gt[s], f"Referência (CA)  t={s*2} min")
    im = depth_ax(axs[1, j], pred[s], f"U-RNN  t={s*2} min")
axs[0, 0].set_ylabel("Referência", fontsize=10)
axs[1, 0].set_ylabel("U-RNN", fontsize=10)
cb = fig.colorbar(im, ax=axs, fraction=0.025, pad=0.02, boundaries=BINS, ticks=BINS)
cb.set_label("Profundidade (m)")
fig.suptitle("Figura 2 — Evolução espaço-temporal da inundação (evento de teste): U-RNN vs referência", fontsize=11)
fig.savefig(os.path.join(FIG, "fig2_evolucao_temporal.png"), dpi=130, bbox_inches="tight"); plt.close()

# ---------- Fig 3: Hidrogramas em pontos selecionados ----------
gt_peak = gt.max(0)
# escolhe 3 pontos molhados com diferentes profundidades
flat = np.argsort(gt_peak.ravel())[::-1]
pts = []
seen = []
for idx in flat:
    y, x = divmod(idx, gt_peak.shape[1])
    if all((abs(y-py)+abs(x-px)) > 6 for py, px in seen):
        seen.append((y, x)); pts.append((y, x))
    if len(pts) == 3: break
tt = np.arange(T) * 2
fig, axs = plt.subplots(1, 3, figsize=(11, 3.2))
for k, (y, x) in enumerate(pts):
    axs[k].plot(tt, gt[:, y, x], "r--", label="Referência (CA)")
    axs[k].plot(tt, pred[:, y, x], "orange", label="U-RNN")
    axs[k].set_title(f"P{k+1}  (linha {y}, col {x})", fontsize=9)
    axs[k].set_xlabel("Tempo (min)"); axs[k].grid(alpha=0.3)
axs[0].set_ylabel("Profundidade (m)"); axs[0].legend(fontsize=8)
fig.suptitle("Figura 3 — Hidrogramas urbanos nowcasted pelo U-RNN vs referência", fontsize=11)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig3_hidrogramas.png"), dpi=130); plt.close()

# ---------- Fig 4: Scatter PR² (profundidade de pico) ----------
pred_peak = pred.max(0)
m = gt_peak > 0.001
x = gt_peak[m]; y = pred_peak[m]
ss_res = np.sum((y-x)**2); ss_tot = np.sum((x-x.mean())**2)+1e-9
pr2 = 1 - ss_res/ss_tot
fig, ax = plt.subplots(figsize=(4.6, 4.4))
ax.scatter(x, y, s=4, alpha=0.3, color="#2171b5")
lim = max(x.max(), y.max())*1.05
ax.plot([0, lim], [0, lim], "k--", lw=1)
ax.set_xlim(0, lim); ax.set_ylim(0, lim)
ax.set_xlabel("Referência (CA) — profundidade de pico (m)")
ax.set_ylabel("U-RNN — profundidade de pico (m)")
ax.set_title(f"Figura 4 — Consistência de pico  PR² = {pr2:.3f}")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig4_scatter_pr2.png"), dpi=130); plt.close()

# ---------- Fig 5: Mapa CSI (TP/FP/FN) extensão máxima ----------
prob = A[f"prob_{ev}"]
pw = prob.max(0) > 0.5
gw = gt_peak > WET
cat = np.zeros_like(dem)  # 0 seco
cat[np.logical_and(pw, gw)] = 3    # TP
cat[np.logical_and(pw, ~gw)] = 2   # FP
cat[np.logical_and(~pw, gw)] = 1   # FN
ccmap = ListedColormap(["#eeeeee", "#d62728", "#ff7f0e", "#1f77b4"])
fig, ax = plt.subplots(figsize=(4.8, 4.6))
ax.imshow(cat, cmap=ccmap, vmin=0, vmax=3)
tp=(cat==3).sum(); fp=(cat==2).sum(); fn=(cat==1).sum()
csi = tp/(tp+fp+fn)
from matplotlib.patches import Patch
leg = [Patch(color="#1f77b4", label="TP (acerto molhado)"),
       Patch(color="#ff7f0e", label="FP (falso positivo)"),
       Patch(color="#d62728", label="FN (falso negativo)")]
ax.legend(handles=leg, fontsize=8, loc="upper right")
ax.set_title(f"Figura 5 — Extensão máxima de inundação  CSI = {csi:.3f}")
ax.set_xticks([]); ax.set_yticks([])
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig5_csi_extensao.png"), dpi=130); plt.close()

# ---------- Fig 6: Curva de perda (treino) ----------
fig, ax = plt.subplots(figsize=(6, 3.4))
ax.plot(np.arange(1, len(hist)+1), hist, color="#08519c")
ax.set_xlabel("Época"); ax.set_ylabel("Perda (multi-tarefa)")
ax.set_title("Figura 6 — Convergência do treino (paradigma SWP, warm restarts a cada bloco)")
ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig6_curva_perda.png"), dpi=130); plt.close()

# ---------- Fig 7: SWP — memória de gradiente vs K (conceitual) ----------
Ks = np.array([1, 4, 8, 16, 28, 48])
mem = Ks / 48.0 * 100   # % da memória do BPTT completo (K=T=48)
fig, ax = plt.subplots(figsize=(6, 3.6))
ax.plot(Ks, mem, "o-", color="#e6550d")
ax.axvline(16, ls="--", color="gray")
ax.text(16.5, 70, "K=16 usado\n(este trabalho)", fontsize=8)
ax.set_xlabel("Tamanho da janela K (passos com gradiente)")
ax.set_ylabel("Memória de gradiente (% do BPTT completo)")
ax.set_title("Figura 7 — Princípio do SWP: memória cresce ~linear com K")
ax.grid(alpha=0.3)
red = 48/16
ax.text(28, 20, f"K=16 → ~{red:.0f}× menos memória\nque o BPTT completo (K=T=48)", fontsize=8,
        bbox=dict(boxstyle="round", fc="#fff3e0"))
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig7_swp_memoria.png"), dpi=130); plt.close()

# ---------- Fig 8: Métricas por evento ----------
pe = R["per_event"]
csis = [m["CSI"] for m in pe]; pr2s = [m["PR2"] for m in pe]
xpos = np.arange(len(pe))
fig, ax = plt.subplots(figsize=(6.4, 3.6))
ax.bar(xpos-0.2, csis, 0.4, label="CSI", color="#3182bd")
ax.bar(xpos+0.2, pr2s, 0.4, label="PR²", color="#e6550d")
ax.set_xticks(xpos); ax.set_xticklabels([f"ev{ i }" for i in xpos])
ax.set_ylim(0, 1); ax.axhline(R["metrics_mean"]["CSI"], ls=":", color="#3182bd")
ax.axhline(R["metrics_mean"]["PR2"], ls=":", color="#e6550d")
ax.set_ylabel("Métrica"); ax.legend(fontsize=8)
ax.set_title(f"Figura 8 — Métricas por evento (teste)  |  média CSI={R['metrics_mean']['CSI']:.2f}, PR²={R['metrics_mean']['PR2']:.2f}")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig8_metricas_evento.png"), dpi=130); plt.close()

print("Figuras geradas em outputs/figures/:")
for f in sorted(os.listdir(FIG)):
    print("  ", f)
