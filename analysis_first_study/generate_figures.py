"""
Regenerate the three result figures from the REAL, corrected Config C
data (post NaN-guard fix, all 8 composite-loss conditions retrained and
pulled from GitHub checkpoint_full_*.jsonl).

Run: python3 generate_figures.py
Outputs:
  fig_ffaith_bars.pdf
  fig_malay_em_f1.pdf
  fig_arabic_failure_modes.pdf
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.size": 10,
    "font.family": "serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

GRAY = "#7f7f7f"
BLUE = "#1f77b4"
GREEN = "#2ca02c"

# ---------------------------------------------------------------------
# Real data, pulled from GitHub experiment_results/checkpoint_full*.jsonl
# ---------------------------------------------------------------------
labels = ["A\n(frozen)", "D\n(full FT)", "B\nQLoRA", "C\nQLoRA",
          "B\nAdaLoRA", "C\nAdaLoRA", "B\nDoRA", "C\nDoRA",
          "B\nVeRA", "C\nVeRA"]
colors = [GRAY, GRAY, BLUE, GREEN, BLUE, GREEN, BLUE, GREEN, BLUE, GREEN]

ar_ffaith = [0.4532, 0.3768, 0.4361, 0.4462, 0.4343, 0.4371, 0.4716, 0.4440, 0.4522, 0.4408]
ms_ffaith = [0.2719, 0.1721, 0.2676, 0.2925, 0.1439, 0.1372, 0.1931, 0.2468, 0.0598, 0.0734]

# ============================== FIGURE 1 ==============================
fig, axes = plt.subplots(1, 2, figsize=(14, 4.2))
x = np.arange(len(labels))

for ax, data, title in zip(axes, [ar_ffaith, ms_ffaith], ["Arabic (XQuAD)", "Malay (Belebele)"]):
    bars = ax.bar(x, data, color=colors, edgecolor="black", linewidth=0.5, width=0.7)
    best_idx = int(np.argmax(data))
    bars[best_idx].set_edgecolor("black")
    bars[best_idx].set_linewidth(1.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel(r"$F_{\mathrm{faith}}$")
    ax.set_title(title, fontsize=11)
    ax.set_ylim(0, max(data) * 1.22)
    for xi, v in zip(x, data):
        ax.text(xi, v + max(data) * 0.02, f"{v:.3f}", ha="center", va="bottom", fontsize=6.5, rotation=90)

legend_handles = [
    plt.Rectangle((0, 0), 1, 1, color=GRAY, label="Frozen / Full FT"),
    plt.Rectangle((0, 0), 1, 1, color=BLUE, label="CE LoRA (Config B)"),
    plt.Rectangle((0, 0), 1, 1, color=GREEN, label="Composite LoRA (Config C)"),
]
fig.legend(handles=legend_handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.06))
plt.tight_layout()
plt.savefig("fig_ffaith_bars.pdf", bbox_inches="tight")
plt.close(fig)
print("Wrote fig_ffaith_bars.pdf")

# ============================== FIGURE 2 ==============================
ms_em = [0.000, 0.000, 0.100, 0.000, 0.000, 0.000, 0.100, 0.000, 0.000, 0.000]
ms_f1 = [0.094, 0.132, 0.224, 0.129, 0.069, 0.026, 0.227, 0.079, 0.022, 0.000]

fig, ax = plt.subplots(figsize=(9, 4.2))
width = 0.35
ax.bar(x - width / 2, ms_em, width, label="EM", color="#d62728", edgecolor="black", linewidth=0.5)
ax.bar(x + width / 2, ms_f1, width, label="F1", color="#9467bd", edgecolor="black", linewidth=0.5)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel("Score")
ax.set_title("Malay (Belebele): Exact Match and F1 by condition")
ax.legend(frameon=False)
for xi, ve, vf in zip(x, ms_em, ms_f1):
    if ve > 0:
        ax.text(xi - width / 2, ve + 0.008, f"{ve:.2f}", ha="center", fontsize=7)
    if vf > 0:
        ax.text(xi + width / 2, vf + 0.008, f"{vf:.2f}", ha="center", fontsize=7)
plt.tight_layout()
plt.savefig("fig_malay_em_f1.pdf", bbox_inches="tight")
plt.close(fig)
print("Wrote fig_malay_em_f1.pdf")

# ============================== FIGURE 3 ==============================
categories = ["Gold answer\nas substring", "Sentinel-token\nleakage\n(<extra_id_N>)", "Topically\nunrelated"]
counts = [0, 28, 22]
pct = [c / 50 * 100 for c in counts]
bar_colors = ["#999999", "#e07b39", "#4c72b0"]

fig, ax = plt.subplots(figsize=(6, 4.2))
bars = ax.bar(categories, counts, color=bar_colors, edgecolor="black", linewidth=0.5, width=0.6)
ax.set_ylabel("Count (of 50 logged generations)")
ax.set_title("Arabic generation failure modes\n(5 examples $\\times$ 10 conditions, incl. corrected Config C)")
for b, c, p in zip(bars, counts, pct):
    ax.text(b.get_x() + b.get_width() / 2, c + 0.6, f"{c}\n({p:.0f}%)", ha="center", fontsize=9)
ax.set_ylim(0, max(counts) * 1.25)
plt.tight_layout()
plt.savefig("fig_arabic_failure_modes.pdf", bbox_inches="tight")
plt.close(fig)
print("Wrote fig_arabic_failure_modes.pdf")
