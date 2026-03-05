"""Compara embeddings de voz: Manuela vs Gabriel vs Isabel + LDA-style."""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from itertools import combinations

emb_dir = "data/embeddings"

# Carrega e normaliza
speakers = {}
files = {
    "Manuela": "Manuela_20260302_200743.npy",
    "Gabriel": "Gabriel_20260302_201126.npy",
    "Isabel":  "Isabel_20260302_220708.npy",
}
colors = {"Manuela": "#E91E63", "Gabriel": "#2196F3", "Isabel": "#4CAF50"}
markers = {"Manuela": "o", "Gabriel": "^", "Isabel": "s"}

for name, fname in files.items():
    emb = np.load(f"{emb_dir}/{fname}")
    emb = emb / (np.linalg.norm(emb) + 1e-8)
    speakers[name] = emb

names = list(speakers.keys())
embs = [speakers[n] for n in names]

# Similarity matrix
print("=" * 50)
print("Cosine Similarity Matrix")
print("=" * 50)
sim_matrix = np.zeros((3, 3))
for i, (na, ea) in enumerate(zip(names, embs)):
    for j, (nb, eb) in enumerate(zip(names, embs)):
        sim_matrix[i, j] = float(np.dot(ea, eb))
        if i <= j:
            print(f"  {na:>8} vs {nb:<8}: {sim_matrix[i,j]:.4f}")

# --- Figure 1: Manuela vs Isabel (mesma estrutura do anterior) ---
man, isa = speakers["Manuela"], speakers["Isabel"]
cosine_mi = float(np.dot(man, isa))

fig1, axes = plt.subplots(2, 2, figsize=(14, 10))
fig1.suptitle(f"Manuela vs Isabel — Cosine Similarity: {cosine_mi:.4f}", fontsize=14, fontweight='bold')

ax = axes[0, 0]
dims = np.arange(len(man))
ax.plot(dims, man, alpha=0.7, linewidth=0.5, label="Manuela", color=colors["Manuela"])
ax.plot(dims, isa, alpha=0.7, linewidth=0.5, label="Isabel", color=colors["Isabel"])
ax.set_xlabel("Dimension")
ax.set_ylabel("Value")
ax.set_title("Embedding Values (192-dim ECAPA-TDNN)")
ax.legend()

ax = axes[0, 1]
diff = man - isa
ax.bar(dims, diff, width=1.0, color=np.where(diff > 0, colors["Manuela"], colors["Isabel"]), alpha=0.7)
ax.set_xlabel("Dimension")
ax.set_ylabel("Manuela − Isabel")
ax.set_title("Per-dimension Difference")
ax.axhline(0, color="black", linewidth=0.5)

ax = axes[1, 0]
ax.hist(man, bins=40, alpha=0.6, label="Manuela", color=colors["Manuela"], density=True)
ax.hist(isa, bins=40, alpha=0.6, label="Isabel", color=colors["Isabel"], density=True)
ax.set_xlabel("Embedding Value")
ax.set_ylabel("Density")
ax.set_title("Value Distribution")
ax.legend()

ax = axes[1, 1]
stacked = np.vstack([man, isa])
pca = PCA(n_components=2)
proj = pca.fit_transform(stacked)
ax.scatter(proj[0, 0], proj[0, 1], s=200, color=colors["Manuela"], marker="o", zorder=5, label="Manuela")
ax.scatter(proj[1, 0], proj[1, 1], s=200, color=colors["Isabel"], marker="s", zorder=5, label="Isabel")
ax.plot([proj[0, 0], proj[1, 0]], [proj[0, 1], proj[1, 1]], 'k--', alpha=0.3)
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
ax.set_title("PCA Projection")
ax.legend()

plt.tight_layout()
fig1.savefig("voice_comparison_manuela_isabel.png", dpi=150)
print("Salvo: voice_comparison_manuela_isabel.png")

# --- Figure 2: LDA-style dashboard com todos os 3 speakers ---
fig2, axes2 = plt.subplots(2, 2, figsize=(15, 12))
fig2.suptitle("Voice Embedding Analysis — Manuela / Gabriel / Isabel", fontsize=15, fontweight='bold')

# 2a) Heatmap de similaridade
ax = axes2[0, 0]
im = ax.imshow(sim_matrix, cmap="RdYlGn", vmin=-0.1, vmax=1.0)
ax.set_xticks(range(3))
ax.set_yticks(range(3))
ax.set_xticklabels(names, fontsize=11)
ax.set_yticklabels(names, fontsize=11)
for i in range(3):
    for j in range(3):
        ax.text(j, i, f"{sim_matrix[i,j]:.3f}", ha="center", va="center",
                fontsize=13, fontweight="bold",
                color="white" if sim_matrix[i,j] > 0.6 else "black")
ax.set_title("Cosine Similarity Matrix")
fig2.colorbar(im, ax=ax, fraction=0.046)

# 2b) Embedding overlay (3 speakers)
ax = axes2[0, 1]
for name in names:
    ax.plot(dims, speakers[name], alpha=0.6, linewidth=0.5, label=name, color=colors[name])
ax.set_xlabel("Dimension")
ax.set_ylabel("Value")
ax.set_title("Embedding Values Overlay")
ax.legend()

# 2c) LDA projection (simulated with per-dimension "variance")
# Com apenas 1 amostra por classe, LDA puro não funciona.
# Simulamos gerando vizinhança gaussiana ao redor de cada embedding e aplicando LDA.
ax = axes2[1, 0]
np.random.seed(42)
n_synthetic = 200
noise_std = 0.03
X_synth, y_synth = [], []
for i, name in enumerate(names):
    base = speakers[name]
    samples = base + np.random.randn(n_synthetic, len(base)) * noise_std
    # Re-normaliza cada amostra
    norms = np.linalg.norm(samples, axis=1, keepdims=True)
    samples = samples / (norms + 1e-8)
    X_synth.append(samples)
    y_synth.extend([i] * n_synthetic)

X_synth = np.vstack(X_synth)
y_synth = np.array(y_synth)

lda = LinearDiscriminantAnalysis(n_components=2)
X_lda = lda.fit_transform(X_synth, y_synth)

for i, name in enumerate(names):
    mask = y_synth == i
    ax.scatter(X_lda[mask, 0], X_lda[mask, 1], alpha=0.3, s=15,
               color=colors[name], label=f"{name} (synthetic)")
    # Centro real
    center_lda = lda.transform(speakers[name].reshape(1, -1))
    ax.scatter(center_lda[0, 0], center_lda[0, 1], s=250, color=colors[name],
               marker=markers[name], edgecolors="black", linewidths=1.5, zorder=10)

ax.set_xlabel("LDA Component 1")
ax.set_ylabel("LDA Component 2")
ax.set_title("LDA Projection (synthetic neighborhood)")
ax.legend(fontsize=9)

# 2d) Bar chart de distâncias (1 - similarity) entre pares
ax = axes2[1, 1]
pairs = list(combinations(names, 2))
pair_labels = [f"{a}\nvs\n{b}" for a, b in pairs]
sims = [float(np.dot(speakers[a], speakers[b])) for a, b in pairs]
dists = [1 - s for s in sims]
bar_colors = ["#9C27B0", "#FF9800", "#607D8B"]

bars = ax.bar(pair_labels, dists, color=bar_colors, alpha=0.8, width=0.5)
for bar, s, d in zip(bars, sims, dists):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f"sim={s:.3f}\ndist={d:.3f}", ha="center", va="bottom", fontsize=10)

ax.set_ylabel("Cosine Distance (1 − similarity)")
ax.set_title("Pairwise Voice Distance")
ax.set_ylim(0, max(dists) * 1.35)
ax.axhline(1 - 0.55, color="red", linestyle="--", alpha=0.5, label="Verifier threshold (0.55)")
ax.legend()

plt.tight_layout()
fig2.savefig("voice_lda_dashboard.png", dpi=150)
print("Salvo: voice_lda_dashboard.png")

plt.show()
