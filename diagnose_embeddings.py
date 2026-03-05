"""Diagnóstico de qualidade dos embeddings de voz.

Testa os embeddings salvos contra áudios reais das sessões,
segmentando em janelas de 3s para simular o comportamento do verifier.
"""
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from speechbrain.inference.speaker import EncoderClassifier
import soundfile as sf

# --- Config ---
EMB_DIR = "data/embeddings"
AUDIO_DIR = "realtime_sessions"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WINDOW_SEC = 3.0       # janela de análise (simula segmento do verifier)
HOP_SEC = 1.5          # passo entre janelas
SR = 16000
THRESHOLD = 0.55       # threshold do verifier

# --- Carrega modelo ---
print("Carregando ECAPA-TDNN...")
classifier = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    run_opts={"device": DEVICE}
)

# --- Carrega embeddings salvos ---
stored = {}
for f in os.listdir(EMB_DIR):
    if f.endswith(".npy"):
        name = f.replace("_auto.npy", "").split("_")[0]  # "Manuela_2026..." → "Manuela"
        # Handle timestamp format
        base = os.path.splitext(f)[0]
        if base.endswith("_auto"):
            name = base[:-5]
        else:
            parts = base.split("_")
            if len(parts) >= 3 and parts[-1].isdigit() and parts[-2].isdigit():
                name = "_".join(parts[:-2])
            else:
                name = parts[0]
        emb = np.load(os.path.join(EMB_DIR, f))
        emb = emb / (np.linalg.norm(emb) + 1e-8)
        stored[name] = torch.from_numpy(emb).float().to(DEVICE)
        print(f"  Loaded: {name} ({f})")

print(f"\n{len(stored)} embeddings carregados: {list(stored.keys())}")

# --- Propriedades dos embeddings ---
print("\n" + "=" * 60)
print("PROPRIEDADES DOS EMBEDDINGS")
print("=" * 60)

names = list(stored.keys())
for name in names:
    emb = stored[name].cpu().numpy()
    print(f"\n  {name}:")
    print(f"    Norma L2: {np.linalg.norm(emb):.4f} (ideal: ~1.0)")
    print(f"    Média: {emb.mean():.4f}")
    print(f"    Std: {emb.std():.4f}")
    print(f"    Min/Max: {emb.min():.4f} / {emb.max():.4f}")
    # Effective dimensionality (quantas dimensões contribuem)
    abs_emb = np.abs(emb)
    p = abs_emb / abs_emb.sum()
    entropy = -np.sum(p * np.log(p + 1e-10))
    max_entropy = np.log(len(emb))
    print(f"    Entropia normalizada: {entropy/max_entropy:.3f} (0=concentrado, 1=espalhado)")

# Cross-similarity
print(f"\n  Cross-similarity:")
for i, n1 in enumerate(names):
    for n2 in names[i+1:]:
        sim = torch.nn.functional.cosine_similarity(stored[n1], stored[n2], dim=0).item()
        print(f"    {n1} vs {n2}: {sim:.4f}")

# --- Testa contra áudios recentes ---
print("\n" + "=" * 60)
print("TESTE CONTRA ÁUDIOS DAS SESSÕES")
print("=" * 60)

# Pega os áudios mais recentes
audio_files = sorted([
    f for f in os.listdir(AUDIO_DIR)
    if f.startswith("audio_") and f.endswith(".wav")
])[-5:]  # últimos 5

all_results = {name: [] for name in names}  # name -> list of (score, audio_file, window_idx)

for audio_file in audio_files:
    path = os.path.join(AUDIO_DIR, audio_file)
    audio, sr = sf.read(path)
    if sr != SR:
        import torchaudio
        audio_t = torch.from_numpy(audio).float().unsqueeze(0)
        audio_t = torchaudio.functional.resample(audio_t, sr, SR)
        audio = audio_t.squeeze().numpy()
        sr = SR

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    duration = len(audio) / SR
    window_samples = int(WINDOW_SEC * SR)
    hop_samples = int(HOP_SEC * SR)

    n_windows = max(1, int((len(audio) - window_samples) / hop_samples) + 1)

    print(f"\n  {audio_file} ({duration:.1f}s, {n_windows} janelas)")

    for w in range(n_windows):
        start = w * hop_samples
        end = min(start + window_samples, len(audio))
        chunk = audio[start:end]

        if len(chunk) < SR * 0.5:
            continue

        # Normaliza
        mx = np.abs(chunk).max()
        if mx < 1e-6:
            continue
        chunk = chunk / mx

        signal = torch.from_numpy(chunk).float().to(DEVICE).unsqueeze(0)
        with torch.no_grad():
            emb = classifier.encode_batch(signal).squeeze()

        # Compara com todos
        scores = {}
        for name, ref in stored.items():
            sim = torch.nn.functional.cosine_similarity(emb, ref, dim=0).item()
            scores[name] = sim
            all_results[name].append(sim)

        best = max(scores, key=scores.get)
        best_score = scores[best]
        indicator = "✅" if best_score >= THRESHOLD else "❌"
        scores_str = " | ".join(f"{n}: {s:.2f}" for n, s in sorted(scores.items(), key=lambda x: -x[1]))
        if w % 3 == 0:  # print every 3rd window to avoid flooding
            print(f"    [{w:2d}] {indicator} {scores_str}")

# --- Estatísticas por pessoa ---
print("\n" + "=" * 60)
print("ESTATÍSTICAS DE MATCHING (todas as janelas)")
print("=" * 60)

stats = {}
for name in names:
    scores = all_results[name]
    if scores:
        arr = np.array(scores)
        stats[name] = {
            "mean": arr.mean(),
            "std": arr.std(),
            "min": arr.min(),
            "max": arr.max(),
            "pct_above_thr": (arr >= THRESHOLD).mean() * 100,
            "n": len(arr)
        }
        s = stats[name]
        print(f"\n  {name} ({s['n']} janelas):")
        print(f"    Média: {s['mean']:.3f}  Std: {s['std']:.3f}")
        print(f"    Min: {s['min']:.3f}  Max: {s['max']:.3f}")
        print(f"    Acima do threshold ({THRESHOLD}): {s['pct_above_thr']:.1f}%")

        if s['mean'] < THRESHOLD:
            print(f"    ⚠️  EMBEDDING FRACO — média abaixo do threshold!")
        elif s['std'] > 0.15:
            print(f"    ⚠️  EMBEDDING INSTÁVEL — variância alta!")
        else:
            print(f"    ✅  EMBEDDING OK")

# --- Gráficos ---
fig, axes = plt.subplots(2, 2, figsize=(15, 11))
fig.suptitle("Voice Embedding Diagnostic", fontsize=14, fontweight="bold")

colors = {"Manuela": "#E91E63", "Gabriel": "#2196F3", "Isabel": "#4CAF50"}

# 1) Histograma de scores por pessoa
ax = axes[0, 0]
for name in names:
    scores = all_results[name]
    if scores:
        c = colors.get(name, "gray")
        ax.hist(scores, bins=30, alpha=0.5, label=name, color=c, density=True)
ax.axvline(THRESHOLD, color="red", linestyle="--", label=f"Threshold ({THRESHOLD})")
ax.set_xlabel("Cosine Similarity")
ax.set_ylabel("Density")
ax.set_title("Score Distribution per Person")
ax.legend()

# 2) Box plot
ax = axes[0, 1]
data = [all_results[n] for n in names]
bp = ax.boxplot(data, labels=names, patch_artist=True)
for patch, name in zip(bp['boxes'], names):
    patch.set_facecolor(colors.get(name, "gray"))
    patch.set_alpha(0.6)
ax.axhline(THRESHOLD, color="red", linestyle="--", label=f"Threshold ({THRESHOLD})")
ax.set_ylabel("Cosine Similarity")
ax.set_title("Score Distribution (Box Plot)")
ax.legend()

# 3) Scores ao longo do tempo (último áudio)
ax = axes[1, 0]
if audio_files:
    last_audio = os.path.join(AUDIO_DIR, audio_files[-1])
    audio, sr = sf.read(last_audio)
    if sr != SR:
        audio_t = torch.from_numpy(audio).float().unsqueeze(0)
        audio_t = torchaudio.functional.resample(audio_t, sr, SR)
        audio = audio_t.squeeze().numpy()
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    window_samples = int(WINDOW_SEC * SR)
    hop_samples = int(HOP_SEC * SR)
    n_win = max(1, int((len(audio) - window_samples) / hop_samples) + 1)

    time_scores = {n: [] for n in names}
    times = []
    for w in range(n_win):
        start = w * hop_samples
        end = min(start + window_samples, len(audio))
        chunk = audio[start:end]
        if len(chunk) < SR * 0.5 or np.abs(chunk).max() < 1e-6:
            continue
        chunk = chunk / np.abs(chunk).max()
        signal = torch.from_numpy(chunk).float().to(DEVICE).unsqueeze(0)
        with torch.no_grad():
            emb = classifier.encode_batch(signal).squeeze()
        times.append(start / SR)
        for name, ref in stored.items():
            sim = torch.nn.functional.cosine_similarity(emb, ref, dim=0).item()
            time_scores[name].append(sim)

    for name in names:
        c = colors.get(name, "gray")
        ax.plot(times, time_scores[name], label=name, color=c, alpha=0.7)
    ax.axhline(THRESHOLD, color="red", linestyle="--", alpha=0.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Cosine Similarity")
    ax.set_title(f"Scores over Time — {audio_files[-1]}")
    ax.legend()

# 4) Resumo: barras de média ± std
ax = axes[1, 1]
means = [stats[n]["mean"] if n in stats else 0 for n in names]
stds = [stats[n]["std"] if n in stats else 0 for n in names]
bars = ax.bar(names, means, yerr=stds, capsize=8,
              color=[colors.get(n, "gray") for n in names], alpha=0.7)
ax.axhline(THRESHOLD, color="red", linestyle="--", label=f"Threshold ({THRESHOLD})")
for bar, m, s in zip(bars, means, stds):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + s + 0.02,
            f"{m:.3f}±{s:.3f}", ha="center", fontsize=10)
ax.set_ylabel("Cosine Similarity")
ax.set_title("Mean ± Std per Embedding")
ax.set_ylim(0, 1.0)
ax.legend()

plt.tight_layout()
plt.savefig("embedding_diagnostic.png", dpi=150)
print(f"\nSalvo: embedding_diagnostic.png")
plt.show()
