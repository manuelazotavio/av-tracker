"""
enroll_speaker.py — Adiciona embeddings de voz para um falante conhecido.

Uso básico:
    python enroll_speaker.py "Isabel" audio.wav
    python enroll_speaker.py "Isabel" audio.wav --start 10 --end 40

Uso com múltiplos arquivos:
    python enroll_speaker.py "Isabel" clip1.wav clip2.wav clip3.wav

O script extrai o embedding ECAPA-TDNN e salva em data/embeddings/.
Cada arquivo gera um .npy separado — o verificador faz a média automaticamente.

Dicas:
  • Use trechos de 10-60 segundos com voz clara e sem música de fundo.
  • Quanto mais clipes de sessões diferentes, melhor a generalização.
  • Execute para cada pessoa que deve ser reconhecida.
"""

import os
import sys
import argparse
import subprocess
import tempfile
import numpy as np
import torch
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EMB_DIR = os.path.join(BASE_DIR, "data", "embeddings")


def load_audio_ffmpeg(path: str, start: float = 0.0, end: float = None, sr: int = 16000) -> np.ndarray:
    """Extrai áudio via ffmpeg, opcionalmente num intervalo de tempo."""
    filters = [f"aresample={sr}", "aformat=sample_fmts=flt:channel_layouts=mono"]
    ffmpeg_cmd = ["ffmpeg", "-y"]
    if start > 0:
        ffmpeg_cmd += ["-ss", str(start)]
    ffmpeg_cmd += ["-i", path]
    if end is not None:
        duration = end - start
        ffmpeg_cmd += ["-t", str(duration)]
    ffmpeg_cmd += ["-af", ",".join(filters), "-f", "f32le", "pipe:1"]

    result = subprocess.run(ffmpeg_cmd, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg falhou:\n{result.stderr.decode()}")
    audio = np.frombuffer(result.stdout, dtype=np.float32).copy()
    if len(audio) == 0:
        raise ValueError("Áudio vazio — verifique o arquivo e o intervalo de tempo.")
    return audio


def extract_embedding(audio: np.ndarray, classifier, device: str) -> np.ndarray:
    signal = torch.from_numpy(audio).float().to(device)
    if signal.dim() == 1:
        signal = signal.unsqueeze(0)
    with torch.no_grad():
        emb = classifier.encode_batch(signal).squeeze().cpu().numpy()
    norm = np.linalg.norm(emb)
    return emb / norm if norm > 0 else emb


def list_existing(name: str):
    """Mostra embeddings já salvos para este nome."""
    files = [f for f in os.listdir(EMB_DIR) if f.endswith(".npy") and
             (f == f"{name}.npy" or f.startswith(f"{name}_"))]
    return files


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def main():
    parser = argparse.ArgumentParser(description="Enrolar falante no verificador de voz")
    parser.add_argument("name", help='Nome do falante (ex: "Isabel")')
    parser.add_argument("files", nargs="+", help="Arquivo(s) de áudio (WAV, MP4, MP3…)")
    parser.add_argument("--start", type=float, default=0.0,
                        help="Início do trecho em segundos (padrão: 0)")
    parser.add_argument("--end", type=float, default=None,
                        help="Fim do trecho em segundos (padrão: até o final)")
    parser.add_argument("--min-sim", type=float, default=0.35,
                        help="Similaridade mínima entre clipes para aceitar como mesma voz (padrão: 0.35)")
    args = parser.parse_args()

    os.makedirs(EMB_DIR, exist_ok=True)

    print(f"\n🧠 Carregando modelo ECAPA-TDNN...")
    from speechbrain.inference.speaker import EncoderClassifier
    device = "cuda" if torch.cuda.is_available() else "cpu"
    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        run_opts={"device": device}
    )
    print(f"   Device: {device}")

    # Mostra embeddings existentes para este nome
    existing = list_existing(args.name)
    if existing:
        print(f"\n📂 Embeddings existentes para '{args.name}': {len(existing)} arquivo(s)")
        for f in existing:
            print(f"   • {f}")
    else:
        print(f"\n📂 Nenhum embedding existente para '{args.name}' — será o primeiro.")

    saved = []
    for audio_path in args.files:
        if not os.path.isfile(audio_path):
            print(f"\n❌ Arquivo não encontrado: {audio_path!r}")
            continue

        print(f"\n🎵 Processando: {audio_path}")
        if args.start > 0 or args.end is not None:
            print(f"   Trecho: {args.start}s → {args.end or 'fim'}")

        try:
            audio = load_audio_ffmpeg(audio_path, start=args.start, end=args.end)
        except Exception as e:
            print(f"   ❌ Erro ao carregar áudio: {e}")
            continue

        duration = len(audio) / 16000
        print(f"   Duração: {duration:.1f}s  |  Amostras: {len(audio)}")
        if duration < 3.0:
            print(f"   ⚠️  Trecho muito curto (< 3s) — qualidade do embedding pode ser baixa.")

        emb = extract_embedding(audio, classifier, device)

        # Verifica consistência com embeddings já salvos para este nome
        existing_now = list_existing(args.name)
        if existing_now:
            sims = []
            for ef in existing_now:
                prev = np.load(os.path.join(EMB_DIR, ef))
                s = cosine_sim(emb, prev)
                sims.append((ef, s))
            min_s = min(s for _, s in sims)
            max_s = max(s for _, s in sims)
            print(f"   Similaridade com embeddings anteriores: min={min_s:.2f}  max={max_s:.2f}")
            if min_s < args.min_sim:
                low = [(f, s) for f, s in sims if s < args.min_sim]
                print(f"   ⚠️  Baixa similaridade com: {', '.join(f'{f}({s:.2f})' for f, s in low)}")
                ans = input(f"   Salvar mesmo assim? [s/N] ").strip().lower()
                if ans not in ("s", "sim", "y", "yes"):
                    print(f"   ⏭️  Pulado.")
                    continue

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_name = f"{args.name}_{ts}.npy"
        out_path = os.path.join(EMB_DIR, out_name)
        np.save(out_path, emb)
        saved.append(out_name)
        print(f"   ✅ Salvo: {out_name}")

    if saved:
        all_files = list_existing(args.name)
        print(f"\n🎉 '{args.name}' agora tem {len(all_files)} embedding(s) no total.")
        if len(all_files) >= 2:
            # Mostra a média resultante (o que o verificador vai usar)
            all_embs = [np.load(os.path.join(EMB_DIR, f)) for f in all_files]
            avg = np.mean(all_embs, axis=0)
            avg /= (np.linalg.norm(avg) + 1e-8)
            sims = [cosine_sim(avg, e) for e in all_embs]
            print(f"   Coerência da média (sim média → centróide): {np.mean(sims):.2f}")
    else:
        print("\n⚠️  Nenhum embedding salvo.")

    print(f"\n💡 Dica: reinicie o tracker para carregar os novos embeddings.")


if __name__ == "__main__":
    main()
