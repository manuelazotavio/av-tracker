"""
enroll_speaker.py -- Adds voice embeddings for a known speaker.

Basic usage:
    python enroll_speaker.py "Isabel" audio.wav
    python enroll_speaker.py "Isabel" audio.wav --start 10 --end 40

Usage with multiple files:
    python enroll_speaker.py "Isabel" clip1.wav clip2.wav clip3.wav

The script extracts the ECAPA-TDNN embedding and saves it to data/embeddings/.
Each file generates a separate .npy -- the verifier averages them automatically.

Tips:
  - Use clips of 10-60 seconds with clear voice and no background music.
  - The more clips from different sessions, the better the generalization.
  - Run for each person that should be recognized.
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
    """Extracts audio via ffmpeg, optionally within a time range."""
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
        raise RuntimeError(f"ffmpeg failed:\n{result.stderr.decode()}")
    audio = np.frombuffer(result.stdout, dtype=np.float32).copy()
    if len(audio) == 0:
        raise ValueError("Empty audio -- check the file and time range.")
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
    """Shows embeddings already saved for this name."""
    files = [f for f in os.listdir(EMB_DIR) if f.endswith(".npy") and
             (f == f"{name}.npy" or f.startswith(f"{name}_"))]
    return files


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def main():
    parser = argparse.ArgumentParser(description="Enroll speaker in the voice verifier")
    parser.add_argument("name", help='Speaker name (e.g.: "Isabel")')
    parser.add_argument("files", nargs="+", help="Audio file(s) (WAV, MP4, MP3...)")
    parser.add_argument("--start", type=float, default=0.0,
                        help="Start of segment in seconds (default: 0)")
    parser.add_argument("--end", type=float, default=None,
                        help="End of segment in seconds (default: until the end)")
    parser.add_argument("--min-sim", type=float, default=0.35,
                        help="Minimum similarity between clips to accept as same voice (default: 0.35)")
    args = parser.parse_args()

    os.makedirs(EMB_DIR, exist_ok=True)

    print(f"\nLoading ECAPA-TDNN model...")
    from speechbrain.inference.speaker import EncoderClassifier
    device = "cuda" if torch.cuda.is_available() else "cpu"
    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        run_opts={"device": device}
    )
    print(f"   Device: {device}")

    # Show existing embeddings for this name
    existing = list_existing(args.name)
    if existing:
        print(f"\nExisting embeddings for '{args.name}': {len(existing)} file(s)")
        for f in existing:
            print(f"   - {f}")
    else:
        print(f"\nNo existing embedding for '{args.name}' -- this will be the first.")

    saved = []
    for audio_path in args.files:
        if not os.path.isfile(audio_path):
            print(f"\nFile not found: {audio_path!r}")
            continue

        print(f"\nProcessing: {audio_path}")
        if args.start > 0 or args.end is not None:
            print(f"   Segment: {args.start}s -> {args.end or 'end'}")

        try:
            audio = load_audio_ffmpeg(audio_path, start=args.start, end=args.end)
        except Exception as e:
            print(f"   Error loading audio: {e}")
            continue

        duration = len(audio) / 16000
        print(f"   Duration: {duration:.1f}s  |  Samples: {len(audio)}")
        if duration < 3.0:
            print(f"   WARNING: Segment too short (< 3s) -- embedding quality may be low.")

        emb = extract_embedding(audio, classifier, device)

        # Check consistency with embeddings already saved for this name
        existing_now = list_existing(args.name)
        if existing_now:
            sims = []
            for ef in existing_now:
                prev = np.load(os.path.join(EMB_DIR, ef))
                s = cosine_sim(emb, prev)
                sims.append((ef, s))
            min_s = min(s for _, s in sims)
            max_s = max(s for _, s in sims)
            print(f"   Similarity with previous embeddings: min={min_s:.2f}  max={max_s:.2f}")
            if min_s < args.min_sim:
                low = [(f, s) for f, s in sims if s < args.min_sim]
                print(f"   WARNING: Low similarity with: {', '.join(f'{f}({s:.2f})' for f, s in low)}")
                ans = input(f"   Save anyway? [y/N] ").strip().lower()
                if ans not in ("s", "sim", "y", "yes"):
                    print(f"   Skipped.")
                    continue

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_name = f"{args.name}_{ts}.npy"
        out_path = os.path.join(EMB_DIR, out_name)
        np.save(out_path, emb)
        saved.append(out_name)
        print(f"   Saved: {out_name}")

    if saved:
        all_files = list_existing(args.name)
        print(f"\n'{args.name}' now has {len(all_files)} embedding(s) total.")
        if len(all_files) >= 2:
            # Show the resulting average (what the verifier will use)
            all_embs = [np.load(os.path.join(EMB_DIR, f)) for f in all_files]
            avg = np.mean(all_embs, axis=0)
            avg /= (np.linalg.norm(avg) + 1e-8)
            sims = [cosine_sim(avg, e) for e in all_embs]
            print(f"   Average coherence (avg sim -> centroid): {np.mean(sims):.2f}")
    else:
        print("\nNo embedding saved.")

    print(f"\nTip: restart the tracker to load the new embeddings.")


if __name__ == "__main__":
    main()
