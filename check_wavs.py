import os
from pathlib import Path

wav_root = Path("vox_300_atores/wav")
wavs_per_actor = {}

for actor_dir in wav_root.iterdir():
    if actor_dir.is_dir():
        wavs = list(actor_dir.glob("**/*.wav"))
        wavs_per_actor[actor_dir.name] = len(wavs)

# Get distribution
counts = {}
for actor, count in wavs_per_actor.items():
    if count not in counts:
        counts[count] = 0
    counts[count] += 1

print("WAVs per actor distribution:")
for count in sorted(counts.keys()):
    print(f"  {count} WAV file(s): {counts[count]} actors")

print(f"\nTotal actors: {len(wavs_per_actor)}")
print(f"Actors with >=3 WAVs: {sum(1 for c in wavs_per_actor.values() if c >= 3)}")
