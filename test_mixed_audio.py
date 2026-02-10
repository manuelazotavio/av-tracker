import sys
sys.path.append('src')
from speaker_identification_with_chunks import extract_speaker_embeddings
from pathlib import Path
import numpy as np

# Test on one file
wav_file = Path('data/mixed_audio/mix_2_speakers/mix_2_speakers_001.wav')

# Extract embeddings for the mixed audio
embeddings = extract_speaker_embeddings(str(wav_file), sr=16000, chunk_duration=2.0)
print(f'Number of chunks extracted: {len(embeddings)}')
if embeddings:
    print(f'Embedding shape: {embeddings[0].shape}')
    # Print ground truth
    gt_file = 'data/mixed_audio/mix_2_speakers/mix_2_speakers_001_ground_truth.txt'
    with open(gt_file, 'r') as f:
        print("\nGround Truth:")
        print(f.read())
else:
    print('No embeddings extracted')
