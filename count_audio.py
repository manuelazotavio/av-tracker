#!/usr/bin/env python3
import os
import glob

def count_audio_files():
    total_count = 0
    folders = ['data/voxceleb2', 'data/voxceleb_females', 'data/voxceleb_females_limited']

    for folder in folders:
        if os.path.exists(folder):
            wav_files = glob.glob(os.path.join(folder, '**', '*.wav'), recursive=True)
            print(f'{folder}: {len(wav_files)} arquivos WAV')
            total_count += len(wav_files)

    # Verificar também outros diretórios
    other_dirs = ['wav', 'vox_100_atores/wav']
    for folder in other_dirs:
        if os.path.exists(folder):
            wav_files = glob.glob(os.path.join(folder, '**', '*.wav'), recursive=True)
            print(f'{folder}: {len(wav_files)} arquivos WAV')
            total_count += len(wav_files)

    print(f'\nTotal de arquivos WAV encontrados: {total_count}')
    return total_count

if __name__ == "__main__":
    count_audio_files()