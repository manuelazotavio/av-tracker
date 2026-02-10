import os
import random
import csv
import torch
import torchaudio
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def load_actors(meta_path, wav_root):
    """Load available actors from CSV and check if they exist in wav_root."""
    actors = {}
    
    # Read metadata
    with open(meta_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            id_ = row['VoxCeleb1 ID']
            name = row['VGGFace1 ID']
            actors[id_] = name

    # Check existence
    available_actors = []
    wav_path = Path(wav_root)
    for id_ in actors:
        actor_dir = wav_path / id_
        if actor_dir.exists() and any(actor_dir.iterdir()):
            # Get the WAV file
            wav_files = list(actor_dir.glob('**/*.wav'))
            if wav_files:
                available_actors.append(id_)
    
    logging.info(f"Found {len(available_actors)} available actors in {wav_root}")
    return available_actors, actors

def load_audio(path, target_sr=16000):
    """Load and resample audio to target sample rate."""
    wav, sr = torchaudio.load(path)
    
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        wav = resampler(wav)
    
    # Convert to mono
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    
    # Normalize energy (RMS to 0.1)
    rms = torch.sqrt(torch.mean(wav**2))
    if rms > 0:
        wav = wav / rms * 0.1
    
    return wav, target_sr

def get_actor_wav(actor_id, wav_root):
    """Get the WAV file for an actor."""
    actor_dir = Path(wav_root) / actor_id
    wav_files = list(actor_dir.glob('**/*.wav'))
    if not wav_files:
        return None
    return wav_files[0]  # Each actor has only 1 WAV file

def mix_audios_escadinha_simple(actor_ids, wav_root, output_path, sr=16000, start_offset=1.0):
    """
    Mix audio files following simple escadinha rule:
    - Ator 1 começa no tempo 0
    - Ator 2 começa no tempo start_offset (1 segundo)
    - Ator 3 começa no tempo 2*start_offset (2 segundos)
    - Etc.
    
    Cada ator fala seu áudio inteiro.
    """
    
    signals = []  # List of (offset, signal) tuples
    
    # Load each actor's audio and assign an offset
    for idx, actor_id in enumerate(actor_ids):
        wav_file = get_actor_wav(actor_id, wav_root)
        if wav_file:
            wav, _ = load_audio(wav_file, sr)
            offset_seconds = idx * start_offset
            offset_samples = int(offset_seconds * sr)
            signals.append((offset_samples, wav))
    
    if not signals:
        return None, None
    
    # Calculate total duration
    # Total = longest audio + last offset
    max_audio_duration = max(signal[1].shape[1] for signal in signals)
    last_offset = signals[-1][0]
    total_samples = last_offset + max_audio_duration
    
    # Initialize mixed signal
    mixed = torch.zeros((1, total_samples))
    
    # Add each speaker at their offset time
    for offset_samples, signal in signals:
        audio_length = signal.shape[1]
        end_sample = offset_samples + audio_length
        mixed[0, offset_samples:end_sample] += signal[0, :]
    
    # Normalize to avoid clipping
    max_val = torch.max(torch.abs(mixed))
    if max_val > 0:
        mixed = mixed / max_val * 0.9
    
    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torchaudio.save(output_path, mixed, sr)
    
    total_duration = total_samples / sr
    logging.info(f"Saved mixed audio to {output_path} (Duration: {total_duration:.2f}s, Speakers: {len(signals)})")
    
    return mixed, total_duration

def main():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    META_PATH = os.path.join(BASE_DIR, "vox1_meta.csv")
    WAV_ROOT = os.path.join(BASE_DIR, "vox_300_atores", "wav")
    OUTPUT_DIR = os.path.join(BASE_DIR, "data", "mixed_audio")

    available_ids, actor_names = load_actors(META_PATH, WAV_ROOT)
    
    if len(available_ids) < 5:
        logging.error("Not enough actors found!")
        return

    # Create subdirectories for each speaker count
    for num_speakers in [2, 3, 4, 5]:
        os.makedirs(os.path.join(OUTPUT_DIR, f"mix_{num_speakers}_speakers"), exist_ok=True)

    # Create mixes following simple escadinha rule
    num_tests_per_group = 50
    start_offset = 1.0  # 1 second between each actor entering
    
    for num_speakers in [2, 3, 4, 5]:
        print(f"\n{'='*60}")
        print(f"Creating {num_speakers}-speaker mixes (simple escadinha)")
        print(f"Rule: Each actor starts {start_offset}s after the previous")
        print(f"{'='*60}")
        
        for test_idx in range(num_tests_per_group):
            selected_ids = random.sample(available_ids, num_speakers)
            selected_names = [actor_names[id_] for id_ in selected_ids]
            
            # Save in subdirectory
            output_file = os.path.join(
                OUTPUT_DIR, 
                f"mix_{num_speakers}_speakers",
                f"mix_{num_speakers}_speakers_{test_idx+1:03d}.wav"
            )
            
            mixed, total_dur = mix_audios_escadinha_simple(
                selected_ids, 
                WAV_ROOT, 
                output_file,
                start_offset=start_offset
            )
            
            if mixed is not None:
                # Save ground truth in the same subdirectory
                gt_file = os.path.join(
                    OUTPUT_DIR,
                    f"mix_{num_speakers}_speakers",
                    f"mix_{num_speakers}_speakers_{test_idx+1:03d}_ground_truth.txt"
                )
                
                with open(gt_file, "w") as f:
                    f.write(f"Speakers: {', '.join(selected_names)}\n")
                    f.write(f"Rule: Simple Escadinha (each speaker enters {start_offset}s after the previous)\n")
                    f.write(f"Total duration: {total_dur:.2f} seconds\n\n")
                    
                    f.write(f"Timeline:\n")
                    for i, name in enumerate(selected_names):
                        start_time = i * start_offset
                        f.write(f"Speaker {i+1} ({name}) enters at {start_time:.1f}s\n")
                    f.write(f"\nDetails:\n")
                    
                    for i, (id_, name) in enumerate(zip(selected_ids, selected_names)):
                        wav_file = get_actor_wav(id_, WAV_ROOT)
                        f.write(f"Speaker {i+1}: {name}\n")
                        f.write(f"  WAV file: {wav_file}\n")
            
            if (test_idx + 1) % 10 == 0:
                print(f"  ✓ Created {test_idx + 1}/{num_tests_per_group} mixes")
        
        print(f"  ✅ Completed {num_speakers}-speaker mixes")
    
    print(f"\n{'='*60}")
    print("✅ ALL MIXES CREATED SUCCESSFULLY!")
    print(f"   Total: {len([2,3,4,5]) * num_tests_per_group} audio files")
    print(f"   Location: {OUTPUT_DIR}")
    print(f"   Rule: Simple escadinha - each speaker enters 1s after previous")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
