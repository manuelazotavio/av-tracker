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
            available_actors.append(id_)
    
    logging.info(f"Found {len(available_actors)} available actors in {wav_root}")
    return available_actors, actors

def get_random_wav(actor_id, wav_root):
    """Get a random wav file for a specific actor."""
    actor_dir = Path(wav_root) / actor_id
    wavs = list(actor_dir.glob('**/*.wav'))
    if not wavs:
        return None
    return random.choice(wavs)

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

def mix_audios_escadinha(wav_paths, output_path, sr=16000, solo_duration=5.0):
    """
    Mix multiple audio files following "escadinha" rule:
    - First speaker talks alone for solo_duration seconds
    - Second speaker enters, both talk for solo_duration seconds
    - Third speaker enters, all three talk for solo_duration seconds
    - And so on...
    
    This creates a cascading effect where speakers enter one by one.
    Each speaker talks long enough to complete at least one phrase.
    """
    
    signals = []
    
    # Load all audio files
    for path in wav_paths:
        wav, _ = load_audio(path, sr)
        signals.append(wav)
    
    if not signals:
        return
    
    num_speakers = len(signals)
    solo_samples = int(solo_duration * sr)
    
    # Calculate total duration:
    # speaker1 alone: solo_duration (5s)
    # speaker2 enters: solo_duration (5s, both talking)
    # speaker3 enters: solo_duration (5s, all three talking)
    # ... and so on
    # Total = solo_duration * num_speakers
    # E.g., 2 speakers = 10s, 3 speakers = 15s, etc.
    total_duration = solo_duration * num_speakers
    total_samples = int(total_duration * sr)
    
    # Initialize mixed signal
    mixed = torch.zeros((1, total_samples))
    
    # Add each speaker at their appropriate time
    for speaker_idx, signal in enumerate(signals):
        # This speaker starts at: solo_duration * speaker_idx
        start_sample = speaker_idx * solo_samples
        
        # Extend signal if needed
        if signal.shape[1] < (total_samples - start_sample):
            # Loop the audio to fill the remaining time
            repeats = int((total_samples - start_sample) / signal.shape[1]) + 1
            extended_signal = signal.repeat(1, repeats)
        else:
            extended_signal = signal
        
        # Trim to exact length needed
        end_sample = min(start_sample + extended_signal.shape[1], total_samples)
        segment_length = end_sample - start_sample
        
        mixed[0, start_sample:end_sample] += extended_signal[0, :segment_length]
    
    # Normalize to avoid clipping
    max_val = torch.max(torch.abs(mixed))
    if max_val > 0:
        mixed = mixed / max_val * 0.9
    
    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torchaudio.save(output_path, mixed, sr)
    logging.info(f"Saved mixed audio (escadinha) to {output_path} (Duration: {total_duration}s, Speakers: {num_speakers})")

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

    # Create mixes following escadinha rule
    num_tests_per_group = 50  # 50 tests per group (2,3,4,5 speakers) = 200 total
    
    for num_speakers in [2, 3, 4, 5]:
        print(f"\n{'='*60}")
        print(f"Creating {num_speakers}-speaker mixes (escadinha rule)")
        print(f"{'='*60}")
        
        for test_idx in range(num_tests_per_group):
            selected_ids = random.sample(available_ids, num_speakers)
            wav_files = []
            selected_names = []

            for aid in selected_ids:
                wav = get_random_wav(aid, WAV_ROOT)
                if wav:
                    wav_files.append(wav)
                    name = actor_names[aid]
                    selected_names.append(name)
            
            if len(wav_files) == num_speakers:
                # Save in subdirectory
                output_file = os.path.join(
                    OUTPUT_DIR, 
                    f"mix_{num_speakers}_speakers",
                    f"mix_{num_speakers}_speakers_{test_idx+1:03d}.wav"
                )
                
                mix_audios_escadinha(wav_files, output_file)
                
                # Save ground truth in the same subdirectory
                gt_file = os.path.join(
                    OUTPUT_DIR,
                    f"mix_{num_speakers}_speakers",
                    f"mix_{num_speakers}_speakers_{test_idx+1:03d}_ground_truth.txt"
                )
                
                with open(gt_file, "w") as f:
                    f.write(f"Speakers: {', '.join(selected_names)}\n")
                    f.write(f"Rule: Escadinha (1 second per speaker alone, then cascading)\n")
                    f.write(f"Total duration: {1.0 * num_speakers} seconds\n\n")
                    for i, (name, path) in enumerate(zip(selected_names, wav_files), 1):
                        solo_start = (i-1) * 1.0
                        solo_end = solo_start + 1.0
                        f.write(f"Speaker {i}: {name}\n")
                        f.write(f"  Audio: {path}\n")
                        f.write(f"  Solo time: {solo_start:.1f}s - {solo_end:.1f}s\n")
                        f.write(f"  Active from: {solo_start:.1f}s onwards\n\n")
                
                if (test_idx + 1) % 10 == 0:
                    print(f"  ✓ Created {test_idx + 1}/{num_tests_per_group} mixes")
        
        print(f"  ✅ Completed {num_speakers}-speaker mixes")
    
    print(f"\n{'='*60}")
    print("✅ ALL MIXES CREATED SUCCESSFULLY!")
    print(f"   Total: {len([2,3,4,5]) * num_tests_per_group} audio files")
    print(f"   Location: {OUTPUT_DIR}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
