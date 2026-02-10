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

def get_multiple_wavs(actor_id, wav_root, num_wavs=3):
    """Get multiple different wav files for a specific actor without repetition."""
    actor_dir = Path(wav_root) / actor_id
    wavs = list(actor_dir.glob('**/*.wav'))
    if not wavs:
        return []
    
    # Get unique files, up to num_wavs
    num_to_get = min(num_wavs, len(wavs))
    return random.sample(wavs, num_to_get)

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

def concatenate_audios(wav_paths, target_sr=16000):
    """
    Concatenate multiple audio files from the same actor.
    Each audio is a different sentence/phrase - no repetition.
    """
    if not wav_paths:
        return None
    
    audios = []
    for path in wav_paths:
        wav, _ = load_audio(path, target_sr)
        audios.append(wav)
    
    # Concatenate all audios
    concatenated = torch.cat(audios, dim=1)
    
    return concatenated

def mix_audios_escadinha(wav_paths, output_path, sr=16000, solo_duration=5.0):
    """
    Mix multiple audio files following "escadinha" rule WITHOUT REPETITION:
    - First speaker talks alone for solo_duration seconds (different sentences)
    - Second speaker enters, both talk for solo_duration seconds
    - Third speaker enters, all three talk for solo_duration seconds
    - And so on...
    
    Each speaker has multiple different phrases (no looping of same audio).
    """
    
    signals = []
    
    # Load all audio files and concatenate them (no repetition)
    for path_list in wav_paths:
        if isinstance(path_list, list):
            # Multiple paths for one actor
            concatenated = concatenate_audios(path_list, sr)
        else:
            # Single path
            concatenated, _ = load_audio(path_list, sr)
        
        if concatenated is not None:
            signals.append(concatenated)
    
    if not signals:
        return
    
    num_speakers = len(signals)
    solo_samples = int(solo_duration * sr)
    
    # Calculate total duration
    total_duration = solo_duration * num_speakers
    total_samples = int(total_duration * sr)
    
    # Initialize mixed signal
    mixed = torch.zeros((1, total_samples))
    
    # Add each speaker at their appropriate time
    for speaker_idx, signal in enumerate(signals):
        # This speaker starts at: solo_duration * speaker_idx
        start_sample = speaker_idx * solo_samples
        
        # Check if we have enough audio
        end_sample = min(start_sample + solo_samples, total_samples)
        segment_length = end_sample - start_sample
        
        # Trim or use what we have
        if signal.shape[1] >= segment_length:
            # We have enough audio - just trim it
            audio_segment = signal[0, :segment_length]
        else:
            # We don't have enough audio - use what we have and pad with silence
            audio_segment = signal[0, :signal.shape[1]]
            # Pad with silence
            padding = segment_length - audio_segment.shape[0]
            if padding > 0:
                audio_segment = torch.cat([audio_segment, torch.zeros(padding)])
        
        mixed[0, start_sample:end_sample] += audio_segment
    
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

    # Create mixes following escadinha rule WITHOUT repetition
    num_tests_per_group = 50
    
    for num_speakers in [2, 3, 4, 5]:
        print(f"\n{'='*60}")
        print(f"Creating {num_speakers}-speaker mixes (escadinha rule, NO repetition)")
        print(f"{'='*60}")
        
        for test_idx in range(num_tests_per_group):
            selected_ids = random.sample(available_ids, num_speakers)
            wav_file_lists = []
            selected_names = []

            for aid in selected_ids:
                # Get 3 different audio files per actor (will be concatenated)
                wav_files = get_multiple_wavs(aid, WAV_ROOT, num_wavs=3)
                if wav_files:
                    wav_file_lists.append(wav_files)
                    name = actor_names[aid]
                    selected_names.append(name)
            
            if len(wav_file_lists) == num_speakers:
                # Save in subdirectory
                output_file = os.path.join(
                    OUTPUT_DIR, 
                    f"mix_{num_speakers}_speakers",
                    f"mix_{num_speakers}_speakers_{test_idx+1:03d}.wav"
                )
                
                mix_audios_escadinha(wav_file_lists, output_file)
                
                # Save ground truth in the same subdirectory
                gt_file = os.path.join(
                    OUTPUT_DIR,
                    f"mix_{num_speakers}_speakers",
                    f"mix_{num_speakers}_speakers_{test_idx+1:03d}_ground_truth.txt"
                )
                
                with open(gt_file, "w") as f:
                    f.write(f"Speakers: {', '.join(selected_names)}\n")
                    f.write(f"Rule: Escadinha (5 seconds per speaker alone, then cascading)\n")
                    f.write(f"Total duration: {5.0 * num_speakers} seconds\n")
                    f.write(f"NO REPETITION: Each speaker has multiple different phrases\n\n")
                    for i, (name, paths) in enumerate(zip(selected_names, wav_file_lists), 1):
                        solo_start = (i-1) * 5.0
                        solo_end = solo_start + 5.0
                        f.write(f"Speaker {i}: {name}\n")
                        for j, path in enumerate(paths, 1):
                            f.write(f"  Phrase {j}: {path}\n")
                        f.write(f"  Solo time: {solo_start:.1f}s - {solo_end:.1f}s\n")
                        f.write(f"  Active from: {solo_start:.1f}s onwards\n\n")
                
                if (test_idx + 1) % 10 == 0:
                    print(f"  ✓ Created {test_idx + 1}/{num_tests_per_group} mixes")
        
        print(f"  ✅ Completed {num_speakers}-speaker mixes")
    
    print(f"\n{'='*60}")
    print("✅ ALL MIXES CREATED SUCCESSFULLY!")
    print(f"   Total: {len([2,3,4,5]) * num_tests_per_group} audio files")
    print(f"   Location: {OUTPUT_DIR}")
    print(f"   Feature: NO REPETITION - each speaker has different phrases")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
