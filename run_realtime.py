#!/usr/bin/env python3
"""
Simple script to run the Real-Time Transcriber
Press ENTER to start, Ctrl+C to stop
"""

import os
import sys
import logging
from src.realtime_transcriber import RealtimeTranscriber
from src.multi_speaker_verifier import MultiSpeakerVerifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    print("\n" + "="*70)
    print("🎙️  REAL-TIME TRANSCRIBER - MICROPHONE RECORDING")
    print("="*70)
    
    # Find embeddings directory
    base_dir = os.path.abspath(os.path.dirname(__file__))
    candidate_dirs = [
        os.path.join(base_dir, "data", "embeddings"),
        os.path.join(base_dir, "data", "embeddings", "individual"),
    ]
    embeddings_dir = next((d for d in candidate_dirs if os.path.exists(d)), None)
    
    if not embeddings_dir:
        print("\n❌ ERROR: Embeddings directory not found!")
        print(f"Searched in:")
        for d in candidate_dirs:
            print(f"  - {d}")
        print("\nRun first: python src/create_embeddings_from_audio.py --help")
        sys.exit(1)
    
    print(f"\n✅ Embeddings found in: {embeddings_dir}")
    print("\n⚙️  Initializing system...")
    print("(This may take 1-3 minutes the first time)\n")
    
    try:
        # Load verifier
        print("📂 Loading speaker verifier...")
        verifier = MultiSpeakerVerifier(embeddings_dir, threshold=0.65)
        print("✅ Verifier loaded\n")
        
        # Create real-time transcriber
        print("🎯 Initializing real-time transcriber...")
        
        # Get HF token
        sys.path.insert(0, os.path.dirname(__file__))
        from src.multi_speaker_verification import HF_TOKEN
        
        transcriber = RealtimeTranscriber(
            verifier,
            HF_TOKEN,
            whisper_size="small",
            device="cpu",  # Forced to CPU (PyTorch CPU-only in this environment)
            use_ai_analysis=True,
            diarization_clustering_threshold=0.45,
            verifier_confidence_min=0.9,
            chunk_duration=2.0,
        )
        print("✅ Transcriber loaded and ready!\n")
        
        # Options menu
        print("="*70)
        print("📋 OPTIONS")
        print("="*70)
        print("\n1️⃣  Normal Mode - Automatically detect speakers")
        print("2️⃣  Custom Mode - Configure parameters")
        print("3️⃣  Exit")

        choice = input("\nChoose an option (1-3): ").strip()
        
        if choice == "3":
            print("\n👋 Goodbye!")
            return
        elif choice == "2":
            print("\n⚙️  CUSTOM CONFIGURATION")
            print("-" * 70)
            
            # Whisper size
            print("\nWhich Whisper size?")
            print("  - tiny   (fast, less accurate, ~1s latency)")
            print("  - small  (default, balanced, ~2s latency)")
            print("  - medium (accurate, slow, ~5s latency)")
            whisper_size = input("Size (tiny/small/medium) [small]: ").strip().lower() or "small"
            if whisper_size not in ["tiny", "small", "base", "medium"]:
                whisper_size = "small"
            
            # Chunk duration
            chunk_input = input("\nChunk duration in seconds [2.0]: ").strip()
            try:
                chunk_duration = float(chunk_input) if chunk_input else 2.0
                chunk_duration = max(1.0, min(5.0, chunk_duration))  # 1-5s
            except ValueError:
                chunk_duration = 2.0
            
            # Confidence
            conf_input = input("\nMinimum confidence for verification [0.9]: ").strip()
            try:
                confidence = float(conf_input) if conf_input else 0.9
                confidence = max(0.5, min(1.0, confidence))  # 0.5-1.0
            except ValueError:
                confidence = 0.9
            
            # Device
            print("\nDevice (cuda/cpu)?")
            print("  - cuda (GPU - much faster)")
            print("  - cpu  (CPU - more compatible)")
            device = input("Device [cuda]: ").strip().lower() or "cuda"
            if device not in ["cuda", "cpu"]:
                device = "cuda"
            
            print(f"\n✨ Configuration:")
            print(f"  - Whisper: {whisper_size}")
            print(f"  - Chunk: {chunk_duration}s")
            print(f"  - Confidence: {confidence:.0%}")
            print(f"  - Device: {device}")
            
            # Recreate transcriber with new config
            transcriber = RealtimeTranscriber(
                verifier,
                HF_TOKEN,
                whisper_size=whisper_size,
                device=device,
                verifier_confidence_min=confidence,
                chunk_duration=chunk_duration,
            )
        
        # Start recording
        print("\n" + "="*70)
        print("🎤 READY TO RECORD!")
        print("="*70)
        print("\n📣 Instructions:")
        print("  1. Make sure your microphone is on")
        print("  2. Speak clearly")
        print("  3. Introduce yourself at the beginning (name, name, ...)")
        print("  4. The transcription will appear in real time")
        print("  5. Press Ctrl+C when done")
        print(f"\n💾 Result will be saved to: realtime_sessions/")

        input("\n▶️  Press ENTER to start recording...\n")
        
        # Start
        output_file = transcriber.start_recording(embeddings_dir)

        print(f"\n✅ Session saved!")
        print(f"📄 View result: {output_file}")
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Aborted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        logger.exception("Fatal error")
        sys.exit(1)


if __name__ == "__main__":
    main()
