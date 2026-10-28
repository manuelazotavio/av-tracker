#!/usr/bin/env python3

import sys
from types import ModuleType

sys.modules['k2'] = ModuleType('k2')
sys.modules['_k2'] = ModuleType('_k2')
sys.modules['flair'] = ModuleType('flair')
sys.modules['flair.data'] = ModuleType('flair.data')
sys.modules['flair.embeddings'] = ModuleType('flair.embeddings')
sys.modules['speechbrain.integrations.k2_fsa'] = ModuleType('speechbrain.integrations.k2_fsa')
sys.modules['speechbrain.integrations.nlp'] = ModuleType('speechbrain.integrations.nlp')
sys.modules['speechbrain.integrations.nlp.flair_embeddings'] = ModuleType('speechbrain.integrations.nlp.flair_embeddings')

import os
import logging
import sounddevice as sd
import numpy as np
from src.realtime_transcriber import RealtimeTranscriber
from src.multi_speaker_verifier import MultiSpeakerVerifier

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def list_audio_devices():
    """Lists all available audio devices"""
    print("\n" + "="*70)
    print("🎧 AVAILABLE AUDIO DEVICES")
    print("="*70)

    devices = sd.query_devices()

    for i, device in enumerate(devices):
        print(f"\n[{i}] {device['name']}")
        print(f"    Input channels: {device['max_input_channels']}")
        print(f"    Output channels: {device['max_output_channels']}")
        print(f"    Sample rate: {device['default_samplerate']} Hz")

        # Mark relevant devices
        if 'loopback' in device['name'].lower() or 'stereo mix' in device['name'].lower():
            print(f"    ⭐ IDEAL FOR CALLS/SYSTEM")
        if 'microphone' in device['name'].lower() or 'mic' in device['name'].lower():
            print(f"    🎤 IDEAL FOR MICROPHONE")

    return len(devices)


def select_devices():
    """Allows the user to select which devices to use"""
    print("\n" + "="*70)
    print("🎯 DEVICE SELECTION")
    print("="*70)

    list_audio_devices()

    print("\n" + "-"*70)

    # Microphone
    mic_input = input("\nEnter the MICROPHONE index (leave blank for default): ").strip()
    mic_device = int(mic_input) if mic_input else None

    # System audio (calls, music, etc)
    system_input = input("Enter the index for SYSTEM/CALL AUDIO (leave blank to skip): ").strip()
    system_device = int(system_input) if system_input else None

    print(f"\n✅ Selected configuration:")
    if mic_device is not None:
        print(f"   Microphone: [{mic_device}]")
    else:
        print(f"   Microphone: [default]")

    if system_device is not None:
        print(f"   System/Calls: [{system_device}]")
    else:
        print(f"   System/Calls: [disabled]")

    return mic_device, system_device


def create_multi_source_recorder(verifier, hf_token, mic_device=None, system_device=None,
                                  whisper_size="small", **kwargs):
    """Creates a recorder that can capture multiple sources"""

    recorder = RealtimeTranscriber(
        verifier,
        hf_token,
        whisper_size=whisper_size,
        device="cuda",
        **kwargs
    )

    # Store device info
    recorder.mic_device = mic_device
    recorder.system_device = system_device
    recorder.audio_queue_system = None

    # If we have a system device, create a separate queue
    if system_device is not None:
        import queue
        recorder.audio_queue_system = queue.Queue(maxsize=100)

    return recorder


def record_multi_source(recorder, embeddings_dir):
    """Records multiple sources in parallel"""
    import threading
    import queue
    import time

    print("\n" + "="*70)
    print("🎤 STARTING MULTI-SOURCE RECORDING")
    print("="*70)

    # Force recording to start
    recorder.running = True

    def _get_wasapi_hostapi_index():
        try:
            for idx, api in enumerate(sd.query_hostapis()):
                if "WASAPI" in api.get("name", ""):
                    return idx
        except Exception:
            return None
        return None

    wasapi_index = _get_wasapi_hostapi_index()
    if wasapi_index is not None:
        logger.info("🎧 WASAPI available (index %s)", wasapi_index)

    def _build_wasapi_settings(loopback):
        if not hasattr(sd, "WasapiSettings"):
            return None
        if not loopback:
            try:
                return sd.WasapiSettings()
            except Exception:
                return None
        try:
            return sd.WasapiSettings(loopback=True)
        except TypeError:
            logger.warning("⚠️ WASAPI loopback not supported in this version of sounddevice")
            return None
        except Exception:
            return None

    def _make_callback(target_queue, label):
        def _callback(indata, frames, time_info, status):
            if status:
                logger.warning(f"⚠️ {label}: {status}")
            try:
                audio_data = indata.copy().flatten().astype(np.float32)
                target_queue.put(audio_data, timeout=0.01)
            except queue.Full:
                logger.debug(f"{label}: queue full, discarding audio")
        return _callback

    def _open_input_stream(device, label, callback, loopback=False):
        # 1. Fetch actual device information
        try:
            device_info = sd.query_devices(device)
            # Use the device's native samplerate
            native_samplerate = device_info.get("default_samplerate", recorder.sample_rate)
            # Read the maximum number of input channels the device supports
            native_channels = int(device_info.get("max_input_channels", 1))
        except Exception as e:
            logger.warning(f"⚠️ {label}: Could not read device {device} info. Using defaults.")
            native_samplerate = recorder.sample_rate
            native_channels = 1

        # Limit to 1 or 2 channels to avoid errors with unusual headsets (e.g. 3 channels)
        if native_channels > 2:
            native_channels = 2

        stream_kwargs = dict(
            channels=native_channels,
            samplerate=native_samplerate,
            blocksize=int(native_samplerate * 0.1),
            dtype=np.float32,
            callback=callback,
        )

        try:
            return sd.InputStream(device=device, **stream_kwargs)
        except Exception as e:
            logger.warning(f"⚠️ {label}: failed to open device {device} with native settings. Trying safe defaults. Error: {e}")

            # Safe fallback: default API (MME/DirectSound), 1 channel, let Python choose the sample rate.
            fallback_kwargs = dict(
                channels=1,
                dtype=np.float32,
                callback=callback,
            )
            try:
                return sd.InputStream(device=None, **fallback_kwargs)
            except Exception as e2:
                logger.error(f"❌ Fatal error trying to open safe audio stream: {e2}")
                raise

    # Thread to capture microphone
    def capture_microphone():
        try:
            logger.info("🎤 Capturing from microphone...")
            callback = _make_callback(recorder.audio_queue, "Microphone")
            with _open_input_stream(recorder.mic_device, "Microphone", callback):
                print("\n" + "🔴 RECORDING (Microphone)")
                while recorder.running:
                    time.sleep(0.2)
        except Exception as e:
            logger.error(f"Error capturing from microphone: {e}")
            import traceback
            traceback.print_exc()

    # Thread to capture system audio
    def capture_system_audio():
        if recorder.system_device is None:
            return

        try:
            logger.info("🔊 Capturing system audio...")
            callback = _make_callback(recorder.audio_queue, "System")
            device_index = recorder.system_device
            try:
                info = sd.query_devices(device_index)
                logger.info(
                    "🔎 System: %s | in=%s out=%s",
                    info.get("name", "?"),
                    info.get("max_input_channels", "?"),
                    info.get("max_output_channels", "?"),
                )
            except Exception:
                logger.warning("⚠️ Could not read device %s info", device_index)

            # Force WASAPI loopback to capture system audio
            extra_settings = _build_wasapi_settings(loopback=True)
            if extra_settings is None:
                logger.warning("⚠️ Loopback not available, skipping system audio")
                return

            with _open_input_stream(device_index, "System", callback, loopback=True):
                print("🔴 RECORDING (System/Calls)")
                while recorder.running:
                    time.sleep(0.2)
        except Exception as e:
            logger.error(f"Error capturing system audio: {e}")
            import traceback
            traceback.print_exc()

    # Start threads
    thread_mic = threading.Thread(target=capture_microphone, daemon=False)
    thread_system = threading.Thread(target=capture_system_audio, daemon=False) if recorder.system_device else None

    thread_mic.start()
    if thread_system:
        thread_system.start()

    processing_thread = threading.Thread(target=recorder._process_audio_chunks, daemon=True)
    processing_thread.start()

    logger.info("✅ Capture threads started")

    # Start normal processing
    print("\n✨ SYSTEM READY - Start speaking!")
    print("Press Ctrl+C to stop\n")

    try:
        # Main loop - keep recording
        last_status = time.time()
        while recorder.running:
            try:
                time.sleep(0.5)

                # Status every 5 seconds
                if time.time() - last_status > 5.0:
                    queue_size = recorder.audio_queue.qsize() if hasattr(recorder.audio_queue, 'qsize') else 0
                    logger.debug(f"📊 Status: {queue_size} frames in queue, running={recorder.running}")
                    last_status = time.time()

            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                break

    except KeyboardInterrupt:
        print("\n⏹️ Stopping recording...")
        logger.info("⏹️ Ctrl+C detected")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        recorder.running = False
        logger.info("Waiting for threads...")
        time.sleep(1)
        thread_mic.join(timeout=3)
        if thread_system:
            thread_system.join(timeout=3)
        processing_thread.join(timeout=3)
        logger.info("✅ Threads finished")


def main():
    print("\n" + "="*70)
    print("🎙️ MULTI-SOURCE TRANSCRIBER (Microphone + Calls)")
    print("="*70)

    # Find embeddings
    base_dir = os.path.abspath(os.path.dirname(__file__))
    embeddings_dir = os.path.join(base_dir, "data", "embeddings")

    if not os.path.exists(embeddings_dir):
        print("\n⚠️ Embeddings directory not found, creating an empty one.")
        os.makedirs(embeddings_dir)

    print(f"\n✅ Embeddings found in: {embeddings_dir}")

    # Select devices
    mic_device, system_device = select_devices()

    print("\n⚙️  Initializing system...")

    try:
        # Load verifier
        logger.info("📂 Loading speaker verifier...")
        verifier = MultiSpeakerVerifier(embeddings_dir, threshold=0.65)

        # Load HF token
        sys.path.insert(0, os.path.dirname(__file__))
        from src.multi_speaker_verification import HF_TOKEN

        # Create multi-source recorder
        logger.info("🎯 Initializing transcriber...")
        recorder = create_multi_source_recorder(
            verifier,
            HF_TOKEN,
            mic_device=mic_device,
            system_device=system_device,
            whisper_size="small",
            use_ai_analysis=True,
            diarization_clustering_threshold=0.45,
            verifier_confidence_min=0.9,
            chunk_duration=2.0,
        )

        print("✅ System ready!\n")

        # Record multiple sources
        record_multi_source(recorder, embeddings_dir)

        # Save result
        output_file = recorder.save_session()
        print(f"\n✅ Session saved to: {output_file}")

    except KeyboardInterrupt:
        print("\n\n⏹️ Interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        logger.exception("Fatal error")
        sys.exit(1)


if __name__ == "__main__":
    main()
