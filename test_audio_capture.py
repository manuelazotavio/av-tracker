#!/usr/bin/env python3
"""
Script simples para testar captura de áudio multi-fonte
Útil para debug
"""

import sounddevice as sd
import numpy as np
import time
import threading
import sys


def test_audio_capture():
    """Testa captura de áudio simples"""
    
    print("\n" + "="*70)
    print("🎧 TESTE DE CAPTURA DE ÁUDIO")
    print("="*70)
    
    # Lista dispositivos
    print("\n📋 Dispositivos disponíveis:\n")
    devices = sd.query_devices()
    
    for i, device in enumerate(devices):
        print(f"[{i}] {device['name']}")
        print(f"    In: {device['max_input_channels']} | Out: {device['max_output_channels']}\n")
    
    # Seleciona dispositivos
    print("-"*70)
    mic_idx = input("Dispositivo para MICROFONE [padrão]: ").strip()
    mic_device = int(mic_idx) if mic_idx else None
    
    sys_idx = input("Dispositivo para SISTEMA [deixe em branco]: ").strip()
    sys_device = int(sys_idx) if sys_idx else None
    
    sample_rate = 16000
    duration = 10  # segundos
    
    print(f"\n⏱️  Gravando por {duration} segundos...")
    print("Fale normalmente para testar!\n")
    
    # Armazena áudio
    audio_data = {'mic': [], 'sys': []}
    lock = threading.Lock()
    running = True
    
    def capture_mic():
        try:
            with sd.InputStream(device=mic_device, channels=1, samplerate=sample_rate, dtype=np.float32) as stream:
                print("🎤 Capturando do microfone...")
                while running:
                    data, _ = stream.read(int(sample_rate * 0.1))
                    with lock:
                        audio_data['mic'].append(data.flatten())
        except Exception as e:
            print(f"❌ Erro microfone: {e}")
    
    def capture_sys():
        if sys_device is None:
            return
        try:
            with sd.InputStream(device=sys_device, channels=1, samplerate=sample_rate, dtype=np.float32) as stream:
                print("🔊 Capturando do sistema...")
                while running:
                    data, _ = stream.read(int(sample_rate * 0.1))
                    with lock:
                        audio_data['sys'].append(data.flatten())
        except Exception as e:
            print(f"❌ Erro sistema: {e}")
    
    # Inicia threads
    t1 = threading.Thread(target=capture_mic, daemon=False)
    t2 = threading.Thread(target=capture_sys, daemon=False) if sys_device else None
    
    t1.start()
    if t2:
        t2.start()
    
    # Aguarda
    try:
        for i in range(duration, 0, -1):
            print(f"⏳ {i}s restantes...", end="\r")
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n⏹️ Interrompido")
    finally:
        running = False
        t1.join(timeout=2)
        if t2:
            t2.join(timeout=2)
    
    # Mostra resultados
    print("\n" + "="*70)
    print("📊 RESULTADOS")
    print("="*70)
    
    with lock:
        mic_samples = sum(len(x) for x in audio_data['mic'])
        sys_samples = sum(len(x) for x in audio_data['sys'])
    
    print(f"\n🎤 Microfone:")
    print(f"   Frames capturados: {len(audio_data['mic'])}")
    print(f"   Amostras: {mic_samples:,}")
    print(f"   Tempo: {mic_samples / sample_rate:.1f}s")
    
    if audio_data['mic']:
        mic_array = np.concatenate(audio_data['mic'])
        mic_rms = np.sqrt(np.mean(mic_array**2))
        print(f"   Volume RMS: {mic_rms:.5f}")
    
    print(f"\n🔊 Sistema/Chamadas:")
    print(f"   Frames capturados: {len(audio_data['sys'])}")
    print(f"   Amostras: {sys_samples:,}")
    print(f"   Tempo: {sys_samples / sample_rate:.1f}s")
    
    if audio_data['sys']:
        sys_array = np.concatenate(audio_data['sys'])
        sys_rms = np.sqrt(np.mean(sys_array**2))
        print(f"   Volume RMS: {sys_rms:.5f}")
    
    # Diagnóstico
    print("\n" + "="*70)
    print("🔍 DIAGNÓSTICO")
    print("="*70)
    
    if mic_samples == 0:
        print("\n❌ Microfone: NÃO CAPTUROU ÁUDIO!")
        print("   → Verifique se o dispositivo está correto")
        print("   → Ou o microfone está da forma (em mudo?)")
    elif mic_samples < sample_rate * 5:  # Menos que 5 segundos
        print(f"\n⚠️  Microfone: CAPTUROU POUCO ({mic_samples / sample_rate:.1f}s)")
        print("   → Verifique a conexão do microfone")
    else:
        print(f"\n✅ Microfone: OK! Capturou {mic_samples / sample_rate:.1f}s")
    
    if sys_device is not None:
        if sys_samples == 0:
            print("\n⚠️  Sistema: NÃO CAPTUROU ÁUDIO")
            print("   → Nenhum áudio de sistema detectado")
            print("   → Reproduza áudio/vídeo ou inicie uma chamada")
        elif sys_samples < sample_rate * 5:
            print(f"\n⚠️  Sistema: CAPTUROU POUCO ({sys_samples / sample_rate:.1f}s)")
        else:
            print(f"\n✅ Sistema: OK! Capturou {sys_samples / sample_rate:.1f}s")
    
    print("\n" + "="*70)
    print("✨ Teste concluído!")
    print("="*70)


if __name__ == "__main__":
    try:
        test_audio_capture()
    except KeyboardInterrupt:
        print("\n\n⏹️ Abortado")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Erro fatal: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
