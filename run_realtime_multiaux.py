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
    """Lista todos os dispositivos de áudio disponíveis"""
    print("\n" + "="*70)
    print("🎧 DISPOSITIVOS DE ÁUDIO DISPONÍVEIS")
    print("="*70)
    
    devices = sd.query_devices()
    
    for i, device in enumerate(devices):
        print(f"\n[{i}] {device['name']}")
        print(f"    Canais entrada: {device['max_input_channels']}")
        print(f"    Canais saída: {device['max_output_channels']}")
        print(f"    Taxa amostra: {device['default_samplerate']} Hz")
        
        # Marca dispositivos relevantes
        if 'loopback' in device['name'].lower() or 'stereo mix' in device['name'].lower():
            print(f"    ⭐ IDEAL PARA CHAMADAS/SISTEMA")
        if 'microphone' in device['name'].lower() or 'mic' in device['name'].lower():
            print(f"    🎤 IDEAL PARA MICROFONE")
    
    return len(devices)


def select_devices():
    """Permite ao user selecionar quais dispositivos usar"""
    print("\n" + "="*70)
    print("🎯 SELEÇÃO DE DISPOSITIVOS")
    print("="*70)
    
    list_audio_devices()
    
    print("\n" + "-"*70)
    
    # Microfone
    mic_input = input("\nDigite o índice do MICROFONE (deixe em branco para padrão): ").strip()
    mic_device = int(mic_input) if mic_input else None
    
    # Áudio do sistema (chamadas, música, etc)
    system_input = input("Digite o índice para ÁUDIO DO SISTEMA/CHAMADAS (deixe em branco para pular): ").strip()
    system_device = int(system_input) if system_input else None
    
    print(f"\n✅ Configuração selecionada:")
    if mic_device is not None:
        print(f"   Microfone: [{mic_device}]")
    else:
        print(f"   Microfone: [padrão]")
    
    if system_device is not None:
        print(f"   Sistema/Chamadas: [{system_device}]")
    else:
        print(f"   Sistema/Chamadas: [desativado]")
    
    return mic_device, system_device


def create_multi_source_recorder(verifier, hf_token, mic_device=None, system_device=None, 
                                  whisper_size="small", **kwargs):
    """Cria um gravador que pode capturar múltiplas fontes"""
    
    recorder = RealtimeTranscriber(
        verifier,
        hf_token,
        whisper_size=whisper_size,
        device="cuda",
        **kwargs
    )
    
    # Armazena info de dispositivos
    recorder.mic_device = mic_device
    recorder.system_device = system_device
    recorder.audio_queue_system = None
    
    # Se temos dispositivo de sistema, cria fila separada
    if system_device is not None:
        import queue
        recorder.audio_queue_system = queue.Queue(maxsize=100)
    
    return recorder


def record_multi_source(recorder, embeddings_dir):
    """Grava múltiplas fontes em paralelo"""
    import threading
    import queue
    import time
    
    print("\n" + "="*70)
    print("🎤 INICIANDO GRAVAÇÃO MULTI-FONTE")
    print("="*70)
    
    # Força gravação a iniciar
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
        logger.info("🎧 WASAPI disponivel (indice %s)", wasapi_index)

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
            logger.warning("⚠️ WASAPI loopback nao suportado nesta versao do sounddevice")
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
                logger.debug(f"{label}: fila cheia, descartando áudio")
        return _callback

    def _open_input_stream(device, label, callback, loopback=False):
        # 1. Busca as informações reais do dispositivo selecionado
        try:
            device_info = sd.query_devices(device)
            # Usa o samplerate nativo do dispositivo
            native_samplerate = device_info.get("default_samplerate", recorder.sample_rate)
            # Lê o número máximo de canais de entrada que o dispositivo suporta
            native_channels = int(device_info.get("max_input_channels", 1))
        except Exception as e:
            logger.warning(f"⚠️ {label}: Não foi possível ler as info do dispositivo {device}. Usando padrão.")
            native_samplerate = recorder.sample_rate
            native_channels = 1
        
        # Limita para 1 ou 2 canais para evitar erros com headsets bizarros (como 3 canais)
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
            logger.warning(f"⚠️ {label}: falha ao abrir dispositivo {device} com as configurações nativas. Tentando padrão seguro. Erro: {e}")
            
            # Fallback seguro: API padrão (MME/DirectSound), 1 canal, e deixa o Python escolher o sample rate.
            fallback_kwargs = dict(
                channels=1,
                dtype=np.float32,
                callback=callback,
            )
            try:
                return sd.InputStream(device=None, **fallback_kwargs)
            except Exception as e2:
                logger.error(f"❌ Erro fatal ao tentar abrir fluxo de áudio seguro: {e2}")
                raise

    # Thread para capturar microfone
    def capture_microphone():
        try:
            logger.info("🎤 Capturando do microfone...")
            callback = _make_callback(recorder.audio_queue, "Microfone")
            with _open_input_stream(recorder.mic_device, "Microfone", callback):
                print("\n" + "🔴 GRAVANDO (Microfone)")
                while recorder.running:
                    time.sleep(0.2)
        except Exception as e:
            logger.error(f"Erro na captura do microfone: {e}")
            import traceback
            traceback.print_exc()
    
    # Thread para capturar áudio do sistema
    def capture_system_audio():
        if recorder.system_device is None:
            return
        
        try:
            logger.info("🔊 Capturando áudio do sistema...")
            callback = _make_callback(recorder.audio_queue, "Sistema")
            device_index = recorder.system_device
            try:
                info = sd.query_devices(device_index)
                logger.info(
                    "🔎 Sistema: %s | in=%s out=%s",
                    info.get("name", "?"),
                    info.get("max_input_channels", "?"),
                    info.get("max_output_channels", "?"),
                )
            except Exception:
                logger.warning("⚠️ Não foi possível ler info do dispositivo %s", device_index)

            # Forca loopback WASAPI para capturar audio do sistema
            extra_settings = _build_wasapi_settings(loopback=True)
            if extra_settings is None:
                logger.warning("⚠️ Loopback nao disponivel, pulando audio do sistema")
                return

            with _open_input_stream(device_index, "Sistema", callback, loopback=True):
                print("🔴 GRAVANDO (Sistema/Chamadas)")
                while recorder.running:
                    time.sleep(0.2)
        except Exception as e:
            logger.error(f"Erro na captura do sistema: {e}")
            import traceback
            traceback.print_exc()
    
    # Inicia threads
    thread_mic = threading.Thread(target=capture_microphone, daemon=False)
    thread_system = threading.Thread(target=capture_system_audio, daemon=False) if recorder.system_device else None
    
    thread_mic.start()
    if thread_system:
        thread_system.start()

    processing_thread = threading.Thread(target=recorder._process_audio_chunks, daemon=True)
    processing_thread.start()
    
    logger.info("✅ Threads de captura iniciadas")
    
    # Inicia processamento normal
    print("\n✨ SISTEMA PRONTO - Comece a falar!")
    print("Pressione Ctrl+C para parar\n")
    
    try:
        # Loop principal - mantém gravando
        last_status = time.time()
        while recorder.running:
            try:
                time.sleep(0.5)
                
                # Status a cada 5 segundos
                if time.time() - last_status > 5.0:
                    queue_size = recorder.audio_queue.qsize() if hasattr(recorder.audio_queue, 'qsize') else 0
                    logger.debug(f"📊 Status: {queue_size} frames na fila, running={recorder.running}")
                    last_status = time.time()
                    
            except Exception as e:
                logger.error(f"Erro no loop principal: {e}")
                break
                
    except KeyboardInterrupt:
        print("\n⏹️ Parando gravação...")
        logger.info("⏹️ Ctrl+C detectado")
    except Exception as e:
        logger.error(f"Erro inesperado: {e}")
        import traceback
        traceback.print_exc()
    finally:
        recorder.running = False
        logger.info("Aguardando threads...")
        time.sleep(1)
        thread_mic.join(timeout=3)
        if thread_system:
            thread_system.join(timeout=3)
        processing_thread.join(timeout=3)
        logger.info("✅ Threads finalizadas")


def main():
    print("\n" + "="*70)
    print("🎙️ TRANSCRITOR MULTI-FONTE (Microfone + Chamadas)")
    print("="*70)
    
    # Encontra embeddings
    base_dir = os.path.abspath(os.path.dirname(__file__))
    embeddings_dir = os.path.join(base_dir, "data", "embeddings")
    
    if not os.path.exists(embeddings_dir):
        print("\n⚠️ Pasta de embeddings não encontrada, criando uma vazia.")
        os.makedirs(embeddings_dir)
    
    print(f"\n✅ Embeddings encontrados em: {embeddings_dir}")
    
    # Seleciona dispositivos
    mic_device, system_device = select_devices()
    
    print("\n⚙️  Inicializando sistema...")
    
    try:
        # Carrega verifier
        logger.info("📂 Carregando verificador de speakers...")
        verifier = MultiSpeakerVerifier(embeddings_dir, threshold=0.65)
        
        # Carrega token HF
        sys.path.insert(0, os.path.dirname(__file__))
        from src.multi_speaker_verification import HF_TOKEN
        
        # Cria gravador multi-source
        logger.info("🎯 Inicializando transcritor...")
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
        
        print("✅ Sistema pronto!\n")
        
        # Grava múltiplas fontes
        record_multi_source(recorder, embeddings_dir)
        
        # Salva resultado
        output_file = recorder.save_session()
        print(f"\n✅ Sessão salva em: {output_file}")
        
    except KeyboardInterrupt:
        print("\n\n⏹️ Interrompido pelo usuário")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ ERRO: {e}")
        logger.exception("Erro fatal")
        sys.exit(1)


if __name__ == "__main__":
    main()
