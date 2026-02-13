#!/usr/bin/env python3
"""
Script simples para executar o Transcritor em Tempo Real
Pressione ENTER para começar, Ctrl+C para parar
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
    print("🎙️  TRANSCRITOR EM TEMPO REAL - GRAVAÇÃO COM MICROFONE")
    print("="*70)
    
    # Encontra diretório de embeddings
    base_dir = os.path.abspath(os.path.dirname(__file__))
    candidate_dirs = [
        os.path.join(base_dir, "data", "embeddings"),
        os.path.join(base_dir, "data", "embeddings", "individual"),
    ]
    embeddings_dir = next((d for d in candidate_dirs if os.path.exists(d)), None)
    
    if not embeddings_dir:
        print("\n❌ ERRO: Pasta de embeddings não encontrada!")
        print(f"Procurei em:")
        for d in candidate_dirs:
            print(f"  - {d}")
        print("\nExecute primeiro: python src/create_embeddings_from_audio.py --help")
        sys.exit(1)
    
    print(f"\n✅ Embeddings encontrados em: {embeddings_dir}")
    print("\n⚙️  Inicializando sistema...")
    print("(Isto pode levar 1-3 minutos na primeira vez)\n")
    
    try:
        # Carrega verificador
        print("📂 Carregando verificador de speakers...")
        verifier = MultiSpeakerVerifier(embeddings_dir, threshold=0.65)
        print("✅ Verificador carregado\n")
        
        # Cria transcritor em tempo real
        print("🎯 Inicializando transcritor em tempo real...")
        
        # Obtém token HF
        sys.path.insert(0, os.path.dirname(__file__))
        from src.multi_speaker_verification import HF_TOKEN
        
        transcriber = RealtimeTranscriber(
            verifier,
            HF_TOKEN,
            whisper_size="small",
            device="cpu",  # Forçado para CPU (PyTorch CPU-only neste ambiente)
            use_ai_analysis=True,
            diarization_clustering_threshold=0.45,
            verifier_confidence_min=0.9,
            chunk_duration=2.0,
        )
        print("✅ Transcritor carregado e pronto!\n")
        
        # Menu de opções
        print("="*70)
        print("📋 OPÇÕES")
        print("="*70)
        print("\n1️⃣  Modo Normal - Detectar automaticamente speakers")
        print("2️⃣  Modo Personalizado - Configurar parâmetros")
        print("3️⃣  Sair")
        
        choice = input("\nEscolha uma opção (1-3): ").strip()
        
        if choice == "3":
            print("\n👋 Até logo!")
            return
        elif choice == "2":
            print("\n⚙️  CONFIGURAÇÃO PERSONALIZADA")
            print("-" * 70)
            
            # Whisper size
            print("\nQual tamanho do Whisper?")
            print("  - tiny   (rápido, menos preciso, ~1s latência)")
            print("  - small  (padrão, equilibrado, ~2s latência)")
            print("  - medium (preciso, lento, ~5s latência)")
            whisper_size = input("Tamanho (tiny/small/medium) [small]: ").strip().lower() or "small"
            if whisper_size not in ["tiny", "small", "base", "medium"]:
                whisper_size = "small"
            
            # Chunk duration
            chunk_input = input("\nDuração do chunk em segundos [2.0]: ").strip()
            try:
                chunk_duration = float(chunk_input) if chunk_input else 2.0
                chunk_duration = max(1.0, min(5.0, chunk_duration))  # 1-5s
            except ValueError:
                chunk_duration = 2.0
            
            # Confidence
            conf_input = input("\nConfiança mínima para verificação [0.9]: ").strip()
            try:
                confidence = float(conf_input) if conf_input else 0.9
                confidence = max(0.5, min(1.0, confidence))  # 0.5-1.0
            except ValueError:
                confidence = 0.9
            
            # Device
            print("\nDispositivo (cuda/cpu)?")
            print("  - cuda (GPU - muito mais rápido)")
            print("  - cpu  (CPU - mais compatível)")
            device = input("Dispositivo [cuda]: ").strip().lower() or "cuda"
            if device not in ["cuda", "cpu"]:
                device = "cuda"
            
            print(f"\n✨ Configuração:")
            print(f"  - Whisper: {whisper_size}")
            print(f"  - Chunk: {chunk_duration}s")
            print(f"  - Confiança: {confidence:.0%}")
            print(f"  - Device: {device}")
            
            # Recria transcritor com novas configs
            transcriber = RealtimeTranscriber(
                verifier,
                HF_TOKEN,
                whisper_size=whisper_size,
                device=device,
                verifier_confidence_min=confidence,
                chunk_duration=chunk_duration,
            )
        
        # Inicia gravação
        print("\n" + "="*70)
        print("🎤 PRONTO PARA GRAVAR!")
        print("="*70)
        print("\n📣 Instruções:")
        print("  1. Certifique-se de que o microfone está ligado")
        print("  2. Fale claramente em português")
        print("  3. Apresente-se no começo (nome, nome, ...)")
        print("  4. A transcrição aparecerá em tempo real")
        print("  5. Pressione Ctrl+C quando terminar")
        print(f"\n💾 Resultado será salvo em: realtime_sessions/")
        
        input("\n▶️  Pressione ENTER para iniciar a gravação...\n")
        
        # Inicia
        output_file = transcriber.start_recording(embeddings_dir)
        
        print(f"\n✅ Sessão salva!")
        print(f"📄 Ver resultado: {output_file}")
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Abortado pelo usuário")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ ERRO: {e}")
        logger.exception("Erro fatal")
        sys.exit(1)


if __name__ == "__main__":
    main()
