#!/usr/bin/env python3
import os
import subprocess
import sys

def run_command(cmd, description):
    """Run a command and return the result"""
    print(f"\n=== {description} ===")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=os.getcwd())
        if result.returncode == 0:
            print("✓ Comando executado com sucesso")
            if result.stdout:
                print(result.stdout[-500:])  # Show last 500 chars of output
        else:
            print(f"✗ Erro ao executar comando (código {result.returncode})")
            if result.stderr:
                print("Erro:", result.stderr[-500:])
        return result.returncode == 0
    except Exception as e:
        print(f"✗ Exceção ao executar comando: {e}")
        return False

def count_audio_files():
    """Count audio files in the project"""
    print("\n=== CONTANDO ARQUIVOS DE ÁUDIO ===")

    import glob

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

    print(f'\n🎵 Total de arquivos WAV encontrados: {total_count}')
    return total_count

def main():
    print("🔄 REFazendo TODOS os gráficos com dados balanceados de gênero")

    # Count audio files first
    audio_count = count_audio_files()

    # Run metrics calculation
    success = run_command("python src/calculate_metrics.py", "Calculando métricas e gerando gráficos")

    if success:
        print("\n✅ Gráficos atualizados com sucesso!")
        print(f"📊 {audio_count} arquivos de áudio foram processados")
        print("\n📁 Gráficos salvos em: analysis_results/")
        print("   - metricas_barra.png")
        print("   - confianca_scores.png")
        print("   - matriz_confusao.png")
        print("   - analise_threshold.png")
        print("   - acuracia_por_speakers.png")
    else:
        print("\n❌ Erro ao gerar gráficos")
        sys.exit(1)

if __name__ == "__main__":
    main()