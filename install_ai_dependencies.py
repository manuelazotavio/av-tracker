"""
Script para instalar dependências de IA para análise contextual
"""
import subprocess
import sys

def run_command(cmd, description):
    print(f"\n{'='*60}")
    print(f"📦 {description}")
    print(f"{'='*60}")
    try:
        subprocess.check_call(cmd, shell=True)
        print(f"✅ {description} - Concluído!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Erro: {e}")
        return False

def main():
    print("""
    ╔════════════════════════════════════════════════════════════╗
    ║  Instalador de Dependências de IA                          ║
    ║  Para Análise Contextual e Detecção de Nomes               ║
    ╚════════════════════════════════════════════════════════════╝
    """)
    
    dependencies = [
        ("pip install spacy", "Instalando spaCy"),
        ("pip install transformers", "Instalando Transformers (Hugging Face)"),
        ("pip install sentencepiece", "Instalando SentencePiece (para LLM)"),
        ("python -m spacy download pt_core_news_sm", "Baixando modelo NER Português (pequeno)"),
    ]
    
    success_count = 0
    
    for cmd, desc in dependencies:
        if run_command(cmd, desc):
            success_count += 1
    
    print(f"\n{'='*60}")
    print(f"📊 Resultado: {success_count}/{len(dependencies)} instalações bem-sucedidas")
    print(f"{'='*60}")
    
    if success_count == len(dependencies):
        print("\n✅ Todas as dependências foram instaladas com sucesso!")
        print("\n💡 Modelo instalado: pt_core_news_sm (pequeno)")
        print("   Para melhor precisão, instale o modelo grande:")
        print("   python -m spacy download pt_core_news_lg")
    else:
        print("\n⚠️ Algumas instalações falharam. Verifique os erros acima.")
    
    print("\n🚀 Agora você pode usar análise de IA no sistema!")

if __name__ == "__main__":
    main()
