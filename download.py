from huggingface_hub import hf_hub_download
import os

# Usando o repositório oficial da Idiap
repo_id = "Idiap/EdgeFace-XXS"
filename = "edgeface_xxs.pt"
dest_dir = "od_model"

print(f"⏳ Baixando {filename} do repositório oficial...")

try:
    path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=dest_dir,
        local_dir_use_symlinks=False  # Crucial para Windows
    )
    
    size = os.path.getsize(path)
    print(f"✅ Download concluído!")
    print(f"📍 Local: {path}")
    print(f"📏 Tamanho: {size / (1024*1024):.2f} MB")
    
    if size < 1000000:
        print("❌ Alerta: O arquivo parece ser apenas um ponteiro (muito pequeno).")
    else:
        print("🚀 Arquivo binário detectado com sucesso!")

except Exception as e:
    print(f"❌ Erro: {e}")