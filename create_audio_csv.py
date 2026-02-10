#!/usr/bin/env python3
"""
Cria um CSV mapeando actor_id -> audio_file para os atores baixados.
"""
import pandas as pd
from pathlib import Path
import sys

def main():
    # Diretório com os atores baixados
    vox_dir = Path("vox_300_atores")
    
    if not vox_dir.exists():
        print("❌ Diretório vox_300_atores não existe!")
        print("Execute primeiro: python src/download.py")
        sys.exit(1)
    
    # Buscar todos os arquivos .wav
    audio_files = list(vox_dir.rglob("*.wav"))
    
    if not audio_files:
        print("❌ Nenhum arquivo .wav encontrado em vox_100_atores/")
        sys.exit(1)
    
    print(f"✅ Encontrados {len(audio_files)} arquivos .wav")
    
    # Criar lista de registros
    records = []
    for audio_path in audio_files:
        # Extrair actor_id do caminho (ex: vox_100_atores/id10001/... -> id10001)
        parts = audio_path.parts
        actor_id = None
        for part in parts:
            if part.startswith("id"):
                actor_id = part
                break
        
        if actor_id:
            records.append({
                'actor_id': actor_id,
                'audio_file': str(audio_path)
            })
    
    print(f"✅ Mapeados {len(records)} arquivos para atores")
    
    # Criar DataFrame e salvar
    df = pd.DataFrame(records)
    
    # Adicionar informação de gênero do vox1_meta.csv se disponível
    meta_file = Path("vox1_meta.csv")
    if meta_file.exists():
        try:
            meta_df = pd.read_csv(meta_file, sep='\t')
            meta_df.columns = ['actor_id', 'name', 'gender', 'nationality', 'set']
            df = df.merge(meta_df[['actor_id', 'gender']], on='actor_id', how='left')
            print(f"✅ Adicionadas informações de gênero")
        except Exception as e:
            print(f"⚠️ Não foi possível adicionar gênero: {e}")
    
    # Contar atores únicos
    n_actors = df['actor_id'].nunique()
    print(f"✅ Total de atores únicos: {n_actors}")
    
    # Salvar CSV
    output_file = Path("data/audio_files.csv")
    output_file.parent.mkdir(exist_ok=True)
    df.to_csv(output_file, index=False)
    
    print(f"\n✅ CSV criado: {output_file}")
    print(f"   - {len(df)} arquivos de áudio")
    print(f"   - {n_actors} atores únicos")
    
    # Mostrar amostra
    print("\n📋 Amostra do CSV:")
    print(df.head(10).to_string(index=False))

if __name__ == '__main__':
    main()
