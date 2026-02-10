#!/usr/bin/env python3
"""
Extrai embeddings individuais do arquivo all_embeddings.pkl
para arquivos .npy separados (um por ator).
"""

import pickle
import numpy as np
from pathlib import Path
import sys

def extract_embeddings(input_pkl: str, output_dir: str):
    """
    Extrai embeddings do arquivo .pkl e salva em arquivos .npy individuais.
    
    Args:
        input_pkl: Caminho para all_embeddings.pkl
        output_dir: Diretório onde salvar os arquivos .npy
    """
    
    # Carrega o arquivo pickle
    print(f"Carregando embeddings de {input_pkl}...")
    with open(input_pkl, 'rb') as f:
        data = pickle.load(f)
    
    embeddings = data['embeddings']
    labels = data['labels']
    
    print(f"  Encontrados: {len(labels)} atores")
    print(f"  Dimensão dos embeddings: {embeddings.shape}")
    
    # Cria o diretório de saída se não existir
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Salva cada embedding em um arquivo separado
    print(f"\nExtraindo para {output_dir}...")
    for i, label in enumerate(labels):
        embedding = embeddings[i]
        output_file = output_path / f"{label}.npy"
        np.save(output_file, embedding)
        
        if (i + 1) % 50 == 0:
            print(f"  Progresso: {i+1}/{len(labels)} atores")
    
    print(f"\n✓ Concluído! {len(labels)} arquivos .npy criados em {output_dir}")
    
    # Lista alguns exemplos
    print("\nExemplos de arquivos criados:")
    for i, label in enumerate(labels[:5]):
        print(f"  - {label}.npy")

if __name__ == "__main__":
    # Configuração
    input_pkl = "data/embeddings/all_embeddings.pkl"
    output_dir = "data/embeddings/individual"
    
    if len(sys.argv) > 1:
        input_pkl = sys.argv[1]
    if len(sys.argv) > 2:
        output_dir = sys.argv[2]
    
    extract_embeddings(input_pkl, output_dir)
