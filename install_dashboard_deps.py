#!/usr/bin/env python3
"""
Script para instalar dependências do dashboard Streamlit
"""

import subprocess
import sys

def install_requirements():
    requirements = [
        "streamlit>=1.28.0",
        "pandas>=2.0.0",
        "plotly>=5.17.0",
        "pillow>=10.0.0",
        "numpy>=1.24.0"
    ]
    
    print("Instalando dependências do dashboard...")
    for package in requirements:
        print(f"\nInstalando {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    
    print("\n✅ Todas as dependências foram instaladas com sucesso!")
    print("\nPara iniciar o dashboard, execute:")
    print("   streamlit run dashboard.py")

if __name__ == "__main__":
    install_requirements()
