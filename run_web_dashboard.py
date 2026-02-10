#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para testar o servidor web localmente e verificar dependencias
"""

import sys
import subprocess

def check_dependencies():
    """Verifica se as dependencias estao instaladas"""
    print("=" * 60)
    print("Verificando Dependencias")
    print("=" * 60)
    
    dependencies = {
        'flask': 'Flask',
        'flask_httpauth': 'Flask-HTTPAuth',
        'pandas': 'Pandas',
        'werkzeug': 'Werkzeug'
    }
    
    missing = []
    
    for module, name in dependencies.items():
        try:
            __import__(module)
            print(f"[OK] {name} instalado")
        except ImportError:
            print(f"[ERRO] {name} NAO instalado")
            missing.append(module)
    
    if missing:
        print("\n" + "=" * 60)
        print("Instalando dependencias faltantes...")
        print("=" * 60)
        subprocess.check_call([
            sys.executable, '-m', 'pip', 'install',
            'flask', 'flask-httpauth', 'pandas', 'werkzeug'
        ])
        print("\n[OK] Dependencias instaladas!")
    else:
        print("\n[OK] Todas as dependencias estao instaladas!")
    
    return True

def start_web_dashboard():
    """Inicia o servidor web"""
    print("\n" + "=" * 60)
    print("Iniciando Servidor Web AV-Tracker")
    print("=" * 60)
    print("\nAcesse em seu navegador:")
    print("  URL: http://localhost:5000")
    print("  Usuario: professor1")
    print("  Senha: senha123")
    print("\nPressione CTRL+C para parar")
    print("=" * 60 + "\n")
    
    subprocess.run([sys.executable, 'web_dashboard.py'])

if __name__ == '__main__':
    try:
        if check_dependencies():
            start_web_dashboard()
    except KeyboardInterrupt:
        print("\n\nServidor interrompido pelo usuario")
        sys.exit(0)
    except Exception as e:
        print(f"\nErro: {e}")
        sys.exit(1)
