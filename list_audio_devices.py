#!/usr/bin/env python3
"""
Script simples para listar e identificar seus dispositivos de áudio
Use isso para saber os índices corretos do microfone e auto-falante
"""

import sounddevice as sd

print("\n" + "="*80)
print("🎧 SEUS DISPOSITIVOS DE ÁUDIO")
print("="*80)

devices = sd.query_devices()

print("\n📋 LISTA COMPLETA:\n")

for i, device in enumerate(devices):
    marker = ""
    name = device['name']
    
    # Identifica tipos de dispositivos
    if 'microphone' in name.lower() or 'mic' in name.lower() or 'entrada' in name.lower():
        marker = " 🎤 ← MICROFONE"
    elif 'loopback' in name.lower() or 'stereo mix' in name.lower():
        marker = " 🔊 ← ÁUDIO DO SISTEMA (Ideal para chamadas!)"
    elif 'vb-audio' in name.lower() or 'virtual cable' in name.lower():
        marker = " 🔊 ← ÁUDIO VIRTUAL (VB-Cable)"
    elif 'voicemeeter' in name.lower():
        marker = " 🎚️  ← VOICEMEETER"
    elif 'speaker' in name.lower() or 'saída' in name.lower():
        marker = " 🔊 ← AUTO-FALANTE"
    elif 'default' in name.lower():
        marker = " ⭐ PADRÃO"
    
    channels_in = device['max_input_channels']
    channels_out = device['max_output_channels']
    
    print(f"[{i:2d}] {name}")
    if channels_in > 0:
        print(f"      Entrada: {channels_in} ch | Taxa: {device['default_samplerate']} Hz{marker}")
    if channels_out > 0:
        print(f"      Saída: {channels_out} ch")
    print()

print("="*80)
print("\n💡 INSTRUÇÕES:")
print("   1. Procure pelos índices com os marcadores 🎤 e 🔊")
print("   2. Se não encontrar 'Stereo Mix' ou 'VB-Cable', deixe em branco")
print("   3. Digite o índice quando o script pedir, ou deixe em branco para padrão")
print("\n📝 EXEMPLO:")
print("   Microfone (enter): 3")
print("   Sistema (enter): 7")
print("\n" + "="*80)
