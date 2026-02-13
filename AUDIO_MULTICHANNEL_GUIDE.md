# 🎙️ GUIA: Capturar Áudio de Chamadas + Microfone

Este guia explica como capturar áudio simultâneo do seu microfone E de chamadas/ligações (Teams, WhatsApp, Zoom, etc).

## 📋 Opções Disponíveis

### Opção 1: Stereo Mix (Windows) ⭐ MAIS FÁCIL
Permite capturar todo o áudio do computador + microfone em um único dispositivo.

**Pré-requisito:**
- Seu driver de áudio deve suportar "Stereo Mix" ou "What U Hear"
- Geralmente disponível em drivers Realtek, NVIDIA, SoundBlaster

**Como ativar:**
1. Clique em `Som` → `Configuração avançada de som`
2. Abra `Volume e configurações avançadas`
3. Vá para `Gravação`
4. Procure por `Stereo Mix` ou `What U Hear`
5. Se estiver desativado:
   - Clique com botão direito
   - `Ativar`
   - Defina como `Dispositivo padrão`

**Depois:**
```bash
python run_realtime_multiaux.py
```
- Quando pedir dispositivos, selecione o Stereo Mix
- O sistema capturará tudo automaticamente

---

### Opção 2: VB-Audio Virtual Cable ⭐ RECOMENDADO
Solução profissional que funciona com qualquer interface de áudio.

**Download:**
- https://vb-audio.com/Cable/
- Versão gratuita disponível

**Como funciona:**
1. Instala um "cabo virtual" entre aplicações
2. Redireciona áudio do Teams/Zoom → Gravador
3. Captura audio separado do microfone

**Instalação:**
```bash
# Baixar e instalar VB-Audio Cable
# Reiniciar o computador

# Depois executar:
python run_realtime_multiaux.py
```

**Configuração:**
1. No Teams/Zoom/WhatsApp → Defina `VB-Audio Virtual Cable` como saída
2. No script: selecione `VB-Audio Virtual Cable` como "Áudio do Sistema"
3. Selecione seu `Microfone` como "Microfone"

---

### Opção 3: Voicemeeter Banana (Gratuito) ⭐ ALTERNATIVA
Mixer de áudio virtual mais robusto.

**Download:**
- https://vb-audio.com/Voicemeeter/
- 100% gratuito

**Vantagens:**
- Controla nível de cada entrada
- Separa áudio para gravação
- Mais configurável que Stereo Mix

**Instalação:**
1. Baixar e instalar Voicemeeter Banana
2. Reiniciar
3. Abrir o Voicemeeter
4. Configurar:
   - `VAIO1` = Áudio do Sistema (Teams, Zoom, etc)
   - `VAIO2` = Microfone

```bash
python run_realtime_multiaux.py
```

---

### Opção 4: Script Separado por Fonte (Avançado)
Se quiser processar áudio do microfone E do sistema separadamente:

**Uso:**
```bash
# Terminal 1 - Apenas microfone
python run_realtime.py

# Terminal 2 - Para capturar só sistema (se VB-Cable instalado)
python -c "import sounddevice; print(sounddevice.query_devices())"
```

---

## 🚀 QUICKSTART

### Passo 1: Verificar seus dispositivos
```bash
python -c "import sounddevice; print(sounddevice.query_devices())"
```

Procure por:
- ✅ `Stereo Mix`
- ✅ `VB-Audio Virtual Cable`
- ✅ `Voicemeeter`
- ✅ `Seu Microfone`

### Passo 2: Executar o script
```bash
python run_realtime_multiaux.py
```

### Passo 3: Selecionar dispositivos
```
Digite o índice do MICROFONE (deixe em branco para padrão): [selecione seu mic]
Digite o índice para ÁUDIO DO SISTEMA/CHAMADAS: [selecione Stereo Mix/VB-Cable]
```

### Passo 4: Começar chamada
1. Inicie uma chamada no Teams, Zoom, WhatsApp, etc
2. Fale normalmente
3. O script vai gravar AMBAS as vozes
4. Pressione Ctrl+C quando terminar

---

## 📊 Comparação das Opções

| Metodo | Windows | Mac | Linux | Facilidade | Qualidade |
|--------|---------|-----|-------|-----------|-----------|
| **Stereo Mix** | ✅ | ❌ | ❌ | ⭐⭐⭐⭐⭐ | Boa |
| **VB-Cable** | ✅ | ⚠️ (pago) | ✅ | ⭐⭐⭐⭐ | Excelente |
| **Voicemeeter** | ✅ | ❌ | ❌ | ⭐⭐⭐ | Excelente |
| **Script Manual** | ✅ | ✅ | ✅ | ⭐⭐ | Boa |

---

## 🔧 TROUBLESHOOTING

### Problema: "Stereo Mix não aparece"
**Solução:**
1. Clique direito no alto-falante
2. `Dispositivos de som` → `Gravação`
3. Clique direito em espaço vazio
4. Marque `Mostrar dispositivos desativados`
5. Procure `Stereo Mix`, clique direito → `Ativar`

### Problema: "Áudio muito baixo da chamada"
**Solução:**
1. Instale VB-Cable (Stereo Mix é muito fraco)
2. Ou ajuste ganho no Voicemeeter
3. Ou aumente volume do Teams/Zoom antes de gravar

### Problema: "Capturando só o microfone, não a chamada"
**Solução:**
1. Verifique se selecionou o dispositivo certo
2. Teste a captura:
   ```bash
   python -c "
   import sounddevice as sd
   # Teste o dispositivo selecionado
   print('Testando...')
   "
   ```

### Problema: "Ruído muito alto"
**Solução:**
- Use o script exclusivamente para a chamada (não deixe música de fundo)
- Ajuste os parâmetros no `run_realtime_multiaux.py`:
  ```python
  diarization_clustering_threshold=0.45,  # ↑ aumenta se muito ruído
  ```

---

## 💡 DICAS PROFISSIONAIS

### 1. Testar antes de gravar importante
```bash
python run_realtime_multiaux.py --test
```

### 2. Configurar nível de áudio
- Teams/Zoom: Teste áudio ANTES de começar
- Verifique que seu microfone e os speakers da chamada têm bom nível

### 3. Usar fones com microfone integrado
- Reduz feedback e eco
- Melhora qualidade da gravação

### 4. Fechar outras aplicações de áudio
- Obs, Discord, StreamLabs interferem
- Feche antes de gravar chamadas importantes

---

## 📝 Resultado da Gravação

Após pressionar Ctrl+C, você terá:

```
realtime_sessions/
├── transcript_YYYYMMDD_HHMMSS.txt  ← Transcrição em texto
└── speakers_YYYYMMDD_HHMMSS.txt    ← Identificação de speakers
```

**Exemplo de saída:**
```
[14:23:45] Speaker001 (João): Olá, tudo bem?
[14:23:47] Speaker002 (Maria): Oi! Tudo certo
[14:23:50] Speaker001 (João): Queria falar sobre o projeto
```

---

## 🆘 Precisa de ajuda?

Se nenhuma opção funcionar, tente:

1. **Verificar drivers de áudio:**
   ```bash
   python -c "
   import sounddevice as sd
   info = sd.query_devices()
   import json
   print(json.dumps(info, indent=2, default=str))
   " > audio_info.txt
   ```

2. **Testar gravação simples:**
   ```bash
   python run_realtime.py  # Só microfone, sem chamada
   ```

3. **Usar alternativa - gravação manual:**
   - OBS Studio (gratuito, funciona para tudo)
   - Exportar o áudio
   - Processar com o script depois

---

**Última atualição:** Fevereiro 2026
**Versão:** v2.0 - Multi-source

