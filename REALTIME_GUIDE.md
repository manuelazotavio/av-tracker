# 🎙️ Guia - Transcritor em Tempo Real

## Visão Geral

O `realtime_transcriber.py` funciona completamente diferente do `multi_speaker_verification.py`:

### ✨ Recursos Principais

- **Captura de microfone em tempo real**: Grava áudio conforme você fala
- **Processamento contínuo**: Transcreções aparecem enquanto você fala
- **Diarização dinâmica**: Identifica speakers automaticamente
- **Separação de áudio**: Resolve overlaps (quando 2+ pessoas falam simultaneamente)
- **Detecção de nomes incremental**: Identifica nomes conforme a conversa avança
- **Embeddings automáticos**: Cria embeddings para novos speakers
- **Streaming real**: Tudo em background threads sem bloquear

---

## 🚀 Instalação de Dependências

### Backend de Áudio (escolha UM):

**Opção A - SoundDevice (recomendado para Windows/Mac):**
```bash
pip install sounddevice
```

**Opção B - PyAudio (mais compatível, mas mais difícil de instalar):**
```bash
pip install PyAudio
```

### Outras dependências necessárias:
```bash
pip install -r requirements.txt
```

---

## 📖 Como Usar

### Modo 1: Script Simples (Recomendado)

```bash
python run_realtime.py
```

O script fará:
1. Inicializar todos os modelos
2. Pedir para você pressionar ENTER
3. Começar a gravar do microfone
4. Exibir transcrições em tempo real
5. Pressione Ctrl+C para parar
6. Salvar tudo automaticamente

### Modo 2: Uso Programático

```python
from src.realtime_transcriber import RealtimeTranscriber
from src.multi_speaker_verifier import MultiSpeakerVerifier

# Carrega verificador
verifier = MultiSpeakerVerifier("data/embeddings", threshold=0.65)

# Cria transcritor
transcriber = RealtimeTranscriber(
    verifier,
    hf_token="seu_token_aqui",
    whisper_size="small",           # ou "tiny" para mais rápido
    device="cuda",                   # ou "cpu"
    chunk_duration=2.0,              # segundos de áudio por chunk
    verifier_confidence_min=0.9,     # confiança mínima para verificação
)

# Inicia gravação
transcriber.start_recording(embeddings_dir="data/embeddings")
```

---

## 🎯 Arquitetura de Processamento

```
FLUXO EM TEMPO REAL:

┌─────────────────┐
│  Microfone      │  
└────────┬────────┘
         │ (100ms chunks)
         ▼
┌─────────────────┐     ┌──────────────────┐
│  Audio Queue    │────▶│ Processing Thread│
└────────┬────────┘     └────────┬─────────┘
         │                       │
         │                       ▼
         │              ┌──────────────────┐
         │              │  Diarização      │
         │              │  (pyannote)      │
         │              └────────┬─────────┘
         │                       │
         │                       ▼
         │    ┌──────────────────────────────┐
         │    │  Detecta Overlap?            │
         │    └──────┬──────────────┬────────┘
         │           │              │
         │    NÃO    │              │ SIM
         │     ▼     │              ▼
         │  ┌────┐   │    ┌──────────────────┐
         │  │Wh  │   │    │  SepFormer       │
         │  │isp │   │    │  (Separação)     │
         │  │er  │   │    └──────┬───────────┘
         │  └────┘   │           │
         │     │     │           ▼
         │     │     │    ┌──────────────────┐
         │     │     │    │ Whisper 1 & 2    │
         │     │     │    │ (2 transcrições) │
         │     │     │    └────────┬─────────┘
         │     │     │             │
         │     └──────────┬────────┘
         │                ▼
         │       ┌──────────────────┐
         │       │ Exibe em Tempo   │
         │       │ Real no Console  │
         │       └────────┬─────────┘
         │                │
         │     ┌──────────┴────────┐
         │     ▼                   ▼
         │ ┌────────────┐  ┌──────────────┐
         │ │ Detecta   │  │ Coleta Áudio │
         │ │ Nomes     │  │ para Embedding
         │ │ (NER+LLM) │  │ (Unknowns)   │
         │ └───────────┘  └──────────────┘
         │
         └──▶  [Continua enquanto grava...]

FIM DA GRAVAÇÃO (Ctrl+C):
         │
         ▼
┌──────────────────────────────────┐
│  Salva Transcrição               │
│  Salva Speakers Identificados    │
│  Cria Embeddings (se houver      │
│  novos speakers com nomes)       │
└──────────────────────────────────┘
```

---

## 📊 Arquivos de Saída

Após a gravação, em `realtime_sessions/`:

### 1. `transcript_YYYYMMDD_HHMMSS.txt`
```
TRANSCRIÇÃO EM TEMPO REAL - 2026-02-13 18:30:45
============================================================

[18:30:45] Speaker_0: Olá, meu nome é Lucas
[18:30:52] Speaker_1: Oi Lucas! Sou o Pedro
[18:31:05] Speaker_0: Prazer, Pedro!
[18:31:15] Speaker_0 (Lucas) + Speaker_1 (Pedro) [Voz 1]: Sobreposição detectada
[18:31:22] Speaker_1 (Pedro): Qualquer coisa que precisar...
```

### 2. `speakers_YYYYMMDD_HHMMSS.txt`
```
SPEAKERS IDENTIFICADOS
============================================================

Speaker_0: Lucas
Speaker_1: Pedro
```

### 3. `embeddings/` (criados automaticamente se houver novos speakers)
```
data/embeddings/
├── Lucas_20260213_183045.npy
└── Pedro_20260213_183045.npy
```

---

## ⚙️ Parâmetros de Configuração

| Parâmetro | Padrão | Descrição |
|-----------|--------|-----------|
| `whisper_size` | "small" | "tiny" (rápido), "base", "small", "medium" |
| `chunk_duration` | 2.0 | Segundos de áudio por chunk (maior = mais latência) |
| `sample_rate` | 16000 | Taxa de amostragem (16kHz é padrão) |
| `verifier_confidence_min` | 0.8 | Confiança mínima (0-1) para verificação |
| `diarization_clustering_threshold` | 0.6 | Threshold para agrupar speakers (menor = mais speakers) |
| `device` | "cuda" | "cuda" (GPU) ou "cpu" |

### Exemplos de Uso:

**Modo RÁPIDO (para testes):**
```python
transcriber = RealtimeTranscriber(
    ...,
    whisper_size="tiny",           # Rápido
    chunk_duration=1.0,             # Baixa latência
    device="cpu"                    # Roda em CPU
)
```

**Modo PRECISO (análise profunda):**
```python
transcriber = RealtimeTranscriber(
    ...,
    whisper_size="medium",          # Mais preciso
    chunk_duration=3.0,             # Mais contexto
    device="cuda",                  # GPU para bem mais rápido
    verifier_confidence_min=0.95    # Mais rigoroso
)
```

---

## 🔧 Troubleshooting

### ❌ "Nenhum backend de áudio disponível"
```bash
# Instale sounddevice:
pip install -U sounddevice

# Ou PyAudio:
pip install PyAudio
```

### ❌ "CUDA out of memory"
```python
# Use CPU:
device="cpu"

# Ou use Whisper menor:
whisper_size="tiny"
```

### ❌ Transcrição muito lenta
```python
# Reduza chunk_duration:
chunk_duration=1.0

# Ou use Whisper menor:
whisper_size="tiny"
```

### ❌ Detecta muitos speakers falsos
```python
# Aumente o threshold de clustering:
diarization_clustering_threshold=0.7  # Antes: 0.45

# Aumente a confiança mínima:
verifier_confidence_min=0.95  # Antes: 0.8
```

### ❌ Token HuggingFace expirado
Veja a mensagem do erro anterior sobre atualizar o token HF_TOKEN.

---

## 📝 Exemplos de Saída em Tempo Real

```
============================================================
🔴 GRAVANDO EM TEMPO REAL
Fale normalmente (a transcrição aparecerá conforme você fala)
Pressione Ctrl+C para parar
============================================================

[18:30:45] Speaker_0: Olá, meu nome é Lucas

[18:30:52] Speaker_1: E eu sou o Pedro

[18:31:05] Speaker_0 (Lucas): Prazer, Pedro!

[18:31:15] Speaker_0 (Lucas) [Voz 1]: Você quer tomar um café?
[18:31:15] Speaker_1 (Pedro) [Voz 2]: Com prazer!

[18:31:22] Speaker_1 (Pedro): Qual é o melhor lugar por aqui?

✨ AUTO-APRESENTAÇÃO: Speaker_0 = Lucas
✨ AUTO-APRESENTAÇÃO: Speaker_1 = Pedro

^C (Ctrl+C pressionado)

⏹️ Parando gravação...
✅ Gravação finalizada!

============================================================
📊 SESSÃO FINALIZADA
  - Transcrições: 7
  - Speakers únicos: 2
  - Nomes detectados: 2
  - Arquivo: realtime_sessions/transcript_20260213_183045.txt
============================================================
```

---

## 🎓 Diferenças vs `multi_speaker_verification.py`

| Aspecto | Arquivo (batch) | Tempo Real (streaming) |
|--------|------------------|----------------------|
| Entrada | Arquivo WAV completo | Microfone em tempo real |
| Processamento | Após completo | Conforme chega |
| Latência | Minutos | Segundos |
| Diarização | Global (melhor) | Chunk-by-chunk |
| Saída | Arquivo único | Múltiplos chunks em tempo real |
| Use quando | Análise pós-gravação | Conversa ao vivo |
| Performance | Mais preciso | Mais rápido |

---

## 💡 Dicas de Uso

1. **Para melhor qualidade**: Use microfone dedicado, não o builtin do notebook
2. **Para melhor diarização**: Fale em turnos (um de cada vez), não sobreposições
3. **Para detecção de nomes**: Apresente-se explicitamente no começo ("Meu nome é Lucas")
4. **Para embeddings**: Fale por 30+ segundos para gerar embedding robusto
5. **Para Ctrl+C**: Pressione Ctrl+C uma vez e aguarde a parada graciosa

---

## 📞 Suporte

Se encontrar problemas:
1. Verifique o arquivo de log (logs/)
2. Teste sua entrada de áudio: `python -c "import sounddevice; print(sounddevice.default_device())"` 
3. Tente com whisper_size="tiny" primeiro para descartar GPU/CPU issues
4. Revise os parâmetros de threshold

---

**Desenvolvido em 2026 - AV Tracker**
