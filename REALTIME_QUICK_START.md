# 🚀 Quick Start - Transcritor em Tempo Real

## 30 Segundos Setup

### 1️⃣ Instale o backend de áudio
```bash
pip install sounddevice
```

### 2️⃣ Execute
```bash
python run_realtime.py
```

### 3️⃣ Pronto!
- Pressione ENTER
- Fale no microfone
- Veja a transcrição em tempo real
- Ctrl+C para parar
- Resultado salvo em `realtime_sessions/`

---

## ⚡ Modo Ultra-Rápido (sem GPU)

Se sua GPU não tem memória suficiente:

```bash
# Edite run_realtime.py, linha ~70, mude:
device="cuda"  → device="cpu"
whisper_size="small" → whisper_size="tiny"

# Depois execute:
python run_realtime.py
```

---

## 🎯 Modo Programático (Código)

```python
from src.realtime_transcriber import RealtimeTranscriber
from src.multi_speaker_verifier import MultiSpeakerVerifier

# Carrega
verifier = MultiSpeakerVerifier("data/embeddings")
transcriber = RealtimeTranscriber(verifier, "seu_hf_token_aqui")

# Grava
transcriber.start_recording("data/embeddings")
```

---

## 📚 Mais Informações

Ver `REALTIME_GUIDE.md` para:
- Configuração detalhada
- Troubleshooting
- Exemplos avançados
- Parâmetros de ajuste

---

## ✅ Checklist Antes de Usar

- [ ] Microfone conectado e funcionando?
- [ ] sounddevice instalado: `pip install sounddevice`
- [ ] data/embeddings/ existe?
- [ ] HF_TOKEN válido em `src/multi_speaker_verification.py`?

---

**Tudo pronto? Execute:** `python run_realtime.py`
