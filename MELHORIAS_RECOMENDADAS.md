# 🚀 Melhorias Recomendadas para Transcrição e Assertividade

## 📝 QUALIDADE DA TRANSCRIÇÃO

### 1. **Parâmetros do Whisper** ⭐ CRÍTICO
```python
# ATUAL (ruim):
segs, _ = self.whisper.transcribe(source_np, language="pt", beam_size=1, vad_filter=True)

# RECOMENDADO:
segs, _ = self.whisper.transcribe(
    source_np, 
    language="pt", 
    beam_size=5,           # Mais opções de decodificação
    best_of=5,             # Tenta 5 variações
    temperature=0.2,       # Menos randomicidade
    vad_filter=False       # VAD estava cortando áudio válido
)
```

### 2. **Compute Type do Whisper** ⭐ IMPORTANTE
```python
# ATUAL (int8 = perde qualidade):
self.whisper = WhisperModel(whisper_size, device="cpu", compute_type="int8")

# RECOMENDADO:
self.whisper = WhisperModel(
    whisper_size, 
    device="cpu", 
    compute_type="float16"  # Melhor qualidade que int8
)
```

### 3. **Normalização de Volume**
Antes de transcrever, normalizar o áudio para evitar clipping e melhorar qualidade:
```python
def normalize_audio(audio, target_db=-20):
    """Normaliza áudio para nível padrão"""
    # Calcular RMS
    rms = np.sqrt(np.mean(audio**2))
    if rms < 1e-6:
        return audio
    
    # Normalizar para target_db
    target_amplitude = 10**(target_db/20)
    scaling_factor = target_amplitude / rms
    return (audio * scaling_factor).astype(np.float32)
```

---

## 🎯 ASSERTIVIDADE DA DIARIZAÇÃO

### 4. **Threshold da Diarização**
```python
# ATUAL:
self.pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")

# RECOMENDADO (ajustar conforme necessário):
diarization = self.pipeline(audio_path, min_duration_off=0.5, min_duration_on=0.5)
```

### 5. **Filtrar segmentos muito curtos**
```python
# Aumentar limite de duração mínima:
if (turn.end - turn.start) < 0.5:  # Aumentar para 0.7s
    continue
```

---

## 🎤 ASSERTIVIDADE DO SPEAKER VERIFICATION

### 6. **Threshold de Confiança**
```python
# ATUAL (threshold=0.0 = aceita tudo):
verifier = MultiSpeakerVerifier(embeddings_dir, threshold=0.0)

# RECOMENDADO:
verifier = MultiSpeakerVerifier(embeddings_dir, threshold=0.5)
# Valores típicos: 0.3-0.7 (maior = mais rigoroso)
```

### 7. **Verificação de Duração Mínima**
```python
# Adicionar verificação antes de processar:
if len(source_np) < sample_rate * 0.5:  # Pelo menos 0.5s
    continue  # Muito curto para identificar
```

---

## 🤖 ASSERTIVIDADE NA DETECÇÃO DE NOMES

### 8. **Melhorar Fallback quando não tem spaCy**
Se NER falhar, usar padrões simples como fallback

### 9. **Confiança do LLM**
Adicionar score de confiança e só usar se > threshold

---

## 📊 MELHORIAS ESTRUTURAIS

### 10. **Estatísticas Finais**
```python
# Adicionar ao final do processamento:
logger.info(f"📊 ESTATÍSTICAS:")
logger.info(f"  - Total de segmentos processados: {total_segments}")
logger.info(f"  - Segmentos com transcrição: {transcribed_segments}")
logger.info(f"  - Speakers identificados: {len(speaker_names)}")
logger.info(f"  - Taxa de cobertura: {transcribed_segments/total_segments*100:.1f}%")
```

### 11. **Logging de Debug**
Adicionar modo verbose para diagnoose quando transcrição falha

### 12. **Cache de Modelos**
Não recarregar modelos a cada processamento

---

## 🔧 PRIORITY (Implementar Nessa Ordem)

1. ⭐⭐⭐ **Beam Size do Whisper** (5 ao invés de 1)
2. ⭐⭐⭐ **Compute Type** (float16 ao invés de int8)
3. ⭐⭐ **Normalização de Volume**
4. ⭐⭐ **Threshold de Verification** (0.5 ao invés de 0.0)
5. ⭐⭐ **Estatísticas Finais**
6. ⭐ **Duração Mínima de Segmentos**
7. ⭐ **Logging de Debug**

---

## 📈 IMPACTO ESPERADO

| Melhoria | Impacto na Qualidade | Tempo Extra |
|----------|-------------------|-----------|
| Beam Size 5 | +20-30% | +5-10% |
| Float16 | +10-15% | 0% |
| Normalização | +5-10% | +2% |
| Threshold 0.5 | -30% false positives | 0% |
| **TOTAL** | **+35-50%** | **+7-12%** |

