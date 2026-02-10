# 🚀 WORKFLOW COMPLETO: DO ZERO ATÉ MÉTRICAS FINAIS

## Pipeline Automático

### 1️⃣ BAIXAR 300 ATORES
```bash
python src/download.py
```
- Baixa 300 atores do VoxCeleb
- Pasta: `vox_100_atores/`
- Com retry automático se cair

### 2️⃣ CRIAR EMBEDDINGS DOS ATORES
```bash
python scripts/create_embeddings_from_csv.py --input vox_100_atores --output data/embeddings
```
- Cria perfil de voz de cada ator
- Salva em `data/embeddings/`

### 3️⃣ MIXAR ÁUDIOS (2, 3, 4, 5 SPEAKERS)
```bash
python scripts/create_and_mix_audio.py --speakers 5 --output data/mixed_audio
```
- Cria misturas automáticas
- 2 speakers: 50 arquivos
- 3 speakers: 50 arquivos
- 4 speakers: 50 arquivos
- 5 speakers: 50 arquivos
- Total: 200 arquivos mixados

### 4️⃣ PROCESSAR TODOS OS MIXADOS
```bash
python scripts/run_verifier.py -e data/embeddings -i data/mixed_audio -o results.csv
```
- Processa cada mixagem
- Identifica quem está falando
- Salva em `results.csv`

### 5️⃣ CALCULAR MÉTRICAS DETALHADAS
```bash
python src/calculate_metrics.py
```
- Acurácia por número de speakers (2, 3, 4, 5)
- Acurácia por gênero (M/F)
- Matriz de confusão
- Gráficos e tabelas

### 6️⃣ GERAR DASHBOARD
```bash
python generate_html_report.py
```
- Dashboard com todas as métricas
- Gráficos interativos
- Análise completa

---

## ⚡ EXECUTAR TUDO DE UMA VEZ

```bash
python workflow_complete.py
```

Este script executa todo o pipeline automaticamente!

---

## 📊 MÉTRICAS GERADAS

### Por Número de Speakers
- ✅ Taxa de acerto com 2 speakers
- ✅ Taxa de acerto com 3 speakers  
- ✅ Taxa de acerto com 4 speakers
- ✅ Taxa de acerto com 5 speakers

### Por Gênero
- ✅ Acurácia em vozes masculinas
- ✅ Acurácia em vozes femininas
- ✅ Comparação M vs F

### Métricas Gerais
- ✅ Precisão, Recall, F1-Score
- ✅ Matriz de Confusão
- ✅ Similaridade média/máx/mín
- ✅ Chunks identificados vs desconhecidos

---

## ⏱️ TEMPO ESTIMADO

- Download: 2-4 horas (300 atores)
- Embeddings: 1-2 horas
- Mixagem: 10-20 minutos
- Processamento: 1-2 horas
- Métricas: 5 minutos
- **TOTAL: ~5-8 horas**

---

## 🎯 ESTRUTURA FINAL

```
vox_100_atores/           ← 300 atores baixados
data/
  embeddings/             ← Perfis de voz (300 .npy files)
  mixed_audio/            ← Áudios mixados
    mix_2_speakers/       ← 50 arquivos
    mix_3_speakers/       ← 50 arquivos
    mix_4_speakers/       ← 50 arquivos
    mix_5_speakers/       ← 50 arquivos
  ground_truth/           ← Quem está em cada mix
results.csv               ← Resultados do processamento
analysis_results/         ← Métricas e gráficos
  metrics_by_speakers.csv
  metrics_by_gender.csv
  confusion_matrix.png
  accuracy_chart.png
```
