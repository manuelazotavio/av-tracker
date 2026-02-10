# 🎯 DOWNLOAD E PROCESSAMENTO INTELIGENTE DO VOXCELEB2

## 1️⃣ INSTALAR DEPENDÊNCIAS

```bash
pip install datasets huggingface-hub
huggingface-cli login
```

Crie uma conta em: https://huggingface.co/join
Token: https://huggingface.co/settings/tokens

---

## 2️⃣ BAIXAR ÁUDIOS DO VOXCELEB2

### Opção A: Baixar primeiros 100 áudios (teste)
```bash
python download_voxceleb2.py --limit 100
```

### Opção B: Baixar mais arquivos
```bash
python download_voxceleb2.py --limit 500 --output-dir data/voxceleb_downloaded
```

### Opção C: Baixar tudo
```bash
python download_voxceleb2.py --limit None --output-dir data/voxceleb_downloaded
```

**Saída esperada:**
```
======================================================================
DOWNLOAD VOXCELEB2 DO HUGGING FACE
======================================================================

📥 Carregando dataset VoxCeleb2...
✅ Dataset carregado (streaming mode)

   [50] Baixados: 50 | Pulados: 0 | Erros: 0
   [100] Baixados: 100 | Pulados: 0 | Erros: 0

======================================================================
RESUMO DO DOWNLOAD
======================================================================
✅ Baixados: 100
⏭️  Pulados (já existem): 0
❌ Erros: 0
📁 Diretório: data/voxceleb_downloaded
📊 Total geral: 100
======================================================================
```

**Estrutura criada:**
```
data/voxceleb_downloaded/
├── speaker_001/
│   ├── video_001.wav
│   ├── video_002.wav
│   └── ...
├── speaker_002/
│   ├── video_001.wav
│   └── ...
└── ...
```

---

## 3️⃣ SISTEMA DE RASTREAMENTO

### Ver Estatísticas
```bash
python processing_tracker.py --stats
```

**Saída:**
```
======================================================================
ESTATÍSTICAS DE PROCESSAMENTO
======================================================================

📊 GERAL:
   Total processados: 10
   Última atualização: 2026-02-05T10:30:45

📁 POR FONTE:
   DEMO:
      Processados: 10
      Pendentes: 258
      Total: 268

   VOXCELEB_DOWNLOADED:
      Processados: 0
      Pendentes: 100
      Total: 100

⏳ RESUMO:
   Pendentes no total: 358
```

### Atualizar Contagem de Arquivos
```bash
python processing_tracker.py --update-sources
```

### Listar Arquivos Pendentes
```bash
python processing_tracker.py --list-pending
```

---

## 4️⃣ PROCESSAR APENAS NOVOS ARQUIVOS

### Processar com rastreamento automático
```bash
python process_with_tracker.py
```

**Características:**
- ✅ Detecta automaticamente novos arquivos
- ✅ Não reprocessa arquivos já feitos
- ✅ Detecta mudanças em arquivos (via hash)
- ✅ Salva histórico em JSON
- ✅ Append aos resultados existentes
- ✅ Estatísticas em tempo real

**Saída esperada:**
```
======================================================================
PROCESSADOR INTELIGENTE COM RASTREAMENTO
======================================================================

🔍 Escaneando fontes...

======================================================================
ESTATÍSTICAS DE PROCESSAMENTO
======================================================================

📊 GERAL:
   Total processados: 10
   Última atualização: 2026-02-05T10:30:45

📁 POR FONTE:
   DEMO:
      Processados: 10
      Pendentes: 258
      Total: 268

   VOXCELEB_DOWNLOADED:
      Processados: 0
      Pendentes: 100
      Total: 100

⏳ RESUMO:
   Pendentes no total: 358

======================================================================
PROCESSAMENTO
======================================================================

[  1/358] 2actors.WAV                                        VAZIO
[  2/358] 3actors.WAV                                        OK ( 2 chunks)
[  3/358] 4actors.WAV                                        OK ( 2 chunks)
...
```

---

## 5️⃣ FLUXO COMPLETO

### Cenário: Você tem data/demo + dados baixados

**Passo 1: Verificar status**
```bash
python processing_tracker.py --stats
```

**Passo 2: Baixar mais áudios**
```bash
python download_voxceleb2.py --limit 200
```

**Passo 3: Atualizar contagem**
```bash
python processing_tracker.py --update-sources
```

**Passo 4: Processar apenas os novos**
```bash
python process_with_tracker.py
```

**Passo 5: Regenerar dashboard**
```bash
python generate_html_report.py
```

**Passo 6: Iniciar servidor**
```bash
python web_dashboard.py
```

---

## 📊 ARQUIVO DE RASTREAMENTO

O arquivo `processing_tracker.json` contém:

```json
{
  "version": "1.0",
  "processed_files": {
    "data/demo/2actors.WAV": {
      "hash": "a1b2c3d4e5f6",
      "size": 1024000,
      "date_processed": "2026-02-05T10:30:45",
      "results": {
        "chunks_processed": 2,
        "identified_count": 1,
        "avg_similarity": "0.4853"
      }
    }
  },
  "processing_stats": {
    "total_processed": 10,
    "last_processed": "2026-02-05T10:30:45"
  },
  "sources": {
    "demo": {"total": 268, "processed": 10},
    "voxceleb_downloaded": {"total": 100, "processed": 5}
  }
}
```

---

## 🔄 O QUE ACONTECE QUANDO...

### Você baixa novos arquivos
1. Script detecta automaticamente
2. Mostra no `--list-pending`
3. Processa apenas os novos
4. Adiciona ao `results.csv` (append, não sobrescreve)

### Você modifica um arquivo existente
1. Hash muda
2. Sistema detecta mudança
3. Reprocessa apenas esse arquivo

### Você quer reprocessar tudo
```bash
rm processing_tracker.json
python process_with_tracker.py
```

---

## ⚡ DICAS DE USO

### Processamento contínuo
```bash
# Em um terminal, deixe rodando em loop
while true; do
    python process_with_tracker.py
    echo "Aguardando 1 hora antes de verificar novos arquivos..."
    sleep 3600
done
```

### Monitorar em tempo real
```bash
# Em outro terminal
while true; do
    python processing_tracker.py --stats
    sleep 30
done
```

### Integração com cron (Linux/Mac)
```bash
# Processar a cada 6 horas
0 */6 * * * cd /caminho/av-tracker && python process_with_tracker.py
```

---

## 🐛 RESOLUÇÃO DE PROBLEMAS

### Erro: "AuthenticationError"
```
Solução: 
  huggingface-cli login
  Copie seu token de https://huggingface.co/settings/tokens
```

### Erro: "CUDA out of memory"
```
Solução:
  - Usar CPU (automático se GPU não disponível)
  - Processar menos arquivos por vez
  - Aumentar intervalo de limpeza de memória
```

### Muitos arquivos "Error"
```
Solução:
  - Verificar se arquivos estão corrompidos
  - Tentar redownload com --limit menor
  - Verificar espaço em disco (data/voxceleb_downloaded/)
```

---

## 📈 RESUMO DE FUNCIONAMENTO

| Ação | Comando | Resultado |
|------|---------|-----------|
| Ver status | `python processing_tracker.py --stats` | Mostra quantos já processou |
| Baixar áudios | `python download_voxceleb2.py --limit 500` | Baixa 500 áudios novos |
| Listar pendentes | `python processing_tracker.py --list-pending` | Mostra o que falta processar |
| Processar | `python process_with_tracker.py` | Processa APENAS os novos |
| Regenerar dashboard | `python generate_html_report.py` | Atualiza o relatório HTML |

---

**Tudo funciona de forma automática e inteligente! 🚀**
