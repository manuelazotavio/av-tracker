# Snapshot do sistema

Este diretório contém o código do AV-Tracker usado como referência para o
experimento. O snapshot foi separado do histórico Git e teve exemplos de nomes
substituídos por identificadores genéricos para preservar a revisão anônima.

## Arquivos principais

- `run_multimodal_tracker.py`: execução integrada dos fluxos de áudio e vídeo;
- `enroll_speaker.py`: cadastro local autorizado de perfis;
- `src/realtime_transcriber.py`: diarização, transcrição e cascata de áudio;
- `src/personid_tracker.py`: associação facial persistente;
- `src/active_speaker_detector.py` e `src/light_asd_detector.py`: locutor ativo;
- `src/database.py`: esquema e persistência das decisões e métricas;
- `app/pages/validate.py`: interface usada na correção humana;
- `scripts/plot_metrics.py`: visualização das métricas armazenadas.

## Dependências e modelos

Os limites mínimos das dependências constam em `requirements.txt` e
`requirements-realtime.txt`. Os identificadores dos modelos e os parâmetros
experimentais estão em `../config/experiment.json`. Pesos de modelos não são
duplicados neste pacote; devem ser obtidos de suas fontes oficiais conforme os
identificadores documentados.

O acesso a modelos restritos do Hugging Face é feito por uma variável de
ambiente `HF_TOKEN`. Nenhum token está incluído no snapshot.

## Dados locais deliberadamente ausentes

O sistema cria ou lê `av_tracker.db`, diretórios de embeddings, gravações e
pesos locais. Esses itens estão bloqueados pelo `.gitignore` da raiz do
artefato e não são necessários para executar `scripts/analyze.py`.
