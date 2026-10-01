# Dicionário dos dados públicos

Todos os arquivos usam UTF-8, cabeçalho na primeira linha, vírgula como
separador e ponto como separador decimal.

## `data/sessions.csv`

| Coluna | Descrição |
|---|---|
| `anonymous_session_id` | Código sequencial sem vínculo com o identificador privado |
| `session_order` | Ordem relativa das sessões, sem data absoluta |
| `expected_participants` | Quantidade de participantes configurada para a sessão |
| `segment_count` | Segmentos aceitos como fala na sessão |
| `whisper_model_size` | Tamanho do modelo de transcrição registrado |
| `sample_rate_hz` | Taxa de amostragem do áudio |
| `verifier_threshold` | Limiar mínimo da verificação vocal |
| `verifier_confidence_min` | Limiar de confiança vocal alta |
| `face_match_threshold` | Limiar de associação facial, quando registrado |
| `face_merge_threshold` | Limiar de fusão de trajetórias, quando registrado |

## `data/segments.csv`

| Coluna | Descrição |
|---|---|
| `anonymous_session_id` | Sessão anônima do segmento |
| `segment_index` | Índice relativo do segmento na sessão |
| `decision_type` | Regra interna que produziu a atribuição |
| `evidence_group` | Agrupamento usado nas tabelas do artigo |
| `raw_confidence` | Confiança bruta disponível para a decisão |
| `risk_group` | `flagged` para risco médio/alto; caso contrário, `unflagged` |
| `identified` | 1 quando o sistema produziu uma identidade; 0 caso contrário |
| `possible_false_negative` | Indicador interno de possível falso negativo |
| `active_speaker_detected` | Indica se houve evidência de locutor ativo |
| `detected_faces` | Quantidade de faces detectadas no momento da decisão |
| `segment_duration_s` | Duração da fala em segundos |
| `transcription_ms` | Latência da transcrição em milissegundos |
| `voice_verification_ms` | Latência da verificação vocal em milissegundos |
| `total_processing_ms` | Latência total registrada em milissegundos |
| `real_time_factor` | `total_processing_ms / (segment_duration_s × 1000)` |

## `data/validation.csv`

| Coluna | Descrição |
|---|---|
| `anonymous_session_id` | Sessão anônima validada |
| `segment_index` | Índice do segmento validado |
| `predicted_named` | 1 quando a previsão é uma identidade não genérica |
| `exact_match` | 1 quando previsão e correção humana coincidem |
| `risk_group` | Grupo de risco associado ao segmento |

Os rótulos previsto e corrigido não são publicados. As duas variáveis binárias
acima preservam exatamente as estatísticas necessárias sem revelar identidades.
