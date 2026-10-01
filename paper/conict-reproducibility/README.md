# Artefato anônimo de reprodutibilidade

Repositório anônimo: [https://github.com/anonimo826/anonimo](https://github.com/anonimo826/anonimo)

Este pacote permite reproduzir, a partir de dados derivados e anônimos, as
tabelas, as métricas de latência e a validação estatística apresentadas no
artigo submetido ao CONICT 2026. O pacote também inclui um snapshot do código
do sistema, os parâmetros registrados por sessão e o protocolo experimental.

## Reprodução rápida

Requisito: Python 3.10 ou superior. A análise usa somente a biblioteca padrão.

```bash
python scripts/analyze.py
```

O comando lê os arquivos de `data/`, confere as invariantes do experimento e
recria em `results/`:

- `table_decisions.csv`;
- `table_evidence_groups.csv`;
- `table_session_groups.csv`;
- `validation_statistics.csv`;
- `summary.json`.

O resultado esperado contém 92 sessões analisadas, 1.660 segmentos, 330
segmentos corrigidos por revisão humana, fator de tempo real abaixo de 1 em
1.488 segmentos e teste exato de Fisher bilateral com
`p = 0.0002936251`.

## Conteúdo

- `data/`: dados derivados, anônimos e suficientes para refazer a análise;
- `scripts/analyze.py`: análise determinística e testes estatísticos;
- `scripts/export_from_private_db.py`: exportação reexecutável pelos autores;
- `results/`: saídas de referência produzidas pelo script;
- `config/experiment.json`: modelos, limiares e hardware documentados;
- `system/`: snapshot anônimo do código necessário ao sistema e à validação;
- `PROTOCOL.md`: desenho experimental, critérios e ordem de execução;
- `DATA_DICTIONARY.md`: descrição de todas as colunas públicas;
- `PRIVACY.md`: itens incluídos, excluídos e procedimento de auditoria.

## Regeneração pelos autores

O banco privado permanece fora do repositório. Em um ambiente autorizado, os
dados públicos podem ser regenerados por:

```bash
python scripts/export_from_private_db.py --db av_tracker.db --output data
python scripts/analyze.py
```

O exportador abre o SQLite em modo somente leitura, troca os identificadores
por códigos `S001` a `S092` e interrompe a execução se as contagens de controle
forem diferentes das registradas no estudo.

## Escopo da reprodutibilidade

O pacote reproduz integralmente a análise quantitativa publicada. A inferência
fim a fim pode ser reexecutada com o código de `system/`, os modelos indicados
em `config/experiment.json` e uma nova coleta autorizada. A mídia original e
os perfis biométricos da base interna não são distribuídos por conterem dados
pessoais. Essa separação não impede a conferência das métricas apresentadas,
pois os dados derivados por segmento estão incluídos.

## Anonimato

O pacote não contém nomes de autores, instituição, URL pessoal, histórico Git
ou identificadores de participantes. Antes do envio, publique esta pasta em um
repositório anônimo ou envie `conict2026-reproducibility.zip` como material
suplementar da submissão.
