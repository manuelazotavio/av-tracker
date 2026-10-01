# Privacidade, anonimato e auditoria

## Dados incluídos

O artefato contém somente identificadores sequenciais, parâmetros técnicos,
tipos de decisão, indicadores binários e tempos de processamento. Esses campos
são suficientes para reproduzir as tabelas e os testes estatísticos.

## Dados excluídos

- nomes, rótulos de locutores e identificadores originais;
- transcrições e conteúdo das falas;
- datas, horários e timestamps absolutos;
- áudio, vídeo e recortes de face;
- representações biométricas de voz e face;
- listas de candidatos e atributos pessoais;
- `.env`, tokens, caminhos locais, histórico Git e metadados de autoria.

## Auditoria automática

`scripts/export_from_private_db.py` abre o banco em modo somente leitura, usa
uma lista fechada de campos públicos e valida contagens de controle. Depois da
exportação, `data/privacy_report.json` registra as categorias removidas, o
número de linhas e o SHA-256 dos três CSVs.

Antes do envio do ZIP, o manifesto de hashes permite verificar que os arquivos
não foram alterados. O pacote deve ser publicado por um serviço de anonimização
de repositórios ou anexado como material suplementar sem histórico Git.
