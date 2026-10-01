# PIBIFSP — PROJETO DE PESQUISA

## TÍTULO DO PROJETO

**AV-Tracker: atribuição e reidentificação multimodal de participantes em reuniões híbridas**

**Área do Conhecimento (Tabela do CNPq):** 1.03.04.00-2 — Sistemas de Computação.

## RESUMO

Este projeto investiga, implementa e avalia o AV-Tracker, um sistema multimodal para atribuir segmentos de fala aos participantes de reuniões presenciais e híbridas. A identificação de quem está falando é uma etapa essencial para produzir transcrições estruturadas, atas automáticas e registros pesquisáveis, mas permanece desafiadora em situações com ruído, sobreposição de vozes, mudanças de enquadramento, oclusões faciais e participantes com pouco material de cadastro. Para reduzir a dependência de uma única modalidade, o AV-Tracker combina diarização e verificação de locutor, reconhecimento facial, rastreamento de pessoas e detecção de locutor ativo por movimento labial. O fluxo de áudio utiliza modelos pré-treinados para diarização, representações vocais e transcrição; o fluxo de vídeo detecta, rastreia e reidentifica faces, além de estimar qual pessoa visível está falando. Uma cascata de decisão integra essas evidências conforme os níveis de confiança e mantém a coerência das identidades ao longo da sessão. O sistema também registra em banco de dados a fonte de cada decisão, as pontuações dos modelos, as latências e os indicadores de risco, permitindo auditoria e análise estatística. A avaliação considera cobertura operacional, desempenho em tempo real e acurácia em segmentos corrigidos manualmente, distinguindo explicitamente uma decisão produzida pelo sistema de uma identidade efetivamente correta. Espera-se obter um protótipo reprodutível, documentar suas limitações e estabelecer um protocolo de validação para futuras versões. O projeto está vinculado ao Grupo de Pesquisas em Tecnologias Inovadoras (i2) do IFSP — Campus Caraguatatuba e é desenvolvido em colaboração com pesquisadores da GN Advanced Science, da GN Hearing A/S (Jabra), e da IT University of Copenhagen (ITU), fortalecendo a cooperação entre academia e indústria.

## METAS

| Meta | Descrição |
|---:|---|
| 1 | Estudar identificação, verificação e diarização de locutores em reuniões. |
| 2 | Revisar a literatura sobre representações vocais, reconhecimento facial, rastreamento e detecção audiovisual de locutor ativo. |
| 3 | Implementar e testar o fluxo de áudio: diarização, transcrição segmentada, representação vocal e verificação de locutor. |
| 4 | Implementar e testar o fluxo de vídeo: detecção e rastreamento de pessoas, representação facial e detecção de locutor ativo. |
| 5 | Integrar os fluxos em uma cascata multimodal de atribuição e reidentificação. |
| 6 | Instrumentar o sistema para registrar decisões, pontuações, configurações e latências em banco de dados. |
| 7 | Entregar o relatório parcial até 01/07/2026. |
| 8 | Executar experimentos, realizar correção humana de uma amostra e analisar os resultados com métodos estatísticos adequados. |
| 9 | Documentar o sistema, preparar artigo para evento interno e/ou externo e organizar o repositório para divulgação responsável. |
| 10 | Entregar o relatório final até 30/11/2026. |

## CRONOGRAMA

| Meta | Mar. | Abr. | Maio | Jun. | Jul. | Ago. | Set. | Out. | Nov. |
|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | X | X |  |  |  |  |  |  |  |
| 2 | X | X | X |  |  |  |  |  |  |
| 3 |  | X | X | X |  |  |  |  |  |
| 4 |  |  | X | X | X |  |  |  |  |
| 5 |  |  |  | X | X | X |  |  |  |
| 6 |  |  |  | X | X | X |  |  |  |
| 7 |  |  |  | X | X |  |  |  |  |
| 8 |  |  |  |  | X | X | X | X |  |
| 9 |  |  |  |  |  | X | X | X | X |
| 10 |  |  |  |  |  |  |  |  | X |

## FUNDAMENTAÇÃO TEÓRICA

### Identificação, verificação e diarização de locutores

A atribuição de fala em reuniões envolve tarefas relacionadas, mas distintas. A verificação de locutor determina se uma amostra vocal é compatível com uma identidade conhecida. A identificação compara o segmento com um conjunto de participantes cadastrados. A diarização, por sua vez, organiza o áudio no tempo e procura responder “quem falou quando”, normalmente utilizando rótulos anônimos. Em uma reunião real, esses rótulos podem variar entre janelas de processamento e não correspondem automaticamente aos nomes dos participantes. Consequentemente, um sistema de atas ou transcrição precisa associar os agrupamentos acústicos às identidades e preservar essa associação ao longo da sessão.

Modelos como o ECAPA-TDNN extraem vetores numéricos, denominados *embeddings*, que concentram características da voz relevantes para a verificação de locutor (DESPLANQUES; THIENPONDT; DEMUYNCK, 2020). A comparação entre o segmento observado e os perfis previamente cadastrados pode ser realizada por similaridade do cosseno. No AV-Tracker, esses modelos são utilizados como componentes pré-treinados; o projeto não pressupõe o treinamento de uma nova arquitetura completa sobre VoxCeleb1 ou VoxCeleb2. O foco está na integração, na calibração dos limiares e na avaliação do comportamento do sistema em reuniões locais.

O pyannote.audio oferece modelos e procedimentos para segmentação e diarização de locutores, com blocos voltados à detecção de fala e à organização temporal dos turnos (BREDIN, 2023). A avaliação desse componente deve permanecer separada da avaliação da identidade final: um segmento pode estar temporalmente correto, mas associado à pessoa errada. Ferramentas como o pyannote.metrics reforçam a necessidade de métricas reprodutíveis e de análises de erro que distingam essas etapas (BREDIN, 2017).

### Transcrição e separação de fontes

Depois da diarização, cada turno pode ser transcrito separadamente. O Whisper demonstrou robustez em reconhecimento de fala multilíngue por meio de treinamento em grande escala (RADFORD et al., 2023). A implementação utiliza o faster-whisper como mecanismo de inferência e aplica filtros de atividade vocal e de ausência de fala antes de registrar um segmento. Dessa forma, os segmentos armazenados representam trechos aceitos operacionalmente como fala; eles não constituem, por si só, rótulos humanos de presença de fala.

Reuniões também apresentam sobreposição, quando duas ou mais pessoas falam simultaneamente. O SepFormer utiliza uma arquitetura baseada em atenção para separar fontes de áudio e pode apoiar o processamento de trechos sobrepostos (SUBAKAN et al., 2021). A separação, contudo, introduz outra possível fonte de erro e precisa ser registrada na análise do fluxo completo.

### Reidentificação visual e rastreamento

A voz pode ser insuficiente quando o trecho é curto, ruidoso ou semelhante ao perfil de outro participante. O vídeo oferece sinais complementares. O AV-Tracker detecta faces ou regiões de interesse com modelos da família YOLO (JOCHER; CHAURASIA; QIU, 2023), mantém trajetórias temporais com ByteTrack (ZHANG et al., 2022) e extrai representações faciais com EdgeFace, modelo eficiente voltado a dispositivos com recursos limitados (GEORGE et al., 2024). Essa combinação permite reconhecer uma pessoa cadastrada e manter sua identidade mesmo quando a detecção oscila entre quadros.

O rastreamento visual não indica necessariamente quem está falando. Para isso, a detecção de locutor ativo procura relacionar o áudio ao movimento facial, principalmente à região da boca. Conjuntos audiovisuais como AVA-ActiveSpeaker estabeleceram referências para essa tarefa (ROTH et al., 2020). Modelos como TalkNet e Light-ASD exploram dependências temporais entre áudio e vídeo para estimar qual face está associada à fala (TAO et al., 2021; LIAO et al., 2023). No AV-Tracker, o Light-ASD é combinado com uma alternativa baseada na variação visual da região da boca quando a pontuação neural não está disponível para a janela analisada.

### Integração multimodal em cascata

O AV-Tracker não utiliza, nesta etapa, um único classificador treinado para fundir todas as modalidades. O sistema emprega uma cascata auditável de regras. Evidências vocais de alta ou moderada confiança podem produzir uma atribuição direta; evidências vocais fracas exigem apoio de outro sinal. Quando a voz não é suficiente, o sistema consulta a detecção de locutor ativo, a identidade facial, indicadores acústicos auxiliares e as associações persistentes estabelecidas durante a sessão. A primeira regra que satisfaz as condições configuradas encerra a decisão.

Essa arquitetura oferece duas vantagens para a pesquisa. Primeiro, permite observar quanto cada modalidade contribui para a cobertura do sistema. Segundo, torna possível identificar qual regra originou um erro. Em contrapartida, a frequência de uso de uma regra não mede sua acurácia. Uma modalidade pode decidir muitos segmentos e ainda assim produzir atribuições incorretas. Por esse motivo, a avaliação deve combinar registros automáticos com correção humana independente.

### Avaliação, reprodutibilidade e análise estatística

A avaliação considera três dimensões. A primeira é a cobertura operacional: proporção de segmentos atribuídos por voz, detecção de locutor ativo, reconhecimento facial ou rastreamento de sessão. A segunda é o desempenho computacional, medido pela latência e pelo fator de tempo real, calculado como a razão entre o tempo total de processamento e a duração do segmento. A terceira é a acurácia da identidade final em segmentos que receberam correção humana.

Proporções de acerto devem ser acompanhadas por intervalos de confiança, especialmente quando os grupos possuem tamanhos diferentes. O intervalo de Wilson é apropriado para proporções binomiais e evita interpretações excessivamente otimistas em amostras pequenas (WILSON, 1927). Comparações entre grupos de risco podem utilizar o teste exato de Fisher. Além disso, a amostragem precisa ser descrita: uma amostra escolhida por conter sessões problemáticas é útil para diagnosticar falhas, mas não estima de forma imparcial o desempenho geral.

Para promover reprodutibilidade, o sistema registra configurações, pontuações, tipo da regra decisória, duração e latência por segmento. O artefato público deverá incluir código, esquema do banco, parâmetros e tabelas agregadas, mas não deverá distribuir nomes, áudios, vídeos ou representações biométricas identificáveis.

### Privacidade e uso responsável

Vozes e faces constituem dados biométricos sensíveis e exigem tratamento responsável. A coleta local deve ocorrer somente com autorização dos participantes e de acordo com os procedimentos éticos e institucionais aplicáveis. Perfis vocais, representações faciais e associações nominais devem permanecer sob controle de acesso. A divulgação científica deve priorizar resultados agregados e anonimização, evitando a publicação de gravações brutas ou vetores que possam facilitar reidentificação não autorizada. O sistema também deve manter a possibilidade de revisão humana, principalmente enquanto sua confiabilidade não estiver demonstrada em uma amostra representativa.

## OBJETIVO GERAL

Investigar, implementar e avaliar um sistema multimodal capaz de atribuir segmentos de fala aos participantes de reuniões presenciais e híbridas, integrando evidências de voz, face, movimento labial e coerência temporal da sessão.

## OBJETIVOS ESPECÍFICOS

Para atingir o objetivo geral, propõem-se os seguintes objetivos específicos:

1. realizar revisão da literatura sobre diarização, identificação e verificação de locutores, reconhecimento facial e detecção audiovisual de locutor ativo;
2. implementar o processamento segmentado de áudio, incluindo detecção de fala, diarização, transcrição e extração de representações vocais;
3. implementar o processamento de vídeo, incluindo detecção, rastreamento, representação facial e análise da atividade labial;
4. integrar as modalidades em uma cascata de decisão configurável e auditável;
5. criar mecanismos de cadastro e comparação de perfis vocais e faciais de participantes consentidos;
6. registrar em banco de dados as decisões, pontuações, configurações e latências necessárias à reprodução dos experimentos;
7. avaliar a cobertura de cada fonte de evidência e o desempenho computacional do sistema;
8. validar manualmente uma amostra de atribuições e calcular métricas acompanhadas de intervalos de confiança;
9. identificar limitações, possíveis vieses da amostra e situações que exigem revisão humana; e
10. documentar os resultados e preparar artigo técnico para apresentação em evento de iniciação científica.

## METODOLOGIA DE EXECUÇÃO DO PROJETO

O projeto possui natureza aplicada, computacional e experimental. As atividades combinam revisão bibliográfica, desenvolvimento de software, integração de modelos pré-treinados, execução de reuniões experimentais e análise quantitativa dos registros.

### 1. Revisão da literatura

Serão estudados trabalhos sobre representações de locutor, diarização, reconhecimento automático de fala, separação de fontes, reidentificação facial, rastreamento de pessoas, detecção de locutor ativo, fusão multimodal e avaliação estatística. A revisão orientará a seleção dos componentes e a interpretação dos resultados, sem transferir diretamente métricas obtidas em bases públicas para o cenário local.

### 2. Preparação do ambiente e cadastro de participantes

O ambiente será organizado em Python, PyTorch, OpenCV, pyannote.audio, SpeechBrain e bibliotecas auxiliares. Participantes autorizados poderão cadastrar amostras vocais e imagens faciais. O cadastro produzirá representações numéricas usadas apenas para comparação durante as sessões. Os dados identificáveis permanecerão fora do artefato público.

### 3. Fluxo de áudio

O áudio será processado em janelas deslizantes de 15 segundos, com avanço de 5 segundos. O pyannote.audio produzirá os turnos de fala, com duração mínima configurada de 0,5 segundo. Quando houver sobreposição detectada, o SepFormer poderá separar as fontes. Cada turno será transcrito separadamente pelo faster-whisper. O ECAPA-TDNN produzirá a representação vocal, que será comparada aos perfis cadastrados por similaridade. Os limiares serão configuráveis e registrados junto aos resultados.

### 4. Fluxo de vídeo

O fluxo de vídeo detectará as pessoas visíveis, manterá suas trajetórias com ByteTrack e comparará as representações faciais extraídas pelo EdgeFace com os perfis cadastrados. O Light-ASD estimará o locutor ativo ao relacionar áudio e vídeo. Quando o modelo não retornar pontuação para uma janela, será usada uma medida auxiliar de variação visual na região da boca. O sistema registrará pontuações, rastros e identidades candidatas.

### 5. Cascata multimodal

Os resultados serão integrados por uma cascata de decisão. A lógica considerará o nível de confiança da voz, a atividade labial, a identidade facial, indicadores acústicos auxiliares e associações previamente estabelecidas na sessão. A saída poderá conter o nome de um participante conhecido ou um rótulo anônimo quando não houver evidência suficiente. Cada decisão armazenará a regra que a produziu, permitindo análise posterior.

### 6. Coleta instrumental e experimentos

As sessões serão executadas em condições variadas de enquadramento, iluminação, nível de ruído, número de participantes e disponibilidade de cadastro. O banco SQLite armazenará somente os dados derivados necessários à auditoria. Até 4 de agosto de 2026, a base instrumental continha 106 sessões, das quais 92 possuíam segmentos aceitos como fala, totalizando 1.660 segmentos. Esses registros foram produzidos por conveniência durante o desenvolvimento e não constituem uma amostra populacional representativa.

### 7. Validação e análise

A análise utilizará a correção humana mais recente de cada sessão validada. A identidade prevista será comparada exatamente ao rótulo corrigido. Serão calculadas a acurácia global, a acurácia entre atribuições nomeadas, intervalos de confiança de Wilson e, quando aplicável, o teste exato de Fisher para comparar grupos sinalizados e não sinalizados como arriscados. Os resultados serão estratificados pelo número esperado de participantes e pela fonte da decisão. Uma etapa futura deverá ampliar a validação com seleção prospectiva e estratificada de sessões.

### 8. Documentação e disseminação

O repositório público incluirá código, instruções de instalação, esquema do banco, parâmetros de execução, scripts de análise e resultados agregados. Arquivos de áudio, vídeo, nomes e representações biométricas não serão publicados. Os achados serão registrados em relatório e artigo científico, incluindo resultados negativos e limitações.

## VIABILIDADE DE EXECUÇÃO

O projeto é viável porque utiliza predominantemente software livre e modelos pré-treinados. O protótipo vem sendo executado em computador com processador Intel Core i5-11400H, 8 GB de memória e GPU NVIDIA RTX 3060 Laptop com 6 GB de memória de vídeo. Embora essa configuração imponha limites ao treinamento de modelos de grande porte, ela é suficiente para integração, inferência e experimentos em escala reduzida. O uso de modelos pré-treinados reduz o custo computacional e concentra o trabalho na engenharia do sistema, na calibração e na avaliação.

O IFSP — Campus Caraguatatuba oferece laboratórios, acesso à internet e orientação acadêmica. O projeto prevê um bolsista de iniciação científica, acompanhado pelos orientadores. A cooperação formal entre o IFSP e a GN Hearing A/S (Jabra), estabelecida por Memorandum of Understanding desde 2023, permite contribuição técnica voluntária de pesquisadores da GN Advanced Science. A colaboração com a IT University of Copenhagen amplia a interlocução científica. Qualquer participação adicional deverá respeitar as regras institucionais, a confidencialidade e a proteção dos dados dos participantes.

Os principais riscos de execução são a limitação do hardware, a dependência de modelos pré-treinados, a variabilidade das condições de reunião e a disponibilidade de rótulos humanos. Esses riscos serão mitigados por processamento segmentado, registro detalhado das configurações, amostragem planejada e divulgação transparente das limitações.

## RESULTADOS ESPERADOS E DISSEMINAÇÃO

Ao final do projeto, espera-se obter:

1. um protótipo multimodal funcional para atribuição de fala em reuniões;
2. um fluxo integrado de diarização, transcrição, verificação de voz, reidentificação facial, rastreamento e detecção de locutor ativo;
3. um banco de dados instrumental que permita auditar a origem e a latência de cada decisão;
4. um protocolo de validação humana e estatística adequado à tarefa;
5. documentação técnica e scripts reprodutíveis sem exposição de dados biométricos;
6. um relatório técnico detalhado;
7. um artigo para o seminário interno de iniciação científica e possível submissão ao CONICT ou a outro evento; e
8. recomendações para aperfeiçoamento do sistema e para uma futura avaliação com amostra prospectiva.

### Resultados parciais disponíveis

A auditoria realizada até 4 de agosto de 2026 confirma que o protótipo processa áudio e vídeo, registra as decisões em SQLite e opera mais rapidamente que a duração do segmento em 89,6% dos 1.660 segmentos observados. A voz sem corroboração decidiu 67 segmentos (4,0%); 76 segmentos (4,6%) utilizaram voz fraca com corroboração, enquanto os demais dependeram de detecção de locutor ativo, evidência facial ou rastreamento de sessão.

Uma amostra de conveniência com 330 segmentos corrigidos manualmente apresentou 11,2% de acurácia exata. Como essa amostra foi selecionada durante o desenvolvimento por conter problemas conhecidos, o resultado não deve ser extrapolado para todas as reuniões. Ele demonstra, contudo, a importância da instrumentação e da revisão humana e orienta as próximas etapas. O resultado esperado não é apenas elevar uma métrica, mas produzir evidência confiável sobre quando o sistema funciona, quando falha e quais modalidades contribuem para cada decisão.

Os resultados parciais e finais poderão ser divulgados na SIC-LN e/ou no CONICT. Uma publicação externa dependerá da qualidade da validação, do cumprimento dos requisitos éticos e da autorização dos responsáveis. A continuidade do trabalho poderá aprofundar a calibração dos modelos, a avaliação de sobreposição de fala, a fusão aprendida entre modalidades e a cooperação entre IFSP, GN Advanced Science e ITU.

## REFERÊNCIAS BIBLIOGRÁFICAS

BREDIN, Hervé. pyannote.audio 2.1 speaker diarization pipeline: principle, benchmark, and recipe. In: **Interspeech 2023**. [S. l.]: ISCA, 2023. p. 1983–1987. DOI: 10.21437/Interspeech.2023-105.

BREDIN, Hervé. pyannote.metrics: a toolkit for reproducible evaluation, diagnostic, and error analysis of speaker diarization systems. In: **Interspeech 2017**. [S. l.]: ISCA, 2017. p. 3587–3591. DOI: 10.21437/Interspeech.2017-411.

DESPLANQUES, Brecht; THIENPONDT, Jenthe; DEMUYNCK, Kris. ECAPA-TDNN: emphasized channel attention, propagation and aggregation in TDNN-based speaker verification. In: **Interspeech 2020**. [S. l.]: ISCA, 2020. p. 3830–3834. DOI: 10.21437/Interspeech.2020-2650.

GEORGE, Anjith et al. EdgeFace: efficient face recognition model for edge devices. **IEEE Transactions on Biometrics, Behavior, and Identity Science**, p. 147–157, 2024. DOI: 10.1109/TBIOM.2024.3352164.

JOCHER, Glenn; CHAURASIA, Ayush; QIU, Jing. **Ultralytics YOLOv8**. [S. l.], 2023. Software. Disponível em: https://github.com/ultralytics/ultralytics. Acesso em: 4 ago. 2026.

LIAO, Junhua et al. A light weight model for active speaker detection. In: **Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition**. [S. l.: s. n.], 2023. p. 22932–22941.

RADFORD, Alec et al. Robust speech recognition via large-scale weak supervision. In: **Proceedings of the 40th International Conference on Machine Learning**. [S. l.]: PMLR, 2023. v. 202, p. 28492–28518.

ROTH, Joseph et al. AVA-ActiveSpeaker: an audio-visual dataset for active speaker detection. In: **IEEE International Conference on Acoustics, Speech and Signal Processing**. [S. l.]: IEEE, 2020. p. 4492–4496. DOI: 10.1109/ICASSP40776.2020.9053900.

SUBAKAN, Cem et al. Attention is all you need in speech separation. In: **IEEE International Conference on Acoustics, Speech and Signal Processing**. [S. l.]: IEEE, 2021. p. 21–25. DOI: 10.1109/ICASSP39728.2021.9413901.

TAO, Ruijie et al. Is someone speaking? Exploring long-term temporal features for audio-visual active speaker detection. In: **Proceedings of the 29th ACM International Conference on Multimedia**. [S. l.: s. n.], 2021. p. 3927–3935.

WILSON, Edwin B. Probable inference, the law of succession, and statistical inference. **Journal of the American Statistical Association**, v. 22, n. 158, p. 209–212, 1927. DOI: 10.1080/01621459.1927.10502953.

ZHANG, Yifu et al. ByteTrack: multi-object tracking by associating every detection box. In: **European Conference on Computer Vision**. [S. l.]: Springer, 2022. p. 1–21.
