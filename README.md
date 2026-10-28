# AV-Tracker

O **AV-Tracker** é um sistema de reconhecimento e acompanhamento de participantes que combina áudio, vídeo e transcrição. O projeto integra identificação por voz, reconhecimento facial, rastreamento de pessoas e detecção de quem está falando em um fluxo multimodal, com ferramentas para executar sessões, consultar resultados e validar as identificações.

A aplicação atende a reuniões presenciais, reuniões capturadas na tela e análise de vídeos gravados. Seu painel em Streamlit reúne a operação e a exploração dos resultados em uma interface organizada.

## Recursos e benefícios

### Reconhecimento de voz e transcrição

- **Perfis vocais com ECAPA-TDNN:** embeddings extraídos com SpeechBrain permitem comparar a voz observada com participantes cadastrados.
- **Cadastro com múltiplas amostras:** o utilitário de inscrição aceita vários arquivos e trechos delimitados, proporcionando uma representação vocal enriquecida por diferentes gravações.
- **Transcrição com Faster-Whisper:** transforma a fala em texto e integra o conteúdo transcrito ao acompanhamento dos participantes.
- **Diarização com Pyannote:** fornece recursos para organizar a fala por locutor no processamento de áudio.
- **Separação com SepFormer:** compõe os recursos de processamento de falas sobrepostas disponíveis no transcritor.
- **Reconhecimento de nomes:** o processamento textual com spaCy e regras de apresentação auxilia a associação de nomes aos participantes.

### Reconhecimento visual e associação multimodal

- **Detecção facial com YOLO:** localiza os rostos presentes nos quadros para alimentar o reconhecimento visual.
- **Embeddings faciais com EdgeFace:** representam as características dos participantes para comparação e identificação.
- **Rastreamento com ByteTrack:** acompanha as detecções ao longo do vídeo e contribui para a continuidade das identidades.
- **Detecção de locutor ativo com Light-ASD:** utiliza evidências de áudio e vídeo para estimar qual participante está falando. O detector também oferece acompanhamento de movimento visual como alternativa operacional.
- **Registro compartilhado de participantes:** os componentes de áudio e vídeo compartilham identificadores e nomes, permitindo que as evidências das duas modalidades se complementem.
- **Visualização anotada:** apresenta identidades e informações do acompanhamento sobre o vídeo, com suporte a nomes acentuados.

### Gestão de sessões e análise

- **Painel Streamlit:** concentra sessões, métricas, validação e consulta de embeddings.
- **Histórico de transcrições:** facilita a consulta aos registros produzidos durante as sessões.
- **Persistência em SQLite:** organiza participantes, embeddings, sessões, segmentos e validações no banco `av_tracker.db`.
- **Métricas em JSON e gráficos:** apoiam a análise de cada sessão e a comparação entre execuções.
- **Validação humana:** permite revisar atribuições de locutores e produzir transcrições validadas e indicadores por participante.
- **Materiais de pesquisa:** o diretório `paper/` reúne artigos, protocolos, resultados e ferramentas de reprodutibilidade.

## Como o sistema funciona

1. **Entrada:** a sessão recebe imagens da webcam, de um arquivo de vídeo ou da captura de tela, junto ao áudio correspondente.
2. **Análise de áudio:** o sistema processa a fala, extrai representações vocais e produz transcrições.
3. **Análise visual:** os rostos são detectados, representados por embeddings e acompanhados entre quadros.
4. **Associação:** o registro compartilhado e a detecção de locutor ativo conectam identidades vocais, participantes visíveis e nomes.
5. **Registro:** a aplicação salva os artefatos da sessão e as informações estruturadas para consulta e validação.
6. **Exploração:** o painel e os scripts oferecem acesso ao histórico, às métricas e à revisão dos resultados.

O ponto de entrada desse fluxo é `run_multimodal_tracker.py`, que coordena os componentes de áudio e vídeo.

## Preparação do ambiente

### Componentes utilizados

Prepare um ambiente com **Python 3.11**, Git e FFmpeg disponível no `PATH`. Para captura ao vivo, utilize uma câmera e um dispositivo de entrada de áudio. O processamento multimodal seleciona CUDA quando uma GPU compatível está disponível e também oferece execução em CPU.

Os exemplos abaixo usam PowerShell e um ambiente Conda chamado `tracker`, o mesmo nome utilizado pelos lançadores Windows do projeto.

```powershell
git clone https://github.com/manuelazotavio/av-tracker.git
cd av-tracker
conda create -n tracker python=3.11 -y
conda activate tracker
python -m pip install -r requirements.txt
python -m pip install -r requirements-realtime.txt
python -m pip install opencv-python ultralytics timm boxmot python-dotenv mss soundfile matplotlib
python -m spacy download en_core_web_sm
python -m spacy download pt_core_news_sm
```

Os arquivos `requirements.txt` e `requirements-realtime.txt` reúnem as dependências da aplicação e do processamento de áudio. O comando complementar prepara os componentes de visão computacional, captura de tela, configuração e gráficos usados no fluxo multimodal.

### Token e modelos

Configure o token do Hugging Face na sessão do PowerShell:

```powershell
$env:HF_TOKEN = "seu_token_hugging_face"
```

O fluxo multimodal também carrega a configuração de um arquivo `.env` na raiz, no formato `HF_TOKEN=seu_token_hugging_face`. Utilize uma conta com acesso autorizado aos modelos Pyannote empregados na execução. Os componentes SpeechBrain e outros modelos integrados utilizam downloads e caches locais durante sua inicialização.

Organize os pesos nos caminhos utilizados pelo código:

| Componente | Caminho esperado | Papel no sistema |
| --- | --- | --- |
| YOLO Face | `od_model/yolov8n-face.pt` | Detecção dos rostos nos quadros |
| EdgeFace | `od_model/edgeface_xxs.pt` | Extração de embeddings faciais |
| Light-ASD | `pretrained_models/Light-ASD-repo/weight/pretrain_AVA_CVPR.model` | Associação audiovisual de fala ativa |

Para usar Light-ASD, disponibilize seu checkpoint no caminho indicado. O projeto também inclui um detector visual de atividade de fala integrado ao mesmo componente de alto nível.

## Executando o AV-Tracker

Execute os comandos a partir da raiz do repositório, com o ambiente ativado e os modelos preparados.

### Painel interativo

```powershell
streamlit run app/app.py
```

No Windows, o arquivo `AV-Tracker.bat` também inicia o painel usando o ambiente Conda `tracker` instalado em `%USERPROFILE%\miniconda3`. O lançamento de sessões pelo painel utiliza essa mesma configuração.

O painel oferece quatro áreas:

| Área | O que oferece |
| --- | --- |
| **Sessions** | Seleção de modo de captura, início de sessões e consulta ao histórico de transcrições |
| **Metrics** | Exploração dos indicadores registrados nas sessões |
| **Validate** | Revisão das atribuições de locutores e acompanhamento da validação |
| **Embeddings** | Consulta e gerenciamento dos perfis de identificação |

### Modo interativo no terminal

```powershell
python run_multimodal_tracker.py
```

O assistente solicita o modo de execução, o dispositivo de áudio e a quantidade de participantes. Durante a visualização, pressione **q** para encerrar a sessão e iniciar o salvamento dos registros.

### Webcam e microfone

```powershell
python run_multimodal_tracker.py --speakers 4
```

Esse modo acompanha os participantes presentes na câmera enquanto processa o áudio do microfone. A opção `--audio-device` permite escolher uma entrada específica pelo índice apresentado na seleção de dispositivos.

### Arquivo de vídeo

```powershell
python run_multimodal_tracker.py --video "reuniao.mp4" --speakers 4
```

O sistema lê os quadros e extrai o áudio do arquivo para alimentar o acompanhamento multimodal. Esse fluxo permite analisar gravações com as mesmas representações de participantes usadas nas sessões ao vivo.

### Reunião com captura de tela

```powershell
python run_multimodal_tracker.py --meeting --monitor 1 --speakers 4
```

O modo reunião captura o monitor selecionado. Na escolha de áudio, selecione a entrada de loopback do sistema, como **Stereo Mix** ou **Mixagem Estéreo**, para processar a fala reproduzida pela reunião.

### Geração de vídeo anotado

```powershell
python run_multimodal_tracker.py --video "reuniao.mp4" --speakers 4 --render
```

A opção `--render`, usada com `--video`, gera um arquivo ao lado do vídeo original com o sufixo `_annotated.mp4`, contendo as anotações visuais do acompanhamento.

### Opções do fluxo multimodal

| Opção | Finalidade |
| --- | --- |
| `--video CAMINHO` | Seleciona um arquivo de vídeo |
| `--speakers N` | Define a quantidade de participantes; `0` representa ausência de limite |
| `--audio-device N` | Seleciona o índice do dispositivo de entrada de áudio |
| `--meeting` | Ativa a captura de tela para reuniões |
| `--monitor N` | Seleciona o monitor; o padrão é `1` |
| `--headless` | Define o modo de integração usado pelo lançamento via interface |
| `--render` | Exporta vídeo anotado a partir de um arquivo |
| `--fresh` | Utiliza diretórios separados de embeddings para experimentar novas gravações |

O idioma da transcrição multimodal é configurado na criação de `RealtimeTranscriber`, em `run_multimodal_tracker.py`: `language="en"` atende ao fluxo atual em inglês e `language="pt"` configura o processamento para português.

## Cadastro de participantes

### Perfis vocais a partir de arquivos

```powershell
python enroll_speaker.py "Ana" "amostra_ana.wav"
python enroll_speaker.py "Ana" "amostra_1.wav" "amostra_2.mp3"
python enroll_speaker.py "Ana" "reuniao.mp4" --start 10 --end 30
```

O utilitário usa FFmpeg para carregar e converter o áudio para 16 kHz, extrai o embedding ECAPA-TDNN e salva a representação em `data/embeddings/`. Ao cadastrar novas amostras de uma mesma pessoa, compara sua similaridade com os perfis anteriores e apresenta informações de coerência entre as representações. Reinicie o tracker para carregar os novos embeddings.

### Ferramentas de criação de perfis

Os módulos de cadastro também podem ser executados diretamente:

```powershell
python src/audio_profile_creator.py
python src/video_profile_creator.py
```

Eles disponibilizam fluxos dedicados à criação de perfis de áudio e imagem. Os módulos `src/speaker_verification.py` e `src/person_verification.py` complementam o projeto com ferramentas de verificação por modalidade.

### Transcrição dedicada a áudio

```powershell
python run_realtime.py
```

Esse ponto de entrada oferece um fluxo interativo de transcrição com opções de tamanho do Whisper, duração dos blocos, confiança de verificação e dispositivo de processamento. O arquivo `run_realtime_multiaux.py` reúne outra interface de execução para os recursos de áudio.

## Resultados e validação

As sessões produzem registros em `realtime_sessions/`, incluindo transcrições, logs e métricas. O salvamento também pode incluir áudio da sessão, conforme o fluxo executado. Os dados estruturados são integrados ao SQLite para consulta pela aplicação.

Para revisar uma transcrição e gerar indicadores de validação:

```powershell
python scripts/validate_session.py
```

O script seleciona a transcrição mais recente. Também é possível informar um arquivo:

```powershell
python scripts/validate_session.py "realtime_sessions/transcript_SESSAO.txt"
```

A revisão produz uma transcrição validada e um JSON com indicadores gerais e por locutor, permitindo acompanhar os resultados com referência humana.

Para visualizar métricas e comparar sessões:

```powershell
python scripts/plot_metrics.py
python scripts/plot_metrics.py "realtime_sessions/metrics_*.json"
```

Os gráficos são salvos em `realtime_sessions/plots/`. A análise reúne medidas de identificação vocal, reconhecimento facial e comportamento das sessões, conforme os dados registrados em cada execução.

## Estrutura do repositório

```text
av-tracker/
├── app/
│   ├── app.py                         # Painel Streamlit
│   └── pages/                         # Sessões, métricas, validação e embeddings
├── src/
│   ├── realtime_transcriber.py        # Transcrição e integração de evidências
│   ├── multi_speaker_verifier.py      # Comparação de perfis vocais
│   ├── multi_speaker_verification.py  # Processamento de áudio de reuniões
│   ├── audio_profile_creator.py       # Cadastro vocal
│   ├── video_profile_creator.py       # Cadastro visual
│   ├── speaker_verification.py        # Verificação por voz
│   ├── person_verification.py         # Verificação visual
│   ├── personid_tracker.py            # Rastreamento e identidade facial
│   ├── yolo_detector.py               # Detecção facial
│   ├── active_speaker_detector.py     # Coordenação da detecção de fala ativa
│   ├── light_asd_detector.py          # Modelo audiovisual Light-ASD
│   └── database.py                    # Persistência SQLite
├── scripts/                           # Preparação, migração, validação e gráficos
├── od_model/                          # Pesos de visão computacional
├── pretrained_models/                 # Modelos e configurações de áudio e ASD
├── data/                              # Perfis e dados gerados localmente
├── realtime_sessions/                 # Artefatos das sessões
├── paper/                             # Artigos e materiais de reprodutibilidade
├── enroll_speaker.py                  # Cadastro vocal a partir de arquivos
├── run_multimodal_tracker.py          # Execução integrada de áudio e vídeo
├── run_realtime.py                    # Transcrição interativa de áudio
├── run_realtime_multiaux.py           # Fluxo adicional de áudio
├── av_tracker.db                      # Banco de dados da aplicação
├── AV-Tracker.bat                     # Lançador do painel no Windows
├── requirements.txt                   # Dependências da aplicação
└── requirements-realtime.txt          # Dependências de áudio em tempo real
```

## Tecnologias

| Tecnologia | Contribuição |
| --- | --- |
| PyTorch e Torchaudio | Base de processamento e inferência dos modelos |
| SpeechBrain / ECAPA-TDNN | Representações vocais e identificação de locutores |
| Pyannote Audio | Diarização de falantes |
| Faster-Whisper | Transcrição de fala |
| SepFormer | Separação de fontes de áudio |
| YOLO / Ultralytics | Detecção de rostos |
| EdgeFace e timm | Representações faciais |
| ByteTrack / BoxMOT | Rastreamento ao longo dos quadros |
| Light-ASD | Detecção audiovisual de locutor ativo |
| OpenCV e Pillow | Processamento de vídeo e composição das anotações |
| spaCy | Processamento textual e reconhecimento de nomes |
| Streamlit, Pandas e Plotly | Interface e exploração de resultados |
| SQLite | Organização persistente dos registros |

## Pesquisa e reprodutibilidade

O AV-Tracker reúne implementação e documentação científica no mesmo repositório. A pasta `paper/` contém materiais do CONICT e WVC, além de pacotes com protocolos experimentais, dicionários de dados, configurações, scripts de análise e resultados agregados.

O pacote [`paper/conict-reproducibility/`](paper/conict-reproducibility/README.md) descreve um fluxo de reprodução dos resultados e inclui uma cópia do sistema associada ao experimento. Esses materiais oferecem contexto para estudar a fusão audiovisual, examinar as decisões de identificação e desenvolver novas avaliações.

## Desenvolvimento

A organização em módulos facilita a evolução dos componentes de áudio, visão, associação de identidades, persistência e interface. Para contribuir, mantenha o estilo dos módulos existentes, documente novos parâmetros e atualize os exemplos de uso quando adicionar funcionalidades. A validação com sessões e os scripts de análise oferecem uma base prática para acompanhar a evolução do projeto.
