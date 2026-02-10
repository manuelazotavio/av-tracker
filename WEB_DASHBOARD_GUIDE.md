# 🌐 Web Dashboard AV-Tracker - Guia Completo

## ✅ O QUE FOI CRIADO

Um **servidor web profissional** com:
- ✅ **Autenticacao** - Login protegido para professores
- ✅ **Auto-refresh** - Dashboard atualiza a cada 10 segundos
- ✅ **Design dark** - Interface profissional
- ✅ **Acesso remoto** - Compartilhavel via URL
- ✅ **Deploy facil** - Opcoes de hospedagem gratuitas

---

## 🚀 USAR LOCALMENTE (Agora)

### Passo 1: Instalar Dependencias
```bash
pip install -r requirements.txt
```

### Passo 2: Iniciar Servidor
```bash
python web_dashboard.py
```

### Passo 3: Acessar no Navegador
```
URL: http://localhost:5000
Usuario: professor1
Senha: senha123
```

### Passo 4: Ver Atualizacoes em Tempo Real
- Inicie o monitor em outro terminal:
```bash
python monitor_report.py --fast
```

- Processe seus audios normalmente
- Dashboard atualiza automaticamente!

---

## 🌍 COLOCAR ONLINE (Para Compartilhar)

### Opcao 1: Render (Recomendado)

**1. Criar conta**: https://render.com

**2. Conectar GitHub**:
- Push seu projeto para GitHub
- Em Render: New → Web Service
- Conectar repositorio

**3. Configurar**:
- Name: `av-tracker-dashboard`
- Runtime: `Python 3`
- Build command: `pip install -r requirements.txt`
- Start command: `python web_dashboard.py`
- Plan: **Free**

**4. Deploy**
- Clique "Create Web Service"
- Sera gerada uma URL como: `https://av-tracker-dashboard.onrender.com`

**5. Compartilhar com Professores**:
```
Acesso: https://av-tracker-dashboard.onrender.com
Usuario: professor1
Senha: senha123
```

**Tempo**: ~5 minutos
**Custo**: Gratis
**Nota**: Pode "dormir" apos 15 min sem atividade (reacorda ao acessar)

---

### Opcao 2: Railway

**1. Criar conta**: https://railway.app

**2. Deploy**:
- New Project → GitHub Repo
- Selecione seu repositorio
- Railway detecta automaticamente Flask
- Deploy automatico

**3. Configurar variaveis** (se necessario):
```
FLASK_ENV=production
```

**4. Acessar**: URL fornecida automaticamente

**Tempo**: ~2 minutos
**Custo**: Gratis ($5/mes credit)

---

### Opcao 3: PythonAnywhere

**1. Criar conta**: https://www.pythonanywhere.com

**2. Upload seu projeto**

**3. Criar Web App**:
- New Web App
- Python 3.x
- Framework: Flask
- Path: `/home/seu_usuario/av-tracker/web_dashboard.py`

**4. Acessar**: `seu_usuario.pythonanywhere.com`

---

## 🔐 CUSTOMIZAR USUARIOS E SENHAS

### Adicionar Novos Usuarios

Edite `web_dashboard.py`:

```python
USERS = {
    'professor1': generate_password_hash('senha123'),
    'professor2': generate_password_hash('senha456'),
    'seu_nome': generate_password_hash('sua_senha')
}
```

Para gerar hash de senha:
```python
from werkzeug.security import generate_password_hash
print(generate_password_hash('sua_senha'))
```

Copie o hash gerado e cole em USERS.

---

## 📊 WORKFLOW RECOMENDADO

### Desenvolvimento (Sua Maquina):

```
Terminal 1:
$ python monitor_report.py --fast
> Monitorando mudancas...

Terminal 2:
$ python web_dashboard.py
> Servidor em http://localhost:5000

Terminal 3:
$ python seus_scripts.py
> Processando audios...
```

### Producao (Online):

```
1. Deploy em Render/Railway (uma unica vez)
2. Mantenha monitor rodando localmente:
   python monitor_report.py --fast
3. Professores acessam a URL publica
4. Dashboard se atualiza em tempo real
```

---

## 🎯 FLUXO COMPLETO

```
1. Professor acessa: https://seu-site.onrender.com
   ↓
2. Faz login com usuario/senha
   ↓
3. Ve dashboard com metricas atualizadas
   ↓
4. Voce processa audios na sua maquina
   ↓
5. Monitor detecta mudancas
   ↓
6. Monitor regenera HTML
   ↓
7. Dashboard web atualiza (a cada 10s)
   ↓
8. Professor ve dados em tempo real!
```

---

## ⚙️ CONFIGURACOES AVANCADAS

### Mudar Intervalo de Auto-Refresh

No `web_dashboard.py`, procure:
```javascript
setInterval(function() {
    // ...
}, 10000);  // <-- AQUI
```

- `5000` = 5 segundos (mais responsivo)
- `10000` = 10 segundos (balanceado)
- `30000` = 30 segundos (menos CPU)

### Adicionar Logo Customizado

No header do template HTML, adicione:
```html
<img src="seu-logo.png" alt="Logo" width="50">
```

### Mudar Cores

Procure no CSS:
```css
background-color: #1a1a1a;  /* Fundo */
color: #f0f0f0;              /* Texto */
border: 1px solid #333;      /* Bordas */
```

---

## 🔄 ATUALIZAR USERS ONLINE

Se fez deploy e quer adicionar mais usuarios:

**Local**:
1. Edite `web_dashboard.py`
2. Adicione usuarios em USERS
3. Faça push para GitHub
4. Deploy automaticamente regenera

---

## 📱 ACESSAR DE QUALQUER LUGAR

O site funciona em:
- ✅ Computador (desktop/notebook)
- ✅ Tablet
- ✅ Celular
- ✅ Qualquer navegador (Chrome, Firefox, Safari, Edge)

Basta acessar a URL publica!

---

## 🚨 IMPORTANTE: MANTER MONITOR RODANDO

Para que o relatorio se atualize, voce precisa manter rodando:

```bash
python monitor_report.py --fast
```

Isto garante que:
- Mudancas nos arquivos sao detectadas
- HTML e regenerado
- Dashboard web ve dados novos

---

## 🐛 TROUBLESHOOTING

| Problema | Solucao |
|----------|---------|
| "Porta 5000 em uso" | Mude a porta: `app.run(port=5001)` |
| Nao consegue conectar | Verifique firewall/antivirus |
| Dashboard nao atualiza | Confirme que monitor esta rodando |
| Login nao funciona | Verifique usuario/senha em USERS |
| Deploy falha no Render | Confirme que requirements.txt existe |
| Imagens nao carregam | Verifique `analysis_results/plots/` |

---

## 📞 ARQUIVOS IMPORTANTES

```
web_dashboard.py          - Servidor Flask (principal)
requirements.txt          - Dependencias
monitor_report.py         - Monitor em tempo real
generate_html_report.py   - Gerador de HTML
Procfile                  - Config para Heroku/Render
render.yaml               - Config especifica Render
WEB_DEPLOY_GUIDE.md       - Guia detalhado
```

---

## 💡 DICAS EXTRAS

### Teste Localmente Antes de Deploy
```bash
python web_dashboard.py
# Acesse http://localhost:5000
# Teste login, navegacao, auto-refresh
```

### Use HTTPS (Automatico no Deploy)
- Render: HTTPS automatico
- Railway: HTTPS automatico
- PythonAnywhere: HTTPS incluido

### Compartilhe a URL
```
Envie um email para seus professores:

Assunto: Link Dashboard AV-Tracker

Acesse aqui: https://av-tracker-dashboard.onrender.com

Usuario: seu_nome
Senha: sua_senha

Valido durante o semestre letivo.
```

---

## 🎉 RESUMO

Voce agora tem:

1. **Dashboard HTML Estatico** - Para compartilhar arquivos
2. **Dashboard Streamlit** - Para explorar interativamente
3. **Dashboard Web** - Para seu time acessar online
4. **Monitor em Tempo Real** - Para atualizar automaticamente

**Escolha conforme necessario!**

---

**Próximo passo**: 
- [ ] Teste localmente (`python web_dashboard.py`)
- [ ] Customize usuarios/senhas
- [ ] Deploy no Render/Railway
- [ ] Compartilhe URL com professores
- [ ] Inicie monitor (`python monitor_report.py --fast`)

**Voce esta pronto!** 🚀
