# 🌐 Deploy do Web Dashboard - Guia Completo

Seu dashboard web com autenticacao esta pronto! Aqui estao as opcoes para colocar online:

## 1. Testar Localmente

```bash
# Instalar dependencias
pip install flask flask-httpauth pandas

# Executar servidor
python web_dashboard.py
```

Acesse: **http://localhost:5000**

Credenciais teste:
- usuario: `professor1` / senha: `senha123`
- usuario: `professor2` / senha: `senha456`
- usuario: `admin` / senha: `admin123`

---

## 2. Deploy Online Gratuito

### Opcao A: Render (Recomendado - Simples)

1. **Criar conta**: https://render.com
2. **Conectar GitHub**: Push seu projeto
3. **Criar novo servico**:
   - Type: Web Service
   - Runtime: Python 3
   - Build command: `pip install flask flask-httpauth pandas`
   - Start command: `python web_dashboard.py`
4. **Deploy** - Sera gerada uma URL publica

**Vantagens:**
- ✅ Gratuito
- ✅ Muito simples
- ✅ Deployment automatico
- ✅ HTTPS incluido

---

### Opcao B: Railway (Super Simples)

1. **Criar conta**: https://railway.app
2. **Conectar GitHub**
3. **Deploy** - E pronto!

**Vantagens:**
- ✅ Interface muito intuitiva
- ✅ Deploy com 1 clique
- ✅ Ambiente automatico

---

### Opcao C: PythonAnywhere (Facil)

1. **Criar conta**: https://www.pythonanywhere.com
2. **Upload seu projeto**
3. **Configurar Web App**:
   - Python 3.x
   - Framework: Flask
   - Path: `/home/seu_usuario/web_dashboard/web_dashboard.py`

---

### Opcao D: Heroku (Antigo - Precisa Pagar)

Antes era gratuito, agora requer cartao de credito.

---

## 3. Configurar Acessos para Professores

### Adicionar Novos Usuarios

Edite `web_dashboard.py` e customize:

```python
USERS = {
    'professor1': generate_password_hash('senha123'),
    'professor2': generate_password_hash('senha456'),
    'seu_usuario': generate_password_hash('sua_senha')
}
```

Execute:
```python
from werkzeug.security import generate_password_hash
print(generate_password_hash('sua_senha'))
```

---

## 4. Compartilhar com Professores

Apos fazer deploy, compartilhe:

**Email Template:**
```
Assunto: Link do Dashboard AV-Tracker

Oi Professores,

Podem acessar o relatorio em tempo real aqui:

URL: https://seu-site.com (substitua)

Usuario: seu_usuario_aqui
Senha: sua_senha_aqui

O relatorio se atualiza automaticamente a cada 10 segundos.

Qualquer problema, me avise!
```

---

## 5. Monitoramento Contínuo

Para que o relatorio se atualize, mantenha o monitor rodando:

```bash
# Terminal separado
python monitor_report.py --fast
```

Isto garante que:
- Arquivos sao monitorados
- HTML e regenerado automaticamente
- Dashboard web mostra dados sempre atualizados

---

## 6. Workflow Recomendado

```
1. Na sua maquina:
   Terminal 1: python monitor_report.py --fast
   Terminal 2: python web_dashboard.py

2. Deploy online (ex: Render)

3. Compartilhe URL com professores

4. Conforme voce processa audios:
   - Monitor detecta mudancas
   - HTML e regenerado
   - Dashboard web atualiza (a cada 10s)
   - Professores veem em tempo real
```

---

## 7. Customizacoes

### Mudar Intervalo de Auto-Refresh

No arquivo `web_dashboard.py`, encontre:

```javascript
setInterval(function() {
    // ... codigo de refresh
}, 10000);  // 10000 = 10 segundos
```

Mude para:
- `5000` = 5 segundos (mais responsivo, mais CPU)
- `30000` = 30 segundos (menos CPU)

### Adicionar Logo/Titulo

Edite o HTML dos templates para adicionar seu logo.

### Mudar Cores

Procure por `#1a1a1a` no CSS e customize as cores.

---

## 8. Variaveis de Ambiente

Para maior seguranca, use variaveis de ambiente:

```python
import os

SECRET_KEY = os.getenv('SECRET_KEY', 'dev-key')
DEBUG = os.getenv('DEBUG', 'False') == 'True'
```

Nas plataformas de deploy (ex: Render):
- Dashboard > Environment
- Adicione variaveis

---

## 9. HTTPS/Certificado

Todos os servicos acima incluem HTTPS automaticamente:
- ✅ Render: HTTPS incluido
- ✅ Railway: HTTPS incluido
- ✅ PythonAnywhere: HTTPS incluido

---

## 10. Troubleshooting

| Problema | Solucao |
|----------|---------|
| Porta 5000 em uso | Altere a porta: `app.run(port=5001)` |
| Conexao recusada | Verifique firewall |
| Dashboard nao atualiza | Verifique se monitor esta rodando |
| Erro de autenticacao | Regenere as senhas com `generate_password_hash()` |
| Imagens nao carregam | Verifique se `analysis_results/plots/` tem arquivos |

---

## 11. Performance e Escalabilidade

Para muitos usuarios:

```python
# Adicione cache
from flask_caching import Cache
cache = Cache(app, config={'CACHE_TYPE': 'simple'})

@app.route('/api/data')
@cache.cached(timeout=5)
def api_data():
    ...
```

---

## 12. Opcion Final: Compartilhamento Simples (Nao Recomendado)

Se nao quiser fazer deploy online, pode usar:
- Google Drive (arquivo HTML)
- OneDrive
- Dropbox
- Email (arquivo HTML)

Desvantagens:
- ❌ Nao se atualiza automaticamente
- ❌ Historico fica bagunçado
- ❌ Sem autenticacao

---

## Resumo

| Metodo | Complexidade | Custo | Autenticacao | Auto-Update |
|--------|--------------|-------|--------------|-------------|
| Local + Arquivo HTML | Baixa | Gratis | Nao | Manual |
| Web Dashboard Local | Media | Gratis | Sim | Sim |
| Render/Railway | Media | Gratis | Sim | Sim |
| PythonAnywhere | Media | Gratis* | Sim | Sim |

*PythonAnywhere oferece plano gratuito basico.

---

**Recomendacao**: Use Render ou Railway + monitor local. E o jeito mais simples e eficiente!
