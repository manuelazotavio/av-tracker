#!/usr/bin/env python3
"""
Servidor Web Flask para Dashboard AV-Tracker com Autenticacao
Compartilhavel com professores via URL
"""

from flask import Flask, render_template_string, request, jsonify, redirect, url_for, session
from flask_httpauth import HTTPBasicAuth
from werkzeug.security import generate_password_hash, check_password_hash
import pandas as pd
from pathlib import Path
from datetime import datetime
import json
import os
import base64
from functools import wraps

app = Flask(__name__)
app.secret_key = 'av-tracker-secret-key-2025'
auth = HTTPBasicAuth()

# Usuarios e senhas (customize aqui)
USERS = {
    'professor1': generate_password_hash('senha123'),
    'professor2': generate_password_hash('senha456'),
    'admin': generate_password_hash('admin123')
}

@auth.verify_password
def verify_password(username, password):
    if username in USERS and check_password_hash(USERS.get(username), password):
        return username
    return None

def load_report_data():
    """Carrega os dados do relatorio"""
    try:
        results_df = pd.read_csv('results.csv')
        
        try:
            detailed_df = pd.read_csv('analysis_results/detailed_results_with_females.csv')
        except:
            detailed_df = pd.read_csv('analysis_results/detailed_results.csv')
        
        # Carregar metricas
        metrics_text = ""
        try:
            with open('analysis_results/metrics_summary.txt', 'r', encoding='utf-8') as f:
                metrics_text = f.read()
        except:
            pass
        
        # Calcular estatisticas
        total_files = len(results_df)
        total_chunks = results_df['chunks_processed'].sum()
        total_identified = results_df['identified_count'].sum()
        total_unknown = results_df['unknown_count'].sum()
        taxa_geral = (total_identified / total_chunks * 100) if total_chunks > 0 else 0
        
        avg_similarity = results_df['avg_similarity'].mean()
        max_similarity = results_df['top_similarity'].max()
        min_similarity = results_df['bottom_similarity'].min()
        
        return {
            'total_files': total_files,
            'total_chunks': int(total_chunks),
            'total_identified': int(total_identified),
            'total_unknown': int(total_unknown),
            'taxa_geral': taxa_geral,
            'avg_similarity': avg_similarity,
            'max_similarity': max_similarity,
            'min_similarity': min_similarity,
            'metrics_text': metrics_text,
            'last_updated': datetime.now().strftime('%d/%m/%Y %H:%M:%S')
        }
    except Exception as e:
        return {
            'error': str(e),
            'last_updated': datetime.now().strftime('%d/%m/%Y %H:%M:%S')
        }

def encode_image_to_base64(image_path):
    """Converte imagem para base64"""
    try:
        with open(image_path, 'rb') as img_file:
            return base64.b64encode(img_file.read()).decode()
    except:
        return None

@app.route('/')
def index():
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        if username in USERS and check_password_hash(USERS.get(username), password):
            session['user'] = username
            return redirect(url_for('dashboard'))
        else:
            return render_template_string(LOGIN_TEMPLATE, error='Usuario ou senha incorretos')
    
    return render_template_string(LOGIN_TEMPLATE)

@app.route('/dashboard')
def dashboard():
    if 'user' not in session:
        return redirect(url_for('login'))
    
    data = load_report_data()
    
    # Carregar imagens
    plots_html = ""
    plots_dir = Path('analysis_results/plots')
    if plots_dir.exists():
        for img_file in sorted(plots_dir.glob('*.png')):
            b64 = encode_image_to_base64(str(img_file))
            if b64:
                title = img_file.stem.replace('_', ' ').title()
                plots_html += f"""
                <div class="plot-container">
                    <h3>{title}</h3>
                    <img src="data:image/png;base64,{b64}" alt="{title}" onclick="openModal(this)" style="cursor: pointer;">
                </div>
                """
    
    return render_template_string(
        DASHBOARD_TEMPLATE,
        username=session['user'],
        data=data,
        plots_html=plots_html
    )

@app.route('/api/data')
def api_data():
    """API para obter dados (para auto-refresh)"""
    if 'user' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    return jsonify(load_report_data())

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/info')
def info():
    """Pagina com informacoes de acesso"""
    return render_template_string(INFO_TEMPLATE)

LOGIN_TEMPLATE = '''
<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AV-Tracker Dashboard - Login</title>
    <link href="https://fonts.googleapis.com/css2?family=Google+Sans:wght@400;500;700&display=swap" rel="stylesheet">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Google Sans', Arial, sans-serif;
            background: linear-gradient(135deg, #1a1a1a 0%, #0a0a0a 100%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        
        .login-container {
            background: #222;
            border: 1px solid #333;
            border-radius: 10px;
            padding: 40px;
            width: 100%;
            max-width: 400px;
            box-shadow: 0 10px 40px rgba(0, 0, 0, 0.5);
        }
        
        .login-header {
            text-align: center;
            margin-bottom: 30px;
        }
        
        .login-header h1 {
            color: #f0f0f0;
            font-size: 1.8em;
            margin-bottom: 5px;
        }
        
        .login-header p {
            color: #999;
            font-size: 0.9em;
        }
        
        .form-group {
            margin-bottom: 20px;
        }
        
        label {
            display: block;
            color: #f0f0f0;
            margin-bottom: 8px;
            font-weight: 500;
        }
        
        input[type="text"],
        input[type="password"] {
            width: 100%;
            padding: 12px;
            background: #1a1a1a;
            border: 1px solid #333;
            border-radius: 5px;
            color: #f0f0f0;
            font-family: 'Google Sans', Arial, sans-serif;
            font-size: 1em;
        }
        
        input[type="text"]:focus,
        input[type="password"]:focus {
            outline: none;
            border-color: #555;
            background: #252525;
        }
        
        button {
            width: 100%;
            padding: 12px;
            background: #000;
            color: #f0f0f0;
            border: 1px solid #333;
            border-radius: 5px;
            font-size: 1em;
            font-weight: 600;
            cursor: pointer;
            font-family: 'Google Sans', Arial, sans-serif;
            transition: all 0.3s ease;
        }
        
        button:hover {
            background: #1a1a1a;
            border-color: #555;
        }
        
        .error {
            background: #3a2a2a;
            color: #ff6b6b;
            padding: 12px;
            border-radius: 5px;
            margin-bottom: 20px;
            border-left: 3px solid #ff6b6b;
        }
        
        .credentials {
            background: #1a1a1a;
            border: 1px solid #333;
            border-radius: 5px;
            padding: 15px;
            margin-top: 20px;
            color: #999;
            font-size: 0.85em;
        }
        
        .credentials strong {
            color: #f0f0f0;
            display: block;
            margin-bottom: 8px;
        }
    </style>
</head>
<body>
    <div class="login-container">
        <div class="login-header">
            <h1>AV-Tracker</h1>
            <p>Dashboard de Re-identificacao de Voz</p>
        </div>
        
        {% if error %}
        <div class="error">{{ error }}</div>
        {% endif %}
        
        <form method="POST">
            <div class="form-group">
                <label for="username">Usuario</label>
                <input type="text" id="username" name="username" required autofocus>
            </div>
            
            <div class="form-group">
                <label for="password">Senha</label>
                <input type="password" id="password" name="password" required>
            </div>
            
            <button type="submit">Entrar</button>
        </form>
        
        <div class="credentials">
            <strong>Credenciais de Teste:</strong>
            professor1 / senha123<br>
            professor2 / senha456<br>
            admin / admin123
        </div>
    </div>
</body>
</html>
'''

DASHBOARD_TEMPLATE = '''
<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AV-Tracker Dashboard</title>
    <link href="https://fonts.googleapis.com/css2?family=Google+Sans:wght@400;500;700&display=swap" rel="stylesheet">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Google Sans', Arial, sans-serif;
            background-color: #1a1a1a;
            color: #f0f0f0;
            padding: 20px;
        }
        
        header {
            background-color: #000;
            border-bottom: 1px solid #333;
            padding: 15px 20px;
            margin: -20px -20px 20px -20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        header h1 {
            font-size: 1.6em;
        }
        
        .header-info {
            display: flex;
            align-items: center;
            gap: 20px;
        }
        
        .user-info {
            color: #999;
            font-size: 0.9em;
        }
        
        .btn-logout {
            background: #333;
            color: #f0f0f0;
            border: none;
            padding: 8px 16px;
            border-radius: 5px;
            cursor: pointer;
            text-decoration: none;
            font-size: 0.9em;
        }
        
        .btn-logout:hover {
            background: #444;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        
        .refresh-info {
            background: #222;
            border: 1px solid #333;
            padding: 12px;
            border-radius: 5px;
            margin-bottom: 20px;
            font-size: 0.85em;
            color: #999;
        }
        
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }
        
        .metric-card {
            background: #222;
            border: 1px solid #333;
            padding: 15px;
            border-radius: 5px;
            text-align: center;
        }
        
        .metric-value {
            font-size: 2em;
            font-weight: 700;
            color: #f0f0f0;
            margin: 8px 0;
        }
        
        .metric-label {
            font-size: 0.75em;
            color: #999;
            text-transform: uppercase;
        }
        
        h2 {
            color: #f0f0f0;
            border-bottom: 1px solid #333;
            padding-bottom: 10px;
            margin: 25px 0 15px 0;
            font-size: 1.3em;
        }
        
        .plots-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }
        
        .plot-container {
            background: #222;
            border: 1px solid #333;
            padding: 12px;
            border-radius: 5px;
        }
        
        .plot-container h3 {
            color: #f0f0f0;
            margin-bottom: 8px;
            font-size: 0.95em;
        }
        
        .plot-container img {
            max-width: 100%;
            height: auto;
            max-height: 250px;
            cursor: pointer;
            border-radius: 3px;
        }
        
        .plot-container img:hover {
            opacity: 0.9;
        }
        
        .modal {
            display: none;
            position: fixed;
            z-index: 1000;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0, 0, 0, 0.95);
            align-items: center;
            justify-content: center;
        }
        
        .modal.show {
            display: flex;
        }
        
        .modal-content {
            position: relative;
            max-width: 90vw;
            max-height: 90vh;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        
        .modal-content img {
            max-width: 100%;
            max-height: 90vh;
            object-fit: contain;
        }
        
        .close {
            position: absolute;
            top: 15px;
            right: 25px;
            font-size: 32px;
            font-weight: bold;
            color: #fff;
            cursor: pointer;
            background-color: rgba(0, 0, 0, 0.7);
            width: 40px;
            height: 40px;
            display: flex;
            align-items: center;
            justify-content: center;
            border-radius: 50%;
            z-index: 1001;
        }
        
        .close:hover {
            background-color: rgba(0, 0, 0, 0.9);
        }
        
        .metrics-text {
            background: #222;
            border: 1px solid #333;
            padding: 12px;
            border-radius: 5px;
            font-family: 'Courier New', monospace;
            white-space: pre-wrap;
            word-wrap: break-word;
            font-size: 0.75em;
            color: #bbb;
            line-height: 1.3;
            max-height: 400px;
            overflow-y: auto;
        }
    </style>
</head>
<body>
    <div id="imageModal" class="modal" onclick="closeModal(event)">
        <div class="modal-content" onclick="event.stopPropagation()">
            <span class="close" onclick="closeModal()">&times;</span>
            <img id="modalImage" src="" alt="">
        </div>
    </div>
    
    <header>
        <h1>AV-Tracker Dashboard</h1>
        <div class="header-info">
            <div class="user-info">
                Conectado como: <strong>{{ username }}</strong><br>
                <span id="last-update">Ultima atualizacao: {{ data.last_updated }}</span>
            </div>
            <a href="{{ url_for('logout') }}" class="btn-logout">Sair</a>
        </div>
    </header>
    
    <div class="container">
        <div class="refresh-info">
            Este relatorio se atualiza automaticamente a cada 10 segundos. Ultima sincronizacao: <strong id="sync-time">{{ data.last_updated }}</strong>
        </div>
        
        <h2>Metricas Gerais</h2>
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-label">Total de Arquivos</div>
                <div class="metric-value" id="total-files">{{ data.total_files }}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Chunks Processados</div>
                <div class="metric-value" id="total-chunks">{{ data.total_chunks }}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Identificados</div>
                <div class="metric-value" id="total-identified">{{ data.total_identified }}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Taxa Geral</div>
                <div class="metric-value" id="taxa-geral">{{ "%.1f"|format(data.taxa_geral) }}%</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Desconhecidos</div>
                <div class="metric-value" id="total-unknown">{{ data.total_unknown }}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Avg Similaridade</div>
                <div class="metric-value" id="avg-similarity">{{ "%.4f"|format(data.avg_similarity) }}</div>
            </div>
        </div>
        
        <h2>Graficos</h2>
        <div class="plots-grid">
            {{ plots_html | safe }}
        </div>
        
        <h2>Metricas Detalhadas</h2>
        <div class="metrics-text">{{ data.metrics_text }}</div>
    </div>
    
    <script>
        function openModal(img) {
            const modal = document.getElementById('imageModal');
            const modalImg = document.getElementById('modalImage');
            modalImg.src = img.src;
            modal.classList.add('show');
            document.body.style.overflow = 'hidden';
        }
        
        function closeModal(event) {
            if (event && event.target.id !== 'imageModal') return;
            const modal = document.getElementById('imageModal');
            modal.classList.remove('show');
            document.body.style.overflow = 'auto';
        }
        
        document.addEventListener('keydown', function(event) {
            if (event.key === 'Escape') {
                closeModal();
            }
        });
        
        // Auto-refresh a cada 10 segundos
        setInterval(function() {
            fetch('/api/data')
                .then(response => response.json())
                .then(data => {
                    if (!data.error) {
                        document.getElementById('total-files').textContent = data.total_files;
                        document.getElementById('total-chunks').textContent = data.total_chunks;
                        document.getElementById('total-identified').textContent = data.total_identified;
                        document.getElementById('total-unknown').textContent = data.total_unknown;
                        document.getElementById('taxa-geral').textContent = data.taxa_geral.toFixed(1) + '%';
                        document.getElementById('avg-similarity').textContent = data.avg_similarity.toFixed(4);
                        document.getElementById('sync-time').textContent = data.last_updated;
                        document.getElementById('last-update').textContent = 'Ultima atualizacao: ' + data.last_updated;
                    }
                });
        }, 10000);
    </script>
</body>
</html>
'''

INFO_TEMPLATE = '''
<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Informacoes - AV-Tracker</title>
    <link href="https://fonts.googleapis.com/css2?family=Google+Sans:wght@400;500;700&display=swap" rel="stylesheet">
    <style>
        body {
            font-family: 'Google Sans', Arial, sans-serif;
            background-color: #1a1a1a;
            color: #f0f0f0;
            padding: 20px;
            line-height: 1.6;
        }
        .container {
            max-width: 800px;
            margin: 0 auto;
            background: #222;
            border: 1px solid #333;
            border-radius: 5px;
            padding: 30px;
        }
        h1 { color: #f0f0f0; }
        a { color: #66b3ff; text-decoration: none; }
        a:hover { text-decoration: underline; }
        code { background: #1a1a1a; padding: 2px 6px; border-radius: 3px; color: #bbb; }
    </style>
</head>
<body>
    <div class="container">
        <h1>AV-Tracker Web Dashboard</h1>
        <p>Servidor rodando com sucesso!</p>
        <p><a href="/">Voltar para Login</a></p>
        <h2>Como Usar</h2>
        <ol>
            <li>Acesse: <code>http://seu-endereco:5000/login</code></li>
            <li>Use as credenciais fornecidas</li>
            <li>O dashboard se atualiza automaticamente</li>
        </ol>
    </div>
</body>
</html>
'''

if __name__ == '__main__':
    print("=" * 60)
    print("AV-Tracker Web Dashboard")
    print("=" * 60)
    print("Acesse: http://localhost:5000")
    print("Login: professor1 / senha123")
    print("=" * 60)
    app.run(debug=True, host='0.0.0.0', port=5000)
