#!/usr/bin/env python3
"""
Gera um relatório HTML estático a partir dos dados do AV-Tracker
Útil para compartilhar sem precisar do Streamlit instalado
"""

import pandas as pd
from pathlib import Path
import json
from datetime import datetime
import base64

def encode_image_to_base64(image_path):
    """Converte uma imagem para base64"""
    try:
        with open(image_path, 'rb') as img_file:
            return base64.b64encode(img_file.read()).decode()
    except:
        return None

def generate_html_report():
    """Gera um relatório HTML completo"""
    
    # Carregar dados
    results_df = pd.read_csv('results.csv')
    
    try:
        detailed_df = pd.read_csv('analysis_results/detailed_results_with_females.csv')
    except:
        detailed_df = pd.read_csv('analysis_results/detailed_results.csv')
    
    metrics_text = ""
    try:
        with open('analysis_results/metrics_summary.txt', 'r', encoding='utf-8') as f:
            metrics_text = f.read()
    except:
        pass
    
    # Calcular estatísticas
    total_files = len(results_df)
    total_chunks = results_df['chunks_processed'].sum()
    total_identified = results_df['identified_count'].sum()
    total_unknown = results_df['unknown_count'].sum()
    taxa_geral = (total_identified / total_chunks * 100) if total_chunks > 0 else 0
    
    avg_similarity = results_df['avg_similarity'].mean()
    max_similarity = results_df['top_similarity'].max()
    min_similarity = results_df['bottom_similarity'].min()
    
    # Encapsular imagens em base64
    plots_dir = Path('analysis_results/plots')
    images_html = ""
    if plots_dir.exists():
        for img_file in sorted(plots_dir.glob('*.png')):
            b64 = encode_image_to_base64(img_file)
            if b64:
                title = img_file.stem.replace('_', ' ').title()
                images_html += f"""
                <div class="plot-container">
                    <h3>{title}</h3>
                    <img src="data:image/png;base64,{b64}" alt="{title}" onclick="openModal(this)" style="cursor: pointer;">
                </div>
                """
    
    # Gerar tabela HTML
    table_html = results_df.to_html(
        classes='table table-striped',
        index=False,
        float_format=lambda x: f'{x:.4f}' if x < 1 else f'{x:.0f}'
    )
    
    # HTML do relatório
    html_content = f"""
    <!DOCTYPE html>
    <html lang="pt-BR">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>AV-Tracker Dashboard Report</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
        <link href="https://fonts.googleapis.com/css2?family=Google+Sans:wght@400;500;700&display=swap" rel="stylesheet">
        <style>
            body {{
                font-family: 'Google Sans', 'Arial', sans-serif;
                background-color: #1a1a1a;
                color: #f0f0f0;
            }}
            
            header {{
                background-color: #000;
                color: #fff;
                padding: 12px 0;
                margin-bottom: 15px;
                border-bottom: 1px solid #333;
            }}
            
            header h1 {{
                font-size: 1.6em;
                font-weight: 700;
                margin: 0;
            }}
            
            header p {{
                font-size: 0.85em;
                margin: 3px 0 0 0;
                color: #bbb;
            }}
            
            .metric-card {{
                background: #222;
                border: 1px solid #333;
                padding: 10px;
                margin-bottom: 8px;
                text-align: center;
            }}
            
            .metric-card:hover {{
                background-color: #2a2a2a;
            }}
            
            .metric-value {{
                font-size: 1.6em;
                font-weight: 700;
                color: #f0f0f0;
                margin: 6px 0;
            }}
            
            .metric-label {{
                font-size: 0.7em;
                color: #999;
                text-transform: uppercase;
                letter-spacing: 0.3px;
            }}
            
            section {{
                margin-bottom: 20px;
            }}
            
            h2 {{
                color: #f0f0f0;
                border-bottom: 1px solid #333;
                padding-bottom: 6px;
                margin-bottom: 12px;
                font-size: 1.3em;
                font-weight: 700;
            }}
            
            .plot-container {{
                background: #222;
                border: 1px solid #333;
                padding: 8px;
                margin-bottom: 10px;
            }}
            
            .plot-container img {{
                max-width: 100%;
                height: auto;
                max-height: 250px;
            }}
            
            .plot-container h3 {{
                color: #f0f0f0;
                margin-bottom: 6px;
                font-size: 0.95em;
            }}
            
            .table-container {{
                background: #222;
                border: 1px solid #333;
                padding: 10px;
                overflow-x: auto;
            }}
            
            .table {{
                margin-bottom: 0;
                font-size: 0.8em;
            }}
            
            .table thead {{
                background-color: #000;
                color: #f0f0f0;
            }}
            
            .table thead th {{
                border: 1px solid #333;
                padding: 6px;
                font-weight: 600;
            }}
            
            .table tbody tr {{
                border-bottom: 1px solid #333;
            }}
            
            .table tbody tr:hover {{
                background-color: #2a2a2a;
            }}
            
            .table tbody td {{
                padding: 6px;
                border: 1px solid #2a2a2a;
                vertical-align: middle;
                color: #f0f0f0;
            }}
            
            .metrics-text {{
                background: #222;
                border: 1px solid #333;
                padding: 10px;
                font-family: 'Courier New', monospace;
                white-space: pre-wrap;
                word-wrap: break-word;
                font-size: 0.75em;
                color: #bbb;
                line-height: 1.3;
            }}
            
            footer {{
                background-color: #000;
                color: #999;
                text-align: center;
                padding: 10px;
                margin-top: 20px;
                border-top: 1px solid #333;
                font-size: 0.8em;
            }}
            
            .section-subtitle {{
                color: #999;
                font-size: 0.8em;
                margin-bottom: 8px;
            }}
            
            .container {{
                max-width: 1200px;
            }}
            
            @media print {{
                body {{
                    background-color: white;
                    color: black;
                }}
                
                .metric-card, .plot-container, .table-container, .metrics-text {{
                    page-break-inside: avoid;
                    border: 1px solid #000;
                    background-color: white;
                    color: black;
                }}
            }}
            
            .modal {{
                display: none;
                position: fixed;
                z-index: 1000;
                left: 0;
                top: 0;
                width: 100%;
                height: 100%;
                background-color: rgba(0, 0, 0, 0.9);
            }}
            
            .modal.show {{
                display: flex;
                align-items: center;
                justify-content: center;
            }}
            
            .modal-content {{
                position: relative;
                max-width: 90vw;
                max-height: 90vh;
                display: flex;
                align-items: center;
                justify-content: center;
            }}
            
            .modal-content img {{
                max-width: 100%;
                max-height: 90vh;
                object-fit: contain;
            }}
            
            .close {{
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
            }}
            
            .close:hover {{
                background-color: rgba(0, 0, 0, 0.9);
                color: #ccc;
            }}
            
            .plot-container img {{
                cursor: pointer;
            }}
            
            .plot-container img:hover {{
                opacity: 0.9;
            }}
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
            <div class="container">
                <h1>AV-Tracker Dashboard Report</h1>
                <p>Sistema de Re-identificação de Voz - Relatório Completo</p>
                <small style="color: #666;">Gerado em {datetime.now().strftime('%d/%m/%Y às %H:%M:%S')}</small>
            </div>
        </header>
        
        <main class="container">
            <!-- SEÇÃO 1: MÉTRICAS GERAIS -->
            <section>
                <h2>Métricas Gerais</h2>
                <p class="section-subtitle">Resumo de desempenho do sistema</p>
                
                <div class="row">
                    <div class="col-md-6 col-lg-3">
                        <div class="metric-card">
                            <div class="metric-label">Total de Arquivos</div>
                            <div class="metric-value">{total_files}</div>
                        </div>
                    </div>
                    <div class="col-md-6 col-lg-3">
                        <div class="metric-card">
                            <div class="metric-label">Chunks Processados</div>
                            <div class="metric-value">{int(total_chunks)}</div>
                        </div>
                    </div>
                    <div class="col-md-6 col-lg-3">
                        <div class="metric-card">
                            <div class="metric-label">Identificados</div>
                            <div class="metric-value">{int(total_identified)}</div>
                        </div>
                    </div>
                    <div class="col-md-6 col-lg-3">
                        <div class="metric-card">
                            <div class="metric-label">Taxa Geral</div>
                            <div class="metric-value">{taxa_geral:.1f}%</div>
                        </div>
                    </div>
                </div>
                
                <div class="row mt-4">
                    <div class="col-md-6 col-lg-3">
                        <div class="metric-card">
                            <div class="metric-label">Desconhecidos</div>
                            <div class="metric-value">{int(total_unknown)}</div>
                        </div>
                    </div>
                    <div class="col-md-6 col-lg-3">
                        <div class="metric-card">
                            <div class="metric-label">Avg Similaridade</div>
                            <div class="metric-value">{avg_similarity:.4f}</div>
                        </div>
                    </div>
                    <div class="col-md-6 col-lg-3">
                        <div class="metric-card">
                            <div class="metric-label">Max Similaridade</div>
                            <div class="metric-value">{max_similarity:.4f}</div>
                        </div>
                    </div>
                    <div class="col-md-6 col-lg-3">
                        <div class="metric-card">
                            <div class="metric-label">Min Similaridade</div>
                            <div class="metric-value">{min_similarity:.4f}</div>
                        </div>
                    </div>
                </div>
            </section>
            
            <!-- SEÇÃO 2: GRÁFICOS -->
            <section>
                <h2>Gráficos Gerados</h2>
                <p class="section-subtitle">Visualizações dos dados de análise</p>
                <div class="row">
                    {images_html}
                </div>
            </section>
            
            <!-- SEÇÃO 3: TABELA DE DADOS -->
            <section>
                <h2>Dados Detalhados</h2>
                <p class="section-subtitle">Tabela completa de resultados</p>
                <div class="table-container">
                    {table_html}
                </div>
            </section>
            
            <!-- SEÇÃO 4: MÉTRICAS COMPLETAS -->
            <section>
                <h2>Análise Estatística Completa</h2>
                <p class="section-subtitle">Resumo detalhado de métricas do sistema</p>
                <div class="metrics-text">{metrics_text if metrics_text else 'Métricas não disponíveis'}</div>
            </section>
            
            <!-- SEÇÃO 5: INFORMAÇÕES DO SISTEMA -->
            <section>
                <h2>Informações do Sistema</h2>
                <div class="row">
                    <div class="col-md-6">
                        <div class="metric-card">
                            <h4 style="color: #000; margin-bottom: 15px;">Sobre AV-Tracker</h4>
                            <p style="text-align: left; font-size: 0.9em;">
                                <strong>Tipo:</strong> Sistema de Re-identificação de Voz<br>
                                <strong>Modelo:</strong> Reconhecimento de padrões de áudio<br>
                                <strong>Aplicação:</strong> Identificação de speakers em múltiplos áudios<br>
                                <strong>Status:</strong> Ativo
                            </p>
                        </div>
                    </div>
                    <div class="col-md-6">
                        <div class="metric-card">
                            <h4 style="color: #000; margin-bottom: 15px;">Dados Processados</h4>
                            <p style="text-align: left; font-size: 0.9em;">
                                <strong>Data da Análise:</strong> {datetime.now().strftime('%d/%m/%Y %H:%M')}<br>
                                <strong>Total de Arquivos:</strong> {total_files}<br>
                                <strong>Total de Chunks:</strong> {int(total_chunks)}<br>
                                <strong>Versão do Relatório:</strong> 1.0
                            </p>
                        </div>
                    </div>
                </div>
            </section>
        </main>
        
        <footer>
            <p>Dashboard AV-Tracker | 2025 | Desenvolvido para análise de dados de áudio</p>
            <small>Relatório gerado automaticamente - {datetime.now().strftime('%d/%m/%Y às %H:%M:%S')}</small>
        </footer>
        
        <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
        <script>
            function openModal(img) {{
                const modal = document.getElementById('imageModal');
                const modalImg = document.getElementById('modalImage');
                modalImg.src = img.src;
                modal.classList.add('show');
                document.body.style.overflow = 'hidden';
            }}
            
            function closeModal(event) {{
                if (event && event.target.id !== 'imageModal') return;
                const modal = document.getElementById('imageModal');
                modal.classList.remove('show');
                document.body.style.overflow = 'auto';
            }}
            
            document.addEventListener('keydown', function(event) {{
                if (event.key === 'Escape') {{
                    closeModal();
                }}
            }});
        </script>
    </body>
    </html>
    """
    
    # Salvar relatório
    output_path = Path('analysis_results/dashboard_report.html')
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✅ Relatório HTML gerado com sucesso!")
    print(f"📁 Local: {output_path}")
    print(f"🌐 Abra em qualquer navegador para visualizar")
    
    return str(output_path)

if __name__ == "__main__":
    try:
        report_path = generate_html_report()
        print(f"\n🎉 Relatório disponível em: {report_path}")
    except Exception as e:
        print(f"❌ Erro ao gerar relatório: {e}")
        import traceback
        traceback.print_exc()
