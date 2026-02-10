import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import json
import numpy as np
from PIL import Image
import os

# Configuração da página
st.set_page_config(
    page_title="AV-Tracker Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilo customizado
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
    .metric-value {
        font-size: 32px;
        font-weight: bold;
        color: #1f77b4;
    }
    .metric-label {
        font-size: 14px;
        color: #666;
        margin-top: 5px;
    }
    h1 {
        color: #1f77b4;
        text-align: center;
    }
    h2 {
        color: #2c3e50;
        border-bottom: 2px solid #1f77b4;
        padding-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Título principal
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown("<h1>📊 AV-Tracker Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #666;'>Sistema de Re-identificação de Voz</p>", unsafe_allow_html=True)

st.divider()

# ===== CARREGAMENTO DE DADOS =====
@st.cache_data
def load_data():
    data = {}
    
    # Carregar CSV de resultados
    try:
        data['results_df'] = pd.read_csv('results.csv')
    except:
        data['results_df'] = None
    
    # Carregar métricas
    try:
        with open('analysis_results/metrics_summary.txt', 'r', encoding='utf-8') as f:
            data['metrics_text'] = f.read()
    except:
        data['metrics_text'] = None
    
    # Carregar CSVs detalhados se existirem
    try:
        data['detailed_df'] = pd.read_csv('analysis_results/detailed_results_with_females.csv')
    except:
        try:
            data['detailed_df'] = pd.read_csv('analysis_results/detailed_results.csv')
        except:
            data['detailed_df'] = None
    
    return data

data = load_data()
results_df = data['results_df']
detailed_df = data['detailed_df']

# ===== SEÇÃO 1: MÉTRICAS GERAIS =====
st.markdown("## 📈 Métricas Gerais do Sistema")

if results_df is not None:
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total de Arquivos", len(results_df))
    with col2:
        total_chunks = results_df['chunks_processed'].sum()
        st.metric("Chunks Processados", int(total_chunks))
    with col3:
        total_identified = results_df['identified_count'].sum()
        st.metric("Identificados", int(total_identified))
    with col4:
        if total_chunks > 0:
            taxa_geral = (total_identified / total_chunks) * 100
            st.metric("Taxa de Identificação", f"{taxa_geral:.1f}%")

st.divider()

# ===== SEÇÃO 2: ANÁLISE DE PERFORMANCE =====
st.markdown("## 🎯 Performance do Sistema")

col1, col2 = st.columns(2)

# Gráfico de identificações vs desconhecidos
with col1:
    if results_df is not None:
        identified = results_df['identified_count'].sum()
        unknown = results_df['unknown_count'].sum()
        
        fig = go.Figure(data=[
            go.Pie(
                labels=['Identificados', 'Desconhecidos'],
                values=[identified, unknown],
                marker=dict(colors=['#1f77b4', '#ff7f0e']),
                textposition='inside',
                textinfo='label+percent'
            )
        ])
        fig.update_layout(
            title="Distribuição: Identificados vs Desconhecidos",
            height=400,
            font=dict(size=12)
        )
        st.plotly_chart(fig, use_container_width=True)

# Gráfico de similaridade média por arquivo
with col2:
    if results_df is not None:
        fig = go.Figure(data=[
            go.Bar(
                x=results_df.index.astype(str),
                y=results_df['avg_similarity'],
                marker=dict(
                    color=results_df['avg_similarity'],
                    colorscale='Viridis',
                    showscale=False
                ),
                text=results_df['avg_similarity'].round(3),
                textposition='outside'
            )
        ])
        fig.update_layout(
            title="Similaridade Média por Arquivo",
            xaxis_title="Arquivo",
            yaxis_title="Similaridade Média",
            height=400,
            showlegend=False,
            xaxis=dict(showticklabels=False)
        )
        st.plotly_chart(fig, use_container_width=True)

st.divider()

# ===== SEÇÃO 3: GRÁFICOS DISPONÍVEIS =====
st.markdown("## 📊 Gráficos Gerados")

plots_dir = Path('analysis_results/plots')
if plots_dir.exists():
    plot_files = sorted(list(plots_dir.glob('*.png')))
    
    if plot_files:
        # Criar colunas para os gráficos
        cols = st.columns(2)
        for idx, plot_file in enumerate(plot_files):
            with cols[idx % 2]:
                st.markdown(f"### {plot_file.stem.replace('_', ' ').title()}")
                image = Image.open(plot_file)
                st.image(image, use_column_width=True)
    else:
        st.info("Nenhum gráfico gerado encontrado.")
else:
    st.warning("Diretório de gráficos não encontrado.")

st.divider()

# ===== SEÇÃO 4: DETALHES DE SPEAKERS =====
if results_df is not None:
    st.markdown("## 🎤 Análise de Speakers")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Taxa de identificação por número de speakers
        speaker_analysis = results_df.groupby('speaker_count').agg({
            'identified_count': 'sum',
            'chunks_processed': 'sum'
        }).reset_index()
        
        speaker_analysis['taxa_id'] = (
            speaker_analysis['identified_count'] / speaker_analysis['chunks_processed']
        ) * 100
        
        fig = go.Figure(data=[
            go.Bar(
                x=speaker_analysis['speaker_count'],
                y=speaker_analysis['taxa_id'],
                marker=dict(color='#1f77b4'),
                text=speaker_analysis['taxa_id'].round(1),
                textposition='outside'
            )
        ])
        fig.update_layout(
            title="Taxa de Identificação por Número de Speakers",
            xaxis_title="Número de Speakers",
            yaxis_title="Taxa de Identificação (%)",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Similaridade média por speaker count
        fig = go.Figure(data=[
            go.Box(
                x=results_df['speaker_count'],
                y=results_df['avg_similarity'],
                marker=dict(color='#2ca02c')
            )
        ])
        fig.update_layout(
            title="Distribuição de Similaridade por Speakers",
            xaxis_title="Número de Speakers",
            yaxis_title="Similaridade Média",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

st.divider()

# ===== SEÇÃO 5: ESTATÍSTICAS DETALHADAS =====
st.markdown("## 📋 Estatísticas Detalhadas")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Similaridade Máxima",
        f"{results_df['top_similarity'].max():.4f}" if results_df is not None else "N/A"
    )
    st.metric(
        "Similaridade Mínima",
        f"{results_df['bottom_similarity'].min():.4f}" if results_df is not None else "N/A"
    )

with col2:
    st.metric(
        "Avg de Chunks/Arquivo",
        f"{results_df['chunks_processed'].mean():.0f}" if results_df is not None else "N/A"
    )
    st.metric(
        "Avg Identificados/Arquivo",
        f"{results_df['identified_count'].mean():.0f}" if results_df is not None else "N/A"
    )

with col3:
    st.metric(
        "Similaridade Média Global",
        f"{results_df['avg_similarity'].mean():.4f}" if results_df is not None else "N/A"
    )
    st.metric(
        "Desvio Padrão",
        f"{results_df['avg_similarity'].std():.4f}" if results_df is not None else "N/A"
    )

st.divider()

# ===== SEÇÃO 6: TABELA DE DADOS =====
st.markdown("## 📊 Dados Detalhados")

if results_df is not None:
    # Opções de filtro
    col1, col2, col3 = st.columns(3)
    
    with col1:
        min_speakers = st.slider(
            "Número mínimo de speakers",
            int(results_df['speaker_count'].min()),
            int(results_df['speaker_count'].max()),
            int(results_df['speaker_count'].min())
        )
    
    with col2:
        min_sim = st.slider(
            "Similaridade mínima",
            0.0,
            1.0,
            0.0
        )
    
    with col3:
        sort_by = st.selectbox(
            "Ordenar por",
            ["avg_similarity", "chunks_processed", "identified_count", "speaker_count"]
        )
    
    # Filtrar dados
    filtered_df = results_df[
        (results_df['speaker_count'] >= min_speakers) &
        (results_df['avg_similarity'] >= min_sim)
    ].sort_values(sort_by, ascending=False)
    
    # Exibir tabela
    st.dataframe(
        filtered_df,
        use_container_width=True,
        height=400
    )
    
    # Opção para download
    csv = filtered_df.to_csv(index=False)
    st.download_button(
        label="📥 Baixar dados como CSV",
        data=csv,
        file_name="resultados_filtrados.csv",
        mime="text/csv"
    )

st.divider()

# ===== SEÇÃO 7: RESUMO DE MÉTRICAS =====
if data['metrics_text']:
    st.markdown("## 📈 Resumo de Métricas Completas")
    
    with st.expander("Ver métricas detalhadas", expanded=False):
        st.markdown("""
        <pre style='background-color: #f5f5f5; padding: 15px; border-radius: 5px; font-size: 12px;'>
        """ + data['metrics_text'].replace('<', '&lt;').replace('>', '&gt;') + """
        </pre>
        """, unsafe_allow_html=True)

st.divider()

# ===== SEÇÃO 8: INFORMAÇÕES GERAIS =====
st.markdown("## ℹ️ Informações do Sistema")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### Sobre o AV-Tracker
    - **Tipo**: Sistema de Re-identificação de Voz
    - **Modelo**: Reconhecimento de padrões de áudio
    - **Aplicação**: Identificação de speakers em múltiplos áudios
    """)

with col2:
    st.markdown(f"""
    ### Dados Carregados
    - **Arquivos processados**: {len(results_df) if results_df is not None else 'N/A'}
    - **Data da análise**: {pd.Timestamp.now().strftime('%d/%m/%Y %H:%M')}
    - **Status**: ✅ Ativo
    """)

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p>Dashboard AV-Tracker | © 2025 | Desenvolvido para análise de dados de áudio</p>
</div>
""", unsafe_allow_html=True)
