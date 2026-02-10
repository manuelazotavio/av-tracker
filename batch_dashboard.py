"""
Dashboard para visualizar resultados do processamento em lote multi-speaker
"""

from flask import Flask, render_template_string
import json
import os
from pathlib import Path
from datetime import datetime

app = Flask(__name__)

def load_latest_batch_results():
    """Carrega o arquivo JSON mais recente (prioriza merged)"""
    logs_dir = Path('logs')
    
    # Verifica se existe arquivo mesclado
    merged_file = logs_dir / 'batch_results_merged.json'
    if merged_file.exists():
        latest_file = merged_file
    else:
        # Pega o mais recente
        json_files = list(logs_dir.glob('batch_results_*.json'))
        if not json_files:
            return None
        latest_file = max(json_files, key=lambda p: p.stat().st_mtime)
    
    with open(latest_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    data['filename'] = latest_file.name
    data['file_date'] = datetime.fromtimestamp(latest_file.stat().st_mtime).strftime('%d/%m/%Y %H:%M:%S')
    
    # Adiciona exemplo de áudio processado
    if 'file_results' in data and len(data['file_results']) > 0:
        # Pega o primeiro resultado com sucesso
        for result in data['file_results']:
            if result.get('success'):
                data['example_audio'] = result
                break
    
    return data

DASHBOARD_HTML = '''
<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AV-Tracker - Multi-Speaker Dashboard</title>
    <link href="https://fonts.googleapis.com/css2?family=Google+Sans:wght@400;500;700&display=swap" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Google Sans', 'Roboto', sans-serif;
            background: #000000;
            color: #ffffff;
            padding: 20px;
            min-height: 100vh;
        }
        
        .container {
            max-width: 1800px;
            margin: 0 auto;
        }
        
        .header {
            background: #0a0a0a;
            padding: 40px;
            border-radius: 8px;
            border: 1px solid #1a1a1a;
            margin-bottom: 30px;
            text-align: center;
        }
        
        .header h1 {
            color: #ffffff;
            font-size: 3em;
            margin-bottom: 10px;
            font-weight: 700;
        }
        
        .header .subtitle {
            color: #999999;
            font-size: 1.2em;
            font-weight: 400;
        }
        
        .header .file-info {
            margin-top: 20px;
            padding: 15px;
            background: #111111;
            border-radius: 6px;
            font-size: 0.95em;
            color: #cccccc;
            border: 1px solid #1a1a1a;
        }
        
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .metric-card {
            background: #0a0a0a;
            padding: 30px;
            border-radius: 8px;
            border: 1px solid #1a1a1a;
            transition: all 0.3s ease;
        }
        
        .metric-card:hover {
            background: #111111;
            border-color: #333333;
        }
        
        .metric-card .label {
            color: #888888;
            font-size: 0.9em;
            text-transform: uppercase;
            letter-spacing: 1.5px;
            margin-bottom: 15px;
            font-weight: 500;
        }
        
        .metric-card .value {
            font-size: 3em;
            font-weight: 700;
            color: #ffffff;
        }
        
        .metric-card .sub-value {
            font-size: 0.9em;
            color: #666666;
            margin-top: 10px;
            font-weight: 400;
        }
        
        .details-section {
            background: #0a0a0a;
            padding: 35px;
            border-radius: 8px;
            border: 1px solid #1a1a1a;
            margin-bottom: 25px;
        }
        
        .details-section h2 {
            color: #ffffff;
            margin-bottom: 25px;
            padding-bottom: 15px;
            border-bottom: 1px solid #1a1a1a;
            font-size: 1.8em;
            font-weight: 700;
        }
        
        .chart-container {
            position: relative;
            height: 450px;
            margin-top: 20px;
        }
        
        .charts-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(550px, 1fr));
            gap: 25px;
            margin-bottom: 30px;
        }
        
        .stats-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }
        
        .stats-table th,
        .stats-table td {
            padding: 15px;
            text-align: left;
            border-bottom: 1px solid #1a1a1a;
        }
        
        .stats-table th {
            background: #111111;
            color: #ffffff;
            font-weight: 600;
            text-transform: uppercase;
            font-size: 0.85em;
            letter-spacing: 1px;
        }
        
        .stats-table tr:hover {
            background: #111111;
        }
        
        .stats-table td {
            color: #cccccc;
        }
        
        .progress-bar {
            width: 100%;
            height: 10px;
            background: #1a1a1a;
            border-radius: 5px;
            overflow: hidden;
            margin-top: 8px;
        }
        
        .progress-fill {
            height: 100%;
            background: #ffffff;
            transition: width 0.5s ease;
        }
        
        .example-section {
            background: #0a0a0a;
            padding: 30px;
            border-radius: 8px;
            border: 1px solid #1a1a1a;
            margin-top: 30px;
        }
        
        .example-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
        }
        
        .example-metrics {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 20px;
        }
        
        .example-metric {
            background: #111111;
            padding: 15px;
            border-radius: 6px;
            border: 1px solid #1a1a1a;
        }
        
        .example-metric .label {
            color: #888888;
            font-size: 0.85em;
            margin-bottom: 8px;
        }
        
        .example-metric .value {
            color: #ffffff;
            font-size: 1.5em;
            font-weight: 600;
        }
        
        .speakers-list {
            margin-top: 20px;
        }
        
        .speaker-row {
            display: flex;
            justify-content: space-between;
            padding: 12px;
            background: #111111;
            margin-bottom: 8px;
            border-radius: 6px;
            border: 1px solid #1a1a1a;
        }
        
        .speaker-row .name {
            color: #ffffff;
            font-weight: 500;
        }
        
        .speaker-row .metrics {
            color: #999999;
            font-size: 0.9em;
        }
    </style>
</head>

        
        
<body>
    <div class="container">
        <div class="header">
            <h1>AV-Tracker Dashboard</h1>
            <div class="subtitle">Multi-Speaker Processing Analysis</div>
            <div class="file-info">
                <strong>{{ data.filename }}</strong> | 
                🕐 <strong>{{ data.file_date }}</strong>
            </div>
        </div>
        
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="label">Total Processed</div>
                <div class="value">{{ data.files_processed }}</div>
                <div class="sub-value">audio files</div>
            </div>
            
            <div class="metric-card">
                <div class="label">Success</div>
                <div class="value">{{ data.file_results|length if data.file_results else 0 }}</div>
                <div class="sub-value">{{ "%.1f"|format(((data.file_results|length) / data.files_processed * 100) if data.files_processed > 0 else 0) }}% success rate</div>
            </div>
            
            <div class="metric-card">
                <div class="label">Speaker Accuracy</div>
                <div class="value">{{ "%.1f"|format(data.overall_metrics.average_speaker_accuracy) }}%</div>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: {{ data.overall_metrics.average_speaker_accuracy }}%"></div>
                </div>
            </div>
            
            
            
            <div class="metric-card">
                <div class="label">Recall</div>
                <div class="value">{{ "%.1f"|format(data.overall_metrics.average_recall) }}%</div>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: {{ data.overall_metrics.average_recall }}%"></div>
                </div>
            </div>
            
            <div class="metric-card">
                <div class="label">F1-Score</div>
                <div class="value">{{ "%.1f"|format(data.overall_metrics.average_f1_score) }}%</div>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: {{ data.overall_metrics.average_f1_score }}%"></div>
                </div>
            </div>
            
            <div class="metric-card">
                <div class="label">DER (Error)</div>
                <div class="value">{{ "%.1f"|format(data.overall_metrics.average_diarization_error_rate) }}%</div>
                <div class="sub-value">Diarization Error Rate</div>
            </div>
            
            <div class="metric-card">
                <div class="label">Speed</div>
                <div class="value">{{ "%.2f"|format(data.overall_metrics.average_processing_speed_ratio) }}x</div>
                <div class="sub-value">real-time</div>
            </div>
        </div>
        
        {% if data.example_audio %}
        <div class="example-section">
            <div class="example-header">
                <h2>Processed audio example</h2>
            </div>
            
            <div style="background: #111111; padding: 20px; border-radius: 6px; margin-bottom: 20px; border: 1px solid #1a1a1a;">
                <div style="color: #ffffff; font-size: 1.3em; font-weight: 600; margin-bottom: 10px;">
                    {{ data.example_audio.audio_file }}
                </div>
                <div style="color: #888888;">
                    Duration: {{ "%.2f"|format(data.example_audio.metrics.audio_duration_seconds) }}s | 
                    Processing: {{ "%.2f"|format(data.example_audio.metrics.processing_time_seconds) }}s | 
                    Volume: {{ data.example_audio.metrics.audio_metrics.volume_category.title() }} ({{ "%.1f"|format(data.example_audio.metrics.audio_metrics.db_level) }} dB)
                </div>
            </div>
            
            <div class="example-metrics">
                <div class="example-metric">
                    <div class="label">Accuracy</div>
                    <div class="value">{{ "%.1f"|format(data.example_audio.metrics.speaker_accuracy) }}%</div>
                </div>
                
                <div class="example-metric">
                    <div class="label">Recall</div>
                    <div class="value">{{ "%.1f"|format(data.example_audio.metrics.speaker_recall) }}%</div>
                </div>
                <div class="example-metric">
                    <div class="label">F1-Score</div>
                    <div class="value">{{ "%.1f"|format(data.example_audio.metrics.speaker_f1_score) }}%</div>
                </div>
                <div class="example-metric">
                    <div class="label">GT Speakers</div>
                    <div class="value">{{ data.example_audio.metrics.num_speakers_ground_truth }}</div>
                </div>
                <div class="example-metric">
                    <div class="label">Detected Speakers</div>
                    <div class="value">{{ data.example_audio.metrics.num_speakers_predicted }}</div>
                </div>
            </div>
            
            <div class="speakers-list">
                <h3 style="color: #ffffff; margin-bottom: 15px; font-size: 1.2em;">Ground Truth vs Predicted Speakers</h3>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
                    <div>
                        <div style="color: #888888; margin-bottom: 10px; font-size: 0.9em; text-transform: uppercase;">Ground Truth</div>
                        {% for speaker in data.example_audio.metrics.ground_truth_speakers %}
                        <div class="speaker-row">
                            <span class="name">{{ speaker }}</span>
                            <span class="metrics">
                                {% if speaker in data.example_audio.metrics.per_speaker_metrics %}
                                {{ data.example_audio.metrics.per_speaker_metrics[speaker].gender|upper }}
                                {% endif %}
                            </span>
                        </div>
                        {% endfor %}
                    </div>
                    <div>
                        <div style="color: #888888; margin-bottom: 10px; font-size: 0.9em; text-transform: uppercase;">Predicted</div>
                        {% for speaker in data.example_audio.metrics.predicted_speakers %}
                        <div class="speaker-row">
                            <span class="name">{{ speaker }}</span>
                            <span class="metrics">
                                {% if speaker in data.example_audio.metrics.per_speaker_metrics %}
                                Prec: {{ "%.0f"|format(data.example_audio.metrics.per_speaker_metrics[speaker].precision) }}% | 
                                Rec: {{ "%.0f"|format(data.example_audio.metrics.per_speaker_metrics[speaker].recall) }}%
                                {% endif %}
                            </span>
                        </div>
                        {% endfor %}
                    </div>
                </div>
            </div>
        </div>
        {% endif %}
        
        <div class="charts-grid">
            <div class="details-section">
                <h2>Key metrics</h2>
                <div class="chart-container">
                    <canvas id="metricsRadarChart"></canvas>
                </div>
            </div>
            
            <div class="details-section">
                <h2>Accuracy by speakers</h2>
                <p style="color: #cccccc; margin-bottom: 15px; font-size: 0.95em;">
                    Detailed comparison of metrics by number of speakers in the audio
                    {% if '2_speakers' not in data.by_number_of_speakers %}
                    <br><strong>Note:</strong> 2-speaker data not included in this batch (files 87-200)
                    {% endif %}
                </p>
                <div class="chart-container">
                    <canvas id="speakersBarChart"></canvas>
                </div>
            </div>
        </div>
        
        <div class="charts-grid">
            <div class="details-section">
                <h2>Accuracy by volume</h2>
                <div class="chart-container">
                    <canvas id="volumeBarChart"></canvas>
                </div>
            </div>
            
            <div class="details-section">
                <h2>Gender distribution</h2>
                <div class="chart-container">
                    <canvas id="genderDoughnutChart"></canvas>
                </div>
            </div>
        </div>
        
        <div class="details-section">
            <h2>By number of speakers</h2>
            <table class="stats-table">
                <thead>
                    <tr>
                        <th>Category</th>
                        <th>Files</th>
                        <th>Average Accuracy</th>
                        <th>Min</th>
                        <th>Max</th>
                        <th>Progress</th>
                    </tr>
                </thead>
                <tbody>
                    {% for key, value in data.by_number_of_speakers.items() %}
                    <tr>
                        <td><strong>{{ key.replace('_', ' ').title() }}</strong></td>
                        <td>{{ value.count }}</td>
                        <td><strong>{{ "%.1f"|format(value.avg_accuracy) }}%</strong></td>
                        <td>{{ "%.1f"|format(value.min_accuracy) }}%</td>
                        <td>{{ "%.1f"|format(value.max_accuracy) }}%</td>
                        <td>
                            <div class="progress-bar">
                                <div class="progress-fill" style="width: {{ value.avg_accuracy }}%"></div>
                            </div>
                        </td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>
        
        <div class="details-section">
            <h2>By audio volume</h2>
            <table class="stats-table">
                <thead>
                    <tr>
                        <th>Volume</th>
                        <th>Files</th>
                        <th>Average Accuracy</th>
                        <th>Min</th>
                        <th>Max</th>
                    </tr>
                </thead>
                <tbody>
                    {% for key, value in data.by_audio_volume.items() %}
                    <tr>
                        <td><strong>{{ key.title() }}</strong></td>
                        <td>{{ value.count }}</td>
                        <td><strong>{{ "%.1f"|format(value.avg_accuracy) }}%</strong></td>
                        <td>{{ "%.1f"|format(value.min_accuracy) }}%</td>
                        <td>{{ "%.1f"|format(value.max_accuracy) }}%</td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>
        
        <div class="details-section">
            <h2>Gender analysis</h2>
            <div class="gender-stats">
                {% for gender, stats in data.by_gender.items() %}
                <div class="gender-card">
                    <h4>{{ 'Male' if gender == 'm' else ('Female' if gender == 'f' else 'Unknown') }}</h4>
                    <div class="stat">
                        <span>Speakers:</span>
                        <strong>{{ stats.total_speakers }}</strong>
                    </div>
                    <div class="stat">
                        <span>Segments:</span>
                        <strong>{{ stats.total_segments }}</strong>
                    </div>
                    <div class="stat">
                        <span>Correct:</span>
                        <strong>{{ stats.total_correct }}</strong>
                    </div>
                    <div class="stat">
                        <span>Accuracy:</span>
                        <strong>{{ "%.1f"|format(stats.avg_accuracy) }}%</strong>
                    </div>
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: {{ stats.avg_accuracy }}%"></div>
                    </div>
                </div>
                {% endfor %}
            </div>
        </div>
        
    </div>
    
    <script>
        // Configuração global dos gráficos
        Chart.defaults.color = '#cbd5e1';
        Chart.defaults.borderColor = 'rgba(99, 102, 241, 0.2)';
        Chart.defaults.font.family = "'Google Sans', 'Roboto', sans-serif";
        
        const gradientPurple = (ctx) => {
            const gradient = ctx.createLinearGradient(0, 0, 0, 400);
            gradient.addColorStop(0, 'rgba(129, 140, 248, 0.8)');
            gradient.addColorStop(1, 'rgba(192, 132, 252, 0.8)');
            return gradient;
        };
        
        // Radar Chart - Métricas Principais
        const radarCtx = document.getElementById('metricsRadarChart').getContext('2d');
        new Chart(radarCtx, {
            type: 'radar',
            data: {
                labels: ['Accuracy','Recall', 'F1-Score', 'DER', 'Processing Speed'],
                datasets: [{
                    label: 'Performance Geral',
                    data: [
                        {{ "%.1f"|format(data.overall_metrics.average_speaker_accuracy) }},
                        {{ "%.1f"|format(data.overall_metrics.average_recall) }},
                        {{ "%.1f"|format(data.overall_metrics.average_f1_score) }},
                        {{ "%.1f"|format(100 - (data.overall_metrics.average_diarization_error_rate or 0)) }},
                        {{ "%.1f"|format(data.overall_metrics.average_processing_speed_ratio * 100) }}
                    ],
                    backgroundColor: 'rgba(255, 255, 255, 0.1)',
                    borderColor: '#ffffff',
                    borderWidth: 2,
                    pointBackgroundColor: '#ffffff',
                    pointBorderColor: '#ffffff',
                    pointBorderWidth: 2,
                    pointRadius: 5,
                    pointHoverRadius: 8,
                    pointHoverBackgroundColor: '#cccccc',
                    pointHoverBorderColor: '#ffffff'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    r: {
                        beginAtZero: true,
                        max: 100,
                        ticks: {
                            stepSize: 10,
                            color: '#ffffff',
                            backdropColor: 'transparent',
                            font: {
                                size: 12
                            }
                        },
                        grid: {
                            color: 'rgba(255, 255, 255, 0.2)'
                        },
                        pointLabels: {
                            color: '#ffffff',
                            font: {
                                size: 13,
                                weight: 'bold'
                            }
                        }
                    }
                },
                plugins: {
                    legend: {
                        display: true,
                        labels: {
                            color: '#ffffff',
                            font: {
                                size: 14
                            }
                        }
                    },
                    tooltip: {
                        backgroundColor: 'rgba(0, 0, 0, 0.9)',
                        titleColor: '#ffffff',
                        bodyColor: '#ffffff',
                        borderColor: '#ffffff',
                        borderWidth: 1
                    }
                }
            }
        });
        
        // Bar Chart - Accuracy por Número de Speakers
        const speakersCtx = document.getElementById('speakersBarChart').getContext('2d');
        new Chart(speakersCtx, {
            type: 'bar',
            data: {
                labels: [{% for key in data.by_number_of_speakers.keys() %}'{{ key.replace("_", " ").title() }}'{% if not loop.last %}, {% endif %}{% endfor %}],
                datasets: [
                    {
                        label: 'Average Accuracy (%)',
                        data: [{% for value in data.by_number_of_speakers.values() %}{{ "%.1f"|format(value.avg_accuracy) }}{% if not loop.last %}, {% endif %}{% endfor %}],
                        backgroundColor: 'rgba(255, 255, 255, 0.9)',
                        borderColor: '#ffffff',
                        borderWidth: 2
                    },
                    {
                        label: 'Minimum Accuracy (%)',
                        data: [{% for value in data.by_number_of_speakers.values() %}{{ "%.1f"|format(value.min_accuracy) }}{% if not loop.last %}, {% endif %}{% endfor %}],
                        backgroundColor: 'rgba(180, 180, 180, 0.9)',
                        borderColor: '#b4b4b4',
                        borderWidth: 2
                    },
                    {
                        label: 'Maximum Accuracy (%)',
                        data: [{% for value in data.by_number_of_speakers.values() %}{{ "%.1f"|format(value.max_accuracy) }}{% if not loop.last %}, {% endif %}{% endfor %}],
                        backgroundColor: 'rgba(100, 100, 100, 0.9)',
                        borderColor: '#666666',
                        borderWidth: 2
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100,
                        ticks: {
                            stepSize: 10,
                            color: '#ffffff',
                            font: {
                                size: 12
                            }
                        },
                        grid: {
                            color: 'rgba(255, 255, 255, 0.2)'
                        }
                    },
                    x: {
                        ticks: {
                            color: '#ffffff',
                            font: {
                                size: 13,
                                weight: 'bold'
                            }
                        },
                        grid: {
                            display: false
                        }
                    }
                },
                plugins: {
                    legend: {
                        display: true,
                        labels: {
                            color: '#ffffff',
                            font: {
                                size: 13
                            }
                        }
                    },
                    tooltip: {
                        backgroundColor: 'rgba(0, 0, 0, 0.9)',
                        titleColor: '#ffffff',
                        bodyColor: '#ffffff',
                        borderColor: '#ffffff',
                        borderWidth: 1,
                        padding: 12,
                        displayColors: true,
                        callbacks: {
                            label: function(context) {
                                return context.dataset.label + ': ' + context.parsed.y.toFixed(1) + '%';
                            }
                        }
                    }
                }
            }
        });
        
        // Bar Chart - Accuracy por Volume
        const volumeCtx = document.getElementById('volumeBarChart').getContext('2d');
        new Chart(volumeCtx, {
            type: 'bar',
            data: {
                labels: [{% for key in data.by_audio_volume.keys() %}'{{ key.title() }}'{% if not loop.last %}, {% endif %}{% endfor %}],
                datasets: [{
                    label: 'Average Accuracy (%)',
                    data: [{% for value in data.by_audio_volume.values() %}{{ "%.1f"|format(value.avg_accuracy) }}{% if not loop.last %}, {% endif %}{% endfor %}],
                    backgroundColor: 'rgba(255, 255, 255, 0.9)',
                    borderColor: '#ffffff',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100,
                        ticks: {
                            stepSize: 10,
                            color: '#ffffff',
                            font: {
                                size: 12
                            }
                        },
                        grid: {
                            color: 'rgba(255, 255, 255, 0.2)'
                        }
                    },
                    x: {
                        ticks: {
                            color: '#ffffff',
                            font: {
                                size: 13,
                                weight: 'bold'
                            }
                        },
                        grid: {
                            display: false
                        }
                    }
                },
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        backgroundColor: 'rgba(0, 0, 0, 0.9)',
                        titleColor: '#ffffff',
                        bodyColor: '#ffffff',
                        borderColor: '#ffffff',
                        borderWidth: 1,
                        padding: 12,
                        displayColors: false
                    }
                }
            }
        });
        
        // Doughnut Chart - Gender Distribution
        const genderCtx = document.getElementById('genderDoughnutChart').getContext('2d');
        new Chart(genderCtx, {
            type: 'doughnut',
            data: {
                labels: [{% for gender in data.by_gender.keys() %}'{{ "Male" if gender == "m" else ("Female" if gender == "f" else "Unknown") }}'{% if not loop.last %}, {% endif %}{% endfor %}],
                datasets: [{
                    label: 'Speakers',
                    data: [{% for value in data.by_gender.values() %}{{ value.total_speakers }}{% if not loop.last %}, {% endif %}{% endfor %}],
                    backgroundColor: [
                        'rgba(255, 255, 255, 0.9)',
                        'rgba(180, 180, 180, 0.9)',
                        'rgba(100, 100, 100, 0.9)'
                    ],
                    borderColor: '#000000',
                    borderWidth: 2,
                    hoverOffset: 15,
                    hoverBorderColor: '#ffffff',
                    hoverBorderWidth: 3
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: {
                            color: '#ffffff',
                            padding: 20,
                            font: {
                                size: 14,
                                weight: 'bold'
                            },
                            usePointStyle: true,
                            pointStyle: 'circle'
                        }
                    },
                    tooltip: {
                        backgroundColor: 'rgba(0, 0, 0, 0.9)',
                        titleColor: '#ffffff',
                        bodyColor: '#ffffff',
                        borderColor: '#ffffff',
                        borderWidth: 1,
                        padding: 12,
                        callbacks: {
                            label: function(context) {
                                const total = context.dataset.data.reduce((a, b) => a + b, 0);
                                const percentage = ((context.parsed / total) * 100).toFixed(1);
                                return context.label + ': ' + context.parsed + ' speakers (' + percentage + '%)';
                            }
                        }
                    }
                }
            }
        });
    </script>
</body>
</html>
'''

@app.route('/')
def dashboard():
    data = load_latest_batch_results()
    
    if data is None:
        return "<h1>Nenhum resultado encontrado</h1><p>Execute o processamento em lote primeiro.</p>"
    
    return render_template_string(DASHBOARD_HTML, data=data)

@app.route('/api/results')
def api_results():
    """Endpoint JSON para integração"""
    data = load_latest_batch_results()
    return data if data else {"error": "No results found"}

if __name__ == '__main__':
    print("=" * 60)
    print("🎤 AV-TRACKER BATCH DASHBOARD")
    print("=" * 60)
    print("\n📊 Dashboard disponível em:")
    print("   http://localhost:5001")
    print("\n📡 API JSON disponível em:")
    print("   http://localhost:5001/api/results")
    print("\n⚡ Pressione Ctrl+C para parar")
    print("=" * 60)
    
    app.run(debug=True, host='0.0.0.0', port=5001)
