"""
Mescla dois arquivos de resultados batch para incluir dados de 2_speakers
"""
import json
from pathlib import Path
from datetime import datetime

def merge_batch_results(file1_path, file2_path, output_path):
    """Mescla dois arquivos JSON de resultados batch"""
    
    # Carregar ambos os arquivos
    with open(file1_path, 'r', encoding='utf-8') as f:
        data1 = json.load(f)
    
    with open(file2_path, 'r', encoding='utf-8') as f:
        data2 = json.load(f)
    
    # Criar estrutura mesclada
    merged = {
        'timestamp': datetime.now().isoformat(),
        'files_processed': data1.get('total_processed', data1.get('files_processed', 0)) + data2.get('total_processed', data2.get('files_processed', 0)),
        'total_duration': data1.get('total_duration', 0) + data2.get('total_duration', 0),
        'total_processing_time': data1.get('total_processing_time', 0) + data2.get('total_processing_time', 0),
        'overall_metrics': {},
        'by_number_of_speakers': {},
        'by_audio_volume': {},
        'by_gender': {},
        'file_results': data1.get('results', data1.get('file_results', [])) + data2.get('results', data2.get('file_results', []))
    }
    
    # Mesclar métricas gerais (fazer média ponderada)
    total_files = merged['files_processed']
    files1 = data1.get('total_processed', data1.get('files_processed', 0))
    files2 = data2.get('total_processed', data2.get('files_processed', 0))
    
    om1 = data1.get('overall_metrics', {})
    om2 = data2.get('overall_metrics', {})
    
    if total_files > 0:
        merged['overall_metrics'] = {
            'average_speaker_accuracy': (om1.get('average_speaker_accuracy', 0) * files1 + 
                                         om2.get('average_speaker_accuracy', 0) * files2) / total_files,
            'average_precision': (om1.get('average_precision', 0) * files1 + 
                                 om2.get('average_precision', 0) * files2) / total_files,
            'average_recall': (om1.get('average_recall', 0) * files1 + 
                              om2.get('average_recall', 0) * files2) / total_files,
            'average_f1_score': (om1.get('average_f1_score', 0) * files1 + 
                                om2.get('average_f1_score', 0) * files2) / total_files,
            'average_diarization_error_rate': (om1.get('average_diarization_error_rate', 0) * files1 + 
                                              om2.get('average_diarization_error_rate', 0) * files2) / total_files,
            'average_processing_speed_ratio': (om1.get('average_processing_speed_ratio', 0) * files1 + 
                                              om2.get('average_processing_speed_ratio', 0) * files2) / total_files
        }
    else:
        merged['overall_metrics'] = om2 if files2 > 0 else om1
    
    # Mesclar by_number_of_speakers
    speakers1 = data1.get('by_number_of_speakers', {})
    speakers2 = data2.get('by_number_of_speakers', {})
    all_speaker_keys = set(speakers1.keys()) | set(speakers2.keys())
    
    for key in all_speaker_keys:
        s1 = speakers1.get(key, {})
        s2 = speakers2.get(key, {})
        count1 = s1.get('count', 0)
        count2 = s2.get('count', 0)
        total_count = count1 + count2
        
        if total_count > 0:
            merged['by_number_of_speakers'][key] = {
                'count': total_count,
                'avg_accuracy': (s1.get('avg_accuracy', 0) * count1 + 
                               s2.get('avg_accuracy', 0) * count2) / total_count,
                'min_accuracy': min(s1.get('min_accuracy', 100), s2.get('min_accuracy', 100)),
                'max_accuracy': max(s1.get('max_accuracy', 0), s2.get('max_accuracy', 0))
            }
    
    # Mesclar by_audio_volume
    vol1 = data1.get('by_audio_volume', {})
    vol2 = data2.get('by_audio_volume', {})
    all_vol_keys = set(vol1.keys()) | set(vol2.keys())
    
    for key in all_vol_keys:
        v1 = vol1.get(key, {})
        v2 = vol2.get(key, {})
        count1 = v1.get('count', 0)
        count2 = v2.get('count', 0)
        total_count = count1 + count2
        
        if total_count > 0:
            merged['by_audio_volume'][key] = {
                'count': total_count,
                'avg_accuracy': (v1.get('avg_accuracy', 0) * count1 + 
                               v2.get('avg_accuracy', 0) * count2) / total_count,
                'min_accuracy': min(v1.get('min_accuracy', 100), v2.get('min_accuracy', 100)),
                'max_accuracy': max(v1.get('max_accuracy', 0), v2.get('max_accuracy', 0))
            }
    
    # Mesclar by_gender
    gen1 = data1.get('by_gender', {})
    gen2 = data2.get('by_gender', {})
    all_gen_keys = set(gen1.keys()) | set(gen2.keys())
    
    for key in all_gen_keys:
        g1 = gen1.get(key, {})
        g2 = gen2.get(key, {})
        
        # Pegar valores ou usar campos alternativos
        speakers1 = g1.get('total_speakers', 0)
        speakers2 = g2.get('total_speakers', 0)
        total_speakers = speakers1 + speakers2
        
        segments1 = g1.get('total_segments', g1.get('count', 0))
        segments2 = g2.get('total_segments', g2.get('count', 0))
        total_count = segments1 + segments2
        
        # Calcular accuracy ponderada
        if total_count > 0:
            acc1 = g1.get('avg_accuracy', 0)
            acc2 = g2.get('avg_accuracy', 0)
            avg_accuracy = (acc1 * segments1 + acc2 * segments2) / total_count
            
            merged['by_gender'][key] = {
                'count': total_count,
                'total_speakers': total_speakers,
                'avg_accuracy': avg_accuracy
            }
    
    # Salvar arquivo mesclado
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Arquivos mesclados com sucesso!")
    print(f"📁 Arquivo de saída: {output_path}")
    print(f"📊 Total de arquivos: {merged['files_processed']}")
    print(f"👥 Categorias de speakers: {list(merged['by_number_of_speakers'].keys())}")

if __name__ == "__main__":
    # Arquivos a mesclar
    file1 = "logs/batch_results_20260205_165118.json"  # Tem 2_speakers
    file2 = "logs/batch_results_20260206_050129.json"  # Tem 3,4,5_speakers
    output = "logs/batch_results_merged.json"
    
    merge_batch_results(file1, file2, output)
