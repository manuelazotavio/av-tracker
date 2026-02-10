@echo off
echo ================================================================================
echo PROCESSAMENTO DE TODOS OS ARQUIVOS DE AUDIO
echo ================================================================================
echo.
echo Embeddings: data\embeddings
echo Input: data\demo  
echo Output: results.csv
echo Threshold: 0.5
echo.
echo ================================================================================
echo Iniciando processamento... Isso pode levar alguns minutos.
echo ================================================================================
echo.

python scripts\run_verifier.py -e data\embeddings -i data\demo -o results.csv -t 0.5

echo.
echo ================================================================================
echo PROCESSAMENTO CONCLUIDO!
echo ================================================================================
echo.
echo Arquivo gerado: results.csv
echo.
echo Proximos passos:
echo 1. Visualizar o dashboard: python generate_html_report.py
echo 2. Iniciar servidor web: python web_dashboard.py
echo.
pause
