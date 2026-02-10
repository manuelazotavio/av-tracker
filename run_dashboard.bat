@echo off
REM Script para iniciar o dashboard AV-Tracker no Windows

echo.
echo ============================================
echo   AV-Tracker Dashboard
echo ============================================
echo.

REM Verificar se o Python está instalado
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python nao encontrado no PATH
    echo Instale Python ou adicione a seu PATH
    pause
    exit /b 1
)

REM Verificar e instalar dependências se necessário
echo Verificando dependências...
python -c "import streamlit" >nul 2>&1
if errorlevel 1 (
    echo.
    echo Streamlit nao esta instalado. Instalando dependencias...
    echo.
    python install_dashboard_deps.py
    if errorlevel 1 (
        echo Erro ao instalar dependencias!
        pause
        exit /b 1
    )
)

echo.
echo Iniciando dashboard...
echo O dashboard sera aberto em http://localhost:8501
echo.
echo Pressione CTRL+C para parar o servidor
echo.

python -m streamlit run dashboard.py

pause
