@echo off
title AV-Tracker
cd /d "%~dp0"
call "%USERPROFILE%\miniconda3\condabin\conda.bat" activate tracker
streamlit run app/app.py --server.port 8501 --browser.gatherUsageStats false
pause
