@echo off
cd /d "C:\Users\manu\av-tracker"
call "%USERPROFILE%\miniconda3\condabin\conda.bat" activate tracker
echo 2| python run_multimodal_tracker.py
pause
