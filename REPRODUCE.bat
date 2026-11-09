@echo off
cd /d "%~dp0"
python -m unittest discover -s tests -v
if errorlevel 1 exit /b 1
python -m label_audit evaluate --out runtime\evaluation
pause
