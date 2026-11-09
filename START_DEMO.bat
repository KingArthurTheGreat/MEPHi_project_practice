@echo off
cd /d "%~dp0"
python -m label_audit serve --open
pause

