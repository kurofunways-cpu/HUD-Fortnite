@echo off
chcp 65001 >nul
cd /d "%~dp0"
title HUD Aliado V4
if not exist ".venv\Scripts\python.exe" (
 echo Instale primeiro usando instalar.bat
 pause
 exit /b 1
)
".venv\Scripts\python.exe" app.py
if errorlevel 1 (
 echo.
 echo O programa encontrou um erro. Envie a mensagem acima.
 pause
)
