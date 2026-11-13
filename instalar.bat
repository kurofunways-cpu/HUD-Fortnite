@echo off
chcp 65001 >nul
cd /d "%~dp0"
title Instalar HUD Aliado V4
set "PY_CMD="
py -3.12 -c "import sys" >nul 2>&1
if not errorlevel 1 set "PY_CMD=py -3.12"
if not defined PY_CMD (
  py -3.11 -c "import sys" >nul 2>&1
  if not errorlevel 1 set "PY_CMD=py -3.11"
)
if not defined PY_CMD (
  echo.
  echo ERRO: Instale Python 3.12 64 bits no site oficial.
  echo https://www.python.org/downloads/release/python-31210/
  echo Python 3.15 pode nao ser compativel com as bibliotecas de IA no Windows.
  pause
  exit /b 1
)
echo Criando ambiente isolado usando %PY_CMD%...
%PY_CMD% -m venv .venv
if errorlevel 1 goto erro
set "ENV_PY=.venv\Scripts\python.exe"
"%ENV_PY%" -m pip install --upgrade pip
if errorlevel 1 goto erro
echo Instalando dependencias diretamente do PyPI...
"%ENV_PY%" -m pip install -r requirements.txt
if errorlevel 1 goto erro
echo.
echo Instalacao concluida! Execute iniciar.bat
pause
exit /b 0
:erro
echo.
echo ERRO durante a instalacao. Copie as ultimas linhas e envie no chat.
pause
exit /b 1
