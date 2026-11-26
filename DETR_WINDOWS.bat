@echo off
setlocal EnableExtensions
cd /d "%~dp0" || exit /b 1
set "ENV_DIR=.venv-detr-315"
set "WHEELS=cu134"
if /I "%~1"=="cpu" (
  set "ENV_DIR=.venv-cpu-315"
  set "WHEELS=cpu"
)
set "EXE=%ENV_DIR%\Scripts\python.exe"
set "CHECK=import sys,struct,platform; assert sys.version_info[:2]==(3,15) and struct.calcsize('P')==8 and platform.machine().lower() in ('amd64','x86_64')"
if exist "%EXE%" goto deps
py -3.15 -c "%CHECK%" >nul 2>&1
if errorlevel 1 goto pathpython
py -3.15 -m venv "%ENV_DIR%" || goto error
goto deps
:pathpython
python -c "%CHECK%" >nul 2>&1 || goto runtime
python -m venv "%ENV_DIR%" || goto error
:deps
"%EXE%" -c "%CHECK%" || goto runtime
if exist "%ENV_DIR%\instalado-315-v1" goto run
"%EXE%" -m pip install --upgrade pip || goto error
"%EXE%" -m pip install "torch==2.15.0+%WHEELS%" "torchvision==0.29.2+%WHEELS%" --extra-index-url https://download.pytorch.org/whl/%WHEELS% || goto error
"%EXE%" -m pip install -r requirements.txt || goto error
"%EXE%" -c "import torch, torchvision, cv2, streamlit, ijson, scipy; from transformers import DeformableDetrForObjectDetection; from ultralytics import YOLOE; torchvision.ops.nms(torch.zeros((1,4)),torch.ones(1),0.5); print('Python 3.15 - dependencias verificadas')" || goto error
type nul > "%ENV_DIR%\instalado-315-v1"
:run
if "%~1"=="" goto train_and_open
if /I "%~1"=="app" goto app
if /I "%~1"=="cpu" goto app
if /I "%~1"=="entrenar" goto train
if /I "%~1"=="completo" goto full
"%EXE%" entrenar_detr.py %* || goto error
goto fin
:train_and_open
echo Iniciando un nuevo ajuste de DETR 2 desde los pesos base...
"%EXE%" entrenar_detr.py || goto error
goto app
:train
"%EXE%" entrenar_detr.py || goto error
goto fin
:full
"%EXE%" entrenar_detr.py --modo completo || goto error
:fin
pause
exit /b 0
:app
"%EXE%" -m streamlit run app.py || goto error
exit /b 0
:runtime
echo Se requiere CPython 3.15 para Windows x64. Compruebe py -3.15 --version o python --version.
:error
pause
exit /b 1
