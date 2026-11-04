@echo off
REM Creates .\.venv next to this script, installs everything, and registers a Jupyter kernel.
REM Usage (Windows, from Command Prompt in the project folder):   setup_env.bat
REM Optional: set CUDA_TAG=cu128 before running (RTX 50-series). Default cu126.
setlocal
cd /d "%~dp0"
if "%CUDA_TAG%"=="" set CUDA_TAG=cu126
set KERNEL_NAME=cmpt819-deepfake

where py >nul 2>nul && (set PY=py -3) || (set PY=python)
echo ^>^> Python version:
%PY% --version
%PY% -m venv .venv || goto :err
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip wheel setuptools || goto :err

where nvidia-smi >nul 2>nul
if %errorlevel%==0 (
    echo ^>^> NVIDIA GPU found - installing CUDA build %CUDA_TAG%
    pip install torch torchvision --index-url https://download.pytorch.org/whl/%CUDA_TAG% || goto :err
) else (
    echo ^>^> No GPU - CPU build
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu || goto :err
)

pip install -r requirements.txt || goto :err
pip install --no-deps facenet-pytorch==2.6.0 || goto :err
python -m ipykernel install --user --name %KERNEL_NAME% --display-name "Python (CMPT819 Deepfake)" || goto :err

python -c "import torch,timm,facenet_pytorch,cv2; from facenet_pytorch import MTCNN; MTCNN(device='cpu'); print('torch',torch.__version__,'| CUDA:',torch.cuda.is_available()); print('Environment OK')" || goto :err
echo ^>^> Done. In VS Code / Jupyter pick the kernel: Python (CMPT819 Deepfake)
exit /b 0

:err
echo !! Setup failed - see the message above.
exit /b 1
