@echo off
REM Site Sentinel - start the FastAPI backend (CPU-only)
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Creating virtual environment...
  python -m venv .venv
  call .venv\Scripts\activate.bat
  python -m pip install --upgrade pip
  pip install torch==2.14.1+cpu torchvision==0.29.1+cpu --index-url https://download.pytorch.org/whl/cpu
  pip install -r requirements.txt
)
echo Starting Site Sentinel API on http://127.0.0.1:8100
.venv\Scripts\python.exe -m uvicorn app.api:app --host 127.0.0.1 --port 8100
