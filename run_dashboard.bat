@echo off
REM Site Sentinel - start the dark Streamlit dashboard
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Run run_api.bat first to create the virtual environment.
  pause
  exit /b 1
)
echo Starting Site Sentinel dashboard on http://127.0.0.1:8501
.venv\Scripts\python.exe -m streamlit run app\dashboard.py --server.port 8501 --server.headless true
