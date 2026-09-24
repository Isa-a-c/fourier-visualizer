@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\pythonw.exe" (
    echo Python environment not found. Please create .venv first.
    pause
    exit /b 1
)
start "" ".venv\Scripts\pythonw.exe" "fourier_visualizer\desktop_app.py"
