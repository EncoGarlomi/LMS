@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Python interpreter not found in .venv.
    echo Run the project setup first.
    pause
    exit /b 1
)

".venv\Scripts\python.exe" -m app.bot
if errorlevel 1 (
    echo Bot exited with an error.
    pause
)

endlocal
