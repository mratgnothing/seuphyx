@echo off
setlocal EnableExtensions

set "APP_DIR=%~dp0"
set "APP_DIR=%APP_DIR:~0,-1%"
set "PYTHON=%APP_DIR%\.venv\Scripts\python.exe"
set "APP=%APP_DIR%\streamlit_app.py"
set "PORT=8501"
set "URL=http://127.0.0.1:%PORT%/"

title SEUPhyX Local Website :%PORT%
cd /d "%APP_DIR%"

echo.
echo ============================================================
echo  SEUPhyX local website
echo ============================================================
echo.
echo  Project: %APP_DIR%
echo  URL:     %URL%
echo.

if not exist "%PYTHON%" (
    echo [ERROR] Python environment not found:
    echo         %PYTHON%
    echo.
    echo The .venv directory is required for local launch.
    pause
    exit /b 1
)

if not exist "%APP%" (
    echo [ERROR] Streamlit entrypoint not found:
    echo         %APP%
    pause
    exit /b 1
)

echo [1/3] Checking port %PORT%...
for /f "tokens=5" %%P in ('netstat -ano ^| findstr /R /C:":%PORT% .*LISTENING"') do (
    echo       Stopping old server process %%P...
    taskkill /PID %%P /F >nul 2>nul
)

set "PYTHONPATH=%APP_DIR%\src"
set "STREAMLIT_SERVER_HEADLESS=true"
set "STREAMLIT_BROWSER_GATHER_USAGE_STATS=false"

echo [2/3] Opening browser...
start "" "%URL%"

echo [3/3] Starting Streamlit server...
echo.
echo Keep this window open while using the website.
echo Close this window, or press Ctrl+C, to stop the local server.
echo.

"%PYTHON%" -m streamlit run "%APP%" --server.port %PORT% --server.address 127.0.0.1 --server.headless true --browser.gatherUsageStats false

echo.
echo [SEUPhyX] Server stopped.
pause
