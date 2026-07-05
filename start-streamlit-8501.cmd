@echo off
cd /d "%~dp0"
set "PYTHONPATH=%CD%\src"
set "STREAMLIT_SERVER_HEADLESS=true"
set "STREAMLIT_BROWSER_GATHER_USAGE_STATS=false"
".venv\Scripts\python.exe" -m streamlit run "%CD%\src\seuphyx\core\oil\app.py" --server.port 8501 --server.headless true --browser.gatherUsageStats false --server.enableCORS false --server.enableXsrfProtection false --server.baseUrlPath /seuphyx/oil 1>>"%CD%\streamlit-8501.out.log" 2>>"%CD%\streamlit-8501.err.log"
