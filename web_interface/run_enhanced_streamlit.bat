@echo off
echo ========================================
echo  Enhanced Skin Cancer Detection System
echo  with Lesion Detection v2.0
echo ========================================
echo.
echo Starting Streamlit application...
echo.
echo The app will open in your browser at:
echo   http://localhost:8502
echo.
echo Press CTRL+C to stop the server
echo ========================================
echo.

cd /d "%~dp0"
streamlit run streamlit_web_app.py --server.port 8502 --server.headless false

pause
