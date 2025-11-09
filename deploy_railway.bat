@echo off
echo ========================================
echo 🚀 Railway Deployment Script (Windows)
echo ========================================
echo.

REM Check if Railway CLI is installed
where railway >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo 📦 Installing Railway CLI...
    npm i -g @railway/cli
) else (
    echo ✅ Railway CLI already installed
)

echo.
echo 🔐 Logging into Railway...
call railway login

echo.
echo 📂 Initializing Railway project...
call railway init

echo.
echo ⚙️ Setting environment variables...
call railway variables set PYTHON_VERSION=3.11.0
call railway variables set MODEL_PATH=./models

echo.
echo 🚀 Deploying to Railway...
call railway up

echo.
echo ========================================
echo ✅ Deployment Complete!
echo ========================================
echo.
echo 🌐 Your app will be available at the Railway-provided URL
echo.
echo 📊 Check status: railway status
echo 📝 View logs: railway logs
echo.
pause
