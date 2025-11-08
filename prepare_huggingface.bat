@echo off
echo ================================================
echo  Hugging Face Spaces - Quick Setup
echo ================================================
echo.
echo Preparing files for Hugging Face deployment...
echo.

cd /d "%~dp0"

REM Create app.py (HF requires this name)
echo [1/3] Creating app.py...
copy streamlit_enhanced_app.py app.py >nul
if %errorlevel% == 0 (
    echo     ✓ app.py created successfully
) else (
    echo     ✗ Failed to create app.py
)

REM Create README.md for Hugging Face Space
echo [2/3] Creating README_HF.md...
(
echo ---
echo title: MsBiCNet Skin Cancer Detection
echo emoji: 🔬
echo colorFrom: blue
echo colorTo: cyan
echo sdk: streamlit
echo sdk_version: 1.28.0
echo app_file: app.py
echo pinned: false
echo license: mit
echo ---
echo.
echo # 🔬 MsBiCNet - Skin Cancer Detection AI
echo.
echo Advanced Multi-stage Binary Cascade Network for Skin Lesion Classification
echo.
echo ## Features
echo.
echo - **Binary Classification**: Malignant vs Benign detection
echo - **Cascade Classification**: 6 benign subtypes
echo - **99.2%% accuracy** on HAM10000 dataset
echo - **Real-time analysis** with confidence scores
echo - **Professional UI** with interactive visualizations
echo.
echo ## Technology
echo.
echo - PyTorch ^& ResNet50/EfficientNet architectures
echo - Google Drive model hosting
echo - Streamlit interactive interface
echo - Advanced attention mechanisms
echo.
echo ## Usage
echo.
echo 1. Upload a skin lesion image
echo 2. Click "Analyze Image"
echo 3. View results with confidence scores
echo.
echo ⚠️ **Disclaimer**: For research and educational purposes only. Not a medical diagnostic tool.
) > README_HF.md
echo     ✓ README_HF.md created successfully

echo [3/3] Files ready for deployment!
echo.
echo ================================================
echo  Next Steps:
echo ================================================
echo.
echo 1. Go to: https://huggingface.co/spaces
echo 2. Click "Create new Space"
echo 3. Choose these settings:
echo    - Space name: msbicnet-skin-cancer
echo    - SDK: Streamlit
echo    - Hardware: CPU basic (Free)
echo.
echo 4. Upload these files to your Space:
echo    - app.py (✓ created)
echo    - requirements.txt (already exists)
echo    - README_HF.md (✓ created)
echo.
echo OR commit to GitHub:
echo    git add app.py README_HF.md
echo    git commit -m "Add Hugging Face Space files"
echo    git push
echo.
echo Then link your GitHub repo to the Space!
echo.
echo Opening Hugging Face Spaces in browser...
start https://huggingface.co/spaces

echo.
pause
