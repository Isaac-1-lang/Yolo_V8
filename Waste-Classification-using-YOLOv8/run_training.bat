@echo off
echo ===============================================
echo Waste Classification - Training Setup
echo ===============================================
echo.

cd /d "%~dp0"

echo Checking Python installation...
python --version
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    pause
    exit /b 1
)

echo.
echo Starting training process...
echo This may take several hours depending on your hardware.
echo.

python setup_and_train.py

echo.
echo Training process completed!
echo Check the results in the runs/detect/waste_classification_new/ directory
echo.
pause

