@echo off
echo Setting up ArcGIS AI Assistant environment...

REM Check if conda is available
where conda >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Error: Conda not found. Please ensure ArcGIS Pro is installed and try again.
    exit /b 1
)

REM Create conda environment from ArcGIS Pro base environment
echo Creating conda environment from ArcGIS Pro base environment...
echo This will clone the arcgispro-py3 environment (with Python 3.11 and ArcPy 3.4)...
call conda create --name arcgis_llm --clone arcgispro-py3 -y
if %ERRORLEVEL% NEQ 0 (
    echo Error: Failed to create conda environment. Please ensure ArcGIS Pro is installed correctly.
    exit /b 1
)

REM Activate the new environment
echo Activating arcgis_llm environment...
call conda activate arcgis_llm
if %ERRORLEVEL% NEQ 0 (
    echo Error: Failed to activate conda environment.
    exit /b 1
)

REM Install conda requirements
echo Installing conda requirements...
call conda install --file conda-requirements.txt -c esri -c conda-forge -y
if %ERRORLEVEL% NEQ 0 (
    echo Warning: Some conda packages may not have installed correctly.
    echo Continuing with pip installations...
)

REM Install pip requirements
echo Installing pip requirements...
pip install -r requirements.txt
if %ERRORLEVEL% NEQ 0 (
    echo Error: Failed to install pip requirements.
    exit /b 1
)

REM Verify the environment
echo Verifying environment...
python -c "import sys, arcpy, fiona, rasterio, langchain, google.generativeai, aiohttp, aiofiles, requests; print(f'Python version: {sys.version}'); print(f'ArcPy version: {arcpy.__version__}'); print('All required libraries successfully imported')"

echo Environment setup complete!
echo To activate the environment, run: conda activate arcgis_llm
echo To run the application, navigate to the project directory and run: python main.py

pause 