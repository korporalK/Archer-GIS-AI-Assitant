# Detailed Environment Setup Guide

This document provides detailed instructions for setting up the Python environment required for the ArcGIS AI Assistant.

## Understanding the Environment Requirements

The ArcGIS AI Assistant requires:

1. **ArcGIS Pro Python Environment**: This provides access to the `arcpy` module (version 3.4) and other ArcGIS-specific packages.
2. **Python 3.11**: The application is built and tested with Python 3.11, which is included with ArcGIS Pro.
3. **GIS Libraries**: The application uses `fiona` and `rasterio` for additional GIS functionality (installed via conda).
4. **LangChain and Google Generative AI**: These provide the AI capabilities of the application (installed via pip).
5. **Asynchronous Libraries**: `aiohttp`, `aiofiles`, and `asyncio` for asynchronous operations (installed via pip).
6. **Standard Library Modules**: Various Python standard library modules like `tkinter`, `traceback`, `inspect`, etc.

## Environment Setup Options

### Option 1: Using the Automated Setup Script (Recommended)

We provide a script that automates the environment creation process:

1. Open a command prompt
2. Navigate to the project directory:
   ```
   cd path\to\ArcGIS_AI\gis_agent_project
   ```
3. Run the setup script:
   ```
   setup_environment.bat
   ```

The script will:
- Check if conda is available
- Create a clone of the ArcGIS Pro Python environment named `arcgis_llm`
- Install required conda packages from the `conda-requirements.txt` file
- Install required pip packages from the `requirements.txt` file
- Verify that all required libraries are correctly installed

### Option 2: Manual Setup

If you prefer to set up the environment manually:

#### Step 1: Clone the ArcGIS Pro Environment

1. Open the ArcGIS Pro Python Command Prompt
2. Create a clone of the ArcGIS Pro environment:
   ```
   conda create --name arcgis_llm --clone arcgispro-py3
   ```
3. Activate the new environment:
   ```
   conda activate arcgis_llm
   ```

#### Step 2: Install Conda Packages

Install the required conda packages with the esri channel taking precedence:

```
conda install --file conda-requirements.txt -c esri -c conda-forge
```

This will install:
- fiona (version 1.9.5)
- rasterio (version 1.3.9)
- Other conda dependencies

#### Step 3: Install Pip Packages

Install the required pip packages:

```
pip install -r requirements.txt
```

This will install:
- langchain and related packages
- google-generativeai
- python-dotenv
- aiohttp and aiofiles
- requests
- asyncio and nest-asyncio
- Other pip dependencies

### Option 3: Using the ArcGIS Pro Environment Directly

If you prefer not to create a separate environment:

1. Open ArcGIS Pro
2. Click on the Project tab
3. Click on Python → Python Command Prompt
4. Install the required packages:
   ```
   conda install -c conda-forge fiona=1.9.5 rasterio=1.3.9
   pip install -r requirements.txt
   ```

**Note**: This approach may modify your ArcGIS Pro Python environment, which could potentially affect other ArcGIS Pro functionality.

### Option 4: Rebuilding outside of ArcGIS

1. Open ArcGIS Python Command Prompt
2. Navigate to the env_confs directory
3. Build the arcgis conda environment:
   ```
   conda env create -f environment_full.yml
   ```
4. Add the Environment to ArcGIS Pro: To use the new environment within ArcGIS Pro:
   - Open ArcGIS Pro.
   - Navigate to Project > Python > Manage Environments.
   - Click Add and browse to the location of the new environment.
   
**Note**: I am have not tested building the environment on my end, I advise you to first try the environment_full, if you get into issues then try environment_nobuild and environment_history.


## Troubleshooting Environment Issues

### Common Issues and Solutions

1. **Conda Not Found**:
   - Ensure ArcGIS Pro is installed correctly
   - Make sure you're using the ArcGIS Pro Python Command Prompt

2. **Clone Creation Fails**:
   - Verify you have sufficient disk space
   - Ensure you have administrator privileges
   - Check that the ArcGIS Pro environment exists and is not corrupted

3. **Package Installation Fails**:
   - Check your internet connection
   - Ensure you have the necessary permissions
   - Try installing packages one by one to identify problematic packages

4. **arcpy Not Found**:
   - Ensure you're using the cloned ArcGIS Pro environment
   - Verify that ArcGIS Pro is installed correctly
   - Check that the environment has access to the ArcGIS Pro installation directory

5. **GIS Libraries Issues**:
   - If fiona or rasterio fail to install, try installing them separately:
     ```
     conda install -c conda-forge fiona=1.9.5
     conda install -c conda-forge rasterio=1.3.9
     ```
   - On Windows, you may need to install the Microsoft Visual C++ Redistributable

6. **Asynchronous Libraries Issues**:
   - If you encounter issues with aiohttp or aiofiles, try installing them separately:
     ```
     pip install aiohttp==3.9.1 aiofiles==23.2.1
     ```

7. **Version Mismatch**:
   - If you see warnings about Python or ArcPy versions not matching, ensure you're using ArcGIS Pro's Python environment
   - The application requires Python 3.11 and ArcPy 3.4

### Verifying Your Environment

To verify that your environment is set up correctly:

1. Activate your environment:
   ```
   conda activate arcgis_llm
   ```

2. Check that arcpy is available with the correct version:
   ```
   python -c "import arcpy; print(arcpy.__version__)"
   ```
   Expected output: `3.4`

3. Check that Python is the correct version:
   ```
   python -c "import sys; print(sys.version)"
   ```
   Expected output: Python 3.11.x

4. Check that GIS libraries are available:
   ```
   python -c "import fiona, rasterio; print('GIS libraries imported successfully')"
   ```

5. Check that AI libraries are available:
   ```
   python -c "import langchain, google.generativeai; print('AI libraries imported successfully')"
   ```

6. Check that asynchronous libraries are available:
   ```
   python -c "import aiohttp, aiofiles, asyncio; print('Async libraries imported successfully')"
   ```

## Channel Priority for Conda Packages

When installing conda packages, we prioritize channels in the following order:
1. **esri**: Contains ArcGIS-specific packages
2. **conda-forge**: Contains many scientific packages
3. **defaults**: The default Anaconda channel

This priority ensures that ArcGIS-specific packages are installed from the esri channel, which is important for compatibility with ArcGIS Pro.

## Package Management Best Practices

1. **Use the Cloned Environment**: This isolates your project dependencies from the main ArcGIS Pro environment.
2. **Keep Requirements Files Updated**: When adding new dependencies, update the requirements files.
3. **Specify Version Numbers**: This ensures reproducibility across different installations.
4. **Separate Conda and Pip Packages**: Install packages available through conda using conda, and use pip only for packages not available through conda.

## API Key Configuration

The ArcGIS AI Assistant requires API keys for various functions. These should be set up before running the application.

### Required API Keys

1. **Google Gemini API Key**
   - Sign up at https://ai.google.dev/
   - Create a new API key in your Google AI Studio dashboard
   - Rate limits may apply based on your account type

### Optional API Keys

2. **Tavily API Key** (for web search capabilities)
   - Sign up at https://tavily.com/
   - Follow the instructions to create an API key
   - Free tier provides a limited number of searches per day

3. **NASA Earthdata Credentials** (for Landsat downloads)
   - Register at https://urs.earthdata.nasa.gov/
   - Create an Earthdata Login account
   - After login, go to "Applications → Authorized Apps"
   - Generate a new application token for use with the ArcGIS AI Assistant

### Setting Environment Variables

The application will look for these API keys in a `.env` file in the project directory or in the settings.json file. You can set these up by:

1. Creating a `.env` file with the following structure:
   ```
   GEMINI_API_KEY=your_key_here
   TAVILY_API_KEY=your_key_here
   EARTHDATA_USER=your_username
   EARTHDATA_PASS=your_password
   EARTHDATA_TOKEN=your_token
   ```

2. Or by using the Environment tab in the application GUI to enter your API keys 