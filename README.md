# ArcGIS AI Assistant

A powerful GIS assistant application that combines ArcGIS Pro capabilities with AI to help with GIS tasks. This application leverages the ArcGIS Pro Python environment and integrates with Google's Gemini API to provide an intelligent assistant for GIS operations.

## Overview

The ArcGIS AI Assistant helps GIS professionals and analysts by:
- Providing a chat interface to interact with GIS data and tools
- Automating common GIS workflows
- Assisting with spatial analysis tasks
- Managing workspaces and directories
- Integrating with external data sources

## Prerequisites

- ArcGIS Pro 3.x installed (with its native conda environment)
- Python 3.11 (included with ArcGIS Pro)
- ArcPy 3.4 (included with ArcGIS Pro)
- Windows 10/11 operating system
- Google Gemini API key (required)
- Tavily API key (optional, for web search functionality)
- NASA Earthdata credentials (optional, for Landsat data download)

## Environment Setup

This project requires the ArcGIS Pro Python environment. We provide multiple ways to set up your environment:

### Quick Setup (Recommended)

1. Run the provided setup script:
   - Windows: `setup_environment.bat`
   - Linux/macOS: `./setup_environment.sh`

2. Follow the on-screen instructions

For detailed environment setup instructions, see [ENVIRONMENT_SETUP.md](ENVIRONMENT_SETUP.md).

## Configuration

### API Keys Setup

1. Copy the `.env.template` file to `.env` in the `gis_agent_project` directory:
   ```
   cp .env.template .env
   ```

2. Edit the `.env` file and add your API keys:
   ```
   GEMINI_API_KEY=your_gemini_api_key_here
   TAVILY_API_KEY=your_tavily_api_key_here
   EARTHDATA_USER=your_earthdata_username
   EARTHDATA_PASS=your_earthdata_password
   EARTHDATA_TOKEN=your_earthdata_token
   ```

### Workspace Configuration

The application uses a workspace directory for storing and accessing GIS data. By default, it will use the current directory, but you can configure a specific workspace in the settings.

1. The workspace can be set in the `settings.json` file
2. You can also set the workspace through the application's Environment tab
3. Make sure the workspace directory exists and is accessible

## Running the Application

1. Activate your environment (if using a clone):
   ```
   conda activate arcgis_llm
   ```

2. Navigate to the project directory:
   ```
   cd path\to\ArcGIS_AI\gis_agent_project
   ```

3. Run the application:
   ```
   python main.py
   ```

## Features

- **Chat Interface**: Interact with the GIS AI assistant using natural language
- **Environment Management**: Configure workspace and directories
- **API Key Management**: Securely store and manage API keys through the GUI
- **Directory Scanning**: Automatically scan directories for GIS files
- **GIS Tools Integration**: Access to a wide range of ArcGIS tools and functions
- **Data Visualization**: View and analyze GIS data
- **Spatial Analysis**: Perform complex spatial analysis tasks
- **Data Download**: Download external data sources like Landsat imagery

## Settings

The application stores settings in a `settings.json` file located in the `gis_agent_project` directory. This includes:
- Workspace location
- Watched directories
- API keys (securely stored)

You can modify these settings directly through the application's interface.

## Troubleshooting

### API Key Issues
1. Check that your `.env` file is properly formatted and located in the `gis_agent_project` directory
2. You can also enter API keys directly in the Environment tab of the application
3. Ensure you have the necessary permissions for the workspace directory

### Environment Issues
1. Make sure ArcGIS Pro is properly installed
2. Verify that the conda environment has access to the `arcpy` package (version 3.4)
3. Check that all dependencies are installed correctly with the versions specified in requirements.txt
4. If using a cloned environment, ensure it was created from the ArcGIS Pro base environment
5. If you encounter issues with the setup scripts, try the manual setup process
6. For detailed environment troubleshooting, see [ENVIRONMENT_SETUP.md](ENVIRONMENT_SETUP.md)

### Workspace Issues
1. Ensure the workspace directory exists and is accessible
2. Check that you have write permissions for the workspace directory
3. Verify that the workspace path is correctly specified in the settings

## Contributing

Contributions to this project are welcome. Please follow these steps:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 