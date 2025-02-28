# GIS Agent Project

A powerful GIS assistant application that combines ArcGIS capabilities with AI to help with GIS tasks.

## Setup

1. Make sure you have ArcGIS Pro installed with Python environment set up.

2. Install required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. API Keys Setup:
   - Copy the `.env.template` file to `.env` in the `gis_agent_project` directory
   - Fill in your API keys in the `.env` file:
     - `GEMINI_API_KEY`: Required for the application to function
     - `TAVILY_API_KEY`: Optional for web search functionality
     - `EARTHDATA_USER`, `EARTHDATA_PASS`, `EARTHDATA_TOKEN`: Optional for Landsat data download

4. Run the application:
   ```
   python main.py
   ```

## Features

- Chat interface to interact with the GIS AI assistant
- Environment management for workspace and directories
- API key management through the GUI
- Directory scanning for GIS files
- Comprehensive GIS tools integration

## Settings

The application stores settings in a `settings.json` file located in the `gis_agent_project` directory. This includes:
- Workspace location
- Watched directories
- API keys (securely stored)

## Troubleshooting

If you encounter issues with API keys:
1. Check that your `.env` file is properly formatted and located in the `gis_agent_project` directory
2. You can also enter API keys directly in the Environment tab of the application
3. Ensure you have the necessary permissions for the workspace directory 