import json
import os
from pathlib import Path
from typing import Dict, List

class SettingsManager:
    def __init__(self):
        # Get the directory where this script is located
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Set settings file to be in the same directory
        self.settings_file = os.path.join(current_dir, "settings.json")
        
        self.default_settings = {
            "workspace": "",
            "watched_directories": [],
            "recent_files": [],
            "api_keys": {
                "gemini_api_key": "",
                "tavily_api_key": "",
                "earthdata_user": "",
                "earthdata_pass": "",
                "earthdata_token": ""
            }
        }
        self.settings = self.load_settings()
        
        # Load API keys from .env file if they exist and not already in settings
        self._load_api_keys_from_env()
    
    def _load_api_keys_from_env(self):
        """Load API keys from .env file if they exist and settings are empty"""
        try:
            # Check if we have any API keys in the settings
            if not any(self.settings["api_keys"].values()):
                # Try to load from .env file in the current directory
                current_dir = os.path.dirname(os.path.abspath(__file__))
                env_path = os.path.join(current_dir, ".env")
                
                if os.path.exists(env_path):
                    with open(env_path, 'r') as f:
                        for line in f:
                            if '=' in line:
                                key, value = line.strip().split('=', 1)
                                key = key.strip()
                                value = value.strip().strip('"\'')
                                
                                if key == "GEMINI_API_KEY" and not self.settings["api_keys"]["gemini_api_key"]:
                                    self.settings["api_keys"]["gemini_api_key"] = value
                                elif key == "TAVILY_API_KEY" and not self.settings["api_keys"]["tavily_api_key"]:
                                    self.settings["api_keys"]["tavily_api_key"] = value
                                elif key == "EARTHDATA_USER" and not self.settings["api_keys"]["earthdata_user"]:
                                    self.settings["api_keys"]["earthdata_user"] = value
                                elif key == "EARTHDATA_PASS" and not self.settings["api_keys"]["earthdata_pass"]:
                                    self.settings["api_keys"]["earthdata_pass"] = value
                                elif key == "EARTHDATA_TOKEN" and not self.settings["api_keys"]["earthdata_token"]:
                                    self.settings["api_keys"]["earthdata_token"] = value
                    
                    # Save the loaded API keys
                    self.save_settings()
                    print(f"Loaded API keys from {env_path}")
        except Exception as e:
            print(f"Error loading API keys from .env: {str(e)}")
    
    def load_settings(self) -> Dict:
        if os.path.exists(self.settings_file):
            try:
                with open(self.settings_file, 'r') as f:
                    settings = json.load(f)
                    
                    # Ensure all required keys exist (for backward compatibility)
                    if "api_keys" not in settings:
                        settings["api_keys"] = self.default_settings["api_keys"]
                    
                    return settings
            except Exception as e:
                print(f"Error loading settings: {str(e)}")
                return self.default_settings.copy()
        return self.default_settings.copy()
    
    def save_settings(self):
        try:
            with open(self.settings_file, 'w') as f:
                json.dump(self.settings, f, indent=2)
        except Exception as e:
            print(f"Error saving settings: {str(e)}")
    
    def set_workspace(self, workspace: str):
        self.settings["workspace"] = workspace
        self.save_settings()
    
    def add_directory(self, directory: str):
        if directory not in self.settings["watched_directories"]:
            self.settings["watched_directories"].append(directory)
            self.save_settings()
    
    def remove_directory(self, directory: str):
        if directory in self.settings["watched_directories"]:
            self.settings["watched_directories"].remove(directory)
            self.save_settings()
    
    def add_recent_file(self, file_path: str):
        if file_path in self.settings["recent_files"]:
            self.settings["recent_files"].remove(file_path)
        self.settings["recent_files"].insert(0, file_path)
        if len(self.settings["recent_files"]) > 10:  # Keep only last 10 files
            self.settings["recent_files"] = self.settings["recent_files"][:10]
        self.save_settings()
    
    def set_api_key(self, key_name: str, value: str):
        """Set an API key in the settings
        
        Args:
            key_name: The name of the API key (gemini_api_key, tavily_api_key, etc.)
            value: The API key value
        """
        if key_name in self.settings["api_keys"]:
            self.settings["api_keys"][key_name] = value
            self.save_settings()
    
    def get_api_key(self, key_name: str) -> str:
        """Get an API key from the settings
        
        Args:
            key_name: The name of the API key
            
        Returns:
            The API key value or empty string if not found
        """
        return self.settings["api_keys"].get(key_name, "")
    
    def get_settings_file_location(self):
        """
        Returns the absolute path to the settings file.
        
        Returns:
            str: The absolute path to the settings file
        """
        return os.path.abspath(self.settings_file) 