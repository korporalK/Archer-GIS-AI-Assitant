import json
import os
from pathlib import Path
from typing import Dict, List

class SettingsManager:
    def __init__(self):
        self.settings_file = Path.home() / ".gis_agent_settings.json"
        self.default_settings = {
            "workspace": "",
            "watched_directories": [],
            "recent_files": []
        }
        self.settings = self.load_settings()
    
    def load_settings(self) -> Dict:
        if self.settings_file.exists():
            try:
                with open(self.settings_file, 'r') as f:
                    return json.load(f)
            except:
                return self.default_settings.copy()
        return self.default_settings.copy()
    
    def save_settings(self):
        with open(self.settings_file, 'w') as f:
            json.dump(self.settings, f, indent=2)
    
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