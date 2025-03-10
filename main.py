import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain_community.callbacks.manager import get_openai_callback
import json
import queue
import threading
import tkinter as tk
from tkinter import scrolledtext, filedialog, ttk
import time
import inspect
import arcpy
import traceback
import io  # Import io module
import contextlib # Import contextlib
import re
from settings_manager import SettingsManager
import tkinter.messagebox


# Import tools properly
from tools import *

# Function to get all tools from the tools module
def get_all_tools():
    """Get all tool functions from the tools module.
    
    Returns:
        A list of all tools defined in the tools module.
    """
    import tools
    import inspect
    
    # Get all members from the tools module that are decorated with @tool
    tool_functions = []
    for name in dir(tools):
        obj = getattr(tools, name)
        # Check if it's a tool function (has 'name', 'description', and 'func' attributes)
        if hasattr(obj, 'name') and hasattr(obj, 'description') and hasattr(obj, 'func'):
            tool_functions.append(obj)
    
    return tool_functions

# New Directory Management Classes
class DirectoryManager:
    """Manages directory scanning and caching for GIS files."""
    def __init__(self, settings_manager, gis_agent):
        self.settings_manager = settings_manager
        self.gis_agent = gis_agent
        self.scan_cache = {}  # Cache scan results by directory
        self.scan_lock = threading.Lock()  # Thread safety for scanning
        self.scan_in_progress = False
        self.last_scan_time = {}  # Track when each directory was last scanned
    
    def scan_directory(self, directory, force_refresh=False):
        """Scan a single directory and cache the results"""
        with self.scan_lock:
            # Check if we have a recent scan (within 5 minutes) and not forcing refresh
            current_time = time.time()
            if (not force_refresh and 
                directory in self.scan_cache and 
                directory in self.last_scan_time and
                current_time - self.last_scan_time[directory] < 300):  # 5 minutes
                return self.scan_cache[directory]
            
            try:
                print(f"Scanning directory: {directory}")
                scan_result_str = scan_external_directory_for_gis_files.invoke(directory)
                try:
                    scan_result = json.loads(scan_result_str)
                    self.scan_cache[directory] = scan_result
                    self.last_scan_time[directory] = current_time
                    
                    # Log the results
                    vector_count = len(scan_result.get("vector_files", []))
                    raster_count = len(scan_result.get("raster_files", []))
                    print(f"Found {vector_count} vector files and {raster_count} raster files in {directory}")
                    
                    return scan_result
                except json.JSONDecodeError as json_err:
                    print(f"Error parsing scan result for {directory}: {str(json_err)}")
                    error_result = {"error": f"Invalid JSON response: {str(json_err)}", "vector_files": [], "raster_files": []}
                    self.scan_cache[directory] = error_result
                    self.last_scan_time[directory] = current_time
                    return error_result
            except Exception as e:
                print(f"Error scanning directory {directory}: {str(e)}")
                error_result = {"error": str(e), "vector_files": [], "raster_files": []}
                self.scan_cache[directory] = error_result
                self.last_scan_time[directory] = current_time
                return error_result
    
    def scan_all_directories(self, callback=None, force_refresh=False):
        """Scan all watched directories with optional progress callback"""
        if self.scan_in_progress:
            return False
        
        self.scan_in_progress = True
        watched_directories = self.settings_manager.settings["watched_directories"]
        results = {}
        
        try:
            total_dirs = len(watched_directories)
            for i, directory in enumerate(watched_directories):
                if callback:
                    callback(f"Scanning directory {i+1}/{total_dirs}: {directory}", (i / total_dirs) * 100)
                
                results[directory] = self.scan_directory(directory, force_refresh)
            
            # Update the GIS agent's environment info with our results
            self.update_gis_agent_cache(results)
            
            if callback:
                callback("Scan completed", 100)
            
            return results
        except Exception as e:
            print(f"Error in scan_all_directories: {str(e)}")
            traceback.print_exc()
            if callback:
                callback(f"Error scanning directories: {str(e)}", -1)
            return {}
        finally:
            self.scan_in_progress = False
    
    def update_gis_agent_cache(self, scan_results):
        """Update the GIS agent's environment info with our scan results"""
        # Only update the external_directories part of the environment info
        if self.gis_agent._environment_info:
            self.gis_agent._environment_info["external_directories"] = scan_results
        else:
            # If no environment info exists yet, create a minimal version
            self.gis_agent._environment_info = {
                "workspace": self.gis_agent.workspace,
                "workspace_inventory": {},
                "external_directories": scan_results
            }


class TreeViewManager:
    """Manages updates to the tree view for displaying GIS files."""
    def __init__(self, tree_view, response_callback=None):
        self.tree_view = tree_view
        self.response_callback = response_callback
        self.update_lock = threading.Lock()
    
    def clear_tree(self):
        """Clear all items from the tree view"""
        self.tree_view.delete(*self.tree_view.get_children())
    
    def update_tree_from_scan_results(self, scan_results):
        """Update the tree view with scan results"""
        with self.update_lock:
            self.clear_tree()
            
            # Track statistics and nodes
            total_vector_files = 0
            total_raster_files = 0
            dir_nodes = []
            vector_nodes = []
            raster_nodes = []
            
            # Process each directory
            for directory, result in scan_results.items():
                # Check for errors
                if "error" in result:
                    if self.response_callback:
                        self.response_callback(f"Error scanning directory {directory}: {result['error']}")
                    dir_node = self.tree_view.insert("", "end", text=directory)
                    self.tree_view.insert(dir_node, "end", text=f"Error: {result['error']}")
                    continue
                
                # Count files
                vector_files = result.get("vector_files", [])
                raster_files = result.get("raster_files", [])
                total_vector_files += len(vector_files)
                total_raster_files += len(raster_files)
                
                # Add directory node
                dir_node = self.tree_view.insert("", "end", 
                                               text=f"{directory} ({len(vector_files) + len(raster_files)} files)")
                dir_nodes.append(dir_node)
                
                # Add vector files
                if vector_files:
                    vector_node = self.tree_view.insert(dir_node, "end", 
                                                      text=f"Vector Files ({len(vector_files)})")
                    vector_nodes.append(vector_node)
                    for file in vector_files:
                        values = (
                            file["name"],
                            file["type"],
                            file["path"],
                            file.get("driver", "-"),
                            file.get("crs", "-"),
                            str(file.get("layer_count", "-")),
                            "-",  # Dimensions (raster-specific)
                            "-",  # Bands (raster-specific)
                        )
                        self.tree_view.insert(vector_node, "end", text="", values=values)
                
                # Add raster files
                if raster_files:
                    raster_node = self.tree_view.insert(dir_node, "end", 
                                                      text=f"Raster Files ({len(raster_files)})")
                    raster_nodes.append(raster_node)
                    for file in raster_files:
                        dimensions = f"{file.get('dimensions', [0, 0])[0]}x{file.get('dimensions', [0, 0])[1]}"
                        values = (
                            file["name"],
                            file["type"],
                            file["path"],
                            str(file.get("driver", "-")),
                            file.get("crs", "-"),
                            "-",  # Layer_count (vector-specific)
                            dimensions,
                            str(file.get("bands", "-")),
                        )
                        self.tree_view.insert(raster_node, "end", text="", values=values)
            
            # Expand all nodes
            for node in dir_nodes:
                self.tree_view.item(node, open=True)
            for node in vector_nodes:
                self.tree_view.item(node, open=True)
            for node in raster_nodes:
                self.tree_view.item(node, open=True)
            
            # Return statistics
            return {
                "total_directories": len(scan_results),
                "total_vector_files": total_vector_files,
                "total_raster_files": total_raster_files
            }


class ScanProgressIndicator:
    """Displays a progress window for directory scanning operations."""
    def __init__(self, parent, response_callback=None):
        self.parent = parent
        self.response_callback = response_callback
        self.progress_window = None
        self.progress_bar = None
        self.status_label = None
    
    def show(self):
        """Show the progress window"""
        if self.progress_window:
            return
        
        self.progress_window = tk.Toplevel(self.parent)
        self.progress_window.title("Scanning Directories")
        self.progress_window.geometry("400x100")
        self.progress_window.transient(self.parent)
        self.progress_window.grab_set()
        
        # Status label
        self.status_label = ttk.Label(self.progress_window, text="Initializing scan...")
        self.status_label.pack(pady=5)
        
        # Progress bar
        self.progress_bar = ttk.Progressbar(self.progress_window, orient="horizontal", 
                                           length=350, mode="determinate")
        self.progress_bar.pack(pady=10, padx=20)
    
    def update(self, message, progress):
        """Update progress indicator"""
        if not self.progress_window:
            return
        
        if progress < 0:  # Error state
            self.status_label.config(text=message, foreground="red")
        else:
            self.status_label.config(text=message, foreground="black")
            self.progress_bar["value"] = progress
        
        # Also update response area if callback provided
        if self.response_callback:
            self.response_callback(message)
        
        # Force update of the window
        self.progress_window.update_idletasks()
    
    def close(self):
        """Close the progress window"""
        if self.progress_window:
            self.progress_window.destroy()
            self.progress_window = None
            self.progress_bar = None
            self.status_label = None

# Environment Setup
def setup_environment():
    """
    Set up the environment for the GIS Agent application.
    
    This function loads environment variables from a .env file and checks for the required API key.
    It also verifies that the ArcPy license is available.
    
    Returns:
        str: The Gemini API key if found
        
    Raises:
        ValueError: If the API key is not found or if ArcPy license is invalid
    """
    try:
        # Get the directory where this script is located
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Path to the .env file in the same directory as the script
        env_path = os.path.join(current_dir, ".env")
        
        # Load environment variables from .env file
        load_dotenv(env_path)
        
        # Check if API key is set
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set. Please set it in the .env file.")
        
        arcpy.env.overwriteOutput = True

        # Check ArcPy license
        if arcpy.ProductInfo() == "NotInitialized":
            raise ValueError("ArcPy license not available. Please check your license.")
            
        return api_key
    except Exception as e:
        raise ValueError(f"Error setting up environment: {str(e)}")

# Prompts
PLANNER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful GIS planning assistant. Your task is to develop a step-by-step plan to solve a GIS problem using ONLY the provided tools.
You must output a JSON array of steps. Each step must be an object with:
  - tool: The exact name of one of the available tools (no other functions allowed).
  - input: A dictionary of input parameters exactly matching the tool's required parameters.
  - description: A clear explanation of why this tool is used in this step.

Available tools and their descriptions:
{tools}
    
Current workspace: {workspace}
Workspace inventory files: {inventory}
External files available: {external_files}

Important Environment Information:
- You have access to files both in the workspace geodatabase and in external directories.
- When planning operations, consider both workspace data and external files.
- If needed, you can import external files into the workspace using appropriate tools.
- Always check if required data already exists before planning data creation or import steps.

Process:
1. Review the user request, workspace inventory, and available external files.
2. Consider whether needed data already exists in either the workspace or external directories.
3. Plan any necessary data import or conversion steps.
4. Identify the necessary tools and parameters required to address the user's request.
5. Construct a logical sequence of steps that utilize the available tools effectively.
6. Use the workspace inventory and external files provided in the context to plan the steps.
7. DO NOT ASSUME FIELD NAME, ALWAYS CHECK USING 'list_fields' tool and provide a placeholder.
8. DO NOT ASSUME FILE NAME, ALWAYS CHECK in the directories and the workspace inventory using "scan_workspace_directory_for_gis_files" and "scan_external_directory_for_gis_files" tool and provide a placeholder "file name to be decided by executor based on directories tool output" in the meanwhile.
9. DO NOT ASSUME that only GIS file directories have the required files ad the workspace inventory is empty, the workspace may also have the required files, use "scan_workspace_directory_for_gis_files" tool to check the workspace inventory as well.
10. DO NOT scan the directories or workspace again if you already have the information of the external files in your context.
11. If user request can be fulfilled using available files, DO NOT PLAN any additional steps of data download. Always use the available files if they already exist.
12. The correct Index formula must be used for the analysis based on the user request. For example, Do not use NDVI if MNDWI is required by the user request.
Do not include any markdown formatting or code blocks; output ONLY the JSON array.

Example format:
[
    {{
        "tool": "scan_workspace_directory_for_gis_files",
        "input": {{"workspace": "D:\\masters_project\\ArcGIS_AI\\Default.gdb"}},
        "description": "List all contents of the current workspace."
    }}
]

    
Output ONLY the JSON array.
"""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

VERIFIER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a GIS plan verifier with a chain-of-thought reasoning process. Your task is to verify a GIS plan for errors with detailed internal reasoning.

Tool Descriptions:
{tools}

Workspace Inventory Files:
{inventory}

External Files:
{external_files}

Input:
- user_request: The original GIS task.
- plan: A JSON array of steps, where each step includes 'tool', 'input', and 'description'.
- tool_descriptions: A list of available tools and their descriptions.
- workspace_inventory: The current inventory of the GIS workspace.
- external_files: Available GIS files in external directories.

Additional Verification Checks:
- Verify that input files referenced in the plan exist either in the workspace or external directories.
- Ensure proper data import steps are included when using external files.
- Check that file formats are compatible with the intended operations.
- Verify spatial reference consistency across all data sources.

Process:
1. Review each step to ensure:
   - Each 'tool' is valid and available.
   - Each 'input' contains all required parameters according to the tool's description.
   - The sequence of steps is logical and coherent in a GIS context.
   - Outputs from previous steps are appropriately utilized in later steps.
   - There are no missing steps essential to achieving the overall goal.
   - Potential data issues (e.g., incorrect spatial reference, missing fields) are accounted for.
   - Redundancies are minimized; for instance, avoid re-downloading data that already exists.
   - Confirm that file formats are compatible with the intended tools.
   - Verify that all datasets use the same spatial reference system or are properly reprojected.
   - Ensure that all required attribute fields exist and are correctly formatted.
   - Check for consistency in field naming conventions and units.
   - Validate that attribute values are within expected ranges and data types (e.g., numeric, string) match tool requirements.
   - Properly handle NoData values, ensuring they are not misinterpreted during processing.
   - When resampling or reclassifying, ensure that the chosen method (nearest neighbor, bilinear, cubic, etc.) is suitable for the data type.
   - Double-check that the selected tool is appropriate for the data type (vector vs. raster) and analysis goal.
   - Ensure that the sequence of steps logically builds upon previous outputs (e.g., using field listings to inform selection criteria).
   - DO NOT ASSUME FIELD NAME, ALWAYS CHECK USING 'list_fields' tool and provide a placeholder "field name to be decided by executor based on list_fields output" in the meanwhile.
   - DO NOT ASSUME FILE NAME, ALWAYS CHECK in the directories and the workspace inventory using "scan_workspace_directory_for_gis_files" and "scan_external_directory_for_gis_files" tool and provide a placeholder "file name to be decided by executor based on directories tool output" in the meanwhile.
   - If plan does not include "scan_workspace_directory_for_gis_files" tool, DO NOT ASSUME that only gis file directories have the required files, the workspace may also have the required files, use "scan_workspace_directory_for_gis_files" tool to check the workspace inventory as well.
   - If any step requires an attribute field (or similar parameter) that cannot be predetermined, verify that the plan includes a step to list or examine fields and a clear placeholder indicating that the executor will determine the appropriate field from the list.
   - When using external files, verify proper import steps are included
   - Confirm data source locations (workspace vs external) are correctly handled
   - The correct Index formula must be used for the analysis based on the user request. For example, Do not use NDVI if MNDWI is required by the user request.
   - If user request can be fulfilled using available files, DO NOT PLAN any additional steps of data download. Always use the available files if they already exist.
   
   
2. As you analyze the plan, produce a detailed chain-of-thought that captures your reasoning process.
3. After completing your reasoning, output a JSON object with exactly two keys:
   - "detailed_thought": Containing the full chain-of-thought reasoning.
   - "validity": A final verdict that is either "valid" if the plan is acceptable, or "invalid" if the plan is incorrect. This key must appear last in the JSON object.

Do not output any markdown or additional formatting; output only the JSON object.
"""),
    ("user", """
Original User Request:
{request}

Plan:
{plan}

Output:
Return a JSON object with exactly two keys:
- "detailed_thought": Your complete chain-of-thought reasoning.
- "validity": "valid" if the plan is correct, and "invalid" if the plan is incorrect.
""")
])

EXECUTOR_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a GIS task executor. Your role is to execute the provided plan step-by-step using the available tools.
Each step in the plan is an object with:
- tool: The name of the tool to execute.
- input: The parameters required for that tool.
- description: A brief explanation of why this step is necessary.

Example format:
[
    {{
        "tool": "buffer_features",
        "input": {{
            "input_features": "input.shp",
            "output_features": "buffered_output.shp",
            "buffer_distance": 100,
            "buffer_unit": "Meters"
            }},
        "description": "Buffer the input features by 100 meters."
    }}
]     

Process:
1. Execute each step sequentially.
2. Reference any results from previous steps if they are used as inputs in later steps.
3. If a placeholder was provided for selecting an attribute field (because the correct field was unknown at planning time), use the output from the corresponding "list_fields" step to determine and substitute the appropriate field.
4. If a placeholder was provided for selecting a file name (because the correct file name was unknown at planning time), use the output from the corresponding "scan_workspace_directory_for_gis_files" and "scan_external_directory_for_gis_files" tool to determine and substitute the appropriate file name.
5. After executing all steps, provide the full plan, and a final summary indicating the the overall success or failure of the plan execution.
6. Reason through your choices based on the results of the previous steps before making a decision when plan requires a decision on the tool parameters.
7. DO NOT HALLUCINATE the tool calls in the summary, only include the tool calls that you used and the input parameters that you used.
8. DO NOT HALLUCINATE the tool outputs in the summary, only include the tool outputs that you got.

Output:
- Should include relevant tool calls.
- Should include the full plan with the tool calls and the input parameters that you used.
- A final summary with the overall status.
"""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "Here is the plan to execute: {input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])


class GISAgent:
    def __init__(self, api_key: str, workspace: str, response_queue: queue.Queue, settings_manager: SettingsManager):
        self.api_key = api_key
        self.workspace = workspace
        self.settings_manager = settings_manager
        self.model = "gemini-2.0-pro-exp-02-05"
        self.model_small = "gemini-2.0-flash-exp"
        self.response_queue = response_queue
        self._environment_info = {}  # Cache for environment info

        # Set environment variables for tools that need them
        os.environ["EARTHDATA_USER"] = self.settings_manager.get_api_key("earthdata_user")
        os.environ["EARTHDATA_PASS"] = self.settings_manager.get_api_key("earthdata_pass")
        os.environ["EARTHDATA_TOKEN"] = self.settings_manager.get_api_key("earthdata_token")
        os.environ["TAVILY_API_KEY"] = self.settings_manager.get_api_key("tavily_api_key")
        
        # Set ArcPy workspace if it exists
        if workspace and os.path.exists(workspace):
            try:
                arcpy.env.workspace = workspace
                arcpy.env.overwriteOutput = True
                print(f"ArcPy workspace set to: {workspace}")
            except Exception as e:
                print(f"Error setting ArcPy workspace: {str(e)}")
        else:
            print(f"Warning: Workspace path does not exist: {workspace}")

        self.tools = get_all_tools()
        # Add tool descriptions
        self.tool_descriptions = [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": {
                    k: str(v.annotation) 
                    for k, v in inspect.signature(tool.func).parameters.items()
                }
            }
            for tool in self.tools
        ]
        
        print(f"Loaded {len(self.tools)} tools:")
        for tool in self.tool_descriptions:
            print(f"- {tool['name']}: {len(tool['parameters'])} parameters")
        
        # Initialize agents AFTER tools are set up
        self.planner = self._create_planner()
        self.verifier = self._create_verifier()
        self.executor = self._create_executor()
    
    def _create_planner(self):
        llm = ChatGoogleGenerativeAI(model=self.model, 
                                    google_api_key=self.api_key,
                                    temperature=0.0,
                                    timeout=60)
        
        memory = ConversationBufferMemory(memory_key="chat_history",
                                        return_messages=True,
                                        input_key="input",
                                        output_key="output")
        agent = create_openai_tools_agent(llm, self.tools, PLANNER_PROMPT)
        
        return AgentExecutor(
            agent=agent,
                           tools=self.tools,
                           verbose=True,
                           memory=memory,
            max_iterations=3,
            early_stopping_method="generate",
        )
    

    
    def _create_verifier(self):
        return ChatGoogleGenerativeAI(model=self.model,
                                    google_api_key=self.api_key,
                                    temperature=0.0,
                                    timeout=60)
    
    def _create_executor(self):
        llm = ChatGoogleGenerativeAI(
            model=self.model_small,
                                    google_api_key=self.api_key,
            temperature=0.0,
            max_retries=3,
            timeout=30
        )
        
        # Add memory for executor
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            input_key="input",
            output_key="output"
        )
        
        agent = create_openai_tools_agent(llm, self.tools, EXECUTOR_PROMPT)
        return AgentExecutor(
            agent=agent,
                           tools=self.tools,
                           verbose=True,
            memory=memory,
            max_iterations=5,
            # early_stopping_method="generate",
        )
    
    def _remove_ansi_escape_codes(self, text):
        """Removes ANSI escape codes from a string."""
        ansi_escape = re.compile(r'\x1b\[[0-9;]*m')
        return ansi_escape.sub('', text)
    
    
    def _extract_plan(self, plan_result: dict) -> str:
        """Extract and validate the plan JSON from the planner's output."""
        try:
            print("\n=== Plan Extraction ===")
            # Get the plan text and clean it
            plan = plan_result['output'].strip()
            print(f"Raw plan text: {plan}")
            
            plan = plan.replace("```json", "").replace("```", "").strip()
            print(f"Cleaned plan text: {plan}")
            
            # Validate JSON structure
            parsed_plan = json.loads(plan)
            print(f"Parsed JSON: {json.dumps(parsed_plan, indent=2)}")
            
            # Validate it's a list of steps
            if not isinstance(parsed_plan, list):
                raise ValueError("Plan must be a JSON array of steps")
            
            # Validate each step has required fields
            for i, step in enumerate(parsed_plan):
                print(f"\nValidating step {i + 1}:")
                if not all(k in step for k in ("tool", "input", "description")):
                    raise ValueError("Each step must have 'tool', 'input', and 'description' fields")
                
                # Validate tool exists
                if step["tool"] not in [t.name for t in self.tools]:
                    raise ValueError(f"Unknown tool: {step['tool']}")
                print(f"Step {i + 1} validated successfully")
            
            return plan
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {str(e)}")
            raise ValueError(f"Invalid JSON format: {str(e)}")
        except Exception as e:
            print(f"Error in plan extraction: {str(e)}")
            raise ValueError(f"Error extracting plan: {str(e)}")

    def get_environment_info(self, force_refresh: bool = False) -> dict:
        """Gather all environment information including workspace and external files.
        
        Args:
            force_refresh: If True, forces a refresh of the cached environment info.
        """
        try:
            # Return cached info if available and no refresh is requested
            if not force_refresh and self._environment_info:
                return self._environment_info
            
            # Get workspace inventory
            try:
                # Make sure we have a valid workspace
                if not self.workspace or not os.path.exists(self.workspace):
                    print(f"Warning: Workspace path does not exist: {self.workspace}")
                    workspace_inventory = {}
                else:
                    # Get the workspace inventory
                    workspace_inventory_str = scan_workspace_directory_for_gis_files.invoke(self.workspace)
                    
                    # Parse the JSON result
                    try:
                        workspace_inventory = json.loads(workspace_inventory_str)
                        print(f"Successfully loaded workspace inventory from {self.workspace}")
                    except json.JSONDecodeError as e:
                        print(f"Error parsing workspace inventory JSON: {str(e)}")
                        print(f"Raw inventory string: {workspace_inventory_str}")
                        workspace_inventory = {}
            except Exception as e:
                print(f"Error getting workspace inventory: {str(e)}")
                traceback.print_exc()
                workspace_inventory = {}
            
            # Get external file inventory - always scan all directories fresh
            external_files = {}
            
            # Get the current list of watched directories from settings
            watched_directories = self.settings_manager.settings["watched_directories"]
            
            # Only scan directories that are actually in the watched list
            for directory in watched_directories:
                try:
                    print(f"Scanning directory: {directory}")
                    scan_result_str = scan_external_directory_for_gis_files.invoke(directory)
                    
                    # Ensure the result is valid JSON
                    try:
                        scan_result = json.loads(scan_result_str)
                        external_files[directory] = scan_result
                        
                        # Log the results
                        vector_count = len(scan_result.get("vector_files", []))
                        raster_count = len(scan_result.get("raster_files", []))
                        print(f"Found {vector_count} vector files and {raster_count} raster files in {directory}")
                        
                    except json.JSONDecodeError as json_err:
                        print(f"Error parsing scan result for {directory}: {str(json_err)}")
                        print(f"Raw result: {scan_result_str}")
                        external_files[directory] = {"error": f"Invalid JSON response: {str(json_err)}"}
                        
                except Exception as e:
                    print(f"Error scanning directory {directory}: {str(e)}")
                    external_files[directory] = {"error": str(e), "vector_files": [], "raster_files": []}
            
            # Cache and return the results
            self._environment_info = {
                "workspace": self.workspace,
                "workspace_inventory": workspace_inventory,
                "external_directories": external_files
            }
            
            # Log the environment info for debugging
            print(f"Environment info workspace: {self.workspace}")
            print(f"Workspace inventory: {json.dumps(workspace_inventory, indent=2)}")
            print(f"External directories count: {len(external_files)}")
            
            return self._environment_info
        except Exception as e:
            print(f"Error gathering environment info: {str(e)}")
            traceback.print_exc()  # Print the full traceback for debugging
            return {}

    def refresh_environment_info(self):
        """Force a refresh of the environment information."""
        return self.get_environment_info(force_refresh=True)

    def process_request(self, user_input: str, max_iterations: int = 5) -> str:
        """Process a user request through the three-agent pipeline."""
        try:
            print("\n=== Starting Request Processing ===")
            print(f"User Input: {user_input}")
            
            # Get complete environment information
            env_info = self.get_environment_info()
            print(f"\nEnvironment Info: {json.dumps(env_info, indent=2)}")
            
            current_iteration = 0
            verification_result = ""
            plan = None
            
            while current_iteration < max_iterations:
                print(f"\n--- Iteration {current_iteration + 1} ---")
                self.response_queue.put(f"\n--- Iteration {current_iteration + 1} ---\n") # Send iteration info to GUI

                # Planning Phase
                print("\n1. PLANNING PHASE")
                self.response_queue.put("Planning...\n") # Send "Planning..." to GUI
                planning_input = {
                    "input": user_input if current_iteration == 0 
                            else f"{user_input}\nPrevious verification feedback: {verification_result}",
                "workspace": self.workspace,
                    "inventory": env_info["workspace_inventory"],
                    "external_files": env_info["external_directories"],
                    "tools": self.tool_descriptions,
                }
                print(f"Planning Input: {json.dumps(planning_input, indent=2)}")
                
                try:
                    plan_result = self.planner.invoke(planning_input)
                    print(f"Raw Planner Output: {plan_result['output']}")
                    plan = self._extract_plan(plan_result)
                    print(f"Extracted Plan: {plan}")
                except ValueError as e:
                    print(f"Planning Error: {str(e)}")
                    return f"Planning failed: {str(e)}"
                
                time.sleep(5)  # Wait for 5 seconds before proceeding to the verification phase

                # Verification Phase
                print("\n2. VERIFICATION PHASE")
                self.response_queue.put("Verifying...\n") # Send "Verifying..." to GUI
                verification_input = {
                "plan": plan,
                "request": user_input,
                    "tools": self.tool_descriptions,
                    "inventory": env_info["workspace_inventory"],
                    "external_files": env_info["external_directories"]
                }

                print(f"Verification Input: {json.dumps(verification_input, indent=2)}")
                
                verification_result = self.verifier.invoke(
                VERIFIER_PROMPT.format_prompt(**verification_input)
                ).content
            
                # Clean the verification result
                verification_result = verification_result.replace("```json", "").replace("```", "").strip()

                # Process verification result as JSON
                try:
                    verification_json = json.loads(verification_result)
                    detailed_thought = verification_json.get("detailed_thought", "")
                    validity = verification_json.get("validity", "invalid")
                    print(f"Verification Detailed Thought: {detailed_thought}")
                    print(f"Verification Validity: {validity}")
                    
                    if validity.lower() == "valid":
                        print("Plan verified as valid!")
                        break  # Exit the while loop if plan is valid
                    else:
                        print(f"Verification invalid: {detailed_thought}")
                        # Provide feedback to the planner for the next iteration
                        planning_input["previous_feedback"] = detailed_thought
                        print(f"Feedback sent to planner: {detailed_thought}")
                        current_iteration += 1  # Increment iteration for the next loop
                        if current_iteration >= max_iterations:
                            print("Maximum iterations reached. Planning failed.")
                            return "Maximum iterations reached. Planning failed."
                        
                        time.sleep(5)
                        continue  # Continue to next iteration
                        
                except json.JSONDecodeError as e:
                    print(f"Verification JSON decode error: {str(e)}")
                    return f"Invalid verification output format: {str(e)}"
            
            # Do not proceed to Execution Phase if the plan is invalid
            if validity.lower() != "valid":
                return "Plan is invalid after maximum iterations. Execution phase skipped."
            
            # Execution Phase
            print("\n3. EXECUTION PHASE")
            print(f"Executing Plan: {plan}")
            print(f"Plan JSON sent to Executor: {plan}")
            
            # Format the EXECUTOR_PROMPT to see the full input
            executor_input = EXECUTOR_PROMPT.format_prompt(input=plan, agent_scratchpad=[], chat_history=[])
            full_executor_input_content = executor_input.to_string()
            print("\n--- Full Input to Executor Agent (Prompt + Plan) ---")
            print(full_executor_input_content)
            print("\n--- End of Full Input to Executor Agent ---")

            # Capture stdout during executor.invoke()
            captured_output = io.StringIO()
            try:
                with contextlib.redirect_stdout(captured_output):
                    execution_result = self.executor.invoke({
                            "input": plan
                })
                    executor_output = captured_output.getvalue()
                    executor_output_cleaned = self._remove_ansi_escape_codes(executor_output)
                    print(f"Execution Result (Summary): {execution_result}")
                    self.response_queue.put(f"Executor Output:\n{executor_output_cleaned}\n")
                return f"Execution completed:\n{execution_result['output']}"
            except Exception as exec_error:
                executor_output = captured_output.getvalue()
                executor_output_cleaned = self._remove_ansi_escape_codes(executor_output)
                print(f"Execution error: {str(exec_error)}")
                error_message = f"Execution failed: {str(exec_error)}"
                self.response_queue.put(f"Executor Output:\n{executor_output_cleaned}\nExecution error:\n{error_message}\n")
                return error_message
            
        except Exception as e:
            print(f"\nERROR: {str(e)}")
            print("Traceback:", traceback.format_exc())
            return f"Error processing request: {str(e)}"
    
    def _update_tree_with_results(self, scan_results):
        """Update the tree view with the scan results.
        
        Args:
            scan_results: Dictionary mapping directory paths to scan results.
        """
        # Use the tree manager to update the tree view
        self.tree_manager.update_tree_from_scan_results(scan_results)
        
        # Calculate statistics
        total_vector_files = 0
        total_raster_files = 0
        
        for directory, result in scan_results.items():
            if "error" not in result:
                total_vector_files += len(result.get("vector_files", []))
                total_raster_files += len(result.get("raster_files", []))
        
        # Show completion message with statistics
        self.update_response_area(
            f"Directory scanning completed. Found {total_vector_files} vector files and "
            f"{total_raster_files} raster files in {len(scan_results)} directories."
        )

# GUI Class
class GISGUI:
    def __init__(self, gis_agent: GISAgent):
        self.gis_agent = gis_agent
        self.settings_manager = gis_agent.settings_manager
        self.response_queue = queue.Queue()
        self.request_queue = queue.Queue()  # Initialize request_queue
        self.root = tk.Tk()
        self.root.title("GIS Agent")
        self.root.geometry("1200x800")
        self.root.minsize(800, 600)
        
        # Set up variables
        self.workspace_var = tk.StringVar(value=self.settings_manager.settings["workspace"])
        self.button_colors = ["#4CAF50", "#5CBF60", "#6CCF70", "#7CDF80", "#8CEF90", "#7CDF80", "#6CCF70", "#5CBF60"]
        
        # Create managers (tree_manager will be initialized after tree view is created)
        self.directory_manager = DirectoryManager(self.settings_manager, self.gis_agent)
        self.tree_manager = None
        
        # Setup GUI
        self.setup_gui()
        
        # Initialize tree manager now that tree view exists
        self.tree_manager = TreeViewManager(self.files_tree, self.update_response_area)
        
        # Start agent thread
        self.start_agent_thread()
        
        # Initial scan of directories
        self.initial_scan()
    
    def initial_scan(self):
        """Perform initial scan of directories in background"""
        if not self.settings_manager.settings["watched_directories"]:
            return
        
        threading.Thread(target=self._background_initial_scan, daemon=True).start()
    
    def _background_initial_scan(self):
        """Background thread for initial directory scan"""
        try:
            self.update_response_area("Performing initial scan of watched directories...")
            scan_results = self.directory_manager.scan_all_directories(
                callback=lambda msg, _: self.update_response_area(msg)
            )
            
            # Update tree view in main thread
            self.root.after(0, lambda: self._update_tree_with_results(scan_results))
        except Exception as e:
            self.update_response_area(f"Error in initial scan: {str(e)}")
    
    def _update_tree_with_results(self, scan_results):
        """Update the tree view with the scan results.
        
        Args:
            scan_results: Dictionary mapping directory paths to scan results.
        """
        # Use the tree manager to update the tree view
        self.tree_manager.update_tree_from_scan_results(scan_results)
        
        # Calculate statistics
        total_vector_files = 0
        total_raster_files = 0
        
        for directory, result in scan_results.items():
            if "error" not in result:
                total_vector_files += len(result.get("vector_files", []))
                total_raster_files += len(result.get("raster_files", []))
        
        # Show completion message with statistics
        self.update_response_area(
            f"Directory scanning completed. Found {total_vector_files} vector files and "
            f"{total_raster_files} raster files in {len(scan_results)} directories."
        )
    
    def setup_gui(self):
        self.root.title("ArcGIS AI Assistant")
        
        # Set window dimensions and position
        screen_width = self.root.winfo_screenwidth()
        screen_height = int(self.root.winfo_screenheight()) - 80
        window_width = int(screen_width * 0.30)  # 30% of screen width
        window_height = screen_height
        
        # Ensure window doesn't go beyond screen edge
        x_position = screen_width - window_width
        y_position = 0
        
        # Set the window size and position
        self.root.geometry(f"{window_width}x{window_height}+{x_position}+{y_position}")
        
        # Prevent window resizing below minimum dimensions
        self.root.minsize(window_width, 600)
        self.root.configure(bg="#f0f0f0")
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Main chat tab
        self.chat_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.chat_frame, text="Chat")
        
        # Environment tab
        self.env_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.env_frame, text="Environment")
        
        # Setup chat frame
        self.setup_chat_frame()
        
        # Setup environment frame
        self.setup_environment_frame()
    
    def setup_chat_frame(self):
        # Main frame with groove relief
        main_frame = tk.Frame(self.chat_frame, bg="#f0f0f0", bd=10, relief="groove")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Output frame with white background
        output_frame = tk.Frame(main_frame, bg="white", bd=2, relief="groove")
        output_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Response area with better styling
        self.response_area = scrolledtext.ScrolledText(
            output_frame, 
            font=("Arial", 11),
            wrap=tk.WORD,
            bg="white",
            state='disabled',
            bd=0
        )
        self.response_area.pack(expand=True, fill=tk.BOTH, padx=5, pady=5)
        
        # Input frame
        input_frame = tk.Frame(main_frame, bg="#f0f0f0", bd=2, relief="groove")
        input_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)
        
        # Input box with placeholder
        self.input_box = tk.Text(
            input_frame,
            height=4,  # Initial height
            width=40,
            wrap=tk.WORD,  # Ensure text wraps
            font=("Arial", 12),
            bd=0,
            highlightthickness=0
        )
        self.input_box.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5, pady=5)
        self.input_box.insert("1.0", "Enter your GIS request")
        self.input_box.config(fg='gray')
        
        # Input box bindings
        self.input_box.bind("<FocusIn>", self.on_entry_click)
        self.input_box.bind("<FocusOut>", self.on_focus_out)
        self.input_box.bind("<Return>", self.submit_request)
        self.input_box.bind("<Shift-Return>", lambda e: self.input_box.insert(tk.INSERT, '\n'))
        
        # Animated send button
        self.send_button = tk.Button(
            input_frame,
            text="Send",
            command=self.submit_request,
            bg=self.button_colors[0],
            fg="white",
            font=("Arial", 12, "bold"),
            bd=0,
            activebackground=self.button_colors[0],
            activeforeground="white"
        )
        self.send_button.pack(side=tk.RIGHT, padx=5, pady=5)
        self.send_button.bind("<Enter>", self.animate_button)
    
    def setup_environment_frame(self):
        """Set up the environment frame with API keys, workspace, and directories sections."""
        # Create a notebook for tabbed sections within the environment tab
        notebook = ttk.Notebook(self.env_frame)
        notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # API Keys section
        api_frame = ttk.Frame(notebook, padding="10")
        notebook.add(api_frame, text="API Keys")
        
        # Initialize dictionary to store API key variables
        self.api_key_vars = {}
        
        # Gemini API Key (primary model)
        ttk.Label(api_frame, text="Gemini API Key:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.api_key_vars["gemini_api_key"] = tk.StringVar(value=self.settings_manager.get_api_key("gemini_api_key"))
        gemini_key_entry = ttk.Entry(api_frame, textvariable=self.api_key_vars["gemini_api_key"], width=50, show="*")
        gemini_key_entry.grid(row=0, column=1, sticky=tk.W, pady=5)
        
        # Show/Hide Gemini API Key
        show_gemini_key = tk.BooleanVar(value=False)
        ttk.Checkbutton(api_frame, text="Show", variable=show_gemini_key, 
                        command=lambda: self.toggle_key_visibility(gemini_key_entry, show_gemini_key)).grid(row=0, column=2, sticky=tk.W, pady=5)
        
        # EarthData User
        ttk.Label(api_frame, text="EarthData User:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.api_key_vars["earthdata_user"] = tk.StringVar(value=self.settings_manager.get_api_key("earthdata_user"))
        ttk.Entry(api_frame, textvariable=self.api_key_vars["earthdata_user"], width=50).grid(row=1, column=1, sticky=tk.W, pady=5)
        
        # EarthData Password
        ttk.Label(api_frame, text="EarthData Password:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.api_key_vars["earthdata_pass"] = tk.StringVar(value=self.settings_manager.get_api_key("earthdata_pass"))
        earthdata_pass_entry = ttk.Entry(api_frame, textvariable=self.api_key_vars["earthdata_pass"], width=50, show="*")
        earthdata_pass_entry.grid(row=2, column=1, sticky=tk.W, pady=5)
        
        # Show/Hide EarthData Password
        show_earthdata_pass = tk.BooleanVar(value=False)
        ttk.Checkbutton(api_frame, text="Show", variable=show_earthdata_pass, 
                        command=lambda: self.toggle_key_visibility(earthdata_pass_entry, show_earthdata_pass)).grid(row=2, column=2, sticky=tk.W, pady=5)
        
        # EarthData Token
        ttk.Label(api_frame, text="EarthData Token:").grid(row=3, column=0, sticky=tk.W, pady=5)
        self.api_key_vars["earthdata_token"] = tk.StringVar(value=self.settings_manager.get_api_key("earthdata_token"))
        earthdata_token_entry = ttk.Entry(api_frame, textvariable=self.api_key_vars["earthdata_token"], width=50, show="*")
        earthdata_token_entry.grid(row=3, column=1, sticky=tk.W, pady=5)
        
        # Show/Hide EarthData Token
        show_earthdata_token = tk.BooleanVar(value=False)
        ttk.Checkbutton(api_frame, text="Show", variable=show_earthdata_token, 
                        command=lambda: self.toggle_key_visibility(earthdata_token_entry, show_earthdata_token)).grid(row=3, column=2, sticky=tk.W, pady=5)
        
        # Tavily API Key
        ttk.Label(api_frame, text="Tavily API Key:").grid(row=4, column=0, sticky=tk.W, pady=5)
        self.api_key_vars["tavily_api_key"] = tk.StringVar(value=self.settings_manager.get_api_key("tavily_api_key"))
        tavily_key_entry = ttk.Entry(api_frame, textvariable=self.api_key_vars["tavily_api_key"], width=50, show="*")
        tavily_key_entry.grid(row=4, column=1, sticky=tk.W, pady=5)
        
        # Show/Hide Tavily API Key
        show_tavily_key = tk.BooleanVar(value=False)
        ttk.Checkbutton(api_frame, text="Show", variable=show_tavily_key, 
                        command=lambda: self.toggle_key_visibility(tavily_key_entry, show_tavily_key)).grid(row=4, column=2, sticky=tk.W, pady=5)
        
        # Save API Keys button
        ttk.Button(api_frame, text="Save API Keys", command=self.save_api_keys).grid(row=5, column=1, sticky=tk.W, pady=10)
        
        # Combined Workspace & Files section
        workspace_files_frame = ttk.Frame(notebook, padding="10")
        notebook.add(workspace_files_frame, text="Workspace & Files")
        
        # Configure the frame layout
        workspace_files_frame.columnconfigure(0, weight=1)
        workspace_files_frame.rowconfigure(0, weight=0)  # Workspace section (fixed height)
        workspace_files_frame.rowconfigure(1, weight=1)  # Directories section (small)
        workspace_files_frame.rowconfigure(2, weight=3)  # Files section (larger)
        
        # Workspace section at the top
        workspace_section = ttk.LabelFrame(workspace_files_frame, text="Workspace Path", padding="10")
        workspace_section.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        workspace_section.columnconfigure(1, weight=1)
        
        # Workspace path
        ttk.Label(workspace_section, text="Path:").grid(row=0, column=0, sticky=tk.W, pady=5, padx=5)
        self.workspace_var = tk.StringVar(value=self.settings_manager.settings.get("workspace", ""))
        workspace_entry = ttk.Entry(workspace_section, textvariable=self.workspace_var, width=50)
        workspace_entry.grid(row=0, column=1, sticky=tk.EW, pady=5, padx=5)
        
        # Browse button for workspace
        ttk.Button(workspace_section, text="Browse", command=self.browse_workspace).grid(row=0, column=2, sticky=tk.W, pady=5, padx=5)
        
        # Save Workspace button
        ttk.Button(workspace_section, text="Save", command=self.save_workspace).grid(row=0, column=3, sticky=tk.W, pady=5, padx=5)
        
        # Directories section (middle)
        directories_frame = ttk.LabelFrame(workspace_files_frame, text="Watched Directories", padding="10")
        directories_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        directories_frame.columnconfigure(0, weight=1)
        directories_frame.rowconfigure(0, weight=1)
        
        # Directory listbox with scrollbar
        dir_list_frame = ttk.Frame(directories_frame)
        dir_list_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        dir_list_frame.columnconfigure(0, weight=1)
        dir_list_frame.rowconfigure(0, weight=1)
        
        self.dir_listbox = tk.Listbox(dir_list_frame, height=4)  # Reduced height
        self.dir_listbox.grid(row=0, column=0, sticky="nsew")
        
        dir_scrollbar = ttk.Scrollbar(dir_list_frame, orient="vertical", command=self.dir_listbox.yview)
        dir_scrollbar.grid(row=0, column=1, sticky="ns")
        self.dir_listbox.configure(yscrollcommand=dir_scrollbar.set)
        
        # Directory buttons
        dir_buttons_frame = ttk.Frame(directories_frame)
        dir_buttons_frame.grid(row=1, column=0, sticky="ew", pady=5)
        
        ttk.Button(dir_buttons_frame, text="Add Directory", command=self.add_directory).pack(side=tk.LEFT, padx=5)
        ttk.Button(dir_buttons_frame, text="Remove Directory", command=self.remove_directory).pack(side=tk.LEFT, padx=5)
        ttk.Button(dir_buttons_frame, text="Scan Directories", command=self.scan_directories).pack(side=tk.LEFT, padx=5)
        
        # Files section (bottom)
        files_frame = ttk.LabelFrame(workspace_files_frame, text="Available GIS Files", padding="10")
        files_frame.grid(row=2, column=0, sticky="nsew", padx=5, pady=5)
        files_frame.columnconfigure(0, weight=1)
        files_frame.rowconfigure(0, weight=1)
        
        # Create a tree view for files with scrollbars
        tree_frame = ttk.Frame(files_frame)
        tree_frame.grid(row=0, column=0, sticky="nsew")
        tree_frame.columnconfigure(0, weight=1)
        tree_frame.rowconfigure(0, weight=1)
        
        # Create the tree view with columns
        self.files_tree = ttk.Treeview(tree_frame, columns=("name", "type", "path", "driver", "crs", "layers", "dimensions", "bands"))
        self.files_tree.grid(row=0, column=0, sticky="nsew")
        
        # Configure the tree view columns
        self.files_tree.heading("#0", text="Directory")
        self.files_tree.heading("name", text="Name")
        self.files_tree.heading("type", text="Type")
        self.files_tree.heading("path", text="Path")
        self.files_tree.heading("driver", text="Driver")
        self.files_tree.heading("crs", text="CRS")
        self.files_tree.heading("layers", text="Layers")
        self.files_tree.heading("dimensions", text="Dimensions")
        self.files_tree.heading("bands", text="Bands")
        
        # Set column widths
        self.files_tree.column("#0", width=200, minwidth=150, stretch=False)
        self.files_tree.column("name", width=150, minwidth=100, stretch=True)
        self.files_tree.column("type", width=80, minwidth=60, stretch=False)
        self.files_tree.column("path", width=250, minwidth=150, stretch=True)
        self.files_tree.column("driver", width=80, minwidth=60, stretch=False)
        self.files_tree.column("crs", width=100, minwidth=80, stretch=False)
        self.files_tree.column("layers", width=60, minwidth=50, stretch=False)
        self.files_tree.column("dimensions", width=100, minwidth=80, stretch=False)
        self.files_tree.column("bands", width=60, minwidth=50, stretch=False)
        
        # Add scrollbars
        tree_y_scrollbar = ttk.Scrollbar(tree_frame, orient="vertical", command=self.files_tree.yview)
        tree_y_scrollbar.grid(row=0, column=1, sticky="ns")
        tree_x_scrollbar = ttk.Scrollbar(tree_frame, orient="horizontal", command=self.files_tree.xview)
        tree_x_scrollbar.grid(row=1, column=0, sticky="ew")
        
        self.files_tree.configure(yscrollcommand=tree_y_scrollbar.set, xscrollcommand=tree_x_scrollbar.set)
        
        # Initialize the tree view manager
        self.tree_manager = TreeViewManager(self.files_tree, self.update_response_area)
        
        # Load watched directories
        self.load_watched_directories()
    
    def browse_workspace(self):
        """Browse and select a workspace directory."""
        # Show initial message to user
        tk.messagebox.showinfo(
            "Workspace Selection",
            "Please select a directory to use as your workspace."
        )
        
        # Use askdirectory to allow selecting a directory
        workspace = filedialog.askdirectory(
            title="Select Workspace Directory",
            initialdir=os.path.dirname(self.workspace_var.get()) if self.workspace_var.get() else os.getcwd()
        )
        
        if workspace:
            # Update the workspace entry
            self.workspace_var.set(workspace)
            
            # We don't save the workspace here - user needs to click the Save button
    
    def update_arcpy_environment(self):
        """Update ArcPy environment settings when workspace changes."""
        arcpy.env.workspace = self.workspace_var.get()
        arcpy.env.overwriteOutput = True  # Always enable overwrite
    
    def add_directory(self):
        """Add a directory to the watched directories list and scan it for GIS files."""
        directory = filedialog.askdirectory(title="Select Directory to Watch")
        if not directory:
            return
        
        # Check if directory already exists in the watched list
        if directory in self.settings_manager.settings["watched_directories"]:
            tk.messagebox.showinfo(
                "Directory Already Watched",
                f"The directory '{directory}' is already being watched."
            )
            return
        
        # Add the directory to settings
        self.settings_manager.add_directory(directory)
        self.load_watched_directories()
        
        # Show progress indicator
        progress = ScanProgressIndicator(self.root, self.update_response_area)
        progress.show()
        
        # Use a separate thread to scan the directory
        def scan_thread():
            try:
                # Scan only the new directory
                progress.update(f"Scanning directory: {directory}", 0)
                scan_result = self.directory_manager.scan_directory(directory, force_refresh=True)
                
                # Create a results dict with just this directory
                new_scan_results = {directory: scan_result}
                
                # Update the GIS agent's cache
                self.directory_manager.update_gis_agent_cache(new_scan_results)
                
                # Update progress
                progress.update("Scan complete, updating tree view...", 90)
                
                # Get all scan results from cache to show all directories
                all_scan_results = {}
                for dir_path in self.settings_manager.settings["watched_directories"]:
                    if dir_path in self.directory_manager.scan_cache:
                        all_scan_results[dir_path] = self.directory_manager.scan_cache[dir_path]
                
                # Update tree view in main thread with all scan results
                self.root.after(0, lambda: self._update_tree_with_results(all_scan_results))
                
                # Show completion message
                self.root.after(0, lambda: tk.messagebox.showinfo(
                    "Directory Added",
                    f"Directory '{directory}' has been added and scanned successfully."
                ))
                
                # Close progress window
                self.root.after(0, progress.close)
                
            except Exception as e:
                error_msg = f"Error scanning directory {directory}: {str(e)}"
                print(error_msg)
                traceback.print_exc()
                
                # Update UI from main thread
                self.root.after(0, lambda: progress.update(error_msg, -1))
                self.root.after(2000, progress.close)  # Close after 2 seconds
                
                self.root.after(0, lambda: tk.messagebox.showerror(
                    "Scan Error",
                    f"Error scanning directory: {str(e)}"
                ))
        
        # Start the scan thread
        threading.Thread(target=scan_thread, daemon=True).start()
    
    def remove_directory(self):
        """Remove a directory from the watched directories list and update the tree view."""
        selection = self.dir_listbox.curselection()
        if not selection:
            return
        
        directory = self.dir_listbox.get(selection[0])
        
        # Confirm removal
        confirm = tk.messagebox.askyesno(
            "Confirm Removal",
            f"Are you sure you want to remove the directory '{directory}' from the watched list?"
        )
        
        if not confirm:
            return
        
        # Update the response area
        self.update_response_area(f"Removing directory: {directory}")
        
        # Remove the directory from settings
        self.settings_manager.remove_directory(directory)
        self.load_watched_directories()
        
        # Remove from cache
        if directory in self.directory_manager.scan_cache:
            del self.directory_manager.scan_cache[directory]
        
        # Update the GIS agent's environment info
        if self.gis_agent._environment_info and "external_directories" in self.gis_agent._environment_info:
            if directory in self.gis_agent._environment_info["external_directories"]:
                del self.gis_agent._environment_info["external_directories"][directory]
        
        # Get current scan results without the removed directory
        scan_results = {}
        for dir_path in self.settings_manager.settings["watched_directories"]:
            if dir_path in self.directory_manager.scan_cache:
                scan_results[dir_path] = self.directory_manager.scan_cache[dir_path]
        
        # Update tree view
        self._update_tree_with_results(scan_results)
        
        # Show confirmation
        self.update_response_area(f"Directory removed: {directory}")
    
    def load_watched_directories(self):
        """Load the watched directories from settings and display them in the listbox."""
        # Clear the current list
        self.dir_listbox.delete(0, tk.END)
        
        # Get the list of watched directories from settings
        watched_directories = self.settings_manager.settings.get("watched_directories", [])
        
        # Add each directory to the listbox
        for directory in watched_directories:
            self.dir_listbox.insert(tk.END, directory)
    
    def scan_directories(self):
        """Scan all watched directories for GIS files and update the tree view."""
        # Get the list of watched directories
        watched_directories = self.settings_manager.settings["watched_directories"]
        
        if not watched_directories:
            self.update_response_area("No directories to scan. Please add directories first.")
            return
        
        # Create a progress indicator
        progress = ScanProgressIndicator(self.root, self.update_response_area)
        progress.show()
        
        # Define the background scanning function
        def background_scan():
            try:
                # Scan all directories
                scan_results = self.directory_manager.scan_all_directories(
                    callback=progress.update,
                    force_refresh=True
                )
                
                # Update the tree view from the main thread
                self.root.after(0, lambda: self._update_tree_with_results(scan_results))
                
                # Close the progress indicator
                self.root.after(0, progress.close)
                
            except Exception as e:
                error_msg = f"Error during directory scanning: {str(e)}"
                print(error_msg)
                traceback.print_exc()
                
                # Update the progress indicator with the error
                self.root.after(0, lambda: progress.update(error_msg, -1))
                
                # Close the progress indicator after a delay
                self.root.after(3000, progress.close)
        
        # Start the background scanning thread
        threading.Thread(target=background_scan, daemon=True).start()
    
    def on_entry_click(self, event):
        """Clear placeholder text when entry is clicked."""
        # For the input box
        if hasattr(self, 'input_box') and event.widget == self.input_box:
            if self.input_box.get("1.0", tk.END).strip() == "Enter your GIS request":
                self.input_box.delete("1.0", tk.END)
                self.input_box.config(fg='black')
    
    def on_focus_out(self, event):
        """Restore placeholder text when focus is lost and the field is empty."""
        # For the input box
        if hasattr(self, 'input_box') and event.widget == self.input_box:
            if not self.input_box.get("1.0", tk.END).strip():
                self.input_box.insert("1.0", "Enter your GIS request")
                self.input_box.config(fg='gray')
    
    def animate_button(self, event):
        current_color_index = self.button_colors.index(self.send_button.cget("bg"))
        next_color_index = (current_color_index + 1) % len(self.button_colors)
        self.send_button.config(
            bg=self.button_colors[next_color_index],
            activebackground=self.button_colors[next_color_index]
        )
    
    def submit_request(self, event=None):
        request = self.input_box.get("1.0", tk.END).strip()
        if request and request != "Enter your GIS request":
            self.request_queue.put(request)
            self.input_box.delete("1.0", tk.END)
            self.update_response_area(f"User: {request}\n")
            self.input_box.focus_set()
    
    def update_response_area(self, message: str):
        """Update the response area with a message.
        
        This method can be called from any thread, as it uses the after method
        to ensure the update happens in the main thread.
        """
        def _update():
            self.response_area.configure(state='normal')
            self.response_area.insert(tk.END, message + "\n")
            self.response_area.see(tk.END)
            self.response_area.configure(state='disabled')
        
        # If called from the main thread, update directly
        if threading.current_thread() is threading.main_thread():
            _update()
        # If called from another thread, schedule the update in the main thread
        else:
            self.root.after(0, _update)
    
    def process_responses(self):
        try:
            while True:
                response = self.response_queue.get_nowait()
                if response is None:  # Check for None
                    continue
                self.update_response_area(response)
                if response == "Exiting...":
                    self.root.destroy()
                    return
        except queue.Empty:
            pass
        self.root.after(100, self.process_responses)
    
    def start_agent_thread(self):
        def agent_worker():
            while True:
                request = self.request_queue.get()
                if request.lower() == "/exit":
                    self.response_queue.put("Exiting...")
                    break
                
                result = self.gis_agent.process_request(request)
                self.response_queue.put(result)
                self.request_queue.task_done()
        
        agent_thread = threading.Thread(target=agent_worker, daemon=True)
        agent_thread.start()
        self.process_responses()
    
    def run(self):
        self.root.mainloop()

    def save_api_keys(self):
        """Save the API keys to the settings."""
        try:
            # Save each API key to the settings manager
            for key_name, var in self.api_key_vars.items():
                self.settings_manager.set_api_key(key_name, var.get())
            
            # Update environment variables
            os.environ["GEMINI_API_KEY"] = self.api_key_vars["gemini_api_key"].get()
            os.environ["EARTHDATA_USER"] = self.api_key_vars["earthdata_user"].get()
            os.environ["EARTHDATA_PASS"] = self.api_key_vars["earthdata_pass"].get()
            os.environ["EARTHDATA_TOKEN"] = self.api_key_vars["earthdata_token"].get()
            os.environ["TAVILY_API_KEY"] = self.api_key_vars["tavily_api_key"].get()
            
            # Update the GIS agent's API key
            self.gis_agent.api_key = self.api_key_vars["gemini_api_key"].get()
            
            # Show confirmation
            self.update_response_area("API keys saved successfully.")
            tk.messagebox.showinfo("Success", "API keys saved successfully.")
        except Exception as e:
            error_msg = f"Error saving API keys: {str(e)}"
            self.update_response_area(error_msg)
            tk.messagebox.showerror("Error", error_msg)

    def save_workspace(self):
        """Save the workspace path to the settings."""
        # Get the workspace path from the entry field
        workspace = self.workspace_var.get()
        
        # Update the settings
        self.settings_manager.settings["workspace"] = workspace
        self.settings_manager.save_settings()
        
        # Update the GIS agent's workspace
        self.gis_agent.workspace = workspace
        
        # Update ArcPy environment
        self.update_arcpy_environment()
        
        # Force refresh of the GIS agent's environment info
        self.gis_agent.refresh_environment_info()
        
        # Show confirmation
        self.update_response_area(f"Workspace saved: {workspace}")
        
        # Show a message to the user
        tk.messagebox.showinfo("Workspace Updated", f"Workspace has been updated to: {workspace}")

    def toggle_key_visibility(self, entry_widget, show_var):
        """Toggle the visibility of an API key in the entry widget."""
        if show_var.get():
            entry_widget.config(show="")
        else:
            entry_widget.config(show="*")

def main():
    try:
        # Create settings manager first
        settings_manager = SettingsManager()
        
        # Try to get API key from settings
        api_key = settings_manager.get_api_key("gemini_api_key")
        
        # If no API key in settings, try to load from .env
        if not api_key:
            try:
                api_key = setup_environment()
                # Save the API key to settings for future use
                settings_manager.set_api_key("gemini_api_key", api_key)
                print("API key loaded from .env file and saved to settings")
            except ValueError as e:
                # If still no API key, prompt the user
                print(f"Error loading API key: {str(e)}")
                # Create a simple dialog to get the API key
                root = tk.Tk()
                root.withdraw()  # Hide the main window
                api_key = tkinter.simpledialog.askstring(
                    "API Key Required", 
                    "Please enter your Gemini API Key:",
                    parent=root
                )
                root.destroy()
                
                if not api_key:
                    raise ValueError("No API key provided. Application cannot start.")
                
                # Save the entered API key to settings
                settings_manager.set_api_key("gemini_api_key", api_key)
        
        # Also load other API keys from .env if they're not in settings
        for key_name in ["earthdata_user", "earthdata_pass", "earthdata_token", "tavily_api_key"]:
            if not settings_manager.get_api_key(key_name):
                # Try to load from environment variables (which may have been set by setup_environment)
                env_value = os.getenv(key_name.upper())
                if env_value:
                    settings_manager.set_api_key(key_name, env_value)
                    print(f"{key_name} loaded from environment and saved to settings")
        
        # Create response queue for agent and GUI to communicate
        response_queue = queue.Queue()

        # Create GIS Agent, passing both response queue and settings manager
        gis_agent = GISAgent(api_key, settings_manager.settings["workspace"], response_queue, settings_manager)

        # Create and run GUI, passing the agent
        gui = GISGUI(gis_agent)
        gui.run()
    except Exception as e:
        print(f"Error starting application: {str(e)}")
        traceback.print_exc()
        # Show error in a message box
        root = tk.Tk()
        root.withdraw()
        tkinter.messagebox.showerror("Error", f"Error starting application: {str(e)}")
        root.destroy()

if __name__ == "__main__":
    main() 