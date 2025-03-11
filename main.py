import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain_community.callbacks.manager import get_openai_callback
from langchain_core.callbacks import BaseCallbackHandler
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
from langchain.agents.format_scratchpad import format_to_openai_functions
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser


# Import tools properly
from less_tools import *

# Function to get all tools from the tools module
def get_all_tools():
    """Get all tool functions from the tools module.
    
    Returns:
        A list of all tools defined in the tools module.
    """
    import less_tools
    
    # Get all members from the tools module that are decorated with @tool
    tool_functions = []
    for name in dir(less_tools):
        obj = getattr(less_tools, name)
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
    ("system", """[Output Requirements]
You are a GIS planning assistant. Your goal is to develop a detailed, step-by-step plan to solve a given GIS problem using ONLY the provided tools. Your output must be a valid JSON array without any markdown, code blocks, or extra commentary. Each element in the array must be a JSON object with exactly these keys:
  - "tool": The exact name of one available tool (do not invent new functions).
  - "input": A dictionary of input parameters that exactly match the required parameters of that tool.
  - "description": A brief explanation (one or two sentences) of why this tool is used in that step.

[Context Information]
You are provided with:
- Available Tools and Their Descriptions: {tools}
- Current Workspace: {workspace}
- Workspace Inventory Files: {inventory}
- External Files Available: {external_files}

[GIS and Geoprocessing Considerations]
GIS tasks often include:
- Data acquisition or verification (e.g., scanning directories for GIS files)
- Checking spatial reference systems, data formats, and field attributes
- Importing or converting data (e.g., importing external files into the workspace)
- Geoprocessing operations such as buffering, overlay analysis, reprojecting, clipping, and attribute querying
- Ensuring that required fields exist and that numeric, string, and date attributes are correctly formatted
- Verifying that any index or analysis formula (e.g., NDVI, MNDWI) is appropriate for the user's request
- Handling cases where the file name or attribute field is unknown by using dedicated scanning or listing tools (e.g., "list_fields", "scan_workspace_directory_for_gis_files", "scan_external_directory_for_gis_files")

[Critical Instruction: Do Not Assume Inputs]

Attribute Fields: If a step requires an attribute field (for example, to extract "atm" values from a financial dataset) but the correct field name is not explicitly provided, do not assume it. Instead, include a step to retrieve the field names using the "list_fields" tool.
Example: Instead of outputting:
{{
    "tool": "calculate_field",
    "input": {{"layer": "financial_services", "field": "some field name", "expression": "some_expression"}},
    "description": "Calculate ATM values from the financial_services layer."
}}
Output instead:
{{
    "tool": "list_fields",
    "input": {{"layer": "financial_services"}},
    "description": "Retrieve the list of attribute fields for the financial_services layer to determine the correct field for ATM values."
}}

File Names: If a step requires a file name (for example, when importing data) and it is unknown, include a step that calls a scanning tool to list available files.
Example: Instead of assuming:
{{
    "tool": "import_csv",
    "input": {{"file": "data.csv"}},
    "description": "Import the CSV file containing GIS data."
}}
Output instead:
{{
    "tool": "scan_workspace_directory_for_gis_files",
    "input": {{"workspace": "<workspace_directory>"}},
    "description": "Scan the workspace directory to list available GIS files and determine the correct CSV file."
}}

Missing Parameters: If any required parameter is unclear or missing, do not guess a default value. Always include a step to retrieve or verify that parameter using the appropriate tool.


[Process Instructions]
1. **Review Request and Context:** Analyze the user request along with the current workspace inventory and external file listings.
2. **Data Verification:** Check if the required GIS data (files, layers, attributes) already exists. Do not plan redundant download or import steps if data is present.
3. **Landsat Data Considerations:** If the user request involves Landsat data, ensure that the plan includes steps to:
   - Verify the availability of Landsat data in the workspace or external files.
   - Check if the required Landsat data is already available in the workspace or external files.
   - If the required Landsat data is not available, plan a step to download the data using the appropriate tool.
3. **Plan Geoprocessing Steps:** Identify necessary steps such as:
   - Importing external files (if required) using the appropriate tool.
   - Converting file formats or reprojecting data to a common spatial reference.
   - Extracting or listing attribute fields when the required field name is unknown. In these cases, include a step that uses a tool (e.g., "list_fields") and use a clear placeholder such as "field name to be decided by executor based on list_fields output".
   - Scanning directories to determine the correct file names. Use placeholders like "file name to be decided by executor based on scan_workspace_directory_for_gis_files output" when necessary.
4. **Tool Selection and Parameter Specification:** For each step:
   - Choose a valid tool from the provided list.
   - Provide the required parameters exactly as specified in the tool's description.
   - Do not assume any file names or attribute names; always plan a step to verify them.
5. **Sequence and Logic:** Ensure that the sequence of steps is logical, that each step builds on the outputs of previous steps, and that no critical dependencies are missing.
6. **Avoid Redundancies:** If a required file or attribute is already available (from inventory or external files), do not plan unnecessary scanning or download steps unless explicitly requested.

[Output]
Output ONLY the JSON array representing the plan.

Example Format (do not include markdown or extra text):
[
    {{
        "tool": "scan_workspace_directory_for_gis_files",
        "input": {{"workspace": "D:\\masters_project\\ArcGIS_AI\\Default.gdb"}},
        "description": "List all contents of the current workspace to verify available GIS files."
    }},
    {{
        "tool": "list_fields",
        "input": {{"layer": "placeholder_layer_name"}},
        "description": "List all fields in the specified layer; use the output to determine the correct attribute field."
    }}
]
"""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])


VERIFIER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """[Role and Task]
You are a GIS plan verifier with a detailed chain-of-thought reasoning process. Your task is to critically evaluate a GIS plan for errors, inconsistencies, and unjustified assumptions. In particular, you must identify if the plan:
- Assumes file names, attribute field names, or other necessary parameters without verification.
- Lacks steps to retrieve or verify required inputs (using tools like "list_fields", "scan_workspace_directory_for_gis_files", or "scan_external_directory_for_gis_files").
- Includes any redundant, illogical, or incomplete sequences of geoprocessing steps.

[Input Details]
You are provided with:
- Original User Request:
- Plan: A JSON array of steps. Each step is an object with exactly these keys:
    - "tool": The name of the tool to be executed.
    - "input": A dictionary of input parameters for that tool.
    - "description": A brief explanation of why the tool is used.
     
[Context Information: Tools and Files]
- Tool Descriptions: {tools}
- Workspace Inventory Files: {inventory}
- External Files Available: {external_files}

[Verification Checklist]
For each step in the plan, perform the following checks:
1. **Tool Validity:**  
   - Verify that the "tool" specified exists among the available tools.
2. **Input Parameter Accuracy:**  
   - Ensure that the "input" dictionary exactly matches the tool's required parameters.
3. **No Unverified Assumptions:**  
   - **File Names:** If a step uses a file name, ensure that the plan does not assume a default value. For example, if a step uses an input such as `"file": "data.shp"`, check that there is a preceding step that scans the directory (e.g., "scan_workspace_directory_for_gis_files") to verify the file name.  
   - **Attribute Fields:** If a step uses an attribute field (e.g., `"field": "atm"`), verify that the plan includes a step (using "list_fields") to confirm the correct field name.  
   - **Other Parameters:** Check for any other required inputs. If a parameter is missing or given as a placeholder (e.g., "to be decided by executor based on ... output"), flag it as a mistake.
   - **General Example:**  
     Instead of a step that states:  
     {{
        "tool": "calculate_field", 
        "input": {{"layer": "<layer_id>", "field": "some field name", "expression": "some_expression"}}, 
        "description": "Calculate ATM values."
     }}
     
     The plan should include a prior step like:  
     {{
        "tool": "list_fields", 
        "input": {{"layer": "<layer_id>"}}, 
        "description": "Retrieve field names for the layer to determine the correct field for ATM values."
     }}
     
4. **Logical Sequence and Completeness:**  
   - Confirm that the steps are arranged in a logical order and that dependencies (such as using output from one tool as input for a subsequent step) are properly addressed.
   - Verify that no essential step is missing to achieve the overall GIS task.
5. **Data Compatibility and Consistency:**  
   - Ensure that file formats, spatial references, and attribute types are appropriate for the intended operations.
   - Verify that if an analysis or index formula is required (e.g., for NDVI or MNDWI), the correct one is planned.
6. **Avoidance of Redundancy:**  
   - If the required data is already available in the workspace or external files, the plan must not include unnecessary steps for re-downloading or re-scanning unless explicitly requested.

[Process]
- Carefully review each step of the plan using the above checklist.
- Document your full internal chain-of-thought reasoning.
- Identify any mistakes, including assumptions or hallucinations where the plan does not verify required inputs.
- Ask if the field names are being assumed in the plan, if so, ask for the list of fields and their names to be added in the plan.
- Ask if the satellite data already exists in the workspace or the external files, if so, ask for it to be used instead of downloading it again.

[Output Format]
After completing your reasoning, output a JSON object with exactly two keys:
- "detailed_thought": A string containing your complete chain-of-thought reasoning.
- "validity": A string that is "valid" if the plan meets all criteria, or "invalid" if there are errors.  
The key "validity" must appear last in the JSON object.

Output ONLY the JSON object with no extra formatting or markdown.
"""),
    ("user", """
Original User Request:
{request}

Plan:
{plan}

Output:
Return a JSON object with exactly two keys:
- "detailed_thought": Your complete reasoning process.
- "validity": "valid" if the plan is correct; "invalid" if not.
""")
])



EXECUTOR_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """[Role and Task]
You are a GIS task executor with the ability to execute multiple tools in sequence. You MUST complete the ENTIRE PLAN by making all necessary tool calls one after another. NEVER STOP after just one tool call.

[Execution Guidelines]
1. You will receive a JSON array of steps. EXECUTE ALL OF THEM in order.
2. For each step:
   - Call the specified tool with the exact parameters from the plan
   - If a parameter contains a placeholder (e.g., "field name to be decided by executor based on list_fields output"), first call the appropriate tool to get the actual value
   - After completing a tool call, IMMEDIATELY CONTINUE to the next step without waiting for further instructions
   - DO NOT STOP after the first tool call - you must keep going until ALL steps are complete

3. When handling field values and string comparisons:
   - Make string comparisons case-insensitive by using UPPER() for SQL WHERE clauses
   - Use exact field names as returned by list_fields tools

4. CRITICALLY IMPORTANT: After each tool call, you MUST CONTINUE to the next step in the plan. Do not wait for confirmation to proceed.

5. Only after ALL steps have been executed, provide a final summary of what was accomplished.

Remember: Your MOST IMPORTANT directive is to KEEP MAKING TOOL CALLS until you've completed ALL steps in the plan. Never stop after just one step!
"""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "Here is the plan to execute: {input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])


# Custom callback handler for tracking tool usage
class ToolTrackingHandler(BaseCallbackHandler):
    def __init__(self):
        super().__init__()
        self.tool_starts = 0
        self.tool_ends = 0
        self.tools_used = []
        
    def on_tool_start(self, serialized, input_str, **kwargs):
        self.tool_starts += 1
        tool_name = serialized.get('name', 'unknown')
        self.tools_used.append(tool_name)
        print(f"🔧 Starting tool call #{self.tool_starts}: {tool_name}")
    
    def on_tool_end(self, output, **kwargs):
        self.tool_ends += 1
        print(f"✅ Completed tool call #{self.tool_ends}")
        
    def on_chain_start(self, serialized, inputs, **kwargs):
        if self.tool_starts > 0:
            print(f"🔄 Continuing execution after {self.tool_starts} tool calls...")
        
    def on_llm_start(self, serialized, prompts, **kwargs):
        if self.tool_starts > 0:
            # Inject a reminder to continue making tool calls if we've already started
            print(f"🔔 Reminder: Still need to complete all remaining steps in the plan!")


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
                                    timeout=300)
        
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
            max_iterations=10,
            early_stopping_method="generate",
        )
    

    
    def _create_verifier(self):
        return ChatGoogleGenerativeAI(model=self.model,
                                    google_api_key=self.api_key,
                                    temperature=0.0,
                                    timeout=300)
    
    def _create_executor(self):
        llm = ChatGoogleGenerativeAI(
            model=self.model_small,
            google_api_key=self.api_key,
            temperature=0.05,  # Slightly increased temperature to encourage exploration
            max_retries=15,
            timeout=600  # Increased timeout for longer operations
        )
        
        # Add memory for executor with a more specific configuration
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            input_key="input",
            output_key="output"
        )
        
        # Create a callback handler to track tool usage
        tool_tracker = ToolTrackingHandler()
        
        # Use the standard tools agent with enhanced configuration
        agent = create_openai_tools_agent(llm, self.tools, EXECUTOR_PROMPT)
        
        return AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            memory=memory,
            max_iterations=15,  # Increased max iterations to allow for more tool calls
            handle_parsing_errors=True,
            return_intermediate_steps=True,  # Return intermediate steps for better debugging
            max_execution_time=600,  # Set maximum execution time to 10 minutes
            callbacks=[tool_tracker],  # Add our custom callback handler
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
            
            # Use our dedicated function to clean the JSON string
            plan = clean_json_string(plan)
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
            
            # Send environment info to detailed output
            self.response_queue.put(f"Environment Info:\n{json.dumps(env_info, indent=2)}\n")
            
            current_iteration = 0
            verification_result = ""
            plan = None
            
            while current_iteration < max_iterations:
                iteration_header = f"\n--- ITERATION {current_iteration + 1} ---\n"
                print(iteration_header)
                self.response_queue.put(iteration_header)

                # Planning Phase
                planning_header = "\n1. PLANNING PHASE\n" + "="*40
                print(planning_header)
                self.response_queue.put("Planning...\n")
                self.response_queue.put(planning_header)
                
                planning_input = {
                    "input": user_input if current_iteration == 0 
                            else f"{user_input}\nPrevious verification feedback: {verification_result}",
                    "workspace": self.workspace,
                    "inventory": env_info["workspace_inventory"],
                    "external_files": env_info["external_directories"],
                    "tools": self.tool_descriptions,
                }
                
                # Send planning input to detailed output
                self.response_queue.put(f"Planning Input:\n{json.dumps(planning_input, indent=2)}\n")
                
                try:
                    plan_result = self.planner.invoke(planning_input)
                    plan = self._extract_plan(plan_result)
                    
                    # Send plan to detailed output
                    self.response_queue.put(f"Generated Plan:\n{json.dumps(json.loads(plan), indent=2)}\n")
                    
                except ValueError as e:
                    error_msg = f"Planning Error: {str(e)}"
                    print(error_msg)
                    self.response_queue.put(f"❌ {error_msg}\n")
                    self.response_queue.put(f"Archer:\n{error_msg}")
                    return ""
                
                time.sleep(2)  # Brief pause before verification

                # Verification Phase
                verification_header = "\n2. VERIFICATION PHASE\n" + "="*40
                print(verification_header)
                self.response_queue.put("Verifying...\n")
                self.response_queue.put(verification_header)
                
                verification_input = {
                    "plan": plan,
                    "request": user_input,
                    "tools": self.tool_descriptions,
                    "inventory": env_info["workspace_inventory"],
                    "external_files": env_info["external_directories"]
                }
                
                # Send verification input to detailed output
                self.response_queue.put(f"Verification Input:\n{json.dumps(verification_input, indent=2)}\n")
                
                verification_result = self.verifier.invoke(
                    VERIFIER_PROMPT.format_prompt(**verification_input)
                ).content
            
                # Clean the verification result using our dedicated function
                verification_result = clean_json_string(verification_result)

                # Process verification result as JSON
                try:
                    self.response_queue.put(f"Verification Result:\n{verification_result}\n")
                    
                    verification_json = json.loads(verification_result)
                    detailed_thought = verification_json.get("detailed_thought", "")
                    validity = verification_json.get("validity", "invalid")
                    
                    self.response_queue.put(f"Verification Thought:\n{detailed_thought}\n")
                    self.response_queue.put(f"Validity: {validity}\n")
                    
                    if validity.lower() == "valid":
                        valid_msg = "✅ Plan verified as valid!"
                        print(valid_msg)
                        self.response_queue.put(valid_msg + "\n")
                        break  # Exit the while loop if plan is valid
                    else:
                        invalid_msg = f"❌ Verification invalid: {detailed_thought}"
                        print(invalid_msg)
                        self.response_queue.put(invalid_msg + "\n")
                        
                        # Provide feedback to the planner for the next iteration
                        planning_input["previous_feedback"] = detailed_thought
                        current_iteration += 1  # Increment iteration for the next loop
                        
                        if current_iteration >= max_iterations:
                            max_iter_msg = "Maximum iterations reached. Planning failed."
                            print(max_iter_msg)
                            self.response_queue.put(f"❌ {max_iter_msg}\n")
                            self.response_queue.put(f"Archer:\n{max_iter_msg}")
                            return ""
                        
                        time.sleep(2)
                        continue  # Continue to next iteration
                        
                except json.JSONDecodeError as e:
                    json_error = f"Verification JSON decode error: {str(e)}"
                    print(json_error)
                    self.response_queue.put(f"❌ {json_error}\n")
                    self.response_queue.put(f"Archer:\nInvalid verification output format: {str(e)}")
                    return ""
            
            # Do not proceed to Execution Phase if the plan is invalid
            if validity.lower() != "valid":
                invalid_plan_msg = "Plan is invalid after maximum iterations. Execution phase skipped."
                self.response_queue.put(f"❌ {invalid_plan_msg}\n")
                self.response_queue.put(f"Archer:\n{invalid_plan_msg}")
                return ""
            
            # Execution Phase
            print("\n3. EXECUTION PHASE")
            
            # Pretty print the plan JSON for better readability
            try:
                # Parse the plan into a Python object if it's a string
                if isinstance(plan, str):
                    plan_obj = json.loads(plan)
                else:
                    plan_obj = plan
                    
                # Format the plan with nice indentation and colors
                formatted_plan = json.dumps(plan_obj, indent=2)
                
                # Add some decorative elements
                print("\n" + "="*80)
                print("📋 EXECUTION PLAN:")
                print("="*80)
                # Print each line with line numbers for easier reference
                for i, line in enumerate(formatted_plan.splitlines(), 1):
                    print(f"{i:3d} | {line}")
                print("="*80 + "\n")
            except Exception as e:
                # Fallback to simple printing if there's an error
                print(f"Plan JSON sent to Executor: {plan}")

            # Format the EXECUTOR_PROMPT to see the full input
            executor_input = EXECUTOR_PROMPT.format_prompt(input=plan, agent_scratchpad=[], chat_history=[])
            full_executor_input_content = executor_input.to_string()
            # print("\n--- Full Input to Executor Agent (Prompt + Plan) ---")
            # print(full_executor_input_content)
            # print("\n--- End of Full Input to Executor Agent ---")

            # Capture stdout during executor.invoke()
            captured_output = io.StringIO()
            try:
                print("\nStarting execution of all plan steps...")
                
                # Force a clear memory before execution to avoid conflicting history
                self.executor.memory.clear()
                
                with contextlib.redirect_stdout(captured_output):
                    # Add a specific message to signal that all steps must be executed
                    execution_result = self.executor.invoke({
                        "input": f"Execute ALL steps in this plan WITHOUT STOPPING. Complete EVERY tool call in sequence: {plan}"
                    })
                    
                # Process the output
                executor_output = captured_output.getvalue()
                executor_output_cleaned = self._remove_ansi_escape_codes(executor_output)
                
                # Check for completed steps in intermediate_steps
                if 'intermediate_steps' in execution_result:
                    completed_tools = [step[0].tool for step in execution_result['intermediate_steps']]
                    print(f"Completed tool calls: {', '.join(completed_tools)}")
                
                # Enhanced display of completed tools
                if 'intermediate_steps' in execution_result:
                    completed_tools = [step[0].tool for step in execution_result['intermediate_steps']]
                    
                    print("\n" + "="*80)
                    print("✅ COMPLETED TOOLS:")
                    print("="*80)
                    for i, tool_name in enumerate(completed_tools, 1):
                        print(f"{i:2d}. {tool_name}")
                    print("="*80)
                
                # First, send the execution output to show progress in the detailed tab
                self.response_queue.put(f"Executor Output:\n{executor_output_cleaned}\n")
                
                # Format and display the execution result
                result_output = execution_result.get('output', '')
                result_display = "\n" + "="*80 + "\n"
                result_display += "🏁 EXECUTION RESULT SUMMARY:\n"
                result_display += "="*80 + "\n"
                result_display += result_output + "\n"
                result_display += "="*80 + "\n"
                
                # Send formatted result to the GUI
                self.response_queue.put(result_display)
                
                # Ensure status is reset at the end of execution
                self.response_queue.put("STATUS_RESET:Awaiting Query:🤖")
                
                # Create the Archer message
                archer_message = f"Archer:\n{result_output}"
                
                # Add Archer message to the response queue
                self.response_queue.put(archer_message)
                
                # Return nothing since we've already queued the message
                return ""
            except Exception as exec_error:
                executor_output = captured_output.getvalue()
                executor_output_cleaned = self._remove_ansi_escape_codes(executor_output)
                
                # Check if we have any completed steps despite the error
                completed_steps = []
                if hasattr(self.executor, "_get_tool_returns") and hasattr(self.executor, "callbacks"):
                    try:
                        completed_steps = self.executor._get_tool_returns(self.executor.callbacks.handlers)
                        if completed_steps:
                            print(f"Completed {len(completed_steps)} tool calls before error")
                    except:
                        pass
                
                print(f"Execution error: {str(exec_error)}")
                error_message = f"Execution failed: {str(exec_error)}"
                
                # Add more details to the error message
                error_details = f"Execution failed after completing {len(completed_steps)} steps.\nError: {str(exec_error)}"
                traceback_str = traceback.format_exc()
                
                self.response_queue.put(f"Executor Output:\n{executor_output_cleaned}\nExecution error:\n{error_details}\n")
                
                # Create a properly formatted error message
                error_message = f"Execution failed: {str(exec_error)}"
                
                # Ensure the error message reaches the chat tab
                self.response_queue.put(f"Archer:\nError: {str(exec_error)}")
                
                # Return an empty string to avoid duplicating the message
                return ""
            
        except Exception as e:
            print(f"\nERROR: {str(e)}")
            print("Traceback:", traceback.format_exc())
            # Queue the error message instead of returning it
            self.response_queue.put(f"Archer:\nError processing request: {str(e)}")
            return ""
    
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
        self.root.title("ARCHER - GIS Agent - Powered by AI")
        self.root.geometry("1200x800")
        self.root.minsize(800, 600)
        
        # Set up variables
        self.workspace_var = tk.StringVar(value=self.settings_manager.settings["workspace"])
        self.button_colors = ["#4CAF50", "#5CBF60", "#6CCF70", "#7CDF80", "#8CEF90", "#7CDF80", "#6CCF70", "#5CBF60"]
        self.agent_thread = None  # Track the agent worker thread
        
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
            self.update_response_area("Performing initial scan of watched directories...", "detail", "info")
            scan_results = self.directory_manager.scan_all_directories(
                callback=lambda msg, _: self.update_response_area(msg, "detail", "info")
            )
            
            # Update tree view in main thread
            self.root.after(0, lambda: self._update_tree_with_results(scan_results))
        except Exception as e:
            self.update_response_area(f"Error in initial scan: {str(e)}", "detail", "error")
    
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
        self.root.title("ARCHER - GIS Agent - Powered by AI")
        
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
        
        # Create ttk style for buttons
        style = ttk.Style()
        style.configure("TButton", font=("Segoe UI", 10))
        style.configure("Accent.TButton", font=("Segoe UI", 10, "bold"))
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Main chat tab
        self.chat_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.chat_frame, text="GIS Assistant")
        
        # Environment tab
        self.env_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.env_frame, text="Environment")
        
        # Setup chat frame
        self.setup_chat_frame()
        
        # Setup environment frame
        self.setup_environment_frame()
    
    def setup_chat_frame(self):
        """Set up the chat interface inside the existing chat_frame"""
        # Note: self.chat_frame is already created in setup_gui
        
        # Create notebook for tabs inside the existing chat_frame
        self.chat_notebook = ttk.Notebook(self.chat_frame)
        self.chat_notebook.pack(fill=tk.BOTH, expand=True)
        
        # Simple Chat Tab
        self.simple_chat_tab = ttk.Frame(self.chat_notebook)
        self.chat_notebook.add(self.simple_chat_tab, text="Conversation")
        
        # Detailed Output Tab
        self.detailed_tab = ttk.Frame(self.chat_notebook)
        self.chat_notebook.add(self.detailed_tab, text="Detailed Output")
        
        # Chat area (simplified) in the Chat tab
        self.response_area = scrolledtext.ScrolledText(self.simple_chat_tab, wrap=tk.WORD, width=60, height=20, font=("Segoe UI", 10))
        self.response_area.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.response_area.config(state="disabled")
        self.response_area.tag_configure("user", foreground="#007acc", font=("Segoe UI", 10, "bold"))
        self.response_area.tag_configure("agent", foreground="#990000", font=("Segoe UI", 10, "bold"))
        self.response_area.tag_configure("info", foreground="#888888", font=("Segoe UI", 9, "italic"))
        self.response_area.tag_configure("error", foreground="#cc0000", font=("Segoe UI", 10, "bold"))
        
        # Detailed output area in the Details tab
        self.detailed_area = scrolledtext.ScrolledText(self.detailed_tab, wrap=tk.WORD, width=60, height=20, font=("Consolas", 9))
        self.detailed_area.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.detailed_area.config(state="disabled")
        # Configure tags for different types of output
        self.detailed_area.tag_configure("planner", foreground="#0066cc", font=("Consolas", 9, "bold"))
        self.detailed_area.tag_configure("verifier", foreground="#9933cc", font=("Consolas", 9, "bold"))
        self.detailed_area.tag_configure("executor", foreground="#cc6600", font=("Consolas", 9, "bold"))
        self.detailed_area.tag_configure("tool", foreground="#009933", font=("Consolas", 9))
        self.detailed_area.tag_configure("error", foreground="#cc0000", font=("Consolas", 9, "bold"))
        self.detailed_area.tag_configure("header", foreground="#000000", background="#f0f0f0", font=("Consolas", 10, "bold"))
        
        # Status indicator frame
        self.status_frame = ttk.Frame(self.chat_frame)
        self.status_frame.pack(fill=tk.X, padx=5, pady=(0, 5))
        
        # Status indicator label with icon
        self.status_icon_label = ttk.Label(self.status_frame, text="🔄")
        self.status_icon_label.pack(side=tk.LEFT, padx=(5, 2), pady=5)
        
        self.status_label = ttk.Label(self.status_frame, text="Ready")
        self.status_label.pack(side=tk.LEFT, padx=2, pady=5)
        
        # Input area with placeholder text
        self.input_frame = ttk.Frame(self.chat_frame)
        self.input_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.input_area = tk.Text(self.input_frame, wrap=tk.WORD, width=50, height=3, font=("Segoe UI", 10), padx=8, pady=8)
        self.input_area.pack(fill=tk.X, expand=True, side=tk.LEFT, padx=(0, 5))
        self.input_area.insert(tk.END, "Enter your GIS task here...")
        self.input_area.config(fg="grey")
        
        # Input bindings for placeholder text behavior
        self.input_area.bind("<FocusIn>", self.on_entry_click)
        self.input_area.bind("<FocusOut>", self.on_focus_out)
        self.input_area.bind("<Return>", self.submit_request)
        
        # Submit Button with some styling
        self.submit_button = ttk.Button(self.input_frame, text="Submit", command=self.submit_request, width=10)
        self.submit_button.pack(side=tk.RIGHT)
        self.submit_button.bind("<Enter>", self.animate_button)
        self.submit_button.bind("<Leave>", self.animate_button)
    
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
        
        # Show progress indicator with callback that sends updates to the detail tab
        progress = ScanProgressIndicator(self.root, 
            lambda msg: self.update_response_area(msg, "detail", "info")
        )
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
        """Clear placeholder text when input area is clicked."""
        # For the input area
        if event.widget == self.input_area:
            if self.input_area.get("1.0", tk.END).strip() == "Enter your GIS task here...":
                self.input_area.delete("1.0", tk.END)
                self.input_area.config(fg='black')
    
    def on_focus_out(self, event):
        """Restore placeholder text when focus is lost and the field is empty."""
        # For the input area
        if event.widget == self.input_area:
            if not self.input_area.get("1.0", tk.END).strip():
                self.input_area.delete("1.0", tk.END)  # Clear first to avoid duplicating
                self.input_area.insert("1.0", "Enter your GIS task here...")
                self.input_area.config(fg='grey')
    
    def animate_button(self, event):
        """Handle mouse hover animation for the submit button"""
        # Use ttk styling instead of direct color manipulation since we're using ttk.Button
        if event.type == tk.EventType.Enter:
            self.submit_button.config(style="Accent.TButton")
        else:
            self.submit_button.config(style="TButton")
    
    def submit_request(self, event=None):
        """Submit a request to the GIS agent for processing."""
        # Get the request text
        request = self.input_area.get("1.0", tk.END).strip()
        
        # Skip empty requests
        if not request and request != "Enter your GIS task here...":
            return
        
        # Add the request to the chat area with user styling
        self.update_response_area(f"You: {request}", "chat", "user")
        
        # Update detailed area with a header for the query
        self.update_response_area("\n" + "="*80, "detail", "header")
        self.update_response_area(f"📝 NEW QUERY: {request}", "detail", "header")
        self.update_response_area("="*80 + "\n", "detail", "header")
        
        # Start the agent thread if it's not already running
        self.start_agent_thread()
        
        # Clear the input area for the next request
        self.input_area.delete("1.0", tk.END)
        
        # Update status
        self.update_status("Processing your request...", "⏳")
        
        # Add the request to the queue for processing
        self.request_queue.put(request)
        
        return "break"  # Prevents the default Return key behavior
    
    def update_response_area(self, message: str, area_type="chat", message_type="agent"):
        """Update one of the response areas with a message.
        
        Args:
            message: The message to display.
            area_type: 'chat' for the chat area, 'detail' for the technical details area.
            message_type: 'user', 'agent', 'info', 'error', 'planner', 'verifier', 'executor', 'header'.
        """
        # Properly identify Archer messages to ensure they're styled correctly
        if message and isinstance(message, str) and message.startswith("Archer:"):
            message_type = "agent"  # Force agent styling for Archer messages
        
        # Defer the update to the main thread
        def _update():
            # Update appropriate area based on type
            target_area = self.response_area if area_type == "chat" else self.detailed_area

            # Insert the message
            target_area.config(state=tk.NORMAL)
            
            # Apply different formats based on message type
            if message_type == "user":
                target_area.insert(tk.END, message + "\n\n", "user")
            elif message_type == "agent":
                # Format agent messages with some styling
                target_area.insert(tk.END, message + "\n\n", "agent")
            elif message_type == "error":
                target_area.insert(tk.END, message + "\n\n", "error")
            elif message_type in ["planner", "verifier", "executor"]:
                # Format different agent components with specific styling
                target_area.insert(tk.END, message + "\n", message_type)
            elif message_type == "header":
                # Format headers with distinct styling
                target_area.insert(tk.END, message + "\n", "header")
            else:
                # Default formatting for information messages
                target_area.insert(tk.END, message + "\n", "info")
            
            # Scroll to the end
            target_area.see(tk.END)
            target_area.config(state=tk.DISABLED)
            
        # Schedule the update
        self.root.after(10, _update)
    
    def update_status(self, status, icon="🔄"):
        """Update the status indicator with the current operation."""
        def _update():
            self.status_icon_label.config(text=icon)
            self.status_label.config(text=status)
        
        self.root.after(10, _update)
    
    def process_responses(self):
        """Process responses from the agent and route them to the appropriate display areas."""
        try:
            # Keep track of execution status to ensure messages appear in correct order
            showing_execution = False
            archer_message = None
            execution_completed_shown = False
            
            # Process all messages in the queue
            while not self.gis_agent.response_queue.empty():
                message = self.gis_agent.response_queue.get_nowait()
                
                # Allow through important planning and verification outputs, filter very verbose ones
                if any(keyword in message for keyword in [
                    "Environment Info:", "Verification Input:", "verification_json", 
                    "THOUGHT:", "Raw plan", "Cleaned plan", "Parsed JSON:",
                    "Maximum iterations reached", "Planning Input:", "Verification Thought:"
                ]):
                    # These are too verbose - don't show them
                    continue
                
                # Show important planning outputs with better formatting
                if message.startswith("Generated Plan:"):
                    cleaned_message = message.replace("Generated Plan:", "GENERATED PLAN:")
                    self.update_response_area(cleaned_message, "detail", "planner")
                    continue
                
                # Show important planning outputs with better formatting
                if message.startswith("Planning Input:"):
                    cleaned_message = message.replace("Planning Input:", "PLANNING INPUT:")
                    self.update_response_area(cleaned_message, "detail", "planner")
                    continue
                
                # Show important verification outputs with better formatting
                if message.startswith("Verification Result:"):
                    cleaned_message = message.replace("Verification Result:", "VERIFICATION RESULT:")
                    self.update_response_area(cleaned_message, "detail", "verifier")
                    continue
                
                # Determine message type and target area based on content
                if message.startswith("Planning..."):
                    self.update_status("Planning...", "🧠")
                    self.update_response_area("Agent is planning...", "chat", "info")
                    self.update_response_area("\n--- PLANNING PHASE ---", "detail", "planner")
                
                elif message.startswith("Verifying..."):
                    self.update_status("Verifying plan...", "🔍")
                    self.update_response_area("Agent is verifying the plan...", "chat", "info")
                    self.update_response_area("\n--- VERIFICATION PHASE ---", "detail", "verifier")
                
                # Show important planning/verification outputs in detailed tab
                elif "PLANNING PHASE" in message:
                    self.update_response_area(message, "detail", "planner")
                
                elif "VERIFICATION PHASE" in message:
                    self.update_response_area(message, "detail", "verifier")
                    
                elif "Plan verified" in message:
                    self.update_response_area(message, "detail", "verifier")
                    
                elif "Validating step" in message or "validated successfully" in message:
                    self.update_response_area(message, "detail", "verifier")
                    
                elif "Plan is invalid" in message or "Validity:" in message:
                    self.update_response_area(message, "detail", "verifier")
                
                elif message.startswith("Executor Output:"):
                    showing_execution = True
                    self.update_status("Executing plan...", "⚙️")
                    self.update_response_area("Agent is executing the plan...", "chat", "info")
                    formatted_message = message.replace("Executor Output:", "\n--- EXECUTION PHASE ---")
                    self.update_response_area(formatted_message, "detail", "executor")
                
                elif message.startswith("Archer:"):
                    # Store Archer message to display after execution message if needed
                    archer_message = message
                    
                    # Only display immediately if we've already shown the execution message
                    if showing_execution:
                        self.update_status("Awaiting Query", "🤖")  # Reset to awaiting status
                        self.update_response_area(archer_message, "chat", "agent")
                        
                        # Only show the execution completed message once
                        if not execution_completed_shown:
                            self.update_response_area("\n--- EXECUTION COMPLETED ---", "detail", "executor")
                            execution_completed_shown = True
                        
                        archer_message = None  # Reset the stored message
                    
                    # Otherwise wait to display it later in the correct order
                
                elif message.startswith("Execution completed:"):
                    # Convert to Archer message and store to display in correct order
                    archer_message = message.replace("Execution completed:", "Archer:")
                    
                    # Only display immediately if we've already shown the execution message
                    if showing_execution:
                        self.update_status("Awaiting Query", "🤖")  # Reset to awaiting status
                        self.update_response_area(archer_message, "chat", "agent")
                        
                        # Only show the execution completed message once
                        if not execution_completed_shown:
                            self.update_response_area("\n--- EXECUTION COMPLETED ---", "detail", "executor")
                            execution_completed_shown = True
                        
                        archer_message = None  # Reset the stored message
                    
                    # Otherwise wait to display it later in the correct order
                
                elif message.startswith("🏁 EXECUTION RESULT SUMMARY:"):
                    # This is just for the detailed tab, not the chat tab
                    self.update_response_area(message, "detail", "executor")
                
                elif message.startswith("Execution failed:") or "error" in message.lower():
                    self.update_status("Error occurred", "❌")
                    clean_message = message.replace("Execution failed:", "Error:")
                    # Error messages are now handled directly in agent_worker for chat tab
                    self.update_response_area("\n--- EXECUTION ERROR ---\n" + message, "detail", "error")
                
                # Route directory scanning and initialization messages to Technical Details tab
                elif any(keyword in message for keyword in [
                    "Scanning directory", "Scan complete", "Found", 
                    "Successfully processed", "Directory scanning", 
                    "Performing initial scan", "COMPLETED TOOLS"
                ]):
                    self.update_response_area(message, "detail", "info")
                    # Optionally update status
                    if "initial scan" in message:
                        self.update_status("Scanning directories...", "🔎")
                    elif "completed" in message:
                        self.update_status("Ready", "✅")
                
                # Handle special status reset message
                elif message.startswith("STATUS_RESET:"):
                    # Format: STATUS_RESET:Status Text:Emoji
                    parts = message.split(":", 2)
                    if len(parts) >= 3:
                        status_text = parts[1]
                        emoji = parts[2]
                        self.update_status(status_text, emoji)
                
                else:
                    # Default to detailed area for any other messages
                    self.update_response_area(message, "detail", "info")
            
            # If we have an Archer message but haven't displayed the execution message yet,
            # we need to ensure the execution message appears first
            if archer_message and not showing_execution:
                self.update_status("Executing plan...", "⚙️")
                self.update_response_area("Agent is executing the plan...", "chat", "info")
                
                # Then display the Archer message
                self.update_status("Awaiting Query", "🤖")
                self.update_response_area(archer_message, "chat", "agent")
                
                # Only show the execution completed message once
                if not execution_completed_shown:
                    self.update_response_area("\n--- EXECUTION COMPLETED ---", "detail", "executor")
                    execution_completed_shown = True
                
        except Exception as e:
            print(f"Error processing responses: {str(e)}")
            self.update_response_area(f"Error processing responses: {str(e)}", "detail", "error")
        
        # Check again after 100ms
        self.root.after(100, self.process_responses)
    
    def start_agent_thread(self):
        """Start the agent worker thread if it's not already running"""
        # Only start a new thread if there isn't one or the existing one is dead
        if self.agent_thread is None or not self.agent_thread.is_alive():
            def agent_worker():
                while True:
                    request = self.request_queue.get()
                    if request.lower() == "/exit":
                        self.response_queue.put("Exiting...")
                        break
                    
                    try:
                        # Get the result from the agent
                        result = self.gis_agent.process_request(request)
                        
                        # The process_request method already adds the Archer message to the queue
                        # We don't need to do anything else here to avoid duplication
                        
                    except Exception as e:
                        error_msg = f"Error processing request: {str(e)}"
                        print(error_msg)
                        self.response_queue.put(error_msg)
                        self.update_response_area(error_msg, "chat", "error")
                    finally:
                        self.request_queue.task_done()
            
            self.agent_thread = threading.Thread(target=agent_worker, daemon=True)
            self.agent_thread.start()
            print("Agent thread started.")
            
        # Always make sure we're processing responses
        self.process_responses()
    
    def run(self):
        # Start processing responses from the agent
        self.process_responses()
        
        # Enter the main event loop
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

def clean_json_string(json_str):
    """
    Clean a string to make it valid JSON.
    
    Args:
        json_str (str): The JSON string to clean
        
    Returns:
        str: A cleaned JSON string that can be parsed
    """
    # Try a series of increasingly aggressive cleaning approaches
    
    # First, basic cleaning
    cleaned = _basic_json_cleaning(json_str)
    
    # Test if it's valid JSON
    try:
        json.loads(cleaned)
        return cleaned
    except json.JSONDecodeError:
        # If basic cleaning failed, try more aggressive approaches
        pass
    
    # Try more aggressive cleaning
    cleaned = _aggressive_json_cleaning(cleaned)
    
    # Final fallback - if all else fails, attempt to extract just the JSON object/array
    try:
        json.loads(cleaned)
        return cleaned
    except json.JSONDecodeError:
        return _extract_json_pattern(cleaned)

def _basic_json_cleaning(json_str):
    """Basic JSON cleaning - handles common issues"""
    # Remove markdown code block syntax
    json_str = json_str.replace("```json", "").replace("```", "").strip()
    
    # Remove control characters
    json_str = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]', '', json_str)
    
    # Replace actual newlines and tabs with spaces
    json_str = re.sub(r'[\n\t\r]+', ' ', json_str)
    
    return json_str

def _aggressive_json_cleaning(json_str):
    """More aggressive JSON cleaning for problematic strings"""
    # Handle unescaped backslashes
    json_str = re.sub(r'([^\\])\\([^\\/"bfnrtu])', r'\1\\\\\2', json_str)
    
    # Fix unescaped quotes within string values
    json_str = re.sub(r'(":.*?)([^\\])"(.*?")', r'\1\2\\\"\3', json_str)
    
    # Attempt to fix unbalanced quotes
    open_quotes = 0
    fixed_str = ""
    for char in json_str:
        if char == '"' and (not fixed_str or fixed_str[-1] != '\\'):
            open_quotes += 1
            if open_quotes % 2 == 0:
                # This is a closing quote
                fixed_str += char
            else:
                # This is an opening quote
                fixed_str += char
        else:
            fixed_str += char
            
    # If we have unbalanced quotes at the end, add a closing quote
    if open_quotes % 2 != 0:
        fixed_str += '"'
    
    return fixed_str

def _extract_json_pattern(text):
    """
    Extract valid JSON object or array from text.
    This is a last resort when other cleaning methods fail.
    """
    # Try to find JSON object pattern
    obj_match = re.search(r'(\{.*\})', text)
    if obj_match:
        extracted = obj_match.group(1)
        try:
            # Test if it's valid
            json.loads(extracted)
            return extracted
        except:
            pass
    
    # Try to find JSON array pattern
    arr_match = re.search(r'(\[.*\])', text)
    if arr_match:
        extracted = arr_match.group(1)
        try:
            # Test if it's valid
            json.loads(extracted)
            return extracted
        except:
            pass
    
    # If we get here, nothing worked - return the original with basic cleaning
    return text

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