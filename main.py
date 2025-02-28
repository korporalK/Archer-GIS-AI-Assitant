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
from tools import (
    add_field,append_features,aspect,buffer_features,calculate_field,clip_features,
    closest_facility,create_feature_class,create_file_geodatabase,dataset_exists,
    define_projection,delete_features,describe_dataset,dissolve_features,
    download_landsat_tool,erase_features,export_to_csv,extract_by_mask,
    get_environment_settings,get_workspace_inventory,hillshade,import_csv,
    intersect_features,list_fields,merge_features,project_features,
    reclassify_raster,repair_geometry,route,select_features,service_area,
    slope,spatial_join,union_features,zonal_statistics,
    scan_directory_for_gis_files
)

# Environment Setup
def setup_environment():
    load_dotenv(r"D:\masters_project\ArcGIS_AI\.env")
    
    API_KEY = os.getenv("GEMINI_API_KEY")
    if not API_KEY:
        raise ValueError("Please set the GEMINI_API_KEY environment variable.")
    
    # Enable overwriteOutput by default
    arcpy.env.overwriteOutput = True
    
    # Check ArcPy license
    if not arcpy.CheckProduct("ArcInfo"):
        raise RuntimeError("Advanced license required")
    
    return API_KEY

# Prompts
PLANNER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful GIS planning assistant. Your task is to develop a step-by-step plan to solve a GIS problem using ONLY the provided tools.
You must output a JSON array of steps. Each step must be an object with:
  - tool: The exact name of one of the available tools (no other functions allowed).
  - input: A dictionary of input parameters exactly matching the tool's required parameters.
  - description: A clear explanation of why this tool is used in this step.

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
6. DO NOT ASSUME FIELD NAME, ALWAYS CHECK USING 'list_fields' tool and provide a placeholder.

Do not include any markdown formatting or code blocks; output ONLY the JSON array.

Example format:
[
    {{
        "tool": "get_workspace_inventory",
        "input": {{"workspace": "D:\\masters_project\\ArcGIS_AI\\Default.gdb"}},
        "description": "List all contents of the current workspace."
    }}
]

Available tools and their descriptions:
{tools}
    
    Current workspace: {workspace}
Workspace inventory: {inventory}
External files available: {external_files}
    
Output ONLY the JSON array.
"""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

VERIFIER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a GIS plan verifier with a chain-of-thought reasoning process. Your task is to verify a GIS plan for errors with detailed internal reasoning.
    
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
   - If any step requires an attribute field (or similar parameter) that cannot be predetermined, verify that the plan includes a step to list or examine fields and a clear placeholder indicating that the executor will determine the appropriate field from the list.
   - When using external files, verify proper import steps are included
   - Confirm data source locations (workspace vs external) are correctly handled
   
2. As you analyze the plan, produce a detailed chain-of-thought that captures your reasoning process.
3. After completing your reasoning, output a JSON object with exactly two keys:
   - "detailed_thought": Containing the full chain-of-thought reasoning.
   - "validity": A final verdict that is either "valid" if the plan is acceptable, or "invalid" if the plan is incorrect. This key must appear last in the JSON object.

Do not output any markdown or additional formatting; output only the JSON object.
"""),
    ("user", """
Original User Request:
{request}

Tool Descriptions:
{tools}

Plan:
{plan}

Workspace Inventory:
{inventory}

External Files:
{external_files}

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
4. After executing all steps, provide a final summary indicating the overall success or failure of the plan execution.

Output:
- Should include relevant tool calls.
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

        self.tools = [
            add_field,append_features,aspect,buffer_features,calculate_field,
            clip_features,closest_facility,create_feature_class,create_file_geodatabase,
            dataset_exists,define_projection,delete_features,describe_dataset,dissolve_features,
            download_landsat_tool,erase_features,export_to_csv,extract_by_mask,
            get_environment_settings,get_workspace_inventory,hillshade,import_csv,
            intersect_features,list_fields,merge_features,project_features,
            reclassify_raster,repair_geometry,route,select_features,service_area,
            slope,spatial_join,union_features,zonal_statistics,
        ]
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
            workspace_inventory = get_workspace_inventory.invoke(self.workspace)
            
            # Get external file inventory - always scan all directories fresh
            external_files = {}
            for directory in self.settings_manager.settings["watched_directories"]:
                try:
                    print(f"Scanning directory: {directory}")
                    scan_result = json.loads(scan_directory_for_gis_files(directory))
                    external_files[directory] = scan_result
                except Exception as e:
                    print(f"Error scanning directory {directory}: {str(e)}")
                    external_files[directory] = {"vector_files": [], "raster_files": []}
            
            # Cache and return the results
            self._environment_info = {
                "workspace": self.workspace,
                "workspace_inventory": workspace_inventory,
                "external_directories": external_files
            }
            return self._environment_info
        except Exception as e:
            print(f"Error gathering environment info: {str(e)}")
            return {}

    def refresh_environment_info(self):
        """Force a refresh of the environment information."""
        return self.get_environment_info(force_refresh=True)

    def process_request(self, user_input: str, max_iterations: int = 3) -> str:
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


            # except Exception as exec_error:
            #     print(f"Execution error: {str(exec_error)}")
            #     # Try to execute the plan directly without the agent
            #     # try:
            #     #     results = []
            #     #     parsed_plan = json.loads(plan)
            #     #     for step in parsed_plan:
            #     #         tool_name = step["tool"]
            #     #         tool = next((t for t in self.tools if t.name == tool_name), None)
            #     #         if tool:
            #     #             result = tool.invoke(step["input"])
            #     #             results.append(f"Step '{tool_name}' result: {result}")
            #     #     return "\n\n".join(results)
            #     # except Exception as direct_error:
            #     #     return f"Execution failed: {str(direct_error)}"
            
        except Exception as e:
            print(f"\nERROR: {str(e)}")
            print("Traceback:", traceback.format_exc())
            return f"Error processing request: {str(e)}"

# GUI Class
class GISGUI:
    def __init__(self, gis_agent: GISAgent):
        self.gis_agent = gis_agent
        self.request_queue = queue.Queue()
        self.response_queue = queue.Queue()
        self.gis_agent.response_queue = self.response_queue
        
        self.root = tk.Tk()
        self.settings_manager = SettingsManager()
        
        # Initialize ArcPy environment settings
        arcpy.env.workspace = self.settings_manager.settings["workspace"]
        arcpy.env.overwriteOutput = True
        
        self.setup_gui()
        self.start_agent_thread()
        
        # Initial environment scan
        self.gis_agent.refresh_environment_info()
    
    def setup_gui(self):
        self.root.title("ArcGIS AI Assistant")
        
        # Set window dimensions and position
        screen_width = self.root.winfo_screenwidth()
        screen_height = int(self.root.winfo_screenheight()) - 80
        window_width = 500
        window_height = screen_height
        x_position = screen_width - window_width
        y_position = 0
        
        self.root.geometry(f"{window_width}x{window_height}+{x_position}+{y_position}")
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
        self.button_colors = ["#4CAF50", "#2196F3", "#f44336", "#FFC107"]
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
        # Workspace section
        workspace_frame = ttk.LabelFrame(self.env_frame, text="Workspace")
        workspace_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.workspace_var = tk.StringVar(value=self.settings_manager.settings["workspace"])
        ttk.Entry(workspace_frame, textvariable=self.workspace_var, state='readonly').pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5, pady=5)
        ttk.Button(workspace_frame, text="Browse", command=self.browse_workspace).pack(side=tk.RIGHT, padx=5, pady=5)
        
        # Watched directories section
        directories_frame = ttk.LabelFrame(self.env_frame, text="Watched Directories")
        directories_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Directory list
        self.dir_listbox = tk.Listbox(directories_frame, selectmode=tk.SINGLE)
        self.dir_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Directory buttons
        dir_buttons_frame = ttk.Frame(directories_frame)
        dir_buttons_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=5)
        
        ttk.Button(dir_buttons_frame, text="Add", command=self.add_directory).pack(fill=tk.X, pady=2)
        ttk.Button(dir_buttons_frame, text="Remove", command=self.remove_directory).pack(fill=tk.X, pady=2)
        ttk.Button(dir_buttons_frame, text="Scan", command=self.scan_directories).pack(fill=tk.X, pady=2)
        
        # Files section with scrollbars
        files_frame = ttk.LabelFrame(self.env_frame, text="Available GIS Files")
        files_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create a frame to hold the tree and scrollbars
        tree_frame = ttk.Frame(files_frame)
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Define all possible columns
        columns = ("Name", "Type", "Path", "Driver", "CRS", "Feature_Count", "Dimensions", "Bands")
        
        # Create tree view for files with all columns
        self.files_tree = ttk.Treeview(tree_frame, columns=columns, show="tree headings", selectmode="browse")
        
        # Configure column widths and stretching
        self.files_tree.column("#0", width=200, minwidth=150, stretch=False)  # Tree column (directory structure)
        column_widths = {
            "Name": (200, 100),
            "Type": (100, 80),
            "Path": (300, 200),
            "Driver": (100, 80),
            "CRS": (100, 50),
            "Feature_Count": (100, 50),
            "Dimensions": (100, 50),
            "Bands": (70, 50),
        }
        
        # Configure all columns with initial widths and no stretching
        for col, (width, minwidth) in column_widths.items():
            self.files_tree.column(col, width=width, minwidth=minwidth, stretch=False)
            self.files_tree.heading(col, text=col.replace("_", " "), anchor=tk.W)
        
        # Add vertical scrollbar
        vsb = ttk.Scrollbar(tree_frame, orient="vertical", command=self.files_tree.yview)
        self.files_tree.configure(yscrollcommand=vsb.set)
        
        # Add horizontal scrollbar
        hsb = ttk.Scrollbar(tree_frame, orient="horizontal", command=self.files_tree.xview)
        self.files_tree.configure(xscrollcommand=hsb.set)
        
        # Grid layout for tree and scrollbars
        self.files_tree.grid(column=0, row=0, sticky='nsew')
        vsb.grid(column=1, row=0, sticky='ns')
        hsb.grid(column=0, row=1, sticky='ew')
        
        # Configure grid weights
        tree_frame.grid_columnconfigure(0, weight=1)
        tree_frame.grid_rowconfigure(0, weight=1)
        
        # Load existing directories
        self.load_watched_directories()
    
    def browse_workspace(self):
        """Browse and select only ArcGIS Geodatabase (.gdb) as workspace."""
        # Show initial message to user
        tk.messagebox.showinfo(
            "Workspace Selection",
            "Please select an ArcGIS Geodatabase (.gdb) folder as your workspace.\nNote: Select the .gdb folder itself, not its contents."
        )
        
        # Use askdirectory to allow selecting the .gdb folder
        workspace = filedialog.askdirectory(
            title="Select ArcGIS Geodatabase (.gdb) folder",
            initialdir=os.path.dirname(self.workspace_var.get())  # Start from current workspace directory
        )
        
        if workspace:
            # Check if selected folder is a .gdb or if parent is a .gdb
            if not workspace.lower().endswith('.gdb'):
                # Check if we're inside a .gdb folder
                parent_dir = os.path.dirname(workspace)
                if parent_dir.lower().endswith('.gdb'):
                    workspace = parent_dir
                else:
                    tk.messagebox.showerror(
                        "Invalid Workspace",
                        "Selected folder is not an ArcGIS Geodatabase. Please select a folder with .gdb extension."
                    )
                    return
            
            try:
                # Validate using arcpy
                if not arcpy.Exists(workspace):
                    tk.messagebox.showerror(
                        "Invalid Workspace",
                        "Selected geodatabase is not valid or cannot be accessed."
                    )
                    return
                
                # Update workspace if all validations pass
                self.workspace_var.set(workspace)
                self.settings_manager.set_workspace(workspace)
                self.gis_agent.workspace = workspace
                # Update ArcPy environment settings
                self.update_arcpy_environment()
                # Refresh environment info
                self.gis_agent.refresh_environment_info()
                
                tk.messagebox.showinfo(
                    "Workspace Updated",
                    f"Workspace successfully set to:\n{workspace}"
                )
            
            except Exception as e:
                tk.messagebox.showerror(
                    "Error",
                    f"Error setting workspace: {str(e)}"
                )
    
    def update_arcpy_environment(self):
        """Update ArcPy environment settings when workspace changes."""
        arcpy.env.workspace = self.workspace_var.get()
        arcpy.env.overwriteOutput = True  # Always enable overwrite
    
    def add_directory(self):
        directory = filedialog.askdirectory(title="Select Directory to Watch")
        if directory:
            self.settings_manager.add_directory(directory)
            self.load_watched_directories()
            # Force a complete refresh of environment info when adding directory
            self.gis_agent._environment_info = {}  # Clear the cache completely
            self.gis_agent.refresh_environment_info()
            # Automatically scan the directories after adding a new one
            self.scan_directories()
    
    def remove_directory(self):
        selection = self.dir_listbox.curselection()
        if selection:
            directory = self.dir_listbox.get(selection[0])
            self.settings_manager.remove_directory(directory)
            self.load_watched_directories()
            # Force a complete refresh of environment info when removing directory
            self.gis_agent._environment_info = {}  # Clear the cache completely
            self.gis_agent.refresh_environment_info()
            # Automatically update the tree view after removing a directory
            self.scan_directories()
    
    def load_watched_directories(self):
        self.dir_listbox.delete(0, tk.END)
        for directory in self.settings_manager.settings["watched_directories"]:
            self.dir_listbox.insert(tk.END, directory)
    
    def scan_directories(self):
        self.files_tree.delete(*self.files_tree.get_children())
        
        # Force a complete refresh of environment info before scanning
        self.gis_agent._environment_info = {}  # Clear the cache completely
        env_info = self.gis_agent.refresh_environment_info()
        
        # Use the refreshed external directories info to update the tree
        for directory, result in env_info.get("external_directories", {}).items():
            dir_node = self.files_tree.insert("", "end", text=directory)
            
            if result.get("vector_files"):
                vector_node = self.files_tree.insert(dir_node, "end", text="Vector Files")
                for file in result["vector_files"]:
                    # Prepare vector file values with placeholders for raster-specific columns
                    values = (
                        file["name"],
                        file["type"],
                        file["path"],
                        file["driver"],
                        file.get("crs", "-"),
                        str(file.get("layer_count", "-")),
                        "-",  # Dimensions (raster-specific)
                        "-",  # Bands (raster-specific)

                    )
                    self.files_tree.insert(vector_node, "end", text="", values=values)
            
            if result.get("raster_files"):
                raster_node = self.files_tree.insert(dir_node, "end", text="Raster Files")
                for file in result["raster_files"]:
                    # Prepare raster file values with placeholders for vector-specific columns
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
                    self.files_tree.insert(raster_node, "end", text="", values=values)
    
    def on_entry_click(self, event):
        if self.input_box.get("1.0", tk.END).strip() == "Enter your GIS request":
            self.input_box.delete("1.0", tk.END)
            self.input_box.config(fg='black')
    
    def on_focus_out(self, event):
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
        self.response_area.configure(state='normal')
        self.response_area.insert(tk.END, message + "\n")
        self.response_area.see(tk.END)
        self.response_area.configure(state='disabled')
    
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

def main():
    # Setup
    api_key = setup_environment()

    # Create settings manager
    settings_manager = SettingsManager()
    
    # Create response queue for agent and GUI to communicate
    response_queue = queue.Queue()

    # Create GIS Agent, passing both response queue and settings manager
    gis_agent = GISAgent(api_key, settings_manager.settings["workspace"], response_queue, settings_manager)

    # Create and run GUI, passing the agent
    gui = GISGUI(gis_agent)
    gui.run()

if __name__ == "__main__":
    main() 