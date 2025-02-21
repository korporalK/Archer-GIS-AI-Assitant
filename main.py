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
from tkinter import scrolledtext
import time
import inspect
import arcpy
import traceback

# Import tools properly
from tools import (
    add_field,
    append_features,
    aspect,
    buffer_features,
    calculate_field,
    clip_features,
    closest_facility,
    create_feature_class,
    create_file_geodatabase,
    dataset_exists,
    define_projection,
    delete_features,
    describe_dataset,
    dissolve_features,
    download_landsat_tool,
    erase_features,
    export_to_csv,
    extract_by_mask,
    get_environment_settings,
    get_workspace_inventory,
    hillshade,
    import_csv,
    intersect_features,
    list_fields,
    merge_features,
    project_features,
    reclassify_raster,
    repair_geometry,
    route,
    select_features,
    service_area,
    slope,
    spatial_join,
    union_features,
    zonal_statistics
)

# Environment Setup
def setup_environment():
    load_dotenv(r"D:\masters_project\ArcGIS_AI\.env")
    
    API_KEY = os.getenv("GEMINI_API_KEY")
    if not API_KEY:
        raise ValueError("Please set the GEMINI_API_KEY environment variable.")
    
    WORKSPACE = r"D:\masters_project\ArcGIS_AI\Default.gdb"
    
    # Add ArcPy environment checks
    if not arcpy.Exists(WORKSPACE):
        raise ValueError(f"Workspace does not exist: {WORKSPACE}")
    
    arcpy.env.workspace = WORKSPACE
    arcpy.env.overwriteOutput = True
    
    # Check ArcPy license
    if not arcpy.CheckProduct("ArcInfo"):
        raise RuntimeError("Advanced license required")
    
    return API_KEY, WORKSPACE

# Prompts
PLANNER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful GIS assistant. Your task is to create a plan to solve a GIS problem, using ONLY the provided tools.
            You MUST output a JSON array of steps. Each step MUST have:
              - tool: The exact name of one of the available tools (no other functions allowed)
              - input: A dictionary of input parameters exactly matching the tool's required parameters
              - description: Why you are using this tool in this step
            
            DO NOT include any markdown formatting or code blocks. Output ONLY the JSON array.

            Example format:
            [
                {{
                    "tool": "get_workspace_inventory",
                    "input": {{"workspace": "D:\masters_project\ArcGIS_AI\Default.gdb"}},
                    "description": "List all contents of the current workspace"
                }}
            ]
            
            Available tools and their descriptions:
            {tools}
            
            Current workspace: {workspace}
            Current inventory: {inventory}
            
            DO NOT include any markdown formatting or code blocks. Output ONLY the JSON array."""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

VERIFIER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a GIS plan verifier. Your task is to check a plan for errors.

    Input:
    - user_request: The original GIS task.
    - plan: A JSON array of steps. Each step has `tool`, `input`, and `description`.
    - tool_descriptions: A list of available tools and descriptions.
    - workspace_inventory: Inventory of the current GIS workspace.

    Output:
    Return a SINGLE STRING.
    - If the plan is valid, ONLY return the string "valid".
    - If the plan is invalid, return a string describing the error.  Do NOT return JSON.

    Checks:
    1. Each 'tool' in the plan must be one of the available tools.
    2. Each 'input' must contain all *required* parameters for the specified tool.
    3. Basic type checking (string, number) for input parameters based on tool descriptions.
    4. Does the order of steps make sense in a GIS context?
    5. Are the outputs of previous steps correctly used as inputs for subsequent steps?
    6. Are there any missing steps that are necessary to achieve the overall goal?
    7. Consider potential data issues (e.g., incorrect spatial reference, data existence).
    8. Avoid redundancy, in context of data as well as geoprocessing steps. 
    9. If data already exists then don't download it again.
    10. Fields for selection exist? If not get the fields first.

    Examples:
    Valid plan: "valid"
    Invalid plan: "Step 1: Missing 'buffer_unit'"
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

     Output:
     Return a SINGLE STRING.
    - If the plan is valid, ONLY return the string "valid".
    - If the plan is invalid, return a string describing the error.  Do NOT return JSON.
     """)
])

EXECUTOR_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a GIS task executor. Your role is to execute the provided plan using the available tools.
Each step in the plan contains:
- tool: The name of the tool to execute
- input: The parameters for the tool
- description: Why this step is needed

Execute each step and report the results. If a step fails, provide the error message.
Use the execution history to understand what steps have been completed and their results."""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "Here is the plan to execute: {input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

class GISAgent:
    def __init__(self, api_key: str, workspace: str):
        self.api_key = api_key
        self.workspace = workspace
        self.model = "gemini-2.0-pro-exp-02-05"
        
        self.tools = [
            add_field,
            append_features,
            aspect,
            buffer_features,
            calculate_field,
            clip_features,
            closest_facility,
            create_feature_class,
            create_file_geodatabase,
            dataset_exists,
            define_projection,
            delete_features,
            describe_dataset,
            dissolve_features,
            download_landsat_tool,
            erase_features,
            export_to_csv,
            extract_by_mask,
            get_environment_settings,
            get_workspace_inventory,
            hillshade,
            import_csv,
            intersect_features,
            list_fields,
            merge_features,
            project_features,
            reclassify_raster,
            repair_geometry,
            route,
            select_features,
            service_area,
            slope,
            spatial_join,
            union_features,
            zonal_statistics,
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
                                    temperature=0.0)
        agent = create_openai_tools_agent(llm, self.tools, PLANNER_PROMPT)
        memory = ConversationBufferMemory(memory_key="chat_history",
                                        return_messages=True,
                                        input_key="input",
                                        output_key="output")
        return AgentExecutor(agent=agent,
                           tools=self.tools,
                           verbose=True,
                           memory=memory,
                           handle_parsing_errors=True)
    
    def _create_verifier(self):
        return ChatGoogleGenerativeAI(model=self.model,
                                    google_api_key=self.api_key,
                                    temperature=0.0)
    
    def _create_executor(self):
        llm = ChatGoogleGenerativeAI(
            model=self.model,
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
            max_iterations=3,
            early_stopping_method="generate",
        )
    
    
    
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

    def process_request(self, user_input: str, max_iterations: int = 5) -> str:
        """Process a user request through the three-agent pipeline."""
        try:
            print("\n=== Starting Request Processing ===")
            print(f"User Input: {user_input}")
            
            inventory = get_workspace_inventory(self.workspace)
            print(f"\nCurrent Workspace Inventory: {inventory}")
            
            current_iteration = 0
            verification_result = ""
            plan = None
            
            while current_iteration < max_iterations:
                print(f"\n--- Iteration {current_iteration + 1} ---")
                
                # Planning Phase
                print("\n1. PLANNING PHASE")
                planning_input = {
                    "input": user_input if current_iteration == 0 
                            else f"{user_input}\nPrevious verification feedback: {verification_result}",
                    "workspace": self.workspace,
                    "inventory": inventory,
                    "tools": self.tool_descriptions  # Using the tool descriptions
                }
                print("Planning Input:")
                print(f"- Input: {planning_input['input']}")
                print(f"- Workspace: {planning_input['workspace']}")
                print(f"- Tools available: {len(self.tool_descriptions)}")
                
                try:
                    plan_result = self.planner.invoke(planning_input)
                    print("Raw Planner Output:")
                    print(plan_result['output'])  # Just print the output string
                    plan = self._extract_plan(plan_result)
                    print(f"Extracted Plan: {plan}")
                except ValueError as e:
                    print(f"Planning Error: {str(e)}")
                    return f"Planning failed: {str(e)}"
                
                time.sleep(5)  # Wait for 5 seconds before proceeding to the verification phase

                # Verification Phase
                print("\n2. VERIFICATION PHASE")
                verification_input = {
                    "plan": plan,
                    "request": user_input,
                    "tools": self.tool_descriptions,
                    "inventory": inventory
                }
                print(f"Verification Input: {json.dumps(verification_input, indent=2)}")
                
                verification_result = self.verifier.invoke(
                    VERIFIER_PROMPT.format_prompt(**verification_input)
                ).content
                print(f"Verification Result: {verification_result}")
                
                time.sleep(5)

                if verification_result.lower() == "valid":
                    print("Plan verified as valid!")
                    break
                
                current_iteration += 1
                if current_iteration == max_iterations:
                    print(f"Failed after {max_iterations} attempts")
                    return f"Failed to create a valid plan after {max_iterations} attempts. Last verification error: {verification_result}"
            
            # Execution Phase with error handling
            print("\n3. EXECUTION PHASE")
            print(f"Executing Plan: {plan}")
            
            try:
                execution_result = self.executor.invoke({
                    "input": plan
                })
                print(f"Execution Result: {json.dumps(execution_result, indent=2)}")
                return f"Execution completed:\n{execution_result['output']}"
            except Exception as exec_error:
                print(f"Execution error: {str(exec_error)}")
                # Try to execute the plan directly without the agent
                try:
                    results = []
                    parsed_plan = json.loads(plan)
                    for step in parsed_plan:
                        tool_name = step["tool"]
                        tool = next((t for t in self.tools if t.name == tool_name), None)
                        if tool:
                            result = tool.invoke(step["input"])
                            results.append(f"Step '{tool_name}' result: {result}")
                    return "\n".join(results)
                except Exception as direct_error:
                    return f"Execution failed: {str(direct_error)}"

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
        
        self.root = tk.Tk()
        self.setup_gui()
        self.start_agent_thread()
    
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
        
        # Main frame with groove relief
        main_frame = tk.Frame(self.root, bg="#f0f0f0", bd=10, relief="groove")
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
            height=1,
            width=40,
            wrap=tk.WORD,
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
    api_key, workspace = setup_environment()
    
    # Create GIS Agent
    gis_agent = GISAgent(api_key, workspace)
    
    # Create and run GUI
    gui = GISGUI(gis_agent)
    gui.run()

if __name__ == "__main__":
    main() 