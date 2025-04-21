# # soc_analyst_agent.py

# # INSTRUCTIONS TO RUN THIS SCRIPT:
# # ---------------------------------
# # 1. **SAVE:** Copy ALL the code below and save it as a file named `soc_analyst_agent.py`.
# # 2. **TERMINAL:** Open your Terminal or Command Prompt.
# # 3. **NAVIGATE:** Go to the directory where you saved `soc_analyst_agent.py`.
# # 4. **INSTALL:** Run this command to install necessary Python libraries:
# #    'pip install langchain langchain_community transformers accelerate torch pandas python-dotenv bitsandbytes'
# # 5. **RUN:** Execute the script by typing: `python soc_analyst_agent.py`
# # 6. **OUTPUT:** Observe the output in the terminal. Focusing on Triage Agent completion.

# import os
# import pandas as pd
# import json
# from langchain.agents import AgentExecutor, Tool, ZeroShotAgent
# from langchain.llms import HuggingFacePipeline
# from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
# # from dotenv import load_dotenv
# from langchain.chains import LLMChain
# import torch
# from langchain_community.llms import HuggingFacePipeline #Correct import 

# # load_dotenv()

# # Force disable TF32 - potentially reduces memory usage (less precision)
# torch.backends.cuda.matmul.allow_tf32 = False
# torch.backends.cudnn.allow_tf32 = False

# model_name = "unsloth/mistral-7b-instruct-v0.1-bnb-4bit"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", load_in_4bit=True)

# pipe = pipeline(
#                 "text-generation",
#                 model=model,
#                 tokenizer=tokenizer,
#                 max_new_tokens=64, # **EXTREME REDUCTION** - SHORTEST POSSIBLE
#                 temperature=0.1,   # Lowest temperature for determinism
#                 top_p=0.75,
#                 do_sample=False, # Force greedy - resolve warnings and simplify
#                 repetition_penalty=1.0 # No repetition penalty for simplest output
#             )

# llm = HuggingFacePipeline(pipeline=pipe)


# triage_tools = [] # REMOVED ALL TOOLS for Triage Agent - starting with NO tools

# triage_prefix = """You are a cybersecurity Triage Agent.
# Analyze the input and decide: ESCALATE or NO ESCALATE.
# Respond with 'ESCALATE' or 'NO ESCALATE'.""" # Minimal Triage Prompt - NO tools mentioned

# triage_suffix = """Begin!

# Input: {input}""" # Input directly in suffix - even simpler

# triage_prompt = ZeroShotAgent.create_prompt(
#     triage_tools,  # NO TOOLS
#     prefix=triage_prefix,
#     suffix=triage_suffix,
#     input_variables=["input", "agent_scratchpad", "tool_names"]
# )

# triage_llm_chain = LLMChain(llm=llm, prompt=triage_prompt)

# triage_agent = ZeroShotAgent(llm_chain=triage_llm_chain, tools=triage_tools, handle_parsing_errors=False) # Agent with NO tools, output_parser=NoopOutputParser() - CORRECTED IMPORT PATH
# triage_agent_executor = AgentExecutor.from_agent_and_tools(agent=triage_agent, tools=triage_tools, verbose=True, handle_parsing_errors=False) # Executor with NO tools, NO error handling


# def run_apa_workflow_df(alert_data_json):
#     print("\n--- Starting DataFrame-based APA Workflow (TRIAGE AGENT ONLY - NO TOOLS - MINIMAL MEMORY) ---") # Updated message

#     try:
#         alerts_list = json.loads(alert_data_json) # JSON PARSING HAPPENS HERE
#         alerts_df_initial = pd.DataFrame(alerts_list)
#         print("\n--- Initial Alerts DataFrame: ---")
#         print(alerts_df_initial)
#     except json.JSONDecodeError as e:
#         print(f"Error decoding JSON: {e}")
#         return {"error": "Invalid JSON input"}

#     print("\n--- Starting Triage Agent ---")
#     triage_tool_names = [tool.name for tool in triage_tools] # Tool names will be empty
#     triage_input = {"input": alerts_df_initial.to_string(), "tool_names": triage_tool_names}

#     print("\n[DEBUG - Triage Agent Input]:", triage_input)

#     print("\n[DEBUG - Before Triage Agent Executor Run]")
#     triage_output = triage_agent_executor.run(triage_input) # RUN TRIAGE AGENT ONLY - NO PARSER
#     print("\n[DEBUG - After Triage Agent Executor Run]")

#     print("\n[DEBUG - Triage Agent Raw Output]:", triage_output) # RAW OUTPUT - See what LLM *actually* says
#     print(f"\n--- Triage Agent Output: ---\n{triage_output}")

#     workflow_results = {
#         "triage_result": triage_output, # Store RAW output
#         "escalated": False,
#         "malicious_findings": {}
#     }

#     if "ESCALATE" in triage_output.upper(): # Basic string check - no parser
#         workflow_results["escalated"] = True
#         print("\n--- Triage Agent Escalated (String Check) ---") # Debug message if escalated
#     else:
#         print("\n--- Triage Agent Did NOT Escalate (String Check) ---") # Debug message if not escalated

#     filename = "security_findings.json"
#     print(f"\n--- Writing Security Findings (Triage Only - Minimal - RAW OUTPUT) to JSON file: {filename} ---") # Updated message
#     with open(filename, 'w') as f:
#         json.dump(workflow_results, f, indent=2)
#     print(f"  - Findings saved to '{filename}'")


#     print("\n--- DataFrame-based APA Workflow (TRIAGE ONLY - MINIMAL - RAW OUTPUT) Complete ---") # Workflow Complete message
#     final_results_json = json.dumps(workflow_results, indent=2)
#     print("\n--- Workflow Results (JSON - TRIAGE ONLY - MINIMAL - RAW OUTPUT): ---") # Updated message
#     print(final_results_json)
#     return workflow_results


# suricata_alert_data_json = """
# [
#   {
#     "timestamp": "2024-01-20T10:00:00Z",
#     "alert": {
#       "category": "Malware C2 Activity",
#       "signature": "Likely Evil DNS",
#       "severity": 4
#     },
#     "src_ip": "192.168.1.100"
#   }
# ]
# """ # Minimal alert data - ONE very simple alert

# print("\n--- Example Suricata Alert Data (JSON - Minimal): ---") # Updated message
# print(suricata_alert_data_json)

# workflow_results_df = run_apa_workflow_df(suricata_alert_data_json)




# soc_analyst_agent.py

# INSTRUCTIONS TO RUN THIS SCRIPT:
# ---------------------------------
# 1. **SAVE:** Copy ALL the code below and save it as a file named `soc_analyst_agent.py`.
# 2. **TERMINAL:** Open your Terminal or Command Prompt.
# 3. **NAVIGATE:** Go to the directory where you saved `soc_analyst_agent.py`.
# 4. **INSTALL:** Run this command to install necessary Python libraries:
#    'pip install langchain langchain_community transformers accelerate torch pandas python-dotenv bitsandbytes'
# 5. **RUN:** Execute the script by typing: `python soc_analyst_agent.py`
# 6. **OUTPUT:** Observe the output in the terminal. Focus is on JSON file creation now.



# soc_analyst_agent.py

# INSTRUCTIONS TO RUN THIS SCRIPT:
# ---------------------------------
# 1. **SAVE:** Copy ALL the code below and save it as a file named `soc_analyst_agent.py`.
# 2. **TERMINAL:** Open your Terminal or Command Prompt.
# 3. **NAVIGATE:** Go to the directory where you saved `soc_analyst_agent.py`.
# 4. **INSTALL:** Run this command to install necessary Python libraries:
#    'pip install langchain langchain_community transformers accelerate torch pandas python-dotenv bitsandbytes'
# 5. **RUN:** Execute the script by typing: `python soc_analyst_agent.py`
# 6. **OUTPUT:** Observe the output in the terminal and check for 'security_findings.json'.

import os
import pandas as pd
import json
from langchain.agents import AgentExecutor, Tool, ZeroShotAgent
from langchain.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from dotenv import load_dotenv
from langchain.chains import LLMChain
import torch
from langchain_community.llms import HuggingFacePipeline

load_dotenv()

# Force disable TF32
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

model_name = "unsloth/mistral-7b-instruct-v0.1-bnb-4bit"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", load_in_4bit=True)

pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=32, # Shortest possible
                temperature=0.1,
                top_p=0.75,
                do_sample=False,
                repetition_penalty=1.0
            )

llm = HuggingFacePipeline(pipeline=pipe)


triage_tools = [] # NO TOOLS

triage_prefix = """You are a cybersecurity Triage Agent.
Analyze the input and decide: ESCALATE or NO ESCALATE.
Respond with 'ESCALATE' or 'NO ESCALATE'."""

triage_suffix = """Begin!

Input: {input}"""

triage_prompt = ZeroShotAgent.create_prompt(
    triage_tools,
    prefix=triage_prefix,
    suffix=triage_suffix,
    input_variables=["input", "agent_scratchpad", "tool_names"]
)

triage_llm_chain = LLMChain(llm=llm, prompt=triage_prompt)

triage_agent = ZeroShotAgent(llm_chain=triage_llm_chain, tools=triage_tools)
triage_agent_executor = AgentExecutor.from_agent_and_tools(agent=triage_agent, tools=triage_tools, verbose=True, handle_parsing_errors=False)


def run_apa_workflow_df(alert_data_json):
    print("\n--- Starting DataFrame-based APA Workflow (TRIAGE AGENT ONLY - NO TOOLS - JSON FILE OUTPUT CHECK) ---")

    try:
        alerts_list = json.loads(alert_data_json)
        alerts_df_initial = pd.DataFrame(alerts_list)
        print("\n--- Initial Alerts DataFrame: ---")
        print(alerts_df_initial)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        return {"error": "Invalid JSON input"}

    print("\n--- Starting Triage Agent ---")
    triage_tool_names = [tool.name for tool in triage_tools]
    triage_input = {"input": alerts_df_initial.to_string(), "tool_names": triage_tool_names}

    print("\n[DEBUG - Triage Agent Input]:", triage_input)

    print("\n[DEBUG - Before Triage Agent Executor Run]")
    triage_output = triage_agent_executor.run(triage_input)
    print("\n[DEBUG - After Triage Agent Executor Run]")

    print("\n[DEBUG - Triage Agent Raw Output]:", triage_output)
    print(f"\n--- Triage Agent Output: ---\n{triage_output}")

    workflow_results = {
        "triage_result": triage_output,
        "escalated": False,
        "malicious_findings": {}
    }

    if "ESCALATE" in triage_output.upper():
        workflow_results["escalated"] = True
        print("\n--- Triage Agent Escalated (String Check) ---")

        filename = "security_findings.json"
        print(f"\n[DEBUG - Before JSON File Write - Escalated=True]: About to write to {filename}")
        try:
            # Get current working directory for debugging
            current_dir = os.getcwd()
            print(f"\n[DEBUG] - Current Working Directory: {current_dir}") # Print current directory

            filepath = os.path.join(current_dir, filename) # Create full file path
            print(f"\n[DEBUG] - Full File Path: {filepath}") # Print full file path

            with open(filename, 'w') as f: # Use filename (not filepath for now, simplify)
                json.dump(workflow_results, f, indent=2)
            print(f"  - Findings saved to '{filename}'")
            print(f"\n[DEBUG - After JSON File Write]: File '{filename}' should now exist in CWD.")
            workflow_results["json_file_created"] = True
        except Exception as e:
            workflow_results["json_file_created"] = False
            workflow_results["json_file_error"] = str(e)
            print(f"\n[ERROR] - Error writing to JSON file: {e}")

    else:
        workflow_results["escalated"] = False
        workflow_results["json_file_created"] = False
        print("\n--- Triage Agent Did NOT Escalate (String Check) ---")
        print("\n[DEBUG - No Escalation - JSON File NOT written]")

    print("\n--- DataFrame-based APA Workflow (TRIAGE ONLY - NO TOOLS - JSON FILE OUTPUT CHECK) Complete ---")
    final_results_json = json.dumps(workflow_results, indent=2)
    print("\n--- Workflow Results (JSON - TRIAGE ONLY - NO TOOLS - JSON FILE OUTPUT CHECK): ---")
    print(final_results_json)
    return workflow_results


suricata_alert_data_json = """
[
  {
    "timestamp": "2024-01-20T10:00:00Z",
    "alert": {
      "category": "Malware C2 Activity",
      "signature": "Likely Evil DNS",
      "severity": 4
    },
    "src_ip": "192.168.1.100"
  }
]
"""

print("\n--- Example Suricata Alert Data (JSON - Minimal): ---")
print(suricata_alert_data_json)

workflow_results_df = run_apa_workflow_df(suricata_alert_data_json)