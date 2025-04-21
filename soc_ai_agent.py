# # soc_analyst_agent.py

# # INSTRUCTIONS TO RUN THIS SCRIPT:
# # ---------------------------------
# # 1. **SAVE:** Copy ALL the code below and save it as a file named `soc_analyst_agent.py`.
# # 2. **TERMINAL:** Open your Terminal or Command Prompt.
# # 3. **NAVIGATE:** Go to the directory where you saved `soc_analyst_agent.py` using the 'cd' command.
# #    For example, if you saved it on your Desktop inside a folder named 'my_project', you would type:
# #    'cd Desktop/my_project' (on macOS/Linux) or 'cd Desktop\my_project' (on Windows).
# # 4. **INSTALL:** Run this command to install necessary Python libraries:
# #    'pip install langchain transformers accelerate torch pandas python-dotenv'
# # 5. **RUN:** Execute the script by typing: `python soc_analyst_agent.py`
# # 6. **OUTPUT:** Observe the results printed in your terminal. It will show the Agentic Process Automation workflow
# #    being executed and the final results in JSON format.

# # NOTE on API Keys (.env file):
# # -----------------------------
# # This script, as it is now, uses a local language model (Mistral 7B) and does NOT require an OpenAI API key.
# # You can ignore any mention of OpenAI API keys or .env files for THIS SPECIFIC SCRIPT.
# # If you were to modify this script to use cloud-based LLMs like OpenAI in the future,
# # you would then need to set up an API key and potentially use a .env file to store it securely.


# import os
# import pandas as pd
# import json
# from langchain.agents import AgentExecutor, Tool, ZeroShotAgent
# from langchain.llms import HuggingFacePipeline
# from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
# from dotenv import load_dotenv
# from langchain.chains import LLMChain

# load_dotenv()

# model_name = "unsloth/mistral-7b-instruct-v0.1-bnb-4bit"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

# pipe = pipeline(
#                 "text-generation",
#                 model=model,
#                 tokenizer=tokenizer,
#                 max_new_tokens=8000,
#                 temperature=0.7,
#                 top_p=0.95,
#                 repetition_penalty=1.15
#             )

# llm = HuggingFacePipeline(pipeline=pipe)

# threat_intel_data = pd.DataFrame({
#     'ioc': ['192.168.1.100', 'malicious.domain.com', 'benign.domain.com', 'evil.domain.net'],
#     'reputation': ['Malicious', 'Malicious', 'Benign', 'Malicious'],
#     'threat_type': ['C2 Server', 'Phishing', 'Informational', 'Malware Distribution'],
#     'confidence': ['High', 'Medium', 'High', 'High']
# })

# machine_details_data = pd.DataFrame({
#     'src_ip': ['192.168.1.100', '192.168.1.101', '192.168.1.102'],
#     'hostname': ['workstation-01', 'server-02', 'laptop-03'],
#     'os': ['Windows 10', 'Linux', 'MacOS'],
#     'user': ['john.doe', 'system', 'jane.doe']
# })

# account_details_data = pd.DataFrame({
#     'username': ['john.doe', 'system', 'jane.doe', 'attacker123'],
#     'department': ['Engineering', 'System', 'Marketing', 'External'],
#     'roles': [['user', 'developer'], ['system'], ['user', 'analyst'], ['malicious']],
#     'last_login': ['2024-01-20T10:00:00Z', 'N/A', '2024-01-19T14:30:00Z', 'Never']
# })

# siem_data = pd.DataFrame({
#     'timestamp': pd.to_datetime(['2024-01-20 10:05:00', '2024-01-20 10:06:00', '2024-01-20 10:07:00', '2024-01-20 10:08:00',
#                                   '2024-01-20 11:00:00', '2024-01-20 11:05:00']),
#     'event_type': ['network_connection', 'dns_query', 'http_request', 'firewall_deny',
#                    'authentication_failure', 'file_access'],
#     'src_ip': ['192.168.1.100', '192.168.1.100', '192.168.1.100', '192.168.1.100',
#                '192.168.1.105', '192.168.1.100'],
#     'dest_ip': ['8.8.8.8', '8.8.8.8', 'evil.domain.net', '192.168.2.10',
#                 '192.168.1.100', 'server-03'],
#     'protocol': ['TCP', 'UDP', 'HTTP', 'TCP',
#                  'SSH', 'SMB'],
#     'port': [53, 53, 80, 443,
#              22, 445],
#     'domain': [None, 'malicious.domain.com', 'evil.domain.net', None,
#                None, None],
#     'username': [None, None, None, None,
#                  'john.doe', 'john.doe'],
#     'file_accessed': [None, None, None, None, None, '/sensitive/data.txt']
# })


# def deduplicate_alerts_df(alerts_df):
#     print("\n[Triage Agent - DeduplicateAlerts Tool]: Deduplicating alerts...")
#     initial_count = len(alerts_df)
#     deduplicated_df = alerts_df.drop_duplicates(subset=['alert.signature', 'src_ip', 'timestamp'], keep='first')
#     deduplicated_count = len(deduplicated_df)
#     print(f"  - Deduplicated {initial_count - deduplicated_count} alerts.")
#     return deduplicated_df

# def group_alerts_by_asset_df(alerts_df):
#     print("\n[Triage Agent - GroupByAsset Tool]: Grouping alerts by asset (src_ip)...")
#     grouped_alerts = alerts_df.groupby('src_ip')
#     print("  - Alerts grouped by 'src_ip'.")
#     return grouped_alerts

# def enrich_ioc_df(alerts_df, threat_intel_df=threat_intel_data):
#     print("\n[Triage Agent - EnrichIOC Tool]: Enriching alerts with threat intelligence...")
#     enriched_alerts_df = alerts_df.copy()

#     def enrich_row(row):
#         ioc_ip = row['src_ip']
#         ioc_domain = row.get('dns.query')

#         ip_intel = threat_intel_df[threat_intel_df['ioc'] == ioc_ip]
#         if not ip_intel.empty:
#             row['ip_reputation'] = ip_intel['reputation'].iloc[0]
#             row['ip_threat_type'] = ip_intel['threat_type'].iloc[0]
#             row['ip_confidence'] = ip_intel['confidence'].iloc[0]

#         if ioc_domain:
#             domain_intel = threat_intel_df[threat_intel_df['ioc'] == ioc_domain]
#             if not domain_intel.empty:
#                 row['domain_reputation'] = domain_intel['reputation'].iloc[0]
#                 row['domain_threat_type'] = domain_intel['threat_type'].iloc[0]
#                 row['domain_confidence'] = domain_intel['confidence'].iloc[0]
#         return row

#     enriched_alerts_df = enriched_alerts_df.apply(enrich_row, axis=1)
#     print("  - Alerts enriched with threat intelligence.")
#     return enriched_alerts_df

# def get_machine_details_df(alerts_df, machine_data=machine_details_data):
#     print("\n[Triage Agent - GetMachineDetails Tool]: Getting machine details...")
#     merged_df = pd.merge(alerts_df, machine_data, on='src_ip', how='left')
#     print("  - Machine details merged into alerts DataFrame based on 'src_ip'.")
#     return merged_df

# def get_account_details_df(alerts_df, account_data=account_details_data):
#     print("\n[Triage Agent - GetAccountDetails Tool]: Getting account details (example based on alert signature keyword)...")
#     alerts_df['username_extract'] = alerts_df['alert.signature'].str.extract(r'user\s+(\w+)')
#     merged_df = pd.merge(alerts_df, account_data, left_on='username_extract', right_on='username', how='left')
#     merged_df = merged_df.drop(columns=['username_extract', 'username_y'])
#     merged_df = merged_df.rename(columns={'username_x': 'username'})
#     print("  - Account details merged (example based on username extraction from signature).")
#     return merged_df


# def query_siem_df(query_description, siem_df=siem_data):
#     print(f"\n[Threat Hunting Agent - QuerySIEM Tool]: Querying SIEM for: '{query_description}'...")
#     if "ip 192.168.1.100" in query_description.lower():
#         results_df = siem_df[siem_df['src_ip'] == '192.168.1.100']
#     elif "domain malicious.domain.com" in query_description.lower():
#         results_df = siem_df[siem_df['domain'] == 'malicious.domain.com']
#     elif "authentication failures for john.doe" in query_description.lower():
#         results_df = siem_df[(siem_df['event_type'] == 'authentication_failure') & (siem_df['username'] == 'john.doe')]
#     else:
#         results_df = pd.DataFrame()
#     print(f"  - SIEM Query returned {len(results_df)} results.")
#     return results_df

# def classify_indicator_df(indicator_value):
#     print(f"\n[Threat Hunting Agent - ClassifyIndicator Tool]: Classifying indicator: '{indicator_value}'...")
#     if indicator_value.startswith(('192.', '10.', '172.', 'domain')):
#         classification = "Atomic Indicator (IP Address or Domain)"
#     elif "hash" in indicator_value.lower():
#         classification = "Computed Indicator (File Hash Value)"
#     elif "unusual activity" in indicator_value.lower() or "login pattern" in indicator_value.lower():
#         classification = "Behavioral Indicator (Activity Pattern)"
#     else:
#         classification = "Unknown Indicator Type (Needs further analysis)"
#     print(f"  - Classified '{indicator_value}' as: {classification}")
#     return classification

# def map_ttp_mitre_attack_df(behavioral_indicator):
#     print(f"\n[Threat Hunting Agent - MapTTPMITREATTACK Tool]: Mapping TTP for: '{behavioral_indicator}'...")
#     ttp_mapping = {
#         "unusual network activity": ["T1071 - Application Layer Protocol", "T1041 - Exfiltration Over C2 Channel"],
#         "multiple failed logins": ["T1110 - Brute Force", "T1133 - External Remote Services"],
#         "file access to sensitive data by unusual user": ["T1003 - OS Credential Dumping", "T1081 - Credentials in Files"]
#     }
#     found_ttps = []
#     for indicator_keyword, ttps in ttp_mapping.items():
#         if indicator_keyword in behavioral_indicator.lower():
#             found_ttps.extend(ttps)

#     if found_ttps:
#         ttp_string = ", ".join(found_ttps)
#         print(f"  - Mapped '{behavioral_indicator}' to MITRE ATT&CK TTPs: {ttp_string}")
#         return f"MITRE ATT&CK TTPs: {ttp_string}"
#     else:
#         print(f"  - No MITRE ATT&CK TTP mapping found for: '{behavioral_indicator}'")
#         return "No MITRE ATT&CK TTP mapping found."


# def isolate_endpoint_action(endpoint_id):
#     print(f"\n[Response Agent - IsolateEndpoint Tool]: **ACTION: Isolating endpoint: {endpoint_id}** (Simulated)")
#     return f"ACTION TAKEN: Endpoint '{endpoint_id}' isolation initiated. (Simulated)"

# def block_ip_address_action(ip_address):
#     print(f"\n[Response Agent - BlockIPAddress Tool]: **ACTION: Blocking IP address: {ip_address}** (Simulated)")
#     return f"ACTION TAKEN: IP address '{ip_address}' blocked on firewall. (Simulated)"

# def collect_forensic_data_action(endpoint_id):
#     print(f"\n[Response Agent - CollectForensicData Tool]: **ACTION: Collecting forensic data from endpoint: {endpoint_id}** (Simulated)")
#     return f"ACTION TAKEN: Forensic data collection initiated from endpoint '{endpoint_id}'. (Simulated)"

# def generate_iac_terraform_action(remediation_steps):
#     print(f"\n[Response Agent - GenerateIACTerraform Tool]: **ACTION: Generating Terraform code for: {remediation_steps}** (Simulated)")
#     terraform_code_example = f"""
#     # Simulated Terraform code for remediation steps: {remediation_steps}
#     resource "null_resource" "remediation" {{
#       provisioner "local-exec" {{
#         command = "echo 'Simulated remediation: {remediation_steps}'"
#       }}
#     }}
#     """
#     print(f"  - Simulated Terraform Code Example:\n```terraform\n{terraform_code_example}\n```")
#     return f"ACTION TAKEN: Terraform code generated for remediation steps. (Simulated)\nExample Code:\n{terraform_code_example}"


# triage_tools = [
#     Tool(name="DeduplicateAlertsDF", func=deduplicate_alerts_df, description="Useful to deduplicate alerts. Input: alerts DataFrame."),
#     Tool(name="GroupByAssetDF", func=group_alerts_by_asset_df, description="Useful to group alerts by asset. Input: alerts DataFrame."),
#     Tool(name="EnrichIOCDF", func=enrich_ioc_df, description="Useful to enrich alerts with threat intel. Input: alerts DataFrame."),
#     Tool(name="GetMachineDetailsDF", func=get_machine_details_df, description="Useful to get machine details. Input: alerts DataFrame."),
#     Tool(name="GetAccountDetailsDF", func=get_account_details_df, description="Useful to get account details. Input: alerts DataFrame.")
# ]

# threat_hunting_tools = [
#     Tool(name="QuerySIEMDF", func=query_siem_df, description="Useful to query SIEM data. Input: query description."),
#     Tool(name="ClassifyIndicatorDF", func=classify_indicator_df, description="Useful to classify indicators. Input: indicator value."),
#     Tool(name="MapTTPMITREATTACKDF", func=map_ttp_mitre_attack_df, description="Useful to map TTPs to MITRE ATT&CK. Input: behavioral indicator.")
# ]

# response_tools = [
#     Tool(name="IsolateEndpointAction", func=isolate_endpoint_action, description="Useful to isolate endpoint. Input: endpoint ID."),
#     Tool(name="BlockIPAddressAction", func=block_ip_address_action, description="Useful to block IP address. Input: IP address."),
#     Tool(name="CollectForensicDataAction", func=collect_forensic_data_action, description="Useful to collect forensic data. Input: endpoint ID."),
#     Tool(name="GenerateIACTerraformAction", func=generate_iac_terraform_action, description="Useful to generate Terraform code. Input: remediation steps description.")
# ]


# triage_prefix = """You are a cybersecurity Triage Agent. Analyze alerts DataFrame, enrich data, decide escalation.

# **Response Format Instructions:**

# You MUST use the following format for your responses:

# **If using a tool:**


# Tools:
# {tool_names}

# Input DataFrame:
# {input}
# """

# triage_suffix = """Begin!"""
# triage_prompt = ZeroShotAgent.create_prompt(
#     triage_tools,
#     prefix=triage_prefix,
#     suffix=triage_suffix,
#     input_variables=["input", "agent_scratchpad", "tool_names"]
# )

# triage_llm_chain = LLMChain(llm=llm, prompt=triage_prompt)

# threat_hunting_prefix = """You are a Reactive Threat Hunting Agent. Investigate triage findings, enriched alerts DataFrame, use SIEM data, indicator analysis.

# **Response Format Instructions:**

# You MUST use the following format for your responses:

# **If using a tool:**


# Tools:
# {tool_names}

# Triage Findings and Enriched Alerts DataFrame:
# {input}

# """
# threat_hunting_suffix = """Begin!"""
# threat_hunting_prompt = ZeroShotAgent.create_prompt(
#     threat_hunting_tools,
#     prefix=threat_hunting_prefix,
#     suffix=threat_hunting_suffix,
#     input_variables=["input", "agent_scratchpad", "tool_names"]
# )
# threat_hunting_llm_chain = LLMChain(llm=llm, prompt=threat_hunting_prompt)

# response_prefix = """You are a Cybersecurity Response Agent. Take action based on Threat Hunting findings to contain and remediate the incident.

# **Response Format Instructions:**

# You MUST use the following format for your responses:

# **If using a tool (taking an action):**


# Tools:
# {tool_names}

# Threat Hunting Findings and Recommendations:
# {input}
# """
# response_suffix = """Begin!"""
# response_prompt = ZeroShotAgent.create_prompt(
#     response_tools,
#     prefix=response_prefix,
#     suffix=response_suffix,
#     input_variables=["input", "agent_scratchpad", "tool_names"]
# )
# response_llm_chain = LLMChain(llm=llm, prompt=response_prompt)

# triage_agent = ZeroShotAgent(llm_chain=triage_llm_chain, tools=triage_tools)
# triage_agent_executor = AgentExecutor.from_agent_and_tools(agent=triage_agent, tools=triage_tools, verbose=True, handle_parsing_errors=True)

# threat_hunting_agent = ZeroShotAgent(llm_chain=threat_hunting_llm_chain, tools=threat_hunting_tools)
# threat_hunting_agent_executor = AgentExecutor.from_agent_and_tools(agent=threat_hunting_agent, tools=threat_hunting_tools, verbose=True, handle_parsing_errors=True)

# response_agent = ZeroShotAgent(llm_chain=response_llm_chain, tools=response_tools)
# response_agent_executor = AgentExecutor.from_agent_and_tools(agent=response_agent, tools=response_tools, verbose=True, handle_parsing_errors=True)


# def run_apa_workflow_df(alert_data_json):
#     print("\n--- Starting DataFrame-based APA Workflow ---")

#     try:
#         alerts_list = json.loads(alert_data_json)
#         alerts_df_initial = pd.DataFrame(alerts_list)
#         print("\n--- Initial Alerts DataFrame: ---")
#         print(alerts_df_initial)
#     except json.JSONDecodeError as e:
#         print(f"Error decoding JSON: {e}")
#         return {"error": "Invalid JSON input"}

#     print("\n--- Starting Triage Agent ---")
#     triage_output = triage_agent_executor.run(alerts_df_initial.to_string())
#     print(f"\n--- Triage Agent Output: ---\n{triage_output}")

#     escalated = False
#     enriched_alerts_df = None
#     if "ESCALATE" in triage_output.upper():
#         escalated = True
#         print("\n--- Escalating to Reactive Threat Hunting Agent ---")

#         enriched_alerts_df = enrich_ioc_df(alerts_df_initial)
#         threat_hunting_input = f"Triage Agent escalated. Enriched Alerts DataFrame:\n{enriched_alerts_df.to_string()}"
#         threat_hunting_output = threat_hunting_agent_executor.run(threat_hunting_input)
#         print(f"\n--- Threat Hunting Agent Output: ---\n{threat_hunting_output}")

#         print("\n--- Proceeding to Response Agent ---")
#         response_input = f"Threat Hunting findings: {threat_hunting_output}. Enriched Alerts DataFrame (for context):\n{enriched_alerts_df.to_string()}"
#         response_output = response_agent_executor.run(response_input)
#         print(f"\n--- Response Agent Output: ---\n{response_output}")

#         return {
#             "triage_result": triage_output,
#             "threat_hunting_result": threat_hunting_output,
#             "response_result": response_output,
#             "escalated": escalated,
#             "enriched_alerts_dataframe": enriched_alerts_df.to_dict() if enriched_alerts_df is not None else None
#         }
#     else:
#         return {
#             "triage_result": triage_output,
#             "threat_hunting_result": "Not Escalated",
#             "response_result": "Not Escalated",
#             "escalated": escalated,
#             "enriched_alerts_dataframe": None
#         }


# suricata_alert_data_json = """
# [
#   {
#     "timestamp": "2024-01-20T10:00:00Z",
#     "alert": {
#       "category": "Malware Command and Control Activity",
#       "signature": "ET MALWARE Observed DNS Query to .top domain - Likely Evil",
#       "severity": 4
#     },
#     "src_ip": "192.168.1.100",
#     "dest_ip": "8.8.8.8",
#     "proto": "UDP",
#     "src_port": 53456,
#     "dest_port": 53,
#     "dns": {
#       "query": "maliciousdomain.top",
#       "type": "A"
#     }
#   },
#   {
#     "timestamp": "2024-01-20T10:01:00Z",
#     "alert": {
#       "category": "Malware Command and Control Activity",
#       "signature": "ET MALWARE Observed DNS Query to .top domain - Likely Evil",
#       "severity": 4
#     },
#     "src_ip": "192.168.1.100",
#     "dest_ip": "8.8.8.8",
#     "proto": "UDP",
#     "src_port": 53457,
#     "dest_port": 53,
#     "dns": {
#       "query": "maliciousdomain.top",
#       "type": "A"
#     }
#   },
#     {
#     "timestamp": "2024-01-20T11:15:00Z",
#     "alert": {
#       "category": "System Access",
#       "signature": "Possible Brute Force SSH Login attempt - user john.doe",
#       "severity": 3
#     },
#     "src_ip": "192.168.1.105",
#     "dest_ip": "192.168.1.100",
#     "proto": "TCP",
#     "src_port": 12345,
#     "dest_port": 22,
#     "username": "john.doe"
#     }
# ]
# """

# print("\n--- Example Suricata Alert Data (JSON): ---")
# print(suricata_alert_data_json)

# workflow_results_df = run_apa_workflow_df(suricata_alert_data_json)

# print("\n--- DataFrame-based APA Workflow Complete ---")
# print("\n--- Workflow Results (JSON): ---")
# print(json.dumps(workflow_results_df, indent=2))
######################################################
####################################################
##################################################
# soc_analyst_agent.py

# INSTRUCTIONS TO RUN THIS SCRIPT:
# ---------------------------------
# 1. **SAVE:** Copy ALL the code below and save it as a file named `soc_analyst_agent.py`.
# 2. **TERMINAL:** Open your Terminal or Command Prompt.
# 3. **NAVIGATE:** Go to the directory where you saved `soc_analyst_agent.py` using the 'cd' command.
#    For example, if you saved it on your Desktop inside a folder named 'my_project', you would type:
#    'cd Desktop/my_project' (on macOS/Linux) or 'cd Desktop\my_project' (on Windows).
# 4. **INSTALL:** Run this command to install necessary Python libraries:
#    'pip install langchain transformers accelerate torch pandas python-dotenv'
# 5. **RUN:** Execute the script by typing: `python soc_analyst_agent.py`
# 6. **OUTPUT:** Observe the results printed in your terminal. It will show the Agentic Process Automation workflow
#    being executed and the final results in JSON format.

# NOTE on API Keys (.env file):
# -----------------------------
# This script, as it is now, uses a local language model (Mistral 7B) and does NOT require an OpenAI API key.
# You can ignore any mention of OpenAI API keys or .env files for THIS SPECIFIC SCRIPT.
# If you were to modify this script to use cloud-based LLMs like OpenAI in the future,
# you would then need to set up an API key and potentially use a .env file to store it securely.


import os
import pandas as pd
import json
from langchain.agents import AgentExecutor, Tool, ZeroShotAgent
from langchain.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
# from dotenv import load_dotenv
from langchain.chains import LLMChain

# load_dotenv()

model_name = "unsloth/Llama-3.2-1B-Instruct-bnb-4bit"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=1024,
                temperature=0.7,
                top_p=0.95,
                repetition_penalty=1.15
            )

llm = HuggingFacePipeline(pipeline=pipe)

threat_intel_data = pd.DataFrame({
    'ioc': ['192.168.1.100', 'malicious.domain.com'],
    'reputation': ['Malicious', 'Malicious'],
    'threat_type': ['C2 Server', 'Phishing'],
    'confidence': ['High', 'Medium']
})

machine_details_data = pd.DataFrame({
    'src_ip': ['192.168.1.100'],
    'hostname': ['workstation-01'],
    'os': ['MacOS'],
    'user': ['jane.doe']
})

account_details_data = pd.DataFrame({
    'username': ['john.doe', 'system'],
    'department': ['Engineering', 'System'],
    'roles': [['user', 'developer'], ['system']],
    'last_login': ['2024-01-20T10:00:00Z',  'Never']
})

siem_data = pd.DataFrame({
    'timestamp': pd.to_datetime(['2024-01-20 10:05:00', '2024-01-20 10:06:00', '2024-01-20 10:07:00', '2024-01-20 10:08:00',
                                  '2024-01-20 11:00:00', '2024-01-20 11:05:00']),
    'event_type': ['network_connection', 'dns_query', 'http_request', 'firewall_deny',
                   'authentication_failure', 'file_access'],
    'src_ip': ['192.168.1.100', '192.168.1.100', '192.168.1.100', '192.168.1.100',
               '192.168.1.105', '192.168.1.100'],
    'dest_ip': ['8.8.8.8', '8.8.8.8', 'evil.domain.net', '192.168.2.10',
                '192.168.1.100', 'server-03'],
    'protocol': ['TCP', 'UDP', 'HTTP', 'TCP',
                 'SSH', 'SMB'],
    'port': [53, 53, 80, 443,
             22, 445],
    'domain': [None, 'malicious.domain.com', 'evil.domain.net', None,
               None, None],
    'username': [None, None, None, None,
                 'john.doe', 'john.doe'],
    'file_accessed': [None, None, None, None, None, '/sensitive/data.txt']
})


def deduplicate_alerts_df(alerts_df):
    print("\n[Triage Agent - AlertDeduplicator Tool]: Deduplicating alerts...")
    initial_count = len(alerts_df)
    deduplicated_df = alerts_df.drop_duplicates(subset=['alert.signature', 'src_ip', 'timestamp'], keep='first')
    deduplicated_count = len(deduplicated_df)
    print(f"  - Deduplicated {initial_count - deduplicated_count} alerts.")
    return deduplicated_df

def group_alerts_by_asset_df(alerts_df):
    print("\n[Triage Agent - AlertGrouperByAsset Tool]: Grouping alerts by asset (src_ip)...")
    grouped_alerts = alerts_df.groupby('src_ip')
    print("  - Alerts grouped by 'src_ip'.")
    return grouped_alerts

def enrich_ioc_df(alerts_df, threat_intel_df=threat_intel_data):
    print("\n[Triage Agent - IOCEnricher Tool]: Enriching alerts with threat intelligence...")
    enriched_alerts_df = alerts_df.copy()

    def enrich_row(row):
        ioc_ip = row['src_ip']
        ioc_domain = row.get('dns.query')

        ip_intel = threat_intel_df[threat_intel_df['ioc'] == ioc_ip]
        if not ip_intel.empty:
            row['ip_reputation'] = ip_intel['reputation'].iloc[0]
            row['ip_threat_type'] = ip_intel['threat_type'].iloc[0]
            row['ip_confidence'] = ip_intel['confidence'].iloc[0]

        if ioc_domain:
            domain_intel = threat_intel_df[threat_intel_df['ioc'] == ioc_domain]
            if not domain_intel.empty:
                row['domain_reputation'] = domain_intel['reputation'].iloc[0]
                row['domain_threat_type'] = domain_intel['threat_type'].iloc[0]
                row['domain_confidence'] = domain_intel['confidence'].iloc[0]
        return row

    enriched_alerts_df = enriched_alerts_df.apply(enrich_row, axis=1)
    print("  - Alerts enriched with threat intelligence.")
    return enriched_alerts_df

def get_machine_details_df(alerts_df, machine_data=machine_details_data):
    print("\n[Triage Agent - MachineDetailsFetcher Tool]: Getting machine details...")
    merged_df = pd.merge(alerts_df, machine_data, on='src_ip', how='left')
    print("  - Machine details merged into alerts DataFrame based on 'src_ip'.")
    return merged_df

def get_account_details_df(alerts_df, account_data=account_details_data):
    print("\n[Triage Agent - AccountDetailsFetcher Tool]: Getting account details (example based on alert signature keyword)...")
    alerts_df['username_extract'] = alerts_df['alert.signature'].str.extract(r'user\s+(\w+)')
    merged_df = pd.merge(alerts_df, account_data, left_on='username_extract', right_on='username', how='left')
    merged_df = merged_df.drop(columns=['username_extract', 'username_y'])
    merged_df = merged_df.rename(columns={'username_x': 'username'})
    print("  - Account details merged (example based on username extraction from signature).")
    return merged_df


def query_siem_df(query_description, siem_df=siem_data):
    print(f"\n[Threat Hunting Agent - SIEMQueryTool]: Querying SIEM for: '{query_description}'...")
    if "ip 192.168.1.100" in query_description.lower():
        results_df = siem_df[siem_df['src_ip'] == '192.168.1.100']
    elif "domain malicious.domain.com" in query_description.lower():
        results_df = siem_df[siem_df['domain'] == 'malicious.domain.com']
    elif "authentication failures for john.doe" in query_description.lower():
        results_df = siem_df[(siem_df['event_type'] == 'authentication_failure') & (siem_df['username'] == 'john.doe')]
    else:
        results_df = pd.DataFrame()
    print(f"  - SIEM Query returned {len(results_df)} results.")
    return results_df

def classify_indicator_df(indicator_value):
    print(f"\n[Threat Hunting Agent - IndicatorClassifier Tool]: Classifying indicator: '{indicator_value}'...")
    if indicator_value.startswith(('192.', '10.', '172.', 'domain')):
        classification = "Atomic Indicator (IP Address or Domain)"
    elif "hash" in indicator_value.lower():
        classification = "Computed Indicator (File Hash Value)"
    elif "unusual activity" in indicator_value.lower() or "login pattern" in indicator_value.lower():
        classification = "Behavioral Indicator (Activity Pattern)"
    else:
        classification = "Unknown Indicator Type (Needs further analysis)"
    print(f"  - Classified '{indicator_value}' as: {classification}")
    return classification

def map_ttp_mitre_attack_df(behavioral_indicator):
    print(f"\n[Threat Hunting Agent - MITREATTACKMapper Tool]: Mapping TTP for: '{behavioral_indicator}'...")
    ttp_mapping = {
        "unusual network activity": ["T1071 - Application Layer Protocol", "T1041 - Exfiltration Over C2 Channel"],
        "multiple failed logins": ["T1110 - Brute Force", "T1133 - External Remote Services"],
        "file access to sensitive data by unusual user": ["T1003 - OS Credential Dumping", "T1081 - Credentials in Files"]
    }
    found_ttps = []
    for indicator_keyword, ttps in ttp_mapping.items():
        if indicator_keyword in behavioral_indicator.lower():
            found_ttps.extend(ttps)

    if found_ttps:
        ttp_string = ", ".join(found_ttps)
        print(f"  - Mapped '{behavioral_indicator}' to MITRE ATT&CK TTPs: {ttp_string}")
        return f"MITRE ATT&CK TTPs: {ttp_string}"
    else:
        print(f"  - No MITRE ATT&CK TTP mapping found for: '{behavioral_indicator}'")
        return "No MITRE ATT&CK TTP mapping found."


def isolate_endpoint_action(endpoint_id):
    print(f"\n[Response Agent - EndpointIsolator Tool]: **ACTION: Isolating endpoint: {endpoint_id}** (Simulated)")
    return f"ACTION TAKEN: Endpoint '{endpoint_id}' isolation initiated. (Simulated)"

def block_ip_address_action(ip_address):
    print(f"\n[Response Agent - IPBlocker Tool]: **ACTION: Blocking IP address: {ip_address}** (Simulated)")
    return f"ACTION TAKEN: IP address '{ip_address}' blocked on firewall. (Simulated)"

def collect_forensic_data_action(endpoint_id):
    print(f"\n[Response Agent - ForensicDataCollector Tool]: **ACTION: Collecting forensic data from endpoint: {endpoint_id}** (Simulated)")
    return f"ACTION TAKEN: Forensic data collection initiated from endpoint '{endpoint_id}'. (Simulated)"

def generate_iac_terraform_action(remediation_steps):
    print(f"\n[Response Agent - TerraformCodeGenerator Tool]: **ACTION: Generating Terraform code for: {remediation_steps}** (Simulated)")
    terraform_code_example = f"""
    # Simulated Terraform code for remediation steps: {remediation_steps}
    resource "null_resource" "remediation" {{
      provisioner "local-exec" {{
        command = "echo 'Simulated remediation: {remediation_steps}'"
      }}
    }}
    """
    print(f"  - Simulated Terraform Code Example:\n```terraform\n{terraform_code_example}\n```")
    return f"ACTION TAKEN: Terraform code generated for remediation steps. (Simulated)\nExample Code:\n{terraform_code_example}"


triage_tools = [
    Tool(name="AlertDeduplicator", func=deduplicate_alerts_df, description="Useful to deduplicate alerts. Input: alerts DataFrame."),
    Tool(name="AlertGrouperByAsset", func=group_alerts_by_asset_df, description="Useful to group alerts by asset. Input: alerts DataFrame."),
    Tool(name="IOCEnricher", func=enrich_ioc_df, description="Useful to enrich alerts with threat intel. Input: alerts DataFrame."),
    Tool(name="MachineDetailsFetcher", func=get_machine_details_df, description="Useful to get machine details. Input: alerts DataFrame."),
    Tool(name="AccountDetailsFetcher", func=get_account_details_df, description="Useful to get account details. Input: alerts DataFrame.")
]

threat_hunting_tools = [
    Tool(name="SIEMQueryTool", func=query_siem_df, description="Useful to query SIEM data. Input: query description."),
    Tool(name="IndicatorClassifier", func=classify_indicator_df, description="Useful to classify indicators. Input: indicator value."),
    Tool(name="MITREATTACKMapper", func=map_ttp_mitre_attack_df, description="Useful to map TTPs to MITRE ATT&CK. Input: behavioral indicator.")
]

response_tools = [
    Tool(name="EndpointIsolator", func=isolate_endpoint_action, description="Useful to isolate endpoint. Input: endpoint ID."),
    Tool(name="IPBlocker", func=block_ip_address_action, description="Useful to block IP address. Input: IP address."),
    Tool(name="ForensicDataCollector", func=collect_forensic_data_action, description="Useful to collect forensic data. Input: endpoint ID."),
    Tool(name="TerraformCodeGenerator", func=generate_iac_terraform_action, description="Useful to generate Terraform code. Input: remediation steps description.")
]


triage_prefix = """You are a cybersecurity Triage Agent. Analyze alerts DataFrame, enrich data, decide escalation.

**Response Format Instructions:**

You MUST use the following format for your responses:

**If using a tool:**


Tools:
{tool_names}

Input DataFrame:
{input}
"""

triage_suffix = """Begin!"""
triage_prompt = ZeroShotAgent.create_prompt(
    triage_tools,
    prefix=triage_prefix,
    suffix=triage_suffix,
    input_variables=["input", "agent_scratchpad", "tool_names"]
)

triage_llm_chain = LLMChain(llm=llm, prompt=triage_prompt)

threat_hunting_prefix = """You are a Reactive Threat Hunting Agent. Investigate triage findings, enriched alerts DataFrame, use SIEM data, indicator analysis.

**Response Format Instructions:**

You MUST use the following format for your responses:

**If using a tool:**


Tools:
{tool_names}

Triage Findings and Enriched Alerts DataFrame:
{input}

"""
threat_hunting_suffix = """Begin!"""
threat_hunting_prompt = ZeroShotAgent.create_prompt(
    threat_hunting_tools,
    prefix=threat_hunting_prefix,
    suffix=threat_hunting_suffix,
    input_variables=["input", "agent_scratchpad", "tool_names"]
)
threat_hunting_llm_chain = LLMChain(llm=llm, prompt=threat_hunting_prompt)

response_prefix = """You are a Cybersecurity Response Agent. Take action based on Threat Hunting findings to contain and remediate the incident.

**Response Format Instructions:**

You MUST use the following format for your responses:

**If using a tool (taking an action):**


Tools:
{tool_names}

Threat Hunting Findings and Recommendations:
{input}
"""
response_suffix = """Begin!"""
response_prompt = ZeroShotAgent.create_prompt(
    response_tools,
    prefix=response_prefix,
    suffix=response_suffix,
    input_variables=["input", "agent_scratchpad", "tool_names"]
)
response_llm_chain = LLMChain(llm=llm, prompt=response_prompt)

triage_agent = ZeroShotAgent(llm_chain=triage_llm_chain, tools=triage_tools)
triage_agent_executor = AgentExecutor.from_agent_and_tools(agent=triage_agent, tools=triage_tools, verbose=True, handle_parsing_errors=True)

threat_hunting_agent = ZeroShotAgent(llm_chain=threat_hunting_llm_chain, tools=threat_hunting_tools)
threat_hunting_agent_executor = AgentExecutor.from_agent_and_tools(agent=threat_hunting_agent, tools=threat_hunting_tools, verbose=True, handle_parsing_errors=True)

response_agent = ZeroShotAgent(llm_chain=response_llm_chain, tools=response_tools)
response_agent_executor = AgentExecutor.from_agent_and_tools(agent=response_agent, tools=response_tools, verbose=True, handle_parsing_errors=True)


def run_apa_workflow_df(alert_data_json):
    print("\n--- Starting DataFrame-based APA Workflow ---")

    try:
        alerts_list = json.loads(alert_data_json)
        alerts_df_initial = pd.DataFrame(alerts_list)
        print("\n--- Initial Alerts DataFrame: ---")
        print(alerts_df_initial)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        return {"error": "Invalid JSON input"}

    print("\n--- Starting Triage Agent ---")
    triage_tool_names = [tool.name for tool in triage_tools] # Extract tool names
    triage_input = {"input": alerts_df_initial.to_string(), "tool_names": triage_tool_names} # Create input dictionary
    triage_output = triage_agent_executor.run(triage_input) # Pass input dictionary
    print(f"\n--- Triage Agent Output: ---\n{triage_output}")

    escalated = False
    enriched_alerts_df = None
    if "ESCALATE" in triage_output.upper():
        escalated = True
        print("\n--- Escalating to Reactive Threat Hunting Agent ---")

        enriched_alerts_df = enrich_ioc_df(alerts_df_initial)
        threat_hunting_input_str = f"Triage Agent escalated. Enriched Alerts DataFrame:\n{enriched_alerts_df.to_string()}"
        threat_hunting_tool_names = [tool.name for tool in threat_hunting_tools] # Extract tool names
        threat_hunting_input = {"input": threat_hunting_input_str, "tool_names": threat_hunting_tool_names} # Create input dictionary
        threat_hunting_output = threat_hunting_agent_executor.run(threat_hunting_input)
        print(f"\n--- Threat Hunting Agent Output: ---\n{threat_hunting_output}")

        print("\n--- Proceeding to Response Agent ---")
        response_input_str = f"Threat Hunting findings: {threat_hunting_output}. Enriched Alerts DataFrame (for context):\n{enriched_alerts_df.to_string()}"
        response_tool_names = [tool.name for tool in response_tools] # Extract tool names
        response_input = {"input": response_input_str, "tool_names": response_tool_names} # Create input dictionary
        response_output = response_agent_executor.run(response_input)
        print(f"\n--- Response Agent Output: ---\n{response_output}")

        return {
            "triage_result": triage_output,
            "threat_hunting_result": threat_hunting_output,
            "response_result": response_output,
            "escalated": escalated,
            "enriched_alerts_dataframe": enriched_alerts_df.to_dict() if enriched_alerts_df is not None else None
        }
    else:
        return {
            "triage_result": triage_output,
            "threat_hunting_result": "Not Escalated",
            "response_result": "Not Escalated",
            "escalated": escalated,
            "enriched_alerts_dataframe": None
        }


suricata_alert_data_json = """
[
  {
    "timestamp": "2024-01-20T10:00:00Z",
    "alert": {
      "category": "Malware Command and Control Activity",
      "signature": "ET MALWARE Observed DNS Query to .top domain - Likely Evil",
      "severity": 4
    },
    "src_ip": "192.168.1.100",
    "dest_ip": "8.8.8.8",
    "proto": "UDP",
    "src_port": 53456,
    "dest_port": 53,
    "dns": {
      "query": "maliciousdomain.top",
      "type": "A"
    }
  },
  {
    "timestamp": "2024-01-20T10:01:00Z",
    "alert": {
      "category": "Malware Command and Control Activity",
      "signature": "ET MALWARE Observed DNS Query to .top domain - Likely Evil",
      "severity": 4
    },
    "src_ip": "192.168.1.100",
    "dest_ip": "8.8.8.8",
    "proto": "UDP",
    "src_port": 53457,
    "dest_port": 53,
    "dns": {
      "query": "maliciousdomain.top",
      "type": "A"
    }
  }
]
"""

print("\n--- Example Suricata Alert Data (JSON): ---")
print(suricata_alert_data_json)

workflow_results_df = run_apa_workflow_df(suricata_alert_data_json)

print("\n--- DataFrame-based APA Workflow Complete ---")
print("\n--- Workflow Results (JSON): ---")
print(json.dumps(workflow_results_df, indent=2))

