# # Cybersecurity AI Agent with Hugging Face Models
# Requirements: Python 3.10+, Linux, NVIDIA GPU (recommended)

# Install dependencies
# !pip install -qU langgraph langchain transformers torch accelerate sentencepiece requests python-iptables

from langgraph.graph import StateGraph, END
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from typing import TypedDict
import requests
import json
import subprocess
from pprint import pprint

# 1. Load Hugging Face Model (Zephyr-7B-beta example)
model_name = "unsloth/mistral-7b-instruct-v0.1-bnb-4bit"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",  # Automatically uses GPU if available
    torch_dtype="auto"
)

# Create text generation pipeline
text_generation_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=1024,
    temperature=0.3,
    repetition_penalty=1.1
)

# Wrap in LangChain interface
llm = HuggingFacePipeline(pipeline=text_generation_pipeline)

# 2. Define Cybersecurity Tools
class CyberSecurityTools:
    @staticmethod
    def check_ioc(indicator: str) -> dict:
        """Mock threat intelligence lookup"""
        ioc_db = {
            "185.142.239.85": {"malicious": True, "threat_type": "C2 Server"},
            "192.168.1.101": {"malicious": False}
        }
        return ioc_db.get(indicator, {"malicious": False})
    
    @staticmethod
    def block_ip(ip: str) -> str:
        """Block IP using iptables"""
        try:
            cmd = f"sudo iptables -A INPUT -s {ip} -j DROP"
            subprocess.run(cmd, shell=True, check=True)
            return f"Blocked {ip} successfully"
        except Exception as e:
            return f"Block failed: {str(e)}"

# 3. Define LangGraph State
class AgentState(TypedDict):
    alert: dict
    ioc_data: dict
    analysis: dict
    action: str

# 4. Build Workflow Nodes
def triage_alert(state: AgentState):
    """Initial alert processing"""
    if state["alert"]["severity"] >= 3:
        return {"decision": "investigate"}
    return {"decision": "ignore"}

def enrich_ioc(state: AgentState):
    """IOC enrichment"""
    ip = state["alert"]["src_ip"]
    return {"ioc_data": CyberSecurityTools.check_ioc(ip)}

def analyze_threat(state: AgentState):
    """LLM-based threat analysis"""
    prompt = f"""Analyze this security alert and map to MITRE ATT&CK:
    
    Alert Details:
    {json.dumps(state["alert"])}
    
    IOC Analysis:
    {json.dumps(state["ioc_data"])}
    
    Respond in JSON format with:
    - ttp_id: MITRE TTP ID
    - confidence: 0-100
    - recommended_action: string"""

    response = llm.invoke(prompt)
    try:
        return {"analysis": json.loads(response)}
    except:
        return {"analysis": {"ttp_id": "T1043", "confidence": 75, "recommended_action": "Block IP"}}

def take_action(state: AgentState):
    """Execute response action"""
    if state["analysis"].get("confidence", 0) > 60:
        result = CyberSecurityTools.block_ip(state["alert"]["src_ip"])
        return {"action": result}
    return {"action": "No action taken"}

# 5. Create Workflow Graph
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("triage", triage_alert)
workflow.add_node("enrich", enrich_ioc)
workflow.add_node("analyze", analyze_threat)
workflow.add_node("act", take_action)

# Define edges
workflow.set_entry_point("triage")

workflow.add_edge("triage", "enrich")
workflow.add_edge("enrich", "analyze")
workflow.add_conditional_edges(
    "analyze",
    lambda x: "act" if x["analysis"].get("confidence", 0) > 60 else END
)
workflow.add_edge("act", END)

app = workflow.compile()

# 6. Test with Sample Suricata Alert
sample_alert = {
    "timestamp": "2025-02-15T09:32:15.123456Z",
    "src_ip": "185.142.239.85",  # Known malicious IP
    "dest_ip": "192.168.1.101",
    "signature": "ET DROP Spamhaus DROP Listed Traffic Inbound",
    "severity": 3
}

print("Processing alert:")
pprint(sample_alert)

result = app.invoke({"alert": sample_alert})

print("\nFinal Result:")
pprint(result)