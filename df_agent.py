import os
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.agents import AgentType
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dotenv import load_dotenv

from transformers import AutoTokenizer, AutoModelForCausalLM

# Load environment variables
load_dotenv()  # Load environment variables from .env file if it exists

# Set a random seed for reproducibility
np.random.seed(42)

# Generate sample data
n_rows = 1000

# Generate dates
start_date = datetime(2022, 1, 1)
dates = [start_date + timedelta(days=i) for i in range(n_rows)]

# Define data categories
makes = ['Toyota', 'Honda', 'Ford', 'Chevrolet', 'Nissan', 'BMW', 'Mercedes', 'Audi', 'Hyundai', 'Kia']
models = ['Sedan', 'SUV', 'Truck', 'Hatchback', 'Coupe', 'Van']
colors = ['Red', 'Blue', 'Black', 'White', 'Silver', 'Gray', 'Green']

# Create the dataset
data = {
    'Date': dates,
    'Make': np.random.choice(makes, n_rows),
    'Model': np.random.choice(models, n_rows),
    'Color': np.random.choice(colors, n_rows),
    'Year': np.random.randint(2015, 2023, n_rows),
    'Price': np.random.uniform(20000, 80000, n_rows).round(2),
    'Mileage': np.random.uniform(0, 100000, n_rows).round(0),
    'EngineSize': np.random.choice([1.6, 2.0, 2.5, 3.0, 3.5, 4.0], n_rows),
    'FuelEfficiency': np.random.uniform(20, 40, n_rows).round(1),
    'SalesPerson': np.random.choice(['Alice', 'Bob', 'Charlie', 'David', 'Eva'], n_rows)
}

# Create DataFrame and sort by date
df = pd.DataFrame(data).sort_values('Date')

# --- Mistral Integration (using Hugging Face transformers) ---
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

# 1. Choose a Mistral model from Hugging Face Hub (e.g., "mistralai/Mistral-7B-Instruct-v0.2")
model_name = "unsloth/mistral-7b-instruct-v0.1-bnb-4bit"  # Replace with your desired model

# 2. Determine the device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# 3. Load the model and tokenizer
model = None
tokenizer = None
try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto") # Let transformers handle device placement

except Exception as e:
    print(f"Error loading model or tokenizer: {e}")
    print(f"Please ensure the model '{model_name}' is available and you have sufficient resources and that bitsandbytes is installed (if required by the model).")
    print(f"Also, make sure CUDA is properly installed if you are trying to use the GPU.")

# 4. Create a text generation pipeline
if model is not None and tokenizer is not None:
    try:
        pipe = pipeline("text-generation",
                          model=model,
                          tokenizer=tokenizer,
                          torch_dtype=torch.float16,
                          max_new_tokens=8000) # Adjust max_new_tokens as needed


        # Create a dummy llm for the agent (we will actually be using the pipeline directly)
        from langchain.llms.base import LLM
        from typing import Any, List, Optional
        from langchain.callbacks.manager import CallbackManagerForLLMRun

        class HFTransformerLLM(LLM):
            pipeline: Any  # The Hugging Face pipeline

            @property
            def _llm_type(self) -> str:
                return "huggingface_pipeline"

            def _call(self, prompt: str, stop: Optional[List[str]] = None, run_manager: Optional[CallbackManagerForLLMRun] = None) -> str:
                try:
                    generated_text = self.pipeline(prompt, max_length=8000, pad_token_id=self.pipeline.tokenizer.eos_token_id)[0]['generated_text']  # Use pipeline for generation
                    return generated_text
                except Exception as e:
                    return f"Error during text generation: {e}"


        llm = HFTransformerLLM(pipeline=pipe)
        # print(llm)

        # Create the Pandas DataFrame agent
        agent = create_pandas_dataframe_agent(
            llm,
            df,
            verbose=True,
            allow_dangerous_code=True,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,  # Change agent type
        )
        print("Data Analysis Agent is ready. You can now ask questions about the data.")

        def ask_agent(question):
            """Function to ask questions to the agent and display the response"""
            try:
                response = agent.run(question)
                # print(f"Question: {question}")
                print(f"Answer %%%%%%%%%%%%%%%%%%: {response}")
            except Exception as e:
                print(f"Error during agent.run(): {e}")  # Detailed error for agent execution
            print("---")

        # Example Usage
        ask_agent("What are the column names in this dataset?")
        ask_agent("What is the average price of a car?")
        # ask_agent("Who are you ?, Who made you?")
        ask_agent("which type of car category manufucture most??")

    except Exception as e:
        print(f"Error creating the pipeline or agent: {e}")

else:
    print("Failed to load the model and tokenizer. Please check the error message and ensure you have the correct model name and dependencies installed.")

# import os
# from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
# from langchain.agents import AgentType
# import pandas as pd
# import numpy as np
# from datetime import datetime, timedelta
# from dotenv import load_dotenv

# from transformers import AutoTokenizer, AutoModelForCausalLM

# # Load environment variables
# load_dotenv()  # Load environment variables from .env file if it exists

# # Set a random seed for reproducibility
# np.random.seed(42)

# # Generate sample data
# n_rows = 1000

# # Generate dates
# start_date = datetime(2022, 1, 1)
# dates = [start_date + timedelta(days=i) for i in range(n_rows)]

# # Define data categories
# makes = ['Toyota', 'Honda', 'Ford', 'Chevrolet', 'Nissan', 'BMW', 'Mercedes', 'Audi', 'Hyundai', 'Kia']
# models = ['Sedan', 'SUV', 'Truck', 'Hatchback', 'Coupe', 'Van']
# colors = ['Red', 'Blue', 'Black', 'White', 'Silver', 'Gray', 'Green']

# # Create the dataset
# data = {
#     'Date': dates,
#     'Make': np.random.choice(makes, n_rows),
#     'Model': np.random.choice(models, n_rows),
#     'Color': np.random.choice(colors, n_rows),
#     'Year': np.random.randint(2015, 2023, n_rows),
#     'Price': np.random.uniform(20000, 80000, n_rows).round(2),
#     'Mileage': np.random.uniform(0, 100000, n_rows).round(0),
#     'EngineSize': np.random.choice([1.6, 2.0, 2.5, 3.0, 3.5, 4.0], n_rows),
#     'FuelEfficiency': np.random.uniform(20, 40, n_rows).round(1),
#     'SalesPerson': np.random.choice(['Alice', 'Bob', 'Charlie', 'David', 'Eva'], n_rows)
# }

# # Create DataFrame and sort by date
# df = pd.DataFrame(data).sort_values('Date')

# # --- Mistral Integration (using Hugging Face transformers) ---
# from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
# import torch

# # 1. Choose a Mistral model from Hugging Face Hub (e.g., "mistralai/Mistral-7B-Instruct-v0.2")
# model_name = "unsloth/mistral-7b-instruct-v0.1-bnb-4bit"  # Replace with your desired model

# # 2. Determine the device
# device = "cuda" if torch.cuda.is_available() else "cpu"
# print(f"Using device: {device}")

# # 3. Load the model and tokenizer
# model = None
# tokenizer = None
# try:
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto") # Let transformers handle device placement

# except Exception as e:
#     print(f"Error loading model or tokenizer: {e}")
#     print(f"Please ensure the model '{model_name}' is available and you have sufficient resources and that bitsandbytes is installed (if required by the model).")
#     print(f"Also, make sure CUDA is properly installed if you are trying to use the GPU.")

# # 4. Create a text generation pipeline
# if model is not None and tokenizer is not None:
#     try:
#         pipe = pipeline("text-generation",
#                           model=model,
#                           tokenizer=tokenizer,
#                           torch_dtype=torch.float16,
#                           max_new_tokens=512,
#                           return_full_text=False)


#         # Create a dummy llm for the agent (we will actually be using the pipeline directly)
#         from langchain.llms.base import LLM
#         from typing import Any, List, Optional
#         from langchain.callbacks.manager import CallbackManagerForLLMRun

#         class HFTransformerLLM(LLM):
#             pipeline: Any  # The Hugging Face pipeline

#             @property
#             def _llm_type(self) -> str:
#                 return "huggingface_pipeline"

#             def _call(self, prompt: str, stop: Optional[List[str]] = None, run_manager: Optional[CallbackManagerForLLMRun] = None) -> str:
#                 try:
#                     generated_text = self.pipeline(prompt, max_length=512, pad_token_id=self.pipeline.tokenizer.eos_token_id)[0]['generated_text']
#                     # Extract content between "Final Answer:" and the next "Question:" or end of string
#                     parts = generated_text.split("Final Answer:")
#                     if len(parts) > 1:
#                         answer_part = parts[1].strip()
#                         if "Question:" in answer_part:
#                             answer = answer_part.split("Question:")[0].strip()
#                         else:
#                             answer = answer_part
#                     else:
#                         answer = generated_text # If "Final Answer:" is missing, return the entire generated text

#                     # Attempt to clean numeric-only responses
#                     if answer.isdigit(): #if answer is just a number return "The answer is" + number
#                         return "The answer is " + answer
#                     else:
#                         return answer

#                 except Exception as e:
#                     return f"Error during text generation: {e}"


#         llm = HFTransformerLLM(pipeline=pipe)


#         # Create the Pandas DataFrame agent
#         agent = create_pandas_dataframe_agent(
#             llm,
#             df,
#             verbose=True,
#             allow_dangerous_code=True,
#             agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION # Changed back to ZERO_SHOT_REACT_DESCRIPTION as it fits better with HF models.
#         )
#         print("Data Analysis Agent is ready. You can now ask questions about the data.")

#         def ask_agent(question):
#             """Function to ask questions to the agent and display the response"""
#             try:
#                 response = agent.run(question)
#                 print(f"Question: {question}")
#                 print(f"Answer: {response}")
#             except Exception as e:
#                 print(f"Error during agent.run(): {e}")
#             print("---")

#         # Example Usage
#         ask_agent("What are the column names in this dataset?")
#         ask_agent("What is the average price of a car?")
#         ask_agent("Which salesperson sold the most cars?")

#     except Exception as e:
#         print(f"Error creating the pipeline or agent: {e}")

# else:
#     print("Failed to load the model and tokenizer. Please check the error message and ensure you have the correct model name and dependencies installed.")