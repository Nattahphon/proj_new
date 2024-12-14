
# -----------------------------------------------------------------------------------------
import os
import pandas as pd
import matplotlib as plt
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType
import re
from F_Data_hub import DataHandler

temperature: float = 0.1

model: bool = [
    {
        "base_url": "https://api.opentyphoon.ai/v1",
        "model_name": "typhoon-v1.5x-70b-instruct"
    },
]

load_dotenv()

class PandasAgent:
    def __init__(self, temperature: float, base_url: str, model_name: str):
        self.handler = DataHandler()
        self.data = self.handler.load_data()
        self.df = self.handler.preprocess_data()
        self.temperature = temperature
        self.base_url = base_url
        self.model = model_name
        self.api_key = os.getenv("PANDAS_API_KEY")
        self.llm = self.initialize_llm()
        self.agent = self.create_agent()

    def initialize_llm(self) -> ChatOpenAI:
        """Initialize the language model."""
        return ChatOpenAI(
            base_url=self.base_url,
            model=self.model,
            api_key=self.api_key,
            temperature=self.temperature,
        )

    def create_agent(self):
        """Create a pandas DataFrame agent."""
        suffix = (
            "You are working with a DataFrame. The columns are: "
            + ", ".join([f"'{col}'" for col in self.df.columns]) + "."
        )
        return create_pandas_dataframe_agent(
            llm=self.llm,
            df=self.df,
            agent_type=AgentType.OPENAI_FUNCTIONS,
            suffix=suffix,
            verbose=True,
            allow_dangerous_code=True,
        )

    def get_dataframe(self):
        """Expose the DataFrame."""
        return self.df
    
    def extract_code_snippet(self, response: str) -> None:
        """
        Extract the Python code snippet from a response.

        Args:
            response (str): The raw response containing a code block.

        Returns:
            str: The extracted Python code snippet, or the entire response if no snippet is found.
        """
        match = re.search(r'```(?:python|code)?\n(.*?)\n```', response, re.DOTALL)
        if match:
            return match.group(1).strip()

        match = re.search(r'(?m)^ {4}(.*)$', response)
        if match:
            return "\n".join(line.strip() for line in response.splitlines() if line.startswith('    '))

    # def run(self, input_user: str):
    #     """Process a user query."""
    #     try:
    #         agent = self.agent
    #         response = agent.invoke({"input": input_user})
    #         code_to_execute = self.extract_code_snippet(response['output'])
    #         print(response['output'])
    #         exec_globals = {"df": self.df, "pd": pd, "plt":plt}  
    #         exec(code_to_execute, exec_globals) 
    #     except Exception as e:
    #         print(f"An error occurred: {e}")

    def run(self, input_user: str):
        """Process a user query."""
        try:
            agent = self.agent
            response = agent.invoke({"input": input_user})
            print(response['output'])
            code_to_execute = self.extract_code_snippet(response['output'])
            # print(f"Extracted Code: {code_to_execute}")
            if not isinstance(code_to_execute, str) or not code_to_execute.strip():
                raise ValueError("Extracted code snippet is not valid Python code.")
            # Prepare execution environment
            exec_globals = {"df": self.df, "pd": pd, "plt": plt}
            # Execute the code
            exec(code_to_execute, exec_globals)
        except Exception as e:
            print(f"An error occurred: {e}")
