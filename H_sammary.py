import os
import pandas as pd
import logging
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType


class SummaryAgent:
    """A class to summarize the outputs of the PandasAgent."""

    def __init__(self, temperature: float, base_url: str, model_name: str):
        self.llm = self.initialize_llm(temperature, base_url, model_name)

    @staticmethod
    def initialize_llm(temperature: float, base_url: str, model_name: str) -> ChatOpenAI:
        """Initialize the LLM for summarization."""
        api_key = os.getenv("PLOT_API_KEY")
        if not api_key:
            raise ValueError("API key is missing. Ensure 'API_KEY' is set in your environment.")
        return ChatOpenAI(
            base_url=base_url,
            model=model_name,
            api_key=api_key,
            temperature=temperature,
        )

    def summarize(self, text: str) -> str:
        """Generate a summary of the given text."""
        prompt = f"Summarize the following content in a concise and clear manner:\n\n{text}"
        try:
            response = self.llm.invoke({"input": prompt})
            return response["output"]
        except Exception as e:
            logging.error(f"Error during summarization: {e}")
            return "An error occurred while generating the summary."
