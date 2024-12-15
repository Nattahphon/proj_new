# -----------------------------------------------------------------------
# this is main
from dotenv import load_dotenv
import os
from langchain.agents import AgentExecutor, create_react_agent
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
from langchain import hub
from H_pandas import PandasAgent
from langchain_core.tools import Tool
from H_sammary import SummaryAgent

class TyphoonAgent:
    def __init__(self, temperature: float, base_url: str, model_name: str, dataset_paths: dict):
        self.temperature = temperature
        self.base_url = base_url
        self.model = model_name
        self.api_key = os.getenv("TYPHOON_API_KEY")
        self.llm = self.initialize_llm()
        self.memory = self.initialize_memory()
        self.pandas_agent = PandasAgent(temperature, base_url, model_name, dataset_paths)
        self.summary_agent = SummaryAgent(temperature, base_url, model_name)
        self.tools = self.initialize_tools()
        self.agent = self.create_agent()
        self.agent_executor = self.create_agent_executor()

    def initialize_llm(self) -> ChatOpenAI:
        """Initialize the language model."""
        return ChatOpenAI(
            base_url=self.base_url,
            model=self.model,
            api_key=self.api_key,
            temperature=self.temperature,
        )

    def initialize_memory(self):
        """Set up memory for the conversation."""
        return ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    def initialize_tools(self):
        """Initialize the tools used by the TyphoonAgent."""
        pandas_tool = Tool(
            name="pandas_agent",
            func=self.query_dataframe,
            description="Usefull when you need to Consult the PandasAgent to analyze and visualize data in a DataFrame.",
        )

        summary_tool = Tool(
            name="summary_agent",
            func=self.summary_answer,
            description="Usefull when you need to summarizing responses from other agents or condensing user input for clarity and concise communication.",
        )
        return [pandas_tool, summary_tool]
    
    def summary_answer(self, user_input: str) -> None:
        return self.summary_agent.summarize(user_input)

    def query_dataframe(self, user_input: str) -> None:
        """
        Delegate the user query to the PandasAgent for processing.

        Args:
            user_input (str): The user's query related to the DataFrame.

        Returns:
            str: The processed result from the PandasAgent.
        """
        return self.pandas_agent.run(user_input)
    

    def create_agent(self):
        """Create a React agent that works with tools."""
        react_prompt = hub.pull(os.getenv("REACT_PROMPT", "hwchase17/react-chat"))
        return create_react_agent(llm=self.llm, tools=self.tools, prompt=react_prompt)

    def create_agent_executor(self):
        """Create the agent executor to handle queries."""
        return AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            memory=self.memory,
            verbose=True,
            max_iterations=int(os.getenv("MAX_ITERATIONS", 20)),
            handle_parsing_errors=True,
        )

    def process_query(self, user_input: str) -> None:
        """Process user input by delegating to the appropriate agent/tool."""
        try:
            print("> Entering TyphoonAgent...")
            response = self.agent_executor.invoke({"input": user_input})
            print("> Finished TyphoonAgent Response:")
            print(response["output"])
        except Exception as e:
            print(f"An error occurred: {e}")

    def run(self):
        """Start the TyphoonAgent loop for user interaction."""
        print("Welcome to the Typhoon Agent. Type 'stop agent' to exit.")
        while True:
            user_input = input("Enter your query: ")
            if user_input.lower() == "stop agent":
                self.memory.clear()
                print("Exiting Typhoon Agent...")
                break
            self.process_query(user_input)

if __name__ == "__main__":
    load_dotenv()
    
    filepaths = ['./McDonald_s_Reviews.csv', './Financials.csv']

    def get_filepath(filepaths: list) -> bool:
        return {filepath.split('/')[-1]: filepath for filepath in filepaths}
    
    file_paths = get_filepath(filepaths=filepaths)

    agent = TyphoonAgent(
        temperature=0.1,
        base_url="https://api.opentyphoon.ai/v1",
        model_name="typhoon-v1.5x-70b-instruct", 
        dataset_paths=file_paths
    )
    agent.run()
