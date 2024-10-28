import os
import time

from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from typing import List, Dict
from dotenv import load_dotenv

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

llm = ChatOpenAI(model="gpt-4o-mini", max_tokens=1000, temperature=0.7)

# Define the base Agent class
class Agent:
    def __init__(self, name: str, role: str, skills: List[str]):
        self.name = name
        self.role = role
        self.skills = skills
        self.llm = llm

    def process(self, task: str, context: List[Dict] = None) -> str:
        messages = [
            SystemMessage(content=f"You are {self.name}, a {self.role}. Your skills include: {', '.join(self.skills)}. Respond to the task based on your role and skills.")
        ]
        
        if context:
            for msg in context:
                if msg['role'] == 'human':
                    messages.append(HumanMessage(content=msg['content']))
                elif msg['role'] == 'ai':
                    messages.append(AIMessage(content=msg['content']))
        
        messages.append(HumanMessage(content=task))
        response = self.llm.invoke(messages)
        return response.content
    

# Define specialized agents: HistoryResearchAgent and DataAnalysisAgent
class HistoryResearchAgent(Agent):
    def __init__(self):
        super().__init__("Clio", "History Research Specialist", ["deep knowledge of historical events", "understanding of historical contexts", "identifying historical trends"])

class DataAnalysisAgent(Agent):
    def __init__(self):
        super().__init__("Data", "Data Analysis Expert", ["interpreting numerical data", "statistical analysis", "data visualization description"])

# Define the different functions for the collaboration system

# Research Historical Context
def research_historical_context(history_agent, task: str, context: list) -> list:
    print("🏛️ History Agent: Researching historical context...")
    history_task = f"Provide relevant historical context and information for the following task: {task}"
    history_result = history_agent.process(history_task)
    context.append({"role": "ai", "content": f"History Agent: {history_result}"})
    print(f"📜 Historical context provided: {history_result[:100]}...\n")
    return context

# Identify Data Needs
def identify_data_needs(data_agent, task: str, context: list) -> list:
    print("📊 Data Agent: Identifying data needs based on historical context...")
    historical_context = context[-1]["content"]
    data_need_task = f"Based on the historical context, what specific data or statistical information would be helpful to answer the original question? Historical context: {historical_context}"
    data_need_result = data_agent.process(data_need_task, context)
    context.append({"role": "ai", "content": f"Data Agent: {data_need_result}"})
    print(f"🔍 Data needs identified: {data_need_result[:100]}...\n")
    return context

# Provide Historical Data
def provide_historical_data(history_agent, task: str, context: list) -> list:
    print("🏛️ History Agent: Providing relevant historical data...")
    data_needs = context[-1]["content"]
    data_provision_task = f"Based on the data needs identified, provide relevant historical data or statistics. Data needs: {data_needs}"
    data_provision_result = history_agent.process(data_provision_task, context)
    context.append({"role": "ai", "content": f"History Agent: {data_provision_result}"})
    print(f"📊 Historical data provided: {data_provision_result[:100]}...\n")
    return context

# Analyze Data
def analyze_data(data_agent, task: str, context: list) -> list:
    print("📈 Data Agent: Analyzing historical data...")
    historical_data = context[-1]["content"]
    analysis_task = f"Analyze the historical data provided and describe any trends or insights relevant to the original task. Historical data: {historical_data}"
    analysis_result = data_agent.process(analysis_task, context)
    context.append({"role": "ai", "content": f"Data Agent: {analysis_result}"})
    print(f"💡 Data analysis results: {analysis_result[:100]}...\n")
    return context

# Synthesize Final Answer
def synthesize_final_answer(history_agent, task: str, context: list) -> str:
    print("🏛️ History Agent: Synthesizing final answer...")
    synthesis_task = "Based on all the historical context, data, and analysis, provide a comprehensive answer to the original task."
    final_result = history_agent.process(synthesis_task, context)
    return final_result


# HistoryDataCollaborationSystem Class
class HistoryDataCollaborationSystem:
    def __init__(self):
        self.history_agent = Agent("Clio", "History Research Specialist", ["deep knowledge of historical events", "understanding of historical contexts", "identifying historical trends"])
        self.data_agent = Agent("Data", "Data Analysis Expert", ["interpreting numerical data", "statistical analysis", "data visualization description"])

    def solve(self, task: str, timeout: int = 300) -> str:
        print(f"\n👥 Starting collaboration to solve: {task}\n")
        
        start_time = time.time()
        context = []
        
        steps = [
            (research_historical_context, self.history_agent),
            (identify_data_needs, self.data_agent),
            (provide_historical_data, self.history_agent),
            (analyze_data, self.data_agent),
            (synthesize_final_answer, self.history_agent)
        ]
        
        for step_func, agent in steps:
            if time.time() - start_time > timeout:
                return "Operation timed out. The process took too long to complete."
            try:
                result = step_func(agent, task, context)
                if isinstance(result, str):
                    return result  # This is the final answer
                context = result
            except Exception as e:
                return f"Error during collaboration: {str(e)}"
        
        print("\n✅ Collaboration complete. Final answer synthesized.\n")
        return context[-1]["content"]
    

# Example Usage
collaboration_system = HistoryDataCollaborationSystem()
# task = "Analyze the impact of the Industrial Revolution on the global economy and social structures."
task = "How will raise in AI and AI taking over jobs affect housing prices?."
result = collaboration_system.solve(task)
print(f"📚 Final Answer: {result}")

