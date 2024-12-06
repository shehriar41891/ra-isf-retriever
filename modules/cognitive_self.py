from crewai import Agent, Task, Crew, Process
from models.llm_model import get_llm
from dotenv import load_dotenv
import os 
import litellm
# from test.test_queries import spotifyQueries

litellm.set_verbose=True

load_dotenv()

openai_api = os.getenv('OPENAI_API')
# os.environ['OPENAI_API'] = openai_api
os.environ['MODEL_NAME'] = 'gpt-3.5-turbo'

litellm.api_key = openai_api

# llm = get_llm()
# print('The llm from cognitive side is ',llm)

Knowledge_Confidence_Evaluator = Agent(
    role='Query Answer Assistant',
    goal="""
    Evaluate the query {query} to identify the specific sector it refers to. For example,
    if the query asks about tracking a shipment but does not specify which company, respond with
    'Not sure'. Only provide an answer 
    if you are 100% certain about the sector being discussed. Otherwise, respond with 'Not sure'.
    """,
    verbose=True,
    memory=True,
    backstory=(
        """
        You are a query Evaluator who is expert in answering 'sure' or 'not sure' based on the query 
        and your knowledge
        """
    ),
    allow_delegation=True
)

ConfidenceEvaluation = Task(
    description=(
        """Analyze the query {query}. Answer only if you are 100% sure. 
        If you are unsure, respond with 'Not sure'. Follow this format:
        """
    ),
    expected_output="A precise and accurate answer or 'Not sure' if you are not 100% confident, using the correct format.",
    agent=Knowledge_Confidence_Evaluator,
    allow_delegation=True
)

crew = Crew(
    agents=[Knowledge_Confidence_Evaluator],
    tasks=[ConfidenceEvaluation],
    verbose=True,
    process=Process.sequential,
    debug=True,
    max_iterations=2
)

def get_result(user_query):
    result = crew.kickoff(inputs={'query' : user_query})
    return result 