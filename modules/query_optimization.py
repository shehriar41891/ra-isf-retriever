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

Query_Optimizater = Agent(
    role='Query Optimizer Assitance',
    goal="""
    You will be give user query: {query} you need to optimzed the query by removing redundant words,
    avoiding unnecessary phrases or words try to focus only on the important part of the query
    """,
    verbose=True,
    memory=True,
    backstory=(
        """
        You are a query optimizer you takes a query in and optimize it by removing redundant words,
        avoiding uncessary details and only focusing and huglighting the important part of the query
        """
    ),
    allow_delegation=True
)

Query_Optimization = Task(
    description=(
        """Optimize the given query: {query} by removing the uncessary details and redundant words
        """
    ),
    expected_output="A precise and to the point query",
    agent=Query_Optimizater,
    allow_delegation=True
)

crew = Crew(
    agents=[Query_Optimizater],
    tasks=[Query_Optimization],
    verbose=True,
    process=Process.sequential,
    debug=True,
    max_iterations=2
)

def get_result(user_query):
    result = crew.kickoff(inputs={'query' : user_query})
    return result 

query = """"
        Can you kindly tell me, in the most clear and detailed manner, what the weather forecast will be 
        like for tomorrow in the city of New York, including any possible chances of rain, snow, or 
        other weather events, and maybe even details like the temperature in both Fahrenheit and 
        Celsius, and wind speed, and if it will be a sunny or cloudy day, or if there are any other
        details that could be relevant for someone planning to go outside tomorrow?
    """

print(get_result(query))