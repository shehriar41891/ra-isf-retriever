import os
import pprint
from dotenv import load_dotenv
from langchain_community.utilities import GoogleSerperAPIWrapper
from crewai import Agent, Task, Crew, Process
import os
import litellm
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.embedding_model import get_embedding_model
from pinecone.grpc import PineconeGRPC as Pinecone


# Enable verbose mode for LiteLLM
litellm.set_verbose = True

# Load environment variables
load_dotenv()
openai_api = os.getenv('OPENAI_API')
pinecone_api_key = os.getenv('PINECONE_API_KEY')

os.environ['OPENAI_API_KEY'] = openai_api
os.environ['MODEL_NAME'] = 'gpt-3.5-turbo'


Short_Responder = Agent(
    role='Short answer giver',
    goal=(
        """"Analyze the document: {text} based on the user query: {query}. Extract **only** the exact
        word or phrase that directly answers the query, with **no supporting phrases or extra words**.
        Return only the specific answer. for example if query asked about what is capital of Pakistan
        then the answer should be 'Islamabad' only nothing else
        """
    ),
    verbose=True,
    memory=True,
    backstory=(
    """You are an expert document evaluator, skilled in extracting precise answers
    from text based on a query, with no added elaboration.
    """
    ),
    allow_delegation=True,
)

# Define the evaluation task
# Define the evaluation task
Short_answering = Task(
    description=(
        """Analyze the provided document `{text}` to extract the exact word or phrase 
        that directly answers the user query `{query}`. Avoid any additional context, 
        explanations, or supporting phrases. Return the precise answer only."""
    ),
    expected_output="A single word or phrase that directly answers the query, with no additional text.",
    agent=Short_Responder,
    allow_delegation=True,
)

# Create the crew
crew = Crew(
    agents=[Short_Responder],
    tasks=[Short_answering],
    verbose=True,
    process=Process.sequential,
    debug=True,
    max_iterations=2,
)

def relevant_answer(results, user_query):
    result = crew.kickoff(inputs={'text': results, 'query': user_query})
    return result

# Example input
text = """
The name Cynthia is of Greek origin and comes from the Greek word Kynthía, which means
"from Mount Cynthus" on the island of Delos. It was originally an epithet for the Greek 
goddess of the moon, Artemis, who was said to have been born on Mount Cynthus. 
"""
query = "what is the origin of the name cynthia?"

# Call the function
# relevant_result = relevant_answer(text,query)

# # Output the result
# print(relevant_result)
