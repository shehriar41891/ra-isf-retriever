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

# Configure LiteLLM with API key
litellm.api_key = openai_api
embedding_model = get_embedding_model()

SERPER_API_KEY = os.getenv('SERPER_API_KEY')

os.environ["SERPER_API_KEY"] = SERPER_API_KEY

def Googlesearch(query):
    search = GoogleSerperAPIWrapper()
    result = search.run(query)
    
    return result


user_query = 'Who was the first president of Pakistan'
results = Googlesearch(user_query)

# Define the evaluator agent
Evaluator = Agent(
    role='Evaluation of the documents',
    goal=(
        """"Evaluate the given document: {text} based on the user query: {query} and only extract the 
        relevant part from the document according to the {query}
        """
    ),
    verbose=True,
    memory=True,
    backstory=(
    """You are a document evaluator who is expert in extracting relevant information from 
    the text based on the given query
    """
    ),
    allow_delegation=True,
)

# Define the evaluation task
Evaluation = Task(
    description=(
        """" Analyze the document: {text} based on the user query: {query} and take only relevant part 
        from the document based on the query: {query}
        """
    ),
    expected_output="small relevant document to the user query",
    agent=Evaluator,
    allow_delegation=True,
)

# Create the crew
crew = Crew(
    agents=[Evaluator],
    tasks=[Evaluation],
    verbose=True,
    process=Process.sequential,
    debug=True,
    max_iterations=2,
)

def filter_top(user_query, results):
    result = crew.kickoff(inputs={'text': results, 'query': user_query})
    return result


#storing the result to pinecone
pc = Pinecone(api_key=pinecone_api_key)
index = pc.Index(host="upwork")

def upsert_to_pinecone():
    metadata = filter_top(user_query,results)
    embeddings = embedding_model.get_embedding(metadata)

    vector = [
    {
    'values' : embeddings,
    'metadata': metadata
    }
    ]
    
    index.upsert(
        vector,
        namespace='namespace6'
    )
    
    
    