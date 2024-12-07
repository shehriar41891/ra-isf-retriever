import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
import litellm

# Enable verbose mode for LiteLLM
litellm.set_verbose = True

# Load environment variables
load_dotenv()
openai_api = os.getenv('OPENAI_API')

os.environ['OPENAI_API_KEY'] = openai_api
os.environ['MODEL_NAME'] = 'gpt-3.5-turbo'

# Define the responder agent
Short_Responder = Agent(
    role='Short answer giver',
    goal=(
        """Answer the user's query: {query}. Extract **only** the exact
        word or phrase that directly answers the query, with **no supporting phrases or extra words**.
        For example, if the query asks about the capital of Pakistan, the answer should be 'Islamabad' only.
        """
    ),
    verbose=True,
    memory=False,
    backstory=(
        """You are an expert answer provider, skilled in extracting precise answers
        from knowledge and queries with no added elaboration."""
    ),
    allow_delegation=True,
)

# Define the task
Short_answering = Task(
    description=(
        """Analyze the user query `{query}` and provide the exact word or phrase 
        that directly answers the query. Avoid any additional context, 
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

def direct_relevant_answer(user_query):
    result = crew.kickoff(inputs={'query': user_query})
    return result

# Example input
query = "What is the origin of the name Cynthia?"

# Call the function
relevant_result = direct_relevant_answer(query)

# Output the result
print(relevant_result)
