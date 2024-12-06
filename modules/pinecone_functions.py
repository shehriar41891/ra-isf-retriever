from dotenv import load_dotenv
from pinecone.grpc import PineconeGRPC as Pinecone
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.embedding_model import get_embedding_model

load_dotenv()

# Retrieve Pinecone credentials from environment variables
pinecone_api_key = os.getenv('PINECONE_API_KEY')
pinecone_env = os.getenv('PINECONE_ENV')

# Validate that required environment variables are set
if not pinecone_api_key or not pinecone_env:
    raise ValueError("Missing Pinecone API key or environment. Please check your .env file.")

try:
    pc = Pinecone(api_key=pinecone_api_key)
    index = pc.Index("upwork")
    print("Pinecone client initialized successfully.")
except Exception as e:
    raise RuntimeError(f"Failed to initialize Pinecone client: {e}")

embedding_model = get_embedding_model()

#query from pinecone
def query_data(query: str, namespace: str = "namespace6", top_k: int = 10):
    """
    Queries the vector database using an embedded version of the input query.

    Args:
        query (str): The text query to search for.
        namespace (str, optional): The namespace within the vector database to query. Defaults to "example-namespace".
        top_k (int, optional): The number of top results to retrieve. Defaults to 10.

    Returns:
        dict: The query results containing matched vectors, metadata, and scores.
    """
    if not query or not isinstance(query, str):
        raise ValueError("Query must be a non-empty string.")
    
    # if not hasattr(embedding_model, 'embed_query') or not callable(embedding_model.embed):
    #     raise AttributeError("Embedding model must have a callable 'embed_query' method.")

    try:
        # Embed the query text
        print(dir(embedding_model))
        embedded_text = embedding_model.get_embedding(query)

        # Query the vector database
        res = index.query(
            namespace=namespace,
            vector=embedded_text,
            top_k=top_k,
            include_values=False,
            include_metadata=True
        )

        return res
    
    except Exception as e:
        raise RuntimeError(f"An error occurred while querying the database: {e}")
    
    
    
print(query_data('How can I check my data usage'))