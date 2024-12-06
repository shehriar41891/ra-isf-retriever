import os
from typing import List
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# Ensure the API key is loaded from the environment
openai_api_key = os.getenv("OPENAI_API")

class OpenAIEmbedding:
    """
    A wrapper class for generating embeddings using OpenAI's API.
    """

    def __init__(self, model: str = "text-embedding-3-small"):
        """
        Initializes the OpenAI client and model.

        Args:
            model (str): The OpenAI embedding model to use. Default is 'text-embedding-3-small'.
        """
        if not openai_api_key:
            raise EnvironmentError("OpenAI API key not found. Please set it in your environment variables.")
        
        self.client = OpenAI(api_key=openai_api_key)
        self.model = model

    def get_embedding(self, text: str) -> List[float]:
        """
        Generates an embedding for a single piece of text.

        Args:
            text (str): Input text to embed.

        Returns:
            List[float]: A list of float values representing the embedding.
        """
        text = text.replace("\n", " ")  # Replace newlines with spaces
        response = self.client.embeddings.create(input=[text], model=self.model)
        return response.data[0].embedding

def get_embedding_model() -> OpenAIEmbedding:
    """
    Factory function to create an instance of the OpenAIEmbedding class.

    Returns:
        OpenAIEmbedding: An instance of OpenAIEmbedding.
    """
    return OpenAIEmbedding()

if __name__ == "__main__":
    # Example usage
    embedding_model = get_embedding_model()
    test_text = "Llama Index is a great framework for information retrieval."
    embedding = embedding_model.get_embedding(test_text)

    print(f"Generated Embedding (truncated): {embedding[:10]} ...")  # Print the first 10 values