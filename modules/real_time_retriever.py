import requests
import os
from dotenv import load_dotenv

load_dotenv()

SERPER_API_KEY = os.getenv('SERPER_API_KEY')

# Function to get job done using Serper API
def get_search_results(query, api_key):
    """
    Function to send a query to Serper API and get search results.

    Parameters:
    - query (str): The search query to be sent to Serper API.
    - api_key (str): The Serper API key for authentication.

    Returns:
    - dict: Parsed JSON response with search results.
    """
    # Define the base URL for Serper API
    url = "https://api.serper.dev/search"

    # Headers for the API request (adding the API key in the headers)
    headers = {
        "Authorization": f"Bearer {api_key}"
    }

    # Parameters for the API request
    params = {
        "q": query,  # The query to search for
    }

    try:
        # Send the GET request to the Serper API
        response = requests.get(url, headers=headers, params=params)

        # Check for successful request
        if response.status_code == 200:
            # Parse the JSON response
            return response.json()
        else:
            # Handle errors if the status code is not 200 (OK)
            return {"error": f"Error: {response.status_code}, Message: {response.text}"}
    except requests.exceptions.RequestException as e:
        # Handle network errors or other request exceptions
        return {"error": f"Request failed: {str(e)}"}

# Example usage
if __name__ == "__main__":
    # Your Serper API key
    api_key = SERPER_API_KEY
    print(api_key)
    # The query you want to search for
    query = "weather forecast in New York"

    # Call the function to get search results
    result = get_search_results(query, api_key)

    # Print the result or handle it as needed
    if "error" in result:
        print(result["error"])
    else:
        # Example: Print first 5 search results (if available)
        print("Top 5 search results:")
        for i, item in enumerate(result.get("organic_results", [])[:5]):
            print(f"{i + 1}. {item['title']}: {item['link']}")
