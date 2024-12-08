import requests

def get_article_revision_before_2019(article_title):
    endpoint = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "titles": article_title,
        "prop": "revisions",
        "rvlimit": 1,  # Limit to the most recent revision before 2019
        "rvend": "2018-12-31T23:59:59Z",  # Ensure revisions before 2019
        "format": "json",
    }

    response = requests.get(endpoint, params=params)
    data = response.json()

    page_id = next(iter(data['query']['pages']))  # Get the page ID
    if 'revisions' in data['query']['pages'][page_id]:
        revision_content = data['query']['pages'][page_id]['revisions'][0]
        return revision_content
    else:
        return "No revisions found for this article before 2019."

# Example usage:
article_title = "Python_(programming_language)"  # Replace with your desired article
revision = get_article_revision_before_2019(article_title)
print(revision)
