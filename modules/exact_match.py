import os
import re

# Function to Normalize Text (removing unimportant parts, including specific words like "The")
def normalize_text(text):
    """
    Preprocess the text to remove unnecessary words, titles, suffixes, extra spaces,
    and common words like "The" before returning a normalized version of the string.
    """
    # Convert to lower case for case-insensitive comparison
    text = text.lower()

    # Remove common suffixes or honorary titles (e.g., 'ForMemRS', 'PhD', 'Dr.')
    text = re.sub(r"(formemrs|phd|dr|prof|sir|mrs|mr|ms|jr|ii|iii|iv|v|m\.d\.|b\.a\.)", "", text)

    # Remove common words like 'the'
    text = re.sub(r"\bthe\b", "", text)

    # Remove extra spaces and punctuation
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation marks

    # Trim leading/trailing spaces
    text = text.strip()

    return text


# Function to check Exact Match after Normalization
def exact_match_score(predicted: str, target: str) -> int:
    """
    Normalize both predicted and target values, then check for exact match.
    Returns:
        1 if predicted matches target exactly, else 0.
    """
    # Normalize the predicted and target values
    normalized_predicted = normalize_text(predicted)
    normalized_target = normalize_text(target)

    # Exact match comparison
    if normalized_predicted == normalized_target:
        return 1
    else:
        return 0


# Define predicted and target values
target_value = 'The falsa'
predicted_value = 'falsa'

# Calculate Exact Match score
result = exact_match_score(predicted_value, target_value)

# Print the result
print(f"Exact match result: {result}")  # Output: 1 if they match after normalization, 0 otherwise
