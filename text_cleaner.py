import re

def clean_text(text):
    # Example: Fix line breaks and remove noise
    text = re.sub(r'\n+', ' ', text)  # Replace multiple newlines with spaces
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    return text.strip()
