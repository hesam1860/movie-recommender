# src/data_preparation.py
import pandas as pd
import json

# Load dataset files
movies = pd.read_csv('data/movies_metadata.csv', low_memory=False)
ratings = pd.read_csv('data/ratings.csv')
credits = pd.read_csv('data/credits.csv')
keywords = pd.read_csv('data/keywords.csv')
links = pd.read_csv('data/links.csv')

# Parse JSON-like columns
def parse_json(column):
    try:
        return [item['name'] for item in json.loads(column.replace("'", '"'))]
    except:
        return []

movies['genres'] = movies['genres'].apply(parse_json)
credits['cast'] = credits['cast'].apply(parse_json)
credits['crew'] = credits['crew'].apply(parse_json)
keywords['keywords'] = keywords['keywords'].apply(parse_json)

# Log initial row counts
print(f"Initial movies rows: {len(movies)}")
print(f"Initial ratings rows: {len(ratings)}")
print(f"Initial credits rows: {len(credits)}")
print(f"Initial keywords rows: {len(keywords)}")
print(f"Initial links rows: {len(links)}")

# Save parsed data for verification
movies.to_csv('data/movies_parsed.csv', index=False)