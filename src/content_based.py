# src/content_based.py
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import ast

# Load preprocessed data
movies = pd.read_csv('data/movies_processed.csv')

# Convert stringified lists to actual lists
movies['genres'] = movies['genres'].apply(ast.literal_eval)
movies['keywords'] = movies['keywords'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else [])
movies['cast'] = movies['cast'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else [])
movies['crew'] = movies['crew'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else [])

# Create text features by combining genres, keywords, top 3 cast, and director
def get_director(crew):
    if isinstance(crew, list) and crew:
        return crew[0]  # Return first crew member as fallback
    return ''

movies['director'] = movies['crew'].apply(get_director)
movies['text_features'] = movies.apply(
    lambda x: ' '.join(
        (x['genres'] if isinstance(x['genres'], list) else []) +
        (x['keywords'] if isinstance(x['keywords'], list) else []) +
        (x['cast'][:3] if isinstance(x['cast'], list) else []) +
        ([x['director']] if x['director'] else [])
    ), axis=1
)

# Debug: Print sample text features
print("Sample text features:", movies['text_features'].head().tolist())

# Compute TF-IDF matrix
tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
tfidf_matrix = tfidf.fit_transform(movies['text_features'].fillna(''))

# Compute cosine similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Content-based recommender
def content_based_recommender(movie_id, k=5):
    idx = movies[movies['movieId'] == movie_id].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:k+1]
    movie_indices = [i[0] for i in sim_scores]
    return movies[['movieId', 'title']].iloc[movie_indices]

# Example: Recommend for The Shawshank Redemption (movieId=318)
recommendations = content_based_recommender(318)
print("Recommendations for The Shawshank Redemption:")
print(recommendations)

# Save recommendations
recommendations.to_csv('models/content_based_318.csv', index=False)