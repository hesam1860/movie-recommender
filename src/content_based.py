# src/content_based.py
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import ast

# Load data
movies = pd.read_csv('data/movies_processed.csv')
ratings = pd.read_csv('data/ratings_small.csv')

# Convert stringified lists
movies['genres'] = movies['genres'].apply(ast.literal_eval)
movies['keywords'] = movies['keywords'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else [])
movies['cast'] = movies['cast'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else [])
movies['crew'] = movies['crew'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else [])

# Create text features
def get_director(crew):
    return crew[0] if isinstance(crew, list) and crew else ''

movies['director'] = movies['crew'].apply(get_director)
movies['text_features'] = movies.apply(
    lambda x: ' '.join(
        (x['genres'] if isinstance(x['genres'], list) else []) +
        (x['keywords'] if isinstance(x['keywords'], list) else []) +
        (x['cast'][:3] if isinstance(x['cast'], list) else []) +
        ([x['director']] if x['director'] else []) +
        ([str(x['overview'])] if pd.notna(x['overview']) else []) +
        ([str(x['tagline'])] if pd.notna(x['tagline']) else [])
    ), axis=1
)

# Compute TF-IDF matrix
tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
tfidf_matrix = tfidf.fit_transform(movies['text_features'].fillna(''))
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Build user profile
def build_user_profile(user_id):
    user_ratings = ratings[ratings['userId'] == user_id]
    if user_ratings.empty:
        return np.zeros(tfidf_matrix.shape[1])  # Cold-start: zero vector
    weighted_vectors = []
    for _, row in user_ratings.iterrows():
        movie_idx = movies[movies['movieId'] == row['movieId']].index
        if not movie_idx.empty:
            rating = row['rating'] - user_ratings['rating'].mean()  # Center ratings
            weighted_vectors.append(tfidf_matrix[movie_idx[0]].toarray() * rating)
    return np.sum(weighted_vectors, axis=0) / len(weighted_vectors) if weighted_vectors else np.zeros(tfidf_matrix.shape[1])

# Explain recommendations
def explain_recommendation(movie_id, rec_movie_id):
    movie = movies[movies['movieId'] == movie_id]
    rec_movie = movies[movies['movieId'] == rec_movie_id]
    shared_genres = set(movie['genres'].iloc[0]) & set(rec_movie['genres'].iloc[0]) if not movie.empty and not rec_movie.empty else set()
    shared_cast = set(movie['cast'].iloc[0][:3]) & set(rec_movie['cast'].iloc[0][:3]) if not movie.empty and not rec_movie.empty else set()
    return f"Shared genres: {shared_genres}, Shared cast: {shared_cast}"

# Content-based recommender
def content_based_recommender(user_id=None, movie_id=None, genres=None, k=5):
    if user_id is not None:
        user_vector = build_user_profile(user_id)
        scores = cosine_similarity(user_vector.reshape(1, -1), tfidf_matrix)[0]
        movie_indices = np.argsort(scores)[::-1][:k]
        recs = movies[['movieId', 'title']].iloc[movie_indices]
        explanations = [explain_recommendation(movies.iloc[i]['movieId'], rec['movieId']) for i, rec in recs.iterrows()]
        return recs.assign(Explanation=explanations)
    elif movie_id is not None:
        idx = movies[movies['movieId'] == movie_id].index[0]
        scores = cosine_sim[idx]
        movie_indices = np.argsort(scores)[::-1][1:k+1]
        recs = movies[['movieId', 'title']].iloc[movie_indices]
        explanations = [explain_recommendation(movie_id, rec['movieId']) for _, rec in recs.iterrows()]
        return recs.assign(Explanation=explanations)
    else:  # Cold-start
        if genres:
            genre_vector = tfidf.transform([' '.join(genres)])
            scores = cosine_similarity(genre_vector, tfidf_matrix)[0]
            movie_indices = np.argsort(scores)[::-1][:k]
            return movies[['movieId', 'title']].iloc[movie_indices]
        return movies[['movieId', 'title']].head(k)  # Fallback to popularity

# Example: Recommend for userId=1
recommendations = content_based_recommender(user_id=1)
print("Content-Based Recommendations for userId=1:")
print(recommendations)

# Save recommendations
recommendations.to_csv('models/content_based_user1.csv', index=False)