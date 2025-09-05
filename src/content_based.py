# src/content_based.py
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import ast
from sentence_transformers import SentenceTransformer


# Load data
movies = pd.read_csv('data/movies_processed.csv')
movies = movies.sample(n=5000, random_state=42)  # Subset to 5000 movies
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
pd.to_pickle(tfidf_matrix, 'data/tfidf_matrix.pkl')
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
    rec_movie = movies[movies['movieId'] == rec_movie_id]
    if movie_id is None:
        return f"Recommended based on user profile: genres {rec_movie['genres'].iloc[0] if not rec_movie.empty else []}"
    movie = movies[movies['movieId'] == movie_id]
    shared_genres = set(movie['genres'].iloc[0]) & set(rec_movie['genres'].iloc[0]) if not movie.empty and not rec_movie.empty else set()
    shared_cast = set(movie['cast'].iloc[0][:3]) & set(rec_movie['cast'].iloc[0][:3]) if not movie.empty and not rec_movie.empty else set()
    return f"Shared genres: {shared_genres}, Shared cast: {shared_cast}"

# Content-based recommender
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import issparse


# Preprocess features column
movies['features'] = movies['title'].fillna('') + ' ' + movies['genres'].fillna('').apply(lambda x: ' '.join(eval(x)) if isinstance(x, str) else '')

def content_based_recommender(user_id, k=20, use_embeddings=False):
    # Load precomputed TF-IDF matrix
    tfidf_matrix = pd.read_pickle('data/tfidf_matrix.pkl')
    if use_embeddings:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        # Sample movies to match tfidf_matrix size (5000)
        sampled_movies = movies.iloc[:tfidf_matrix.shape[0]].reset_index(drop=True)
        movie_features = model.encode(sampled_movies['features'].tolist())
        user_ratings = ratings[ratings['userId'] == user_id]
        if len(user_ratings) == 0:
            return pd.DataFrame({'movieId': sampled_movies['movieId'].sample(k), 'score': [1/k]*k})
        # Map user ratings to sampled movies and get valid indices within sampled subset
        valid_movie_ids = sampled_movies['movieId'].values
        user_indices = [i for i, mid in enumerate(sampled_movies['movieId']) 
                        if mid in user_ratings['movieId'].values]
        if not user_indices:
            return pd.DataFrame({'movieId': sampled_movies['movieId'].sample(k), 'score': [1/k]*k})
        user_profile = np.mean([movie_features[i] for i in user_indices], axis=0)
        similarities = cosine_similarity([user_profile], movie_features)[0]
        recommendations = pd.DataFrame({
            'movieId': sampled_movies['movieId'],
            'score': similarities
        }).sort_values(by='score', ascending=False).head(k)
    else:
        user_ratings = ratings[ratings['userId'] == user_id]
        if len(user_ratings) == 0:
            return pd.DataFrame({'movieId': movies['movieId'].sample(k), 'score': [1/k]*k})
        user_indices = [movies[movies['movieId'] == mid].index[0]
                        for mid in user_ratings['movieId'] if mid in movies['movieId'].values]
        valid_indices = [i for i in user_indices if i < tfidf_matrix.shape[0]]
        if not valid_indices:
            return pd.DataFrame({'movieId': movies['movieId'].sample(k), 'score': [1/k]*k})
        user_profile = tfidf_matrix[valid_indices].mean(axis=0)
        user_profile = np.asarray(user_profile).reshape(1, -1)
        similarities = cosine_similarity(user_profile, tfidf_matrix)[0]
        recommendations = pd.DataFrame({
            'movieId': movies['movieId'],
            'score': similarities
        }).sort_values(by='score', ascending=False).head(k)
    return recommendations

# Example: Recommend for userId=1
print(movies['movieId'].duplicated().sum())
movies.drop_duplicates(subset='movieId')

recommendations = content_based_recommender(user_id=1)
print("Content-Based Recommendations for userId=1:")
print(recommendations)

# Save recommendations
recommendations.to_csv('models/content_based_user1.csv', index=False)