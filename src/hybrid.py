import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from surprise import SVD, Dataset, Reader
from config import SEED
import ast

# Load data
try:
    movies = pd.read_csv('data/movies_processed.csv')
    ratings = pd.read_csv('data/ratings_small.csv')
except FileNotFoundError as e:
    print(f"Error: {e}. Ensure 'data/movies_processed.csv' and 'data/ratings_small.csv' exist.")
    exit(1)

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

# Build user profile for content-based
def build_user_profile(user_id):
    user_ratings = ratings[ratings['userId'] == user_id]
    if user_ratings.empty:
        return np.zeros(tfidf_matrix.shape[1])
    weighted_vectors = []
    for _, row in user_ratings.iterrows():
        movie_idx = movies[movies['movieId'] == row['movieId']].index
        if not movie_idx.empty:
            rating = row['rating'] - user_ratings['rating'].mean()
            weighted_vectors.append(tfidf_matrix[movie_idx[0]].toarray() * rating)
    return np.sum(weighted_vectors, axis=0) / len(weighted_vectors) if weighted_vectors else np.zeros(tfidf_matrix.shape[1])

# Collaborative Filtering (SVD)
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
trainset = data.build_full_trainset()
algo = SVD(n_factors=100, reg_all=0.1, random_state=SEED)
algo.fit(trainset)

# Hybrid Recommender
def hybrid_recommender(user_id, k=5, alpha=0.5):
    user_ratings = ratings[ratings['userId'] == user_id]
    if user_ratings.empty:
        print(f"No ratings for userId={user_id}. Returning top k movies.")
        return movies[['movieId', 'title']].head(k)
    # Content-based scores
    user_vector = build_user_profile(user_id)
    cb_scores = cosine_similarity(user_vector.reshape(1, -1), tfidf_matrix)[0]
    # Collaborative filtering scores
    cf_scores = [algo.predict(user_id, m_id).est for m_id in movies['movieId']]
    # Hybrid scores
    hybrid_scores = alpha * np.array(cf_scores) + (1 - alpha) * cb_scores
    # Get top k
    movie_indices = np.argsort(hybrid_scores)[::-1][:k]
    return movies[['movieId', 'title']].iloc[movie_indices]

# Example: Recommend for userId=1 with alpha=0.5
recommendations = hybrid_recommender(user_id=1, alpha=0.5)
print("Hybrid Recommendations for userId=1 (alpha=0.5):")
print(recommendations)

# Save recommendations
recommendations.to_csv('models/hybrid_user1.csv', index=False)