# src/collaborative_filtering.py
import pandas as pd
import numpy as np
from scipy.sparse import load_npz
from sklearn.metrics.pairwise import cosine_similarity
from config import SEED

# Load data
ratings = pd.read_csv('data/ratings_small.csv')
user_item_matrix = load_npz('data/user_item_matrix.npz').toarray()
movies = pd.read_csv('data/movies_processed.csv')

# Compute user-user similarity
user_similarity = cosine_similarity(user_item_matrix)

# Collaborative filtering recommender
def collaborative_recommender(user_id, k=5, n_neighbors=10):
    # Get similarity scores for the user
    user_idx = ratings[ratings['userId'] == user_id].index[0]
    sim_scores = user_similarity[user_idx]
    # Find top N similar users
    similar_users = np.argsort(sim_scores)[-n_neighbors-1:-1][::-1]
    # Get ratings of similar users
    similar_ratings = user_item_matrix[similar_users]
    # Compute weighted average ratings
    weights = sim_scores[similar_users]
    weighted_ratings = np.average(similar_ratings, axis=0, weights=weights)
    # Find unrated movies for the user
    user_ratings = user_item_matrix[user_idx]
    unrated_indices = np.where(user_ratings == 0)[0]
    # Sort unrated movies by predicted rating
    recommendations = sorted(
        [(idx, weighted_ratings[idx]) for idx in unrated_indices],
        key=lambda x: x[1],
        reverse=True
    )[:k]
    movie_indices = [movies[movies['movieId'] == ratings.iloc[idx]['movieId']].index[0] for idx, _ in recommendations]
    return movies[['movieId', 'title']].iloc[movie_indices]

# Example: Recommend for userId=1
recommendations = collaborative_recommender(user_id=1)
print("Recommendations for userId=1:")
print(recommendations)

# Save recommendations
recommendations.to_csv('models/collaborative_user1.csv', index=False)