import pandas as pd
import numpy as np
from surprise import SVD, Dataset, Reader
from config import SEED

# Load data
try:
    ratings = pd.read_csv('data/ratings_small.csv')
    movies = pd.read_csv('data/movies_processed.csv')
except FileNotFoundError as e:
    print(f"Error: {e}. Ensure 'data/ratings_small.csv' and 'data/movies_processed.csv' exist.")
    exit(1)

# Prepare data for Surprise
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
trainset = data.build_full_trainset()

# Train SVD
algo = SVD(n_factors=100, reg_all=0.1, random_state=SEED)
algo.fit(trainset)

# Recommend
def mf_recommender(user_id, k=5):
    all_movies = movies['movieId'].unique()
    predictions = [algo.predict(user_id, movie_id).est for movie_id in all_movies]
    movie_scores = sorted(zip(all_movies, predictions), key=lambda x: x[1], reverse=True)[:k]
    movie_indices = []
    for movie_id, _ in movie_scores:
        idx = movies[movies['movieId'] == movie_id].index
        if not idx.empty:
            movie_indices.append(idx[0])
    if not movie_indices:
        print(f"No valid movie indices found for userId={user_id}. Returning top k movies.")
        return movies[['movieId', 'title']].head(k)
    return movies[['movieId', 'title']].iloc[movie_indices]

# Example: userId=1
recommendations = mf_recommender(user_id=1)
print("Matrix Factorization Recommendations for userId=1:")
print(recommendations)

# Save
recommendations.to_csv('models/mf_user1.csv', index=False)