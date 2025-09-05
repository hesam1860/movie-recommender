# src/collaborative_filtering.py
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
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
def collaborative_recommender(user_id, k=5, n_neighbors=10, similarity='cosine'):
    # Check if user_id exists and map to index within bounds
    user_mask = ratings['userId'] == user_id
    if not user_mask.any():
        return pd.DataFrame({'movieId': movies['movieId'].sample(k), 'title': movies['title'].sample(k)})
    user_idx = ratings[user_mask].index[0]
    if user_idx >= user_item_matrix.shape[0]:
        return pd.DataFrame({'movieId': movies['movieId'].sample(k), 'title': movies['title'].sample(k)})
    
    # Compute similarity scores
    if similarity == 'cosine':
        sim_scores = user_similarity[user_idx]
    else:  # Pearson
        sim_scores = np.array([pearsonr(user_item_matrix[user_idx].flatten(), 
                                       user_item_matrix[i].flatten())[0] 
                              for i in range(user_item_matrix.shape[0])])
        sim_scores = np.nan_to_num(sim_scores, 0)  # Handle NaN
    
    # Get similar users
    similar_users = np.argsort(sim_scores)[-n_neighbors-1:-1][::-1]
    weights = sim_scores[similar_users]
    similar_ratings = user_item_matrix[similar_users]
    
    # Compute weighted ratings
    weighted_ratings = np.average(similar_ratings, axis=0, weights=weights)
    user_ratings = user_item_matrix[user_idx].flatten()
    unrated_indices = np.where(user_ratings == 0)[0]
    
    # Generate recommendations
    recommendations = sorted(
        [(idx, weighted_ratings[idx]) for idx in unrated_indices],
        key=lambda x: x[1],
        reverse=True
    )[:k]
    movie_indices = [movies[movies['movieId'] == ratings.iloc[idx]['movieId']].index[0] 
                     for idx, _ in recommendations if idx < movies.shape[0]]
    
    return movies[['movieId', 'title']].iloc[movie_indices] if movie_indices else pd.DataFrame({'movieId': movies['movieId'].sample(k), 'title': movies['title'].sample(k)})

# Example: Recommend for userId=1
recommendations = collaborative_recommender(user_id=1, similarity='cosine')
print("User-User Cosine Recommendations for userId=1:")
print(recommendations)
recommendations.to_csv('models/collaborative_user1.csv', index=False)

pearson_recommendations = collaborative_recommender(user_id=1, similarity='pearson')
print("User-User Pearson Recommendations for userId=1:")
print(pearson_recommendations)
pearson_recommendations.to_csv('models/collaborative_pearson_user1.csv', index=False)


# Item-item collaborative filtering recommender
def item_item_recommender(user_id, k=5, n_neighbors=10, similarity='cosine'):
    # Sample a subset of items to reduce computation
    max_items = 1000
    item_indices = np.random.choice(user_item_matrix.shape[1], size=min(max_items, user_item_matrix.shape[1]), replace=False)
    sampled_matrix = user_item_matrix[:, item_indices]
    sampled_movie_ids = ratings.pivot(index='userId', columns='movieId', values='rating').columns[item_indices]
    
    # Compute item-item similarity
    if similarity == 'cosine':
        item_similarity = cosine_similarity(sampled_matrix.T)
    else:  # Pearson
        item_similarity = np.zeros((len(item_indices), len(item_indices)))
        for i in range(len(item_indices)):
            for j in range(len(item_indices)):
                item_similarity[i, j] = pearsonr(sampled_matrix[:, i], sampled_matrix[:, j])[0]
        item_similarity = np.nan_to_num(item_similarity, 0)
    
    # Get user ratings
    user_idx = ratings[ratings['userId'] == user_id].index[0]
    user_ratings = user_item_matrix[user_idx]
    rated_indices = np.where(user_ratings > 0)[0]
    rated_in_sample = [i for i, idx in enumerate(item_indices) if idx in rated_indices]
    
    # Predict scores for unrated movies
    scores = np.zeros(len(item_indices))
    for i in range(len(item_indices)):
        if item_indices[i] not in rated_indices:
            sim_scores = item_similarity[i]
            valid_similar = [j for j in np.argsort(sim_scores)[-n_neighbors-1:-1][::-1] if j in rated_in_sample]
            if valid_similar:
                weights = sim_scores[valid_similar]
                ratings_valid = user_ratings[item_indices[valid_similar]]
                scores[i] = np.average(ratings_valid, weights=weights) if weights.sum() > 0 else 0
    
    # Sort unrated movies by predicted score
    unrated_indices = [i for i in range(len(item_indices)) if item_indices[i] not in rated_indices]
    recommendations = sorted(
        [(i, scores[i]) for i in unrated_indices],
        key=lambda x: x[1],
        reverse=True
    )[:k]
    
    # Map to movieId
    movie_indices = []
    for i, _ in recommendations:
        movie_id = sampled_movie_ids[i]
        movie_idx = movies[movies['movieId'] == movie_id].index
        if not movie_idx.empty:
            movie_indices.append(movie_idx[0])
    if not movie_indices:
        print(f"No valid movie indices found for userId={user_id}. Returning top k movies.")
        return movies[['movieId', 'title']].head(k)
    return movies[['movieId', 'title']].iloc[movie_indices]

# Example: Recommend for userId=1
item_recommendations = item_item_recommender(user_id=1, similarity='cosine')
print("Item-Item Cosine Recommendations for userId=1:")
print(item_recommendations)
item_recommendations.to_csv('models/item_item_collaborative_user1.csv', index=False)

item_pearson_recommendations = item_item_recommender(user_id=1, similarity='pearson')
print("Item-Item Pearson Recommendations for userId=1:")
print(item_pearson_recommendations)
item_pearson_recommendations.to_csv('models/item_item_pearson_user1.csv', index=False)