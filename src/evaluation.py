# src/evaluation.py
import pandas as pd
import numpy as np
from config import SEED

# Load data
ratings = pd.read_csv('data/ratings_small.csv')
global_pop = pd.read_csv('models/global_popularity.csv')
drama_pop = pd.read_csv('models/drama_popularity.csv')
content_rec = pd.read_csv('models/content_based_318.csv')
collab_rec = pd.read_csv('models/collaborative_user1.csv')

# Create test set (last 20% of ratings for each user)
ratings = ratings.sort_values(by=['userId', 'timestamp'])
test_size = 0.2
test_ratings = ratings.groupby('userId').apply(
    lambda x: x.iloc[int(len(x)*(1-test_size)):], include_groups=False
).reset_index()
test_ratings = test_ratings[test_ratings['rating'] >= 4.0]  # Consider ratings >= 4 as relevant

# Debug: Check test set for userId=1
user1_test = test_ratings[test_ratings['userId'] == 1]
print(f"Test set size for userId=1: {len(user1_test)}")
print(f"Test set movieIds for userId=1: {user1_test['movieId'].tolist()}")

# Precision@k metric
def precision_at_k(recommended, relevant, k=5):
    recommended_k = recommended['movieId'].head(k).tolist()
    relevant_k = relevant['movieId'].tolist()
    hits = len(set(recommended_k) & set(relevant_k))
    return hits / k if k > 0 else 0

# Evaluate for multiple users
users = test_ratings['userId'].unique()[:10]  # Test first 10 users
results = {'Global': [], 'Drama': [], 'Content': [], 'Collaborative': []}
for user_id in users:
    user_test = test_ratings[test_ratings['userId'] == user_id]
    if len(user_test) > 0:  # Only evaluate users with test ratings
        results['Global'].append(precision_at_k(global_pop, user_test))
        results['Drama'].append(precision_at_k(drama_pop, user_test))
        results['Content'].append(precision_at_k(content_rec, user_test))
        results['Collaborative'].append(precision_at_k(collab_rec, user_test))

# Compute average Precision@5
global_precision = np.mean(results['Global']) if results['Global'] else 0
drama_precision = np.mean(results['Drama']) if results['Drama'] else 0
content_precision = np.mean(results['Content']) if results['Content'] else 0
collab_precision = np.mean(results['Collaborative']) if results['Collaborative'] else 0

print(f"Average Global Popularity Precision@5: {global_precision:.3f}")
print(f"Average Drama Popularity Precision@5: {drama_precision:.3f}")
print(f"Average Content-Based Precision@5: {content_precision:.3f}")
print(f"Average Collaborative Filtering Precision@5: {collab_precision:.3f}")

# Save evaluation results
results_df = pd.DataFrame({
    'Model': ['Global Popularity', 'Drama Popularity', 'Content-Based', 'Collaborative Filtering'],
    'Precision@5': [global_precision, drama_precision, content_precision, collab_precision]
})
results_df.to_csv('report/evaluation_results.csv', index=False)