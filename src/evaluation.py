import pandas as pd
import numpy as np
from sklearn.metrics import ndcg_score
from scipy.stats import bootstrap
from config import SEED

# Load data
try:
    ratings = pd.read_csv('data/ratings_small.csv')
    movies = pd.read_csv('data/movies_processed.csv')
    global_pop = pd.read_csv('models/global_popularity.csv')
    drama_pop = pd.read_csv('models/drama_popularity.csv')
    content_rec = pd.read_csv('models/content_based_user1.csv')
    collab_rec = pd.read_csv('models/collaborative_user1.csv')
    mf_rec = pd.read_csv('models/mf_user1.csv')
    hybrid_rec = pd.read_csv('models/hybrid_user1.csv')
except FileNotFoundError as e:
    print(f"Error: {e}. Ensure all required files exist in 'data/' and 'models/'.")
    exit(1)

# Create test set (last 20% of ratings, sorted by timestamp)
ratings = ratings.sort_values(by=['userId', 'timestamp'])
test_size = 0.2
test_ratings = ratings.groupby('userId').apply(
    lambda x: x.iloc[int(len(x)*(1-test_size)):], include_groups=False
).reset_index()
test_ratings = test_ratings[test_ratings['rating'] >= 4.0]

# Debug: Check test set for userId=1
user1_test = test_ratings[test_ratings['userId'] == 1]
print(f"Test set size for userId=1: {len(user1_test)}")
print(f"Test set movieIds for userId=1: {user1_test['movieId'].tolist()}")

# Metrics
def precision_at_k(recommended, relevant, k):
    recommended_k = recommended['movieId'].head(k).tolist()
    relevant_k = relevant['movieId'].tolist()
    hits = len(set(recommended_k) & set(relevant_k))
    return hits / k if k > 0 else 0

def recall_at_k(recommended, relevant, k):
    recommended_k = recommended['movieId'].head(k).tolist()
    relevant_k = relevant['movieId'].tolist()
    hits = len(set(recommended_k) & set(relevant_k))
    return hits / len(relevant_k) if len(relevant_k) > 0 else 0

def ndcg_at_k(recommended, relevant, k):
    recommended_k = recommended['movieId'].head(k).tolist()
    relevant_k = relevant['movieId'].tolist()
    if len(relevant_k) < 2:  # NDCG requires at least 2 items
        return 0
    relevance = [1 if m in relevant_k else 0 for m in recommended_k]
    ideal_relevance = [1] * min(len(relevant_k), k)
    try:
        return ndcg_score([ideal_relevance], [relevance], k=k)
    except ValueError:
        return 0

def catalog_coverage(recommended, all_movies):
    recommended_movies = set(recommended['movieId'].head(20).tolist())
    return len(recommended_movies) / len(all_movies)

# Evaluate across models
users = test_ratings['userId'].unique()[:10]  # Limit to 10 users
results = {
    'Model': [],
    'Precision@10': [], 'Recall@10': [], 'NDCG@10': [],
    'Precision@20': [], 'Recall@20': [], 'NDCG@20': [], 'Coverage@20': []
}
for model_name, rec in [
    ('Global', global_pop), ('Drama', drama_pop),
    ('Content', content_rec), ('Collaborative', collab_rec),
    ('Matrix Factorization', mf_rec), ('Hybrid', hybrid_rec)
]:
    prec_10, rec_10, ndcg_10, prec_20, rec_20, ndcg_20, cov_20 = [], [], [], [], [], [], []
    for user_id in users:
        user_test = test_ratings[test_ratings['userId'] == user_id]
        if len(user_test) > 0:
            prec_10.append(precision_at_k(rec, user_test, 10))
            rec_10.append(recall_at_k(rec, user_test, 10))
            ndcg_10.append(ndcg_at_k(rec, user_test, 10))
            prec_20.append(precision_at_k(rec, user_test, 20))
            rec_20.append(recall_at_k(rec, user_test, 20))
            ndcg_20.append(ndcg_at_k(rec, user_test, 20))
            cov_20.append(catalog_coverage(rec, movies['movieId']))
    results['Model'].append(model_name)
    results['Precision@10'].append(np.mean(prec_10) if prec_10 else 0)
    results['Recall@10'].append(np.mean(rec_10) if rec_10 else 0)
    results['NDCG@10'].append(np.mean(ndcg_10) if ndcg_10 else 0)
    results['Precision@20'].append(np.mean(prec_20) if prec_20 else 0)
    results['Recall@20'].append(np.mean(rec_20) if rec_20 else 0)
    results['NDCG@20'].append(np.mean(ndcg_20) if ndcg_20 else 0)
    results['Coverage@20'].append(np.mean(cov_20) if cov_20 else 0)

# Confidence intervals
def bootstrap_ci(data, n_bootstraps=1000):
    rng = np.random.default_rng(SEED)
    bootstrapped_means = [np.mean(rng.choice(data, len(data), replace=True)) for _ in range(n_bootstraps)]
    return np.percentile(bootstrapped_means, [2.5, 97.5])

for metric in ['Precision@10', 'Recall@10', 'NDCG@10', 'Precision@20', 'Recall@20', 'NDCG@20']:
    results[f'{metric}_CI'] = [bootstrap_ci([results[metric][i]], 1000) if results[metric][i] > 0 else [0, 0] for i in range(len(results['Model']))]

# Save results
results_df = pd.DataFrame(results)
print(results_df)
results_df.to_csv('report/evaluation_results.csv', index=False)