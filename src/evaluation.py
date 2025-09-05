import pandas as pd
import numpy as np
from sklearn.metrics import ndcg_score
from sklearn.metrics.pairwise import cosine_similarity
from hybrid import hybrid_recommender
from config import SEED
from content_based import content_based_recommender
from collaborative_filtering import collaborative_recommender
from matrix_factorization import mf_recommender
from hybrid import switching_hybrid_recommender
from hybrid import retrieve_rerank_recommender
from hybrid import ltr_hybrid_recommender

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
    switching_rec = pd.read_csv('models/switching_hybrid_user1.csv')
    retrieve_rerank_rec = pd.read_csv('models/retrieve_rerank_hybrid_user1.csv')
    ltr_rec = pd.read_csv('models/ltr_hybrid_user1.csv')
    tfidf = pd.read_pickle('data/tfidf_matrix.pkl')
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
    print(f"Recommended k: {recommended_k}")
    print(f"Relevant k: {relevant_k}")
    return hits / k if k > 0 else 0

def recall_at_k(recommended, relevant, k):
    recommended_k = recommended['movieId'].head(k).tolist()
    relevant_k = relevant['movieId'].tolist()
    hits = len(set(recommended_k) & set(relevant_k))
    return hits / len(relevant_k) if len(relevant_k) > 0 else 0

def ndcg_at_k(recommended, relevant, k):
    recommended_k = recommended['movieId'].head(k).tolist()
    relevant_k = relevant['movieId'].tolist()
    if not relevant_k:
        return 0
    relevance = [1 if m in relevant_k else 0 for m in recommended_k]
    ideal_relevance = [1] * min(len(relevant_k), k)
    if len(relevant_k) == 1:
        for i, rel in enumerate(relevance):
            if rel == 1:
                dcg = 1 / np.log2(i + 2)  # Rank i+1 (1-based)
                idcg = 1  # Ideal DCG (rank 1)
                return dcg / idcg
        return 0.0
    try:
        return ndcg_score([ideal_relevance], [relevance], k=k)
    except ValueError:
        return 0

def catalog_coverage(recommended, all_movies):
    recommended_movies = set(recommended['movieId'].head(20).tolist())
    return len(recommended_movies) / len(all_movies)

def novelty(recommended, ratings, k):
    recommended_k = recommended['movieId'].head(k).tolist()
    total_ratings = len(ratings)
    movie_counts = ratings['movieId'].value_counts()
    novelty_scores = [-np.log2((movie_counts.get(m, 1) / total_ratings) + 1e-10) for m in recommended_k]
    return np.mean(novelty_scores) if novelty_scores else 0

def intra_list_diversity(recommended, k):
    recommended_k = recommended['movieId'].head(k).tolist()
    movie_indices = [
        movies[movies['movieId'] == m].index[0]
        for m in recommended_k
        if m in movies['movieId'].values and movies[movies['movieId'] == m].index[0] < tfidf.shape[0]
    ]
    if len(movie_indices) < 2:
        return 0
    sim_matrix = cosine_similarity(tfidf[movie_indices])
    return 1 - np.mean(sim_matrix[np.triu_indices(len(movie_indices), k=1)])

# Evaluate for multiple users
user_ids = ratings['userId'].unique()[:10]  # Top 10 users
results = {
    'Model': [], 'UserId': [],
    'Precision@10': [], 'Recall@10': [], 'NDCG@10': [],
    'Precision@20': [], 'Recall@20': [], 'NDCG@20': [],
    'Coverage@20': [], 'Novelty@10': [], 'Diversity@10': []
}
for user_id in user_ids:
    user_test = test_ratings[test_ratings['userId'] == user_id]
    print(f"Test set size for userId={user_id}: {len(user_test)}")
    print(f"Test set movieIds for userId={user_id}: {user_test['movieId'].tolist()}")
    for model_name, rec in [
        ('Global', global_pop), ('Drama', drama_pop),
        ('Content', content_rec if user_id == 1 else content_based_recommender(user_id, k=20)),
        ('Content-Embeddings', content_based_recommender(user_id, k=20, use_embeddings=True)),
        ('Collaborative', collab_rec if user_id == 1 else collaborative_recommender(user_id, k=20)),
        ('Matrix Factorization', mf_rec if user_id == 1 else mf_recommender(user_id, k=20)),
        ('Hybrid', hybrid_rec if user_id == 1 else hybrid_recommender(user_id, k=20)),
        ('Switching Hybrid', switching_rec if user_id == 1 else switching_hybrid_recommender(user_id, k=20)),
        ('Retrieve-Rerank Hybrid', retrieve_rerank_rec if user_id == 1 else retrieve_rerank_recommender(user_id, k=20)),
        ('Learning-to-Rank Hybrid', ltr_rec if user_id == 1 else ltr_hybrid_recommender(user_id, k=20)),
        ('Hybrid-CB', hybrid_recommender(user_id, k=5, alpha=0)),
        ('Hybrid-CF', hybrid_recommender(user_id, k=5, alpha=1)),
        ('Hybrid-Alpha0.3', hybrid_recommender(user_id, k=5, alpha=0.3)),
        ('Hybrid-Alpha0.7', hybrid_recommender(user_id, k=5, alpha=0.7))
    ]:
        if len(user_test) > 0:
            prec_10 = precision_at_k(rec, user_test, 10)
            rec_10 = recall_at_k(rec, user_test, 10)
            ndcg_10 = ndcg_at_k(rec, user_test, 10)
            prec_20 = precision_at_k(rec, user_test, 20)
            rec_20 = recall_at_k(rec, user_test, 20)
            ndcg_20 = ndcg_at_k(rec, user_test, 20)
            cov_20 = catalog_coverage(rec, movies['movieId'])
            nov_10 = novelty(rec, ratings, 10)
            div_10 = intra_list_diversity(rec, 10)
        else:
            prec_10 = rec_10 = ndcg_10 = prec_20 = rec_20 = ndcg_20 = cov_20 = nov_10 = div_10 = 0
        results['Model'].append(model_name)
        results['UserId'].append(user_id)
        results['Precision@10'].append(prec_10)
        results['Recall@10'].append(rec_10)
        results['NDCG@10'].append(ndcg_10)
        results['Precision@20'].append(prec_20)
        results['Recall@20'].append(rec_20)
        results['NDCG@20'].append(ndcg_20)
        results['Coverage@20'].append(cov_20)
        results['Novelty@10'].append(nov_10)
        results['Diversity@10'].append(div_10)

# Aggregate results
results_df = pd.DataFrame(results)
results_agg = results_df.groupby('Model').mean().reset_index()
print("Aggregated Results:")
print(results_agg)
results_agg.to_csv('report/evaluation_results_aggregated.csv', index=False)

# Confidence intervals
for metric in ['Precision@10', 'Recall@10', 'NDCG@10', 'Precision@20', 'Recall@20', 'NDCG@20']:
    results_agg[f'{metric}_CI'] = results_df.groupby('Model')[metric].apply(
        lambda x: [x.mean() - 1.96 * x.std() / np.sqrt(len(x)), x.mean() + 1.96 * x.std() / np.sqrt(len(x))]
    ).tolist()

# Visualization
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 6))
results_agg.plot(x='Model', y=['Precision@10', 'Recall@10', 'NDCG@10'], kind='bar')
plt.title('Model Comparison: Precision@10, Recall@10, NDCG@10')
plt.ylabel('Score')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('report/metric_comparison.png')
plt.close()