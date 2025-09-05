import pandas as pd
import json
import numpy as np
from scipy.stats.mstats import winsorize
from scipy import sparse
import matplotlib.pyplot as plt
import seaborn as sns
from config import SEED

# Load dataset files
movies = pd.read_csv('data/movies_metadata.csv', low_memory=False)
ratings = pd.read_csv('data/ratings_small.csv')
credits = pd.read_csv('data/credits.csv')
keywords = pd.read_csv('data/keywords.csv')
links = pd.read_csv('data/links.csv')

# Parse JSON-like columns
def parse_json(column, key='name'):
    try:
        if isinstance(column, str):
            return [item[key] for item in json.loads(column.replace("'", '"'))]
        return []
    except:
        return []

movies['genres'] = movies['genres'].apply(lambda x: parse_json(x))
credits['cast'] = credits['cast'].apply(lambda x: parse_json(x))
credits['crew'] = credits['crew'].apply(lambda x: parse_json(x))
keywords['keywords'] = keywords['keywords'].apply(lambda x: parse_json(x))

# Convert id columns to int
movies = movies[movies['id'].str.isnumeric()]
non_numeric_rows = movies[~movies['id'].str.isnumeric()]
print(f"Rows dropped due to non-numeric ID: {len(non_numeric_rows)}")
movies['id'] = movies['id'].astype(int)
keywords['id'] = keywords['id'].astype(int)
credits['id'] = credits['id'].astype(int)

# Debug: Print sample parsed data
print("Sample genres:", movies['genres'].head().tolist())
print("Sample keywords:", keywords['keywords'].head().tolist())
print("Sample cast:", credits['cast'].head().tolist())
print("Sample crew:", credits['crew'].head().tolist())

# Log initial row counts
print(f"Initial movies rows: {len(movies)}")
print(f"Initial ratings rows: {len(ratings)}")
print(f"Initial credits rows: {len(credits)}")
print(f"Initial keywords rows: {len(keywords)}")
print(f"Initial links rows: {len(links)}")

# Merge keywords and credits
movies = movies.merge(keywords[['id', 'keywords']], left_on='id', right_on='id', how='left')
print(f"After merging keywords: {len(movies)}")
movies = movies.merge(credits[['id', 'cast', 'crew']], left_on='id', right_on='id', how='left')
print(f"After merging credits: {len(movies)}")

# Align IDs with links
links['tmdbId'] = links['tmdbId'].fillna(0).astype(int)
movies = movies.merge(links[['movieId', 'tmdbId']], left_on='id', right_on='tmdbId', how='inner')
print(f"After ID alignment: {len(movies)}")

# Handle missing/outliers
movies['budget'] = pd.to_numeric(movies['budget'], errors='coerce')
movies['budget'] = winsorize(movies['budget'].fillna(0), limits=[0.05, 0.05])
movies['popularity'] = pd.to_numeric(movies['popularity'], errors='coerce')
movies['popularity'] = np.log1p(movies['popularity'].fillna(movies['popularity'].median()))
movies['vote_count'] = np.log1p(movies['vote_count'].fillna(movies['vote_count'].median()))

# EDA: Rating histogram
plt.figure(figsize=(8, 6))
sns.histplot(ratings['rating'], bins=10)
plt.title('Rating Distribution')
plt.savefig('report/rating_histogram.png')
plt.close()

# EDA: Sparsity heatmap
user_item_matrix = ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)
plt.figure(figsize=(10, 8))
sns.heatmap(user_item_matrix.iloc[:100, :100], cmap='Blues')
plt.title('User-Item Interaction Sparsity (Sample)')
plt.savefig('report/sparsity_heatmap.png')
plt.close()

# EDA: Long-tail of movies (by vote count)
movie_counts = movies['vote_count'].value_counts().sort_index()
plt.figure(figsize=(10, 6))
plt.plot(movie_counts.index, movie_counts.values, 'b-')
plt.title('Long-tail Distribution of Movie Vote Counts')
plt.xlabel('Vote Count')
plt.ylabel('Number of Movies')
plt.yscale('log')
plt.savefig('report/movie_long_tail.png')
plt.close()

# EDA: Temporal dynamics (ratings over time)
ratings['timestamp'] = pd.to_datetime(ratings['timestamp'], unit='s')
ratings_by_time = ratings.groupby(ratings['timestamp'].dt.year)['rating'].count()
plt.figure(figsize=(10, 6))
plt.plot(ratings_by_time.index, ratings_by_time.values, 'r-')
plt.title('Temporal Dynamics of Ratings')
plt.xlabel('Year')
plt.ylabel('Number of Ratings')
plt.savefig('report/ratings_over_time.png')
plt.close()

# EDA: Coverage across languages
plt.figure(figsize=(10, 6))
sns.countplot(data=movies, x='original_language', order=movies['original_language'].value_counts().index[:10])
plt.title('Top 10 Languages Coverage')
plt.xticks(rotation=45)
plt.savefig('report/language_coverage.png')
plt.close()

# Save preprocessed data
movies.to_csv('data/movies_processed.csv', index=False)
sparse.save_npz('data/user_item_matrix.npz', sparse.csr_matrix(user_item_matrix))