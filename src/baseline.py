# src/baseline.py
import pandas as pd
import numpy as np
import ast
from config import SEED

# Load preprocessed movies
movies = pd.read_csv('data/movies_processed.csv')
movies['genres'] = movies['genres'].apply(ast.literal_eval)  # Convert string to list

# Debug: Check genres column
print("Sample genres:", movies['genres'].head().tolist())

# IMDb weighted-rating formula: WR = (v / (v + m)) * R + (m / (v + m)) * C
def compute_weighted_rating(df, m_percentile=0.8):
    C = df['vote_average'].mean()  # Global mean rating
    m = df['vote_count'].quantile(m_percentile)  # Minimum votes threshold
    df['weighted_rating'] = (df['vote_count'] / (df['vote_count'] + m)) * df['vote_average'] + \
                            (m / (df['vote_count'] + m)) * C
    return df

# Global popularity recommender
movies = compute_weighted_rating(movies)
global_top_k = movies[['movieId', 'title', 'weighted_rating']].sort_values(by='weighted_rating', ascending=False).head(10)
print("Global Top 10 Movies:")
print(global_top_k)

# Per-genre popularity recommender
def per_genre_recommender(df, genre, k=5):
    df_exploded = df.explode('genres')
    genre_df = df_exploded[df_exploded['genres'] == genre]
    genre_top_k = genre_df[['movieId', 'title', 'weighted_rating']].sort_values(by='weighted_rating', ascending=False).head(k)
    return genre_top_k

# Example genres
genres = ['Drama', 'Action', 'Comedy']
for genre in genres:
    print(f"\nTop 5 {genre} Movies:")
    print(per_genre_recommender(movies, genre))

# Save recommendations
global_top_k.to_csv('models/global_popularity.csv', index=False)
for genre in genres:
    per_genre_recommender(movies, genre).to_csv(f'models/{genre.lower()}_popularity.csv', index=False)