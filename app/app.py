import gradio as gr
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import ast

# Load data
try:
    movies = pd.read_csv('data/movies_processed.csv')
    ratings = pd.read_csv('data/ratings_small.csv')
except FileNotFoundError as e:
    print(f"Error: {e}. Ensure 'data/movies_processed.csv' and 'data/ratings_small.csv' exist.")
    exit(1)

# Ensure consistent data types for movieId
movies['movieId'] = movies['movieId'].astype('int64')
ratings['movieId'] = ratings['movieId'].astype('int64')

# Log mismatched movieIds
missing_movie_ids = set(ratings['movieId']) - set(movies['movieId'])
if missing_movie_ids:
    print(f"Warning: {len(missing_movie_ids)} movieIds in ratings not in movies_processed.csv")

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
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

def explain_recommendation(user_rated_movie_ids, rec_movie_id):
    rec_movie = movies[movies['movieId'] == rec_movie_id]
    if rec_movie.empty:
        return "No metadata for recommended movie."
    rec_genres = set(rec_movie['genres'].iloc[0])
    rec_cast = set(rec_movie['cast'].iloc[0][:3])
    shared_genres, shared_cast = set(), set()
    for rated_movie_id in user_rated_movie_ids:
        rated_movie = movies[movies['movieId'] == rated_movie_id]
        if not rated_movie.empty:
            rated_genres = set(rated_movie['genres'].iloc[0])
            rated_cast = set(rated_movie['cast'].iloc[0][:3])
            shared_genres.update(rec_genres & rated_genres)
            shared_cast.update(rec_cast & rated_cast)
    return f"Shared genres: {shared_genres}, Shared cast: {shared_cast}"

def recommend_movies(user_id=None, movie_title=None, genres=None, k=5):
    if user_id:
        try:
            user_id = int(user_id)  # Ensure integer
            user_ratings = ratings[ratings['userId'] == user_id]
            if user_ratings.empty:
                return movies[['movieId', 'title']].head(k).to_string() + "\nNote: No ratings for this user, showing popular movies."
            # Filter ratings to only include movieIds present in movies
            user_ratings = user_ratings[user_ratings['movieId'].isin(movies['movieId'])]
            print(f"Debug: Filtered userId={user_id} ratings count: {len(user_ratings)}")
            if user_ratings.empty:
                return movies[['movieId', 'title']].head(k).to_string() + "\nNote: No valid ratings found for this user, showing popular movies."
            user_vector = np.zeros(tfidf_matrix.shape[1])  # Shape (5000,)
            mean_rating = user_ratings['rating'].mean()
            valid_ratings = 0
            user_rated_movie_ids = user_ratings['movieId'].tolist()
            for _, row in user_ratings.iterrows():
                movie_id = int(row['movieId'])  # Ensure integer
                print(f"Debug: Processing movieId={movie_id} for userId={user_id}")
                movie_indices = movies.index[movies['movieId'] == movie_id].tolist()
                if movie_indices:
                    movie_idx = movie_indices[0]
                    vector = tfidf_matrix[movie_idx].toarray().flatten()  # Flatten to (5000,)
                    user_vector += vector * (row['rating'] - mean_rating)
                    valid_ratings += 1
                else:
                    print(f"Warning: movieId {movie_id} not found in movies_processed.csv")
            if valid_ratings == 0:
                return movies[['movieId', 'title']].head(k).to_string() + "\nNote: No valid ratings found for this user, showing popular movies."
            user_vector = user_vector / valid_ratings
            scores = cosine_similarity(user_vector.reshape(1, -1), tfidf_matrix)[0]
            movie_indices = np.argsort(scores)[::-1][:k]
            recs = movies[['movieId', 'title']].iloc[movie_indices]
            explanations = [explain_recommendation(user_rated_movie_ids, rec['movieId']) for _, rec in recs.iterrows()]
            return recs.assign(Explanation=explanations).to_string()
        except Exception as e:
            return f"Error in user-based recommendation: {str(e)}"
    elif movie_title:
        try:
            movie_id = movies[movies['title'] == movie_title]['movieId'].iloc[0] if movie_title in movies['title'].values else 318
            idx = movies[movies['movieId'] == movie_id].index[0]
            scores = cosine_sim[idx]
            movie_indices = np.argsort(scores)[::-1][1:k+1]
            recs = movies[['movieId', 'title']].iloc[movie_indices]
            explanations = [explain_recommendation([movie_id], rec['movieId']) for _, rec in recs.iterrows()]
            return recs.assign(Explanation=explanations).to_string()
        except Exception as e:
            return f"Error in movie-based recommendation: {str(e)}"
    elif genres:
        try:
            genre_vector = tfidf.transform([' '.join(genres)])
            scores = cosine_similarity(genre_vector, tfidf_matrix)[0]
            movie_indices = np.argsort(scores)[::-1][:k]
            return movies[['movieId', 'title']].iloc[movie_indices].to_string()
        except Exception as e:
            return f"Error in genre-based recommendation: {str(e)}"
    return movies[['movieId', 'title']].head(k).to_string() + "\nNote: No input provided, showing popular movies."

# Gradio interface
iface = gr.Interface(
    fn=recommend_movies,
    inputs=[
        gr.Number(label="User ID (optional)", value=None),
        gr.Dropdown(choices=movies['title'].tolist(), label="Movie Title (optional)"),
        gr.CheckboxGroup(choices=['Action', 'Comedy', 'Drama', 'Romance'], label="Genres (optional)")
    ],
    outputs="text",
    title="Movie Recommender System"
)
iface.launch()