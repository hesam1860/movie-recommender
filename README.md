# Movie Recommender System

## Overview
This is a capstone project for a Machine Learning course, building a movie recommender system using The Movies Dataset (TMDb and MovieLens). The system implements:
- Global and per-genre popularity baselines (via `baseline.py`).
- Content-based recommender with TF-IDF features and user profiles (via `content_based.py`).
- Collaborative filtering with k-NN and matrix factorization (via `collaborative_filtering.py` and `matrix_factorization.py`).
- A hybrid model blending content-based and collaborative filtering (via `hybrid.py`).
- Evaluation with Precision@K, Recall@K, NDCG@K, and confidence intervals (via `evaluation.py`).
- A Gradio app for interactive recommendations (deployed on Hugging Face Spaces, with local runtime via `app/`).

The project is structured with directories `src/`, `data/`, `models/`, `app/`, and `report/` for reproducibility. Data is sourced from [The Movie Database (TMDb)](https://www.themoviedb.org/) and [GroupLens MovieLens](https://grouplens.org/datasets/movielens/).

## Setup
1. **Clone the Repository**:
   
   git clone https://github.com/hesam1860/movie-recommender
   cd movie-recommender

## Install Dependencies:

Ensure you have Python 3.11 installed.
Install required packages using the provided requirements.txt:
pip install -r requirements.txt



## Prepare the Dataset:

Place the Movies Dataset files (e.g., credits.csv, keywords.csv, links.csv, links_small.csv, movies_metadata.csv, ratings.csv, ratings_small.csv) in the data/ directory. Download them from TMDb and MovieLens if not already included.
Run the data preparation script to process the data and generate movies_parsed.csv, movies_processed.csv, tfidf_matrix.pkl, and user_item_matrix.npz:
python src/data_preparation.py




## Training

Train the models using the training script:
- python src/matrix_factorization.py  # For matrix factorization
- python src/collaborative_filtering.py  # For k-NN collaborative filtering
- python src/content_based.py  # For content-based recommender
- python src/hybrid.py  # For hybrid model

This will generate and save model artifacts (e.g., factor matrices) in the models/ directory. Adjust commands if a unified train.py is added.



## Evaluation

Evaluate the models with various metrics (Precision@10/20, Recall@10/20, NDCG@10/20, coverage):
- python src/evaluation.py

Results, including confidence intervals, will be saved in the report/ directory or printed to the console.



## Running the App Locally

Launch the Gradio app to interact with the recommender system:
- gradio app/app.py

The app allows title or user ID input and displays top-k recommendations with explanations and posters. Open your browser at the provided local URL (e.g., http://127.0.0.1:7860).
Note: Ensure pre-computed artifacts (tfidf_matrix.pkl, user_item_matrix.npz) in data/ and models in models/ are available.



## Notes

Random seeds are set in src/config.py for reproducibility (SEED = 42).
Pre-computed artifacts (tfidf_matrix.pkl, user_item_matrix.npz) are stored in data/ and model outputs in models/ to manage memory.
The Hugging Face Spaces deployment is accessible here (https://huggingface.co/spaces/Vector1860/movie-recommender-hesam).

Attribution
This project uses data from:

The Movie Database (TMDb)
GroupLens MovieLens