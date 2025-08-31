# Movie Recommender System

## Setup
1. Clone the repository: `git clone https://github.com/hesam1860/movie-recommender`
2. Create and activate virtual environment:
   ```bash
   python -m venv venv
   source venv/Scripts/activate  # On Windows with Git Bash
   ```
3. Install dependencies: `pip install -r requirements.txt`
4. Place dataset files in `data/` directory.

## Training
Run `python src/train.py` to train models.

## Evaluation
Run `python src/evaluate.py` to compute metrics.

## App
Run `python app/app.py` to launch the Gradio app locally.
Deploy to Hugging Face Spaces: `<your-space-url>`

## Dataset
The Movies Dataset from TMDb and MovieLens[](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset).
