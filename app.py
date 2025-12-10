import os
import sys

from pathlib import Path
from make_recommendation import *

# Add the project root (parent of 'deploy') to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

print(PROJECT_ROOT)
import streamlit as st
import requests

module_path = os.path.abspath(os.path.join(os.path.dirname('app.py'), '.'))
sys.path.insert(0, module_path)


def fetch_movie_poster(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{map_id_to_tmdb[movie_id]}?api_key=9f1339eba7ad0596ca0b8b7aaae99d64"
    data=requests.get(url)
    data=data.json()
    poster_path = data['poster_path']
    full_path = f"https://image.tmdb.org/t/p/w500/{poster_path}"
    return full_path

# experiment_path = os.path.abspath(os.path.join(module_path, path, experiment, folder))
movies = load_data_from_pickle(module_path, 'movies', log=False)
map_movie_to_title = load_data_from_pickle(module_path, 'map_movie_to_title', log=False)
links = load_data_from_pickle(module_path, 'links', log=False)
map_title_to_movie = {map_movie_to_title[idx]: idx for idx in map_movie_to_title.keys()}
map_id_to_tmdb = {movieId: tmdbId for movieId, tmdbId in zip(links["movieId"].to_list(), links["tmdbId"].to_list())}

movies_list = movies['title']
ratings_list = list(np.arange(0, 5.5, 0.5))

st.set_page_config(layout='wide')
st.header("Movie Recommender Sysytem Using ALS Method and Matrix Factorization")


# Hyperparameters for recommendation
bias_factors_params = load_data_from_pickle(module_path, 'params_bias_factors').best_params
LAMBDA, GAMMA, TAU = bias_factors_params['LAMBDA'], bias_factors_params['GAMMA'], bias_factors_params['TAU']
K = 16

#################################
st.subheader("Select movies and provide your ratings")

# User can select multiple movies
selected_movies = st.multiselect(
    "Select one or more movies",
    movies_list,
    max_selections=10  # optional limit
)

user_ratings = {}

# For each selected movie, show a slider for rating
for title in selected_movies:
    user_ratings[title] = st.slider(
        f"Rating for: {title}",
        min_value=0.0,
        max_value=5.0,
        value=5.0,
        step=0.5,
        key=f"rating_{title}"
    )

model_choice = st.radio(
    "Select model:",
    ["Bias + Factors", "Bias Only"],
    index=0,
    horizontal=True
)

def render_title_fixed_height(title: str, max_lines: int = 2):
    line_height_em = 1.5
    height_em = max_lines * line_height_em
    st.markdown(
        f"""
        <div style="
            height:{height_em}em;
            overflow:hidden;
            text-overflow:ellipsis;
            display:-webkit-box;
            -webkit-line-clamp:{max_lines};
            -webkit-box-orient:vertical;
            font-size:0.9rem;
        ">
            {title}
        </div>
        """,
        unsafe_allow_html=True,
    )


def display_recommendations(recommendations, recommendation_posters, columns_per_row: int = 5):
    """
    Show recommendations in a clean grid.
    """
    num_recs = len(recommendations)

    for start in range(0, num_recs, columns_per_row):
        end = start + columns_per_row
        row_titles = recommendations[start:end]
        row_posters = recommendation_posters[start:end]

        cols = st.columns(len(row_titles))

        for col, title, poster_url in zip(cols, row_titles, row_posters):
            with col:
                render_title_fixed_height(title, max_lines=2)
                st.image(poster_url, width='stretch')


if st.button("Show Recommendation"):
    movie_ids = [map_title_to_movie[title] for title in selected_movies]
    movie_ratings = [user_ratings[title] for title in selected_movies]

    if model_choice == "Bias + Factors":
        bias_factors_params = load_data_from_pickle(
            module_path,
            "params_bias_factors",
            log=False,
        ).best_params
        LAMBDA = bias_factors_params["LAMBDA"]
        GAMMA = bias_factors_params["GAMMA"]
        TAU = bias_factors_params["TAU"]

        recommendations = recommend_movies_streamlit(
            base_path=module_path,
            movie_ids=movie_ids,
            movie_ratings=movie_ratings,
            top_n=10,
            LAMBDA=LAMBDA,
            GAMMA=GAMMA,
            TAU=TAU,
        )

    else:  # Bias Only
        bias_only_params = load_data_from_pickle(
            module_path,
            "params_bias",       # or whatever you saved it as
            log=False,
        ).best_params
        LAMBDA = bias_only_params["LAMBDA"]
        GAMMA = bias_only_params["GAMMA"]

        recommendations = recommend_movies_bias_only_streamlit(
            base_path=module_path,
            movie_ids=movie_ids,
            movie_ratings=movie_ratings,
            top_n=10,
            LAMBDA=LAMBDA,
            GAMMA=GAMMA,
        )

    recommendation_posters = [
        fetch_movie_poster(map_title_to_movie[movie_title])
        for movie_title in recommendations
    ]

    st.markdown(
        """
        <div style="font-size:1.4rem; font-weight:600; margin-top:1em;">
            You may also like:
        </div>
        """,
        unsafe_allow_html=True,
    )

    display_recommendations(recommendations, recommendation_posters, columns_per_row=5)
