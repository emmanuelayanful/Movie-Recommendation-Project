import os
import sys
import pickle
import numpy as np
import streamlit as st
import requests


path = os.path.abspath(os.path.join(os.path.dirname('app.py')))
sys.path.insert(0, path)

def load_data_from_pickle(save_path, filename):
    infile = open(os.path.join(save_path, f'{filename}.pkl'), 'rb')
    data = pickle.load(infile)
    infile.close()
    return data

def load_numpy_array(save_path, filename, log=True):
    return np.load(os.path.join(save_path, f'{filename}.npy'), allow_pickle=True)

def recommend_movies_streamlit(path, movie_id, movie_rating, top_n, LAMBDA, GAMMA, TAU):
    # Define paths for loading saved files
    
    # Load model files
    movie_embeddings = load_numpy_array(path, 'movie_embeddings')
    map_movie_to_title = load_data_from_pickle(path, 'map_movie_to_title')
    map_movie_to_id = load_data_from_pickle(path,'map_movie_to_id')
    map_id_to_movie = load_data_from_pickle(path,'map_id_to_movie')
    movie_biases = load_numpy_array(path, 'movie_biases')
    
    # Update user embeddings
    movie_idx = map_movie_to_id[movie_id]
    user_movie_embedding = movie_embeddings[movie_idx]
    user_movie_bias = movie_biases[movie_idx]
    user_embedding = np.zeros((1, movie_embeddings.shape[1]))
    user_bias = LAMBDA * (movie_rating - (user_embedding @ user_movie_embedding + user_movie_bias)) / (LAMBDA  + GAMMA)
    
    for _ in range(5):
        inverse_term = LAMBDA * (user_movie_embedding.T @ user_movie_embedding) + TAU * np.eye(movie_embeddings.shape[1])
        other_term = user_movie_embedding.T * (movie_rating - (user_movie_bias + user_bias))
        user_embedding = np.linalg.solve(inverse_term, other_term)
        user_bias = LAMBDA * (movie_rating - (user_embedding.T @ user_movie_embedding + user_movie_bias)) / (LAMBDA  + GAMMA)
    
    # Compute scores for all items
    scores = (
        np.inner(user_embedding, movie_embeddings) +
        0.05 * movie_biases
    )

    # Get  the indices of the top_n scores
    top_indices = np.argsort(scores)[-top_n:][::-1]

    # Map back to movie IDs
    top_movie_ids = [map_id_to_movie[i] for i in top_indices]

    # Map back to movie Titles
    item_titles = [map_movie_to_title[item_id] for item_id in top_movie_ids]
    
    return item_titles

def fetch_movie_poster(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{map_id_to_tmdb[movie_id]}?api_key=9f1339eba7ad0596ca0b8b7aaae99d64"
    data=requests.get(url)
    data=data.json()
    poster_path = data['poster_path']
    full_path = f"https://image.tmdb.org/t/p/w500/{poster_path}"
    return full_path


movies = load_data_from_pickle(path, 'movies')
map_movie_to_title = load_data_from_pickle(path, 'map_movie_to_title')
links = load_data_from_pickle(path, 'links')
map_title_to_movie = {map_movie_to_title[idx]: idx for idx in map_movie_to_title.keys()}
map_id_to_tmdb = {movieId: tmdbId for movieId, tmdbId in zip(links["movieId"].to_list(), links["tmdbId"].to_list())}

movies_list = movies['title']
ratings_list = list(np.arange(0.5, 5.5, 0.5))

st.set_page_config(layout='wide')
st.header("Movie Recommender Sysytem Using ALS Method and Matrix Factorization")
select_movie = st.selectbox("Select movie from dropdown", movies_list)
select_rating = st.selectbox("Select rating (0.5-5)", ratings_list)

movie_id = map_title_to_movie[select_movie]

# Hyperparameters for recommendation
LAMBDA = 0.0317
GAMMA = 0.0007
TAU = 0.6060
K = 16

if st.button("Show Recommendation"):
    recommendations = recommend_movies_streamlit(path, movie_id, select_rating, 10, LAMBDA, GAMMA, TAU)
    recommendation_posters = [fetch_movie_poster(map_title_to_movie[movie_title]) for movie_title in recommendations]
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.text(recommendations[0])
        st.image(recommendation_posters[0])
        st.text(recommendations[5])
        st.image(recommendation_posters[5])
    with col2:
        st.text(recommendations[1])
        st.image(recommendation_posters[1])
        st.text(recommendations[6])
        st.image(recommendation_posters[6])
    with col3:
        st.text(recommendations[2])
        st.image(recommendation_posters[2])
        st.text(recommendations[7])
        st.image(recommendation_posters[7])
    with col4:
        st.text(recommendations[3])
        st.image(recommendation_posters[3])
        st.text(recommendations[8])
        st.image(recommendation_posters[8])
    with col5:
        st.text(recommendations[4])
        st.image(recommendation_posters[4])
        st.text(recommendations[9])
        st.image(recommendation_posters[9])