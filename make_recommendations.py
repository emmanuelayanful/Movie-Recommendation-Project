import os
import numpy as np
import data_preprocessing as dp


def recommend_movies(model_path, movie_id, movie_rating, top_n, LAMBDA, GAMMA, TAU):
    # Define paths for loading saved files
    process_path = os.path.join(model_path, 'data preprocess')
    model_files = os.path.join(model_path, 'model files')
    
    # Load model files
    movie_embeddings = dp.load_numpy_array(model_files, 'movie_embeddings', log=False)
    map_movie_to_title = dp.load_data_from_pickle(process_path, 'map_movie_to_title', log=False)
    map_movie_to_id = dp.load_data_from_pickle(process_path,'map_movie_to_id', log=False)
    map_id_to_movie = dp.load_data_from_pickle(process_path,'map_id_to_movie', log=False)
    movie_biases = dp.load_numpy_array(model_files, 'movie_biases', log=False)
    
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
    
    movie_title = map_movie_to_title[movie_id]

    print(f"Since you liked '{movie_title}', \nYou may also like: \n")
    for i, title in enumerate(item_titles):
        print(f"{i+1}. {title}")
        

def recommend_movies_streamlit(model_path, movie_id, movie_rating, top_n, LAMBDA, GAMMA, TAU):
    # Define paths for loading saved files
    process_path = os.path.join(model_path, 'data preprocess')
    model_files = os.path.join(model_path, 'model files')
    
    # Load model files
    movie_embeddings = dp.load_numpy_array(model_files, 'movie_embeddings', log=False)
    map_movie_to_title = dp.load_data_from_pickle(process_path, 'map_movie_to_title', log=False)
    map_movie_to_id = dp.load_data_from_pickle(process_path,'map_movie_to_id', log=False)
    map_id_to_movie = dp.load_data_from_pickle(process_path,'map_id_to_movie', log=False)
    movie_biases = dp.load_numpy_array(model_files, 'movie_biases', log=False)
    
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