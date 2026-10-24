import os
import numpy as np
import pickle

def load_data_from_pickle(save_path: str, filename: str, log=True):
    """
    Load a data object from a pickle file.
    
    Args:
        save_path (str): Directory where the pickle file is located.
        filename (str): Name of the pickle file without the extension.
        log (bool): If True, print a message indicating the file loaded.

    Returns:
        object: The loaded data object.
    """
    infile = open(os.path.join(save_path, f'{filename}.pkl'), 'rb')
    data = pickle.load(infile)
    infile.close()
    if log:
        print(f'Loaded {filename} from .pkl file at {save_path}')
    return data

def load_numpy_array(save_path: str, filename: str, log:bool=True):
    """
    Load a NumPy array from file.
    
    Args:
        save_path (str): Directory where the file is located.
        filename (str): Name of the file without the extension.
        log (bool): If True, print a message indicating the file loaded.

    Returns:
        numpy.ndarray: The loaded NumPy array.
    """
    if log:
        print(f'Loaded {filename} from .npy file at {save_path}')
        return np.load(os.path.join(save_path, f'{filename}.npy'), allow_pickle=True)
    else:
        return np.load(os.path.join(save_path, f'{filename}.npy'), allow_pickle=True)

def load_numpy_array(save_path:str, filename:str, log:bool=True) -> np.ndarray:
    return np.load(os.path.join(save_path, f'{filename}.npy'), allow_pickle=True)

def recommend_movies_bias_only_streamlit(base_path: str, movie_ids, movie_ratings, top_n: int, LAMBDA: float, GAMMA: float):
    """
    Recommend movies for a *new* user based on rated movies.
    Args:
        base_path (str): Base directory where the ratings file would be.
        movie_ids (list) : List of movieId values that the user has rated.
        movie_ratings (list) : Corresponding ratings the user gave to those movie_ids.
        top_n (int) : Number of recommendations to return.
        LAMBDA, GAMMA (float): Hyperparameters matching your training setup.
    """

    # load mappings and model parameters
    map_movie_to_id = load_data_from_pickle(base_path, 'map_movie_to_id', log=False)
    map_movie_to_title = load_data_from_pickle(base_path,'map_movie_to_title', log=False)
    map_id_to_movie = load_data_from_pickle(base_path, 'map_id_to_movie', log=False)
    movie_biases = load_numpy_array(base_path, 'movie_biases', log=False)

    # movie_biases
    movie_biases = load_numpy_array(base_path, "movie_biases_b", log=False)
    # global_mean: scalar
    ratings = load_data_from_pickle(base_path, 'ratings', log=False)
    global_mean = ratings['rating'].mean()

    # build arrays for rated movies
    rated_indices = [map_movie_to_id[m_id] for m_id in movie_ids]
    r = np.array(movie_ratings, dtype=float)          # (n,)
    b_i = movie_biases[rated_indices]                 # (n,)
    n_rated = len(rated_indices)

    # compute new user bias b_u 
    residuals = r - (global_mean + b_i)               # (n,)
    user_bias = LAMBDA * residuals.sum() / (LAMBDA * n_rated + GAMMA)

    # predict score for all movies
    scores = global_mean + movie_biases + user_bias   # (num_movies,)

    # don't recommend already-rated movies
    scores[rated_indices] = -np.inf

    # top-N
    top_indices = np.argsort(scores)[-top_n:][::-1]

    top_movie_ids = [map_id_to_movie[i] for i in top_indices]
    top_titles = [map_movie_to_title[m_id] for m_id in top_movie_ids]

    return top_titles




def recommend_movies_streamlit(base_path: str, movie_ids: list, movie_ratings: list, top_n: int, LAMBDA: float, GAMMA: float, TAU: float):
    """
    Recommend movies for a *new* user based on rated movies.

    Args:
        base_path (str): Base directory where the ratings file would be.
        movie_ids (list) : List of movieId values that the user has rated.
        movie_ratings (list) : Corresponding ratings the user gave to those movie_ids.
        top_n (int) : Number of recommendations to return.
        LAMBDA, GAMMA, TAU  (float): Hyperparameters matching your training setup.
    """    

    # load mappings and model parameters
    map_movie_to_id = load_data_from_pickle(base_path, 'map_movie_to_id', log=False)
    map_movie_to_title = load_data_from_pickle(base_path,'map_movie_to_title', log=False)
    movie_embeddings = load_numpy_array(base_path, 'movie_embeddings', log=False)
    map_id_to_movie = load_data_from_pickle(base_path, 'map_id_to_movie', log=False)
    movie_biases = load_numpy_array(base_path, 'movie_biases', log=False)

    k = movie_embeddings.shape[1]   # embedding dimension

    # Build matrices for the rated movies
    # Map movie_ids to internal indices
    rated_indices = [map_movie_to_id[m_id] for m_id in movie_ids]

    # Stack embeddings and biases
    E = movie_embeddings[rated_indices]    
    b = movie_biases[rated_indices]       
    r = np.array(movie_ratings, dtype=float)

    n_rated = E.shape[0]

    # Initialize user embedding and bias
    user_embedding = np.zeros((k,))
    user_bias = 0.0

    # Iterative closed-form updates
    I_k = np.eye(k)

    for _ in range(5):
        # Predicted ratings with current user_embedding & bias (for rated items)
        pred = E @ user_embedding + b + user_bias
        residual = r - pred                        

        # Update user_embedding:
        inverse_term = LAMBDA * (E.T @ E) + TAU * I_k   
        other_term = E.T @ (LAMBDA * (r - (b + user_bias)))
        user_embedding = np.linalg.solve(inverse_term, other_term)

        # Update user_bias
        pred_no_ubias = E @ user_embedding + b
        residual_no_ubias = r - pred_no_ubias
        user_bias = LAMBDA * residual_no_ubias.sum() / (LAMBDA * n_rated + GAMMA)

    # Compute scores for all movies 
    # shape: (num_movies,)
    scores = movie_embeddings @ user_embedding + user_bias + 0.05 * movie_biases

    # Don't recommend the movies the user already rated
    mask = np.ones_like(scores, dtype=bool)
    mask[rated_indices] = False

    scores_filtered = scores.copy()
    scores_filtered[~mask] = -np.inf  # so they never appear in top-N

    # Top-N indices
    top_indices = np.argsort(scores_filtered)[-top_n:][::-1]

    # Map to movie IDs and titles
    top_movie_ids = [map_id_to_movie[i] for i in top_indices]
    item_titles = [map_movie_to_title[m_id] for m_id in top_movie_ids]

    return item_titles