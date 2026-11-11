"""Inference-only functions for a deployed MovieLens ALS recommender.

This module does not download MovieLens data or train models. It reads the
saved movie parameters and trusted parameter dictionaries from ``assets/``.

Rating-prediction models (no explicit global-mean intercept):
    Bias-only:       r_hat(u, i) = b_u + b_i
    Matrix factors:  r_hat(u, i) = b_u + b_i + p_u @ q_i

The Matrix Factorization ranking uses a post-training heuristic:
    rank_score(u, i) = p_u @ q_i + bias_weight * b_i.

The constant new-user bias does not change the ranking of unseen movies.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

DEFAULT_BIAS_WEIGHT = 0.05


@dataclass(frozen=True)
class BiasOnlyModel:
    movie_biases: np.ndarray
    lambda_reg: float
    gamma: float


@dataclass(frozen=True)
class FactorModel:
    movie_biases: np.ndarray
    movie_embeddings: np.ndarray
    lambda_reg: float
    gamma: float
    tau: float


def _parameters(path: Path, required: set[str]) -> dict[str, float]:
    """Load a trusted pickle dictionary containing model hyperparameters.

    The dictionary must use the lowercase keys ``lambda_reg``, ``gamma`` and,
    for the factor model, ``tau``. No Optuna study object is needed at inference.
    """
    with path.open("rb") as file:
        params = pickle.load(file)  # Only load trusted deployment-owned files.

    if not isinstance(params, dict):
        raise TypeError(f"Expected a hyperparameter dictionary in {path}.")

    missing = required - params.keys()
    if missing:
        raise ValueError(f"Missing hyperparameters in {path}: {sorted(missing)}")

    values = {key: float(params[key]) for key in required}
    if not all(np.isfinite(value) for value in values.values()):
        raise ValueError(f"Non-finite hyperparameters in {path}.")
    if values["lambda_reg"] <= 0 or values["gamma"] < 0:
        raise ValueError("lambda_reg must be positive and gamma nonnegative.")
    if "tau" in values and values["tau"] <= 0:
        raise ValueError("tau must be positive.")
    return values


def load_bias_model(asset_dir: str | Path, n_movies: int) -> BiasOnlyModel:
    """Load a Bias-Only model, validating its array length."""
    root = Path(asset_dir)
    params = _parameters(root / "params_bias.pkl", {"lambda_reg", "gamma"})
    biases = np.load(root / "movie_biases_b.npy", allow_pickle=False)
    if biases.shape != (n_movies,) or not np.isfinite(biases).all():
        raise ValueError("Bias-Only movie biases do not match the movie mappings.")
    biases.flags.writeable = False
    return BiasOnlyModel(movie_biases=biases, **params)


def load_factor_model(asset_dir: str | Path, n_movies: int) -> FactorModel:
    """Load a factor model, validating embedding and bias dimensions."""
    root = Path(asset_dir)
    params = _parameters(
        root / "params_bias_factors.pkl", {"lambda_reg", "gamma", "tau"}
    )
    biases = np.load(root / "movie_biases.npy", allow_pickle=False)
    embeddings = np.load(root / "movie_embeddings.npy", allow_pickle=False)
    if (
        biases.shape != (n_movies,)
        or embeddings.ndim != 2
        or embeddings.shape[0] != n_movies
        or embeddings.shape[1] < 1
    ):
        raise ValueError("Factor-model arrays do not match the movie mappings.")
    if not np.isfinite(biases).all() or not np.isfinite(embeddings).all():
        raise ValueError("Factor-model arrays contain non-finite values.")
    biases.flags.writeable = False
    embeddings.flags.writeable = False
    return FactorModel(movie_biases=biases, movie_embeddings=embeddings, **params)


def _validate_inputs(
    movie_ids: Sequence[int],
    movie_ratings: Sequence[float],
    movie_id_to_index: Mapping[int, int],
    top_n: int,
) -> tuple[np.ndarray, np.ndarray]:
    if not movie_ids or len(movie_ids) != len(movie_ratings):
        raise ValueError("Select at least one movie with one rating per movie.")
    if len(set(movie_ids)) != len(movie_ids):
        raise ValueError("Each selected movie must be unique.")
    if not isinstance(top_n, int) or top_n < 1:
        raise ValueError("top_n must be a positive integer.")
    missing = [movie_id for movie_id in movie_ids if movie_id not in movie_id_to_index]
    if missing:
        raise ValueError(f"Selected movies are missing from the model: {missing}")
    ratings = np.asarray(movie_ratings, dtype=np.float64)
    if not np.isfinite(ratings).all() or np.any((ratings < 0.5) | (ratings > 5)):
        raise ValueError("Movie ratings must be between 0.5 and 5.0.")
    indices = np.asarray([movie_id_to_index[x] for x in movie_ids], dtype=np.int64)
    return indices, ratings


def _rank(
    scores: np.ndarray,
    rated_indices: np.ndarray,
    index_to_movie_id: Sequence[int],
    movie_titles: Mapping[int, str],
    top_n: int,
) -> list[dict[str, object]]:
    """Return ranked movie IDs, titles, and scores, excluding rated movies."""
    candidate_scores = np.asarray(scores, dtype=np.float64).copy()
    candidate_scores[rated_indices] = -np.inf
    n = min(top_n, len(candidate_scores) - len(set(rated_indices.tolist())))
    if n <= 0:
        return []
    # Stable descending sort ensures deterministic ties.
    chosen = np.argsort(-candidate_scores, kind="stable")[:n]
    return [
        {
            "movieId": int(index_to_movie_id[int(idx)]),
            "title": movie_titles[int(index_to_movie_id[int(idx)])],
            "score": float(candidate_scores[idx]),
        }
        for idx in chosen
    ]


def recommend_bias_only(
    model: BiasOnlyModel,
    movie_ids: Sequence[int],
    movie_ratings: Sequence[float],
    movie_id_to_index: Mapping[int, int],
    index_to_movie_id: Sequence[int],
    movie_titles: Mapping[int, str],
    top_n: int = 10,
) -> list[dict[str, object]]:
    """
    Rank unseen movies by their learned item biases.

    With a bias-only model, changing a new user's rating values does not
    change item ranking; rating inputs only identify movies to exclude.
    """
    indices, _ = _validate_inputs(movie_ids, movie_ratings, movie_id_to_index, top_n)
    return _rank(model.movie_biases, indices, index_to_movie_id, movie_titles, top_n)


def recommend_factors(
    model: FactorModel,
    movie_ids: Sequence[int],
    movie_ratings: Sequence[float],
    movie_id_to_index: Mapping[int, int],
    index_to_movie_id: Sequence[int],
    movie_titles: Mapping[int, str],
    top_n: int = 10,
    bias_weight: float = DEFAULT_BIAS_WEIGHT,
) -> list[dict[str, object]]:
    """
    Fit a new user with five ALS updates and rank unseen movies.
    """
    indices, ratings = _validate_inputs(movie_ids, movie_ratings, movie_id_to_index, top_n)
    if not np.isfinite(bias_weight) or bias_weight < 0:
        raise ValueError("bias_weight must be finite and nonnegative.")

    E = model.movie_embeddings[indices]
    item_biases = model.movie_biases[indices]
    n_rated, n_factors = E.shape
    user_embedding = np.zeros(n_factors, dtype=np.float64)
    user_bias = 0.0

    normal_matrix = model.lambda_reg * (E.T @ E) + model.tau * np.eye(n_factors)
    for _ in range(5):
        rhs = model.lambda_reg * (E.T @ (ratings - item_biases - user_bias))
        user_embedding = np.linalg.solve(normal_matrix, rhs)
        residuals = ratings - E @ user_embedding - item_biases
        user_bias = model.lambda_reg * residuals.sum() / (
            model.lambda_reg * n_rated + model.gamma
        )

    # User bias is constant across candidate movies and doesn't affect ranking.
    scores = model.movie_embeddings @ user_embedding + bias_weight * model.movie_biases
    return _rank(scores, indices, index_to_movie_id, movie_titles, top_n)
