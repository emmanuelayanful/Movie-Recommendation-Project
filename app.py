"""Streamlit inference-only interface for pre-trained MovieLens ALS models.

Run from the deployment repository root:
    streamlit run app.py

All local model and movie metadata files are read from ``assets/``.
No training code, dataset selection, or generated catalog is required.
"""

from __future__ import annotations

import os
import pickle
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path

import requests
import streamlit as st

from make_recommendation import (
    DEFAULT_BIAS_WEIGHT,
    load_bias_model,
    load_factor_model,
    recommend_bias_only,
    recommend_factors,
)

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets"
MAX_SELECTED = 10
MAX_SEARCH_RESULTS = 25

st.set_page_config(page_title="Movie Recommendation System", layout="wide")
st.title("Movie Recommendation System")
st.caption("Alternating Least Squares · Bias-Only and Matrix Factorization")


@st.cache_resource(show_spinner=False)
def load_movie_mappings():
    """Load the original MovieLens ID mappings and build a compact search list.

    Internal indices come from map_movie_to_id.pkl/map_id_to_movie.pkl,
    never from sorting movie titles or reconstructing indices from movies.pkl.
    This guarantees correct alignment with the saved model arrays.
    """
    def read_pickle(name: str):
        with (ASSETS / f"{name}.pkl").open("rb") as file:
            return pickle.load(file)  # Only trust repository-owned assets.

    movie_to_index = read_pickle("map_movie_to_id")
    index_to_movie = read_pickle("map_id_to_movie")
    movie_to_title = read_pickle("map_movie_to_title")

    n_movies = len(index_to_movie)
    if not n_movies or len(movie_to_index) != n_movies:
        raise ValueError("Movie mapping sizes do not agree.")

    try:
        index_to_id = [int(index_to_movie[idx]) for idx in range(n_movies)]
    except KeyError as exc:
        raise ValueError("Internal movie indices must cover 0 through N-1.") from exc

    if any(movie_to_index.get(movie_id) != idx for idx, movie_id in enumerate(index_to_id)):
        raise ValueError("Movie index mappings are not mutually consistent.")

    # Only movies represented in the model can be selected or recommended.
    titles = {
        movie_id: str(movie_to_title.get(movie_id, f"Movie {movie_id}"))
        for movie_id in index_to_id
    }
    searchable = [(movie_id, title.casefold()) for movie_id, title in titles.items()]
    return index_to_id, movie_to_index, titles, searchable


@st.cache_resource(show_spinner=False)
def load_tmdb_links():
    """Read the optional local links.pkl metadata (a Polars DataFrame)."""
    path = ASSETS / "links.pkl"
    if not path.is_file():
        return {}
    with path.open("rb") as file:
        frame = pickle.load(file)  # Trusted local file; requires Polars.
    return {
        int(movie_id): int(tmdb_id)
        for movie_id, tmdb_id in zip(
            frame["movieId"].to_list(), frame["tmdbId"].to_list()
        )
        if tmdb_id is not None
    }


@st.cache_resource(show_spinner=False)
def get_model(model_name: str, n_movies: int):
    """Load the chosen trained model only once per running Streamlit server."""
    if model_name == "Bias + Factors":
        return load_factor_model(ASSETS, n_movies)
    if model_name == "Bias Only":
        return load_bias_model(ASSETS, n_movies)
    raise ValueError(f"Unknown recommendation model: {model_name}")


def _tmdb_api_key() -> str:
    key = os.environ.get("TMDB_API_KEY", "")
    try:
        key = st.secrets.get("TMDB_API_KEY", key)
    except Exception:
        pass
    return str(key).strip()


@lru_cache(maxsize=2048)
def poster_url(tmdb_id: int, api_key: str) -> str | None:
    """Look up a poster URL with a bounded network timeout."""
    try:
        response = requests.get(
            f"https://api.themoviedb.org/3/movie/{tmdb_id}",
            params={"api_key": api_key},
            timeout=(2.5, 5.0),
        )
        response.raise_for_status()
        path = response.json().get("poster_path")
        return f"https://image.tmdb.org/t/p/w342{path}" if path else None
    except (requests.RequestException, ValueError):
        return None


def posters_for(
    recommendations: list[dict[str, object]], api_key: str
) -> dict[int, str | None]:
    """Fetch only recommended movie posters, in a small thread pool."""
    if not api_key:
        return {}
    try:
        movie_to_tmdb = load_tmdb_links()
    except (OSError, ValueError, ImportError):
        return {}
    tmdb_ids = {
        int(rec["movieId"]): movie_to_tmdb[int(rec["movieId"])]
        for rec in recommendations
        if int(rec["movieId"]) in movie_to_tmdb
    }
    if not tmdb_ids:
        return {}

    with ThreadPoolExecutor(max_workers=5) as executor:
        images = list(executor.map(lambda tmdb_id: poster_url(tmdb_id, api_key), tmdb_ids.values()))
    return dict(zip(tmdb_ids, images))


def find_movies(term: str, searchable: list[tuple[int, str]]) -> list[int]:
    """Filter on the server; send at most 25 movie choices to the browser."""
    query = term.strip().casefold()
    if len(query) < 2:
        return []
    matches = []
    for movie_id, lowercase_title in searchable:
        if query in lowercase_title:
            matches.append(movie_id)
            if len(matches) == MAX_SEARCH_RESULTS:
                break
    return matches


try:
    index_to_id, id_to_index, titles, searchable = load_movie_mappings()
except (OSError, KeyError, ValueError, TypeError, ModuleNotFoundError) as exc:
    st.error(f"Could not load movie mappings from assets/: {exc}")
    st.stop()

if "selected_movies" not in st.session_state:
    st.session_state.selected_movies = {}  # External MovieLens ID -> rating.
if "search_matches" not in st.session_state:
    st.session_state.search_matches = []

st.subheader("Search for movies")
with st.form("search_form"):
    query = st.text_input(
        "Movie title", placeholder="Try: Toy Story, Lord of the Rings, Twilight"
    )
    do_search = st.form_submit_button("Search")

if do_search:
    st.session_state.search_matches = find_movies(query, searchable)
    if len(query.strip()) < 2:
        st.info("Enter at least two characters to search.")
    elif not st.session_state.search_matches:
        st.info("No matching movie found in the trained model.")

if st.session_state.search_matches:
    with st.form("add_movie_form"):
        selected_id = st.selectbox(
            "Matching movies (up to 25)",
            st.session_state.search_matches,
            format_func=lambda movie_id: f"{titles[movie_id]} · ID {movie_id}",
        )
        add_clicked = st.form_submit_button("Add movie")

    if add_clicked:
        if selected_id in st.session_state.selected_movies:
            st.info("This movie is already selected.")
        elif len(st.session_state.selected_movies) >= MAX_SELECTED:
            st.warning(f"Choose at most {MAX_SELECTED} movies.")
        else:
            st.session_state.selected_movies[selected_id] = 5.0
            st.session_state.pop("recommendations", None)
            st.rerun()

st.subheader("Your selected movies and ratings")
if not st.session_state.selected_movies:
    st.info("Search and add at least one movie to begin.")
else:
    for movie_id, rating in list(st.session_state.selected_movies.items()):
        rating_col, remove_col = st.columns([6, 1])
        with rating_col:
            new_rating = st.slider(
                f"{titles[movie_id]} (ID {movie_id})",
                min_value=0.5,
                max_value=5.0,
                value=rating,
                step=0.5,
                key=f"rating_{movie_id}",
            )
            if new_rating != rating:
                st.session_state.selected_movies[movie_id] = new_rating
                st.session_state.pop("recommendations", None)
        with remove_col:
            if st.button("Remove", key=f"remove_{movie_id}"):
                st.session_state.selected_movies.pop(movie_id, None)
                st.session_state.pop("recommendations", None)
                st.rerun()

model_choice = st.radio(
    "Recommendation model", ("Bias + Factors", "Bias Only"), horizontal=True
)
if model_choice == "Bias Only":
    st.caption(
        "Bias-Only ranks unseen movies by learned item biases. Changing the rating "
        "values does not change the ranking."
    )
else:
    st.caption(
        "Matrix Factorization uses personalized latent factors with "
        f"a movie-bias ranking weight of {DEFAULT_BIAS_WEIGHT}."
    )

if st.button(
    "Show recommendations", type="primary",
    disabled=not bool(st.session_state.selected_movies),
):
    movie_ids = list(st.session_state.selected_movies)
    movie_ratings = list(st.session_state.selected_movies.values())
    try:
        with st.spinner("Generating recommendations..."):
            model = get_model(model_choice, len(index_to_id))
            if model_choice == "Bias + Factors":
                recommendations = recommend_factors(
                    model, movie_ids, movie_ratings,
                    id_to_index, index_to_id, titles, top_n=10,
                )
            else:
                recommendations = recommend_bias_only(
                    model, movie_ids, movie_ratings,
                    id_to_index, index_to_id, titles, top_n=10,
                )
            posters = posters_for(recommendations, _tmdb_api_key())
        # Store the input state to avoid displaying stale results after changes.
        signature = (model_choice, tuple(zip(movie_ids, movie_ratings)))
        st.session_state.recommendations = (signature, recommendations, posters)
    except (OSError, ValueError, KeyError, TypeError, ModuleNotFoundError) as exc:
        st.error(f"Could not generate recommendations: {exc}")

if "recommendations" in st.session_state:
    signature, recommendations, posters = st.session_state.recommendations
    current_signature = (
        model_choice,
        tuple(st.session_state.selected_movies.items()),
    )
    if signature == current_signature:
        st.subheader("You may also like")
        for start in range(0, len(recommendations), 5):
            row = recommendations[start:start + 5]
            cols = st.columns(len(row))
            for col, recommendation in zip(cols, row):
                with col:
                    movie_id = int(recommendation["movieId"])
                    st.caption(str(recommendation["title"]))
                    poster = posters.get(movie_id)
                    if poster:
                        st.image(poster, width="stretch")
                    else:
                        st.caption("Poster unavailable")
                    st.caption(f"MovieLens ID {movie_id}")
