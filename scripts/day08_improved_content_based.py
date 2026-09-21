
from pathlib import Path
import re
import string

import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MOVIE_FEATURES_PATH = (
    PROJECT_ROOT / "processed_data" / "movie_features.csv"
)

TFIDF_MATRIX_PATH = (
    PROJECT_ROOT / "processed_data" / "movie_tfidf_matrix.npz"
)


# ============================================================
# 2. LOAD DATA
# ============================================================

def load_data():
    """
    Load movie features and TF-IDF matrix.
    """

    if not MOVIE_FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"Movie features file not found: {MOVIE_FEATURES_PATH}"
        )

    if not TFIDF_MATRIX_PATH.exists():
        raise FileNotFoundError(
            f"TF-IDF matrix file not found: {TFIDF_MATRIX_PATH}"
        )

    movies = pd.read_csv(MOVIE_FEATURES_PATH)
    tfidf_matrix = load_npz(TFIDF_MATRIX_PATH)

    if len(movies) != tfidf_matrix.shape[0]:
        raise ValueError(
            "Movie count and TF-IDF matrix row count do not match."
        )

    return movies, tfidf_matrix


# ============================================================
# 3. TITLE NORMALIZATION
# ============================================================

def normalize_title(title):
    """
    Normalize a movie title for better matching.

    Operations:
    1. Convert to lowercase
    2. Move article from end to beginning
    3. Remove release year
    4. Remove punctuation
    5. Remove extra spaces
    """

    if pd.isna(title):
        return ""

    title = str(title).strip().lower()

    # Convert "movie, the" into "the movie"
    article_pattern = r"^(.*),\s*(the|a|an)$"
    article_match = re.match(article_pattern, title)

    if article_match:
        main_title = article_match.group(1).strip()
        article = article_match.group(2).strip()
        title = f"{article} {main_title}"

    # Remove year such as (1995) or [1995]
    title = re.sub(r"[\(\[]\d{4}[\)\]]", "", title)

    # Remove standalone four-digit release year
    title = re.sub(r"\b\d{4}\b", "", title)

    # Remove punctuation
    title = title.translate(
        str.maketrans("", "", string.punctuation)
    )

    # Remove extra spaces
    title = re.sub(r"\s+", " ", title).strip()

    return title


# ============================================================
# 4. PREPARE SEARCH COLUMNS
# ============================================================

def prepare_movie_titles(movies):
    """
    Create normalized title columns.
    """

    movies = movies.copy()

    movies["normalized_title"] = movies["title"].apply(
        normalize_title
    )

    movies["clean_title_normalized"] = movies[
        "clean_title"
    ].apply(normalize_title)

    return movies


# ============================================================
# 5. FIND MOVIE CANDIDATES
# ============================================================

def find_movie_candidates(movies, user_query):
    """
    Find movies using:
    1. Exact normalized title
    2. Partial normalized title
    """

    normalized_query = normalize_title(user_query)

    if not normalized_query:
        return pd.DataFrame()

    exact_matches = movies[
        (movies["normalized_title"] == normalized_query)
        |
        (
            movies["clean_title_normalized"]
            == normalized_query
        )
    ]

    if not exact_matches.empty:
        return exact_matches

    partial_matches = movies[
        movies["normalized_title"].str.contains(
            normalized_query,
            case=False,
            na=False,
            regex=False
        )
        |
        movies["clean_title_normalized"].str.contains(
            normalized_query,
            case=False,
            na=False,
            regex=False
        )
    ]

    return partial_matches


# ============================================================
# 6. DISPLAY SEARCH RESULTS
# ============================================================

def display_search_results(candidates):
    """
    Display possible movie matches.
    """

    if candidates.empty:
        print("\nNo movie found.")
        return

    print("\nPossible movie matches:\n")

    for index, row in candidates.head(10).iterrows():
        print(
            f"Movie ID: {row['movie_id']} | "
            f"Title: {row['title']} | "
            f"Genres: {row['genre_text']}"
        )


# ============================================================
# 7. GET MOVIE INDEX
# ============================================================

def get_movie_index(movies, selected_movie_id):
    """
    Get the DataFrame index for a movie ID.
    """

    matching_rows = movies[
        movies["movie_id"] == selected_movie_id
    ]

    if matching_rows.empty:
        return None

    return matching_rows.index[0]


# ============================================================
# 8. RECOMMENDATION FUNCTION
# ============================================================

def recommend_movies(
    movies,
    tfidf_matrix,
    movie_id,
    top_k=10,
    min_similarity=0.05,
    genre_filter=None
):
    """
    Generate improved content-based recommendations.

    Parameters:
    movies:
        Movie feature DataFrame.

    tfidf_matrix:
        Sparse TF-IDF matrix.

    movie_id:
        Selected movie ID.

    top_k:
        Number of recommendations.

    min_similarity:
        Minimum allowed similarity score.

    genre_filter:
        Optional genre string such as "comedy".
    """

    movie_index = get_movie_index(
        movies,
        movie_id
    )

    if movie_index is None:
        print("Selected movie ID was not found.")
        return pd.DataFrame()

    selected_vector = tfidf_matrix[movie_index]

    similarity_scores = cosine_similarity(
        selected_vector,
        tfidf_matrix
    ).flatten()

    recommendations = movies.copy()

    recommendations["similarity_score"] = similarity_scores

    # Do not recommend the selected movie itself
    recommendations = recommendations[
        recommendations["movie_id"] != movie_id
    ]

    # Apply minimum similarity threshold
    recommendations = recommendations[
        recommendations["similarity_score"]
        >= min_similarity
    ]

    # Optional genre filter
    if genre_filter is not None:
        genre_filter = genre_filter.lower().strip()

        recommendations = recommendations[
            recommendations["genre_text"].str.lower().str.contains(
                genre_filter,
                na=False,
                regex=False
            )
        ]

    # Sort by similarity score
    recommendations = recommendations.sort_values(
        by="similarity_score",
        ascending=False
    )

    # Return only top K
    recommendations = recommendations.head(top_k)

    return recommendations[
        [
            "movie_id",
            "title",
            "genre_text",
            "similarity_score"
        ]
    ]


# ============================================================
# 9. RUN INTERACTIVE RECOMMENDATION
# ============================================================

def run_recommendation_system(
    movies,
    tfidf_matrix
):
    """
    Run an interactive recommendation flow.
    """

    user_query = input(
        "\nEnter a movie title: "
    ).strip()

    if not user_query:
        print("Movie title cannot be empty.")
        return

    candidates = find_movie_candidates(
        movies,
        user_query
    )

    if candidates.empty:
        print(
            "\nMovie not found. "
            "Please try another title."
        )
        return

    display_search_results(candidates)

    selected_movie_id = candidates.iloc[0]["movie_id"]

    if len(candidates) > 1:
        print(
            "\nUsing the first matching movie:"
        )

    selected_movie = movies[
        movies["movie_id"] == selected_movie_id
    ].iloc[0]

    print(
        f"\nSelected movie: {selected_movie['title']}"
    )

    genre_filter = input(
        "Optional genre filter "
        "(press Enter to skip): "
    ).strip()

    if genre_filter == "":
        genre_filter = None

    recommendations = recommend_movies(
        movies=movies,
        tfidf_matrix=tfidf_matrix,
        movie_id=selected_movie_id,
        top_k=10,
        min_similarity=0.05,
        genre_filter=genre_filter
    )

    if recommendations.empty:
        print(
            "\nNo recommendations found "
            "with the selected filters."
        )
        return

    print("\nTop Recommendations:\n")

    for rank, (_, row) in enumerate(
        recommendations.iterrows(),
        start=1
    ):
        print(
            f"{rank}. {row['title']} | "
            f"Genres: {row['genre_text']} | "
            f"Similarity: "
            f"{row['similarity_score']:.4f}"
        )


# ============================================================
# 10. MAIN FUNCTION
# ============================================================

def main():
    print("=" * 60)
    print("DAY 8 - IMPROVED CONTENT-BASED RECOMMENDER")
    print("=" * 60)

    movies, tfidf_matrix = load_data()

    movies = prepare_movie_titles(movies)

    print(f"\nMovies loaded: {movies.shape}")
    print(
        f"TF-IDF matrix shape: "
        f"{tfidf_matrix.shape}"
    )

    # Automatic test
    test_query = "Toy Story (1995)"

    print(
        f"\nTesting title search with: "
        f"{test_query}"
    )

    test_candidates = find_movie_candidates(
        movies,
        test_query
    )

    display_search_results(test_candidates)

    # Interactive system
    run_recommendation_system(
        movies,
        tfidf_matrix
    )

    print("\nDay 8 script completed successfully.")


if __name__ == "__main__":
    main()