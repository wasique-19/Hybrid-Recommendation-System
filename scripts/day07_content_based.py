
from pathlib import Path

import pandas as pd

from scipy.sparse import load_npz
from sklearn.metrics.pairwise import cosine_similarity


# --------------------------------------------------
# 1. Project Paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DATA_DIR = PROJECT_ROOT / "processed_data"

MOVIE_FEATURES_FILE = (
    PROCESSED_DATA_DIR / "movie_features.csv"
)

TFIDF_MATRIX_FILE = (
    PROCESSED_DATA_DIR / "movie_tfidf_matrix.npz"
)


# --------------------------------------------------
# 2. Load Data
# --------------------------------------------------

def load_data():
    print("Loading movie features and TF-IDF matrix...")

    movies = pd.read_csv(
        MOVIE_FEATURES_FILE
    )

    tfidf_matrix = load_npz(
        TFIDF_MATRIX_FILE
    )

    print(f"Movies shape: {movies.shape}")
    print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")

    return movies, tfidf_matrix


# --------------------------------------------------
# 3. Normalize Movie Title
# --------------------------------------------------

def normalize_title(title):
    return (
        str(title)
        .lower()
        .strip()
    )


# --------------------------------------------------
# 4. Find Movie Index
# --------------------------------------------------

def find_movie_index(movies, movie_title):
    normalized_input = normalize_title(
        movie_title
    )

    # Search in clean_title
    exact_matches = movies[
        movies["clean_title"].apply(
            normalize_title
        ) == normalized_input
    ]

    if not exact_matches.empty:
        return exact_matches.index[0]

    # Search in original title
    original_matches = movies[
        movies["title"].apply(
            normalize_title
        ) == normalized_input
    ]

    if not original_matches.empty:
        return original_matches.index[0]

    # Partial title matching
    partial_matches = movies[
        movies["clean_title"]
        .fillna("")
        .str.lower()
        .str.contains(
            normalized_input,
            regex=False
        )
    ]

    if not partial_matches.empty:
        print("\nExact title not found.")
        print("Using first partial match:")
        print(
            partial_matches[
                ["movie_id", "title"]
            ]
            .head(5)
            .to_string(index=False)
        )

        return partial_matches.index[0]

    return None


# --------------------------------------------------
# 5. Content-Based Recommendation
# --------------------------------------------------

def recommend_movies(
    movie_title,
    movies,
    tfidf_matrix,
    top_k=10
):
    movie_index = find_movie_index(
        movies,
        movie_title
    )

    if movie_index is None:
        print(
            f"\nMovie not found: {movie_title}"
        )

        print(
            "\nTry another movie title."
        )

        return pd.DataFrame()

    # Calculate similarity with all movies
    similarity_scores = cosine_similarity(
        tfidf_matrix[movie_index],
        tfidf_matrix
    ).flatten()

    # Sort indices by descending similarity
    similar_indices = similarity_scores.argsort()[
        ::-1
    ]

    recommendations = []

    for index in similar_indices:
        # Exclude the input movie itself
        if index == movie_index:
            continue

        recommendations.append(
            {
                "movie_id": movies.iloc[index]["movie_id"],
                "title": movies.iloc[index]["title"],
                "genre_text": movies.iloc[index]["genre_text"],
                "similarity_score": similarity_scores[index]
            }
        )

        if len(recommendations) >= top_k:
            break

    recommendations_df = pd.DataFrame(
        recommendations
    )

    return recommendations_df


# --------------------------------------------------
# 6. Display Recommendations
# --------------------------------------------------

def display_recommendations(
    movie_title,
    recommendations
):
    print("\n" + "=" * 60)
    print(
        f"Recommendations similar to: "
        f"{movie_title}"
    )
    print("=" * 60)

    if recommendations.empty:
        print("No recommendations available.")
        return

    for position, row in enumerate(
        recommendations.itertuples(index=False),
        start=1
    ):
        print(
            f"{position}. {row.title}"
        )

        print(
            f"   Genres: {row.genre_text}"
        )

        print(
            f"   Similarity: "
            f"{row.similarity_score:.4f}"
        )

        print("-" * 60)


# --------------------------------------------------
# 7. Main Function
# --------------------------------------------------

def main():
    print("=" * 60)
    print("DAY 7 - CONTENT-BASED FILTERING")
    print("=" * 60)

    movies, tfidf_matrix = load_data()

    # Test movie
    test_movie = "Toy Story"

    recommendations = recommend_movies(
        movie_title=test_movie,
        movies=movies,
        tfidf_matrix=tfidf_matrix,
        top_k=10
    )

    display_recommendations(
        test_movie,
        recommendations
    )

    print(
        "\nDay 7 content-based filtering "
        "completed successfully!"
    )


if __name__ == "__main__":
    main()