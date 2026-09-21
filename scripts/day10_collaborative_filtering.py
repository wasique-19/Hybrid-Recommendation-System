
from pathlib import Path

import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RATINGS_PATH = (
    PROJECT_ROOT / "processed_data" / "ratings_clean.csv"
)

MOVIES_PATH = (
    PROJECT_ROOT / "processed_data" / "movie_features.csv"
)

EVALUATION_DIR = PROJECT_ROOT / "evaluation"

RECOMMENDATIONS_PATH = (
    EVALUATION_DIR / "collaborative_recommendations.csv"
)

SUMMARY_PATH = (
    EVALUATION_DIR / "collaborative_summary.csv"
)

EVALUATION_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. LOAD DATA
# ============================================================

def load_data():
    """
    Load ratings and movie information.
    """

    if not RATINGS_PATH.exists():
        raise FileNotFoundError(
            f"Ratings file not found: {RATINGS_PATH}"
        )

    if not MOVIES_PATH.exists():
        raise FileNotFoundError(
            f"Movies file not found: {MOVIES_PATH}"
        )

    ratings = pd.read_csv(RATINGS_PATH)
    movies = pd.read_csv(MOVIES_PATH)

    required_rating_columns = {
        "user_id",
        "movie_id",
        "rating"
    }

    missing_columns = (
        required_rating_columns
        - set(ratings.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Missing rating columns: {missing_columns}"
        )

    return ratings, movies


# ============================================================
# 3. CREATE USER-MOVIE MATRIX
# ============================================================

def create_user_movie_matrix(ratings):
    """
    Create a user-movie matrix.

    Rows    = users
    Columns = movies
    Values  = ratings
    """

    user_movie_matrix = ratings.pivot_table(
        index="user_id",
        columns="movie_id",
        values="rating",
        aggfunc="mean",
        fill_value=0
    )

    return user_movie_matrix


# ============================================================
# 4. CREATE ITEM SIMILARITY MATRIX
# ============================================================

def create_item_similarity_matrix(user_movie_matrix):
    """
    Calculate movie-to-movie cosine similarity.

    Input matrix shape:
        users x movies

    Transposed matrix shape:
        movies x users
    """

    movie_user_matrix = user_movie_matrix.T

    movie_similarity = cosine_similarity(
        movie_user_matrix
    )

    movie_similarity_df = pd.DataFrame(
        movie_similarity,
        index=movie_user_matrix.index,
        columns=movie_user_matrix.index
    )

    return movie_similarity_df


# ============================================================
# 5. FIND MOVIE ID
# ============================================================

def find_movie_id(movies, movie_title):
    """
    Find the first movie matching the title.
    """

    matches = movies[
        movies["title"].str.contains(
            movie_title,
            case=False,
            na=False,
            regex=False
        )
    ]

    if matches.empty:
        return None

    return int(matches.iloc[0]["movie_id"])


# ============================================================
# 6. GET RATED MOVIES FOR USER
# ============================================================

def get_user_rated_movies(
    ratings,
    user_id
):
    """
    Return movie IDs already rated by a user.
    """

    user_ratings = ratings[
        ratings["user_id"] == user_id
    ]

    return set(
        user_ratings["movie_id"].astype(int)
    )


# ============================================================
# 7. RECOMMEND SIMILAR MOVIES
# ============================================================

def recommend_similar_movies(
    movie_id,
    movie_similarity_df,
    movies,
    top_k=10
):
    """
    Recommend movies similar to the selected movie.
    """

    if movie_id not in movie_similarity_df.index:
        print(
            "Movie ID not found in similarity matrix."
        )
        return pd.DataFrame()

    similarity_scores = (
        movie_similarity_df[movie_id]
        .sort_values(
            ascending=False
        )
    )

    # Remove the selected movie itself
    similarity_scores = similarity_scores[
        similarity_scores.index != movie_id
    ]

    recommendations = similarity_scores.head(
        top_k
    ).reset_index()

    recommendations.columns = [
        "movie_id",
        "similarity_score"
    ]

    recommendations["movie_id"] = (
        recommendations["movie_id"].astype(int)
    )

    recommendations = recommendations.merge(
        movies[
            [
                "movie_id",
                "title",
                "genre_text"
            ]
        ],
        on="movie_id",
        how="left"
    )

    return recommendations[
        [
            "movie_id",
            "title",
            "genre_text",
            "similarity_score"
        ]
    ]


# ============================================================
# 8. PERSONALIZED ITEM-BASED RECOMMENDATIONS
# ============================================================

def personalized_recommendations(
    user_id,
    ratings,
    movies,
    movie_similarity_df,
    top_k=10
):
    """
    Generate recommendations using movies already
    rated by the selected user.

    Similarity scores from all rated movies
    are combined using the maximum score.
    """

    rated_movie_ids = get_user_rated_movies(
        ratings,
        user_id
    )

    if not rated_movie_ids:
        print(
            "This user has not rated any movies."
        )
        return pd.DataFrame()

    candidate_scores = {}

    for rated_movie_id in rated_movie_ids:

        if rated_movie_id not in movie_similarity_df.index:
            continue

        similarity_scores = (
            movie_similarity_df[rated_movie_id]
        )

        for candidate_movie_id, score in (
            similarity_scores.items()
        ):

            candidate_movie_id = int(
                candidate_movie_id
            )

            # Do not recommend already-rated movies
            if candidate_movie_id in rated_movie_ids:
                continue

            # Ignore self-comparison
            if candidate_movie_id == rated_movie_id:
                continue

            score = float(score)

            if (
                candidate_movie_id not in candidate_scores
                or score > candidate_scores[
                    candidate_movie_id
                ]
            ):
                candidate_scores[
                    candidate_movie_id
                ] = score

    if not candidate_scores:
        return pd.DataFrame()

    recommendations = pd.DataFrame(
        [
            {
                "movie_id": movie_id,
                "similarity_score": score
            }
            for movie_id, score in candidate_scores.items()
        ]
    )

    recommendations = recommendations.sort_values(
        by="similarity_score",
        ascending=False
    ).head(top_k)

    recommendations = recommendations.merge(
        movies[
            [
                "movie_id",
                "title",
                "genre_text"
            ]
        ],
        on="movie_id",
        how="left"
    )

    return recommendations[
        [
            "movie_id",
            "title",
            "genre_text",
            "similarity_score"
        ]
    ]


# ============================================================
# 9. CALCULATE MATRIX STATISTICS
# ============================================================

def calculate_matrix_statistics(
    ratings,
    user_movie_matrix
):
    """
    Calculate basic collaborative filtering statistics.
    """

    total_users = user_movie_matrix.shape[0]
    total_movies = user_movie_matrix.shape[1]

    total_possible_interactions = (
        total_users * total_movies
    )

    observed_interactions = len(ratings)

    sparsity = 1 - (
        observed_interactions
        / total_possible_interactions
    )

    statistics = {
        "total_users": total_users,
        "total_movies": total_movies,
        "observed_ratings": observed_interactions,
        "possible_interactions": (
            total_possible_interactions
        ),
        "matrix_sparsity": sparsity,
        "average_rating": ratings["rating"].mean()
    }

    return statistics


# ============================================================
# 10. MAIN FUNCTION
# ============================================================

def main():
    print("=" * 60)
    print("DAY 10 - COLLABORATIVE FILTERING")
    print("=" * 60)

    ratings, movies = load_data()

    print(
        f"\nRatings shape: {ratings.shape}"
    )

    print(
        f"Movies shape: {movies.shape}"
    )

    # Create user-movie matrix
    user_movie_matrix = create_user_movie_matrix(
        ratings
    )

    print(
        "\nUser-Movie Matrix Shape: "
        f"{user_movie_matrix.shape}"
    )

    # Create item similarity matrix
    movie_similarity_df = (
        create_item_similarity_matrix(
            user_movie_matrix
        )
    )

    print(
        "Movie Similarity Matrix Shape: "
        f"{movie_similarity_df.shape}"
    )

    # Matrix statistics
    statistics = calculate_matrix_statistics(
        ratings,
        user_movie_matrix
    )

    print("\nMatrix Statistics:")

    for key, value in statistics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

    # --------------------------------------------------------
    # Test 1: Similar movie recommendations
    # --------------------------------------------------------

    test_movie_title = "Toy Story"

    test_movie_id = find_movie_id(
        movies,
        test_movie_title
    )

    if test_movie_id is not None:
        print(
            f"\nSimilar movies for: "
            f"{test_movie_title}"
        )

        similar_movies = recommend_similar_movies(
            movie_id=test_movie_id,
            movie_similarity_df=movie_similarity_df,
            movies=movies,
            top_k=10
        )

        if similar_movies.empty:
            print(
                "No similar movies found."
            )
        else:
            print(
                similar_movies.to_string(
                    index=False
                )
            )

    # --------------------------------------------------------
    # Test 2: Personalized recommendations
    # --------------------------------------------------------

    test_user_id = 1

    print(
        f"\nPersonalized recommendations "
        f"for User ID: {test_user_id}"
    )

    personalized_results = (
        personalized_recommendations(
            user_id=test_user_id,
            ratings=ratings,
            movies=movies,
            movie_similarity_df=movie_similarity_df,
            top_k=10
        )
    )

    if personalized_results.empty:
        print(
            "No personalized recommendations found."
        )
    else:
        print(
            personalized_results.to_string(
                index=False
            )
        )

        personalized_results[
            "user_id"
        ] = test_user_id

        personalized_results.to_csv(
            RECOMMENDATIONS_PATH,
            index=False
        )

        print(
            "\nPersonalized recommendations saved: "
            f"{RECOMMENDATIONS_PATH}"
        )

    # Save summary
    summary_df = pd.DataFrame([statistics])

    summary_df.to_csv(
        SUMMARY_PATH,
        index=False
    )

    print(
        f"\nCollaborative summary saved: "
        f"{SUMMARY_PATH}"
    )

    print(
        "\nDay 10 script completed successfully."
    )


if __name__ == "__main__":
    main()