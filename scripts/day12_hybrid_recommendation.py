"""
Day 12: Basic Hybrid Recommendation Engine

Combines:
1. Content-based filtering
2. Collaborative filtering

Pipeline:
Content Score
      +
Collaborative Score
      ↓
Normalization
      ↓
Weighted Fusion
      ↓
Hybrid Score
      ↓
Top-K Recommendations
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RATINGS_PATH = (
    PROJECT_ROOT
    / "processed_data"
    / "ratings_clean.csv"
)

MOVIES_PATH = (
    PROJECT_ROOT
    / "processed_data"
    / "movie_features.csv"
)

TFIDF_PATH = (
    PROJECT_ROOT
    / "processed_data"
    / "movie_tfidf_matrix.npz"
)

OUTPUT_DIR = PROJECT_ROOT / "evaluation"

OUTPUT_DIR.mkdir(
    exist_ok=True
)


# ============================================================
# 2. CONFIGURATION
# ============================================================

TEST_USER_ID = 1

TOP_K = 10

CONTENT_WEIGHT = 0.4

COLLABORATIVE_WEIGHT = 0.6

MIN_SIMILARITY = 0.05


# ============================================================
# 3. LOAD DATA
# ============================================================

print("=" * 75)
print("DAY 12 - BASIC HYBRID RECOMMENDATION ENGINE")
print("=" * 75)

ratings = pd.read_csv(
    RATINGS_PATH
)

movies = pd.read_csv(
    MOVIES_PATH
)

tfidf_matrix = load_npz(
    TFIDF_PATH
)

print("\nRatings shape:", ratings.shape)

print(
    "Movies shape:",
    movies.shape
)

print(
    "TF-IDF shape:",
    tfidf_matrix.shape
)


# ============================================================
# 4. CREATE USER-MOVIE MATRIX
# ============================================================

user_movie_matrix = ratings.pivot_table(
    index="user_id",
    columns="movie_id",
    values="rating",
    aggfunc="mean",
    fill_value=0
)

print(
    "\nUser-Movie Matrix Shape:",
    user_movie_matrix.shape
)


# ============================================================
# 5. CREATE COLLABORATIVE SIMILARITY
# ============================================================

print(
    "\nCalculating collaborative similarity..."
)

movie_similarity = cosine_similarity(
    user_movie_matrix.T
)

movie_similarity_df = pd.DataFrame(
    movie_similarity,
    index=user_movie_matrix.columns,
    columns=user_movie_matrix.columns
)

print(
    "Movie Similarity Shape:",
    movie_similarity_df.shape
)


# ============================================================
# 6. TITLE NORMALIZATION
# ============================================================

def normalize_title(title):
    """
    Normalize title for easier matching.
    """

    title = str(title).lower().strip()

    return title


# ============================================================
# 7. FIND MOVIE ID
# ============================================================

def find_movie_id(title):
    """
    Find movie ID from title.
    """

    normalized_query = normalize_title(
        title
    )

    movie_titles = (
        movies["title"]
        .astype(str)
        .str.lower()
    )

    # Exact match
    exact_matches = movies[
        movie_titles == normalized_query
    ]

    if not exact_matches.empty:

        return int(
            exact_matches.iloc[0]["movie_id"]
        )

    # Partial match
    partial_matches = movies[
        movie_titles.str.contains(
            normalized_query,
            regex=False,
            na=False
        )
    ]

    if not partial_matches.empty:

        return int(
            partial_matches.iloc[0]["movie_id"]
        )

    return None


# ============================================================
# 8. MIN-MAX NORMALIZATION
# ============================================================

def min_max_normalize(score_dict):
    """
    Normalize dictionary scores between 0 and 1.
    """

    if not score_dict:

        return {}

    values = np.array(
        list(score_dict.values()),
        dtype=float
    )

    min_value = values.min()

    max_value = values.max()

    # Avoid division by zero
    if np.isclose(
        max_value,
        min_value
    ):

        return {
            movie_id: 1.0
            for movie_id in score_dict
        }

    normalized = {}

    for movie_id, score in score_dict.items():

        normalized[movie_id] = (
            (score - min_value)
            / (max_value - min_value)
        )

    return normalized


# ============================================================
# 9. CONTENT-BASED USER PROFILE
# ============================================================

def generate_content_scores(
    user_id,
    rated_movies
):
    """
    Generate content-based scores using
    the user's positively rated movies.

    Ratings >= 4 are treated as positive preferences.
    """

    content_scores = {}

    # --------------------------------------------------------
    # Select positive ratings
    # --------------------------------------------------------

    positive_ratings = rated_movies[
        rated_movies["rating"] >= 4
    ].copy()

    if positive_ratings.empty:

        return content_scores

    # --------------------------------------------------------
    # Movie ID -> TF-IDF row mapping
    # --------------------------------------------------------

    movie_id_to_index = {
        int(movie_id): index
        for index, movie_id
        in enumerate(
            movies["movie_id"]
        )
    }

    # --------------------------------------------------------
    # Process each positively rated movie
    # --------------------------------------------------------

    for _, row in positive_ratings.iterrows():

        movie_id = int(
            row["movie_id"]
        )

        rating = float(
            row["rating"]
        )

        if movie_id not in movie_id_to_index:
            continue

        source_index = (
            movie_id_to_index[movie_id]
        )

        source_vector = (
            tfidf_matrix[source_index]
        )

        similarities = cosine_similarity(
            source_vector,
            tfidf_matrix
        ).flatten()

        # Rating weight
        rating_weight = (
            (rating - 3.0) / 2.0
        )

        for index, similarity in enumerate(
            similarities
        ):

            candidate_movie_id = int(
                movies.iloc[index]["movie_id"]
            )

            # Skip source movie
            if candidate_movie_id == movie_id:
                continue

            if similarity < MIN_SIMILARITY:
                continue

            contribution = (
                similarity
                * rating_weight
            )

            content_scores[
                candidate_movie_id
            ] = (
                content_scores.get(
                    candidate_movie_id,
                    0.0
                )
                + contribution
            )

    return content_scores


# ============================================================
# 10. COLLABORATIVE USER PROFILE
# ============================================================

def generate_collaborative_scores(
    user_id,
    rated_movies
):
    """
    Generate collaborative scores using
    rating-weighted movie similarity.
    """

    collaborative_scores = {}

    for _, row in rated_movies.iterrows():

        movie_id = int(
            row["movie_id"]
        )

        rating = float(
            row["rating"]
        )

        if movie_id not in movie_similarity_df.columns:
            continue

        preference = (
            (rating - 3.0) / 2.0
        )

        similar_movies = (
            movie_similarity_df[movie_id]
        )

        for candidate_movie_id, similarity in (
            similar_movies.items()
        ):

            candidate_movie_id = int(
                candidate_movie_id
            )

            similarity = float(
                similarity
            )

            # Skip source movie
            if candidate_movie_id == movie_id:
                continue

            if similarity < MIN_SIMILARITY:
                continue

            contribution = (
                preference
                * similarity
            )

            collaborative_scores[
                candidate_movie_id
            ] = (
                collaborative_scores.get(
                    candidate_movie_id,
                    0.0
                )
                + contribution
            )

    return collaborative_scores


# ============================================================
# 11. HYBRID RECOMMENDATION FUNCTION
# ============================================================

def hybrid_recommendations(
    user_id,
    top_k=10
):
    """
    Generate hybrid recommendations.

    Steps:
    1. Get user's ratings
    2. Generate content scores
    3. Generate collaborative scores
    4. Normalize both
    5. Weighted fusion
    6. Remove watched movies
    7. Return Top-K
    """

    # --------------------------------------------------------
    # Check user
    # --------------------------------------------------------

    user_ratings = ratings[
        ratings["user_id"] == user_id
    ][
        [
            "movie_id",
            "rating"
        ]
    ].copy()

    if user_ratings.empty:

        print(
            f"\nUser {user_id} has no ratings."
        )

        print(
            "Using popularity fallback."
        )

        return popularity_fallback(
            top_k
        )

    # --------------------------------------------------------
    # Already rated movies
    # --------------------------------------------------------

    rated_movie_ids = set(
        user_ratings["movie_id"]
    )

    # --------------------------------------------------------
    # Generate content scores
    # --------------------------------------------------------

    print(
        "\nGenerating content scores..."
    )

    content_scores = (
        generate_content_scores(
            user_id,
            user_ratings
        )
    )

    print(
        "Content candidates:",
        len(content_scores)
    )

    # --------------------------------------------------------
    # Generate collaborative scores
    # --------------------------------------------------------

    print(
        "Generating collaborative scores..."
    )

    collaborative_scores = (
        generate_collaborative_scores(
            user_id,
            user_ratings
        )
    )

    print(
        "Collaborative candidates:",
        len(collaborative_scores)
    )

    # --------------------------------------------------------
    # Normalize scores
    # --------------------------------------------------------

    normalized_content = (
        min_max_normalize(
            content_scores
        )
    )

    normalized_collaborative = (
        min_max_normalize(
            collaborative_scores
        )
    )

    # --------------------------------------------------------
    # Create union of candidates
    # --------------------------------------------------------

    all_candidates = (
        set(normalized_content.keys())
        |
        set(normalized_collaborative.keys())
    )

    # Remove already rated movies
    all_candidates -= rated_movie_ids

    # --------------------------------------------------------
    # Calculate hybrid score
    # --------------------------------------------------------

    results = []

    for movie_id in all_candidates:

        content_score = (
            normalized_content.get(
                movie_id,
                0.0
            )
        )

        collaborative_score = (
            normalized_collaborative.get(
                movie_id,
                0.0
            )
        )

        hybrid_score = (
            CONTENT_WEIGHT
            * content_score
            +
            COLLABORATIVE_WEIGHT
            * collaborative_score
        )

        results.append(
            {
                "movie_id": movie_id,
                "content_score": content_score,
                "collaborative_score": (
                    collaborative_score
                ),
                "hybrid_score": hybrid_score
            }
        )

    # --------------------------------------------------------
    # No candidates
    # --------------------------------------------------------

    if not results:

        return popularity_fallback(
            top_k
        )

    # --------------------------------------------------------
    # Create DataFrame
    # --------------------------------------------------------

    result_df = pd.DataFrame(
        results
    )

    # --------------------------------------------------------
    # Add movie information
    # --------------------------------------------------------

    result_df = result_df.merge(
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

    # --------------------------------------------------------
    # Sort by hybrid score
    # --------------------------------------------------------

    result_df = result_df.sort_values(
        "hybrid_score",
        ascending=False
    )

    result_df = result_df.head(
        top_k
    ).reset_index(drop=True)

    result_df[
        "recommendation_type"
    ] = "basic_hybrid"

    return result_df[
        [
            "movie_id",
            "title",
            "genre_text",
            "content_score",
            "collaborative_score",
            "hybrid_score",
            "recommendation_type"
        ]
    ]


# ============================================================
# 12. POPULARITY FALLBACK
# ============================================================

def popularity_fallback(top_k=10):
    """
    Popularity fallback for cold-start users.
    """

    popularity = (
        ratings.groupby("movie_id")
        .agg(
            rating_count=("rating", "count"),
            average_rating=("rating", "mean")
        )
        .reset_index()
    )

    popularity = popularity[
        popularity["rating_count"] >= 20
    ].copy()

    popularity["popularity_score"] = (
        popularity["average_rating"]
        * np.log1p(
            popularity["rating_count"]
        )
    )

    popularity = popularity.sort_values(
        "popularity_score",
        ascending=False
    )

    result = popularity.head(
        top_k
    ).merge(
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

    result[
        "content_score"
    ] = 0.0

    result[
        "collaborative_score"
    ] = 0.0

    result[
        "hybrid_score"
    ] = result["popularity_score"]

    result[
        "recommendation_type"
    ] = "popularity_fallback"

    return result[
        [
            "movie_id",
            "title",
            "genre_text",
            "content_score",
            "collaborative_score",
            "hybrid_score",
            "recommendation_type"
        ]
    ]


# ============================================================
# 13. TEST HYBRID MODEL
# ============================================================

print("\n" + "=" * 75)
print("HYBRID RECOMMENDATIONS")
print("=" * 75)

print(
    f"\nTest User: {TEST_USER_ID}"
)

print(
    f"Content Weight: {CONTENT_WEIGHT}"
)

print(
    f"Collaborative Weight: "
    f"{COLLABORATIVE_WEIGHT}"
)

recommendations = hybrid_recommendations(
    user_id=TEST_USER_ID,
    top_k=TOP_K
)

print(
    "\nTop Hybrid Recommendations:\n"
)

print(
    recommendations.to_string(
        index=False
    )
)


# ============================================================
# 14. COLD-START TEST
# ============================================================

print("\n" + "=" * 75)
print("COLD-START TEST")
print("=" * 75)

COLD_START_USER_ID = 99999

cold_start_results = hybrid_recommendations(
    user_id=COLD_START_USER_ID,
    top_k=5
)

print(
    cold_start_results.to_string(
        index=False
    )
)


# ============================================================
# 15. SAVE HYBRID RESULTS
# ============================================================

hybrid_output = (
    OUTPUT_DIR
    / "day12_hybrid_recommendations.csv"
)

recommendations.to_csv(
    hybrid_output,
    index=False
)

cold_start_output = (
    OUTPUT_DIR
    / "day12_hybrid_cold_start.csv"
)

cold_start_results.to_csv(
    cold_start_output,
    index=False
)

print(
    "\nSaved hybrid recommendations:"
)

print(hybrid_output)

print(
    "\nSaved cold-start recommendations:"
)

print(cold_start_output)


# ============================================================
# 16. CREATE SUMMARY
# ============================================================

summary = pd.DataFrame(
    {
        "metric": [
            "total_users",
            "total_movies",
            "total_ratings",
            "content_weight",
            "collaborative_weight",
            "test_user_id",
            "test_user_rating_count",
            "content_candidate_count",
            "collaborative_candidate_count",
            "hybrid_recommendation_count"
        ],
        "value": [
            ratings["user_id"].nunique(),
            ratings["movie_id"].nunique(),
            len(ratings),
            CONTENT_WEIGHT,
            COLLABORATIVE_WEIGHT,
            TEST_USER_ID,
            len(
                ratings[
                    ratings["user_id"]
                    == TEST_USER_ID
                ]
            ),
            len(
                generate_content_scores(
                    TEST_USER_ID,
                    ratings[
                        ratings["user_id"]
                        == TEST_USER_ID
                    ][
                        [
                            "movie_id",
                            "rating"
                        ]
                    ]
                )
            ),
            len(
                generate_collaborative_scores(
                    TEST_USER_ID,
                    ratings[
                        ratings["user_id"]
                        == TEST_USER_ID
                    ][
                        [
                            "movie_id",
                            "rating"
                        ]
                    ]
                )
            ),
            len(recommendations)
        ]
    }
)

summary_output = (
    OUTPUT_DIR
    / "day12_hybrid_summary.csv"
)

summary.to_csv(
    summary_output,
    index=False
)

print(
    "\nSummary:"
)

print(
    summary.to_string(
        index=False
    )
)


# ============================================================
# 17. COMPLETION MESSAGE
# ============================================================

print("\n" + "=" * 75)

print(
    "DAY 12 COMPLETED SUCCESSFULLY"
)

print("=" * 75)