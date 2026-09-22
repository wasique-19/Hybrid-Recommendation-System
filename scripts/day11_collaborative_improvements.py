"""
Day 11: Collaborative Filtering Improvements

Improvements over Day 10:
1. Rating-weighted collaborative scoring
2. Positive preference modeling
3. Already-rated movie filtering
4. Cold-start handling
5. Popularity fallback
6. Recommendation evaluation summary
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# 1. PATH CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RATINGS_PATH = PROJECT_ROOT / "processed_data" / "ratings_clean.csv"
MOVIES_PATH = PROJECT_ROOT / "processed_data" / "movie_features.csv"

OUTPUT_DIR = PROJECT_ROOT / "evaluation"
OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================
# 2. LOAD DATA
# ============================================================

print("=" * 70)
print("DAY 11 - COLLABORATIVE FILTERING IMPROVEMENTS")
print("=" * 70)

ratings = pd.read_csv(RATINGS_PATH)
movies = pd.read_csv(MOVIES_PATH)

print("\nRatings shape:", ratings.shape)
print("Movies shape:", movies.shape)


# ============================================================
# 3. CREATE USER-MOVIE MATRIX
# ============================================================

user_movie_matrix = ratings.pivot_table(
    index="user_id",
    columns="movie_id",
    values="rating",
    aggfunc="mean",
    fill_value=0
)

print("\nUser-Movie Matrix Shape:", user_movie_matrix.shape)


# ============================================================
# 4. MOVIE SIMILARITY MATRIX
# ============================================================

print("\nCalculating movie similarity...")

movie_similarity = cosine_similarity(
    user_movie_matrix.T
)

movie_similarity_df = pd.DataFrame(
    movie_similarity,
    index=user_movie_matrix.columns,
    columns=user_movie_matrix.columns
)

print("Movie Similarity Matrix Shape:", movie_similarity_df.shape)


# ============================================================
# 5. HELPER: FIND MOVIE ID
# ============================================================

def find_movie_id(movie_title):
    """
    Find movie_id from movie title.

    Uses case-insensitive partial matching.
    """

    title = str(movie_title).strip().lower()

    matches = movies[
        movies["title"]
        .astype(str)
        .str.lower()
        .str.contains(title, regex=False, na=False)
    ]

    if matches.empty:
        return None

    return int(matches.iloc[0]["movie_id"])


# ============================================================
# 6. RATING TO PREFERENCE
# ============================================================

def rating_to_preference(rating):
    """
    Convert 1-5 rating into preference score.

    1 -> -1.0
    2 -> -0.5
    3 ->  0.0
    4 ->  0.5
    5 ->  1.0
    """

    return (rating - 3.0) / 2.0


# ============================================================
# 7. GET POPULAR MOVIES
# ============================================================

def get_popular_movies(top_k=10):
    """
    Popularity-based fallback.

    Used especially for cold-start users.
    """

    popularity = (
        ratings.groupby("movie_id")
        .agg(
            rating_count=("rating", "count"),
            average_rating=("rating", "mean")
        )
        .reset_index()
    )

    # Require at least 20 ratings
    popularity = popularity[
        popularity["rating_count"] >= 20
    ].copy()

    # Balanced popularity score
    popularity["popularity_score"] = (
        popularity["average_rating"]
        * np.log1p(popularity["rating_count"])
    )

    popularity = popularity.sort_values(
        "popularity_score",
        ascending=False
    )

    result = popularity.head(top_k).merge(
        movies[["movie_id", "title", "genre_text"]],
        on="movie_id",
        how="left"
    )

    return result[
        [
            "movie_id",
            "title",
            "genre_text",
            "average_rating",
            "rating_count",
            "popularity_score"
        ]
    ]


# ============================================================
# 8. RATING-WEIGHTED PERSONALIZED RECOMMENDATIONS
# ============================================================

def rating_weighted_recommendations(
    user_id,
    top_k=10,
    min_similarity=0.05
):
    """
    Generate personalized collaborative recommendations.

    Method:
    1. Get user's rated movies.
    2. Convert ratings into preference scores.
    3. Compare candidate movies with rated movies.
    4. Weight similarities by user preference.
    5. Remove already-rated movies.
    6. Return Top-K.
    """

    # --------------------------------------------------------
    # Check whether user exists
    # --------------------------------------------------------

    if user_id not in user_movie_matrix.index:

        print(
            f"\nUser {user_id} not found."
        )

        print(
            "Using popularity-based cold-start fallback..."
        )

        fallback = get_popular_movies(top_k)

        fallback["recommendation_type"] = "cold_start"

        return fallback


    # --------------------------------------------------------
    # Get user's ratings
    # --------------------------------------------------------

    user_ratings = ratings[
        ratings["user_id"] == user_id
    ][
        ["movie_id", "rating"]
    ].copy()

    if user_ratings.empty:

        print(
            f"\nUser {user_id} has no ratings."
        )

        fallback = get_popular_movies(top_k)

        fallback["recommendation_type"] = "cold_start"

        return fallback


    # --------------------------------------------------------
    # Convert ratings to preferences
    # --------------------------------------------------------

    user_ratings["preference"] = (
        user_ratings["rating"]
        .apply(rating_to_preference)
    )


    # --------------------------------------------------------
    # Store already-rated movies
    # --------------------------------------------------------

    rated_movie_ids = set(
        user_ratings["movie_id"]
    )


    # --------------------------------------------------------
    # Candidate score dictionary
    # --------------------------------------------------------

    candidate_scores = {}

    candidate_weights = {}


    # --------------------------------------------------------
    # Calculate weighted similarity
    # --------------------------------------------------------

    for _, row in user_ratings.iterrows():

        movie_id = int(row["movie_id"])
        preference = float(row["preference"])

        if movie_id not in movie_similarity_df.columns:
            continue

        similar_movies = movie_similarity_df[
            movie_id
        ]

        for candidate_movie_id, similarity in (
            similar_movies.items()
        ):

            candidate_movie_id = int(
                candidate_movie_id
            )

            similarity = float(similarity)

            # Skip itself
            if candidate_movie_id == movie_id:
                continue

            # Skip movies already rated
            if candidate_movie_id in rated_movie_ids:
                continue

            # Ignore very weak similarities
            if similarity < min_similarity:
                continue

            weighted_score = (
                preference * similarity
            )

            candidate_scores[candidate_movie_id] = (
                candidate_scores.get(
                    candidate_movie_id,
                    0.0
                )
                + weighted_score
            )

            candidate_weights[candidate_movie_id] = (
                candidate_weights.get(
                    candidate_movie_id,
                    0.0
                )
                + abs(preference) * similarity
            )


    # --------------------------------------------------------
    # No candidates found
    # --------------------------------------------------------

    if not candidate_scores:

        fallback = get_popular_movies(top_k)

        fallback["recommendation_type"] = (
            "popularity_fallback"
        )

        return fallback


    # --------------------------------------------------------
    # Normalize candidate scores
    # --------------------------------------------------------

    recommendation_rows = []

    for movie_id, score in candidate_scores.items():

        weight = candidate_weights.get(
            movie_id,
            0.0
        )

        if weight == 0:
            continue

        normalized_score = score / weight

        recommendation_rows.append(
            {
                "movie_id": movie_id,
                "collaborative_score": normalized_score
            }
        )


    recommendations = pd.DataFrame(
        recommendation_rows
    )


    # --------------------------------------------------------
    # Sort recommendations
    # --------------------------------------------------------

    recommendations = recommendations.sort_values(
        "collaborative_score",
        ascending=False
    )


    # --------------------------------------------------------
    # Add movie information
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # Return Top-K
    # --------------------------------------------------------

    recommendations = recommendations.head(
        top_k
    ).reset_index(drop=True)

    recommendations[
        "recommendation_type"
    ] = "rating_weighted_collaborative"

    return recommendations


# ============================================================
# 9. TEST USER
# ============================================================

TEST_USER_ID = 1

print("\n" + "=" * 70)
print("TEST USER:", TEST_USER_ID)
print("=" * 70)


# Show user's highest rated movies

test_user_ratings = ratings[
    ratings["user_id"] == TEST_USER_ID
].sort_values(
    "rating",
    ascending=False
).head(10)

test_user_ratings = test_user_ratings.merge(
    movies[
        [
            "movie_id",
            "title"
        ]
    ],
    on="movie_id",
    how="left"
)

print("\nUser's Top Rated Movies:")

print(
    test_user_ratings[
        [
            "movie_id",
            "title",
            "rating"
        ]
    ].to_string(index=False)
)


# ============================================================
# 10. GENERATE RECOMMENDATIONS
# ============================================================

recommendations = rating_weighted_recommendations(
    user_id=TEST_USER_ID,
    top_k=10
)

print("\n" + "=" * 70)
print("RATING-WEIGHTED RECOMMENDATIONS")
print("=" * 70)

print(
    recommendations.to_string(index=False)
)


# ============================================================
# 11. COLD START TEST
# ============================================================

COLD_START_USER_ID = 99999

print("\n" + "=" * 70)
print("COLD-START TEST")
print("=" * 70)

cold_start_recommendations = (
    rating_weighted_recommendations(
        user_id=COLD_START_USER_ID,
        top_k=5
    )
)

print(
    cold_start_recommendations.to_string(
        index=False
    )
)


# ============================================================
# 12. SAVE RESULTS
# ============================================================

output_path = (
    OUTPUT_DIR
    / "day11_rating_weighted_recommendations.csv"
)

recommendations.to_csv(
    output_path,
    index=False
)

cold_start_path = (
    OUTPUT_DIR
    / "day11_cold_start_recommendations.csv"
)

cold_start_recommendations.to_csv(
    cold_start_path,
    index=False
)

print("\nSaved:")
print(output_path)
print(cold_start_path)


# ============================================================
# 13. SUMMARY
# ============================================================

summary = pd.DataFrame(
    {
        "metric": [
            "total_users",
            "total_movies",
            "total_ratings",
            "user_movie_matrix_rows",
            "user_movie_matrix_columns",
            "test_user_id",
            "test_user_rating_count",
            "recommendation_count"
        ],
        "value": [
            ratings["user_id"].nunique(),
            ratings["movie_id"].nunique(),
            len(ratings),
            user_movie_matrix.shape[0],
            user_movie_matrix.shape[1],
            TEST_USER_ID,
            len(test_user_ratings),
            len(recommendations)
        ]
    }
)

summary_path = (
    OUTPUT_DIR
    / "day11_collaborative_summary.csv"
)

summary.to_csv(
    summary_path,
    index=False
)

print(summary.to_string(index=False))

print("\n" + "=" * 70)
print("DAY 11 COMPLETED SUCCESSFULLY")
print("=" * 70)