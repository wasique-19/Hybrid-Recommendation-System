"""
Day 13 - Improved Hybrid Recommendation System

Improvements over Day 12:
1. Candidate union from content + collaborative models
2. Consistent 0-1 score normalization
3. Configurable weights
4. Popularity normalization for cold-start
5. Already-rated movie filtering
6. Duplicate removal
7. Minimum hybrid score threshold
8. Genre diversity analysis
9. Weight experiment
"""

import os
import numpy as np
import pandas as pd

from scipy.sparse import load_npz
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RATINGS_PATH = os.path.join(
    BASE_DIR,
    "processed_data",
    "ratings_clean.csv"
)

MOVIES_PATH = os.path.join(
    BASE_DIR,
    "processed_data",
    "movie_features.csv"
)

TFIDF_PATH = os.path.join(
    BASE_DIR,
    "processed_data",
    "movie_tfidf_matrix.npz"
)

OUTPUT_HYBRID = os.path.join(
    BASE_DIR,
    "evaluation",
    "day13_improved_hybrid.csv"
)

OUTPUT_COLD_START = os.path.join(
    BASE_DIR,
    "evaluation",
    "day13_cold_start.csv"
)

OUTPUT_EXPERIMENT = os.path.join(
    BASE_DIR,
    "evaluation",
    "day13_weight_experiment.csv"
)


# ============================================================
# MODEL SETTINGS
# ============================================================

TEST_USER_ID = 1

TOP_K = 10

CONTENT_WEIGHT = 0.4
COLLABORATIVE_WEIGHT = 0.6

MIN_HYBRID_SCORE = 0.05

POSITIVE_RATING_THRESHOLD = 4.0


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("=" * 75)
    print("DAY 13 - IMPROVED HYBRID RECOMMENDATION SYSTEM")
    print("=" * 75)

    ratings = pd.read_csv(RATINGS_PATH)
    movies = pd.read_csv(MOVIES_PATH)

    tfidf_matrix = load_npz(TFIDF_PATH)

    print("\nRatings shape:", ratings.shape)
    print("Movies shape:", movies.shape)
    print("TF-IDF shape:", tfidf_matrix.shape)

    return ratings, movies, tfidf_matrix


# ============================================================
# CREATE USER-MOVIE MATRIX
# ============================================================

def create_user_movie_matrix(ratings):

    user_movie_matrix = ratings.pivot_table(
        index="user_id",
        columns="movie_id",
        values="rating",
        aggfunc="mean",
        fill_value=0
    )

    print("\nUser-Movie Matrix Shape:",
          user_movie_matrix.shape)

    return user_movie_matrix


# ============================================================
# NORMALIZATION
# ============================================================

def min_max_normalize(series):

    series = series.astype(float)

    if len(series) == 0:
        return series

    min_value = series.min()
    max_value = series.max()

    if max_value == min_value:
        return pd.Series(
            np.ones(len(series)),
            index=series.index
        )

    return (series - min_value) / (max_value - min_value)


# ============================================================
# CONTENT SCORE
# ============================================================

def generate_content_scores(
    user_id,
    ratings,
    movies,
    tfidf_matrix
):

    user_ratings = ratings[
        ratings["user_id"] == user_id
    ]

    positive_ratings = user_ratings[
        user_ratings["rating"] >= POSITIVE_RATING_THRESHOLD
    ]

    if positive_ratings.empty:
        return pd.DataFrame(
            columns=["movie_id", "content_score"]
        )

    rated_movie_ids = set(
        user_ratings["movie_id"]
    )

    candidate_scores = {}

    for _, row in positive_ratings.iterrows():

        movie_id = int(row["movie_id"])
        rating = float(row["rating"])

        movie_positions = movies.index[
            movies["movie_id"] == movie_id
        ].tolist()

        if not movie_positions:
            continue

        movie_index = movie_positions[0]

        similarities = cosine_similarity(
            tfidf_matrix[movie_index],
            tfidf_matrix
        ).flatten()

        rating_weight = (rating - 3.0) / 2.0

        for index, similarity in enumerate(similarities):

            candidate_movie_id = int(
                movies.iloc[index]["movie_id"]
            )

            if candidate_movie_id in rated_movie_ids:
                continue

            if candidate_movie_id == movie_id:
                continue

            score = float(similarity) * rating_weight

            if score > 0:

                if (
                    candidate_movie_id not in candidate_scores
                    or score > candidate_scores[candidate_movie_id]
                ):
                    candidate_scores[candidate_movie_id] = score

    content_df = pd.DataFrame(
        list(candidate_scores.items()),
        columns=["movie_id", "content_score"]
    )

    if content_df.empty:
        return content_df

    content_df["content_score"] = min_max_normalize(
        content_df["content_score"]
    )

    return content_df


# ============================================================
# COLLABORATIVE SCORE
# ============================================================

def generate_collaborative_scores(
    user_id,
    ratings,
    movies,
    user_movie_matrix
):

    if user_id not in user_movie_matrix.index:

        return pd.DataFrame(
            columns=["movie_id", "collaborative_score"]
        )

    user_ratings = ratings[
        ratings["user_id"] == user_id
    ]

    rated_movie_ids = set(
        user_ratings["movie_id"]
    )

    positive_ratings = user_ratings[
        user_ratings["rating"] >= POSITIVE_RATING_THRESHOLD
    ]

    if positive_ratings.empty:

        return pd.DataFrame(
            columns=["movie_id", "collaborative_score"]
        )

    print("\nCalculating collaborative similarity...")

    movie_similarity = cosine_similarity(
        user_movie_matrix.T
    )

    movie_ids = list(
        user_movie_matrix.columns
    )

    movie_id_to_index = {
        movie_id: index
        for index, movie_id in enumerate(movie_ids)
    }

    candidate_scores = {}

    for _, row in positive_ratings.iterrows():

        movie_id = int(row["movie_id"])

        if movie_id not in movie_id_to_index:
            continue

        movie_index = movie_id_to_index[movie_id]

        rating_weight = (
            float(row["rating"]) - 3.0
        ) / 2.0

        similarities = movie_similarity[
            movie_index
        ]

        for index, similarity in enumerate(similarities):

            candidate_movie_id = int(
                movie_ids[index]
            )

            if candidate_movie_id in rated_movie_ids:
                continue

            if candidate_movie_id == movie_id:
                continue

            score = (
                float(similarity)
                * rating_weight
            )

            if score > 0:

                if (
                    candidate_movie_id not in candidate_scores
                    or score > candidate_scores[candidate_movie_id]
                ):
                    candidate_scores[
                        candidate_movie_id
                    ] = score

    collaborative_df = pd.DataFrame(
        list(candidate_scores.items()),
        columns=[
            "movie_id",
            "collaborative_score"
        ]
    )

    if collaborative_df.empty:
        return collaborative_df

    collaborative_df[
        "collaborative_score"
    ] = min_max_normalize(
        collaborative_df["collaborative_score"]
    )

    return collaborative_df


# ============================================================
# POPULARITY SCORE
# ============================================================

def calculate_popularity(movies, ratings):

    popularity = (
        ratings.groupby("movie_id")
        .agg(
            average_rating=("rating", "mean"),
            rating_count=("rating", "count")
        )
        .reset_index()
    )

    popularity["raw_popularity"] = (
        popularity["average_rating"]
        * np.log1p(popularity["rating_count"])
    )

    popularity["popularity_score"] = (
        min_max_normalize(
            popularity["raw_popularity"]
        )
    )

    popularity = popularity.merge(
        movies[
            ["movie_id", "title", "genre_text"]
        ],
        on="movie_id",
        how="left"
    )

    return popularity


# ============================================================
# HYBRID RECOMMENDATIONS
# ============================================================

def generate_hybrid_recommendations(
    user_id,
    ratings,
    movies,
    tfidf_matrix,
    user_movie_matrix,
    content_weight=0.4,
    collaborative_weight=0.6,
    top_k=10
):

    print("\n" + "=" * 75)
    print("GENERATING IMPROVED HYBRID RECOMMENDATIONS")
    print("=" * 75)

    content_df = generate_content_scores(
        user_id,
        ratings,
        movies,
        tfidf_matrix
    )

    collaborative_df = generate_collaborative_scores(
        user_id,
        ratings,
        movies,
        user_movie_matrix
    )

    print("\nContent candidates:",
          len(content_df))

    print("Collaborative candidates:",
          len(collaborative_df))

    # --------------------------------------------------------
    # Candidate UNION
    # --------------------------------------------------------

    hybrid_df = pd.merge(
        content_df,
        collaborative_df,
        on="movie_id",
        how="outer"
    )

    hybrid_df["content_score"] = (
        hybrid_df["content_score"]
        .fillna(0)
    )

    hybrid_df["collaborative_score"] = (
        hybrid_df["collaborative_score"]
        .fillna(0)
    )

    # --------------------------------------------------------
    # HYBRID SCORE
    # --------------------------------------------------------

    hybrid_df["hybrid_score"] = (
        content_weight
        * hybrid_df["content_score"]
        +
        collaborative_weight
        * hybrid_df["collaborative_score"]
    )

    # --------------------------------------------------------
    # REMOVE LOW SCORE
    # --------------------------------------------------------

    hybrid_df = hybrid_df[
        hybrid_df["hybrid_score"]
        >= MIN_HYBRID_SCORE
    ]

    # --------------------------------------------------------
    # REMOVE DUPLICATES
    # --------------------------------------------------------

    hybrid_df = hybrid_df.drop_duplicates(
        subset=["movie_id"]
    )

    # --------------------------------------------------------
    # ADD MOVIE INFORMATION
    # --------------------------------------------------------

    hybrid_df = hybrid_df.merge(
        movies[
            ["movie_id", "title", "genre_text"]
        ],
        on="movie_id",
        how="left"
    )

    # --------------------------------------------------------
    # SORT
    # --------------------------------------------------------

    hybrid_df = hybrid_df.sort_values(
        "hybrid_score",
        ascending=False
    )

    hybrid_df = hybrid_df.head(top_k)

    hybrid_df["recommendation_type"] = (
        "improved_hybrid"
    )

    return hybrid_df


# ============================================================
# COLD START
# ============================================================

def cold_start_recommendations(
    ratings,
    movies,
    top_k=10
):

    print("\n" + "=" * 75)
    print("COLD-START RECOMMENDATIONS")
    print("=" * 75)

    popularity = calculate_popularity(
        movies,
        ratings
    )

    recommendations = popularity.sort_values(
        "popularity_score",
        ascending=False
    ).head(top_k).copy()

    recommendations["content_score"] = 0.0

    recommendations["collaborative_score"] = 0.0

    recommendations["hybrid_score"] = (
        recommendations["popularity_score"]
    )

    recommendations["recommendation_type"] = (
        "popularity_fallback"
    )

    return recommendations[
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
# DIVERSITY ANALYSIS
# ============================================================

def calculate_genre_diversity(recommendations):

    if recommendations.empty:
        return 0

    genres = set()

    for genre_text in recommendations[
        "genre_text"
    ].fillna(""):

        for genre in str(
            genre_text
        ).split():

            if genre.strip():
                genres.add(
                    genre.strip()
                )

    return len(genres)


# ============================================================
# WEIGHT EXPERIMENT
# ============================================================

def run_weight_experiment(
    user_id,
    ratings,
    movies,
    tfidf_matrix,
    user_movie_matrix
):

    print("\n" + "=" * 75)
    print("WEIGHT EXPERIMENT")
    print("=" * 75)

    experiments = [
        (0.2, 0.8),
        (0.3, 0.7),
        (0.4, 0.6),
        (0.5, 0.5),
        (0.6, 0.4),
        (0.7, 0.3),
        (0.8, 0.2)
    ]

    results = []

    for content_weight, collaborative_weight in experiments:

        recommendations = generate_hybrid_recommendations(
            user_id,
            ratings,
            movies,
            tfidf_matrix,
            user_movie_matrix,
            content_weight,
            collaborative_weight,
            TOP_K
        )

        diversity = calculate_genre_diversity(
            recommendations
        )

        average_score = (
            recommendations["hybrid_score"]
            .mean()
            if not recommendations.empty
            else 0
        )

        results.append(
            {
                "content_weight": content_weight,
                "collaborative_weight":
                    collaborative_weight,
                "recommendation_count":
                    len(recommendations),
                "average_hybrid_score":
                    average_score,
                "genre_diversity":
                    diversity
            }
        )

    return pd.DataFrame(results)


# ============================================================
# MAIN
# ============================================================

def main():

    ratings, movies, tfidf_matrix = load_data()

    user_movie_matrix = create_user_movie_matrix(
        ratings
    )

    # --------------------------------------------------------
    # MAIN HYBRID MODEL
    # --------------------------------------------------------

    recommendations = generate_hybrid_recommendations(
        TEST_USER_ID,
        ratings,
        movies,
        tfidf_matrix,
        user_movie_matrix,
        CONTENT_WEIGHT,
        COLLABORATIVE_WEIGHT,
        TOP_K
    )

    print("\n" + "=" * 75)
    print("TOP HYBRID RECOMMENDATIONS")
    print("=" * 75)

    print(
        recommendations[
            [
                "movie_id",
                "title",
                "genre_text",
                "content_score",
                "collaborative_score",
                "hybrid_score"
            ]
        ].to_string(index=False)
    )

    diversity = calculate_genre_diversity(
        recommendations
    )

    print("\nGenre Diversity:", diversity)

    # --------------------------------------------------------
    # SAVE HYBRID RESULTS
    # --------------------------------------------------------

    recommendations.to_csv(
        OUTPUT_HYBRID,
        index=False
    )

    print(
        "\nSaved:",
        OUTPUT_HYBRID
    )

    # --------------------------------------------------------
    # COLD START
    # --------------------------------------------------------

    cold_start = cold_start_recommendations(
        ratings,
        movies,
        TOP_K
    )

    print(
        "\n",
        cold_start.to_string(index=False)
    )

    cold_start.to_csv(
        OUTPUT_COLD_START,
        index=False
    )

    print(
        "\nSaved:",
        OUTPUT_COLD_START
    )

    # --------------------------------------------------------
    # WEIGHT EXPERIMENT
    # --------------------------------------------------------

    experiment_results = run_weight_experiment(
        TEST_USER_ID,
        ratings,
        movies,
        tfidf_matrix,
        user_movie_matrix
    )

    print("\n" + "=" * 75)
    print("WEIGHT EXPERIMENT RESULTS")
    print("=" * 75)

    print(
        experiment_results.to_string(
            index=False
        )
    )

    experiment_results.to_csv(
        OUTPUT_EXPERIMENT,
        index=False
    )

    print(
        "\nSaved:",
        OUTPUT_EXPERIMENT
    )

    # --------------------------------------------------------
    # FINAL STATUS
    # --------------------------------------------------------

    print("\n" + "=" * 75)
    print("DAY 13 COMPLETED SUCCESSFULLY")
    print("=" * 75)


if __name__ == "__main__":
    main()