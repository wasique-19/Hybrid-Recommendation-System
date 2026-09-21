
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

EVALUATION_DIR = PROJECT_ROOT / "evaluation"
PLOTS_DIR = EVALUATION_DIR / "plots"

RECOMMENDATIONS_OUTPUT_PATH = (
    EVALUATION_DIR / "content_recommendations.csv"
)

SUMMARY_OUTPUT_PATH = (
    EVALUATION_DIR / "content_evaluation_summary.csv"
)

PLOTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. LOAD DATA
# ============================================================

def load_data():
    """
    Load movie features and TF-IDF matrix.
    """

    if not MOVIE_FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"Missing file: {MOVIE_FEATURES_PATH}"
        )

    if not TFIDF_MATRIX_PATH.exists():
        raise FileNotFoundError(
            f"Missing file: {TFIDF_MATRIX_PATH}"
        )

    movies = pd.read_csv(MOVIE_FEATURES_PATH)
    tfidf_matrix = load_npz(TFIDF_MATRIX_PATH)

    if len(movies) != tfidf_matrix.shape[0]:
        raise ValueError(
            "Movie rows and TF-IDF rows do not match."
        )

    return movies, tfidf_matrix


# ============================================================
# 3. FIND MOVIE
# ============================================================

def find_movie_id(movies, movie_title):
    """
    Find a movie ID using a case-insensitive title search.
    """

    title_matches = movies[
        movies["title"].str.contains(
            movie_title,
            case=False,
            na=False,
            regex=False
        )
    ]

    if title_matches.empty:
        return None

    return title_matches.iloc[0]["movie_id"]


# ============================================================
# 4. RECOMMEND MOVIES
# ============================================================

def recommend_movies(
    movies,
    tfidf_matrix,
    movie_id,
    top_k=10
):
    """
    Generate Top-K content-based recommendations.
    """

    matching_rows = movies[
        movies["movie_id"] == movie_id
    ]

    if matching_rows.empty:
        return pd.DataFrame()

    movie_index = matching_rows.index[0]

    selected_vector = tfidf_matrix[movie_index]

    similarity_scores = cosine_similarity(
        selected_vector,
        tfidf_matrix
    ).flatten()

    recommendations = movies.copy()

    recommendations["similarity_score"] = (
        similarity_scores
    )

    # Exclude the selected movie
    recommendations = recommendations[
        recommendations["movie_id"] != movie_id
    ]

    recommendations = recommendations.sort_values(
        by="similarity_score",
        ascending=False
    )

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
# 5. CALCULATE GENRE DIVERSITY
# ============================================================

def calculate_genre_diversity(recommendations):
    """
    Calculate unique genres in recommendations.
    """

    all_genres = []

    for genre_text in recommendations["genre_text"].dropna():
        genres = str(genre_text).split()

        for genre in genres:
            if genre != "unknown":
                all_genres.append(genre)

    unique_genres = sorted(set(all_genres))

    return len(unique_genres), unique_genres


# ============================================================
# 6. CALCULATE RECOMMENDATION STATISTICS
# ============================================================

def calculate_statistics(recommendations):
    """
    Calculate basic recommendation statistics.
    """

    if recommendations.empty:
        return {
            "recommendation_count": 0,
            "average_similarity": 0.0,
            "minimum_similarity": 0.0,
            "maximum_similarity": 0.0,
            "genre_diversity_count": 0,
            "unique_genres": ""
        }

    genre_count, unique_genres = (
        calculate_genre_diversity(recommendations)
    )

    statistics = {
        "recommendation_count": len(recommendations),
        "average_similarity": (
            recommendations["similarity_score"].mean()
        ),
        "minimum_similarity": (
            recommendations["similarity_score"].min()
        ),
        "maximum_similarity": (
            recommendations["similarity_score"].max()
        ),
        "genre_diversity_count": genre_count,
        "unique_genres": ", ".join(unique_genres)
    }

    return statistics


# ============================================================
# 7. PLOT SIMILARITY SCORES
# ============================================================

def create_similarity_plot(
    recommendations,
    movie_title
):
    """
    Create a bar chart of recommendation similarity scores.
    """

    if recommendations.empty:
        print(
            "No recommendations available for plotting."
        )
        return

    plot_data = recommendations.sort_values(
        by="similarity_score",
        ascending=True
    )

    plt.figure(figsize=(10, 6))

    plt.barh(
        plot_data["title"],
        plot_data["similarity_score"]
    )

    plt.xlabel("Cosine Similarity Score")
    plt.ylabel("Movie Title")
    plt.title(
        f"Similarity Scores for: {movie_title}"
    )

    plt.tight_layout()

    output_path = (
        PLOTS_DIR / "content_similarity_scores.png"
    )

    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"Similarity plot saved: {output_path}")


# ============================================================
# 8. EVALUATE TEST MOVIES
# ============================================================

def evaluate_test_movies(
    movies,
    tfidf_matrix,
    test_titles,
    top_k=10
):
    """
    Generate recommendations for multiple test movies.
    """

    all_recommendations = []
    summary_rows = []

    for movie_title in test_titles:
        print("\n" + "=" * 60)
        print(f"Evaluating: {movie_title}")
        print("=" * 60)

        movie_id = find_movie_id(
            movies,
            movie_title
        )

        if movie_id is None:
            print(
                f"Movie not found: {movie_title}"
            )
            continue

        recommendations = recommend_movies(
            movies=movies,
            tfidf_matrix=tfidf_matrix,
            movie_id=movie_id,
            top_k=top_k
        )

        if recommendations.empty:
            print("No recommendations found.")
            continue

        print("\nTop Recommendations:")

        for rank, (_, row) in enumerate(
            recommendations.iterrows(),
            start=1
        ):
            print(
                f"{rank}. {row['title']} | "
                f"Similarity: "
                f"{row['similarity_score']:.4f}"
            )

        statistics = calculate_statistics(
            recommendations
        )

        statistics["input_movie"] = movie_title
        statistics["input_movie_id"] = movie_id

        summary_rows.append(statistics)

        recommendations = recommendations.copy()
        recommendations["input_movie"] = movie_title
        recommendations["input_movie_id"] = movie_id
        recommendations["rank"] = range(
            1,
            len(recommendations) + 1
        )

        all_recommendations.append(
            recommendations
        )

        create_similarity_plot(
            recommendations,
            movie_title
        )

    if all_recommendations:
        final_recommendations = pd.concat(
            all_recommendations,
            ignore_index=True
        )
    else:
        final_recommendations = pd.DataFrame()

    summary_df = pd.DataFrame(summary_rows)

    return final_recommendations, summary_df


# ============================================================
# 9. CATALOG COVERAGE
# ============================================================

def calculate_catalog_coverage(
    recommendations,
    total_movies
):
    """
    Calculate catalog coverage based on unique
    recommended movie IDs.
    """

    if recommendations.empty or total_movies == 0:
        return 0.0

    unique_recommended_movies = (
        recommendations["movie_id"].nunique()
    )

    coverage = (
        unique_recommended_movies / total_movies
    )

    return coverage


# ============================================================
# 10. MAIN FUNCTION
# ============================================================

def main():
    print("=" * 60)
    print("DAY 9 - CONTENT-BASED EVALUATION")
    print("=" * 60)

    movies, tfidf_matrix = load_data()

    print(f"\nMovies loaded: {movies.shape}")
    print(
        f"TF-IDF matrix shape: "
        f"{tfidf_matrix.shape}"
    )

    test_titles = [
        "Toy Story",
        "GoldenEye",
        "Contact",
        "Fargo",
        "Aladdin"
    ]

    recommendations_df, summary_df = (
        evaluate_test_movies(
            movies=movies,
            tfidf_matrix=tfidf_matrix,
            test_titles=test_titles,
            top_k=10
        )
    )

    if not recommendations_df.empty:
        recommendations_df.to_csv(
            RECOMMENDATIONS_OUTPUT_PATH,
            index=False
        )

        print(
            f"\nRecommendations saved: "
            f"{RECOMMENDATIONS_OUTPUT_PATH}"
        )

    if not summary_df.empty:
        coverage = calculate_catalog_coverage(
            recommendations_df,
            total_movies=len(movies)
        )

        summary_df["catalog_coverage"] = coverage

        summary_df.to_csv(
            SUMMARY_OUTPUT_PATH,
            index=False
        )

        print(
            f"Summary saved: "
            f"{SUMMARY_OUTPUT_PATH}"
        )

        print("\nEvaluation Summary:")
        print(summary_df.to_string(index=False))

        print(
            f"\nCatalog Coverage: "
            f"{coverage:.4f}"
        )

    print("\nDay 9 script completed successfully.")


if __name__ == "__main__":
    main()