
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# ==========================================
# 1. Project Paths
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PROCESSED_DATA_DIR = PROJECT_ROOT / "processed_data"
PLOTS_DIR = PROJECT_ROOT / "evaluation" / "plots"

PLOTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ==========================================
# 2. Load Cleaned Data
# ==========================================

ratings_file = PROCESSED_DATA_DIR / "ratings_clean.csv"
movies_file = PROCESSED_DATA_DIR / "movies_clean.csv"
users_file = PROCESSED_DATA_DIR / "users_clean.csv"

ratings_df = pd.read_csv(ratings_file)
movies_df = pd.read_csv(movies_file)
users_df = pd.read_csv(users_file)

print("=" * 60)
print("DAY 4 - EXPLORATORY DATA ANALYSIS")
print("=" * 60)

print("\nData Loaded Successfully")

print("Ratings shape:", ratings_df.shape)
print("Movies shape:", movies_df.shape)
print("Users shape:", users_df.shape)


# ==========================================
# 3. Rating Distribution
# ==========================================

rating_counts = (
    ratings_df["rating"]
    .value_counts()
    .sort_index()
)

print("\n--- Rating Distribution ---")
print(rating_counts)

plt.figure(figsize=(8, 5))

rating_counts.plot(
    kind="bar"
)

plt.title("Movie Rating Distribution")
plt.xlabel("Rating")
plt.ylabel("Number of Ratings")
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig(
    PLOTS_DIR / "rating_distribution.png",
    dpi=150
)

plt.close()


# ==========================================
# 4. User Activity Analysis
# ==========================================

user_activity = (
    ratings_df
    .groupby("user_id")
    .size()
    .sort_values(ascending=False)
)

print("\n--- User Activity ---")

print("Most active users:")
print(user_activity.head(10))

print(
    "Average ratings per user:",
    round(user_activity.mean(), 2)
)

print(
    "Maximum ratings by one user:",
    user_activity.max()
)

plt.figure(figsize=(8, 5))

plt.hist(
    user_activity,
    bins=30
)

plt.title("Distribution of Ratings per User")
plt.xlabel("Number of Ratings")
plt.ylabel("Number of Users")
plt.tight_layout()

plt.savefig(
    PLOTS_DIR / "user_activity_distribution.png",
    dpi=150
)

plt.close()


# ==========================================
# 5. Movie Popularity
# ==========================================

movie_rating_counts = (
    ratings_df
    .groupby("movie_id")
    .size()
    .reset_index(name="rating_count")
)

popular_movies = (
    movie_rating_counts
    .merge(
        movies_df[["movie_id", "title"]],
        on="movie_id",
        how="left"
    )
    .sort_values(
        "rating_count",
        ascending=False
    )
    .head(10)
)

print("\n--- Top 10 Most-Rated Movies ---")
print(popular_movies)


plt.figure(figsize=(10, 6))

popular_movies_sorted = popular_movies.sort_values(
    "rating_count"
)

plt.barh(
    popular_movies_sorted["title"],
    popular_movies_sorted["rating_count"]
)

plt.title("Top 10 Most-Rated Movies")
plt.xlabel("Number of Ratings")
plt.ylabel("Movie Title")
plt.tight_layout()

plt.savefig(
    PLOTS_DIR / "top_10_popular_movies.png",
    dpi=150
)

plt.close()


# ==========================================
# 6. Average Rating by Movie
# ==========================================

movie_average_ratings = (
    ratings_df
    .groupby("movie_id")["rating"]
    .agg(["mean", "count"])
    .reset_index()
)

movie_average_ratings = movie_average_ratings.rename(
    columns={
        "mean": "average_rating",
        "count": "rating_count"
    }
)

# Only consider movies with at least 50 ratings.
reliable_movies = movie_average_ratings[
    movie_average_ratings["rating_count"] >= 50
].copy()

top_rated_movies = (
    reliable_movies
    .merge(
        movies_df[["movie_id", "title"]],
        on="movie_id",
        how="left"
    )
    .sort_values(
        "average_rating",
        ascending=False
    )
    .head(10)
)

print("\n--- Top-Rated Movies with at Least 50 Ratings ---")
print(
    top_rated_movies[
        ["movie_id", "title", "average_rating", "rating_count"]
    ]
)


# ==========================================
# 7. Genre Distribution
# ==========================================

genre_columns = [
    "unknown",
    "action",
    "adventure",
    "animation",
    "children",
    "comedy",
    "crime",
    "documentary",
    "drama",
    "fantasy",
    "film_noir",
    "horror",
    "musical",
    "mystery",
    "romance",
    "sci_fi",
    "thriller",
    "war",
    "western"
]

available_genres = [
    column
    for column in genre_columns
    if column in movies_df.columns
]

genre_counts = movies_df[available_genres].sum().sort_values(
    ascending=False
)

print("\n--- Genre Distribution ---")
print(genre_counts)


plt.figure(figsize=(10, 6))

genre_counts.sort_values().plot(
    kind="barh"
)

plt.title("Movie Genre Distribution")
plt.xlabel("Number of Movies")
plt.ylabel("Genre")
plt.tight_layout()

plt.savefig(
    PLOTS_DIR / "genre_distribution.png",
    dpi=150
)

plt.close()


# ==========================================
# 8. Gender Distribution
# ==========================================

gender_counts = users_df["gender"].value_counts()

print("\n--- User Gender Distribution ---")
print(gender_counts)


plt.figure(figsize=(6, 5))

gender_counts.plot(
    kind="bar"
)

plt.title("User Gender Distribution")
plt.xlabel("Gender")
plt.ylabel("Number of Users")
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig(
    PLOTS_DIR / "gender_distribution.png",
    dpi=150
)

plt.close()


# ==========================================
# 9. Summary Statistics
# ==========================================

summary = {
    "total_users": users_df["user_id"].nunique(),
    "total_movies": movies_df["movie_id"].nunique(),
    "total_ratings": len(ratings_df),
    "average_rating": ratings_df["rating"].mean(),
    "minimum_rating": ratings_df["rating"].min(),
    "maximum_rating": ratings_df["rating"].max(),
    "average_ratings_per_user": user_activity.mean(),
    "average_ratings_per_movie": (
        ratings_df.groupby("movie_id").size().mean()
    )
}

summary_df = pd.DataFrame(
    [summary]
)

summary_df.to_csv(
    PROJECT_ROOT / "evaluation" / "eda_summary.csv",
    index=False
)

print("\n--- EDA Summary ---")
print(summary_df.T)


# ==========================================
# 10. Completion Message
# ==========================================

print("\nGenerated Plot Files:")

for plot_file in sorted(PLOTS_DIR.glob("*.png")):
    print(plot_file.name)

print("\n" + "=" * 60)
print("EDA COMPLETED SUCCESSFULLY")
print("=" * 60)