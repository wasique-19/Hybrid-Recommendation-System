
from pathlib import Path
import pandas as pd
import numpy as np


# --------------------------------------------------
# 1. Project Paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DATA_DIR = PROJECT_ROOT / "processed_data"

RATINGS_FILE = PROCESSED_DATA_DIR / "ratings_clean.csv"
MOVIES_FILE = PROCESSED_DATA_DIR / "movies_clean.csv"
USERS_FILE = PROCESSED_DATA_DIR / "users_clean.csv"


# --------------------------------------------------
# 2. Load Cleaned Data
# --------------------------------------------------

def load_data():
    print("Loading cleaned datasets...")

    ratings = pd.read_csv(RATINGS_FILE)
    movies = pd.read_csv(MOVIES_FILE)
    users = pd.read_csv(USERS_FILE)

    print(f"Ratings shape: {ratings.shape}")
    print(f"Movies shape: {movies.shape}")
    print(f"Users shape: {users.shape}")

    return ratings, movies, users


# --------------------------------------------------
# 3. Movie Feature Engineering
# --------------------------------------------------

def create_movie_features(movies):
    print("\nCreating movie features...")

    movie_features = movies.copy()

    # Ensure title is string
    movie_features["title"] = (
        movie_features["title"]
        .fillna("unknown")
        .astype(str)
        .str.strip()
    )

    # Extract title without year
    movie_features["clean_title"] = (
        movie_features["title"]
        .str.replace(r"\s*\(\d{4}\)\s*$", "", regex=True)
        .str.strip()
        .str.lower()
    )

    # Extract year from title if available
    title_year = movie_features["title"].str.extract(
        r"\((\d{4})\)\s*$"
    )[0]

    title_year = pd.to_numeric(title_year, errors="coerce")

    # Fill missing release year using title year
    movie_features["release_year"] = (
        movie_features["release_year"]
        .fillna(title_year)
    )

    # Genre columns available in MovieLens dataset
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

    # Keep only genre columns that exist
    available_genres = [
        genre for genre in genre_columns
        if genre in movie_features.columns
    ]

    # Convert genre indicator columns into genre text
    def build_genre_text(row):
        selected_genres = []

        for genre in available_genres:
            try:
                if int(row[genre]) == 1:
                    selected_genres.append(
                        genre.lower().replace("-", "_")
                    )
            except (ValueError, TypeError):
                continue

        if not selected_genres:
            return "unknown"

        return " ".join(selected_genres)

    movie_features["genre_text"] = movie_features.apply(
        build_genre_text,
        axis=1
    )

    # Number of genres for each movie
    movie_features["genre_count"] = 0

    for genre in available_genres:
        movie_features["genre_count"] += pd.to_numeric(
            movie_features[genre],
            errors="coerce"
        ).fillna(0)

    # Current reference year for age calculation
    reference_year = 1998

    movie_features["movie_age"] = (
        reference_year - movie_features["release_year"]
    )

    movie_features["movie_age"] = (
        movie_features["movie_age"]
        .clip(lower=0)
        .fillna(0)
    )

    # Select useful movie columns
    selected_columns = [
        "movie_id",
        "title",
        "clean_title",
        "release_year",
        "genre_text",
        "genre_count",
        "movie_age"
    ]

    movie_features = movie_features[
        [
            column for column in selected_columns
            if column in movie_features.columns
        ]
    ]

    movie_features = movie_features.drop_duplicates(
        subset=["movie_id"]
    )

    movie_features = movie_features.sort_values(
        by="movie_id"
    )

    movie_features = movie_features.reset_index(
        drop=True
    )

    print(f"Movie features shape: {movie_features.shape}")

    return movie_features


# --------------------------------------------------
# 4. User Feature Engineering
# --------------------------------------------------

def create_user_features(users):
    print("\nCreating user features...")

    user_features = users.copy()

    # Clean text columns
    user_features["gender"] = (
        user_features["gender"]
        .fillna("unknown")
        .astype(str)
        .str.strip()
        .str.upper()
    )

    user_features["occupation"] = (
        user_features["occupation"]
        .fillna("unknown")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    # Convert age into numeric values
    user_features["age"] = pd.to_numeric(
        user_features["age"],
        errors="coerce"
    )

    user_features["age"] = user_features["age"].fillna(
        user_features["age"].median()
    )

    # Create age groups
    def assign_age_group(age):
        if age < 18:
            return "teenager"
        elif age < 30:
            return "young_adult"
        elif age < 45:
            return "adult"
        elif age < 60:
            return "middle_aged"
        else:
            return "senior"

    user_features["age_group"] = (
        user_features["age"]
        .apply(assign_age_group)
    )

    # Age normalization
    user_features["age_normalized"] = (
        user_features["age"] / 100
    )

    selected_columns = [
        "user_id",
        "age",
        "gender",
        "occupation",
        "age_group",
        "age_normalized"
    ]

    user_features = user_features[
        [
            column for column in selected_columns
            if column in user_features.columns
        ]
    ]

    user_features = user_features.drop_duplicates(
        subset=["user_id"]
    )

    user_features = user_features.sort_values(
        by="user_id"
    )

    user_features = user_features.reset_index(
        drop=True
    )

    print(f"User features shape: {user_features.shape}")

    return user_features


# --------------------------------------------------
# 5. Rating Feature Engineering
# --------------------------------------------------

def create_rating_features(ratings):
    print("\nCreating rating features...")

    rating_features = ratings.copy()

    rating_features["rating"] = pd.to_numeric(
        rating_features["rating"],
        errors="coerce"
    )

    # Rating category
    def assign_rating_category(rating):
        if rating <= 2:
            return "negative"
        elif rating == 3:
            return "neutral"
        else:
            return "positive"

    rating_features["rating_category"] = (
        rating_features["rating"]
        .apply(assign_rating_category)
    )

    # Binary positive rating flag
    rating_features["positive_rating"] = (
        rating_features["rating"] >= 4
    ).astype(int)

    # Binary negative rating flag
    rating_features["negative_rating"] = (
        rating_features["rating"] <= 2
    ).astype(int)

    # Rating normalized to 0-1 scale
    rating_features["rating_normalized"] = (
        (rating_features["rating"] - 1) / 4
    )

    rating_features = rating_features.dropna(
        subset=["user_id", "movie_id", "rating"]
    )

    rating_features = rating_features.reset_index(
        drop=True
    )

    print(f"Rating features shape: {rating_features.shape}")

    return rating_features


# --------------------------------------------------
# 6. Save Feature-Engineered Data
# --------------------------------------------------

def save_features(
    movie_features,
    user_features,
    rating_features
):
    PROCESSED_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    movie_output = (
        PROCESSED_DATA_DIR / "movie_features.csv"
    )

    user_output = (
        PROCESSED_DATA_DIR / "user_features.csv"
    )

    rating_output = (
        PROCESSED_DATA_DIR / "rating_features.csv"
    )

    movie_features.to_csv(
        movie_output,
        index=False
    )

    user_features.to_csv(
        user_output,
        index=False
    )

    rating_features.to_csv(
        rating_output,
        index=False
    )

    print("\nFeature files saved successfully:")
    print(movie_output)
    print(user_output)
    print(rating_output)


# --------------------------------------------------
# 7. Display Sample Data
# --------------------------------------------------

def display_samples(
    movie_features,
    user_features,
    rating_features
):
    print("\nMovie Feature Sample:")
    print(movie_features.head(5).to_string(index=False))

    print("\nUser Feature Sample:")
    print(user_features.head(5).to_string(index=False))

    print("\nRating Feature Sample:")
    print(rating_features.head(5).to_string(index=False))


# --------------------------------------------------
# 8. Main Function
# --------------------------------------------------

def main():
    print("=" * 60)
    print("DAY 5 - FEATURE ENGINEERING")
    print("=" * 60)

    ratings, movies, users = load_data()

    movie_features = create_movie_features(movies)

    user_features = create_user_features(users)

    rating_features = create_rating_features(ratings)

    save_features(
        movie_features,
        user_features,
        rating_features
    )

    display_samples(
        movie_features,
        user_features,
        rating_features
    )

    print("\nDay 5 feature engineering completed successfully!")


if __name__ == "__main__":
    main()