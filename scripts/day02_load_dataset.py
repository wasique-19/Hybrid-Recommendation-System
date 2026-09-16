
from pathlib import Path
import pandas as pd


# ==========================================
# 1. Project and Dataset Paths
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset" / "ml-100k"

RATINGS_FILE = DATASET_DIR / "u.data"
MOVIES_FILE = DATASET_DIR / "u.item"
USERS_FILE = DATASET_DIR / "u.user"


# ==========================================
# 2. Check Dataset Files
# ==========================================

print("=" * 60)
print("MOVIELENS 100K DATASET LOADING")
print("=" * 60)

if not DATASET_DIR.exists():
    raise FileNotFoundError(
        f"Dataset directory not found: {DATASET_DIR}"
    )

required_files = [RATINGS_FILE, MOVIES_FILE, USERS_FILE]

for file_path in required_files:
    if not file_path.exists():
        raise FileNotFoundError(
            f"Required file not found: {file_path}"
        )

print("\nAll required dataset files found.")


# ==========================================
# 3. Load Ratings Data
# ==========================================

ratings_columns = [
    "user_id",
    "movie_id",
    "rating",
    "timestamp"
]

ratings_df = pd.read_csv(
    RATINGS_FILE,
    sep="\t",
    names=ratings_columns,
    encoding="latin-1"
)


# ==========================================
# 4. Load Movies Data
# ==========================================

movie_columns = [
    "movie_id",
    "title",
    "release_date",
    "video_release_date",
    "imdb_url",
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

movies_df = pd.read_csv(
    MOVIES_FILE,
    sep="|",
    names=movie_columns,
    encoding="latin-1"
)


# ==========================================
# 5. Load Users Data
# ==========================================

user_columns = [
    "user_id",
    "age",
    "gender",
    "occupation",
    "zip_code"
]

users_df = pd.read_csv(
    USERS_FILE,
    sep="|",
    names=user_columns,
    encoding="latin-1"
)


# ==========================================
# 6. Display Basic Information
# ==========================================

print("\n--- Ratings Data ---")
print(ratings_df.head())
print("Shape:", ratings_df.shape)

print("\n--- Movies Data ---")
print(movies_df[["movie_id", "title", "release_date"]].head())
print("Shape:", movies_df.shape)

print("\n--- Users Data ---")
print(users_df.head())
print("Shape:", users_df.shape)


# ==========================================
# 7. Dataset Statistics
# ==========================================

print("\n--- Dataset Statistics ---")

print("Total ratings:", len(ratings_df))
print("Total movies:", movies_df["movie_id"].nunique())
print("Total users:", users_df["user_id"].nunique())

print(
    "Minimum rating:",
    ratings_df["rating"].min()
)

print(
    "Maximum rating:",
    ratings_df["rating"].max()
)

print(
    "Average rating:",
    round(ratings_df["rating"].mean(), 2)
)


# ==========================================
# 8. Missing Values
# ==========================================

print("\n--- Missing Values ---")

print("Ratings missing values:")
print(ratings_df.isnull().sum())

print("\nMovies missing values:")
print(movies_df.isnull().sum())

print("\nUsers missing values:")
print(users_df.isnull().sum())


# ==========================================
# 9. Duplicate Records
# ==========================================

print("\n--- Duplicate Records ---")

print(
    "Duplicate ratings:",
    ratings_df.duplicated().sum()
)

print(
    "Duplicate movies:",
    movies_df.duplicated().sum()
)

print(
    "Duplicate users:",
    users_df.duplicated().sum()
)


# ==========================================
# 10. Completion Message
# ==========================================

print("\n" + "=" * 60)
print("DATASET LOADED SUCCESSFULLY")
print("=" * 60)