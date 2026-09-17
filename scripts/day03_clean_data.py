
from pathlib import Path
import pandas as pd


# ==========================================
# 1. Project Paths
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DATA_DIR = PROJECT_ROOT / "dataset" / "ml-100k"
PROCESSED_DATA_DIR = PROJECT_ROOT / "processed_data"

PROCESSED_DATA_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ==========================================
# 2. File Paths
# ==========================================

RATINGS_FILE = RAW_DATA_DIR / "u.data"
MOVIES_FILE = RAW_DATA_DIR / "u.item"
USERS_FILE = RAW_DATA_DIR / "u.user"


# ==========================================
# 3. Load Ratings
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
# 4. Load Movies
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
# 5. Load Users
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
# 6. Initial Information
# ==========================================

print("=" * 60)
print("DAY 3 - DATA CLEANING AND PREPROCESSING")
print("=" * 60)

print("\nInitial Shapes:")
print("Ratings:", ratings_df.shape)
print("Movies:", movies_df.shape)
print("Users:", users_df.shape)


# ==========================================
# 7. Convert Data Types
# ==========================================

ratings_df["user_id"] = pd.to_numeric(
    ratings_df["user_id"],
    errors="coerce"
)

ratings_df["movie_id"] = pd.to_numeric(
    ratings_df["movie_id"],
    errors="coerce"
)

ratings_df["rating"] = pd.to_numeric(
    ratings_df["rating"],
    errors="coerce"
)

movies_df["movie_id"] = pd.to_numeric(
    movies_df["movie_id"],
    errors="coerce"
)

users_df["user_id"] = pd.to_numeric(
    users_df["user_id"],
    errors="coerce"
)


# ==========================================
# 8. Handle Missing Values
# ==========================================

print("\nMissing Values Before Cleaning:")

print("\nRatings:")
print(ratings_df.isnull().sum())

print("\nMovies:")
print(movies_df.isnull().sum())

print("\nUsers:")
print(users_df.isnull().sum())


# Ratings with missing essential fields are removed.
ratings_df = ratings_df.dropna(
    subset=["user_id", "movie_id", "rating"]
).copy()


# Movies must have a title.
movies_df = movies_df.dropna(
    subset=["movie_id", "title"]
).copy()


# Users must have a user ID.
users_df = users_df.dropna(
    subset=["user_id"]
).copy()


# ==========================================
# 9. Validate Ratings
# ==========================================

invalid_ratings = ratings_df[
    ~ratings_df["rating"].between(1, 5)
]

print(
    "\nInvalid Ratings Before Removal:",
    len(invalid_ratings)
)

ratings_df = ratings_df[
    ratings_df["rating"].between(1, 5)
].copy()


# ==========================================
# 10. Remove Duplicate Records
# ==========================================

print("\nDuplicate Records Before Cleaning:")
print("Ratings:", ratings_df.duplicated().sum())
print("Movies:", movies_df.duplicated().sum())
print("Users:", users_df.duplicated().sum())

ratings_df = ratings_df.drop_duplicates().copy()
movies_df = movies_df.drop_duplicates(
    subset=["movie_id"]
).copy()
users_df = users_df.drop_duplicates(
    subset=["user_id"]
).copy()


# ==========================================
# 11. Process Movie Release Date
# ==========================================

movies_df["release_date"] = pd.to_datetime(
    movies_df["release_date"],
    format="%d-%b-%Y",
    errors="coerce"
)

movies_df["release_year"] = (
    movies_df["release_date"].dt.year
)


# ==========================================
# 12. Remove Unnecessary Movie Columns
# ==========================================

movies_df = movies_df.drop(
    columns=["video_release_date"],
    errors="ignore"
)


# ==========================================
# 13. Clean Text Fields
# ==========================================

movies_df["title"] = (
    movies_df["title"]
    .astype(str)
    .str.strip()
)

users_df["gender"] = (
    users_df["gender"]
    .astype(str)
    .str.strip()
)

users_df["occupation"] = (
    users_df["occupation"]
    .astype(str)
    .str.strip()
)

users_df["zip_code"] = (
    users_df["zip_code"]
    .astype(str)
    .str.strip()
)


# ==========================================
# 14. Reset Indexes
# ==========================================

ratings_df = ratings_df.reset_index(drop=True)
movies_df = movies_df.reset_index(drop=True)
users_df = users_df.reset_index(drop=True)


# ==========================================
# 15. Save Processed Data
# ==========================================

ratings_output = PROCESSED_DATA_DIR / "ratings_clean.csv"
movies_output = PROCESSED_DATA_DIR / "movies_clean.csv"
users_output = PROCESSED_DATA_DIR / "users_clean.csv"

ratings_df.to_csv(
    ratings_output,
    index=False
)

movies_df.to_csv(
    movies_output,
    index=False
)

users_df.to_csv(
    users_output,
    index=False
)


# ==========================================
# 16. Final Validation
# ==========================================

print("\nFinal Shapes:")
print("Ratings:", ratings_df.shape)
print("Movies:", movies_df.shape)
print("Users:", users_df.shape)

print("\nFinal Missing Values:")
print("Ratings:", ratings_df.isnull().sum().sum())
print("Movies:", movies_df.isnull().sum().sum())
print("Users:", users_df.isnull().sum().sum())

print("\nSaved Files:")
print(ratings_output)
print(movies_output)
print(users_output)

print("\n" + "=" * 60)
print("DATA CLEANING COMPLETED SUCCESSFULLY")
print("=" * 60)