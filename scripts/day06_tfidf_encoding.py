
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from scipy.sparse import save_npz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer


# --------------------------------------------------
# 1. Project Paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DATA_DIR = PROJECT_ROOT / "processed_data"
MODELS_DIR = PROJECT_ROOT / "models"

MOVIE_FEATURES_FILE = (
    PROCESSED_DATA_DIR / "movie_features.csv"
)

TFIDF_MATRIX_FILE = (
    PROCESSED_DATA_DIR / "movie_tfidf_matrix.npz"
)

TFIDF_NAMES_FILE = (
    PROCESSED_DATA_DIR / "tfidf_feature_names.csv"
)

GENRE_ENCODED_FILE = (
    PROCESSED_DATA_DIR / "genre_encoded.csv"
)

VECTORIZER_FILE = (
    MODELS_DIR / "tfidf_vectorizer.joblib"
)


# --------------------------------------------------
# 2. Load Movie Features
# --------------------------------------------------

def load_movie_features():
    print("Loading movie features...")

    movie_features = pd.read_csv(
        MOVIE_FEATURES_FILE
    )

    print(
        f"Movie features shape: "
        f"{movie_features.shape}"
    )

    return movie_features


# --------------------------------------------------
# 3. Prepare Text Data
# --------------------------------------------------

def prepare_text_data(movie_features):
    print("\nPreparing movie text data...")

    movie_features = movie_features.copy()

    movie_features["clean_title"] = (
        movie_features["clean_title"]
        .fillna("unknown")
        .astype(str)
        .str.lower()
        .str.strip()
    )

    movie_features["genre_text"] = (
        movie_features["genre_text"]
        .fillna("unknown")
        .astype(str)
        .str.lower()
        .str.strip()
    )

    # Combine title and genre information
    movie_features["combined_text"] = (
        movie_features["clean_title"]
        + " "
        + movie_features["genre_text"]
    )

    movie_features["combined_text"] = (
        movie_features["combined_text"]
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    print("\nCombined text sample:")

    print(
        movie_features[
            [
                "title",
                "combined_text"
            ]
        ]
        .head(10)
        .to_string(index=False)
    )

    return movie_features


# --------------------------------------------------
# 4. TF-IDF Vectorization
# --------------------------------------------------

def create_tfidf_matrix(movie_features):
    print("\nCreating TF-IDF matrix...")

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95
    )

    tfidf_matrix = vectorizer.fit_transform(
        movie_features["combined_text"]
    )

    feature_names = vectorizer.get_feature_names_out()

    print(
        f"TF-IDF matrix shape: "
        f"{tfidf_matrix.shape}"
    )

    print(
        f"Total TF-IDF features: "
        f"{len(feature_names)}"
    )

    return vectorizer, tfidf_matrix, feature_names


# --------------------------------------------------
# 5. Save TF-IDF Data
# --------------------------------------------------

def save_tfidf_data(
    vectorizer,
    tfidf_matrix,
    feature_names
):
    PROCESSED_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    save_npz(
        TFIDF_MATRIX_FILE,
        tfidf_matrix
    )

    joblib.dump(
        vectorizer,
        VECTORIZER_FILE
    )

    feature_names_df = pd.DataFrame(
        {
            "feature_index": np.arange(
                len(feature_names)
            ),
            "feature_name": feature_names
        }
    )

    feature_names_df.to_csv(
        TFIDF_NAMES_FILE,
        index=False
    )

    print("\nTF-IDF files saved:")
    print(TFIDF_MATRIX_FILE)
    print(VECTORIZER_FILE)
    print(TFIDF_NAMES_FILE)


# --------------------------------------------------
# 6. Genre Multi-Hot Encoding
# --------------------------------------------------

def create_genre_encoding(movie_features):
    print("\nCreating genre multi-hot encoding...")

    genre_lists = (
        movie_features["genre_text"]
        .fillna("unknown")
        .astype(str)
        .str.split()
    )

    multilabel_encoder = MultiLabelBinarizer()

    genre_matrix = multilabel_encoder.fit_transform(
        genre_lists
    )

    genre_columns = (
        multilabel_encoder.classes_
    )

    genre_encoded = pd.DataFrame(
        genre_matrix,
        columns=genre_columns
    )

    genre_encoded.insert(
        0,
        "movie_id",
        movie_features["movie_id"].values
    )

    genre_encoded.to_csv(
        GENRE_ENCODED_FILE,
        index=False
    )

    print(
        f"Genre encoded shape: "
        f"{genre_encoded.shape}"
    )

    print("\nGenre columns:")
    print(list(genre_columns))

    print("\nGenre encoding sample:")
    print(
        genre_encoded
        .head(5)
        .to_string(index=False)
    )

    return genre_encoded


# --------------------------------------------------
# 7. Display TF-IDF Sample
# --------------------------------------------------

def display_tfidf_sample(
    tfidf_matrix,
    feature_names
):
    print("\nTF-IDF sample information:")

    sample_vector = tfidf_matrix[0]

    nonzero_indices = (
        sample_vector.nonzero()[1]
    )

    print(
        f"Non-zero values in first movie: "
        f"{len(nonzero_indices)}"
    )

    print("\nFirst movie's important features:")

    for index in nonzero_indices[:10]:
        value = sample_vector[0, index]

        print(
            f"{feature_names[index]}: "
            f"{value:.4f}"
        )


# --------------------------------------------------
# 8. Main Function
# --------------------------------------------------

def main():
    print("=" * 60)
    print("DAY 6 - TF-IDF & TEXT ENCODING")
    print("=" * 60)

    movie_features = load_movie_features()

    movie_features = prepare_text_data(
        movie_features
    )

    (
        vectorizer,
        tfidf_matrix,
        feature_names
    ) = create_tfidf_matrix(
        movie_features
    )

    save_tfidf_data(
        vectorizer,
        tfidf_matrix,
        feature_names
    )

    create_genre_encoding(
        movie_features
    )

    display_tfidf_sample(
        tfidf_matrix,
        feature_names
    )

    print("\nDay 6 completed successfully!")


if __name__ == "__main__":
    main()