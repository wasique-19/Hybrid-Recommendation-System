import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from scipy.sparse import load_npz
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch.utils.data import Dataset, DataLoader


# ============================================================
# DAY 15 - DEEP EMBEDDING FUSION
# ============================================================

print("=" * 75)
print("DAY 15 - DEEP EMBEDDING FUSION")
print("=" * 75)


# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PROCESSED_DIR = os.path.join(BASE_DIR, "processed_data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
EVALUATION_DIR = os.path.join(BASE_DIR, "evaluation")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(EVALUATION_DIR, exist_ok=True)

RANDOM_SEED = 42

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"\nUsing device: {DEVICE}")


# ============================================================
# 2. LOAD DATA
# ============================================================

print("\n" + "=" * 75)
print("LOADING DATA")
print("=" * 75)

ratings_path = os.path.join(
    PROCESSED_DIR,
    "ratings_clean.csv"
)

movies_path = os.path.join(
    PROCESSED_DIR,
    "movie_features.csv"
)

tfidf_path = os.path.join(
    PROCESSED_DIR,
    "movie_tfidf_matrix.npz"
)

ratings = pd.read_csv(ratings_path)
movies = pd.read_csv(movies_path)

tfidf_matrix = load_npz(tfidf_path)

print("\nRatings shape:")
print(ratings.shape)

print("\nMovies shape:")
print(movies.shape)

print("\nTF-IDF shape:")
print(tfidf_matrix.shape)


# ============================================================
# 3. CREATE USER AND MOVIE INDEX MAPPINGS
# ============================================================

print("\n" + "=" * 75)
print("CREATING INDEX MAPPINGS")
print("=" * 75)

unique_users = sorted(ratings["user_id"].unique())
unique_movies = sorted(ratings["movie_id"].unique())

user_to_index = {
    user_id: index
    for index, user_id in enumerate(unique_users)
}

movie_to_index = {
    movie_id: index
    for index, movie_id in enumerate(unique_movies)
}

ratings["user_index"] = ratings["user_id"].map(user_to_index)
ratings["movie_index"] = ratings["movie_id"].map(movie_to_index)

print(f"\nNumber of users: {len(unique_users)}")
print(f"Number of movies: {len(unique_movies)}")


# ============================================================
# 4. CREATE CONTENT EMBEDDINGS
# ============================================================

print("\n" + "=" * 75)
print("CREATING CONTENT EMBEDDINGS")
print("=" * 75)

# TF-IDF contains thousands of dimensions.
# We reduce it to a compact 32-dimensional representation.

CONTENT_DIM = 32

svd = TruncatedSVD(
    n_components=CONTENT_DIM,
    random_state=RANDOM_SEED
)

content_embeddings = svd.fit_transform(tfidf_matrix)

print("\nOriginal TF-IDF dimensions:")
print(tfidf_matrix.shape[1])

print("\nReduced content embedding shape:")
print(content_embeddings.shape)

print(
    f"\nExplained variance ratio: "
    f"{svd.explained_variance_ratio_.sum():.4f}"
)


# Save content embeddings

content_embedding_path = os.path.join(
    MODEL_DIR,
    "day15_content_embeddings.npy"
)

np.save(
    content_embedding_path,
    content_embeddings
)

print(
    f"\nSaved content embeddings:\n"
    f"{content_embedding_path}"
)


# ============================================================
# 5. ALIGN CONTENT EMBEDDINGS WITH MOVIE INDEX
# ============================================================

print("\nAligning movie content embeddings...")

movie_content_matrix = np.zeros(
    (len(unique_movies), CONTENT_DIM),
    dtype=np.float32
)

movie_id_to_movie_row = {
    movie_id: index
    for index, movie_id in enumerate(movies["movie_id"])
}

for movie_id, movie_index in movie_to_index.items():

    if movie_id in movie_id_to_movie_row:

        source_index = movie_id_to_movie_row[movie_id]

        movie_content_matrix[movie_index] = (
            content_embeddings[source_index]
        )

print(
    "Movie content matrix shape:",
    movie_content_matrix.shape
)


# ============================================================
# 6. TRAIN / TEST SPLIT
# ============================================================

print("\n" + "=" * 75)
print("TRAIN / TEST SPLIT")
print("=" * 75)

ratings = ratings.sample(
    frac=1,
    random_state=RANDOM_SEED
).reset_index(drop=True)

split_index = int(len(ratings) * 0.90)

train_df = ratings.iloc[:split_index].copy()
test_df = ratings.iloc[split_index:].copy()

print("\nTrain samples:")
print(len(train_df))

print("\nTest samples:")
print(len(test_df))


# ============================================================
# 7. PYTORCH DATASET
# ============================================================

class MovieRatingDataset(Dataset):

    def __init__(self, dataframe):

        self.users = torch.tensor(
            dataframe["user_index"].values,
            dtype=torch.long
        )

        self.movies = torch.tensor(
            dataframe["movie_index"].values,
            dtype=torch.long
        )

        self.ratings = torch.tensor(
            dataframe["rating"].values,
            dtype=torch.float32
        )

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, index):

        return (
            self.users[index],
            self.movies[index],
            self.ratings[index]
        )


train_dataset = MovieRatingDataset(train_df)
test_dataset = MovieRatingDataset(test_df)

train_loader = DataLoader(
    train_dataset,
    batch_size=512,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=512,
    shuffle=False
)


# ============================================================
# 8. DEEP EMBEDDING FUSION MODEL
# ============================================================

class DeepEmbeddingFusionModel(nn.Module):

    def __init__(
        self,
        num_users,
        num_movies,
        embedding_dim=32,
        content_dim=32
    ):

        super().__init__()

        # Collaborative user embedding
        self.user_embedding = nn.Embedding(
            num_users,
            embedding_dim
        )

        # Collaborative movie embedding
        self.movie_embedding = nn.Embedding(
            num_movies,
            embedding_dim
        )

        # Content projection layer
        self.content_projection = nn.Sequential(

            nn.Linear(
                content_dim,
                32
            ),

            nn.ReLU(),

            nn.Linear(
                32,
                32
            )
        )

        # Fusion network
        fusion_input_dim = (
            embedding_dim
            + embedding_dim
            + 32
        )

        self.fusion_network = nn.Sequential(

            nn.Linear(
                fusion_input_dim,
                128
            ),

            nn.ReLU(),

            nn.Dropout(0.2),

            nn.Linear(
                128,
                64
            ),

            nn.ReLU(),

            nn.Dropout(0.2),

            nn.Linear(
                64,
                32
            ),

            nn.ReLU(),

            nn.Linear(
                32,
                1
            )
        )

    def forward(
        self,
        user_ids,
        movie_ids,
        content_features
    ):

        user_vector = self.user_embedding(
            user_ids
        )

        movie_vector = self.movie_embedding(
            movie_ids
        )

        content_vector = self.content_projection(
            content_features
        )

        fused_vector = torch.cat(
            [
                user_vector,
                movie_vector,
                content_vector
            ],
            dim=1
        )

        prediction = self.fusion_network(
            fused_vector
        )

        return prediction.squeeze(1)


# ============================================================
# 9. CREATE MODEL
# ============================================================

model = DeepEmbeddingFusionModel(
    num_users=len(unique_users),
    num_movies=len(unique_movies),
    embedding_dim=32,
    content_dim=CONTENT_DIM
)

model = model.to(DEVICE)

print("\n" + "=" * 75)
print("MODEL ARCHITECTURE")
print("=" * 75)

print(model)


# ============================================================
# 10. OPTIMIZER AND LOSS
# ============================================================

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,
    weight_decay=1e-5
)


# ============================================================
# 11. CONTENT TENSOR
# ============================================================

content_tensor = torch.tensor(
    movie_content_matrix,
    dtype=torch.float32
).to(DEVICE)


# ============================================================
# 12. TRAINING FUNCTION
# ============================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion
):

    model.train()

    total_loss = 0.0

    for (
        user_ids,
        movie_ids,
        ratings_batch
    ) in loader:

        user_ids = user_ids.to(DEVICE)

        movie_ids = movie_ids.to(DEVICE)

        ratings_batch = ratings_batch.to(DEVICE)

        content_features = content_tensor[
            movie_ids
        ]

        optimizer.zero_grad()

        predictions = model(
            user_ids,
            movie_ids,
            content_features
        )

        loss = criterion(
            predictions,
            ratings_batch
        )

        loss.backward()

        optimizer.step()

        total_loss += (
            loss.item()
            * len(ratings_batch)
        )

    return total_loss / len(loader.dataset)


# ============================================================
# 13. EVALUATION FUNCTION
# ============================================================

def evaluate(
    model,
    loader
):

    model.eval()

    predictions = []
    actuals = []

    with torch.no_grad():

        for (
            user_ids,
            movie_ids,
            ratings_batch
        ) in loader:

            user_ids = user_ids.to(DEVICE)

            movie_ids = movie_ids.to(DEVICE)

            content_features = content_tensor[
                movie_ids
            ]

            outputs = model(
                user_ids,
                movie_ids,
                content_features
            )

            predictions.extend(
                outputs.cpu().numpy()
            )

            actuals.extend(
                ratings_batch.numpy()
            )

    predictions = np.array(predictions)

    actuals = np.array(actuals)

    predictions = np.clip(
        predictions,
        1,
        5
    )

    rmse = np.sqrt(
        mean_squared_error(
            actuals,
            predictions
        )
    )

    mae = mean_absolute_error(
        actuals,
        predictions
    )

    return rmse, mae, predictions, actuals


# ============================================================
# 14. TRAIN MODEL
# ============================================================

print("\n" + "=" * 75)
print("TRAINING DEEP EMBEDDING FUSION MODEL")
print("=" * 75)

EPOCHS = 8

history = []

for epoch in range(EPOCHS):

    train_loss = train_one_epoch(
        model,
        train_loader,
        optimizer,
        criterion
    )

    rmse, mae, _, _ = evaluate(
        model,
        test_loader
    )

    history.append(
        {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "test_rmse": rmse,
            "test_mae": mae
        }
    )

    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Test RMSE: {rmse:.4f} | "
        f"Test MAE: {mae:.4f}"
    )


# ============================================================
# 15. SAVE TRAINING HISTORY
# ============================================================

history_df = pd.DataFrame(history)

history_path = os.path.join(
    EVALUATION_DIR,
    "day15_training_history.csv"
)

history_df.to_csv(
    history_path,
    index=False
)

print(
    f"\nTraining history saved:\n"
    f"{history_path}"
)


# ============================================================
# 16. FINAL EVALUATION
# ============================================================

print("\n" + "=" * 75)
print("FINAL MODEL EVALUATION")
print("=" * 75)

rmse, mae, predictions, actuals = evaluate(
    model,
    test_loader
)

print(f"\nRMSE: {rmse:.4f}")
print(f"MAE:  {mae:.4f}")


# ============================================================
# 17. SAMPLE PREDICTIONS
# ============================================================

print("\n" + "=" * 75)
print("SAMPLE PREDICTIONS")
print("=" * 75)

sample_count = min(
    10,
    len(test_df)
)

sample_df = test_df.iloc[
    :sample_count
].copy()

sample_df["predicted_rating"] = (
    predictions[:sample_count]
)

print(
    sample_df[
        [
            "user_id",
            "movie_id",
            "rating",
            "predicted_rating"
        ]
    ].to_string(index=False)
)


# ============================================================
# 18. EXTRACT LEARNED EMBEDDINGS
# ============================================================

print("\n" + "=" * 75)
print("EXTRACTING LEARNED EMBEDDINGS")
print("=" * 75)

model.eval()

user_embeddings = (
    model.user_embedding.weight
    .detach()
    .cpu()
    .numpy()
)

movie_embeddings = (
    model.movie_embedding.weight
    .detach()
    .cpu()
    .numpy()
)

print(
    "\nUser embedding matrix:",
    user_embeddings.shape
)

print(
    "Movie embedding matrix:",
    movie_embeddings.shape
)


# ============================================================
# 19. SAVE EMBEDDINGS
# ============================================================

user_embedding_path = os.path.join(
    MODEL_DIR,
    "day15_user_embeddings.npy"
)

movie_embedding_path = os.path.join(
    MODEL_DIR,
    "day15_movie_embeddings.npy"
)

np.save(
    user_embedding_path,
    user_embeddings
)

np.save(
    movie_embedding_path,
    movie_embeddings
)

print(
    f"\nSaved user embeddings:\n"
    f"{user_embedding_path}"
)

print(
    f"\nSaved movie embeddings:\n"
    f"{movie_embedding_path}"
)


# ============================================================
# 20. SAVE MODEL
# ============================================================

model_path = os.path.join(
    MODEL_DIR,
    "day15_deep_embedding_fusion_model.pth"
)

torch.save(
    {
        "model_state_dict": model.state_dict(),

        "num_users": len(unique_users),

        "num_movies": len(unique_movies),

        "embedding_dim": 32,

        "content_dim": CONTENT_DIM,

        "user_to_index": user_to_index,

        "movie_to_index": movie_to_index
    },
    model_path
)

print(
    f"\nModel saved:\n"
    f"{model_path}"
)


# ============================================================
# 21. SAVE EVALUATION RESULTS
# ============================================================

evaluation_results = pd.DataFrame(
    [
        {
            "model": "Deep Embedding Fusion",
            "RMSE": rmse,
            "MAE": mae,
            "embedding_dimension": 32,
            "content_dimension": CONTENT_DIM
        }
    ]
)

evaluation_path = os.path.join(
    EVALUATION_DIR,
    "day15_deep_embedding_fusion_results.csv"
)

evaluation_results.to_csv(
    evaluation_path,
    index=False
)

print(
    f"\nEvaluation results saved:\n"
    f"{evaluation_path}"
)


# ============================================================
# 22. COMPLETION MESSAGE
# ============================================================

print("\n" + "=" * 75)
print("DAY 15 COMPLETED SUCCESSFULLY")
print("=" * 75)