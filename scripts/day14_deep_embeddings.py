"""
Day 14 - Deep User and Movie Embeddings

Goal:
Build the foundation of a neural collaborative filtering model.

Architecture:

User ID
   ↓
User Embedding
   ↓
User Vector
        \
         Concatenate
        /
Movie Vector
   ↑
Movie Embedding
   ↑
Movie ID

Concatenated Vector
   ↓
Dense Layer
   ↓
ReLU
   ↓
Dropout
   ↓
Dense Layer
   ↓
Predicted Rating
"""

import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

RATINGS_PATH = os.path.join(
    BASE_DIR,
    "processed_data",
    "ratings_clean.csv"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "day14_deep_embedding_model.pth"
)

USER_MAP_PATH = os.path.join(
    BASE_DIR,
    "models",
    "day14_user_id_mapping.csv"
)

MOVIE_MAP_PATH = os.path.join(
    BASE_DIR,
    "models",
    "day14_movie_id_mapping.csv"
)


# Model settings
EMBEDDING_DIM = 32
HIDDEN_DIM = 64

BATCH_SIZE = 256
EPOCHS = 5

LEARNING_RATE = 0.001

TEST_SIZE = 0.10
RANDOM_SEED = 42


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# DATASET CLASS
# ============================================================

class MovieLensDataset(Dataset):

    def __init__(
        self,
        dataframe
    ):

        self.user_ids = torch.tensor(
            dataframe["user_index"].values,
            dtype=torch.long
        )

        self.movie_ids = torch.tensor(
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
            self.user_ids[index],
            self.movie_ids[index],
            self.ratings[index]
        )


# ============================================================
# DEEP EMBEDDING MODEL
# ============================================================

class DeepEmbeddingModel(nn.Module):

    def __init__(
        self,
        num_users,
        num_movies,
        embedding_dim=32,
        hidden_dim=64
    ):

        super().__init__()

        # User embedding
        self.user_embedding = nn.Embedding(
            num_users,
            embedding_dim
        )

        # Movie embedding
        self.movie_embedding = nn.Embedding(
            num_movies,
            embedding_dim
        )

        # Fully connected layers
        self.fc1 = nn.Linear(
            embedding_dim * 2,
            hidden_dim
        )

        self.relu = nn.ReLU()

        self.dropout = nn.Dropout(
            p=0.2
        )

        self.fc2 = nn.Linear(
            hidden_dim,
            32
        )

        self.fc3 = nn.Linear(
            32,
            1
        )

    def forward(
        self,
        user_ids,
        movie_ids
    ):

        # Get user embedding
        user_vector = self.user_embedding(
            user_ids
        )

        # Get movie embedding
        movie_vector = self.movie_embedding(
            movie_ids
        )

        # Concatenate both embeddings
        combined_vector = torch.cat(
            [
                user_vector,
                movie_vector
            ],
            dim=1
        )

        # Neural network
        x = self.fc1(
            combined_vector
        )

        x = self.relu(x)

        x = self.dropout(x)

        x = self.fc2(x)

        x = self.relu(x)

        output = self.fc3(x)

        # Convert [batch, 1]
        # into [batch]
        output = output.squeeze(1)

        return output


# ============================================================
# LOAD RATINGS
# ============================================================

def load_ratings():

    print("=" * 75)
    print("DAY 14 - DEEP USER & MOVIE EMBEDDINGS")
    print("=" * 75)

    ratings = pd.read_csv(
        RATINGS_PATH
    )

    print("\nRatings shape:")
    print(ratings.shape)

    print("\nFirst 5 rows:")
    print(
        ratings.head().to_string(
            index=False
        )
    )

    return ratings


# ============================================================
# CREATE ID MAPPINGS
# ============================================================

def create_id_mappings(
    ratings
):

    unique_users = sorted(
        ratings["user_id"].unique()
    )

    unique_movies = sorted(
        ratings["movie_id"].unique()
    )

    user_to_index = {
        user_id: index
        for index, user_id in enumerate(
            unique_users
        )
    }

    movie_to_index = {
        movie_id: index
        for index, movie_id in enumerate(
            unique_movies
        )
    }

    ratings = ratings.copy()

    ratings["user_index"] = (
        ratings["user_id"]
        .map(user_to_index)
    )

    ratings["movie_index"] = (
        ratings["movie_id"]
        .map(movie_to_index)
    )

    print("\nNumber of users:")
    print(len(user_to_index))

    print("\nNumber of movies:")
    print(len(movie_to_index))

    print("\nUser index range:")
    print(
        ratings["user_index"].min(),
        "to",
        ratings["user_index"].max()
    )

    print("\nMovie index range:")
    print(
        ratings["movie_index"].min(),
        "to",
        ratings["movie_index"].max()
    )

    return (
        ratings,
        user_to_index,
        movie_to_index
    )


# ============================================================
# SAVE MAPPINGS
# ============================================================

def save_mappings(
    user_to_index,
    movie_to_index
):

    user_mapping = pd.DataFrame(
        {
            "user_id":
                list(user_to_index.keys()),
            "user_index":
                list(user_to_index.values())
        }
    )

    movie_mapping = pd.DataFrame(
        {
            "movie_id":
                list(movie_to_index.keys()),
            "movie_index":
                list(movie_to_index.values())
        }
    )

    user_mapping.to_csv(
        USER_MAP_PATH,
        index=False
    )

    movie_mapping.to_csv(
        MOVIE_MAP_PATH,
        index=False
    )

    print("\nSaved user mapping:")
    print(USER_MAP_PATH)

    print("\nSaved movie mapping:")
    print(MOVIE_MAP_PATH)


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

def create_train_test_split(
    ratings
):

    shuffled = ratings.sample(
        frac=1.0,
        random_state=RANDOM_SEED
    ).reset_index(
        drop=True
    )

    test_count = int(
        len(shuffled) * TEST_SIZE
    )

    test_df = shuffled[
        :test_count
    ].copy()

    train_df = shuffled[
        test_count:
    ].copy()

    print("\nTrain samples:")
    print(len(train_df))

    print("\nTest samples:")
    print(len(test_df))

    return train_df, test_df


# ============================================================
# TRAIN ONE EPOCH
# ============================================================

def train_one_epoch(
    model,
    dataloader,
    optimizer,
    loss_function
):

    model.train()

    total_loss = 0.0

    for (
        user_ids,
        movie_ids,
        ratings
    ) in dataloader:

        user_ids = user_ids.to(
            DEVICE
        )

        movie_ids = movie_ids.to(
            DEVICE
        )

        ratings = ratings.to(
            DEVICE
        )

        # Reset gradients
        optimizer.zero_grad()

        # Prediction
        predictions = model(
            user_ids,
            movie_ids
        )

        # Loss
        loss = loss_function(
            predictions,
            ratings
        )

        # Backpropagation
        loss.backward()

        # Update parameters
        optimizer.step()

        total_loss += (
            loss.item()
            * len(ratings)
        )

    average_loss = (
        total_loss
        / len(dataloader.dataset)
    )

    return average_loss


# ============================================================
# EVALUATE
# ============================================================

def evaluate(
    model,
    dataloader,
    loss_function
):

    model.eval()

    total_loss = 0.0

    with torch.no_grad():

        for (
            user_ids,
            movie_ids,
            ratings
        ) in dataloader:

            user_ids = user_ids.to(
                DEVICE
            )

            movie_ids = movie_ids.to(
                DEVICE
            )

            ratings = ratings.to(
                DEVICE
            )

            predictions = model(
                user_ids,
                movie_ids
            )

            loss = loss_function(
                predictions,
                ratings
            )

            total_loss += (
                loss.item()
                * len(ratings)
            )

    average_loss = (
        total_loss
        / len(dataloader.dataset)
    )

    return average_loss


# ============================================================
# SAMPLE PREDICTIONS
# ============================================================

def show_sample_predictions(
    model,
    test_df,
    count=10
):

    model.eval()

    sample = test_df.head(
        count
    ).copy()

    user_tensor = torch.tensor(
        sample["user_index"].values,
        dtype=torch.long
    ).to(DEVICE)

    movie_tensor = torch.tensor(
        sample["movie_index"].values,
        dtype=torch.long
    ).to(DEVICE)

    with torch.no_grad():

        predictions = model(
            user_tensor,
            movie_tensor
        )

    sample[
        "predicted_rating"
    ] = predictions.cpu().numpy()

    print("\n" + "=" * 75)
    print("SAMPLE PREDICTIONS")
    print("=" * 75)

    print(
        sample[
            [
                "user_id",
                "movie_id",
                "rating",
                "predicted_rating"
            ]
        ].to_string(
            index=False
        )
    )


# ============================================================
# INSPECT EMBEDDINGS
# ============================================================

def inspect_embeddings(
    model
):

    model.eval()

    print("\n" + "=" * 75)
    print("EMBEDDING SHAPES")
    print("=" * 75)

    user_embeddings = (
        model.user_embedding.weight
    )

    movie_embeddings = (
        model.movie_embedding.weight
    )

    print(
        "User embedding matrix:",
        tuple(
            user_embeddings.shape
        )
    )

    print(
        "Movie embedding matrix:",
        tuple(
            movie_embeddings.shape
        )
    )

    print("\nFirst user embedding:")

    print(
        user_embeddings[0]
        .detach()
        .cpu()
        .numpy()
    )

    print("\nFirst movie embedding:")

    print(
        movie_embeddings[0]
        .detach()
        .cpu()
        .numpy()
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\nUsing device:", DEVICE)

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    ratings = load_ratings()

    # --------------------------------------------------------
    # Create mappings
    # --------------------------------------------------------

    (
        ratings,
        user_to_index,
        movie_to_index
    ) = create_id_mappings(
        ratings
    )

    save_mappings(
        user_to_index,
        movie_to_index
    )

    # --------------------------------------------------------
    # Train / Test split
    # --------------------------------------------------------

    (
        train_df,
        test_df
    ) = create_train_test_split(
        ratings
    )

    # --------------------------------------------------------
    # PyTorch datasets
    # --------------------------------------------------------

    train_dataset = (
        MovieLensDataset(
            train_df
        )
    )

    test_dataset = (
        MovieLensDataset(
            test_df
        )
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------

    num_users = len(
        user_to_index
    )

    num_movies = len(
        movie_to_index
    )

    model = DeepEmbeddingModel(
        num_users=num_users,
        num_movies=num_movies,
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM
    )

    model = model.to(
        DEVICE
    )

    print("\nModel:")
    print(model)

    # --------------------------------------------------------
    # Loss and optimizer
    # --------------------------------------------------------

    loss_function = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    print("\n" + "=" * 75)
    print("TRAINING")
    print("=" * 75)

    for epoch in range(
        EPOCHS
    ):

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            loss_function
        )

        test_loss = evaluate(
            model,
            test_loader,
            loss_function
        )

        train_rmse = np.sqrt(
            train_loss
        )

        test_rmse = np.sqrt(
            test_loss
        )

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train RMSE: {train_rmse:.4f} | "
            f"Test Loss: {test_loss:.4f} | "
            f"Test RMSE: {test_rmse:.4f}"
        )

    # --------------------------------------------------------
    # Inspect embeddings
    # --------------------------------------------------------

    inspect_embeddings(
        model
    )

    # --------------------------------------------------------
    # Sample predictions
    # --------------------------------------------------------

    show_sample_predictions(
        model,
        test_df,
        count=10
    )

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    torch.save(
        {
            "model_state_dict":
                model.state_dict(),

            "num_users":
                num_users,

            "num_movies":
                num_movies,

            "embedding_dim":
                EMBEDDING_DIM,

            "hidden_dim":
                HIDDEN_DIM
        },
        MODEL_PATH
    )

    print("\n" + "=" * 75)
    print("MODEL SAVED")
    print("=" * 75)

    print(MODEL_PATH)

    print("\n" + "=" * 75)
    print("DAY 14 COMPLETED SUCCESSFULLY")
    print("=" * 75)


if __name__ == "__main__":
    main()