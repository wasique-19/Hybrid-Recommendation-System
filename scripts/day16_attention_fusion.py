import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_SEED = 42

EMBEDDING_DIM = 32
HIDDEN_DIM = 64

BATCH_SIZE = 256
EPOCHS = 8
LEARNING_RATE = 0.001

DROPOUT = 0.2

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

PROCESSED_DATA_DIR = os.path.join(
    PROJECT_ROOT, "processed_data"
)

MODELS_DIR = os.path.join(
    PROJECT_ROOT, "models"
)

EVALUATION_DIR = os.path.join(
    PROJECT_ROOT, "evaluation"
)

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(EVALUATION_DIR, exist_ok=True)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", DEVICE)


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)


# ============================================================
# DATASET CLASS
# ============================================================

class MovieRatingDataset(Dataset):

    def __init__(
        self,
        user_ids,
        movie_ids,
        ratings,
        content_embeddings
    ):
        self.user_ids = torch.tensor(
            user_ids,
            dtype=torch.long
        )

        self.movie_ids = torch.tensor(
            movie_ids,
            dtype=torch.long
        )

        self.ratings = torch.tensor(
            ratings,
            dtype=torch.float32
        )

        self.content_embeddings = torch.tensor(
            content_embeddings,
            dtype=torch.float32
        )

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, index):

        movie_id = self.movie_ids[index]

        return (
            self.user_ids[index],
            movie_id,
            self.content_embeddings[movie_id],
            self.ratings[index]
        )


# ============================================================
# ATTENTION MODULE
# ============================================================

class AttentionFusion(nn.Module):

    def __init__(self, embedding_dim):

        super().__init__()

        self.attention_network = nn.Sequential(

            nn.Linear(
                embedding_dim,
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
        content_embedding,
        collaborative_embedding
    ):

        # ----------------------------------------------------
        # Calculate attention score for content
        # ----------------------------------------------------

        content_score = self.attention_network(
            content_embedding
        )

        # ----------------------------------------------------
        # Calculate attention score for collaborative
        # ----------------------------------------------------

        collaborative_score = self.attention_network(
            collaborative_embedding
        )

        # ----------------------------------------------------
        # Stack scores
        # Shape:
        # [batch_size, 2]
        # ----------------------------------------------------

        scores = torch.cat(
            [
                content_score,
                collaborative_score
            ],
            dim=1
        )

        # ----------------------------------------------------
        # Convert scores into probabilities
        # ----------------------------------------------------

        attention_weights = torch.softmax(
            scores,
            dim=1
        )

        content_weight = attention_weights[:, 0:1]

        collaborative_weight = attention_weights[:, 1:2]

        # ----------------------------------------------------
        # Weighted fusion
        # ----------------------------------------------------

        fused_embedding = (
            content_weight * content_embedding
            +
            collaborative_weight * collaborative_embedding
        )

        return (
            fused_embedding,
            attention_weights
        )


# ============================================================
# COMPLETE ATTENTION FUSION MODEL
# ============================================================

class AttentionFusionModel(nn.Module):

    def __init__(
        self,
        num_users,
        num_movies,
        embedding_dim=32,
        hidden_dim=64,
        dropout=0.2
    ):

        super().__init__()

        # ----------------------------------------------------
        # User embedding
        # ----------------------------------------------------

        self.user_embedding = nn.Embedding(
            num_users,
            embedding_dim
        )

        # ----------------------------------------------------
        # Collaborative movie embedding
        # ----------------------------------------------------

        self.movie_embedding = nn.Embedding(
            num_movies,
            embedding_dim
        )

        # ----------------------------------------------------
        # Attention fusion
        # ----------------------------------------------------

        self.attention = AttentionFusion(
            embedding_dim
        )

        # ----------------------------------------------------
        # Deep fusion network
        # ----------------------------------------------------

        self.fusion_network = nn.Sequential(

            nn.Linear(
                embedding_dim * 2,
                hidden_dim
            ),

            nn.ReLU(),

            nn.Dropout(dropout),

            nn.Linear(
                hidden_dim,
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
        content_embeddings
    ):

        # ----------------------------------------------------
        # User representation
        # ----------------------------------------------------

        user_embedding = self.user_embedding(
            user_ids
        )

        # ----------------------------------------------------
        # Collaborative movie representation
        # ----------------------------------------------------

        movie_embedding = self.movie_embedding(
            movie_ids
        )

        # ----------------------------------------------------
        # Attention fusion
        # ----------------------------------------------------

        fused_movie_embedding, attention_weights = (
            self.attention(
                content_embeddings,
                movie_embedding
            )
        )

        # ----------------------------------------------------
        # Combine user + fused movie
        # ----------------------------------------------------

        final_input = torch.cat(
            [
                user_embedding,
                fused_movie_embedding
            ],
            dim=1
        )

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        prediction = self.fusion_network(
            final_input
        )

        return (
            prediction.squeeze(1),
            attention_weights
        )


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("DAY 16 - ATTENTION-BASED DEEP EMBEDDING FUSION")
print("=" * 70)

ratings_path = os.path.join(
    PROCESSED_DATA_DIR,
    "ratings_clean.csv"
)

content_path = os.path.join(
    MODELS_DIR,
    "day15_content_embeddings.npy"
)

print("\nLoading ratings...")

ratings = pd.read_csv(
    ratings_path
)

print("Ratings shape:", ratings.shape)

print("\nLoading content embeddings...")

content_embeddings = np.load(
    content_path
)

print(
    "Content embedding shape:",
    content_embeddings.shape
)


# ============================================================
# CREATE INDEX MAPPINGS
# ============================================================

print("\nCreating ID mappings...")

unique_users = sorted(
    ratings["user_id"].unique()
)

unique_movies = sorted(
    ratings["movie_id"].unique()
)

user_to_index = {
    user_id: index
    for index, user_id in enumerate(unique_users)
}

movie_to_index = {
    movie_id: index
    for index, movie_id in enumerate(unique_movies)
}


ratings["user_index"] = ratings[
    "user_id"
].map(user_to_index)

ratings["movie_index"] = ratings[
    "movie_id"
].map(movie_to_index)


print(
    "Number of users:",
    len(unique_users)
)

print(
    "Number of movies:",
    len(unique_movies)
)


# ============================================================
# VALIDATE CONTENT EMBEDDINGS
# ============================================================

if content_embeddings.shape[0] != len(unique_movies):

    raise ValueError(
        "Content embedding movie count does not match "
        "MovieLens movie count."
    )

if content_embeddings.shape[1] != EMBEDDING_DIM:

    raise ValueError(
        "Expected 32-dimensional content embeddings."
    )


# ============================================================
# TRAIN TEST SPLIT
# ============================================================

train_df, test_df = train_test_split(
    ratings,
    test_size=0.10,
    random_state=RANDOM_SEED
)

print("\nTrain samples:", len(train_df))
print("Test samples:", len(test_df))


# ============================================================
# CREATE DATASETS
# ============================================================

train_dataset = MovieRatingDataset(

    train_df["user_index"].values,

    train_df["movie_index"].values,

    train_df["rating"].values,

    content_embeddings
)


test_dataset = MovieRatingDataset(

    test_df["user_index"].values,

    test_df["movie_index"].values,

    test_df["rating"].values,

    content_embeddings
)


# ============================================================
# DATA LOADERS
# ============================================================

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


# ============================================================
# MODEL
# ============================================================

model = AttentionFusionModel(

    num_users=len(unique_users),

    num_movies=len(unique_movies),

    embedding_dim=EMBEDDING_DIM,

    hidden_dim=HIDDEN_DIM,

    dropout=DROPOUT
).to(DEVICE)


print("\n" + "=" * 70)
print("MODEL ARCHITECTURE")
print("=" * 70)

print(model)


# ============================================================
# LOSS + OPTIMIZER
# ============================================================

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate(model, loader):

    model.eval()

    predictions = []

    actuals = []

    attention_values = []

    with torch.no_grad():

        for (
            user_ids,
            movie_ids,
            content_batch,
            ratings_batch
        ) in loader:

            user_ids = user_ids.to(DEVICE)

            movie_ids = movie_ids.to(DEVICE)

            content_batch = content_batch.to(
                DEVICE
            )

            ratings_batch = ratings_batch.to(
                DEVICE
            )

            preds, attention = model(
                user_ids,
                movie_ids,
                content_batch
            )

            predictions.extend(
                preds.cpu().numpy()
            )

            actuals.extend(
                ratings_batch.cpu().numpy()
            )

            attention_values.append(
                attention.cpu().numpy()
            )

    predictions = np.array(
        predictions
    )

    actuals = np.array(
        actuals
    )

    attention_values = np.vstack(
        attention_values
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

    return (
        rmse,
        mae,
        predictions,
        actuals,
        attention_values
    )


# ============================================================
# TRAINING
# ============================================================

print("\n" + "=" * 70)
print("TRAINING ATTENTION FUSION MODEL")
print("=" * 70)

history = []

for epoch in range(EPOCHS):

    model.train()

    total_loss = 0.0

    for (
        user_ids,
        movie_ids,
        content_batch,
        ratings_batch
    ) in train_loader:

        user_ids = user_ids.to(DEVICE)

        movie_ids = movie_ids.to(DEVICE)

        content_batch = content_batch.to(
            DEVICE
        )

        ratings_batch = ratings_batch.to(
            DEVICE
        )

        optimizer.zero_grad()

        predictions, _ = model(
            user_ids,
            movie_ids,
            content_batch
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

    average_train_loss = (
        total_loss
        / len(train_dataset)
    )

    (
        test_rmse,
        test_mae,
        _,
        _,
        _
    ) = evaluate(
        model,
        test_loader
    )

    history.append({

        "epoch": epoch + 1,

        "train_loss": average_train_loss,

        "test_rmse": test_rmse,

        "test_mae": test_mae
    })

    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"Train Loss: {average_train_loss:.4f} | "
        f"Test RMSE: {test_rmse:.4f} | "
        f"Test MAE: {test_mae:.4f}"
    )


# ============================================================
# FINAL EVALUATION
# ============================================================

(
    final_rmse,
    final_mae,
    predictions,
    actuals,
    attention_values
) = evaluate(
    model,
    test_loader
)


print("\n" + "=" * 70)
print("FINAL MODEL EVALUATION")
print("=" * 70)

print(
    f"\nRMSE: {final_rmse:.4f}"
)

print(
    f"MAE:  {final_mae:.4f}"
)


# ============================================================
# ATTENTION ANALYSIS
# ============================================================

content_attention = attention_values[:, 0]

collaborative_attention = attention_values[:, 1]

average_content_attention = (
    content_attention.mean()
)

average_collaborative_attention = (
    collaborative_attention.mean()
)


print("\n" + "=" * 70)
print("ATTENTION ANALYSIS")
print("=" * 70)

print(
    f"\nAverage Content Attention: "
    f"{average_content_attention:.4f}"
)

print(
    f"Average Collaborative Attention: "
    f"{average_collaborative_attention:.4f}"
)

print(
    "\nAttention sum:",
    (
        average_content_attention
        +
        average_collaborative_attention
    )
)


# ============================================================
# SAVE ATTENTION RESULTS
# ============================================================

attention_results = pd.DataFrame({

    "content_attention":
        content_attention,

    "collaborative_attention":
        collaborative_attention,

    "actual_rating":
        actuals,

    "predicted_rating":
        predictions
})


attention_results_path = os.path.join(
    EVALUATION_DIR,
    "day16_attention_results.csv"
)

attention_results.to_csv(
    attention_results_path,
    index=False
)

print(
    "\nAttention results saved:"
)

print(
    attention_results_path
)


# ============================================================
# SAVE TRAINING HISTORY
# ============================================================

history_df = pd.DataFrame(
    history
)

history_path = os.path.join(
    EVALUATION_DIR,
    "day16_training_history.csv"
)

history_df.to_csv(
    history_path,
    index=False
)

print(
    "\nTraining history saved:"
)

print(
    history_path
)


# ============================================================
# SAVE MODEL
# ============================================================

model_path = os.path.join(
    MODELS_DIR,
    "day16_attention_fusion_model.pth"
)

torch.save(
    {
        "model_state_dict":
            model.state_dict(),

        "num_users":
            len(unique_users),

        "num_movies":
            len(unique_movies),

        "embedding_dim":
            EMBEDDING_DIM,

        "hidden_dim":
            HIDDEN_DIM,

        "final_rmse":
            final_rmse,

        "final_mae":
            final_mae
    },
    model_path
)

print(
    "\nModel saved:"
)

print(
    model_path
)


# ============================================================
# SAVE LEARNED EMBEDDINGS
# ============================================================

user_embeddings = (
    model.user_embedding
    .weight
    .detach()
    .cpu()
    .numpy()
)

movie_embeddings = (
    model.movie_embedding
    .weight
    .detach()
    .cpu()
    .numpy()
)


user_embedding_path = os.path.join(
    MODELS_DIR,
    "day16_user_embeddings.npy"
)

movie_embedding_path = os.path.join(
    MODELS_DIR,
    "day16_movie_embeddings.npy"
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
    "\nUser embeddings saved:"
)

print(
    user_embedding_path
)

print(
    "\nMovie embeddings saved:"
)

print(
    movie_embedding_path
)


# ============================================================
# SAVE METRICS
# ============================================================

metrics_df = pd.DataFrame({

    "metric": [

        "RMSE",

        "MAE",

        "Average Content Attention",

        "Average Collaborative Attention"
    ],

    "value": [

        final_rmse,

        final_mae,

        average_content_attention,

        average_collaborative_attention
    ]
})


metrics_path = os.path.join(
    EVALUATION_DIR,
    "day16_attention_metrics.csv"
)

metrics_df.to_csv(
    metrics_path,
    index=False
)


# ============================================================
# PLOT TRAINING RMSE
# ============================================================

plt.figure(
    figsize=(8, 5)
)

plt.plot(
    history_df["epoch"],
    history_df["test_rmse"],
    marker="o"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Test RMSE"
)

plt.title(
    "Day 16 Attention Fusion - Test RMSE"
)

plt.grid(
    True,
    alpha=0.3
)

rmse_plot_path = os.path.join(
    EVALUATION_DIR,
    "day16_attention_rmse.png"
)

plt.savefig(
    rmse_plot_path,
    dpi=150,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# PLOT ATTENTION DISTRIBUTION
# ============================================================

plt.figure(
    figsize=(8, 5)
)

plt.hist(
    content_attention,
    bins=30,
    alpha=0.7,
    label="Content Attention"
)

plt.hist(
    collaborative_attention,
    bins=30,
    alpha=0.7,
    label="Collaborative Attention"
)

plt.xlabel(
    "Attention Weight"
)

plt.ylabel(
    "Frequency"
)

plt.title(
    "Content vs Collaborative Attention"
)

plt.legend()

plt.grid(
    True,
    alpha=0.3
)

attention_plot_path = os.path.join(
    EVALUATION_DIR,
    "day16_attention_distribution.png"
)

plt.savefig(
    attention_plot_path,
    dpi=150,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# SAMPLE PREDICTIONS
# ============================================================

sample_size = min(
    10,
    len(test_df)
)

sample_results = test_df.iloc[
    :sample_size
].copy()

sample_results[
    "predicted_rating"
] = predictions[:sample_size]

sample_results[
    "content_attention"
] = content_attention[
    :sample_size
]

sample_results[
    "collaborative_attention"
] = collaborative_attention[
    :sample_size
]


print("\n" + "=" * 70)
print("SAMPLE PREDICTIONS")
print("=" * 70)

print(
    sample_results[
        [
            "user_id",
            "movie_id",
            "rating",
            "predicted_rating",
            "content_attention",
            "collaborative_attention"
        ]
    ].to_string(index=False)
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("DAY 16 COMPLETED SUCCESSFULLY")
print("=" * 70)

print(
    "\nFinal RMSE:",
    round(final_rmse, 4)
)

print(
    "Final MAE:",
    round(final_mae, 4)
)

print(
    "\nAverage Content Attention:",
    round(
        average_content_attention,
        4
    )
)

print(
    "Average Collaborative Attention:",
    round(
        average_collaborative_attention,
        4
    )
)

print(
    "\nSaved model:",
    model_path
)

print(
    "\nSaved metrics:",
    metrics_path
)

print(
    "\nSaved plots:"
)

print(
    rmse_plot_path
)

print(
    attention_plot_path
)