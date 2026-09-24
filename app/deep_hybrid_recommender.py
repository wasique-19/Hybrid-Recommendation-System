import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# ============================================================
# ATTENTION FUSION
# ============================================================

class AttentionFusion(nn.Module):

    def __init__(self, embedding_dim=32):
        super().__init__()

        self.attention_network = nn.Sequential(
            nn.Linear(
                embedding_dim,
                embedding_dim
            ),
            nn.ReLU(),
            nn.Linear(
                embedding_dim,
                1
            )
        )

    def forward(
        self,
        content_embedding,
        collaborative_embedding
    ):

        content_score = self.attention_network(
            content_embedding
        )

        collaborative_score = self.attention_network(
            collaborative_embedding
        )

        attention_scores = torch.cat(
            [
                content_score,
                collaborative_score
            ],
            dim=1
        )

        attention_weights = torch.softmax(
            attention_scores,
            dim=1
        )

        content_weight = attention_weights[:, 0:1]

        collaborative_weight = attention_weights[:, 1:2]

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
# ATTENTION FUSION MODEL
# ============================================================

class AttentionFusionModel(nn.Module):

    def __init__(
        self,
        num_users,
        num_movies,
        embedding_dim=32
    ):

        super().__init__()

        self.user_embedding = nn.Embedding(
            num_users,
            embedding_dim
        )

        self.movie_embedding = nn.Embedding(
            num_movies,
            embedding_dim
        )

        self.attention = AttentionFusion(
            embedding_dim
        )

        self.fusion_network = nn.Sequential(

            nn.Linear(
                embedding_dim * 2,
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
        content_embeddings
    ):

        user_emb = self.user_embedding(
            user_ids
        )

        movie_emb = self.movie_embedding(
            movie_ids
        )

        content_emb = content_embeddings[
            movie_ids
        ]

        fused_movie_embedding, attention_weights = (
            self.attention(
                content_emb,
                movie_emb
            )
        )

        combined = torch.cat(
            [
                user_emb,
                fused_movie_embedding
            ],
            dim=1
        )

        prediction = self.fusion_network(
            combined
        )

        return (
            prediction.squeeze(1),
            attention_weights
        )


# ============================================================
# DEEP HYBRID RECOMMENDER
# ============================================================

class DeepHybridRecommender:

    def __init__(self):

        # ----------------------------------------------------
        # DEVICE
        # ----------------------------------------------------

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        print(
            f"Using device: {self.device}"
        )

        # ----------------------------------------------------
        # PROJECT DIRECTORIES
        # ----------------------------------------------------

        base_dir = os.path.dirname(
            os.path.dirname(
                os.path.abspath(__file__)
            )
        )

        self.base_dir = base_dir

        processed_dir = os.path.join(
            base_dir,
            "processed_data"
        )

        models_dir = os.path.join(
            base_dir,
            "models"
        )

        # ----------------------------------------------------
        # LOAD RATINGS
        # ----------------------------------------------------

        ratings_path = os.path.join(
            processed_dir,
            "ratings_clean.csv"
        )

        print(
            f"Loading ratings: {ratings_path}"
        )

        self.ratings = pd.read_csv(
            ratings_path
        )

        # ----------------------------------------------------
        # LOAD MOVIES
        # ----------------------------------------------------

        movies_path = os.path.join(
            processed_dir,
            "movie_features.csv"
        )

        print(
            f"Loading movie features: {movies_path}"
        )

        self.movies = pd.read_csv(
            movies_path
        )

        # ----------------------------------------------------
        # LOAD CONTENT EMBEDDINGS
        # ----------------------------------------------------

        content_path = os.path.join(
            models_dir,
            "day15_content_embeddings.npy"
        )

        print(
            f"Loading content embeddings: {content_path}"
        )

        self.content_embeddings = np.load(
            content_path
        )

        print()
        print("Ratings shape:")
        print(self.ratings.shape)

        print()
        print("Movies shape:")
        print(self.movies.shape)

        print()
        print("Content embedding shape:")
        print(self.content_embeddings.shape)

        # ----------------------------------------------------
        # CREATE MAPPINGS
        # ----------------------------------------------------

        self.user_mapping = (
            self._create_user_mapping()
        )

        self.movie_mapping = (
            self._create_movie_mapping()
        )

        self.reverse_movie_mapping = {
            index: movie_id
            for movie_id, index
            in self.movie_mapping.items()
        }

        self.num_users = len(
            self.user_mapping
        )

        self.num_movies = len(
            self.movie_mapping
        )

        print()
        print("Number of users:")
        print(self.num_users)

        print("Number of movies:")
        print(self.num_movies)

        # ----------------------------------------------------
        # VERIFY CONTENT EMBEDDINGS
        # ----------------------------------------------------

        if self.content_embeddings.ndim != 2:

            raise ValueError(
                "Content embeddings must be a 2D matrix."
            )

        if self.content_embeddings.shape[0] != self.num_movies:

            raise ValueError(
                "Content embedding count does not match "
                "number of movies.\n"
                f"Content embeddings: "
                f"{self.content_embeddings.shape[0]}\n"
                f"Movies in mapping: "
                f"{self.num_movies}"
            )

        if self.content_embeddings.shape[1] != 32:

            raise ValueError(
                "Content embedding dimension mismatch.\n"
                f"Found: {self.content_embeddings.shape[1]}\n"
                f"Expected: 32"
            )

        # ----------------------------------------------------
        # CREATE MODEL
        # ----------------------------------------------------

        self.model = AttentionFusionModel(
            num_users=self.num_users,
            num_movies=self.num_movies,
            embedding_dim=32
        )

        # ----------------------------------------------------
        # LOAD DAY 16 CHECKPOINT
        # ----------------------------------------------------

        model_path = os.path.join(
            models_dir,
            "day16_attention_fusion_model.pth"
        )

        print()
        print("Loading Day 16 model:")
        print(model_path)

        # IMPORTANT:
        # Day 16 saved a complete checkpoint dictionary.
        #
        # PyTorch 2.6+ defaults to weights_only=True.
        # We explicitly use weights_only=False because this
        # checkpoint contains metadata along with model weights.
        #
        # Use this ONLY for a trusted checkpoint created by you.

        checkpoint = torch.load(
            model_path,
            map_location=self.device,
            weights_only=False
        )

        # ----------------------------------------------------
        # HANDLE CHECKPOINT FORMAT
        # ----------------------------------------------------

        if (
            isinstance(checkpoint, dict)
            and "model_state_dict" in checkpoint
        ):

            print(
                "Checkpoint format: full training checkpoint"
            )

            saved_num_users = checkpoint.get(
                "num_users"
            )

            saved_num_movies = checkpoint.get(
                "num_movies"
            )

            saved_embedding_dim = checkpoint.get(
                "embedding_dim",
                32
            )

            # Check user count
            if (
                saved_num_users is not None
                and saved_num_users != self.num_users
            ):

                raise ValueError(
                    "User count mismatch.\n"
                    f"Checkpoint: {saved_num_users}\n"
                    f"Current data: {self.num_users}"
                )

            # Check movie count
            if (
                saved_num_movies is not None
                and saved_num_movies != self.num_movies
            ):

                raise ValueError(
                    "Movie count mismatch.\n"
                    f"Checkpoint: {saved_num_movies}\n"
                    f"Current data: {self.num_movies}"
                )

            # Check embedding dimension
            if saved_embedding_dim != 32:

                raise ValueError(
                    "Embedding dimension mismatch.\n"
                    f"Checkpoint: {saved_embedding_dim}\n"
                    f"Expected: 32"
                )

            state_dict = checkpoint[
                "model_state_dict"
            ]

        else:

            print(
                "Checkpoint format: raw state_dict"
            )

            state_dict = checkpoint

        # ----------------------------------------------------
        # LOAD MODEL WEIGHTS
        # ----------------------------------------------------

        try:

            self.model.load_state_dict(
                state_dict,
                strict=True
            )

        except RuntimeError as error:

            print()
            print(
                "ERROR: Model architecture does not match "
                "the Day 16 checkpoint."
            )

            raise error

        self.model.to(
            self.device
        )

        self.model.eval()

        # ----------------------------------------------------
        # CONTENT TENSOR
        # ----------------------------------------------------

        self.content_tensor = torch.tensor(
            self.content_embeddings,
            dtype=torch.float32,
            device=self.device
        )

        print()
        print(
            "Deep Hybrid Recommender loaded successfully."
        )

        print(
            f"Model device: {self.device}"
        )

    # ========================================================
    # USER MAPPING
    # ========================================================

    def _create_user_mapping(self):

        user_ids = sorted(
            self.ratings[
                "user_id"
            ].unique()
        )

        return {
            int(user_id): index
            for index, user_id
            in enumerate(user_ids)
        }

    # ========================================================
    # MOVIE MAPPING
    # ========================================================

    def _create_movie_mapping(self):

        movie_ids = sorted(
            self.ratings[
                "movie_id"
            ].unique()
        )

        return {
            int(movie_id): index
            for index, movie_id
            in enumerate(movie_ids)
        }

    # ========================================================
    # GET WATCHED MOVIES
    # ========================================================

    def _get_watched_movies(
        self,
        user_id
    ):

        watched = self.ratings[
            self.ratings[
                "user_id"
            ] == user_id
        ][
            "movie_id"
        ].tolist()

        return set(
            int(movie_id)
            for movie_id in watched
        )

    # ========================================================
    # RECOMMEND
    # ========================================================

    def recommend(
        self,
        user_id,
        top_k=10
    ):

        user_id = int(user_id)

        # ----------------------------------------------------
        # COLD START
        # ----------------------------------------------------

        if user_id not in self.user_mapping:

            print()
            print(
                f"User {user_id} not found."
            )

            print(
                "Using popularity-based "
                "cold-start fallback..."
            )

            return self._cold_start_recommendations(
                top_k
            )

        # ----------------------------------------------------
        # USER INDEX
        # ----------------------------------------------------

        user_index = self.user_mapping[
            user_id
        ]

        # ----------------------------------------------------
        # WATCHED MOVIES
        # ----------------------------------------------------

        watched_movies = (
            self._get_watched_movies(
                user_id
            )
        )

        # ----------------------------------------------------
        # CANDIDATE MOVIES
        # ----------------------------------------------------

        candidate_movies = [
            movie_id
            for movie_id in self.movie_mapping
            if movie_id not in watched_movies
        ]

        if len(candidate_movies) == 0:

            return pd.DataFrame(
                columns=[
                    "movie_id",
                    "title",
                    "genre_text",
                    "release_year",
                    "predicted_rating",
                    "content_attention",
                    "collaborative_attention"
                ]
            )

        # ----------------------------------------------------
        # CREATE TENSORS
        # ----------------------------------------------------

        user_indices = [
            user_index
            for _ in candidate_movies
        ]

        movie_indices = [
            self.movie_mapping[movie_id]
            for movie_id in candidate_movies
        ]

        user_tensor = torch.tensor(
            user_indices,
            dtype=torch.long,
            device=self.device
        )

        movie_tensor = torch.tensor(
            movie_indices,
            dtype=torch.long,
            device=self.device
        )

        # ----------------------------------------------------
        # MODEL PREDICTION
        # ----------------------------------------------------

        with torch.no_grad():

            predictions, attention = self.model(
                user_tensor,
                movie_tensor,
                self.content_tensor
            )

        # ----------------------------------------------------
        # CONVERT TO NUMPY
        # ----------------------------------------------------

        predictions = (
            predictions
            .detach()
            .cpu()
            .numpy()
        )

        attention = (
            attention
            .detach()
            .cpu()
            .numpy()
        )

        # ----------------------------------------------------
        # CREATE RESULT
        # ----------------------------------------------------

        result = pd.DataFrame({

            "movie_id":
                candidate_movies,

            "predicted_rating":
                predictions,

            "content_attention":
                attention[:, 0],

            "collaborative_attention":
                attention[:, 1]
        })

        # ----------------------------------------------------
        # SORT
        # ----------------------------------------------------

        result = result.sort_values(
            "predicted_rating",
            ascending=False
        )

        result = result.head(
            top_k
        )

        # ----------------------------------------------------
        # MERGE MOVIE INFORMATION
        # ----------------------------------------------------

        movie_columns = [
            "movie_id",
            "title",
            "genre_text",
            "release_year"
        ]

        available_columns = [
            column
            for column in movie_columns
            if column in self.movies.columns
        ]

        result = result.merge(
            self.movies[
                available_columns
            ],
            on="movie_id",
            how="left"
        )

        # ----------------------------------------------------
        # FINAL COLUMNS
        # ----------------------------------------------------

        columns = [
            "movie_id",
            "title",
            "genre_text",
            "release_year",
            "predicted_rating",
            "content_attention",
            "collaborative_attention"
        ]

        columns = [
            column
            for column in columns
            if column in result.columns
        ]

        return result[
            columns
        ].reset_index(
            drop=True
        )

    # ========================================================
    # COLD START
    # ========================================================

    def _cold_start_recommendations(
        self,
        top_k=10
    ):

        popularity = (
            self.ratings
            .groupby("movie_id")
            .agg(
                average_rating=(
                    "rating",
                    "mean"
                ),
                rating_count=(
                    "rating",
                    "count"
                )
            )
            .reset_index()
        )

        popularity[
            "popularity_score"
        ] = (
            popularity[
                "average_rating"
            ]
            *
            np.log1p(
                popularity[
                    "rating_count"
                ]
            )
        )

        popularity = popularity.sort_values(
            "popularity_score",
            ascending=False
        )

        result = popularity.head(
            top_k
        ).copy()

        # ----------------------------------------------------
        # MERGE MOVIE INFORMATION
        # ----------------------------------------------------

        movie_columns = [
            "movie_id",
            "title",
            "genre_text",
            "release_year"
        ]

        available_columns = [
            column
            for column in movie_columns
            if column in self.movies.columns
        ]

        result = result.merge(
            self.movies[
                available_columns
            ],
            on="movie_id",
            how="left"
        )

        result[
            "recommendation_type"
        ] = "cold_start"

        # ----------------------------------------------------
        # FINAL COLUMNS
        # ----------------------------------------------------

        columns = [
            "movie_id",
            "title",
            "genre_text",
            "release_year",
            "average_rating",
            "rating_count",
            "popularity_score",
            "recommendation_type"
        ]

        columns = [
            column
            for column in columns
            if column in result.columns
        ]

        return result[
            columns
        ].reset_index(
            drop=True
        )