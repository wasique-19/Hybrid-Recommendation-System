# ================================================================
# DAY 18 - DEEP HYBRID MODEL EVALUATION
# ================================================================

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error


# ================================================================
# CONFIGURATION
# ================================================================

TOP_K = 10
RELEVANCE_THRESHOLD = 4.0
TEST_SIZE = 0.10
RANDOM_STATE = 42

EMBEDDING_DIM = 32


# ================================================================
# MODEL ARCHITECTURE
# SAME ARCHITECTURE USED IN DAY 16
# ================================================================

class AttentionFusion(nn.Module):

    def __init__(self, embedding_dim=32):

        super().__init__()

        self.attention_network = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 1)
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


# ================================================================
# MAIN EVALUATOR
# ================================================================

class Day18Evaluator:

    def __init__(self):

        print("=" * 70)
        print("DAY 18 - DEEP HYBRID MODEL EVALUATION")
        print("=" * 70)

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        print()
        print("Using device:", self.device)

        # --------------------------------------------------------
        # PATHS
        # --------------------------------------------------------

        base_dir = os.path.dirname(
            os.path.dirname(
                os.path.abspath(__file__)
            )
        )

        self.processed_dir = os.path.join(
            base_dir,
            "processed_data"
        )

        self.models_dir = os.path.join(
            base_dir,
            "models"
        )

        self.evaluation_dir = os.path.join(
            base_dir,
            "evaluation"
        )

        os.makedirs(
            self.evaluation_dir,
            exist_ok=True
        )

        # --------------------------------------------------------
        # LOAD RATINGS
        # --------------------------------------------------------

        ratings_path = os.path.join(
            self.processed_dir,
            "ratings_clean.csv"
        )

        print()
        print("Loading ratings:")
        print(ratings_path)

        self.ratings = pd.read_csv(
            ratings_path
        )

        print()
        print("Ratings shape:")
        print(self.ratings.shape)

        # --------------------------------------------------------
        # LOAD MOVIES
        # --------------------------------------------------------

        movies_path = os.path.join(
            self.processed_dir,
            "movie_features.csv"
        )

        print()
        print("Loading movie features:")
        print(movies_path)

        self.movies = pd.read_csv(
            movies_path
        )

        print()
        print("Movies shape:")
        print(self.movies.shape)

        # --------------------------------------------------------
        # LOAD CONTENT EMBEDDINGS
        # --------------------------------------------------------

        content_path = os.path.join(
            self.models_dir,
            "day15_content_embeddings.npy"
        )

        print()
        print("Loading content embeddings:")
        print(content_path)

        self.content_embeddings = np.load(
            content_path
        )

        print()
        print("Content embedding shape:")
        print(self.content_embeddings.shape)

        # --------------------------------------------------------
        # CREATE MAPPINGS
        # --------------------------------------------------------

        self.user_ids = sorted(
            self.ratings["user_id"].unique()
        )

        self.movie_ids = sorted(
            self.ratings["movie_id"].unique()
        )

        self.user_mapping = {
            user_id: index
            for index, user_id
            in enumerate(self.user_ids)
        }

        self.movie_mapping = {
            movie_id: index
            for index, movie_id
            in enumerate(self.movie_ids)
        }

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

        # --------------------------------------------------------
        # LOAD MODEL
        # --------------------------------------------------------

        self.model = AttentionFusionModel(
            self.num_users,
            self.num_movies,
            EMBEDDING_DIM
        )

        model_path = os.path.join(
            self.models_dir,
            "day16_attention_fusion_model.pth"
        )

        print()
        print("Loading Day 16 model:")
        print(model_path)

        checkpoint = torch.load(
            model_path,
            map_location=self.device,
            weights_only=False
        )

        # --------------------------------------------------------
        # HANDLE CHECKPOINT FORMAT
        # --------------------------------------------------------

        if isinstance(
            checkpoint,
            dict
        ) and "model_state_dict" in checkpoint:

            print(
                "Checkpoint format: full training checkpoint"
            )

            state_dict = checkpoint[
                "model_state_dict"
            ]

        else:

            print(
                "Checkpoint format: state dictionary"
            )

            state_dict = checkpoint

        self.model.load_state_dict(
            state_dict
        )

        self.model.to(
            self.device
        )

        self.model.eval()

        # --------------------------------------------------------
        # CONTENT TENSOR
        # --------------------------------------------------------

        self.content_tensor = torch.tensor(
            self.content_embeddings,
            dtype=torch.float32,
            device=self.device
        )

        print()
        print("Model loaded successfully.")

        # --------------------------------------------------------
        # TRAIN / TEST SPLIT
        # --------------------------------------------------------

        (
            self.train_ratings,
            self.test_ratings
        ) = train_test_split(
            self.ratings,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE
        )

        print()
        print("Train samples:")
        print(len(self.train_ratings))

        print("Test samples:")
        print(len(self.test_ratings))


    # ============================================================
    # PREDICT TEST RATINGS
    # ============================================================

    def evaluate_rating_prediction(self):

        print()
        print("=" * 70)
        print("RATING PREDICTION EVALUATION")
        print("=" * 70)

        valid_rows = []

        for _, row in self.test_ratings.iterrows():

            if (
                row["user_id"] in self.user_mapping
                and
                row["movie_id"] in self.movie_mapping
            ):

                valid_rows.append(row)

        test_df = pd.DataFrame(
            valid_rows
        )

        user_indices = [
            self.user_mapping[user_id]
            for user_id
            in test_df["user_id"]
        ]

        movie_indices = [
            self.movie_mapping[movie_id]
            for movie_id
            in test_df["movie_id"]
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

        with torch.no_grad():

            predictions, attention = self.model(
                user_tensor,
                movie_tensor,
                self.content_tensor
            )

        predictions = (
            predictions
            .cpu()
            .numpy()
        )

        attention = (
            attention
            .cpu()
            .numpy()
        )

        actual = (
            test_df["rating"]
            .values
        )

        rmse = np.sqrt(
            mean_squared_error(
                actual,
                predictions
            )
        )

        mae = mean_absolute_error(
            actual,
            predictions
        )

        average_content_attention = (
            attention[:, 0].mean()
        )

        average_collaborative_attention = (
            attention[:, 1].mean()
        )

        print()
        print("RMSE:", round(rmse, 4))

        print("MAE :", round(mae, 4))

        print(
            "Average Content Attention:",
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

        return {
            "rmse": rmse,
            "mae": mae,
            "average_content_attention":
                average_content_attention,
            "average_collaborative_attention":
                average_collaborative_attention
        }


    # ============================================================
    # GENERATE RECOMMENDATIONS
    # ============================================================

    def generate_recommendations(
        self,
        user_id,
        top_k=10
    ):

        if user_id not in self.user_mapping:

            return pd.DataFrame()

        user_index = self.user_mapping[
            user_id
        ]

        watched_movies = set(
            self.ratings[
                self.ratings["user_id"] == user_id
            ]["movie_id"]
            .tolist()
        )

        candidate_movies = [
            movie_id
            for movie_id in self.movie_mapping
            if movie_id not in watched_movies
        ]

        if len(candidate_movies) == 0:

            return pd.DataFrame()

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

        with torch.no_grad():

            predictions, attention = self.model(
                user_tensor,
                movie_tensor,
                self.content_tensor
            )

        predictions = (
            predictions
            .cpu()
            .numpy()
        )

        attention = (
            attention
            .cpu()
            .numpy()
        )

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

        result = result.sort_values(
            "predicted_rating",
            ascending=False
        )

        result = result.head(
            top_k
        )

        result = result.merge(
            self.movies[
                [
                    "movie_id",
                    "title",
                    "genre_text",
                    "release_year"
                ]
            ],
            on="movie_id",
            how="left"
        )

        columns = [
            "movie_id",
            "title",
            "genre_text",
            "release_year",
            "predicted_rating",
            "content_attention",
            "collaborative_attention"
        ]

        return result[
            columns
        ].reset_index(
            drop=True
        )


    # ============================================================
    # PRECISION / RECALL / HIT RATE / NDCG
    # ============================================================

    def evaluate_top_k(
        self,
        user_ids,
        top_k=10
    ):

        print()
        print("=" * 70)
        print(
            f"TOP-{top_k} RECOMMENDATION EVALUATION"
        )
        print("=" * 70)

        precisions = []
        recalls = []
        hit_rates = []
        ndcgs = []

        evaluated_users = 0

        for user_id in user_ids:

            if user_id not in self.user_mapping:
                continue

            # ----------------------------------------------------
            # Actual relevant movies from TEST SET
            # ----------------------------------------------------

            user_test = self.test_ratings[
                self.test_ratings["user_id"] == user_id
            ]

            relevant_movies = set(
                user_test[
                    user_test["rating"] >= RELEVANCE_THRESHOLD
                ]["movie_id"]
            )

            if len(relevant_movies) == 0:
                continue

            recommendations = (
                self.generate_recommendations(
                    user_id,
                    top_k
                )
            )

            if recommendations.empty:
                continue

            recommended_movies = (
                recommendations[
                    "movie_id"
                ]
                .tolist()
            )

            hits = [
                movie_id
                for movie_id
                in recommended_movies
                if movie_id in relevant_movies
            ]

            hit_count = len(hits)

            precision = (
                hit_count / top_k
            )

            recall = (
                hit_count /
                len(relevant_movies)
            )

            hit_rate = (
                1.0
                if hit_count > 0
                else 0.0
            )

            # ----------------------------------------------------
            # NDCG
            # ----------------------------------------------------

            dcg = 0.0

            for rank, movie_id in enumerate(
                recommended_movies,
                start=1
            ):

                if movie_id in relevant_movies:

                    dcg += (
                        1.0 /
                        np.log2(rank + 1)
                    )

            ideal_hits = min(
                len(relevant_movies),
                top_k
            )

            idcg = sum(
                1.0 /
                np.log2(rank + 1)
                for rank
                in range(
                    1,
                    ideal_hits + 1
                )
            )

            if idcg > 0:

                ndcg = dcg / idcg

            else:

                ndcg = 0.0

            precisions.append(
                precision
            )

            recalls.append(
                recall
            )

            hit_rates.append(
                hit_rate
            )

            ndcgs.append(
                ndcg
            )

            evaluated_users += 1

        if evaluated_users == 0:

            print(
                "No users available for Top-K evaluation."
            )

            return {
                "precision_at_k": 0.0,
                "recall_at_k": 0.0,
                "hit_rate_at_k": 0.0,
                "ndcg_at_k": 0.0,
                "evaluated_users": 0
            }

        metrics = {

            "precision_at_k":
                np.mean(precisions),

            "recall_at_k":
                np.mean(recalls),

            "hit_rate_at_k":
                np.mean(hit_rates),

            "ndcg_at_k":
                np.mean(ndcgs),

            "evaluated_users":
                evaluated_users
        }

        print()
        print(
            f"Precision@{top_k}:",
            round(
                metrics["precision_at_k"],
                4
            )
        )

        print(
            f"Recall@{top_k}:",
            round(
                metrics["recall_at_k"],
                4
            )
        )

        print(
            f"Hit Rate@{top_k}:",
            round(
                metrics["hit_rate_at_k"],
                4
            )
        )

        print(
            f"NDCG@{top_k}:",
            round(
                metrics["ndcg_at_k"],
                4
            )
        )

        print(
            "Evaluated users:",
            evaluated_users
        )

        return metrics


    # ============================================================
    # COVERAGE
    # ============================================================

    def calculate_coverage(
        self,
        user_ids,
        top_k=10
    ):

        recommended_movies = set()

        evaluated_users = 0

        for user_id in user_ids:

            recommendations = (
                self.generate_recommendations(
                    user_id,
                    top_k
                )
            )

            if recommendations.empty:
                continue

            evaluated_users += 1

            recommended_movies.update(
                recommendations[
                    "movie_id"
                ].tolist()
            )

        if self.num_movies == 0:

            coverage = 0.0

        else:

            coverage = (
                len(recommended_movies)
                /
                self.num_movies
            )

        print()
        print(
            f"Catalog Coverage@{top_k}:",
            round(
                coverage,
                4
            )
        )

        print(
            "Unique recommended movies:",
            len(recommended_movies)
        )

        return coverage


    # ============================================================
    # GENRE DIVERSITY
    # ============================================================

    def calculate_genre_diversity(
        self,
        user_id,
        top_k=10
    ):

        recommendations = (
            self.generate_recommendations(
                user_id,
                top_k
            )
        )

        if recommendations.empty:

            return 0

        genres = set()

        for genre_text in (
            recommendations["genre_text"]
        ):

            if pd.isna(genre_text):
                continue

            for genre in str(
                genre_text
            ).split():

                genres.add(
                    genre
                )

        diversity = len(
            genres
        )

        print()
        print(
            f"Genre Diversity@{top_k}:",
            diversity
        )

        print(
            "Genres:",
            ", ".join(
                sorted(genres)
            )
        )

        return diversity


    # ============================================================
    # SAVE FINAL RESULTS
    # ============================================================

    def save_results(
        self,
        rating_metrics,
        top_k_metrics,
        coverage,
        genre_diversity
    ):

        results = {

            "rmse":
                rating_metrics["rmse"],

            "mae":
                rating_metrics["mae"],

            "average_content_attention":
                rating_metrics[
                    "average_content_attention"
                ],

            "average_collaborative_attention":
                rating_metrics[
                    "average_collaborative_attention"
                ],

            f"precision_at_{TOP_K}":
                top_k_metrics[
                    "precision_at_k"
                ],

            f"recall_at_{TOP_K}":
                top_k_metrics[
                    "recall_at_k"
                ],

            f"hit_rate_at_{TOP_K}":
                top_k_metrics[
                    "hit_rate_at_k"
                ],

            f"ndcg_at_{TOP_K}":
                top_k_metrics[
                    "ndcg_at_k"
                ],

            f"catalog_coverage_at_{TOP_K}":
                coverage,

            f"genre_diversity_at_{TOP_K}":
                genre_diversity,

            "evaluated_users":
                top_k_metrics[
                    "evaluated_users"
                ]
        }

        results_df = pd.DataFrame(
            [results]
        )

        output_path = os.path.join(
            self.evaluation_dir,
            "day18_deep_hybrid_metrics.csv"
        )

        results_df.to_csv(
            output_path,
            index=False
        )

        print()
        print(
            "Metrics saved:"
        )

        print(
            output_path
        )

        return results_df


# ================================================================
# MAIN
# ================================================================

if __name__ == "__main__":

    evaluator = Day18Evaluator()

    # ------------------------------------------------------------
    # 1. RATING PREDICTION
    # ------------------------------------------------------------

    rating_metrics = (
        evaluator.evaluate_rating_prediction()
    )

    # ------------------------------------------------------------
    # 2. TOP-K EVALUATION
    # ------------------------------------------------------------

    evaluation_users = sorted(
        evaluator.test_ratings[
            "user_id"
        ].unique()
    )

    top_k_metrics = (
        evaluator.evaluate_top_k(
            evaluation_users,
            TOP_K
        )
    )

    # ------------------------------------------------------------
    # 3. COVERAGE
    # ------------------------------------------------------------

    # Use a smaller user sample for recommendation-based
    # evaluation to avoid unnecessary CPU time.

    coverage_users = evaluation_users[
        :min(
            100,
            len(evaluation_users)
        )
    ]

    coverage = (
        evaluator.calculate_coverage(
            coverage_users,
            TOP_K
        )
    )

    # ------------------------------------------------------------
    # 4. GENRE DIVERSITY
    # ------------------------------------------------------------

    test_user_id = 1

    genre_diversity = (
        evaluator.calculate_genre_diversity(
            test_user_id,
            TOP_K
        )
    )

    # ------------------------------------------------------------
    # 5. FINAL RESULTS
    # ------------------------------------------------------------

    results_df = evaluator.save_results(
        rating_metrics,
        top_k_metrics,
        coverage,
        genre_diversity
    )

    # ------------------------------------------------------------
    # 6. DISPLAY FINAL SUMMARY
    # ------------------------------------------------------------

    print()
    print("=" * 70)
    print("DAY 18 FINAL EVALUATION SUMMARY")
    print("=" * 70)

    print()

    print(
        results_df.to_string(
            index=False
        )
    )

    print()
    print("=" * 70)
    print("DAY 18 COMPLETED SUCCESSFULLY")
    print("=" * 70)