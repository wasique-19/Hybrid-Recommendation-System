import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# ================================================================
# DAY 19 - IMPROVED RECOMMENDATION EVALUATION
# ================================================================

print("=" * 70)
print("DAY 19 - IMPROVED RECOMMENDATION EVALUATION")
print("=" * 70)


# ================================================================
# 1. DEVICE
# ================================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"\nUsing device: {device}")


# ================================================================
# 2. PATHS
# ================================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

PROCESSED_DIR = os.path.join(
    BASE_DIR,
    "processed_data"
)

MODELS_DIR = os.path.join(
    BASE_DIR,
    "models"
)

EVALUATION_DIR = os.path.join(
    BASE_DIR,
    "evaluation"
)

PLOTS_DIR = os.path.join(
    EVALUATION_DIR,
    "plots"
)

os.makedirs(EVALUATION_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)


# ================================================================
# 3. FILE PATHS
# ================================================================

RATINGS_PATH = os.path.join(
    PROCESSED_DIR,
    "ratings_clean.csv"
)

MOVIES_PATH = os.path.join(
    PROCESSED_DIR,
    "movie_features.csv"
)

CONTENT_EMBEDDINGS_PATH = os.path.join(
    MODELS_DIR,
    "day15_content_embeddings.npy"
)

MODEL_PATH = os.path.join(
    MODELS_DIR,
    "day16_attention_fusion_model.pth"
)


# ================================================================
# 4. LOAD DATA
# ================================================================

print("\n" + "=" * 70)
print("LOADING DATA")
print("=" * 70)

print(f"\nLoading ratings:")
print(RATINGS_PATH)

ratings = pd.read_csv(
    RATINGS_PATH
)

print("\nRatings shape:")
print(ratings.shape)


print("\nLoading movie features:")
print(MOVIES_PATH)

movies = pd.read_csv(
    MOVIES_PATH
)

print("\nMovies shape:")
print(movies.shape)


print("\nLoading content embeddings:")
print(CONTENT_EMBEDDINGS_PATH)

content_embeddings = np.load(
    CONTENT_EMBEDDINGS_PATH
)

print("\nContent embedding shape:")
print(content_embeddings.shape)


# ================================================================
# 5. BASIC INFORMATION
# ================================================================

all_user_ids = sorted(
    ratings["user_id"].unique()
)

all_movie_ids = sorted(
    ratings["movie_id"].unique()
)

print("\nNumber of users:")
print(len(all_user_ids))

print("Number of movies:")
print(len(all_movie_ids))


# ================================================================
# 6. TRAIN / TEST SPLIT
# ================================================================
#
# IMPORTANT:
# We create a deterministic split.
#
# For every user:
# - last 20% ratings -> test
# - first 80% ratings -> train
#
# This gives us test movies that were NOT in the training history.
# ================================================================

print("\n" + "=" * 70)
print("CREATING TRAIN / TEST SPLIT")
print("=" * 70)


ratings = ratings.sort_values(
    ["user_id", "timestamp"]
).reset_index(
    drop=True
)


train_parts = []
test_parts = []


for user_id, user_data in ratings.groupby(
    "user_id"
):

    user_data = user_data.sort_values(
        "timestamp"
    )

    n = len(user_data)

    if n < 5:
        train_parts.append(
            user_data.iloc[:-1]
        )

        test_parts.append(
            user_data.iloc[-1:]
        )

    else:
        split_index = int(
            n * 0.8
        )

        train_parts.append(
            user_data.iloc[
                :split_index
            ]
        )

        test_parts.append(
            user_data.iloc[
                split_index:
            ]
        )


train_ratings = pd.concat(
    train_parts,
    ignore_index=True
)

test_ratings = pd.concat(
    test_parts,
    ignore_index=True
)


print("\nTrain samples:")
print(len(train_ratings))

print("Test samples:")
print(len(test_ratings))


# ================================================================
# 7. CREATE ID MAPPINGS
# ================================================================

print("\n" + "=" * 70)
print("CREATING ID MAPPINGS")
print("=" * 70)


user_mapping = {
    user_id: index
    for index, user_id
    in enumerate(all_user_ids)
}

movie_mapping = {
    movie_id: index
    for index, movie_id
    in enumerate(all_movie_ids)
}


num_users = len(
    user_mapping
)

num_movies = len(
    movie_mapping
)


print("\nUsers:")
print(num_users)

print("Movies:")
print(num_movies)


# ================================================================
# 8. MODEL ARCHITECTURE
# ================================================================

class AttentionFusion(nn.Module):

    def __init__(
        self,
        embedding_dim=32
    ):
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

        content_score = (
            self.attention_network(
                content_embedding
            )
        )

        collaborative_score = (
            self.attention_network(
                collaborative_embedding
            )
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

        content_weight = (
            attention_weights[:, 0:1]
        )

        collaborative_weight = (
            attention_weights[:, 1:2]
        )

        fused_embedding = (
            content_weight
            * content_embedding
            +
            collaborative_weight
            * collaborative_embedding
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

            nn.Dropout(
                0.2
            ),

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

        content_emb = (
            content_embeddings[
                movie_ids
            ]
        )

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
# 9. LOAD DAY 16 CHECKPOINT
# ================================================================

print("\n" + "=" * 70)
print("LOADING DAY 16 MODEL")
print("=" * 70)

print("\nModel path:")
print(MODEL_PATH)


model = AttentionFusionModel(
    num_users=num_users,
    num_movies=num_movies,
    embedding_dim=32
)


checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=False
)


# Day 16 saved a full checkpoint.
if isinstance(
    checkpoint,
    dict
) and "model_state_dict" in checkpoint:

    print("\nCheckpoint format:")
    print("Full training checkpoint")

    state_dict = checkpoint[
        "model_state_dict"
    ]

else:

    print("\nCheckpoint format:")
    print("State dictionary")

    state_dict = checkpoint


model.load_state_dict(
    state_dict
)

model.to(
    device
)

model.eval()

print("\nModel loaded successfully.")


# ================================================================
# 10. CONTENT TENSOR
# ================================================================

content_tensor = torch.tensor(
    content_embeddings,
    dtype=torch.float32,
    device=device
)


# ================================================================
# 11. TRAIN HISTORY
# ================================================================

train_history = {}

for user_id, group in train_ratings.groupby(
    "user_id"
):

    train_history[user_id] = set(
        group["movie_id"].tolist()
    )


# ================================================================
# 12. TEST RELEVANT ITEMS
# ================================================================
#
# Rating >= 4 is considered relevant.
#
# This is the key improvement over Day 18.
# ================================================================

test_relevant = {}

for user_id, group in test_ratings.groupby(
    "user_id"
):

    relevant = set(
        group[
            group["rating"] >= 4
        ]["movie_id"].tolist()
    )

    if len(relevant) > 0:

        test_relevant[user_id] = (
            relevant
        )


print("\nUsers with relevant test items:")
print(
    len(test_relevant)
)


# ================================================================
# 13. RECOMMENDATION FUNCTION
# ================================================================

def generate_recommendations(
    user_id,
    top_k=10
):

    if user_id not in user_mapping:

        return []

    user_index = user_mapping[
        user_id
    ]

    watched_movies = train_history.get(
        user_id,
        set()
    )

    # IMPORTANT:
    # Only remove movies seen in TRAIN.
    # Test movies must remain candidates.
    candidate_movies = [
        movie_id
        for movie_id in all_movie_ids
        if movie_id not in watched_movies
    ]

    if len(candidate_movies) == 0:

        return []

    user_indices = [
        user_index
        for _ in candidate_movies
    ]

    movie_indices = [
        movie_mapping[movie_id]
        for movie_id in candidate_movies
    ]

    user_tensor = torch.tensor(
        user_indices,
        dtype=torch.long,
        device=device
    )

    movie_tensor = torch.tensor(
        movie_indices,
        dtype=torch.long,
        device=device
    )

    with torch.no_grad():

        predictions, attention = model(
            user_tensor,
            movie_tensor,
            content_tensor
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

    return result.head(
        top_k
    )


# ================================================================
# 14. METRIC FUNCTIONS
# ================================================================

def precision_at_k(
    recommended,
    relevant,
    k=10
):

    recommended = recommended[:k]

    if len(recommended) == 0:
        return 0.0

    hits = sum(
        movie_id in relevant
        for movie_id in recommended
    )

    return hits / len(recommended)


def recall_at_k(
    recommended,
    relevant,
    k=10
):

    if len(relevant) == 0:
        return 0.0

    recommended = recommended[:k]

    hits = sum(
        movie_id in relevant
        for movie_id in recommended
    )

    return hits / len(relevant)


def hit_rate_at_k(
    recommended,
    relevant,
    k=10
):

    recommended = recommended[:k]

    for movie_id in recommended:

        if movie_id in relevant:

            return 1.0

    return 0.0


def ndcg_at_k(
    recommended,
    relevant,
    k=10
):

    recommended = recommended[:k]

    dcg = 0.0

    for index, movie_id in enumerate(
        recommended
    ):

        if movie_id in relevant:

            rank = index + 1

            dcg += (
                1.0
                /
                np.log2(
                    rank + 1
                )
            )

    ideal_hits = min(
        len(relevant),
        k
    )

    if ideal_hits == 0:

        return 0.0

    idcg = sum(

        1.0
        /
        np.log2(
            rank + 1
        )

        for rank in range(
            1,
            ideal_hits + 1
        )
    )

    if idcg == 0:

        return 0.0

    return dcg / idcg


def average_precision_at_k(
    recommended,
    relevant,
    k=10
):

    if len(relevant) == 0:

        return 0.0

    recommended = recommended[:k]

    score = 0.0
    hits = 0

    for index, movie_id in enumerate(
        recommended
    ):

        if movie_id in relevant:

            hits += 1

            precision = (
                hits
                /
                (index + 1)
            )

            score += precision

    denominator = min(
        len(relevant),
        k
    )

    if denominator == 0:

        return 0.0

    return score / denominator


# ================================================================
# 15. GENRE DIVERSITY
# ================================================================

def calculate_genre_diversity(
    recommendation_ids
):

    if len(recommendation_ids) == 0:

        return 0, set()

    selected_movies = movies[
        movies["movie_id"].isin(
            recommendation_ids
        )
    ]

    genres = set()

    for genre_text in selected_movies[
        "genre_text"
    ].fillna(""):

        for genre in str(
            genre_text
        ).split():

            if genre.strip():

                genres.add(
                    genre.strip()
                )

    return (
        len(genres),
        genres
    )


# ================================================================
# 16. EVALUATION
# ================================================================

print("\n" + "=" * 70)
print("TOP-10 RECOMMENDATION EVALUATION")
print("=" * 70)


user_results = []

all_recommended_movies = set()

content_attention_values = []
collaborative_attention_values = []


for counter, user_id in enumerate(
    test_relevant.keys(),
    start=1
):

    relevant_movies = test_relevant[
        user_id
    ]

    recommendations = (
        generate_recommendations(
            user_id,
            top_k=10
        )
    )

    if len(recommendations) == 0:

        continue

    recommended_ids = (
        recommendations[
            "movie_id"
        ].tolist()
    )

    all_recommended_movies.update(
        recommended_ids
    )

    content_attention_values.extend(
        recommendations[
            "content_attention"
        ].tolist()
    )

    collaborative_attention_values.extend(
        recommendations[
            "collaborative_attention"
        ].tolist()
    )

    precision = precision_at_k(
        recommended_ids,
        relevant_movies,
        10
    )

    recall = recall_at_k(
        recommended_ids,
        relevant_movies,
        10
    )

    hit_rate = hit_rate_at_k(
        recommended_ids,
        relevant_movies,
        10
    )

    ndcg = ndcg_at_k(
        recommended_ids,
        relevant_movies,
        10
    )

    map_score = average_precision_at_k(
        recommended_ids,
        relevant_movies,
        10
    )

    user_results.append({

        "user_id":
            user_id,

        "relevant_item_count":
            len(relevant_movies),

        "recommendation_count":
            len(recommended_ids),

        "precision_at_10":
            precision,

        "recall_at_10":
            recall,

        "hit_rate_at_10":
            hit_rate,

        "ndcg_at_10":
            ndcg,

        "map_at_10":
            map_score
    })


    if counter <= 5:

        print(
            f"\nUser {user_id}"
        )

        print(
            "Relevant movies:",
            sorted(
                relevant_movies
            )
        )

        print(
            "Recommended movies:",
            recommended_ids
        )

        print(
            f"Precision@10: {precision:.4f}"
        )

        print(
            f"Recall@10: {recall:.4f}"
        )

        print(
            f"Hit Rate@10: {hit_rate:.4f}"
        )

        print(
            f"NDCG@10: {ndcg:.4f}"
        )

        print(
            f"MAP@10: {map_score:.4f}"
        )


# ================================================================
# 17. USER-LEVEL DATAFRAME
# ================================================================

user_results_df = pd.DataFrame(
    user_results
)


if len(user_results_df) == 0:

    print(
        "\nERROR: No users could be evaluated."
    )

    raise SystemExit


# ================================================================
# 18. GLOBAL METRICS
# ================================================================

precision_mean = (
    user_results_df[
        "precision_at_10"
    ].mean()
)

recall_mean = (
    user_results_df[
        "recall_at_10"
    ].mean()
)

hit_rate_mean = (
    user_results_df[
        "hit_rate_at_10"
    ].mean()
)

ndcg_mean = (
    user_results_df[
        "ndcg_at_10"
    ].mean()
)

map_mean = (
    user_results_df[
        "map_at_10"
    ].mean()
)


# ================================================================
# 19. CATALOG COVERAGE
# ================================================================

catalog_coverage = (
    len(all_recommended_movies)
    /
    len(all_movie_ids)
)


# ================================================================
# 20. GENRE DIVERSITY
# ================================================================

genre_diversity, unique_genres = (
    calculate_genre_diversity(
        list(all_recommended_movies)
    )
)


# ================================================================
# 21. ATTENTION
# ================================================================

average_content_attention = (
    np.mean(
        content_attention_values
    )
)

average_collaborative_attention = (
    np.mean(
        collaborative_attention_values
    )
)


# ================================================================
# 22. FINAL METRICS
# ================================================================

metrics = {

    "precision_at_10":
        precision_mean,

    "recall_at_10":
        recall_mean,

    "hit_rate_at_10":
        hit_rate_mean,

    "ndcg_at_10":
        ndcg_mean,

    "map_at_10":
        map_mean,

    "catalog_coverage_at_10":
        catalog_coverage,

    "unique_recommended_movies":
        len(all_recommended_movies),

    "genre_diversity_at_10":
        genre_diversity,

    "average_content_attention":
        average_content_attention,

    "average_collaborative_attention":
        average_collaborative_attention,

    "evaluated_users":
        len(user_results_df)
}


# ================================================================
# 23. SAVE USER RESULTS
# ================================================================

user_results_path = os.path.join(
    EVALUATION_DIR,
    "day19_user_evaluation.csv"
)

user_results_df.to_csv(
    user_results_path,
    index=False
)


# ================================================================
# 24. SAVE METRICS
# ================================================================

metrics_df = pd.DataFrame(
    [metrics]
)

metrics_path = os.path.join(
    EVALUATION_DIR,
    "day19_recommendation_metrics.csv"
)

metrics_df.to_csv(
    metrics_path,
    index=False
)


# ================================================================
# 25. SAVE RECOMMENDATION DETAILS
# ================================================================

recommendation_rows = []


for user_id in test_relevant.keys():

    recommendations = (
        generate_recommendations(
            user_id,
            top_k=10
        )
    )

    if len(recommendations) == 0:

        continue

    for rank, row in enumerate(
        recommendations.itertuples(
            index=False
        ),
        start=1
    ):

        recommendation_rows.append({

            "user_id":
                user_id,

            "rank":
                rank,

            "movie_id":
                row.movie_id,

            "predicted_rating":
                row.predicted_rating,

            "content_attention":
                row.content_attention,

            "collaborative_attention":
                row.collaborative_attention
        })


recommendations_df = pd.DataFrame(
    recommendation_rows
)


recommendations_path = os.path.join(
    EVALUATION_DIR,
    "day19_recommendations.csv"
)

recommendations_df.to_csv(
    recommendations_path,
    index=False
)


# ================================================================
# 26. OPTIONAL PLOTS
# ================================================================

try:

    import matplotlib.pyplot as plt


    # ------------------------------------------------------------
    # Precision / Recall / Hit Rate
    # ------------------------------------------------------------

    plt.figure(
        figsize=(8, 5)
    )

    metric_names = [
        "Precision@10",
        "Recall@10",
        "Hit Rate@10"
    ]

    metric_values = [
        precision_mean,
        recall_mean,
        hit_rate_mean
    ]

    plt.bar(
        metric_names,
        metric_values
    )

    plt.ylabel(
        "Score"
    )

    plt.title(
        "Day 19 Recommendation Metrics"
    )

    plt.tight_layout()

    precision_recall_path = os.path.join(
        PLOTS_DIR,
        "day19_precision_recall.png"
    )

    plt.savefig(
        precision_recall_path,
        dpi=150
    )

    plt.close()


    # ------------------------------------------------------------
    # NDCG / MAP
    # ------------------------------------------------------------

    plt.figure(
        figsize=(8, 5)
    )

    metric_names = [
        "NDCG@10",
        "MAP@10"
    ]

    metric_values = [
        ndcg_mean,
        map_mean
    ]

    plt.bar(
        metric_names,
        metric_values
    )

    plt.ylabel(
        "Score"
    )

    plt.title(
        "Day 19 Ranking Metrics"
    )

    plt.tight_layout()

    ndcg_path = os.path.join(
        PLOTS_DIR,
        "day19_ndcg.png"
    )

    plt.savefig(
        ndcg_path,
        dpi=150
    )

    plt.close()


    # ------------------------------------------------------------
    # Catalog Coverage
    # ------------------------------------------------------------

    plt.figure(
        figsize=(6, 5)
    )

    plt.bar(
        ["Catalog Coverage@10"],
        [catalog_coverage]
    )

    plt.ylabel(
        "Coverage"
    )

    plt.title(
        "Day 19 Catalog Coverage"
    )

    plt.tight_layout()

    coverage_path = os.path.join(
        PLOTS_DIR,
        "day19_catalog_coverage.png"
    )

    plt.savefig(
        coverage_path,
        dpi=150
    )

    plt.close()


    print("\nPlots saved:")
    print(precision_recall_path)
    print(ndcg_path)
    print(coverage_path)

except Exception as e:

    print(
        "\nPlot generation skipped:"
    )

    print(e)


# ================================================================
# 27. FINAL OUTPUT
# ================================================================

print("\n" + "=" * 70)
print("DAY 19 FINAL EVALUATION SUMMARY")
print("=" * 70)

print(
    f"\nPrecision@10: "
    f"{precision_mean:.4f}"
)

print(
    f"Recall@10: "
    f"{recall_mean:.4f}"
)

print(
    f"Hit Rate@10: "
    f"{hit_rate_mean:.4f}"
)

print(
    f"NDCG@10: "
    f"{ndcg_mean:.4f}"
)

print(
    f"MAP@10: "
    f"{map_mean:.4f}"
)

print(
    f"Catalog Coverage@10: "
    f"{catalog_coverage:.4f}"
)

print(
    f"Unique recommended movies: "
    f"{len(all_recommended_movies)}"
)

print(
    f"Genre Diversity@10: "
    f"{genre_diversity}"
)

print(
    f"Genres: "
    f"{', '.join(sorted(unique_genres))}"
)

print(
    f"Average Content Attention: "
    f"{average_content_attention:.4f}"
)

print(
    f"Average Collaborative Attention: "
    f"{average_collaborative_attention:.4f}"
)

print(
    f"Evaluated users: "
    f"{len(user_results_df)}"
)


print("\nMetrics saved:")
print(metrics_path)

print("\nUser evaluation saved:")
print(user_results_path)

print("\nRecommendation details saved:")
print(recommendations_path)


print("\n" + "=" * 70)
print("DAY 19 COMPLETED SUCCESSFULLY")
print("=" * 70)