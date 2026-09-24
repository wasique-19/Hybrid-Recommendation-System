import os
import sys


# ============================================================
# PROJECT ROOT
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


if BASE_DIR not in sys.path:

    sys.path.insert(
        0,
        BASE_DIR
    )


# ============================================================
# IMPORT RECOMMENDER
# ============================================================

from app.deep_hybrid_recommender import (
    DeepHybridRecommender
)


# ============================================================
# DAY 17
# ============================================================

print("=" * 70)
print("DAY 17 - DEEP HYBRID RECOMMENDATION ENGINE")
print("=" * 70)


# ============================================================
# LOAD RECOMMENDER
# ============================================================

print()
print("Loading Deep Hybrid Recommender...")


recommender = DeepHybridRecommender()


# ============================================================
# TEST USER
# ============================================================

test_user_id = 1


print()
print("=" * 70)
print(
    f"RECOMMENDATIONS FOR USER {test_user_id}"
)
print("=" * 70)


recommendations = recommender.recommend(
    user_id=test_user_id,
    top_k=10
)


print()
print(
    recommendations.to_string(
        index=False
    )
)


# ============================================================
# COLD START TEST
# ============================================================

cold_start_user = 99999


print()
print("=" * 70)
print("COLD-START TEST")
print("=" * 70)


cold_recommendations = recommender.recommend(
    user_id=cold_start_user,
    top_k=10
)


print()
print(
    cold_recommendations.to_string(
        index=False
    )
)


# ============================================================
# SAVE RESULTS
# ============================================================

evaluation_dir = os.path.join(
    BASE_DIR,
    "evaluation"
)


os.makedirs(
    evaluation_dir,
    exist_ok=True
)


user_output_path = os.path.join(
    evaluation_dir,
    "day17_deep_hybrid_recommendations.csv"
)


cold_output_path = os.path.join(
    evaluation_dir,
    "day17_cold_start_recommendations.csv"
)


recommendations.to_csv(
    user_output_path,
    index=False
)


cold_recommendations.to_csv(
    cold_output_path,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 70)
print("DAY 17 COMPLETED SUCCESSFULLY")
print("=" * 70)


print()
print(
    "User recommendations saved:"
)


print(
    user_output_path
)


print()
print(
    "Cold-start recommendations saved:"
)


print(
    cold_output_path
)


print()
print(
    f"Recommendation count: "
    f"{len(recommendations)}"
)


if len(recommendations) > 0:

    print()

    print(
        "Average predicted rating: "
        f"{recommendations['predicted_rating'].mean():.4f}"
    )

    print(
        "Average content attention: "
        f"{recommendations['content_attention'].mean():.4f}"
    )

    print(
        "Average collaborative attention: "
        f"{recommendations['collaborative_attention'].mean():.4f}"
    )