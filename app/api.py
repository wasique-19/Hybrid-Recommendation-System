import os
import sys
import traceback

from fastapi import FastAPI, HTTPException, Query

# ---------------------------------------------------------
# PROJECT PATH
# ---------------------------------------------------------

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)


# ---------------------------------------------------------
# IMPORT RECOMMENDER
# ---------------------------------------------------------

try:
    from app.deep_hybrid_recommender import DeepHybridRecommender
except Exception as e:
    print("ERROR: Could not import DeepHybridRecommender")
    print(e)
    traceback.print_exc()
    raise


# ---------------------------------------------------------
# FASTAPI APP
# ---------------------------------------------------------

app = FastAPI(
    title="Hybrid Recommendation System API",
    description=(
        "Day 21 API for the Deep Hybrid Movie "
        "Recommendation System"
    ),
    version="1.0.0"
)


# ---------------------------------------------------------
# LOAD MODEL ONCE
# ---------------------------------------------------------

print("=" * 70)
print("DAY 21 - STARTING RECOMMENDATION API")
print("=" * 70)

try:
    recommender = DeepHybridRecommender()

    print("Deep Hybrid Recommender loaded successfully.")

except Exception as e:
    print("ERROR: Failed to load recommender.")
    print(e)
    traceback.print_exc()
    raise


# ---------------------------------------------------------
# ROOT / HEALTH CHECK
# ---------------------------------------------------------

@app.get("/")
def root():

    return {
        "status": "success",
        "message": "Hybrid Recommendation System API is running",
        "day": 21,
        "model": "Deep Hybrid Recommendation Engine",
        "device": str(recommender.device)
    }


# ---------------------------------------------------------
# HEALTH ENDPOINT
# ---------------------------------------------------------

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model_loaded": True,
        "device": str(recommender.device),
        "num_users": recommender.num_users,
        "num_movies": recommender.num_movies
    }


# ---------------------------------------------------------
# PERSONALIZED RECOMMENDATIONS
# ---------------------------------------------------------

@app.get("/recommend/{user_id}")
def recommend(
    user_id: int,
    top_k: int = Query(
        default=10,
        ge=1,
        le=100
    )
):

    try:

        recommendations = recommender.recommend(
            user_id=user_id,
            top_k=top_k
        )

        return {
            "status": "success",
            "user_id": user_id,
            "recommendation_count": len(
                recommendations
            ),
            "recommendations": (
                recommendations
                .replace({float("nan"): None})
                .to_dict(
                    orient="records"
                )
            )
        }

    except Exception as e:

        print(
            f"Recommendation error for user "
            f"{user_id}: {e}"
        )

        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# ---------------------------------------------------------
# COLD START
# ---------------------------------------------------------

@app.get("/cold-start")
def cold_start(
    top_k: int = Query(
        default=10,
        ge=1,
        le=100
    )
):

    try:

        recommendations = (
            recommender._cold_start_recommendations(
                top_k
            )
        )

        return {
            "status": "success",
            "recommendation_type": "cold_start",
            "recommendation_count": len(
                recommendations
            ),
            "recommendations": (
                recommendations
                .replace({float("nan"): None})
                .to_dict(
                    orient="records"
                )
            )
        }

    except Exception as e:

        print(
            f"Cold-start error: {e}"
        )

        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# ---------------------------------------------------------
# MOVIE DETAILS
# ---------------------------------------------------------

@app.get("/movie/{movie_id}")
def movie_details(
    movie_id: int
):

    try:

        movie = recommender.movies[
            recommender.movies["movie_id"]
            == movie_id
        ]

        if movie.empty:

            raise HTTPException(
                status_code=404,
                detail=(
                    f"Movie ID {movie_id} "
                    f"not found"
                )
            )

        result = (
            movie
            .replace({float("nan"): None})
            .iloc[0]
            .to_dict()
        )

        return {
            "status": "success",
            "movie": result
        }

    except HTTPException:
        raise

    except Exception as e:

        print(
            f"Movie lookup error: {e}"
        )

        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# ---------------------------------------------------------
# USER CHECK
# ---------------------------------------------------------

@app.get("/user/{user_id}")
def user_details(
    user_id: int
):

    if user_id not in recommender.user_mapping:

        return {
            "status": "success",
            "user_id": user_id,
            "exists": False,
            "recommendation_type": "cold_start"
        }

    watched_movies = (
        recommender._get_watched_movies(
            user_id
        )
    )

    return {
        "status": "success",
        "user_id": user_id,
        "exists": True,
        "watched_movie_count": len(
            watched_movies
        )
    }


# ---------------------------------------------------------
# RUN MESSAGE
# ---------------------------------------------------------

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "app.api:app",
        host="127.0.0.1",
        port=8000,
        reload=False
    )