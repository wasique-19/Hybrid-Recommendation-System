import os
import time
import json
import requests
import pandas as pd


# ============================================================
# DAY 21 - API TESTING
# ============================================================

print("=" * 70)
print("DAY 21 - RECOMMENDATION API TESTING")
print("=" * 70)


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

EVALUATION_DIR = os.path.join(
    BASE_DIR,
    "evaluation"
)

os.makedirs(
    EVALUATION_DIR,
    exist_ok=True
)


# ============================================================
# API CONFIGURATION
# ============================================================

BASE_URL = "http://127.0.0.1:8000"

TIMEOUT = 60


# ============================================================
# TEST CONFIGURATION
# ============================================================

TEST_USER_ID = 1
COLD_START_TOP_K = 10
RECOMMENDATION_TOP_K = 10
TEST_MOVIE_ID = 50


# ============================================================
# TEST RESULTS
# ============================================================

results = []


# ============================================================
# HELPER FUNCTION
# ============================================================

def test_endpoint(
    test_name,
    method,
    endpoint,
    expected_status=200
):
    """
    Test one API endpoint and store the result.
    """

    url = BASE_URL + endpoint

    print("\n" + "-" * 70)
    print(f"TEST: {test_name}")
    print(f"METHOD: {method}")
    print(f"URL: {url}")
    print("-" * 70)

    start_time = time.perf_counter()

    try:

        if method.upper() == "GET":

            response = requests.get(
                url,
                timeout=TIMEOUT
            )

        else:

            raise ValueError(
                f"Unsupported HTTP method: {method}"
            )

        end_time = time.perf_counter()

        latency_ms = (
            end_time - start_time
        ) * 1000

        status_code = response.status_code

        success = (
            status_code == expected_status
        )

        response_size = len(
            response.content
        )

        print(
            f"Status Code: {status_code}"
        )

        print(
            f"Expected Status: {expected_status}"
        )

        print(
            f"Latency: {latency_ms:.2f} ms"
        )

        print(
            f"Response Size: {response_size} bytes"
        )

        # ----------------------------------------------------
        # Parse JSON
        # ----------------------------------------------------

        json_data = None

        try:

            json_data = response.json()

        except Exception:

            json_data = None

        if success:

            print("Result: PASS")

        else:

            print("Result: FAIL")

            print(
                "Response:",
                response.text[:500]
            )

        # ----------------------------------------------------
        # Extract useful API information
        # ----------------------------------------------------

        recommendation_count = None
        movie_id = None
        user_id = None
        api_status = None

        if isinstance(
            json_data,
            dict
        ):

            api_status = json_data.get(
                "status"
            )

            user_id = json_data.get(
                "user_id"
            )

            movie = json_data.get(
                "movie"
            )

            if isinstance(
                movie,
                dict
            ):

                movie_id = movie.get(
                    "movie_id"
                )

            recommendation_count = (
                json_data.get(
                    "recommendation_count"
                )
            )

        # ----------------------------------------------------
        # Save result
        # ----------------------------------------------------

        results.append({

            "test_name":
                test_name,

            "method":
                method.upper(),

            "endpoint":
                endpoint,

            "expected_status":
                expected_status,

            "actual_status":
                status_code,

            "success":
                success,

            "latency_ms":
                round(
                    latency_ms,
                    3
                ),

            "response_size_bytes":
                response_size,

            "api_status":
                api_status,

            "user_id":
                user_id,

            "movie_id":
                movie_id,

            "recommendation_count":
                recommendation_count
        })

        return response

    except requests.exceptions.ConnectionError as e:

        end_time = time.perf_counter()

        latency_ms = (
            end_time - start_time
        ) * 1000

        print(
            "ERROR: Could not connect to API."
        )

        print(
            "Make sure the FastAPI server is running:"
        )

        print(
            "python -m uvicorn app.api:app "
            "--host 127.0.0.1 --port 8000"
        )

        results.append({

            "test_name":
                test_name,

            "method":
                method.upper(),

            "endpoint":
                endpoint,

            "expected_status":
                expected_status,

            "actual_status":
                None,

            "success":
                False,

            "latency_ms":
                round(
                    latency_ms,
                    3
                ),

            "response_size_bytes":
                0,

            "api_status":
                None,

            "user_id":
                None,

            "movie_id":
                None,

            "recommendation_count":
                None
        })

        return None

    except Exception as e:

        end_time = time.perf_counter()

        latency_ms = (
            end_time - start_time
        ) * 1000

        print(
            f"ERROR: {e}"
        )

        results.append({

            "test_name":
                test_name,

            "method":
                method.upper(),

            "endpoint":
                endpoint,

            "expected_status":
                expected_status,

            "actual_status":
                None,

            "success":
                False,

            "latency_ms":
                round(
                    latency_ms,
                    3
                ),

            "response_size_bytes":
                0,

            "api_status":
                None,

            "user_id":
                None,

            "movie_id":
                None,

            "recommendation_count":
                None
        })

        return None


# ============================================================
# TEST 1 - ROOT
# ============================================================

root_response = test_endpoint(
    test_name="Root Endpoint",
    method="GET",
    endpoint="/",
    expected_status=200
)


# ============================================================
# TEST 2 - HEALTH
# ============================================================

health_response = test_endpoint(
    test_name="Health Endpoint",
    method="GET",
    endpoint="/health",
    expected_status=200
)


# ============================================================
# TEST 3 - PERSONALIZED RECOMMENDATIONS
# ============================================================

recommend_response = test_endpoint(
    test_name="Personalized Recommendations",
    method="GET",
    endpoint=(
        f"/recommend/{TEST_USER_ID}"
        f"?top_k={RECOMMENDATION_TOP_K}"
    ),
    expected_status=200
)


# ============================================================
# TEST 4 - COLD START
# ============================================================

cold_start_response = test_endpoint(
    test_name="Cold Start Recommendations",
    method="GET",
    endpoint=(
        f"/cold-start"
        f"?top_k={COLD_START_TOP_K}"
    ),
    expected_status=200
)


# ============================================================
# TEST 5 - MOVIE DETAILS
# ============================================================

movie_response = test_endpoint(
    test_name="Movie Details",
    method="GET",
    endpoint=(
        f"/movie/{TEST_MOVIE_ID}"
    ),
    expected_status=200
)


# ============================================================
# TEST 6 - USER DETAILS
# ============================================================

user_response = test_endpoint(
    test_name="User Details",
    method="GET",
    endpoint=(
        f"/user/{TEST_USER_ID}"
    ),
    expected_status=200
)


# ============================================================
# TEST 7 - INVALID MOVIE
# ============================================================

invalid_movie_response = test_endpoint(
    test_name="Invalid Movie Handling",
    method="GET",
    endpoint="/movie/999999",
    expected_status=404
)


# ============================================================
# TEST 8 - INVALID USER
# ============================================================

invalid_user_response = test_endpoint(
    test_name="Cold Start User Handling",
    method="GET",
    endpoint="/user/99999",
    expected_status=200
)


# ============================================================
# CREATE RESULTS DATAFRAME
# ============================================================

results_df = pd.DataFrame(
    results
)


# ============================================================
# SAVE TEST RESULTS
# ============================================================

results_path = os.path.join(
    EVALUATION_DIR,
    "day21_api_test_results.csv"
)

results_df.to_csv(
    results_path,
    index=False
)


# ============================================================
# TEST SUMMARY
# ============================================================

total_tests = len(
    results_df
)

passed_tests = int(
    results_df["success"].sum()
)

failed_tests = (
    total_tests
    - passed_tests
)

pass_rate = (
    passed_tests / total_tests * 100
    if total_tests > 0
    else 0
)

average_latency = (
    results_df["latency_ms"].mean()
    if total_tests > 0
    else 0
)

successful_latencies = (
    results_df.loc[
        results_df["success"] == True,
        "latency_ms"
    ]
)

if len(successful_latencies) > 0:

    average_success_latency = (
        successful_latencies.mean()
    )

    maximum_success_latency = (
        successful_latencies.max()
    )

    minimum_success_latency = (
        successful_latencies.min()
    )

else:

    average_success_latency = 0
    maximum_success_latency = 0
    minimum_success_latency = 0


# ============================================================
# PRINT TEST RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("DAY 21 API TEST SUMMARY")
print("=" * 70)

print(
    f"Total tests: {total_tests}"
)

print(
    f"Passed tests: {passed_tests}"
)

print(
    f"Failed tests: {failed_tests}"
)

print(
    f"Pass rate: {pass_rate:.2f}%"
)

print(
    f"Average latency: "
    f"{average_latency:.2f} ms"
)

print(
    f"Average successful latency: "
    f"{average_success_latency:.2f} ms"
)

print(
    f"Minimum successful latency: "
    f"{minimum_success_latency:.2f} ms"
)

print(
    f"Maximum successful latency: "
    f"{maximum_success_latency:.2f} ms"
)


# ============================================================
# PRINT DETAILED TABLE
# ============================================================

print("\n")
print("=" * 70)
print("ENDPOINT TEST RESULTS")
print("=" * 70)

display_columns = [
    "test_name",
    "endpoint",
    "actual_status",
    "success",
    "latency_ms"
]

print(
    results_df[
        display_columns
    ].to_string(
        index=False
    )
)


# ============================================================
# SAVE SUMMARY CSV
# ============================================================

summary_data = {

    "total_tests":
        total_tests,

    "passed_tests":
        passed_tests,

    "failed_tests":
        failed_tests,

    "pass_rate":
        round(
            pass_rate,
            4
        ),

    "average_latency_ms":
        round(
            average_latency,
            4
        ),

    "average_success_latency_ms":
        round(
            average_success_latency,
            4
        ),

    "minimum_success_latency_ms":
        round(
            minimum_success_latency,
            4
        ),

    "maximum_success_latency_ms":
        round(
            maximum_success_latency,
            4
        )
}

summary_df = pd.DataFrame(
    [
        summary_data
    ]
)

summary_path = os.path.join(
    EVALUATION_DIR,
    "day21_api_test_summary.csv"
)

summary_df.to_csv(
    summary_path,
    index=False
)


# ============================================================
# SAVE JSON REPORT
# ============================================================

json_report = {

    "day": 21,

    "api_base_url":
        BASE_URL,

    "total_tests":
        total_tests,

    "passed_tests":
        passed_tests,

    "failed_tests":
        failed_tests,

    "pass_rate":
        round(
            pass_rate,
            4
        ),

    "average_latency_ms":
        round(
            average_latency,
            4
        ),

    "average_success_latency_ms":
        round(
            average_success_latency,
            4
        ),

    "tests":
        results
}

json_path = os.path.join(
    EVALUATION_DIR,
    "day21_api_test_report.json"
)

with open(
    json_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        json_report,
        f,
        indent=4
    )


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n")
print("=" * 70)
print("SAVED FILES")
print("=" * 70)

print(
    f"API test results:"
    f"\n{results_path}"
)

print(
    f"\nAPI test summary:"
    f"\n{summary_path}"
)

print(
    f"\nAPI JSON report:"
    f"\n{json_path}"
)


# ============================================================
# FINAL STATUS
# ============================================================

print("\n")
print("=" * 70)

if failed_tests == 0:

    print(
        "DAY 21 API TESTING COMPLETED SUCCESSFULLY"
    )

else:

    print(
        "DAY 21 API TESTING COMPLETED "
        "WITH FAILURES"
    )

print("=" * 70)