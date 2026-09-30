"""
===========================================================================
DAY 23 - API MONITORING
===========================================================================

Purpose:
    Monitor the Day 21 Recommendation API.

Features:
    1. Health monitoring
    2. Endpoint latency monitoring
    3. Status-code monitoring
    4. Recommendation API monitoring
    5. Cold-start monitoring
    6. Movie/User endpoint monitoring
    7. Multiple requests per endpoint
    8. Error-rate calculation
    9. Average / min / max / P95 latency
    10. Monitoring CSV reports
    11. Monitoring JSON report
    12. Summary CSV
    13. Latency plot
    14. Status/error plot
    15. Endpoint comparison plot

API:
    http://127.0.0.1:8000

Run:
    python scripts/day23_api_monitoring.py

Make sure Day 21 API is running before starting this script.
"""

import os
import json
import time
import statistics
from datetime import datetime

import requests
import pandas as pd
import matplotlib.pyplot as plt


# ===========================================================================
# CONFIGURATION
# ===========================================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
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


API_BASE_URL = "http://127.0.0.1:8000"

REQUEST_TIMEOUT = 10

REQUESTS_PER_ENDPOINT = 5


# ===========================================================================
# ENDPOINT CONFIGURATION
# ===========================================================================

ENDPOINTS = [
    {
        "name": "Root Endpoint",
        "method": "GET",
        "path": "/",
        "expected_status": 200,
    },
    {
        "name": "Health Endpoint",
        "method": "GET",
        "path": "/health",
        "expected_status": 200,
    },
    {
        "name": "Personalized Recommendations",
        "method": "GET",
        "path": "/recommend/1?top_k=10",
        "expected_status": 200,
    },
    {
        "name": "Cold Start Recommendations",
        "method": "GET",
        "path": "/cold-start?top_k=10",
        "expected_status": 200,
    },
    {
        "name": "Movie Details",
        "method": "GET",
        "path": "/movie/50",
        "expected_status": 200,
    },
    {
        "name": "User Details",
        "method": "GET",
        "path": "/user/1",
        "expected_status": 200,
    },
    {
        "name": "Invalid Movie Handling",
        "method": "GET",
        "path": "/movie/999999",
        "expected_status": 404,
    },
    {
        "name": "Cold Start User Handling",
        "method": "GET",
        "path": "/user/99999",
        "expected_status": 200,
    },
]


# ===========================================================================
# HELPER FUNCTIONS
# ===========================================================================

def percentile(values, p):
    """
    Calculate percentile without requiring numpy.
    """

    if not values:
        return 0.0

    values = sorted(values)

    if len(values) == 1:
        return float(values[0])

    k = (len(values) - 1) * (p / 100.0)

    lower = int(k)
    upper = min(lower + 1, len(values) - 1)

    weight = k - lower

    return (
        values[lower]
        + weight * (values[upper] - values[lower])
    )


def safe_json(response):
    """
    Safely parse JSON response.
    """

    try:
        return response.json()
    except Exception:
        return None


def response_size(response):
    """
    Return response size in bytes.
    """

    try:
        return len(response.content)
    except Exception:
        return 0


def check_api_available():
    """
    Check whether API is running.
    """

    url = API_BASE_URL + "/health"

    print("=" * 75)
    print("CHECKING API AVAILABILITY")
    print("=" * 75)

    print(f"API URL: {API_BASE_URL}")
    print(f"Health URL: {url}")

    try:

        start = time.perf_counter()

        response = requests.get(
            url,
            timeout=REQUEST_TIMEOUT
        )

        latency = (
            time.perf_counter() - start
        ) * 1000

        print(
            f"Status Code: {response.status_code}"
        )

        print(
            f"Latency: {latency:.2f} ms"
        )

        if response.status_code == 200:

            print("API Status: ONLINE")
            print()

            return True

        print("API Status: UNHEALTHY")
        print()

        return False

    except requests.exceptions.RequestException as exc:

        print("API Status: OFFLINE")
        print(f"Error: {exc}")
        print()

        return False


# ===========================================================================
# SINGLE REQUEST
# ===========================================================================

def perform_request(endpoint):
    """
    Execute one endpoint request.
    """

    url = API_BASE_URL + endpoint["path"]

    start_time = time.perf_counter()

    try:

        response = requests.request(
            method=endpoint["method"],
            url=url,
            timeout=REQUEST_TIMEOUT
        )

        end_time = time.perf_counter()

        latency_ms = (
            end_time - start_time
        ) * 1000

        actual_status = response.status_code

        success = (
            actual_status
            == endpoint["expected_status"]
        )

        data = safe_json(response)

        return {
            "timestamp": datetime.now().isoformat(),
            "test_name": endpoint["name"],
            "method": endpoint["method"],
            "endpoint": endpoint["path"],
            "url": url,
            "expected_status": endpoint["expected_status"],
            "actual_status": actual_status,
            "success": success,
            "latency_ms": round(latency_ms, 3),
            "response_size_bytes": response_size(response),
            "response_json": (
                json.dumps(data)
                if data is not None
                else ""
            ),
            "error": "",
        }

    except requests.exceptions.Timeout as exc:

        end_time = time.perf_counter()

        latency_ms = (
            end_time - start_time
        ) * 1000

        return {
            "timestamp": datetime.now().isoformat(),
            "test_name": endpoint["name"],
            "method": endpoint["method"],
            "endpoint": endpoint["path"],
            "url": url,
            "expected_status": endpoint["expected_status"],
            "actual_status": 0,
            "success": False,
            "latency_ms": round(latency_ms, 3),
            "response_size_bytes": 0,
            "response_json": "",
            "error": f"Timeout: {exc}",
        }

    except requests.exceptions.RequestException as exc:

        end_time = time.perf_counter()

        latency_ms = (
            end_time - start_time
        ) * 1000

        return {
            "timestamp": datetime.now().isoformat(),
            "test_name": endpoint["name"],
            "method": endpoint["method"],
            "endpoint": endpoint["path"],
            "url": url,
            "expected_status": endpoint["expected_status"],
            "actual_status": 0,
            "success": False,
            "latency_ms": round(latency_ms, 3),
            "response_size_bytes": 0,
            "response_json": "",
            "error": f"Request error: {exc}",
        }

    except Exception as exc:

        end_time = time.perf_counter()

        latency_ms = (
            end_time - start_time
        ) * 1000

        return {
            "timestamp": datetime.now().isoformat(),
            "test_name": endpoint["name"],
            "method": endpoint["method"],
            "endpoint": endpoint["path"],
            "url": url,
            "expected_status": endpoint["expected_status"],
            "actual_status": 0,
            "success": False,
            "latency_ms": round(latency_ms, 3),
            "response_size_bytes": 0,
            "response_json": "",
            "error": f"Unexpected error: {exc}",
        }


# ===========================================================================
# MONITOR ALL ENDPOINTS
# ===========================================================================

def monitor_endpoints():
    """
    Execute monitoring requests for all configured endpoints.
    """

    print("=" * 75)
    print("DAY 23 - API MONITORING")
    print("=" * 75)

    print()
    print(f"API Base URL: {API_BASE_URL}")
    print(
        f"Requests per endpoint: "
        f"{REQUESTS_PER_ENDPOINT}"
    )

    print()

    all_results = []

    for endpoint in ENDPOINTS:

        print("-" * 75)

        print(
            f"Monitoring: "
            f"{endpoint['name']}"
        )

        print(
            f"Endpoint: "
            f"{endpoint['path']}"
        )

        print(
            f"Expected Status: "
            f"{endpoint['expected_status']}"
        )

        print()

        for request_number in range(
            1,
            REQUESTS_PER_ENDPOINT + 1
        ):

            result = perform_request(
                endpoint
            )

            result["request_number"] = (
                request_number
            )

            all_results.append(result)

            status_text = (
                "PASS"
                if result["success"]
                else "FAIL"
            )

            print(
                f"Request "
                f"{request_number}/"
                f"{REQUESTS_PER_ENDPOINT} | "
                f"Status: "
                f"{result['actual_status']} | "
                f"Latency: "
                f"{result['latency_ms']:.2f} ms | "
                f"{status_text}"
            )

    print()

    return pd.DataFrame(all_results)


# ===========================================================================
# ENDPOINT SUMMARY
# ===========================================================================

def create_endpoint_summary(results_df):
    """
    Create endpoint-level monitoring summary.
    """

    rows = []

    for endpoint_name, group in results_df.groupby(
        "test_name"
    ):

        latencies = (
            group["latency_ms"]
            .astype(float)
            .tolist()
        )

        total_requests = len(group)

        successful_requests = int(
            group["success"].sum()
        )

        failed_requests = (
            total_requests
            - successful_requests
        )

        success_rate = (
            successful_requests
            / total_requests
            * 100
            if total_requests > 0
            else 0
        )

        error_rate = (
            failed_requests
            / total_requests
            * 100
            if total_requests > 0
            else 0
        )

        rows.append(
            {
                "test_name": endpoint_name,
                "endpoint": group[
                    "endpoint"
                ].iloc[0],
                "total_requests": total_requests,
                "successful_requests": (
                    successful_requests
                ),
                "failed_requests": (
                    failed_requests
                ),
                "success_rate": round(
                    success_rate,
                    4
                ),
                "error_rate": round(
                    error_rate,
                    4
                ),
                "average_latency_ms": round(
                    statistics.mean(latencies),
                    4
                ),
                "median_latency_ms": round(
                    statistics.median(latencies),
                    4
                ),
                "min_latency_ms": round(
                    min(latencies),
                    4
                ),
                "max_latency_ms": round(
                    max(latencies),
                    4
                ),
                "p95_latency_ms": round(
                    percentile(
                        latencies,
                        95
                    ),
                    4
                ),
                "average_response_size_bytes": round(
                    group[
                        "response_size_bytes"
                    ].mean(),
                    2
                ),
            }
        )

    return pd.DataFrame(rows)


# ===========================================================================
# OVERALL SUMMARY
# ===========================================================================

def create_overall_summary(
    results_df,
    endpoint_summary
):
    """
    Create overall monitoring summary.
    """

    total_requests = len(results_df)

    successful_requests = int(
        results_df["success"].sum()
    )

    failed_requests = (
        total_requests
        - successful_requests
    )

    latencies = (
        results_df["latency_ms"]
        .astype(float)
        .tolist()
    )

    success_rate = (
        successful_requests
        / total_requests
        * 100
        if total_requests > 0
        else 0
    )

    error_rate = (
        failed_requests
        / total_requests
        * 100
        if total_requests > 0
        else 0
    )

    status_200_count = int(
        (results_df["actual_status"] == 200).sum()
    )

    status_404_count = int(
        (results_df["actual_status"] == 404).sum()
    )

    status_500_count = int(
        (results_df["actual_status"] == 500).sum()
    )

    summary = {
        "monitoring_timestamp": datetime.now().isoformat(),
        "api_base_url": API_BASE_URL,
        "total_endpoints": len(
            endpoint_summary
        ),
        "total_requests": total_requests,
        "successful_requests": successful_requests,
        "failed_requests": failed_requests,
        "success_rate": round(
            success_rate,
            4
        ),
        "error_rate": round(
            error_rate,
            4
        ),
        "average_latency_ms": round(
            statistics.mean(latencies),
            4
        ) if latencies else 0,
        "median_latency_ms": round(
            statistics.median(latencies),
            4
        ) if latencies else 0,
        "minimum_latency_ms": round(
            min(latencies),
            4
        ) if latencies else 0,
        "maximum_latency_ms": round(
            max(latencies),
            4
        ) if latencies else 0,
        "p95_latency_ms": round(
            percentile(
                latencies,
                95
            ),
            4
        ) if latencies else 0,
        "status_200_count": status_200_count,
        "status_404_count": status_404_count,
        "status_500_count": status_500_count,
    }

    return pd.DataFrame([summary])


# ===========================================================================
# LATENCY ANALYSIS
# ===========================================================================

def create_latency_analysis(results_df):
    """
    Create latency statistics table.
    """

    rows = []

    for endpoint_name, group in results_df.groupby(
        "test_name"
    ):

        values = (
            group["latency_ms"]
            .astype(float)
            .tolist()
        )

        rows.append(
            {
                "test_name": endpoint_name,
                "average_ms": round(
                    statistics.mean(values),
                    4
                ),
                "median_ms": round(
                    statistics.median(values),
                    4
                ),
                "min_ms": round(
                    min(values),
                    4
                ),
                "max_ms": round(
                    max(values),
                    4
                ),
                "p95_ms": round(
                    percentile(
                        values,
                        95
                    ),
                    4
                ),
                "p99_ms": round(
                    percentile(
                        values,
                        99
                    ),
                    4
                ),
            }
        )

    return pd.DataFrame(rows)


# ===========================================================================
# PLOT 1 - ENDPOINT LATENCY
# ===========================================================================

def plot_endpoint_latency(endpoint_summary):

    print()
    print("=" * 75)
    print("GENERATING ENDPOINT LATENCY PLOT")
    print("=" * 75)

    plt.figure(figsize=(14, 7))

    plt.bar(
        endpoint_summary["test_name"],
        endpoint_summary["average_latency_ms"]
    )

    plt.xlabel(
        "Endpoint"
    )

    plt.ylabel(
        "Average Latency (ms)"
    )

    plt.title(
        "Day 23 - API Endpoint Average Latency"
    )

    plt.xticks(
        rotation=45,
        ha="right"
    )

    plt.tight_layout()

    output_path = os.path.join(
        PLOTS_DIR,
        "day23_endpoint_latency.png"
    )

    plt.savefig(
        output_path,
        dpi=150
    )

    plt.close()

    print(
        f"Plot saved:\n{output_path}"
    )


# ===========================================================================
# PLOT 2 - P95 LATENCY
# ===========================================================================

def plot_p95_latency(endpoint_summary):

    print()
    print("=" * 75)
    print("GENERATING P95 LATENCY PLOT")
    print("=" * 75)

    plt.figure(figsize=(14, 7))

    plt.bar(
        endpoint_summary["test_name"],
        endpoint_summary["p95_latency_ms"]
    )

    plt.xlabel(
        "Endpoint"
    )

    plt.ylabel(
        "P95 Latency (ms)"
    )

    plt.title(
        "Day 23 - API Endpoint P95 Latency"
    )

    plt.xticks(
        rotation=45,
        ha="right"
    )

    plt.tight_layout()

    output_path = os.path.join(
        PLOTS_DIR,
        "day23_p95_latency.png"
    )

    plt.savefig(
        output_path,
        dpi=150
    )

    plt.close()

    print(
        f"Plot saved:\n{output_path}"
    )


# ===========================================================================
# PLOT 3 - SUCCESS / ERROR RATE
# ===========================================================================

def plot_success_error_rate(
    endpoint_summary
):

    print()
    print("=" * 75)
    print("GENERATING SUCCESS / ERROR RATE PLOT")
    print("=" * 75)

    plt.figure(figsize=(14, 7))

    x = range(
        len(endpoint_summary)
    )

    plt.bar(
        x,
        endpoint_summary[
            "success_rate"
        ],
        label="Success Rate (%)"
    )

    plt.bar(
        x,
        endpoint_summary[
            "error_rate"
        ],
        bottom=endpoint_summary[
            "success_rate"
        ],
        label="Error Rate (%)"
    )

    plt.xticks(
        list(x),
        endpoint_summary[
            "test_name"
        ],
        rotation=45,
        ha="right"
    )

    plt.ylabel(
        "Percentage"
    )

    plt.title(
        "Day 23 - API Success / Error Rate"
    )

    plt.legend()

    plt.tight_layout()

    output_path = os.path.join(
        PLOTS_DIR,
        "day23_success_error_rate.png"
    )

    plt.savefig(
        output_path,
        dpi=150
    )

    plt.close()

    print(
        f"Plot saved:\n{output_path}"
    )


# ===========================================================================
# PLOT 4 - REQUEST LATENCY DISTRIBUTION
# ===========================================================================

def plot_latency_distribution(
    results_df
):

    print()
    print("=" * 75)
    print("GENERATING LATENCY DISTRIBUTION PLOT")
    print("=" * 75)

    plt.figure(figsize=(12, 7))

    plt.hist(
        results_df["latency_ms"],
        bins=15
    )

    plt.xlabel(
        "Latency (ms)"
    )

    plt.ylabel(
        "Request Count"
    )

    plt.title(
        "Day 23 - API Request Latency Distribution"
    )

    plt.tight_layout()

    output_path = os.path.join(
        PLOTS_DIR,
        "day23_latency_distribution.png"
    )

    plt.savefig(
        output_path,
        dpi=150
    )

    plt.close()

    print(
        f"Plot saved:\n{output_path}"
    )


# ===========================================================================
# PLOT 5 - STATUS CODES
# ===========================================================================

def plot_status_codes(
    results_df
):

    print()
    print("=" * 75)
    print("GENERATING STATUS CODE PLOT")
    print("=" * 75)

    status_counts = (
        results_df[
            "actual_status"
        ]
        .value_counts()
        .sort_index()
    )

    plt.figure(figsize=(10, 6))

    plt.bar(
        status_counts.index.astype(str),
        status_counts.values
    )

    plt.xlabel(
        "HTTP Status Code"
    )

    plt.ylabel(
        "Request Count"
    )

    plt.title(
        "Day 23 - HTTP Status Code Distribution"
    )

    plt.tight_layout()

    output_path = os.path.join(
        PLOTS_DIR,
        "day23_status_codes.png"
    )

    plt.savefig(
        output_path,
        dpi=150
    )

    plt.close()

    print(
        f"Plot saved:\n{output_path}"
    )


# ===========================================================================
# SAVE REPORTS
# ===========================================================================

def save_reports(
    results_df,
    endpoint_summary,
    overall_summary,
    latency_analysis
):

    print()
    print("=" * 75)
    print("SAVING MONITORING REPORTS")
    print("=" * 75)

    results_path = os.path.join(
        EVALUATION_DIR,
        "day23_api_monitoring_results.csv"
    )

    endpoint_summary_path = os.path.join(
        EVALUATION_DIR,
        "day23_endpoint_monitoring_summary.csv"
    )

    overall_summary_path = os.path.join(
        EVALUATION_DIR,
        "day23_api_monitoring_summary.csv"
    )

    latency_path = os.path.join(
        EVALUATION_DIR,
        "day23_latency_analysis.csv"
    )

    json_path = os.path.join(
        EVALUATION_DIR,
        "day23_api_monitoring_report.json"
    )

    results_df.to_csv(
        results_path,
        index=False
    )

    endpoint_summary.to_csv(
        endpoint_summary_path,
        index=False
    )

    overall_summary.to_csv(
        overall_summary_path,
        index=False
    )

    latency_analysis.to_csv(
        latency_path,
        index=False
    )

    report = {
        "day": 23,
        "project": "Hybrid Recommendation System",
        "monitoring_type": "Recommendation API Monitoring",
        "timestamp": datetime.now().isoformat(),
        "api_base_url": API_BASE_URL,
        "requests_per_endpoint": (
            REQUESTS_PER_ENDPOINT
        ),
        "overall_summary": (
            overall_summary
            .to_dict(orient="records")
        ),
        "endpoint_summary": (
            endpoint_summary
            .to_dict(orient="records")
        ),
        "latency_analysis": (
            latency_analysis
            .to_dict(orient="records")
        ),
    }

    with open(
        json_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            report,
            file,
            indent=4
        )

    print(
        f"Detailed results saved:\n"
        f"{results_path}"
    )

    print(
        f"\nEndpoint summary saved:\n"
        f"{endpoint_summary_path}"
    )

    print(
        f"\nOverall summary saved:\n"
        f"{overall_summary_path}"
    )

    print(
        f"\nLatency analysis saved:\n"
        f"{latency_path}"
    )

    print(
        f"\nJSON report saved:\n"
        f"{json_path}"
    )


# ===========================================================================
# PRINT FINAL SUMMARY
# ===========================================================================

def print_final_summary(
    overall_summary,
    endpoint_summary
):

    row = overall_summary.iloc[0]

    print()
    print("=" * 75)
    print("DAY 23 FINAL API MONITORING SUMMARY")
    print("=" * 75)

    print(
        f"API Base URL: "
        f"{API_BASE_URL}"
    )

    print(
        f"Total endpoints monitored: "
        f"{int(row['total_endpoints'])}"
    )

    print(
        f"Total requests: "
        f"{int(row['total_requests'])}"
    )

    print(
        f"Successful requests: "
        f"{int(row['successful_requests'])}"
    )

    print(
        f"Failed requests: "
        f"{int(row['failed_requests'])}"
    )

    print(
        f"Success rate: "
        f"{row['success_rate']:.2f}%"
    )

    print(
        f"Error rate: "
        f"{row['error_rate']:.2f}%"
    )

    print(
        f"Average latency: "
        f"{row['average_latency_ms']:.2f} ms"
    )

    print(
        f"Median latency: "
        f"{row['median_latency_ms']:.2f} ms"
    )

    print(
        f"Minimum latency: "
        f"{row['minimum_latency_ms']:.2f} ms"
    )

    print(
        f"Maximum latency: "
        f"{row['maximum_latency_ms']:.2f} ms"
    )

    print(
        f"P95 latency: "
        f"{row['p95_latency_ms']:.2f} ms"
    )

    print()
    print(
        "HTTP 200 responses: "
        f"{int(row['status_200_count'])}"
    )

    print(
        "HTTP 404 responses: "
        f"{int(row['status_404_count'])}"
    )

    print(
        "HTTP 500 responses: "
        f"{int(row['status_500_count'])}"
    )

    print()
    print("=" * 75)
    print("ENDPOINT SUMMARY")
    print("=" * 75)

    display_columns = [
        "test_name",
        "total_requests",
        "successful_requests",
        "failed_requests",
        "success_rate",
        "average_latency_ms",
        "p95_latency_ms",
    ]

    print(
        endpoint_summary[
            display_columns
        ].to_string(
            index=False
        )
    )


# ===========================================================================
# MAIN
# ===========================================================================

def main():

    print()
    print("=" * 75)
    print("DAY 23 - API MONITORING")
    print("=" * 75)

    print(
        "Hybrid Recommendation System"
    )

    print(
        f"Base directory:\n{BASE_DIR}"
    )

    print(
        f"Evaluation directory:\n"
        f"{EVALUATION_DIR}"
    )

    print()

    # -----------------------------------------------------------------------
    # CHECK API
    # -----------------------------------------------------------------------

    api_online = check_api_available()

    if not api_online:

        print("=" * 75)
        print("ERROR")
        print("=" * 75)

        print(
            "Recommendation API is not available."
        )

        print()
        print(
            "Start Day 21 API first:"
        )

        print(
            "python scripts/day21_api.py"
        )

        print()

        print(
            "Then run Day 23 again."
        )

        return

    # -----------------------------------------------------------------------
    # MONITOR ENDPOINTS
    # -----------------------------------------------------------------------

    results_df = monitor_endpoints()

    # -----------------------------------------------------------------------
    # CREATE ENDPOINT SUMMARY
    # -----------------------------------------------------------------------

    print()
    print("=" * 75)
    print("CREATING ENDPOINT SUMMARY")
    print("=" * 75)

    endpoint_summary = (
        create_endpoint_summary(
            results_df
        )
    )

    print(
        endpoint_summary.to_string(
            index=False
        )
    )

    # -----------------------------------------------------------------------
    # CREATE OVERALL SUMMARY
    # -----------------------------------------------------------------------

    print()
    print("=" * 75)
    print("CREATING OVERALL SUMMARY")
    print("=" * 75)

    overall_summary = (
        create_overall_summary(
            results_df,
            endpoint_summary
        )
    )

    print(
        overall_summary.to_string(
            index=False
        )
    )

    # -----------------------------------------------------------------------
    # LATENCY ANALYSIS
    # -----------------------------------------------------------------------

    print()
    print("=" * 75)
    print("CREATING LATENCY ANALYSIS")
    print("=" * 75)

    latency_analysis = (
        create_latency_analysis(
            results_df
        )
    )

    print(
        latency_analysis.to_string(
            index=False
        )
    )

    # -----------------------------------------------------------------------
    # SAVE CSV / JSON REPORTS
    # -----------------------------------------------------------------------

    save_reports(
        results_df,
        endpoint_summary,
        overall_summary,
        latency_analysis
    )

    # -----------------------------------------------------------------------
    # GENERATE PLOTS
    # -----------------------------------------------------------------------

    plot_endpoint_latency(
        endpoint_summary
    )

    plot_p95_latency(
        endpoint_summary
    )

    plot_success_error_rate(
        endpoint_summary
    )

    plot_latency_distribution(
        results_df
    )

    plot_status_codes(
        results_df
    )

    # -----------------------------------------------------------------------
    # FINAL SUMMARY
    # -----------------------------------------------------------------------

    print_final_summary(
        overall_summary,
        endpoint_summary
    )

    # -----------------------------------------------------------------------
    # FINAL STATUS
    # -----------------------------------------------------------------------

    print()
    print("=" * 75)
    print("DAY 23 COMPLETED SUCCESSFULLY")
    print("=" * 75)

    print()
    print(
        f"Monitoring results:\n"
        f"{EVALUATION_DIR}"
    )

    print()
    print(
        f"Plots:\n"
        f"{PLOTS_DIR}"
    )

    print()
    print(
        "API monitoring completed successfully."
    )


# ===========================================================================
# ENTRY POINT
# ===========================================================================

if __name__ == "__main__":
    main()