# ======================================================================
# DAY 22 - API PERFORMANCE & LOAD EVALUATION
# ======================================================================
#
# This script tests the Day 21 Recommendation API.
#
# Required running API:
#     http://127.0.0.1:8000
#
# Start Day 21 API first:
#     python scripts/day21_api.py
#
# Then run this script in another terminal:
#     python scripts/day22_api_performance.py
#
# Outputs:
#     evaluation/day22_api_performance_results.csv
#     evaluation/day22_api_performance_summary.csv
#
#     evaluation/plots/day22_latency_comparison.png
#     evaluation/plots/day22_throughput.png
#     evaluation/plots/day22_success_rate.png
#
# ======================================================================

import os
import time
import json
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import pandas as pd
import matplotlib.pyplot as plt


# ======================================================================
# CONFIGURATION
# ======================================================================

BASE_URL = "http://127.0.0.1:8000"

TOP_K = 10

# Number of requests for normal endpoint testing
REQUESTS_PER_ENDPOINT = 10

# Concurrent load settings
CONCURRENT_REQUESTS = 50

# Timeout for every API request
REQUEST_TIMEOUT = 60

# Test endpoints
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
        "path": f"/recommend/1?top_k={TOP_K}",
        "expected_status": 200,
    },
    {
        "name": "Cold Start Recommendations",
        "method": "GET",
        "path": f"/cold-start?top_k={TOP_K}",
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


# ======================================================================
# DIRECTORY SETUP
# ======================================================================

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

os.makedirs(
    EVALUATION_DIR,
    exist_ok=True
)

os.makedirs(
    PLOTS_DIR,
    exist_ok=True
)


RESULTS_PATH = os.path.join(
    EVALUATION_DIR,
    "day22_api_performance_results.csv"
)

SUMMARY_PATH = os.path.join(
    EVALUATION_DIR,
    "day22_api_performance_summary.csv"
)

LATENCY_PLOT_PATH = os.path.join(
    PLOTS_DIR,
    "day22_latency_comparison.png"
)

THROUGHPUT_PLOT_PATH = os.path.join(
    PLOTS_DIR,
    "day22_throughput.png"
)

SUCCESS_RATE_PLOT_PATH = os.path.join(
    PLOTS_DIR,
    "day22_success_rate.png"
)


# ======================================================================
# PRINT HEADER
# ======================================================================

print("=" * 70)
print("DAY 22 - API PERFORMANCE & LOAD EVALUATION")
print("=" * 70)

print()
print("Base URL:")
print(BASE_URL)

print()
print("Evaluation directory:")
print(EVALUATION_DIR)

print()
print("Requests per endpoint:")
print(REQUESTS_PER_ENDPOINT)

print()
print("Concurrent requests:")
print(CONCURRENT_REQUESTS)

print()


# ======================================================================
# HELPER FUNCTIONS
# ======================================================================

def build_url(path):
    """
    Build complete API URL.
    """

    return BASE_URL.rstrip("/") + "/" + path.lstrip("/")


def perform_request(
    endpoint,
    request_number=1
):
    """
    Perform one HTTP request and collect performance information.
    """

    url = build_url(
        endpoint["path"]
    )

    start_time = time.perf_counter()

    status_code = None
    response_size = 0
    error_message = ""

    try:

        response = requests.get(
            url,
            timeout=REQUEST_TIMEOUT
        )

        status_code = response.status_code

        try:
            response_size = len(
                response.content
            )
        except Exception:
            response_size = 0

        success = (
            status_code
            == endpoint["expected_status"]
        )

    except requests.exceptions.Timeout:

        success = False

        error_message = (
            "Request timeout"
        )

    except requests.exceptions.ConnectionError:

        success = False

        error_message = (
            "Connection error"
        )

    except requests.exceptions.RequestException as exc:

        success = False

        error_message = str(
            exc
        )

    except Exception as exc:

        success = False

        error_message = str(
            exc
        )

    end_time = time.perf_counter()

    latency_ms = (
        end_time
        - start_time
    ) * 1000

    return {
        "endpoint_name": endpoint["name"],
        "method": endpoint["method"],
        "path": endpoint["path"],
        "request_number": request_number,
        "expected_status": endpoint[
            "expected_status"
        ],
        "actual_status": status_code,
        "success": success,
        "latency_ms": latency_ms,
        "response_size_bytes": response_size,
        "error": error_message,
    }


def calculate_percentile(
    values,
    percentile
):
    """
    Calculate percentile safely.
    """

    if not values:
        return 0.0

    series = pd.Series(
        values,
        dtype=float
    )

    return float(
        series.quantile(
            percentile / 100.0
        )
    )


# ======================================================================
# CHECK API AVAILABILITY
# ======================================================================

print("=" * 70)
print("CHECKING API AVAILABILITY")
print("=" * 70)

try:

    health_url = build_url(
        "/health"
    )

    health_start = time.perf_counter()

    health_response = requests.get(
        health_url,
        timeout=REQUEST_TIMEOUT
    )

    health_end = time.perf_counter()

    health_latency = (
        health_end
        - health_start
    ) * 1000

    print()
    print(
        f"Health status: "
        f"{health_response.status_code}"
    )

    print(
        f"Health latency: "
        f"{health_latency:.2f} ms"
    )

    if health_response.status_code != 200:

        print()
        print(
            "ERROR: API health check failed."
        )

        print(
            "Start the Day 21 API first."
        )

        raise SystemExit(1)

except Exception as exc:

    print()
    print(
        "ERROR: Could not connect to API."
    )

    print(
        f"Details: {exc}"
    )

    print()
    print(
        "Start Day 21 API using:"
    )

    print(
        "python scripts/day21_api.py"
    )

    raise SystemExit(1)


print()
print(
    "API is available."
)


# ======================================================================
# NORMAL ENDPOINT PERFORMANCE TEST
# ======================================================================

print()
print("=" * 70)
print("ENDPOINT PERFORMANCE TEST")
print("=" * 70)


all_results = []


for endpoint in ENDPOINTS:

    print()
    print("-" * 70)

    print(
        f"Testing: {endpoint['name']}"
    )

    print(
        f"METHOD: {endpoint['method']}"
    )

    print(
        f"URL: {build_url(endpoint['path'])}"
    )

    print(
        f"Requests: {REQUESTS_PER_ENDPOINT}"
    )

    print("-" * 70)

    endpoint_results = []

    for request_number in range(
        1,
        REQUESTS_PER_ENDPOINT + 1
    ):

        result = perform_request(
            endpoint,
            request_number
        )

        endpoint_results.append(
            result
        )

        all_results.append(
            result
        )

    successful_results = [
        result
        for result in endpoint_results
        if result["success"]
    ]

    latencies = [
        result["latency_ms"]
        for result in successful_results
    ]

    success_count = len(
        successful_results
    )

    total_count = len(
        endpoint_results
    )

    success_rate = (
        success_count
        / total_count
        * 100
        if total_count > 0
        else 0
    )

    if latencies:

        avg_latency = statistics.mean(
            latencies
        )

        min_latency = min(
            latencies
        )

        max_latency = max(
            latencies
        )

        median_latency = statistics.median(
            latencies
        )

        p95_latency = calculate_percentile(
            latencies,
            95
        )

    else:

        avg_latency = 0
        min_latency = 0
        max_latency = 0
        median_latency = 0
        p95_latency = 0

    print(
        f"Successful requests: "
        f"{success_count}/{total_count}"
    )

    print(
        f"Success rate: "
        f"{success_rate:.2f}%"
    )

    print(
        f"Average latency: "
        f"{avg_latency:.2f} ms"
    )

    print(
        f"Median latency: "
        f"{median_latency:.2f} ms"
    )

    print(
        f"Minimum latency: "
        f"{min_latency:.2f} ms"
    )

    print(
        f"Maximum latency: "
        f"{max_latency:.2f} ms"
    )

    print(
        f"P95 latency: "
        f"{p95_latency:.2f} ms"
    )


# ======================================================================
# REQUEST LEVEL DATAFRAME
# ======================================================================

results_df = pd.DataFrame(
    all_results
)


print()
print("=" * 70)
print("REQUEST LEVEL RESULTS")
print("=" * 70)

print()

if not results_df.empty:

    display_columns = [
        "endpoint_name",
        "request_number",
        "actual_status",
        "success",
        "latency_ms",
        "response_size_bytes",
    ]

    print(
        results_df[
            display_columns
        ].to_string(
            index=False
        )
    )


# ======================================================================
# ENDPOINT SUMMARY
# ======================================================================

print()
print("=" * 70)
print("CREATING ENDPOINT SUMMARY")
print("=" * 70)


summary_rows = []


for endpoint in ENDPOINTS:

    endpoint_name = endpoint[
        "name"
    ]

    endpoint_df = results_df[
        results_df[
            "endpoint_name"
        ]
        == endpoint_name
    ]

    total_requests = len(
        endpoint_df
    )

    successful_requests = int(
        endpoint_df[
            "success"
        ].sum()
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

    successful_latencies = endpoint_df[
        endpoint_df["success"]
    ]["latency_ms"].tolist()

    if successful_latencies:

        average_latency = statistics.mean(
            successful_latencies
        )

        median_latency = statistics.median(
            successful_latencies
        )

        min_latency = min(
            successful_latencies
        )

        max_latency = max(
            successful_latencies
        )

        p95_latency = calculate_percentile(
            successful_latencies,
            95
        )

        p99_latency = calculate_percentile(
            successful_latencies,
            99
        )

        average_response_size = (
            endpoint_df[
                endpoint_df["success"]
            ]["response_size_bytes"]
            .mean()
        )

    else:

        average_latency = 0
        median_latency = 0
        min_latency = 0
        max_latency = 0
        p95_latency = 0
        p99_latency = 0
        average_response_size = 0

    summary_rows.append(
        {
            "endpoint_name": endpoint_name,
            "method": endpoint["method"],
            "path": endpoint["path"],
            "expected_status": endpoint[
                "expected_status"
            ],
            "total_requests": total_requests,
            "successful_requests": successful_requests,
            "failed_requests": failed_requests,
            "success_rate_percent": success_rate,
            "average_latency_ms": average_latency,
            "median_latency_ms": median_latency,
            "minimum_latency_ms": min_latency,
            "maximum_latency_ms": max_latency,
            "p95_latency_ms": p95_latency,
            "p99_latency_ms": p99_latency,
            "average_response_size_bytes": average_response_size,
        }
    )


summary_df = pd.DataFrame(
    summary_rows
)


print()

print(
    summary_df.to_string(
        index=False
    )
)


# ======================================================================
# SAVE REQUEST LEVEL RESULTS
# ======================================================================

results_df.to_csv(
    RESULTS_PATH,
    index=False
)

print()
print(
    "Request-level results saved:"
)

print(
    RESULTS_PATH
)


# ======================================================================
# CONCURRENT LOAD TEST
# ======================================================================

print()
print("=" * 70)
print("CONCURRENT LOAD TEST")
print("=" * 70)


# We use the recommendation endpoint because
# it represents the main API workload.

load_endpoint = {
    "name": "Personalized Recommendations Load Test",
    "method": "GET",
    "path": f"/recommend/1?top_k={TOP_K}",
    "expected_status": 200,
}


print()
print(
    "Endpoint:"
)

print(
    build_url(
        load_endpoint["path"]
    )
)

print()
print(
    f"Concurrent requests: "
    f"{CONCURRENT_REQUESTS}"
)


load_start = time.perf_counter()

load_results = []


with ThreadPoolExecutor(
    max_workers=CONCURRENT_REQUESTS
) as executor:

    futures = [
        executor.submit(
            perform_request,
            load_endpoint,
            request_number
        )
        for request_number
        in range(
            1,
            CONCURRENT_REQUESTS + 1
        )
    ]

    for future in as_completed(
        futures
    ):

        result = future.result()

        load_results.append(
            result
        )


load_end = time.perf_counter()


load_duration_seconds = (
    load_end
    - load_start
)


load_successful = [
    result
    for result in load_results
    if result["success"]
]


load_failed = [
    result
    for result in load_results
    if not result["success"]
]


load_latencies = [
    result["latency_ms"]
    for result in load_results
]


load_success_rate = (
    len(load_successful)
    / len(load_results)
    * 100
    if load_results
    else 0
)


load_throughput = (
    len(load_results)
    / load_duration_seconds
    if load_duration_seconds > 0
    else 0
)


load_average_latency = (
    statistics.mean(
        load_latencies
    )
    if load_latencies
    else 0
)


load_median_latency = (
    statistics.median(
        load_latencies
    )
    if load_latencies
    else 0
)


load_p95_latency = calculate_percentile(
    load_latencies,
    95
)


load_p99_latency = calculate_percentile(
    load_latencies,
    99
)


print()
print(
    f"Total requests: "
    f"{len(load_results)}"
)

print(
    f"Successful requests: "
    f"{len(load_successful)}"
)

print(
    f"Failed requests: "
    f"{len(load_failed)}"
)

print(
    f"Success rate: "
    f"{load_success_rate:.2f}%"
)

print(
    f"Total duration: "
    f"{load_duration_seconds:.4f} seconds"
)

print(
    f"Throughput: "
    f"{load_throughput:.2f} requests/sec"
)

print(
    f"Average latency: "
    f"{load_average_latency:.2f} ms"
)

print(
    f"Median latency: "
    f"{load_median_latency:.2f} ms"
)

print(
    f"P95 latency: "
    f"{load_p95_latency:.2f} ms"
)

print(
    f"P99 latency: "
    f"{load_p99_latency:.2f} ms"
)


# ======================================================================
# OVERALL PERFORMANCE METRICS
# ======================================================================

print()
print("=" * 70)
print("OVERALL PERFORMANCE")
print("=" * 70)


total_requests = len(
    results_df
)

successful_requests = int(
    results_df[
        "success"
    ].sum()
)

failed_requests = (
    total_requests
    - successful_requests
)


overall_success_rate = (
    successful_requests
    / total_requests
    * 100
    if total_requests > 0
    else 0
)


successful_latency_values = (
    results_df[
        results_df["success"]
    ]["latency_ms"]
    .tolist()
)


if successful_latency_values:

    overall_average_latency = (
        statistics.mean(
            successful_latency_values
        )
    )

    overall_median_latency = (
        statistics.median(
            successful_latency_values
        )
    )

    overall_min_latency = min(
        successful_latency_values
    )

    overall_max_latency = max(
        successful_latency_values
    )

    overall_p95_latency = calculate_percentile(
        successful_latency_values,
        95
    )

else:

    overall_average_latency = 0
    overall_median_latency = 0
    overall_min_latency = 0
    overall_max_latency = 0
    overall_p95_latency = 0


print()

print(
    f"Total requests: "
    f"{total_requests}"
)

print(
    f"Successful requests: "
    f"{successful_requests}"
)

print(
    f"Failed requests: "
    f"{failed_requests}"
)

print(
    f"Success rate: "
    f"{overall_success_rate:.2f}%"
)

print(
    f"Average latency: "
    f"{overall_average_latency:.2f} ms"
)

print(
    f"Median latency: "
    f"{overall_median_latency:.2f} ms"
)

print(
    f"Minimum latency: "
    f"{overall_min_latency:.2f} ms"
)

print(
    f"Maximum latency: "
    f"{overall_max_latency:.2f} ms"
)

print(
    f"P95 latency: "
    f"{overall_p95_latency:.2f} ms"
)


# ======================================================================
# CREATE FINAL SUMMARY TABLE
# ======================================================================

print()
print("=" * 70)
print("CREATING FINAL PERFORMANCE SUMMARY")
print("=" * 70)


final_summary_rows = []


for _, row in summary_df.iterrows():

    final_summary_rows.append(
        {
            "metric_type": "endpoint",
            "endpoint_name": row[
                "endpoint_name"
            ],
            "total_requests": row[
                "total_requests"
            ],
            "successful_requests": row[
                "successful_requests"
            ],
            "failed_requests": row[
                "failed_requests"
            ],
            "success_rate_percent": row[
                "success_rate_percent"
            ],
            "average_latency_ms": row[
                "average_latency_ms"
            ],
            "median_latency_ms": row[
                "median_latency_ms"
            ],
            "minimum_latency_ms": row[
                "minimum_latency_ms"
            ],
            "maximum_latency_ms": row[
                "maximum_latency_ms"
            ],
            "p95_latency_ms": row[
                "p95_latency_ms"
            ],
            "p99_latency_ms": row[
                "p99_latency_ms"
            ],
            "throughput_requests_per_second": None,
        }
    )


final_summary_rows.append(
    {
        "metric_type": "concurrent_load",
        "endpoint_name": load_endpoint[
            "name"
        ],
        "total_requests": len(
            load_results
        ),
        "successful_requests": len(
            load_successful
        ),
        "failed_requests": len(
            load_failed
        ),
        "success_rate_percent": load_success_rate,
        "average_latency_ms": load_average_latency,
        "median_latency_ms": load_median_latency,
        "minimum_latency_ms": (
            min(load_latencies)
            if load_latencies
            else 0
        ),
        "maximum_latency_ms": (
            max(load_latencies)
            if load_latencies
            else 0
        ),
        "p95_latency_ms": load_p95_latency,
        "p99_latency_ms": load_p99_latency,
        "throughput_requests_per_second": load_throughput,
    }
)


final_summary_df = pd.DataFrame(
    final_summary_rows
)


print()

print(
    final_summary_df.to_string(
        index=False
    )
)


final_summary_df.to_csv(
    SUMMARY_PATH,
    index=False
)


print()
print(
    "Final performance summary saved:"
)

print(
    SUMMARY_PATH
)


# ======================================================================
# PLOT 1 - LATENCY COMPARISON
# ======================================================================

print()
print("=" * 70)
print("GENERATING LATENCY COMPARISON PLOT")
print("=" * 70)


plot_df = summary_df.copy()


plt.figure(
    figsize=(12, 7)
)

plt.bar(
    plot_df[
        "endpoint_name"
    ],
    plot_df[
        "average_latency_ms"
    ]
)

plt.xlabel(
    "API Endpoint"
)

plt.ylabel(
    "Average Latency (ms)"
)

plt.title(
    "Day 22 - API Endpoint Latency Comparison"
)

plt.xticks(
    rotation=45,
    ha="right"
)

plt.tight_layout()

plt.savefig(
    LATENCY_PLOT_PATH,
    dpi=150
)

plt.close()


print(
    "Plot saved:"
)

print(
    LATENCY_PLOT_PATH
)


# ======================================================================
# PLOT 2 - THROUGHPUT
# ======================================================================

print()
print("=" * 70)
print("GENERATING THROUGHPUT PLOT")
print("=" * 70)


endpoint_throughputs = []


for _, row in summary_df.iterrows():

    request_count = row[
        "successful_requests"
    ]

    average_latency = row[
        "average_latency_ms"
    ]

    if (
        request_count > 0
        and average_latency > 0
    ):

        throughput = (
            1000
            / average_latency
        )

    else:

        throughput = 0

    endpoint_throughputs.append(
        throughput
    )


throughput_plot_df = pd.DataFrame(
    {
        "endpoint_name":
            summary_df[
                "endpoint_name"
            ],

        "throughput":
            endpoint_throughputs,
    }
)


plt.figure(
    figsize=(12, 7)
)

plt.bar(
    throughput_plot_df[
        "endpoint_name"
    ],
    throughput_plot_df[
        "throughput"
    ]
)

plt.xlabel(
    "API Endpoint"
)

plt.ylabel(
    "Estimated Requests / Second"
)

plt.title(
    "Day 22 - API Endpoint Throughput"
)

plt.xticks(
    rotation=45,
    ha="right"
)

plt.tight_layout()

plt.savefig(
    THROUGHPUT_PLOT_PATH,
    dpi=150
)

plt.close()


print(
    "Plot saved:"
)

print(
    THROUGHPUT_PLOT_PATH
)


# ======================================================================
# PLOT 3 - SUCCESS RATE
# ======================================================================

print()
print("=" * 70)
print("GENERATING SUCCESS RATE PLOT")
print("=" * 70)


plt.figure(
    figsize=(12, 7)
)

plt.bar(
    summary_df[
        "endpoint_name"
    ],
    summary_df[
        "success_rate_percent"
    ]
)

plt.xlabel(
    "API Endpoint"
)

plt.ylabel(
    "Success Rate (%)"
)

plt.title(
    "Day 22 - API Endpoint Success Rate"
)

plt.ylim(
    0,
    105
)

plt.xticks(
    rotation=45,
    ha="right"
)

plt.tight_layout()

plt.savefig(
    SUCCESS_RATE_PLOT_PATH,
    dpi=150
)

plt.close()


print(
    "Plot saved:"
)

print(
    SUCCESS_RATE_PLOT_PATH
)


# ======================================================================
# SAVE LOAD TEST RESULTS
# ======================================================================

load_results_path = os.path.join(
    EVALUATION_DIR,
    "day22_concurrent_load_results.csv"
)


load_results_df = pd.DataFrame(
    load_results
)


load_results_df.to_csv(
    load_results_path,
    index=False
)


print()
print(
    "Concurrent load results saved:"
)

print(
    load_results_path
)


# ======================================================================
# CREATE JSON REPORT
# ======================================================================

report_path = os.path.join(
    EVALUATION_DIR,
    "day22_api_performance_report.json"
)


report = {
    "day": 22,
    "title": (
        "API Performance & Load Evaluation"
    ),
    "base_url": BASE_URL,

    "configuration": {
        "requests_per_endpoint":
            REQUESTS_PER_ENDPOINT,

        "concurrent_requests":
            CONCURRENT_REQUESTS,

        "top_k":
            TOP_K,

        "request_timeout_seconds":
            REQUEST_TIMEOUT,
    },

    "overall": {
        "total_requests":
            total_requests,

        "successful_requests":
            successful_requests,

        "failed_requests":
            failed_requests,

        "success_rate_percent":
            overall_success_rate,

        "average_latency_ms":
            overall_average_latency,

        "median_latency_ms":
            overall_median_latency,

        "minimum_latency_ms":
            overall_min_latency,

        "maximum_latency_ms":
            overall_max_latency,

        "p95_latency_ms":
            overall_p95_latency,
    },

    "concurrent_load": {
        "total_requests":
            len(load_results),

        "successful_requests":
            len(load_successful),

        "failed_requests":
            len(load_failed),

        "success_rate_percent":
            load_success_rate,

        "duration_seconds":
            load_duration_seconds,

        "throughput_requests_per_second":
            load_throughput,

        "average_latency_ms":
            load_average_latency,

        "median_latency_ms":
            load_median_latency,

        "p95_latency_ms":
            load_p95_latency,

        "p99_latency_ms":
            load_p99_latency,
    },

    "files": {
        "request_results":
            RESULTS_PATH,

        "summary":
            SUMMARY_PATH,

        "load_results":
            load_results_path,

        "latency_plot":
            LATENCY_PLOT_PATH,

        "throughput_plot":
            THROUGHPUT_PLOT_PATH,

        "success_rate_plot":
            SUCCESS_RATE_PLOT_PATH,
    },
}


with open(
    report_path,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        report,
        file,
        indent=4
    )


print()
print(
    "JSON report saved:"
)

print(
    report_path
)


# ======================================================================
# FINAL SUMMARY
# ======================================================================

print()
print("=" * 70)
print("DAY 22 FINAL PERFORMANCE SUMMARY")
print("=" * 70)

print()

print(
    f"Total endpoint requests: "
    f"{total_requests}"
)

print(
    f"Successful endpoint requests: "
    f"{successful_requests}"
)

print(
    f"Failed endpoint requests: "
    f"{failed_requests}"
)

print(
    f"Overall success rate: "
    f"{overall_success_rate:.2f}%"
)

print(
    f"Average latency: "
    f"{overall_average_latency:.2f} ms"
)

print(
    f"Median latency: "
    f"{overall_median_latency:.2f} ms"
)

print(
    f"P95 latency: "
    f"{overall_p95_latency:.2f} ms"
)

print()

print(
    f"Concurrent load requests: "
    f"{len(load_results)}"
)

print(
    f"Concurrent load success rate: "
    f"{load_success_rate:.2f}%"
)

print(
    f"Concurrent throughput: "
    f"{load_throughput:.2f} requests/sec"
)

print(
    f"Concurrent average latency: "
    f"{load_average_latency:.2f} ms"
)

print(
    f"Concurrent P95 latency: "
    f"{load_p95_latency:.2f} ms"
)

print()

print("=" * 70)
print("SAVED FILES")
print("=" * 70)

print()

print(
    "Request-level results:"
)

print(
    RESULTS_PATH
)

print()

print(
    "Performance summary:"
)

print(
    SUMMARY_PATH
)

print()

print(
    "Concurrent load results:"
)

print(
    load_results_path
)

print()

print(
    "JSON report:"
)

print(
    report_path
)

print()

print(
    "Latency plot:"
)

print(
    LATENCY_PLOT_PATH
)

print()

print(
    "Throughput plot:"
)

print(
    THROUGHPUT_PLOT_PATH
)

print()

print(
    "Success rate plot:"
)

print(
    SUCCESS_RATE_PLOT_PATH
)

print()

print("=" * 70)
print(
    "DAY 22 COMPLETED SUCCESSFULLY"
)
print("=" * 70)