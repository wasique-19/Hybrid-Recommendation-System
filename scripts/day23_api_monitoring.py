"""
===========================================================================
DAY 23 - API MONITORING & PERFORMANCE ANALYSIS
===========================================================================

Purpose:
    Analyze Day 22 API performance/load-test results.

Inputs:
    evaluation/day22_api_performance_results.csv
    evaluation/day22_api_performance_summary.csv
    evaluation/day22_concurrent_load_results.csv
    evaluation/day22_api_performance_report.json

Outputs:
    evaluation/day23_endpoint_analysis.csv
    evaluation/day23_performance_summary.csv
    evaluation/day23_monitoring_report.json

Plots:
    evaluation/plots/day23_endpoint_latency.png
    evaluation/plots/day23_latency_distribution.png
    evaluation/plots/day23_success_rate.png
    evaluation/plots/day23_throughput.png
    evaluation/plots/day23_performance_dashboard.png
===========================================================================

Run from project root:

    python scripts/day23_api_monitoring.py
"""

import os
import json
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


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

os.makedirs(
    EVALUATION_DIR,
    exist_ok=True
)

os.makedirs(
    PLOTS_DIR,
    exist_ok=True
)


# ===========================================================================
# FILE PATHS
# ===========================================================================

DAY22_REQUEST_RESULTS = os.path.join(
    EVALUATION_DIR,
    "day22_api_performance_results.csv"
)

DAY22_PERFORMANCE_SUMMARY = os.path.join(
    EVALUATION_DIR,
    "day22_api_performance_summary.csv"
)

DAY22_CONCURRENT_RESULTS = os.path.join(
    EVALUATION_DIR,
    "day22_concurrent_load_results.csv"
)

DAY22_JSON_REPORT = os.path.join(
    EVALUATION_DIR,
    "day22_api_performance_report.json"
)


DAY23_ENDPOINT_ANALYSIS = os.path.join(
    EVALUATION_DIR,
    "day23_endpoint_analysis.csv"
)

DAY23_PERFORMANCE_SUMMARY = os.path.join(
    EVALUATION_DIR,
    "day23_performance_summary.csv"
)

DAY23_JSON_REPORT = os.path.join(
    EVALUATION_DIR,
    "day23_monitoring_report.json"
)


# ===========================================================================
# PLOT PATHS
# ===========================================================================

LATENCY_PLOT = os.path.join(
    PLOTS_DIR,
    "day23_endpoint_latency.png"
)

DISTRIBUTION_PLOT = os.path.join(
    PLOTS_DIR,
    "day23_latency_distribution.png"
)

SUCCESS_RATE_PLOT = os.path.join(
    PLOTS_DIR,
    "day23_success_rate.png"
)

THROUGHPUT_PLOT = os.path.join(
    PLOTS_DIR,
    "day23_throughput.png"
)

DASHBOARD_PLOT = os.path.join(
    PLOTS_DIR,
    "day23_performance_dashboard.png"
)


# ===========================================================================
# HELPER FUNCTIONS
# ===========================================================================

def print_header(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def safe_float(value, default=np.nan):
    try:
        if pd.isna(value):
            return default

        return float(value)

    except Exception:
        return default


def safe_int(value, default=0):
    try:
        if pd.isna(value):
            return default

        return int(value)

    except Exception:
        return default


def classify_latency(latency):
    """
    Classify endpoint latency.

    Thresholds:
        < 50 ms       -> Excellent
        50-100 ms     -> Good
        100-200 ms    -> Moderate
        200-500 ms    -> Slow
        > 500 ms      -> Critical
    """

    latency = safe_float(latency)

    if np.isnan(latency):
        return "Unknown"

    if latency < 50:
        return "Excellent"

    if latency < 100:
        return "Good"

    if latency < 200:
        return "Moderate"

    if latency < 500:
        return "Slow"

    return "Critical"


def classify_success_rate(success_rate):
    """
    Classify endpoint reliability.
    """

    success_rate = safe_float(success_rate)

    if np.isnan(success_rate):
        return "Unknown"

    if success_rate >= 99:
        return "Excellent"

    if success_rate >= 95:
        return "Good"

    if success_rate >= 90:
        return "Moderate"

    if success_rate >= 75:
        return "Poor"

    return "Critical"


def classify_endpoint(avg_latency, success_rate):
    """
    Combined endpoint health classification.
    """

    latency_status = classify_latency(avg_latency)
    success_status = classify_success_rate(success_rate)

    if (
        latency_status == "Critical"
        or success_status == "Critical"
    ):
        return "Critical"

    if (
        latency_status == "Slow"
        or success_status == "Poor"
    ):
        return "Needs Attention"

    if (
        latency_status == "Moderate"
        or success_status == "Moderate"
    ):
        return "Monitor"

    return "Healthy"


def load_csv(path, name):
    """
    Load CSV safely.
    """

    print(f"Loading {name}:")
    print(path)

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Required file not found:\n{path}"
        )

    df = pd.read_csv(path)

    print(f"{name} shape:")
    print(df.shape)

    return df


def load_json(path, name):
    """
    Load JSON safely.
    """

    print(f"Loading {name}:")
    print(path)

    if not os.path.exists(path):
        print("JSON file not found. Continuing without it.")
        return {}

    try:
        with open(
            path,
            "r",
            encoding="utf-8"
        ) as file:
            return json.load(file)

    except Exception as exc:
        print(
            f"Warning: Could not parse JSON: {exc}"
        )

        return {}


# ===========================================================================
# START
# ===========================================================================

print_header(
    "DAY 23 - API MONITORING & PERFORMANCE ANALYSIS"
)

print("Base directory:")
print(BASE_DIR)

print()

print("Evaluation directory:")
print(EVALUATION_DIR)

print()

print("Plots directory:")
print(PLOTS_DIR)


# ===========================================================================
# LOAD DAY 22 DATA
# ===========================================================================

print_header(
    "LOADING DAY 22 PERFORMANCE DATA"
)

request_results = load_csv(
    DAY22_REQUEST_RESULTS,
    "Day 22 request-level results"
)

performance_summary = load_csv(
    DAY22_PERFORMANCE_SUMMARY,
    "Day 22 performance summary"
)

concurrent_results = load_csv(
    DAY22_CONCURRENT_RESULTS,
    "Day 22 concurrent load results"
)

day22_report = load_json(
    DAY22_JSON_REPORT,
    "Day 22 JSON report"
)


# ===========================================================================
# DISPLAY INPUT INFORMATION
# ===========================================================================

print_header(
    "INPUT DATA OVERVIEW"
)

print(
    f"Request-level records: "
    f"{len(request_results)}"
)

print(
    f"Performance summary records: "
    f"{len(performance_summary)}"
)

print(
    f"Concurrent load records: "
    f"{len(concurrent_results)}"
)


# ===========================================================================
# NORMALIZE COLUMN NAMES
# ===========================================================================

request_results.columns = [
    str(column).strip()
    for column in request_results.columns
]

performance_summary.columns = [
    str(column).strip()
    for column in performance_summary.columns
]

concurrent_results.columns = [
    str(column).strip()
    for column in concurrent_results.columns
]


# ===========================================================================
# REQUEST-LEVEL DATA CLEANING
# ===========================================================================

print_header(
    "CLEANING REQUEST-LEVEL DATA"
)

required_request_columns = [
    "endpoint_name",
    "request_number",
    "actual_status",
    "success",
    "latency_ms",
    "response_size_bytes"
]

missing_request_columns = [
    column
    for column in required_request_columns
    if column not in request_results.columns
]

if missing_request_columns:

    raise ValueError(
        "Missing columns in request-level results: "
        + ", ".join(missing_request_columns)
    )


request_results["latency_ms"] = pd.to_numeric(
    request_results["latency_ms"],
    errors="coerce"
)

request_results["actual_status"] = pd.to_numeric(
    request_results["actual_status"],
    errors="coerce"
)

request_results["response_size_bytes"] = pd.to_numeric(
    request_results["response_size_bytes"],
    errors="coerce"
)

request_results["success"] = (
    request_results["success"]
    .astype(str)
    .str.lower()
    .isin(
        [
            "true",
            "1",
            "yes"
        ]
    )
)


request_results = request_results.dropna(
    subset=[
        "endpoint_name",
        "latency_ms"
    ]
).copy()


print(
    "Clean request-level records:",
    len(request_results)
)


# ===========================================================================
# REQUEST-LEVEL BASIC STATISTICS
# ===========================================================================

print_header(
    "REQUEST-LEVEL STATISTICS"
)

total_requests = len(
    request_results
)

successful_requests = int(
    request_results["success"].sum()
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

overall_average_latency = (
    request_results["latency_ms"].mean()
)

overall_median_latency = (
    request_results["latency_ms"].median()
)

overall_min_latency = (
    request_results["latency_ms"].min()
)

overall_max_latency = (
    request_results["latency_ms"].max()
)

overall_p95_latency = (
    request_results["latency_ms"].quantile(
        0.95
    )
)

overall_p99_latency = (
    request_results["latency_ms"].quantile(
        0.99
    )
)


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

print(
    f"P99 latency: "
    f"{overall_p99_latency:.2f} ms"
)


# ===========================================================================
# ENDPOINT ANALYSIS
# ===========================================================================

print_header(
    "CREATING ENDPOINT MONITORING ANALYSIS"
)

endpoint_rows = []


for endpoint_name, group in request_results.groupby(
    "endpoint_name"
):

    total = len(group)

    successful = int(
        group["success"].sum()
    )

    failed = (
        total
        - successful
    )

    success_rate = (
        successful
        / total
        * 100
        if total > 0
        else 0
    )

    average_latency = (
        group["latency_ms"].mean()
    )

    median_latency = (
        group["latency_ms"].median()
    )

    minimum_latency = (
        group["latency_ms"].min()
    )

    maximum_latency = (
        group["latency_ms"].max()
    )

    p95_latency = (
        group["latency_ms"].quantile(
            0.95
        )
    )

    p99_latency = (
        group["latency_ms"].quantile(
            0.99
        )
    )

    average_response_size = (
        group["response_size_bytes"].mean()
    )

    average_status = (
        group["actual_status"].mean()
    )

    latency_status = classify_latency(
        average_latency
    )

    reliability_status = classify_success_rate(
        success_rate
    )

    health_status = classify_endpoint(
        average_latency,
        success_rate
    )

    endpoint_rows.append(
        {
            "endpoint_name": endpoint_name,
            "total_requests": total,
            "successful_requests": successful,
            "failed_requests": failed,
            "success_rate_percent": success_rate,
            "average_latency_ms": average_latency,
            "median_latency_ms": median_latency,
            "minimum_latency_ms": minimum_latency,
            "maximum_latency_ms": maximum_latency,
            "p95_latency_ms": p95_latency,
            "p99_latency_ms": p99_latency,
            "average_response_size_bytes": average_response_size,
            "average_http_status": average_status,
            "latency_status": latency_status,
            "reliability_status": reliability_status,
            "health_status": health_status
        }
    )


endpoint_analysis = pd.DataFrame(
    endpoint_rows
)


endpoint_analysis = endpoint_analysis.sort_values(
    "average_latency_ms",
    ascending=False
).reset_index(
    drop=True
)


print()
print(
    endpoint_analysis.to_string(
        index=False
    )
)


# ===========================================================================
# BOTTLENECK IDENTIFICATION
# ===========================================================================

print_header(
    "IDENTIFYING PERFORMANCE BOTTLENECKS"
)

if len(endpoint_analysis) > 0:

    slowest_endpoint = (
        endpoint_analysis.iloc[0]
    )

    fastest_endpoint = (
        endpoint_analysis.sort_values(
            "average_latency_ms",
            ascending=True
        ).iloc[0]
    )

    highest_p95_endpoint = (
        endpoint_analysis.sort_values(
            "p95_latency_ms",
            ascending=False
        ).iloc[0]
    )

    lowest_success_endpoint = (
        endpoint_analysis.sort_values(
            "success_rate_percent",
            ascending=True
        ).iloc[0]
    )

    print(
        "Slowest endpoint:"
    )

    print(
        f"  {slowest_endpoint['endpoint_name']}"
    )

    print(
        f"  Average latency: "
        f"{slowest_endpoint['average_latency_ms']:.2f} ms"
    )

    print()

    print(
        "Fastest endpoint:"
    )

    print(
        f"  {fastest_endpoint['endpoint_name']}"
    )

    print(
        f"  Average latency: "
        f"{fastest_endpoint['average_latency_ms']:.2f} ms"
    )

    print()

    print(
        "Highest P95 latency endpoint:"
    )

    print(
        f"  {highest_p95_endpoint['endpoint_name']}"
    )

    print(
        f"  P95 latency: "
        f"{highest_p95_endpoint['p95_latency_ms']:.2f} ms"
    )

    print()

    print(
        "Lowest success-rate endpoint:"
    )

    print(
        f"  {lowest_success_endpoint['endpoint_name']}"
    )

    print(
        f"  Success rate: "
        f"{lowest_success_endpoint['success_rate_percent']:.2f}%"
    )


# ===========================================================================
# LATENCY THRESHOLD ANALYSIS
# ===========================================================================

print_header(
    "LATENCY THRESHOLD ANALYSIS"
)

latency_thresholds = {
    "under_50ms": (
        request_results["latency_ms"] < 50
    ),
    "50_to_100ms": (
        (request_results["latency_ms"] >= 50)
        &
        (request_results["latency_ms"] < 100)
    ),
    "100_to_200ms": (
        (request_results["latency_ms"] >= 100)
        &
        (request_results["latency_ms"] < 200)
    ),
    "200_to_500ms": (
        (request_results["latency_ms"] >= 200)
        &
        (request_results["latency_ms"] < 500)
    ),
    "over_500ms": (
        request_results["latency_ms"] >= 500
    )
}


latency_threshold_rows = []


for threshold_name, mask in latency_thresholds.items():

    count = int(
        mask.sum()
    )

    percentage = (
        count
        / total_requests
        * 100
        if total_requests > 0
        else 0
    )

    latency_threshold_rows.append(
        {
            "latency_bucket": threshold_name,
            "request_count": count,
            "percentage": percentage
        }
    )


latency_threshold_df = pd.DataFrame(
    latency_threshold_rows
)


print(
    latency_threshold_df.to_string(
        index=False
    )
)


# ===========================================================================
# CONCURRENT LOAD ANALYSIS
# ===========================================================================

print_header(
    "CONCURRENT LOAD ANALYSIS"
)

concurrent_analysis = {}

if len(concurrent_results) > 0:

    concurrent_results["latency_ms"] = pd.to_numeric(
        concurrent_results.get(
            "latency_ms",
            pd.Series(dtype=float)
        ),
        errors="coerce"
    )

    if "success" in concurrent_results.columns:

        concurrent_results["success"] = (
            concurrent_results["success"]
            .astype(str)
            .str.lower()
            .isin(
                [
                    "true",
                    "1",
                    "yes"
                ]
            )
        )

    else:

        concurrent_results["success"] = True


    concurrent_total = len(
        concurrent_results
    )

    concurrent_successful = int(
        concurrent_results["success"].sum()
    )

    concurrent_failed = (
        concurrent_total
        - concurrent_successful
    )

    concurrent_success_rate = (
        concurrent_successful
        / concurrent_total
        * 100
        if concurrent_total > 0
        else 0
    )

    concurrent_average_latency = (
        concurrent_results["latency_ms"].mean()
    )

    concurrent_median_latency = (
        concurrent_results["latency_ms"].median()
    )

    concurrent_p95_latency = (
        concurrent_results["latency_ms"].quantile(
            0.95
        )
    )

    concurrent_p99_latency = (
        concurrent_results["latency_ms"].quantile(
            0.99
        )
    )

    concurrent_analysis = {
        "total_requests": concurrent_total,
        "successful_requests": concurrent_successful,
        "failed_requests": concurrent_failed,
        "success_rate_percent": concurrent_success_rate,
        "average_latency_ms": concurrent_average_latency,
        "median_latency_ms": concurrent_median_latency,
        "p95_latency_ms": concurrent_p95_latency,
        "p99_latency_ms": concurrent_p99_latency
    }


    print(
        f"Concurrent requests: "
        f"{concurrent_total}"
    )

    print(
        f"Successful: "
        f"{concurrent_successful}"
    )

    print(
        f"Failed: "
        f"{concurrent_failed}"
    )

    print(
        f"Success rate: "
        f"{concurrent_success_rate:.2f}%"
    )

    print(
        f"Average latency: "
        f"{concurrent_average_latency:.2f} ms"
    )

    print(
        f"Median latency: "
        f"{concurrent_median_latency:.2f} ms"
    )

    print(
        f"P95 latency: "
        f"{concurrent_p95_latency:.2f} ms"
    )

    print(
        f"P99 latency: "
        f"{concurrent_p99_latency:.2f} ms"
    )


# ===========================================================================
# THROUGHPUT EXTRACTION
# ===========================================================================

print_header(
    "THROUGHPUT ANALYSIS"
)

throughput = np.nan

if "throughput_requests_per_second" in performance_summary.columns:

    throughput_values = pd.to_numeric(
        performance_summary[
            "throughput_requests_per_second"
        ],
        errors="coerce"
    ).dropna()

    if len(throughput_values) > 0:

        throughput = float(
            throughput_values.iloc[-1]
        )


if np.isnan(throughput):

    if isinstance(day22_report, dict):

        possible_keys = [
            "throughput_requests_per_second",
            "concurrent_throughput",
            "throughput"
        ]

        for key in possible_keys:

            if key in day22_report:

                value = safe_float(
                    day22_report[key]
                )

                if not np.isnan(value):

                    throughput = value
                    break


if np.isnan(throughput):

    if (
        concurrent_analysis
        and concurrent_analysis.get(
            "total_requests",
            0
        ) > 0
    ):

        # If total duration is available in Day 22
        # JSON/report, use it.
        duration = np.nan

        if isinstance(day22_report, dict):

            possible_duration_keys = [
                "concurrent_duration_seconds",
                "total_duration_seconds",
                "duration_seconds"
            ]

            for key in possible_duration_keys:

                if key in day22_report:

                    duration = safe_float(
                        day22_report[key]
                    )

                    if not np.isnan(duration):
                        break

        if (
            not np.isnan(duration)
            and duration > 0
        ):

            throughput = (
                concurrent_analysis[
                    "total_requests"
                ]
                / duration
            )


print(
    "Throughput:"
)

if np.isnan(throughput):

    print(
        "  Not available in source data."
    )

else:

    print(
        f"  {throughput:.2f} requests/sec"
    )


# ===========================================================================
# ENDPOINT HEALTH SUMMARY
# ===========================================================================

print_header(
    "ENDPOINT HEALTH SUMMARY"
)

health_counts = (
    endpoint_analysis[
        "health_status"
    ]
    .value_counts()
    .to_dict()
)

for status, count in health_counts.items():

    print(
        f"{status}: {count}"
    )


healthy_endpoints = int(
    (
        endpoint_analysis[
            "health_status"
        ] == "Healthy"
    ).sum()
)

attention_endpoints = int(
    (
        endpoint_analysis[
            "health_status"
        ] == "Needs Attention"
    ).sum()
)

monitor_endpoints = int(
    (
        endpoint_analysis[
            "health_status"
        ] == "Monitor"
    ).sum()
)

critical_endpoints = int(
    (
        endpoint_analysis[
            "health_status"
        ] == "Critical"
    ).sum()
)


# ===========================================================================
# PERFORMANCE SCORE
# ===========================================================================

print_header(
    "CALCULATING PERFORMANCE SCORE"
)

if len(endpoint_analysis) > 0:

    average_endpoint_success = (
        endpoint_analysis[
            "success_rate_percent"
        ].mean()
    )

    average_endpoint_latency = (
        endpoint_analysis[
            "average_latency_ms"
        ].mean()
    )

    latency_score = max(
        0,
        100
        - min(
            average_endpoint_latency,
            100
        )
    )

    success_score = min(
        max(
            average_endpoint_success,
            0
        ),
        100
    )

    health_score = (
        0.6 * success_score
        +
        0.4 * latency_score
    )

else:

    average_endpoint_success = 0
    average_endpoint_latency = 0
    health_score = 0


print(
    f"Average endpoint success rate: "
    f"{average_endpoint_success:.2f}%"
)

print(
    f"Average endpoint latency: "
    f"{average_endpoint_latency:.2f} ms"
)

print(
    f"Performance score: "
    f"{health_score:.2f}/100"
)


# ===========================================================================
# SAVE ENDPOINT ANALYSIS
# ===========================================================================

print_header(
    "SAVING ENDPOINT ANALYSIS"
)

endpoint_analysis.to_csv(
    DAY23_ENDPOINT_ANALYSIS,
    index=False
)

print(
    "Endpoint analysis saved:"
)

print(
    DAY23_ENDPOINT_ANALYSIS
)


# ===========================================================================
# FINAL PERFORMANCE SUMMARY TABLE
# ===========================================================================

print_header(
    "CREATING FINAL PERFORMANCE SUMMARY"
)

summary_rows = []


for _, row in endpoint_analysis.iterrows():

    summary_rows.append(
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
            "average_response_size_bytes": row[
                "average_response_size_bytes"
            ],
            "throughput_requests_per_second": np.nan,
            "health_status": row[
                "health_status"
            ]
        }
    )


if concurrent_analysis:

    summary_rows.append(
        {
            "metric_type": "concurrent_load",
            "endpoint_name": "Concurrent Load Test",
            "total_requests": concurrent_analysis[
                "total_requests"
            ],
            "successful_requests": concurrent_analysis[
                "successful_requests"
            ],
            "failed_requests": concurrent_analysis[
                "failed_requests"
            ],
            "success_rate_percent": concurrent_analysis[
                "success_rate_percent"
            ],
            "average_latency_ms": concurrent_analysis[
                "average_latency_ms"
            ],
            "median_latency_ms": concurrent_analysis[
                "median_latency_ms"
            ],
            "minimum_latency_ms": np.nan,
            "maximum_latency_ms": np.nan,
            "p95_latency_ms": concurrent_analysis[
                "p95_latency_ms"
            ],
            "p99_latency_ms": concurrent_analysis[
                "p99_latency_ms"
            ],
            "average_response_size_bytes": np.nan,
            "throughput_requests_per_second": throughput,
            "health_status": classify_endpoint(
                concurrent_analysis[
                    "average_latency_ms"
                ],
                concurrent_analysis[
                    "success_rate_percent"
                ]
            )
        }
    )


summary_rows.append(
    {
        "metric_type": "overall",
        "endpoint_name": "All Endpoints",
        "total_requests": total_requests,
        "successful_requests": successful_requests,
        "failed_requests": failed_requests,
        "success_rate_percent": overall_success_rate,
        "average_latency_ms": overall_average_latency,
        "median_latency_ms": overall_median_latency,
        "minimum_latency_ms": overall_min_latency,
        "maximum_latency_ms": overall_max_latency,
        "p95_latency_ms": overall_p95_latency,
        "p99_latency_ms": overall_p99_latency,
        "average_response_size_bytes": (
            request_results[
                "response_size_bytes"
            ].mean()
        ),
        "throughput_requests_per_second": np.nan,
        "health_status": classify_endpoint(
            overall_average_latency,
            overall_success_rate
        )
    }
)


performance_summary_df = pd.DataFrame(
    summary_rows
)


print(
    performance_summary_df.to_string(
        index=False
    )
)


performance_summary_df.to_csv(
    DAY23_PERFORMANCE_SUMMARY,
    index=False
)

print()

print(
    "Performance summary saved:"
)

print(
    DAY23_PERFORMANCE_SUMMARY
)


# ===========================================================================
# PLOT 1 - ENDPOINT LATENCY
# ===========================================================================

print_header(
    "GENERATING ENDPOINT LATENCY PLOT"
)

plot_data = endpoint_analysis.sort_values(
    "average_latency_ms",
    ascending=True
)

plt.figure(
    figsize=(12, 7)
)

plt.barh(
    plot_data[
        "endpoint_name"
    ],
    plot_data[
        "average_latency_ms"
    ]
)

plt.xlabel(
    "Average Latency (ms)"
)

plt.ylabel(
    "Endpoint"
)

plt.title(
    "Day 23 - Average Endpoint Latency"
)

plt.tight_layout()

plt.savefig(
    LATENCY_PLOT,
    dpi=150,
    bbox_inches="tight"
)

plt.close()

print(
    "Plot saved:"
)

print(
    LATENCY_PLOT
)


# ===========================================================================
# PLOT 2 - LATENCY DISTRIBUTION
# ===========================================================================

print_header(
    "GENERATING LATENCY DISTRIBUTION PLOT"
)

plt.figure(
    figsize=(11, 7)
)

plt.hist(
    request_results[
        "latency_ms"
    ],
    bins=25
)

plt.xlabel(
    "Latency (ms)"
)

plt.ylabel(
    "Number of Requests"
)

plt.title(
    "Day 23 - API Latency Distribution"
)

plt.tight_layout()

plt.savefig(
    DISTRIBUTION_PLOT,
    dpi=150,
    bbox_inches="tight"
)

plt.close()

print(
    "Plot saved:"
)

print(
    DISTRIBUTION_PLOT
)


# ===========================================================================
# PLOT 3 - SUCCESS RATE
# ===========================================================================

print_header(
    "GENERATING SUCCESS RATE PLOT"
)

plot_data = endpoint_analysis.sort_values(
    "success_rate_percent",
    ascending=True
)

plt.figure(
    figsize=(12, 7)
)

plt.barh(
    plot_data[
        "endpoint_name"
    ],
    plot_data[
        "success_rate_percent"
    ]
)

plt.xlabel(
    "Success Rate (%)"
)

plt.ylabel(
    "Endpoint"
)

plt.title(
    "Day 23 - Endpoint Success Rate"
)

plt.xlim(
    0,
    105
)

plt.tight_layout()

plt.savefig(
    SUCCESS_RATE_PLOT,
    dpi=150,
    bbox_inches="tight"
)

plt.close()

print(
    "Plot saved:"
)

print(
    SUCCESS_RATE_PLOT
)


# ===========================================================================
# PLOT 4 - THROUGHPUT
# ===========================================================================

print_header(
    "GENERATING THROUGHPUT PLOT"
)

plt.figure(
    figsize=(10, 6)
)

if np.isnan(throughput):

    plt.text(
        0.5,
        0.5,
        "Throughput data not available",
        ha="center",
        va="center",
        fontsize=14
    )

    plt.xlim(
        0,
        1
    )

    plt.ylim(
        0,
        1
    )

else:

    plt.bar(
        ["Concurrent Load"],
        [throughput]
    )

    plt.ylabel(
        "Requests / Second"
    )

    plt.title(
        "Day 23 - Concurrent API Throughput"
    )


plt.tight_layout()

plt.savefig(
    THROUGHPUT_PLOT,
    dpi=150,
    bbox_inches="tight"
)

plt.close()

print(
    "Plot saved:"
)

print(
    THROUGHPUT_PLOT
)


# ===========================================================================
# PLOT 5 - PERFORMANCE DASHBOARD
# ===========================================================================

print_header(
    "GENERATING PERFORMANCE DASHBOARD"
)

fig = plt.figure(
    figsize=(15, 10)
)

ax1 = fig.add_subplot(
    2,
    2,
    1
)

ax2 = fig.add_subplot(
    2,
    2,
    2
)

ax3 = fig.add_subplot(
    2,
    2,
    3
)

ax4 = fig.add_subplot(
    2,
    2,
    4
)


# Average latency

latency_plot_data = endpoint_analysis.sort_values(
    "average_latency_ms",
    ascending=False
)

ax1.bar(
    latency_plot_data[
        "endpoint_name"
    ],
    latency_plot_data[
        "average_latency_ms"
    ]
)

ax1.set_title(
    "Average Latency"
)

ax1.set_ylabel(
    "ms"
)

ax1.tick_params(
    axis="x",
    rotation=70
)


# P95 latency

ax2.bar(
    latency_plot_data[
        "endpoint_name"
    ],
    latency_plot_data[
        "p95_latency_ms"
    ]
)

ax2.set_title(
    "P95 Latency"
)

ax2.set_ylabel(
    "ms"
)

ax2.tick_params(
    axis="x",
    rotation=70
)


# Success rate

success_plot_data = endpoint_analysis.sort_values(
    "success_rate_percent",
    ascending=False
)

ax3.bar(
    success_plot_data[
        "endpoint_name"
    ],
    success_plot_data[
        "success_rate_percent"
    ]
)

ax3.set_title(
    "Success Rate"
)

ax3.set_ylabel(
    "%"
)

ax3.set_ylim(
    0,
    105
)

ax3.tick_params(
    axis="x",
    rotation=70
)


# Latency histogram

ax4.hist(
    request_results[
        "latency_ms"
    ],
    bins=20
)

ax4.set_title(
    "Latency Distribution"
)

ax4.set_xlabel(
    "Latency (ms)"
)

ax4.set_ylabel(
    "Requests"
)


fig.suptitle(
    "Day 23 - API Performance Monitoring Dashboard",
    fontsize=16
)

fig.tight_layout(
    rect=[
        0,
        0,
        1,
        0.96
    ]
)

fig.savefig(
    DASHBOARD_PLOT,
    dpi=150,
    bbox_inches="tight"
)

plt.close(
    fig
)

print(
    "Dashboard saved:"
)

print(
    DASHBOARD_PLOT
)


# ===========================================================================
# GENERATE MONITORING RECOMMENDATIONS
# ===========================================================================

print_header(
    "GENERATING MONITORING INSIGHTS"
)

insights = []


if overall_success_rate >= 99:

    insights.append(
        "API reliability is high based on the observed request success rate."
    )

else:

    insights.append(
        "API reliability requires attention because failed requests were observed."
    )


if overall_p95_latency < 50:

    insights.append(
        "Overall P95 latency is below 50 ms."
    )

elif overall_p95_latency < 100:

    insights.append(
        "Overall P95 latency is below 100 ms but should continue to be monitored."
    )

elif overall_p95_latency < 200:

    insights.append(
        "Overall P95 latency is moderate and should be monitored."
    )

else:

    insights.append(
        "Overall P95 latency is high and may require performance optimization."
    )


if len(endpoint_analysis) > 0:

    slow_endpoint = endpoint_analysis.iloc[0]

    insights.append(
        "Slowest endpoint by average latency: "
        f"{slow_endpoint['endpoint_name']} "
        f"({slow_endpoint['average_latency_ms']:.2f} ms)."
    )


if concurrent_analysis:

    if (
        concurrent_analysis[
            "success_rate_percent"
        ] >= 99
    ):

        insights.append(
            "Concurrent load testing achieved a high success rate."
        )

    else:

        insights.append(
            "Concurrent load testing showed failed requests."
        )


    if not np.isnan(
        concurrent_analysis[
            "p95_latency_ms"
        ]
    ):

        insights.append(
            "Concurrent P95 latency was "
            f"{concurrent_analysis['p95_latency_ms']:.2f} ms."
        )


if throughput is not None and not np.isnan(
    throughput
):

    insights.append(
        f"Observed concurrent throughput: "
        f"{throughput:.2f} requests/sec."
    )


for index, insight in enumerate(
    insights,
    start=1
):

    print(
        f"{index}. {insight}"
    )


# ===========================================================================
# CREATE JSON REPORT
# ===========================================================================

print_header(
    "CREATING MONITORING JSON REPORT"
)

report = {
    "day": 23,
    "title": (
        "API Monitoring & Performance Analysis"
    ),
    "generated_at": datetime.now().isoformat(),
    "project_directory": BASE_DIR,

    "input_files": {
        "request_results": DAY22_REQUEST_RESULTS,
        "performance_summary": DAY22_PERFORMANCE_SUMMARY,
        "concurrent_results": DAY22_CONCURRENT_RESULTS,
        "json_report": DAY22_JSON_REPORT
    },

    "overall_metrics": {
        "total_requests": total_requests,
        "successful_requests": successful_requests,
        "failed_requests": failed_requests,
        "success_rate_percent": (
            round(
                overall_success_rate,
                4
            )
        ),
        "average_latency_ms": (
            round(
                overall_average_latency,
                4
            )
        ),
        "median_latency_ms": (
            round(
                overall_median_latency,
                4
            )
        ),
        "minimum_latency_ms": (
            round(
                overall_min_latency,
                4
            )
        ),
        "maximum_latency_ms": (
            round(
                overall_max_latency,
                4
            )
        ),
        "p95_latency_ms": (
            round(
                overall_p95_latency,
                4
            )
        ),
        "p99_latency_ms": (
            round(
                overall_p99_latency,
                4
            )
        )
    },

    "endpoint_health": {
        "total_endpoints": int(
            len(endpoint_analysis)
        ),
        "healthy": healthy_endpoints,
        "monitor": monitor_endpoints,
        "needs_attention": attention_endpoints,
        "critical": critical_endpoints
    },

    "performance_score": round(
        health_score,
        4
    ),

    "latency_distribution": (
        latency_threshold_df.to_dict(
            orient="records"
        )
    ),

    "concurrent_load": (
        concurrent_analysis
        if concurrent_analysis
        else {}
    ),

    "throughput_requests_per_second": (
        None
        if np.isnan(throughput)
        else round(
            throughput,
            4
        )
    ),

    "slowest_endpoint": (
        slowest_endpoint[
            "endpoint_name"
        ]
        if len(endpoint_analysis) > 0
        else None
    ),

    "fastest_endpoint": (
        fastest_endpoint[
            "endpoint_name"
        ]
        if len(endpoint_analysis) > 0
        else None
    ),

    "highest_p95_endpoint": (
        highest_p95_endpoint[
            "endpoint_name"
        ]
        if len(endpoint_analysis) > 0
        else None
    ),

    "insights": insights,

    "output_files": {
        "endpoint_analysis": DAY23_ENDPOINT_ANALYSIS,
        "performance_summary": DAY23_PERFORMANCE_SUMMARY,
        "monitoring_report": DAY23_JSON_REPORT,
        "endpoint_latency_plot": LATENCY_PLOT,
        "latency_distribution_plot": DISTRIBUTION_PLOT,
        "success_rate_plot": SUCCESS_RATE_PLOT,
        "throughput_plot": THROUGHPUT_PLOT,
        "dashboard_plot": DASHBOARD_PLOT
    }
}


with open(
    DAY23_JSON_REPORT,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        report,
        file,
        indent=4,
        default=lambda value: (
            float(value)
            if isinstance(
                value,
                (np.float32, np.float64)
            )
            else int(value)
            if isinstance(
                value,
                (np.int32, np.int64)
            )
            else value
        )
    )


print(
    "Monitoring report saved:"
)

print(
    DAY23_JSON_REPORT
)


# ===========================================================================
# FINAL SUMMARY
# ===========================================================================

print_header(
    "DAY 23 FINAL MONITORING SUMMARY"
)

print(
    f"Total requests analyzed: "
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

print(
    f"P99 latency: "
    f"{overall_p99_latency:.2f} ms"
)

print(
    f"Endpoints analyzed: "
    f"{len(endpoint_analysis)}"
)

print(
    f"Healthy endpoints: "
    f"{healthy_endpoints}"
)

print(
    f"Endpoints requiring monitoring: "
    f"{monitor_endpoints}"
)

print(
    f"Endpoints needing attention: "
    f"{attention_endpoints}"
)

print(
    f"Critical endpoints: "
    f"{critical_endpoints}"
)

if not np.isnan(throughput):

    print(
        f"Concurrent throughput: "
        f"{throughput:.2f} requests/sec"
    )

if concurrent_analysis:

    print(
        f"Concurrent success rate: "
        f"{concurrent_analysis['success_rate_percent']:.2f}%"
    )

    print(
        f"Concurrent average latency: "
        f"{concurrent_analysis['average_latency_ms']:.2f} ms"
    )

    print(
        f"Concurrent P95 latency: "
        f"{concurrent_analysis['p95_latency_ms']:.2f} ms"
    )


print(
    f"Performance score: "
    f"{health_score:.2f}/100"
)


# ===========================================================================
# SAVED FILES
# ===========================================================================

print_header(
    "SAVED FILES"
)

print(
    "Endpoint analysis:"
)

print(
    DAY23_ENDPOINT_ANALYSIS
)

print()

print(
    "Performance summary:"
)

print(
    DAY23_PERFORMANCE_SUMMARY
)

print()

print(
    "Monitoring JSON report:"
)

print(
    DAY23_JSON_REPORT
)

print()

print(
    "Endpoint latency plot:"
)

print(
    LATENCY_PLOT
)

print()

print(
    "Latency distribution plot:"
)

print(
    DISTRIBUTION_PLOT
)

print()

print(
    "Success rate plot:"
)

print(
    SUCCESS_RATE_PLOT
)

print()

print(
    "Throughput plot:"
)

print(
    THROUGHPUT_PLOT
)

print()

print(
    "Performance dashboard:"
)

print(
    DASHBOARD_PLOT
)


# ===========================================================================
# COMPLETE
# ===========================================================================

print_header(
    "DAY 23 COMPLETED SUCCESSFULLY"
)