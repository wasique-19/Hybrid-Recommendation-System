# ============================================================================
# DAY 20 - MODEL COMPARISON
# ============================================================================
#
# Purpose:
#   1. Load Day 18 evaluation metrics
#   2. Load Day 19 evaluation metrics
#   3. Load Day 19 user-level evaluation
#   4. Load Day 19 recommendation-level results
#   5. Create model comparison table
#   6. Create Day 18 vs Day 19 improvement analysis
#   7. Generate comparison plots
#   8. Perform user-level analysis
#   9. Perform recommendation-level analysis
#  10. Create final consolidated CSV
#
# Expected project structure:
#
# Hybrid Recommendation System/
# ├── evaluation/
# │   ├── day18_deep_hybrid_metrics.csv
# │   ├── day19_recommendation_metrics.csv
# │   ├── day19_user_evaluation.csv
# │   ├── day19_recommendations.csv
# │   └── plots/
# └── scripts/
#     └── day20_model_comparison.py
#
# ============================================================================

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================================
# CONFIGURATION
# ============================================================================

warnings.filterwarnings("ignore")

# Get project root from this script location.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)

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


# ============================================================================
# FILE PATHS
# ============================================================================

DAY18_METRICS_PATH = os.path.join(
    EVALUATION_DIR,
    "day18_deep_hybrid_metrics.csv"
)

DAY19_METRICS_PATH = os.path.join(
    EVALUATION_DIR,
    "day19_recommendation_metrics.csv"
)

DAY19_USER_EVALUATION_PATH = os.path.join(
    EVALUATION_DIR,
    "day19_user_evaluation.csv"
)

DAY19_RECOMMENDATIONS_PATH = os.path.join(
    EVALUATION_DIR,
    "day19_recommendations.csv"
)

COMPARISON_TABLE_PATH = os.path.join(
    EVALUATION_DIR,
    "day20_comparison_table.csv"
)

FINAL_SUMMARY_PATH = os.path.join(
    EVALUATION_DIR,
    "day20_final_summary.csv"
)

USER_ANALYSIS_PATH = os.path.join(
    EVALUATION_DIR,
    "day20_user_analysis.csv"
)

FINAL_COMPARISON_PATH = os.path.join(
    EVALUATION_DIR,
    "day20_model_comparison.csv"
)

RMSE_MAE_PLOT_PATH = os.path.join(
    PLOTS_DIR,
    "day20_rmse_mae_comparison.png"
)

RANKING_PLOT_PATH = os.path.join(
    PLOTS_DIR,
    "day20_ranking_metrics_comparison.png"
)

COVERAGE_PLOT_PATH = os.path.join(
    PLOTS_DIR,
    "day20_coverage_diversity_comparison.png"
)

ATTENTION_PLOT_PATH = os.path.join(
    PLOTS_DIR,
    "day20_attention_comparison.png"
)


# ============================================================================
# PRINT HELPERS
# ============================================================================

def print_separator():
    print("=" * 78)


def print_section(title):
    print()
    print_separator()
    print(title)
    print_separator()


def print_path(label, path):
    print(f"{label}:")
    print(path)


# ============================================================================
# SAFE FILE LOADING
# ============================================================================

def load_csv(path, description):
    """
    Safely load a CSV file.
    """

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"\nRequired file not found:\n{path}\n"
            f"Please make sure the previous day generated this file."
        )

    print(f"Loading {description}:")
    print(path)

    dataframe = pd.read_csv(path)

    print(f"{description} shape:")
    print(dataframe.shape)

    return dataframe


# ============================================================================
# SAFE COLUMN ACCESS
# ============================================================================

def get_value(dataframe, column, default=np.nan):
    """
    Safely retrieve the first value from a DataFrame column.

    Returns default if:
      - column does not exist
      - dataframe is empty
      - value cannot be converted
    """

    if dataframe is None:
        return default

    if not isinstance(dataframe, pd.DataFrame):
        return default

    if dataframe.empty:
        return default

    if column not in dataframe.columns:
        return default

    value = dataframe.iloc[0][column]

    if pd.isna(value):
        return default

    return value


def numeric_value(dataframe, column, default=np.nan):
    """
    Get numeric value safely.
    """

    value = get_value(
        dataframe,
        column,
        default
    )

    try:
        return float(value)
    except (TypeError, ValueError):
        return default


# ============================================================================
# REQUIRED COLUMN CHECK
# ============================================================================

def check_columns(dataframe, required_columns, dataframe_name):
    """
    Check whether required columns exist.
    """

    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError(
            f"{dataframe_name} is not a pandas DataFrame."
        )

    missing_columns = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            f"\nMissing columns in {dataframe_name}:\n"
            f"{missing_columns}\n"
            f"Available columns:\n"
            f"{list(dataframe.columns)}"
        )


# ============================================================================
# LOAD ALL DATA
# ============================================================================

def load_all_data():

    print_section(
        "DAY 20 - MODEL COMPARISON"
    )

    print()
    print("Base directory:")
    print(BASE_DIR)

    print()
    print("Evaluation directory:")
    print(EVALUATION_DIR)

    # ------------------------------------------------------------------------
    # Day 18 metrics
    # ------------------------------------------------------------------------

    print()
    print_section(
        "LOADING DAY 18 METRICS"
    )

    day18_metrics_df = load_csv(
        DAY18_METRICS_PATH,
        "Day 18 metrics"
    )

    # ------------------------------------------------------------------------
    # Day 19 metrics
    # ------------------------------------------------------------------------

    print()
    print_section(
        "LOADING DAY 19 METRICS"
    )

    day19_metrics_df = load_csv(
        DAY19_METRICS_PATH,
        "Day 19 metrics"
    )

    # ------------------------------------------------------------------------
    # Day 19 user evaluation
    # ------------------------------------------------------------------------

    print()
    print_section(
        "LOADING DAY 19 USER EVALUATION"
    )

    day19_user_evaluation_df = load_csv(
        DAY19_USER_EVALUATION_PATH,
        "Day 19 user evaluation"
    )

    # ------------------------------------------------------------------------
    # Day 19 recommendations
    # ------------------------------------------------------------------------

    print()
    print_section(
        "LOADING DAY 19 RECOMMENDATIONS"
    )

    day19_recommendations_df = load_csv(
        DAY19_RECOMMENDATIONS_PATH,
        "Day 19 recommendations"
    )

    return (
        day18_metrics_df,
        day19_metrics_df,
        day19_user_evaluation_df,
        day19_recommendations_df
    )


# ============================================================================
# VALIDATE DATA
# ============================================================================

def validate_data(
    day18_metrics_df,
    day19_metrics_df,
    day19_user_evaluation_df,
    day19_recommendations_df
):

    print_section(
        "VALIDATING INPUT DATA"
    )

    day18_required = [
        "rmse",
        "mae",
        "precision_at_10",
        "recall_at_10",
        "hit_rate_at_10",
        "ndcg_at_10",
        "catalog_coverage_at_10",
        "genre_diversity_at_10",
        "average_content_attention",
        "average_collaborative_attention",
        "evaluated_users"
    ]

    day19_required = [
        "precision_at_10",
        "recall_at_10",
        "hit_rate_at_10",
        "ndcg_at_10",
        "map_at_10",
        "catalog_coverage_at_10",
        "unique_recommended_movies",
        "genre_diversity_at_10",
        "average_content_attention",
        "average_collaborative_attention",
        "evaluated_users"
    ]

    user_required = [
        "user_id",
        "relevant_item_count",
        "recommendation_count",
        "precision_at_10",
        "recall_at_10",
        "hit_rate_at_10",
        "ndcg_at_10",
        "map_at_10"
    ]

    recommendation_required = [
        "user_id",
        "rank",
        "movie_id",
        "predicted_rating",
        "content_attention",
        "collaborative_attention"
    ]

    check_columns(
        day18_metrics_df,
        day18_required,
        "Day 18 metrics"
    )

    check_columns(
        day19_metrics_df,
        day19_required,
        "Day 19 metrics"
    )

    check_columns(
        day19_user_evaluation_df,
        user_required,
        "Day 19 user evaluation"
    )

    check_columns(
        day19_recommendations_df,
        recommendation_required,
        "Day 19 recommendations"
    )

    print()
    print("All required columns are available.")

    print()
    print("Day 19 user evaluation columns:")
    print(
        list(
            day19_user_evaluation_df.columns
        )
    )

    print()
    print("Day 19 recommendation columns:")
    print(
        list(
            day19_recommendations_df.columns
        )
    )


# ============================================================================
# DISPLAY LOADED METRICS
# ============================================================================

def display_loaded_metrics(
    day18_metrics_df,
    day19_metrics_df
):

    print_section(
        "LOADED METRICS"
    )

    print()
    print("Day 18 metrics:")
    print(
        day18_metrics_df.to_string(
            index=False
        )
    )

    print()
    print("Day 19 metrics:")
    print(
        day19_metrics_df.to_string(
            index=False
        )
    )


# ============================================================================
# BUILD MODEL COMPARISON TABLE
# ============================================================================

def create_model_comparison_table(
    day18_metrics_df,
    day19_metrics_df
):

    print_section(
        "CREATING MODEL COMPARISON TABLE"
    )

    # ------------------------------------------------------------------------
    # Day 18 values
    # ------------------------------------------------------------------------

    day18_row = {
        "model":
            "Day 18 - Deep Hybrid",

        "rmse":
            numeric_value(
                day18_metrics_df,
                "rmse"
            ),

        "mae":
            numeric_value(
                day18_metrics_df,
                "mae"
            ),

        "precision_at_10":
            numeric_value(
                day18_metrics_df,
                "precision_at_10"
            ),

        "recall_at_10":
            numeric_value(
                day18_metrics_df,
                "recall_at_10"
            ),

        "hit_rate_at_10":
            numeric_value(
                day18_metrics_df,
                "hit_rate_at_10"
            ),

        "ndcg_at_10":
            numeric_value(
                day18_metrics_df,
                "ndcg_at_10"
            ),

        "map_at_10":
            np.nan,

        "catalog_coverage_at_10":
            numeric_value(
                day18_metrics_df,
                "catalog_coverage_at_10"
            ),

        "genre_diversity_at_10":
            numeric_value(
                day18_metrics_df,
                "genre_diversity_at_10"
            ),

        "average_content_attention":
            numeric_value(
                day18_metrics_df,
                "average_content_attention"
            ),

        "average_collaborative_attention":
            numeric_value(
                day18_metrics_df,
                "average_collaborative_attention"
            ),

        "evaluated_users":
            numeric_value(
                day18_metrics_df,
                "evaluated_users"
            )
    }

    # ------------------------------------------------------------------------
    # Day 19 values
    # ------------------------------------------------------------------------

    day19_row = {
        "model":
            "Day 19 - Improved Evaluation",

        "rmse":
            np.nan,

        "mae":
            np.nan,

        "precision_at_10":
            numeric_value(
                day19_metrics_df,
                "precision_at_10"
            ),

        "recall_at_10":
            numeric_value(
                day19_metrics_df,
                "recall_at_10"
            ),

        "hit_rate_at_10":
            numeric_value(
                day19_metrics_df,
                "hit_rate_at_10"
            ),

        "ndcg_at_10":
            numeric_value(
                day19_metrics_df,
                "ndcg_at_10"
            ),

        "map_at_10":
            numeric_value(
                day19_metrics_df,
                "map_at_10"
            ),

        "catalog_coverage_at_10":
            numeric_value(
                day19_metrics_df,
                "catalog_coverage_at_10"
            ),

        "genre_diversity_at_10":
            numeric_value(
                day19_metrics_df,
                "genre_diversity_at_10"
            ),

        "average_content_attention":
            numeric_value(
                day19_metrics_df,
                "average_content_attention"
            ),

        "average_collaborative_attention":
            numeric_value(
                day19_metrics_df,
                "average_collaborative_attention"
            ),

        "evaluated_users":
            numeric_value(
                day19_metrics_df,
                "evaluated_users"
            )
    }

    comparison_df = pd.DataFrame(
        [
            day18_row,
            day19_row
        ]
    )

    # Round numeric columns for clean output.
    numeric_columns = [
        column
        for column in comparison_df.columns
        if column != "model"
    ]

    comparison_df[numeric_columns] = (
        comparison_df[numeric_columns]
        .apply(
            pd.to_numeric,
            errors="coerce"
        )
        .round(4)
    )

    print()
    print("Final model comparison:")
    print(
        comparison_df.to_string(
            index=False
        )
    )

    comparison_df.to_csv(
        COMPARISON_TABLE_PATH,
        index=False
    )

    print()
    print("Comparison table saved:")
    print(COMPARISON_TABLE_PATH)

    return comparison_df


# ============================================================================
# IMPROVEMENT ANALYSIS
# ============================================================================

def calculate_percentage_change(
    old_value,
    new_value
):

    if pd.isna(old_value) or pd.isna(new_value):
        return np.nan

    if old_value == 0:
        return np.nan

    return (
        (new_value - old_value)
        / abs(old_value)
    ) * 100.0


def classify_change(
    metric,
    old_value,
    new_value
):

    if pd.isna(old_value) or pd.isna(new_value):
        return "Not available"

    difference = new_value - old_value

    # For RMSE / MAE, lower is better.
    if metric in [
        "RMSE",
        "MAE"
    ]:

        if difference < 0:
            return "Improved"

        if difference > 0:
            return "Worsened"

        return "No change"

    # For all ranking / coverage / diversity metrics,
    # higher is better.
    if difference > 0:
        return "Improved"

    if difference < 0:
        return "Worsened"

    return "No change"


def create_improvement_analysis(
    day18_metrics_df,
    day19_metrics_df
):

    print_section(
        "DAY 19 IMPROVEMENT ANALYSIS"
    )

    metric_mapping = [
        (
            "RMSE",
            "rmse",
            True
        ),
        (
            "MAE",
            "mae",
            True
        ),
        (
            "Precision@10",
            "precision_at_10",
            False
        ),
        (
            "Recall@10",
            "recall_at_10",
            False
        ),
        (
            "Hit Rate@10",
            "hit_rate_at_10",
            False
        ),
        (
            "NDCG@10",
            "ndcg_at_10",
            False
        ),
        (
            "MAP@10",
            "map_at_10",
            False
        ),
        (
            "Catalog Coverage@10",
            "catalog_coverage_at_10",
            False
        ),
        (
            "Genre Diversity@10",
            "genre_diversity_at_10",
            False
        )
    ]

    rows = []

    for display_name, column_name, _ in metric_mapping:

        day18_value = numeric_value(
            day18_metrics_df,
            column_name
        )

        day19_value = numeric_value(
            day19_metrics_df,
            column_name
        )

        difference = np.nan

        if not pd.isna(day18_value) and not pd.isna(day19_value):
            difference = (
                day19_value
                - day18_value
            )

        percentage_change = (
            calculate_percentage_change(
                day18_value,
                day19_value
            )
        )

        interpretation = classify_change(
            display_name,
            day18_value,
            day19_value
        )

        rows.append(
            {
                "metric":
                    display_name,

                "day18_value":
                    day18_value,

                "day19_value":
                    day19_value,

                "difference":
                    difference,

                "percentage_change":
                    percentage_change,

                "interpretation":
                    interpretation
            }
        )

    improvement_df = pd.DataFrame(
        rows
    )

    numeric_columns = [
        "day18_value",
        "day19_value",
        "difference",
        "percentage_change"
    ]

    improvement_df[numeric_columns] = (
        improvement_df[numeric_columns]
        .apply(
            pd.to_numeric,
            errors="coerce"
        )
        .round(4)
    )

    print()
    print(
        improvement_df.to_string(
            index=False
        )
    )

    improvement_df.to_csv(
        FINAL_SUMMARY_PATH,
        index=False
    )

    print()
    print("Improvement summary saved:")
    print(FINAL_SUMMARY_PATH)

    return improvement_df


# ============================================================================
# RMSE / MAE PLOT
# ============================================================================

def generate_rmse_mae_plot(
    comparison_df
):

    print_section(
        "GENERATING RMSE / MAE PLOT"
    )

    models = comparison_df["model"].tolist()

    rmse_values = (
        comparison_df["rmse"]
        .astype(float)
        .tolist()
    )

    mae_values = (
        comparison_df["mae"]
        .astype(float)
        .tolist()
    )

    x = np.arange(
        len(models)
    )

    width = 0.35

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    ax.bar(
        x - width / 2,
        np.nan_to_num(
            rmse_values,
            nan=0.0
        ),
        width,
        label="RMSE"
    )

    ax.bar(
        x + width / 2,
        np.nan_to_num(
            mae_values,
            nan=0.0
        ),
        width,
        label="MAE"
    )

    ax.set_xlabel(
        "Model"
    )

    ax.set_ylabel(
        "Error"
    )

    ax.set_title(
        "Day 18 vs Day 19 - RMSE / MAE Comparison"
    )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        models,
        rotation=15,
        ha="right"
    )

    ax.legend()

    ax.grid(
        axis="y",
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        RMSE_MAE_PLOT_PATH,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    print("Plot saved:")
    print(RMSE_MAE_PLOT_PATH)


# ============================================================================
# RANKING METRICS PLOT
# ============================================================================

def generate_ranking_metrics_plot(
    comparison_df
):

    print_section(
        "GENERATING RANKING METRICS PLOT"
    )

    metrics = [
        "precision_at_10",
        "recall_at_10",
        "hit_rate_at_10",
        "ndcg_at_10",
        "map_at_10"
    ]

    display_names = [
        "Precision@10",
        "Recall@10",
        "Hit Rate@10",
        "NDCG@10",
        "MAP@10"
    ]

    x = np.arange(
        len(metrics)
    )

    width = 0.35

    fig, ax = plt.subplots(
        figsize=(12, 7)
    )

    day18_values = []

    day19_values = []

    for metric in metrics:

        day18_value = (
            comparison_df.loc[
                comparison_df["model"]
                == "Day 18 - Deep Hybrid",
                metric
            ]
        )

        day19_value = (
            comparison_df.loc[
                comparison_df["model"]
                == "Day 19 - Improved Evaluation",
                metric
            ]
        )

        if len(day18_value) > 0:
            value18 = float(
                day18_value.iloc[0]
            )
        else:
            value18 = np.nan

        if len(day19_value) > 0:
            value19 = float(
                day19_value.iloc[0]
            )
        else:
            value19 = np.nan

        day18_values.append(
            value18
        )

        day19_values.append(
            value19
        )

    ax.bar(
        x - width / 2,
        np.nan_to_num(
            day18_values,
            nan=0.0
        ),
        width,
        label="Day 18"
    )

    ax.bar(
        x + width / 2,
        np.nan_to_num(
            day19_values,
            nan=0.0
        ),
        width,
        label="Day 19"
    )

    ax.set_xlabel(
        "Ranking Metric"
    )

    ax.set_ylabel(
        "Score"
    )

    ax.set_title(
        "Day 18 vs Day 19 - Ranking Metrics"
    )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        display_names,
        rotation=15,
        ha="right"
    )

    ax.legend()

    ax.grid(
        axis="y",
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        RANKING_PLOT_PATH,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    print("Plot saved:")
    print(RANKING_PLOT_PATH)


# ============================================================================
# COVERAGE / DIVERSITY PLOT
# ============================================================================

def generate_coverage_diversity_plot(
    comparison_df
):

    print_section(
        "GENERATING COVERAGE / DIVERSITY PLOT"
    )

    metrics = [
        "catalog_coverage_at_10",
        "genre_diversity_at_10"
    ]

    display_names = [
        "Catalog Coverage@10",
        "Genre Diversity@10"
    ]

    x = np.arange(
        len(metrics)
    )

    width = 0.35

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    day18_values = []
    day19_values = []

    for metric in metrics:

        row18 = comparison_df[
            comparison_df["model"]
            == "Day 18 - Deep Hybrid"
        ]

        row19 = comparison_df[
            comparison_df["model"]
            == "Day 19 - Improved Evaluation"
        ]

        value18 = (
            float(
                row18.iloc[0][metric]
            )
            if not row18.empty
            and not pd.isna(
                row18.iloc[0][metric]
            )
            else 0.0
        )

        value19 = (
            float(
                row19.iloc[0][metric]
            )
            if not row19.empty
            and not pd.isna(
                row19.iloc[0][metric]
            )
            else 0.0
        )

        day18_values.append(
            value18
        )

        day19_values.append(
            value19
        )

    ax.bar(
        x - width / 2,
        day18_values,
        width,
        label="Day 18"
    )

    ax.bar(
        x + width / 2,
        day19_values,
        width,
        label="Day 19"
    )

    ax.set_xlabel(
        "Metric"
    )

    ax.set_ylabel(
        "Value"
    )

    ax.set_title(
        "Day 18 vs Day 19 - Coverage and Diversity"
    )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        display_names,
        rotation=15,
        ha="right"
    )

    ax.legend()

    ax.grid(
        axis="y",
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        COVERAGE_PLOT_PATH,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    print("Plot saved:")
    print(COVERAGE_PLOT_PATH)


# ============================================================================
# ATTENTION COMPARISON PLOT
# ============================================================================

def generate_attention_plot(
    comparison_df
):

    print_section(
        "GENERATING ATTENTION COMPARISON PLOT"
    )

    models = comparison_df["model"].tolist()

    content_values = (
        comparison_df[
            "average_content_attention"
        ]
        .fillna(0)
        .astype(float)
        .tolist()
    )

    collaborative_values = (
        comparison_df[
            "average_collaborative_attention"
        ]
        .fillna(0)
        .astype(float)
        .tolist()
    )

    x = np.arange(
        len(models)
    )

    width = 0.35

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    ax.bar(
        x - width / 2,
        content_values,
        width,
        label="Content Attention"
    )

    ax.bar(
        x + width / 2,
        collaborative_values,
        width,
        label="Collaborative Attention"
    )

    ax.set_xlabel(
        "Model"
    )

    ax.set_ylabel(
        "Attention Weight"
    )

    ax.set_title(
        "Day 18 vs Day 19 - Attention Comparison"
    )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        models,
        rotation=15,
        ha="right"
    )

    ax.legend()

    ax.grid(
        axis="y",
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        ATTENTION_PLOT_PATH,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    print("Plot saved:")
    print(ATTENTION_PLOT_PATH)


# ============================================================================
# USER-LEVEL ANALYSIS
# ============================================================================

def create_user_level_analysis(
    day19_user_evaluation_df
):

    print_section(
        "USER-LEVEL ANALYSIS"
    )

    # IMPORTANT:
    # Do NOT name this variable "day19_users" and later overwrite it.
    # We keep the original DataFrame intact throughout the function.

    user_df = (
        day19_user_evaluation_df
        .copy()
    )

    required_columns = [
        "user_id",
        "relevant_item_count",
        "recommendation_count",
        "precision_at_10",
        "recall_at_10",
        "hit_rate_at_10",
        "ndcg_at_10",
        "map_at_10"
    ]

    check_columns(
        user_df,
        required_columns,
        "Day 19 user evaluation"
    )

    print()
    print("Available Day 19 user evaluation columns:")
    print(
        list(
            user_df.columns
        )
    )

    # ------------------------------------------------------------------------
    # Numeric conversion
    # ------------------------------------------------------------------------

    numeric_columns = [
        "user_id",
        "relevant_item_count",
        "recommendation_count",
        "precision_at_10",
        "recall_at_10",
        "hit_rate_at_10",
        "ndcg_at_10",
        "map_at_10"
    ]

    for column in numeric_columns:
        user_df[column] = pd.to_numeric(
            user_df[column],
            errors="coerce"
        )

    # ------------------------------------------------------------------------
    # Sort users
    # ------------------------------------------------------------------------

    user_df = user_df.sort_values(
        "user_id"
    ).reset_index(
        drop=True
    )

    # ------------------------------------------------------------------------
    # Select output columns
    # ------------------------------------------------------------------------

    user_analysis_columns = [
        "user_id",
        "precision_at_10",
        "recall_at_10",
        "hit_rate_at_10",
        "ndcg_at_10",
        "map_at_10"
    ]

    user_analysis_df = (
        user_df[
            user_analysis_columns
        ]
        .copy()
    )

    # ------------------------------------------------------------------------
    # Round values
    # ------------------------------------------------------------------------

    metric_columns = [
        "precision_at_10",
        "recall_at_10",
        "hit_rate_at_10",
        "ndcg_at_10",
        "map_at_10"
    ]

    user_analysis_df[
        metric_columns
    ] = (
        user_analysis_df[
            metric_columns
        ]
        .round(4)
    )

    print()
    print("User-level evaluation:")

    print(
        user_analysis_df.head(10)
        .to_string(
            index=False
        )
    )

    # ------------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------------

    user_analysis_df.to_csv(
        USER_ANALYSIS_PATH,
        index=False
    )

    print()
    print("User analysis saved:")
    print(USER_ANALYSIS_PATH)

    # ------------------------------------------------------------------------
    # Average metrics
    # ------------------------------------------------------------------------

    user_average_metrics = (
        user_analysis_df[
            metric_columns
        ]
        .mean(
            numeric_only=True
        )
    )

    print()
    print("User-level average metrics:")

    for metric_name, value in (
        user_average_metrics.items()
    ):
        print(
            f"{metric_name}    "
            f"{value:.4f}"
        )

    return (
        user_analysis_df,
        user_average_metrics
    )


# ============================================================================
# RECOMMENDATION-LEVEL ANALYSIS
# ============================================================================

def create_recommendation_level_analysis(
    day19_recommendations_df
):

    print_section(
        "RECOMMENDATION-LEVEL ANALYSIS"
    )

    recommendation_df = (
        day19_recommendations_df
        .copy()
    )

    required_columns = [
        "user_id",
        "rank",
        "movie_id",
        "predicted_rating",
        "content_attention",
        "collaborative_attention"
    ]

    check_columns(
        recommendation_df,
        required_columns,
        "Day 19 recommendations"
    )

    print()
    print("Day 19 recommendation columns:")
    print(
        list(
            recommendation_df.columns
        )
    )

    # ------------------------------------------------------------------------
    # Convert numeric columns
    # ------------------------------------------------------------------------

    numeric_columns = [
        "user_id",
        "rank",
        "movie_id",
        "predicted_rating",
        "content_attention",
        "collaborative_attention"
    ]

    for column in numeric_columns:

        recommendation_df[column] = (
            pd.to_numeric(
                recommendation_df[column],
                errors="coerce"
            )
        )

    # ------------------------------------------------------------------------
    # Basic statistics
    # ------------------------------------------------------------------------

    total_rows = len(
        recommendation_df
    )

    unique_movies = (
        recommendation_df[
            "movie_id"
        ]
        .nunique()
    )

    average_predicted_rating = (
        recommendation_df[
            "predicted_rating"
        ]
        .mean()
    )

    average_content_attention = (
        recommendation_df[
            "content_attention"
        ]
        .mean()
    )

    average_collaborative_attention = (
        recommendation_df[
            "collaborative_attention"
        ]
        .mean()
    )

    print()
    print("Total recommendation rows:")
    print(total_rows)

    print()
    print("Unique recommended movies:")
    print(unique_movies)

    print()
    print("Average predicted rating:")
    print(
        f"{average_predicted_rating:.4f}"
    )

    print()
    print("Recommendation average content attention:")
    print(
        f"{average_content_attention:.4f}"
    )

    print()
    print("Recommendation average collaborative attention:")
    print(
        f"{average_collaborative_attention:.4f}"
    )

    # ------------------------------------------------------------------------
    # Recommendation statistics DataFrame
    # ------------------------------------------------------------------------

    recommendation_summary_df = pd.DataFrame(
        [
            {
                "total_recommendation_rows":
                    total_rows,

                "unique_recommended_movies":
                    unique_movies,

                "average_predicted_rating":
                    average_predicted_rating,

                "average_content_attention":
                    average_content_attention,

                "average_collaborative_attention":
                    average_collaborative_attention
            }
        ]
    )

    return (
        recommendation_df,
        recommendation_summary_df
    )


# ============================================================================
# CREATE FINAL CONSOLIDATED CSV
# ============================================================================

def create_final_comparison_csv(
    comparison_df,
    recommendation_summary_df
):

    print_section(
        "CREATING FINAL DAY 20 CSV"
    )

    final_df = comparison_df.copy()

    # ------------------------------------------------------------------------
    # Add experiment names
    # ------------------------------------------------------------------------

    final_df.insert(
        0,
        "experiment",
        [
            "Day 18",
            "Day 19"
        ]
    )

    # ------------------------------------------------------------------------
    # Add evaluation type
    # ------------------------------------------------------------------------

    final_df[
        "evaluation_type"
    ] = [
        "baseline_deep_hybrid",
        "improved_recommendation_evaluation"
    ]

    # ------------------------------------------------------------------------
    # Add recommendation-level statistics
    # ------------------------------------------------------------------------
    #
    # These values belong to Day 19, therefore Day 18 receives NaN.
    #

    total_rows = float(
        recommendation_summary_df.iloc[0][
            "total_recommendation_rows"
        ]
    )

    unique_movies = float(
        recommendation_summary_df.iloc[0][
            "unique_recommended_movies"
        ]
    )

    avg_predicted_rating = float(
        recommendation_summary_df.iloc[0][
            "average_predicted_rating"
        ]
    )

    final_df[
        "recommendation_rows"
    ] = [
        np.nan,
        total_rows
    ]

    final_df[
        "unique_recommended_movies"
    ] = [
        np.nan,
        unique_movies
    ]

    final_df[
        "average_predicted_rating"
    ] = [
        np.nan,
        avg_predicted_rating
    ]

    # ------------------------------------------------------------------------
    # Reorder columns
    # ------------------------------------------------------------------------

    preferred_columns = [
        "experiment",
        "model",
        "rmse",
        "mae",
        "precision_at_10",
        "recall_at_10",
        "hit_rate_at_10",
        "ndcg_at_10",
        "map_at_10",
        "catalog_coverage_at_10",
        "genre_diversity_at_10",
        "average_content_attention",
        "average_collaborative_attention",
        "evaluated_users",
        "recommendation_rows",
        "unique_recommended_movies",
        "average_predicted_rating",
        "evaluation_type"
    ]

    final_columns = [
        column
        for column in preferred_columns
        if column in final_df.columns
    ]

    final_df = (
        final_df[
            final_columns
        ]
    )

    # ------------------------------------------------------------------------
    # Round
    # ------------------------------------------------------------------------

    for column in final_df.columns:

        if column not in [
            "experiment",
            "model",
            "evaluation_type"
        ]:

            final_df[column] = (
                pd.to_numeric(
                    final_df[column],
                    errors="coerce"
                )
                .round(4)
            )

    # ------------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------------

    final_df.to_csv(
        FINAL_COMPARISON_PATH,
        index=False
    )

    print()
    print("Final comparison CSV saved:")
    print(FINAL_COMPARISON_PATH)

    return final_df


# ============================================================================
# FINAL SUMMARY
# ============================================================================

def print_final_summary(
    final_comparison_df,
    day18_metrics_df,
    day19_metrics_df
):

    print_section(
        "DAY 20 FINAL SUMMARY"
    )

    print()
    print("Model Comparison:")

    print(
        final_comparison_df.to_string(
            index=False
        )
    )

    # ------------------------------------------------------------------------
    # Day 18
    # ------------------------------------------------------------------------

    print()
    print("Key Day 18 metrics:")

    print(
        "RMSE:",
        round(
            numeric_value(
                day18_metrics_df,
                "rmse"
            ),
            4
        )
    )

    print(
        "MAE:",
        round(
            numeric_value(
                day18_metrics_df,
                "mae"
            ),
            4
        )
    )

    print(
        "Precision@10:",
        round(
            numeric_value(
                day18_metrics_df,
                "precision_at_10"
            ),
            4
        )
    )

    print(
        "Recall@10:",
        round(
            numeric_value(
                day18_metrics_df,
                "recall_at_10"
            ),
            4
        )
    )

    print(
        "Hit Rate@10:",
        round(
            numeric_value(
                day18_metrics_df,
                "hit_rate_at_10"
            ),
            4
        )
    )

    print(
        "NDCG@10:",
        round(
            numeric_value(
                day18_metrics_df,
                "ndcg_at_10"
            ),
            4
        )
    )

    print(
        "Catalog Coverage@10:",
        round(
            numeric_value(
                day18_metrics_df,
                "catalog_coverage_at_10"
            ),
            4
        )
    )

    print(
        "Genre Diversity@10:",
        round(
            numeric_value(
                day18_metrics_df,
                "genre_diversity_at_10"
            ),
            4
        )
    )

    # ------------------------------------------------------------------------
    # Day 19
    # ------------------------------------------------------------------------

    print()
    print("Key Day 19 metrics:")

    print(
        "Precision@10:",
        round(
            numeric_value(
                day19_metrics_df,
                "precision_at_10"
            ),
            4
        )
    )

    print(
        "Recall@10:",
        round(
            numeric_value(
                day19_metrics_df,
                "recall_at_10"
            ),
            4
        )
    )

    print(
        "Hit Rate@10:",
        round(
            numeric_value(
                day19_metrics_df,
                "hit_rate_at_10"
            ),
            4
        )
    )

    print(
        "NDCG@10:",
        round(
            numeric_value(
                day19_metrics_df,
                "ndcg_at_10"
            ),
            4
        )
    )

    print(
        "MAP@10:",
        round(
            numeric_value(
                day19_metrics_df,
                "map_at_10"
            ),
            4
        )
    )

    print(
        "Catalog Coverage@10:",
        round(
            numeric_value(
                day19_metrics_df,
                "catalog_coverage_at_10"
            ),
            4
        )
    )

    print(
        "Genre Diversity@10:",
        round(
            numeric_value(
                day19_metrics_df,
                "genre_diversity_at_10"
            ),
            4
        )
    )

    # ------------------------------------------------------------------------
    # Generated files
    # ------------------------------------------------------------------------

    print()
    print("Generated files:")

    output_files = [
        FINAL_COMPARISON_PATH,
        COMPARISON_TABLE_PATH,
        FINAL_SUMMARY_PATH,
        USER_ANALYSIS_PATH,
        RMSE_MAE_PLOT_PATH,
        RANKING_PLOT_PATH,
        COVERAGE_PLOT_PATH,
        ATTENTION_PLOT_PATH
    ]

    for file_path in output_files:

        if os.path.exists(file_path):

            print(
                "[OK]",
                file_path
            )

        else:

            print(
                "[MISSING]",
                file_path
            )

    # ------------------------------------------------------------------------
    # File sizes
    # ------------------------------------------------------------------------

    print()
    print("Output file sizes:")

    for file_path in output_files:

        if os.path.exists(file_path):

            size_kb = (
                os.path.getsize(
                    file_path
                )
                / 1024
            )

            print(
                f"{os.path.basename(file_path)}: "
                f"{size_kb:.2f} KB"
            )


# ============================================================================
# MAIN
# ============================================================================

def main():

    try:

        # --------------------------------------------------------------------
        # LOAD
        # --------------------------------------------------------------------

        (
            day18_metrics_df,
            day19_metrics_df,
            day19_user_evaluation_df,
            day19_recommendations_df
        ) = load_all_data()

        # --------------------------------------------------------------------
        # VALIDATE
        # --------------------------------------------------------------------

        validate_data(
            day18_metrics_df,
            day19_metrics_df,
            day19_user_evaluation_df,
            day19_recommendations_df
        )

        # --------------------------------------------------------------------
        # DISPLAY
        # --------------------------------------------------------------------

        display_loaded_metrics(
            day18_metrics_df,
            day19_metrics_df
        )

        # --------------------------------------------------------------------
        # MODEL COMPARISON
        # --------------------------------------------------------------------

        comparison_df = (
            create_model_comparison_table(
                day18_metrics_df,
                day19_metrics_df
            )
        )

        # --------------------------------------------------------------------
        # IMPROVEMENT ANALYSIS
        # --------------------------------------------------------------------

        improvement_df = (
            create_improvement_analysis(
                day18_metrics_df,
                day19_metrics_df
            )
        )

        # Prevent unused-variable warning / make intent explicit.
        _ = improvement_df

        # --------------------------------------------------------------------
        # PLOTS
        # --------------------------------------------------------------------

        generate_rmse_mae_plot(
            comparison_df
        )

        generate_ranking_metrics_plot(
            comparison_df
        )

        generate_coverage_diversity_plot(
            comparison_df
        )

        generate_attention_plot(
            comparison_df
        )

        # --------------------------------------------------------------------
        # USER LEVEL
        # --------------------------------------------------------------------

        (
            user_analysis_df,
            user_average_metrics
        ) = create_user_level_analysis(
            day19_user_evaluation_df
        )

        _ = user_analysis_df
        _ = user_average_metrics

        # --------------------------------------------------------------------
        # RECOMMENDATION LEVEL
        # --------------------------------------------------------------------

        (
            recommendation_df,
            recommendation_summary_df
        ) = create_recommendation_level_analysis(
            day19_recommendations_df
        )

        _ = recommendation_df

        # --------------------------------------------------------------------
        # FINAL CSV
        # --------------------------------------------------------------------

        final_comparison_df = (
            create_final_comparison_csv(
                comparison_df,
                recommendation_summary_df
            )
        )

        # --------------------------------------------------------------------
        # FINAL SUMMARY
        # --------------------------------------------------------------------

        print_final_summary(
            final_comparison_df,
            day18_metrics_df,
            day19_metrics_df
        )

        # --------------------------------------------------------------------
        # SUCCESS
        # --------------------------------------------------------------------

        print()
        print_separator()
        print("DAY 20 COMPLETED SUCCESSFULLY")
        print_separator()

    except FileNotFoundError as error:

        print()
        print_separator()
        print("DAY 20 FAILED - FILE NOT FOUND")
        print_separator()
        print(str(error))

        raise

    except ValueError as error:

        print()
        print_separator()
        print("DAY 20 FAILED - DATA VALIDATION ERROR")
        print_separator()
        print(str(error))

        raise

    except Exception as error:

        print()
        print_separator()
        print("DAY 20 FAILED")
        print_separator()
        print(
            f"{type(error).__name__}: {error}"
        )

        raise


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()