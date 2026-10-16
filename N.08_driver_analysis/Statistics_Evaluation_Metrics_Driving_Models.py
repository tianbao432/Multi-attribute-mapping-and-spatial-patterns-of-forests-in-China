#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Collect model evaluation metrics for all forest attributes into one table.

The script reads:
    <RESULT_ROOT>/<target_folder>/spatial_cv_metrics_overall.csv

It exports:
    <RESULT_ROOT>/model_evaluation_summary.csv
"""

from pathlib import Path
import numpy as np
import pandas as pd


# =============================================================================
# Configuration
# =============================================================================
RESULT_ROOT = Path("/data/China_mapping/driver/result/")

OUTPUT_CSV = RESULT_ROOT / "model_evaluation_summary.csv"

TARGET_CONFIGS = [
    {
        "folder": "Density",
        "attribute": "Density",
        "abbreviation": "DEN",
    },
    {
        "folder": "DBH",
        "attribute": "Mean diameter at breast height",
        "abbreviation": "DBH",
    },
    {
        "folder": "BA",
        "attribute": "Basal area",
        "abbreviation": "BA",
    },
    {
        "folder": "GSV",
        "attribute": "Growing stock volume",
        "abbreviation": "GSV",
    },
    {
        "folder": "AGB",
        "attribute": "Aboveground biomass",
        "abbreviation": "AGB",
    },
    {
        "folder": "Shannon",
        "attribute": "Shannon diversity index",
        "abbreviation": "SHA",
    },
    {
        "folder": "CV_DBH",
        "alternative_folders": [
            "CV DBH",
            "CV-DBH",
        ],
        "attribute": "Coefficient of variation in DBH",
        "abbreviation": "CV-DBH",
    },
    {
        "folder": "Age",
        "attribute": "Mean forest age",
        "abbreviation": "AGE",
    },
    {
        "folder": "Height",
        "attribute": "Mean tree height",
        "abbreviation": "H",
    },
]

METRICS_FILENAME = "spatial_cv_metrics_overall.csv"


# =============================================================================
# Utilities
# =============================================================================
def resolve_target_directory(config):
    candidate_folders = [config["folder"]]
    candidate_folders.extend(config.get("alternative_folders", []))

    for folder in candidate_folders:
        target_dir = RESULT_ROOT / folder
        metrics_path = target_dir / METRICS_FILENAME

        if metrics_path.exists():
            return target_dir, metrics_path

    return None, None


def safely_get_numeric(row, possible_names):
    for name in possible_names:
        if name in row.index:
            value = pd.to_numeric(
                pd.Series([row[name]]),
                errors="coerce",
            ).iloc[0]

            if pd.notna(value):
                return float(value)

    return np.nan


def safely_get_value(row, possible_names, default=np.nan):
    for name in possible_names:
        if name in row.index:
            return row[name]

    return default


# =============================================================================
# Metrics collection
# =============================================================================
def collect_model_metrics():
    summary_records = []
    missing_records = []

    for order, config in enumerate(TARGET_CONFIGS, start=1):
        target_dir, metrics_path = resolve_target_directory(config)

        if metrics_path is None:
            missing_records.append({
                "Abbreviation": config["abbreviation"],
                "Forest_Attribute": config["attribute"],
                "Expected_Folder": config["folder"],
                "Expected_File": METRICS_FILENAME,
            })

            summary_records.append({
                "Order": order,
                "Forest_Attribute": config["attribute"],
                "Abbreviation": config["abbreviation"],
                "N": np.nan,
                "R2": np.nan,
                "RMSE": np.nan,
                "Tuning_Informed_Evaluation": np.nan,
                "Status": "Missing",
            })

            print(
                f"Warning: metrics file not found for "
                f"{config['abbreviation']}"
            )
            continue

        metrics_df = pd.read_csv(metrics_path)

        if metrics_df.empty:
            summary_records.append({
                "Order": order,
                "Forest_Attribute": config["attribute"],
                "Abbreviation": config["abbreviation"],
                "N": np.nan,
                "R2": np.nan,
                "RMSE": np.nan,
                "Tuning_Informed_Evaluation": np.nan,
                "Status": "Empty file",
            })

            print(f"Warning: empty metrics file -> {metrics_path}")
            continue

        row = metrics_df.iloc[0]

        n_value = safely_get_numeric(
            row,
            ["N", "Sample_N", "Valid_N"],
        )

        r2_value = safely_get_numeric(
            row,
            ["R2", "R2", "R_squared"],
        )

        rmse_value = safely_get_numeric(
            row,
            ["RMSE", "Root_Mean_Squared_Error"],
        )

        tuning_informed = safely_get_value(
            row,
            ["Tuning_Informed_Evaluation"],
            default=np.nan,
        )

        summary_records.append({
            "Order": order,
            "Forest_Attribute": config["attribute"],
            "Abbreviation": config["abbreviation"],
            "N": (
                int(n_value)
                if np.isfinite(n_value)
                else np.nan
            ),
            "R2": r2_value,
            "RMSE": rmse_value,
            "Tuning_Informed_Evaluation": tuning_informed,
            "Status": "Success",
        })

        print(
            f"Loaded {config['abbreviation']}: "
            f"N={n_value:.0f}, "
            f"R2={r2_value:.4f}, "
            f"RMSE={rmse_value:.4f}"
        )

    summary_df = pd.DataFrame(summary_records)

    summary_df = summary_df.sort_values(
        "Order",
        ascending=True,
    ).reset_index(drop=True)

    return summary_df, pd.DataFrame(missing_records)


# =============================================================================
# Main
# =============================================================================
def main():
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)

    summary_df, missing_df = collect_model_metrics()

    # Preserve full numerical precision in the exported analysis table.
    summary_df.to_csv(
        OUTPUT_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    # Export a publication-friendly rounded version.
    publication_df = summary_df[
        [
            "Forest_Attribute",
            "Abbreviation",
            "N",
            "R2",
            "RMSE",
        ]
    ].copy()

    publication_df["R2"] = publication_df["R2"].round(3)
    publication_df["RMSE"] = publication_df["RMSE"].round(3)

    publication_csv = (
        RESULT_ROOT
        / "model_evaluation_summary_publication.csv"
    )

    publication_df.to_csv(
        publication_csv,
        index=False,
        encoding="utf-8-sig",
    )

    if not missing_df.empty:
        missing_csv = (
            RESULT_ROOT
            / "missing_model_evaluation_files.csv"
        )

        missing_df.to_csv(
            missing_csv,
            index=False,
            encoding="utf-8-sig",
        )

        print(f"Missing-file report saved: {missing_csv}")

    print("\nModel evaluation summary:")
    print(publication_df.to_string(index=False))

    print("\nFiles saved successfully:")
    print(OUTPUT_CSV)
    print(publication_csv)


if __name__ == "__main__":
    main()