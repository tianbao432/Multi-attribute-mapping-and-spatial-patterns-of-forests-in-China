#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Part 1: Extract raster values and assign 10-fold K-means spatial CV IDs.

Outputs
-------
A single CSV containing:
- Renamed forest attribute columns
- Fifteen extracted driver variables
- A 10-fold K-means spatial fold ID
- No longitude or latitude columns

The spatial folds are created from projected plot coordinates before the
coordinate columns are removed from the exported CSV.
"""

from __future__ import annotations

import json
import logging
import sys
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import rasterio
import sklearn

from pyproj import CRS, Transformer
from sklearn.cluster import KMeans

warnings.filterwarnings("ignore")


# =============================================================================
# Configuration
# =============================================================================
@dataclass
class Config:
    input_csv: str = "/data/China_mapping/foerst_plot/PLOT_Data_Climate.csv"
    raster_dir: str = "/data/China_mapping/driving_variables/"
    output_csv: str = "/data/China_mapping/driver/plot_driver_kmeans10.csv"
    output_log: str = "/data/China_mapping/driver/part1_extraction.log"
    output_summary: str = "/data/China_mapping/driver/raster_extraction_summary.csv"
    output_config: str = "/data/China_mapping/driver/part1_configuration.json"

    lon_col: str = "Longitude"
    lat_col: str = "Latitude"
    fold_col: str = "KMeans_Fold"

    n_folds: int = 10
    random_seed: int = 42

    drivers: List[str] = field(default_factory=lambda: [
        "HMI",
        "MAT",
        "ISO",
        "MTCM",
        "MTDQ",
        "MAP",
        "PDM",
        "PS",
        "ELEV",
        "TWI",
        "VPD",
        "SWVL3",
        "TN",
        "PH",
        "CLAY",
    ])

    rename_mapping: Dict[str, str] = field(default_factory=lambda: {
        "V": "GSV",
        "N": "Density",
        "Shannon.Index": "Shannon",
        "H": "Height",
        "AGE": "Age",
        "CV_DBH": "CV DBH",
    })

    projected_crs: str = (
        "+proj=aea +lat_1=25 +lat_2=47 +lat_0=0 "
        "+lon_0=105 +datum=WGS84 +units=m +no_defs"
    )


CONFIG = Config()


# =============================================================================
# Logging and utilities
# =============================================================================
def ensure_parent(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def setup_logger(path: str) -> logging.Logger:
    ensure_parent(path)

    logger = logging.getLogger("part1_extraction")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(path, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def save_configuration() -> None:
    ensure_parent(CONFIG.output_config)
    with open(CONFIG.output_config, "w", encoding="utf-8") as file:
        json.dump(asdict(CONFIG), file, indent=2, ensure_ascii=False)


# =============================================================================
# Coordinate handling
# =============================================================================
def validate_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    result[CONFIG.lon_col] = pd.to_numeric(
        result[CONFIG.lon_col],
        errors="coerce",
    )
    result[CONFIG.lat_col] = pd.to_numeric(
        result[CONFIG.lat_col],
        errors="coerce",
    )

    valid_mask = (
        result[CONFIG.lon_col].between(-180.0, 180.0)
        & result[CONFIG.lat_col].between(-90.0, 90.0)
    )

    return result.loc[valid_mask].reset_index(drop=True)


def project_coordinates(df: pd.DataFrame) -> np.ndarray:
    source_crs = CRS.from_epsg(4326)
    target_crs = CRS.from_proj4(CONFIG.projected_crs)

    transformer = Transformer.from_crs(
        source_crs,
        target_crs,
        always_xy=True,
    )

    x_coord, y_coord = transformer.transform(
        df[CONFIG.lon_col].to_numpy(dtype=np.float64),
        df[CONFIG.lat_col].to_numpy(dtype=np.float64),
    )

    coordinates = np.column_stack([x_coord, y_coord]).astype(np.float64)

    if not np.isfinite(coordinates).all():
        raise ValueError("Projected coordinates contain non-finite values.")

    return coordinates


def assign_kmeans_folds(coordinates_m: np.ndarray) -> np.ndarray:
    if coordinates_m.shape[0] < CONFIG.n_folds:
        raise ValueError(
            f"Only {coordinates_m.shape[0]} valid plots are available, "
            f"which is fewer than {CONFIG.n_folds} folds."
        )

    coordinate_std = np.std(coordinates_m, axis=0, ddof=0)
    coordinate_std = np.where(coordinate_std > 0.0, coordinate_std, 1.0)

    standardized = (
        coordinates_m - np.mean(coordinates_m, axis=0, keepdims=True)
    ) / coordinate_std

    model = KMeans(
        n_clusters=CONFIG.n_folds,
        random_state=CONFIG.random_seed,
        n_init=100,
        algorithm="lloyd",
    )

    labels = model.fit_predict(standardized)

    # Export fold IDs as 1-10 rather than 0-9.
    return labels.astype(int) + 1


# =============================================================================
# Raster extraction
# =============================================================================
def raster_path(driver: str) -> Path:
    return Path(CONFIG.raster_dir) / f"{driver}.tif"


def extract_single_raster(
    path: Path,
    longitude: np.ndarray,
    latitude: np.ndarray,
) -> np.ndarray:
    values = np.full(longitude.shape[0], np.nan, dtype=np.float64)
    geographic_crs = CRS.from_epsg(4326)

    with rasterio.open(path) as src:
        if src.crs is None:
            raise ValueError(f"Raster CRS is missing: {path}")

        raster_crs = CRS.from_user_input(src.crs)
        transformer = Transformer.from_crs(
            geographic_crs,
            raster_crs,
            always_xy=True,
        )

        x_coord, y_coord = transformer.transform(
            longitude,
            latitude,
        )

        inside = (
            np.isfinite(x_coord)
            & np.isfinite(y_coord)
            & (x_coord >= src.bounds.left)
            & (x_coord <= src.bounds.right)
            & (y_coord >= src.bounds.bottom)
            & (y_coord <= src.bounds.top)
        )

        positions = np.where(inside)[0]
        if positions.size == 0:
            return values

        sample_coordinates = [
            (float(x_coord[index]), float(y_coord[index]))
            for index in positions
        ]

        sampled_values = []

        for sample in src.sample(sample_coordinates, masked=True):
            if np.ma.isMaskedArray(sample) and np.any(sample.mask):
                sampled_values.append(np.nan)
            elif sample.size == 0:
                sampled_values.append(np.nan)
            else:
                sampled_values.append(float(sample[0]))

        sampled = np.asarray(sampled_values, dtype=np.float64)

        invalid = ~np.isfinite(sampled)

        if src.nodata is not None and np.isfinite(src.nodata):
            invalid |= np.isclose(sampled, src.nodata)

        sampled[invalid] = np.nan
        values[positions] = sampled

    return values


def extract_all_rasters(
    df: pd.DataFrame,
    logger: logging.Logger,
) -> pd.DataFrame:
    result = df.copy()

    longitude = result[CONFIG.lon_col].to_numpy(dtype=np.float64)
    latitude = result[CONFIG.lat_col].to_numpy(dtype=np.float64)

    summary_records = []

    for driver in CONFIG.drivers:
        path = raster_path(driver)

        if not path.exists():
            logger.warning("Raster not found for %s: %s", driver, path)
            result[driver] = np.nan

            summary_records.append({
                "Driver": driver,
                "Raster_Path": str(path),
                "Status": "missing_file",
                "Valid_N": 0,
                "Missing_N": len(result),
                "Minimum": np.nan,
                "Maximum": np.nan,
            })
            continue

        try:
            values = extract_single_raster(
                path=path,
                longitude=longitude,
                latitude=latitude,
            )

            result[driver] = values

            summary_records.append({
                "Driver": driver,
                "Raster_Path": str(path),
                "Status": "success",
                "Valid_N": int(np.isfinite(values).sum()),
                "Missing_N": int((~np.isfinite(values)).sum()),
                "Minimum": (
                    float(np.nanmin(values))
                    if np.isfinite(values).any()
                    else np.nan
                ),
                "Maximum": (
                    float(np.nanmax(values))
                    if np.isfinite(values).any()
                    else np.nan
                ),
            })

            logger.info(
                "%s extracted: valid=%d, missing=%d",
                driver,
                np.isfinite(values).sum(),
                (~np.isfinite(values)).sum(),
            )

        except Exception as exc:
            logger.exception("Extraction failed for %s: %s", driver, exc)
            result[driver] = np.nan

            summary_records.append({
                "Driver": driver,
                "Raster_Path": str(path),
                "Status": f"failed: {exc}",
                "Valid_N": 0,
                "Missing_N": len(result),
                "Minimum": np.nan,
                "Maximum": np.nan,
            })

    ensure_parent(CONFIG.output_summary)
    pd.DataFrame(summary_records).to_csv(
        CONFIG.output_summary,
        index=False,
    )

    return result


# =============================================================================
# Column preparation
# =============================================================================
def rename_columns_safely(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    duplicate_targets = [
        new_name
        for old_name, new_name in CONFIG.rename_mapping.items()
        if old_name in result.columns
        and new_name in result.columns
        and old_name != new_name
    ]

    if duplicate_targets:
        raise ValueError(
            "Renaming would create duplicate columns: "
            f"{duplicate_targets}"
        )

    available_mapping = {
        old_name: new_name
        for old_name, new_name in CONFIG.rename_mapping.items()
        if old_name in result.columns
    }

    return result.rename(columns=available_mapping)


def reorder_output_columns(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    preferred_targets = [
        "Density",
        "DBH",
        "BA",
        "GSV",
        "AGB",
        "Shannon",
        "CV DBH",
        "Age",
        "Height",
    ]

    target_columns = [
        column for column in preferred_targets
        if column in result.columns
    ]

    driver_columns = [
        column for column in CONFIG.drivers
        if column in result.columns
    ]

    excluded = {
        CONFIG.lon_col,
        CONFIG.lat_col,
        CONFIG.fold_col,
        *target_columns,
        *driver_columns,
    }

    other_columns = [
        column for column in result.columns
        if column not in excluded
    ]

    ordered_columns = (
        other_columns
        + target_columns
        + driver_columns
        + [CONFIG.fold_col]
    )

    return result.loc[:, ordered_columns]


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    logger = setup_logger(CONFIG.output_log)
    save_configuration()

    logger.info("Python: %s", sys.version.split()[0])
    logger.info("NumPy: %s", np.__version__)
    logger.info("pandas: %s", pd.__version__)
    logger.info("scikit-learn: %s", sklearn.__version__)

    input_path = Path(CONFIG.input_csv)

    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    data = pd.read_csv(input_path)

    missing_coordinate_columns = [
        column
        for column in [CONFIG.lon_col, CONFIG.lat_col]
        if column not in data.columns
    ]

    if missing_coordinate_columns:
        raise KeyError(
            f"Missing coordinate columns: {missing_coordinate_columns}"
        )

    original_n = len(data)
    data = validate_coordinates(data)

    logger.info(
        "Valid coordinate rows: %d of %d",
        len(data),
        original_n,
    )

    coordinates_m = project_coordinates(data)

    data[CONFIG.fold_col] = assign_kmeans_folds(
        coordinates_m=coordinates_m,
    )

    fold_summary = (
        data[CONFIG.fold_col]
        .value_counts()
        .sort_index()
        .rename_axis(CONFIG.fold_col)
        .reset_index(name="N")
    )

    fold_summary_path = (
        Path(CONFIG.output_csv).parent
        / "kmeans_fold_summary.csv"
    )
    fold_summary.to_csv(fold_summary_path, index=False)

    data = extract_all_rasters(
        df=data,
        logger=logger,
    )

    data = rename_columns_safely(data)

    # Coordinates are removed only after the K-means fold IDs are assigned
    # and all raster values are extracted.
    data = data.drop(
        columns=[CONFIG.lon_col, CONFIG.lat_col],
        errors="ignore",
    )

    data = reorder_output_columns(data)

    ensure_parent(CONFIG.output_csv)
    data.to_csv(CONFIG.output_csv, index=False)

    logger.info("Output CSV saved: %s", CONFIG.output_csv)
    logger.info("Rows: %d", len(data))
    logger.info("Columns: %d", data.shape[1])
    logger.info("Fold IDs: %s", sorted(data[CONFIG.fold_col].unique()))


if __name__ == "__main__":
    main()
