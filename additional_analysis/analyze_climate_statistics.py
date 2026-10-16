# -*- coding: utf-8 -*-

import os

# Fix for pyproj CRSError (Internal Proj Error)
PROJ_DATA_DIR = "/home/ubuntu/miniconda3/envs/china/share/proj"
os.environ["PROJ_LIB"] = PROJ_DATA_DIR

import pyproj

pyproj.datadir.set_data_dir(PROJ_DATA_DIR)

import re
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import ExitStack

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from rasterio.windows import Window
from rasterio.windows import bounds as window_bounds


# ==========================================
# 1. Configuration
# ==========================================
RASTER_PATHS = [
    "/data/China_mapping/production/fina_production/China_National_V_MEAN.tif",
    "/data/China_mapping/production/fina_production/China_National_Trees_N_MEAN.tif",
    "/data/China_mapping/production/fina_production/China_National_shannon_MEAN.tif",
    "/data/China_mapping/production/fina_production/China_National_H_MEAN.tif",
    "/data/China_mapping/production/fina_production/China_National_DBH_MEAN.tif",
    "/data/China_mapping/production/fina_production/China_National_CV_DBH_MEAN.tif",
    "/data/China_mapping/production/fina_production/China_National_BA_MEAN.tif",
    "/data/China_mapping/production/fina_production/China_National_AGE_MEAN.tif",
    "/data/China_mapping/production/fina_production/China_National_AGB_MEAN.tif",
]

VAR_NAMES = [
    "V",
    "Trees_N",
    "shannon",
    "H",
    "DBH",
    "CV_DBH",
    "BA",
    "AGE",
    "AGB",
]

SHP_PATH = "/data/China_mapping/climate_zone/clmate_zone.shp"
ZONE_FIELD = "clim_abbr"
OUT_DIR = "/data/China_mapping/Climate_statistics/"

CHUNK_SIZE = 4096
SUBSAMPLE_RATE = 0.2

os.makedirs(OUT_DIR, exist_ok=True)


def clean_name(name):
    return re.sub(r"[^a-zA-Z0-9_]", "_", str(name))


# Load vector data globally so forked worker processes can access it in memory
print("Loading climate zone shapefile into shared memory...")
GDF = gpd.read_file(SHP_PATH)

# Get reference CRS and dimensions from the first raster
with rasterio.open(RASTER_PATHS[0]) as tmp_src:
    REF_CRS = tmp_src.crs
    REF_TRANSFORM = tmp_src.transform
    WIDTH = tmp_src.width
    HEIGHT = tmp_src.height

if GDF.crs != REF_CRS:
    print("Reprojecting shapefile...")
    GDF = GDF.to_crs(REF_CRS)

UNIQUE_ZONES = GDF[ZONE_FIELD].dropna().unique()
ZONE_TO_ID = {
    zone: index + 1
    for index, zone in enumerate(UNIQUE_ZONES)
}
ID_TO_ZONE = {
    index + 1: zone
    for index, zone in enumerate(UNIQUE_ZONES)
}

GDF["zone_id"] = GDF[ZONE_FIELD].map(ZONE_TO_ID)
SINDEX = GDF.sindex


# ==========================================
# 2. Worker Function
# ==========================================
def process_window(window):
    """Extract sampled raster values for one raster window."""
    try:
        with ExitStack() as stack:
            srcs = [
                stack.enter_context(rasterio.open(path))
                for path in RASTER_PATHS
            ]

            nodata_vals = [src.nodata for src in srcs]
            arrays = [
                src.read(1, window=window)
                for src in srcs
            ]

            # Create a common valid-data mask
            valid = np.ones(arrays[0].shape, dtype=bool)

            for array, nodata_value in zip(arrays, nodata_vals):
                if nodata_value is not None:
                    if np.isnan(nodata_value):
                        valid &= ~np.isnan(array)
                    else:
                        valid &= array != nodata_value

                if array.dtype.kind in "fc":
                    valid &= np.isfinite(array)

            if not valid.any():
                return {}

            valid_data = np.column_stack(
                [array[valid] for array in arrays]
            )

            # Identify climate zones intersecting this raster window
            win_transform = rasterio.windows.transform(
                window,
                REF_TRANSFORM,
            )
            bounds = window_bounds(
                window,
                REF_TRANSFORM,
            )
            intersect_idx = list(
                SINDEX.intersection(bounds)
            )

            if intersect_idx:
                intersect_gdf = GDF.iloc[intersect_idx]

                if len(intersect_gdf) > 0:
                    shapes = (
                        (geometry, zone_id)
                        for geometry, zone_id in zip(
                            intersect_gdf.geometry,
                            intersect_gdf["zone_id"],
                        )
                        if geometry is not None
                        and not geometry.is_empty
                    )

                    zone_arr = rasterize(
                        shapes=shapes,
                        out_shape=arrays[0].shape,
                        transform=win_transform,
                        fill=0,
                        dtype=np.uint16,
                    )
                    valid_zones = zone_arr[valid]
                else:
                    valid_zones = np.zeros(
                        valid_data.shape[0],
                        dtype=np.uint16,
                    )
            else:
                valid_zones = np.zeros(
                    valid_data.shape[0],
                    dtype=np.uint16,
                )

            # Randomly subsample valid pixels
            if SUBSAMPLE_RATE < 1.0:
                n_samples = int(
                    valid_data.shape[0] * SUBSAMPLE_RATE
                )

                if n_samples <= 0:
                    return {}

                sample_indices = np.random.choice(
                    valid_data.shape[0],
                    size=n_samples,
                    replace=False,
                )

                valid_data = valid_data[sample_indices]
                valid_zones = valid_zones[sample_indices]

            # Organize national and climate-zone results
            result = {"National": valid_data}

            for zone_id in np.unique(valid_zones):
                if zone_id == 0:
                    continue

                zone_name = clean_name(
                    ID_TO_ZONE[int(zone_id)]
                )
                zone_data = valid_data[
                    valid_zones == zone_id
                ]

                if zone_data.size > 0:
                    result[zone_name] = zone_data

            return result

    except Exception as error:
        print(
            f"Window processing failed for {window}: {error}",
            flush=True,
        )
        return {}


# ==========================================
# 3. Main Execution Block
# ==========================================
def main():
    windows = []

    for row_off in range(0, HEIGHT, CHUNK_SIZE):
        for col_off in range(0, WIDTH, CHUNK_SIZE):
            width = min(
                CHUNK_SIZE,
                WIDTH - col_off,
            )
            height = min(
                CHUNK_SIZE,
                HEIGHT - row_off,
            )

            windows.append(
                Window(
                    col_off,
                    row_off,
                    width,
                    height,
                )
            )

    total_windows = len(windows)

    # Use 80% of available CPU cores
    n_cores = max(
        1,
        int(multiprocessing.cpu_count() * 0.8),
    )

    print(
        f"Starting multiprocessing extraction using "
        f"{n_cores} CPU cores..."
    )

    master_results = {"National": []}

    for zone in UNIQUE_ZONES:
        master_results[clean_name(zone)] = []

    completed = 0

    with ProcessPoolExecutor(
        max_workers=n_cores
    ) as executor:
        futures = {
            executor.submit(
                process_window,
                window,
            ): window
            for window in windows
        }

        for future in as_completed(futures):
            window = futures[future]

            try:
                result = future.result()
            except Exception as error:
                print(
                    f"Worker failed for {window}: {error}",
                    flush=True,
                )
                result = {}

            for key, data_array in result.items():
                master_results[key].append(
                    data_array
                )

            completed += 1

            if (
                completed % 20 == 0
                or completed == total_windows
            ):
                print(
                    f"Progress: {completed}/"
                    f"{total_windows} windows processed..."
                )

    # ==========================================
    # 4. Aggregation and Statistics
    # ==========================================
    print(
        "\nExtraction complete. "
        "Merging data in memory and writing to disk..."
    )

    for key, array_list in master_results.items():
        if len(array_list) == 0:
            continue

        print(f"Calculating stats for: {key}")

        merged_array = np.vstack(array_list)
        dataframe = pd.DataFrame(
            merged_array,
            columns=VAR_NAMES,
        )

        # Save raw sampled data for plotting
        raw_csv_path = os.path.join(
            OUT_DIR,
            f"{key}_raw_data.csv",
        )
        dataframe.to_csv(
            raw_csv_path,
            index=False,
        )

        # Save descriptive statistics
        stats = dataframe.describe()
        stats_path = os.path.join(
            OUT_DIR,
            f"{key}_boxplot_stats.csv",
        )
        stats.to_csv(stats_path)

        # Save correlation matrix
        correlation = dataframe.corr()
        correlation_path = os.path.join(
            OUT_DIR,
            f"{key}_correlation.csv",
        )
        correlation.to_csv(
            correlation_path
        )

    print(
        f"\nAll operations finished. "
        f"Files are ready in {OUT_DIR}"
    )


if __name__ == "__main__":
    main()
