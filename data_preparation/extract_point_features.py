import os
import glob
import gc
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import rasterio
from rasterio.crs import CRS
from rasterio.warp import transform as rio_transform

# Suppress invalid value warnings inside numpy computations
warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("GDAL_NUM_THREADS", "1")

# ==========================================
# 1. Configuration & Paths
# ==========================================
CSV_PATH = "/data/China_mapping/foerst_plot/PLOT_Data_Climate.csv"
OUT_DIR = "/data/China_mapping/foerst_plot"
OUT_CSV_PATH = os.path.join(OUT_DIR, "PLOT_Data_Climate_Extracted_Fast.csv")

VAR_DIR = "/data/China_mapping/variable"

DEM_FILES = [
    "/data/China_mapping/Tope/aspect.tif",
    "/data/China_mapping/Tope/DEM.tif",
    "/data/China_mapping/Tope/slop.tif"
]

VAR_FILES = glob.glob(os.path.join(VAR_DIR, "*.tif"))
ALL_RASTERS = DEM_FILES + VAR_FILES

cpu_count = os.cpu_count() or 4
MAX_WORKERS = max(1, min(8, cpu_count - 2))

WGS84 = CRS.from_epsg(4326)

# ==========================================
# 2. Worker Global Initialization
# ==========================================
GLOBAL_XS = None
GLOBAL_YS = None


def init_worker(xs, ys):
    """
    Initialize global coordinate arrays inside worker processes.
    """
    global GLOBAL_XS, GLOBAL_YS
    GLOBAL_XS = xs
    GLOBAL_YS = ys


# ==========================================
# 3. Helper Functions
# ==========================================
def is_wgs84_like(crs):

    if crs is None:
        return False

    try:
        epsg = crs.to_epsg(confidence_threshold=25)
        if epsg == 4326:
            return True

        if crs == WGS84:
            return True

        wkt = crs.to_wkt().upper()
        if crs.is_geographic and ("WGS" in wkt) and ("84" in wkt):
            return True

    except Exception:
        return False

    return False


def get_unique_raster_tasks(raster_paths):

    tasks = []
    name_counter = {}

    for path in raster_paths:
        if not os.path.exists(path):
            continue

        base_name = os.path.splitext(os.path.basename(path))[0]

        if base_name not in name_counter:
            name_counter[base_name] = 1
            col_name = base_name
        else:
            name_counter[base_name] += 1
            col_name = f"{base_name}_{name_counter[base_name]}"

        tasks.append((path, col_name))

    return tasks


def transform_points_if_needed(xs, ys, src_crs, col_name, worker_pid):

    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)

    if src_crs is None:
        print(
            f"[Worker PID: {worker_pid}] WARNING: {col_name} has no CRS. "
            f"Assuming coordinates are WGS84 lon/lat.",
            flush=True
        )
        return xs, ys

    if is_wgs84_like(src_crs):
        return xs, ys

    print(
        f"[Worker PID: {worker_pid}] INFO: Transforming points from EPSG:4326 "
        f"to raster CRS for {col_name}: {src_crs}",
        flush=True
    )

    tx = np.full_like(xs, np.nan, dtype=np.float64)
    ty = np.full_like(ys, np.nan, dtype=np.float64)

    valid_xy = np.isfinite(xs) & np.isfinite(ys)

    if np.any(valid_xy):
        x_list, y_list = rio_transform(
            WGS84,
            src_crs,
            xs[valid_xy].tolist(),
            ys[valid_xy].tolist()
        )
        tx[valid_xy] = np.asarray(x_list, dtype=np.float64)
        ty[valid_xy] = np.asarray(y_list, dtype=np.float64)

    return tx, ty


# ==========================================
# 4. Worker Function
# ==========================================
def process_single_raster(task):
    raster_path, col_name = task

    xs = GLOBAL_XS
    ys = GLOBAL_YS

    worker_pid = os.getpid()

    print(
        f"[Worker PID: {worker_pid}] STARTED extracting raster: {col_name}",
        flush=True
    )

    try:
        with rasterio.Env(GDAL_CACHEMAX=256, CPL_DEBUG="OFF"):
            with rasterio.open(raster_path) as src:

                if src.count < 1:
                    raise ValueError(f"{col_name} has no raster band.")

                sample_xs, sample_ys = transform_points_if_needed(
                    xs, ys, src.crs, col_name, worker_pid
                )

                extracted_values = np.full(len(xs), np.nan, dtype=np.float32)

                bounds = src.bounds

                valid_mask = (
                    np.isfinite(sample_xs) &
                    np.isfinite(sample_ys) &
                    (sample_xs >= bounds.left) &
                    (sample_xs <= bounds.right) &
                    (sample_ys >= bounds.bottom) &
                    (sample_ys <= bounds.top)
                )

                valid_indices = np.where(valid_mask)[0]

                print(
                    f"[Worker PID: {worker_pid}] INFO for {col_name}: "
                    f"Found {len(valid_indices)}/{len(xs)} points within bounds.",
                    flush=True
                )

                if len(valid_indices) > 0:
                    valid_coords = [
                        (float(sample_xs[i]), float(sample_ys[i]))
                        for i in valid_indices
                    ]

                    sampled_gen = src.sample(valid_coords)

                    sampled_data = np.fromiter(
                        (val[0] for val in sampled_gen),
                        dtype=np.float32,
                        count=len(valid_coords)
                    )

                    nodata_val = src.nodata

                    if nodata_val is not None:
                        try:
                            if np.isfinite(nodata_val):
                                sampled_data[np.isclose(sampled_data, nodata_val)] = np.nan
                            else:
                                sampled_data[~np.isfinite(sampled_data)] = np.nan
                        except Exception:
                            sampled_data[sampled_data == nodata_val] = np.nan

                    sampled_data[~np.isfinite(sampled_data)] = np.nan

                    extracted_values[valid_indices] = sampled_data

        gc.collect()

        print(
            f"[Worker PID: {worker_pid}] FINISHED extracting raster: {col_name}",
            flush=True
        )

        return {
            "status": "success",
            "col_name": col_name,
            "data": extracted_values
        }

    except Exception as e:
        print(
            f"[Worker PID: {worker_pid}] ERROR on raster {col_name}: {str(e)}",
            flush=True
        )

        return {
            "status": "error",
            "col_name": col_name,
            "error": str(e)
        }


# ==========================================
# 5. Main Execution Block
# ==========================================
if __name__ == "__main__":

    print("--- INITIALIZING SCRIPT ---", flush=True)
    print(f"Reading point data: {CSV_PATH}", flush=True)

    df = pd.read_csv(CSV_PATH)
    total_points = len(df)

    print(f"Total points loaded from CSV: {total_points}", flush=True)

    if "Longitude" not in df.columns or "Latitude" not in df.columns:
        raise ValueError("CSV must contain 'Longitude' and 'Latitude' columns.")

    df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")
    df["Latitude"] = pd.to_numeric(df["Latitude"], errors="coerce")

    xs = df["Longitude"].to_numpy(dtype=np.float64)
    ys = df["Latitude"].to_numpy(dtype=np.float64)

    invalid_xy_count = np.sum(~np.isfinite(xs) | ~np.isfinite(ys))

    if invalid_xy_count > 0:
        print(
            f"WARNING: Found {invalid_xy_count} records with invalid coordinates. "
            f"Extracted raster values for these records will be NaN.",
            flush=True
        )

    print("Scanning for raster files...", flush=True)

    tasks = get_unique_raster_tasks(ALL_RASTERS)
    total_rasters = len(tasks)

    print(f"Found {total_rasters} valid raster files. Missing files skipped.", flush=True)
    print(f"Starting parallel extraction using {MAX_WORKERS} workers...", flush=True)
    print("-" * 50, flush=True)

    extracted_results = {}

    with ProcessPoolExecutor(
        max_workers=MAX_WORKERS,
        initializer=init_worker,
        initargs=(xs, ys)
    ) as executor:

        futures = {
            executor.submit(process_single_raster, task): task
            for task in tasks
        }

        completed = 0

        for future in as_completed(futures):

            raster_path, col_name = futures.pop(future)

            try:
                res = future.result()
            except Exception as e:
                res = {
                    "status": "error",
                    "col_name": col_name,
                    "error": str(e)
                }

            completed += 1
            progress_pct = completed / total_rasters * 100

            if res["status"] == "success":
                extracted_results[res["col_name"]] = res["data"]

                print(
                    f"[MAIN THREAD] Progress: {completed}/{total_rasters} "
                    f"({progress_pct:.1f}%) | Successfully saved {res['col_name']}.",
                    flush=True
                )

            else:
                print(
                    f"[MAIN THREAD] Progress: {completed}/{total_rasters} "
                    f"({progress_pct:.1f}%) | FAILED on {res['col_name']} - {res['error']}",
                    flush=True
                )

            if completed % 5 == 0:
                print("[MAIN THREAD] Performing scheduled garbage collection...", flush=True)
                gc.collect()

    # ==========================================
    # 6. Merge Data and Save
    # ==========================================
    print("-" * 50, flush=True)
    print("All tasks completed. Merging extracted columns into main DataFrame...", flush=True)

    extracted_df = pd.DataFrame(extracted_results)

    if len(extracted_df) != len(df):
        raise ValueError(
            f"Row mismatch: extracted_df has {len(extracted_df)} rows, "
            f"but original df has {len(df)} rows."
        )

    df_out = pd.concat([df, extracted_df], axis=1)

    print("Executing final garbage collection before disk write...", flush=True)

    del extracted_results, extracted_df
    gc.collect()

    print("Saving finalized dataset to disk...", flush=True)

    os.makedirs(OUT_DIR, exist_ok=True)
    df_out.to_csv(OUT_CSV_PATH, index=False)

    print("--- SCRIPT COMPLETED SUCCESSFULLY ---", flush=True)
    print(f"Optimized dataset saved to: {OUT_CSV_PATH}", flush=True)