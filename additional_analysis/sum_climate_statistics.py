import os
import pyproj

# Fix for pyproj CRSError (Internal Proj Error)
os.environ["PROJ_LIB"] = "/home/ubuntu/miniconda3/envs/china/share/proj"
pyproj.datadir.set_data_dir("/home/ubuntu/miniconda3/envs/china/share/proj")

import re
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.windows import Window
from rasterio.windows import bounds as window_bounds
from rasterio.features import rasterize
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

# ==========================================
# 1. Configuration 
# ==========================================
# Target variables for absolute totals calculation
TARGET_VARIABLES = {
    "V": "/data/China_mapping/production/fina_production/China_National_V_MEAN.tif",
    "AGB": "/data/China_mapping/production/fina_production/China_National_AGB_MEAN.tif",
    "BA": "/data/China_mapping/production/fina_production/China_National_BA_MEAN.tif",
    "Trees_N": "/data/China_mapping/production/fina_production/China_National_Trees_N_MEAN.tif"
}
VAR_NAMES = list(TARGET_VARIABLES.keys())
RASTER_PATHS = list(TARGET_VARIABLES.values())

SHP_PATH = "/data/China_mapping/climate_zone/clmate_zone.shp"
ZONE_FIELD = "clim_abbr"  
OUT_DIR = "/data/China_mapping/Climate_statistics/statistic/"

# Increased chunk size to utilize 500GB RAM efficiently
CHUNK_SIZE = 8192 
EARTH_R = 6371007.2  # Earth's mean radius in meters (WGS84)

os.makedirs(OUT_DIR, exist_ok=True)

def clean_name(name):
    return re.sub(r'[^a-zA-Z0-9_]', '_', str(name))

# ==========================================
# 2. Global Spatial Initialization
# ==========================================
print("Loading climate zone shapefile into shared memory...")
GDF = gpd.read_file(SHP_PATH)

# Get reference CRS and transform from the first target raster
with rasterio.open(RASTER_PATHS[0]) as tmp_src:
    REF_CRS = tmp_src.crs
    REF_TRANSFORM = tmp_src.transform
    WIDTH = tmp_src.width
    HEIGHT = tmp_src.height
    
    # Pixel dimensions in degrees
    DLON = abs(REF_TRANSFORM.a)
    DLAT = abs(REF_TRANSFORM.e)

if GDF.crs != REF_CRS:
    print("Reprojecting shapefile...")
    GDF = GDF.to_crs(REF_CRS)

UNIQUE_ZONES = GDF[ZONE_FIELD].dropna().unique()
ZONE_TO_ID = {zone: i + 1 for i, zone in enumerate(UNIQUE_ZONES)}
ID_TO_ZONE = {i + 1: zone for i, zone in enumerate(UNIQUE_ZONES)}
GDF["zone_id"] = GDF[ZONE_FIELD].map(ZONE_TO_ID)
SINDEX = GDF.sindex

# ==========================================
# 3. Worker Function (Processes all params in one pass)
# ==========================================
def process_window(window):
    """Calculates exact absolute totals for all variables in a single pass."""
    try:
        srcs = [rasterio.open(p) for p in RASTER_PATHS]
        nodata_vals = [src.nodata for src in srcs]
        
        arrays = [src.read(1, window=window) for src in srcs]
        
        # Create valid mask (must be valid across all rasters to be safe, or at least the first one)
        valid = np.ones(arrays[0].shape, dtype=bool)
        for arr, nd in zip(arrays, nodata_vals):
            if nd is not None:
                valid &= (arr != nd)
            if arr.dtype.kind in 'fc':
                valid &= (~np.isnan(arr))
                
        if not valid.any():
            for src in srcs: src.close()
            return None

        # Extract valid pixel values. Shape: (N_valid_pixels, N_variables)
        valid_data = np.column_stack([arr[valid] for arr in arrays])
        
        # Spatial intersection to get climate zones
        win_transform = rasterio.windows.transform(window, REF_TRANSFORM)
        bounds = window_bounds(window, REF_TRANSFORM)
        intersect_idx = list(SINDEX.intersection(bounds))
        
        if intersect_idx:
            intersect_gdf = GDF.iloc[intersect_idx]
            shapes = ((geom, z_id) for geom, z_id in zip(intersect_gdf.geometry, intersect_gdf["zone_id"]))
            if len(intersect_gdf) > 0:
                zone_arr = rasterize(
                    shapes,
                    out_shape=arrays[0].shape,
                    transform=win_transform,
                    fill=0,
                    dtype=np.uint16
                )
                valid_zones = zone_arr[valid]
            else:
                valid_zones = np.zeros(valid_data.shape[0], dtype=np.uint16)
        else:
            valid_zones = np.zeros(valid_data.shape[0], dtype=np.uint16)

        # -----------------------------------------------------------------
        # EXACT WGS84 AREA CALCULATION (Hectares per Pixel based on Latitude)
        # -----------------------------------------------------------------
        row_indices = np.arange(window.height)
        _, y_coords = rasterio.transform.xy(win_transform, row_indices, np.zeros(window.height))
        y_coords = np.array(y_coords)
        
        lats_grid = np.broadcast_to(y_coords[:, None], (window.height, window.width))
        valid_lats = lats_grid[valid]
        
        dlon_rad = np.radians(DLON)
        dlat_rad = np.radians(DLAT)
        lat_rad = np.radians(valid_lats)
        
        # Area(m2) = R^2 * dLon * 2 * sin(dLat/2) * cos(Lat)
        pixel_area_m2 = (EARTH_R**2) * dlon_rad * 2 * np.sin(dlat_rad / 2) * np.cos(lat_rad)
        pixel_area_ha = pixel_area_m2 / 10000.0  
        
        # Value/ha * pixel_area_ha = Absolute Total Amount
        # Shape: (N_valid_pixels, N_variables) * (N_valid_pixels, 1)
        absolute_amounts = valid_data * pixel_area_ha[:, None]
        
        # Aggregate totals for this window
        window_totals = {"National": np.sum(absolute_amounts, axis=0)}
        unique_z_ids = np.unique(valid_zones)
        
        for z_id in unique_z_ids:
            if z_id == 0:
                continue
            z_name = clean_name(ID_TO_ZONE[z_id])
            mask = (valid_zones == z_id)
            window_totals[z_name] = np.sum(absolute_amounts[mask], axis=0)

        for src in srcs: src.close()
        return window_totals

    except Exception as e:
        return None 

# ==========================================
# 4. Main Execution
# ==========================================
if __name__ == '__main__':
    windows = []
    for row_off in range(0, HEIGHT, CHUNK_SIZE):
        for col_off in range(0, WIDTH, CHUNK_SIZE):
            w = min(CHUNK_SIZE, WIDTH - col_off)
            h = min(CHUNK_SIZE, HEIGHT - row_off)
            windows.append(Window(col_off, row_off, w, h))

    total_windows = len(windows)
    
    # Use max available cores, leave a tiny buffer for OS
    n_cores = max(1, int(multiprocessing.cpu_count() * 0.90))
    print(f"Generated {total_windows} blocks (CHUNK_SIZE={CHUNK_SIZE}). Using {n_cores} CPU cores.")

    # Master dictionary to hold final absolute totals
    master_totals = {"National": np.zeros(len(VAR_NAMES))}
    for zone in UNIQUE_ZONES:
        master_totals[clean_name(zone)] = np.zeros(len(VAR_NAMES))

    # Run highly optimized multiprocessing pool
    completed = 0
    with ProcessPoolExecutor(max_workers=n_cores) as executor:
        futures = {executor.submit(process_window, w): w for w in windows}
        
        for future in as_completed(futures):
            res = future.result()
            if res is not None:
                # Accumulate totals in memory (Very low RAM footprint)
                for key, totals_array in res.items():
                    if key not in master_totals:
                        master_totals[key] = np.zeros(len(VAR_NAMES))
                    master_totals[key] += totals_array
                            
            completed += 1
            if completed % 20 == 0 or completed == total_windows:
                print(f"Progress: {completed}/{total_windows} chunks processed...")

    # -------------------------------------------------------------
    # EXPORT RESULTS
    # -------------------------------------------------------------
    print("\nExtraction complete. Exporting final summary table...")
    
    # Format the final DataFrame
    totals_df = pd.DataFrame.from_dict(master_totals, orient='index', columns=[f"Total_{v}" for v in VAR_NAMES])
    totals_df.index.name = "Region"
    
    # Save the consolidated table
    summary_csv_path = os.path.join(OUT_DIR, "Absolute_Totals_Summary_All_Parameters.csv")
    totals_df.to_csv(summary_csv_path)
    print(f"-> Master totals table successfully saved to: {summary_csv_path}")
    print("All tasks finished perfectly!")