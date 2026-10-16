# ==============================================================================
# Script: Massive Multi-Index Forest Structure Calculator (Sentinel-2)
# Status: Production Ready for National-Scale 30m Imagery
# Description: 
#   1. Calculates 10 critical forest structural indices simultaneously.
#   2. Bypasses /dev/shm to prevent system crash on national-scale data.
#   3. Strict float32 enforcement to prevent NumPy implicit memory doubling.
#   4. Aggressive garbage collection at the micro-operation level.
# ==============================================================================

import os
import gc
import shutil
import numpy as np
import rasterio
from rasterio.windows import Window
from concurrent.futures import ProcessPoolExecutor
import time
import random
import subprocess

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================
# Paths
BASE_DIR = "/data/China_mapping/variable/"
SEASON = "Winter"

# CRITICAL FIX: Removed /dev/shm. National 30m data will generate terabytes of 
# temporary chunks. Must write to physical SSD storage to prevent system crash.
TEMP_PATH = os.path.join(BASE_DIR, f"temp_chunks_VIs_{SEASON}")

# Input Band Definitions
REFERENCE_BAND = os.path.join(BASE_DIR, f"B8_{SEASON}.tif")

INPUT_BANDS = {
    'B2': os.path.join(BASE_DIR, f"B2_{SEASON}.tif"),
    'B3': os.path.join(BASE_DIR, f"B3_{SEASON}.tif"),  
    'B4': os.path.join(BASE_DIR, f"B4_{SEASON}.tif"),
    'B5': os.path.join(BASE_DIR, f"B5_{SEASON}.tif"),
    'B6': os.path.join(BASE_DIR, f"B6_{SEASON}.tif"),
    'B8': REFERENCE_BAND,
    'B11': os.path.join(BASE_DIR, f"B11_{SEASON}.tif")
}

# System & Memory Parameters
# For 512GB RAM, 16 workers x 1024 rows on a 180,000 col national image is safe.
# Max peak memory per worker ~15-20GB. Total peak ~320GB.
WORKERS = 16          
CHUNK_ROWS = 1024     

TARGET_VIS = [
    "NDVI", "EVI", "SAVI", "MSAVI2", "NDRE1", 
    "NDRE2", "LSWI", "BSI", "GNDVI", "GLI"
]

os.makedirs(TEMP_PATH, exist_ok=True)
for vi in TARGET_VIS:
    os.makedirs(os.path.join(TEMP_PATH, vi), exist_ok=True)

# ==============================================================================
# 2. VEGETATION INDEX CALCULATOR ENGINE (Memory-Strict)
# ==============================================================================
def calculate_all_vis(bands_dict):
    """
    Computes 10 VIs with extreme memory efficiency.
    Strictly forces float32 to prevent numpy memory bloat.
    Aggressively deletes intermediate arrays.
    """
    b2 = bands_dict['B2']
    b3 = bands_dict['B3']
    b4 = bands_dict['B4']
    b5 = bands_dict['B5']
    b6 = bands_dict['B6']
    b8 = bands_dict['B8']
    b11 = bands_dict['B11']

    # Force float32 constants to prevent float64 implicit casting
    f32 = np.float32 

    valid_mask = (~np.isnan(b8)) & (~np.isnan(b4)) & (~np.isnan(b3))
    results = {}
    
    with np.errstate(divide='ignore', invalid='ignore'):
        # 1. NDVI
        denom = b8 + b4
        results['NDVI'] = np.where(valid_mask & (denom != 0), (b8 - b4) / denom, np.nan).astype(f32)
        del denom # Destroy intermediate immediately

        # 2. EVI
        denom = b8 + f32(6.0) * b4 - f32(7.5) * b2 + f32(1.0)
        results['EVI'] = np.where(valid_mask & (denom != 0), f32(2.5) * (b8 - b4) / denom, np.nan).astype(f32)
        del denom

        # 3. SAVI
        denom = b8 + b4 + f32(0.5)
        results['SAVI'] = np.where(valid_mask & (denom != 0), f32(1.5) * (b8 - b4) / denom, np.nan).astype(f32)
        del denom

        # 4. MSAVI2
        term1 = f32(2.0) * b8 + f32(1.0)
        term2 = np.sqrt(np.maximum(f32(0), term1**2 - f32(8.0) * (b8 - b4)))
        results['MSAVI2'] = np.where(valid_mask, f32(0.5) * (term1 - term2), np.nan).astype(f32)
        del term1, term2

        # 5. NDRE1
        denom = b8 + b5
        results['NDRE1'] = np.where(valid_mask & (~np.isnan(b5)) & (denom != 0), (b8 - b5) / denom, np.nan).astype(f32)
        del denom

        # 6. NDRE2
        denom = b8 + b6
        results['NDRE2'] = np.where(valid_mask & (~np.isnan(b6)) & (denom != 0), (b8 - b6) / denom, np.nan).astype(f32)
        del denom

        # 7. LSWI
        denom = b8 + b11
        results['LSWI'] = np.where(valid_mask & (~np.isnan(b11)) & (denom != 0), (b8 - b11) / denom, np.nan).astype(f32)
        del denom

        # 8. BSI
        swir_red = b11 + b4
        nir_blue = b8 + b2
        denom = swir_red + nir_blue
        results['BSI'] = np.where(valid_mask & (~np.isnan(b11)) & (~np.isnan(b2)) & (denom != 0), 
                                  (swir_red - nir_blue) / denom, np.nan).astype(f32)
        del swir_red, nir_blue, denom

        # 9. GNDVI
        denom = b8 + b3
        results['GNDVI'] = np.where(valid_mask & (denom != 0), (b8 - b3) / denom, np.nan).astype(f32)
        del denom
        
        # 10. GLI
        denom = f32(2.0) * b3 + b4 + b2
        results['GLI'] = np.where(valid_mask & (~np.isnan(b2)) & (denom != 0), 
                                  (f32(2.0) * b3 - b4 - b2) / denom, np.nan).astype(f32)
        del denom

    # Clean up outliers safely
    for vi_name in results.keys():
        results[vi_name] = np.clip(results[vi_name], f32(-2.0), f32(2.0))

    return results

# ==============================================================================
# 3. WORKER FUNCTION
# ==============================================================================
def process_multi_band_chunk(args):
    row_off, height, src_width, total_height, crs, transform = args
    
    # Stagger startups to avoid massive concurrent disk seeks
    time.sleep(random.uniform(0.1, 2.0))
    
    all_exist = True
    for vi in TARGET_VIS:
        out_name = os.path.join(TEMP_PATH, vi, f"{vi}_chunk_{row_off}.tif")
        if not (os.path.exists(out_name) and os.path.getsize(out_name) > 1024):
            all_exist = False
            break
    if all_exist: return True

    read_window = Window(col_off=0, row_off=row_off, width=src_width, height=height)
    win_transform = rasterio.windows.transform(read_window, transform)
    
    chunk_data = {}
    max_retries = 3
    
    with rasterio.Env(GDAL_CACHEMAX=1024, NUM_THREADS='1'):
        for attempt in range(max_retries):
            try:
                for b_name, b_path in INPUT_BANDS.items():
                    with rasterio.open(b_path) as src:
                        data = src.read(1, window=read_window).astype(np.float32)
                        nodata = src.nodata
                        if nodata is not None:
                            data[data == nodata] = np.nan
                        chunk_data[b_name] = data
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(random.uniform(2, 6))
                    continue
                else:
                    print(f"Error reading chunk {row_off}: {e}")
                    return False

        if np.all(np.isnan(chunk_data['B8'])) or np.all(chunk_data['B8'] == 0):
            del chunk_data
            gc.collect()
            return True 

        # Compute
        try:
            vi_results = calculate_all_vis(chunk_data)
        except Exception as e:
            print(f"Calculation error in chunk {row_off}: {e}")
            return False
            
        # VERY IMPORTANT: Release input band memory immediately before writing outputs
        del chunk_data
        gc.collect()

        # Write output chunks (Compression set to DEFLATE/ZSTD for faster I/O balance on SSD)
        profile = {
            'driver': 'GTiff', 
            'height': height, 
            'width': src_width,
            'count': 1, 
            'dtype': rasterio.float32, 
            'crs': crs,
            'transform': win_transform, 
            'compress': 'deflate',  # Faster than LZW for temporary writes
            'zlevel': 1,            # Lowest compression level for maximum write speed
            'nodata': np.nan,
            'bigtiff': 'yes'
        }

        for vi_name in TARGET_VIS:
            out_name = os.path.join(TEMP_PATH, vi_name, f"{vi_name}_chunk_{row_off}.tif")
            temp_out = out_name + ".tmp"
            
            try:
                with rasterio.open(temp_out, 'w', **profile) as dst:
                    dst.write(vi_results[vi_name], 1)
                os.rename(temp_out, out_name)
            except Exception as e:
                print(f"Error writing {vi_name} chunk {row_off}: {e}")
                if os.path.exists(temp_out): os.remove(temp_out)

        del vi_results
        gc.collect()
        return True

# ==============================================================================
# 4. MAIN EXECUTION
# ==============================================================================
if __name__ == "__main__":
    
    for b_name, b_path in INPUT_BANDS.items():
        if not os.path.exists(b_path):
            raise FileNotFoundError(f"Missing input band: {b_path}")

    with rasterio.open(REFERENCE_BAND) as src:
        total_height = src.height
        src_width = src.width
        crs = src.crs
        transform = src.transform
        
    offsets = list(range(0, total_height, CHUNK_ROWS))
    print(f"Image Width: {src_width}, Total Height: {total_height}")
    print(f"Chunk Size: {CHUNK_ROWS}, Total Chunks: {len(offsets)}")
    
    tasks = []
    for off in offsets:
        h = min(CHUNK_ROWS, total_height - off)
        tasks.append((off, h, src_width, total_height, crs, transform))

    # Phase 1: Heavy Computation & Direct SSD Write
    print(f"\n>>> Phase 1: Calculating 10 VIs across {WORKERS} workers <<<")
    start_time = time.time()
    
    with ProcessPoolExecutor(max_workers=WORKERS) as executor:
        results = list(executor.map(process_multi_band_chunk, tasks))
        
    calc_time = time.time()
    print(f"  Calculations finished in {(calc_time - start_time)/60:.1f} mins.")

    # Phase 2: Sequential VRT Build and Translate
    print(f"\n>>> Phase 2: Merging temporary chunks to final BigTIFFs <<<")
    
    for metric in TARGET_VIS:
        print(f"  Merging {metric}_{SEASON}.tif ...")
        metric_dir = os.path.join(TEMP_PATH, metric)
        final_output = os.path.join(BASE_DIR, f"{metric}_{SEASON}.tif")
        temp_vrt = os.path.join(metric_dir, "merge.vrt")
        txt_list = os.path.join(metric_dir, "chunk_list.txt")
        
        chunk_files = [os.path.join(metric_dir, f) for f in os.listdir(metric_dir) if f.endswith('.tif')]
        if not chunk_files:
            print(f"  WARNING: No chunks found for {metric}. Skipping.")
            continue
            
        with open(txt_list, 'w') as f:
            for chunk in chunk_files:
                f.write(f"{chunk}\n")
        
        try:
            # Build VRT
            subprocess.run(["gdalbuildvrt", "-input_file_list", txt_list, temp_vrt], check=True)
            
            # Translate to final BigTIFF (Using ALL_CPUS for multi-threaded compression)
            subprocess.run([
                "gdal_translate", 
                "-co", "COMPRESS=LZW", 
                "-co", "BIGTIFF=YES", 
                "-co", "TILED=YES",
                "-co", "NUM_THREADS=ALL_CPUS", 
                temp_vrt, 
                final_output
            ], check=True)
            
            # Clean up the specific metric folder to free SSD space progressively
            print(f"  Freeing SSD space for {metric}...")
            shutil.rmtree(metric_dir)
            
        except subprocess.CalledProcessError as e:
            print(f"  ERROR during GDAL merge for {metric}: {e}")
            continue

    if os.path.exists(TEMP_PATH):
        shutil.rmtree(TEMP_PATH)

    print(f"\n=== ALL PROCESSING COMPLETE. Total Time: {(time.time() - start_time)/60:.1f} mins ===")