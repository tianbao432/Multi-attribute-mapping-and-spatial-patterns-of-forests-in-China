# ==============================================================================
# Script: Extreme-Speed GLCM Texture Extraction (30m Resolution)
# Status: Fully Optimized for 512GB RAM Server
# Description: 
#   1. Routes all temporary chunk I/O strictly through /dev/shm (RAM Disk).
#   2. High concurrency (24 workers) to fully utilize multicore CPUs.
#   3. Strict memory cleanup after each metric to prevent RAM disk overflow.
#   4. Preserves GDAL single-threading per worker to prevent thread collision.
# ==============================================================================

import os
import gc
import shutil
import numpy as np
import rasterio
from rasterio.windows import Window
from concurrent.futures import ProcessPoolExecutor
from numba import jit
import time
import random
import math
import subprocess

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================
# Paths
INPUT_FILE ="/data/China_mapping/variable/B8_Winter.tif"
OUTPUT_PATH = "/data/China_mapping/variable/"

# [CRITICAL UPDATE]: Route temporary chunks directly to ultra-fast RAM Disk
TEMP_PATH = "/dev/shm/temp_chunks_B8_Winter"

# System & Memory Parameters
# [CRITICAL UPDATE]: Scaled up for heavy multi-core processing
WORKERS = 24          
CHUNK_ROWS = 1024    

# GLCM Parameters
GLCM_WINDOW_SIZE = 3 
BUFFER_SIZE = 4       
GLCM_LEVELS = 32      

TARGET_METRICS = [
    "mean", "variance", "correlation",  
    "contrast", "dissimilarity", "homogeneity", 
    "asm", "entropy" 
]

os.makedirs(TEMP_PATH, exist_ok=True)

# ==============================================================================
# 2. NUMBA OPTIMIZED GLCM ENGINE
# ==============================================================================
@jit(nopython=True, parallel=False, fastmath=True)
def calculate_glcm_metric(data, metric_code, levels):
    rows, cols = data.shape
    result = np.full((rows, cols), np.nan, dtype=np.float32)
    
    dr = np.array([0, 1, 1, 1])
    dc = np.array([1, 1, 0, -1])
    
    quantized = np.floor(data * levels).astype(np.int32)
    hist_matrix = np.zeros((levels, levels), dtype=np.float32)

    for r in range(1, rows - 1):
        for c in range(1, cols - 1):
            val_center = quantized[r, c]
            if val_center < 0 or val_center >= levels: continue 
            
            valid_pairs = 0
            
            if metric_code == 3 or metric_code == 4:
                hist_matrix[:] = 0.0
                for k in range(4):
                    nr, nc = r + dr[k], c + dc[k]
                    val_neighbor = quantized[nr, nc]
                    if val_neighbor >= 0 and val_neighbor < levels:
                        hist_matrix[val_center, val_neighbor] += 1
                        hist_matrix[val_neighbor, val_center] += 1
                        valid_pairs += 2 

                if valid_pairs > 0:
                    metric_val = 0.0
                    inv_N = 1.0 / valid_pairs
                    
                    if metric_code == 3: 
                        for i in range(levels):
                            for j in range(levels):
                                count = hist_matrix[i, j]
                                if count > 0:
                                    p = count * inv_N
                                    metric_val += p * p
                                    
                    elif metric_code == 4: 
                        for i in range(levels):
                            for j in range(levels):
                                count = hist_matrix[i, j]
                                if count > 0:
                                    p = count * inv_N
                                    metric_val -= p * math.log(p) 
                    
                    result[r, c] = metric_val

            else:
                sum_diff_sq = 0.0   
                sum_diff_abs = 0.0  
                sum_inv_diff = 0.0  
                sum_val = 0.0       
                sum_sq_val = 0.0    
                sum_prod = 0.0      
                count = 0.0

                for k in range(4):
                    nr, nc = r + dr[k], c + dc[k]
                    val_neighbor = quantized[nr, nc]
                    
                    if val_neighbor >= 0 and val_neighbor < levels:
                        i = float(val_center)
                        j = float(val_neighbor)
                        
                        diff = i - j
                        diff_sq = diff * diff
                        
                        if metric_code == 0: sum_diff_sq += diff_sq
                        elif metric_code == 1: sum_diff_abs += abs(diff)
                        elif metric_code == 2: sum_inv_diff += 1.0 / (1.0 + diff_sq)
                        
                        if metric_code >= 5:
                            sum_val += i + j
                            sum_sq_val += (i*i) + (j*j)
                            sum_prod += (i * j) * 2.0 
                            
                        count += 2.0 

                if count > 0:
                    if metric_code == 0: result[r, c] = sum_diff_sq / count
                    elif metric_code == 1: result[r, c] = sum_diff_abs / count
                    elif metric_code == 2: result[r, c] = sum_inv_diff / count
                    elif metric_code >= 5:
                        mean = sum_val / count
                        variance = (sum_sq_val / count) - (mean * mean)
                        
                        if metric_code == 5: result[r, c] = mean / levels 
                        elif metric_code == 6: result[r, c] = variance
                        elif metric_code == 7: 
                            covariance = (sum_prod / count) - (mean * mean)
                            if variance > 1e-6:
                                result[r, c] = covariance / variance
                            else:
                                result[r, c] = 0.0

    return result

def run_glcm(image_array, metric_name):
    codes = {
        "contrast": 0, "dissimilarity": 1, "homogeneity": 2, 
        "asm": 3, "entropy": 4, "mean": 5, "variance": 6, "correlation": 7
    }
    if metric_name not in codes: return None
    return calculate_glcm_metric(image_array, codes[metric_name], GLCM_LEVELS)

# ==============================================================================
# 3. WORKER FUNCTION
# ==============================================================================
def process_chunk_buffered(args):
    row_idx, write_off, write_height, total_height, metric_name, input_path, out_dir, buffer_size = args
    
    # Stagger startups slightly to avoid initial hit on the physical read disk
    time.sleep(random.uniform(0.5, 3.0))
    
    out_name = os.path.join(out_dir, f"{metric_name}_chunk_{row_idx}.tif")
    
    if os.path.exists(out_name) and os.path.getsize(out_name) > 1024: 
        return out_name 
    
    temp_out_name = out_name + ".tmp"
    if os.path.exists(temp_out_name):
        try: os.remove(temp_out_name)
        except OSError: pass

    max_retries = 5
    data = None
    
    with rasterio.Env(GDAL_CACHEMAX=512, NUM_THREADS='1'):
        for attempt in range(max_retries):
            try:
                with rasterio.open(input_path) as src_handle:
                    read_off = max(0, write_off - buffer_size)
                    read_bottom = min(total_height, write_off + write_height + buffer_size)
                    read_height = read_bottom - read_off
                    
                    pad_top = write_off - read_off
                    pad_bottom = read_bottom - (write_off + write_height)
                    
                    read_window = Window(col_off=0, row_off=read_off, width=src_handle.width, height=read_height)
                    data = src_handle.read(1, window=read_window)
                    win_transform = src_handle.window_transform(read_window)
                    src_crs = src_handle.crs
                    src_nodata = src_handle.nodata
                    src_width = src_handle.width
                break 
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(random.uniform(1, 4)) 
                    continue
                else:
                    return None

        try:
            if data is None or data.size == 0: return None
            data = data.astype(np.float32)
            
            if src_nodata is not None:
                data[data == src_nodata] = np.nan
            
            data = np.clip(data, 0.0, 1.0)
            
            if np.all(np.isnan(data)): 
                del data
                gc.collect()
                return None
            
            data = np.ascontiguousarray(data)
            tex_result = run_glcm(data, metric_name)
            
            del data
            gc.collect()
            
            if tex_result is None or np.all(np.isnan(tex_result)): 
                del tex_result
                gc.collect()
                return None

            valid_data = tex_result[pad_top : read_height - pad_bottom, :]
            if valid_data.shape[0] != write_height:
                valid_data = tex_result[pad_top : pad_top + write_height, :]

            new_top_y = win_transform.f + (pad_top * win_transform.e)
            dst_transform = rasterio.Affine(win_transform.a, win_transform.b, win_transform.c,
                                            win_transform.d, win_transform.e, new_top_y)

            profile = {
                'driver': 'GTiff', 
                'height': write_height, 
                'width': src_width,
                'count': 1, 
                'dtype': rasterio.float32, 
                'crs': src_crs,
                'transform': dst_transform, 
                'compress': 'lzw', 
                'nodata': np.nan,
                'bigtiff': 'yes'
            }
            
            with rasterio.open(temp_out_name, 'w', **profile) as dst:
                dst.write(valid_data, 1)
            
            os.rename(temp_out_name, out_name)
                
            del tex_result, valid_data
            gc.collect()
                
            return out_name

        except Exception as e:
            if os.path.exists(temp_out_name): os.remove(temp_out_name)
            return None

# ==============================================================================
# 4. MAIN EXECUTION
# ==============================================================================
if __name__ == "__main__":
    
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    with rasterio.open(INPUT_FILE) as src:
        total_height = src.height
        offsets = list(range(0, total_height, CHUNK_ROWS))
    
    print(f"Total Height: {total_height}, Chunk Size: {CHUNK_ROWS}, Total Chunks: {len(offsets)}")
    print(f"Metrics to process: {TARGET_METRICS}")

    for metric in TARGET_METRICS:
        print(f"\n>>> Starting Metric: {metric} <<<")
        start_time = time.time()
        
        metric_dir = os.path.join(TEMP_PATH, metric)
        os.makedirs(metric_dir, exist_ok=True)
        
        tasks = []
        for i, off in enumerate(offsets):
            h = min(CHUNK_ROWS, total_height - off)
            tasks.append((i, off, h, total_height, metric, INPUT_FILE, metric_dir, BUFFER_SIZE))
        
        with ProcessPoolExecutor(max_workers=WORKERS) as executor:
            results = list(executor.map(process_chunk_buffered, tasks))
            chunk_files = [r for r in results if r is not None]
        
        if not chunk_files:
            print(f"  WARNING: No valid chunks generated for {metric}. Skipping merge.")
            continue

        print(f"  Merging {len(chunk_files)} chunks into physical storage...")
        final_output = os.path.join(OUTPUT_PATH, f"{metric}_Winter.tif")
        temp_vrt = os.path.join(metric_dir, "merge.vrt")
        txt_list = os.path.join(metric_dir, "chunk_list.txt")
        
        if os.path.exists(temp_vrt): os.remove(temp_vrt)
        
        with open(txt_list, 'w') as f:
            for chunk in chunk_files:
                f.write(f"{chunk}\n")
        
        try:
            subprocess.run(["gdalbuildvrt", "-input_file_list", txt_list, temp_vrt], check=True)
            subprocess.run([
                "gdal_translate", 
                "-co", "COMPRESS=LZW", 
                "-co", "BIGTIFF=YES", 
                "-co", "TILED=YES", 
                temp_vrt, 
                final_output
            ], check=True)
            
            # ==================================================================
            # [CRITICAL UPDATE]: Instant RAM Disk Cleanup
            # Prevents /dev/shm from overflowing during multi-metric processing
            # ==================================================================
            print(f"  Cleaning up RAM disk for {metric}...")
            if os.path.exists(metric_dir):
                shutil.rmtree(metric_dir)
                
        except subprocess.CalledProcessError as e:
            print(f"  ERROR during GDAL merge for {metric}: {e}")
            continue
        
        end_time = time.time()
        print(f"  Finished {metric} in {(end_time - start_time)/60:.1f} mins. Saved to {final_output}")

    # Final sweep of the master temp directory just in case
    if os.path.exists(TEMP_PATH):
        shutil.rmtree(TEMP_PATH)

    print("\n=== ALL PROCESSING COMPLETE ===")