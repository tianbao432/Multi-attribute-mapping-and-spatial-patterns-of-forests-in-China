# -*- coding: utf-8 -*-
"""
National Mosaicking Pipeline
Optimizations applied:
  - Glob file discovery across all climate zones.
  - Efficient merge using rasterio path references (avoids open file handle limits).
  - LZW compression with Predictor 3 (optimal for floating-point raster data reduction).
  - Tiled and BigTIFF enabled for fast I/O and GIS rendering.
  - Safe temp file writing before overriding final outputs.
  - Immediate garbage collection after each major matrix operation.
"""
import os
import glob
import gc
import logging
import rasterio
from rasterio.merge import merge
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("National_Mosaic")

# ==============================================================================
# Configuration
# ==============================================================================
INPUT_BASE = "/data/China_mapping/fina_mapping/"
OUTPUT_BASE = "/data/China_mapping/production/fina_production/"
# Target variables (AGE/ is prioritized as requested. Add others when ready.)
TARGET_VARS = ["DBH"] 
STATS = ["MEAN", "CV"]

# ==============================================================================
# Processing Logic
# ==============================================================================
def create_national_mosaic(target_var, input_base, output_base):
    os.makedirs(output_base, exist_ok=True)

    for stat in STATS:
        logger.info(f"========== Starting National Mosaic for: {target_var} [{stat}] ==========")
        
        # Path logic matching previous script's output structure
        # E.g., /data/China_mapping/fina_mapping/*/AGE/Final_Mosaic/Zone*_AGE_MEAN_Mosaic.tif
        search_pattern = os.path.join(
            input_base, 
            "*", 
            target_var, 
            "Final_Mosaic", 
            f"Zone*_{target_var}_{stat}_Mosaic.tif"
        )
        tif_files = sorted(glob.glob(search_pattern))
        
        if not tif_files:
            logger.warning(f"No regional files found for {target_var} [{stat}].")
            logger.warning(f"Checked pattern: {search_pattern}")
            continue
            
        logger.info(f"Found {len(tif_files)} regional mosaics to merge.")
        
        out_filename = f"China_National_{target_var}_{stat}.tif"
        out_filepath = os.path.join(output_base, out_filename)
        
        if os.path.exists(out_filepath):
            logger.info(f"Output already exists, skipping: {out_filepath}\n")
            continue

        try:
            # Read base metadata from the first file
            with rasterio.open(tif_files[0]) as src0:
                out_meta = src0.meta.copy()

            logger.info("Executing rasterio merge (this may take a while depending on RAM)...")
            
            # Pass file paths directly to avoid hitting OS 'too many open files' limit
            mosaic, out_trans = merge(tif_files, nodata=-9999)
            
            # Update metadata for GIS optimization and minimal file size
            out_meta.update({
                "driver": "GTiff",
                "height": mosaic.shape[1],
                "width": mosaic.shape[2],
                "transform": out_trans,
                "count": 1,
                "dtype": 'float32',
                "nodata": -9999,
                "compress": "lzw",
                "predictor": 3,
                "bigtiff": "YES",
                "tiled": True,
                "blockxsize": 256,
                "blockysize": 256
            })
            
            logger.info("Writing compressed raster to disk...")
            tmp_filepath = out_filepath + ".tmp"
            
            with rasterio.open(tmp_filepath, "w", **out_meta) as dest:
                dest.write(mosaic)
            
            # Atomic rename for safety
            os.replace(tmp_filepath, out_filepath)
            logger.info(f"Successfully finalized: {out_filepath}\n")
            
        except Exception as e:
            logger.error(f"Failed to merge {target_var} [{stat}]. Error: {e}\n")
            
        finally:
            if 'mosaic' in locals():
                del mosaic
            if 'out_trans' in locals():
                del out_trans
            gc.collect()

# ==============================================================================
# Main
# ==============================================================================
if __name__ == "__main__":
    logger.info("Starting National Aggregation Pipeline...")
    
    for var in TARGET_VARS:
        create_national_mosaic(var, INPUT_BASE, OUTPUT_BASE)
        
    logger.info("All requested national mosaics completed.")