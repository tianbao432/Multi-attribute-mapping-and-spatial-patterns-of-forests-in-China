import os
import glob
import subprocess
import tempfile

# ==========================================
#Spring,Summer,Autumn,Winter
# ==========================================
INPUT_DIR ="/data/China_mapping/S1/VV_summer/"
SEARCH_PATTERN = "*.tif"
OUTPUT_FILEPATH = "/data/China_mapping/variable/VV_Summer.tif"

def mosaic_rasters_optimized(input_dir, pattern, output_path):
    search_criteria = os.path.join(input_dir, pattern)
    tif_files = glob.glob(search_criteria)

    if not tif_files:
        raise FileNotFoundError(f"No files matching '{pattern}' found in {input_dir}")

    print(f"Found {len(tif_files)} raster files. Starting multi-threaded parallel processing...")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Use a temporary file to store the list of paths to prevent command line length limits
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as list_file:
        for f in tif_files:
            list_file.write(f"{f}\n")
        file_list_path = list_file.name

    vrt_path = output_path.replace('.tif', '.vrt')

    # ---------------------------------------------------------
    # Step A: Build Virtual Raster (VRT spatial index)
    # ---------------------------------------------------------
    build_vrt_cmd = [
        "gdalbuildvrt",
        "-input_file_list", file_list_path,
        vrt_path
    ]
    print("-> Building VRT spatial index...")
    subprocess.run(build_vrt_cmd, check=True)

    # ---------------------------------------------------------
    # Step B: Multi-threaded block-by-block translation (Memory & I/O optimized)
    # ---------------------------------------------------------
    # [Core Memory Management] Allocate 150,000 MB (150GB) cache to GDAL
    # Prevents massive pixel overflow to swap space, eliminating I/O bottlenecks
    os.environ['GDAL_CACHEMAX'] = '150000' 
    
    # Force GDAL underlying operations to use all CPU cores
    os.environ['GDAL_NUM_THREADS'] = 'ALL_CPUS'

    translate_cmd = [
        "gdal_translate",
        vrt_path,
        output_path,
        "-of", "GTiff",
        "-co", "BIGTIFF=YES",           # Break the 4GB file size limit
        "-co", "TILED=YES",             # Enable tiled storage for faster subsequent reads
        "-co", "BLOCKXSIZE=512",        # Tile width set to 512
        "-co", "BLOCKYSIZE=512",        # Tile height set to 512
        "-co", "COMPRESS=ZSTD",         # [Optimization] Use ZSTD compression, highly efficient for multi-threading
        "-co", "ZSTD_LEVEL=1",          # [Optimization] ZSTD level 1: extremely fast with excellent size control
        "-co", "NUM_THREADS=ALL_CPUS",  # Use all CPU cores during writing
    ]

    print("-> Starting multi-threaded writing of the final massive image...")
    subprocess.run(translate_cmd, check=True)

    # ---------------------------------------------------------
    # Step C: Garbage collection and cleanup
    # ---------------------------------------------------------
    os.remove(file_list_path)
    os.remove(vrt_path)
    print(f"\nProcessing complete! Lossless mosaic saved to: {output_path}")

if __name__ == "__main__":
    mosaic_rasters_optimized(
        input_dir=INPUT_DIR,
        pattern=SEARCH_PATTERN,
        output_path=OUTPUT_FILEPATH
    )