import rasterio
from rasterio.vrt import WarpedVRT
import numpy as np
import pandas as pd
import os
import gc

# ==========================================
# 1. Configuration & Paths
# ==========================================
output_dir = "/data/China_mapping/compare_other/plot/"
os.makedirs(output_dir, exist_ok=True)
output_csv = os.path.join(output_dir, "All_Products_Comparison_Sampled_1M.csv")

target_samples = 1000000

# Master raster (Our AGB) used to determine CRS, resolution, and valid mask
master_file = "/data/China_mapping/production/fina_production/China_National_AGB_MEAN.tif"

# Dictionary of other products to extract (Name: Path)
other_products = {
    "Our AGE": "/data/China_mapping/production/fina_production/China_National_AGE_MEAN.tif",
    "Our H": "/data/China_mapping/production/fina_production/China_National_H_MEAN.tif",
    "Cai AGB": "/data/China_mapping/compare_other/AGB_2020.tif",
    "CCI AGB": "/data/China_mapping/compare_other/CCAGB/Merged_CCAGB.tif",
    "Cheng H": "/data/China_mapping/compare_other/Ha_China.tif",
    "Lang H": "/data/China_mapping/compare_other/ETH_GlobalCanopyHeight_2020_1km_Bilinear.tif",
    "Cheng AGE": "/data/China_mapping/compare_other/China_Forest_Age.tif",
    "Bonan AGE": "/data/China_mapping/compare_other/ForestAge_TC000.tif"
}

# Dictionary to store the final extracted data
extracted_data = {}

# ==========================================
# 2. Read Master Raster & Generate Random Samples
# ==========================================
print("Loading master raster to establish baseline mask and coordinates...")
with rasterio.open(master_file) as src_master:
    master_data = src_master.read(1)
    master_nodata = src_master.nodata
    master_transform = src_master.transform
    master_crs = src_master.crs
    master_height, master_width = src_master.height, src_master.width

print("Filtering valid pixels from master raster...")
valid_mask = (master_data != master_nodata) & (master_data > 0) & ~np.isnan(master_data)
row_indices, col_indices = np.nonzero(valid_mask)
total_valid = len(row_indices)
print(f"Total valid pixels found: {total_valid:,}")

del valid_mask
gc.collect()

# Random Sampling
if total_valid > target_samples:
    print(f"Randomly sampling {target_samples:,} points...")
    np.random.seed(42)  # Fix random seed for reproducibility
    sample_indices = np.random.choice(total_valid, target_samples, replace=False)
    
    row_indices = row_indices[sample_indices]
    col_indices = col_indices[sample_indices]
else:
    print("Valid pixels are fewer than target samples. Using all available valid pixels.")

# Extract sampled values for Master AGB
extracted_data["Our AGB"] = master_data[row_indices, col_indices].astype(float)

del master_data
gc.collect()

# ==========================================
# 3. Calculate Coordinates (Lon, Lat)
# ==========================================
print("Calculating spatial coordinates for sampled points...")
xs, ys = master_transform * (col_indices, row_indices)
extracted_data["Lon"] = np.array(xs)
extracted_data["Lat"] = np.array(ys)

# Package coordinates into a list of tuples for rasterio.sample
coords = list(zip(extracted_data["Lon"], extracted_data["Lat"]))

# ==========================================
# 4. Extract Values from All Other Rasters via VRT
# ==========================================
for name, filepath in other_products.items():
    print(f"Processing and aligning: {name} ...")
    if not os.path.exists(filepath):
        print(f"  [WARNING] File not found: {filepath}. Filling column with NaN.")
        extracted_data[name] = np.full(len(row_indices), np.nan)
        continue

    with rasterio.open(filepath) as src:
        # Use WarpedVRT to dynamically align the target raster
        with WarpedVRT(src, crs=master_crs, transform=master_transform, 
                       width=master_width, height=master_height) as vrt:
            
            vrt_nodata = vrt.nodata
            
            # Use vrt.sample() to extract exactly the values at the coordinates
            # This prevents loading the entire warped raster into memory
            sampled_values = np.array([val[0] for val in vrt.sample(coords)]).astype(float)
            
            # Convert nodata or invalid values to np.nan
            if vrt_nodata is not None:
                sampled_values[sampled_values == vrt_nodata] = np.nan
            sampled_values[(sampled_values <= 0) | np.isinf(sampled_values)] = np.nan
            
            extracted_data[name] = sampled_values

# ==========================================
# 5. Assemble Data and Export to CSV
# ==========================================
print("Assembling all data into a DataFrame...")

# Define the requested column order
columns_order = [
    "Lon", "Lat", 
    "Our AGB", "Our AGE", "Our H", 
    "Cai AGB", "CCI AGB", 
    "Cheng H", "Lang H", 
    "Cheng AGE", "Bonan AGE"
]

df = pd.DataFrame(extracted_data)
# Reorder columns
df = df[columns_order]

print(f"Exporting {len(df):,} records to CSV...")
df.to_csv(output_csv, index=False)
print(f"Done! Data successfully saved to: {output_csv}")