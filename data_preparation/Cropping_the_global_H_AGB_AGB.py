import os
import geopandas as gpd
import rasterio
from rasterio.mask import mask

# 1. Define input and output paths
vector_mask_path = "/data/China_mapping/climate_zone/clmate_zone.shp"
output_dir = "/data/China_mapping/compare_other/"

input_rasters = [
    "/data/global_mapping/models_V100/compare_other_study/ForestAge_TC000.tif",
    "/data/global_mapping/models_V100/compare_other_study/ETH_GlobalCanopyHeight_2020_1km_Bilinear.tif",
    "/data/global_mapping/models_V100/compare_other_study/Crowther_Density_Stems_per_Ha.tif"
]

# Create output directory if it does not exist
os.makedirs(output_dir, exist_ok=True)

# 2. Read the vector shapefile
print("Loading vector mask...")
vector_data = gpd.read_file(vector_mask_path)

# 3. Iterate through each raster, apply mask, and export separately
for raster_path in input_rasters:
    file_name = os.path.basename(raster_path)
    output_path = os.path.join(output_dir, file_name)
    
    print(f"Processing: {file_name}")
    
    with rasterio.open(raster_path) as src:
        # Check and match Coordinate Reference System (CRS)
        if vector_data.crs != src.crs:
            print(f"  Reprojecting vector to match raster CRS: {src.crs}")
            vector_proj = vector_data.to_crs(src.crs)
        else:
            vector_proj = vector_data
            
        # Extract geometries from the GeoDataFrame
        geometries = vector_proj.geometry.values
        
        try:
            # Perform the spatial masking (cropping to vector extent)
            out_image, out_transform = mask(src, geometries, crop=True)
            out_meta = src.meta.copy()
            
            # Update metadata for the new masked output
            out_meta.update({
                "driver": "GTiff",
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform
            })
            
            # Write the masked array to a new TIFF file
            with rasterio.open(output_path, "w", **out_meta) as dest:
                dest.write(out_image)
                
            print(f"  Successfully exported to: {output_path}")
            
        except ValueError as e:
            print(f"  Error processing {file_name}: {e}")

print("All masking tasks completed successfully.")