import os
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from shapely.geometry import mapping

shapefile_path = "/data/China_mapping/climate_zone/clmate_zone.shp"
output_dir = "/data/China_mapping/driving_variables/"

os.makedirs(output_dir, exist_ok=True)

selected_rasters = {
    # Human influence
    "HMI": "/data/global_mapping/models_V100/1km_traditional_bilinear_no_forestslip/HMI.tif",

    # Temperature
    "MAT": "/data/global_mapping/models_V100/1km_traditional_bilinear_no_forestslip/MAT.tif",
    "ISO": "/data/global_mapping/models_V100/1km_traditional_bilinear_no_forestslip/ISO.tif",
    "MTCM": "/data/global_mapping/models_V100/1km_traditional_bilinear_no_forestslip/MTCM.tif",
    "MTDQ": "/data/global_mapping/models_V100/1km_traditional_bilinear_no_forestslip/MTDQ.tif",

    # Precipitation
    "MAP": "/data/global_mapping/models_V100/1km_traditional_bilinear_no_forestslip/MAP.tif",
    "PDM": "/data/global_mapping/models_V100/1km_traditional_bilinear_no_forestslip/PDM.tif",
    "PS": "/data/global_mapping/models_V100/1km_traditional_bilinear_no_forestslip/PS.tif",

    # Topography
    "ELEV": "/data/global_mapping/models_V100/1km_traditional_bilinear_no_forestslip/ELEV.tif",
    "TWI": "/data/global_mapping/models_V100/1km_traditional_bilinear_no_forestslip/TWI.tif",

    # Biophysical conditions
    "VPD": "/data/global_mapping/models_V100/1km_traditional_bilinear_no_forestslip/VPD.tif",
    "SWVL3": "/data/global_mapping/models_V100/1km_traditional_bilinear_no_forestslip/SWVL3.tif",

    # Soil properties
    "TN": "/data/global_mapping/models_V100/1km_traditional_bilinear_no_forestslip/TN.tif",
    "PH": "/data/global_mapping/models_V100/1km_traditional_bilinear_no_forestslip/PH.tif",
    "CLAY": "/data/global_mapping/models_V100/1km_traditional_bilinear_no_forestslip/CLAY.tif"
}

gdf = gpd.read_file(shapefile_path)

for var_name, raster_path in selected_rasters.items():
    if not os.path.exists(raster_path):
        print(f"Warning: File not found, skipping {raster_path}")
        continue
        
    output_path = os.path.join(output_dir, f"{var_name}.tif")
    
    with rasterio.open(raster_path) as src:
        if gdf.crs != src.crs:
            gdf_proj = gdf.to_crs(src.crs)
        else:
            gdf_proj = gdf
            
        geometries = [mapping(geom) for geom in gdf_proj.geometry]
        
        try:
            out_image, out_transform = mask(src, geometries, crop=True)
            out_meta = src.meta.copy()
            
            out_meta.update({
                "driver": "GTiff",
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform,
                "compress": "lzw" 
            })
            
            with rasterio.open(output_path, "w", **out_meta) as dest:
                dest.write(out_image)
                
            print(f"Success: Processed and saved {output_path}")
            
        except Exception as e:
            print(f"Error processing {var_name}: {str(e)}")