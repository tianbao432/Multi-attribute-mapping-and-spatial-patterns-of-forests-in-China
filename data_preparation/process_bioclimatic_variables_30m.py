import os
import glob
import rasterio
from rasterio.warp import reproject, Resampling

input_dir = "/data/China_mapping/bio_climate/"
output_dir = "/data/China_mapping/variable/"
os.makedirs(output_dir, exist_ok=True)

target_res_deg = 30.0 / 111320.0 

for input_file in glob.glob(os.path.join(input_dir, "*.tif")):
    output_file = os.path.join(output_dir, os.path.basename(input_file))
    
    with rasterio.open(input_file) as src:
        if src.crs is None:
            print(f"Warning: {input_file} has no CRS. Assuming WGS84.")
        
        scale_x = src.res[0] / target_res_deg
        scale_y = src.res[1] / target_res_deg
        
        new_width = int(round(src.width * scale_x))
        new_height = int(round(src.height * scale_y))
        

        new_transform = src.transform * src.transform.scale(
            1.0 / scale_x, 1.0 / scale_y
        )
        
        profile = src.profile.copy()
        profile.update({
            "height": new_height,
            "width": new_width,
            "transform": new_transform,
            "crs": src.crs,           
            "compress": "lzw",
            "tiled": True,
            "blockxsize": 256,
            "blockysize": 256,
            "bigtiff": "yes"
        })
        
        with rasterio.open(output_file, "w", **profile) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=new_transform,
                    dst_crs=src.crs,      
                    resampling=Resampling.bilinear
                )
                
    print(f"Finished: {os.path.basename(input_file)} (WGS84 Preserved)")

print("All rasters completed.")