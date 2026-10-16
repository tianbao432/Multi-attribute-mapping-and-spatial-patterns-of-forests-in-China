# -*- coding: utf-8 -*-
"""
Update Climate Zone Shapefile
Reads the climate zone vector file, repairs invalid geometries, 
generates standard abbreviations and full English names, 
reprojects to WGS 84, and saves to a new file.
"""
import os
import logging
import geopandas as gpd

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("Vector_Updater")

def update_climate_shapefile(input_path: str, output_path: str):
    if not os.path.exists(input_path):
        logger.error(f"Input file not found: {input_path}")
        return

    logger.info(f"Reading vector file: {input_path}")
    gdf = gpd.read_file(input_path)

    # Check for required column
    if 'name' not in gdf.columns:
        logger.error("The 'name' column was not found in the attribute table!")
        return

    # Define mapping dictionary as ID
    ID = {
        'H': 'H',        # Highland
        'I': 'CT',       # Cold Temperate
        'II': 'MT',      # Mid-Temperate
        'III': 'WT',     # Warm Temperate
        'IV': 'NS',      # North Subtropical
        'V': 'MS',       # Mid-Subtropical
        'VI': 'SS',      # South Subtropical
        'VII': 'MTrop',  # Marginal Tropical
        'VIII': 'MidTr'  # Mid-Tropical
    }

    eng_mapping = {
        'H': 'Highland',
        'I': 'ColdTemp',
        'II': 'MidTemp',
        'III': 'WarmTemp',
        'IV': 'NorthSub',
        'V': 'MidSub',
        'VI': 'SouthSub',
        'VII': 'MargTrop',
        'VIII': 'MidTrop'
    }

    logger.info("Appending 'clim_abbr' and 'clim_full' columns...")
    
    # Clean whitespace and map columns
    gdf['name'] = gdf['name'].str.strip()
    gdf['clim_abbr'] = gdf['name'].map(ID)
    gdf['clim_full'] = gdf['name'].map(eng_mapping)

    # Repair invalid geometries (Critical for avoiding Cutline polygon errors)
    logger.info("Repairing invalid geometries...")
    gdf['geometry'] = gdf['geometry'].buffer(0)

    # Reproject to WGS 84 (EPSG:4326)
    logger.info("Reprojecting to WGS 84 (EPSG:4326)...")
    gdf = gdf.to_crs(epsg=4326)

    # Diagnostic check for unmapped values
    unmapped_records = gdf[gdf['clim_abbr'].isna()]['name'].unique()
    if len(unmapped_records) > 0:
        logger.warning(f"Unmapped 'name' values detected: {unmapped_records}. These will be NULL.")

    logger.info(f"Saving updated data to: {output_path}")
    try:
        # Save to a new path
        gdf.to_file(output_path, encoding='utf-8')
        logger.info("Update and export completed successfully.")
    except Exception as e:
        logger.error(f"Export failed: {e}")

if __name__ == "__main__":
    # Define input and output paths
    INPUT_PATH = "/data/China_mapping/climate_zone/I_clmate.shp"
    OUTPUT_PATH = "/data/China_mapping/climate_zone/clmate_zone.shp"
    
    update_climate_shapefile(INPUT_PATH, OUTPUT_PATH)