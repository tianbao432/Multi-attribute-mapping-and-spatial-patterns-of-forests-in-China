import os
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

# ================= 1. Configure File Paths =================
CSV_PATH = "/data/China_mapping/foerst_plot/PLOT_Data.csv"
SHP_PATH = "/data/China_mapping/climate_zone/I_clmate.shp"
OUTPUT_CSV_PATH = "/data/China_mapping/foerst_plot/PLOT_Data_Climate.csv"

# Check if input files exist
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"CSV file not found: {CSV_PATH}")
if not os.path.exists(SHP_PATH):
    raise FileNotFoundError(f"Shapefile not found: {SHP_PATH}")

print("Loading data...")

# ================= 2. Load Data and Construct Geometry Objects =================
# Read CSV file
df = pd.read_csv(CSV_PATH)

# Ensure necessary coordinate columns exist in the dataframe
if 'Longitude' not in df.columns or 'Latitude' not in df.columns:
    raise ValueError("The CSV file must contain 'Longitude' and 'Latitude' columns.")

# Create a collection of Shapely Point objects using longitude and latitude
geometry = [Point(xy) for xy in zip(df['Longitude'], df['Latitude'])]

# Convert the standard pandas DataFrame to a GeoDataFrame with spatial information
gdf_points = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

# Read the climate zone Shapefile
print("Loading vector climate zone data...")
gdf_climate = gpd.read_file(SHP_PATH)

# ================= 3. Unify Coordinate Systems and Spatial Join =================
if gdf_points.crs != gdf_climate.crs:
    print(f"CRS mismatch! Converting climate zone CRS ({gdf_climate.crs}) to point data CRS ({gdf_points.crs})...")
    gdf_climate = gdf_climate.to_crs(gdf_points.crs)

print("Executing spatial join to extract climate attributes...")
# Execute Spatial Join
joined_gdf = gpd.sjoin(gdf_points, gdf_climate[['geometry', 'name']], how="left", predicate="intersects")

# ================= 4. Map Climate Zones to English Abbreviations =================
print("Mapping climate zones to standard English abbreviations...")

# Dictionary for Standard English Abbreviations
abbr_mapping = {
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

# Dictionary for Full English Names (Optional but good for metadata)
eng_mapping = {
    'H': 'Highland',
    'I': 'Cold Temperate',
    'II': 'Mid-Temperate',
    'III': 'Warm Temperate',
    'IV': 'North Subtropical',
    'V': 'Mid-Subtropical',
    'VI': 'South Subtropical',
    'VII': 'Marginal Tropical',
    'VIII': 'Mid-Tropical'
}

# Create new columns based on the original 'name' column
# .map() will match the keys in the dictionary and return the corresponding values
joined_gdf['climate_abbr'] = joined_gdf['name'].map(abbr_mapping)
joined_gdf['climate_eng'] = joined_gdf['name'].map(eng_mapping)

# ================= 5. Clean up and Export Data =================
# Drop the internal helper column 'index_right' if it exists
if 'index_right' in joined_gdf.columns:
    joined_gdf = joined_gdf.drop(columns=['index_right'])

# Drop the geometry column, revert to standard dataframe
final_df = pd.DataFrame(joined_gdf.drop(columns=['geometry']))

# Rename 'name' column to 'climate_code' for better clarity in the final CSV (Optional)
final_df = final_df.rename(columns={'name': 'climate_code'})

# Export to a new file in the original directory
final_df.to_csv(OUTPUT_CSV_PATH, index=False)

print("\n" + "="*60)
print("Processing successful! Climate attributes added to dataset.")
print("Added columns: 'climate_code', 'climate_abbr', 'climate_eng'.")
print(f"New file saved to: {OUTPUT_CSV_PATH}")
print("="*60)