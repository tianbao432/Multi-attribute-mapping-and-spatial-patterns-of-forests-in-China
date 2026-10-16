import os
import logging
import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.cluster import KMeans
import warnings

# Suppress warnings for cleaner logs
warnings.filterwarnings("ignore")

CONFIG = {
    "base_dir": "/data/China_mapping/",
    "output_dir": "/data/China_mapping/fina_modelA/",
    "target_vars": ["AGE", "BA", "Trees_N", "AGB", "V", "shannon", "CV_DBH", "DBH", "H"],
    "Climate_Zone": ["WT", "MS", "MT", "SS", "NS", "CT", "H", "MTrop", "MidTr"],
    "k_folds": 10,
    "seed": 54321
}

# Setup logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("CV_Data_Prep")

# Set global seed for reproducibility
np.random.seed(CONFIG["seed"])

def main():
    os.makedirs(CONFIG["output_dir"], exist_ok=True)

    # Reversing the loop order to group by target_var first, matching the requested output structure
    for t_var in CONFIG["target_vars"]:
        logger.info(f"========== Processing Target Variable: {t_var} ==========")
        
        for b_id in CONFIG["Climate_Zone"]:
            input_path = os.path.join(CONFIG["base_dir"], "forest_data", t_var, f"{b_id}", "data.csv")
            
            # Define specific output directory for this parameter and climate zone
            target_out_dir = os.path.join(CONFIG["output_dir"], t_var, f"{b_id}")
            
            if not os.path.exists(input_path):
                logger.warning(f"File not found: {input_path}")
                continue
            
            try:
                df_raw = pd.read_csv(input_path, low_memory=False)
            except Exception as e:
                logger.error(f"Error reading {input_path}: {e}")
                continue
            
            # Modification 1: Rename climate_abbr to Climate_Zone
            if "climate_abbr" in df_raw.columns:
                df_raw = df_raw.rename(columns={"climate_abbr": "Climate_Zone"})
            
            # Fallback: if Climate_Zone is missing, fill it with the loop variable
            if "Climate_Zone" not in df_raw.columns:
                df_raw["Climate_Zone"] = b_id
                
            # Drop rows missing coordinates or target variable
            df = df_raw.dropna(subset=["Longitude", "Latitude", t_var]).copy()
            
            if len(df) < (CONFIG["k_folds"] * 5):
                logger.warning(f"Insufficient data for: {t_var} in Biome {b_id}")
                continue

            # Spatial projection logic
            gdf = gpd.GeoDataFrame(
                df, 
                geometry=gpd.points_from_xy(df["Longitude"], df["Latitude"]),
                crs="EPSG:4326"
            )
            
            lon_range = df["Longitude"].max() - df["Longitude"].min()
            
            if lon_range > 10:
                target_crs = "+proj=moll +lon_0=0 +datum=WGS84 +units=m +no_defs"
            else:
                zone = int(np.floor((df["Longitude"].mean() + 180) / 6) + 1)
                target_crs = f"+proj=utm +zone={zone} +datum=WGS84 +units=m +no_defs"
                
            gdf_proj = gdf.to_crs(target_crs)
            coords_mat = np.column_stack((gdf_proj.geometry.x, gdf_proj.geometry.y))
            
            # Generate Random Folds
            n_samples = len(df)
            fold_ids = np.tile(np.arange(1, CONFIG["k_folds"] + 1), int(np.ceil(n_samples / CONFIG["k_folds"])))
            df["fold_id_random"] = np.random.permutation(fold_ids)[:n_samples]
            
            # Generate K-Means Spatial Folds
            kmeans = KMeans(
                n_clusters=CONFIG["k_folds"], 
                n_init=50, 
                max_iter=1000, 
                random_state=CONFIG["seed"],
                algorithm="lloyd"
            )
            # Add 1 to match R's 1-based indexing for folds
            df["fold_id_kmeans"] = kmeans.fit_predict(coords_mat) + 1
            
            # Clean up ALL coordinate columns before saving
            # This drops geopandas geometry, and the original Longitude/Latitude
            columns_to_drop = ["geometry", "Longitude", "Latitude"]
            df = df.drop(columns=[col for col in columns_to_drop if col in df.columns], errors="ignore")
            
            # Ensure the specific output directory exists
            os.makedirs(target_out_dir, exist_ok=True)
            
            # Save the file in the specific parameter/biome folder
            output_csv = os.path.join(target_out_dir, "data_with_all_folds.csv")
            df.to_csv(output_csv, index=False)
            
            logger.info(f"Successfully processed and saved (No Coordinates): {t_var} in Biome {b_id} -> {output_csv}")

if __name__ == "__main__":
    main()