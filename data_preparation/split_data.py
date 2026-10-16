import pandas as pd
import numpy as np
import os
import chardet


def process_data_no_split():
    # ================= CONFIGURATION =================
    input_path ="/data/China_mapping/foerst_plot/PLOT_Data_Climate_Extracted_Fast.csv"
    output_root = "/data/China_mapping/forest_data/"

    # Target variables (using the new renamed names)
    target_columns = ["DBH", "H", "AGE", "BA", "Trees_N", "AGB", "V", "shannon", "CV_DBH"]

    # Grouping variable (categorical)
    group_col = "climate_abbr"

    # ================= 1. LOAD DATA =================
    print(f"Loading raw data: {input_path}")

    # Automatically detect file encoding to prevent read errors
    with open(input_path, "rb") as f:
        raw = f.read(1024 * 1024)
        enc = chardet.detect(raw)["encoding"] or "utf-8"

    try:
        df = pd.read_csv(input_path, encoding=enc, low_memory=False)
    except Exception as e:
        print(f"Failed with detected encoding, trying latin1: {e}")
        df = pd.read_csv(input_path, encoding="latin1", low_memory=False)

    print(f"Data loaded. Total rows: {len(df)}")

    # ================= 2. RENAME VARIABLES =================
    # Renaming specific columns as requested:
    # Shannon.Index -> shannon
    # N -> Trees_N
    rename_dict = {
        "Shannon.Index": "shannon",
        "N": "Trees_N",
    }
    df.rename(columns=rename_dict, inplace=True)
    print("Variables renamed: 'Shannon.Index' -> 'shannon', 'N' -> 'Trees_N'")

    # ================= 3. EXPLICIT FEATURE LIST =================
    # Predictors (environment + remote sensing) listed explicitly,
    # grouped by category for readability. Order does not matter.
    env_features = [
    "B11_Autumn", "B11_Spring", "B11_Summer", "B11_Winter",
    "B2_Autumn", "B2_Spring", "B2_Summer", "B2_Winter",
    "B3_Autumn", "B3_Spring", "B3_Summer", "B3_Winter",
    "B4_Autumn", "B4_Spring", "B4_Summer", "B4_Winter",
    "B5_Autumn", "B5_Spring", "B5_Summer", "B5_Winter",
    "B6_Autumn", "B6_Spring", "B6_Summer", "B6_Winter",
    "B8_Autumn", "B8_Spring", "B8_Summer", "B8_Winter",
    "BSI_Autumn", "BSI_Spring", "BSI_Summer", "BSI_Winter",
    "EVI_Autumn", "EVI_Spring", "EVI_Summer", "EVI_Winter",
    "GLI_Autumn", "GLI_Spring", "GLI_Summer", "GLI_Winter",
    "GNDVI_Autumn", "GNDVI_Spring", "GNDVI_Summer", "GNDVI_Winter",
    "ISO", 
    "LSWI_Autumn", "LSWI_Spring", "LSWI_Summer", "LSWI_Winter",
    "MAP", "MAT", "MDR",
    "MSAVI2_Autumn", "MSAVI2_Spring", "MSAVI2_Summer", "MSAVI2_Winter",
    "MTCM", "MTCQ", "MTDQ", "MTWM", "MTWaQ", "MTWeQ",
    "NDRE1_Autumn", "NDRE1_Spring", "NDRE1_Summer", "NDRE1_Winter",
    "NDRE2_Autumn", "NDRE2_Spring", "NDRE2_Summer", "NDRE2_Winter",
    "NDVI_Autumn", "NDVI_Spring", "NDVI_Summer", "NDVI_Winter",
    "PCQ", "PDM", "PDQ", "PS", "PWaQ", "PWM", "PWeQ",
    "SAVI_Autumn", "SAVI_Spring", "SAVI_Summer", "SAVI_Winter",
    "TAR", "TS",
    "VH_Autumn", "VH_Spring", "VH_Summer", "VH_Winter",
    "VV_Autumn", "VV_Spring", "VV_Summer", "VV_Winter",
    "asm_Autumn", "asm_Spring", "asm_Summer", "asm_Winter",
    "contrast_Autumn", "contrast_Spring", "contrast_Summer", "contrast_Winter",
    "correlation_Autumn", "correlation_Spring", "correlation_Summer", "correlation_Winter",
    "dissimilarity_Autumn", "dissimilarity_Spring", "dissimilarity_Summer", "dissimilarity_Winter",
    "entropy_Autumn", "entropy_Spring", "entropy_Summer", "entropy_Winter",
    "homogeneity_Autumn", "homogeneity_Spring", "homogeneity_Summer", "homogeneity_Winter",
    "mean_Autumn", "mean_Spring", "mean_Summer", "mean_Winter",
    "variance_Autumn", "variance_Spring", "variance_Summer", "variance_Winter","slop","DEM","aspect",
    ]
    assert len(env_features) == 130

    # Coordinates are kept alongside the predictors
    feature_columns = env_features + ["Latitude", "Longitude"]

    # Sanity check: warn about any predictor missing from the CSV
    missing = [c for c in feature_columns if c not in df.columns]
    if missing:
        print(f"[WARN] {len(missing)} predictor(s) not found in CSV and will be skipped: {missing}")
    feature_columns = [c for c in feature_columns if c in df.columns]

    print(f"Using {len(feature_columns)} predictors (including Latitude/Longitude).")

    # ================= 4. PROCESS LOOP =================
    for target in target_columns:
        # Skip if target variable is not found in the dataframe
        if target not in df.columns:
            print(f"\n[SKIP] Target '{target}' not found in CSV columns.")
            continue

        print(f"\n>>> Processing Target: {target}")

        # Identify unique BIOMEs (dropping NAs to avoid errors)
        unique_biomes = df[group_col].dropna().unique()

        for biome_val in unique_biomes:
            # --- Format Folder Name (standardize integer IDs) ---
            try:
                if float(biome_val).is_integer():
                    biome_str = str(int(biome_val))
                else:
                    biome_str = str(biome_val)
            except (ValueError, TypeError):
                biome_str = str(biome_val)

            biome_folder_name = f"{biome_str}"

            # 1. Filter data for the specific current BIOME
            biome_raw_df = df[df[group_col] == biome_val]

            # 2. Select required columns (Target + All Predictors)
            cols_to_keep = [target] + feature_columns

            # Ensure only columns that actually exist in the dataframe are used
            available_cols = [c for c in cols_to_keep if c in biome_raw_df.columns]
            subset_df = biome_raw_df[available_cols]

            # 3. Drop NA rows
            # Rows must have valid data for the target AND all environmental predictors
            clean_df = subset_df.dropna()

            # Logging statistics for the current BIOME
            original_count = len(subset_df)
            clean_count = len(clean_df)
            dropped_count = original_count - clean_count

            # Skip output generation if no valid data rows exist
            if clean_count < 1:
                continue

            # 4. Save Data (Organized by Target and BIOME)
            # Path structure: root / TargetName / BIOME_X / data.csv
            save_dir = os.path.join(output_root, target, biome_folder_name)
            os.makedirs(save_dir, exist_ok=True)

            output_file = os.path.join(save_dir, "data.csv")
            clean_df.to_csv(output_file, index=False)

            print(f"   - {biome_folder_name:<10}: Raw {original_count} -> Clean {clean_count} (Dropped {dropped_count})")

    print("\n" + "=" * 50)
    print("DONE! Process completed.")
    print(f"Output Path: {output_root}")
    print("=" * 50)


if __name__ == "__main__":
    process_data_no_split()