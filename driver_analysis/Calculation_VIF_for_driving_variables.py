import os
import numpy as np
import pandas as pd
import rasterio
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
import warnings

# Suppress runtime warnings from statsmodels
warnings.filterwarnings("ignore", category=RuntimeWarning)

input_dir = "/data/China_mapping/driving_variables/"

# Updated to the 9 golden variables
var_names =  [
        "HMI",
        "MAT",
        "ISO",
        "MTCM",
        "MTDQ",
        "MAP",
        "PDM",
        "PS",
        "ELEV",
        "TWI",
        "VPD",
        "SWVL3",
        "TN",
        "PH",
        "CLAY",
    ]

# Changed reference raster from MAT to MAP since MAT is removed
ref_raster_path = os.path.join(input_dir, "MAP.tif")

with rasterio.open(ref_raster_path) as src:
    data = src.read(1)
    nodata = src.nodata
    
    if nodata is not None and np.isnan(nodata):
        valid_mask = ~np.isnan(data)
    elif nodata is not None:
        valid_mask = (data != nodata)
    else:
        valid_mask = ~np.isnan(data)
        
    valid_rows, valid_cols = np.where(valid_mask)
    
    np.random.seed(42)
    sample_size = min(10000, len(valid_rows))
    sample_indices = np.random.choice(len(valid_rows), size=sample_size, replace=False)
    
    sample_rows = valid_rows[sample_indices]
    sample_cols = valid_cols[sample_indices]
    
    coords = [src.xy(r, c) for r, c in zip(sample_rows, sample_cols)]

print(f"Generated {len(coords)} valid random points.")

df_data = {}
for var in var_names:
    raster_path = os.path.join(input_dir, f"{var}.tif")
    if not os.path.exists(raster_path):
        print(f"Warning: {raster_path} not found, skipping.")
        continue
        
    with rasterio.open(raster_path) as src:
        values = [val[0] for val in src.sample(coords)]
        df_data[var] = values

df = pd.DataFrame(df_data)

df.replace([np.inf, -np.inf], np.nan, inplace=True)
df = df.dropna()
print(f"Data shape after removing NaNs: {df.shape}")

def get_vif(dataframe):
    X = add_constant(dataframe)
    vif_df = pd.DataFrame()
    vif_df["variable"] = X.columns
    vif_df["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif_df[vif_df["variable"] != "const"].reset_index(drop=True)

print("\n--- Starting Stepwise VIF Selection ---")
current_vars = list(df.columns)

while len(current_vars) > 5:
    current_df = df[current_vars]
    vif_result = get_vif(current_df)
    
    max_vif_idx = vif_result['VIF'].idxmax()
    max_vif_val = vif_result.loc[max_vif_idx, 'VIF']
    max_vif_var = vif_result.loc[max_vif_idx, 'variable']
    
    if max_vif_val > 10.0:
        print(f"Removing {max_vif_var} (VIF: {max_vif_val:.2f})")
        current_vars.remove(max_vif_var)
    else:
        print("\nAll remaining variables have VIF <= 10.")
        break

print("\n--- Final Selected Variables ---")
final_vif = get_vif(df[current_vars])
final_vif = final_vif.sort_values(by="VIF", ascending=False).reset_index(drop=True)
print(final_vif.to_string(index=False))

output_csv = os.path.join(input_dir, "vif_selected_samples.csv")
df[current_vars].to_csv(output_csv, index=False)
print(f"\nFinal valid variables saved to {output_csv}")