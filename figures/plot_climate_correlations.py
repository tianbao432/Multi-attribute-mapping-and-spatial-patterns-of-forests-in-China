import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import Normalize

# ==========================================
# 1. Configuration & Top-Tier Styling
# ==========================================
INPUT_DIR = "/data/China_mapping/Climate_statistics/"
OUTPUT_DIR = os.path.join(INPUT_DIR, "Figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# High-impact academic journal styling (Clean sans-serif, NO bold fonts)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['svg.fonttype'] = 'none'

# Optimal font sizes for clarity without bold weight
plt.rcParams['font.size'] = 14
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['xtick.labelsize'] =12
plt.rcParams['ytick.labelsize'] = 12

# Balanced axes lines for modern aesthetic
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['xtick.major.width'] = 1.5
plt.rcParams['ytick.major.width'] = 1.5
plt.rcParams['xtick.major.size'] = 5.0
plt.rcParams['ytick.major.size'] = 5.0

def save_high_res_plot(filepath_without_ext):
    """Saves plots in ultra-high resolution PNG and TIF formats."""
    plt.savefig(filepath_without_ext + ".png", dpi=600, bbox_inches='tight', transparent=False)
    plt.savefig(filepath_without_ext + ".tif", dpi=600, bbox_inches='tight', transparent=False, format='tiff')

# ==========================================
# 2. National Correlation Heatmap (Figure 1)
# ==========================================
def generate_national_correlation_heatmap(data_dir, output_dir):
    """
    Generates a single, clean national correlation heatmap without titles or subplots.
    """
    national_file = os.path.join(data_dir, "National_correlation.csv")
    if not os.path.exists(national_file):
        print("National correlation file not found.")
        return

    df_corr = pd.read_csv(national_file, index_col=0)
    df_corr.rename(columns={'Trees_N': 'Density'}, index={'Trees_N': 'Density'}, inplace=True)
    df_corr.rename(columns={'V': 'GSV'}, index={'V': 'GSV'}, inplace=True)
    df_corr.rename(columns={'H': 'Height'}, index={'H': 'Height'}, inplace=True)
    # Lower triangle masking
    df_plot = df_corr.iloc[1:, :-1]
    full_mask = np.triu(np.ones_like(df_corr, dtype=bool))
    mask_plot = full_mask[1:, :-1]
    
    fig, ax = plt.subplots(figsize=(6, 6))
    cmap = plt.colormaps['viridis'].copy()
    
    sns.heatmap(df_plot, mask=mask_plot, cmap=cmap, vmax=1.0, vmin=-1.0, center=0,
                square=True, linewidths=1.5, linecolor='white', cbar=False,
                annot=True, fmt=".2f", annot_kws={"size": 13, "weight": "normal"}, ax=ax)
    
    # Strictly no title, no subplots, clean appearance
    ax.tick_params(axis='x', rotation=45)
    ax.tick_params(axis='y', rotation=0)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "Figure_National_Correlation")
    save_high_res_plot(out_path)
    plt.close()
    print(f"National correlation heatmap saved: {out_path}")

# ==========================================
# 3. 9-Zone Gridded Correlation Heatmaps (Figure 2)
# ==========================================
def generate_gridded_heatmaps(data_dir, output_dir):
    """
    Combines regional correlation matrices into a single multi-panel figure 
    with a shared colorbar, without any bold fonts or subplot letters.
    """
    corr_files = sorted(glob.glob(os.path.join(data_dir, "*_correlation.csv")))
    # Exclude National if present to focus purely on the 9 regional groups
    corr_files = [f for f in corr_files if "National" not in os.path.basename(f)]
    
    if not corr_files:
        print("No regional correlation files found.")
        return

    n_files = len(corr_files)
    cols = 3
    rows = int(np.ceil(n_files / cols))
    
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    axes = axes.flatten()
    
    cmap ="YlGnBu_r"
    
    for i, file_path in enumerate(corr_files):
        filename = os.path.basename(file_path)
        zone_name = filename.replace("_correlation.csv", "")
        
        df_corr = pd.read_csv(file_path, index_col=0)
        df_corr.rename(columns={'Trees_N': 'Density'}, index={'Trees_N': 'Density'}, inplace=True)
        df_corr.rename(columns={'V': 'GSV'}, index={'V': 'GSV'}, inplace=True)
        df_corr.rename(columns={'H': 'Height'}, index={'H': 'Height'}, inplace=True)
        
        df_plot = df_corr.iloc[1:, :-1]
        full_mask = np.triu(np.ones_like(df_corr, dtype=bool))
        mask_plot = full_mask[1:, :-1]
        
        ax = axes[i]
        
        sns.heatmap(df_plot, mask=mask_plot, cmap=cmap, vmax=1.0, vmin=-1.0, center=0,
                    square=True, linewidths=1.2, linecolor='white', cbar=False,
                    annot=True, fmt=".2f", annot_kws={"size": 8, "weight": "normal"}, ax=ax)
        
        # Clean zone title without bolding
        ax.set_title(zone_name, pad=12, fontweight='normal', fontsize=14)
        ax.tick_params(axis='x', rotation=45)
        ax.tick_params(axis='y', rotation=0)

    # Remove empty subplots if any
    for i in range(n_files, len(axes)):
        fig.delaxes(axes[i])
        
    plt.tight_layout()
    
    out_path = os.path.join(output_dir, "Figure_Regional_Correlation_Grid")
    save_high_res_plot(out_path)
    plt.close()
    print(f"Regional gridded correlation heatmaps saved: {out_path}")

# ==========================================
# 4. Main Execution
# ==========================================
if __name__ == '__main__':
    print("Initializing correlation visualizations...")
    
    generate_national_correlation_heatmap(INPUT_DIR, OUTPUT_DIR)
    generate_gridded_heatmaps(INPUT_DIR, OUTPUT_DIR)
    
    print("Correlation figures successfully generated.")