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

# High-impact academic journal styling
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['svg.fonttype'] = 'none'

# Optimal font sizes for scaled-down readability in papers
plt.rcParams['font.size'] = 16
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16

# Bold axes lines for "striking" visual weight
plt.rcParams['axes.linewidth'] = 2.0
plt.rcParams['xtick.major.width'] = 2.0
plt.rcParams['ytick.major.width'] = 2.0
plt.rcParams['xtick.major.size'] = 6.0
plt.rcParams['ytick.major.size'] = 6.0

def save_high_res_plot(filepath_without_ext):
    """Saves plots in ultra-high resolution PNG and TIF formats."""
    plt.savefig(filepath_without_ext + ".png", dpi=600, bbox_inches='tight', transparent=False)
    plt.savefig(filepath_without_ext + ".tif", dpi=600, bbox_inches='tight', transparent=False, format='tiff')

# ==========================================
# 2. Correlation Visualizations
# ==========================================
def generate_summary_target_heatmap(data_dir, output_dir, target_var="Density"):
    """
    Creates a single, highly condensed heatmap showing how the target_var 
    correlates with all other variables across all spatial zones.
    """
    corr_files = glob.glob(os.path.join(data_dir, "*_correlation.csv"))
    if not corr_files:
        print("No correlation files found for summary.")
        return

    summary_data = {}
    
    for file_path in corr_files:
        filename = os.path.basename(file_path)
        raw_zone = filename.replace("_correlation.csv", "")
        zone_name = "National" if raw_zone == "National" else raw_zone
        
        df_corr = pd.read_csv(file_path, index_col=0)
        df_corr.rename(columns={'Trees_N': 'Density'}, index={'Trees_N': 'Density'}, inplace=True)
        
        if target_var in df_corr.index:
            correlations = df_corr.loc[target_var].drop(target_var)
            summary_data[zone_name] = correlations
            
    df_summary = pd.DataFrame(summary_data).T
    
    # Ensure 'National' is placed at the top for reference
    if "National" in df_summary.index:
        zones = [z for z in df_summary.index if z != "National"]
        df_summary = df_summary.reindex(["National"] + sorted(zones))

    fig, ax = plt.subplots(figsize=(12, max(5, len(df_summary) * 0.8)))
    
    # Diverging colormap for correlations (-1 to 1)
    cmap = "RdBu_r"
    
    sns.heatmap(df_summary, cmap=cmap, vmax=1.0, vmin=-1.0, center=0,
                square=True, linewidths=2.0, linecolor='white', 
                cbar_kws={"shrink": 0.8, "label": "Correlation Coefficient"},
                annot=True, fmt=".2f", annot_kws={"size": 14, "weight": "bold"}, ax=ax)
    
    ax.set_title(f"Correlation Drivers of {target_var} Across Regions", pad=20, fontweight='bold')
    ax.tick_params(axis='x', rotation=45)
    ax.tick_params(axis='y', rotation=0)
    
    out_path = os.path.join(output_dir, f"Figure_Summary_{target_var}_Correlation")
    save_high_res_plot(out_path)
    plt.close()
    print(f"Summary heatmap saved: {out_path}")

def generate_gridded_heatmaps(data_dir, output_dir):
    """
    Combines all lower-triangle correlation matrices into a single multi-panel 
    figure with one shared colorbar to reduce redundancy and save space.
    """
    corr_files = sorted(glob.glob(os.path.join(data_dir, "*_correlation.csv")))
    if not corr_files:
        print("No correlation files found for gridded plot.")
        return

    n_files = len(corr_files)
    cols = 3
    rows = int(np.ceil(n_files / cols))
    
    fig, axes = plt.subplots(rows, cols, figsize=(8 * cols, 7 * rows))
    axes = axes.flatten()
    
    cmap = "RdBu_r"
    
    for i, file_path in enumerate(corr_files):
        filename = os.path.basename(file_path)
        raw_zone = filename.replace("_correlation.csv", "")
        zone_name = "National" if raw_zone == "National" else raw_zone
        
        df_corr = pd.read_csv(file_path, index_col=0)
        df_corr.rename(columns={'Trees_N': 'Density'}, index={'Trees_N': 'Density'}, inplace=True)
        
        # Lower triangle masking
        df_plot = df_corr.iloc[1:, :-1]
        full_mask = np.triu(np.ones_like(df_corr, dtype=bool))
        mask_plot = full_mask[1:, :-1]
        
        ax = axes[i]
        
        sns.heatmap(df_plot, mask=mask_plot, cmap=cmap, vmax=1.0, vmin=-1.0, center=0,
                    square=True, linewidths=1.5, linecolor='white', cbar=False,
                    annot=True, fmt=".2f", annot_kws={"size": 13, "weight": "bold"}, ax=ax)
        
        ax.set_title(zone_name, pad=15, fontweight='bold', fontsize=22)
        
        # Subplot letter notation
        letter = chr(97 + i)
        ax.text(-0.05, 1.05, letter, transform=ax.transAxes,
                fontsize=24, fontweight='bold', va='bottom', ha='right')

    # Remove empty subplots
    for i in range(n_files, len(axes)):
        fig.delaxes(axes[i])
        
    plt.tight_layout()
    
    # Add a single shared colorbar on the right
    plt.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.93, 0.15, 0.02, 0.7]) 
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=Normalize(vmin=-1, vmax=1))
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Correlation Coefficient', rotation=270, labelpad=25, fontweight='bold')
    
    out_path = os.path.join(output_dir, "Figure_All_Zones_Correlation_Grid")
    save_high_res_plot(out_path)
    plt.close()
    print(f"Gridded heatmaps saved: {out_path}")

# ==========================================
# 3. Boxplot Function
# ==========================================
def generate_comparative_boxplots(data_dir, output_dir):
    """
    Generates a consolidated grid of boxplots for all variables.
    Applies a uniform, aesthetic color to all boxes to maintain academic elegance.
    """
    stats_files = glob.glob(os.path.join(data_dir, "*_boxplot_stats.csv"))
    if not stats_files:
        print("No boxplot stats files found.")
        return

    zones = []
    df_dict = {}
    for f in stats_files:
        raw_zone = os.path.basename(f).replace("_boxplot_stats.csv", "")
        zone_name = "All" if raw_zone == "National" else raw_zone
        zones.append(zone_name)
        
        df = pd.read_csv(f, index_col=0)
        df.rename(columns={'Trees_N': 'Density'}, inplace=True)
        df_dict[zone_name] = df

    if "All" in zones:
        zones.remove("All")
        zones = ["All"] + sorted(zones)
    else:
        zones = sorted(zones)

    variables = df_dict[zones[0]].columns.tolist()
    n_vars = len(variables)
    
    cols = 3
    rows = int(np.ceil(n_vars / cols))
    
    fig, axes = plt.subplots(rows, cols, figsize=(8 * cols, 7 * rows))
    axes = axes.flatten()

    # Single aesthetic color for all boxes (Muted Academic Teal/Blue)
    box_color = '#5A94B5' 
    
    for i, var in enumerate(variables):
        ax = axes[i]
        bxp_stats = []
        
        for j, zone in enumerate(zones):
            df = df_dict[zone]
            if var in df.columns:
                bxp_stats.append({
                    'label': zone,
                    'med': df.loc['50%', var],
                    'q1': df.loc['25%', var],
                    'q3': df.loc['75%', var],
                    'whislo': df.loc['min', var],
                    'whishi': df.loc['max', var]
                })
        
        bplot = ax.bxp(bxp_stats, showfliers=False, patch_artist=True, widths=0.6)
        
        # Apply uniform styling to all boxes
        for patch in bplot['boxes']:
            patch.set_facecolor(box_color)
            patch.set_alpha(0.85)
            patch.set_edgecolor('black')
            patch.set_linewidth(2.0)
            
        for median in bplot['medians']:
            median.set_color('#D73027') # Striking red for contrast
            median.set_linewidth(3.5)
            
        for whisker in bplot['whiskers']:
            whisker.set_color('black')
            whisker.set_linewidth(2.0)
            whisker.set_linestyle('-')
            
        for cap in bplot['caps']:
            cap.set_color('black')
            cap.set_linewidth(2.0)

        # Aesthetics
        ax.set_title(var, pad=15, fontweight='bold', fontsize=22)
        ax.grid(axis='y', linestyle='--', alpha=0.6, linewidth=1.2)
        ax.set_xticklabels([stat['label'] for stat in bxp_stats], rotation=45, ha='right')
        
        # Clean frame limits
        sns.despine(ax=ax, trim=False)
        ax.spines['bottom'].set_linewidth(2.0)
        ax.spines['left'].set_linewidth(2.0)
        
        # Subplot letters
        letter = chr(97 + i)
        ax.text(-0.10, 1.05, letter, transform=ax.transAxes,
                fontsize=26, fontweight='bold', va='bottom', ha='right')

    # Remove unused axes
    for i in range(n_vars, len(axes)):
        fig.delaxes(axes[i])

    plt.subplots_adjust(hspace=0.6, wspace=0.3)
    out_path = os.path.join(output_dir, "Figure_Variables_Boxplots_Unified")
    save_high_res_plot(out_path)
    plt.close()
    print(f"Comparative boxplots saved: {out_path}")

# ==========================================
# 4. Main Execution
# ==========================================
if __name__ == '__main__':
    print("Initializing top-tier visualizations...")
    
    # 1. Generate the highly condensed summary heatmap
    generate_summary_target_heatmap(INPUT_DIR, OUTPUT_DIR, target_var="Density")
    
    # 2. Generate the full gridded heatmaps with shared colorbar
    generate_gridded_heatmaps(INPUT_DIR, OUTPUT_DIR)
    
    # 3. Generate the unified color boxplot grid
    generate_comparative_boxplots(INPUT_DIR, OUTPUT_DIR)
    
    print("Visualizations finished. High-DPI outputs generated successfully.")