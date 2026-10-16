import os
import gc
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy import stats

import geopandas as gpd
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.mpl.geoaxes
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# Ignore environmental warnings
warnings.filterwarnings("ignore", category=UserWarning)

# ==========================================
# 1. Premium Journal Configuration
# ==========================================
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 13,
    'axes.labelsize': 14,
    'axes.titlesize': 14, 
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'axes.linewidth': 1.0,
    'xtick.major.width': 1.0,
    'ytick.major.width': 1.0,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'axes.facecolor': 'white',
    'figure.facecolor': 'white'
})

# Paths
DATA_PATH = "/data/China_mapping/validation_plot/Validation_Plot_Data_Extracted.csv"
CHINA_SHP = "/data/China_mapping/forest_boundary/china.shp"
NINEDASH_SHP = "/data/China_mapping/forest_boundary/JDX.shp"
OUTPUT_DIR = "/data/China_mapping/validation_plot/figures/"

os.makedirs(OUTPUT_DIR, exist_ok=True)

BIAS_MAP_PNG = os.path.join(OUTPUT_DIR, "Spatial_Bias_Maps.png")
BIAS_MAP_PDF = os.path.join(OUTPUT_DIR, "Spatial_Bias_Maps.pdf")
BIAS_MAP_TIF = os.path.join(OUTPUT_DIR, "Spatial_Bias_Maps.tif")

VARIABLES = [
    'AGB', 'AGE', 'BA', 'CV_DBH', 'DBH', 'H', 'N', 'Shannon.Index', 'V'
]

METADATA = {
    'AGB': {'title': 'AGB', 'unit': 'Mg ha$^{-1}$'},
    'AGE': {'title': 'Age', 'unit': 'years'},
    'BA': {'title': 'BA', 'unit': 'm$^{2}$ ha$^{-1}$'},
    'CV_DBH': {'title': 'CV DBH', 'unit': ''},
    'DBH': {'title': 'DBH', 'unit': 'cm'},
    'H': {'title': 'Height', 'unit': 'm'},
    'N': {'title': 'Density', 'unit': 'stems ha$^{-1}$'},
    'Shannon.Index': {'title': 'Shannon', 'unit': ''},
    'V': {'title': 'GSV', 'unit': 'm$^{3}$ ha$^{-1}$'}
}

def clean_formatter(x, pos=None):
    if x == 0:
        return "0"
    return f"{x:g}"

# ==========================================
# 2. High-Resolution Bias Maps
# ==========================================
def create_bias_maps(df):
    print("Reading shapefiles...")
    try:
        china_boundary = gpd.read_file(CHINA_SHP)
        ninedash_line = gpd.read_file(NINEDASH_SHP)
    except Exception as e:
        print(f"Shapefile reading error: {e}")
        return

    # Standard China Projection
    plot_proj = ccrs.AlbersEqualArea(central_longitude=105, standard_parallels=(25, 47))
    point_proj = ccrs.PlateCarree()

    print("Generating High-Resolution Bias Maps...")
    
    # Large figure size to ensure crisp details
    fig = plt.figure(figsize=(14, 12), dpi=300) 
    
    GRID_RES = 0.25
    lon_bins = np.arange(73, 136 + GRID_RES, GRID_RES)
    lat_bins = np.arange(15, 55 + GRID_RES, GRID_RES)
    
    for idx, var in enumerate(VARIABLES):
        ax = fig.add_subplot(3, 3, idx + 1, projection=plot_proj)
        meta = METADATA.get(var, {})
        
        obs_col = var
        pred_col = f"Pred_{var}"
        
        if obs_col not in df.columns or pred_col not in df.columns:
            ax.set_visible(False)
            continue
            
        valid_df = df[['Longitude', 'Latitude', obs_col, pred_col]].dropna()
        lons = valid_df['Longitude'].values
        lats = valid_df['Latitude'].values
        bias = valid_df[pred_col].values - valid_df[obs_col].values
        
        vmax = np.percentile(np.abs(bias), 92) 
        vmin = -vmax

        ret = stats.binned_statistic_2d(lons, lats, bias, statistic='mean', bins=[lon_bins, lat_bins])
        bias_grid = ret.statistic.T
        X, Y = np.meshgrid(ret.x_edge, ret.y_edge)

        # Extremely tight bounding box to center the mainland and maximize map size
        ax.set_extent([73.5, 134.5, 18.0, 53.8], crs=point_proj)
        
        for spine in ax.spines.values():
            spine.set_visible(False)
            
        ax.add_feature(cfeature.LAND, facecolor='#EAEAEA', edgecolor='none', zorder=1)
        ax.add_feature(cfeature.OCEAN, facecolor='#FFFFFF', edgecolor='none', zorder=1)
            
        pm = ax.pcolormesh(X, Y, bias_grid, cmap='coolwarm', vmin=vmin, vmax=vmax, 
                           transform=point_proj, zorder=2, rasterized=True)
        
        china_boundary.boundary.plot(ax=ax, edgecolor='#111111', linewidth=0.75, alpha=0.9, transform=plot_proj, zorder=3)
        
        unit_str = f" ({meta.get('unit')})" if meta.get('unit') else ""
        
        panel_label = chr(97 + idx)
        ax.text(0.02, 0.96, panel_label, transform=ax.transAxes, 
                fontsize=20, fontweight='bold', va='top', ha='left', zorder=10,
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, pad=3))
        
        cbar_ax = inset_axes(ax, width="60%", height="3.5%", loc='lower center', 
                             bbox_to_anchor=(0, -0.12, 1, 1), bbox_transform=ax.transAxes, borderpad=0)
        cbar = fig.colorbar(pm, cax=cbar_ax, orientation='horizontal', extend='both')
        cbar_ax.set_title(f"{meta.get('title')} Bias{unit_str}", fontsize=12, pad=6, loc='center', color='#222222', fontweight='normal')
        
        cbar.ax.xaxis.set_major_formatter(ticker.FuncFormatter(clean_formatter))
        cbar.ax.tick_params(labelsize=11, length=3, width=0.8)
        cbar.outline.set_linewidth(0.8)

        # Inset map precisely tucked into the bottom right corner
        inset_ax = inset_axes(ax, width="11%", height="25%", loc='lower right', 
                              axes_class=cartopy.mpl.geoaxes.GeoAxes, 
                              axes_kwargs=dict(projection=plot_proj), 
                              bbox_to_anchor=(-0.02, 0.05, 1, 1), bbox_transform=ax.transAxes, borderpad=0)
        
        inset_ax.set_extent([106, 124, 2, 24], crs=point_proj)
        
        inset_ax.add_feature(cfeature.LAND, facecolor='#EAEAEA', edgecolor='none', zorder=0)
        inset_ax.add_feature(cfeature.OCEAN, facecolor='#FFFFFF', edgecolor='none', zorder=0)
        
        inset_ax.pcolormesh(X, Y, bias_grid, cmap='coolwarm', vmin=vmin, vmax=vmax, 
                            transform=point_proj, zorder=2, rasterized=True)
        
        china_boundary.boundary.plot(ax=inset_ax, edgecolor='#111111', linewidth=0.4, alpha=0.9, transform=plot_proj, zorder=3)
        ninedash_line.plot(ax=inset_ax, edgecolor='#111111', linewidth=0.5, transform=plot_proj, zorder=3)
        
        for spine in inset_ax.spines.values():
            spine.set_linewidth(0.4)
            spine.set_edgecolor('#444444')

    # Tight layout configuration to minimize empty margins
    plt.subplots_adjust(left=0.01, right=0.99, top=0.98, bottom=0.05, wspace=0.02, hspace=0.15)
    
    print(f"Saving Output Files to: {OUTPUT_DIR}")
    plt.savefig(BIAS_MAP_PNG, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(BIAS_MAP_PDF, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(BIAS_MAP_TIF, dpi=600, bbox_inches='tight', format='tiff', 
                pil_kwargs={"compression": "tiff_lzw"}, facecolor='white')
    
    plt.close(fig)
    gc.collect()

# ==========================================
# 3. Main Execution
# ==========================================
def main():
    print(f"Loading data from {DATA_PATH}...")
    try:
        df = pd.read_csv(DATA_PATH)
    except Exception as e:
        print(f"Failed to load CSV data: {e}")
        return

    create_bias_maps(df)
    print("Top-tier map visualization successfully completed!")

if __name__ == '__main__':
    main()