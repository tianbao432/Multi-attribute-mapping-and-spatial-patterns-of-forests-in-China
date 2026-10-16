import os
import gc
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import rasterio
from cartopy.mpl.geoaxes import GeoAxes
from rasterio.enums import Resampling
import geopandas as gpd
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 1. Configuration & Paths
# ==========================================
base_dir = "/data/China_mapping/production/fina_production/"
china_shp_path = "/data/China_mapping/forest_boundary/china.shp"
ninedash_shp_path = "/data/China_mapping/forest_boundary/JDX.shp"
output_dir = "/data/China_mapping/production/VIS/"
os.makedirs(output_dir, exist_ok=True)

output_png = os.path.join(output_dir, "Nine_Parameters_CV_Continuous.png")
output_tif = os.path.join(output_dir, "Nine_Parameters_CV_Continuous.tif")

var_config = {
    'AGB': {
        'file': "China_National_AGB_CV.tif",
        'label': 'a',
        'cbar_title': 'CV of Aboveground Biomass'
    },
    'AGE': {
        'file': "China_National_AGE_CV.tif",
        'label': 'b',
        'cbar_title': 'CV of Forest Age'
    },
    'BA': {
        'file': "China_National_BA_CV.tif",
        'label': 'c',
        'cbar_title': 'CV of Basal Area'
    },
    'CV_DBH': {
        'file': "China_National_CV_DBH_CV.tif", 
        'label': 'd',
        'cbar_title': 'CV of DBH Variation'
    },
    'DBH': {
        'file': "China_National_DBH_CV.tif", 
        'label': 'e',
        'cbar_title': 'CV of Diameter at Breast Height'
    },
    'H': {
        'file': "China_National_H_CV.tif",
        'label': 'f',
        'cbar_title': 'CV of Tree Height'
    },
    'Trees_N': {
        'file': "China_National_Trees_N_CV.tif",
        'label': 'g',
        'cbar_title': 'CV of Tree Density'
    },
    'Shannon': {
        'file': "China_National_shannon_CV.tif",
        'label': 'h',
        'cbar_title': 'CV of Shannon Index'
    },
    'V': {
        'file': "China_National_V_CV.tif",
        'label': 'i',
        'cbar_title': 'CV of Growing Stock Volume'
    }
}

# Publication-grade style settings
CBAR_LABEL_SIZE = 16
TICK_SIZE = 12
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.linewidth'] = 1.0

# ==========================================
# 2. Read Vector Data
# ==========================================
print("Reading boundary and Nine-Dash Line shapefiles...")
china_boundary = gpd.read_file(china_shp_path, engine="fiona")
ninedash_line = gpd.read_file(ninedash_shp_path, engine="fiona")

albers_crs = "+proj=aea +lat_1=25 +lat_2=47 +lon_0=105 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
china_boundary.set_crs(albers_crs, allow_override=True, inplace=True)
ninedash_line.set_crs(albers_crs, allow_override=True, inplace=True)

plot_proj = ccrs.Robinson(central_longitude=105)
data_proj = ccrs.PlateCarree()

# ==========================================
# 3. Canvas Setup (3 rows x 3 cols)
# ==========================================
print("Initializing canvas...")
fig = plt.figure(figsize=(16, 18))
downscale_factor = 25 

keys = list(var_config.keys())

for idx in range(9): 
    ax = fig.add_subplot(3, 3, idx + 1, projection=plot_proj)
        
    var_key = keys[idx]
    cfg = var_config[var_key]
    raster_path = os.path.join(base_dir, cfg['file'])
    
    print(f"Processing ({cfg['label']}) {var_key} CV...")
    
    # 3.1 Read and Downsample Raster
    with rasterio.open(raster_path) as src:
        raster_crs = src.crs
        if idx == 0:
            cb_reproj = china_boundary.to_crs(raster_crs)
            nd_reproj = ninedash_line.to_crs(raster_crs)
        
        new_height = int(src.height // downscale_factor)
        new_width = int(src.width // downscale_factor)
        
        data = src.read(
            1, 
            out_shape=(1, new_height, new_width), 
            resampling=Resampling.bilinear
        )
        
        transform = src.transform * src.transform.scale(
            (src.width / data.shape[-1]),
            (src.height / data.shape[-2])
        )
        
        nodata_value = src.nodata
        if nodata_value is not None:
            data = np.ma.masked_equal(data, nodata_value)
        else:
            data = np.ma.masked_less_equal(data, 0) 
            
        extent = [transform[2], transform[2] + transform[0] * new_width,
                  transform[5] + transform[4] * new_height, transform[5]]

    # 3.2 Map Layout Settings
    ax.set_extent([73, 136, 15, 55], crs=data_proj)
    for spine in ax.spines.values():
        spine.set_visible(False)
        
    # 3.3 Dynamic Continuous Colormap Setup (Thresholding Bottom 30%)
    valid_data = data.filled(np.nan)
    p2 = np.nanpercentile(valid_data, 2)
    p30 = np.nanpercentile(valid_data, 30)
    p98 = np.nanpercentile(valid_data, 98)
    
    if p98 == p2:
        p98 = p2 + 1e-5
        
    # Normalize the 30th percentile position strictly between [0, 1]
    p30_norm = (p30 - p2) / (p98 - p2)
    p30_norm = np.clip(p30_norm, 0.05, 0.95)
    
    base_cmap = plt.colormaps['YlOrRd']
    
    # Force the 0% to 30% range into light pale colors, and scale the rest
    custom_colors = [
        '#ffffff',                             # 0.0: Pure White
        '#fff7bc',                             # p30_norm: Pale Yellow
        base_cmap(0.5),                        # Mid-high gradient
        base_cmap(0.8),                        # High gradient
        base_cmap(1.0)                         # 1.0: Maximum Red
    ]
    
    color_nodes = [
        0.0, 
        p30_norm, 
        p30_norm + (1.0 - p30_norm) * 0.33, 
        p30_norm + (1.0 - p30_norm) * 0.66, 
        1.0
    ]
    
    cmap_dynamic = mcolors.LinearSegmentedColormap.from_list(f"custom_cmap_{var_key}", list(zip(color_nodes, custom_colors)))
    cmap_dynamic.set_bad(color='white')
    
    # Plot Raster
    img = ax.imshow(data, cmap=cmap_dynamic, extent=extent, origin='upper', 
                    vmin=p2, vmax=p98, transform=data_proj)
    
    # Plot Boundaries
    cb_reproj.boundary.plot(ax=ax, edgecolor='black', linewidth=0.8, alpha=0.9, transform=data_proj)
    
    # Add Alphabet Label (a, b, c...)
    ax.text(0.02, 0.98, cfg['label'], transform=ax.transAxes, 
            fontsize=24, fontweight='bold', va='top', ha='left', zorder=10)
    
    # Add Horizontal Colorbar
    cbar_ax = inset_axes(ax, width="70%", height="4%", loc='lower center', 
                         bbox_to_anchor=(0, -0.12, 1, 1), bbox_transform=ax.transAxes, borderpad=0)
    cbar = fig.colorbar(img, cax=cbar_ax, orientation='horizontal', extend='both')
    cbar_ax.set_title(cfg['cbar_title'], fontsize=CBAR_LABEL_SIZE, pad=10)
    cbar.ax.tick_params(labelsize=TICK_SIZE)
    cbar.outline.set_linewidth(0.8)

    # 3.4 Add South China Sea Inset
    inset_ax = inset_axes(ax, width="20%", height="28%", loc='lower right', 
                          axes_class=GeoAxes, 
                          axes_kwargs=dict(projection=plot_proj), borderpad=1)
    inset_ax.set_extent([105, 125, 0, 25], crs=data_proj)
    inset_ax.add_feature(cfeature.LAND, facecolor='whitesmoke', edgecolor='none')
    inset_ax.add_feature(cfeature.OCEAN, facecolor='white', edgecolor='none')
    cb_reproj.boundary.plot(ax=inset_ax, edgecolor='black', linewidth=0.4, alpha=0.9, transform=data_proj)
    nd_reproj.plot(ax=inset_ax, edgecolor='black', linewidth=0.6, transform=data_proj)
    
    for spine in inset_ax.spines.values():
        spine.set_linewidth(0.6)
        spine.set_edgecolor('black')

    # Memory cleanup per loop to maintain stability during raster processing
    del data, valid_data
    gc.collect()

# ==========================================
# 4. Export Images
# ==========================================
print(f"Exporting combined CV layout to {output_dir}...")
plt.subplots_adjust(wspace=0.05, hspace=0.05) 

plt.savefig(output_png, dpi=300, bbox_inches='tight', pad_inches=0.1, transparent=False)
plt.savefig(output_tif, dpi=900, bbox_inches='tight', pad_inches=0.1, format='tiff', 
            pil_kwargs={"compression": "tiff_lzw"})

plt.close(fig)
gc.collect()
print("Execution complete. Visualizations successfully exported.")