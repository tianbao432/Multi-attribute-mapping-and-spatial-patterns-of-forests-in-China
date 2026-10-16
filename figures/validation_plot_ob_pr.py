import os
import gc
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy import stats
from sklearn.metrics import mean_squared_error, r2_score

# Ignore environmental warnings
warnings.filterwarnings("ignore", category=UserWarning)

# ==========================================
# 1. Premium Journal Configuration
# ==========================================
# Configure matplotlib for ultra-clean, publication-ready output
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 16,             
    'axes.labelsize': 18,        
    'axes.titlesize': 20,        
    'xtick.labelsize': 15,       
    'ytick.labelsize': 15,       
    'axes.linewidth': 1.2,
    'xtick.major.width': 1.2,
    'ytick.major.width': 1.2,
    'xtick.major.size': 5.0,
    'ytick.major.size': 5.0,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'axes.facecolor': 'white',
    'figure.facecolor': 'white'
})

# Paths
DATA_PATH = "/data/China_mapping/validation_plot/Validation_Plot_Data_Extracted.csv"
OUTPUT_DIR = "/data/China_mapping/validation_plot/figures/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SCATTER_PNG = os.path.join(OUTPUT_DIR, "Validation_Scatter_Plots.png")
SCATTER_PDF = os.path.join(OUTPUT_DIR, "Validation_Scatter_Plots.pdf")
SCATTER_TIF = os.path.join(OUTPUT_DIR, "Validation_Scatter_Plots.tif")

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
# 2. Premium Scatter Plots
# ==========================================
def create_scatter_plots(df):
    print("Generating Premium Validation Scatter Plots...")
    
    # Create figure with optimal spacing for a 3x3 layout
    fig, axes = plt.subplots(3, 3, figsize=(14, 14), dpi=300)
    axes = axes.flatten()
    
    # New Top-Tier Academic Color Palette
    colormap = 'YlGnBu'          # Stunning, universally readable scientific colormap
    reg_color = '#D62828'         # Elegant deep red for the thin trendline
    one_to_one_color = '#888888'  # Soft grey for the reference line
    
    for i, var in enumerate(VARIABLES):
        ax = axes[i]
        obs_col = var
        pred_col = f"Pred_{var}"
        
        if obs_col not in df.columns or pred_col not in df.columns:
            ax.set_visible(False)
            continue
            
        valid_data = df[[obs_col, pred_col]].dropna()
        x = valid_data[obs_col].values
        y = valid_data[pred_col].values
        
        if len(x) == 0:
            ax.set_visible(False)
            continue
            
        n_samples = len(x)
        r2 = r2_score(x, y)
        rmse = np.sqrt(mean_squared_error(x, y))
        bias = np.mean(y - x)
        
        # Calculate dynamic axes limits with a clean 5% buffer
        min_val = min(np.min(x), np.min(y))
        max_val = max(np.max(x), np.max(y))
        buffer = (max_val - min_val) * 0.05
        ax_min = min_val - buffer
        ax_max = max_val + buffer
        
        # Remove top and right spines for a clean, open aesthetic
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # High-definition Hexbin with Logarithmic scaling
        hb = ax.hexbin(x, y, gridsize=60, cmap=colormap, mincnt=1, bins='log',
                       extent=[ax_min, ax_max, ax_min, ax_max], 
                       linewidths=0.2, edgecolors='none', zorder=2, rasterized=True)
        
        # 1:1 Line (thinner)
        ax.plot([ax_min, ax_max], [ax_min, ax_max], color=one_to_one_color, 
                linestyle='--', linewidth=1.0, zorder=3)
        
        # Regression Line (thinner and elegant)
        slope, intercept, _, _, _ = stats.linregress(x, y)
        x_reg = np.array([ax_min, ax_max])
        y_reg = slope * x_reg + intercept
        ax.plot(x_reg, y_reg, color=reg_color, linewidth=1.2, zorder=4)
        
        # Formatting limits and aspect ratio
        ax.set_xlim(ax_min, ax_max)
        ax.set_ylim(ax_min, ax_max)
        ax.set_aspect('equal', adjustable='box')
        
        # Professional statistics inset - borderless, larger font, N without commas
        stats_text = (f"$N = {n_samples}$\n"
                      f"$R^2 = {r2:.2f}$\n"
                      f"RMSE = {rmse:.2f}\n"
                      f"Bias = {bias:.2f}")
                      
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=14,
                verticalalignment='top', bbox=dict(boxstyle='square,pad=0', facecolor='none', edgecolor='none'), zorder=5)
        
        # Titles and Axis Labels
        meta = METADATA.get(var, {})
        title_text = f"{meta.get('title', var)}"
        unit_text = f" ({meta.get('unit')})" if meta.get('unit') else ""
        
        # Clean title without bold font and without panel labels
        ax.set_title(title_text, pad=12, loc='center')
        ax.set_xlabel(f"Observed {title_text}{unit_text}")
        ax.set_ylabel(f"Predicted {title_text}{unit_text}")
        
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(clean_formatter))
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(clean_formatter))

    # Perfect the layout spacing
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.35, wspace=0.30)
    
    print(f"Saving Output Files to: {OUTPUT_DIR}")
    # Save standard PNG
    plt.savefig(SCATTER_PNG, dpi=300, bbox_inches='tight', facecolor='white')
    # Save Vector PDF
    plt.savefig(SCATTER_PDF, dpi=300, bbox_inches='tight', facecolor='white')
    # Save High-Res TIF (LZW Compressed)
    plt.savefig(SCATTER_TIF, dpi=600, bbox_inches='tight', format='tiff', pil_kwargs={"compression": "tiff_lzw"}, facecolor='white')
    plt.close()
    
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

    create_scatter_plots(df)
    print("Top-tier visualization successfully completed!")

if __name__ == '__main__':
    main()