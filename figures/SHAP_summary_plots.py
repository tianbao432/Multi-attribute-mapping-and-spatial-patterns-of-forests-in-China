import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.cm import ScalarMappable
from scipy.stats import gaussian_kde

# =============================================================================
# Configuration
# =============================================================================
RESULT_ROOT = "/data/China_mapping/driver/result"
OUTPUT_DIR = "/data/China_mapping/driver/figures"
OUTPUT_BASENAME = "Combined_SHAP_Beeswarm_9Attributes_Top5"

TOP_N_FEATURES = 10
MAX_POINTS_PER_FEATURE = 80000
RANDOM_SEED = 42
DPI = 600

TARGET_CONFIGS = [
    {"folder": "AGB",      "title": "AGB"},
    {"folder": "Age",      "title": "Age"},
    {"folder": "BA",       "title": "BA"},
    {"folder": "CV_DBH",   "title": "CV DBH"},
    {"folder": "DBH",      "title": "DBH"},
    {"folder": "Height",   "title": "Height"},
    {"folder": "Density",  "title": "Density"},
    {"folder": "Shannon",  "title": "Shannon"},
    {"folder": "GSV",      "title": "GSV"},
]

FEATURE_NAME_MAP = {
    "HMI": "HMI",
    "MAT": "MAT",
    "ISO": "ISO",
    "MTCM": "MTCM",
    "MTDQ": "MTDQ",
    "MAP": "MAP",
    "PDM": "PDM",
    "PS": "PS",
    "ELEV": "ELEV",
    "TWI": "TWI",
    "VPD": "VPD",
    "SWVL3": "SWVL3",
    "TN": "TN",
    "PH": "pH",
    "CLAY": "CLAY",
}

# Define SHAP colormap 
SHAP_CMAP = LinearSegmentedColormap.from_list(
    "muted_teal_coral",
    [
        "#1F8A8A",
        "#58AAA5",
        "#A9D4CD",
        "#F2F2F2",
        "#F7C6B7",
        "#F28E72",
        "#E96B4C",
    ],
    N=256,
)

# =============================================================================
# Plot style (Nature Journal Guidelines)
# =============================================================================
def set_plot_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7,                
        "axes.titlesize": 8,           
        "axes.labelsize": 7,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 7,
        "axes.linewidth": 0.5,         
        "xtick.direction": "out",      
        "ytick.direction": "out",
        "xtick.major.width": 0.5,      
        "ytick.major.width": 0.5,
        "xtick.major.size": 3.0,       
        "ytick.major.size": 3.0,
        "savefig.dpi": DPI,
        "pdf.fonttype": 42,            
        "ps.fonttype": 42,
    })

# =============================================================================
# Utilities
# =============================================================================
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def load_target_data(target_folder):
    target_dir = os.path.join(RESULT_ROOT, target_folder)

    shap_path = os.path.join(target_dir, "final_shap_values_and_features.csv")
    imp_path = os.path.join(target_dir, "final_shap_importance.csv")

    if not os.path.exists(shap_path):
        raise FileNotFoundError(f"Missing file: {shap_path}")
    if not os.path.exists(imp_path):
        raise FileNotFoundError(f"Missing file: {imp_path}")

    shap_df = pd.read_csv(shap_path)
    imp_df = pd.read_csv(imp_path)

    return shap_df, imp_df

def robust_normalize(values):
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(values)

    if valid.sum() == 0:
        return np.full(values.shape, 0.5, dtype=float)

    v = values[valid]
    vmin = np.nanpercentile(v, 5)
    vmax = np.nanpercentile(v, 95)

    if np.isclose(vmax, vmin):
        return np.full(values.shape, 0.5, dtype=float)

    out = (values - vmin) / (vmax - vmin)
    out = np.clip(out, 0.0, 1.0)
    out[~valid] = 0.5
    return out

def subsample_points(x, c, max_points, rng):
    n = len(x)
    if n <= max_points:
        return x, c

    idx = rng.choice(n, size=max_points, replace=False)
    return x[idx], c[idx]

def compute_density_jitter(x, max_spread=0.35):
    x = np.asarray(x, dtype=float)
    valid = np.isfinite(x)
    if valid.sum() <= 1:
        return np.zeros_like(x)

    xv = x[valid]
    if np.std(xv) < 1e-6:
        offsets = np.random.uniform(-max_spread, max_spread, size=len(x))
        offsets[~valid] = 0
        return offsets

    kde = gaussian_kde(xv)
    density = kde(xv)
    density = density / np.max(density)
    
    jitter = np.random.uniform(-1, 1, size=len(xv)) * density * max_spread
    
    offsets = np.zeros_like(x)
    offsets[valid] = jitter
    return offsets

def get_feature_order(imp_df, top_n):
    if "Feature" not in imp_df.columns:
        raise ValueError("The importance file must contain a 'Feature' column.")
    return imp_df["Feature"].head(top_n).tolist()

def prettify_feature_names(features):
    return [FEATURE_NAME_MAP.get(f, f) for f in features]

# =============================================================================
# Panel plotting
# =============================================================================
def plot_beeswarm_panel(ax, shap_df, imp_df, panel_title, rng):
    top_features = get_feature_order(imp_df, TOP_N_FEATURES)
    top_features_pretty = prettify_feature_names(top_features)

    all_abs_vals = []
    for feat in top_features:
        shap_col = f"SHAP_{feat}"
        if shap_col not in shap_df.columns:
            raise ValueError(f"Missing SHAP column: {shap_col}")
        all_abs_vals.extend(np.abs(shap_df[shap_col].to_numpy(dtype=float)))

    all_abs_vals = np.asarray(all_abs_vals, dtype=float)
    all_abs_vals = all_abs_vals[np.isfinite(all_abs_vals)]

    if all_abs_vals.size == 0:
        x_limit = 1.0
    else:
        x_limit = np.nanpercentile(all_abs_vals, 99.5)
        x_limit = max(x_limit, 1e-6)

    x_limit *= 1.1
    n_features = len(top_features)

    for i, feat in enumerate(top_features):
        shap_col = f"SHAP_{feat}"

        if feat not in shap_df.columns:
            raise ValueError(f"Missing feature column: {feat}")

        x = shap_df[shap_col].to_numpy(dtype=float)
        c = shap_df[feat].to_numpy(dtype=float)

        valid = np.isfinite(x) & np.isfinite(c)
        x = x[valid]
        c = c[valid]

        if len(x) == 0:
            continue

        x, c = subsample_points(x, c, MAX_POINTS_PER_FEATURE, rng)
        c_norm = robust_normalize(c)

        offsets = compute_density_jitter(x, max_spread=0.35)
        y = np.full(len(x), i, dtype=float) + offsets

        ax.scatter(
            x,
            y,
            c=c_norm,
            cmap=SHAP_CMAP,
            s=4,           
            alpha=0.8,
            linewidths=0.0,
            rasterized=True,
            zorder=3
        )

    # Reference line at zero
    ax.axvline(0, color="black", linestyle="-", linewidth=0.5, zorder=2, alpha=0.7)

    # Horizontal guide lines 
    for yline in range(n_features):
        ax.hlines(
            y=yline,
            xmin=-x_limit,
            xmax=x_limit,
            colors="gray",
            linestyles=":",
            linewidth=0.5,
            alpha=0.4,
            zorder=1
        )

    ax.set_xlim(-x_limit, x_limit)
    ax.set_ylim(-0.6, n_features - 0.4)
    ax.invert_yaxis()

    ax.set_yticks(np.arange(n_features))
    ax.set_yticklabels(top_features_pretty)

    ax.set_title(panel_title, loc="left", pad=8, fontsize=8)
    
    ax.set_xlabel("SHAP value")
    
    ax.xaxis.set_label_coords(0.5, -0.15)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)
    
    ax.tick_params(top=False, right=False, direction="out", length=3.0, width=0.5)

# =============================================================================
# Main
# =============================================================================
def main():
    ensure_dir(OUTPUT_DIR)
    set_plot_style()

    rng = np.random.default_rng(RANDOM_SEED)

    # Nature double-column width ~7.2 inches
    fig, axes = plt.subplots(
        nrows=3,
        ncols=3,
        figsize=(6,6),  
        constrained_layout=False
    )
    axes = axes.flatten()

    for idx, cfg in enumerate(TARGET_CONFIGS):
        shap_df, imp_df = load_target_data(cfg["folder"])
        
        plot_beeswarm_panel(
            ax=axes[idx],
            shap_df=shap_df,
            imp_df=imp_df,
            panel_title=cfg["title"],
            rng=rng
        )

    plt.subplots_adjust(
        left=0.08,
        right=0.96,
        top=0.94,
        bottom=0.12, 
        wspace=0.35,  
        hspace=0.45   
    )

    # Horizontal colorbar at the bottom center
    cbar_ax = fig.add_axes([0.35, 0.03, 0.3, 0.012])
    sm = ScalarMappable(norm=Normalize(vmin=0, vmax=1), cmap=SHAP_CMAP)
    sm.set_array([])

    cbar = fig.colorbar(
        sm,
        cax=cbar_ax,
        orientation="horizontal"
    )
    cbar.set_label("Feature value", labelpad=5, fontsize=7)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["Low", "High"])
    
    cbar.outline.set_linewidth(0.5)
    cbar_ax.tick_params(size=0) 

    png_path = os.path.join(OUTPUT_DIR, f"{OUTPUT_BASENAME}.png")
    tif_path = os.path.join(OUTPUT_DIR, f"{OUTPUT_BASENAME}.tif")

    fig.savefig(png_path, dpi=DPI, bbox_inches="tight", transparent=False, facecolor='white')
    fig.savefig(
        tif_path,
        dpi=DPI,
        bbox_inches="tight",
        transparent=False,
        facecolor='white',
        pil_kwargs={"compression": "tiff_lzw"}
    )

    plt.close(fig)

    print("Figure saved successfully:")
    print(png_path)
    print(tif_path)


if __name__ == "__main__":
    main()