# -*- coding: utf-8 -*-
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from scipy.stats import gaussian_kde

warnings.filterwarnings("ignore")


# =============================================================================
# Configuration
# =============================================================================
RESULT_ROOT = "/data/China_mapping/driver/result"
OUTPUT_DIR = "/data/China_mapping/driver/figures"
OUTPUT_BASENAME = "Top1_SHAP_Response_Density_9Attributes"

DPI = 600
RANDOM_SEED = 42

# Maximum number of points used in each panel to prevent overplotting.
MAX_POINTS_PER_PANEL = 100000

# Robust display ranges.
X_LOWER_PERCENTILE = 1.0
X_UPPER_PERCENTILE = 99.0
Y_LOWER_PERCENTILE = 1.0
Y_UPPER_PERCENTILE = 99.0

# Density calculation settings.
KDE_BW_METHOD = "scott"
DENSITY_CLIP_PERCENTILE = 99.0

# Point appearance optimized for large datasets.
POINT_SIZE = 4.0
POINT_ALPHA = 0.6
POINT_EDGE_WIDTH = 0.0

# Figure arrangement.
N_ROWS = 3
N_COLS = 3
FIGURE_SIZE = (9, 9)  # Slightly tightened for better proportions

# Define SHAP colormap (using viridis for academic standard)
SHAP_CMAP = plt.get_cmap("viridis")
TARGET_CONFIGS = [
    {"folder": "AGB",     "title": "AGB"},
    {"folder": "Age",     "title": "Age"},
    {"folder": "BA",      "title": "BA"},
    {"folder": "CV_DBH",  "alternative_folders": ["CV DBH", "CV DBH"], "title": "CV DBH"},
    {"folder": "DBH",     "title": "DBH"},
    {"folder": "Height",  "title": "Height"},
    {"folder": "Density", "title": "Density"},
    {"folder": "Shannon", "title": "Shannon"},
    {"folder": "GSV",     "title": "GSV"},
]

FEATURE_LABELS = {
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
    "CLAY": "Clay"
}

# Define units for each feature to append to the X-axis label
# Using unicode escapes (\u00B0 for degree, \u00B3 for cubed) to completely avoid file encoding errors.
FEATURE_UNITS = {
    "HMI": "",          
    "MAT": "\u00B0C",      
    "ISO": "%",
    "MTCM": "\u00B0C",     
    "MTDQ": "\u00B0C",     
    "MAP": "mm",
    "PDM": "mm",
    "PS": "",
    "ELEV": "m",
    "TWI": "",          
    "VPD": "kPa",
    "SWVL3": "m\u00B3/m\u00B3", 
    "TN": "g/kg",       
    "PH": "",           
    "CLAY": "%"
}


# =============================================================================
# Plot style
# =============================================================================
def set_plot_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "axes.linewidth": 0.5,           # Ultra-thin global axes lines
        "axes.spines.top": False,        # Top-tier journals remove top/right spines
        "axes.spines.right": False,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 4.0,
        "ytick.major.size": 4.0,
        "xtick.major.width": 0.5,        # Thinner ticks
        "ytick.major.width": 0.5,
        "figure.dpi": 150,
        "savefig.dpi": DPI,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "pdf.fonttype": 42,
        "ps.fonttype": 42
    })


# =============================================================================
# File utilities
# =============================================================================
def ensure_directory(path):
    os.makedirs(path, exist_ok=True)


def resolve_target_directory(config):
    candidates = [config["folder"]]
    candidates.extend(config.get("alternative_folders", []))

    for folder in candidates:
        target_dir = os.path.join(RESULT_ROOT, folder)

        importance_path = os.path.join(
            target_dir,
            "final_shap_importance.csv"
        )
        shap_values_path = os.path.join(
            target_dir,
            "final_shap_values_and_features.csv"
        )

        if os.path.exists(importance_path) and os.path.exists(
            shap_values_path
        ):
            return target_dir

    candidate_text = ", ".join(candidates)

    raise FileNotFoundError(
        f"No valid result directory was found for: {candidate_text}"
    )


def load_top1_data(config):
    target_dir = resolve_target_directory(config)

    importance_path = os.path.join(
        target_dir,
        "final_shap_importance.csv"
    )
    shap_values_path = os.path.join(
        target_dir,
        "final_shap_values_and_features.csv"
    )

    importance_df = pd.read_csv(importance_path)
    shap_df = pd.read_csv(shap_values_path)

    if "Feature" not in importance_df.columns:
        raise KeyError(
            f"'Feature' is missing from: {importance_path}"
        )

    importance_columns = [
        "Mean_Absolute_SHAP",
        "Mean_Abs_SHAP"
    ]

    available_importance_columns = [
        column
        for column in importance_columns
        if column in importance_df.columns
    ]

    if available_importance_columns:
        importance_column = available_importance_columns[0]

        importance_df = importance_df.sort_values(
            importance_column,
            ascending=False
        ).reset_index(drop=True)

    top_feature = str(importance_df.loc[0, "Feature"])
    shap_column = f"SHAP_{top_feature}"

    if top_feature not in shap_df.columns:
        raise KeyError(
            f"Feature column '{top_feature}' is missing from: "
            f"{shap_values_path}"
        )

    if shap_column not in shap_df.columns:
        raise KeyError(
            f"SHAP column '{shap_column}' is missing from: "
            f"{shap_values_path}"
        )

    plot_df = shap_df[
        [top_feature, shap_column]
    ].copy()

    plot_df.columns = [
        "Feature_Value",
        "SHAP_Value"
    ]

    plot_df["Feature_Value"] = pd.to_numeric(
        plot_df["Feature_Value"],
        errors="coerce"
    )
    plot_df["SHAP_Value"] = pd.to_numeric(
        plot_df["SHAP_Value"],
        errors="coerce"
    )

    plot_df = (
        plot_df
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .reset_index(drop=True)
    )

    if len(plot_df) < 3:
        raise ValueError(
            f"Insufficient valid points for {config['title']}."
        )

    return plot_df, top_feature, target_dir


# =============================================================================
# Density estimation
# =============================================================================
def subsample_dataframe(df, max_points, seed):
    if max_points is None or len(df) <= max_points:
        return df.reset_index(drop=True)

    return df.sample(
        n=max_points,
        replace=False,
        random_state=seed
    ).reset_index(drop=True)


def calculate_density(x_values, y_values):
    x_values = np.asarray(x_values, dtype=float)
    y_values = np.asarray(y_values, dtype=float)

    coordinates = np.vstack([
        x_values,
        y_values
    ])

    try:
        density = gaussian_kde(
            coordinates,
            bw_method=KDE_BW_METHOD
        )(coordinates)

    except Exception:
        rng = np.random.default_rng(RANDOM_SEED)

        x_scale = max(
            float(np.nanstd(x_values)),
            1e-8
        )
        y_scale = max(
            float(np.nanstd(y_values)),
            1e-8
        )

        x_jittered = x_values + rng.normal(
            loc=0.0,
            scale=x_scale * 1e-6,
            size=len(x_values)
        )

        y_jittered = y_values + rng.normal(
            loc=0.0,
            scale=y_scale * 1e-6,
            size=len(y_values)
        )

        jittered_coordinates = np.vstack([
            x_jittered,
            y_jittered
        ])

        density = gaussian_kde(
            jittered_coordinates,
            bw_method=KDE_BW_METHOD
        )(jittered_coordinates)

    density = np.asarray(density, dtype=float)
    density[~np.isfinite(density)] = 0.0

    upper_density = np.nanpercentile(
        density,
        DENSITY_CLIP_PERCENTILE
    )

    if not np.isfinite(upper_density) or upper_density <= 0:
        upper_density = np.nanmax(density)

    if not np.isfinite(upper_density) or upper_density <= 0:
        normalized_density = np.zeros_like(density)
    else:
        clipped_density = np.clip(
            density,
            0.0,
            upper_density
        )

        normalized_density = (
            clipped_density / upper_density
        )

    return normalized_density


def sort_points_by_density(x_values, y_values, density):
    order = np.argsort(density)

    return (
        x_values[order],
        y_values[order],
        density[order]
    )


# =============================================================================
# Axis utilities
# =============================================================================
def robust_axis_limits(
    values,
    lower_percentile,
    upper_percentile,
    padding_fraction=0.04
):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    lower = float(
        np.nanpercentile(values, lower_percentile)
    )
    upper = float(
        np.nanpercentile(values, upper_percentile)
    )

    if np.isclose(lower, upper):
        center = 0.5 * (lower + upper)
        spread = max(abs(center) * 0.05, 1e-6)

        lower = center - spread
        upper = center + spread

    padding = (upper - lower) * padding_fraction

    return lower - padding, upper + padding


def format_feature_label(feature):
    label = FEATURE_LABELS.get(feature, feature)
    unit = FEATURE_UNITS.get(feature, "")
    if unit:
        return f"{label} ({unit})"
    return label


# =============================================================================
# Panel plotting
# =============================================================================
def plot_density_panel(
    ax,
    config,
    seed
):
    plot_df, top_feature, target_dir = load_top1_data(config)

    plot_df = subsample_dataframe(
        df=plot_df,
        max_points=MAX_POINTS_PER_PANEL,
        seed=seed
    )

    x_values = plot_df["Feature_Value"].to_numpy(
        dtype=float
    )
    y_values = plot_df["SHAP_Value"].to_numpy(
        dtype=float
    )

    density = calculate_density(
        x_values=x_values,
        y_values=y_values
    )

    x_values, y_values, density = sort_points_by_density(
        x_values=x_values,
        y_values=y_values,
        density=density
    )

    scatter = ax.scatter(
        x_values,
        y_values,
        c=density,
        cmap=SHAP_CMAP,
        norm=Normalize(vmin=0.0, vmax=1.0),
        s=POINT_SIZE,
        alpha=POINT_ALPHA,
        edgecolors="none",
        linewidths=POINT_EDGE_WIDTH,
        rasterized=True
    )

    # Reference line at zero - made ultra-thin (0.5) to match aesthetic
    ax.axhline(
        y=0.0,
        color="black",
        linewidth=0.5,
        linestyle="-",
        zorder=0,
        alpha=0.6
    )

    x_lower, x_upper = robust_axis_limits(
        values=x_values,
        lower_percentile=X_LOWER_PERCENTILE,
        upper_percentile=X_UPPER_PERCENTILE
    )

    y_lower, y_upper = robust_axis_limits(
        values=y_values,
        lower_percentile=Y_LOWER_PERCENTILE,
        upper_percentile=Y_UPPER_PERCENTILE
    )

    ax.set_xlim(x_lower, x_upper)
    ax.set_ylim(y_lower, y_upper)

    # X-axis label: Now unbolded ("normal") and includes units
    ax.set_xlabel(
        format_feature_label(top_feature),
        fontweight="normal"
    )
    
    # Y-axis label
    ax.set_ylabel("SHAP value", fontweight="normal")
    
    ax.xaxis.set_label_coords(0.5, -0.15)

    # Title: Removed Driver Factor parentheses, Left-aligned, Bold
    title_text = config['title']
    ax.set_title(
        title_text,
        loc="left",
        fontsize=11,
        pad=10
    )

    ax.tick_params(
        axis="both",
        which="major",
        pad=4
    )

    return {
        "Target": config["title"],
        "Top_Feature": top_feature,
        "Sample_N": len(plot_df),
        "Source_Directory": target_dir
    }, scatter


# =============================================================================
# Main
# =============================================================================
def main():
    ensure_directory(OUTPUT_DIR)
    set_plot_style()

    figure, axes = plt.subplots(
        nrows=N_ROWS,
        ncols=N_COLS,
        figsize=FIGURE_SIZE,
        constrained_layout=False
    )

    axes = axes.flatten()
    summary_records = []

    for index, config in enumerate(TARGET_CONFIGS):
        summary, scatter = plot_density_panel(
            ax=axes[index],
            config=config,
            seed=RANDOM_SEED + index
        )

        summary_records.append(summary)

    # Adjusted spacing to keep subplots compact
    plt.subplots_adjust(
        left=0.08,
        right=0.96,
        bottom=0.12,
        top=0.93,
        wspace=0.30,
        hspace=0.45
    )

    # Horizontal shared density colorbar.
    colorbar_axis = figure.add_axes([0.35, 0.03, 0.3, 0.015])
    scalar_mappable = ScalarMappable(
        norm=Normalize(vmin=0.0, vmax=1.0),
        cmap=SHAP_CMAP
    )
    scalar_mappable.set_array([])

    colorbar = figure.colorbar(
        scalar_mappable,
        cax=colorbar_axis,
        orientation="horizontal"
    )

    colorbar.set_label(
        "",
        labelpad=6,
        fontweight="normal"
    )

    # Removed the "Medium" tick and label
    colorbar.set_ticks([0.0, 1.0])
    colorbar.set_ticklabels([
        "Low",
        "High"
    ])
    
    # Ultra-thin colorbar outline
    colorbar.outline.set_linewidth(0.5)
    colorbar_axis.tick_params(size=0)

    figure.suptitle(
        "",
        fontsize=13,
        fontweight="normal",
        y=0.98
    )

    output_png = os.path.join(
        OUTPUT_DIR,
        f"{OUTPUT_BASENAME}.png"
    )

    output_tif = os.path.join(
        OUTPUT_DIR,
        f"{OUTPUT_BASENAME}.tif"
    )

    output_csv = os.path.join(
        OUTPUT_DIR,
        f"{OUTPUT_BASENAME}_Top1_Summary.csv"
    )

    figure.savefig(
        output_png,
        dpi=DPI,
        bbox_inches="tight",
        facecolor="white"
    )

    figure.savefig(
        output_tif,
        dpi=DPI,
        bbox_inches="tight",
        facecolor="white",
        pil_kwargs={
            "compression": "tiff_lzw"
        }
    )

    plt.close(figure)

    pd.DataFrame(summary_records).to_csv(
        output_csv,
        index=False
    )

    print("Combined figure saved successfully:")
    print(output_png)
    print(output_tif)
    print(output_csv)


if __name__ == "__main__":
    main()