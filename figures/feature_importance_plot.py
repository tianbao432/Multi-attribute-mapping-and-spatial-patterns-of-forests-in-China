# -*- coding: utf-8 -*-
"""
Publication-Quality SHAP Feature Importance Visualization (Nature/Science Style)
Outputs exactly 10 high-resolution TIFF images:
  - 1 x 3x3 panel: Mean variable importance for all 9 target variables.
  - 9 x 3x3 panel: Variable importance across 9 climate zones (one for each target).
"""

import os
import string
import warnings
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

warnings.filterwarnings("ignore")

# ==============================================================================
# Configuration
# ==============================================================================
BASE_DIR   = "/data/China_mapping/fina_modelA/"
OUTPUT_DIR = "/data/China_mapping/fina_modelA/cv_plot/VA_important/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Original target variable folders/names for data loading
TARGET_VARS = ["AGE", "BA", "Trees_N", "AGB", "V", "shannon", "CV_DBH", "DBH", "H"]
CLIMATE_ZONES = ["WT", "MS", "MT", "SS", "NS", "CT", "H", "MTrop", "MidTr"]

# Academic standard display names for plot titles
TARGET_NAMES_MAP = {
    "Trees_N": "Density",
    "AGB":     "AGB",
    "V":       "GSV",
    "BA":      "BA",
    "H":       "Height",
    "DBH":     "DBH",
    "AGE":     "Age",
    "shannon": "Shannon",
    "CV_DBH":  "CV of DBH"
}

TOP_N_FEATURES = 10   # Number of top features to show per bar chart

# ==============================================================================
# Aesthetics (Nature / Science journal style) - Updated for larger, black fonts
# ==============================================================================
FONT_FAMILY = "Arial"
mpl.rcParams.update({
    "font.family"       : FONT_FAMILY,
    "font.size"         : 11,           # Base font size increased
    "axes.labelsize"    : 12,           # Axis labels size increased
    "axes.titlesize"    : 14,           # Subplot titles size increased
    "xtick.labelsize"   : 10.5,         # X-axis ticks size increased
    "ytick.labelsize"   : 10.5,         # Y-axis ticks size increased
    "text.color"        : "black",      # Forced pure black text globally
    "axes.labelcolor"   : "black",      # Forced pure black axis labels
    "xtick.color"       : "black",      # Forced pure black x ticks
    "ytick.color"       : "black",      # Forced pure black y ticks
    "figure.dpi"        : 300,
    "savefig.dpi"       : 300,
    "axes.linewidth"    : 0.8,          # Slightly thicker axes lines for clarity
    "xtick.major.width" : 0.8,
    "ytick.major.width" : 0.8,
    "xtick.direction"   : "out",
    "ytick.direction"   : "out",
    "axes.spines.top"   : False,
    "axes.spines.right" : False,
})

# Zone palette - visually distinct and professional (Bars retain color, text is black)
ZONE_PALETTE = {
    "WT"   : "#3A7DC9", "MS"   : "#2CA07A", "MT"   : "#8B6BB1",
    "SS"   : "#D4622A", "NS"   : "#C75985", "CT"   : "#5B9E3A",
    "H"    : "#7A7A7A", "MTrop": "#C98A1A", "MidTr": "#C23B3B",
}

# ==============================================================================
# Data Loading
# ==============================================================================
def load_shap_data(target_var: str) -> pd.DataFrame:
    """Load and concatenate all zone SHAP.csv for one target variable."""
    records = []
    for zone in CLIMATE_ZONES:
        path = os.path.join(BASE_DIR, target_var, zone, "SHAP.csv")
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        df.columns = [c.strip() for c in df.columns]
        if "Feature" not in df.columns or "Importance" not in df.columns:
            continue
        df["Zone"]   = zone
        df["Target"] = target_var
        records.append(df[["Feature", "Importance", "Zone", "Target"]])
    
    if not records:
        return pd.DataFrame()
    return pd.concat(records, ignore_index=True)


# ==============================================================================
# Figure 1: 9 Parameters Layout (1 Image)
# ==============================================================================
def plot_all_targets_summary(all_data: dict):
    """
    Plots a 3x3 grid showing the overall top variable importance for each of the 9 targets.
    """
    targets_present = [t for t in TARGET_VARS if t in all_data and not all_data[t].empty]
    if not targets_present:
        print("No data available for the 9-parameter summary plot.")
        return

    # Slightly enlarged figsize to accommodate larger fonts
    fig, axes = plt.subplots(3, 3, figsize=(14, 10), constrained_layout=True)
    axes = axes.flatten()

    cmap = plt.cm.viridis
    norm = Normalize(vmin=0, vmax=TOP_N_FEATURES - 1)

    for i, target in enumerate(TARGET_VARS):
        ax = axes[i]
        if target not in targets_present:
            ax.set_visible(False)
            continue

        df = all_data[target]
        mean_imp = (df.groupby("Feature")["Importance"]
                    .mean()
                    .nlargest(TOP_N_FEATURES)
                    .reset_index()
                    .sort_values("Importance", ascending=True))

        colors = [cmap(norm(rank)) for rank in range(len(mean_imp))]
        
        bars = ax.barh(range(len(mean_imp)), mean_imp["Importance"],
                       color=colors, height=0.65, edgecolor="none", alpha=0.85)

        x_max = mean_imp["Importance"].max()
        for bar, val in zip(bars, mean_imp["Importance"]):
            # Text values forced to pure black and larger font
            ax.text(val + x_max * 0.02, bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}", va="center", ha="left", fontsize=10, color="black")

        ax.set_yticks(range(len(mean_imp)))
        ax.set_yticklabels(mean_imp["Feature"])
        ax.set_xlabel("Mean |SHAP| value")
        
        panel_label = string.ascii_lowercase[i]
        # Translate the target name to its publication-ready format
        display_name = TARGET_NAMES_MAP.get(target, target)
        ax.set_title(f"{display_name}", loc="left", color="black")
        
        ax.set_xlim(0, x_max * 1.30)  # Extended limit slightly for larger text
        ax.spines["left"].set_linewidth(0.8)
        ax.spines["bottom"].set_linewidth(0.8)

    out_file = os.path.join(OUTPUT_DIR, "01_All_Parameters_Summary.tif")
    fig.savefig(out_file, format="tiff", bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    print(f"? Saved Summary Plot: {out_file}")


# ==============================================================================
# Figures 2-10: 9 Climate Zones per Parameter (9 Images)
# ==============================================================================
def plot_zones_for_target(df: pd.DataFrame, target_var: str):
    """
    Plots a 3x3 grid showing feature importance in 9 climate zones for a single target.
    """
    fig, axes = plt.subplots(3, 3, figsize=(14, 10), constrained_layout=True)
    axes = axes.flatten()

    # Apply the abbreviation directly as the suptitle
    display_name = TARGET_NAMES_MAP.get(target_var, target_var)
    fig.suptitle(display_name, fontsize=18,color="black")

    for i, zone in enumerate(CLIMATE_ZONES):
        ax = axes[i]
        sub = df[df["Zone"] == zone]
        
        if sub.empty:
            ax.text(0.5, 0.5, "No Data", ha="center", va="center", color="black", fontsize=12)
            ax.set_title(f" {zone}", loc="left", color="black")
            ax.axis('off')
            continue

        sub = (sub.sort_values("Importance", ascending=False)
                  .head(TOP_N_FEATURES)
                  .reset_index(drop=True)
                  .iloc[::-1])

        bar_color = ZONE_PALETTE.get(zone, "#555555")
        bars = ax.barh(range(len(sub)), sub["Importance"],
                       color=bar_color, height=0.65, edgecolor="none", alpha=0.85)

        x_max = sub["Importance"].max()
        for bar, val in zip(bars, sub["Importance"]):
            # Text values forced to pure black and larger font
            ax.text(val + x_max * 0.02, bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}", va="center", ha="left", fontsize=10, color="black")

        ax.set_yticks(range(len(sub)))
        ax.set_yticklabels(sub["Feature"])
        ax.set_xlabel("Mean |SHAP| value")
        
        panel_label = string.ascii_lowercase[i]
        # Title forced to pure black instead of matching the zone color
        ax.set_title(f"{zone}", loc="left", color="black")
        
        ax.set_xlim(0, x_max * 1.30)
        ax.spines["left"].set_linewidth(0.8)
        ax.spines["bottom"].set_linewidth(0.8)

    # Output file name remains unchanged to maintain system compatibility
    out_file = os.path.join(OUTPUT_DIR, f"02_Zone_Importance_{target_var}.tif")
    fig.savefig(out_file, format="tiff", bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    print(f"? Saved Zone Plot:   {out_file}")


# ==============================================================================
# Main Execution
# ==============================================================================
def main():
    print("=" * 60)
    print("SHAP Importance Visualization - Outputting 10 High-Res TIFs")
    print(f"Output Directory: {OUTPUT_DIR}")
    print("=" * 60)

    all_data = {}

    for target in TARGET_VARS:
        df = load_shap_data(target)
        if not df.empty:
            all_data[target] = df
        else:
            print(f"??  No data found for {target}")

    if not all_data:
        print("Error: No data loaded at all. Check your BASE_DIR and paths.")
        return

    plot_all_targets_summary(all_data)

    for target in TARGET_VARS:
        if target in all_data:
            plot_zones_for_target(all_data[target], target)

    print("\n" + "=" * 60)
    print("?? All 10 TIF figures successfully generated!")
    print("=" * 60)

if __name__ == "__main__":
    main()