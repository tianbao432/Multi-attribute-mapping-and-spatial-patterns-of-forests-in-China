import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import string 

# ================= 1. Top-tier Journal Aesthetics =================
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['xtick.major.width'] = 1.5
plt.rcParams['ytick.major.width'] = 1.5
plt.rcParams['xtick.major.size'] = 6
plt.rcParams['ytick.major.size'] = 6

plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16

# ================= 2. File Path Configuration =================
DATA_PATH = "/data/China_mapping/foerst_plot/PLOT_Data_Climate.csv"
OUTPUT_DIR = "/data/China_mapping/foerst_plot"

OUTPUT_PDF = os.path.join(OUTPUT_DIR, "Forest_Parameters_TopTier.pdf")
OUTPUT_PNG = os.path.join(OUTPUT_DIR, "Forest_Parameters_TopTier.tif")

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Load data
df = pd.read_csv(DATA_PATH)

# ================= 3. Parameters, Formatting & Data Prep =================
VAR_MAP = {
    'AGB': r'AGB ($Mg\ ha^{-1}$)',
    'AGE': 'Age (years)',
    'BA': r'BA ($m^2\ ha^{-1}$)',
    'CV_DBH': 'CV of DBH',
    'DBH': 'DBH (cm)',
    'H': 'Height (m)',
    'N': r'Density ($Stems\ ha^{-1}$)',
    'Shannon.Index': 'Shannon Index',
    'V': r'GSV ($m^3\ ha^{-1}$)'
}
variables = [
    'AGB', 'AGE', 'BA',
    'CV_DBH', 'DBH', 'H',
    'N', 'Shannon.Index', 'V'
]

climate_order = ['H', 'CT', 'MT', 'WT', 'NS', 'MS', 'SS', 'MTrop', 'MidTr']
existing_climates = df['climate_abbr'].dropna().unique()
plot_order = [c for c in climate_order if c in existing_climates]

# ================= 4. Plotting the 3x3 Advanced Grid =================
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(16, 8), dpi=300)
axes = axes.flatten()

for i, var in enumerate(variables):
    ax = axes[i]

    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle='--', linewidth=0.6, color='#CCCCCC', alpha=0.7)
    ax.xaxis.grid(False)

    valid_counts = (
        df.loc[df[var].notna() & df['climate_abbr'].notna()]
        .groupby('climate_abbr')
        .size()
        .reindex(plot_order, fill_value=0)
    )

    # Calculate the upper whisker tip for placing the text dynamically
    whisker_tops = {}
    for c in plot_order:
        group_data = df.loc[(df['climate_abbr'] == c) & df[var].notna(), var]
        if not group_data.empty:
            q1 = group_data.quantile(0.25)
            q3 = group_data.quantile(0.75)
            iqr = q3 - q1
            # Maximum value within 1.5 * IQR from Q3 (excluding fliers)
            top_val = group_data[group_data <= q3 + 1.5 * iqr].max()
            whisker_tops[c] = top_val
        else:
            whisker_tops[c] = np.nan

    # Draw boxplot with a single black color
    sns.boxplot(
        x='climate_abbr',
        y=var,
        data=df,
        order=plot_order,
        color='#333333', # Solid soft black color
        width=0.6,
        ax=ax,
        zorder=3,
        showfliers=False,
        boxprops=dict(alpha=0.95, edgecolor='#111111', linewidth=1.5),
        medianprops=dict(color='white', linewidth=2.0), # Changed to white for visibility
        whiskerprops=dict(color='#111111', linewidth=1.5),
        capprops=dict(color='#111111', linewidth=1.5)
    )

    means = (
        df.groupby('climate_abbr')[var]
        .mean()
        .reindex(plot_order)
        .values
    )

    ax.scatter(
        range(len(plot_order)),
        means,
        marker='D',
        color='white',
        edgecolor='#111111',
        s=60, 
        linewidth=1.2,
        zorder=10,
        label='Mean' if i == 0 else "" 
    )

    # Expand Y-axis significantly at the top to fit the tilted text
    ymin, ymax = ax.get_ylim()
    yrange = ymax - ymin
    ax.set_ylim(ymin - yrange * 0.05, ymax + yrange * 0.28)

    # Add sample sizes tilted above whiskers
    for j, c in enumerate(plot_order):
        count = int(valid_counts.loc[c])
        if count > 0 and not np.isnan(whisker_tops[c]):
            y_pos = whisker_tops[c] + (yrange * 0.02)
            ax.text(
                x=j - 0.15, 
                y=y_pos, 
                s=f"n={count}",
                rotation=90, 
                ha='left', 
                va='bottom', 
                fontsize=12, 
                color='black',        
                #fontweight='bold',    
                fontstyle='italic', 
                zorder=15
            )

    if i < 3:
        line_ymin, line_ymax = -0.15, 1.0
    elif i < 6:
        line_ymin, line_ymax = -0.15, 1.15
    else:
        line_ymin, line_ymax = 0.0, 1.15

    for j in range(len(plot_order)):
        ax.axvline(
            x=j,
            color='#999999',
            linestyle='--',
            linewidth=1.0,
            alpha=0.4, 
            zorder=1,
            ymin=line_ymin,
            ymax=line_ymax,
            clip_on=False
        )

    ax.set_xticks(range(len(plot_order)))
    ax.set_xlabel('')

    formatted_label = VAR_MAP.get(var, var)
    ax.set_ylabel(formatted_label, labelpad=8)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#111111')

    if i < 6:
        ax.set_xticklabels([])
        ax.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
        ax.spines['bottom'].set_visible(False)
    else:
        ax.set_xticklabels(plot_order, rotation=40, ha='right', fontsize=12)
        ax.tick_params(axis='x', which='both', bottom=True, labelbottom=True)
        ax.spines['bottom'].set_visible(True)
        ax.spines['bottom'].set_color('#111111')

# ================= 5. Layout Optimization and Saving =================
plt.subplots_adjust(
    left=0.06,
    right=0.98,
    top=0.94,
    bottom=0.10,
    hspace=0.15,
    wspace=0.30 
)

plt.savefig(OUTPUT_PDF, bbox_inches='tight', dpi=300)
plt.savefig(OUTPUT_PNG, bbox_inches='tight', dpi=600)
plt.close()

print(f"Plotting complete. Advanced multi-panel plot saved in: {OUTPUT_DIR}")