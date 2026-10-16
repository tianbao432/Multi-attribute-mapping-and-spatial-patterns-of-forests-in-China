#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import matplotlib

# Required for Linux servers without a display.
matplotlib.use("Agg")

import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator, ScalarFormatter


# =============================================================================
# 1. Paths and layout
# =============================================================================
BASE_DIR = Path(
    os.environ.get(
        "CHINA_CLIMATE_STATS_DIR",
        "/data/China_mapping/Climate_statistics",
    )
)

DONUT_DATA_PATH = (
    BASE_DIR / "statistic" / "Absolute_Totals_Summary_All_Parameters.csv"
)
BOXPLOT_DATA_DIR = BASE_DIR
OUTPUT_DIR = BASE_DIR / "Figures"
OUTPUT_STEM = OUTPUT_DIR / "Figure_Combined_Comprehensive_Analysis_premium_black"

FIGURE_WIDTH_INCHES = 7.60
FIGURE_HEIGHT_INCHES = 8.25
RASTER_DPI = 600

# =============================================================================
# 1. Variable plotting order
# =============================================================================

# Variable order for donut charts
DONUT_VARIABLE_ORDER = [
    "AGB",
    "BA",
    "Trees_N",
    "V",
]

# Variable order for boxplots
BOX_VARIABLE_ORDER = [
    "AGB",
    "AGE",
    "BA",
    "CV_DBH",
    "DBH",
    "H",
    "Trees_N",
    "shannon",
    "V",
]

# Optional explicit top-to-bottom order for climate zones in the boxplots.
# Leave empty to use alphabetical order with All first.
ZONE_ORDER: List[str] = []

# Donut-label thresholds.
MIN_PERCENT_TO_LABEL = 1.0
MIN_PERCENT_FOR_INSIDE_LABEL = 7.5
OUTSIDE_LABEL_MIN_GAP = 0.135


# =============================================================================
# 2. Variable names and units
# =============================================================================

NAME_MAP: Mapping[str, str] = {
    "AGB": "AGB",
    "AGE": "Age",
    "BA": "BA",
    "CV_DBH": "CV DBH",
    "DBH": "DBH",
    "H": "Height",
    "Trees_N": "Density",
    "shannon": "Shannon",
    "V": "GSV",
}
NAME_MAP_DONUT: Mapping[str, str] = {
    "V": "GSV",
    "AGB": "AGB",
    "BA": "BA",
    "Trees_N": "Trees",
}
UNIT_MAP_DONUT: Mapping[str, str] = {
    "AGB": r"$\mathrm{Mg}$",
    "BA": r"$\mathrm{m^2}$",
    "Trees_N": r"$\mathrm{stems}$",
    "V": r"$\mathrm{m^3}$",
}

UNIT_MAP_BOX: Mapping[str, str] = {
    "AGB": r"$\mathrm{Mg\,ha^{-1}}$",
    "AGE": r"$\mathrm{years}$",
    "BA": r"$\mathrm{m^2\,ha^{-1}}$",
    "CV_DBH": "",
    "DBH": r"$\mathrm{cm}$",
    "H": r"$\mathrm{m}$",
    "Trees_N": r"$\mathrm{stems\,ha^{-1}}$",
    "shannon": "",
    "V": r"$\mathrm{m^3\,ha^{-1}}$",
}


# =============================================================================
# 3. Visual system
# =============================================================================
# Restrained, low-saturation climate-zone palette.
ZONE_PALETTE = [
    "#E64B35",  
    "#4DBBD5", 
    "#00A087", 
    "#3C5488",  
    "#F39B7F", 
    "#8491B4",  
    "#91D1C2", 
    "#FFD125", 
    "#7E6148",  
    "#B09C85",  
    "#DC0000", 
    "#5F559B", 
]
# All text is pure black, as requested.
TEXT_COLOR = "#000000"
SECONDARY_TEXT_COLOR = "#000000"

FIGURE_BACKGROUND = "#FFFFFF"
SPINE_COLOR = "#8B9195"
GRID_COLOR = "#DDE1E3"
ROW_BAND_COLOR = "#F4F5F4"
SEPARATOR_COLOR = "#D7DADD"

ZONE_BOX_FACE = "#00A087"
ZONE_BOX_EDGE = "#5B7471"
ALL_BOX_FACE = "#334F58"
ALL_BOX_EDGE = "#263D44"
MEAN_FACE = "#FFFFFF"


def choose_sans_serif_font() -> str:
    """Choose a clean publication font available on the current system."""
    preferred = ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"]
    available = {font.name for font in font_manager.fontManager.ttflist}
    for name in preferred:
        if name in available:
            return name
    return "DejaVu Sans"


def configure_matplotlib() -> None:
    """Set typography and export-safe rendering parameters."""
    selected_font = choose_sans_serif_font()

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [selected_font],
            "font.weight": "normal",
            "font.size": 8.0,
            "text.color": TEXT_COLOR,
            "axes.titlesize": 8.9,
            "axes.titleweight": "normal",
            "axes.titlecolor": TEXT_COLOR,
            "axes.labelsize": 7.8,
            "axes.labelweight": "normal",
            "axes.labelcolor": TEXT_COLOR,
            "axes.edgecolor": SPINE_COLOR,
            "axes.linewidth": 0.58,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.4,
            "xtick.color": TEXT_COLOR,
            "ytick.color": TEXT_COLOR,
            "xtick.major.size": 2.7,
            "ytick.major.size": 2.7,
            "xtick.major.width": 0.58,
            "ytick.major.width": 0.58,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "legend.fontsize": 6.9,
            "legend.frameon": False,
            "legend.labelcolor": TEXT_COLOR,
            "figure.facecolor": FIGURE_BACKGROUND,
            "axes.facecolor": FIGURE_BACKGROUND,
            "savefig.facecolor": FIGURE_BACKGROUND,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "mathtext.fontset": "dejavusans",
            "axes.unicode_minus": False,
        }
    )


# =============================================================================
# 4. General helpers
# =============================================================================
def clean_variable_name(raw_name: str) -> str:
    return raw_name[len("Total_") :] if raw_name.startswith("Total_") else raw_name


def display_name(raw_name: str) -> str:
    clean_name = clean_variable_name(raw_name)
    return NAME_MAP.get(clean_name, clean_name)


def variable_unit(raw_name: str, context: str) -> str:
    clean_name = clean_variable_name(raw_name)
    if context == "donut":
        return UNIT_MAP_DONUT.get(clean_name, "")
    if context == "box":
        return UNIT_MAP_BOX.get(clean_name, "")
    raise ValueError(f"Unknown unit context: {context}")


def format_scientific(value: float) -> str:
    """Format totals with compact scientific notation."""
    if pd.isna(value):
        return "NA"
    if np.isclose(value, 0.0):
        return "0"

    sign = "?" if value < 0 else ""
    absolute_value = abs(float(value))
    exponent = int(np.floor(np.log10(absolute_value)))
    coefficient = absolute_value / (10.0**exponent)

    if -1 <= exponent <= 3:
        if absolute_value >= 100:
            return f"{sign}{absolute_value:,.0f}"
        if absolute_value >= 10:
            return f"{sign}{absolute_value:,.1f}"
        return f"{sign}{absolute_value:,.2f}"

    return rf"{sign}{coefficient:.2f}$\times10^{{{exponent}}}$"


def spread_label_positions(
    raw_y: Sequence[float],
    lower: float = -0.98,
    upper: float = 0.98,
    minimum_gap: float = OUTSIDE_LABEL_MIN_GAP,
) -> np.ndarray:
    """Vertically separate outside labels while preserving angular order."""
    values = np.asarray(raw_y, dtype=float)
    if values.size <= 1:
        return np.clip(values, lower, upper)

    order = np.argsort(values)
    sorted_y = np.clip(values[order], lower, upper)

    for index in range(1, len(sorted_y)):
        sorted_y[index] = max(sorted_y[index], sorted_y[index - 1] + minimum_gap)

    if sorted_y[-1] > upper:
        sorted_y -= sorted_y[-1] - upper

    for index in range(len(sorted_y) - 2, -1, -1):
        sorted_y[index] = min(sorted_y[index], sorted_y[index + 1] - minimum_gap)

    if sorted_y[0] < lower:
        sorted_y += lower - sorted_y[0]

    adjusted = np.empty_like(sorted_y)
    adjusted[order] = sorted_y
    return adjusted


def order_zones(zones: Sequence[str], all_label: str = "All") -> List[str]:
    unique_zones = list(dict.fromkeys(str(zone) for zone in zones))
    non_all = [zone for zone in unique_zones if zone != all_label]

    if ZONE_ORDER:
        preferred = [zone for zone in ZONE_ORDER if zone in non_all]
        remaining = sorted(zone for zone in non_all if zone not in preferred)
        non_all = preferred + remaining
    else:
        non_all = sorted(non_all)

    return ([all_label] if all_label in unique_zones else []) + non_all


def style_numeric_axis(ax: plt.Axes) -> None:
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-2, 3))
    formatter.set_useOffset(False)
    ax.xaxis.set_major_formatter(formatter)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))

    offset_text = ax.xaxis.get_offset_text()
    offset_text.set_size(6.1)
    offset_text.set_color(TEXT_COLOR)
    offset_text.set_fontweight("normal")


def format_box_title(variable: str) -> str:
    name = display_name(variable)
    unit = variable_unit(variable, context="box")
    return f"{name} ({unit})" if unit else name


def style_box_axis(ax: plt.Axes, show_y_labels: bool) -> None:
    ax.set_axisbelow(True)
    ax.grid(axis="x", color=GRID_COLOR, linewidth=0.47, alpha=0.84)
    ax.grid(axis="y", visible=False)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color(SPINE_COLOR)
    ax.spines["left"].set_color(SPINE_COLOR)
    ax.spines["bottom"].set_linewidth(0.58)
    ax.spines["left"].set_linewidth(0.58)

    ax.tick_params(axis="x", pad=1.8, labelsize=7.05, colors=TEXT_COLOR)
    if show_y_labels:
        ax.tick_params(axis="y", pad=2.7, length=0, labelsize=7.55, colors=TEXT_COLOR)
    else:
        ax.tick_params(axis="y", left=False, labelleft=False)
        ax.spines["left"].set_visible(False)

    style_numeric_axis(ax)


# =============================================================================
# 5. Data loading and validation
# =============================================================================
def load_donut_data() -> Tuple[pd.DataFrame, List[str], List[str]]:
    if not DONUT_DATA_PATH.exists():
        raise FileNotFoundError(f"Donut data not found: {DONUT_DATA_PATH}")

    data = pd.read_csv(DONUT_DATA_PATH, index_col=0)
    data.index = data.index.astype(str).str.strip()
    data.columns = data.columns.astype(str).str.strip()

    if "National" not in data.index:
        raise ValueError(
            f"The donut table must contain a 'National' row: {DONUT_DATA_PATH}"
        )

    preferred_columns = [f"Total_{name}" for name in DONUT_VARIABLE_ORDER]
    target_columns = [column for column in preferred_columns if column in data.columns]

    if not target_columns:
        target_columns = [
            column for column in data.columns if column.startswith("Total_")
        ][:4]

    if not target_columns:
        raise ValueError("No 'Total_*' columns were found in the donut table.")

    target_columns = target_columns[:4]
    zones = [index for index in data.index if index != "National"]
    if not zones:
        raise ValueError("No climate-zone rows were found in the donut table.")

    for column in target_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    return data, target_columns, zones


def load_boxplot_data() -> Tuple[Dict[str, pd.DataFrame], List[str], List[str]]:
    stats_files = sorted(BOXPLOT_DATA_DIR.glob("*_boxplot_stats.csv"))
    if not stats_files:
        raise FileNotFoundError(
            f"No '*_boxplot_stats.csv' files were found in: {BOXPLOT_DATA_DIR}"
        )

    data_by_zone: Dict[str, pd.DataFrame] = {}
    for file_path in stats_files:
        raw_zone = file_path.name.replace("_boxplot_stats.csv", "")
        zone_name = "All" if raw_zone == "National" else raw_zone

        frame = pd.read_csv(file_path, index_col=0)
        frame.index = frame.index.astype(str).str.strip()
        frame.columns = frame.columns.astype(str).str.strip()
        frame = frame.apply(pd.to_numeric, errors="coerce")
        data_by_zone[zone_name] = frame

    zones_top_to_bottom = order_zones(list(data_by_zone))

    available_variables = set().union(
        *(set(frame.columns) for frame in data_by_zone.values())
    )
    variables = [name for name in BOX_VARIABLE_ORDER if name in available_variables]

    if not variables:
        first_frame = next(iter(data_by_zone.values()))
        variables = list(first_frame.columns[:9])

    return data_by_zone, zones_top_to_bottom, variables[:9]


def find_stat_row(frame: pd.DataFrame, candidates: Sequence[str]) -> str | None:
    index_lookup = {str(index).strip().lower(): str(index) for index in frame.index}
    for candidate in candidates:
        matched = index_lookup.get(candidate.strip().lower())
        if matched is not None:
            return matched
    return None


def extract_box_statistics(
    frame: pd.DataFrame,
    variable: str,
) -> Dict[str, float] | None:
    if variable not in frame.columns:
        return None

    row_aliases = {
        "med": ["50%", "median", "q50"],
        "q1": ["25%", "q1", "q25"],
        "q3": ["75%", "q3", "q75"],
        "whislo": ["min", "minimum", "whislo", "lower_whisker"],
        "whishi": ["max", "maximum", "whishi", "upper_whisker"],
        "mean": ["mean", "average"],
    }

    result: Dict[str, float] = {}
    for key, aliases in row_aliases.items():
        row = find_stat_row(frame, aliases)
        result[key] = np.nan if row is None else float(frame.loc[row, variable])

    required = ["med", "q1", "q3", "whislo", "whishi"]
    if any(not np.isfinite(result[key]) for key in required):
        return None

    result["q1"], result["q3"] = sorted([result["q1"], result["q3"]])
    result["whislo"] = min(result["whislo"], result["q1"])
    result["whishi"] = max(result["whishi"], result["q3"])
    result["med"] = float(np.clip(result["med"], result["q1"], result["q3"]))
    return result


# =============================================================================
# 6. Panel a: donut charts
# =============================================================================
def add_donut_annotations(
    ax: plt.Axes,
    wedges: Sequence,
    values: np.ndarray,
) -> None:
    """Add black percentage labels with collision-reduced positioning."""
    total = float(np.nansum(values))
    if not np.isfinite(total) or total <= 0:
        return

    left_items: List[Dict[str, float]] = []
    right_items: List[Dict[str, float]] = []

    for wedge, value in zip(wedges, values):
        percentage = 100.0 * float(value) / total
        if percentage < MIN_PERCENT_TO_LABEL:
            continue

        angle_deg = 0.5 * (wedge.theta1 + wedge.theta2)
        angle_rad = np.deg2rad(angle_deg)
        x_unit = float(np.cos(angle_rad))
        y_unit = float(np.sin(angle_rad))

        if percentage >= MIN_PERCENT_FOR_INSIDE_LABEL:
            # Place only sufficiently large percentages on the ring.
            radius = 0.845
            label = ax.text(
                radius * x_unit,
                radius * y_unit,
                f"{percentage:.1f}%",
                ha="center",
                va="center",
                fontsize=6.75,
                fontweight="normal",
                color=TEXT_COLOR,
                zorder=6,
                clip_on=False,
            )
            # A subtle white halo keeps black labels readable on every color
            # without introducing visible boxes.
            label.set_path_effects(
                [
                    path_effects.Stroke(linewidth=2.4, foreground="#FFFFFF"),
                    path_effects.Normal(),
                ]
            )
        else:
            item = {
                "percentage": percentage,
                "x_edge": 0.99 * x_unit,
                "y_edge": 0.99 * y_unit,
                "raw_y": 1.05 * y_unit,
            }
            (right_items if x_unit >= 0 else left_items).append(item)

    for side, items in (("left", left_items), ("right", right_items)):
        if not items:
            continue

        adjusted_y = spread_label_positions([item["raw_y"] for item in items])
        x_text = -1.13 if side == "left" else 1.13
        alignment = "right" if side == "left" else "left"

        for item, y_text in zip(items, adjusted_y):
            annotation = ax.annotate(
                f"{item['percentage']:.1f}%",
                xy=(item["x_edge"], item["y_edge"]),
                xytext=(x_text, y_text),
                ha=alignment,
                va="center",
                fontsize=6.65,
                fontweight="normal",
                color=TEXT_COLOR,
                annotation_clip=False,
                arrowprops={
                    "arrowstyle": "-",
                    "color": "#9A9FA3",
                    "linewidth": 0.48,
                    "shrinkA": 1.0,
                    "shrinkB": 1.0,
                    "connectionstyle": "angle3,angleA=0,angleB=90",
                },
            )
            annotation.set_path_effects(
                [
                    path_effects.Stroke(linewidth=2.0, foreground="#FFFFFF"),
                    path_effects.Normal(),
                ]
            )


def draw_donut_panel(
    fig: plt.Figure,
    subplot_spec,
    data: pd.DataFrame,
    target_columns: Sequence[str],
    zones: Sequence[str],
    zone_colors: Mapping[str, str],
) -> List[plt.Axes]:
    """Draw four balanced donut charts in a compact horizontal row."""
    subgrid = subplot_spec.subgridspec(1, len(target_columns), wspace=0.025)

    axes: List[plt.Axes] = []
    for column_index, column in enumerate(target_columns):
        ax = fig.add_subplot(subgrid[0, column_index])
        axes.append(ax)

        zone_series = data.loc[list(zones), column].astype(float)
        valid_mask = np.isfinite(zone_series.to_numpy()) & (zone_series.to_numpy() > 0)
        plot_zones = np.asarray(zones, dtype=object)[valid_mask]
        values = zone_series.to_numpy(dtype=float)[valid_mask]

        if values.size == 0 or np.nansum(values) <= 0:
            ax.text(
                0.5,
                0.5,
                "No positive data",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=6.6,
                color=TEXT_COLOR,
            )
            ax.axis("off")
            continue

        colors = [zone_colors[str(zone)] for zone in plot_zones]
        wedges, _ = ax.pie(
            values,
            colors=colors,
            startangle=90,
            counterclock=False,
            radius=1.0,
            normalize=True,
            wedgeprops={
                "width": 0.275,
                "edgecolor": FIGURE_BACKGROUND,
                "linewidth": 0.78,
                "antialiased": True,
            },
        )

        add_donut_annotations(ax, wedges, values)

        national_total = float(data.loc["National", column])
        zone_total = float(np.nansum(values))
        if np.isfinite(national_total) and national_total != 0:
            relative_difference = abs(zone_total - national_total) / abs(national_total)
            if relative_difference > 0.02:
                warnings.warn(
                    f"For {column}, climate-zone sum ({zone_total:g}) differs from "
                    f"National ({national_total:g}) by {relative_difference:.1%}."
                )

        name = display_name(column)
        unit = variable_unit(column, context="donut")
        total_text = format_scientific(national_total)

        ax.text(
            0,
            0.205,
            name,
            ha="center",
            va="center",
            fontsize=9.15,
            fontweight="normal",
            color=TEXT_COLOR,
        )
        ax.text(
            0,
            -0.005,
            total_text,
            ha="center",
            va="center",
            fontsize=8.15,
            fontweight="normal",
            color=TEXT_COLOR,
        )
        ax.text(
            0,
            -0.215,
            unit,
            ha="center",
            va="center",
            fontsize=6.45,
            fontweight="normal",
            color=TEXT_COLOR,
        )

        ax.set_aspect("equal")
        ax.set_xlim(-1.22, 1.22)
        ax.set_ylim(-1.13, 1.13)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    return axes


# =============================================================================
# 7. Panel b: summary boxplots
# =============================================================================
def draw_single_boxplot(
    ax: plt.Axes,
    variable: str,
    data_by_zone: Mapping[str, pd.DataFrame],
    zones_top_to_bottom: Sequence[str],
    show_y_labels: bool,
) -> None:
    """Draw one horizontal summary boxplot panel."""
    zones_bottom_to_top = list(reversed(zones_top_to_bottom))
    position_by_zone = {
        zone: position for position, zone in enumerate(zones_bottom_to_top)
    }

    valid_zones: List[str] = []
    bxp_stats: List[Dict[str, float | str]] = []
    positions: List[int] = []
    means: List[float] = []

    for zone in zones_bottom_to_top:
        statistics = extract_box_statistics(data_by_zone[zone], variable)
        if statistics is None:
            continue

        valid_zones.append(zone)
        positions.append(position_by_zone[zone])
        means.append(statistics["mean"])
        bxp_stats.append(
            {
                "label": zone,
                "med": statistics["med"],
                "q1": statistics["q1"],
                "q3": statistics["q3"],
                "whislo": statistics["whislo"],
                "whishi": statistics["whishi"],
            }
        )

    if not bxp_stats:
        ax.text(
            0.5,
            0.5,
            "No valid summary statistics",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=6.7,
            color=TEXT_COLOR,
        )
    else:
        artists = ax.bxp(
            bxp_stats,
            positions=positions,
            widths=0.50,
            vert=False,
            patch_artist=True,
            showfliers=False,
            manage_ticks=False,
            zorder=3,
        )

        for patch, zone in zip(artists["boxes"], valid_zones):
            if zone == "All":
                patch.set_facecolor(ALL_BOX_FACE)
                patch.set_edgecolor(ALL_BOX_EDGE)
                patch.set_alpha(0.98)
            else:
                patch.set_facecolor(ZONE_BOX_FACE)
                patch.set_edgecolor(ZONE_BOX_EDGE)
                patch.set_alpha(0.87)
            patch.set_linewidth(0.62)

        for median, zone in zip(artists["medians"], valid_zones):
            # Median lines are graphical elements rather than text.
            median.set_color("#FFFFFF" if zone == "All" else "#303030")
            median.set_linewidth(1.08)
            median.set_solid_capstyle("butt")

        for whisker in artists["whiskers"]:
            whisker.set_color(ZONE_BOX_EDGE)
            whisker.set_linewidth(0.58)

        for cap in artists["caps"]:
            cap.set_color(ZONE_BOX_EDGE)
            cap.set_linewidth(0.58)

        finite_mean_mask = np.isfinite(np.asarray(means, dtype=float))
        if np.any(finite_mean_mask):
            mean_values = np.asarray(means, dtype=float)[finite_mean_mask]
            mean_positions = np.asarray(positions, dtype=float)[finite_mean_mask]
            ax.scatter(
                mean_values,
                mean_positions,
                marker="o",
                s=14,
                facecolor=MEAN_FACE,
                edgecolor="#000000",
                linewidth=0.54,
                zorder=5,
                clip_on=True,
            )

    ax.set_ylim(-0.52, len(zones_bottom_to_top) - 0.48)
    ax.set_yticks(range(len(zones_bottom_to_top)))
    ax.set_yticklabels(zones_bottom_to_top, color=TEXT_COLOR)

    ax.set_title(
        format_box_title(variable),
        pad=3.9,
        color=TEXT_COLOR,
        fontsize=9.0,
        fontweight="normal",
    )
    ax.set_xlabel("")
    ax.margins(x=0.038)

    style_box_axis(ax, show_y_labels=show_y_labels)


def draw_boxplot_panel(
    fig: plt.Figure,
    subplot_spec,
    data_by_zone: Mapping[str, pd.DataFrame],
    zones_top_to_bottom: Sequence[str],
    variables: Sequence[str],
) -> List[plt.Axes]:
    subgrid = subplot_spec.subgridspec(3, 3, wspace=0.185, hspace=0.335)

    axes: List[plt.Axes] = []
    first_axis_by_row: Dict[int, plt.Axes] = {}

    for variable_index in range(9):
        row, column = divmod(variable_index, 3)
        shared_axis = first_axis_by_row.get(row) if column > 0 else None
        ax = fig.add_subplot(subgrid[row, column], sharey=shared_axis)
        axes.append(ax)

        if column == 0:
            first_axis_by_row[row] = ax

        if variable_index >= len(variables):
            ax.axis("off")
            continue

        draw_single_boxplot(
            ax=ax,
            variable=variables[variable_index],
            data_by_zone=data_by_zone,
            zones_top_to_bottom=zones_top_to_bottom,
            show_y_labels=(column == 0),
        )

    return axes


# =============================================================================
# 8. Panel labels, top legend, and export
# =============================================================================
def add_panel_letter(ax: plt.Axes, panel_letter: str) -> None:
    """Add a bold black panel letter without a following title."""
    ax.axis("off")
    ax.text(
        -0.05,
        0.50,
        panel_letter,
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=12.4,
        fontweight="bold",
        color=TEXT_COLOR,
    )


def filter_legend_zones_by_percentage(
    data: pd.DataFrame,
    target_columns: Sequence[str],
    zones: Sequence[str],
    minimum_percentage: float = MIN_PERCENT_TO_LABEL,
) -> List[str]:
    """Keep zones contributing at least the threshold in one donut variable."""
    visible_zones: List[str] = []

    for zone in zones:
        keep_zone = False
        for column in target_columns:
            values = pd.to_numeric(data.loc[list(zones), column], errors="coerce")
            total = float(np.nansum(values.to_numpy(dtype=float)))
            zone_value = float(pd.to_numeric(data.loc[zone, column], errors="coerce"))

            if (
                np.isfinite(zone_value)
                and zone_value > 0
                and np.isfinite(total)
                and total > 0
                and 100.0 * zone_value / total >= minimum_percentage
            ):
                keep_zone = True
                break

        if keep_zone:
            visible_zones.append(zone)

    return visible_zones


def add_single_row_zone_legend(
    ax: plt.Axes,
    zones: Sequence[str],
    zone_colors: Mapping[str, str],
) -> None:
    """Draw the climate-zone legend in exactly one horizontal row."""
    ax.axis("off")

    handles = [
        Patch(facecolor=zone_colors[zone], edgecolor="none", label=zone)
        for zone in zones
    ]

    total_characters = sum(len(str(zone)) for zone in zones)
    if len(zones) >= 10 or total_characters > 120:
        legend_fontsize = 5.9
        column_spacing = 0.56
    elif len(zones) >= 8 or total_characters > 90:
        legend_fontsize = 8.45
        column_spacing = 0.78
    else:
        legend_fontsize = 7.05
        column_spacing = 1.02

    legend = ax.legend(
        handles=handles,
        loc="center",
        bbox_to_anchor=(0.0, 0.0, 1.0, 1.0),
        mode="expand",
        ncol=len(handles),
        fontsize=legend_fontsize,
        columnspacing=column_spacing,
        handlelength=1.08,
        handleheight=0.72,
        handletextpad=0.36,
        borderaxespad=0.0,
        labelspacing=0.0,
        frameon=False,
    )

    for text in legend.get_texts():
        text.set_color(TEXT_COLOR)
        text.set_fontweight("normal")


def add_subtle_separator(ax: plt.Axes) -> None:
    """Add a thin separator above panel b for a cleaner visual transition."""
    ax.axis("off")
    ax.plot(
        [0.0, 1.0],
        [0.52, 0.52],
        transform=ax.transAxes,
        color=SEPARATOR_COLOR,
        linewidth=0.55,
        solid_capstyle="butt",
        clip_on=False,
    )


def save_figure(fig: plt.Figure) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    common_kwargs = {
        "bbox_inches": "tight",
        "pad_inches": 0.018,
        "facecolor": FIGURE_BACKGROUND,
        "edgecolor": "none",
        "transparent": False,
    }

    fig.savefig(OUTPUT_STEM.with_suffix(".png"), dpi=RASTER_DPI, **common_kwargs)
    fig.savefig(
        OUTPUT_STEM.with_suffix(".pdf"),
        format="pdf",
        metadata={
            "Title": "National forest totals and climate-zone distributions",
            "Creator": "Matplotlib",
        },
        **common_kwargs,
    )
    fig.savefig(OUTPUT_STEM.with_suffix(".svg"), format="svg", **common_kwargs)
    fig.savefig(
        OUTPUT_STEM.with_suffix(".tif"),
        format="tiff",
        dpi=RASTER_DPI,
        pil_kwargs={"compression": "tiff_lzw"},
        **common_kwargs,
    )


# =============================================================================
# 9. Figure assembly
# =============================================================================
def main() -> None:
    configure_matplotlib()

    donut_data, donut_columns, donut_zones = load_donut_data()
    box_data, box_zones_top_to_bottom, box_variables = load_boxplot_data()

    zone_colors = {
        zone: ZONE_PALETTE[index % len(ZONE_PALETTE)]
        for index, zone in enumerate(donut_zones)
    }

    fig = plt.figure(figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES), dpi=150)

    # Layout structure:
    # a / donuts / single-row legend / separator / b / 3¡Á3 boxplots
    main_grid = fig.add_gridspec(
        nrows=6,
        ncols=1,
        height_ratios=[0.105, 1.82, 0.255, 0.045, 0.105, 5.00],
        hspace=0.018,
        left=0.086,
        right=0.993,
        top=0.990,
        bottom=0.040,
    )

    panel_a_axis = fig.add_subplot(main_grid[0])
    add_panel_letter(panel_a_axis, "a")

    draw_donut_panel(
        fig=fig,
        subplot_spec=main_grid[1],
        data=donut_data,
        target_columns=donut_columns,
        zones=donut_zones,
        zone_colors=zone_colors,
    )

    legend_zones = filter_legend_zones_by_percentage(
        data=donut_data,
        target_columns=donut_columns,
        zones=donut_zones,
        minimum_percentage=MIN_PERCENT_TO_LABEL,
    )

    zone_legend_axis = fig.add_subplot(main_grid[2])
    add_single_row_zone_legend(
        zone_legend_axis,
        zones=legend_zones,
        zone_colors=zone_colors,
    )

    separator_axis = fig.add_subplot(main_grid[3])
    add_subtle_separator(separator_axis)

    panel_b_axis = fig.add_subplot(main_grid[4])
    add_panel_letter(panel_b_axis, "b")

    draw_boxplot_panel(
        fig=fig,
        subplot_spec=main_grid[5],
        data_by_zone=box_data,
        zones_top_to_bottom=box_zones_top_to_bottom,
        variables=box_variables,
    )

    save_figure(fig)
    plt.close(fig)

    print("Premium all-black-typography composite figure generated successfully.")
    print(f"PNG : {OUTPUT_STEM.with_suffix('.png')}")
    print(f"PDF : {OUTPUT_STEM.with_suffix('.pdf')}")
    print(f"SVG : {OUTPUT_STEM.with_suffix('.svg')}")
    print(f"TIFF: {OUTPUT_STEM.with_suffix('.tif')}")


if __name__ == "__main__":
    main()
