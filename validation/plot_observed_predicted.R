# ==============================================================================
# Top-Journal Style Scatter Plot for Observed vs Predicted (RSE Style)
# China Climate Zones Edition
# Features: 
#   - Plotmath Expressions for Units (100% immune to encoding issues)
#   - Boxed Annotations to prevent overlap with scatter points
#   - Percentile Rank Mapping with "MIN" tie-breaking (Solves ghosting)
#   - Perceptually Uniform Gradient (Turbo) & Linear Trend Lines
#   - Shared Bottom Legend (Placed strictly below the X-axis label)
#   - Plain font for legend text/title
#   - Metrics: R, RMSE, rRMSE (%)
# ==============================================================================

suppressPackageStartupMessages({
  library(tidyverse)
  library(ggplot2)
  library(cowplot)
  library(scales)
  library(grid)
})

# Check dependencies
HAS_POINTDENSITY <- requireNamespace("ggpointdensity", quietly = TRUE)
HAS_HEXBIN       <- requireNamespace("hexbin", quietly = TRUE)
HAS_RAGG         <- requireNamespace("ragg", quietly = TRUE)

# ==============================================================================
# 0. Global settings
# ==============================================================================

FONT_FAMILY <- "Arial"
base_family_use <- FONT_FAMILY

PLOT_MAX_POINTS <- 120000

USE_ROBUST_AXIS <- TRUE
AXIS_QUANTILE   <- 0.995

# Height adjusted for bottom legend layout
FIG_WIDTH  <- 7.15
FIG_HEIGHT <- 7.50 
FIG_DPI    <- 600

# ==============================================================================
# 1. Paths and metadata
# ==============================================================================

BASE_DIR    <- "/data/China_mapping/fina_modelA/"
OUTPUT_ROOT <- "/data/China_mapping/fina_modelA/cv_plot/"

VAR_META <- list(
  "Trees_N" = list(x_lbl = expression("Observed Density (stems" ~ ha^{-1} * ")"),
                   y_lbl = expression("Predicted Density (stems" ~ ha^{-1} * ")"), is_large = TRUE),
  "AGB"     = list(x_lbl = expression("Observed AGB (Mg" ~ ha^{-1} * ")"),
                   y_lbl = expression("Predicted AGB (Mg" ~ ha^{-1} * ")"), is_large = TRUE),
  "V"       = list(x_lbl = expression("Observed GSV (" * m^3 ~ ha^{-1} * ")"),
                   y_lbl = expression("Predicted GSV (" * m^3 ~ ha^{-1} * ")"), is_large = TRUE),
  "BA"      = list(x_lbl = expression("Observed BA (" * m^2 ~ ha^{-1} * ")"),
                   y_lbl = expression("Predicted BA (" * m^2 ~ ha^{-1} * ")"), is_large = TRUE),
  "H"       = list(x_lbl = expression("Observed Height (m)"),
                   y_lbl = expression("Predicted Height (m)"), is_large = TRUE),
  "DBH"     = list(x_lbl = expression("Observed DBH (cm)"),
                   y_lbl = expression("Predicted DBH (cm)"), is_large = TRUE),
  "AGE"     = list(x_lbl = expression("Observed Age (yr)"),
                   y_lbl = expression("Predicted Age (yr)"), is_large = TRUE),
  "shannon" = list(x_lbl = expression("Observed Shannon Index"),
                   y_lbl = expression("Predicted Shannon Index"), is_large = FALSE),
  "CV_DBH"  = list(x_lbl = expression("Observed CV DBH"),
                   y_lbl = expression("Predicted CV DBH"), is_large = FALSE)
)

CLIMATE_ZONES <- c("WT", "MS", "MT", "SS", "NS", "CT", "H", "MTrop", "MidTr")
LINE_11_COLOR <- "grey50" 

# ==============================================================================
# 2. Theme
# ==============================================================================

theme_rse_scatter <- function() {
  theme_classic(base_size = 9, base_family = base_family_use) +
    theme(
      plot.background  = element_rect(fill = "white", color = NA),
      panel.background = element_rect(fill = "white", color = NA),
      
      panel.border = element_rect(color = "black", fill = NA, linewidth = 0.6),
      axis.line = element_blank(),
      
      axis.ticks = element_line(color = "black", linewidth = 0.5),
      axis.ticks.length = unit(3, "pt"),
      
      axis.text = element_text(color = "black", size = 8),
      axis.title = element_blank(), 
      
      panel.grid.major = element_blank(),
      panel.grid.minor = element_blank(),
      
      plot.margin = margin(3, 3, 3, 3),
      legend.position = "none" # Default to none, overridden for legend extraction
    )
}

# ==============================================================================
# 3. Helper functions
# ==============================================================================

calculate_metrics <- function(obs, pred) {
  valid <- complete.cases(obs, pred) & is.finite(obs) & is.finite(pred)
  obs  <- obs[valid]
  pred <- pred[valid]
  
  if (length(obs) < 2) return(c(R = NA, RMSE = NA, rRMSE = NA))
  
  rmse_val <- sqrt(mean((obs - pred)^2))
  mean_obs <- mean(obs)
  rrmse_val <- (rmse_val / mean_obs) * 100
  
  c(
    R     = suppressWarnings(cor(obs, pred)),
    RMSE  = rmse_val,
    rRMSE = rrmse_val
  )
}

format_metric <- function(x, is_large) {
  if (!is.finite(x)) return("NA")
  if (isTRUE(is_large)) {
    formatC(x, format = "f", digits = 0, big.mark = "")
  } else {
    formatC(x, format = "f", digits = 2) 
  }
}

get_axis_formatter <- function(is_large) {
  if (isTRUE(is_large)) {
    label_number(accuracy = 1, big.mark = "")
  } else {
    label_number(accuracy = 0.1)
  }
}

read_zone_data <- function(zone_id, strategy_col, target_var) {
  file_path <- file.path(BASE_DIR, target_var, zone_id, paste0("Z_", zone_id, "_Preds.csv"))
  if (!file.exists(file_path)) return(NULL)
  
  df <- tryCatch(
    readr::read_csv(file_path, show_col_types = FALSE, progress = FALSE),
    error = function(e) return(NULL)
  )
  if (is.null(df)) return(NULL)
  
  out <- tibble(
    Zone      = zone_id,
    Observed  = suppressWarnings(as.numeric(df[["obs"]])),
    Predicted = suppressWarnings(as.numeric(df[[strategy_col]]))
  ) %>%
    filter(is.finite(Observed), is.finite(Predicted))
  
  if (nrow(out) < 5) return(NULL)
  out
}

sample_for_plot <- function(df) {
  if (nrow(df) <= PLOT_MAX_POINTS) return(df)
  df[sample.int(nrow(df), PLOT_MAX_POINTS), , drop = FALSE]
}

make_axis_limits <- function(all_data) {
  values <- c(all_data$Observed, all_data$Predicted)
  values <- values[is.finite(values)]
  if (length(values) < 2) return(c(0, 1))
  
  lower <- min(0, min(values, na.rm = TRUE))
  if (isTRUE(USE_ROBUST_AXIS)) {
    upper <- quantile(values, probs = AXIS_QUANTILE, na.rm = TRUE, names = FALSE)
  } else {
    upper <- max(values, na.rm = TRUE)
  }
  
  if (!is.finite(upper) || upper <= lower) upper <- max(values, na.rm = TRUE)
  
  span <- upper - lower
  c(lower, upper + 0.05 * span) 
}

# ==============================================================================
# 4. Panel plotting
# ==============================================================================

create_zone_plot <- function(zone_id, zone_data, axis_limits, target_var) {
  
  var_conf <- VAR_META[[target_var]]
  is_large <- if (!is.null(var_conf)) var_conf$is_large else FALSE
  
  if (is.null(zone_data) || nrow(zone_data) < 5) {
    return(ggplot() + theme_void())
  }
  
  plot_data <- sample_for_plot(zone_data)
  metrics <- calculate_metrics(zone_data$Observed, zone_data$Predicted)
  
  label_stats <- paste0(
    "Zone: ", zone_id, "\n",
    "R = ", sprintf("%.2f", metrics["R"]), "\n",
    "RMSE = ", sprintf("%.2f", metrics["RMSE"])#, "\n"
    #"rRMSE = ", sprintf("%.1f%%", metrics["rRMSE"])
  )
  
  axis_formatter <- get_axis_formatter(is_large)
  x_min <- axis_limits[1]
  x_max <- axis_limits[2]
  span  <- x_max - x_min
  
  stat_x <- x_min + 0.05 * span
  stat_y <- x_max - 0.03 * span
  
  p <- ggplot() +
    geom_abline(
      intercept = 0, slope = 1, 
      linetype = "dashed", color = LINE_11_COLOR, linewidth = 0.5, alpha = 0.8
    )
  
  if (HAS_POINTDENSITY) {
    p <- p + ggpointdensity::geom_pointdensity(
      data = plot_data, 
      aes(x = Observed, y = Predicted, 
          color = after_stat( (rank(density, ties.method = "min") - 1) / max(length(density) - 1, 1) )
      ),
      size = 0.6, alpha = 0.9, adjust = 1.2, shape = 16
    ) + 
    # Perceptually uniform 'turbo' scale"viridis"
    scale_color_viridis_c(
      option ="viridis", 
      limits = c(0, 1), 
      name = "Relative Density",
      breaks = c(0, 0.5, 1),
      labels = c("Low", "Med", "High")
    )
  } else {
    p <- p + geom_point(
      data = plot_data, aes(x = Observed, y = Predicted),
      size = 0.5, alpha = 0.4, color = "#00a1d5"
    )
  }
  
  # Add linear trend line
  p <- p + geom_smooth(
    data = plot_data, 
    aes(x = Observed, y = Predicted), 
    method = "lm", 
    formula = y ~ x, 
    se = FALSE, 
    color = "black", 
    linewidth = 0.6
  )
  
  # Anti-overlap annotation box
  p + annotate(
      "label",
      x = stat_x, y = stat_y, 
      label = label_stats,
      hjust = 0, vjust = 1,
      family = base_family_use, size = 3.2, lineheight = 1.3, 
      color = "black", 
      fill = alpha("white", 0.90), 
      label.size = 0.3,            
      label.padding = unit(0.35, "lines")
    ) +
    scale_x_continuous(labels = axis_formatter, breaks = pretty_breaks(n = 4)) +
    scale_y_continuous(labels = axis_formatter, breaks = pretty_breaks(n = 4)) +
    coord_equal(xlim = axis_limits, ylim = axis_limits, expand = FALSE, clip = "on") +
    theme_rse_scatter()
}

# ==============================================================================
# 5. Figure assembly
# ==============================================================================

generate_global_figure <- function(strategy_col, filename_suffix, main_title, target_var) {
  
  save_dir <- file.path(OUTPUT_ROOT, target_var)
  if (!dir.exists(save_dir)) dir.create(save_dir, recursive = TRUE)
  
  cat(sprintf("\n>>> [%s] Processing: %s\n", target_var, main_title))
  
  zone_data_list <- lapply(CLIMATE_ZONES, function(z) read_zone_data(z, strategy_col, target_var))
  names(zone_data_list) <- CLIMATE_ZONES
  
  all_data <- bind_rows(zone_data_list)
  if (nrow(all_data) < 5) return(invisible(NULL))
  
  axis_limits <- make_axis_limits(all_data)
  var_conf <- VAR_META[[target_var]]
  
  plots <- lapply(CLIMATE_ZONES, function(z) {
    create_zone_plot(zone_id = z, zone_data = zone_data_list[[z]], axis_limits = axis_limits, target_var = target_var)
  })
  
  # -------------------------------------------------------------------------
  # Legend Extraction and Assembly
  # -------------------------------------------------------------------------
  # Force the first plot to display the legend horizontally
  legend_plot <- plots[[1]] + 
    theme(
      legend.position = "bottom",
      legend.justification = "center",
      legend.key.width = unit(2.5, "cm"),
      legend.key.height = unit(0.3, "cm"),
      # Changed to face = "plain" to avoid bold text
      legend.title = element_text(size = 10, face = "plain", vjust = 1),
      legend.text = element_text(size = 9, face = "plain")
    )
  
  shared_legend <- cowplot::get_legend(legend_plot)
  
  # Strip legends from individual plots to keep the grid clean
  plots_clean <- lapply(plots, function(p) p + theme(legend.position = "none"))
  
  # Build the 3x3 grid WITHOUT auto-tagging
  grid_plot <- plot_grid(
    plotlist = plots_clean, 
    ncol = 3, 
    align = "hv", 
    axis = "tblr"
  )
  
  # -------------------------------------------------------------------------
  # Final Canvas Packaging (ggdraw layout mapping)
  # -------------------------------------------------------------------------
  # Layout structure (Top to Bottom):
  # y = 0.97         : Main Title
  # y = 0.11 to 0.95 : 3x3 Grid Plot
  # y = 0.08         : X-axis Label (Observed)
  # y = 0.01         : Legend (Strictly below the X-axis label)
  
  final_plot <- ggdraw() +
    draw_plot(grid_plot, x = 0.04, y = 0.11, width = 0.95, height = 0.84) +
    draw_label(main_title, x = 0.515, y = 0.97, fontfamily = base_family_use, size = 13, color = "black") +
    draw_label(var_conf$y_lbl, x = 0.015, y = 0.53, angle = 90, fontfamily = base_family_use, size = 11, color = "black") +
    # X-axis label moved slightly down to leave space for the legend below it
    draw_label(var_conf$x_lbl, x = 0.515, y = 0.08, fontfamily = base_family_use, size = 11, color = "black") +
    # Shared Legend anchored at the very bottom
    draw_plot(shared_legend, x = 0, y = 0.01, width = 1, height = 0.05)
  
  out_pdf <- file.path(save_dir, paste0("RSE_Style_", target_var, "_", filename_suffix, ".pdf"))
  out_tif <- file.path(save_dir, paste0("RSE_Style_", target_var, "_", filename_suffix, ".tif"))
  
  ggsave(filename = out_pdf, plot = final_plot, width = FIG_WIDTH, height = FIG_HEIGHT, units = "in", device = cairo_pdf, bg = "white")
  
  if (HAS_RAGG) {
    ggsave(filename = out_tif, plot = final_plot, width = FIG_WIDTH, height = FIG_HEIGHT, units = "in", dpi = FIG_DPI, device = ragg::agg_tiff, compression = "lzw", bg = "white")
  } else {
    ggsave(filename = out_tif, plot = final_plot, width = FIG_WIDTH, height = FIG_HEIGHT, units = "in", dpi = FIG_DPI, device = "tiff", compression = "lzw", bg = "white")
  }
  
  cat(sprintf("Saved PDF: %s\n", out_pdf))
}

# ==============================================================================
# 6. Run
# ==============================================================================

for (target in names(VAR_META)) {
  tryCatch({
    generate_global_figure(strategy_col = "Random_E", filename_suffix = "RandomCV", main_title = "Random Cross-Validation", target_var = target)
    generate_global_figure(strategy_col = "KMeans_E", filename_suffix = "SpatialCV", main_title = " K-Means Spatial Cross-Validation", target_var = target)
  }, error = function(e) cat(sprintf("Error in %s: %s\n", target, e$message)))
}

cat("\nProcessing complete.\n")