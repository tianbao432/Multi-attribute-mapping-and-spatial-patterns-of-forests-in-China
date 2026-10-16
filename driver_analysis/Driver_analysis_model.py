#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Part 2: GPU XGBoost driver analysis using preassigned K-means fold IDs.

Input
-----
The CSV created by Part 1. It must contain:
- Fifteen driver variables
- Renamed target variables
- KMeans_Fold values from 1 to 10
- No coordinate columns are required

Workflow
--------
1. Read the prepared CSV.
2. Run one 50-trial Optuna TPE search using the fixed 10-fold K-means IDs.
3. Generate 10-fold out-of-fold predictions with the selected parameters.
4. Report pooled R2 and RMSE.
5. Fit a final GPU XGBoost model using all valid samples.
6. Export a SHAP beeswarm plot.
7. Export the top-ranked driver's SHAP response plot.
8. Calculate fold-model SHAP importance and ranking stability.
"""

from __future__ import annotations

import gc
import json
import logging
import re
import sys
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import shap
import sklearn
import xgboost as xgb

from matplotlib.ticker import MaxNLocator
from pygam import LinearGAM, s
from sklearn.metrics import mean_squared_error, r2_score

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)


# =============================================================================
# Configuration
# =============================================================================
@dataclass
class Config:
    input_csv: str = "/data/China_mapping/driver/plot_driver_kmeans10.csv"
    output_root: str = "/data/China_mapping/driver/result_fina/"
    fold_col: str = "KMeans_Fold"

    drivers: List[str] = field(default_factory=lambda: [
        "HMI",
        "MAT",
        "ISO",
        "MTCM",
        "MTDQ",
        "MAP",
        "PDM",
        "PS",
        "ELEV",
        "TWI",
        "VPD",
        "SWVL3",
        "TN",
        "PH",
        "CLAY",
    ])

    targets: List[str] = field(default_factory=lambda: [
        "Density",
        "DBH",
        "BA",
        "GSV",
        "AGB",
        "Shannon",
        "CV DBH",
        "Age",
        "Height",
    ])

    expected_folds: int = 10
    optuna_trials: int =50
    random_seed: int = 42

    min_total_samples: int = 50
    min_train_samples: int = 30
    min_valid_samples: int = 3

    max_shap_samples: int = 5000
    shap_max_display: int = 15

    use_gpu: bool = True
    gpu_id: int = 0

    save_fold_models: bool = False
    save_final_model: bool = True

    figure_dpi: int = 600


CONFIG = Config()


# =============================================================================
# Logging, plotting, and utilities
# =============================================================================
def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def setup_logger(path: Path) -> logging.Logger:
    ensure_dir(path.parent)

    logger = logging.getLogger("part2_modeling")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(path, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def set_plot_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "axes.linewidth": 0.9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "figure.dpi": 150,
        "savefig.dpi": CONFIG.figure_dpi,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def sanitize_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))


def stable_seed(*parts: Any) -> int:
    text = "|".join(str(part) for part in parts)
    offset = sum((index + 1) * ord(char) for index, char in enumerate(text))
    return int((CONFIG.random_seed + offset) % (2**31 - 1))


def cleanup_memory() -> None:
    gc.collect()


def save_json(data: Dict[str, Any], path: Path) -> None:
    converted = {}

    for key, value in data.items():
        if isinstance(value, np.integer):
            converted[key] = int(value)
        elif isinstance(value, np.floating):
            converted[key] = float(value)
        elif isinstance(value, np.ndarray):
            converted[key] = value.tolist()
        else:
            converted[key] = value

    with open(path, "w", encoding="utf-8") as file:
        json.dump(converted, file, indent=2, ensure_ascii=False)


def requested_device() -> str:
    if not CONFIG.use_gpu:
        return "cpu"
    return f"cuda:{CONFIG.gpu_id}"


def verify_device(logger: logging.Logger) -> str:
    device = requested_device()

    if device == "cpu":
        logger.info("CPU mode selected.")
        return "cpu"

    try:
        model = xgb.XGBRegressor(
            n_estimators=2,
            max_depth=2,
            objective="reg:squarederror",
            tree_method="hist",
            device=device,
            n_jobs=1,
            verbosity=0,
        )

        x_test = np.arange(8, dtype=np.float32).reshape(-1, 1)
        y_test = np.arange(8, dtype=np.float32)
        model.fit(x_test, y_test)

        logger.info("CUDA verification succeeded: %s", device)
        return device

    except Exception as exc:
        logger.warning(
            "CUDA verification failed. CPU fallback will be used. Error: %s",
            exc,
        )
        return "cpu"


# =============================================================================
# Metrics and folds
# =============================================================================
def compute_r2_rmse(
    observed: Sequence[float],
    predicted: Sequence[float],
) -> Dict[str, float]:
    observed_array = np.asarray(observed, dtype=np.float64)
    predicted_array = np.asarray(predicted, dtype=np.float64)

    valid = np.isfinite(observed_array) & np.isfinite(predicted_array)
    observed_array = observed_array[valid]
    predicted_array = predicted_array[valid]

    if observed_array.size < 2:
        return {
            "N": int(observed_array.size),
            "R2": np.nan,
            "RMSE": np.nan,
        }

    return {
        "N": int(observed_array.size),
        "R2": float(r2_score(observed_array, predicted_array)),
        "RMSE": float(
            np.sqrt(mean_squared_error(observed_array, predicted_array))
        ),
    }


def validate_fold_ids(fold_series: pd.Series) -> np.ndarray:
    numeric = pd.to_numeric(fold_series, errors="coerce")

    if numeric.isna().any():
        raise ValueError("The fold column contains missing or non-numeric values.")

    fold_ids = numeric.astype(int).to_numpy()
    unique_folds = np.sort(np.unique(fold_ids))

    if unique_folds.size != CONFIG.expected_folds:
        raise ValueError(
            f"Expected {CONFIG.expected_folds} folds, "
            f"but found {unique_folds.tolist()}."
        )

    expected = np.arange(1, CONFIG.expected_folds + 1)

    if not np.array_equal(unique_folds, expected):
        raise ValueError(
            f"Fold IDs must be 1-{CONFIG.expected_folds}, "
            f"but found {unique_folds.tolist()}."
        )

    return fold_ids


# =============================================================================
# XGBoost and Optuna
# =============================================================================
def build_xgb_model(
    params: Dict[str, Any],
    device: str,
    seed: int,
) -> xgb.XGBRegressor:
    model_params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "tree_method": "hist",
        "device": device,
        "n_jobs": 1 if device.startswith("cuda") else -1,
        "random_state": seed,
        "verbosity": 0,
    }

    model_params.update(params)
    return xgb.XGBRegressor(**model_params)


def suggest_hyperparameters(
    trial: optuna.Trial,
) -> Dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int(
            "n_estimators",
            300,
            1500,
            step=100,
        ),
        "learning_rate": trial.suggest_float(
            "learning_rate",
            0.01,
            0.15,
            log=True,
        ),
        "max_depth": trial.suggest_int(
            "max_depth",
            2,
            8,
        ),
        "min_child_weight": trial.suggest_float(
            "min_child_weight",
            1.0,
            20.0,
            log=True,
        ),
        "subsample": trial.suggest_float(
            "subsample",
            0.60,
            1.00,
        ),
        "colsample_bytree": trial.suggest_float(
            "colsample_bytree",
            0.60,
            1.00,
        ),
        "gamma": trial.suggest_float(
            "gamma",
            1e-5,
            2.0,
            log=True,
        ),
        "reg_alpha": trial.suggest_float(
            "reg_alpha",
            1e-5,
            10.0,
            log=True,
        ),
        "reg_lambda": trial.suggest_float(
            "reg_lambda",
            0.1,
            20.0,
            log=True,
        ),
        "max_bin": trial.suggest_categorical(
            "max_bin",
            [128, 256, 512],
        ),
    }


def spatial_cv_predictions(
    X: pd.DataFrame,
    y: pd.Series,
    fold_ids: np.ndarray,
    params: Dict[str, Any],
    device: str,
    seed_prefix: str,
    return_models: bool,
) -> Tuple[np.ndarray, pd.DataFrame, List[xgb.XGBRegressor], List[int]]:
    predictions = np.full(len(y), np.nan, dtype=np.float64)
    fold_records = []
    fold_models: List[xgb.XGBRegressor] = []
    modeled_fold_ids: List[int] = []

    for fold_id in np.sort(np.unique(fold_ids)):
        valid_index = np.where(fold_ids == fold_id)[0]
        train_index = np.where(fold_ids != fold_id)[0]

        if (
            train_index.size < CONFIG.min_train_samples
            or valid_index.size < CONFIG.min_valid_samples
        ):
            fold_records.append({
                "Fold": int(fold_id),
                "Train_N": int(train_index.size),
                "Valid_N": int(valid_index.size),
                "R2": np.nan,
                "RMSE": np.nan,
                "Status": "skipped_insufficient_samples",
            })
            continue

        model = build_xgb_model(
            params=params,
            device=device,
            seed=stable_seed(seed_prefix, fold_id),
        )

        model.fit(
            X.iloc[train_index],
            y.iloc[train_index],
        )

        fold_prediction = model.predict(X.iloc[valid_index])
        predictions[valid_index] = fold_prediction

        metrics = compute_r2_rmse(
            y.iloc[valid_index].to_numpy(dtype=np.float64),
            fold_prediction,
        )

        fold_records.append({
            "Fold": int(fold_id),
            "Train_N": int(train_index.size),
            "Valid_N": int(valid_index.size),
            "R2": metrics["R2"],
            "RMSE": metrics["RMSE"],
            "Status": "success",
        })

        if return_models:
            fold_models.append(model)
            modeled_fold_ids.append(int(fold_id))
        else:
            del model
            cleanup_memory()

    return (
        predictions,
        pd.DataFrame(fold_records),
        fold_models,
        modeled_fold_ids,
    )


def tune_xgb_once(
    X: pd.DataFrame,
    y: pd.Series,
    fold_ids: np.ndarray,
    device: str,
    target_dir: Path,
    target_name: str,
) -> Tuple[Dict[str, Any], float]:
    def objective(trial: optuna.Trial) -> float:
        params = suggest_hyperparameters(trial)

        predictions, fold_metrics, _, _ = spatial_cv_predictions(
            X=X,
            y=y,
            fold_ids=fold_ids,
            params=params,
            device=device,
            seed_prefix=f"{target_name}_trial_{trial.number}",
            return_models=False,
        )

        successful_folds = int(
            (fold_metrics["Status"] == "success").sum()
        )
        trial.set_user_attr("successful_folds", successful_folds)

        valid = np.isfinite(predictions)

        if successful_folds < 2 or valid.sum() < 2:
            return np.inf

        return float(
            np.sqrt(mean_squared_error(
                y.to_numpy(dtype=np.float64)[valid],
                predictions[valid],
            ))
        )

    sampler = optuna.samplers.TPESampler(
        seed=stable_seed("optuna", target_name),
        n_startup_trials=min(15, CONFIG.optuna_trials),
        multivariate=True,
    )

    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
    )

    study.optimize(
        objective,
        n_trials=CONFIG.optuna_trials,
        n_jobs=1,
        gc_after_trial=True,
        show_progress_bar=False,
    )

    trial_records = []

    for trial in study.trials:
        record = {
            "Trial": int(trial.number),
            "State": trial.state.name,
            "Pooled_RMSE": trial.value,
            "Successful_Folds": trial.user_attrs.get(
                "successful_folds",
                np.nan,
            ),
        }
        record.update(trial.params)
        trial_records.append(record)

    pd.DataFrame(trial_records).to_csv(
        target_dir / "optuna_trials.csv",
        index=False,
    )

    best_params = dict(study.best_params)
    best_rmse = float(study.best_value)

    save_json(
        {
            "Best_Spatial_CV_RMSE": best_rmse,
            **best_params,
        },
        target_dir / "best_hyperparameters.json",
    )

    return best_params, best_rmse


# =============================================================================
# SHAP analysis
# =============================================================================
def select_shap_rows(
    X: pd.DataFrame,
    seed: int,
) -> pd.DataFrame:
    if len(X) <= CONFIG.max_shap_samples:
        return X.reset_index(drop=True)

    rng = np.random.default_rng(seed)
    selected = rng.choice(
        len(X),
        size=CONFIG.max_shap_samples,
        replace=False,
    )

    return X.iloc[selected].reset_index(drop=True)


def compute_shap_explanation(
    model: xgb.XGBRegressor,
    X_data: pd.DataFrame,
) -> shap.Explanation:
    explainer = shap.TreeExplainer(model)

    return explainer(
        X_data,
        check_additivity=False,
    )


def calculate_shap_importance(
    explanation: shap.Explanation,
    feature_names: Sequence[str],
) -> pd.DataFrame:
    mean_absolute_shap = np.mean(
        np.abs(explanation.values),
        axis=0,
    )

    total = float(np.sum(mean_absolute_shap))

    contribution = (
        100.0 * mean_absolute_shap / total
        if total > 0.0
        else np.full_like(mean_absolute_shap, np.nan)
    )

    importance = pd.DataFrame({
        "Feature": list(feature_names),
        "Mean_Absolute_SHAP": mean_absolute_shap,
        "Contribution_Percent": contribution,
    }).sort_values(
        "Mean_Absolute_SHAP",
        ascending=False,
    ).reset_index(drop=True)

    importance["Rank"] = np.arange(1, len(importance) + 1)
    return importance


def save_final_shap_outputs(
    model: xgb.XGBRegressor,
    X: pd.DataFrame,
    target_name: str,
    target_dir: Path,
) -> pd.DataFrame:
    X_shap = select_shap_rows(
        X=X,
        seed=stable_seed("final_shap", target_name),
    )

    explanation = compute_shap_explanation(
        model=model,
        X_data=X_shap,
    )

    importance = calculate_shap_importance(
        explanation=explanation,
        feature_names=X_shap.columns,
    )

    importance.to_csv(
        target_dir / "final_shap_importance.csv",
        index=False,
    )

    shap_values = pd.DataFrame(
        explanation.values,
        columns=[
            f"SHAP_{feature}"
            for feature in X_shap.columns
        ],
    )

    pd.concat(
        [X_shap.reset_index(drop=True), shap_values],
        axis=1,
    ).to_csv(
        target_dir / "final_shap_values_and_features.csv",
        index=False,
    )

    set_plot_style()
    plt.figure(figsize=(7.3, 5.8))

    shap.plots.beeswarm(
        explanation,
        max_display=min(
            CONFIG.shap_max_display,
            X_shap.shape[1],
        ),
        show=False,
        plot_size=None,
    )

    plt.xlabel("SHAP value")
    plt.title(
        f"Predictor contributions to {target_name}",
        pad=10,
    )
    plt.tight_layout()

    plt.savefig(target_dir / "final_shap_beeswarm.png")
    plt.savefig(target_dir / "final_shap_beeswarm.pdf")
    plt.close()

    top_feature = str(importance.loc[0, "Feature"])
    feature_index = X_shap.columns.get_loc(top_feature)

    x_values = X_shap[top_feature].to_numpy(dtype=np.float64)
    shap_values_array = explanation.values[:, feature_index].astype(
        np.float64
    )

    finite = np.isfinite(x_values) & np.isfinite(shap_values_array)
    x_values = x_values[finite]
    shap_values_array = shap_values_array[finite]

    raw_df = pd.DataFrame({
        top_feature: x_values,
        "SHAP_Value": shap_values_array,
    })

    raw_df.to_csv(
        target_dir
        / f"top1_dependence_{sanitize_name(top_feature)}_raw.csv",
        index=False,
    )

    curve_df = pd.DataFrame()
    gam_success = False

    if x_values.size >= 10 and np.unique(x_values).size >= 5:
        try:
            order = np.argsort(x_values)
            x_sorted = x_values[order]
            shap_sorted = shap_values_array[order]

            spline_count = int(
                np.clip(np.unique(x_sorted).size // 5, 5, 20)
            )

            gam = LinearGAM(
                s(0, n_splines=spline_count)
            ).gridsearch(
                x_sorted.reshape(-1, 1),
                shap_sorted,
                progress=False,
            )

            lower = float(np.quantile(x_sorted, 0.01))
            upper = float(np.quantile(x_sorted, 0.99))
            x_grid = np.linspace(lower, upper, 300)

            prediction = gam.predict(
                x_grid.reshape(-1, 1)
            )

            confidence = gam.confidence_intervals(
                x_grid.reshape(-1, 1),
                width=0.95,
            )

            curve_df = pd.DataFrame({
                top_feature: x_grid,
                "GAM_SHAP": prediction,
                "CI_Lower": confidence[:, 0],
                "CI_Upper": confidence[:, 1],
            })

            curve_df.to_csv(
                target_dir
                / f"top1_dependence_{sanitize_name(top_feature)}_gam.csv",
                index=False,
            )

            gam_success = True

        except Exception:
            gam_success = False

    set_plot_style()
    fig, ax = plt.subplots(figsize=(5.4, 4.5))

    ax.scatter(
        x_values,
        shap_values_array,
        s=24,
        alpha=0.58,
        edgecolors="none",
    )

    if gam_success:
        ax.fill_between(
            curve_df[top_feature],
            curve_df["CI_Lower"],
            curve_df["CI_Upper"],
            alpha=0.18,
            linewidth=0,
        )

        ax.plot(
            curve_df[top_feature],
            curve_df["GAM_SHAP"],
            color="black",
            linewidth=1.8,
        )

    ax.axhline(
        0.0,
        color="black",
        linewidth=0.8,
        linestyle="--",
    )

    ax.set_xlabel(top_feature)
    ax.set_ylabel("SHAP value")
    ax.set_title(
        f"{top_feature} contribution to {target_name}"
    )

    fig.tight_layout()

    fig.savefig(
        target_dir
        / f"top1_dependence_{sanitize_name(top_feature)}.png"
    )
    fig.savefig(
        target_dir
        / f"top1_dependence_{sanitize_name(top_feature)}.pdf"
    )
    plt.close(fig)

    return importance


def calculate_fold_shap_stability(
    fold_models: List[xgb.XGBRegressor],
    modeled_fold_ids: List[int],
    X: pd.DataFrame,
    fold_ids: np.ndarray,
    target_name: str,
    target_dir: Path,
) -> pd.DataFrame:
    records = []

    if len(fold_models) != len(modeled_fold_ids):
        raise ValueError(
            "Fold model count does not match modeled fold ID count."
        )

    for model, fold_id in zip(fold_models, modeled_fold_ids):
        valid_index = np.where(fold_ids == fold_id)[0]
        X_valid = X.iloc[valid_index].reset_index(drop=True)

        X_valid = select_shap_rows(
            X=X_valid,
            seed=stable_seed(
                "fold_shap",
                target_name,
                fold_id,
            ),
        )

        explanation = compute_shap_explanation(
            model=model,
            X_data=X_valid,
        )

        importance = calculate_shap_importance(
            explanation=explanation,
            feature_names=X_valid.columns,
        )

        importance["Fold"] = int(fold_id)
        records.append(importance)

    long_table = pd.concat(
        records,
        ignore_index=True,
    )

    long_table.to_csv(
        target_dir / "fold_shap_importance_long.csv",
        index=False,
    )

    stability = (
        long_table
        .groupby("Feature", as_index=False)
        .agg(
            Mean_Absolute_SHAP_Mean=(
                "Mean_Absolute_SHAP",
                "mean",
            ),
            Mean_Absolute_SHAP_SD=(
                "Mean_Absolute_SHAP",
                "std",
            ),
            Contribution_Percent_Mean=(
                "Contribution_Percent",
                "mean",
            ),
            Contribution_Percent_SD=(
                "Contribution_Percent",
                "std",
            ),
            Rank_Mean=("Rank", "mean"),
            Rank_SD=("Rank", "std"),
            Rank_Median=("Rank", "median"),
            Rank_Min=("Rank", "min"),
            Rank_Max=("Rank", "max"),
            Top1_Frequency=(
                "Rank",
                lambda values: int(np.sum(values == 1)),
            ),
            Top3_Frequency=(
                "Rank",
                lambda values: int(np.sum(values <= 3)),
            ),
            Top5_Frequency=(
                "Rank",
                lambda values: int(np.sum(values <= 5)),
            ),
        )
        .sort_values(
            [
                "Rank_Median",
                "Rank_Mean",
                "Mean_Absolute_SHAP_Mean",
            ],
            ascending=[True, True, False],
        )
        .reset_index(drop=True)
    )

    stability["Stable_Rank"] = np.arange(
        1,
        len(stability) + 1,
    )

    stability.to_csv(
        target_dir / "fold_shap_importance_stability.csv",
        index=False,
    )

    set_plot_style()

    plot_df = stability.sort_values(
        "Mean_Absolute_SHAP_Mean",
        ascending=True,
    )

    fig, ax = plt.subplots(figsize=(6.2, 5.0))

    ax.barh(
        plot_df["Feature"],
        plot_df["Mean_Absolute_SHAP_Mean"],
        xerr=plot_df["Mean_Absolute_SHAP_SD"].fillna(0.0),
        capsize=2.5,
    )

    ax.set_xlabel("Mean absolute SHAP value across folds")
    ax.set_ylabel("")
    ax.set_title(
        f"Spatial-fold SHAP stability: {target_name}"
    )
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))

    fig.tight_layout()
    fig.savefig(
        target_dir / "fold_shap_importance_stability.png"
    )
    fig.savefig(
        target_dir / "fold_shap_importance_stability.pdf"
    )
    plt.close(fig)

    return stability


# =============================================================================
# Evaluation plot
# =============================================================================
def save_observed_predicted_plot(
    observed: np.ndarray,
    predicted: np.ndarray,
    target_name: str,
    target_dir: Path,
) -> None:
    valid = np.isfinite(observed) & np.isfinite(predicted)
    observed_valid = observed[valid]
    predicted_valid = predicted[valid]

    if observed_valid.size < 2:
        return

    metrics = compute_r2_rmse(
        observed_valid,
        predicted_valid,
    )

    lower = float(
        min(observed_valid.min(), predicted_valid.min())
    )
    upper = float(
        max(observed_valid.max(), predicted_valid.max())
    )

    padding = 0.03 * max(upper - lower, 1e-12)
    lower -= padding
    upper += padding

    set_plot_style()
    fig, ax = plt.subplots(figsize=(5.1, 4.9))

    ax.scatter(
        observed_valid,
        predicted_valid,
        s=28,
        alpha=0.70,
        edgecolors="white",
        linewidths=0.30,
    )

    ax.plot(
        [lower, upper],
        [lower, upper],
        color="black",
        linewidth=1.1,
        linestyle="--",
    )

    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)
    ax.set_aspect("equal", adjustable="box")

    ax.set_xlabel("Observed")
    ax.set_ylabel("Predicted")
    ax.set_title(
        f"10-fold K-means spatial CV: {target_name}"
    )

    ax.text(
        0.04,
        0.96,
        (
            f"R2 = {metrics['R2']:.2f}\n"
            f"RMSE = {metrics['RMSE']:.2f}"
        ),
        transform=ax.transAxes,
        va="top",
        ha="left",
    )

    fig.tight_layout()
    fig.savefig(target_dir / "observed_vs_predicted.png")
    fig.savefig(target_dir / "observed_vs_predicted.pdf")
    plt.close(fig)


# =============================================================================
# Target workflow
# =============================================================================
def process_target(
    full_df: pd.DataFrame,
    target_name: str,
    device: str,
    logger: logging.Logger,
) -> None:
    logger.info("Starting target: %s", target_name)

    target_dir = (
        Path(CONFIG.output_root)
        / sanitize_name(target_name)
    )
    ensure_dir(target_dir)

    required_columns = [
        target_name,
        CONFIG.fold_col,
        *CONFIG.drivers,
    ]

    valid_df = (
        full_df
        .dropna(subset=required_columns)
        .copy()
        .reset_index(drop=True)
    )

    for column in [
        target_name,
        CONFIG.fold_col,
        *CONFIG.drivers,
    ]:
        valid_df[column] = pd.to_numeric(
            valid_df[column],
            errors="coerce",
        )

    valid_df = (
        valid_df
        .replace([np.inf, -np.inf], np.nan)
        .dropna(subset=required_columns)
        .reset_index(drop=True)
    )

    if len(valid_df) < CONFIG.min_total_samples:
        logger.warning(
            "%s skipped: only %d complete samples.",
            target_name,
            len(valid_df),
        )
        return

    X = valid_df[CONFIG.drivers].astype(np.float32)
    y = valid_df[target_name].astype(np.float32)
    fold_ids = validate_fold_ids(valid_df[CONFIG.fold_col])

    valid_df.to_csv(
        target_dir / "analysis_dataset.csv",
        index=False,
    )

    fold_summary = (
        pd.Series(fold_ids, name="Fold")
        .value_counts()
        .sort_index()
        .rename_axis("Fold")
        .reset_index(name="N")
    )

    fold_summary.to_csv(
        target_dir / "fold_summary.csv",
        index=False,
    )

    logger.info(
        "%s: running one %d-trial Optuna search.",
        target_name,
        CONFIG.optuna_trials,
    )

    best_params, tuning_rmse = tune_xgb_once(
        X=X,
        y=y,
        fold_ids=fold_ids,
        device=device,
        target_dir=target_dir,
        target_name=target_name,
    )

    logger.info(
        "%s: best tuning RMSE = %.4f",
        target_name,
        tuning_rmse,
    )

    (
        predictions,
        fold_metrics,
        fold_models,
        modeled_fold_ids,
    ) = spatial_cv_predictions(
        X=X,
        y=y,
        fold_ids=fold_ids,
        params=best_params,
        device=device,
        seed_prefix=f"{target_name}_final_cv",
        return_models=True,
    )

    overall_metrics = compute_r2_rmse(
        y.to_numpy(dtype=np.float64),
        predictions,
    )

    pd.DataFrame([{
        "N": overall_metrics["N"],
        "R2": overall_metrics["R2"],
        "RMSE": overall_metrics["RMSE"],
        "Tuning_Informed_Evaluation": True,
    }]).to_csv(
        target_dir / "spatial_cv_metrics_overall.csv",
        index=False,
    )

    fold_metrics.to_csv(
        target_dir / "spatial_cv_metrics_by_fold.csv",
        index=False,
    )

    prediction_df = pd.DataFrame({
        "Observed": y.to_numpy(dtype=np.float64),
        "KMeans_Fold": fold_ids,
        "Predicted": predictions,
    })

    prediction_df["Residual"] = (
        prediction_df["Predicted"]
        - prediction_df["Observed"]
    )

    prediction_df.to_csv(
        target_dir / "spatial_cv_predictions.csv",
        index=False,
    )

    save_observed_predicted_plot(
        observed=y.to_numpy(dtype=np.float64),
        predicted=predictions,
        target_name=target_name,
        target_dir=target_dir,
    )

    logger.info(
        "%s spatial CV | N=%d | R2=%.4f | RMSE=%.4f",
        target_name,
        overall_metrics["N"],
        overall_metrics["R2"],
        overall_metrics["RMSE"],
    )

    calculate_fold_shap_stability(
        fold_models=fold_models,
        modeled_fold_ids=modeled_fold_ids,
        X=X,
        fold_ids=fold_ids,
        target_name=target_name,
        target_dir=target_dir,
    )

    if CONFIG.save_fold_models:
        fold_model_dir = target_dir / "fold_models"
        ensure_dir(fold_model_dir)

        for model, fold_id in zip(
            fold_models,
            modeled_fold_ids,
        ):
            joblib.dump(
                model,
                fold_model_dir / f"fold_{fold_id}_model.joblib",
                compress=3,
            )

            model.get_booster().save_model(
                fold_model_dir / f"fold_{fold_id}_model.json"
            )

    final_model = build_xgb_model(
        params=best_params,
        device=device,
        seed=stable_seed("final_model", target_name),
    )

    final_model.fit(X, y)

    if CONFIG.save_final_model:
        joblib.dump(
            final_model,
            target_dir / "final_model.joblib",
            compress=3,
        )

        final_model.get_booster().save_model(
            target_dir / "final_model.json"
        )

    save_final_shap_outputs(
        model=final_model,
        X=X,
        target_name=target_name,
        target_dir=target_dir,
    )

    for model in fold_models:
        del model

    del final_model
    cleanup_memory()

    logger.info("Completed target: %s", target_name)


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    output_root = Path(CONFIG.output_root)
    ensure_dir(output_root)

    logger = setup_logger(
        output_root / "part2_modeling.log"
    )

    device = verify_device(logger)

    logger.info("Python: %s", sys.version.split()[0])
    logger.info("NumPy: %s", np.__version__)
    logger.info("pandas: %s", pd.__version__)
    logger.info("scikit-learn: %s", sklearn.__version__)
    logger.info("XGBoost: %s", xgb.__version__)
    logger.info("SHAP: %s", shap.__version__)
    logger.info("Optuna: %s", optuna.__version__)
    logger.info("Device: %s", device)

    configuration = asdict(CONFIG)
    configuration["resolved_device"] = device

    with open(
        output_root / "part2_configuration.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            configuration,
            file,
            indent=2,
            ensure_ascii=False,
        )

    input_path = Path(CONFIG.input_csv)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input CSV not found: {input_path}"
        )

    data = pd.read_csv(input_path)

    forbidden_coordinate_columns = [
        column
        for column in [
            "Longitude",
            "Latitude",
            "Lon",
            "Lat",
            "longitude",
            "latitude",
        ]
        if column in data.columns
    ]

    if forbidden_coordinate_columns:
        logger.warning(
            "Coordinate columns are present but will not be used: %s",
            forbidden_coordinate_columns,
        )

    missing_drivers = [
        driver
        for driver in CONFIG.drivers
        if driver not in data.columns
    ]

    if missing_drivers:
        raise KeyError(
            f"Missing driver columns: {missing_drivers}"
        )

    if CONFIG.fold_col not in data.columns:
        raise KeyError(
            f"Missing fold column: {CONFIG.fold_col}"
        )

    available_targets = [
        target
        for target in CONFIG.targets
        if target in data.columns
    ]

    missing_targets = [
        target
        for target in CONFIG.targets
        if target not in data.columns
    ]

    if missing_targets:
        logger.warning(
            "Missing target columns: %s",
            missing_targets,
        )

    for target_name in available_targets:
        try:
            process_target(
                full_df=data,
                target_name=target_name,
                device=device,
                logger=logger,
            )

        except Exception as exc:
            logger.exception(
                "Target %s failed: %s",
                target_name,
                exc,
            )

            failure_path = output_root / "failed_targets.csv"

            pd.DataFrame([{
                "Target": target_name,
                "Error": str(exc),
            }]).to_csv(
                failure_path,
                mode="a",
                header=not failure_path.exists(),
                index=False,
            )

    logger.info("All requested targets have been processed.")


if __name__ == "__main__":
    main()
