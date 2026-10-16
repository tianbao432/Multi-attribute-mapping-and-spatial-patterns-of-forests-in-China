# -*- coding: utf-8 -*-
"""
TRA Inversion / Mapping Pipeline
Optimizations applied:
  - Smart RAM Disk Batching: Prevents RAM overflow by processing files in safe chunks.
  - Relaxed Masking: Allows partial feature missing (XGBoost native NaN handling).
  - Target-First Loops: (Variable -> Zone -> Batch) as strictly requested.
  - Immediate RAM Release: Clears tmpfs instantly after each batch.
  - One-time Model Loading: Models loaded once per zone/variable, reused across batches.
  - GPU acceleration for XGBoost inference (device="cuda").
  - Concurrent processing via ThreadPoolExecutor with n_jobs=2 safety.
  - JSON model fallback & Resuming protection.
"""
import os
import glob
import gc
import shutil
import joblib
import numpy as np
import pandas as pd
import rasterio
import warnings
import logging
from rasterio.merge import merge
from tqdm import tqdm
from sklearn.base import BaseEstimator, RegressorMixin
import xgboost as xgb
from concurrent.futures import ThreadPoolExecutor, as_completed

warnings.filterwarnings("ignore")
logging.getLogger('xgboost').setLevel(logging.ERROR)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("TRA_Mapping")

# ==============================================================================
# Configuration
# ==============================================================================
MASTER_RASTER_BANDS = []  
MIN_MODEL_SUCCESS_RATIO = 0.8

# Set to 4 to prevent CUDA Context Collision (OOM) on RTX 3090 with 100 loaded models
MAX_WORKERS = 20
RAMDISK_BATCH_SIZE = 400  


class XGBWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, **kwargs):
        self.model = None

    def predict(self, X):
        return self.model.predict(X)


def normalize_name(name):
    if not name:
        return ""
    base = str(name).replace(".tif", "").replace(".TIF", "")
    return base.lower().strip()


# ==============================================================================
# Model loading
# ==============================================================================
def load_one_model(joblib_path, use_gpu=False):
    device_type = "cuda" if use_gpu else "cpu"
    cpu_threads = 2 
    
    try:
        wrapper = joblib.load(joblib_path)
        if hasattr(wrapper, 'model') and wrapper.model is not None:
            wrapper.model.set_params(device=device_type, n_jobs=cpu_threads)
        return wrapper
    except (OSError, EOFError, KeyError, AttributeError, ValueError, TypeError) as e:
        logger.warning(
            f"  joblib load failed for {os.path.basename(joblib_path)} "
            f"({type(e).__name__}: {e}); trying JSON fallback."
        )

    json_path = joblib_path.replace(".joblib", ".json")
    if not os.path.exists(json_path):
        logger.warning(f"  No JSON fallback for {os.path.basename(joblib_path)}.")
        return None
        
    try:
        booster = xgb.XGBRegressor()
        booster.load_model(json_path)
        booster.set_params(device=device_type, n_jobs=cpu_threads)
        wrapper = XGBWrapper()
        wrapper.model = booster
        return wrapper
    except Exception as e:
        logger.warning(
            f"  JSON fallback failed for {os.path.basename(json_path)}: "
            f"{type(e).__name__}: {e}"
        )
        return None


# ==============================================================================
# Per-raster prediction
# ==============================================================================
def predict_raster(raster_path, loaded_models, output_dir, feature_indices, use_gpu):
    base = os.path.splitext(os.path.basename(raster_path))[0]
    out_m = os.path.join(output_dir, f"{base}_MEAN.tif")
    out_c = os.path.join(output_dir, f"{base}_CV.tif")

    if os.path.exists(out_m) and os.path.exists(out_c):
        return

    if feature_indices is None or not feature_indices:
        logger.error(f"{base}: empty feature_indices, skipping.")
        return

    try:
        with rasterio.open(raster_path) as src:
            meta = src.meta.copy()
            data = src.read()
    except (rasterio.errors.RasterioIOError, OSError) as e:
        logger.error(f"Error reading {raster_path} skipped. {type(e).__name__}: {e}")
        return

    try:
        selected_data = data[feature_indices, :, :]
    except IndexError as e:
        logger.error(f"{base}: Index error slicing bands. Check feature mapping. {e}")
        return

    C, H, W = selected_data.shape
    flat = np.moveaxis(selected_data, 0, -1).reshape(-1, C)

    nodata = meta.get('nodata')
    
    if nodata is None or (isinstance(nodata, float) and np.isnan(nodata)):
        mask = ~np.all(np.isnan(flat), axis=1)
    else:
        mask = ~np.all(flat == nodata, axis=1)
    
    idx = np.where(mask)[0]

    if len(idx) == 0:
        logger.warning(f"{base}: no valid pixels.")
        return

    X_v = flat[idx].astype(np.float32)

    if nodata is not None and not (isinstance(nodata, float) and np.isnan(nodata)):
        X_v[X_v == nodata] = np.nan
    X_v[np.isinf(X_v)] = np.nan

    if use_gpu:
        import cupy as cp
        X_v_target = cp.asarray(X_v)
    else:
        X_v_target = X_v

    all_preds = []
    fail_count = 0
    fail_reasons = {}
    
    for m in loaded_models:
        try:
            p = m.predict(X_v_target)
            if use_gpu and not isinstance(p, cp.ndarray):
                p = cp.asarray(p)
            all_preds.append(p)
        except (ValueError, TypeError, AttributeError, xgb.core.XGBoostError) as e:
            fail_count += 1
            reason = type(e).__name__
            fail_reasons[reason] = fail_reasons.get(reason, 0) + 1

    n_total = len(loaded_models)
    n_ok = len(all_preds)
    
    if fail_count > 0:
        logger.warning(
            f"{base}: {fail_count}/{n_total} models failed during predict | "
            f"reasons: {fail_reasons}")

    if n_total == 0 or n_ok / n_total < MIN_MODEL_SUCCESS_RATIO:
        logger.error(
            f"{base}: only {n_ok}/{n_total} models succeeded. Skipping output.")
        return

    if use_gpu:
        all_preds_gpu = cp.array(all_preds)
        mean_val = cp.nanmean(all_preds_gpu, axis=0).get()
        std_val = cp.nanstd(all_preds_gpu, axis=0).get()
        del all_preds_gpu, X_v_target
    else:
        all_preds = np.array(all_preds)
        mean_val = np.nanmean(all_preds, axis=0)
        std_val = np.nanstd(all_preds, axis=0)

    mean_val[np.isnan(mean_val)] = -9999
    
    abs_mean = np.abs(mean_val)
    cv_threshold = np.percentile(abs_mean[abs_mean != 9999], 1) if np.any(abs_mean != 9999) else 0.01
    valid_for_cv = (abs_mean > cv_threshold) & (mean_val != -9999)
    
    cv_val = np.full_like(mean_val, -9999, dtype=np.float32)
    cv_val[valid_for_cv] = (std_val[valid_for_cv] / abs_mean[valid_for_cv]) * 100.0
    cv_val[np.isnan(cv_val)] = -9999

    meta.update({
        'count': 1,
        'dtype': 'float32',
        'nodata': -9999,
        'compress': 'lzw',
        'bigtiff': 'YES',
        'driver': 'GTiff',
        'tiled': True,
        'blockxsize': 256,
        'blockysize': 256,
        'predictor': 3,
    })

    for v, p in [(mean_val, out_m), (cv_val, out_c)]:
        out_data = np.full(H * W, -9999, dtype=np.float32)
        out_data[idx] = v
        tmp_p = p + ".tmp"
        with rasterio.open(tmp_p, 'w', **meta) as dst:
            dst.write(out_data.reshape(H, W), 1)
        os.replace(tmp_p, p)

    del data, selected_data, flat, X_v, all_preds, mean_val, std_val, cv_val


# ==============================================================================
# Mosaic
# ==============================================================================
def run_mosaic(pred_dir, z_id, var):
    out_v = os.path.join(pred_dir, "Final_Mosaic")
    os.makedirs(out_v, exist_ok=True)

    for s in ["MEAN", "CV"]:
        out_path = os.path.join(out_v, f"Zone{z_id}_{var}_{s}_Mosaic.tif")
        if os.path.exists(out_path):
            logger.info(f"  Mosaic exists, skip: {os.path.basename(out_path)}")
            continue

        fs = sorted(glob.glob(os.path.join(pred_dir, f"*_{s}.tif")))
        if not fs:
            continue

        with rasterio.open(fs[0]) as src0:
            out_meta = src0.meta.copy()

        try:
            mos, out_trans = merge(fs, nodata=-9999)

            out_meta.update({
                "height": mos.shape[1],
                "width": mos.shape[2],
                "transform": out_trans,
                "bigtiff": "YES",
                "compress": "lzw",
                "nodata": -9999,
                "tiled": True,
                "blockxsize": 256,
                "blockysize": 256,
                "predictor": 3,
            })

            tmp_path = out_path + ".tmp"
            with rasterio.open(tmp_path, "w", **out_meta) as dst:
                dst.write(mos)
            os.replace(tmp_path, out_path)
            
            del mos, out_trans
            gc.collect()
            
        except Exception as e:
            logger.error(f"Failed to create mosaic for {out_path}. Error: {e}")


# ==============================================================================
# Resume helpers
# ==============================================================================
def zone_target_already_done(out_dir, z_id, var):
    mos_dir = os.path.join(out_dir, "Final_Mosaic")
    mean_p = os.path.join(mos_dir, f"Zone{z_id}_{var}_MEAN_Mosaic.tif")
    cv_p = os.path.join(mos_dir, f"Zone{z_id}_{var}_CV_Mosaic.tif")
    return os.path.exists(mean_p) and os.path.exists(cv_p)


# ==============================================================================
# Main
# ==============================================================================
if __name__ == "__main__":
    CFG = {
        "raster_base": "/data/China_mapping/fina_variables/",
        "ramdisk_base": "/dev/shm/fast_raster/",
        "model_base": "/data/China_mapping/fina_modelA/",
        "out_base": "/data/China_mapping/fina_mapping/",
        "vars": ["AGE","AGB", "H", "BA", "Trees_N",  "V", "shannon", "CV_DBH", "DBH"],
        "climate_zones": ["CT", "WT", "MS", "MT", "SS", "NS", "H", "MTrop", "MidTr"]
    }

    try:
        import cupy as cp
        _GPU_AVAILABLE = cp.cuda.is_available()
        if _GPU_AVAILABLE:
            logger.info("GPU detected via CuPy. Hardware acceleration ENABLED.")
    except Exception:
        _GPU_AVAILABLE = False
        logger.info("CuPy not found or GPU unavailable. CPU mode ENABLED.")

    # 1st Loop: Target Variables
    for current_var in CFG['vars']:
        logger.info(f"========== STARTING PROCESSING FOR TARGET: {current_var} ==========")
        
        # 2nd Loop: Climate Zones
        for z_id in CFG['climate_zones']:
            logger.info(f">>>>>> Checking Zone {z_id} for target {current_var} <<<<<<")
            
            o_d = os.path.join(CFG['out_base'], z_id, current_var)
            if zone_target_already_done(o_d, z_id, current_var):
                logger.info(f"SKIP Zone {z_id} [{current_var}] - mosaics already produced.")
                continue

            orig_r_d = os.path.join(CFG['raster_base'], z_id)
            orig_tifs = sorted(glob.glob(os.path.join(orig_r_d, f"{z_id}_*.tif")))
            
            if not orig_tifs:
                logger.error(f"Original raster directory missing or empty: {orig_r_d}")
                continue

            feat_path = os.path.join(CFG['model_base'], current_var, z_id, "Global_Selected_Features.csv")
            m_d = os.path.join(CFG['model_base'], current_var, z_id, "Final_Models")
            os.makedirs(o_d, exist_ok=True)

            if not os.path.exists(feat_path):
                logger.error(f"Feature selection file missing: {feat_path}")
                continue

            selected_df = pd.read_csv(feat_path)
            model_features_raw = selected_df['Feature'].tolist()
            model_features_norm = [normalize_name(f) for f in model_features_raw]

            m_paths = sorted(glob.glob(os.path.join(m_d, "*.joblib")))
            if not m_paths:
                logger.warning(f"No model files for Zone {z_id} [{current_var}]")
                continue

            # Read metadata straight from the mechanical HDD just once to calibrate bands
            tif_band_names = []
            try:
                with rasterio.open(orig_tifs[0]) as src:
                    raw_descs = list(src.descriptions)
                    if not any(raw_descs) or raw_descs[0] is None:
                        logger.error("No embedded band descriptions found in TIF.")
                        continue
                    tif_band_names = [normalize_name(d) for d in raw_descs]
            except (rasterio.errors.RasterioIOError, OSError) as e:
                logger.error(f"Failed to read raster metadata from {orig_tifs[0]}: {e}")
                continue

            feature_indices = []
            missing_feats = []
            for m_feat_raw, m_feat_norm in zip(model_features_raw, model_features_norm):
                if m_feat_norm in tif_band_names:
                    idx = tif_band_names.index(m_feat_norm)
                    feature_indices.append(idx)
                else:
                    missing_feats.append(m_feat_raw)

            if missing_feats:
                logger.error(f"[CRITICAL] Features missing in raster for {current_var}")
                continue
            if len(feature_indices) != len(model_features_raw):
                continue
            
            # Pre-load all 100 models ONCE for the entire Zone to save massive amounts of RAM and Time
            logger.info(f"Pre-loading {len(m_paths)} models for Zone {z_id} [{current_var}]...")
            loaded_models = []
            for mp in m_paths:
                wrapper = load_one_model(mp, use_gpu=_GPU_AVAILABLE)
                if wrapper is not None:
                    loaded_models.append(wrapper)

            if not loaded_models:
                logger.error(f"No valid models loaded for {z_id} [{current_var}]. Skipping.")
                continue

            ram_r_d = os.path.join(CFG['ramdisk_base'], z_id)
            batches = [orig_tifs[i:i + RAMDISK_BATCH_SIZE] for i in range(0, len(orig_tifs), RAMDISK_BATCH_SIZE)]
            total_batches = len(batches)
            
            logger.info(f"Zone {z_id} contains {len(orig_tifs)} files. Proceeding with {total_batches} batch(es).")

            # 3rd Loop: Batches
            for b_idx, batch_tifs in enumerate(batches, 1):
                logger.info(f"--- [Variable: {current_var} | Zone: {z_id}] Batch {b_idx}/{total_batches} ---")
                
                # Setup clean RAM disk
                shutil.rmtree(ram_r_d, ignore_errors=True)
                os.makedirs(ram_r_d, exist_ok=True)
                
                # Sync batch to RAM
                for src_file in tqdm(batch_tifs, desc=f"Loading Batch {b_idx} to RAM"):
                    dst_file = os.path.join(ram_r_d, os.path.basename(src_file))
                    try:
                        shutil.copy2(src_file, dst_file)
                    except Exception as e:
                        logger.error(f"Failed to copy {src_file}: {e}")

                # Gather successfully copied RAM tifs
                ram_tifs = sorted(glob.glob(os.path.join(ram_r_d, f"{z_id}_*.tif")))
                if not ram_tifs:
                    logger.warning(f"No files successfully copied to RAM disk for batch {b_idx}.")
                    continue

                # Execute concurrent predictions
                logger.info(f"Executing inference with {MAX_WORKERS} workers...")
                with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                    futures = [
                        executor.submit(predict_raster, tif, loaded_models, o_d, feature_indices, _GPU_AVAILABLE) 
                        for tif in ram_tifs
                    ]
                    for future in tqdm(as_completed(futures), total=len(ram_tifs), desc=f"Predicting"):
                        try:
                            future.result()
                        except Exception as e:
                            logger.error(f"Thread execution failed: {e}")

                # IMMEDIATE RELEASE OF RAM DISK AS REQUESTED
                logger.info(f"Batch {b_idx} complete. Releasing RAM disk immediately...")
                shutil.rmtree(ram_r_d, ignore_errors=True)
                
                # Force GC cycle after each batch
                gc.collect()
                if _GPU_AVAILABLE:
                    try:
                        cp.get_default_memory_pool().free_all_blocks()
                    except Exception:
                        pass

            # Unload models from memory after all batches for this Zone are complete
            del loaded_models
            gc.collect()

            # Run Mosaic for the completed Variable/Zone
            logger.info(f"Creating Final Mosaic for {current_var} Zone {z_id}...")
            run_mosaic(o_d, z_id, current_var)
            logger.info(f"--- Finished Zone {z_id} for target {current_var} ---\n")

        logger.info(f"========== FINISHED ALL ZONES FOR TARGET: {current_var} ==========\n")