# -*- coding: utf-8 -*-

import os
import sys
import glob
import math
import time
import logging
import platform
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box
from shapely.prepared import prep
from pyproj import CRS
from osgeo import gdal

gdal.UseExceptions()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("Stacked_Block_Builder")


VAR_DIR = "/data/China_mapping/variable"
MODEL_DIR = "/data/China_mapping/fina_model/"
SHP_PATH = "/data/China_mapping/climate_zone/clmate_zone.shp"
FOREST_MASK_PATH = "/data/China_mapping/forest_boundary/forest_bound.tif"
OUT_DIR = "/data/China_mapping/fina_variables"

DEM_FILES = [
    "/data/China_mapping/Tope/aspect.tif",
    "/data/China_mapping/Tope/DEM.tif",
    "/data/China_mapping/Tope/slop.tif"
]

ZONE_FIELD = os.environ.get("ZONE_FIELD", "clim_abbr")

STACK_BLOCK_SIZE = int(os.environ.get("STACK_BLOCK_SIZE", "512"))

GTIFF_BLOCK_SIZE = int(os.environ.get("GTIFF_BLOCK_SIZE", "256"))

TILE_WORKERS = int(os.environ.get("TILE_WORKERS", str(max(1, mp.cpu_count() - 1))))
TILE_WORKERS = max(1, min(TILE_WORKERS, mp.cpu_count()))

GDAL_CACHEMAX = os.environ.get("GDAL_CACHEMAX", "2048")
WARP_MEMORY_MB = int(os.environ.get("WARP_MEMORY_MB", "2048"))

OUTPUT_NODATA = float(os.environ.get("OUTPUT_NODATA", "-9999"))

OUTPUT_COMPRESS = os.environ.get("OUTPUT_COMPRESS", "none").lower()
ZSTD_LEVEL = int(os.environ.get("ZSTD_LEVEL", "1"))

RESAMPLE_ALG = os.environ.get("RESAMPLE_ALG", "near")
SKIP_EXISTING = os.environ.get("SKIP_EXISTING", "1") == "1"
OVERWRITE = os.environ.get("OVERWRITE", "0") == "1"

REBUILD_STACK_VRT = os.environ.get("REBUILD_STACK_VRT", "1") == "1"
SPARSE_OK = os.environ.get("SPARSE_OK", "YES")

DEFAULT_MP_METHOD = "fork" if platform.system().lower() == "linux" else "spawn"
MP_START_METHOD = os.environ.get("MP_START_METHOD", DEFAULT_MP_METHOD).strip().lower()

ONLY_ZONE = os.environ.get("ONLY_ZONE", "").strip()
QUIET = os.environ.get("QUIET", "0") == "1"


os.environ["GDAL_CACHEMAX"] = str(GDAL_CACHEMAX)
os.environ["GDAL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VSI_CACHE"] = os.environ.get("VSI_CACHE", "FALSE")
os.environ["GDAL_DISABLE_READDIR_ON_OPEN"] = "EMPTY_DIR"

gdal.SetConfigOption("GDAL_CACHEMAX", str(GDAL_CACHEMAX))
gdal.SetConfigOption("GDAL_NUM_THREADS", "1")
gdal.SetConfigOption("VSI_CACHE", os.environ.get("VSI_CACHE", "FALSE"))
gdal.SetConfigOption("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")


G_STACK_VRT = None
G_BAND_NAMES = None
G_REF_PROJ = None


def get_mp_context():
    try:
        return mp.get_context(MP_START_METHOD)
    except ValueError:
        logger.warning("Invalid MP_START_METHOD=%s, using spawn", MP_START_METHOD)
        return mp.get_context("spawn")


def safe_sql_value(value):
    return str(value).replace("'", "''")


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def get_reference_info(reference_path):
    ds = gdal.Open(reference_path, gdal.GA_ReadOnly)
    if ds is None:
        raise RuntimeError(f"Cannot open reference raster: {reference_path}")

    gt = ds.GetGeoTransform()
    proj = ds.GetProjection()
    width = ds.RasterXSize
    height = ds.RasterYSize

    ds = None

    if abs(gt[2]) > 1e-12 or abs(gt[4]) > 1e-12:
        raise RuntimeError("Rotated geotransform is not supported")

    if not proj:
        raise RuntimeError("Reference raster has no projection")

    return {
        "geotransform": gt,
        "projection": proj,
        "width": int(width),
        "height": int(height),
        "xres": abs(float(gt[1])),
        "yres": abs(float(gt[5]))
    }


def window_bounds(geotransform, col_off, row_off, width, height):
    x0 = geotransform[0] + col_off * geotransform[1]
    x1 = geotransform[0] + (col_off + width) * geotransform[1]
    y0 = geotransform[3] + row_off * geotransform[5]
    y1 = geotransform[3] + (row_off + height) * geotransform[5]

    minx = min(x0, x1)
    maxx = max(x0, x1)
    miny = min(y0, y1)
    maxy = max(y0, y1)

    return (float(minx), float(miny), float(maxx), float(maxy))


def bounds_to_aligned_window(bounds, reference_info):
    gt = reference_info["geotransform"]
    ref_width = reference_info["width"]
    ref_height = reference_info["height"]

    minx, miny, maxx, maxy = bounds

    col_values = [
        (minx - gt[0]) / gt[1],
        (maxx - gt[0]) / gt[1]
    ]
    row_values = [
        (miny - gt[3]) / gt[5],
        (maxy - gt[3]) / gt[5]
    ]

    col0 = int(math.floor(min(col_values)))
    col1 = int(math.ceil(max(col_values)))
    row0 = int(math.floor(min(row_values)))
    row1 = int(math.ceil(max(row_values)))

    col0 = max(0, min(ref_width, col0))
    col1 = max(0, min(ref_width, col1))
    row0 = max(0, min(ref_height, row0))
    row1 = max(0, min(ref_height, row1))

    return col0, row0, max(0, col1 - col0), max(0, row1 - row0)


def align_down(value, block_size):
    return (value // block_size) * block_size


def collect_all_variable_rasters():
    raster_items = []
    used_names = set()

    def add_raster(name, path):
        if not os.path.exists(path):
            logger.warning("Missing raster: %s", path)
            return

        if name in used_names:
            logger.warning("Duplicate raster name skipped: %s | %s", name, path)
            return

        used_names.add(name)
        raster_items.append((name, path))

    for path in DEM_FILES:
        name = os.path.splitext(os.path.basename(path))[0]
        add_raster(name, path)

    for path in sorted(glob.glob(os.path.join(VAR_DIR, "*.tif"))):
        name = os.path.splitext(os.path.basename(path))[0]
        if os.path.abspath(path) == os.path.abspath(FOREST_MASK_PATH):
            continue
        add_raster(name, path)

    if len(raster_items) == 0:
        raise RuntimeError("No independent variable rasters found")

    return raster_items


def write_band_manifest(raster_items):
    manifest_path = os.path.join(OUT_DIR, "band_manifest.csv")
    rows = []

    for idx, (name, path) in enumerate(raster_items, start=1):
        rows.append({
            "band_index": idx,
            "band_name": name,
            "source_path": path
        })

    pd.DataFrame(rows).to_csv(manifest_path, index=False)
    logger.info("Saved band manifest: %s", manifest_path)


def build_stack_vrt(raster_items):
    vrt_dir = os.path.join(OUT_DIR, "_vrt")
    ensure_dir(vrt_dir)

    vrt_path = os.path.join(vrt_dir, "all_independent_variables_stack.vrt")

    if os.path.exists(vrt_path) and not REBUILD_STACK_VRT:
        logger.info("Using existing stack VRT: %s", vrt_path)
        return vrt_path

    if os.path.exists(vrt_path):
        os.remove(vrt_path)

    raster_paths = [path for _, path in raster_items]

    logger.info("Building stack VRT with %d bands", len(raster_paths))

    vrt_options = gdal.BuildVRTOptions(
        separate=True,
        resolution="highest"
    )

    ds = gdal.BuildVRT(vrt_path, raster_paths, options=vrt_options)

    if ds is None:
        raise RuntimeError("Failed to build stack VRT")

    for band_index, (name, _) in enumerate(raster_items, start=1):
        band = ds.GetRasterBand(band_index)
        if band is not None:
            band.SetDescription(name)

    ds.FlushCache()
    ds = None

    logger.info("Saved stack VRT: %s", vrt_path)
    return vrt_path


def read_zones(reference_projection):
    gdf = gpd.read_file(SHP_PATH)

    if ZONE_FIELD not in gdf.columns:
        raise RuntimeError(f"Zone field not found: {ZONE_FIELD}")

    if gdf.crs is None:
        raise RuntimeError("Climate zone shapefile has no CRS")

    ref_crs = CRS.from_wkt(reference_projection)
    gdf = gdf.to_crs(ref_crs)

    if ONLY_ZONE:
        gdf = gdf[gdf[ZONE_FIELD].astype(str) == ONLY_ZONE]

    if gdf.empty:
        raise RuntimeError("No climate zones found")

    zones = []

    for zone_value, sub in gdf.groupby(ZONE_FIELD):
        zone_name = str(zone_value)
        geom = sub.geometry.union_all()

        if geom is None or geom.is_empty:
            continue

        zones.append({
            "zone": zone_name,
            "geometry": geom,
            "bounds": geom.bounds
        })

    zones = sorted(zones, key=lambda item: item["zone"])

    logger.info("Found zones: %s", [item["zone"] for item in zones])
    return zones


def build_zone_tasks(zone_info, reference_info):
    zone = zone_info["zone"]
    geometry = zone_info["geometry"]
    prepared_geometry = prep(geometry)

    col0, row0, width, height = bounds_to_aligned_window(
        zone_info["bounds"],
        reference_info
    )

    if width <= 0 or height <= 0:
        return []

    col_start = align_down(col0, STACK_BLOCK_SIZE)
    row_start = align_down(row0, STACK_BLOCK_SIZE)
    col_end = min(reference_info["width"], col0 + width)
    row_end = min(reference_info["height"], row0 + height)

    tasks = []
    block_id = 0
    gt = reference_info["geotransform"]

    zone_block_dir = os.path.join(OUT_DIR, zone)
    ensure_dir(zone_block_dir)

    for row_off in range(row_start, row_end, STACK_BLOCK_SIZE):
        block_height = min(STACK_BLOCK_SIZE, reference_info["height"] - row_off)

        if block_height <= 0:
            continue

        for col_off in range(col_start, col_end, STACK_BLOCK_SIZE):
            block_width = min(STACK_BLOCK_SIZE, reference_info["width"] - col_off)

            if block_width <= 0:
                continue

            bounds = window_bounds(gt, col_off, row_off, block_width, block_height)
            block_box = box(*bounds)

            if not prepared_geometry.intersects(block_box):
                continue

            out_name = (
                f"{zone}_r{row_off:06d}_c{col_off:06d}_"
                f"h{block_height}_w{block_width}.tif"
            )
            out_path = os.path.join(zone_block_dir, out_name)

            task = {
                "zone": zone,
                "block_id": block_id,
                "row_off": int(row_off),
                "col_off": int(col_off),
                "width": int(block_width),
                "height": int(block_height),
                "bounds": bounds,
                "out_path": out_path
            }

            tasks.append(task)
            block_id += 1

    logger.info("%s candidate blocks: %d", zone, len(tasks))
    return tasks


def get_creation_options():
    options = [
        "TILED=YES",
        f"BLOCKXSIZE={GTIFF_BLOCK_SIZE}",
        f"BLOCKYSIZE={GTIFF_BLOCK_SIZE}",
        "BIGTIFF=YES",
        f"SPARSE_OK={SPARSE_OK}"
    ]

    if OUTPUT_COMPRESS == "none":
        return options

    if OUTPUT_COMPRESS == "zstd":
        options.extend([
            "COMPRESS=ZSTD",
            f"ZSTD_LEVEL={ZSTD_LEVEL}",
            "PREDICTOR=3",
            "NUM_THREADS=1"
        ])
    elif OUTPUT_COMPRESS == "lzw":
        options.extend([
            "COMPRESS=LZW",
            "PREDICTOR=3",
            "NUM_THREADS=1"
        ])
    elif OUTPUT_COMPRESS == "deflate":
        options.extend([
            "COMPRESS=DEFLATE",
            "PREDICTOR=3",
            "NUM_THREADS=1"
        ])
    else:
        options.append(f"COMPRESS={OUTPUT_COMPRESS.upper()}")

    return options


def init_tile_worker(stack_vrt_path, band_names, reference_projection):
    global G_STACK_VRT
    global G_BAND_NAMES
    global G_REF_PROJ

    G_STACK_VRT = stack_vrt_path
    G_BAND_NAMES = band_names
    G_REF_PROJ = reference_projection

    gdal.SetConfigOption("GDAL_CACHEMAX", str(GDAL_CACHEMAX))
    gdal.SetConfigOption("GDAL_NUM_THREADS", "1")
    gdal.SetConfigOption("VSI_CACHE", os.environ.get("VSI_CACHE", "FALSE"))
    gdal.SetConfigOption("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")


def warp_mask_for_block(task):
    zone_sql = f"{ZONE_FIELD} = '{safe_sql_value(task['zone'])}'"

    options = gdal.WarpOptions(
        format="MEM",
        outputBounds=list(task["bounds"]),
        width=int(task["width"]),
        height=int(task["height"]),
        dstSRS=G_REF_PROJ,
        cutlineDSName=SHP_PATH,
        cutlineWhere=zone_sql,
        cropToCutline=False,
        dstNodata=OUTPUT_NODATA,
        outputType=gdal.GDT_Float32,
        resampleAlg="near",
        warpMemoryLimit=WARP_MEMORY_MB,
        multithread=False
    )

    ds = gdal.Warp("", FOREST_MASK_PATH, options=options)

    if ds is None:
        raise RuntimeError("Mask warp failed")

    band = ds.GetRasterBand(1)
    arr = band.ReadAsArray()

    if arr is None:
        ds = None
        raise RuntimeError("Mask array read failed")

    arr = np.asarray(arr, dtype=np.float32)
    ds = None

    return arr


def apply_forest_mask(tile_path, mask_arr):
    ds = gdal.Open(tile_path, gdal.GA_Update)

    if ds is None:
        raise RuntimeError(f"Cannot open output tile for update: {tile_path}")

    nonforest = (mask_arr != 1)

    total_bands = ds.RasterCount

    for band_index in range(1, total_bands + 1):
        band = ds.GetRasterBand(band_index)

        if band is None:
            continue

        if band_index <= len(G_BAND_NAMES):
            band.SetDescription(G_BAND_NAMES[band_index - 1])

        band.SetNoDataValue(OUTPUT_NODATA)

        arr = band.ReadAsArray()

        if arr is None:
            raise RuntimeError(f"Failed to read band {band_index} from {tile_path}")

        arr = np.asarray(arr, dtype=np.float32)
        arr[nonforest] = OUTPUT_NODATA
        arr[~np.isfinite(arr)] = OUTPUT_NODATA

        band.WriteArray(arr)
        band.FlushCache()

        del arr

    ds.FlushCache()
    ds = None

    del nonforest


def process_tile(task):
    t0 = time.time()

    out_path = task["out_path"]
    tmp_path = out_path + f".pid{os.getpid()}.tmp.tif"

    try:
        old_tmps = glob.glob(out_path + "*.tmp.tif")
        for orphan in old_tmps:
            os.remove(orphan)
    except Exception:
        pass

    if SKIP_EXISTING and os.path.exists(out_path) and not OVERWRITE:
        return {
            "zone": task["zone"],
            "block_id": task["block_id"],
            "row_off": task["row_off"],
            "col_off": task["col_off"],
            "width": task["width"],
            "height": task["height"],
            "out_path": out_path,
            "status": "exists",
            "forest_pixels": -1,
            "elapsed_seconds": 0.0,
            "error": ""
        }

    try:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

        if OVERWRITE and os.path.exists(out_path):
            os.remove(out_path)

        mask_arr = warp_mask_for_block(task)
        forest_pixels = int(np.count_nonzero(mask_arr == 1))

        if forest_pixels == 0:
            return {
                "zone": task["zone"],
                "block_id": task["block_id"],
                "row_off": task["row_off"],
                "col_off": task["col_off"],
                "width": task["width"],
                "height": task["height"],
                "out_path": "",
                "status": "skipped_empty_forest",
                "forest_pixels": 0,
                "elapsed_seconds": time.time() - t0,
                "error": ""
            }

        zone_sql = f"{ZONE_FIELD} = '{safe_sql_value(task['zone'])}'"

        warp_options = gdal.WarpOptions(
            format="GTiff",
            outputBounds=list(task["bounds"]),
            width=int(task["width"]),
            height=int(task["height"]),
            dstSRS=G_REF_PROJ,
            cutlineDSName=SHP_PATH,
            cutlineWhere=zone_sql,
            cropToCutline=False,
            dstNodata=OUTPUT_NODATA,
            outputType=gdal.GDT_Float32,
            resampleAlg=RESAMPLE_ALG,
            creationOptions=get_creation_options(),
            warpMemoryLimit=WARP_MEMORY_MB,
            multithread=False
        )

        ds = gdal.Warp(tmp_path, G_STACK_VRT, options=warp_options)

        if ds is None:
            raise RuntimeError("Stack warp failed")

        for band_index, band_name in enumerate(G_BAND_NAMES, start=1):
            band = ds.GetRasterBand(band_index)
            if band is not None:
                band.SetDescription(band_name)
                band.SetNoDataValue(OUTPUT_NODATA)

        ds.FlushCache()
        ds = None

        apply_forest_mask(tmp_path, mask_arr)

        os.replace(tmp_path, out_path)

        return {
            "zone": task["zone"],
            "block_id": task["block_id"],
            "row_off": task["row_off"],
            "col_off": task["col_off"],
            "width": task["width"],
            "height": task["height"],
            "out_path": out_path,
            "status": "created",
            "forest_pixels": forest_pixels,
            "elapsed_seconds": time.time() - t0,
            "error": ""
        }

    except Exception as exc:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass

        return {
            "zone": task["zone"],
            "block_id": task["block_id"],
            "row_off": task["row_off"],
            "col_off": task["col_off"],
            "width": task["width"],
            "height": task["height"],
            "out_path": out_path,
            "status": "error",
            "forest_pixels": -1,
            "elapsed_seconds": time.time() - t0,
            "error": repr(exc)
        }


def run_zone(zone_info, reference_info, stack_vrt_path, band_names):
    zone = zone_info["zone"]
    zone_dir = os.path.join(OUT_DIR, zone)
    ensure_dir(zone_dir)

    tasks = build_zone_tasks(zone_info, reference_info)

    if not tasks:
        logger.warning("%s has no candidate blocks", zone)
        return

    manifest_path = os.path.join(zone_dir, f"{zone}_block_manifest.csv")

    logger.info(
        "%s start | tasks=%d | workers=%d | block_size=%d | bands=%d",
        zone,
        len(tasks),
        TILE_WORKERS,
        STACK_BLOCK_SIZE,
        len(band_names)
    )

    ctx = get_mp_context()
    results = []

    with ProcessPoolExecutor(
        max_workers=TILE_WORKERS,
        mp_context=ctx,
        initializer=init_tile_worker,
        initargs=(stack_vrt_path, band_names, reference_info["projection"])
    ) as executor:

        future_to_task = {
            executor.submit(process_tile, task): task
            for task in tasks
        }

        completed = 0
        created = 0
        skipped = 0
        existing = 0
        errors = 0
        t0 = time.time()

        for future in as_completed(future_to_task):
            result = future.result()
            results.append(result)

            completed += 1
            status = result["status"]

            if status == "created":
                created += 1
            elif status == "exists":
                existing += 1
            elif status.startswith("skipped"):
                skipped += 1
            elif status == "error":
                errors += 1
                logger.error(
                    "%s block error | row=%s | col=%s | error=%s",
                    zone,
                    result["row_off"],
                    result["col_off"],
                    result["error"]
                )

            if not QUIET and (completed % 10 == 0 or completed == len(tasks)):
                elapsed = time.time() - t0
                rate = completed / max(1e-6, elapsed)
                logger.info(
                    "%s progress | %d/%d | created=%d | existing=%d | skipped=%d | errors=%d | rate=%.2f blocks/s",
                    zone,
                    completed,
                    len(tasks),
                    created,
                    existing,
                    skipped,
                    errors,
                    rate
                )

    df = pd.DataFrame(results)
    df = df.sort_values(["row_off", "col_off", "block_id"])
    df.to_csv(manifest_path, index=False)

    logger.info(
        "%s finished | manifest=%s | created=%d | existing=%d | skipped=%d | errors=%d",
        zone,
        manifest_path,
        int((df["status"] == "created").sum()),
        int((df["status"] == "exists").sum()),
        int(df["status"].astype(str).str.startswith("skipped").sum()),
        int((df["status"] == "error").sum())
    )


def main():
    ensure_dir(OUT_DIR)

    logger.info("=" * 80)
    logger.info("Independent variable block stack builder")
    logger.info("Output directory: %s", OUT_DIR)
    logger.info("Stack block size: %d", STACK_BLOCK_SIZE)
    logger.info("Tile workers: %d", TILE_WORKERS)
    logger.info("GDAL_CACHEMAX: %s", GDAL_CACHEMAX)
    logger.info("WARP_MEMORY_MB: %d", WARP_MEMORY_MB)
    logger.info("Output compression: %s", OUTPUT_COMPRESS)
    logger.info("Resampling: %s", RESAMPLE_ALG)
    logger.info("MP start method: %s", MP_START_METHOD)
    logger.info("=" * 80)

    reference_info = get_reference_info(FOREST_MASK_PATH)

    raster_items = collect_all_variable_rasters()
    band_names = [name for name, _ in raster_items]

    logger.info("Total output bands: %d", len(band_names))
    if len(band_names) > 0:
        logger.info("Band 1: %s", band_names[0])

    write_band_manifest(raster_items)
    stack_vrt_path = build_stack_vrt(raster_items)

    zones = read_zones(reference_info["projection"])

    for zone_info in zones:
        run_zone(
            zone_info=zone_info,
            reference_info=reference_info,
            stack_vrt_path=stack_vrt_path,
            band_names=band_names
        )

    logger.info("All zone stacks completed")


if __name__ == "__main__":
    main()