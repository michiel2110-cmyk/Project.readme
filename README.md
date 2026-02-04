
# **Polder analysis from ArcGIS**

This project is to show the code to get to the results of the 'Shifting Shores: Historical Inundation and Land‑Cover Change' guided-research report. Created by Michiel van Dijk under supervision of Dr. Jim van Belzen (NIOZ) and Prof. Dr. Maarten Kleinhans (Utrecht University). Each section has a header with what code it contains and some 'how to use' parts. 


## Imports
```python
import os
import math
import time
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
from rasterio.features import rasterize, shapes
from rasterio.transform import from_origin
import matplotlib.pyplot as plt
from typing import Optional, Dict, Any
import numpy as np
import math
import warnings
from matplotlib.ticker import ScalarFormatter
from matplotlib.collections import LineCollection
from matplotlib.animation import FuncAnimation
from scipy.ndimage import label
from scipy.stats import norm
from typing import Dict, List, Tuple, Any, Optional
from tqdm import trange
from shapely.ops import unary_union
from shapely.geometry import LineString
from matplotlib.patches import Patch
from shapely.geometry import shape as shp_shape
import json
```
## **Helper functions**
These are loaded and are later used, this part includes the base folder etc. which need to be changed to the users pathing of ArcGIS etc.
```python

# scipy functions required for component detection/dilation and distances
try:
    from scipy.ndimage import binary_dilation, label, distance_transform_edt, generate_binary_structure
except Exception:
    binary_dilation = None
    label = None
    distance_transform_edt = None
    generate_binary_structure = None

# tqdm fallback
try:
    from tqdm import tqdm
except Exception:
    def tqdm(it, **kwargs):
        for x in it:
            yield x

# Pathing to GDB
BASE_FOLDER = r"C:\Users\michi\Documents\NIOZ\Historische analyze"
GDB_PATH = r"C:\Users\michi\Documents\ArcGIS Pro 3.3\ArcGISPro\ArcGIS\Projects\MyProject\MyProject.gdb"

ISLANDS = [
    "Tholen", "Noord_Beveland", "Zuid_Beveland",
    "Schouwen_Duiveland", "Walcheren", "Zeeuws_Vlaanderen"
]

# Starting values etc
TARGET_CRS = 28992  # RD_new
# Base raster settings
GRID_RES = 100             # size of pixels in meters (used to compute hectare area)
SUBSAMPLE_FACTOR = 3       # sub-pixel resolution for coverage rasterization
COVERAGE_THRESHOLD = 0.5   # fraction required to declare parent pixel "land" or 'inundated'

START_YEAR_CAP = 1250  # startyear for the analysis
TARGET_YEAR = 1950  # end year
YEARS = np.arange(START_YEAR_CAP, TARGET_YEAR + 1)
YEARS_REL = YEARS - START_YEAR_CAP

SHORT_INUND_YEARS = 2  # episodes <= this are "short" and excluded from inundated-pool analyses as described in methods
SHOW_PLOTS = True
SAVE_PLOTS = False
OUT_DIR = "pixel_survival_output_recurrent"

# Reclaimed detection parameters (used inside run)
STEP_YEARS_DETECT = 25
AREA_FACTOR = 1.1
MAX_DILATION_ITERS = 5
CONNECTIVITY = 2  # 1 -> 4-connectivity, 2 -> 8-connectivity

# Safety limit for high-resolution raster size
MAX_HR_PIXELS = 8000 * 8000
PRINT_TO_CONSOLE = True
A5_DENOM_EPS = 1e-12  # stability denominator for A5

# Marsh staged-growth parameters
ENABLE_MARSH_STAGED_GROWTH = True
INITIAL_MARSH_SEED_FRACTION = 0.10  # fraction of earliest snapshot mask seeded at start_ref
ALLOW_SHRINKAGE = True
PER_ISLAND_GROWTH = True
PREFER_SEED_ORDER = "polder_then_marsh"  # use land/polder as primary seed

# --- New params for pre-reclaim per-polder growth ---
ENABLE_PRE_RECLAIM_GROWTH = True
PRE_RECLAIM_BUFFER_FACTOR = 1.10   # grow to 1.1 * eventual polder area before reclamation (salt marsh)
PRE_RECLAIM_PER_POLY = True        # per-polygon (per-polder) growth 
PRE_RECLAIM_MIN_LEAD_YEARS = 1     # if a polder starts < this many years ahead, will still attempt but years_remaining >=1

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

# ------------------ helper: deterministic candidate selection ------------------
def add_pixels_closest(seed_mask, candidate_mask, n_add, distance_from_seed=None):
    """
    Deterministically select up to n_add pixels from candidate_mask, preferring pixels
    with smallest distance_from_seed (smaller better). If distance_from_seed is None,
    it will be computed as distance_transform_edt(~seed_mask) (i.e., distance to nearest seed).
    Returns a boolean mask with chosen pixels set True.
    """
    if n_add <= 0:
        return np.zeros_like(candidate_mask, dtype=bool)
    if distance_from_seed is None:
        if distance_transform_edt is None:
            raise RuntimeError("scipy.ndimage.distance_transform_edt not available")
        # distance to nearest seed: distance_transform_edt on inverse of seed_mask
        dist = distance_transform_edt(~seed_mask)
    else:
        dist = distance_from_seed

    # get candidate indices
    cand_idxs = np.argwhere(candidate_mask)
    if cand_idxs.size == 0:
        return np.zeros_like(candidate_mask, dtype=bool)

    # extract distances for candidates
    cand_dist = dist[cand_idxs[:,0], cand_idxs[:,1]]
    # sort ascending by distance (closest first)
    order = np.argsort(cand_dist, kind='stable')
    k = min(n_add, len(order))
    chosen_idxs = cand_idxs[order[:k]]
    sel = np.zeros(candidate_mask.size, dtype=bool)
    flat_idx = chosen_idxs[:,0] * candidate_mask.shape[1] + chosen_idxs[:,1]
    sel[flat_idx] = True
    return sel.reshape(candidate_mask.shape)

# ------------------ helper: blob-style deterministic growth (original centroid-based) ------------------
def add_pixels_blob(seeds_mask, candidate_mask, n_add, distance_from_seed=None, connectivity=2):
    """
    Deterministic 'blob' growth (centroid-based partial-ring ordering).

    Algorithm:
      - Grow by successive dilation rings from seeds using scipy.ndimage.binary_dilation.
      - If a whole ring fits the quota, take it; otherwise select a deterministic subset
        of the ring prioritized by closeness to the seed centroid (closest first).
      - Falls back to add_pixels_closest or row-major selection when scipy is not available.
    """
    if n_add <= 0:
        return np.zeros_like(candidate_mask, dtype=bool)

    # fallback if scipy missing or candidate empty
    if binary_dilation is None or generate_binary_structure is None:
        try:
            return add_pixels_closest(seeds_mask, candidate_mask, n_add, distance_from_seed=distance_from_seed)
        except Exception:
            cand_idxs = np.argwhere(candidate_mask)
            sel = np.zeros(candidate_mask.size, dtype=bool)
            if cand_idxs.size == 0:
                return sel.reshape(candidate_mask.shape)
            k = min(n_add, len(cand_idxs))
            flat_idx = cand_idxs[:k,0] * candidate_mask.shape[1] + cand_idxs[:k,1]
            sel[flat_idx] = True
            return sel.reshape(candidate_mask.shape)

    H, W = candidate_mask.shape
    chosen = np.zeros_like(candidate_mask, dtype=bool)
    remaining = candidate_mask.copy()

    seeds = seeds_mask.copy().astype(bool)

    # If there are no seeds, fall back to nearest selection
    if not seeds.any():
        try:
            return add_pixels_closest(seeds_mask, candidate_mask, n_add, distance_from_seed=distance_from_seed)
        except Exception:
            cand_idxs = np.argwhere(candidate_mask)
            sel = np.zeros(candidate_mask.size, dtype=bool)
            if cand_idxs.size == 0:
                return sel.reshape(candidate_mask.shape)
            k = min(n_add, len(cand_idxs))
            flat_idx = cand_idxs[:k,0] * candidate_mask.shape[1] + cand_idxs[:k,1]
            sel[flat_idx] = True
            return sel.reshape(candidate_mask.shape)

    struct = generate_binary_structure(2, connectivity)
    added = 0

    while added < n_add:
        # ring = one-step dilation excluding existing seeds, restricted to remaining candidates
        ring = binary_dilation(seeds, structure=struct) & (~seeds) & remaining
        if not ring.any():
            # fallback: pick nearest remaining candidates
            remaining_quota = n_add - added
            try:
                pick = add_pixels_closest(seeds, remaining, remaining_quota, distance_from_seed=distance_from_seed)
                chosen |= pick
                added += int(pick.sum())
            except Exception:
                idxs = np.argwhere(remaining)
                if idxs.size:
                    k = min(remaining_quota, len(idxs))
                    flat_idx = idxs[:k,0] * W + idxs[:k,1]
                    sel = np.zeros(candidate_mask.size, dtype=bool)
                    sel[flat_idx] = True
                    sel = sel.reshape(candidate_mask.shape)
                    chosen |= sel
                    added += int(sel.sum())
            break

        ring_count = int(ring.sum())
        need = n_add - added
        if ring_count <= need:
            # take whole ring
            chosen |= ring
            remaining[ring] = False
            seeds |= ring
            added += ring_count
            continue
        else:
            # partial ring: choose deterministic subset near seed centroid
            ys_seed, xs_seed = np.nonzero(seeds)
            if ys_seed.size == 0:
                idxs = np.argwhere(ring)
                k = min(need, idxs.shape[0])
                flat_idx = idxs[:k,0] * W + idxs[:k,1]
                sel = np.zeros(candidate_mask.size, dtype=bool)
                sel[flat_idx] = True
                sel = sel.reshape(candidate_mask.shape)
                chosen |= sel
                added += int(sel.sum())
                break

            cy = ys_seed.mean(); cx = xs_seed.mean()
            ys, xs = np.nonzero(ring)
            d2 = (ys - cy)**2 + (xs - cx)**2
            order = np.argsort(d2, kind='stable')
            pick_idx = order[:need]
            sel = np.zeros(candidate_mask.size, dtype=bool)
            flat_idx = ys[pick_idx] * W + xs[pick_idx]
            sel[flat_idx] = True
            sel = sel.reshape(candidate_mask.shape)
            chosen |= sel
            added += int(sel.sum())
            break

    return chosen

# -----------------------------------------------------------------------------
# Loading polygons and raster helpers
def load_and_combine_polygons(base_folder, gdb_path, islands, target_crs=TARGET_CRS):
    all_gdfs = []
    for isl in islands:
        shp_path = os.path.join(base_folder, f"polders_{isl}_timeline.shp")
        if not os.path.exists(shp_path):
            warnings.warn(f"{shp_path} not found — skipping {isl}")
            continue
        gdf = gpd.read_file(shp_path)
        if gdf.crs is None:
            raise RuntimeError(f"{shp_path} has no CRS")
        gdf = gdf.to_crs(target_crs)
        if "StartYear" in gdf.columns:
            gdf = gdf.rename(columns={"StartYear": "start_year"})
        if "EndYear" in gdf.columns:
            gdf = gdf.rename(columns={"EndYear": "end_year"})
        gdf["Island"] = isl
        if "Polder" not in gdf.columns:
            gdf["Polder"] = [f"{isl}_poly_{i}" for i in range(len(gdf))]
        all_gdfs.append(gdf)

    # optional extra layer of the inundated file
    try:
        gdf_am = gpd.read_file(gdb_path, layer="Inundated_polders_Amersfoort")
        if gdf_am is not None and not gdf_am.empty:
            if gdf_am.crs is None:
                raise RuntimeError("Amersfoort layer has no CRS")
            gdf_am = gdf_am.to_crs(target_crs)
            if "Start_year" in gdf_am.columns:
                gdf_am = gdf_am.rename(columns={"Start_year":"start_year"})
            if "End_year" in gdf_am.columns:
                gdf_am = gdf_am.rename(columns={"End_year":"end_year"})
            if "Island" not in gdf_am.columns:
                gdf_am["Island"] = "Amersfoort"
            if "Polder" not in gdf_am.columns:
                gdf_am["Polder"] = [f"Amersfoort_{i}" for i in range(len(gdf_am))]
            all_gdfs.append(gdf_am)
    except Exception:
        pass

    if not all_gdfs:
        raise RuntimeError("No polygon inputs found.")
    df_all = pd.concat(all_gdfs, ignore_index=True, sort=False)
    df_all = gpd.GeoDataFrame(df_all, geometry='geometry', crs=target_crs)
    if 'start_year' not in df_all.columns: df_all['start_year'] = np.nan
    if 'end_year' not in df_all.columns: df_all['end_year'] = np.nan
    df_all['start_year'] = pd.to_numeric(df_all['start_year'], errors='coerce')
    df_all['end_year'] = pd.to_numeric(df_all['end_year'], errors='coerce')
    df_all = df_all.reset_index(drop=False).rename(columns={"index":"orig_idx"})
    df_all.loc[df_all['start_year'] < START_YEAR_CAP, 'start_year'] = START_YEAR_CAP
    return df_all

def estimate_raster_shape_transform(df_all, grid_res):
    minx, miny, maxx, maxy = df_all.total_bounds
    buf = grid_res
    minx -= buf; miny -= buf; maxx += buf; maxy += buf
    width = int(math.ceil((maxx - minx) / grid_res))
    height = int(math.ceil((maxy - miny) / grid_res))
    transform = from_origin(minx, maxy, grid_res, grid_res)
    return width, height, transform, (minx, miny, maxx, maxy)

def rasterize_union_year(df_all, year, width, height, transform, sf=1, coverage_threshold=0.5):
    active = df_all[(df_all['start_year'] <= year) & (df_all['end_year'] > year)].reset_index(drop=True)
    if active.empty:
        return np.zeros((height, width), dtype=bool), active
    if sf <= 1:
        shapes = ((geom, 1) for geom in active.geometry)
        labels = rasterize(shapes, out_shape=(height, width), transform=transform, fill=0, all_touched=False, dtype=np.uint8)
        return labels > 0, active
    hr_pixel_size = transform.a / sf
    minx = transform.c; maxy = transform.f
    hr_transform = from_origin(minx, maxy, hr_pixel_size, hr_pixel_size)
    hr_width = width * sf; hr_height = height * sf
    if hr_width * hr_height > MAX_HR_PIXELS:
        raise MemoryError("HR raster too large - reduce SUBSAMPLE_FACTOR or increase GRID_RES.")
    shapes = ((geom, 1) for geom in active.geometry)
    labels_hr = rasterize(shapes, out_shape=(hr_height, hr_width), transform=hr_transform, fill=0, all_touched=False, dtype=np.uint8)
    try:
        lr = labels_hr.reshape(height, sf, width, sf)
        lr = lr.transpose(0,2,1,3).reshape(height, width, sf*sf)
    except Exception:
        lr = np.empty((height, width, sf*sf), dtype=labels_hr.dtype)
        for i in range(height):
            for j in range(width):
                lr[i,j,:] = labels_hr[i*sf:(i+1)*sf, j*sf:(j+1)*sf].ravel()
    sub_counts = (lr != 0).sum(axis=2)
    threshold = math.ceil(sf * sf * float(coverage_threshold))
    land_coarse = sub_counts >= threshold
    return land_coarse, active

def rasterize_island_ids(df_all, width, height, transform):
    dissolved = df_all.dissolve(by='Island', as_index=False)
    islands = list(dissolved['Island'])
    island_to_id = {name: i+1 for i, name in enumerate(islands)}
    shapes = [(row.geometry, island_to_id[row.Island]) for row in dissolved.itertuples()]
    island_raster = rasterize(shapes, out_shape=(height, width), transform=transform, fill=0, all_touched=False, dtype=np.int32)
    return island_raster, island_to_id
```
## **Main run and creating the pixels and salt marsh growth**
This run function takes a while to run as it first loads all polygon data from the GIS and then backtracks the salt marsh growth on these maps. Firstly the polygon data is transferred to pixels (rasterization). Then, years are added from bathymetry maps with 'best-match' years. Additionally, vlas fall under 'beschrijvi' column which could not be changed within ArcGIS. It starts the salt marsh as a seedling from which it then grows. It is all transferred to globals() in order to keep it in session. Can play with the json or saved_paths part to keep this data saved, so you do not have to run it multiple times, as it takes a while. 
```python
def run():
    if SAVE_PLOTS:
        ensure_dir(OUT_DIR)

    print("Loading polygons...")
    df_all = load_and_combine_polygons(BASE_FOLDER, GDB_PATH, ISLANDS)
    width, height, transform, _ = estimate_raster_shape_transform(df_all, GRID_RES)
    px_area_ha = (GRID_RES * GRID_RES) / 10000.0

    # rasterize island ids early so precompute can use island IDs
    island_raster, island_to_id = rasterize_island_ids(df_all, width, height, transform)
    id_to_island = {v:k for k,v in island_to_id.items()}

    # ---------------- Precompute per-polder pre-reclaim targets (ownership) ----------------
    pre_reclaim_targets = {}  # keyed by start_year -> list of target dicts
    # ownership raster: 0 = unassigned, >0 assignment id index (1..N)
    ownership_raster = np.zeros((height, width), dtype=np.int32)
    target_list = []

    # select candidate polygon rows that represent polders with finite start_year within simulation window
    candidates = df_all[df_all['start_year'].notnull() & (df_all['start_year'] >= START_YEAR_CAP)].copy()
    # sort ascending by start_year so earliest gets ownership first
    candidates = candidates.sort_values(['start_year', 'orig_idx']).reset_index(drop=True)

    if ENABLE_PRE_RECLAIM_GROWTH and not candidates.empty:
        print("Precomputing per-polder targets for pre-reclaim growth...")
        tid = 1
        for i, row in tqdm(candidates.iterrows(), total=len(candidates), desc="precompute targets"):
            sy = int(row['start_year']) if not np.isnan(row['start_year']) else None
            if sy is None:
                continue
            if sy > TARGET_YEAR:
                # Skip targets completely outside simulation horizon (optional)
                continue
            geom = row.geometry
            # rasterize single polygon to mask (respect SUBSAMPLE_FACTOR)
            try:
                # rasterize this single geom
                mask = np.zeros((height, width), dtype=bool)
                if SUBSAMPLE_FACTOR <= 1:
                    labels = rasterize(((geom, 1),), out_shape=(height, width), transform=transform, fill=0, all_touched=False, dtype=np.uint8)
                    mask = labels > 0
                else:
                    hr_pixel_size = transform.a / SUBSAMPLE_FACTOR
                    minx = transform.c; maxy = transform.f
                    hr_transform = from_origin(minx, maxy, hr_pixel_size, hr_pixel_size)
                    hr_width = width * SUBSAMPLE_FACTOR; hr_height = height * SUBSAMPLE_FACTOR
                    labels_hr = rasterize(((geom,1),), out_shape=(hr_height, hr_width), transform=hr_transform, fill=0, all_touched=False, dtype=np.uint8)
                    try:
                        lr = labels_hr.reshape(height, SUBSAMPLE_FACTOR, width, SUBSAMPLE_FACTOR)
                        lr = lr.transpose(0,2,1,3).reshape(height, width, SUBSAMPLE_FACTOR*SUBSAMPLE_FACTOR)
                    except Exception:
                        lr = np.empty((height, width, SUBSAMPLE_FACTOR*SUBSAMPLE_FACTOR), dtype=labels_hr.dtype)
                        for ii in range(height):
                            for jj in range(width):
                                lr[ii,jj,:] = labels_hr[ii*SUBSAMPLE_FACTOR:(ii+1)*SUBSAMPLE_FACTOR, jj*SUBSAMPLE_FACTOR:(jj+1)*SUBSAMPLE_FACTOR].ravel()
                    sub_counts = (lr != 0).sum(axis=2)
                    threshold = math.ceil(SUBSAMPLE_FACTOR * SUBSAMPLE_FACTOR * float(COVERAGE_THRESHOLD))
                    mask = sub_counts >= threshold
            except Exception as exc:
                warnings.warn(f"Rasterizing pre-reclaim polygon failed for row {i}: {exc}")
                continue

            if not mask.any():
                continue
            target_px = int(mask.sum())
            buffer_px = int(math.ceil(float(target_px) * float(PRE_RECLAIM_BUFFER_FACTOR)))

            # island id for the polygon if available
            isl_name = row.get('Island', None)
            isl_id = island_to_id.get(isl_name, 0) if isl_name is not None else 0

            # enforce ownership: assign pixels not yet claimed (earliest start_year wins)
            owned_mask = mask & (ownership_raster == 0)
            if not owned_mask.any():
                # nothing left unique to this polygon -> skip
                continue
            # mark ownership raster with tid for owned pixels
            ownership_raster[owned_mask] = tid

            entry = {
                "target_id": int(tid),
                "orig_idx": int(row.get("orig_idx", -1)),
                "start_year": int(sy),
                "island_name": isl_name,
                "island_id": int(isl_id),
                "mask": mask,
                "owned_mask": owned_mask,
                "target_px": int(target_px),
                "buffer_px": int(buffer_px),
                "geom": geom
            }
            pre_reclaim_targets.setdefault(int(sy), []).append(entry)
            target_list.append(entry)
            tid += 1

        print(f"  pre_reclaim_targets computed: {len(target_list)} targets; ownership assigned to raster")
    else:
        pre_reclaim_targets = {}
        ownership_raster = np.zeros((height, width), dtype=np.int32)
        target_list = []

    # export pre_reclaim structures for inspection
    globals()['pre_reclaim_targets'] = pre_reclaim_targets
    globals()['pre_reclaim_ownership_raster'] = ownership_raster
    globals()['pre_reclaim_target_list'] = target_list

    # --- Begin paste block: load marsh snapshots from GDB and build preseed_targets ---
    # map from GDB layer year -> actual snapshot year to use in simulation
    year_map = {
        1600: 1620,
        1800: 1821,
        1900: 1899
    }

    # attribute values in the marsh layers: 1 = 'kwelders' (salt marsh), 2 = 'droogvallend bij eb'
    KEEP_BESCHRIJVI = ['kwelders']   # change to [1,2] if you want both types

    print("Loading marsh (bathy) snapshot layers from GDB and rasterizing to preseed masks...")
    marsh_gdfs = []
    for gdb_key, actual_year in year_map.items():
        layer_name = f"Bathy_{gdb_key}_Clipped"
        try:
            mg = gpd.read_file(GDB_PATH, layer=layer_name)
            if mg is None or mg.empty:
                warnings.warn(f"Layer {layer_name} empty or not found in GDB — skipping.")
                continue
            # reproject to target CRS of the model
            try:
                mg = mg.to_crs(TARGET_CRS)
            except Exception:
                pass

            # keep only requested attribute classes if column exists
            if 'beschrijvi' in mg.columns:
                # robust handling: accept label text or numeric codes
                vals = mg['beschrijvi'].dropna().unique()
                if all(isinstance(v, (int, np.integer, float)) for v in vals):
                    try:
                        numeric_keep = [int(x) for x in KEEP_BESCHRIJVI]
                        mg = mg[mg['beschrijvi'].isin(numeric_keep)].copy()
                    except Exception:
                        # if KEEP_BESCHRIJVI are strings like 'kwelders', try label match below
                        pass
                else:
                    mg['__besch_lc'] = mg['beschrijvi'].astype(str).str.lower()
                    keep_lc = [str(x).lower() for x in KEEP_BESCHRIJVI]
                    mask = mg['__besch_lc'].apply(lambda s: any(kl == s or kl in s for kl in keep_lc))
                    mg = mg[mask].copy()
                    mg.drop(columns=['__besch_lc'], inplace=True, errors='ignore')

            if mg.empty:
                warnings.warn(f"No features left after filtering {layer_name} — skipping.")
                continue

            mg['snapshot_year'] = int(actual_year)
            marsh_gdfs.append(mg)
            print(f"  loaded {layer_name} -> year {actual_year} ({len(mg)} features)")
        except Exception as e:
            warnings.warn(f"Could not read layer {layer_name} from {GDB_PATH}: {e}")
            continue

    if marsh_gdfs:
        df_marsh = gpd.GeoDataFrame(pd.concat(marsh_gdfs, ignore_index=True, sort=False), geometry='geometry', crs=TARGET_CRS)
        globals()['df_marsh'] = df_marsh

        def _rasterize_gdf_to_mask(gdf, width, height, transform, sf=SUBSAMPLE_FACTOR, coverage_threshold=COVERAGE_THRESHOLD):
            if gdf is None or gdf.empty:
                return np.zeros((height, width), dtype=bool)
            if sf <= 1:
                shapes_iter = ((geom, 1) for geom in gdf.geometry)
                labels = rasterize(shapes_iter, out_shape=(height, width), transform=transform, fill=0, all_touched=False, dtype=np.uint8)
                return labels > 0
            hr_pixel_size = transform.a / sf
            minx = transform.c; maxy = transform.f
            hr_transform = from_origin(minx, maxy, hr_pixel_size, hr_pixel_size)
            hr_width = width * sf; hr_height = height * sf
            shapes_iter = ((geom, 1) for geom in gdf.geometry)
            labels_hr = rasterize(shapes_iter, out_shape=(hr_height, hr_width), transform=hr_transform, fill=0, all_touched=False, dtype=np.uint8)
            try:
                lr = labels_hr.reshape(height, sf, width, sf)
                lr = lr.transpose(0,2,1,3).reshape(height, width, sf*sf)
            except Exception:
                lr = np.empty((height, width, sf*sf), dtype=labels_hr.dtype)
                for i in range(height):
                    for j in range(width):
                        lr[i,j,:] = labels_hr[i*sf:(i+1)*sf, j*sf:(j+1)*sf].ravel()
            sub_counts = (lr != 0).sum(axis=2)
            threshold = math.ceil(sf * sf * float(coverage_threshold))
            mask = sub_counts >= threshold
            return mask

        preseed_targets = {}
        for snap_year in sorted(df_marsh['snapshot_year'].unique()):
            gdf_snap = df_marsh[df_marsh['snapshot_year'] == int(snap_year)].reset_index(drop=True)
            mask_snap = _rasterize_gdf_to_mask(gdf_snap, width, height, transform, sf=SUBSAMPLE_FACTOR, coverage_threshold=COVERAGE_THRESHOLD)
            entry = {
                'orig_idx': None,
                'island': None,
                'start_year': int(snap_year),
                'geom': None,
                'mask': mask_snap,
                'area_pixels': int(mask_snap.sum()),
                'target_pixels': int(mask_snap.sum())
            }
            preseed_targets.setdefault(int(snap_year), []).append(entry)
            print(f"  preseed target for {snap_year}: {int(mask_snap.sum())} pixels ({int(mask_snap.sum()) * px_area_ha:.0f} ha)")

        globals()['preseed_targets'] = preseed_targets
        print("Marsh preseed_targets built and placed in globals()['preseed_targets'].")
    else:
        preseed_targets = {}
        print("No marsh layers loaded; globals()['preseed_targets'] not created.")
    # --- End marsh loading block ---

    # Prepare land at start baseline for polder episodes
    start_ref = YEARS[0]
    land_start, _ = rasterize_union_year(df_all, start_ref, width, height, transform, sf=SUBSAMPLE_FACTOR, coverage_threshold=COVERAGE_THRESHOLD)
    land_start_flat = land_start.ravel()

    # per-pixel state trackers with -1 if it was never observed
    first_inund = np.full((height, width), -1, dtype=np.int32)
    first_reclaim = np.full((height, width), -1, dtype=np.int32)

    current_polder_start = np.full((height, width), -1, dtype=np.int32)
    current_inund_start = np.full((height, width), -1, dtype=np.int32)

    # initialize current_polder_start for pixels that are land at start_ref
    current_polder_start[land_start] = start_ref

    ever_inund = np.zeros((height, width), dtype=bool)
    cumulative_ever_inund = np.zeros((height, width), dtype=bool)
    land_prev = land_start.copy()

    # ---------------- MARSH state initialization ----------------
    current_marsh_mask = np.zeros((height, width), dtype=bool)
    current_marsh_start = np.full((height, width), -1, dtype=np.int32)

    # initial seeding before first snapshot (if requested)
    if ENABLE_MARSH_STAGED_GROWTH and preseed_targets:
        earliest_snap = int(min(preseed_targets.keys()))
        # combine masks for earliest snapshot
        earliest_mask = np.zeros((height, width), dtype=bool)
        for e in preseed_targets[earliest_snap]:
            earliest_mask |= e.get('mask', np.zeros((height, width), dtype=bool))
        # For each island, seed INITIAL_MARSH_SEED_FRACTION of earliest_mask using land_start as seeds
        if INITIAL_MARSH_SEED_FRACTION > 0:
            for iid in np.unique(island_raster):
                if iid == 0:
                    continue
                isl_mask = (island_raster == iid)
                cand = earliest_mask & (~land_start) & isl_mask
                n_cand = int(cand.sum())
                n_seed = int(round(n_cand * float(INITIAL_MARSH_SEED_FRACTION)))
                if n_seed <= 0:
                    continue
                # seed from land_start within island if available, otherwise from cand centroid
                seeds = land_start & isl_mask
                if not seeds.any():
                    # fallback: if there is any current_marsh (unlikely now), use it, else select central candidate
                    if current_marsh_mask.any():
                        seeds = current_marsh_mask & isl_mask
                    else:
                        # choose a central candidate pixel as seed
                        idxs = np.argwhere(cand)
                        if idxs.size:
                            rr, cc = idxs[len(idxs)//2]
                            seeds = np.zeros_like(cand); seeds[rr,cc] = True
                # compute distance-from-seed map and pick closest
                if distance_transform_edt is None:
                    raise RuntimeError("scipy.ndimage.distance_transform_edt required for seeding")
                dist_map = distance_transform_edt(~(seeds.astype(bool)))
                chosen = add_pixels_blob(seeds, cand, n_seed, distance_from_seed=dist_map, connectivity=CONNECTIVITY)
                if chosen.any():
                    current_marsh_mask |= chosen
                    current_marsh_start[chosen] = start_ref

    # ---------------- end marsh initialization ----------------

    per_year_rows = []
    marsh_rows = []   # collect per-year marsh area/time series
    marsh_archive = {}
    # episode collectors
    episodes_polder = []  # tuples: (pixel_index, island_id, start, end_or_-1, duration, event)
    episodes_inund = []

    print("Processing years:", YEARS[0], "->", YEARS[-1])
    t0 = time.time()
    # Main year loop
    for year in tqdm(YEARS, desc="years"):
        land, active = rasterize_union_year(df_all, year, width, height, transform, sf=SUBSAMPLE_FACTOR, coverage_threshold=COVERAGE_THRESHOLD)

        # If this exact year is a snapshot, ensure target mask is applied (force presence inside target)
        if ENABLE_MARSH_STAGED_GROWTH and preseed_targets and (year in preseed_targets):
            for entry in preseed_targets[year]:
                mask_target = entry.get('mask')
                if mask_target is None:
                    continue
                # apply mask for this year but do not overwrite polders (land takes precedence)
                apply_mask = mask_target & (~land)
                newly_started = apply_mask & (~current_marsh_mask)
                if newly_started.any():
                    current_marsh_start[newly_started] = year
                current_marsh_mask |= apply_mask
                # ensure marsh not on land
                current_marsh_mask &= (~land)

        # ---------------- pre-reclaim per-polder staged growth (blob-style) ----------------
        if ENABLE_PRE_RECLAIM_GROWTH and pre_reclaim_targets:
            # consider targets strictly after current year
            future_years = sorted([int(y) for y in pre_reclaim_targets.keys() if int(y) > int(year)])
            if future_years:
                for target_year in future_years:
                    for entry in pre_reclaim_targets.get(target_year, []):
                        owned = entry.get('owned_mask', None)
                        if owned is None or not owned.any():
                            continue
                        # restrict to island and not currently land
                        isl_mask = (island_raster == int(entry.get('island_id', 0))) if entry.get('island_id', 0) != 0 else np.ones((height, width), dtype=bool)
                        # candidates inside owned target not yet marsh and not land
                        candidates = owned & (~current_marsh_mask) & (~land) & isl_mask
                        # current marsh pixels inside owned mask (count towards target)
                        cur_px = int(np.sum(current_marsh_mask & owned))
                        target_buff_px = int(entry.get('buffer_px', entry.get('target_px', 0)))
                        if int(year) >= int(entry['start_year']):
                            # target_year reached or passed: nothing to do (land rasterization will handle reclamation)
                            continue
                        years_remaining = max(PRE_RECLAIM_MIN_LEAD_YEARS, int(entry['start_year'] - year))
                        need = target_buff_px - cur_px
                        if need <= 0:
                            # already reached buffered target, nothing to grow for this entry
                            continue
                        add_this_year = int(math.ceil(float(need) / float(years_remaining)))
                        if add_this_year <= 0:
                            continue
                        if not candidates.any():
                            # no available candidates this year; carry-over to later years
                            continue
                        # pick seeds preferring polder/land then marsh
                        seeds = (land | current_marsh_mask) & isl_mask
                        if not seeds.any():
                            seeds = current_marsh_mask & isl_mask
                        if not seeds.any():
                            # fallback to any owned pixel center
                            idxs = np.argwhere(owned)
                            if idxs.size:
                                rr, cc = idxs[len(idxs)//2]
                                seeds = np.zeros_like(owned); seeds[rr,cc] = True
                        # compute distance map from seeds
                        if distance_transform_edt is None:
                            raise RuntimeError("scipy.ndimage.distance_transform_edt required for pre-reclaim growth")
                        dist_map = distance_transform_edt(~seeds)
                        # Use blob growth helper (expands from seeds inward)
                        chosen = add_pixels_blob(seeds, candidates, add_this_year, distance_from_seed=dist_map, connectivity=CONNECTIVITY)
                        if chosen.any():
                            newly_started = chosen & (~current_marsh_mask)
                            if newly_started.any():
                                current_marsh_start[newly_started] = year
                            # add to shared marsh mask (avoid double-counting because owned masks are exclusive)
                            current_marsh_mask |= chosen
                            # ensure marsh not on land
                            current_marsh_mask &= (~land)
        # ---------------- end pre-reclaim growth ----------------

        # ---------------- staged deterministic growth/shrink toward next snapshot (blob-style) ----------------
        if ENABLE_MARSH_STAGED_GROWTH and preseed_targets:
            # find next snapshot strictly after current year
            future_snaps = sorted([int(y) for y in preseed_targets.keys() if int(y) > int(year)])
            if future_snaps:
                next_snap = int(future_snaps[0])

                # iterate per-island (recommended) or global
                island_ids = np.unique(island_raster) if PER_ISLAND_GROWTH else np.array([0])
                for iid in island_ids:
                    if iid == 0 and PER_ISLAND_GROWTH:
                        continue
                    isl_mask = (island_raster == iid) if PER_ISLAND_GROWTH else np.ones((height, width), dtype=bool)

                    # build combined target mask for this island and next snapshot
                    target_mask = np.zeros((height, width), dtype=bool)
                    for e in preseed_targets[next_snap]:
                        target_mask |= e.get('mask', np.zeros((height, width), dtype=bool))
                    target_mask &= isl_mask

                    # restrict target to not be on polder (marsh cannot be on land)
                    target_mask &= (~land)

                    cur_mask = current_marsh_mask & isl_mask
                    cur_px = int(cur_mask.sum())
                    target_px = int(target_mask.sum())

                    if year >= next_snap:
                        continue  # don't adjust after snapshot year

                    years_remaining = max(1, next_snap - year)
                    if target_px > cur_px:
                        # add pixels
                        need = target_px - cur_px
                        add_this_year = int(math.ceil(float(need) / float(years_remaining)))
                        # candidates inside target not yet marsh and not land
                        candidates = target_mask & (~current_marsh_mask) & (~land) & isl_mask
                        if candidates.any():
                            # define seeds: prefer land + existing marsh (so growth from polder first)
                            seeds = (current_marsh_mask | land) & isl_mask
                            if not seeds.any():
                                # fallback: use any existing marsh in whole domain or land_start
                                seeds = (current_marsh_mask & isl_mask)
                                if not seeds.any():
                                    seeds = (land_start & isl_mask)
                                if not seeds.any():
                                    # last fallback: choose central target pixel
                                    idxs = np.argwhere(candidates)
                                    if idxs.size:
                                        rr, cc = idxs[len(idxs)//2]
                                        seeds = np.zeros_like(candidates); seeds[rr,cc] = True
                            # compute dist_map from seeds (smaller -> closer)
                            if distance_transform_edt is None:
                                raise RuntimeError("scipy.ndimage.distance_transform_edt required for staged growth")
                            dist_map = distance_transform_edt(~seeds)
                            # Use blob growth helper here as well
                            chosen = add_pixels_blob(seeds, candidates, add_this_year, distance_from_seed=dist_map, connectivity=CONNECTIVITY)
                            if chosen.any():
                                newly_started = chosen & (~current_marsh_mask)
                                if newly_started.any():
                                    current_marsh_start[newly_started] = year
                                current_marsh_mask |= chosen
                                # ensure marsh not on land
                                current_marsh_mask &= (~land)
                        # otherwise no candidates this year; remainder carried forward
                    elif ALLOW_SHRINKAGE and target_px < cur_px:
                        # remove pixels (shrink)
                        need = cur_px - target_px
                        rem_this_year = int(math.ceil(float(need) / float(years_remaining)))
                        to_remove_pool = current_marsh_mask & (~target_mask) & isl_mask
                        if to_remove_pool.any():
                            # compute distance to land; remove farthest-from-land pixels first
                            if distance_transform_edt is None:
                                raise RuntimeError("scipy.ndimage.distance_transform_edt required for shrinkage")
                            dist_to_land = distance_transform_edt(~land)
                            idxs = np.argwhere(to_remove_pool)
                            if idxs.size:
                                vals = dist_to_land[to_remove_pool]
                                k = min(rem_this_year, vals.size)
                                # select k largest distances
                                order_desc = np.argsort(-vals, kind='stable')[:k]
                                sel_coords = idxs[order_desc]
                                chosen_mask = np.zeros_like(to_remove_pool, dtype=bool)
                                for (rr, cc) in sel_coords:
                                    chosen_mask[rr, cc] = True
                                if chosen_mask.any():
                                    current_marsh_mask[chosen_mask] = False
                                    current_marsh_start[chosen_mask] = -1
                    # end per-island adjustments
        # ---------------- end staged adjustment ----------------

        # polder -> inundation (newly inundated)
        newly_inund = land_prev & (~land)
        # record first_inund if not set
        mask_first_in = newly_inund & (first_inund == -1)
        if mask_first_in.any():
            first_inund[mask_first_in] = year
        # close polder episodes for newly inundated pixels
        idxs_newly = np.argwhere(newly_inund)
        for (r,c) in idxs_newly:
            pix_idx = int(r * width + c)
            island_id = int(island_raster[r,c])
            start = int(current_polder_start[r,c]) if current_polder_start[r,c] != -1 else None
            if start is not None and start != -1:
                dur = year - start
                episodes_polder.append((pix_idx, island_id, int(start), int(year), int(dur), 1))
            current_polder_start[r,c] = -1
            current_inund_start[r,c] = year

        cumulative_ever_inund |= newly_inund
        ever_inund |= newly_inund

        # inundation -> polder (reclaimed)
        reclaimed = ever_inund & land
        mask_first_reclaim = reclaimed & (first_reclaim == -1)
        if mask_first_reclaim.any():
            first_reclaim[mask_first_reclaim] = year

        # --- REPLACED reclamation loop: retrofill marsh one year before polder creation ---
        idxs_reclaim = np.argwhere(reclaimed)
        for (r, c) in idxs_reclaim:
            start = int(current_inund_start[r, c]) if current_inund_start[r, c] != -1 else None
            pix_idx = int(r * width + c)
            island_id = int(island_raster[r, c])

            # Retrofill: mark pixel as marsh one year before it becomes polder
            prev_year = int(year - 1)
            if prev_year >= START_YEAR_CAP:
                # only retrofill if not already recorded as marsh
                if current_marsh_start[r, c] == -1:
                    current_marsh_start[r, c] = prev_year
                    try:
                        # update marsh_rows entry for prev_year (increment global + per-island counts)
                        found = False
                        for row in marsh_rows:
                            if row.get("Year") == prev_year:
                                row["Marsh_area_px"] = int(row.get("Marsh_area_px", 0)) + 1
                                row["Marsh_area_ha"] = float(row.get("Marsh_area_ha", 0.0)) + px_area_ha
                                col_px = f"island_{island_id}_marsh_px"
                                col_ha = f"island_{island_id}_marsh_ha"
                                row[col_px] = int(row.get(col_px, 0)) + 1
                                row[col_ha] = float(row.get(col_ha, 0.0)) + px_area_ha
                                found = True
                                break
                        if not found:
                            # Fallback: create a row for prev_year if it doesn't exist (rare)
                            newrow = {"Year": prev_year, "Marsh_area_px": 1, "Marsh_area_ha": px_area_ha,
                                      f"island_{island_id}_marsh_px": 1, f"island_{island_id}_marsh_ha": px_area_ha}
                            marsh_rows.append(newrow)
                    except Exception as exc:
                        warnings.warn(f"Retrofill update failed for pixel {(r,c)} prev_year={prev_year}: {exc}")

            # record the reclamation episode (same as before)
            if start is not None and start != -1:
                dur = year - start
                episodes_inund.append((pix_idx, island_id, int(start), int(year), int(dur), 1))

            # finalize state
            current_inund_start[r, c] = -1
            current_polder_start[r, c] = year
        # --- end retrofill block ---

        # update ever_inund: remove reclaimed pixels
        ever_inund &= (~land)

        # per-year metrics (existing)
        per_year_rows.append({
            "Year": int(year),
            "Area_polders_ha": float(land.sum()) * px_area_ha,
            "Newly_inundated_ha": float(newly_inund.sum()) * px_area_ha,
            "Reclaimed_ha": float(reclaimed.sum()) * px_area_ha,
            "Currently_inundated_ha": float(ever_inund.sum()) * px_area_ha,
            "Cumulative_ever_inundated_ha": float(cumulative_ever_inund.sum()) * px_area_ha
        })

        # record marsh metrics this year (global + per-island)
        total_marsh_px = int(current_marsh_mask.sum())
        total_marsh_ha = float(total_marsh_px) * px_area_ha
        row_m = {"Year": int(year), "Marsh_area_ha": total_marsh_ha, "Marsh_area_px": total_marsh_px}
        # per-island areas
        for iid, name in id_to_island.items():
            isl_px = int(np.sum(current_marsh_mask & (island_raster == iid)))
            row_m[f"island_{iid}_marsh_px"] = isl_px
            row_m[f"island_{iid}_marsh_ha"] = float(isl_px) * px_area_ha
        marsh_archive[int(year)] = current_marsh_mask.copy()
        marsh_rows.append(row_m)

        land_prev = land

    # End year loop
    last_year = YEARS[-1]
    # close ongoing inund episodes as censored
    ongoing_inund_idxs = np.argwhere(current_inund_start != -1)
    for (r,c) in ongoing_inund_idxs:
        start = int(current_inund_start[r,c])
        pix_idx = int(r * width + c)
        island_id = int(island_raster[r,c])
        dur = last_year - start
        episodes_inund.append((pix_idx, island_id, int(start), -1, int(dur), 0))
        current_inund_start[r,c] = -1

    # close ongoing polder episodes as censored
    ongoing_polder_idxs = np.argwhere(current_polder_start != -1)
    for (r,c) in ongoing_polder_idxs:
        start = int(current_polder_start[r,c])
        pix_idx = int(r * width + c)
        island_id = int(island_raster[r,c])
        dur = last_year - start
        episodes_polder.append((pix_idx, island_id, int(start), -1, int(dur), 0))
        current_polder_start[r,c] = -1

    # build DataFrames
    df_yearly = pd.DataFrame(per_year_rows)
    df_marsh_by_year = pd.DataFrame(marsh_rows).set_index('Year') if marsh_rows else pd.DataFrame()
    episodes_polder_df = pd.DataFrame(episodes_polder, columns=['pixel_index','island_id','start_year','end_year','duration_years','event_observed'])
    episodes_inund_df = pd.DataFrame(episodes_inund, columns=['pixel_index','island_id','start_year','end_year','duration_years','event_observed'])
    if not episodes_polder_df.empty:
        episodes_polder_df['island'] = episodes_polder_df['island_id'].map(id_to_island)
    if not episodes_inund_df.empty:
        episodes_inund_df['island'] = episodes_inund_df['island_id'].map(id_to_island)

    # Build pixel-level summary (first events + episode counts) same as before
    rows = []
    mask_pixels = np.where(island_raster.ravel() > 0)[0]
    fi_flat = first_inund.ravel()
    fr_flat = first_reclaim.ravel()
    for idx in tqdm(mask_pixels, desc="pixels"):
        r = idx // width; c = idx % width
        island_id = int(island_raster.ravel()[idx])
        island_name = id_to_island.get(island_id, "Unknown")
        rows.append({
            "pixel_index": int(idx),
            "row": int(r),
            "col": int(c),
            "island_id": int(island_id),
            "island": island_name,
            "first_inundation": int(fi_flat[idx]),
            "first_reclaim": int(fr_flat[idx])
        })
    df_pixels = pd.DataFrame(rows)
    # export into globals
    globals()['marsh_archive'] = marsh_archive
    globals()['df_yearly'] = df_yearly
    globals()['df_pixels'] = df_pixels
    globals()['episodes_polder_df'] = episodes_polder_df
    globals()['episodes_inund_df'] = episodes_inund_df
    globals()['df_marsh_by_year'] = df_marsh_by_year
    globals()['current_marsh_mask'] = current_marsh_mask
    globals()['current_marsh_start'] = current_marsh_start
    globals()['preseed_targets'] = preseed_targets
    globals()['pre_reclaim_targets'] = pre_reclaim_targets
    globals()['pre_reclaim_ownership_raster'] = ownership_raster
    globals()['df_all'] = df_all
    globals()['width'] = width
    globals()['height'] = height
    globals()['transform'] = transform
    globals()['island_raster'] = island_raster
    globals()['id_to_island'] = id_to_island

    # --- Save run outputs to disk (Parquet for DataFrames, NPZ for arrays) ---
    try:
        import json, pickle, os
        save_dir = os.path.join(OUT_DIR, "saved_run_parquet")
        os.makedirs(save_dir, exist_ok=True)

        def _save_df_parquet_or_pickle(df, name):
            if df is None:
                return None
            path_parq = os.path.join(save_dir, f"{name}.parquet")
            path_pkl = os.path.join(save_dir, f"{name}.pkl")
            try:
                df.to_parquet(path_parq, index=False)
                return path_parq
            except Exception:
                df.to_pickle(path_pkl)
                return path_pkl

        def _save_array_npz(arr, name):
            if arr is None:
                return None
            path = os.path.join(save_dir, f"{name}.npz")
            if getattr(arr, "dtype", None) is not None and arr.dtype == np.bool_:
                np.savez_compressed(path, data=arr.astype(np.uint8))
            else:
                np.savez_compressed(path, data=arr)
            return path

        saved_paths = {}
        # DataFrames
        saved_paths['df_yearly'] = _save_df_parquet_or_pickle(df_yearly, 'df_yearly') if 'df_yearly' in locals() else None
        saved_paths['df_pixels'] = _save_df_parquet_or_pickle(df_pixels, 'df_pixels') if 'df_pixels' in locals() else None
        saved_paths['episodes_polder_df'] = _save_df_parquet_or_pickle(episodes_polder_df, 'episodes_polder_df') if 'episodes_polder_df' in locals() else None
        saved_paths['episodes_inund_df'] = _save_df_parquet_or_pickle(episodes_inund_df, 'episodes_inund_df') if 'episodes_inund_df' in locals() else None
        saved_paths['df_marsh_by_year'] = _save_df_parquet_or_pickle(df_marsh_by_year, 'df_marsh_by_year') if 'df_marsh_by_year' in locals() else None
        # optional outputs from downstream analysis
        saved_paths['inundation_events_df'] = _save_df_parquet_or_pickle(globals().get('inundation_events_df'), 'inundation_events_df') if globals().get('inundation_events_df') is not None else None
        saved_paths['inundation_bin_summary'] = _save_df_parquet_or_pickle(globals().get('inundation_bin_summary'), 'inundation_bin_summary') if globals().get('inundation_bin_summary') is not None else None
        saved_paths['inundation_year_summary'] = _save_df_parquet_or_pickle(globals().get('inundation_year_summary'), 'inundation_year_summary') if globals().get('inundation_year_summary') is not None else None

        # Arrays / masks / rasters
        saved_paths['island_raster'] = _save_array_npz(island_raster, 'island_raster') if 'island_raster' in locals() else None
        saved_paths['current_marsh_mask'] = _save_array_npz(current_marsh_mask, 'current_marsh_mask') if 'current_marsh_mask' in locals() else None
        saved_paths['current_marsh_start'] = _save_array_npz(current_marsh_start, 'current_marsh_start') if 'current_marsh_start' in locals() else None
        saved_paths['pre_reclaim_ownership_raster'] = _save_array_npz(ownership_raster, 'pre_reclaim_ownership_raster') if 'ownership_raster' in locals() else None

        # marsh_archive is a dict year -> boolean mask; save as one NPZ with keys y_<year>
        if isinstance(globals().get('marsh_archive', None), dict) and globals()['marsh_archive']:
            ma = globals()['marsh_archive']
            arrdict = {}
            for yr, mask in ma.items():
                key = f"y_{int(yr)}"
                arrdict[key] = np.asarray(mask).astype(np.uint8)
            path_ma = os.path.join(save_dir, "marsh_archive.npz")
            np.savez_compressed(path_ma, **arrdict)
            saved_paths['marsh_archive'] = path_ma

        # small dicts and metadata
        if 'id_to_island' in globals():
            with open(os.path.join(save_dir, 'id_to_island.json'), 'w', encoding='utf-8') as fh:
                json.dump(globals()['id_to_island'], fh, ensure_ascii=False)
            saved_paths['id_to_island'] = os.path.join(save_dir, 'id_to_island.json')

        # save transform and other simple metadata with pickle
        metadata = {
            'width': width,
            'height': height,
            'px_area_ha': px_area_ha,
            'GRID_RES': GRID_RES,
            'YEARS': YEARS
        }
        with open(os.path.join(save_dir, "metadata.pkl"), 'wb') as fh:
            pickle.dump(metadata, fh)
        saved_paths['metadata'] = os.path.join(save_dir, "metadata.pkl")

        print("Saved run outputs to:", save_dir)
        for k, v in saved_paths.items():
            if v:
                print("  ", k, "->", v)
    except Exception as exc:
        print("Warning: saving run outputs failed:", exc)
    # --- end save block ---

    print("\nInteractive run complete. Marsh timeseries and masks available in globals:")
    if not df_marsh_by_year.empty:
        print(" df_marsh_by_year (years):", df_marsh_by_year.index.min(), "->", df_marsh_by_year.index.max())
    print(" current_marsh_mask, current_marsh_start, preseed_targets, pre_reclaim_targets, df_marsh (original polygons)")

    return df_yearly, df_pixels, episodes_polder_df, episodes_inund_df, df_marsh_by_year

if __name__ == "__main__":
    run()
```

## **Zeeland (Province) analysis code**
Firstly it computes the A, I, S. The S is taken from snapshots in salt marsh and then calculates the slope between the two points. 
```python
#%% A,I,S,dA/dt,dS/dt
def _infer_px_area_ha(g):
    if 'px_area_ha' in g:
        return float(g['px_area_ha'])
    if 'GRID_RES' in g:
        try:
            return (float(g['GRID_RES']) ** 2) / 10000.0
        except Exception:
            pass
    if 'transform' in g:
        try:
            tr = g['transform']
            return (float(tr.a) * float(tr.a)) / 10000.0
        except Exception:
            pass
    # fallback
    return 1.0

def compute_area_changes(per_island: bool = True,
                         use_cached_land_masks: bool = True,
                         start_year: Optional[int] = None,
                         end_year: Optional[int] = None,
                         verbose: bool = True):
    """
    Compute A(t), I(t), and finite-difference rates, plus marsh slope estimates.

    Args:
      per_island: if True compute series per island (returns dict keyed by island id)
      use_cached_land_masks: if True prefer globals()['land_masks_by_year'] and skip rasterizing
      start_year, end_year: optional restrict year range (inclusive)
      verbose: print a short summary

    Returns:
      dict with results (see module docstring)
    """
    g = globals()
    # required globals for fast path
    if use_cached_land_masks:
        land_masks = g.get('land_masks_by_year') or g.get('land_masks') or None
    else:
        land_masks = None

    df_all = g.get('df_all')
    rasterize_fn = g.get('rasterize_union_year')
    marsh_archive = g.get('marsh_archive', {})
    island_raster = g.get('island_raster', None)
    id_to_island = g.get('id_to_island', None)

    px_area_ha = _infer_px_area_ha(g)

    # determine years to use
    if land_masks:
        yrs = sorted(int(y) for y in land_masks.keys())
    elif 'YEARS' in g:
        yrs = list(map(int, g['YEARS']))
    elif marsh_archive:
        yrs = sorted(int(y) for y in marsh_archive.keys())
    else:
        raise RuntimeError("Could not infer years. Provide land_masks_by_year or YEARS or marsh_archive in globals.")

    if start_year is not None:
        yrs = [y for y in yrs if y >= int(start_year)]
    if end_year is not None:
        yrs = [y for y in yrs if y <= int(end_year)]
    if len(yrs) < 2:
        raise ValueError("Need at least two years to compute differences. Years available: %s" % (yrs,))

    years = sorted(yrs)
    n = len(years)

    # obtain land masks for all years (fast: from cache; fallback: rasterize once per year)
    masks = {}
    if land_masks is not None:
        # validate shapes quickly
        example = next(iter(land_masks.values()))
        H,W = example.shape
        for y in years:
            if int(y) not in land_masks:
                raise RuntimeError(f"Cached land_masks missing year {y}")
            masks[int(y)] = land_masks[int(y)].astype(bool)
    else:
        # fallback: rasterize each year (requires rasterize_union_year & df_all)
        if rasterize_fn is None or df_all is None:
            raise RuntimeError("No cached land masks and rasterize function or df_all missing in globals.")
        for y in years:
            lm, _ = rasterize_fn(df_all, int(y), int(g['width']), int(g['height']), g['transform'], sf= g.get('SUBSAMPLE_FACTOR',1), coverage_threshold=float(g.get('COVERAGE_THRESHOLD',0.5)))
            masks[int(y)] = lm.astype(bool)

    H,W = next(iter(masks.values())).shape

    # Build earlier_land cumulative union for I(t)
    earlier_union = np.zeros((H,W), dtype=bool)
    A_list = []
    I_list = []
    S_list = []
    for y in years:
        land = masks[int(y)]
        A_ha = float(np.sum(land)) * px_area_ha
        # inundation defined as earlier_union & (~land)
        I_mask = earlier_union & (~land)
        I_ha = float(np.sum(I_mask)) * px_area_ha
        # marsh from archive if present (mask expected same dims)
        m = marsh_archive.get(int(y), None)
        if m is None:
            S_ha = np.nan
        else:
            # if marsh mask shape differs but same H,W we still handle; else try to coerce
            try:
                S_ha = float(np.sum(m.astype(bool))) * px_area_ha
            except Exception:
                S_ha = np.nan

        A_list.append(A_ha)
        I_list.append(I_ha)
        S_list.append(S_ha)

        # update cumulative union AFTER computing I for this year
        earlier_union = earlier_union | land

    A_arr = np.array(A_list, dtype=float)
    I_arr = np.array(I_list, dtype=float)
    S_arr = np.array(S_list, dtype=float)  # may contain nan for years with no marsh snapshot

    # finite differences (year-to-year). dX_dt has length n-1 and corresponds to years[i] -> years[i+1]
    dA = np.diff(A_arr)
    dI = np.diff(I_arr)

    # Marsh snapshot analysis: collect available snapshot years & areas (total)
    snap_years = sorted([int(y) for y in marsh_archive.keys()])
    S_snap = []
    S_snap_yrs = []
    for y in snap_years:
        m = marsh_archive.get(int(y))
        if m is None:
            continue
        try:
            S_snap.append(float(np.sum(m.astype(bool))) * px_area_ha)
            S_snap_yrs.append(int(y))
        except Exception:
            continue
    S_snap = np.array(S_snap, dtype=float) if S_snap else np.array([], dtype=float)
    S_snap_yrs = np.array(S_snap_yrs, dtype=int) if len(S_snap_yrs) else np.array([], dtype=int)

    # helper to compute linear slope (ha per year) when at least 2 points
    def _linear_slope(xyrs, yvals):
        if len(xyrs) < 2:
            return np.nan
        # only finite yvals
        mask = np.isfinite(yvals)
        if mask.sum() < 2:
            return np.nan
        x = np.asarray(xyrs)[mask].astype(float)
        y = np.asarray(yvals)[mask].astype(float)
        # linear fit y = a*x + b => slope = a
        a, b = np.polyfit(x, y, 1)
        return float(a)

    S_slope_all = _linear_slope(S_snap_yrs, S_snap)
    # compute slope restricted to [1620,1900] using available snapshot years in that interval
    yr_lo, yr_hi = 1620, 1900
    mask_range = (S_snap_yrs >= yr_lo) & (S_snap_yrs <= yr_hi) if S_snap_yrs.size else np.array([], dtype=bool)
    if mask_range.any():
        S_slope_1620_1900 = _linear_slope(S_snap_yrs[mask_range], S_snap[mask_range])
    else:
        # fallback: try to use nearest available points around that interval
        # find earliest >=1620 and latest <=1900
        yrs_in = S_snap_yrs[(S_snap_yrs >= yr_lo) & (S_snap_yrs <= yr_hi)]
        if yrs_in.size >= 2:
            S_slope_1620_1900 = _linear_slope(yrs_in, S_snap[[np.where(S_snap_yrs == y)[0][0] for y in yrs_in]])
        else:
            # pick first snapshot >=1620 (if any) and last <=1900
            cand_lo = S_snap_yrs[S_snap_yrs >= yr_lo]
            cand_hi = S_snap_yrs[S_snap_yrs <= yr_hi]
            if cand_lo.size and cand_hi.size:
                ylo = int(cand_lo.min())
                yhi = int(cand_hi.max())
                if yhi > ylo:
                    slo = S_snap[np.where(S_snap_yrs == ylo)[0][0]]
                    shi = S_snap[np.where(S_snap_yrs == yhi)[0][0]]
                    S_slope_1620_1900 = float(shi - slo) / float(yhi - ylo)
                else:
                    S_slope_1620_1900 = np.nan
            else:
                S_slope_1620_1900 = np.nan

    # If per_island requested, compute per-island series similarly
    per_island = {}
    if per_island:
        if island_raster is None or id_to_island is None:
            raise RuntimeError("per_island=True requires island_raster and id_to_island in globals.")
        uniq_ids = sorted([int(x) for x in np.unique(island_raster) if x != 0])
        # build island masks
        island_masks = {iid: (island_raster == iid) for iid in uniq_ids}
        for iid in uniq_ids:
            mask_i = island_masks[iid]
            A_i = []
            I_i = []
            S_i_snap = []
            # rebuild earlier union per-island
            earlier_union_i = np.zeros((H,W), dtype=bool)
            for y in years:
                land = masks[int(y)]
                A_i.append(float(np.sum((land & mask_i))) * px_area_ha)
                I_mask_i = earlier_union_i & (~land)
                I_i.append(float(np.sum(I_mask_i & mask_i)) * px_area_ha)
                m = marsh_archive.get(int(y), None)
                if m is None:
                    s = np.nan
                else:
                    s = float(np.sum(m.astype(bool) & mask_i)) * px_area_ha
                earlier_union_i = earlier_union_i | land
                # (note: earlier_union_i uses global "land" union; this mirrors global definition)
            # marsh snapshots per island
            for y in S_snap_yrs:
                m = marsh_archive.get(int(y))
                if m is None:
                    S_i_snap.append(np.nan)
                else:
                    S_i_snap.append(float(np.sum(m.astype(bool) & mask_i)) * px_area_ha)
            per_island[iid] = {
                "years": np.array(years),
                "A_ha": np.array(A_i),
                "I_ha": np.array(I_i),
                "dA_ha": np.diff(np.array(A_i)) if len(A_i) >= 2 else np.array([]),
                "dI_ha": np.diff(np.array(I_i)) if len(I_i) >= 2 else np.array([]),
                "S_snap_yrs": np.array(S_snap_yrs),
                "S_snap_ha": np.array(S_i_snap),
                "S_slope_all": _linear_slope(S_snap_yrs, np.array(S_i_snap)) if S_snap_yrs.size else np.nan,
                "S_slope_1620_1900": None  # compute same logic if desired per-island
            }
            # per-island slope for 1620..1900
            mask_range_i = (per_island[iid]['S_snap_yrs'] >= yr_lo) & (per_island[iid]['S_snap_yrs'] <= yr_hi) if per_island[iid]['S_snap_yrs'].size else np.array([], dtype=bool)
            if mask_range_i.any():
                per_island[iid]['S_slope_1620_1900'] = _linear_slope(per_island[iid]['S_snap_yrs'][mask_range_i], per_island[iid]['S_snap_ha'][mask_range_i])
            else:
                # attempt pair fallback
                yrs_in = per_island[iid]['S_snap_yrs'][(per_island[iid]['S_snap_yrs'] >= yr_lo) & (per_island[iid]['S_snap_yrs'] <= yr_hi)]
                if yrs_in.size >= 2:
                    per_island[iid]['S_slope_1620_1900'] = _linear_slope(yrs_in, per_island[iid]['S_snap_ha'][[np.where(per_island[iid]['S_snap_yrs']==y)[0][0] for y in yrs_in]])
                else:
                    per_island[iid]['S_slope_1620_1900'] = np.nan

    # Verbose summary
    if verbose:
        print("Computed area series for years:", years[0], "..", years[-1], f"({n} timesteps). px_area_ha={px_area_ha:.4f}")
        print(f"Total A (first..last) = {A_arr[0]:.1f} ha -> {A_arr[-1]:.1f} ha ; total change = {A_arr[-1]-A_arr[0]:.1f} ha")
        print(f"Total I (first..last) = {I_arr[0]:.1f} ha -> {I_arr[-1]:.1f} ha ; total change = {I_arr[-1]-I_arr[0]:.1f} ha")
        print(f"Marsh snapshots found: {list(S_snap_yrs)} ; S_slope_all = {S_slope_all:.4f} ha/yr ; S_slope_1620_1900 = {S_slope_1620_1900:.4f} ha/yr")
        if per_island:
            print("Per-island results available in results['per_island'] (keys = island ids)")

    results = {
        "years": np.array(years),
        "A_ha": A_arr,
        "I_ha": I_arr,
        "dA_ha": dA,
        "dI_ha": dI,
        "px_area_ha": px_area_ha,
        "marsh_snap_years": S_snap_yrs,
        "S_snap_ha": S_snap,
        "S_slope_all": S_slope_all,
        "S_slope_1620_1900": S_slope_1620_1900,
        "per_island": per_island if per_island else None
    }
    return results

# If executed as a script (requires run() already executed in the same session OR you saved
# land_masks_by_year and marsh_archive into a persistent python pickle etc.), you can call:
if __name__ == "__main__":
    try:
        res = compute_area_changes(per_island=False, use_cached_land_masks=True, verbose=True)
        # quick example prints
        yrs = res['years']
        print("years count:", len(yrs))
        print("first 5 A (ha):", res['A_ha'][:5])
        print("first 5 dA (ha/yr):", res['dA_ha'][:5])
        print("marsh snapshot years:", list(res['marsh_snap_years']))
        print("marsh slopes (all):", res['S_slope_all'], "1620-1900:", res['S_slope_1620_1900'])
    except Exception as e:
        print("compute_area_changes failed:", e)
#%% Create alpha and beta values per island

def _infer_px_area_ha(g):
    if 'px_area_ha' in g:
        return float(g['px_area_ha'])
    if 'GRID_RES' in g:
        try:
            return (float(g['GRID_RES']) ** 2) / 10000.0
        except Exception:
            pass
    if 'transform' in g:
        try:
            tr = g['transform']
            return (float(tr.a) * float(tr.a)) / 10000.0
        except Exception:
            pass
    return 1.0

def _summarize(arr):
    arr = np.asarray(arr, dtype=float)
    ok = np.isfinite(arr)
    cnt = int(ok.sum())
    if cnt == 0:
        return {"mean": np.nan, "median": np.nan, "std": np.nan, "iqr": np.nan, "count": 0}
    vals = arr[ok]
    mean = float(np.mean(vals))
    med = float(np.median(vals))
    std = float(np.std(vals, ddof=1)) if vals.size > 1 else float(0.0)
    q75, q25 = float(np.percentile(vals, 75)), float(np.percentile(vals, 25))
    return {"mean": mean, "median": med, "std": std, "iqr": q75 - q25, "count": cnt}

def _fit_alpha_beta_nnls(A, I, dI):
    """
    Solve dI = alpha * A - beta * I  for alpha,beta with non-negativity constraints.
    Returns (alpha, beta, resid, r2, method)
    """
    A = np.asarray(A, dtype=float)
    I = np.asarray(I, dtype=float)
    dI = np.asarray(dI, dtype=float)
    mask = np.isfinite(A) & np.isfinite(I) & np.isfinite(dI)
    if mask.sum() < 2:
        return (np.nan, np.nan, None, np.nan, "insufficient")
    y = dI[mask]
    X = np.column_stack([A[mask], -I[mask]])  # coefficients map to [alpha, beta]
    try:
        if nnls is not None:
            sol, rnorm = nnls(X, y)
            alpha_fit, beta_fit = float(sol[0]), float(sol[1])
            ypred = X.dot(sol)
            resid = y - ypred
            ss_res = float((resid**2).sum())
            ss_tot = float(((y - y.mean())**2).sum())
            r2 = 1.0 - ss_res/ss_tot if ss_tot > 0 else np.nan
            return (alpha_fit, beta_fit, resid, r2, "nnls")
        else:
            # fallback least squares (may produce negative values)
            sol, *_ = np.linalg.lstsq(X, y, rcond=None)
            alpha_fit, beta_fit = float(sol[0]), float(sol[1])
            ypred = X.dot(sol)
            resid = y - ypred
            ss_res = float((resid**2).sum())
            ss_tot = float(((y - y.mean())**2).sum())
            r2 = 1.0 - ss_res/ss_tot if ss_tot > 0 else np.nan
            return (alpha_fit, beta_fit, resid, r2, "lstsq")
    except Exception as e:
        warnings.warn(f"Fitting alpha/beta failed: {e}")
        return (np.nan, np.nan, None, np.nan, "error")

def compute_alpha_beta(per_island: bool = False,
                       min_area_ha: float = 1.0,
                       min_inund_ha: float = 1.0,
                       years: Optional[list] = None,
                       fit_global: bool = True,
                       plot: bool = True) -> Dict[str, Any]:
    """
    Compute per-interval alpha and beta series and optionally fit global alpha/beta.

    Args:
      per_island: compute per-island series (requires island_raster & id_to_island)
      min_area_ha: minimum A(t) to compute alpha_t for an interval
      min_inund_ha: minimum I(t) to compute beta_t for an interval
      years: optional list of years to analyze (inclusive). If None inferred from df_yearly/land_masks_by_year/YEARS.
      fit_global: if True, fit alpha/beta across all valid intervals (global) using NNLS/LS
      plot: show diagnostic plots (requires matplotlib)

    Returns:
      dict with computed series and summaries (see module docstring)
    """
    g = globals()
    px_area_ha = _infer_px_area_ha(g)

    df_yearly = g.get('df_yearly', None)
    land_masks_cache = g.get('land_masks_by_year') or g.get('land_masks') or None

    # Determine years list
    if years is None:
        if df_yearly is not None and 'Year' in df_yearly.columns:
            years_list = sorted(int(x) for x in df_yearly['Year'].values)
        elif land_masks_cache is not None:
            years_list = sorted(int(x) for x in land_masks_cache.keys())
        elif 'YEARS' in g:
            years_list = list(map(int, g['YEARS']))
        else:
            raise RuntimeError("Could not infer years. Ensure df_yearly, land_masks_by_year or YEARS present in globals.")
    else:
        years_list = sorted(int(y) for y in years)

    if len(years_list) < 2:
        raise ValueError("Need at least two years to compute per-interval quantities.")

    interval_years = np.array(years_list[:-1], dtype=int)
    n_intervals = len(interval_years)

    # Acquire masks dict for requested years (prefer cached)
    if land_masks_cache is not None:
        masks = {int(y): land_masks_cache[int(y)].astype(bool) for y in years_list}
    else:
        # fallback to rasterize if possible
        rasterize_fn = g.get('rasterize_union_year')
        df_all = g.get('df_all')
        if rasterize_fn is None or df_all is None:
            raise RuntimeError("No cached land masks and cannot rasterize (missing rasterize_union_year or df_all).")
        masks = {}
        for y in years_list:
            lm, _ = rasterize_fn(df_all, int(y), int(g['width']), int(g['height']), g['transform'], sf=g.get('SUBSAMPLE_FACTOR',1), coverage_threshold=float(g.get('COVERAGE_THRESHOLD',0.5)))
            masks[int(y)] = lm.astype(bool)

    # Build earlier land union and compute A, I per year
    example = next(iter(masks.values()))
    H, W = example.shape
    earlier_union = np.zeros((H, W), dtype=bool)
    A_list = []
    I_list = []
    for y in years_list:
        land = masks[int(y)]
        A_ha = float(np.sum(land)) * px_area_ha
        I_mask = earlier_union & (~land)
        I_ha = float(np.sum(I_mask)) * px_area_ha
        A_list.append(A_ha)
        I_list.append(I_ha)
        earlier_union = earlier_union | land

    A = np.array(A_list, dtype=float)            # len = n_years
    I = np.array(I_list, dtype=float)            # len = n_years
    dI = np.diff(I)                              # len = n_years-1

    # newly_inund per interval (from masks)
    newly = np.zeros(n_intervals, dtype=float)
    for idx, y in enumerate(interval_years):
        lm0 = masks[int(y)]
        next_year = int(years_list[idx + 1])   # use next element from years_list (not y+1)
        lm1 = masks[int(next_year)]
        newly_px = int(np.sum(lm0 & (~lm1)))
        newly[idx] = float(newly_px) * px_area_ha

    # alpha per interval
    alpha = np.full(n_intervals, np.nan, dtype=float)
    for i in range(n_intervals):
        At = A[i]
        if At >= float(min_area_ha) and At > 0:
            alpha[i] = newly[i] / At
        else:
            alpha[i] = np.nan

    # beta per interval using dI eq: dI = alpha*A - beta*I  => beta = (alpha*A - dI) / I
    beta = np.full(n_intervals, np.nan, dtype=float)
    for i in range(n_intervals):
        It = I[i]
        if It >= float(min_inund_ha) and It > 0 and np.isfinite(alpha[i]):
            beta[i] = (alpha[i] * A[i] - dI[i]) / It
        else:
            beta[i] = np.nan

    summary = {
        "alpha": _summarize(alpha),
        "beta": _summarize(beta)
    }

    fitted = None
    if fit_global:
        # build arrays of valid rows for fitting (require finite A,I,dI)
        valid_mask = np.isfinite(A[:-1]) & np.isfinite(I[:-1]) & np.isfinite(dI)
        if valid_mask.sum() >= 2:
            A_fit = A[:-1][valid_mask]
            I_fit = I[:-1][valid_mask]
            dI_fit = dI[valid_mask]
            alpha_fit, beta_fit, resid, r2, method = _fit_alpha_beta_nnls(A_fit, I_fit, dI_fit)
            fitted = {
                "alpha_fit": alpha_fit,
                "beta_fit": beta_fit,
                "method": method,
                "r2": r2,
                "residuals": resid,
                "n_used": int(valid_mask.sum())
            }
        else:
            fitted = {"alpha_fit": np.nan, "beta_fit": np.nan, "method": "insufficient", "r2": np.nan, "residuals": None, "n_used": int(valid_mask.sum())}

    # Per-island computations if requested
    # prepare storage for per-island results (do NOT overwrite the parameter 'per_island')
    per_island_results = None
    if per_island:
        island_raster = g.get('island_raster', None)
        id_to_island = g.get('id_to_island', None)
        if island_raster is None or id_to_island is None:
            raise RuntimeError("per_island requires island_raster and id_to_island in globals.")
        uniq_ids = sorted([int(x) for x in np.unique(island_raster) if x != 0])
        per_island_results = {}
        for iid in uniq_ids:
            isl_mask = (island_raster == iid)
            # build A_i, I_i, newly_i similarly
            earlier_union_i = np.zeros((H,W), dtype=bool)
            A_i = np.zeros(len(years_list), dtype=float)
            I_i = np.zeros(len(years_list), dtype=float)
            for j, y in enumerate(years_list):
                land = masks[int(y)]
                A_i[j] = float(np.sum((land & isl_mask))) * px_area_ha
                I_mask_i = earlier_union_i & (~land)
                I_i[j] = float(np.sum(I_mask_i & isl_mask)) * px_area_ha
                earlier_union_i = earlier_union_i | land
            dI_i = np.diff(I_i)
            # --- per-island newly_inund loop (inside per-island block) ---
            newly_i = np.zeros(n_intervals, dtype=float)
            for idx, y in enumerate(interval_years):
                lm0 = masks[int(y)]
                next_year = int(years_list[idx + 1])   # use next element from years_list
                lm1 = masks[int(next_year)]
                newly_px = int(np.sum((lm0 & (~lm1) & isl_mask)))
                newly_i[idx] = float(newly_px) * px_area_ha
            alpha_i = np.full(n_intervals, np.nan, dtype=float)
            beta_i = np.full(n_intervals, np.nan, dtype=float)
            for k in range(n_intervals):
                if A_i[k] >= float(min_area_ha) and A_i[k] > 0:
                    alpha_i[k] = newly_i[k] / A_i[k]
                else:
                    alpha_i[k] = np.nan
                if I_i[k] >= float(min_inund_ha) and I_i[k] > 0 and np.isfinite(alpha_i[k]):
                    beta_i[k] = (alpha_i[k] * A_i[k] - dI_i[k]) / I_i[k]
                else:
                    beta_i[k] = np.nan
    
            # per-island fitted alpha/beta
            fitted_i = None
            if fit_global:
                valid_mask_i = np.isfinite(A_i[:-1]) & np.isfinite(I_i[:-1]) & np.isfinite(dI_i)
                if int(valid_mask_i.sum()) >= 2:
                    a_fit_i, b_fit_i, resid_i, r2_i, method_i = _fit_alpha_beta_nnls(A_i[:-1][valid_mask_i], I_i[:-1][valid_mask_i], dI_i[valid_mask_i])
                    fitted_i = {"alpha_fit": a_fit_i, "beta_fit": b_fit_i, "method": method_i, "r2": r2_i, "n_used": int(valid_mask_i.sum())}
                else:
                    fitted_i = {"alpha_fit": np.nan, "beta_fit": np.nan, "method": "insufficient", "r2": np.nan, "n_used": int(valid_mask_i.sum())}
    
            per_island_results[iid] = {
                "island_name": id_to_island.get(iid, f"id_{iid}"),
                "years": np.array(years_list),
                "A_ha": A_i,
                "I_ha": I_i,
                "dI_ha": dI_i,
                "newly_inund_ha": newly_i,
                "alpha_series": alpha_i,
                "beta_series": beta_i,
                "summary": {"alpha": _summarize(alpha_i), "beta": _summarize(beta_i)},
                "fitted": fitted_i
            }

   
    results = {
        "years": interval_years,
        "A_series_ha": A[:-1],
        "newly_inund_ha": newly,
        "alpha_series": alpha,
        "I_series_ha": I[:-1],
        "dI_ha": dI,
        "beta_series": beta,
        "summary": summary,
        "fitted": fitted,
        "per_island": per_island_results,   
        "px_area_ha": px_area_ha
    }
    return results

# If executed directly (requires run() executed earlier)
if __name__ == "__main__":
    try:
        res = compute_alpha_beta(per_island=True, min_area_ha=1.0, min_inund_ha=1.0, fit_global=True, plot=True)
        print("Global alpha summary:", res['summary']['alpha'])
        alpha = res['summary']['alpha']['mean']
        print("Global beta summary:", res['summary']['beta'])
        beta = res['summary']['beta']['mean']
        id_to_island = globals().get('id_to_island', {})
        alpha_island = []
        beta_island = []
        if res.get('per_island') is not None:
            for iid, info in sorted(res['per_island'].items()):
                name = id_to_island.get(iid, f"id_{iid}")
                alpha_mean = info.get('summary', {}).get('alpha', {}).get('mean', float('nan'))
                alpha_island.append((name, alpha_mean))
                beta_mean = info.get('summary', {}).get('beta', {}).get('mean', float('nan'))
                beta_island.append((name, beta_mean))
        if res['fitted'] is not None:
            print("Global fitted params:", res['fitted'])
        if res['per_island'] is not None:
            for iid, info in res['per_island'].items():
                print(iid, info['island_name'], "alpha_median:", info['summary']['alpha']['median'], "beta_median:", info['summary']['beta']['median'])
    except Exception as e:
        print("compute_alpha_beta failed:", e)
```
# **calculate the rA**
```python
#%% rA for each island seperated
# Compute rA (interval-wise + fitted global) with min_S_ha = 1.0
import numpy as np, warnings
g = globals()

def compute_rA_and_fits(min_S_ha=1.0, do_nnls=True):
    res = g.get('res')
    marsh_archive = g.get('marsh_archive')
    island_raster = g.get('island_raster')
    if not (isinstance(res, dict) and isinstance(res.get('per_island'), dict)):
        raise RuntimeError("res['per_island'] missing.")
    if not isinstance(marsh_archive, dict):
        raise RuntimeError("marsh_archive missing.")
    if island_raster is None:
        raise RuntimeError("island_raster missing.")

    # infer px area if needed
    px = g.get('px_area_ha')
    if px is None:
        px = 1.0
        g['px_area_ha'] = px

    # normalize per_island keys -> ints
    per_raw = res['per_island']
    per = {}
    for k, v in per_raw.items():
        try:
            per[int(k)] = v
        except Exception:
            continue
    island_ids = sorted(per.keys())
    if not island_ids:
        raise RuntimeError("No islands found in res['per_island'].")

    # get years
    sample = per[island_ids[0]]
    years = np.asarray(sample.get('years'), dtype=int)
    n_years = years.size
    if n_years < 2:
        raise RuntimeError("Need >=2 snapshot years.")

    # build S_map from marsh_archive & island_raster (snapshot-aligned, in ha)
    island_arr = np.asarray(island_raster)
    S_map = {}
    for iid in island_ids:
        mask_i = (island_arr == iid)
        S_vals = np.full(n_years, np.nan, dtype=float)
        for idx, y in enumerate(years):
            m = marsh_archive.get(int(y))
            if m is None:
                S_vals[idx] = np.nan
            else:
                S_vals[idx] = float(np.sum(np.asarray(m, dtype=bool) & mask_i)) * float(px)
        S_map[iid] = S_vals
    g['S_map_est'] = S_map

    # collect interval estimates and fit arrays
    all_interval_vals = []
    rA_by_island = {}
    rA_summary_by_island = {}
    fit_y = []
    fit_X = []

    for iid in island_ids:
        info = per[iid]
        A = np.asarray(info.get('A_ha', []), dtype=float)
        alpha_raw = info.get('alpha_series', None)
        alpha = np.asarray(alpha_raw, dtype=float) if alpha_raw is not None else np.array([], dtype=float)
        S = np.asarray(S_map.get(iid, np.full_like(A, np.nan)), dtype=float)

        if A.size < 2 or A.size != S.size:
            rA_by_island[iid] = np.full(max(0, A.size - 1), np.nan)
            rA_summary_by_island[iid] = {"count": 0, "mean": np.nan, "median": np.nan}
            continue

        dA = np.diff(A)                     # length n_intervals
        S_start = S[:-1]                    # length n_intervals
        n_intervals = dA.size

        # align alpha to intervals
        if alpha.size == n_intervals:
            alpha_al = alpha
        elif alpha.size > n_intervals:
            alpha_al = alpha[:n_intervals]
        else:
            alpha_al = np.full(n_intervals, np.nan)

        rA_t = np.full(n_intervals, np.nan)
        for j in range(n_intervals):
            Sj = S_start[j]
            if not (np.isfinite(Sj) and Sj >= float(min_S_ha) and Sj > 0.0):
                continue
            aj = alpha_al[j]
            if not np.isfinite(aj):
                continue
            At = A[j]
            val = (dA[j] + aj * At) / Sj
            if np.isfinite(val):
                rA_t[j] = float(val)
                all_interval_vals.append(val)
                fit_y.append(dA[j] + aj * At)
                fit_X.append(Sj)

        ok = np.isfinite(rA_t)
        if ok.sum() > 0:
            rA_summary_by_island[iid] = {"count": int(ok.sum()), "mean": float(np.nanmean(rA_t[ok])), "median": float(np.nanmedian(rA_t[ok]))}
        else:
            rA_summary_by_island[iid] = {"count": 0, "mean": np.nan, "median": np.nan}
        rA_by_island[iid] = rA_t

    # interval-based global summary
    all_interval_vals = np.asarray(all_interval_vals, dtype=float)
    interval_global_mean = float(np.nanmean(all_interval_vals)) if all_interval_vals.size else float('nan')
    interval_global_median = float(np.nanmedian(all_interval_vals)) if all_interval_vals.size else float('nan')
    interval_count = int(np.isfinite(all_interval_vals).sum())

    # global fit (least squares): fit y = rA * X
    if len(fit_X) > 0:
        X = np.asarray(fit_X, dtype=float)
        y = np.asarray(fit_y, dtype=float)
        denom = np.sum(X * X)
        if denom != 0:
            rA_ls = float(np.sum(X * y) / denom)
        else:
            rA_ls = float('nan')
        # try nnls
        rA_nnls = None
        if do_nnls:
            try:
                from scipy.optimize import nnls
                A_mat = X.reshape(-1,1)
                sol, resnorm = nnls(A_mat, y)
                rA_nnls = float(sol[0])
            except Exception:
                rA_nnls = float('nan')
    else:
        rA_ls = float('nan')
        rA_nnls = float('nan')

    # pick a recommended global rA: prefer non-negative NNLS if valid, else LS, else interval mean
    chosen_rA = None
    if np.isfinite(rA_nnls) and rA_nnls >= 0:
        chosen_rA = rA_nnls
        chosen_method = 'nnls'
    elif np.isfinite(rA_ls):
        chosen_rA = rA_ls
        chosen_method = 'ls'
    elif np.isfinite(interval_global_mean):
        chosen_rA = interval_global_mean
        chosen_method = 'interval_mean'
    else:
        chosen_rA = float('nan')
        chosen_method = 'none'

    # export to globals
    g['rA_by_island'] = rA_by_island
    g['rA_summary_by_island'] = rA_summary_by_island
    g['rA_interval_global'] = {"mean": interval_global_mean, "median": interval_global_median, "n": interval_count}
    g['rA_fit_ls'] = rA_ls
    g['rA_fit_nnls'] = rA_nnls
    g['rA'] = chosen_rA
    # build rA_island list for quick view
    id_to_island = g.get('id_to_island', {})
    rA_island_list = [(id_to_island.get(iid, f"id_{iid}"), rA_summary_by_island[iid]['mean']) for iid in island_ids]
    g['rA_island'] = rA_island_list
    # print summary
    print("Interval-based global: mean={:.6g}, median={:.6g}, n={}".format(interval_global_mean, interval_global_median, interval_count))
    print("Fit LS rA:", rA_ls, "NNLS rA:", rA_nnls)
    print("Chosen rA (exported to globals()['rA']):", chosen_rA, "method:", chosen_method)
    print("Per-island means:")
    for nm, v in rA_island_list:
        print("  {:20s} {:8.6g}".format(nm, v))
    return {
        "interval_global": g['rA_interval_global'],
        "fit_ls": rA_ls,
        "fit_nnls": rA_nnls,
        "chosen_rA": chosen_rA,
        "rA_island_list": rA_island_list
    }

# run with min_S_ha=1.0
_out = compute_rA_and_fits(min_S_ha=1.0, do_nnls=True)
```
# **Do all for Zeeland as a whole, so alpha rates etc.** 
```python
#%% Zeeland as a single aggregated system (area + rates + rA + RK4)
g = globals()
def compute_alpha_beta_zeeland(
    min_area_ha: float = 1.0,
    min_inund_ha: float = 1.0,
    years: Optional[list] = None,
    fit_global: bool = True,
    plot: bool = True,
) -> Dict[str, Any]:
    """
    Compute per-interval alpha and beta series for Zeeland as a single unified entity.

    Args:
      min_area_ha: Minimum A(t) to compute alpha_t for an interval.
      min_inund_ha: Minimum I(t) to compute beta_t for an interval.
      years: List of years to analyze; if None, inferred from globals.
      fit_global: Fit global alpha/beta across all valid intervals (optional).
      plot: Show diagnostic plots (requires matplotlib).

    Returns:
      dict with computed series and summaries (years, A_zeeland, newly_inund_zeeland, alpha_zeeland, beta_zeeland, etc.).
    """
    g = globals()
    px_area_ha = _infer_px_area_ha(g)

    df_yearly = g.get("df_yearly", None)
    land_masks_cache = g.get("land_masks_by_year") or g.get("land_masks") or None

    if years is None:
        if df_yearly is not None and "Year" in df_yearly.columns:
            years_list = sorted(int(x) for x in df_yearly["Year"].values)
        elif land_masks_cache is not None:
            years_list = sorted(int(x) for x in land_masks_cache.keys())
        elif "YEARS" in g:
            years_list = list(map(int, g["YEARS"]))
        else:
            raise RuntimeError("Could not infer years. Ensure df_yearly, land_masks_by_year, or YEARS exist in globals.")
    else:
        years_list = sorted(int(y) for y in years)

    if len(years_list) < 2:
        raise ValueError("Need at least two years to compute per-interval quantities.")

    interval_years = np.array(years_list[:-1], dtype=int)
    n_intervals = len(interval_years)

    if land_masks_cache is not None:
        masks = {int(y): land_masks_cache[int(y)].astype(bool) for y in years_list}
    else:
        rasterize_fn = g.get("rasterize_union_year")
        df_all = g.get("df_all")
        if rasterize_fn is None or df_all is None:
            raise RuntimeError("No cached land masks and cannot rasterize (missing rasterize_union_year or df_all).")
        masks = {}
        for y in years_list:
            lm, _ = rasterize_fn(
                df_all,
                int(y),
                int(g["width"]),
                int(g["height"]),
                g["transform"],
                sf=g.get("SUBSAMPLE_FACTOR", 1),
                coverage_threshold=float(g.get("COVERAGE_THRESHOLD", 0.5)),
            )
            masks[int(y)] = lm.astype(bool)

    example = next(iter(masks.values()))
    H, W = example.shape
    earlier_union = np.zeros((H, W), dtype=bool)
    A_list = []
    I_list = []

    # Compute total land area (A) and inundation (I) for Zeeland as a whole
    for y in years_list:
        land = masks[int(y)]
        A_ha = float(np.sum(land)) * px_area_ha
        I_mask = earlier_union & (~land)
        I_ha = float(np.sum(I_mask)) * px_area_ha
        A_list.append(A_ha)
        I_list.append(I_ha)
        earlier_union = earlier_union | land

    A_zeeland = np.array(A_list, dtype=float)
    I_zeeland = np.array(I_list, dtype=float)
    dI_zeeland = np.diff(I_zeeland)

    # Compute newly inundated area per year
    newly_zeeland = np.zeros(n_intervals, dtype=float)
    for idx, y in enumerate(interval_years):
        lm0 = masks[int(y)]
        next_year = int(years_list[idx + 1])
        lm1 = masks[next_year]
        newly_px = int(np.sum(lm0 & (~lm1)))
        newly_zeeland[idx] = float(newly_px) * px_area_ha

    # Compute alpha_zeeland for each interval
    alpha_zeeland = np.full(n_intervals, np.nan, dtype=float)
    for i in range(n_intervals):
        At = A_zeeland[i]
        if At >= min_area_ha and At > 0:
            alpha_zeeland[i] = newly_zeeland[i] / At

    # Compute beta_zeeland for each interval
    beta_zeeland = np.full(n_intervals, np.nan, dtype=float)
    for i in range(n_intervals):
        It = I_zeeland[i]
        if It >= min_inund_ha and It > 0 and np.isfinite(alpha_zeeland[i]):
            beta_zeeland[i] = (alpha_zeeland[i] * A_zeeland[i] - dI_zeeland[i]) / It

    # Summarize alpha and beta for Zeeland
    summary_zeeland = {
        "alpha": _summarize(alpha_zeeland),
        "beta": _summarize(beta_zeeland),
    }

    # Fit global alpha/beta for Zeeland
    fitted_zeeland = None
    if fit_global:
        valid_mask = np.isfinite(A_zeeland[:-1]) & np.isfinite(I_zeeland[:-1]) & np.isfinite(dI_zeeland)
        if valid_mask.sum() >= 2:
            A_fit = A_zeeland[:-1][valid_mask]
            I_fit = I_zeeland[:-1][valid_mask]
            dI_fit = dI_zeeland[valid_mask]
            alpha_fit, beta_fit, resid, r2, method = _fit_alpha_beta_nnls(A_fit, I_fit, dI_fit)
            fitted_zeeland = {
                "alpha_fit": alpha_fit,
                "beta_fit": beta_fit,
                "method": method,
                "r2": r2,
                "residuals": resid,
                "n_used": int(valid_mask.sum()),
            }

    # Compile results for Zeeland
    results = {
        "years": interval_years,
        "A_series_ha": A_zeeland[:-1],
        "newly_inund_ha": newly_zeeland,
        "alpha_series": alpha_zeeland,
        "I_series_ha": I_zeeland[:-1],
        "dI_ha": dI_zeeland,
        "beta_series": beta_zeeland,
        "summary": summary_zeeland,
        "fitted": fitted_zeeland,
        "px_area_ha": px_area_ha,
    }

    return results
res_zeeland = compute_alpha_beta_zeeland(
    min_area_ha=1.0,
    min_inund_ha=1.0,
    fit_global=True  # Enable global fitting
)
# ------------------------------------------------------------------
# 1. Compute observed totals and rates  
res_area  = compute_area_changes(per_island=False)
res_rates = compute_alpha_beta(per_island=False)

# --- observed totals ------------------------------------------------
years = np.asarray(res_area['years'], dtype=int)
A_obs = np.asarray(res_area['A_ha'], dtype=float)
I_obs = np.asarray(res_area['I_ha'], dtype=float)
S_obs = np.asarray(
    res_area.get('S_snap_ha', np.full_like(A_obs, np.nan)),
    dtype=float
)

n_years = len(years)

# ------------------------------------------------------------------
# 2. Extract global alpha & beta (Zeeland-wide)
alpha_zeeland = np.nan
beta_zeeland  = np.nan

if not np.isfinite(alpha_zeeland) or not np.isfinite(beta_zeeland):
    alpha_zeeland = res_zeeland['summary']['alpha']['mean']
    beta_zeeland  = res_zeeland['summary']['beta']['mean']

alpha_zeeland = float(alpha_zeeland)
beta_zeeland  = float(beta_zeeland)

# ------------------------------------------------------------------
# 3. Compute rA for Zeeland as a whole
#    rA_t = (A_{t+1} - A_t + alpha * A_t) / S_t
# ------------------------------------------------------------------
rA_vals_z = []

for t in range(n_years - 1):
    At, At1 = A_obs[t], A_obs[t+1]
    St = S_obs[t]

    if np.isfinite(At) and np.isfinite(At1) and np.isfinite(St) and St > 0:
        rA_t = (At1 - At + alpha_zeeland * At) / St
        if np.isfinite(rA_t):
            rA_vals_z.append(rA_t)

rA_vals_z = np.asarray(rA_vals_z, dtype=float)
rA_z = float(np.nanmean(rA_vals_z)) if rA_vals_z.size else 0.0

# store for later reuse
g['rA_interval_global'] = {
    "mean": rA_z,
    "values": rA_vals_z
}

# ------------------------------------------------------------------
# 4. Aggregate g and S_max (collapsed islands)
g_by_id     = g.get('g_by_id')
S_max_by_id = g.get('S_max_by_id')

g_total = 0.0
if isinstance(g_by_id, dict) and isinstance(S_max_by_id, dict):
    num = 0.0
    den = 0.0
    for iid, gv in g_by_id.items():
        try:
            smax = float(S_max_by_id[iid]['area_ha'])
            gv   = float(gv)
        except Exception:
            continue
        if smax > 0 and np.isfinite(gv):
            num += gv * smax
            den += smax
    if den > 0:
        g_total = num / den

# fallback
if not np.isfinite(g_total):
    g_total = 0.0

S_max_total = 2.934*10**5 # area zeeland province total, so land+water

if S_max_total <= 0:
    finite_S = S_obs[np.isfinite(S_obs)]
    S_max_total = float(np.nanmax(finite_S)) if finite_S.size else max(1.0, np.nanmax(A_obs))

# ------------------------------------------------------------------
# 5. Aggregated ODE system
def deriv(A, I, S):
    dS_growth = g_total * S * (1 - S / S_max_total)
    dA = rA_z * S - alpha_zeeland * A
    dI = alpha_zeeland * A - beta_zeeland * I
    dS = dS_growth + beta_zeeland * I - rA_z * S
    return dA, dI, dS

def rk4_step(A, I, S, dt=1.0):
    k1 = deriv(A, I, S)
    k2 = deriv(A + 0.5*dt*k1[0], I + 0.5*dt*k1[1], S + 0.5*dt*k1[2])
    k3 = deriv(A + 0.5*dt*k2[0], I + 0.5*dt*k2[1], S + 0.5*dt*k2[2])
    k4 = deriv(A + dt*k3[0],     I + dt*k3[1],     S + dt*k3[2])

    A_n = A + (dt/6)*(k1[0] + 2*k2[0] + 2*k3[0] + k4[0])
    I_n = I + (dt/6)*(k1[1] + 2*k2[1] + 2*k3[1] + k4[1])
    S_n = S + (dt/6)*(k1[2] + 2*k2[2] + 2*k3[2] + k4[2])

    A_n = max(0.0, A_n)
    I_n = max(0.0, I_n)
    S_n = min(max(0.0, S_n), S_max_total)

    return A_n, I_n, S_n

# ------------------------------------------------------------------
# 6. Run simulation
A_sim = np.zeros(n_years)
I_sim = np.zeros(n_years)
S_sim = np.zeros(n_years)

A_sim[0] = np.nan_to_num(A_obs[0])
I_sim[0] = np.nan_to_num(I_obs[0])
S_sim[0] = np.nan_to_num(S_obs[0])

for t in range(n_years - 1):
    A_sim[t+1], I_sim[t+1], S_sim[t+1] = rk4_step(
        A_sim[t], I_sim[t], S_sim[t]
    )

# ------------------------------------------------------------------
# 7. RMSE diagnostics
def rmse(obs, sim):
    m = np.isfinite(obs)
    return np.sqrt(np.mean((obs[m] - sim[m])**2)) if m.any() else np.nan

rmse_A = rmse(A_obs, A_sim)
rmse_I = rmse(I_obs, I_sim)
rmse_S = rmse(S_obs, S_sim)

print("Zeeland aggregated parameters:")
print(f" alpha_Z = {alpha_zeeland:.4e}")
print(f" beta_Z  = {beta_zeeland:.4e}")
print(f" rA_z    = {rA_z:.4e}")
print(f" g     = {g_total:.4e}")
print(f" Smax  = {S_max_total:.1f} ha")

print(f"RMSE (ha): A={rmse_A:.1f}, I={rmse_I:.1f}, S={rmse_S:.1f}")

# save to globals
g['A_sim_total'] = A_sim
g['I_sim_total'] = I_sim
g['S_sim_total'] = S_sim
g['rmse_totals'] = {'A': rmse_A, 'I': rmse_I, 'S': rmse_S}
```

## **Growth factor g and S_max** 
```python
#%% Compute growth factor g for Zeeland as a whole (list-based S)
# Compute Zeeland-wide g from total S (single S_max)
g = globals()

# 0) helper
def _to_1d_array(x, n=None):
    """Coerce x to a 1D numpy array. If n provided, pad/truncate to length n."""
    if x is None:
        return None
    a = np.asarray(x, dtype=float).ravel()
    if n is not None:
        if a.size < n:
            a = np.concatenate([a, np.full(n - a.size, np.nan)])
        elif a.size > n:
            a = a[:n]
    return a

# 1) get years
years = np.asarray(g.get('years', []), dtype=float)
if years.size < 2:
    raise RuntimeError("globals()['years'] missing or too short. Run compute_area_changes first.")
n_years = years.size

# 2) build S_total (prefer S_total in globals, else sum per-island S)
if 'S_total' in g:
    S_total = np.asarray(g['S_total'], dtype=float).ravel()
    S_total = _to_1d_array(S_total, n_years)
else:
    # Candidate sources
    S_src = g.get('S') or g.get('S_dict') or g.get('S_by_id')
    if S_src is None:
        raise RuntimeError("No S data found in globals (checked 'S','S_dict','S_by_id').")
    # If S_src is a 1D array matching years -> treat as total series
    arr = np.asarray(S_src)
    if arr.ndim == 1 and arr.size == n_years:
        S_total = _to_1d_array(arr, n_years)
    else:
        # sum per-island series: accept dict (iid -> arr) or list-of-arrays (index == island id)
        S_total = np.zeros(n_years, dtype=float)
        contributed = np.zeros(n_years, dtype=int)
        if isinstance(S_src, dict):
            iterable = S_src.items()
        else:
            iterable = enumerate(S_src)
        for k, val in iterable:
            if val is None:
                continue
            try:
                a = _to_1d_array(val, n_years)
            except Exception:
                continue
            mask = np.isfinite(a)
            if not mask.any():
                continue
            S_total[mask] += a[mask]
            contributed[mask] += 1
        if contributed.sum() == 0:
            raise RuntimeError("No finite per-island S values found to sum into S_total.")
        # (we keep sum, not mean)
    # store S_total for convenience
    g['S_total'] = S_total

S_total = np.asarray(S_total, dtype=float).ravel()
if S_total.size != n_years:
    S_total = _to_1d_array(S_total, n_years)

# 3) choose single S_max (override -> sum S_max_by_id -> observed peak)
S_max_zeeland = S_max_total

# 4) compute dS/dt and g_series
if np.any(np.diff(years) <= 0):
    raise RuntimeError("years must be strictly increasing.")

dt = np.diff(years)
S0 = S_total[:-1]
S1 = S_total[1:]
dS_dt = (S1 - S0) / dt

denom = S0 * (1.0 - (S0 / float(S_max_zeeland)))
g_series = np.full_like(dS_dt, np.nan, dtype=float)

# valid where denom finite and sufficiently large (avoid division by ~0)
valid = np.isfinite(dS_dt) & np.isfinite(denom) & (np.abs(denom) > 1e-12)
g_series[valid] = dS_dt[valid] / denom[valid]

# 5) summary stats (robust)
vals = g_series[np.isfinite(g_series)]
n_valid = vals.size
g_mean = float(np.nanmean(vals)) if n_valid else np.nan
g_median = float(np.nanmedian(vals)) if n_valid else np.nan
g_trimmed = np.nan
if n_valid:
    lo, hi = np.nanpercentile(vals, [2.5, 97.5])
    trimmed = vals[(vals >= lo) & (vals <= hi)]
    if trimmed.size:
        g_trimmed = float(np.nanmean(trimmed))

# choose g_total: prefer trimmed mean, else median, else mean, else 0.0
if np.isfinite(g_trimmed):
    g_total = g_trimmed
elif np.isfinite(g_median):
    g_total = g_median
elif np.isfinite(g_mean):
    g_total = g_mean
else:
    warnings.warn("No valid g estimates from S_total; defaulting g_total to 0.0", UserWarning)
    g_total = 0.0

# store results
g['S_snap_total'] = S_total
g['S_max_total'] = float(S_max_zeeland)
g['g_series_total'] = g_series
g['g_total'] = float(g_total)

# 6) print summary
print("Computed Zeeland-wide growth factor g:")
print(" S_max used:", S_max_zeeland)
print(" valid g points:", n_valid)
print(" g_mean:", g_mean, " g_median:", g_median, " g_trimmed:", g_trimmed)
print(" chosen g_total (saved to globals()['g_total']):", g_total)
```
## **Now I put A S and I in a simple year system so they are all in session, just a preferation for me too keep sanity**
```python
#%% A, I, S for Zeeland as a whole (snapshot-aligned)
g = globals()

# ------------------------------------------------------------------
# 1) Get area results (Zeeland-wide)
# ------------------------------------------------------------------
res_area = g.get('res_area') or g.get('res')
if not isinstance(res_area, dict):
    raise RuntimeError("Run compute_area_changes(per_island=False) and store as res_area")

years = np.asarray(res_area['years'], dtype=int)
n_years = years.size
if n_years < 2:
    raise RuntimeError("Need at least 2 snapshot years")

A = np.asarray(res_area['A_ha'], dtype=float)
I = np.asarray(res_area['I_ha'], dtype=float)

# ------------------------------------------------------------------
# 2) Build S(t) for Zeeland
# ------------------------------------------------------------------
S = None

# preferred: snapshot salt-marsh areas already computed
if 'S_snap_ha' in res_area:
    S = np.asarray(res_area['S_snap_ha'], dtype=float)

# fallback: compute from marsh_archive
if S is None or not np.any(np.isfinite(S)):
    marsh_archive = g.get('marsh_archive')
    px_area = g.get('px_area_ha', 1.0)

    if isinstance(marsh_archive, dict):
        S_vals = np.full(n_years, np.nan, dtype=float)
        for i, y in enumerate(years):
            m = marsh_archive.get(int(y))
            if m is not None:
                S_vals[i] = float(np.sum(np.asarray(m, dtype=bool))) * px_area
        S = S_vals
    else:
        S = np.full(n_years, np.nan, dtype=float)

# ------------------------------------------------------------------
# 3) Enforce snapshot alignment (safety)
# ------------------------------------------------------------------
def _align(arr, n):
    arr = np.asarray(arr, dtype=float)
    if arr.size == n:
        return arr
    if arr.size == n - 1:
        return np.append(arr, arr[-1])
    out = np.full(n, np.nan)
    out[:min(n, arr.size)] = arr[:min(n, arr.size)]
    return out

A = _align(A, n_years)
I = _align(I, n_years)
S = _align(S, n_years)

# ------------------------------------------------------------------
# 4) Store in globals (single-system convention)
# ------------------------------------------------------------------
g['years_zeeland'] = years
g['A_total'] = A
g['I_total'] = I
g['S_total'] = S

# ------------------------------------------------------------------
# 5) Quick sanity check
# ------------------------------------------------------------------
print("Zeeland-wide A/I/S built:")
print(" years[0:5]:", years[:5])
print(" A[0:5]:", A[:5])
print(" I[0:5]:", I[:5])
print(" S[0:5]:", S[:5])
print(" lengths:", A.size, I.size, S.size)
```
## **Simulation runs**
This part is where i used observed data and the quantile starting points and let the simulation run from different initial values. It uses the g, alpha, beta and rA values
```python
#%% Zeeland-wide simulation with quantile & mean starting points
n_years = len(yrs)
dt = 1.0

# Observed totals
A_obs = np.asarray(A, dtype=float)
I_obs = np.asarray(I, dtype=float)
S_obs = np.asarray(S, dtype=float)

# Define starting points: quantiles + mean
quantiles = [0.0, 0.25, 0.5, 0.75, 1.0]
quantile_labels = ['Q0', 'Q25', 'Q50', 'Q75', 'Q100']

A_start_vals = np.quantile(A_obs, quantiles)
I_start_vals = np.quantile(I_obs, quantiles)
S_start_vals = np.quantile(S_obs, quantiles)

# Add mean as extra starting point
A_start_vals = np.append(A_start_vals, np.nanmean(A_obs))
I_start_vals = np.append(I_start_vals, np.nanmean(I_obs))
S_start_vals = np.append(S_start_vals, np.nanmean(S_obs))
labels = quantile_labels + ['Mean']

sim_results_A = []
sim_results_I = []
sim_results_S = []

# Run simulations for each starting points
for A0, I0, S0 in zip(A_start_vals, I_start_vals, S_start_vals):
    A_sim = np.zeros(n_years)
    I_sim = np.zeros(n_years)
    S_sim = np.zeros(n_years)

    A_sim[0] = A0
    I_sim[0] = I0
    S_sim[0] = S0

    for t in range(n_years-1):
        dA = rA_z*S_sim[t] - alpha_zeeland*A_sim[t]
        dI = alpha_zeeland*A_sim[t] - beta_zeeland*I_sim[t]
        dS = g_total*S_sim[t]*(1 - S_sim[t]/S_max_total) + beta_zeeland*I_sim[t] - rA_z*S_sim[t]

        A_sim[t+1] = A_sim[t] + dA*dt
        I_sim[t+1] = I_sim[t] + dI*dt
        S_sim[t+1] = S_sim[t] + dS*dt

    sim_results_A.append(A_sim)
    sim_results_I.append(I_sim)
    sim_results_S.append(S_sim)
```
## **Kaplan-meier for each island**
You need to run this before you run the next part, some helper functions in this code are used in the zeeland as a whole code. 
```Python
#%% A4 Kaplan-Meier survival rate 
from typing import Tuple, Any, Dict, List
try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

# try to import chi2 for exact CI; fallback if unavailable
try:
    from scipy.stats import chi2
    SCIPY_CHI2 = True
except Exception:
    SCIPY_CHI2 = False


def _compute_A4_mle_from_episodes(durations, events, alpha: float = 0.05) -> Tuple[Any, Any, Tuple[Any, Any], int, int]:
    dur = np.asarray(durations, dtype=float)
    ev = np.asarray(events, dtype=int)
    n_total = dur.size
    n_events = int(np.sum(ev))
    if n_total == 0 or n_events <= 0:
        return None, None, (None, None), n_events, n_total

    # Total time at risk (sum of observed durations including censored)
    T = float(np.sum(dur))
    # MLE for exponential rate lambda_hat = r / T  => tau_hat = 1 / lambda_hat = T / r
    lambda_hat = float(n_events) / float(T)
    tau_hat = float(T) / float(n_events)

    # approximate variance via delta method: var(tau_hat) = tau_hat^2 / r
    var_tau = (tau_hat ** 2) / float(max(1, n_events))
    se_tau = math.sqrt(var_tau)

    # Confidence interval: use chi-square inversion for lambda if possible (exact)
    ci_low_tau = None
    ci_high_tau = None
    if SCIPY_CHI2 and n_events > 0:
        # CI for lambda: [chi2.ppf(alpha/2, 2r) / (2T), chi2.ppf(1-alpha/2, 2r) / (2T)]
        # invert to get tau CI: [2T / chi2.ppf(1-alpha/2, 2r), 2T / chi2.ppf(alpha/2, 2r)]
        df = 2 * n_events
        # handle extreme cases: ppf(alpha/2) can be 0 when df small and alpha tiny, guard it
        lower_chi = chi2.ppf(alpha / 2.0, df) if (alpha / 2.0) > 0 else 0.0
        upper_chi = chi2.ppf(1.0 - alpha / 2.0, df)
        try:
            if upper_chi > 0:
                ci_low_tau = 2.0 * T / upper_chi
            else:
                ci_low_tau = None
            if lower_chi > 0:
                ci_high_tau = 2.0 * T / lower_chi
            else:
                ci_high_tau = None
        except Exception:
            ci_low_tau = None
            ci_high_tau = None

    # fallback to normal approx on tau if chi2 unavailable or failed
    if (ci_low_tau is None or ci_high_tau is None) and se_tau is not None:
        z = 1.96  # 95% CI
        ci_low_tau = max(0.0, tau_hat - z * se_tau)
        ci_high_tau = tau_hat + z * se_tau

    return float(tau_hat), float(var_tau), (float(ci_low_tau) if ci_low_tau is not None else None,
                                           float(ci_high_tau) if ci_high_tau is not None else None), n_events, n_total


# Reuse the episode extraction function (same as before)
def extract_marsh_episodes_from_archive(marsh_archive, years, island_raster) -> pd.DataFrame:
    years = np.asarray(sorted(years), dtype=int)
    if len(years) == 0:
        return pd.DataFrame(columns=['pixel_index', 'island_id', 'start_year', 'end_year', 'duration_years', 'event_observed'])
    height, width = island_raster.shape
    n_years = len(years)
    stack = np.zeros((n_years, height, width), dtype=bool)
    for i, y in enumerate(years):
        mask = marsh_archive.get(int(y), None)
        if mask is None:
            mask = np.zeros((height, width), dtype=bool)
        stack[i] = mask
    rows = []
    flat_idxs = np.nonzero(island_raster.ravel() > 0)[0]
    for flat in flat_idxs:
        r = int(flat // width); c = int(flat % width)
        island_id = int(island_raster[r, c])
        ts = stack[:, r, c]
        i = 0
        while i < n_years:
            if not ts[i]:
                i += 1
                continue
            j = i
            while j + 1 < n_years and ts[j + 1]:
                j += 1
            start_year = int(years[i])
            if j < n_years - 1:
                end_year = int(years[j + 1])
                duration = int(end_year - start_year)
                event = 1
            else:
                end_year = -1
                duration = int(years[-1] - start_year)
                event = 0
            pix_index = int(r * width + c)
            rows.append((pix_index, island_id, start_year, end_year, duration, event))
            i = j + 1
    df = pd.DataFrame(rows, columns=['pixel_index', 'island_id', 'start_year', 'end_year', 'duration_years', 'event_observed'])
    return df


def compute_and_plot_all_islands_A4_mle(years=None, save_csv=None, px_area_ha=None, random_seed=None):
    """
    Compute Kaplan–Meier tables and overlay the exponential parametric curve derived from
    the exponential MLE (tau = T/r) that properly accounts for right-censoring.

    Returns:
      figs: dict island_name -> matplotlib.Figure
      df_summary: pandas.DataFrame summarizing A4-MLE results per island
    """
    g = globals()
    required = ('island_raster', 'id_to_island', 'marsh_archive')
    for name in required:
        if name not in g:
            raise RuntimeError(f"Required global '{name}' not found. Make sure island_raster, id_to_island and marsh_archive are available.")

    if random_seed is not None:
        np.random.seed(int(random_seed))

    # determine years
    if years is None:
        res = g.get('res', {})
        if isinstance(res, dict) and res.get('per_island'):
            sample_iid = next(iter(res['per_island'].keys()))
            years = list(res['per_island'][sample_iid]['years'])
        else:
            years = sorted([int(y) for y in list(g.get('marsh_archive', {}).keys())])
    years = sorted([int(y) for y in years])

    df_polder = g.get('episodes_polder_df', pd.DataFrame())
    df_inund = g.get('episodes_inund_df', pd.DataFrame())
    marsh_archive = g.get('marsh_archive', {})

    island_raster = g['island_raster']
    id_to_island = g['id_to_island']

    figs = {}
    summary_rows: List[Dict[str, Any]] = []

    print("Extracting marsh episodes from marsh_archive...")
    df_marsh_episodes = extract_marsh_episodes_from_archive(marsh_archive, years, island_raster)
    if not df_marsh_episodes.empty and 'island' not in df_marsh_episodes.columns:
        df_marsh_episodes['island'] = df_marsh_episodes['island_id'].map(id_to_island)

    for iid, island_name in sorted([(k, v) for k, v in id_to_island.items()], key=lambda x: x[1]):
        print(f"Processing island: {island_name} (id={iid})")
        mask_p = (df_polder['island_id'] == iid) if (not df_polder.empty and 'island_id' in df_polder.columns) else np.array([], dtype=bool)
        mask_i = (df_inund['island_id'] == iid) if (not df_inund.empty and 'island_id' in df_inund.columns) else np.array([], dtype=bool)
        dfp = df_polder[mask_p] if not df_polder.empty else pd.DataFrame()
        dfi = df_inund[mask_i] if not df_inund.empty else pd.DataFrame()
        dfm = df_marsh_episodes[df_marsh_episodes['island_id'] == iid] if not df_marsh_episodes.empty else pd.DataFrame()

        dur_p = np.asarray(dfp['duration_years'].values) if not dfp.empty else np.array([])
        ev_p = np.asarray(dfp['event_observed'].values) if not dfp.empty else np.array([])
        dur_i = np.asarray(dfi['duration_years'].values) if not dfi.empty else np.array([])
        ev_i = np.asarray(dfi['event_observed'].values) if not dfi.empty else np.array([])
        dur_m = np.asarray(dfm['duration_years'].values) if not dfm.empty else np.array([])
        ev_m = np.asarray(dfm['event_observed'].values) if not dfm.empty else np.array([])

        # KM tables
        km_p = _kaplan_meier_table(dur_p, ev_p) if dur_p.size else pd.DataFrame()
        km_i = _kaplan_meier_table(dur_i, ev_i) if dur_i.size else pd.DataFrame()
        km_m = _kaplan_meier_table(dur_m, ev_m) if dur_m.size else pd.DataFrame()

        # A4 MLE (exponential MLE with censoring)
        a4_mle_p = _compute_A4_mle_from_episodes(dur_p, ev_p)
        a4_mle_i = _compute_A4_mle_from_episodes(dur_i, ev_i)
        a4_mle_m = _compute_A4_mle_from_episodes(dur_m, ev_m)

        # Plot KM + exponential (MLE) overlay
        if plt is None:
            print("matplotlib not available; skipping plotting.")
            fig = None
        else:
            fig, axes = plt.subplots(3, 1, figsize=(8, 10), squeeze=False)
            fig.suptitle(f"Island: {island_name}", fontsize=14)
            ax_p = axes[0][0]; ax_i = axes[1][0]; ax_m = axes[2][0]

            def _plot_km_with_mle(ax, km_df, a4_tuple, color, label):
                if km_df.empty:
                    ax.text(0.5, 0.5, "No episodes", transform=ax.transAxes, ha='center', va='center')
                    ax.set_xlim(0, max(1.0, float(years[-1] - years[0])))
                    ax.set_ylim(0, 1.02)
                    ax.set_title(label)
                    fig.suptitle(f"Island: {island_name}", fontsize=14)
                    return
                xs = [0.0]; ys = [1.0]
                for t, S in zip(km_df['time'].values, km_df['survival'].values):
                    xs.extend([t, t]); ys.extend([ys[-1], S])
                tmax = max(1.0, float(years[-1] - years[0]))
                if xs[-1] < tmax:
                    xs.append(tmax); ys.append(ys[-1])
                ax.plot(xs, ys, drawstyle='steps-post', color=color, lw=2, label='survival rate')

                tau_hat, var_tau, ci_tau, n_events, n_total = a4_tuple
                if tau_hat is not None and n_events > 0 and np.isfinite(tau_hat):
                    tvals = np.linspace(0.0, tmax, 300)
                    S_exp = np.exp(-tvals / float(tau_hat))
                    ax.plot(tvals, S_exp, color=color, ls='--', lw=1.5, label=f"mean τ={tau_hat:.1f}")
                    low, high = ci_tau
                    if low is not None and high is not None and low > 0 and high > 0:
                        S_low = np.exp(-tvals / float(high))
                        S_high = np.exp(-tvals / float(low))
                        ax.fill_between(tvals, S_low, S_high, color=color, alpha=0.12)
                ax.set_xlabel("Years")
                ax.set_ylabel("Survival S(t)")
                ax.set_xlim(0, tmax)
                ax.set_ylim(0.0, 1.03)
                ax.grid(alpha=0.25)
                ax.legend(fontsize=8)
                ax.set_title(label)

            _plot_km_with_mle(ax_p, km_p, a4_mle_p, 'orange', 'Polder (A)')
            _plot_km_with_mle(ax_i, km_i, a4_mle_i, 'tab:blue', 'Inundated (I)')
            _plot_km_with_mle(ax_m, km_m, a4_mle_m, 'green', 'Marsh (S)')

            plt.tight_layout(rect=[0, 0, 1, 0.97])
            figs[island_name] = fig

        # assemble summary
        tauA_p, varA_p, ciA_p, events_p, n_p_total = a4_mle_p
        tauA_i, varA_i, ciA_i, events_i, n_i_total = a4_mle_i
        tauA_m, varA_m, ciA_m, events_m, n_m_total = a4_mle_m

        summary_rows.append({
            "island": island_name,
            "n_polder": int(dur_p.size), "events_polder": int(np.sum(ev_p)),
            "A4MLE_polder_tau": float(tauA_p) if tauA_p is not None else None,
            "A4MLE_polder_ci_low": float(ciA_p[0]) if ciA_p and ciA_p[0] is not None else None,
            "A4MLE_polder_ci_high": float(ciA_p[1]) if ciA_p and ciA_p[1] is not None else None,

            "n_inund": int(dur_i.size), "events_inund": int(np.sum(ev_i)),
            "A4MLE_inund_tau": float(tauA_i) if tauA_i is not None else None,
            "A4MLE_inund_ci_low": float(ciA_i[0]) if ciA_i and ciA_i[0] is not None else None,
            "A4MLE_inund_ci_high": float(ciA_i[1]) if ciA_i and ciA_i[1] is not None else None,

            "n_marsh": int(dur_m.size), "events_marsh": int(np.sum(ev_m)),
            "A4MLE_marsh_tau": float(tauA_m) if tauA_m is not None else None,
            "A4MLE_marsh_ci_low": float(ciA_m[0]) if ciA_m and ciA_m[0] is not None else None,
            "A4MLE_marsh_ci_high": float(ciA_m[1]) if ciA_m and ciA_m[1] is not None else None,
        })

    df_summary = pd.DataFrame(summary_rows)
    if save_csv:
        df_summary.to_csv(save_csv, index=False)
        print("Saved summary to", save_csv)
    g['figs_A4_MLE'] = figs
    g['df_summary_A4_MLE'] = df_summary
    print("Completed A4-MLE analysis. Figures in 'figs_A4_MLE', summary in 'df_summary_A4_MLE'.")
    return figs, df_summary


# Reuse the previous Kaplan-Meier builder for consistency
def _kaplan_meier_table(durations, events):
    if durations.size == 0:
        return pd.DataFrame(columns=["time","n_at_risk","events","censored","survival","var_surv"])
    dur = np.asarray(durations, dtype=float)
    ev = np.asarray(events, dtype=int)
    uniq_times = np.sort(np.unique(dur))
    S = 1.0
    var_acc = 0.0
    rows = []
    for t in uniq_times:
        at_risk = int(np.sum(dur >= t))
        d_i = int(np.sum((dur == t) & (ev == 1)))
        c_i = int(np.sum((dur == t) & (ev == 0)))
        if at_risk > 0 and d_i > 0:
            q = 1.0 - (d_i / float(at_risk))
            S = S * q
            if at_risk - d_i > 0:
                var_acc += d_i / (float(at_risk) * float(at_risk - d_i))
        varS = (var_acc * (S**2)) if at_risk > 0 else 0.0
        rows.append((float(t), at_risk, d_i, c_i, float(S), float(varS)))
    km = pd.DataFrame(rows, columns=["time","n_at_risk","events","censored","survival","var_surv"])
    return km


if __name__ == "__main__":
    compute_and_plot_all_islands_A4_mle()
```
## **Kaplan-meier survival rates for zeeland as whole**
This is the part for survival analysis. Is uses 'blobs' or patches, which are areas that merges pixels if they are the same as neighbor. This way we do not have the issue of very small (less than 3 pixels) events and focus on the larger system dynamics. Additionally it is way faster this way. 
These patches are used for statistical survival analysis (e.g., Kaplan-Meier estimation) to understand the persistence of land features like marshes, inundation, or polders.
```python
#%%  Add a function that creates merged neighbour pixel 'blobs' in order to not see them all as seperate events. 
# Full integrated code: detect patches + KM with pointwise 95% CI + plotting (Zeeland)
# ----------------------------
# helper: detect change patches
# ----------------------------
def detect_change_patches_per_year(mask_stack: Dict[int, np.ndarray],
                                   years: List[int],
                                   change_type: str = "loss",
                                   connectivity: int = 2,
                                   min_area: int = 3,
                                   island_raster: np.ndarray = None) -> Tuple[pd.DataFrame, Dict[int, np.ndarray]]:
    """
    Detect spatial patches of pixels that change cover between consecutive years.
    Returns df_patches (one row per patch) and labeled_maps: year -> labeled map (global patch ids, 0 background).
    Only marks patches for the 'current' year in pair (prev -> cur): i.e. change observed at cur.
    """
    years = [int(y) for y in years]
    if len(years) < 2:
        return pd.DataFrame(columns=[
            "patch_id", "year", "change", "pixel_indices", "area_pixels", "island_ids", "centroid", "bbox"
        ]), {}

    # connectivity structure
    if connectivity == 1:
        structure = np.array([[0,1,0],[1,1,1],[0,1,0]], dtype=int)
    else:
        structure = np.ones((3,3), dtype=int)

    patch_rows = []
    labeled_maps = {}
    global_patch_id = 0

    # iterate year pairs
    sample_shape = None
    for y in mask_stack:
        sample_shape = mask_stack[y].shape
        break
    if sample_shape is None:
        return pd.DataFrame([], columns=[
            "patch_id", "year", "change", "pixel_indices", "area_pixels", "island_ids", "centroid", "bbox"
        ]), {}

    for i in range(1, len(years)):
        y_prev = years[i - 1]
        y_cur = years[i]
        mask_prev = mask_stack.get(y_prev, None)
        mask_cur = mask_stack.get(y_cur, None)
        if mask_prev is None or mask_cur is None:
            labeled_maps[y_cur] = np.zeros(sample_shape, dtype=int)
            continue

        if change_type in ("loss", "both"):
            loss_mask = (mask_prev.astype(bool)) & (~mask_cur.astype(bool))
        else:
            loss_mask = np.zeros_like(mask_prev, dtype=bool)
        if change_type in ("gain", "both"):
            gain_mask = (~mask_prev.astype(bool)) & (mask_cur.astype(bool))
        else:
            gain_mask = np.zeros_like(mask_prev, dtype=bool)

        labeled_map = np.zeros(sample_shape, dtype=int)

        labeled_loss, n_loss = label(loss_mask, structure=structure)
        for lab in range(1, n_loss + 1):
            comp_mask = (labeled_loss == lab)
            area = int(comp_mask.sum())
            if area < min_area:
                continue
            pix_inds = np.nonzero(comp_mask.ravel())[0].tolist()
            rows = (np.array(pix_inds) // sample_shape[1]).astype(int)
            cols = (np.array(pix_inds) % sample_shape[1]).astype(int)
            island_ids = []
            if island_raster is not None:
                island_vals = island_raster[rows, cols]
                island_vals = island_vals[island_vals > 0]
                if island_vals.size > 0:
                    island_ids = list(np.unique(island_vals.astype(int)))
            rmin = int(rows.min()); rmax = int(rows.max()); cmin = int(cols.min()); cmax = int(cols.max())
            centroid = (float(rows.mean()), float(cols.mean()))
            for p in pix_inds:
                labeled_map.flat[p] = global_patch_id + 1
            patch_rows.append({
                "patch_id": int(global_patch_id),
                "year": int(y_cur),
                "change": "loss",
                "pixel_indices": pix_inds,
                "area_pixels": area,
                "island_ids": island_ids,
                "centroid": centroid,
                "bbox": (rmin, rmax, cmin, cmax)
            })
            global_patch_id += 1

        labeled_gain, n_gain = label(gain_mask, structure=structure)
        for lab in range(1, n_gain + 1):
            comp_mask = (labeled_gain == lab)
            area = int(comp_mask.sum())
            if area < min_area:
                continue
            pix_inds = np.nonzero(comp_mask.ravel())[0].tolist()
            rows = (np.array(pix_inds) // sample_shape[1]).astype(int)
            cols = (np.array(pix_inds) % sample_shape[1]).astype(int)
            island_ids = []
            if island_raster is not None:
                island_vals = island_raster[rows, cols]
                island_vals = island_vals[island_vals > 0]
                if island_vals.size > 0:
                    island_ids = list(np.unique(island_vals.astype(int)))
            rmin = int(rows.min()); rmax = int(rows.max()); cmin = int(cols.min()); cmax = int(cols.max())
            centroid = (float(rows.mean()), float(cols.mean()))
            for p in pix_inds:
                labeled_map.flat[p] = global_patch_id + 1
            patch_rows.append({
                "patch_id": int(global_patch_id),
                "year": int(y_cur),
                "change": "gain",
                "pixel_indices": pix_inds,
                "area_pixels": area,
                "island_ids": island_ids,
                "centroid": centroid,
                "bbox": (rmin, rmax, cmin, cmax)
            })
            global_patch_id += 1

        labeled_maps[y_cur] = labeled_map

    df_patches = pd.DataFrame(patch_rows, columns=[
        "patch_id", "year", "change", "pixel_indices", "area_pixels", "island_ids", "centroid", "bbox"
    ])
    return df_patches, labeled_maps

# ---------------------------------------------------------
# Kaplan-Meier with Greenwood standard error and pointwise CI
# ---------------------------------------------------------
def kaplan_meier_with_ci(durations: np.ndarray, events: np.ndarray, alpha: float = 0.05) -> pd.DataFrame:
    """
    Build Kaplan-Meier table with Greenwood standard error and pointwise (1-alpha) CI.
    Returns DataFrame with columns: time, survival, se, ci_lower, ci_upper
    durations : array-like (time-to-event or censor time)
    events : array-like (1 event occurred, 0 censored)
    """
    durations = np.asarray(durations, dtype=float)
    events = np.asarray(events, dtype=int)
    if durations.size == 0:
        return pd.DataFrame(columns=['time','survival','se','ci_lower','ci_upper'])

    z = norm.ppf(1 - alpha/2)

    # unique event times (sorted)
    event_times = np.sort(np.unique(durations[events == 1]))
    surv = 1.0
    cum_var = 0.0
    rows = []
    for t in event_times:
        di = int(np.sum((durations == t) & (events == 1)))        # events at time t
        yi = int(np.sum(durations >= t))                          # at risk just before t
        if yi <= 0:
            continue
        q = di / yi
        # update survival
        surv *= (1.0 - q)
        # Greenwood increment: di / (yi * (yi - di))
        if yi - di > 0:
            cum_var += di / (yi * (yi - di))
        else:
            # if yi==di then variance term is undefined; skip increment (rare)
            cum_var += 0.0
        var_S = (surv ** 2) * cum_var
        se = np.sqrt(var_S) if var_S >= 0 else 0.0
        ci_low = max(0.0, surv - z * se)
        ci_high = min(1.0, surv + z * se)
        rows.append((float(t), float(surv), float(se), float(ci_low), float(ci_high)))

    # include t=0 row
    df = pd.DataFrame(rows, columns=['time','survival','se','ci_lower','ci_upper'])
    df = pd.concat([pd.DataFrame([{'time':0.0,'survival':1.0,'se':0.0,'ci_lower':1.0,'ci_upper':1.0}]), df], ignore_index=True)
    return df.reset_index(drop=True)

def make_step_arrays(km_df: pd.DataFrame, tmax: float) -> Tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
    """
    Given km_df with columns ['time','survival','ci_lower','ci_upper'] (time increasing),
    return stepwise arrays xs, ys, ci_low_step, ci_high_step suitable for steps-post plotting and fill_between.
    """
    xs = [0.0]; ys = [1.0]; cl = [1.0]; ch = [1.0]
    for t, S, low, high in zip(km_df['time'].values, km_df['survival'].values,
                                km_df['ci_lower'].values, km_df['ci_upper'].values):
        xs.extend([t, t])
        ys.extend([ys[-1], S])
        cl.extend([cl[-1], low])
        ch.extend([ch[-1], high])
    if xs[-1] < tmax:
        xs.append(tmax); ys.append(ys[-1]); cl.append(cl[-1]); ch.append(ch[-1])
    return np.array(xs), np.array(ys), np.array(cl), np.array(ch)

# ---------------------------------------------------------
# Main function: compute and plot Zeeland A4-MLE with KM CIs
# ---------------------------------------------------------
def compute_and_plot_zeeland_A4_mle(years=None, save_csv=None, px_area_ha=None, random_seed=None,
                                    min_patch_area: int = 3, connectivity: int = 2):
    """
    Pool episodes from all islands but restrict observed events to those belonging to spatial
    patches of change >= min_patch_area (8-neighbor connectivity by default). This reduces
    influence of tiny, noisy single-pixel changes.

    Plots KM step curves with pointwise 95% Greenwood CI and overlays exponential mean + its CI.
    """
    g = globals()
    required = ('island_raster', 'id_to_island', 'marsh_archive')
    for name in required:
        if name not in g:
            raise RuntimeError(f"Required global '{name}' not found. Make sure island_raster, id_to_island and marsh_archive are available.")

    if random_seed is not None:
        np.random.seed(int(random_seed))

    # Determine years
    if years is None:
        res = g.get('res', {})
        if isinstance(res, dict) and res.get('per_island'):
            sample_iid = next(iter(res['per_island'].keys()))
            years = list(res['per_island'][sample_iid]['years'])
        else:
            years = sorted([int(y) for y in list(g.get('marsh_archive', {}).keys())])
    years = sorted([int(y) for y in years])

    # get archives and episode tables
    marsh_archive = g.get('marsh_archive', {})
    polder_archive = g.get('polder_archive', None)   # optional
    inund_archive = g.get('inund_archive', None)     # optional

    df_polder = g.get('episodes_polder_df', pd.DataFrame()).copy()
    df_inund = g.get('episodes_inund_df', pd.DataFrame()).copy()
    # build marsh per-pixel episodes if not present
    df_marsh_episodes = g.get('episodes_marsh_df', None)
    if df_marsh_episodes is None or df_marsh_episodes.empty:
        # try using existing helper
        df_marsh_episodes = extract_marsh_episodes_from_archive(marsh_archive, years, g['island_raster'])

    island_raster = g['island_raster']
    id_to_island = g['id_to_island']

    # Helper to get set of pixel indices that are part of kept patches for a cover
    def get_kept_patch_pixel_set(archive, cover_name: str):
        if archive is None:
            return pd.DataFrame(), set()  # signal no patch filtering possible
        # build mask_stack: year -> boolean mask
        mask_stack = {}
        for y in years:
            m = archive.get(int(y), None)
            if m is None:
                # create empty mask array of same shape if possible
                if mask_stack:
                    shp = next(iter(mask_stack.values())).shape
                    mask_stack[int(y)] = np.zeros(shp, dtype=bool)
                else:
                    # try to get sample shape from island raster
                    shp = island_raster.shape
                    mask_stack[int(y)] = np.zeros(shp, dtype=bool)
            else:
                mask_stack[int(y)] = m.astype(bool)
        df_patches, labeled_maps = detect_change_patches_per_year(mask_stack, years,
                                                                  change_type='loss',
                                                                  connectivity=connectivity,
                                                                  min_area=min_patch_area,
                                                                  island_raster=island_raster)
        if df_patches.empty:
            return df_patches, set()
        # keep only patches with area >= min_patch_area (detect_change already applied)
        kept_pixels = set()
        for pix_list in df_patches['pixel_indices'].values:
            # pix_list already filtered by min_area in detection
            kept_pixels.update(int(p) for p in pix_list)
        return df_patches, kept_pixels

    # detect patches for marsh (required), polder/inund optional
    _, kept_marsh_pixels = get_kept_patch_pixel_set(marsh_archive, 'marsh')
    if polder_archive is not None:
        _, kept_polder_pixels = get_kept_patch_pixel_set(polder_archive, 'polder')
    else:
        kept_polder_pixels = None
    if inund_archive is not None:
        _, kept_inund_pixels = get_kept_patch_pixel_set(inund_archive, 'inund')
    else:
        kept_inund_pixels = None

    # Helper to build durations/events arrays from episodes df after filtering by kept pixels set
    def make_arrays_from_episodes(df_eps: pd.DataFrame, kept_pixel_set):
        """
        If kept_pixel_set is None => return arrays unfiltered (use df as-is).
        If kept_pixel_set is set => include:
           - all censored episodes (event_observed==0) always
           - event episodes only if their pixel_index in kept_pixel_set
        Assumes df_eps has columns: 'duration_years', 'event_observed' and optionally 'pixel_index'.
        If 'pixel_index' not present and kept_pixel_set not None, we cannot filter and return unfiltered arrays.
        """
        if df_eps is None or df_eps.empty:
            return np.array([], dtype=float), np.array([], dtype=int)

        if kept_pixel_set is None:
            durations = np.asarray(df_eps['duration_years'].values, dtype=float)
            events = np.asarray(df_eps['event_observed'].values, dtype=int)
            return durations, events

        if 'pixel_index' not in df_eps.columns:
            # cannot filter by patch pixels; fall back to unfiltered
            durations = np.asarray(df_eps['duration_years'].values, dtype=float)
            events = np.asarray(df_eps['event_observed'].values, dtype=int)
            return durations, events

        # keep censored always
        censored_mask = (df_eps['event_observed'].astype(int) == 0)
        # event mask and pixel in kept set
        event_mask = (df_eps['event_observed'].astype(int) == 1)
        pix_vals = df_eps.loc[event_mask, 'pixel_index'].astype(int).values
        event_keep_mask = event_mask.copy()
        # build boolean array for event rows
        event_keep = np.array([ (int(p) in kept_pixel_set) for p in pix_vals ], dtype=bool)
        # map that back into full-length boolean mask
        event_idx = np.where(event_mask)[0]
        event_keep_mask[event_idx] = event_keep

        # final keep rows: censored OR event_keep_mask
        keep_rows = censored_mask | event_keep_mask
        df_sel = df_eps.loc[keep_rows].copy()
        durations = np.asarray(df_sel['duration_years'].values, dtype=float)
        events = np.asarray(df_sel['event_observed'].values, dtype=int)
        return durations, events

    # Build arrays per cover
    dur_p, ev_p = make_arrays_from_episodes(df_polder, kept_polder_pixels)
    dur_i, ev_i = make_arrays_from_episodes(df_inund, kept_inund_pixels)
    dur_m, ev_m = make_arrays_from_episodes(df_marsh_episodes, kept_marsh_pixels)

    # Compute KM tables with pointwise CIs and A4-MLEs
    km_p = kaplan_meier_with_ci(dur_p, ev_p) if dur_p.size else pd.DataFrame()
    km_i = kaplan_meier_with_ci(dur_i, ev_i) if dur_i.size else pd.DataFrame()
    km_m = kaplan_meier_with_ci(dur_m, ev_m) if dur_m.size else pd.DataFrame()

    a4_mle_p = _compute_A4_mle_from_episodes(dur_p, ev_p)
    a4_mle_i = _compute_A4_mle_from_episodes(dur_i, ev_i)
    a4_mle_m = _compute_A4_mle_from_episodes(dur_m, ev_m)  
    
    # summary values from A4-MLE tuples
    tauA_p, varA_p, ciA_p, events_p, n_p_total = a4_mle_p
    tauA_i, varA_i, ciA_i, events_i, n_i_total = a4_mle_i
    tauA_m, varA_m, ciA_m, events_m, n_m_total = a4_mle_m
    
    summary_row = {
        "region": "Zeeland",
        "n_polder": int(dur_p.size), "events_polder": int(np.sum(ev_p)),
        "A4MLE_polder_tau": float(tauA_p) if tauA_p is not None else None,
        "A4MLE_polder_ci_low": float(ciA_p[0]) if ciA_p and ciA_p[0] is not None else None,
        "A4MLE_polder_ci_high": float(ciA_p[1]) if ciA_p and ciA_p[1] is not None else None,
    
        "n_inund": int(dur_i.size), "events_inund": int(np.sum(ev_i)),
        "A4MLE_inund_tau": float(tauA_i) if tauA_i is not None else None,
        "A4MLE_inund_ci_low": float(ciA_i[0]) if ciA_i and ciA_i[0] is not None else None,
        "A4MLE_inund_ci_high": float(ciA_i[1]) if ciA_i and ciA_i[1] is not None else None,
    
        "n_marsh": int(dur_m.size), "events_marsh": int(np.sum(ev_m)),
        "A4MLE_marsh_tau": float(tauA_m) if tauA_m is not None else None,
        "A4MLE_marsh_ci_low": float(ciA_m[0]) if ciA_m and ciA_m[0] is not None else None,
        "A4MLE_marsh_ci_high": float(ciA_m[1]) if ciA_m and ciA_m[1] is not None else None,
    }
    
    df_summary_row = pd.DataFrame([summary_row])
    if save_csv:
        try:
            header = not os.path.exists(save_csv)
            df_summary_row.to_csv(save_csv, mode='a', index=False, header=header)
            print("Appended Zeeland summary to", save_csv)
        except Exception:
            df_summary_row.to_csv(save_csv, index=False)
            print("Saved Zeeland summary to", save_csv)
    
    return fig, summary_row
```
## **CCDF and manually add 1953**
The 1953 event is added manually but this must only be run once otherwise it adds multiple events. If this is done, uncomment the part in the end to remove this last entry. 
This code block calculates the CCDF based on the inundation events and sizes. We take base year = 1250 and extract the data from the globals again. It was built on the fact we have islands, so if islands=True set above it does it combines events across islands, otherwise it does not. 
```python
#%% CCDF (per-year-aggregated option) and add 1953 manually, only run this once otherwise multiple 1953 events!! 
# create per-year combined inundation totals and store in globals()['inundations_combined']
# --- build inundation_events_df_all from df_yearly ---
if 'df_yearly' not in globals():
    raise RuntimeError("df_yearly not found in globals()")

# If year is index (most likely)
if df_yearly.index.name is not None or np.issubdtype(df_yearly.index.dtype, np.integer):
    years = df_yearly.index.to_numpy()
else:
    raise RuntimeError("df_yearly index must represent year offset from 1250")

BASE_YEAR = 1250

events = []

for yr_offset, area in zip(years, df_yearly['Newly_inundated_ha'].to_numpy()):
    if not np.isfinite(area) or area <= 0:
        continue

    events.append({
        'year': int(BASE_YEAR + yr_offset),
        'area_ha': float(area),
        'island_id': 'ALL',              # aggregated over islands
        'source': 'df_yearly'
    })

inundation_events_df = pd.DataFrame(events)

# store in globals explicitly
globals()['inundation_events_df_all'] = inundation_events_df

print(
    f"Built inundation_events_df with "
    f"{len(inundation_events_df)} yearly events "
    f"({inundation_events_df['year'].min()}–"
    f"{inundation_events_df['year'].max()})"
)

def add_manual_inundation(year: int, area_ha: float, island_id=None, note="manual_add", store_key_prefer="inundation_events_df_all"):
    g = globals()

    # pick existing events key (prefer the '_all' table if present)
    if g.get(store_key_prefer) is not None:
        events_key = store_key_prefer
    elif g.get('inundation_events_df') is not None:
        events_key = 'inundation_events_df'
    else:
        # no existing events table: create a new one
        events_key = store_key_prefer

    # load or create DataFrame
    if g.get(events_key) is not None:
        events_df = pd.DataFrame(g[events_key]).copy()
    else:
        # create minimal events table if none existed
        events_df = pd.DataFrame(columns=['year', 'area_ha', 'island_id'])

    # build the new row (keep other columns untouched)
    new_row = {'year': int(year), 'area_ha': float(area_ha), 'manual_flag': note}
    if island_id is not None:
        new_row['island_id'] = island_id

    # append safely
    events_df = pd.concat([events_df, pd.DataFrame([new_row])], ignore_index=True)

    # store back to globals under the chosen key
    g[events_key] = events_df

    # rebuild per-year aggregated table (inundations_combined) from the updated events table
    df_ev = events_df[events_df['area_ha'].notna() & (events_df['area_ha'] > 0)].copy()
    df_comb = df_ev.groupby('year', as_index=False).agg(
        total_inundation_ha=('area_ha', 'sum'),
        n_events=('area_ha', 'count')
    ).sort_values('year').reset_index(drop=True)

    # preserve island counts if island_id exists
    if 'island_id' in df_ev.columns:
        n_islands = df_ev.groupby('year')['island_id'].nunique().reset_index(name='n_islands')
        df_comb = df_comb.merge(n_islands, on='year', how='left')
    else:
        df_comb['n_islands'] = np.nan

    # store aggregated table
    g['inundations_combined'] = df_comb

    # summary printout
    print(f"Appended manual event to '{events_key}': year={year}, area_ha={area_ha}, island_id={island_id}, note={note}")
    print(f"Events table rows now: {len(events_df)}")
    print(f"Aggregated per-year table 'inundations_combined' now has {len(df_comb)} rows.")
    print("Per-year total for that year:", float(df_comb.loc[df_comb['year']==year, 'total_inundation_ha']) 
          if (df_comb['year']==year).any() else "year not present (unexpected)")

    # return the new row and updated df_comb for convenience
    return new_row, df_comb

# Example usage: change year/area as needed
add_manual_inundation(1953, 38000, island_id="manual_island_1")
# IF YOU RUN THIS MORE THAN ONCE IT ADDS IT TWICE --> inundation_events_df_all = inundation_events_df_all.drop(index=[-1]) to remove last 1953 entry if needed
```
## **CCDF for different regimes ** 
This is the final code block with the CCDF for the different regimes. It calculates the CCDF per period. Each period is defined and the data is split accordingly. Labels are added to highlight which event it is and a clauset function is called again with the n_boot and min_tail setting. 
```python
#%% CCDF panels with PL overlays + KS marker + event-year annotations (per-year summed totals, recompute fits)
g = globals()

# --- 1) Build per-year combined table (one row per year = sum across islands) ---
# explicit None checks to avoid ambiguous DataFrame truth-value errors
if g.get('inundation_events_df_all') is not None:
    events_all = g['inundation_events_df_all']

if events_all is None:
    raise RuntimeError("No 'inundation_events_df_all' or 'inundation_events_df' found in globals().")

df_ev = pd.DataFrame(events_all)
df_ev = df_ev[df_ev['area_ha'].notna() & (df_ev['area_ha'] > 0)].copy()

df_comb = df_ev.groupby('year', as_index=False).agg(
    total_inundation_ha=('area_ha', 'sum'),
    n_events=('area_ha', 'count')
).sort_values('year').reset_index(drop=True)

if 'island_id' in df_ev.columns:
    n_islands = df_ev.groupby('year')['island_id'].nunique().reset_index(name='n_islands')
    df_comb = df_comb.merge(n_islands, on='year', how='left')
else:
    df_comb['n_islands'] = np.nan

# store for reuse
g['inundations_combined'] = df_comb
print(f"Built 'inundations_combined' with {len(df_comb)} years (range {df_comb['year'].min()}–{df_comb['year'].max()})")

# --- 2) Define windows and highlighted years ---
windows = [
    ('Total', '1250–1950', None, None),
    ('Regional Water Boards (waterschappen)', '1250 – 1500', 1250, 1500),
    ('Provincial Authorities (States of Zeeland)', '1500 – 1798', 1500, 1798),
    ('National Water Authority (Rijkswaterstaat)', '1798 - present', 1798, 1955),
]

# put years you want labelled here
highlight_events = [
    {'year':1808, 'label':'1808'},
    {'year':1404, 'label':"1st Sint-Elisabeth's Day Flood 1404"},
    {'year':1530, 'label':'St Felix quade saterdach 1530'},
    {'year':1421, 'label':"2nd Sint-Elisabeth's Day Flood 1421"},
    {'year':1532, 'label':"All Saints' Flood 1532"},
    {'year':1625, 'label':'1625'},
    {'year':1707, 'label':'Christmas Flood 1707'},
    {'year':1551, 'label':"Pontiaans' Flood 1551"},
    {'year':1570, 'label':"All Saints' Flood 1570"},
    {'year':1682, 'label':'1682'},
    {'year':1299, 'label':'1299'},   # this one will be placed top-right
    {'year':1334, 'label':'1334'},
    {'year':1825, 'label':'1825'},
    {'year':1906, 'label':'1906'},
    {'year':1953, 'label':'1953'},
]

# --- 3) Prepare df_for_analysis (rename summed column to area_ha) ---
df_for_analysis = df_comb.rename(columns={'total_inundation_ha': 'area_ha'})[['year', 'area_ha']].copy()

# --- 4) Recompute Clauset fits on per-year summed data (ensure consistency) ---
n_boot = 5000
min_tail = 10

print("Running Clauset fits on per-year summed data (this may take some time)...")
# analyze_windows_clauset must be available in the environment
results = analyze_windows_clauset(df_for_analysis, windows, n_boot=n_boot, min_tail=min_tail,
                                  seed=12345, annotate_axes=None, progress=True, aggregate='per_event')
print("Clauset fits completed. Summary:")
for title, _, _, _ in windows:
    r = results.get(title, {})
    print(f" {title}: n={r.get('n')}, xmin={r.get('xmin')}, alpha={r.get('alpha')}, n_tail={r.get('n_tail')}, D={r.get('D')}, p={r.get('gof_p')}")

# --- 5) Build CCDFs for plotting (from df_comb total_inundation_ha) ---
ccdfs = {}
df_sizes_by_window = {}
for title, time_label, y0, y1 in windows:
    mask = np.ones(len(df_comb), dtype=bool)
    if y0 is not None:
        mask &= (df_comb['year'] >= int(y0))
    if y1 is not None:
        mask &= (df_comb['year'] < int(y1))
    sizes = df_comb.loc[mask, 'total_inundation_ha'].to_numpy(dtype=float)
    sizes = sizes[np.isfinite(sizes) & (sizes > 0)]
    sizes.sort()
    df_sizes_by_window[title] = sizes
    n = sizes.size
    if n == 0:
        ccdfs[title] = (np.array([]), np.array([]))
    else:
        ccdfs[title] = (sizes, np.arange(n, 0, -1).astype(float) / float(n))

# --- 6) Determine plot limits ---
all_x = np.concatenate([x for x, c in ccdfs.values() if x.size > 0]) if any(x.size > 0 for x, c in ccdfs.values()) else np.array([1.0])
xmin_plot = max(min(all_x.min(), 1.0), 1e-6)
xmax_plot = max(all_x.max(), xmin_plot * 10.0)
all_ccdf_vals = np.concatenate([c for x, c in ccdfs.values() if c.size > 0]) if any(x.size > 0 for x, c in ccdfs.values()) else np.array([1.0])
ymin_plot = max(all_ccdf_vals.min(), 1e-6)
ymax_plot = 1.0

# --- 7) Plot panels, using results computed on df_for_analysis (per-year totals) ---
fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=False, sharey=False)
axes = axes.ravel()
panel_colors = ['k', 'C0', 'C4', 'C2']
x_ref, c_ref = ccdfs.get('Total', (np.array([]), np.array([])))

for j, (ax, (title, time_label, y0, y1), color) in enumerate(zip(axes, windows, panel_colors)):
    x, c = ccdfs.get(title, (np.array([]), np.array([])))

    # reference Total curve in non-total panels
    if title != 'Total' and x_ref.size > 0:
        ax.loglog(x_ref, c_ref, ls='-', lw=1.0, color='0.6', alpha=0.8, label='Total')

    # empirical CCDF (per-year totals)
    if x.size > 0:
        ax.loglog(x, c, ls='None', marker='o', ms=6, color=color, alpha=0.95, label=f'{time_label} (n={len(x)})')
    else:
        ax.text(0.5, 0.5, 'no years', transform=ax.transAxes, ha='center', va='center')

    # Clauset fit for this window
    res = results.get(title, {})
    xmin = res.get('xmin', None)
    alpha = res.get('alpha', None)
    n_tail = res.get('n_tail', 0)
    n_total = res.get('n', max(1, len(x)))
    gof_p = res.get('gof_p', None)

    if xmin is not None and alpha is not None and x.size > 0:
        ax.axvline(xmin, color='0.6', linestyle='-', linewidth=0.8, alpha=0.9)

        tail_mask = x >= xmin
        if tail_mask.any():
            ax.plot(x[tail_mask], c[tail_mask], ls='None', marker='s', ms=6, color='orange',
                    markeredgecolor='k', markeredgewidth=0.3, alpha=0.95, label='tail data')

        x_max_for_model = x.max()
        if x_max_for_model > xmin:
            tvals = np.logspace(np.log10(max(xmin, 1e-12)), np.log10(x_max_for_model), 400)
            model_ccdf = (float(n_tail) / float(max(1, n_total))) * (tvals / float(xmin)) ** (1.0 - float(alpha))
            ax.plot(tvals, model_ccdf, color='orange', ls='--', lw=1.6, alpha=0.95, label='Power-Law fit')

        # values box top-left (unchanged)
        ks_display = res.get('D', np.nan)
        txt = (f"xmin={xmin:.1f}\nα={alpha:.2f}\nn_tail={n_tail}\nKS={ks_display:.3f}\np={gof_p:.3f}"
               if gof_p is not None else f"xmin={xmin:.1f}\nα={alpha:.2f}\nn_tail={n_tail}\nKS={ks_display:.3f}")
        ax.text(0.05, 0.4, txt, transform=ax.transAxes, ha='left', va='bottom',
                fontsize=14, bbox=dict(facecolor='white', alpha=0.75, edgecolor='none'))

    # --- Annotate highlighted years (labels placed lower-left by default, but 1299 top-right) ---
    sizes_array = df_sizes_by_window[title]
    n_panel = sizes_array.size
    if n_panel > 0 and highlight_events:
        # canonicalize highlight list
        canonical = []
        for h in highlight_events:
            if isinstance(h, dict) and 'year' in h:
                dd = h.copy()
                dd.setdefault('label', str(dd['year']))
                dd.setdefault('offset', None)
                canonical.append(dd)
            else:
                try:
                    yr = int(h)
                    canonical.append({'year': yr, 'label': str(yr), 'offset': None})
                except Exception:
                    continue

        annots = []
        for ev in canonical:
            year = int(ev['year'])
            if y0 is not None and year < y0:
                continue
            if y1 is not None and year >= y1:
                continue
            row = df_comb.loc[df_comb['year'] == year]
            if row.empty:
                continue
            s_val = float(row['total_inundation_ha'].iloc[0])
            c_emp = float(np.sum(sizes_array >= s_val) / max(1, n_panel))
            # Per-label offsets: default lower-left (factors < 1)
            if ev.get('offset') is None:
                # special-case year 1299: place top-right initially
                if year == 1299:
                    xfact, yfact = 1.4, 1.4   # top-right ( > 1 )
                else:
                    xfact, yfact = 0.7, 0.7   # lower-left ( < 1 )
            else:
                try:
                    xfact, yfact = float(ev['offset'][0]), float(ev['offset'][1])
                except Exception:
                    xfact, yfact = (1.4, 1.4) if year == 1299 else (0.7, 0.7)
            tx = s_val * xfact
            ty = c_emp * yfact
            annots.append({'year': year, 'x': s_val, 'y': c_emp, 'tx': tx, 'ty': ty, 'label': ev['label']})

        if annots:
            text_artists = []
            for a in annots:
                # default: lower-left -> ha='right', va='top'
                if a['year'] == 1299:
                    ha = 'left'; va = 'bottom'    # top-right placement relative to anchor
                    prefer_up = True
                else:
                    ha = 'right'; va = 'top'      # lower-left placement relative to anchor
                    prefer_up = False

                t = ax.text(a['tx'], a['ty'], a['label'],
                            fontsize=11, color='black',
                            bbox=dict(facecolor='white', alpha=0.9, edgecolor='none'),
                            horizontalalignment=ha, verticalalignment=va, zorder=25)
                text_artists.append({'artist': t, 'anchor_x': a['x'], 'anchor_y': a['y'], 'prefer_up': prefer_up})

            # de-overlap in display coords: prefer downward shifts for defaults, upward for 1299
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            bboxes = [ta['artist'].get_window_extent(renderer=renderer) for ta in text_artists]

            assigned_bboxes = []
            max_layers = 40
            pad = 3
            for ta, bb in zip(text_artists, bboxes):
                overlap = any(bb.overlaps(ab) for ab in assigned_bboxes)
                if not overlap:
                    assigned_bboxes.append(bb)
                    continue
                placed = False
                signs_order = (-1, 1) if not ta.get('prefer_up', False) else (1, -1)
                for layer in range(1, max_layers + 1):
                    for sign in signs_order:
                        dy = sign * layer * (bb.height + pad)
                        bb_candidate = bb.translated(0, dy)
                        if not any(bb_candidate.overlaps(ab) for ab in assigned_bboxes):
                            cur_pos_disp = ax.transData.transform(ta['artist'].get_position())
                            new_pos_disp = (cur_pos_disp[0], cur_pos_disp[1] + dy)
                            new_data_pos = ax.transData.inverted().transform(new_pos_disp)
                            ta['artist'].set_position((new_data_pos[0], new_data_pos[1]))
                            fig.canvas.draw()
                            renderer = fig.canvas.get_renderer()
                            new_bb = ta['artist'].get_window_extent(renderer=renderer)
                            assigned_bboxes.append(new_bb)
                            placed = True
                            break
                    if placed:
                        break
                if not placed:
                    assigned_bboxes.append(bb)

            # draw markers and arrows
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            for ta in text_artists:
                artist = ta['artist']
                lbl = artist.get_text()
                is_1953 = (lbl == '1953')
                anchor_x = ta['anchor_x']
                anchor_y = ta['anchor_y']

                ax.plot(
                    anchor_x, anchor_y,
                    marker='*',
                    ms=16 if is_1953 else 10,
                    color='magenta' if is_1953 else 'red',
                    markeredgecolor='k',
                    markeredgewidth=0.9 if is_1953 else 0.8,
                    zorder=35 if is_1953 else 30
                )
                txt_x, txt_y = artist.get_position()
                ax.annotate('', xy=(anchor_x, anchor_y), xytext=(txt_x, txt_y),
                            arrowprops=dict(arrowstyle='->', color='red', lw=0.9, shrinkA=0, shrinkB=0, mutation_scale=18),
                            zorder=29)

    # axes scales, grid, labels
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlim(xmin_plot * 0.8, xmax_plot * 1.2)
    ax.set_ylim(ymin_plot * 0.8, ymax_plot * 1.05)
    ax.grid(which='both', alpha=0.25)
    ax.set_title(title, fontsize = 18)
    if j % 2 == 0:
        ax.set_ylabel('P(X ≥ x)', fontsize=16)
    if j >= 2:
        ax.set_xlabel('Total Inundation (ha)', fontsize=18)
    if j % 2 == 1:
        ax.tick_params(axis='y', which='both', labelleft=False)
    else:
        ax.tick_params(axis='y', which='both', labelleft=True)
    if j < 2:
        ax.tick_params(axis='x', which='both', labelbottom=False)
    else:
        ax.tick_params(axis='x', which='both', labelbottom=True)

    # legend dedupe
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        seen = set(); new_h = []; new_l = []
        for h, lab in zip(handles, labels):
            if lab not in seen:
                seen.add(lab); new_h.append(h); new_l.append(lab)
        ax.legend(new_h, new_l, fontsize=14, loc='lower left', frameon=True, framealpha=0.9)

plt.suptitle("CCDF of Inundation Events Zeeland", y=0.97, fontsize=24)
plt.tight_layout(rect=[0.03, 0.03, 1.0, 0.95])

# save results JSON
try:
    with open("clauset_results_zeeland.json", "w") as fh:
        json.dump(results, fh, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
    print("Saved Clauset results to clauset_results_zeeland.json")
except Exception as e:
    print("Could not save Clauset results JSON:", e)
plt.show()
``` 
## **Authors**
- Michiel van Dijk

