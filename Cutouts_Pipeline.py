import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from astropy.visualization import simple_norm
from astropy.io import fits
from scipy.ndimage import uniform_filter
import sep
import pandas as pd
import gelsa
from astropy.convolution import Gaussian2DKernel
import glob
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("fits_path", help="Path to the input FITS file")
parser.add_argument(
    "--det-code",
    default=None,
    help="Two-character detector code to process, e.g. '11' or '24'. "
         "If omitted, all detectors in the file are processed.",
)
parser.add_argument(
    "--force",
    action="store_true",
    help="Reprocess a detector even if its detections CSV already exists.",
)
args = parser.parse_args()
fits_path = args.fits_path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CALIB_DIR = os.path.join(SCRIPT_DIR, "Official-Roman-Artifact-Detection")
G_nb = gelsa.Gelsa(
    os.path.join(CALIB_DIR, "calib/gelsa_config.json"),
    workdir=CALIB_DIR,
    zero_order_catalog=None,
)
gelsa_frame = G_nb.load_spec_frame(os.path.abspath(fits_path))

tilt = gelsa_frame.params['tilt']
print(f"Grism tilt: {tilt}°")


CUTOUT_DIR = 'cutouts'
os.makedirs(CUTOUT_DIR, exist_ok=True)


# ── Detection parameters  ───────────────────
COLD_KERNEL       = Gaussian2DKernel(x_stddev=3, y_stddev=1).array
COLD_MINAREA      = 50
COLD_DEBLEND      = 0.05

HOT_KERNEL        = Gaussian2DKernel(0.7).array
HOT_MINAREA       = 15
HOT_DEBLEND       = 0.05
HOT_NTHRESH       = 64

DARK_MIN_AREA     = 140  # area threshold for merged dark sources

MIN_HOT_IN_COLD   = 2    # >= this many hot in a cold footprint → cold wins
thresh            = 0.7
sep.set_sub_object_limit(4096)

# ─────────────────── Cutout Parameters ───────────────────
SCALE_COMPACT    = 2.0
SCALE_ELONGATED  = 1.2
ELONGATED_THRESH = 3.0
TINY_MAX_DIM     = 12
BBOX_PAD         = 5

# ── ZO matching parameters (from main.py) ─────────────────────────────────
ZO_MIN_DST_THRESHOLD = 0.75
ZO_H_SCALE           = 5.0
ZO_W_SCALE           = 20.0

BIGDIGEST_PATH = os.path.join(
    SCRIPT_DIR,
    "Official-Roman-Artifact-Detection",
    "Euclid Extracted Spectra Data",
    "bigdigest 5614B8BE632B859E72E37665FB9914E3.csv",
)
ZO_MIN_BRIGHTNESS_MAG = 20

zo_table = pd.read_csv(BIGDIGEST_PATH)
zo_table = zo_table[zo_table['Magnitude'] < ZO_MIN_BRIGHTNESS_MAG].reset_index(drop=True)
print(f"Bright ZO sources across all pointings: {len(zo_table):,}")

# ── Project RA/Dec → detector pixel coords via gelsa ──────────────────────
zo_x, zo_y, zo_det = gelsa_frame.radec_to_pixel(
    zo_table['RIGHT_ASCENSION'].values,
    zo_table['DECLINATION'].values,
    15000 * np.ones(len(zo_table)),
    dispersion_order=0
)

class DetIndex:
    def __init__(self, thing):
        if type(thing) is str:
            self.code = thing
            self.idx = (int(self.code[1]) - 1) * 4 + (int(self.code[0]) - 1)
        elif type(thing) is int:
            self.idx = thing
            self.code = f"{self.idx % 4 + 1}{self.idx // 4 + 1}"
        else:
            raise Exception("Expected a 2-char string code or an int index")

DetIndex.ALL_DETS = [DetIndex(i) for i in range(16)]

def get_grism_angle(header):
    return header["GWA_TILT"] + float(header["GWA_POS"][-3:])

# ── Inpaint hot pixels ─────────────────────────────────────────────────────
def inpaint_hot_pixels(image, hot_mask, box_size=64, min_good_fraction=0.3):
    img  = image.copy().astype(np.float64)
    good = ~hot_mask & np.isfinite(image)
    good_vals    = img[good]
    global_bg    = float(np.median(good_vals)) if good_vals.size else 0.0
    global_mad   = float(np.median(np.abs(good_vals - global_bg))) if good_vals.size else 1.0
    global_sigma = max(global_mad * 1.4826, 1e-6)
    gf    = good.astype(np.float64)
    clean = np.where(good, img, global_bg)
    cnt   = uniform_filter(gf,             size=box_size, mode='reflect')
    bg    = uniform_filter(clean * gf,     size=box_size, mode='reflect')
    sq    = uniform_filter(clean**2 * gf,  size=box_size, mode='reflect')
    safe  = cnt > 0
    bg    = np.where(safe, bg / (cnt + 1e-30), global_bg)
    lvar  = np.where(safe, sq / (cnt + 1e-30) - bg**2, global_sigma**2)
    sigma = np.sqrt(np.abs(lvar))
    use_local = cnt >= min_good_fraction * (box_size ** 2)
    bg    = np.where(use_local, bg,    global_bg)
    sigma = np.where(use_local, sigma, global_sigma)
    rng = np.random.default_rng(42)
    img[hot_mask] = bg[hot_mask] + sigma[hot_mask] * rng.normal(0.0, 1.0, img.shape)[hot_mask]
    return img

# ──  collapse a group of dicts into one merged box ─────────────────
def _merge_group(group, group_id):
    minc = min(o['bbox'][0] for o in group)
    minr = min(o['bbox'][1] for o in group)
    maxc = max(o['bbox'][2] for o in group)
    maxr = max(o['bbox'][3] for o in group)
    total_area = sum(o['area'] for o in group) or 1
    cx     = sum(o['centroid'][0] * o['area'] for o in group) / total_area
    cy     = sum(o['centroid'][1] * o['area'] for o in group) / total_area
    w, h   = maxc - minc, maxr - minr
    bbox_a = w * h
    return {
        'bbox'          : [minc, minr, maxc, maxr],
        'centroid'      : [cx, cy],
        'area'          : total_area,
        'width'         : w,
        'height'        : h,
        'aspect_ratio'  : w / h if h > 0 else float('inf'),
        'fill_ratio'    : total_area / bbox_a if bbox_a > 0 else 0.0,
        'label'         : group_id,
        'fragment_count': len(group),
        'sub_centroids' : [o['centroid'] for o in group],
    }


# ── Union-Find merger ──────────────────────────────────────────────────────
def merge_objects(object_list, h_proximity=15, v_proximity=5,
                  overlap_threshold=0.1, horizontal_overlap_threshold=0.3,
                  aspect_ratio_tolerance=4,
                  elongated_thresh=3.0, compact_proximity_scale=1.0):
    """
    elongated_thresh        — aspect ratio above which an object is 'elongated'
                              (spectrum fragment); elongated pairs use the full
                              h_proximity / v_proximity.
    compact_proximity_scale — proximity multiplier for compact-compact pairs
                              (round ZO sources); < 1.0 makes them merge less.
    """
    if not object_list:
        return []
    n      = len(object_list)
    parent = list(range(n))
    rank   = [0] * n

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx == ry: return
        if rank[rx] < rank[ry]: parent[rx] = ry
        elif rank[rx] > rank[ry]: parent[ry] = rx
        else: parent[ry] = rx; rank[rx] += 1

    bboxes   = np.array([o['bbox']         for o in object_list])
    aspects  = np.array([o['aspect_ratio'] for o in object_list])
    sort_idx = np.argsort(bboxes[:, 0])

    for i in range(n):
        ii = sort_idx[i]
        minc_i, minr_i, maxc_i, maxr_i = bboxes[ii]
        w_i, h_i = maxc_i - minc_i, maxr_i - minr_i
        for j in range(i + 1, n):
            jj = sort_idx[j]
            minc_j, minr_j, maxc_j, maxr_j = bboxes[jj]
            # early exit uses full h_proximity so we never skip elongated pairs
            if minc_j > maxc_i + h_proximity + w_i:
                break
            w_j, h_j = maxc_j - minc_j, maxr_j - minr_j
            ar_i, ar_j = aspects[ii], aspects[jj]
            if ar_i > 1e-6 and ar_j > 1e-6:
                if max(ar_i, ar_j) / min(ar_i, ar_j) > aspect_ratio_tolerance:
                    continue

            # Adaptive proximity: elongated pairs (horizontal OR vertical)
            # get the full budget; compact-compact pairs get a smaller one
            # so round ZO sources don't over-merge.
            # Use max(ar, 1/ar) so vertical artifacts (ar << 1) are caught.
            # Swap h/v proximity for vertical objects so the large budget
            # is applied in the direction the artifact actually extends.
            elongation_i = max(ar_i, 1/ar_i) if ar_i > 1e-6 else 1.0
            elongation_j = max(ar_j, 1/ar_j) if ar_j > 1e-6 else 1.0
            max_elong = max(elongation_i, elongation_j)
            if max_elong >= elongated_thresh:
                # Determine orientation from the more elongated of the two
                ref_ar = ar_i if elongation_i >= elongation_j else ar_j
                if ref_ar < 1.0:  # vertical artifact: large budget in y
                    eff_h_prox = v_proximity
                    eff_v_prox = h_proximity
                else:             # horizontal artifact: large budget in x
                    eff_h_prox = h_proximity
                    eff_v_prox = v_proximity
            else:
                eff_h_prox = h_proximity * compact_proximity_scale
                eff_v_prox = v_proximity * compact_proximity_scale

            ic_min, ic_max = max(minc_i, minc_j), min(maxc_i, maxc_j)
            ir_min, ir_max = max(minr_i, minr_j), min(maxr_i, maxr_j)
            has_overlap = ic_max > ic_min and ir_max > ir_min
            overlap_merge = False
            if has_overlap:
                min_w, min_h = min(w_i, w_j), min(h_i, h_j)
                hor = (ic_max - ic_min) / min_w if min_w > 0 else 0
                ver = (ir_max - ir_min) / min_h if min_h > 0 else 0
                overlap_merge = hor >= horizontal_overlap_threshold and ver >= overlap_threshold
            h_dist = max(0, minc_j - maxc_i) if maxc_i < minc_j else max(0, minc_i - maxc_j)
            v_dist = max(0, minr_j - maxr_i) if maxr_i < minr_j else max(0, minr_i - maxr_j)
            proximity_merge = h_dist <= eff_h_prox and v_dist <= eff_v_prox
            if overlap_merge or proximity_merge:
                union(ii, jj)

    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(object_list[i])
    return [_merge_group(grp, gid) for gid, grp in enumerate(groups.values())]


# ── Convert SEP structured array → list of dicts ──────────────────────────
def sep_to_objlist(objs):
    result = []
    for obj in objs:
        w = float(obj['xmax'] - obj['xmin'])
        h = float(obj['ymax'] - obj['ymin'])
        bbox_a = w * h
        npix   = float(obj['npix'])
        result.append({
            'bbox'        : [float(obj['xmin']), float(obj['ymin']),
                             float(obj['xmax']), float(obj['ymax'])],
            'centroid'    : [float(obj['x']), float(obj['y'])],
            'area'        : npix,
            'width'       : w,
            'height'      : h,
            'aspect_ratio': w / h if h > 0 else float('inf'),
            'fill_ratio'  : npix / bbox_a if bbox_a > 0 else 0.0,
            'label'       : 0,
        })
    return result


def bbox_of(obj):
    """Return (xmin, ymin, xmax, ymax) from either a SEP row or a merged dict."""
    if hasattr(obj, 'dtype'):   # SEP structured array element
        return obj['xmin'], obj['ymin'], obj['xmax'], obj['ymax']
    return obj['bbox']          # merged dict


def make_cutout(box, image):
    """Square cutout for all sources. Compact: SCALE_COMPACT × longest side.
    Elongated (aspect > ELONGATED_THRESH): SCALE_ELONGATED × longest side."""
    cx  = (box['xmin'] + box['xmax']) // 2
    cy  = (box['ymin'] + box['ymax']) // 2
    bw  = box['xmax'] - box['xmin']
    bh  = box['ymax'] - box['ymin']
    aspect = bw / bh if bh > 0 else float('inf')
    scale  = SCALE_ELONGATED if aspect > ELONGATED_THRESH else SCALE_COMPACT
    half   = max(int(scale * max(bw, bh)) // 2, 20)
    side   = 2 * half
    x0 = np.clip(cx - half, 0, image.shape[1] - side)
    y0 = np.clip(cy - half, 0, image.shape[0] - side)
    cutout = image[y0:y0 + side, x0:x0 + side].copy()
    h, w = cutout.shape
    lxmin = np.clip(box['xmin'] - x0, 0, w)
    lymin = np.clip(box['ymin'] - y0, 0, h)
    lxmax = np.clip(box['xmax'] - x0, 0, w)
    lymax = np.clip(box['ymax'] - y0, 0, h)
    return cutout, lxmin, lymin, lxmax, lymax, x0, y0


parts = os.path.basename(fits_path).split('_')
fits_id = '_'.join(parts[4:7])

detectors = {}


with fits.open(fits_path, memmap=False) as hdul:
    grism_angle = get_grism_angle(hdul[0].header)
    print(f"Grism angle: {grism_angle:.1f} deg")
    det_indices = [DetIndex(h.name[3:-4]) for h in hdul if h.name.endswith('.SCI')]
    print(f"Found {len(det_indices)} detectors: {[d.code for d in det_indices]}")

    for det in det_indices:
        sci_name = f"DET{det.code}.SCI"
        var_name = f"DET{det.code}.VAR"
        dq_name  = f"DET{det.code}.DQ"

        all_names = [h.name for h in hdul]

        sci_data   = hdul[sci_name].data.astype(np.float32)
        sci_header = hdul[sci_name].header

        var_data = hdul[var_name].data.astype(np.float32) if var_name in all_names else None
        if var_name not in all_names:
            print(f"  WARNING: '{var_name}' not found")

        if dq_name in all_names:
            dq_raw  = hdul[dq_name].data.astype(np.int64)
            dq_data = np.where(dq_raw < 0, dq_raw + 2**32, dq_raw).astype(np.uint32)
        else:
            print(f"  WARNING: '{dq_name}' not found")
            dq_data = None

        detectors[det.code] = {
            'sci'   : sci_data,
            'var'   : var_data,
            'dq'    : dq_data,
            'header': sci_header,
            'index' : det.idx,
        }

print(f"\nLoaded {len(detectors)} detectors. Keys in each entry: {list(next(iter(detectors.values())).keys())}")

det_codes_to_process = [args.det_code] if args.det_code is not None else list(detectors.keys())
print(f"Processing detectors: {det_codes_to_process}")

for DET_CODE in det_codes_to_process:
    out_csv = f"detections_{fits_id}_DET{DET_CODE}.csv"
    if os.path.exists(out_csv) and not args.force:
        print(f"Skipping DET{DET_CODE} ({out_csv} already exists; use --force to reprocess)")
        continue

    for f in glob.glob(os.path.join(CUTOUT_DIR, f"cutout_{fits_id}_DET{DET_CODE}_*.npy")):
        os.remove(f)

    det = detectors[DET_CODE]

    sci = det['sci'].astype(np.float64)
    var = det['var'].astype(np.float64)
    dq  = det['dq']
    
    # ── Hot pixel mask from DQ ─────────────────────────────────────────────────
    hot_mask = (dq > 0)
    print(f"DET{DET_CODE}: {hot_mask.sum():,} hot pixels ({100*hot_mask.mean():.3f}%)")

    sci_masked = sci.copy()
    sci_masked[hot_mask] = np.nan

    # ── Continuum subtraction via gelsa ────────────────────────────────────────


    contsub_image, cs_invalid = gelsa_frame.median_filter(
        sci_masked.copy(), filter_size_pix=40, tilt=tilt
    )
    print("Continuum subtraction complete.")

    

    cs_inpainted = inpaint_hot_pixels(contsub_image, hot_mask)
    print("Inpainting complete.")

    # ── Prepare SEP inputs ─────────────────────────────────────────────────────
    img_sep = np.ascontiguousarray(cs_inpainted, dtype=np.float64)
    var_sep = np.ascontiguousarray(var,          dtype=np.float64)

    
        
    # ── Background estimate ────────────────────────────────────────────────────
    bkg = sep.Background(img_sep)

    # ── Run cold + hot passes ──────────────────────────────────────────────────
    cold_objs = sep.extract(
        img_sep, thresh=thresh, var=var_sep,
        filter_type='conv', filter_kernel=COLD_KERNEL,
        deblend_cont=COLD_DEBLEND, minarea=COLD_MINAREA,
    )
    hot_objs = sep.extract(
        img_sep, thresh=thresh, var=var_sep,
        filter_type='conv', filter_kernel=HOT_KERNEL,
        deblend_cont=HOT_DEBLEND, deblend_nthresh=HOT_NTHRESH,
        minarea=HOT_MINAREA,
    )
    dark_objs = sep.extract(
        -img_sep, thresh=thresh, var=var_sep,
        filter_type='conv', filter_kernel=COLD_KERNEL,
        deblend_cont=COLD_DEBLEND, minarea=COLD_MINAREA,
    )
    hot_objs  = hot_objs[hot_objs['npix']   >= 15]
    cold_objs = cold_objs[cold_objs['npix']  >= 15]
    dark_objs = dark_objs[dark_objs['npix']  >= 15]
    print(f"Cold: {len(cold_objs)} sources   Hot: {len(hot_objs)} sources   Dark: {len(dark_objs)} sources")

    # ── Bounding-box overlap matrix (n_cold × n_hot) ──────────────────────────
    x_overlap    = (cold_objs['xmin'][:, None] < hot_objs['xmax'][None, :]) & \
                (cold_objs['xmax'][:, None] > hot_objs['xmin'][None, :])
    y_overlap    = (cold_objs['ymin'][:, None] < hot_objs['ymax'][None, :]) & \
                (cold_objs['ymax'][:, None] > hot_objs['ymin'][None, :])
    bbox_overlap = x_overlap & y_overlap

    n_hot_per_cold = bbox_overlap.sum(axis=1)
    overlaps_hot   = n_hot_per_cold > 0



    # ── Keep/suppress logic ───────────────────────────────────────────────────
    cold_keep         = ~overlaps_hot | (n_hot_per_cold >= MIN_HOT_IN_COLD)
    cold_final        = cold_objs[cold_keep]
    oversplit_mask    = n_hot_per_cold >= MIN_HOT_IN_COLD
    suppress_hot_mask = bbox_overlap[oversplit_mask].any(axis=0)
    hot_final         = hot_objs[~suppress_hot_mask]

    print(f"Cold kept:    {int(cold_keep.sum()):4d}  "
        f"({int(oversplit_mask.sum())} over-split, {int((~overlaps_hot).sum())} no-hot-overlap)")
    print(f"Cold removed: {int((~cold_keep).sum()):4d}")
    print(f"Hot kept:     {len(hot_final):4d}  ({int(suppress_hot_mask.sum())} suppressed)")
    print(f"Combined:     {len(cold_final) + len(hot_final):4d} sources")
    # ── Plot combined result ──────────────────────────────────────────────────
    pad = 5
    
    hot_dicts   = sep_to_objlist(hot_final)
    merged_hot  = merge_objects(hot_dicts,  h_proximity=30, v_proximity=10,
                                elongated_thresh=3.0, compact_proximity_scale=1)
    print(f"Hot before merge:  {len(hot_final):4d}   after: {len(merged_hot)}")

    cold_dicts  = sep_to_objlist(cold_final)
    merged_cold = merge_objects(cold_dicts, h_proximity=30, v_proximity=10,
                                elongated_thresh=3.0, compact_proximity_scale=1)
    print(f"Cold before merge: {len(cold_final):4d}   after: {len(merged_cold)}")

    dark_dicts  = sep_to_objlist(dark_objs)
    merged_dark = merge_objects(dark_dicts, h_proximity=30, v_proximity=10,
                                elongated_thresh=3.0, compact_proximity_scale=1)
    merged_dark = [o for o in merged_dark if o['area'] > DARK_MIN_AREA]
    print(f"Dark before merge: {len(dark_objs):4d}   after: {len(merged_dark)} (area > {DARK_MIN_AREA})")

    all_merged = merged_cold + merged_hot + merged_dark

    norm_vis = simple_norm(cs_inpainted, stretch='asinh', min_percent=1, max_percent=99)

    # Write the array straight to a raster via imsave — no Figure/Axes, no dpi,
    # no bbox cropping. imshow()+savefig() still resamples through Agg even at
    # nominal 1:1 scale (verified: a single bright pixel bleeds across 2 output
    # rows even with interpolation='none'), so imsave is the only path that's
    # actually pixel-exact: output is exactly cs_inpainted.shape, one raster
    # pixel per array element, no antialiasing. origin='upper' keeps raster
    # row 0 == array row 0, matching the unflipped array-index convention used
    # everywhere downstream (SAM cutout display, sam_x0/y0, cutout_x0/y0, mask
    # arrays) — coordinates paste directly, no transform needed on reload.
    rgba = plt.cm.gray(norm_vis(cs_inpainted))

    field = fits_id.split('_')[0]
    noboxes_dir = os.path.join(SCRIPT_DIR, field, DET_CODE)
    os.makedirs(noboxes_dir, exist_ok=True)
    noboxes_path = os.path.join(noboxes_dir, f"{fits_id}_DET{DET_CODE}_noboxes.png")
    plt.imsave(noboxes_path, rgba, origin='upper')

    # Draw detection boxes (same pixel coordinates that go into the detections CSV)
    # directly in pixel space with PIL — also no resampling — onto a second,
    # boxed copy at the same exact detector-pixel dimensions.
    boxed = Image.open(noboxes_path).convert('RGB')
    draw = ImageDraw.Draw(boxed)
    for obj in all_merged:
        xmin, ymin, xmax, ymax = bbox_of(obj)
        draw.rectangle(
            [xmin - pad, ymin - pad, xmax + pad, ymax + pad],
            outline=(65, 105, 225),  # royalblue
            width=1,
        )
    boxed.save(os.path.join(noboxes_dir, f"{fits_id}_DET{DET_CODE}.png"))

    # ---------- Cutouts ----------

    dark_bbox_set = {tuple(int(v) for v in o['bbox']) for o in merged_dark}
    all_boxes = [
        {'xmin': int(o['bbox'][0]), 'ymin': int(o['bbox'][1]),
         'xmax': int(o['bbox'][2]), 'ymax': int(o['bbox'][3]),
         'source_type': 'dark' if tuple(int(v) for v in o['bbox']) in dark_bbox_set else 'bright',
         'centers': o['sub_centroids']}
        for o in all_merged
    ]

    normal_boxes = [b for b in all_boxes
                if max(b['xmax'] - b['xmin'], b['ymax'] - b['ymin']) >= TINY_MAX_DIM]
    tiny_boxes   = [b for b in all_boxes
                    if max(b['xmax'] - b['xmin'], b['ymax'] - b['ymin']) <  TINY_MAX_DIM]
    print(f"Normal: {len(normal_boxes)}   Tiny (< {TINY_MAX_DIM} px): {len(tiny_boxes)}")


    cutouts            = [make_cutout(b, cs_inpainted) for b in normal_boxes]
    cutouts_precontsub = [make_cutout(b, sci_masked)   for b in normal_boxes]

    for box, (cutout, lxmin, lymin, lxmax, lymax, crop_x0, crop_y0), (precontsub, *_) in zip(normal_boxes, cutouts, cutouts_precontsub):
        x0, y0 = box['xmin'], box['ymin']
        x1, y1 = box['xmax'], box['ymax']
        h_co, w_co = cutout.shape
        lxmin_pad = max(0,    lxmin - BBOX_PAD)
        lymin_pad = max(0,    lymin - BBOX_PAD)
        lxmax_pad = min(w_co, lxmax + BBOX_PAD)
        lymax_pad = min(h_co, lymax + BBOX_PAD)
        stem = f"cutout_{fits_id}_DET{DET_CODE}_{x0}_{x1}_{y0}_{y1}"
        np.save(os.path.join(CUTOUT_DIR, f"{stem}.npy"),            cutout.astype(np.float32))
        np.save(os.path.join(CUTOUT_DIR, f"{stem}_precontsub.npy"), precontsub.astype(np.float32))
        np.save(os.path.join(CUTOUT_DIR, f"{stem}_bbox.npy"),
                np.array([lxmin_pad, lymin_pad, lxmax_pad, lymax_pad], dtype=np.float32))
        local_centers = np.array(
            [[cx - crop_x0, cy - crop_y0] for cx, cy in box['centers']], dtype=np.float32)
        np.save(os.path.join(CUTOUT_DIR, f"{stem}_points.npy"), local_centers)
        if box.get('source_type') == 'dark':
            np.save(os.path.join(CUTOUT_DIR, f"{stem}_dark.npy"), np.array(True))


    on_frame = zo_det >= 0
    print(f"On this FITS frame: {on_frame.sum():,} / {len(zo_table):,}")

    det_idx = DetIndex(DET_CODE).idx
    on_this_det = (zo_det == det_idx)
    print(f"On DET{DET_CODE}: {on_this_det.sum()}")

    zo_x_det = zo_x[on_this_det]
    zo_y_det = zo_y[on_this_det]
    if len(zo_x_det):
        print(f"  x range: {zo_x_det.min():.1f} – {zo_x_det.max():.1f}")
        print(f"  y range: {zo_y_det.min():.1f} – {zo_y_det.max():.1f}")

    # Rotation + scale matrix (same as main.py process_detector)
    tilt_rad = np.deg2rad(grism_angle)
    A_inv = np.linalg.inv(np.array([
        [ZO_W_SCALE * np.cos(tilt_rad), -ZO_H_SCALE * np.sin(tilt_rad)],
        [ZO_W_SCALE * np.sin(tilt_rad),  ZO_H_SCALE * np.cos(tilt_rad)]
    ]))

    zpos = np.column_stack((zo_x_det, zo_y_det)) if len(zo_x_det) else np.empty((0, 2))

    # ── Match each box against the ZO catalog ─────────────────────────────────
    zo_matched = []
    unmatched  = []

    for obj in all_merged:
        cx, cy = obj['centroid']
        if len(zpos) == 0:
            unmatched.append(obj)
            continue
        delta = zpos - np.array([cx, cy])           # (N, 2) offsets to all ZO sources
        transformed = (A_inv @ delta.T).T           # apply elliptical scaling + rotation
        dists = np.linalg.norm(transformed, axis=1)
        if dists.min() < ZO_MIN_DST_THRESHOLD:
            zo_matched.append(obj)
        else:
            unmatched.append(obj)

    print(f"ZO matched: {len(zo_matched)}   Unmatched: {len(unmatched)}")

    # ── Build detection CSV ────────────────────────────────────────────────────
    # All coordinates are in the individual detector's own pixel space.
    #
    # Columns:
    #   source_fits                   — original FITS basename (trace back to main file + detector)
    #   det_code                      — two-char detector code, e.g. "11"
    #   det_xmin/ymin/xmax/ymax/cx/cy — merged bbox in detector pixel coords
    #   cutout_x0/y0/x1/y1/cx/cy     — cutout window in detector pixel coords
    #   local_xmin/ymin/xmax/ymax/cx/cy — bbox relative to cutout origin
    #   zeroth_order                  — 1 if matched to ZO catalog, 0 otherwise

    zo_matched_bboxes = {tuple(int(v) for v in o['bbox']) for o in zo_matched}

    fits_basename = os.path.basename(fits_path)
    rows = []

    for obj in all_merged:
        xmin, ymin, xmax, ymax = [int(v) for v in obj['bbox']]
        if max(xmax - xmin, ymax - ymin) < TINY_MAX_DIM:
            continue

        cx, cy   = (xmin + xmax) // 2, (ymin + ymax) // 2
        bw, bh   = xmax - xmin, ymax - ymin
        aspect   = bw / bh if bh > 0 else float('inf')
        scale    = SCALE_ELONGATED if aspect > ELONGATED_THRESH else SCALE_COMPACT
        half     = max(int(scale * max(bw, bh)) // 2, 20)
        side     = 2 * half
        x0 = int(np.clip(cx - half, 0, cs_inpainted.shape[1] - side))
        y0 = int(np.clip(cy - half, 0, cs_inpainted.shape[0] - side))

        lxmin = int(np.clip(xmin - x0, 0, side))
        lymin = int(np.clip(ymin - y0, 0, side))
        lxmax = int(np.clip(xmax - x0, 0, side))
        lymax = int(np.clip(ymax - y0, 0, side))
        lxmin_pad = max(0,    lxmin - BBOX_PAD)
        lymin_pad = max(0,    lymin - BBOX_PAD)
        lxmax_pad = min(side, lxmax + BBOX_PAD)
        lymax_pad = min(side, lymax + BBOX_PAD)

        rows.append({
            'source_fits' : fits_basename,
            'det_code'    : DET_CODE,
            # merged bbox in detector pixel coords
            'det_xmin'    : xmin,
            'det_ymin'    : ymin,
            'det_xmax'    : xmax,
            'det_ymax'    : ymax,
            'det_centerx'      : cx,
            'det_centery'      : cy,
            # cutout window in detector pixel coords
            'cutout_x0'   : x0,
            'cutout_y0'   : y0,
            'cutout_x1'   : x0 + side,
            'cutout_y1'   : y0 + side,
            'cutout_centerx'   : x0 + side // 2,
            'cutout_centery'   : y0 + side // 2,
            # bbox relative to cutout origin (padded, matches _bbox.npy)
            'local_xmin'  : lxmin_pad,
            'local_ymin'  : lymin_pad,
            'local_xmax'  : lxmax_pad,
            'local_ymax'  : lymax_pad,
            'local_centerx'    : (lxmin_pad + lxmax_pad) // 2,
            'local_centery'    : (lymin_pad + lymax_pad) // 2,
            # ZO match label
            'zeroth_order': 1 if (xmin, ymin, xmax, ymax) in zo_matched_bboxes else 0,
            'source_type' : 'dark' if (xmin, ymin, xmax, ymax) in dark_bbox_set else 'bright',
            # stem links this row to its .npy cutout and SAM mask
            'cutout_stem' : f"cutout_{fits_id}_DET{DET_CODE}_{xmin}_{xmax}_{ymin}_{ymax}",
        })

    df_det = pd.DataFrame(rows)
    df_det.to_csv(out_csv, index=False)
    print(f"Saved {len(df_det)} rows → {out_csv}  "
        f"(ZO matched: {df_det['zeroth_order'].sum()}, unmatched: {(df_det['zeroth_order']==0).sum()})")