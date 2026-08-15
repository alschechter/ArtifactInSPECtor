import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from segment_anything import sam_model_registry, SamPredictor
import glob
import os
import pandas as pd
import torch
import argparse
from concurrent.futures import ThreadPoolExecutor
from scipy import ndimage
from astropy.visualization import ZScaleInterval
from astropy.stats import mad_std
from MaskEncoder import MaskEncoder

parser = argparse.ArgumentParser()
parser.add_argument('--no-plots', action='store_true', help='Skip saving result JPGs (faster)')
parser.add_argument('--force', action='store_true', help='Recompute masks even if already present, overwriting existing results')
parser.add_argument('--min-mask-size', type=int, default=5, help='Drop connected components smaller than this (pixels) to remove salt-and-pepper noise; 0 disables')
parser.add_argument('--max-peaks', type=int, default=10, help='Max number of distinct bright-source clusters to prompt SAM with')
parser.add_argument('--max-box-frac', type=float, default=0.6, help='Reject a candidate mask if it covers more than this fraction of the box area (guards against SAM filling the whole box on low-contrast/diffuse sources); falls back to the smallest candidate if all three exceed it')
parser.add_argument('--output-dir', type=str, default='sam_results', help='Directory to save masks/plots/CSV into')
args = parser.parse_args()

sam_checkpoint = "/users/7/aimees/AI_Inspector/sam_vit_h_4b8939.pth"
model_type = "vit_h"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

predictor = SamPredictor(sam)

plot_executor = ThreadPoolExecutor(max_workers=2)


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def clean_mask(mask, min_size=8):
    """Drop connected components smaller than min_size to remove salt-and-pepper noise."""
    if min_size <= 0:
        return mask
    structure = np.ones((3, 3))  # 8-connectivity
    labeled, num_features = ndimage.label(mask, structure=structure)
    sizes = ndimage.sum(mask, labeled, range(1, num_features + 1))

    cleaned = np.zeros_like(mask)
    for i, size in enumerate(sizes, start=1):
        if size >= min_size:
            cleaned[labeled == i] = 1
    return cleaned


zscale = ZScaleInterval()


def _grow_cluster(bbox_region, seed, n):
    """Grow an n-pixel 8-connected cluster from seed=(y, x), greedily adding
    whichever touching neighbor is brightest at each step."""
    h, w = bbox_region.shape
    y0, x0 = seed
    selected = [(y0, x0)]
    selected_set = {(y0, x0)}
    for _ in range(n - 1):
        candidates = set()
        for (yy, xx) in selected:
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx = yy + dy, xx + dx
                    if 0 <= ny < h and 0 <= nx < w and (ny, nx) not in selected_set:
                        candidates.add((ny, nx))
        if not candidates:
            break
        best = max(candidates, key=lambda p: bbox_region[p[0], p[1]])
        selected.append(best)
        selected_set.add(best)
    return selected


def top_bright_cluster(bbox_region, n=3):
    """Return up to n pixel coords (y, x) forming one 8-connected cluster,
    grown greedily from the brightest pixel by repeatedly adding the
    brightest neighbor touching the current cluster."""
    y0, x0 = np.unravel_index(np.argmax(bbox_region), bbox_region.shape)
    return _grow_cluster(bbox_region, (y0, x0), n)


def find_bright_clusters(bbox_region, max_peaks=3, points_per_peak=3, min_peak_distance=5, sigma_thresh=3.0):
    """Find up to max_peaks distinct bright source regions in bbox_region.

    Each candidate peak is grown into a points_per_peak-pixel 8-connected
    cluster (same greedy growth as top_bright_cluster). A cluster is only
    kept if *every* pixel in it clears a background-noise threshold — a lone
    hot pixel has dim immediate neighbors, so its cluster fails this check
    and gets rejected instead of producing a spurious point prompt. A real
    source has correlated brightness across its neighboring pixels and
    passes.
    """
    h, w = bbox_region.shape
    thresh = np.median(bbox_region) + sigma_thresh * mad_std(bbox_region)

    work = bbox_region.copy()
    clusters = []
    while len(clusters) < max_peaks:
        y0, x0 = np.unravel_index(np.argmax(work), work.shape)
        if work[y0, x0] <= thresh:
            break  # no more significant peaks left

        cluster = _grow_cluster(bbox_region, (y0, x0), points_per_peak)
        is_real = len(cluster) == points_per_peak and min(bbox_region[p] for p in cluster) > thresh

        # suppress this neighborhood so the next iteration finds a different peak
        y_lo, y_hi = max(0, y0 - min_peak_distance), min(h, y0 + min_peak_distance + 1)
        x_lo, x_hi = max(0, x0 - min_peak_distance), min(w, x0 + min_peak_distance + 1)
        work[y_lo:y_hi, x_lo:x_hi] = -np.inf

        if is_real:
            clusters.append(cluster)

    return clusters

os.makedirs(args.output_dir, exist_ok=True)

imageslist = sorted(
    (f for f in glob.glob('cutouts/*.npy') if '_bbox' not in f \
     and '_precontsub' not in f and '_dark' not in f and '_points' not in f),
    key=os.path.getsize,
    reverse=True,
)
imageslist = imageslist
print(f"Running SAM on {len(imageslist)} cutouts → saving to {args.output_dir}/")

rows = []
prompt_stats = []

# Pre-load rows for any masks that were already computed in a prior run
for fname in imageslist:
    stem = os.path.splitext(os.path.basename(fname))[0]
    mask_path = os.path.join(args.output_dir, f'{stem}_mask.npy')
    if os.path.exists(mask_path) and not args.force:
        mask = np.load(mask_path)
        ys, xs = np.where(mask)
        h = np.shape(mask)[0]
        w = np.shape(mask)[1]
        pad = 10
        if len(xs):
            rows.append({
                'stem'        : stem,
                'sam_x0'      : max(0, int(xs.min()) - pad),
                'sam_y0'      : max(0, int(ys.min()) - pad),
                'sam_x1'      : min(w, int(xs.max()) + pad),
                'sam_y1'      : min(h, int(ys.max()) + pad),
                'mask_h'      : h,
                'mask_w'      : w,
                'mask_encoded': MaskEncoder(w,h).encode(mask),
            })

for fname in imageslist:
    stem = os.path.splitext(os.path.basename(fname))[0]
    if os.path.exists(os.path.join(args.output_dir, f'{stem}_mask.npy')) and not args.force:
        print(f"Skipping {stem} (already done)")
        continue

    raw = np.load(fname).astype(np.float32)
    if raw.ndim != 2:
        print(f"Skipping {stem} (unexpected shape {raw.shape})")
        continue
    dark_flag = fname.replace('.npy', '_dark.npy')
    if os.path.exists(dark_flag):
        raw = -raw

    finite = raw[np.isfinite(raw)]
    try:
        vmin, vmax = zscale.get_limits(finite)
    except Exception:
        vmin, vmax = np.nanpercentile(raw, 1), np.nanpercentile(raw, 99)
    normed = np.clip((raw - vmin) / (vmax - vmin + 1e-12), 0, 1)
    normed = np.nan_to_num(normed, nan=0.0)
    scaled = (normed * 255).astype(np.uint8)
    image = np.stack([scaled, scaled, scaled], axis=-1)
    h, w = image.shape[:2]
    size_tag = f"  *** LARGE {h}×{w} ***" if max(h, w) > 500 else f"{h}×{w}"
    print(f"{os.path.basename(fname)}  {size_tag}")

    predictor.set_image(image)

    bbox_file = fname.replace('.npy', '_bbox.npy')
    if os.path.exists(bbox_file):
        bbox_local = np.load(bbox_file)
        x0b, y0b, x1b, y1b = bbox_local
        box = np.array([[max(x0b,0), max(y0b,0), min(x1b,w), min(y1b,h)]], dtype=np.float32)
    else:
        box = np.array([[0, 0, w, h]], dtype=np.float32)

    x0b_i, y0b_i = int(box[0, 0]), int(box[0, 1])
    x1b_i, y1b_i = int(box[0, 2]), int(box[0, 3])
    box_area = max((x1b_i - x0b_i) * (y1b_i - y0b_i), 1)
    bbox_region = normed[y0b_i:y1b_i, x0b_i:x1b_i]
    clusters = find_bright_clusters(bbox_region, max_peaks=args.max_peaks)
    if not clusters:
        # nothing cleared the noise threshold; still give SAM at least one prompt
        clusters = [top_bright_cluster(bbox_region, n=3)]

    # Same statistical test as find_bright_clusters (median + sigma*mad_std).
    # Used below to pick negative (background) prompt points, so SAM is told
    # what is *not* the source instead of only ever being shown where it is.
    thresh_val = np.median(bbox_region) + 3.0 * mad_std(bbox_region)

    def background_points(exclude, n=4, min_dist=3):
        """Up to n background-pixel coords (y, x), tried near the box's
        corners/edges (typical background locations for a roughly-centered
        source), kept only if they're actually below-threshold and not too
        close to any point in `exclude` (the positive cluster prompts)."""
        h_r, w_r = bbox_region.shape
        candidates = [(0, 0), (0, w_r - 1), (h_r - 1, 0), (h_r - 1, w_r - 1),
                      (0, w_r // 2), (h_r - 1, w_r // 2), (h_r // 2, 0), (h_r // 2, w_r - 1)]
        picked = []
        for (y, x) in candidates:
            if bbox_region[y, x] >= thresh_val:
                continue
            if any(abs(y - py) < min_dist and abs(x - px) < min_dist for (py, px) in exclude):
                continue
            picked.append((y, x))
            if len(picked) >= n:
                break
        return picked

    # Predict each source separately (rather than all points in one call) so SAM
    # isn't tempted to draw one big mask spanning every point at once when the
    # sources are spatially separated. Union the per-source masks together, but
    # stop accumulating once the union itself starts swallowing the box — a
    # single extended/diffuse source can get fragmented into several "distinct"
    # peaks, each individually reasonable, whose union still overruns the box.
    # find_bright_clusters returns peaks brightest-first, so the dominant source
    # is always accepted before this cap can cut anything off.
    mask = np.zeros((h, w), dtype=np.uint8)
    for cluster in clusters:
        neg_local = background_points(cluster)
        coords = cluster + neg_local
        labels = [1] * len(cluster) + [0] * len(neg_local)
        point_coords = np.array([[x + x0b_i, y + y0b_i] for (y, x) in coords])
        point_labels = np.array(labels)
        cand_masks, cand_scores, _ = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=box,
            multimask_output=True,
        )
        areas = cand_masks.reshape(cand_masks.shape[0], -1).sum(axis=1)
        smallest_idx = int(np.argmin(areas))
        not_box_filling = areas <= args.max_box_frac * box_area
        if not_box_filling.any():
            # among candidates that don't just swallow the whole box, take the
            # smallest rather than trusting SAM's confidence score — the score
            # tracks how "complete" SAM thinks the object is, not how tight the
            # mask is, and measured over this dataset it picks a candidate a
            # median 1.9x (mean 3.1x) larger than the smallest available option
            # ~88% of the time, which is exactly the bloat we're trying to avoid
            valid_idx = np.where(not_box_filling)[0]
            best = valid_idx[np.argmin(areas[valid_idx])]
            fallback_used = False
        else:
            # every candidate still fills the box despite the negative prompts —
            # fall back to the smallest as the least-wrong SAM output
            best = smallest_idx
            fallback_used = True
        mask |= cand_masks[best].astype(np.uint8)

        prompt_stats.append({
            'stem': stem,
            'n_pos': len(cluster), 'n_neg': len(neg_local),
            'area0': int(areas[0]), 'area1': int(areas[1]), 'area2': int(areas[2]),
            'score0': float(cand_scores[0]), 'score1': float(cand_scores[1]), 'score2': float(cand_scores[2]),
            'box_area': box_area,
            'chosen_idx': int(best), 'smallest_idx': smallest_idx,
            'chose_smallest': bool(best == smallest_idx),
            'fallback_used': fallback_used,
        })

        if mask.sum() > args.max_box_frac * box_area:
            break

    mask = clean_mask(mask, min_size=args.min_mask_size)

    np.save(os.path.join(args.output_dir, f'{stem}_mask.npy'), mask)

    ys, xs = np.where(mask)
    if len(xs):
        sam_x0, sam_y0, sam_x1, sam_y1 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
    else:
        sam_x0, sam_y0, sam_x1, sam_y1 = int(box[0,0]), int(box[0,1]), int(box[0,2]), int(box[0,3])

    if not args.no_plots:
        def save_plot(image, mask, box, out_path):
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            axes[0].imshow(image, origin='upper', cmap='gray')
            x0b, y0b, x1b, y1b = box[0]
            rect = plt.matplotlib.patches.Rectangle(
                xy=(x0b, y0b), width=x1b - x0b, height=y1b - y0b,
                edgecolor='limegreen', facecolor='none', linewidth=1.5, linestyle='--'
            )
            rect.set_path_effects([pe.Stroke(linewidth=1.8, foreground='black'), pe.Normal()])
            axes[0].add_patch(rect)
            axes[0].set_title('Data')
            axes[0].axis('off')
            for spine in axes[0].spines.values():
                spine.set_edgecolor('limegreen'); spine.set_linewidth(2); spine.set_visible(True)
            axes[1].imshow(image, origin='upper', cmap='gray')
            show_mask(mask, axes[1])
            axes[1].set_title('Segmentation Map') #Segmentation Map
            axes[1].axis('off')
            for spine in axes[1].spines.values():
                spine.set_edgecolor('limegreen'); spine.set_linewidth(2); spine.set_visible(True)
            plt.tight_layout()
            plt.savefig(out_path, bbox_inches='tight')
            plt.close(fig)

        out_path = os.path.join(args.output_dir, f'{stem}_result.jpg')
        plot_executor.submit(save_plot, image.copy(), mask.copy(), box.copy(), out_path)

    pad = 10
    rows.append({
        'stem'        : stem,
        'sam_x0'      : max(0, sam_x0 - pad),
        'sam_y0'      : max(0, sam_y0 - pad),
        'sam_x1'      : min(w, sam_x1 + pad),
        'sam_y1'      : min(h, sam_y1 + pad),
        'mask_h'      : h,
        'mask_w'      : w,
        'mask_encoded': MaskEncoder(w,h).encode(mask),
    })

plot_executor.shutdown(wait=True)

stats_csv = os.path.join(args.output_dir, 'prompt_stats.csv')
stats_df = pd.DataFrame(prompt_stats)
stats_df.to_csv(stats_csv, index=False)
if len(stats_df):
    print(f"Chose non-smallest candidate: {(~stats_df['chose_smallest']).mean():.1%} of clusters")
    print(f"Fallback (all 3 box-filling) used: {stats_df['fallback_used'].mean():.1%} of clusters")

out_csv = os.path.join(args.output_dir, 'sam_masks.csv')
sam_df = pd.DataFrame(rows)
sam_df.to_csv(out_csv, index=False)
print(f"Saved {len(rows)} mask rows → {out_csv}")

group_ids = set()
for stem in sam_df['stem']:
    after_cutout = stem[len('cutout_'):]
    fits_id, rest = after_cutout.split('_DET', 1)
    det_code = rest.split('_', 1)[0]
    group_ids.add(f"{fits_id}_DET{det_code}")

for group_id in group_ids:
    in_det_csv = f"detections_{group_id}.csv"
    out_det_csv = in_det_csv
    merge_cols = ['sam_x0', 'sam_y0', 'sam_x1', 'sam_y1', 'mask_h', 'mask_w', 'mask_encoded']
    mask_cols = sam_df[['stem'] + merge_cols].rename(columns={'stem': 'cutout_stem'})

    if os.path.exists(in_det_csv):
        det_df = pd.read_csv(in_det_csv)
        det_df = det_df.drop(columns=[c for c in merge_cols if c in det_df.columns])
        det_df = det_df.merge(mask_cols, on='cutout_stem', how='left')
    else:
        det_df = mask_cols

    det_df.to_csv(out_det_csv, index=False)
    n_merged = det_df['mask_encoded'].notna().sum()
    print(f"  Wrote {n_merged}/{len(det_df)} masks → {out_det_csv}")

print("Done.")
